"""Time-series forecasting of the churn rate with ARIMA and SARIMA (train -> validate -> test -> future)."""
import itertools
import warnings

import numpy as np
import pandas as pd

from .metrics import ts_metrics

warnings.filterwarnings("ignore")


def _libs():
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError as e:
        raise RuntimeError("ARIMA / SARIMA need statsmodels. Install with:  pip install statsmodels\n"
                           f"(original error: {e})")
    return ARIMA, SARIMAX


def build_series(df, target, date_col=None):
    """Monthly churn-rate series.
    - If the data has a usable date column -> real calendar months.
    - Otherwise (the public Netflix churn CSV has no dates) -> acquisition-order cohorts: rows are cut into
      equal consecutive blocks (ordered by customer_id) and each block is treated as one period."""
    if date_col and date_col in df:
        d = pd.to_datetime(df[date_col], errors="coerce")
        g = df.assign(_p=d.dt.to_period("M")).dropna(subset=["_p"]).groupby("_p")[target].agg(["mean", "size"])
        if len(g) >= 24:
            return dict(kind="calendar", labels=[str(p) for p in g.index], values=g["mean"].values.astype(float),
                        counts=g["size"].values.astype(int),
                        note=f"Monthly churn rate from '{date_col}'.")
    n_per = int(np.clip(len(df) // 40, 24, 60))
    chunks = np.array_split(np.arange(len(df)), n_per)
    vals = np.array([df[target].values[c].mean() for c in chunks], float)
    return dict(kind="cohort", labels=[f"P{i + 1:02d}" for i in range(n_per)], values=vals,
                counts=np.array([len(c) for c in chunks]),
                note=("No date column found, so rows (ordered by customer_id) are grouped into "
                      f"{n_per} equal acquisition cohorts; each cohort is treated as one period."))


def _fit(kind, y, order, seasonal, h):
    ARIMA, SARIMAX = _libs()
    y = np.asarray(y, float)
    if kind == "arima":
        res = ARIMA(y, order=order).fit()
    else:
        trend = "c" if (order[1] == 0 and seasonal[1] == 0) else None
        res = SARIMAX(y, order=order, seasonal_order=seasonal, trend=trend,
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    fc = res.get_forecast(steps=h)
    mean = np.asarray(fc.predicted_mean, float)
    ci = np.asarray(fc.conf_int(alpha=0.05), float)
    return dict(mean=mean, lo=ci[:, 0], hi=ci[:, 1], fitted=np.asarray(res.fittedvalues, float), aic=float(res.aic))


def _grid(kind, s):
    if kind == "arima":
        return [((p, d, q), (0, 0, 0, 0)) for p, d, q in itertools.product([0, 1, 2], [0, 1], [0, 1, 2])]
    seas = [(1, 0, 0, s), (0, 1, 1, s), (1, 1, 0, s)]
    return [((p, d, q), ss) for (p, d, q), ss in itertools.product(itertools.product([0, 1], [0, 1], [0, 1]), seas)]


def _clip(a):
    return np.clip(a, 0, 1)


def run_forecast(series, horizon, tracker):
    y = series["values"]
    n = len(y)
    n_tr, n_va = int(round(n * 0.70)), int(round(n * 0.15))
    n_te = n - n_tr - n_va
    s = 12 if n >= 36 else 4
    y_tr, y_va, y_te = y[:n_tr], y[n_tr:n_tr + n_va], y[n_tr + n_va:]
    out = {}
    for kind, step in (("arima", "arima"), ("sarima", "sarima")):
        tracker.start(step, f"grid search on validation ({len(_grid(kind, s))} candidates)")
        best = None
        
        first_error = None

        for order, seas in _grid(kind, s):
            try:
                r = _fit(kind, y_tr, order, seas, n_va)
                rmse = ts_metrics(y_va, r["mean"])["rmse"]
            except Exception as e:
                if first_error is None:
                    first_error = (
                        f"{type(e).__name__}: {e}"
                    )
                continue

            if np.isfinite(rmse) and (best is None or rmse < best[0]):
                best = (rmse, order, seas)

        if best is None:
            raise RuntimeError(
                f"No {kind.upper()} configuration could be fitted. "
                f"First underlying error: {first_error}"
            )
        _, order, seas = best
        r_tr = _fit(kind, y_tr, order, seas, n_va)                         # validation forecast
        r_tv = _fit(kind, np.concatenate([y_tr, y_va]), order, seas, n_te)  # test forecast
        r_all = _fit(kind, y, order, seas, horizon)                         # future forecast
        warm = order[1] + seas[1] * seas[3]                                 # first obs are not predictable after differencing
        fitted = r_tr["fitted"].copy()
        fitted[:max(warm, 1)] = np.nan
        pred = np.concatenate([fitted, r_tr["mean"], r_tv["mean"]])
        lo = np.concatenate([np.full(n_tr, np.nan), r_tr["lo"], r_tv["lo"]])
        hi = np.concatenate([np.full(n_tr, np.nan), r_tr["hi"], r_tv["hi"]])
        naive = np.full(n_te, np.concatenate([y_tr, y_va])[-1])
        out[kind] = dict(
            order=list(order), seasonal=list(seas), aic=r_all["aic"],
            pred=_clip(pred), lo=_clip(lo), hi=_clip(hi),
            future=_clip(r_all["mean"]), future_lo=_clip(r_all["lo"]), future_hi=_clip(r_all["hi"]),
            val=ts_metrics(y_va, _clip(r_tr["mean"])), test=ts_metrics(y_te, _clip(r_tv["mean"])),
            naive_test=ts_metrics(y_te, naive))
        label = f"ARIMA{tuple(order)}" if kind == "arima" else f"SARIMA{tuple(order)}x{tuple(seas)}"
        tracker.log(f"{label}: val RMSE={out[kind]['val']['rmse']:.4f}  test RMSE={out[kind]['test']['rmse']:.4f}")
        tracker.done(step, label)

    labels = list(series["labels"])
    if series["kind"] == "calendar":
        last = pd.Period(labels[-1], "M")
        fut_labels = [str(last + i) for i in range(1, horizon + 1)]
    else:
        fut_labels = [f"F{i:02d}" for i in range(1, horizon + 1)]
    split = ["train"] * n_tr + ["validation"] * n_va + ["test"] * n_te
    rows = []
    for i in range(n):
        rows.append(dict(period=labels[i], t_index=i + 1, split=split[i], actual_churn_rate=float(y[i]),
                         n_customers=int(series["counts"][i]),
                         **{f"{k}_{c}": (None if np.isnan(out[k][c][i]) else float(out[k][c][i]))
                            for k in ("arima", "sarima") for c in ("pred", "lo", "hi")}))
    for i in range(horizon):
        rows.append(dict(period=fut_labels[i], t_index=n + i + 1, split="forecast", actual_churn_rate=None, n_customers=None,
                         **{f"{k}_{c}": float(out[k][f"future{'' if c == 'pred' else '_' + c}"][i])
                            for k in ("arima", "sarima") for c in ("pred", "lo", "hi")}))
    for r in rows:
        r["arima_order"] = str(tuple(out["arima"]["order"]))
        r["sarima_order"] = f"{tuple(out['sarima']['order'])}x{tuple(out['sarima']['seasonal'])}"
    ui = dict(kind=series["kind"], note=series["note"], seasonal_period=s, horizon=horizon,
              labels=labels + fut_labels, actual=[float(v) for v in y], split_sizes=[n_tr, n_va, n_te],
              models={k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()}
                      for k, v in out.items()})
    return rows, ui
