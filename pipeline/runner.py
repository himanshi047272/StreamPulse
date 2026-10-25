"""Orchestrates: load -> features -> SMOTE -> TRAIN -> VALIDATE -> TEST -> SHAP/LIME -> ARIMA/SARIMA -> 3 CSVs."""
import os
import re
import threading
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import data as D
from . import forecast as F
from . import models as M
from . import xai as X
from .config import CSV_FILES, DEFAULTS, OUTPUT_DIR, RANDOM_STATE
from .metrics import CLS_FORMULAS, TS_FORMULAS, classification_metrics, plug_numbers, roc_points
from .retention import recommend, risk_level

STEP_DEFS = [
    ("load", "Load & clean data", "train"),
    ("features", "Feature engineering & selection", "train"),
    ("balance", "Scale + SMOTE balancing", "train"),
    ("fit", "Train 6 models", "train"),
    ("validate", "Validate & tune thresholds", "validate"),
    ("select", "Select best model", "validate"),
    ("test_prep", "Prepare test file", "test"),
    ("test_pred", "Predict on test file", "test"),
    ("test_eval", "Compute test metrics", "test"),
    ("shap", "SHAP explanations", "explain"),
    ("lime", "LIME explanations", "explain"),
    ("arima", "ARIMA forecast", "explain"),
    ("sarima", "SARIMA forecast", "explain"),
    ("export", "Write the 3 CSV files", "explain"),
]


class Tracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        with self.lock:
            self.state = "idle"
            self.steps = [dict(id=i, title=t, phase=p, status="pending", detail="", secs=None, _t0=None) for i, t, p in STEP_DEFS]
            self.models = {}
            self.logs = []
            self.error = None
            self.started = None
            self.finished = None

    def _step(self, sid):
        return next(s for s in self.steps if s["id"] == sid)

    def start(self, sid, detail=""):
        with self.lock:
            s = self._step(sid)
            s.update(status="running", detail=detail, _t0=time.time())

    def done(self, sid, detail=None):
        with self.lock:
            s = self._step(sid)
            s["status"] = "done"
            s["secs"] = round(time.time() - (s["_t0"] or time.time()), 1)
            if detail is not None:
                s["detail"] = detail

    def detail(self, sid, detail):
        with self.lock:
            self._step(sid)["detail"] = detail

    def model(self, name, status, pct):
        with self.lock:
            self.models[name] = dict(name=name, status=status, pct=pct)

    def log(self, msg):
        with self.lock:
            self.logs.append(f"{datetime.now().strftime('%H:%M:%S')}  {msg}")
            self.logs = self.logs[-300:]

    def snapshot(self):
        with self.lock:
            done = sum(s["status"] == "done" for s in self.steps)
            return dict(
                state=self.state, error=self.error, progress=round(100 * done / len(self.steps)),
                elapsed=round((self.finished or time.time()) - self.started, 1) if self.started else 0,
                steps=[{k: v for k, v in s.items() if k != "_t0"} for s in self.steps],
                models=list(self.models.values()), logs=list(self.logs))


def _slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _split(df, target, opts, has_test):
    strat = lambda d: d[target]
    if has_test:
        tr, va = train_test_split(df, test_size=opts["val_size"], stratify=strat(df), random_state=RANDOM_STATE)
        return tr, va, None
    hold = opts["holdout_size"]
    trva, te = train_test_split(df, test_size=hold, stratify=strat(df), random_state=RANDOM_STATE)
    tr, va = train_test_split(trva, test_size=hold / (1 - hold), stratify=strat(trva), random_state=RANDOM_STATE)
    return tr, va, te


def run_pipeline(train_raw, test_raw, opts, tr_name, te_name, tracker, bundle_out):
    opts = {**DEFAULTS, **(opts or {})}
    t_start = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── TRAIN: load & clean ────────────────────────────────────────────────
    tracker.start("load")
    df, info = D.clean(train_raw)
    target = info["target"]
    if target is None:
        raise ValueError("Train file needs a churn label column (e.g. 'churned'). Columns found: " + ", ".join(train_raw.columns[:15]))
    if df[target].nunique() < 2:
        raise ValueError("The churn column has only one class - cannot train a classifier.")
    id_col = D.find_id(df)
    df = D.chronological(df, id_col)
    te_df = None
    if test_raw is not None:
        te_df, te_info = D.clean(test_raw)
    tr, va, te_hold = _split(df, target, opts, te_df is not None)
    if te_df is None:
        te_df = te_hold
        te_name = "hold-out split of train file"
    test_has_labels = target in te_df.columns
    tracker.log(f"Train file '{tr_name}': {info['rows_in']} rows, {info['duplicates_removed']} duplicates removed.")
    tracker.done("load", f"{len(tr)} train / {len(va)} validation / {len(te_df)} test rows")

    # ── features ───────────────────────────────────────────────────────────
    tracker.start("features")
    fb = D.FeatureBuilder().fit(tr, target)
    Xtr_all = fb.transform(tr, selected=False)
    kept, dropped = fb.select(Xtr_all, tr[target].values)
    Xtr, Xva = fb.transform(tr), fb.transform(va)
    ytr, yva = tr[target].values, va[target].values
    tracker.log(f"{len(kept)} features kept, {len(dropped)} dropped (|corr| < 0.01): {', '.join(dropped) or 'none'}")
    tracker.done("features", f"{len(kept)} features")

    # ── scale + SMOTE (train only) ─────────────────────────────────────────
    tracker.start("balance")
    scaler = StandardScaler().fit(Xtr.values)
    Xtr_s, Xva_s = scaler.transform(Xtr.values), scaler.transform(Xva.values)
    Xtr_b, ytr_b, smote_used = M.balance(Xtr_s, ytr)
    tracker.log(f"SMOTE {'applied' if smote_used else 'skipped (classes already balanced)'}: "
                f"{len(ytr)} -> {len(ytr_b)} training rows" + ("" if M.SMOTE_AVAILABLE or not smote_used else " (built-in SMOTE)"))
    tracker.done("balance", "SMOTE applied" if smote_used else "not needed")

    # ── TRAIN + VALIDATE ───────────────────────────────────────────────────
    tracker.start("fit", "6 models")
    fitted, rows = M.train_and_validate(Xtr_b, ytr_b, Xva_s, yva, tracker)
    tracker.done("fit")
    tracker.start("validate")
    tracker.done("validate", "threshold tuned per model (max F1)")
    tracker.start("select")
    ranked = sorted(rows, key=lambda n: rows[n]["val"]["roc_auc"], reverse=True)
    best = ranked[0]
    tracker.log(f"Best model on validation ROC-AUC: {best}")
    tracker.done("select", best)

    # ── TEST ───────────────────────────────────────────────────────────────
    tracker.start("test_prep")
    Xte = fb.transform(te_df)
    Xte_s = scaler.transform(Xte.values)
    yte = te_df[target].values if test_has_labels else None
    tracker.done("test_prep", f"{len(te_df)} rows" + ("" if test_has_labels else " (no labels - metrics skipped)"))
    tracker.start("test_pred")
    test_proba = {n: fitted[n].predict_proba(Xte_s)[:, 1] for n in rows}
    tracker.done("test_pred")
    tracker.start("test_eval")
    plug = {}
    for n in rows:
        r = rows[n]
        r["test"] = classification_metrics(yte, test_proba[n], r["threshold"]) if test_has_labels else None
        r["test_roc"] = roc_points(yte, test_proba[n]) if test_has_labels else None
        plug[n] = {"val": plug_numbers(r["val"]), "test": plug_numbers(r["test"]) if r["test"] else None}
        if r["test"]:
            tracker.log(f"TEST  {n}: acc={r['test']['accuracy']:.4f} f1={r['test']['f1']:.4f} auc={r['test']['roc_auc']:.4f}")
    tracker.done("test_eval")

    # ── XAI (SHAP + LIME) on best model ────────────────────────────────────
    ids_te = te_df[id_col].astype(str).values if id_col and id_col in te_df else np.array([f"ROW{i + 1}" for i in range(len(te_df))])
    xai_rows, xai_ui = X.run_xai(best, fitted[best], Xtr_s, Xte_s, Xte.reset_index(drop=True), list(Xte.columns),
                                 test_proba[best], yte, ids_te, tracker, opts["explain_n"])

    # ── Time series ────────────────────────────────────────────────────────
    
    # Time-series forecasting: keep classification results if forecasting fails.
    series = F.build_series(df, target, fb.date_col)

    try:
        ts_rows, ts_ui = F.run_forecast(
            series, int(opts["horizon"]), tracker
        )
    except Exception as e:
        forecast_error = f"{type(e).__name__}: {e}"
        tracker.log(
            f"Forecasting unavailable; classification results will be preserved. "
            f"Reason: {forecast_error}"
        )

        ts_rows = [{
            "period": "Forecast unavailable",
            "t_index": None,
            "split": "unavailable",
            "actual_churn_rate": None,
            "n_customers": None,
            "forecast_status": "unavailable",
            "forecast_error": forecast_error,
        }]

        ts_ui = {
            "unavailable": True,
            "error": forecast_error,
            "kind": series["kind"],
            "note": series["note"],
            "seasonal_period": 12 if len(series["values"]) >= 36 else 4,
            "horizon": int(opts["horizon"]),
            "labels": [],
            "actual": [],
            "split_sizes": [0, 0, 0],
            "models": {},
        }


    # ── CSV exports ────────────────────────────────────────────────────────
    tracker.start("export")
    ids_va = va[id_col].astype(str).values if id_col and id_col in va else np.array([f"VAL{i + 1}" for i in range(len(va))])
    raw_cols = [c for c in ("subscription_type", "monthly_fee", "last_login_days", "watch_hours", "number_of_profiles") if c in df]

    def block(split, ids, raw, y, proba):
        d = pd.DataFrame({"record_id": ids, "split": split, "actual_churn": y if y is not None else ""})
        for n in rows:
            thr = rows[n]["threshold"]
            d[f"{_slug(n)}_proba"] = np.round(proba[n], 6)
            d[f"{_slug(n)}_pred"] = (proba[n] >= thr).astype(int)
        bp = proba[best]
        d["best_model"] = best
        d["best_proba"] = np.round(bp, 6)
        d["best_pred"] = (bp >= rows[best]["threshold"]).astype(int)
        d["risk_level"] = [risk_level(p) for p in bp]
        rr = raw.reset_index(drop=True)
        d["recommendation"] = [recommend(rr.iloc[i].to_dict(), bp[i]) for i in range(len(rr))]
        for c in raw_cols:
            if c in raw:
                d[c] = rr[c].values
        return d

    val_proba = {n: rows[n]["val_proba"] for n in rows}
    csv1 = pd.concat([block("validation", ids_va, va, yva, val_proba),
                      block("test", ids_te, te_df, yte, test_proba)], ignore_index=True)
    csv2 = pd.DataFrame(xai_rows)
    csv3 = pd.DataFrame(ts_rows)
    csv3["n_customers"] = csv3["n_customers"].astype("Int64")
    outputs = {}
    for key, df_out in (("predictions", csv1), ("xai", csv2), ("forecast", csv3)):
        path = os.path.join(OUTPUT_DIR, CSV_FILES[key])
        df_out.to_csv(path, index=False, float_format="%.6f")
        outputs[key] = dict(file=CSV_FILES[key], rows=int(len(df_out)), cols=list(df_out.columns),
                            kb=round(os.path.getsize(path) / 1024, 1))
        tracker.log(f"Wrote {CSV_FILES[key]} ({len(df_out)} rows)")
    tracker.done("export", "3 files")

    # ── Retention summary (test rows) ──────────────────────────────────────
    te_part = csv1[csv1["split"] == "test"]
    risk_counts = te_part["risk_level"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0).to_dict()
    rec_counts = te_part[te_part["risk_level"] != "Low"]["recommendation"].value_counts().to_dict()
    top = te_part.sort_values("best_proba", ascending=False).head(15)
    keep = ["record_id", "best_proba", "risk_level", "recommendation"] + [c for c in raw_cols if c in top]
    retention = dict(risk_counts=risk_counts, rec_counts=rec_counts, top=top[keep].to_dict(orient="records"),
                     threshold=rows[best]["threshold"], n=int(len(te_part)))

    # ── Assemble the UI payload ────────────────────────────────────────────
    model_list = []
    for rank, n in enumerate(ranked, 1):
        r = rows[n]
        model_list.append(dict(name=n, family=r["family"], desc=r["desc"], rank=rank, is_best=n == best,
                               train_seconds=r["train_seconds"], threshold=r["threshold"], val=r["val"], test=r["test"],
                               val_roc=r["val_roc"], test_roc=r["test_roc"]))
    y_all = df[target]
    results = dict(
        meta=dict(app_run_at=datetime.now().strftime("%d %b %Y, %H:%M"), train_name=tr_name, test_name=te_name,
                  test_from_file=te_name != "hold-out split of train file", test_has_labels=test_has_labels,
                  rows_train_file=int(info["rows_in"]), duplicates_removed=int(info["duplicates_removed"]),
                  n_train=int(len(tr)), n_val=int(len(va)), n_test=int(len(te_df)),
                  n_train_balanced=int(len(ytr_b)), smote=bool(smote_used), smote_lib=M.SMOTE_AVAILABLE,
                  churn_rate=float(y_all.mean()), churn_train=float(ytr.mean()), churn_val=float(yva.mean()),
                  churn_test=float(yte.mean()) if test_has_labels else None,
                  features=kept, dropped=dropped, feature_corr={k: v for k, v in fb.corr.items()},
                  target=target, id_col=id_col, best=best, seconds=round(time.time() - t_start, 1)),
        models=model_list, evaluation=dict(cls_formulas=CLS_FORMULAS, ts_formulas=TS_FORMULAS, plug=plug),
        xai=xai_ui, forecast=ts_ui, retention=retention, outputs=outputs)

    bundle_out.update(builder=fb, scaler=scaler, model=fitted[best], best=best, threshold=rows[best]["threshold"],
                      all_models=fitted, thresholds={n: rows[n]["threshold"] for n in rows})
    return results
