"""Explainable AI: SHAP (Kernel / Linear explainer) and LIME (tabular) on the best model."""
import numpy as np
from scipy.stats import spearmanr

from .config import RANDOM_STATE


def _import_libs():
    try:
        import shap
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError as e:
        raise RuntimeError(
            "SHAP and LIME are required for the explainability step. Install them with:\n"
            "    pip install shap lime\n"
            f"(original error: {e})")
    return shap, LimeTabularExplainer


def _as_matrix(sv, n_features):
    """Normalise whatever shap returns to an (n_samples, n_features) array for the churn class."""
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv.reshape(-1, n_features)


def compute_shap(shap, model, model_name, X_bg, X_explain, tracker, nsamples=150):
    f = lambda A: model.predict_proba(A)[:, 1]
    nf = X_explain.shape[1]
    if model_name == "Logistic Regression":
        try:
            sv = shap.LinearExplainer(model, X_bg).shap_values(X_explain)
            tracker.log("SHAP: LinearExplainer (exact, log-odds units)")
            return _as_matrix(sv, nf), "LinearExplainer"
        except Exception as e:  # fall through to the model-agnostic explainer
            tracker.log(f"SHAP LinearExplainer failed ({e}); using KernelExplainer")
    ex = shap.KernelExplainer(f, X_bg)
    sv = ex.shap_values(X_explain, nsamples=nsamples, silent=True)
    tracker.log("SHAP: KernelExplainer (model-agnostic, probability units)")
    return _as_matrix(sv, nf), "KernelExplainer"


def compute_lime(LimeTabularExplainer, model, X_train, X_explain, feature_names, tracker, num_samples=800):
    exp = LimeTabularExplainer(X_train, feature_names=feature_names, class_names=["Stay", "Churn"],
                               mode="classification", discretize_continuous=False, random_state=RANDOM_STATE)
    W = np.zeros((len(X_explain), len(feature_names)))
    for i, x in enumerate(X_explain):
        e = exp.explain_instance(x, model.predict_proba, num_features=len(feature_names),
                                 num_samples=num_samples, labels=(1,))
        for j, w in e.as_map()[1]:
            W[i, j] = w
    return W


def pick_records(proba, n, seed=RANDOM_STATE):
    """Half highest-risk test customers, half random others, so both extremes and typical cases are explained."""
    n = min(n, len(proba))
    order = np.argsort(-proba)
    top = list(order[: n // 2])
    rest = np.setdiff1d(np.arange(len(proba)), top)
    rng = np.random.default_rng(seed)
    other = list(rng.choice(rest, n - len(top), replace=False)) if len(rest) else []
    return np.array(top + other, dtype=int)


def global_table(vals, feature_names):
    mean_abs = np.abs(vals).mean(axis=0)
    mean_sgn = vals.mean(axis=0)
    order = np.argsort(-mean_abs)
    return [{"feature": feature_names[j], "mean_abs": float(mean_abs[j]), "mean": float(mean_sgn[j]), "rank": r + 1}
            for r, j in enumerate(order)]


def run_xai(model_name, model, X_train_s, X_test_s, raw_test, feature_names, proba_test, y_test, ids, tracker, n_explain=40):
    shap, LimeTabularExplainer = _import_libs()
    rng = np.random.default_rng(RANDOM_STATE)
    bg = X_train_s[rng.choice(len(X_train_s), min(25, len(X_train_s)), replace=False)]
    sel = pick_records(proba_test, n_explain)
    Xe = X_test_s[sel]

    tracker.start("shap", f"{len(sel)} records")
    shap_vals, shap_kind = compute_shap(shap, model, model_name, bg, Xe, tracker)
    tracker.done("shap", shap_kind)

    tracker.start("lime", f"{len(sel)} records")
    lime_vals = compute_lime(LimeTabularExplainer, model, X_train_s, Xe, feature_names, tracker)
    tracker.done("lime", "LimeTabularExplainer")

    g_shap, g_lime = global_table(shap_vals, feature_names), global_table(lime_vals, feature_names)
    rho = spearmanr(np.abs(shap_vals).mean(0), np.abs(lime_vals).mean(0))[0]
    top5 = lambda g: {r["feature"] for r in g[:5]}

    # ── rows for CSV #2 ──
    rows = []
    for method, g in (("SHAP", g_shap), ("LIME", g_lime)):
        for r in g:
            rows.append(dict(method=method, scope="global", model=model_name, record_id="", rank=r["rank"],
                             feature=r["feature"], feature_value="", attribution=r["mean"],
                             abs_attribution=r["mean_abs"], direction="", predicted_probability="", actual_churn=""))
    for method, vals in (("SHAP", shap_vals), ("LIME", lime_vals)):
        for k, ridx in enumerate(sel):
            order = np.argsort(-np.abs(vals[k]))
            for rank, j in enumerate(order, 1):
                rows.append(dict(method=method, scope="local", model=model_name, record_id=ids[ridx], rank=rank,
                                 feature=feature_names[j], feature_value=raw_test.iloc[ridx][feature_names[j]],
                                 attribution=vals[k, j], abs_attribution=abs(vals[k, j]),
                                 direction="raises churn risk" if vals[k, j] > 0 else "lowers churn risk",
                                 predicted_probability=round(float(proba_test[ridx]), 4),
                                 actual_churn=int(y_test[ridx]) if y_test is not None else ""))

    # ── local examples for the UI (5 highest-risk explained records) ──
    local = []
    for k, ridx in enumerate(sel[:5]):
        def top(vals):
            o = np.argsort(-np.abs(vals[k]))[:8]
            return [{"feature": feature_names[j], "value": float(raw_test.iloc[ridx][feature_names[j]]),
                     "attr": float(vals[k, j])} for j in o]
        local.append({"record_id": str(ids[ridx]), "proba": float(proba_test[ridx]),
                      "actual": int(y_test[ridx]) if y_test is not None else None,
                      "shap": top(shap_vals), "lime": top(lime_vals)})

    summary = {
        "model": model_name, "shap_kind": shap_kind, "n_records": int(len(sel)),
        "shap_global": g_shap[:15], "lime_global": g_lime[:15],
        "spearman": None if np.isnan(rho) else float(rho),
        "top5_overlap": len(top5(g_shap) & top5(g_lime)), "local": local,
        "shap_units": "log-odds" if shap_kind == "LinearExplainer" else "probability",
    }
    return rows, summary
