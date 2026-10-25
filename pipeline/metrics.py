"""Evaluation metrics (classification + time-series) with their formulas and worked examples."""
import math
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

EPS = 1e-15


def _div(a, b):
    return float(a) / float(b) if b else 0.0


def confusion(y, pred):
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    return (int(((y == 0) & (pred == 0)).sum()), int(((y == 0) & (pred == 1)).sum()),
            int(((y == 1) & (pred == 0)).sum()), int(((y == 1) & (pred == 1)).sum()))  # tn, fp, fn, tp


def classification_metrics(y, proba, thr):
    y = np.asarray(y).astype(int)
    proba = np.asarray(proba, float)
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion(y, pred)
    n = tn + fp + fn + tp
    acc = _div(tp + tn, n)
    prec, rec, spec = _div(tp, tp + fp), _div(tp, tp + fn), _div(tn, tn + fp)
    f1 = _div(2 * prec * rec, prec + rec)
    npv = _div(tn, tn + fn)
    den = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0))
    mcc = _div(tp * tn - fp * fn, den)
    pe = _div((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn), n * n)
    kappa = _div(acc - pe, 1 - pe)
    p = np.clip(proba, EPS, 1 - EPS)
    two_classes = len(np.unique(y)) == 2
    return {
        "threshold": round(float(thr), 4), "n": n, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": acc, "precision": prec, "recall": rec, "specificity": spec, "f1": f1,
        "balanced_accuracy": (rec + spec) / 2, "npv": npv, "fpr": 1 - spec, "fnr": 1 - rec,
        "mcc": mcc, "kappa": kappa,
        "roc_auc": float(roc_auc_score(y, proba)) if two_classes else None,
        "pr_auc": float(average_precision_score(y, proba)) if two_classes else None,
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "brier": float(np.mean((proba - y) ** 2)),
    }


def best_threshold(y, proba):
    """Threshold (0.05-0.95) that maximises F1 on the VALIDATION set."""
    y = np.asarray(y).astype(int)
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        pred = proba >= t
        tp = int(((y == 1) & pred).sum()); fp = int(((y == 0) & pred).sum()); fn = int(((y == 1) & ~pred).sum())
        f = _div(2 * tp, 2 * tp + fp + fn)
        if f > best_f + 1e-12:
            best_t, best_f = float(t), f
    return round(best_t, 2)


def roc_points(y, proba, max_pts=70):
    fpr, tpr, _ = roc_curve(y, proba)
    idx = np.unique(np.linspace(0, len(fpr) - 1, min(max_pts, len(fpr))).astype(int))
    return [[round(float(fpr[i]), 4), round(float(tpr[i]), 4)] for i in idx]


CLS_FORMULAS = [
    {"key": "accuracy", "name": "Accuracy", "formula": "(TP + TN) / (TP + TN + FP + FN)", "note": "Share of all predictions that are correct."},
    {"key": "precision", "name": "Precision", "formula": "TP / (TP + FP)", "note": "Of customers flagged as churners, how many really churned."},
    {"key": "recall", "name": "Recall (Sensitivity)", "formula": "TP / (TP + FN)", "note": "Of real churners, how many we caught."},
    {"key": "specificity", "name": "Specificity", "formula": "TN / (TN + FP)", "note": "Of real stayers, how many we correctly left alone."},
    {"key": "f1", "name": "F1 Score", "formula": "2 x Precision x Recall / (Precision + Recall)", "note": "Harmonic mean of precision and recall."},
    {"key": "balanced_accuracy", "name": "Balanced Accuracy", "formula": "(Recall + Specificity) / 2", "note": "Accuracy that is fair under class imbalance."},
    {"key": "mcc", "name": "Matthews Corr. Coef.", "formula": "(TP x TN - FP x FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))", "note": "-1 to +1 correlation between prediction and truth."},
    {"key": "kappa", "name": "Cohen's Kappa", "formula": "(p_o - p_e) / (1 - p_e)", "note": "Agreement beyond chance (p_o = accuracy)."},
    {"key": "roc_auc", "name": "ROC-AUC", "formula": "Area under the TPR-vs-FPR curve", "note": "Ranking quality across all thresholds."},
    {"key": "pr_auc", "name": "PR-AUC", "formula": "Area under the Precision-vs-Recall curve (average precision)", "note": "Useful when churners are the minority."},
    {"key": "log_loss", "name": "Log Loss", "formula": "-1/N x sum[ y*ln(p) + (1-y)*ln(1-p) ]", "note": "Penalises confident wrong probabilities."},
    {"key": "brier", "name": "Brier Score", "formula": "1/N x sum (p - y)^2", "note": "Mean squared error of the probabilities."},
]


def plug_numbers(m):
    """Return {metric_key: 'formula with the numbers substituted = value'} for one metric dict."""
    tn, fp, fn, tp, n = m["tn"], m["fp"], m["fn"], m["tp"], m["n"]
    f = lambda v: f"{v:.4f}"
    return {
        "accuracy": f"({tp} + {tn}) / {n} = {f(m['accuracy'])}",
        "precision": f"{tp} / ({tp} + {fp}) = {f(m['precision'])}",
        "recall": f"{tp} / ({tp} + {fn}) = {f(m['recall'])}",
        "specificity": f"{tn} / ({tn} + {fp}) = {f(m['specificity'])}",
        "f1": f"2 x {f(m['precision'])} x {f(m['recall'])} / ({f(m['precision'])} + {f(m['recall'])}) = {f(m['f1'])}",
        "balanced_accuracy": f"({f(m['recall'])} + {f(m['specificity'])}) / 2 = {f(m['balanced_accuracy'])}",
        "mcc": f"({tp}x{tn} - {fp}x{fn}) / sqrt(({tp}+{fp})({tp}+{fn})({tn}+{fp})({tn}+{fn})) = {f(m['mcc'])}",
        "kappa": f"(p_o - p_e) / (1 - p_e) = {f(m['kappa'])}",
        "roc_auc": f"integral of ROC curve = {f(m['roc_auc'])}" if m["roc_auc"] is not None else "n/a",
        "pr_auc": f"integral of PR curve = {f(m['pr_auc'])}" if m["pr_auc"] is not None else "n/a",
        "log_loss": f"-mean(y ln p + (1-y) ln(1-p)) over {n} rows = {f(m['log_loss'])}",
        "brier": f"mean((p - y)^2) over {n} rows = {f(m['brier'])}",
    }


# ── time-series metrics ──────────────────────────────────────────────────────
TS_FORMULAS = [
    {"key": "mae", "name": "MAE", "formula": "1/n x sum |y - y_hat|", "note": "Average absolute error, in churn-rate units."},
    {"key": "rmse", "name": "RMSE", "formula": "sqrt( 1/n x sum (y - y_hat)^2 )", "note": "Like MAE but punishes big misses more."},
    {"key": "mape", "name": "MAPE (%)", "formula": "100/n x sum |(y - y_hat) / y|", "note": "Average percentage error."},
    {"key": "smape", "name": "sMAPE (%)", "formula": "100/n x sum 2|y - y_hat| / (|y| + |y_hat|)", "note": "Symmetric percentage error (bounded)."},
    {"key": "r2", "name": "R-squared", "formula": "1 - sum (y - y_hat)^2 / sum (y - mean(y))^2", "note": "Below 0 means worse than predicting the mean."},
]


def ts_metrics(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    err = y - yhat
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mape": float(100 * np.mean(np.abs(err) / np.clip(np.abs(y), EPS, None))),
        "smape": float(100 * np.mean(2 * np.abs(err) / np.clip(np.abs(y) + np.abs(yhat), EPS, None))),
        "r2": float(1 - (err ** 2).sum() / ss_tot) if ss_tot > 0 else None,
        "n": int(len(y)),
    }
