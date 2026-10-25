"""
Recompute every evaluation metric FROM THE CSV FILES (no model needed).

    python evaluate_outputs.py                 # uses outputs/ folder
    python evaluate_outputs.py path/to/outputs

CSV 1 -> classification metrics per model (validation + test) using the stored 0/1 predictions
         and probabilities.  CSV 3 -> ARIMA / SARIMA error metrics on the validation and test windows.
"""
import math
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def cls_metrics(y, pred, proba):
    y, pred = np.asarray(y, int), np.asarray(pred, int)
    tp = int(((y == 1) & (pred == 1)).sum()); tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum()); fn = int(((y == 1) & (pred == 0)).sum())
    d = lambda a, b: a / b if b else 0.0
    prec, rec, spec = d(tp, tp + fp), d(tp, tp + fn), d(tn, tn + fp)
    mcc = d(tp * tn - fp * fn, math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return dict(TP=tp, TN=tn, FP=fp, FN=fn, Accuracy=d(tp + tn, len(y)), Precision=prec, Recall=rec, Specificity=spec,
                F1=d(2 * prec * rec, prec + rec), BalancedAcc=(rec + spec) / 2, MCC=mcc, ROC_AUC=roc_auc_score(y, proba))


def main():
    p1 = os.path.join(OUT, "1_model_predictions.csv")
    p3 = os.path.join(OUT, "3_timeseries_arima_sarima.csv")
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.4f}".format)
    d1 = pd.read_csv(p1)
    models = sorted({c[:-6] for c in d1.columns if c.endswith("_proba") and c != "best_proba"})
    for split in ("validation", "test"):
        part = d1[(d1["split"] == split) & d1["actual_churn"].notna() & (d1["actual_churn"] != "")]
        if part.empty:
            print(f"\n[{split}] no labelled rows - skipped"); continue
        rows = {m: cls_metrics(part["actual_churn"].astype(int), part[f"{m}_pred"], part[f"{m}_proba"]) for m in models}
        print(f"\n=== CLASSIFICATION · {split.upper()} ({len(part)} rows) ===")
        print(pd.DataFrame(rows).T)
    d3 = pd.read_csv(p3)
    print("\n=== TIME SERIES · ARIMA vs SARIMA ===")
    out = {}
    for split in ("validation", "test"):
        part = d3[d3["split"] == split].dropna(subset=["actual_churn_rate"])
        for m in ("arima", "sarima"):
            y, f = part["actual_churn_rate"].values, part[f"{m}_pred"].values
            e = y - f
            out[f"{m.upper()} {split}"] = dict(MAE=np.abs(e).mean(), RMSE=math.sqrt((e ** 2).mean()),
                                               MAPE_pct=100 * np.mean(np.abs(e) / np.abs(y)),
                                               sMAPE_pct=100 * np.mean(2 * np.abs(e) / (np.abs(y) + np.abs(f))),
                                               R2=1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum())
    print(pd.DataFrame(out).T)


if __name__ == "__main__":
    main()
