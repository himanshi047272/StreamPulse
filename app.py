"""
StreamPulse - Netflix churn prediction, explainability (SHAP + LIME) and forecasting (ARIMA + SARIMA).

Run:   python app.py
Open:  http://localhost:5000
"""
import io
import json
import math
import os
import threading
import time
import traceback
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request, send_from_directory

from pipeline import data as D
from pipeline.config import APP_NAME, APP_TAGLINE, CSV_FILES, DEFAULTS, MODEL_DIR, OUTPUT_DIR
from pipeline.retention import recommend, risk_level
from pipeline.runner import Tracker, run_pipeline

app = Flask(__name__)
TRACKER = Tracker()
STATE = {"results": None, "bundle": None}
RESULTS_PATH = os.path.join(OUTPUT_DIR, "results_cache.json")
BUNDLE_PATH = os.path.join(MODEL_DIR, "bundle.joblib")


def jsonable(o):
    """Recursively convert numpy / NaN values so jsonify never emits invalid JSON."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(o) or math.isinf(o)) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def load_cached():
    try:
        if os.path.exists(RESULTS_PATH):
            STATE["results"] = json.load(open(RESULTS_PATH, encoding="utf-8"))
        if os.path.exists(BUNDLE_PATH):
            STATE["bundle"] = joblib.load(BUNDLE_PATH)
    except Exception as e:  # stale cache from an older version - ignore
        print("Could not load cached results:", e)


def lib_status():
    out = {}
    for mod in ("shap", "lime", "statsmodels", "imblearn"):
        try:
            __import__(mod)
            out[mod] = True
        except ImportError:
            out[mod] = False
    return out


# ─── pages & config ──────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", app_name=APP_NAME, tagline=APP_TAGLINE)


@app.get("/api/config")
def api_config():
    return jsonify(app_name=APP_NAME, defaults=DEFAULTS, libs=lib_status(), csv=CSV_FILES,
                   has_results=STATE["results"] is not None)


# ─── run the pipeline in a background thread ─────────────────────────────────
def _job(train_df, test_df, opts, tr_name, te_name):
    bundle = {}
    try:
        res = run_pipeline(train_df, test_df, opts, tr_name, te_name, TRACKER, bundle)
        res = jsonable(res)
        STATE["results"], STATE["bundle"] = res, bundle
        json.dump(res, open(RESULTS_PATH, "w", encoding="utf-8"))
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(bundle, BUNDLE_PATH)
        TRACKER.log("Pipeline finished successfully.")
        TRACKER.state = "done"
    except Exception as e:
        traceback.print_exc()
        TRACKER.error = str(e)
        TRACKER.log("ERROR: " + str(e))
        TRACKER.state = "error"
    finally:
        TRACKER.finished = time.time()


@app.post("/api/run")
def api_run():
    if TRACKER.state == "running":
        return jsonify(error="A run is already in progress."), 409
    try:
        use_sample = request.form.get("use_sample") == "1"
        f_tr, f_te = request.files.get("train_csv"), request.files.get("test_csv")
        if use_sample:
            train_df, tr_name = D.make_sample(5000, 7, 1), "built-in sample (train)"
            test_df, te_name = D.make_sample(1500, 11, 5001), "built-in sample (test)"
        else:
            if not f_tr or not f_tr.filename:
                return jsonify(error="Please upload a train CSV (or tick 'use built-in sample')."), 400
            train_df, tr_name = pd.read_csv(f_tr), f_tr.filename
            test_df, te_name = (pd.read_csv(f_te), f_te.filename) if f_te and f_te.filename else (None, None)
        opts = {
            "val_size": float(request.form.get("val_size", DEFAULTS["val_size"])),
            "horizon": int(request.form.get("horizon", DEFAULTS["horizon"])),
            "explain_n": int(request.form.get("explain_n", DEFAULTS["explain_n"])),
        }
        opts["val_size"] = min(max(opts["val_size"], 0.1), 0.4)
    except Exception as e:
        return jsonify(error=f"Could not read the upload: {e}"), 400
    TRACKER.reset()
    TRACKER.state, TRACKER.started = "running", time.time()
    TRACKER.log("Pipeline started.")
    threading.Thread(target=_job, args=(train_df, test_df, opts, tr_name, te_name), daemon=True).start()
    return jsonify(status="started")


@app.get("/api/status")
def api_status():
    return jsonify(jsonable(TRACKER.snapshot()))


@app.get("/api/results")
def api_results():
    if STATE["results"] is None:
        return jsonify(error="No results yet - run the pipeline first."), 404
    return jsonify(STATE["results"])


# ─── CSV downloads / previews ────────────────────────────────────────────────
@app.get("/api/download/<key>")
def api_download(key):
    if key not in CSV_FILES or not os.path.exists(os.path.join(OUTPUT_DIR, CSV_FILES[key])):
        return jsonify(error="File not generated yet."), 404
    return send_from_directory(OUTPUT_DIR, CSV_FILES[key], as_attachment=True)


@app.get("/api/preview/<key>")
def api_preview(key):
    path = os.path.join(OUTPUT_DIR, CSV_FILES.get(key, "_"))
    if not os.path.exists(path):
        return jsonify(error="File not generated yet."), 404
    df = pd.read_csv(path, nrows=10)
    return jsonify(jsonable({"columns": list(df.columns), "rows": df.values.tolist()}))


@app.get("/api/sample/<which>")
def api_sample(which):
    df = D.make_sample(5000, 7, 1) if which == "train" else D.make_sample(1500, 11, 5001)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=netflix_{which}_sample.csv"})


# ─── single-customer prediction ──────────────────────────────────────────────
@app.post("/api/predict")
def api_predict():
    b = STATE["bundle"]
    if not b:
        return jsonify(error="Run the pipeline first."), 400
    try:
        def _num(v):  # numeric strings from the form -> numbers
            try:
                return float(v)
            except (TypeError, ValueError):
                return v
        row = D.standardize_columns(pd.DataFrame([{k: _num(v) for k, v in request.get_json(force=True).items()}]))
        X = b["scaler"].transform(b["builder"].transform(row).values)
        p = float(b["model"].predict_proba(X)[0, 1])
        return jsonify(probability=round(p, 4), risk=risk_level(p), at_risk=bool(p >= b["threshold"]),
                       threshold=b["threshold"], model=b["best"], recommendation=recommend(row.iloc[0].to_dict(), p))
    except Exception as e:
        return jsonify(error=str(e)), 400


load_cached()

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\n" + "=" * 56)
    print(f"  {APP_NAME}  -  http://localhost:5000")
    print("=" * 56 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
