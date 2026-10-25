"""Central configuration. Change APP_NAME here to rename the whole app."""
import os

APP_NAME = "StreamPulse"
APP_TAGLINE = "Netflix churn prediction, explainability & forecasting"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

RANDOM_STATE = 42

# The three CSV deliverables produced by every run
CSV_FILES = {
    "predictions": "1_model_predictions.csv",        # normal model outputs
    "xai":         "2_xai_shap_lime.csv",            # explainable-AI outputs (SHAP + LIME)
    "forecast":    "3_timeseries_arima_sarima.csv",  # time-series forecasting (ARIMA + SARIMA)
}

DEFAULTS = {
    "val_size": 0.20,       # share of the TRAIN file used for validation (when a test file is given)
    "holdout_size": 0.15,   # share used for validation AND for test when no test file is uploaded
    "horizon": 12,          # forecast periods ahead
    "explain_n": 40,        # number of test records explained by SHAP / LIME
}
