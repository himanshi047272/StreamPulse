# StreamPulse
### Netflix customer-churn prediction · Explainable AI · Time-series forecasting

StreamPulse (formerly *ChurnAI*) is a Flask dashboard that works on the **Netflix customer churn dataset**.
It follows a proper **train → validate → test** flow, then explains the best model with **SHAP + LIME**
and forecasts the churn rate with **ARIMA + SARIMA**.

## Run it
```bash
pip install -r requirements.txt
python app.py            # open http://localhost:5000
```
1. Open **Data & Run**, upload your Netflix churn CSV as the *train file* (needs a `churned` column).
2. Optionally upload a *test file*. If you don't, 15 % of the train file is held out as the test set.
3. Click **Run pipeline**. No data handy? Tick *use built-in sample* (synthetic, same column schema).

Expected columns (public Netflix churn dataset): `customer_id, age, gender, subscription_type, watch_hours,
last_login_days, region, device, monthly_fee, churned, payment_method, number_of_profiles,
avg_watch_time_per_day, favorite_genre`. Extra / missing columns are handled automatically.

## Pipeline
| Stage | What happens |
|---|---|
| **Train** | clean → feature engineering (fit on train only) → correlation feature selection → StandardScaler → SMOTE (train only, only if minority < 40 %) → fit 6 models |
| **Validate** | score every model on the validation split, tune each model's decision threshold (max F1), rank by ROC-AUC, pick the best |
| **Test** | run all models on the test file with the validation-tuned thresholds and compute every metric |
| **Explain** | SHAP (Linear/Kernel explainer) + LIME (tabular) on the best model |
| **Forecast** | ARIMA + SARIMA grid-searched on a validation window, scored on a test window, refit on everything for the future forecast |

## Models (6, no Random Forest / tree ensembles)
Logistic Regression · Linear Discriminant Analysis · Gaussian Naive Bayes · K-Nearest Neighbors · SVM (RBF) · Neural Network (MLP)

## The 3 CSV outputs (`outputs/`)
| File | Content |
|---|---|
| `1_model_predictions.csv` | normal outputs: per-customer probability + 0/1 prediction for all 6 models (validation & test rows), actual label, best-model risk level and retention recommendation |
| `2_xai_shap_lime.csv` | global + per-customer **SHAP** and **LIME** attributions (feature, value, attribution, direction) |
| `3_timeseries_arima_sarima.csv` | actual churn rate, **ARIMA** and **SARIMA** predictions + 95 % intervals for train / validation / test / future periods |

## Evaluation metrics & formulas
Shown in the **Evaluation** page with the numbers plugged in: Accuracy, Precision, Recall, Specificity, F1,
Balanced Accuracy, MCC, Cohen's Kappa, ROC-AUC, PR-AUC, Log-loss, Brier score; time series: MAE, RMSE, MAPE, sMAPE, R².
To recompute them straight from the CSVs: `python evaluate_outputs.py`.

## Time-series note
The public Netflix churn CSV has **no date column**. If one is found (`*_date`, `*_at`, `timestamp`) real calendar
months are used; otherwise rows ordered by `customer_id` are cut into equal acquisition **cohorts** and each cohort
is one period. The UI states which of the two was used.

## Rename the app
Change `APP_NAME` in `pipeline/config.py`.

## Layout
```
app.py                 Flask routes + background job
pipeline/              data.py · models.py · metrics.py · xai.py · forecast.py · retention.py · runner.py
templates/ static/     dashboard UI (no JS libraries, works offline)
outputs/               the 3 CSVs (+ results cache)    models/  saved best model
evaluate_outputs.py    recompute metrics from the CSVs
```
