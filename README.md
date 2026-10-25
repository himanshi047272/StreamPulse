# StreamPulse 📊

### Customer Churn Prediction, Explainable AI & Time-Series Forecasting

StreamPulse is an end-to-end machine learning analytics project designed to analyze customer churn, generate churn predictions, explain model behavior, and forecast future churn trends.

It brings predictive analytics and time-series modeling together in a single dashboard, helping users explore customer retention patterns and understand the factors influencing model predictions.

## ✨ Key Features

- **Customer Churn Prediction:** Generate customer-level churn predictions using a machine learning pipeline.
- **Explainable AI (XAI):** Produce model explainability outputs using SHAP and LIME to help interpret predictions.
- **Time-Series Forecasting:** Use ARIMA and SARIMA models to analyze historical churn trends and generate forecasts.
- **Interactive Dashboard:** Explore prediction results, explainability outputs, and forecasting insights through a web interface.
- **Exportable Results:** Generate CSV outputs for predictions, explainability, and time-series forecasting.
- **Modular Pipeline:** Organize data processing, model execution, forecasting, and result generation into reusable components.

## 🛠️ Tech Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Explainable AI:** SHAP, LIME
- **Time-Series Modeling:** Statsmodels, ARIMA, SARIMA
- **Web Application:** Flask
- **Frontend:** HTML, CSS, JavaScript
- **Data Format:** CSV, Excel (`.xlsx`) for source data where supported

> The exact algorithms, evaluation metrics, and supported input columns depend on the project's implementation and configuration.

## 📂 Project Structure

```text
StreamPulse/
├── app.py
├── requirements.txt
├── pipeline/
│   ├── runner.py
│   └── forecast.py
├── static/
│   └── js/
│       └── app.js
├── templates/
│   └── index.html
├── .gitignore
└── README.md
```

*Additional files and folders may be present in the repository.*

## ⚙️ Getting Started

### Prerequisites

- Python 3.11 recommended for the tested Windows setup
- Git
- Visual Studio Code (recommended)

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/StreamPulse.git
cd StreamPulse
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Run the application

```powershell
python app.py
```

Open the local URL printed in the terminal, typically:

```text
http://127.0.0.1:5000
```

Follow the dashboard instructions to provide the dataset and run the pipeline.

## 📊 Generated Outputs

The pipeline generates CSV files such as:

| File | Purpose |
|---|---|
| `1_model_predictions.csv` | Customer-level model predictions |
| `2_xai_shap_lime.csv` | Explainability results |
| `3_timeseries_arima_sarima.csv` | Time-series forecasting results |

The exact columns and metrics depend on the selected dataset and pipeline configuration.

## 🔍 Explainable AI

StreamPulse includes SHAP/LIME-related outputs to support model interpretation. These techniques can help investigate which features contribute to model predictions.

Explanations should be interpreted in the context of the trained model and dataset; they do not establish causation.

## 📈 Forecasting

The forecasting module uses ARIMA/SARIMA time-series models to estimate future churn trends from an aggregated historical series.

Forecast quality depends on data quality, the length and frequency of the time series, and the selected model configuration. Forecasts should be evaluated against held-out observations before being used for business decisions.

## 🧪 Dataset

StreamPulse can be used with a customer churn dataset that matches the pipeline's expected schema.

Before running the project:

1. Identify the customer identifier, churn target, and any relevant date column.
2. Verify that the churn target is encoded in a format supported by the pipeline.
3. Check for missing values, duplicate records, and inconsistent data types.
4. Ensure that any time-series date column is parsed correctly.

**Dataset note:** Only use datasets you are authorized to access and share. Do not commit private customer information or other sensitive data.

## ⚠️ Limitations

- Results depend on the quality and representativeness of the input dataset.
- Model performance must be evaluated on appropriate held-out data.
- SHAP/LIME explanations describe model behavior and are not proof of causal relationships.
- Time-series forecasts are estimates and may be unreliable when historical data is limited or patterns change.
- This project is an analytics prototype, not a production-ready customer retention platform.

## 🚀 Future Improvements

- Add model comparison and hyperparameter tuning.
- Include more comprehensive classification metrics and validation reports.
- Improve forecast evaluation and confidence-interval visualization.
- Add data validation and automated preprocessing checks.
- Package the application for easier deployment.
- Add automated tests and continuous integration.

## 👩‍💻 Author

**Himanshi**

GitHub: [@himanshi047272](https://github.com/himanshi047272)

---

*Built as a machine learning and analytics portfolio project.*
