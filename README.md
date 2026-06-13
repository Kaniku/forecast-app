# 📈 Forecast App

An end-to-end, self-service forecasting platform built with Streamlit. Upload your data (or query BigQuery directly), configure campaigns and holidays, and let the app automatically train, tune, and compare **5 forecasting models** to find the best fit for your data.

---

## ✨ Features

- **Flexible data input** — upload CSV/XLSX or run a BigQuery SQL query
- **Multi-series forecasting** — forecast multiple metrics (e.g. products, regions) at once
- **Campaign & holiday awareness** — Prophet uses campaign windows and e-commerce holidays as regressors
- **5 models compared automatically**:
  | Model | Best for |
  |---|---|
  | Prophet | Strong seasonal patterns, holidays, campaigns |
  | ARIMA (auto-tuned) | Stable series, statistical baseline |
  | ETS | Weighted recent history, robust fallback |
  | Linear Regression | Interpretable, feature-based |
  | XGBoost | Complex non-linear patterns |
- **Automatic model selection** — picks the best model based on your chosen accuracy metric and thresholds
- **Validation & future forecasts** — see how the model performed historically, then view its forecast for your chosen date range
- **Downloadable results** — export validation and forecast data as CSV

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/your-username/forecast-app.git
cd forecast-app
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> ⚠️ **Python version:** Use **Python 3.11 or 3.12**. Prophet's Stan backend may not yet support newer Python versions reliably.

### 3. Run the app
```bash
streamlit run app.py
```

---

## 📁 Required Files

For campaign and holiday features to work, add these two CSV files **in the same folder as `app.py`**:

### `campaign_periods.csv`
```csv
start,end,label
2024-01-01,2024-01-07,New Year Sale
2024-11-29,2024-12-02,Black Friday
2024-12-20,2024-12-26,Christmas Campaign
```

### `ecommerce_holidays.csv`
```csv
holiday,ds
New Year,2024-01-01
Valentine's Day,2024-02-14
Black Friday,2024-11-29
Cyber Monday,2024-12-02
Christmas,2024-12-25
```

---

## 🧭 How to Use

The app follows a simple step-by-step workflow:

1. **Data source** — Upload a file or connect to BigQuery
2. **Schema mapping** — Tell the app which column is the date and which column(s) to forecast
3. **Seasonality & holidays** — Select campaigns and holidays relevant to your data
4. **Model settings** — Choose which models to run, your accuracy metric, thresholds, and forecast date range
5. **Run forecast** — The app trains, tunes, and compares all selected models
6. **Results** — View charts, validation accuracy, and download forecasts

---

## 📊 Data Requirements

- At least one **date column** and one **numeric metric column**
- **Minimum 45 days** of history for Linear Regression / XGBoost to run
- **180+ days recommended** for reliable forecasts across all models

---

## 🔐 BigQuery Setup (Optional)

If using the BigQuery option, you'll need to authenticate via Google OAuth in the app. For production deployments where multiple users share the app, consider using a **service account** stored in Streamlit Secrets instead of personal credentials — see `docs/bigquery-setup.md` *(if included)*.

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — UI framework
- [Prophet](https://facebook.github.io/prophet/) — time series forecasting
- [pmdarima](https://alkaline-ml.com/pmdarima/) — auto ARIMA
- [statsmodels](https://www.statsmodels.org/) — ETS / Exponential Smoothing
- [scikit-learn](https://scikit-learn.org/) — Linear Regression
- [XGBoost](https://xgboost.readthedocs.io/) — gradient boosting
- [pandas-gbq](https://googleapis.dev/python/pandas-gbq/latest/) — BigQuery integration

---

## ⚠️ Known Limitations

- Linear Regression and XGBoost future forecasts use a simplified recursive approach (continuously refined)
- ETS, Linear Regression, and XGBoost confidence intervals are approximate (±10%)
- 180-day minimum history is recommended but not strictly enforced

---

## 📄 License

This project is for portfolio/demonstration purposes.
