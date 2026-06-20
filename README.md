# 📈 Forecast App

An end-to-end, self-service forecasting platform built with Streamlit. Upload your data, configure campaigns and holidays, and let the app automatically train, tune, and compare **5 forecasting models** to find the best fit for your data.

---

## ✨ Features

- **Flexible data input** — upload CSV or XLSX files directly in the browser
- **Multi-series forecasting** — forecast multiple metrics (e.g. products, regions) at once
- **Campaign & holiday awareness** — Prophet uses campaign windows and e-commerce holidays as regressors; other models do not use these inputs
- **5 models compared automatically**:
  | Model | Best for |
  |---|---|
  | Prophet | Strong seasonal patterns, holidays, campaigns |
  | ARIMA (auto-tuned) | Stable series, statistical baseline |
  | ETS | Weighted recent history, robust fallback |
  | Linear Regression | Interpretable, feature-based |
  | XGBoost | Complex non-linear patterns |
- **Automatic per-series model selection** — each metric independently picks its best model based on your chosen accuracy metric (MAPE, MAE, RMSE, or MSE)
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

1. **Data source** — Upload a CSV or XLSX file
2. **Data preview & schema mapping** — Preview your uploaded data, then tell the app which column is the date and which column(s) to forecast
3. **Seasonality & holidays** — Select campaigns and holidays relevant to your data (used by Prophet only)
4. **Model selection & evaluation settings** — Choose which models to run, your accuracy metric, and forecast date range
5. **Run forecast** — The app trains, tunes, and compares all selected models
6. **Results** — View per-series model rankings, charts, validation accuracy, and download forecasts

---

## 📊 Data Requirements

- At least one **date column** and one **numeric metric column** (the app will flag an error if no numeric columns are detected on upload)
- **Minimum 45 days** of history for Linear Regression / XGBoost to run
- **180+ days recommended** for reliable forecasts across all models

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — UI framework
- [Prophet](https://facebook.github.io/prophet/) — time series forecasting
- [pmdarima](https://alkaline-ml.com/pmdarima/) — auto ARIMA
- [statsmodels](https://www.statsmodels.org/) — ETS / Exponential Smoothing
- [scikit-learn](https://scikit-learn.org/) — Linear Regression
- [XGBoost](https://xgboost.readthedocs.io/) — gradient boosting

---

## ⚠️ Known Limitations

- **Long-horizon accuracy degrades** — all models use recursive (iterative) prediction, meaning each step's error compounds; forecasts beyond 30–60 days should be treated as directional
- **Linear Regression and XGBoost are less reliable past 30 days** — lag and rolling features go stale over time; a dashed reliability line marks the 30-day point on the chart
- **Confidence intervals are approximate** — they use bootstrapped/analytical estimates rather than full simulation, so bands may understate true uncertainty
- **180-day history is recommended, not required** — shorter datasets will still run but may produce noisier models
- **Campaigns and holidays only affect Prophet** — ARIMA, ETS, Linear Regression, and XGBoost do not use these inputs
- **Per-series model selection** — when forecasting multiple metrics, each series independently picks its best model; a single global winner is shown only when all series agree

---

## 📄 License

This project is for portfolio/demonstration purposes.
