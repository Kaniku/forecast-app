"""
Single-page Streamlit forecasting app â€” layout and workflow shell.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
import pandas_gbq as pd_gbq
import streamlit as st

# --- Constants -----------------------------------------------------------------

MODEL_OPTIONS = ["Prophet", "ARIMA", "ETS", "Linear Regression", "XGBoost"]
METRIC_OPTIONS = ["MAPE", "MAE", "RMSE", "MSE"]


def _init_session_state() -> None:
    defaults: dict[str, Any] = {
        "df_raw": None,
        "df_mapped": None,
        "load_error": None,
        "schema_error": None,
        "date_col": None,
        "metric_col": None,
        "metric_cols": [],
        "campaign_periods": pd.DataFrame(columns=["start", "end", "label"]),
        "campaign_manual": pd.DataFrame(columns=["start", "end", "label"]),
        "selected_campaign_labels": [],
        "campaign_load_error": None,
        "ecommerce_holiday_defaults": pd.DataFrame(columns=["holiday", "ds"]),
        "selected_ecommerce_holidays": [],
        "ecommerce_holiday_manual": pd.DataFrame(columns=["holiday", "ds"]),
        "ecommerce_holiday_load_error": None,
        "ecommerce_holidays": pd.DataFrame(columns=["holiday", "ds"]),
        "selected_models": list(MODEL_OPTIONS),
        "selected_metric": "MAPE",
        "auto_select_all_models": True,
        "forecast_start_date": None,
        "forecast_end_date": None,
        "forecast_date_error": None,
        "forecast_run_done": False,
        "results_placeholder": None,
        "bq_credentials": None,
        "bq_authenticated": False,
        "campaign_manual_list": [],
        "holiday_manual_list": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _load_uploaded_file(uploaded: Any) -> pd.DataFrame:
    name = (uploaded.name or "").lower()
    raw = uploaded.getvalue()
    if not raw:
        raise ValueError("The file is empty.")
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(raw))
    raise ValueError("Unsupported format. Please upload a .csv or .xlsx file.")


def _load_bigquery(sql: str, project_id: str | None, credentials: Any = None) -> pd.DataFrame:
    if not sql.strip():
        raise ValueError("Enter a SQL query.")

    proj = project_id.strip() if project_id and str(project_id).strip() else "tbdproject-334912"
    
    configuration = {
        'query': {
            "useQueryCache": True
        }
    }

    try:
        # pandas-gbq will use credentials if provided, else trigger interactive login
        df = pd_gbq.read_gbq(
            sql, 
            project_id=proj, 
            dialect='standard', 
            configuration=configuration, 
            progress_bar_type='tqdm',
            credentials=credentials
        )
        return df
    except Exception as e:
        raise RuntimeError(f"BigQuery request failed: {e}") from e


def _dry_run_bigquery(sql: str, project_id: str, credentials: Any = None) -> str:
    """Estimates the number of bytes processed by the query."""
    try:
        from google.cloud import bigquery
        client = bigquery.Client(project=project_id, credentials=credentials)
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        query_job = client.query(sql, job_config=job_config)
        bytes_processed = query_job.total_bytes_processed
        
        if bytes_processed == 0:
            return "Estimated: 0 B (Cached or empty)"
        elif bytes_processed < 1024:
            return f"Estimated: {bytes_processed} B"
        elif bytes_processed < 1024**2:
            return f"Estimated: {bytes_processed / 1024:.2f} KB"
        elif bytes_processed < 1024**3:
            return f"Estimated: {bytes_processed / 1024**2:.2f} MB"
        else:
            return f"Estimated: {bytes_processed / 1024**3:.2f} GB"
    except Exception as e:
        return f"Query check failed: {e}"


def _normalize_ds_to_calendar_date(df: pd.DataFrame) -> pd.DataFrame:
    """Parse `ds` as datetime and drop time-of-day (calendar date at midnight).

    Accepts plain dates or datetimes; datetimes are converted to the same calendar day.
    """
    out = df.copy()
    parsed = pd.to_datetime(out["ds"], errors="coerce")
    out["ds"] = parsed.dt.normalize()
    return out


def _load_campaign_periods_csv() -> pd.DataFrame:
    """Load campaign periods from campaign_periods.csv in the app folder."""
    csv_path = Path(__file__).resolve().parent / "campaign_periods.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Campaign backend file not found: {csv_path}. Create campaign_periods.csv next to app.py "
            "with columns: start, end, label."
        )

    df = pd.read_csv(csv_path, sep=None, engine="python")
    required = {"start", "end", "label"}
    missing = required.difference(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(
            f"campaign_periods.csv is missing required columns: {missing_cols}. "
            "Expected columns: start, end, label."
        )

    out = df.copy()
    out["start"] = pd.to_datetime(out["start"], errors="coerce", dayfirst=True).dt.normalize()
    out["end"] = pd.to_datetime(out["end"], errors="coerce", dayfirst=True).dt.normalize()
    out["label"] = out["label"].astype(str).str.strip()
    out = out.dropna(subset=["start", "end"])
    out = out[out["label"] != ""]
    out = out.sort_values(["start", "end", "label"]).reset_index(drop=True)
    return out


def _load_ecommerce_holidays_csv() -> pd.DataFrame:
    """Load default e-commerce holidays from ecommerce_holidays.csv in app folder."""
    csv_path = Path(__file__).resolve().parent / "ecommerce_holidays.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"E-commerce holidays backend file not found: {csv_path}. Create ecommerce_holidays.csv next to app.py "
            "with columns: holiday, ds."
        )

    df = pd.read_csv(csv_path, sep=None, engine="python")
    required = {"holiday", "ds"}
    missing = required.difference(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(
            f"ecommerce_holidays.csv is missing required columns: {missing_cols}. "
            "Expected columns: holiday, ds."
        )

    out = df.copy()
    out["holiday"] = out["holiday"].astype(str).str.strip()
    out["ds"] = pd.to_datetime(out["ds"], errors="coerce", dayfirst=True).dt.normalize()
    out = out.dropna(subset=["holiday", "ds"])
    out = out[out["holiday"] != ""]
    out = out.drop_duplicates(subset=["holiday", "ds"])
    out = out.sort_values(["ds", "holiday"]).reset_index(drop=True)
    return out


def _validate_mapped_df(df: pd.DataFrame) -> tuple[bool, str]:
    if df.empty:
        return False, "No data found after processing. Please check your file and try again."
    if "ds" not in df.columns or "y" not in df.columns:
        return False, "Your data needs a date column and a value column. Please go back to Step 2 and map your columns."
    ds = pd.to_datetime(df["ds"], errors="coerce")
    if ds.isna().all():
        return False, "The selected date column doesn't contain valid dates. Please choose a different column in Step 2."
    y = pd.to_numeric(df["y"], errors="coerce")
    if y.notna().sum() == 0:
        return False, "The selected metric column doesn't contain numbers. Please choose a different column in Step 2."
    na_frac = y.isna().mean()
    if na_frac > 0.2:
        return False, f"Too much missing data found ({na_frac:.0%} of rows are empty). Please check your data and try again."
    if "series" in df.columns:
        counts = df.groupby("series")["ds"].nunique()
        min_days = counts.min()
        if min_days < 180:
            return (
                False,
                f"Not enough history — your data only has {min_days} days. "
                "We need at least 180 days for accurate forecasts."
            )
    
    return True, ""


def _build_campaign_days(
    campaign_periods: pd.DataFrame, selected_campaign_labels: list[str] | None
) -> pd.DatetimeIndex:
    """Expand campaign periods into a set of daily dates for regressor."""
    if campaign_periods is None or campaign_periods.empty:
        return pd.DatetimeIndex([], name="ds")

    df = campaign_periods.copy()
    if selected_campaign_labels is not None:
        df = df[df["label"].isin(selected_campaign_labels)]
    if df.empty:
        return pd.DatetimeIndex([], name="ds")

    all_days: list[pd.DatetimeIndex] = []
    for _, row in df.iterrows():
        start = row.get("start")
        end = row.get("end")
        if pd.isna(start) or pd.isna(end):
            continue
        start_ts = pd.to_datetime(start).normalize()
        end_ts = pd.to_datetime(end).normalize()
        if end_ts < start_ts:
            continue
        all_days.append(pd.date_range(start=start_ts, end=end_ts, freq="D"))

    if not all_days:
        return pd.DatetimeIndex([], name="ds")
    days = pd.DatetimeIndex(np.unique(np.concatenate([d.values for d in all_days]))).normalize()
    return days


def _compute_metric(metric_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if metric_name.upper() == "MAE":
        return float(np.mean(np.abs(y_true - y_pred)))
    if metric_name.upper() == "RMSE":
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    if metric_name.upper() == "MSE":
        return float(np.mean((y_true - y_pred) ** 2))
    if metric_name.upper() == "MAPE":
        zero_mask = y_true == 0
        zero_frac = float(np.mean(zero_mask)) if len(y_true) else 0.0
        if zero_frac > 0.1:
            warnings.warn(
                "MAPE may be unreliable because more than 10% of actual values are zero. "
                "Consider using MAE or RMSE instead.",
                RuntimeWarning,
                stacklevel=2,
            )
        denom = np.where(y_true == 0, np.nan, y_true)
        pct = np.abs((y_true - y_pred) / denom)
        val = np.nanmean(pct)
        return float(val) if np.isfinite(val) else float("inf")


def _ets_forecast_intervals(
    model_fit: Any,
    forecast_y: Any,
    forecast_horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ETS forecast intervals using simulation, with a silent fallback."""
    forecast_array = np.asarray(forecast_y, dtype=float)

    try:
        simulation = model_fit.simulate(
            nsimulations=forecast_horizon,
            repetitions=100,
            error="add",
        )
        simulation_array = np.asarray(simulation, dtype=float)
        yhat_lower = np.percentile(simulation_array, 5, axis=1)
        yhat_upper = np.percentile(simulation_array, 95, axis=1)
        return yhat_lower, yhat_upper
    except Exception:
        return forecast_array * 0.9, forecast_array * 1.1

    raise ValueError(f"Unsupported metric: {metric_name}")


def _fit_predict_prophet_validation(
    df_series: pd.DataFrame,
    campaign_days: pd.DatetimeIndex,
    holidays_df: pd.DataFrame | None,
    *,
    metric_name: str,
    changepoint_prior_scale_val: float,
    seasonality_prior_scale_val: float,
    seasonality_mode: str,
    growth_type: str = "flat",
) -> tuple[float, pd.DataFrame, Any]:
    """Fit Prophet on train portion and return validation forecast."""
    try:
        from prophet import Prophet  # type: ignore
    except ImportError as e:
        raise RuntimeError("Prophet is not installed. Run `pip install prophet`.") from e

    df_series = df_series.sort_values("ds").reset_index(drop=True).copy()
    df_series["ds"] = pd.to_datetime(df_series["ds"], errors="coerce").dt.normalize()
    df_series["y"] = pd.to_numeric(df_series["y"], errors="coerce").astype(float).clip(lower=0)
    df_series = df_series.dropna(subset=["ds", "y"])
    if df_series.empty:
        return float("inf"), pd.DataFrame(), None

    # Choose validation horizon based on series length.
    n = len(df_series)
    validation_days = int(min(30, max(7, n * 0.2)))
    validation_days = max(1, validation_days)
    if n <= validation_days + 2:
        return float("inf"), pd.DataFrame(), None

    train_df = df_series.iloc[: n - validation_days]
    val_df = df_series.iloc[n - validation_days :]

    train_prophet = train_df[["ds", "y"]].copy()
    val_prophet = val_df[["ds", "y"]].copy()
    train_prophet["is_campaign"] = train_prophet["ds"].isin(campaign_days).astype(int)
    val_prophet["is_campaign"] = val_prophet["ds"].isin(campaign_days).astype(int)

    holidays_arg = None
    if holidays_df is not None and not holidays_df.empty:
        # Prophet expects at least ['ds', 'holiday']
        needed = {"ds", "holiday"}
        if needed.issubset(set(holidays_df.columns)):
            holidays_arg = holidays_df[["ds", "holiday"]].copy()
        else:
            raise ValueError("holidays_df must contain columns: 'ds' and 'holiday'.")

    model = Prophet(
        growth=growth_type,
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale_val,
        seasonality_prior_scale=seasonality_prior_scale_val,
        holidays=holidays_arg,
    )
    model.add_regressor("is_campaign")
    model.add_seasonality(name="monthly_trend", period=30, fourier_order=3)

    model.fit(train_prophet)

    future_val = val_prophet[["ds", "is_campaign"]].copy()
    forecast = model.predict(future_val)

    forecast_val = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    forecast_val["yhat"] = forecast_val["yhat"].clip(lower=0)
    forecast_val["yhat_lower"] = forecast_val["yhat_lower"].clip(lower=0)
    forecast_val["yhat_upper"] = forecast_val["yhat_upper"].clip(lower=0)

    y_true = val_df["y"].to_numpy()
    y_pred = forecast_val["yhat"].to_numpy()
    metric = _compute_metric(metric_name, y_true, y_pred)
    return metric, forecast_val, model





def _fit_predict_arima_validation(
    df_series: pd.DataFrame,
    *,
    metric_name: str,
) -> tuple[float, pd.DataFrame, Any]:
    """Fit ARIMA on train portion and return validation forecast."""
    df_series = df_series.sort_values("ds").reset_index(drop=True).copy()
    df_series["ds"] = pd.to_datetime(df_series["ds"], errors="coerce").dt.normalize()
    df_series["y"] = pd.to_numeric(df_series["y"], errors="coerce").astype(float).clip(lower=0)
    df_series = df_series.dropna(subset=["ds", "y"])
    if df_series.empty:
        return float("inf"), pd.DataFrame(), None

    # Choose validation horizon based on series length.
    n = len(df_series)
    validation_days = int(min(30, max(7, n * 0.2)))
    validation_days = max(1, validation_days)
    if n <= validation_days + 2:
        return float("inf"), pd.DataFrame(), None

    train_y = df_series.iloc[: n - validation_days]["y"].to_numpy()
    val_y = df_series.iloc[n - validation_days :]["y"].to_numpy()
    val_ds = df_series.iloc[n - validation_days :]["ds"].to_numpy()

    try:
        import pmdarima as pm  # type: ignore
    except ImportError as e:
        raise RuntimeError("pmdarima is not installed. Run `pip install pmdarima`.") from e

    try:
        model = pm.auto_arima(
            train_y,
            start_p=0, max_p=3,
            start_q=0, max_q=3,
            d=None,
            seasonal=False,
            stepwise=True,
            information_criterion='aic',
            suppress_warnings=True,
            error_action='ignore'
        )

        forecast_y, conf_int = model.predict(n_periods=validation_days, return_conf_int=True)
        metric = _compute_metric(metric_name, val_y, forecast_y)
        
        forecast_val = pd.DataFrame({
            "ds": val_ds,
            "yhat": forecast_y,
            "yhat_lower": conf_int[:, 0],
            "yhat_upper": conf_int[:, 1],
        })
    except Exception:
        return float("inf"), pd.DataFrame(), None

    forecast_val["yhat"] = forecast_val["yhat"].clip(lower=0)
    forecast_val["yhat_lower"] = forecast_val["yhat_lower"].clip(lower=0)
    forecast_val["yhat_upper"] = forecast_val["yhat_upper"].clip(lower=0)

    return metric, forecast_val, model


def _tune_ets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    metric_name: str,
) -> tuple[Any, dict[str, Any], float, pd.DataFrame]:
    """Perform grid search for ETS (ExponentialSmoothing) hyperparameters."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing  # type: ignore

    train_y = train_df["y"].to_numpy()
    val_y = val_df["y"].to_numpy()
    val_ds = val_df["ds"].to_numpy()
    n_val = len(val_df)

    trends = ["add", "mul", None]
    seasonals = ["add", "mul", None]
    seasonal_periods_list = [7, 12, 30]

    best_metric = float("inf")
    best_model = None
    best_params = {}
    best_forecast = pd.DataFrame()

    for t in trends:
        for s in seasonals:
            # If seasonal is None, seasonal_periods doesn't matter, use a dummy loop
            periods = seasonal_periods_list if s is not None else [None]
            for p in periods:
                try:
                    # Mul trend/seasonal requires all positive values
                    if (t == "mul" or s == "mul") and (train_y <= 0).any():
                        continue
                    
                    # Not enough data for seasonal period
                    if s is not None and p is not None and len(train_y) < 2 * p:
                        continue

                    # Try with "estimated" first
                    try:
                        model_fit = ExponentialSmoothing(
                            train_y,
                            trend=t,
                            seasonal=s,
                            seasonal_periods=p,
                            initialization_method="estimated",
                        ).fit(disp=False)
                    except:
                        # Fallback to "heuristic" initialization if "estimated" fails
                        model_fit = ExponentialSmoothing(
                            train_y,
                            trend=t,
                            seasonal=s,
                            seasonal_periods=p,
                            initialization_method="heuristic",
                        ).fit(disp=False)

                    forecast_y = model_fit.forecast(n_val)
                    current_metric = _compute_metric(metric_name, val_y, forecast_y)

                    if np.isfinite(current_metric) and current_metric < best_metric:
                        best_metric = current_metric
                        best_model = model_fit
                        best_params = {"trend": t, "seasonal": s, "seasonal_periods": p}
                        yhat_lower, yhat_upper = _ets_forecast_intervals(model_fit, forecast_y, n_val)
                        best_forecast = pd.DataFrame(
                            {
                                "ds": val_ds,
                                "yhat": forecast_y,
                                "yhat_lower": yhat_lower,
                                "yhat_upper": yhat_upper,
                            }
                        )
                except:  # noqa: E722 â€” ignore convergence/fitting errors during grid search
                    continue

    # Fallback: Simple Exponential Smoothing if nothing worked
    if best_model is None:
        try:
            # Simple Smoothing is most robust
            model_fit = ExponentialSmoothing(
                train_y,
                trend=None,
                seasonal=None,
                initialization_method="heuristic",
            ).fit(disp=False)
            best_model = model_fit
            best_params = {"trend": None, "seasonal": None, "seasonal_periods": None}
            forecast_y = model_fit.forecast(n_val)
            yhat_lower, yhat_upper = _ets_forecast_intervals(model_fit, forecast_y, n_val)
            best_metric = _compute_metric(metric_name, val_y, forecast_y)
            best_forecast = pd.DataFrame(
                {
                    "ds": val_ds,
                    "yhat": forecast_y,
                    "yhat_lower": yhat_lower,
                    "yhat_upper": yhat_upper,
                }
            )
        except Exception: # noqa: E722
            pass

    # Final Fallback: Naive (Mean) if even SES failed
    if best_model is None:
        mean_val = float(np.mean(train_y))
        best_params = {"trend": None, "seasonal": None, "seasonal_periods": None, "is_naive": True}
        forecast_y = np.full(n_val, mean_val)
        best_metric = _compute_metric(metric_name, val_y, forecast_y)
        best_model = "naive"
        best_forecast = pd.DataFrame({
            "ds": val_ds,
            "yhat": forecast_y,
            "yhat_lower": forecast_y * 0.8,
            "yhat_upper": forecast_y * 1.2,
        })

    return best_model, best_params, best_metric, best_forecast


def _fit_predict_ets_validation(
    df_series: pd.DataFrame,
    *,
    metric_name: str,
) -> tuple[float, pd.DataFrame, Any, dict[str, Any]]:
    """Fit ETS on train portion and return validation forecast."""
    df_series = df_series.sort_values("ds").reset_index(drop=True).copy()
    df_series["ds"] = pd.to_datetime(df_series["ds"], errors="coerce").dt.normalize()
    df_series["y"] = pd.to_numeric(df_series["y"], errors="coerce").astype(float).clip(lower=0)
    df_series = df_series.dropna(subset=["ds", "y"])
    
    if df_series.empty:
        return float("inf"), pd.DataFrame(), None, {}

    n = len(df_series)
    validation_days = int(min(30, max(7, n * 0.2)))
    validation_days = max(1, validation_days)
    if n <= validation_days + 2:  # Reduced requirement from +5 to +2
        return float("inf"), pd.DataFrame(), None, {}

    train_df = df_series.iloc[: n - validation_days]
    val_df = df_series.iloc[n - validation_days :]

    model, params, metric, forecast_val = _tune_ets(train_df, val_df, metric_name)
    
    if model is None:
        return float("inf"), pd.DataFrame(), None, {}

    forecast_val["yhat"] = forecast_val["yhat"].clip(lower=0)
    forecast_val["yhat_lower"] = forecast_val["yhat_lower"].clip(lower=0)
    forecast_val["yhat_upper"] = forecast_val["yhat_upper"].clip(lower=0)

    return metric, forecast_val, model, params


def train_linear_regression(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    metric_name: str,
) -> tuple[Any, float, pd.DataFrame]:
    """Train Linear Regression with time-series features and return results."""
    from sklearn.linear_model import LinearRegression  # type: ignore

    # Combine for feature engineering (we need lags/rolling from train to predict val)
    df = pd.concat([train_df, val_df], ignore_index=True).copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)

    # 1. Feature Engineering
    df["trend"] = range(len(df))
    df["day_of_week"] = df["ds"].dt.dayofweek
    df["month"] = df["ds"].dt.month
    
    # Lags
    df["lag_1"] = df["y"].shift(1)
    df["lag_7"] = df["y"].shift(7)
    df["lag_30"] = df["y"].shift(30)
    
    # Rolling means
    df["rolling_mean_7"] = df["y"].shift(1).rolling(window=7).mean()
    df["rolling_mean_30"] = df["y"].shift(1).rolling(window=30).mean()
    
    # Drop rows where we don't have enough history for lags/rolling
    df_clean = df.dropna(subset=["lag_30", "rolling_mean_30"]).copy()
    
    if df_clean.empty:
        return None, float("inf"), pd.DataFrame()

    # Split back using dates
    val_dates = val_df["ds"].unique()
    train_final = df_clean[~df_clean["ds"].isin(val_dates)]
    val_final = df_clean[df_clean["ds"].isin(val_dates)]
    
    if train_final.empty or val_final.empty:
        return None, float("inf"), pd.DataFrame()

    features = [
        "trend", "day_of_week", "month", 
        "lag_1", "lag_7", "lag_30", 
        "rolling_mean_7", "rolling_mean_30"
    ]
    
    X_train = train_final[features]
    y_train = train_final["y"]
    X_val = val_final[features]
    y_val = val_final["y"]

    # 2. Train
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 3. Predict
    y_pred = model.predict(X_val)
    y_pred = np.clip(y_pred, a_min=0, a_max=None) # Logical lower bound

    # 4. Prepare results
    forecast_val = pd.DataFrame({
        "ds": val_final["ds"].to_numpy(),
        "yhat": y_pred,
        "yhat_lower": y_pred * 0.9, # Simple placeholder for CI
        "yhat_upper": y_pred * 1.1, # Simple placeholder for CI
    })

    metric_score = _compute_metric(metric_name, y_val.to_numpy(), y_pred)
    
    return model, metric_score, forecast_val


def _fit_predict_lr_validation(
    df_series: pd.DataFrame,
    *,
    metric_name: str,
) -> tuple[float, pd.DataFrame, Any]:
    """Prepare data and call train_linear_regression."""
    df_series = df_series.sort_values("ds").reset_index(drop=True).copy()
    df_series["ds"] = pd.to_datetime(df_series["ds"], errors="coerce").dt.normalize()
    df_series["y"] = pd.to_numeric(df_series["y"], errors="coerce").astype(float).clip(lower=0)
    df_series = df_series.dropna(subset=["ds", "y"])
    
    if df_series.empty:
        return float("inf"), pd.DataFrame(), None

    n = len(df_series)
    # We need at least 30 days of history for the features + some training data
    if n < 45:
        return float("inf"), pd.DataFrame(), None

    validation_days = int(min(30, max(7, n * 0.2)))
    validation_days = max(1, validation_days)
    
    train_df = df_series.iloc[: n - validation_days]
    val_df = df_series.iloc[n - validation_days :]

    model, metric, forecast_val = train_linear_regression(train_df, val_df, metric_name)
    if model is None:
        return float("inf"), pd.DataFrame(), None
        
    return metric, forecast_val, model


def train_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    metric_name: str,
    n_trials: int = 20,
) -> tuple[Any, float, pd.DataFrame]:
    """Train XGBoost Regressor with time-series features and return results."""
    try:
        from xgboost import XGBRegressor  # type: ignore
        from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
    except ImportError as e:
        raise RuntimeError("xgboost or scikit-learn is not installed. Run `pip install xgboost scikit-learn`.") from e

    # Combine for feature engineering
    df = pd.concat([train_df, val_df], ignore_index=True).copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)

    # 1. Feature Engineering (same logic as Linear Regression)
    df["trend"] = range(len(df))
    df["day_of_week"] = df["ds"].dt.dayofweek
    df["month"] = df["ds"].dt.month
    
    # Lags
    df["lag_1"] = df["y"].shift(1)
    df["lag_7"] = df["y"].shift(7)
    df["lag_30"] = df["y"].shift(30)
    
    # Rolling means
    df["rolling_mean_7"] = df["y"].shift(1).rolling(window=7).mean()
    df["rolling_mean_30"] = df["y"].shift(1).rolling(window=30).mean()
    
    # Drop rows where we don't have enough history
    df_clean = df.dropna(subset=["lag_30", "rolling_mean_30"]).copy()
    
    if df_clean.empty:
        return None, float("inf"), pd.DataFrame()

    # Split back
    val_dates = val_df["ds"].unique()
    train_final = df_clean[~df_clean["ds"].isin(val_dates)]
    val_final = df_clean[df_clean["ds"].isin(val_dates)]
    
    if train_final.empty or val_final.empty:
        return None, float("inf"), pd.DataFrame()

    features = [
        "trend", "day_of_week", "month", 
        "lag_1", "lag_7", "lag_30", 
        "rolling_mean_7", "rolling_mean_30"
    ]
    
    X_train = train_final[features]
    y_train = train_final["y"]
    X_val = val_final[features]
    y_val = val_final["y"]

    # 2. Randomized Search CV for XGBoost
    param_dist = {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 10],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    }
    
    tscv = TimeSeriesSplit(n_splits=3)
    xgb = XGBRegressor(objective="reg:squarederror", random_state=42)
    
    random_search = RandomizedSearchCV(
        estimator=xgb, 
        param_distributions=param_dist, 
        n_iter=n_trials, 
        cv=tscv, 
        scoring='neg_mean_squared_error', 
        n_jobs=-1, 
        random_state=42,
        verbose=0
    )
    
    random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_

    # 3. Predict
    y_pred = best_model.predict(X_val)
    y_pred = np.clip(y_pred, a_min=0, a_max=None)

    # 4. Prepare results
    forecast_val = pd.DataFrame({
        "ds": val_final["ds"].to_numpy(),
        "yhat": y_pred,
        "yhat_lower": y_pred * 0.9,
        "yhat_upper": y_pred * 1.1,
    })

    metric_score = _compute_metric(metric_name, y_val.to_numpy(), y_pred)
    
    return best_model, metric_score, forecast_val


def _fit_predict_xgboost_validation(
    df_series: pd.DataFrame,
    *,
    metric_name: str,
) -> tuple[float, pd.DataFrame, Any]:
    """Prepare data and call train_xgboost."""
    df_series = df_series.sort_values("ds").reset_index(drop=True).copy()
    df_series["ds"] = pd.to_datetime(df_series["ds"], errors="coerce").dt.normalize()
    df_series["y"] = pd.to_numeric(df_series["y"], errors="coerce").astype(float).clip(lower=0)
    df_series = df_series.dropna(subset=["ds", "y"])
    
    if df_series.empty:
        return float("inf"), pd.DataFrame(), None

    n = len(df_series)
    if n < 45: # Need 30 for features + some training data
        return float("inf"), pd.DataFrame(), None

    validation_days = int(min(30, max(7, n * 0.2)))
    validation_days = max(1, validation_days)
    
    train_df = df_series.iloc[: n - validation_days]
    val_df = df_series.iloc[n - validation_days :]

    model, metric, forecast_val = train_xgboost(train_df, val_df, metric_name)
    if model is None:
        return float("inf"), pd.DataFrame(), None
        
    return metric, forecast_val, model


def _tune_prophet_grid_search(
    df_mapped: pd.DataFrame,
    campaign_days: pd.DatetimeIndex,
    holidays_df: pd.DataFrame | None,
    *,
    metric_name: str,
    growth_strategy: str = "flat",
) -> tuple[pd.DataFrame, dict[str, Any], float]:
    """Replaces manual search with a grid search for Prophet."""
    series_names = sorted(df_mapped["series"].dropna().unique().tolist())

    # Grid search parameters
    changepoint_prior_scales = [0.01, 0.05, 0.1, 0.5]
    seasonality_prior_scales = [1, 5, 10]
    seasonality_modes = ['additive', 'multiplicative']
    
    # Growth types to try based on strategy
    if growth_strategy == "auto":
        growth_types = ["linear", "flat"]
    else:
        growth_types = [growth_strategy]

    best_score = float("inf")
    best_params = {}
    best_validation_forecast = pd.DataFrame()

    for cp_scale in changepoint_prior_scales:
        for sp_scale in seasonality_prior_scales:
            for s_mode in seasonality_modes:
                for g_type in growth_types:
                    current_params = {
                        "changepoint_prior_scale": cp_scale,
                        "seasonality_prior_scale": sp_scale,
                        "seasonality_mode": s_mode,
                        "growth": g_type,
                    }

                    try:
                        metrics = []
                        forecast_parts = []
                        for s in series_names:
                            df_series = df_mapped[df_mapped["series"] == s]
                            metric, forecast_val, _model = _fit_predict_prophet_validation(
                                df_series,
                                campaign_days,
                                holidays_df,
                                metric_name=metric_name,
                                changepoint_prior_scale_val=cp_scale,
                                seasonality_prior_scale_val=sp_scale,
                                seasonality_mode=s_mode,
                                growth_type=g_type,
                            )
                            if np.isfinite(metric):
                                metrics.append(metric)
                                forecast_val["series"] = s
                                forecast_parts.append(forecast_val)
                            else:
                                metrics.append(float("inf"))
                        
                        current_score = np.mean(metrics) if metrics else float("inf")

                        if current_score < best_score:
                            best_score = current_score
                            best_params = current_params
                            if forecast_parts:
                                best_validation_forecast = pd.concat(forecast_parts, ignore_index=True)

                    except Exception:  # Broad exception to catch any failure during a trial
                        continue

    return best_validation_forecast, best_params, best_score


def _fit_prophet_full(
    df_mapped: pd.DataFrame,
    campaign_days: pd.DatetimeIndex,
    holidays_df: pd.DataFrame | None,
    best_params: dict[str, Any],
    forecast_start_date: Any,
    forecast_end_date: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit Prophet on full data and forecast future."""
    from prophet import Prophet # type: ignore
    
    series_names = sorted(df_mapped["series"].dropna().unique().tolist())
    forecast_parts = []
    trained_models = {}

    last_data_date = pd.to_datetime(df_mapped["ds"]).max()
    forecast_horizon = (pd.to_datetime(forecast_end_date) - last_data_date).days
    # Horizon must be at least 1
    forecast_horizon = max(1, forecast_horizon)

    for s in series_names:
        df_series = df_mapped[df_mapped["series"] == s].sort_values("ds").reset_index(drop=True)
        df_series["is_campaign"] = df_series["ds"].isin(campaign_days).astype(int)

        holidays_arg = None
        if holidays_df is not None and not holidays_df.empty:
            holidays_arg = holidays_df[["ds", "holiday"]].copy()

        model = Prophet(
            growth=best_params.get("growth", "flat"),
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode=best_params["seasonality_mode"],
            changepoint_prior_scale=best_params["changepoint_prior_scale"],
            seasonality_prior_scale=best_params["seasonality_prior_scale"],
            holidays=holidays_arg,
        )
        model.add_regressor("is_campaign")
        model.add_seasonality(name="monthly_trend", period=30, fourier_order=3)

        model.fit(df_series[["ds", "y", "is_campaign"]])
        trained_models[s] = model

        # Future dataframe
        future = model.make_future_dataframe(periods=forecast_horizon, freq="D")
        future["is_campaign"] = future["ds"].isin(campaign_days).astype(int)
        
        forecast = model.predict(future)
        
        # Filter for requested range
        start_ts = pd.to_datetime(forecast_start_date)
        end_ts = pd.to_datetime(forecast_end_date)
        forecast_filtered = forecast[(forecast["ds"] >= start_ts) & (forecast["ds"] <= end_ts)].copy()
        forecast_filtered["series"] = s
        forecast_parts.append(forecast_filtered[["ds", "series", "yhat", "yhat_lower", "yhat_upper"]])

    if not forecast_parts:
        return pd.DataFrame(columns=["ds", "series", "yhat", "yhat_lower", "yhat_upper"]), trained_models

    full_forecast = pd.concat(forecast_parts, ignore_index=True)
    return full_forecast, trained_models


def _fit_arima_full(
    df_mapped: pd.DataFrame,
    best_params_per_series: dict[str, dict[str, int]],
    forecast_start_date: Any,
    forecast_end_date: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit ARIMA on full data per series and forecast future."""
    import pmdarima as pm # type: ignore
    
    series_names = sorted(df_mapped["series"].dropna().unique().tolist())
    forecast_parts = []
    trained_models = {}

    last_data_date = pd.to_datetime(df_mapped["ds"]).max()
    forecast_horizon = (pd.to_datetime(forecast_end_date) - last_data_date).days
    forecast_horizon = max(1, forecast_horizon)

    for s in series_names:
        df_series = df_mapped[df_mapped["series"] == s].sort_values("ds").reset_index(drop=True)
        y = df_series["y"].to_numpy()
        params = best_params_per_series.get(s, {"p": 1, "d": 1, "q": 1})
        
        model = pm.ARIMA(order=(params["p"], params["d"], params["q"]), suppress_warnings=True)
        model.fit(y)
        trained_models[s] = model

        forecast_y, conf_int = model.predict(n_periods=forecast_horizon, return_conf_int=True)
        
        future_dates = pd.date_range(start=last_data_date + pd.Timedelta(days=1), periods=forecast_horizon, freq="D")
        
        forecast_df = pd.DataFrame({
            "ds": future_dates,
            "series": s,
            "yhat": forecast_y,
            "yhat_lower": conf_int[:, 0],
            "yhat_upper": conf_int[:, 1],
        })

        # Filter for requested range
        start_ts = pd.to_datetime(forecast_start_date)
        end_ts = pd.to_datetime(forecast_end_date)
        forecast_filtered = forecast_df[(forecast_df["ds"] >= start_ts) & (forecast_df["ds"] <= end_ts)].copy()
        forecast_parts.append(forecast_filtered)

    if not forecast_parts:
        return pd.DataFrame(columns=["ds", "series", "yhat", "yhat_lower", "yhat_upper"]), trained_models

    full_forecast = pd.concat(forecast_parts, ignore_index=True)
    return full_forecast, trained_models


def _fit_ets_full(
    df_mapped: pd.DataFrame,
    best_params_per_series: dict[str, dict[str, Any]],
    forecast_start_date: Any,
    forecast_end_date: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit ETS on full data per series and forecast future."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing # type: ignore
    
    series_names = sorted(df_mapped["series"].dropna().unique().tolist())
    forecast_parts = []
    trained_models = {}

    last_data_date = pd.to_datetime(df_mapped["ds"]).max()
    forecast_horizon = (pd.to_datetime(forecast_end_date) - last_data_date).days
    forecast_horizon = max(1, forecast_horizon)

    for s in series_names:
        df_series = df_mapped[df_mapped["series"] == s].sort_values("ds").reset_index(drop=True)
        y = df_series["y"].to_numpy()
        params = best_params_per_series.get(s, {"trend": None, "seasonal": None, "seasonal_periods": None})
        
        if params.get("is_naive"):
            # Use Naive forecast (mean)
            mean_val = float(np.mean(y))
            forecast_y = np.full(forecast_horizon, mean_val)
            model = "naive"
        else:
            try:
                model = ExponentialSmoothing(
                    y,
                    trend=params["trend"],
                    seasonal=params["seasonal"],
                    seasonal_periods=params["seasonal_periods"],
                    initialization_method="estimated",
                ).fit(disp=False)
            except:
                model = ExponentialSmoothing(
                    y,
                    trend=params["trend"],
                    seasonal=params["seasonal"],
                    seasonal_periods=params["seasonal_periods"],
                    initialization_method="heuristic",
                ).fit(disp=False)
            
            forecast_y = model.forecast(forecast_horizon)
        
        trained_models[s] = model

        future_dates = pd.date_range(start=last_data_date + pd.Timedelta(days=1), periods=forecast_horizon, freq="D")
        if params.get("is_naive"):
            yhat_lower = np.asarray(forecast_y, dtype=float) * 0.9
            yhat_upper = np.asarray(forecast_y, dtype=float) * 1.1
        else:
            yhat_lower, yhat_upper = _ets_forecast_intervals(model, forecast_y, forecast_horizon)
        
        forecast_df = pd.DataFrame({
            "ds": future_dates,
            "series": s,
            "yhat": forecast_y,
            "yhat_lower": yhat_lower,
            "yhat_upper": yhat_upper,
        })

        # Filter for requested range
        start_ts = pd.to_datetime(forecast_start_date)
        end_ts = pd.to_datetime(forecast_end_date)
        forecast_filtered = forecast_df[(forecast_df["ds"] >= start_ts) & (forecast_df["ds"] <= end_ts)].copy()
        forecast_parts.append(forecast_filtered)

    if not forecast_parts:
        return pd.DataFrame(columns=["ds", "series", "yhat", "yhat_lower", "yhat_upper"]), trained_models

    full_forecast = pd.concat(forecast_parts, ignore_index=True)
    return full_forecast, trained_models


def _fit_ml_full(
    df_mapped: pd.DataFrame,
    best_models_per_series: dict[str, Any],
    forecast_start_date: Any,
    forecast_end_date: Any,
) -> pd.DataFrame:
    """ML models (LR, XGB) recursive forecasting."""
    series_names = sorted(df_mapped["series"].dropna().unique().tolist())
    forecast_parts = []

    last_data_date = pd.to_datetime(df_mapped["ds"]).max()
    forecast_horizon = (pd.to_datetime(forecast_end_date) - last_data_date).days
    forecast_horizon = max(1, forecast_horizon)

    for s in series_names:
        df_series = df_mapped[df_mapped["series"] == s].sort_values("ds").reset_index(drop=True)
        future_dates = pd.date_range(start=last_data_date + pd.Timedelta(days=1), periods=forecast_horizon, freq="D")
        model = best_models_per_series.get(s)
        known_values = df_series["y"].tolist()

        if model is None:
            # Fallback to mean forecast if no trained model exists for this series.
            fallback_val = float(np.mean(known_values)) if known_values else 0.0
            forecast_df = pd.DataFrame({
                "ds": future_dates,
                "series": s,
                "yhat": [fallback_val] * forecast_horizon,
                "yhat_lower": [fallback_val * 0.9] * forecast_horizon,
                "yhat_upper": [fallback_val * 1.1] * forecast_horizon,
            })
            start_ts = pd.to_datetime(forecast_start_date)
            end_ts = pd.to_datetime(forecast_end_date)
            forecast_filtered = forecast_df[(forecast_df["ds"] >= start_ts) & (forecast_df["ds"] <= end_ts)].copy()
            forecast_parts.append(forecast_filtered)
            continue

        if len(known_values) < 30:
            # Fallback to mean forecast if we don't have enough history for lag features.
            fallback_val = float(np.mean(known_values)) if known_values else 0.0
            forecast_df = pd.DataFrame({
                "ds": future_dates,
                "series": s,
                "yhat": [fallback_val] * forecast_horizon,
                "yhat_lower": [fallback_val * 0.9] * forecast_horizon,
                "yhat_upper": [fallback_val * 1.1] * forecast_horizon,
            })
        else:
            yhat_list = []
            current_len = len(df_series)
            for i, future_date in enumerate(future_dates):
                # Construct features exactly matching what train_linear_regression uses
                X_future = pd.DataFrame({
                    "trend": [current_len + i],
                    "day_of_week": [future_date.dayofweek],
                    "month": [future_date.month],
                    "lag_1": [known_values[-1]],
                    "lag_7": [known_values[-7]],
                    "lag_30": [known_values[-30]],
                    "rolling_mean_7": [np.mean(known_values[-7:])],
                    "rolling_mean_30": [np.mean(known_values[-30:])],
                })
                pred = model.predict(X_future)[0]
                pred = max(0.0, float(pred))
                
                yhat_list.append(pred)
                known_values.append(pred)
            
            yhat_array = np.array(yhat_list)
            forecast_df = pd.DataFrame({
                "ds": future_dates,
                "series": s,
                "yhat": yhat_array,
                "yhat_lower": yhat_array * 0.9,
                "yhat_upper": yhat_array * 1.1,
            })

        # Filter for requested range
        start_ts = pd.to_datetime(forecast_start_date)
        end_ts = pd.to_datetime(forecast_end_date)
        forecast_filtered = forecast_df[(forecast_df["ds"] >= start_ts) & (forecast_df["ds"] <= end_ts)].copy()
        forecast_parts.append(forecast_filtered)

    if not forecast_parts:
        return pd.DataFrame(columns=["ds", "series", "yhat", "yhat_lower", "yhat_upper"])

    return pd.concat(forecast_parts, ignore_index=True)


# --- UI ------------------------------------------------------------------------

st.set_page_config(
    page_title="Forecast App",
    page_icon="ðŸ“ˆ",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_init_session_state()

METRIC_DISPLAY_NAMES = {
    "MAPE": "Accuracy Score (MAPE)",
    "MAE": "Mean Error (MAE)",
    "RMSE": "Error Spread (RMSE)",
    "MSE": "Squared Error (MSE)",
}


def _friendly_ui_error(error_msg: str) -> str:
    message = str(error_msg or "").strip()
    lowered = message.lower()

    if "valueerror" in lowered or "keyerror" in lowered:
        return "Something went wrong — please check your data."
    if "nan" in message:
        message = message.replace("NaN", "empty value")
    if "nat" in lowered:
        message = message.replace("NaT", "empty value")
    if "dtype" in lowered:
        message = message.replace("dtype", "data type")
    if "DataFrame" in message:
        message = message.replace("DataFrame", "data table")
    return message


def _friendly_config_error(error_msg: str, *, kind: str) -> str:
    message = str(error_msg or "").strip()
    lowered = message.lower()

    if kind == "campaign" and "campaign_periods.csv" in lowered and "not found" in lowered:
        return "Campaign file not found. Please contact your administrator to set up campaign_periods.csv."
    if kind == "holiday" and "ecommerce_holidays.csv" in lowered and "not found" in lowered:
        return "Holiday file not found. Please contact your administrator to set up ecommerce_holidays.csv."

    return _friendly_ui_error(message)

if st.session_state.get("bq_authenticated") is True:
    st.success("\U0001F510 Google Cloud authenticated", icon="\u2705")

st.markdown(
    """
<div style="
    background-color: #0072FF;
    border-radius: 12px;
    padding: 32px;
    text-align: center;
    margin-bottom: 24px;
">
    <h1 style="color: white; margin: 0; font-size: 32px;">
        Forecast App
    </h1>
    <p style="color: #E0EFFF; margin-top: 8px; font-size: 15px;">
        An end-to-end forecasting tool that automatically trains and 
compares 5 machine learning models, selects the best performer, 
and generates business ready forecasts with campaign and 
holiday adjustments.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- 1
st.header("1. Data source")
st.markdown("Upload a CSV or XLSX file. Your data will load into a data table for review.")

st.session_state["load_error"] = None

uploaded = st.file_uploader(
    "Choose a file",
    type=["csv", "xlsx", "xls"],
    key="file_uploader",
)
if st.button("Load data from file", type="primary", key="btn_load_file"):
    if uploaded is None:
        st.session_state["load_error"] = "Choose a file first."
    else:
        try:
            st.session_state["df_raw"] = _load_uploaded_file(uploaded)
            st.session_state["df_mapped"] = None
            st.session_state["schema_error"] = None
            st.session_state["forecast_run_done"] = False
            st.session_state["results_placeholder"] = None
        except Exception as e:  # noqa: BLE001 — user-facing parse / IO errors
            st.session_state["load_error"] = str(e)
            st.session_state["df_raw"] = None

if st.session_state["load_error"]:
    error_msg = st.session_state["load_error"]
    if "codec" in error_msg.lower() or "encode" in error_msg.lower():
        st.error(
            "❌ Could not read this file — it may be corrupted or saved in an unsupported format. "
            "Try resaving as .csv or .xlsx and upload again."
        )
    elif "empty" in error_msg.lower():
        st.error(
            "❌ This file appears to be empty. Please check the file and try again."
        )
    elif "column" in error_msg.lower():
        st.error(f"❌ Column issue: {error_msg}")
    else:
        st.error(f"❌ Something went wrong: {error_msg}")

st.divider()

# --------------------------------------------------------------------------- 2
st.header("2. Data preview & schema mapping")
df_raw = st.session_state["df_raw"]

if df_raw is None:
    st.info("Load data in step 1 to preview columns and map **Date** and **Metric**.")
else:
    st.subheader("Preview")
    st.dataframe(df_raw.head(5), use_container_width=True)

    numeric_cols = df_raw.select_dtypes(include=["number"]).columns.tolist()
    all_cols = list(df_raw.columns)

    c1, c2 = st.columns(2)
    with c1:
        date_pick = st.selectbox("Date column", options=all_cols, key="pick_date")
    with c2:
        metric_default = [numeric_cols[0]] if numeric_cols else [all_cols[0]]
        metric_pick = st.multiselect(
            "Metric columns (one or more)",
            options=all_cols,
            default=st.session_state.get("metric_cols") or metric_default,
            key="pick_metrics",
        )

    if st.button("Apply schema for forecasting", key="btn_schema", type="primary"):
        st.session_state["date_col"] = date_pick
        st.session_state["metric_cols"] = metric_pick
        st.session_state["metric_col"] = metric_pick[0] if metric_pick else None
        try:
            if not metric_pick:
                st.session_state["schema_error"] = "Select at least one metric column."
                st.session_state["df_mapped"] = None
            else:
                wide = df_raw[[date_pick, *metric_pick]].copy()
                wide = wide.rename(columns={date_pick: "ds"})
                
                # Use a temporary value name to avoid collision if any column is already named 'y'
                # or if the user selected multiple metrics that were renamed to 'y' previously.
                # We'll melt with original names to keep the 'series' column descriptive.
                slim = wide.melt(
                    id_vars=["ds"],
                    value_vars=metric_pick,
                    var_name="series",
                    value_name="_y_value_tmp",
                )
                slim = slim.rename(columns={"_y_value_tmp": "y"})
                
                slim = _normalize_ds_to_calendar_date(slim)
                slim = slim.sort_values(["series", "ds"]).reset_index(drop=True)
                ok, msg = _validate_mapped_df(slim)
                if not ok:
                    st.session_state["schema_error"] = msg
                    st.session_state["df_mapped"] = None
                else:
                    st.session_state["df_mapped"] = slim
                    st.session_state["schema_error"] = None
        except KeyError as e:
            st.session_state["schema_error"] = f"Column missing: {e}"
            st.session_state["df_mapped"] = None

    if st.session_state["schema_error"]:
        st.error(_friendly_ui_error(st.session_state["schema_error"]))
    elif st.session_state["df_mapped"] is not None:
        df_mapped = st.session_state["df_mapped"]
        series_count = df_mapped["series"].nunique()
        min_days = df_mapped.groupby("series")["ds"].nunique().min()
        if min_days < 180:
            st.warning(
                f"\u26A0\uFE0F Data loaded \u2014 {series_count} metric(s) detected but only {min_days} days of history found. "
                "We recommend at least 180 days for reliable forecasts."
            )
        else:
            st.success(
                f"\u2705 Data loaded successfully \u2014 {series_count} metric(s) detected with at least {min_days} days of history. "
                "You're ready to forecast!"
            )
        with st.expander("Preview processed data"):
            st.dataframe(st.session_state["df_mapped"].head(5), use_container_width=True)

st.divider()

# --------------------------------------------------------------------------- 3
st.header("3. Seasonality & holidays")
st.markdown("Capture **campaign windows** and **e-commerce holidays** for Prophet regressors / holidays (stored for later use).")
st.info(
    "📌 Both campaigns and holidays are optional but "
    "recommended. Campaigns help the model understand "
    "sales spikes from promotions. Holidays help it "
    "account for special dates that affect buying behavior. "
    "You can fill in one, both, or neither."
)

tab1, tab2 = st.tabs(["\U0001F4C5 Campaign Periods", "\U0001F389 E-commerce Holidays"])

with tab1:
    st.subheader("Campaign periods")
    st.caption("Date ranges when you ran promotions or sales events — e.g. 11.11, Black Friday, Harbolnas.")

    if st.button("Refresh campaign periods from backend CSV", key="btn_refresh_campaigns"):
        try:
            st.session_state["campaign_periods"] = _load_campaign_periods_csv()
            st.session_state["campaign_load_error"] = None
        except Exception as e:  # noqa: BLE001
            st.session_state["campaign_load_error"] = str(e)
            st.session_state["campaign_periods"] = pd.DataFrame(columns=["start", "end", "label"])

    if st.session_state.get("campaign_periods", pd.DataFrame()).empty and not st.session_state.get("campaign_load_error"):
        try:
            st.session_state["campaign_periods"] = _load_campaign_periods_csv()
        except Exception as e:  # noqa: BLE001
            st.session_state["campaign_load_error"] = str(e)

    if st.session_state["campaign_load_error"]:
        st.error(_friendly_config_error(st.session_state["campaign_load_error"], kind="campaign"))
        st.info("Campaign options will appear here once the campaign file is available.")
    else:
        # Load defaults
        default_camps = st.session_state.get("campaign_periods", pd.DataFrame()).copy()
        if default_camps.empty:
            default_camps = pd.DataFrame(columns=["start", "end", "label"])

        if "selected_campaign_labels" not in st.session_state or st.session_state["selected_campaign_labels"] is None:
            st.session_state["selected_campaign_labels"] = default_camps["label"].dropna().unique().tolist()

        manual_camps_list = st.session_state.get("campaign_manual_list", [])

        all_campaign_rows: list[dict[str, Any]] = []
        for _, row in default_camps.iterrows():
            all_campaign_rows.append(
                {
                    "label": row["label"],
                    "start": row["start"],
                    "end": row["end"],
                    "source": "default",
                }
            )
        for idx, m_row in enumerate(manual_camps_list):
            all_campaign_rows.append(
                {
                    "label": m_row["label"],
                    "start": m_row["start"],
                    "end": m_row["end"],
                    "source": "manual",
                    "manual_index": idx,
                }
            )

        for i in range(len(all_campaign_rows)):
            if f"chk_camp_{i}" not in st.session_state:
                st.session_state[f"chk_camp_{i}"] = True

        all_selected = all(
            st.session_state.get(f"chk_camp_{i}", True) for i in range(len(all_campaign_rows))
        )
        btn_label = "Deselect all" if all_selected else "Select all"
        if st.button(btn_label, key="btn_select_all_campaigns"):
            new_val = not all_selected
            for i in range(len(all_campaign_rows)):
                st.session_state[f"chk_camp_{i}"] = new_val
            st.rerun()

        st.write("")
        col_inc, col_lbl, col_start, col_end, col_src, col_del = st.columns([0.5, 3, 2, 2, 1, 0.5])
        col_inc.markdown("**✓**")
        col_lbl.markdown("**Campaign Name**")
        col_start.markdown("**Start Date**")
        col_end.markdown("**End Date**")
        col_src.markdown("**Source**")
        col_del.markdown("")
        st.markdown('<hr style="margin: 4px 0 8px 0;">', unsafe_allow_html=True)

        new_manual_list = []

        with st.container(height=300):
            for i, row in enumerate(all_campaign_rows):
                row_cols = st.columns([0.5, 3, 2, 2, 1, 0.5])
                with row_cols[0]:
                    checked = st.checkbox("", key=f"chk_camp_{i}", label_visibility="collapsed")
                with row_cols[1]:
                    st.write(row["label"])
                with row_cols[2]:
                    st.write(str(row["start"])[:10])
                with row_cols[3]:
                    st.write(str(row["end"])[:10])
                with row_cols[4]:
                    if row.get("source", "default") == "manual":
                        st.markdown(
                            '<span style="font-size:11px; background:#EFF6FF; color:#1D4ED8; padding:2px 8px; border-radius:6px;">manual</span>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            '<span style="font-size:11px; background:#F3F4F6; color:#6B7280; padding:2px 8px; border-radius:6px;">default</span>',
                            unsafe_allow_html=True,
                        )
                with row_cols[5]:
                    if row.get("source", "default") == "manual":
                        if st.button(
                            "✕",
                            key=f"del_camp_{i}",
                            help="Remove this campaign",
                            type="secondary",
                        ):
                            st.session_state["campaign_manual_list"].pop(row["manual_index"])
                            st.rerun()

                if row.get("source") == "manual":
                    new_manual_list.append(
                        {
                            "start": row["start"],
                            "end": row["end"],
                            "label": row["label"],
                            "include": checked,
                        }
                    )

        st.session_state["selected_campaign_labels"] = [
            all_campaign_rows[i]["label"]
            for i in range(len(all_campaign_rows))
            if st.session_state.get(f"chk_camp_{i}", True)
        ]
        st.session_state["campaign_manual_list"] = new_manual_list

        manual_df_rows = [r for r in new_manual_list if r["include"]]
        st.session_state["campaign_manual"] = pd.DataFrame(manual_df_rows, columns=["start", "end", "label"])

        # PART 2 — Add Custom Campaign
        st.markdown('<hr style="margin: 16px 0;">', unsafe_allow_html=True)
        st.markdown("### Add Custom Campaign")
        col1, col2, col3 = st.columns([2, 2, 3])
        with col1:
            new_start = st.date_input("Start date", key="new_camp_start")
        with col2:
            new_end = st.date_input("End date", key="new_camp_end")
        with col3:
            new_label = st.text_input(
                "Campaign name",
                placeholder="e.g. Harbolnas 12.12",
                key="new_camp_label",
            )
            st.caption("👆 Click '➕ Add campaign' button below to save.")

        if st.button("➕ Add campaign", key="btn_add_campaign", use_container_width=True, type="secondary"):
            if not new_label.strip():
                st.error("Please enter a campaign name.")
            elif new_end < new_start:
                st.error("End date must be after start date.")
            else:
                st.session_state["campaign_manual_list"].append({
                    "label": new_label.strip(),
                    "start": pd.to_datetime(new_start).normalize(),
                    "end": pd.to_datetime(new_end).normalize(),
                    "source": "manual",
                    "include": True,
                })
                if "new_camp_label" in st.session_state:
                    del st.session_state["new_camp_label"]
                st.rerun()

with tab2:
    st.subheader("E-commerce Holidays")
    st.caption("Single dates that affect buying behavior — e.g. Ramadhan, Lebaran, Christmas.")

    if st.button("Refresh default e-commerce holidays from backend CSV", key="btn_refresh_ecomm_holidays"):
        try:
            st.session_state["ecommerce_holiday_defaults"] = _load_ecommerce_holidays_csv()
            st.session_state["ecommerce_holiday_load_error"] = None
        except Exception as e:  # noqa: BLE001
            st.session_state["ecommerce_holiday_load_error"] = str(e)
            st.session_state["ecommerce_holiday_defaults"] = pd.DataFrame(columns=["holiday", "ds"])

    if st.session_state.get("ecommerce_holiday_defaults", pd.DataFrame()).empty and not st.session_state.get(
        "ecommerce_holiday_load_error"
    ):
        try:
            st.session_state["ecommerce_holiday_defaults"] = _load_ecommerce_holidays_csv()
        except Exception as e:  # noqa: BLE001
            st.session_state["ecommerce_holiday_load_error"] = str(e)

    if st.session_state["ecommerce_holiday_load_error"]:
        st.error(_friendly_config_error(st.session_state["ecommerce_holiday_load_error"], kind="holiday"))
        st.info("Holiday options will appear here once the holiday file is available.")
    else:
        # Load defaults
        default_hols = st.session_state.get("ecommerce_holiday_defaults", pd.DataFrame()).copy()
        if default_hols.empty:
            default_hols = pd.DataFrame(columns=["holiday", "ds"])

        if "selected_ecommerce_holidays" not in st.session_state or st.session_state["selected_ecommerce_holidays"] is None:
            st.session_state["selected_ecommerce_holidays"] = default_hols["holiday"].dropna().unique().tolist()

        manual_hols_list = st.session_state.get("holiday_manual_list", [])

        all_holiday_rows: list[dict[str, Any]] = []
        for _, row in default_hols.iterrows():
            all_holiday_rows.append(
                {
                    "holiday": row["holiday"],
                    "ds": row["ds"],
                    "source": "default",
                }
            )
        for idx, m_row in enumerate(manual_hols_list):
            all_holiday_rows.append(
                {
                    "holiday": m_row["holiday"],
                    "ds": m_row["ds"],
                    "source": "manual",
                    "manual_index": idx,
                }
            )

        for i in range(len(all_holiday_rows)):
            if f"chk_hol_{i}" not in st.session_state:
                st.session_state[f"chk_hol_{i}"] = True

        all_selected = all(
            st.session_state.get(f"chk_hol_{i}", True) for i in range(len(all_holiday_rows))
        )
        btn_label = "Deselect all" if all_selected else "Select all"
        if st.button(btn_label, key="btn_select_all_holidays"):
            new_val = not all_selected
            for i in range(len(all_holiday_rows)):
                st.session_state[f"chk_hol_{i}"] = new_val
            st.rerun()

        st.write("")
        col_inc, col_lbl, col_date, col_src, col_del = st.columns([0.5, 3, 2, 1, 0.5])
        col_inc.markdown("**✓**")
        col_lbl.markdown("**Holiday Name**")
        col_date.markdown("**Date**")
        col_src.markdown("**Source**")
        col_del.markdown("")
        st.markdown('<hr style="margin: 4px 0 8px 0;">', unsafe_allow_html=True)

        new_manual_hols_list = []

        with st.container(height=300):
            for i, row in enumerate(all_holiday_rows):
                row_cols = st.columns([0.5, 3, 2, 1, 0.5])
                with row_cols[0]:
                    checked = st.checkbox("", key=f"chk_hol_{i}", label_visibility="collapsed")
                with row_cols[1]:
                    st.write(row["holiday"])
                with row_cols[2]:
                    st.write(str(row["ds"])[:10])
                with row_cols[3]:
                    if row.get("source", "default") == "manual":
                        st.markdown(
                            '<span style="font-size:11px; background:#EFF6FF; color:#1D4ED8; padding:2px 8px; border-radius:6px;">manual</span>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            '<span style="font-size:11px; background:#F3F4F6; color:#6B7280; padding:2px 8px; border-radius:6px;">default</span>',
                            unsafe_allow_html=True,
                        )
                with row_cols[4]:
                    if row.get("source", "default") == "manual":
                        if st.button(
                            "✕",
                            key=f"del_hol_{i}",
                            help="Remove this holiday",
                            type="secondary",
                        ):
                            st.session_state["holiday_manual_list"].pop(row["manual_index"])
                            st.rerun()

                if row.get("source") == "manual":
                    new_manual_hols_list.append(
                        {
                            "holiday": row["holiday"],
                            "ds": row["ds"],
                            "include": checked,
                        }
                    )

        st.session_state["selected_ecommerce_holidays"] = [
            all_holiday_rows[i]["holiday"]
            for i in range(len(all_holiday_rows))
            if st.session_state.get(f"chk_hol_{i}", True)
        ]
        st.session_state["holiday_manual_list"] = new_manual_hols_list

        manual_holidays_df_rows = [r for r in new_manual_hols_list if r["include"]]
        manual_holidays = pd.DataFrame(manual_holidays_df_rows, columns=["holiday", "ds"])
        st.session_state["ecommerce_holiday_manual"] = manual_holidays

        # Perform combined_holidays calculation
        default_holidays = st.session_state.get("ecommerce_holiday_defaults", pd.DataFrame())
        selected_defaults = default_holidays[default_holidays["holiday"].isin(st.session_state["selected_ecommerce_holidays"])]
        combined_holidays = (
            pd.concat([selected_defaults, manual_holidays], ignore_index=True)
            .drop_duplicates(subset=["holiday", "ds"])
            .sort_values(["ds", "holiday"])
            .reset_index(drop=True)
        )
        st.session_state["ecommerce_holidays"] = combined_holidays

        # PART 2 — Add Custom Holiday
        st.markdown('<hr style="margin: 16px 0;">', unsafe_allow_html=True)
        st.markdown("### Add Custom Holiday")
        col1, col2 = st.columns([3, 2])
        with col1:
            new_holiday = st.text_input(
                "Holiday name",
                placeholder="e.g. Lebaran 2025",
                key="new_hol_name",
            )
            st.caption("👆 Click '➕ Add holiday' button below to save.")
        with col2:
            new_date = st.date_input("Date", key="new_hol_date")

        if st.button("➕ Add holiday", key="btn_add_holiday", use_container_width=True, type="secondary"):
            if not new_holiday.strip():
                st.error("Please enter a holiday name.")
            else:
                st.session_state["holiday_manual_list"].append({
                    "holiday": new_holiday.strip(),
                    "ds": pd.to_datetime(new_date).normalize(),
                    "source": "manual",
                    "include": True,
                })
                if "new_hol_name" in st.session_state:
                    del st.session_state["new_hol_name"]
                st.rerun()

st.divider()

# --------------------------------------------------------------------------- 4
st.header("4. Model selection & evaluation settings")

st.subheader("Forecast date range")
df_mapped = st.session_state.get("df_mapped")
if df_mapped is not None and not df_mapped.empty:
    try:
        last_data_date = pd.to_datetime(df_mapped["ds"]).max().date()
        default_start = last_data_date + pd.Timedelta(days=1)
        default_end = default_start + pd.Timedelta(days=30)
        
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            start_date = st.date_input(
                "Forecast Start Date",
                value=st.session_state.get("forecast_start_date") or default_start,
                min_value=default_start,
                key="input_start_date",
            )
            st.session_state["forecast_start_date"] = start_date

        stored_end = st.session_state.get("forecast_end_date")
        if stored_end is not None and stored_end <= start_date:
            st.session_state["forecast_end_date"] = start_date + pd.Timedelta(days=30)

        with col_d2:
            end_date = st.date_input(
                "Forecast End Date",
                value=st.session_state.get("forecast_end_date") or default_end,
                min_value=start_date,
                key="input_end_date",
            )
            st.session_state["forecast_end_date"] = end_date
        
        # Validation Logic
        st.session_state["forecast_date_error"] = None
        if start_date <= last_data_date:
            st.caption(f"📅 Forecast must start after your last data point ({last_data_date}).")
        elif end_date <= start_date:
            st.session_state["forecast_date_error"] = "Forecast End Date must be after the Forecast Start Date."
        
        if st.session_state["forecast_date_error"]:
            st.error(st.session_state["forecast_date_error"])
        else:
            # Calculate periods for internal use
            forecast_horizon = (end_date - last_data_date).days
            st.caption(f"Forecast will cover **{forecast_horizon}** days beyond the last historical date ({last_data_date}).")
            
            # Long horizon warning (> 3 years / 1095 days)
            if forecast_horizon > 1095:
                st.warning("Forecast horizon too long (> 3 years). Forecast accuracy may decrease.")
    except Exception:
        st.session_state["forecast_start_date"] = None
        st.session_state["forecast_end_date"] = None
        st.error(
            "⚠️ Date range was reset because of a conflict. Please select your forecast dates again."
        )
else:
    st.info("Map your data in step 2 to select a forecast date range.")

for m in MODEL_OPTIONS:
    if f"model_chk_{m}" not in st.session_state:
        st.session_state[f"model_chk_{m}"] = True

MODEL_SHORT_DESC = {
    "Prophet": "Seasonal + campaigns",
    "ARIMA": "Statistical baseline",
    "ETS": "Weighted recent history",
    "Linear Regression": "Interpretable, 45+ days",
    "XGBoost": "Complex patterns",
}

selected_count = sum(
    1 for m in MODEL_OPTIONS if st.session_state.get(f"model_chk_{m}", True)
)
all_selected = selected_count == len(MODEL_OPTIONS)

col_title, col_btn = st.columns([4, 1])
with col_title:
    st.markdown("**Models to compare**")
with col_btn:
    btn_label = "Deselect all" if all_selected else "Select all"
    if st.button(btn_label, key="btn_toggle_all_models", use_container_width=True):
        new_val = not all_selected
        for m in MODEL_OPTIONS:
            st.session_state[f"model_chk_{m}"] = new_val
        st.rerun()

rows = [MODEL_OPTIONS[i : i + 3] for i in range(0, len(MODEL_OPTIONS), 3)]

for row in rows:
    cols = st.columns(3)
    for j, model_name in enumerate(row):
        with cols[j]:
            with st.container(border=True):
                checked = st.checkbox(
                    f"**{model_name}**",
                    key=f"model_chk_{model_name}",
                )
                st.caption(MODEL_SHORT_DESC.get(model_name, ""))

selected_models = [
    m for m in MODEL_OPTIONS if st.session_state.get(f"model_chk_{m}", True)
]
st.session_state["selected_models"] = selected_models

st.caption(f"{len(selected_models)} of {len(MODEL_OPTIONS)} models selected")

if not selected_models:
    st.warning("Please select at least one model to run the forecast.")

with st.expander("What do these models mean? \U0001F4A1"):
    st.markdown(
        "- Prophet: Best for data with strong weekly or seasonal patterns. Handles holidays and campaigns natively.\n"
        "- ARIMA: Good for stable time series without strong seasonality. Works purely from past values.\n"
        "- ETS: Similar to ARIMA but focuses on weighted averages - gives more importance to recent data.\n"
        "- Linear Regression: Uses time-based features like day of week and recent values to find patterns. Needs 45+ days.\n"
        "- XGBoost: Most powerful for complex patterns. Same features as Linear Regression but uses hundreds of decision trees."
    )

st.session_state["selected_metric"] = st.selectbox(
    "Accuracy metric",
    options=METRIC_OPTIONS,
    index=METRIC_OPTIONS.index(st.session_state.get("selected_metric", "MAPE")),
    key="select_metric",
    format_func=lambda metric: METRIC_DISPLAY_NAMES.get(metric, metric),
)

METRIC_INFO = {
    "MAPE": {
        "name": "Mean Absolute Percentage Error",
        "explain": "Measures average % error. Easy to interpret across different data scales.",
        "bands": [
            (0.10, "✅ Excellent", "Model is off by less than 10% on average."),
            (0.20, "✅ Good", "Acceptable for most forecasting use cases."),
            (0.35, "⚠️ Fair", "Consider adding more history or adjusting campaigns."),
            (float("inf"), "❌ Poor", "Forecast may not be reliable. Check your data quality."),
        ],
    },
    "MAE": {
        "name": "Mean Absolute Error",
        "explain": "Average error in the same unit as your data (e.g. if forecasting sales in IDR, MAE is in IDR).",
        "bands": None,
    },
    "RMSE": {
        "name": "Root Mean Squared Error",
        "explain": "Like MAE but penalises large errors more heavily. Same unit as your data.",
        "bands": None,
    },
    "MSE": {
        "name": "Mean Squared Error",
        "explain": "Squared unit — best used for comparing models, not for interpreting the actual error size.",
        "bands": None,
    },
}

m = st.session_state["selected_metric"]
info = METRIC_INFO[m]
st.caption(f"**{info['name']}** — {info['explain']}")
if info["bands"]:
    st.caption(
        "Score guide: "
        + " · ".join(
            [f"{label} (< {int(b * 100)}%)" for b, label, _ in info["bands"][:-1]]
            + ["❌ Poor (above 35%)"]
        )
    )
st.divider()

# --------------------------------------------------------------------------- 5
st.header("5. Run forecast")
st.markdown("We'll train each selected model, compare their accuracy, and automatically pick the best one for your data.")
st.caption("\u23F1 Estimated run time: Prophet + XGBoost may take 2-5 minutes on large datasets or many series. ARIMA, ETS, and Linear Regression are typically faster.")

run_disabled = (
    st.session_state["df_mapped"] is None
    or not st.session_state["selected_models"]
    or st.session_state.get("forecast_date_error") is not None
)
if run_disabled:
    if st.session_state.get("forecast_date_error"):
        st.warning("Fix the forecast date errors in step 4 to enable the run.")
    elif st.session_state.get("schema_error"):
        st.warning(f"Fix the data issue in Step 2: {_friendly_ui_error(st.session_state['schema_error'])}")
    else:
        st.warning("Complete schema mapping (step 2) and select at least one model (step 4) to enable the run.")

if st.button("Run forecast", type="primary", disabled=run_disabled, key="btn_run_forecast"):
    with st.spinner("Running models, tuning, and selection..."):
        selected_models = st.session_state.get("selected_models") or []
        mapped = st.session_state.get("df_mapped")

        placeholder_forecast = pd.DataFrame(
            {
                "ds": [],
                "series": [],
                "yhat": [],
                "yhat_lower": [],
                "yhat_upper": [],
            }
        )

        if mapped is None:
            st.session_state["forecast_run_done"] = True
            st.session_state["results_placeholder"] = {
                "best_model": None,
                "metric_scores": {},
                "series_names": [],
                "forecast_df": placeholder_forecast,
                "primary_metric": None,
                "message": "No mapped data found. Please run schema mapping first.",
            }
            st.stop()

        primary_metric = st.session_state.get("selected_metric", "MAPE")
        model_results = {}
        total_steps = len(selected_models)
        completed = 0
        progress_bar = st.progress(0, text="Starting forecast run...")

        def _advance_progress(model_name: str) -> None:
            # use a mutable container to avoid nonlocal/global issues at module scope
            completed["value"] += 1
            progress_bar.progress(
                completed["value"] / total_steps,
                text=f"Completed {completed['value']}/{total_steps} models â€” last finished: {model_name}",
            )

        # 1. Prophet
        if "Prophet" in selected_models:
            try:
                with st.spinner("Running grid search for Prophetâ€¦"):
                    campaign_periods = st.session_state.get("campaign_periods")
                    if campaign_periods is None:
                        campaign_periods = pd.DataFrame()
                    selected_campaign_labels = st.session_state.get("selected_campaign_labels") or []
                    manual_campaigns = st.session_state.get("campaign_manual")
                    if manual_campaigns is None:
                        manual_campaigns = pd.DataFrame()
                    ecommerce_holidays_df = st.session_state.get("ecommerce_holidays")

                    # Combine campaigns
                    selected_csv_campaigns = campaign_periods[campaign_periods["label"].isin(selected_campaign_labels)]
                    combined_campaigns = pd.concat([selected_csv_campaigns, manual_campaigns], ignore_index=True)
                    campaign_days = _build_campaign_days(combined_campaigns, None)

                    validation_forecast, best_params, best_score = _tune_prophet_grid_search(
                        mapped,
                        campaign_days,
                        ecommerce_holidays_df if ecommerce_holidays_df is not None and not ecommerce_holidays_df.empty else None,
                        metric_name=primary_metric,
                        growth_strategy="auto",
                    )
                    
                    # Evaluate against all available metrics
                    all_metric_results = {}
                    for m in METRIC_OPTIONS:
                        # For Prophet, we currently tune based on primary_metric
                        # To be thorough, we compute all metrics for the best model found
                        y_true = []
                        y_pred = []
                        for s in mapped["series"].unique():
                            s_mapped = mapped[mapped["series"] == s]
                            # Simple split matching the internal prophet validation
                            n = len(s_mapped)
                            vd = int(min(30, max(7, n * 0.2)))
                            val_y = s_mapped.iloc[n-vd:]["y"].to_numpy()
                            s_fc = validation_forecast[validation_forecast["series"] == s]
                            if not s_fc.empty:
                                y_true.extend(val_y)
                                y_pred.extend(s_fc["yhat"].to_numpy())
                        
                        if y_true and y_pred:
                            all_metric_results[m] = _compute_metric(m, np.array(y_true), np.array(y_pred))

                    model_results["Prophet"] = {
                        "score": best_score,
                        "all_metrics": all_metric_results,
                        "validation_forecast": validation_forecast,
                        "best_params": best_params,
                        "message": f"Prophet tuned (CP={best_params.get('changepoint_prior_scale', 0):.4g}, SP={best_params.get('seasonality_prior_scale', 0):.4g}, Mode={best_params.get('seasonality_mode', 'n/a')}, Growth={best_params.get('growth', 'flat')}).",
                    }
            except Exception as e:  # noqa: BLE001
                st.warning(f"Prophet training failed: {e}")
            finally:
                completed += 1
                progress_bar.progress(
                    completed / total_steps,
                    text=f"Completed {completed}/{total_steps} - Last finished: Prophet",
                )

        # 2. ARIMA
        if "ARIMA" in selected_models:
            try:
                with st.spinner("Running ARIMA (grid search p,d,q)..."):
                    series_names = sorted(mapped["series"].dropna().unique().tolist())
                    forecast_parts = []
                    scores = []
                    params_per_series = {}
                    all_y_true = []
                    all_y_pred = []
                    for s in series_names:
                        try:
                            df_series = mapped[mapped["series"] == s]
                            score, forecast_val, model = _fit_predict_arima_validation(
                                df_series, metric_name=primary_metric
                            )
                            if np.isfinite(score):
                                forecast_val["series"] = s
                                forecast_parts.append(forecast_val)
                                scores.append(score)
                                
                                # Collect for aggregate metrics
                                n = len(df_series)
                                vd = int(min(30, max(7, n * 0.2)))
                                all_y_true.extend(df_series.iloc[n-vd:]["y"].to_numpy())
                                all_y_pred.extend(forecast_val["yhat"].to_numpy())

                                if hasattr(model, 'order'):
                                    params_per_series[s] = {"p": model.order[0], "d": model.order[1], "q": model.order[2]}
                        except Exception as e:
                            st.warning(f"ARIMA failed for series {s}: {e}")
                            continue

                    if scores:
                        avg_score = float(np.mean(scores))
                        all_metric_results = {}
                        for m in METRIC_OPTIONS:
                            all_metric_results[m] = _compute_metric(m, np.array(all_y_true), np.array(all_y_pred))

                        validation_forecast = pd.concat(forecast_parts, ignore_index=True)
                        model_results["ARIMA"] = {
                            "score": avg_score,
                            "all_metrics": all_metric_results,
                            "validation_forecast": validation_forecast,
                            "best_params": params_per_series,
                            "message": f"ARIMA grid-searched (p,d,q) across {len(scores)} series.",
                        }
            except Exception as e:  # noqa: BLE001
                st.warning(f"ARIMA training failed: {e}")
            finally:
                completed += 1
                progress_bar.progress(
                    completed / total_steps,
                    text=f"Completed {completed}/{total_steps} - Last finished: ARIMA",
                )

        # 3. ETS
        if "ETS" in selected_models:
            try:
                with st.spinner("Running ETS (ExponentialSmoothing grid search)..."):
                    series_names = sorted(mapped["series"].dropna().unique().tolist())
                    forecast_parts = []
                    scores = []
                    params_per_series = {}
                    all_y_true = []
                    all_y_pred = []
                    for s in series_names:
                        try:
                            df_series = mapped[mapped["series"] == s]
                            score, forecast_val, model, params = _fit_predict_ets_validation(
                                df_series, metric_name=primary_metric
                            )
                            if np.isfinite(score):
                                forecast_val["series"] = s
                                forecast_parts.append(forecast_val)
                                scores.append(score)
                                
                                n = len(df_series)
                                vd = int(min(30, max(7, n * 0.2)))
                                all_y_true.extend(df_series.iloc[n-vd:]["y"].to_numpy())
                                all_y_pred.extend(forecast_val["yhat"].to_numpy())

                                params_per_series[s] = params
                        except Exception as e:
                            st.warning(f"ETS failed for series {s}: {e}")
                            continue

                    if scores:
                        avg_score = float(np.mean(scores))
                        all_metric_results = {}
                        for m in METRIC_OPTIONS:
                            all_metric_results[m] = _compute_metric(m, np.array(all_y_true), np.array(all_y_pred))

                        validation_forecast = pd.concat(forecast_parts, ignore_index=True)
                        model_results["ETS"] = {
                            "score": avg_score,
                            "all_metrics": all_metric_results,
                            "validation_forecast": validation_forecast,
                            "best_params": params_per_series,
                            "message": f"ETS tuned via grid search across {len(scores)} series.",
                        }
            except Exception as e:  # noqa: BLE001
                st.warning(f"ETS training failed: {e}")
            finally:
                completed += 1
                progress_bar.progress(
                    completed / total_steps,
                    text=f"Completed {completed}/{total_steps} - Last finished: ETS",
                )

        # 4. Linear Regression
        if "Linear Regression" in selected_models:
            try:
                with st.spinner("Running Linear Regression (feature engineering)..."):
                    series_names = sorted(mapped["series"].dropna().unique().tolist())
                    forecast_parts = []
                    scores = []
                    models_per_series = {}
                    all_y_true = []
                    all_y_pred = []
                    for s in series_names:
                        try:
                            df_series = mapped[mapped["series"] == s]
                            score, forecast_val, model = _fit_predict_lr_validation(
                                df_series, metric_name=primary_metric
                            )
                            if np.isfinite(score):
                                forecast_val["series"] = s
                                forecast_parts.append(forecast_val)
                                scores.append(score)
                                models_per_series[s] = model
                                
                                # For LR, we need to match the indices used in train_linear_regression
                                # Instead of complex re-splitting, we'll use the forecast_val dates
                                y_val_actual = df_series[df_series["ds"].isin(forecast_val["ds"])]["y"].to_numpy()
                                all_y_true.extend(y_val_actual)
                                all_y_pred.extend(forecast_val["yhat"].to_numpy())
                        except Exception as e:
                            st.warning(f"Linear Regression failed for series {s}: {e}")
                            continue

                    if scores:
                        avg_score = float(np.mean(scores))
                        all_metric_results = {}
                        for m in METRIC_OPTIONS:
                            all_metric_results[m] = _compute_metric(m, np.array(all_y_true), np.array(all_y_pred))

                        validation_forecast = pd.concat(forecast_parts, ignore_index=True)
                        model_results["Linear Regression"] = {
                            "score": avg_score,
                            "all_metrics": all_metric_results,
                            "validation_forecast": validation_forecast,
                            "best_params": models_per_series,
                            "message": f"Linear Regression trained across {len(scores)} series.",
                        }
            except Exception as e:  # noqa: BLE001
                st.warning(f"Linear Regression training failed: {e}")
            finally:
                completed += 1
                progress_bar.progress(
                    completed / total_steps,
                    text=f"Completed {completed}/{total_steps} - Last finished: Linear Regression",
                )

        # 5. XGBoost
        if "XGBoost" in selected_models:
            try:
                with st.spinner("Running XGBoost (gradient boosting)..."):
                    series_names = sorted(mapped["series"].dropna().unique().tolist())
                    forecast_parts = []
                    scores = []
                    models_per_series = {}
                    all_y_true = []
                    all_y_pred = []
                    for s in series_names:
                        try:
                            df_series = mapped[mapped["series"] == s]
                            score, forecast_val, model = _fit_predict_xgboost_validation(
                                df_series, metric_name=primary_metric
                            )
                            if np.isfinite(score):
                                forecast_val["series"] = s
                                forecast_parts.append(forecast_val)
                                scores.append(score)
                                models_per_series[s] = model
                                
                                y_val_actual = df_series[df_series["ds"].isin(forecast_val["ds"])]["y"].to_numpy()
                                all_y_true.extend(y_val_actual)
                                all_y_pred.extend(forecast_val["yhat"].to_numpy())
                        except Exception as e:
                            st.warning(f"XGBoost failed for series {s}: {e}")
                            continue

                    if scores:
                        avg_score = float(np.mean(scores))
                        all_metric_results = {}
                        for m in METRIC_OPTIONS:
                            all_metric_results[m] = _compute_metric(m, np.array(all_y_true), np.array(all_y_pred))

                        validation_forecast = pd.concat(forecast_parts, ignore_index=True)
                        model_results["XGBoost"] = {
                            "score": avg_score,
                            "all_metrics": all_metric_results,
                            "validation_forecast": validation_forecast,
                            "best_params": models_per_series,
                            "message": f"XGBoost trained across {len(scores)} series.",
                        }
            except Exception as e:  # noqa: BLE001
                st.warning(f"XGBoost training failed: {e}")
            finally:
                completed += 1
                progress_bar.progress(
                    completed / total_steps,
                    text=f"Completed {completed}/{total_steps} - Last finished: XGBoost",
                )

        progress_bar.progress(1.0, text="All models completed \u2705")

        # Check if we have any results
        if not model_results:
            st.session_state["forecast_run_done"] = True
            st.session_state["results_placeholder"] = {
                "best_model": None,
                "metric_scores": {},
                "series_names": [],
                "forecast_df": placeholder_forecast,
                "primary_metric": primary_metric,
                "message": "All selected models failed to train.",
            }
            st.stop()

        best_model_name = min(model_results, key=lambda k: model_results[k]["score"])

        best_res = model_results[best_model_name]
        best_score = best_res["score"]
        validation_forecast = best_res["validation_forecast"]
        best_params = best_res["best_params"]
        
        # --- NEW: Retrain best model on full data & forecast future ---
        with st.spinner(f"Retraining best model ({best_model_name}) on full data..."):
            future_forecast = pd.DataFrame()
            f_start = st.session_state["forecast_start_date"]
            f_end = st.session_state["forecast_end_date"]
            
            if best_model_name == "Prophet":
                campaign_periods = st.session_state.get("campaign_periods")
                if campaign_periods is None:
                    campaign_periods = pd.DataFrame()
                selected_campaign_labels = st.session_state.get("selected_campaign_labels") or []
                manual_campaigns = st.session_state.get("campaign_manual")
                if manual_campaigns is None:
                    manual_campaigns = pd.DataFrame()
                
                selected_csv_campaigns = campaign_periods[campaign_periods["label"].isin(selected_campaign_labels)]
                combined_campaigns = pd.concat([selected_csv_campaigns, manual_campaigns], ignore_index=True)
                campaign_days = _build_campaign_days(combined_campaigns, None)
                
                holidays_df = st.session_state.get("ecommerce_holidays")
                future_forecast, _ = _fit_prophet_full(mapped, campaign_days, holidays_df, best_params, f_start, f_end)
            elif best_model_name == "ARIMA":
                future_forecast, _ = _fit_arima_full(mapped, best_params, f_start, f_end)
            elif best_model_name == "ETS":
                future_forecast, _ = _fit_ets_full(mapped, best_params, f_start, f_end)
            elif best_model_name in ["Linear Regression", "XGBoost"]:
                ml_horizon = (pd.to_datetime(f_end) - pd.to_datetime(f_start)).days
                if ml_horizon > 30:
                    st.warning(
                        f"⚠️ {best_model_name} is most accurate for short-term forecasts (up to 30 days). "
                        f"Your selected horizon is {ml_horizon} days — consider using Prophet or ARIMA for "
                        f"longer forecasts, or reduce your forecast end date."
                    )
                future_forecast = _fit_ml_full(mapped, best_params, f_start, f_end)

        series_names = sorted(
            validation_forecast["series"].dropna().unique().tolist()
            if validation_forecast is not None and not validation_forecast.empty
            else []
        )

        future_forecast = future_forecast.rename(columns={
            "ds": "Date",
            "series": "Metric",
            "yhat": "Forecast",
            "yhat_lower": "Forecast (Low)",
            "yhat_upper": "Forecast (High)",
        })
        validation_forecast = validation_forecast.rename(columns={
            "ds": "Date",
            "series": "Metric",
            "yhat": "Predicted",
            "yhat_lower": "Predicted (Low)",
            "yhat_upper": "Predicted (High)",
        })

        msg = best_res["message"]
        other_selected = [m for m in selected_models if m not in model_results]
        if other_selected:
            msg += f" Note: other selected models ({', '.join(other_selected)}) failed."

        st.session_state["forecast_run_done"] = True
        st.session_state["results_placeholder"] = {
            "best_model": best_model_name,
            "metric_scores": best_res["all_metrics"],
            "all_model_results": model_results, # Store all results for details table
            "primary_metric": primary_metric,
            "series_names": series_names,
            "validation_df": validation_forecast,
            "forecast_df": future_forecast, # This is the future forecast
            "message": msg,
        }


st.divider()

# --------------------------------------------------------------------------- 6
import plotly.graph_objects as go

st.header("6. Results")

if not st.session_state.get("forecast_run_done"):
    st.info("Run the forecast in step 5 to see the best model, scores, and chart here.")
else:
    res = st.session_state.get("results_placeholder") or {}
    series_names = res.get("series_names") or []
    st.subheader("Best model evaluation")
    
    # Primary Metric
    primary_m = res.get("primary_metric") or "MAPE"
    primary_val = (res.get("metric_scores") or {}).get(primary_m, 0)
    
    c1, c2 = st.columns(2)
    with c1:
        st.metric(label="Selected model", value=res.get("best_model") or "-")
    with c2:
        st.metric(label=f"📈 {METRIC_DISPLAY_NAMES.get(primary_m, primary_m)}", value=f"{primary_val:.4g}")

    score = primary_val
    if primary_m == "MAPE":
        if score < 0.10:
            st.success(f"✅ Excellent — model is off by {score:.1%} on average.")
        elif score < 0.20:
            st.success(f"✅ Good — {score:.1%} average error, acceptable for most use cases.")
        elif score < 0.35:
            st.warning(f"⚠️ Fair — {score:.1%} average error. Consider adding more history.")
        else:
            st.error(f"❌ Poor — {score:.1%} average error. Forecast may not be reliable.")
    else:
        st.info(
            f"**{primary_m} = {score:,.2f}** — measured in the same unit as your data. "
            "Compare across models to find the best performer."
        )

    all_models = res.get("all_model_results", {})
    best_model_name = res.get("best_model", "").replace(" (Best Available)", "")

    if all_models and len(all_models) > 1:
        model_scores = {
            name: info.get("all_metrics", {}).get(primary_m, float("inf"))
            for name, info in all_models.items()
            if primary_m in info.get("all_metrics", {})
        }

        model_scores = {
            name: score
            for name, score in model_scores.items()
            if np.isfinite(score)
        }

        if len(model_scores) > 1 and best_model_name in model_scores:
            best_score = model_scores[best_model_name]
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1])

            unique_scores = {score for _, score in sorted_models}
            if len(unique_scores) == 1:
                st.info(f"All models performed equally on {primary_m}.")
            else:
                runner_up = [(name, score) for name, score in sorted_models if name != best_model_name]

                if runner_up:
                    runner_name, runner_score = runner_up[0]
                    if runner_score > 0 and np.isfinite(runner_score):
                        improvement = ((runner_score - best_score) / runner_score) * 100
                        st.info(
                            f"🏆 **{best_model_name}** is the best performer — **{improvement:.0f}% more "
                            f"accurate** than the runner-up ({runner_name}) based on {primary_m}."
                        )
            with st.expander("See full model ranking", expanded=False):
                col_rank, col_name, col_score, col_diff = st.columns([0.5, 2, 1.5, 1.5])
                with col_rank:
                    st.markdown("**#**")
                with col_name:
                    st.markdown("**Model**")
                with col_score:
                    st.markdown(f"**{primary_m} Score**")
                with col_diff:
                    st.markdown("**vs Best**")

                st.divider()

                for rank, (name, score) in enumerate(sorted_models, 1):
                    if not np.isfinite(score):
                        continue

                    medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"{rank}.")

                    if name == best_model_name:
                        diff_str = "← best"
                        diff_color = "green"
                    else:
                        if unique_scores and len(unique_scores) == 1:
                            diff_str = "← tied"
                        elif best_score > 0 and np.isfinite(best_score):
                            pct_worse = ((score - best_score) / best_score) * 100
                            diff_str = f"+{pct_worse:.0f}% worse"
                        else:
                            diff_str = "worse"
                        diff_color = "red"

                    col_rank, col_name, col_score, col_diff = st.columns([0.5, 2, 1.5, 1.5])
                    with col_rank:
                        st.write(medal)
                    with col_name:
                        st.write(f"**{name}**" if name == best_model_name else name)
                    with col_score:
                        if primary_m == "MAPE":
                            st.write(f"{score:.1%}")
                        else:
                            st.write(f"{score:,.2f} {primary_m}")
                    with col_diff:
                        st.markdown(f":{diff_color}[{diff_str}]")
        elif len(model_scores) <= 1:
            st.info("Run more models to compare.")

    df_val = res.get("validation_df")
    df_future = res.get("forecast_df")

    st.subheader("Download")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            label="Download validation CSV",
            data=df_val.to_csv(index=False).encode("utf-8") if df_val is not None else b"",
            file_name="forecast_validation.csv",
            mime="text/csv",
            key="dl_val",
        )
    with col_dl2:
        st.download_button(
            label="Download future forecast CSV",
            data=df_future.to_csv(index=False).encode("utf-8") if df_future is not None else b"",
            file_name="forecast_future.csv",
            mime="text/csv",
            key="dl_future",
        )

    st.divider()

    if primary_m == "MAPE" and (not np.isfinite(primary_val) or primary_val > 1):
        st.warning(
            "MAPE could not be calculated reliably because some actual values are zero. "
            "Consider switching to MAE or RMSE in step 4."
        )

    if series_names:
        st.caption(f"Detected metric series: {', '.join(series_names)}")

    if res.get("message"):
        st.warning(res["message"])

    best_model_name = (res.get("best_model") or "").replace(" (Best Available)", "")
    forecast_start_date = st.session_state.get("forecast_start_date")
    forecast_end_date = st.session_state.get("forecast_end_date")
    if (
        best_model_name in ["Linear Regression", "XGBoost"]
        and forecast_start_date is not None
        and forecast_end_date is not None
        and (pd.to_datetime(forecast_end_date) - pd.to_datetime(forecast_start_date)).days > 30
    ):
        st.info(
            "💡 This forecast extends beyond 30 days. ML models like Linear Regression and XGBoost "
            "may become less reliable over longer horizons as prediction errors compound over time. "
            "The first 30 days are most reliable."
        )

    # --- NEW: Model Details Expander ---
    all_models = res.get("all_model_results")
    if all_models:
        with st.expander("Model Details", expanded=False):
            st.markdown("Comparison of all evaluated models based on their validation metrics and parameters.")
            
            def format_params(model_name, params):
                if model_name == "Prophet":
                    return (
                        f"CP={params.get('changepoint_prior_scale', '?')}, "
                        f"SP={params.get('seasonality_prior_scale', '?')}, "
                        f"Mode={params.get('seasonality_mode', '?')}, "
                        f"Growth={params.get('growth', '?')}"
                    )
                if model_name == "ARIMA":
                    if isinstance(params, dict):
                        first = next(iter(params.values()), {})
                        return f"p={first.get('p', '?')}, d={first.get('d', '?')}, q={first.get('q', '?')}"
                    return "auto-selected"
                if model_name == "ETS":
                    if isinstance(params, dict):
                        first = next(iter(params.values()), {})
                        return (
                            f"Trend={first.get('trend', 'None')}, "
                            f"Seasonal={first.get('seasonal', 'None')}, "
                            f"Period={first.get('seasonal_periods', 'None')}"
                        )
                    return "auto-selected"
                if model_name in ["Linear Regression", "XGBoost"]:
                    return "feature-based (lag 1/7/30, rolling mean)"
                return str(params)[:50]

            summary_data = []
            for m_name, m_info in all_models.items():
                params = m_info.get("best_params", {})
                param_str = format_params(m_name, params)
                row = {
                    "Model": m_name,
                    "Parameters": param_str,
                }
                # Add all computed metrics
                for metric, score in m_info.get("all_metrics", {}).items():
                    row[metric] = round(score, 6)
                
                summary_data.append(row)
            
            df_summary = pd.DataFrame(summary_data)
            best_model_raw = res.get("best_model", "").replace(" (Best Available)", "")
            df_summary["Model"] = df_summary["Model"].apply(
                lambda x: f"★ {x}" if x == best_model_raw else x
            )

            st.dataframe(df_summary, use_container_width=True, hide_index=True)

    st.subheader("Actual vs Forecast Analysis")
    res = st.session_state.get("results_placeholder") or {}
    df_mapped = st.session_state.get("df_mapped")

    if df_mapped is not None and df_future is not None:
        for s in series_names:
            with st.expander(f"Analysis: {s}", expanded=True):
                # 1. Prepare Chart Data
                # Historical data
                s_hist = df_mapped[df_mapped["series"] == s][["ds", "y"]].rename(columns={"y": "Actual"}).set_index("ds")
                # Future forecast data
                s_fut = df_future[df_future["Metric"] == s][["Date", "Forecast"]].set_index("Date")
                
                # Combine for plotting
                s_plot = s_hist.join(s_fut, how="outer")
                
                last_hist_date = s_hist.index.max()
                first_forecast_date = s_fut.index.min() if not s_fut.empty else None
                gap_days = None
                forecast_x = s_fut.index
                forecast_y = s_fut["Forecast"]
                if first_forecast_date is not None:
                    gap_days = (first_forecast_date - pd.Timestamp(last_hist_date)).days

                    if gap_days <= 3:
                        # Connect smoothly — forecast starts right after historical
                        if last_hist_date in s_plot.index:
                            last_val = s_plot.loc[last_hist_date, "Actual"]
                            s_plot.loc[last_hist_date, "Forecast"] = last_val
                        bridge = pd.DataFrame(
                            {"Forecast": [s_hist["Actual"].iloc[-1]]},
                            index=[s_hist.index.max()],
                        )
                        s_fut_connected = pd.concat([bridge, s_fut])
                        forecast_x = s_fut_connected.index.astype(str)
                        forecast_y = s_fut_connected["Forecast"]
                    else:
                        # Gap exists — do NOT draw connecting line
                        # Leave forecast disconnected from historical so user sees there is a gap period
                        pass

                st.markdown(f"### {s} - Forecast Trend (Historical + Future)")
                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=s_hist.index.astype(str),
                        y=s_hist["Actual"],
                        name="Actual",
                        line=dict(color="#1f77b4", width=2),
                        mode="lines",
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=pd.Index(forecast_x).astype(str),
                        y=forecast_y,
                        name="Forecast",
                        line=dict(color="#ff7f0e", width=2),
                        mode="lines",
                    )
                )

                all_values = pd.concat([s_hist["Actual"], forecast_y]).dropna()
                if not all_values.empty:
                    y_min = all_values.min() * 0.95
                    y_max = all_values.max() * 1.05
                    fig.update_layout(
                        yaxis=dict(
                            range=[y_min, y_max]
                        )
                    )

                if not s_fut.empty:
                    forecast_start = s_fut.index.min()
                    forecast_start_str = str(forecast_start.date())
                    fig.add_shape(
                        type="line",
                        x0=forecast_start_str,
                        x1=forecast_start_str,
                        y0=0,
                        y1=1,
                        xref="x",
                        yref="paper",
                        line=dict(
                            dash="dash",
                            color="gray",
                            width=1,
                        ),
                    )
                    fig.add_annotation(
                        x=forecast_start_str,
                        y=1,
                        xref="x",
                        yref="paper",
                        text="Forecast starts",
                        showarrow=False,
                        xanchor="left",
                        yanchor="bottom",
                        font=dict(size=11, color="gray"),
                    )

                if (
                    best_model_name in ["Linear Regression", "XGBoost"]
                    and forecast_start_date is not None
                    and forecast_end_date is not None
                    and (pd.to_datetime(forecast_end_date) - pd.to_datetime(forecast_start_date)).days > 30
                ):
                    thirty_day_mark = pd.to_datetime(forecast_start_date) + pd.Timedelta(days=30)
                    thirty_day_mark_str = str(thirty_day_mark.date())
                    fig.add_shape(
                        type="line",
                        x0=thirty_day_mark_str,
                        x1=thirty_day_mark_str,
                        y0=0,
                        y1=1,
                        xref="x",
                        yref="paper",
                        line=dict(
                            dash="dot",
                            color="orange",
                            width=1,
                        ),
                    )
                    fig.add_annotation(
                        x=thirty_day_mark_str,
                        y=1,
                        xref="x",
                        yref="paper",
                        text="Reliability drops here",
                        showarrow=False,
                        xanchor="left",
                        yanchor="bottom",
                        font=dict(size=10, color="orange"),
                    )

                if first_forecast_date is not None and gap_days is not None and gap_days > 3:
                    gap_start = (last_hist_date + pd.Timedelta(days=1)).strftime("%d %b %Y")
                    gap_end = first_forecast_date.strftime("%d %b %Y")
                    st.caption(
                        f"⚠️ Gap between {gap_start} and {gap_end} is not forecasted — the model predicts from your selected start date onwards."
                    )

                fig.update_layout(
                    title=f"{s} — Price Forecast",
                    title_font_size=14,
                    hovermode="x unified",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.2,
                        xanchor="left",
                        x=0,
                    ),
                    margin=dict(l=0, r=0, t=40, b=0),
                    height=400,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(
                        showgrid=True,
                        gridcolor="rgba(128,128,128,0.1)",
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor="rgba(128,128,128,0.1)",
                        tickformat=",",
                    ),
                )

                st.plotly_chart(fig, use_container_width=True)

                # 2. Validation Comparison Chart (Actual vs Projected)
                if df_val is not None and not df_val.empty:
                    s_val = df_val[df_val["Metric"] == s].copy()
                    s_actuals_comp = df_mapped[df_mapped["series"] == s][["ds", "y"]]
                    
                    # Merge to get Date, Actual, Projected
                    comp_df = pd.merge(s_actuals_comp, s_val[["Date", "Predicted"]], left_on="ds", right_on="Date", how="inner")
                    comp_df = comp_df.rename(columns={"y": "Actual", "Predicted": "Projected"}).drop(columns=["ds"]).set_index("Date")
                    
                    if not comp_df.empty:
                        st.markdown(f"### {s} - Validation Comparison (Actual vs Projected)")
                        st.line_chart(comp_df, color=["#1f77b4", "#ff7f0e"])

                # 3. Prepare Detailed Table (Actual vs Forecast)
                # We want to show the validation period where we have both Actual and Forecast
                if df_val is not None and not df_val.empty:
                    s_val = df_val[df_val["Metric"] == s].copy()
                    # Get corresponding actuals from df_mapped
                    s_actuals = df_mapped[df_mapped["series"] == s][["ds", "y"]]
                    
                    # Merge to get Date, Actual, Forecast
                    table_df = pd.merge(s_actuals, s_val[["Date", "Predicted"]], left_on="ds", right_on="Date", how="inner")
                    table_df = table_df.rename(columns={"y": "Actual", "Predicted": "Forecast"}).drop(columns=["ds"])
                    table_df["Date"] = pd.to_datetime(table_df["Date"]).dt.strftime("%Y-%m-%d")
                    
                    # Calculate Error
                    table_df["Error"] = table_df["Actual"] - table_df["Forecast"]
                    table_df["Error %"] = (table_df["Error"] / table_df["Actual"]).abs() * 100

                    def get_status(error_pct):
                        if error_pct <= 10:
                            return "✅ Good"
                        if error_pct <= 20:
                            return "⚠️ Fair"
                        return "❌ High"

                    table_df["Status"] = table_df["Error %"].apply(get_status)
                    
                    st.markdown("### Validation Data (Actual vs Predicted)")
                    st.dataframe(
                        table_df.style.format({
                            "Actual": "{:,.2f}",
                            "Forecast": "{:,.2f}",
                            "Error": "{:,.2f}",
                            "Error %": "{:.2f}%"
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No validation data available for table display.")

