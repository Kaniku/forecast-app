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

MODEL_OPTIONS = ["Prophet", "ARIMA", "SARIMA", "ETS", "Linear Regression", "XGBoost"]
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
        "_dedup_count": 0,
        "_df_pre_dedup": None,
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
        df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(io.BytesIO(raw))
    else:
        raise ValueError("Unsupported format. Please upload a .csv or .xlsx file.")
    if df.select_dtypes(include="number").empty:
        raise ValueError(
            "No numeric columns found. Please upload a file with at least one numeric column to use as a forecast metric."
        )
    return df


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
    validation_days = int(min(90, max(7, n * 0.2)))
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
    validation_days = int(min(90, max(7, n * 0.2)))
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


def _fit_predict_sarima_validation(
    df_series: pd.DataFrame,
    *,
    metric_name: str,
) -> tuple[float, pd.DataFrame, Any]:
    """Fit SARIMA(p,d,q)(P,D,Q,7) on train portion and return validation forecast."""
    df_series = df_series.sort_values("ds").reset_index(drop=True).copy()
    df_series["ds"] = pd.to_datetime(df_series["ds"], errors="coerce").dt.normalize()
    df_series["y"] = pd.to_numeric(df_series["y"], errors="coerce").astype(float).clip(lower=0)
    df_series = df_series.dropna(subset=["ds", "y"])
    if df_series.empty:
        return float("inf"), pd.DataFrame(), None

    n = len(df_series)
    validation_days = int(min(90, max(7, n * 0.2)))
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
            seasonal=True,
            m=7,
            start_P=0, max_P=2,
            start_Q=0, max_Q=2,
            D=None,
            stepwise=True,
            information_criterion="aic",
            suppress_warnings=True,
            error_action="ignore",
        )

        forecast_y, conf_int = model.predict(n_periods=validation_days, return_conf_int=True)
        metric = _compute_metric(metric_name, val_y, forecast_y)

        forecast_val = pd.DataFrame({
            "ds": val_ds,
            "yhat": np.clip(forecast_y, 0, None),
            "yhat_lower": np.clip(conf_int[:, 0], 0, None),
            "yhat_upper": np.clip(conf_int[:, 1], 0, None),
        })
    except Exception:
        return float("inf"), pd.DataFrame(), None

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
    validation_days = int(min(90, max(7, n * 0.2)))
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

    n_train = len(train_df)

    # 1. Feature Engineering
    df["trend"] = range(len(df))
    df["day_of_week"] = df["ds"].dt.dayofweek
    df["month"] = df["ds"].dt.month

    # Build lag/rolling features using only training-period y values.
    # Val-period positions are filled with the last known training value so that
    # rolling windows cannot peek into the held-out data.
    y_for_features = df["y"].copy()
    y_for_features.iloc[n_train:] = y_for_features.iloc[n_train - 1]

    df["lag_1"] = y_for_features.shift(1)
    df["lag_7"] = y_for_features.shift(7)
    df["lag_30"] = y_for_features.shift(30)
    df["rolling_mean_7"] = y_for_features.shift(1).rolling(window=7).mean()
    df["rolling_mean_30"] = y_for_features.shift(1).rolling(window=30).mean()

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

    validation_days = int(min(90, max(7, n * 0.2)))
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

    n_train = len(train_df)

    # 1. Feature Engineering (same logic as Linear Regression)
    df["trend"] = range(len(df))
    df["day_of_week"] = df["ds"].dt.dayofweek
    df["month"] = df["ds"].dt.month

    # Build lag/rolling features using only training-period y values.
    # Val-period positions are filled with the last known training value so that
    # rolling windows cannot peek into the held-out data.
    y_for_features = df["y"].copy()
    y_for_features.iloc[n_train:] = y_for_features.iloc[n_train - 1]

    df["lag_1"] = y_for_features.shift(1)
    df["lag_7"] = y_for_features.shift(7)
    df["lag_30"] = y_for_features.shift(30)
    df["rolling_mean_7"] = y_for_features.shift(1).rolling(window=7).mean()
    df["rolling_mean_30"] = y_for_features.shift(1).rolling(window=30).mean()

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

    validation_days = int(min(90, max(7, n * 0.2)))
    validation_days = max(1, validation_days)
    
    train_df = df_series.iloc[: n - validation_days]
    val_df = df_series.iloc[n - validation_days :]

    model, metric, forecast_val = train_xgboost(train_df, val_df, metric_name)
    if model is None:
        return float("inf"), pd.DataFrame(), None
        
    return metric, forecast_val, model


def _run_model_validation(
    model_name: str,
    validation_fn,
    mapped: pd.DataFrame,
    primary_metric: str,
    params_fn,
    message_fn,
) -> "dict | None":
    """Run per-series validation for a model; return a model_results entry or None if every series fails.

    validation_fn: called as validation_fn(df_series, metric_name=primary_metric)
                   must return (score, forecast_val, *extra).
    params_fn:     called as params_fn(*extra); return None to skip storing params for that series.
    message_fn:    called as message_fn(n_series_succeeded) -> str.
    """
    series_names = sorted(mapped["series"].dropna().unique().tolist())
    forecast_parts: list = []
    scores: list = []
    scores_per_series: dict = {}
    params_per_series: dict = {}
    all_y_true: list = []
    all_y_pred: list = []

    for s in series_names:
        try:
            df_series = mapped[mapped["series"] == s]
            result = validation_fn(df_series, metric_name=primary_metric)
            score, forecast_val = result[0], result[1]
            if np.isfinite(score):
                forecast_val = forecast_val.copy()
                forecast_val["series"] = s
                forecast_parts.append(forecast_val)
                scores.append(score)
                scores_per_series[s] = float(score)
                y_actual = df_series[df_series["ds"].isin(forecast_val["ds"])]["y"].to_numpy()
                all_y_true.extend(y_actual)
                all_y_pred.extend(forecast_val["yhat"].to_numpy())
                params = params_fn(*result[2:])
                if params is not None:
                    params_per_series[s] = params
        except Exception as e:
            st.warning(f"{model_name} failed for series {s}: {e}")
            continue

    if not scores:
        return None

    yt = np.array(all_y_true)
    yp = np.array(all_y_pred)
    return {
        "score": float(np.mean(scores)),
        "scores_per_series": scores_per_series,
        "all_metrics": {m: _compute_metric(m, yt, yp) for m in METRIC_OPTIONS},
        "validation_forecast": pd.concat(forecast_parts, ignore_index=True),
        "best_params": params_per_series,
        "message": message_fn(len(scores)),
    }


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


def _fit_sarima_full(
    df_mapped: pd.DataFrame,
    best_params_per_series: dict[str, dict],
    forecast_start_date: Any,
    forecast_end_date: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit SARIMA on full data per series using saved (p,d,q)(P,D,Q,m) params and forecast future."""
    import pmdarima as pm  # type: ignore

    series_names = sorted(df_mapped["series"].dropna().unique().tolist())
    forecast_parts = []
    trained_models: dict[str, Any] = {}

    last_data_date = pd.to_datetime(df_mapped["ds"]).max()
    forecast_horizon = (pd.to_datetime(forecast_end_date) - last_data_date).days
    forecast_horizon = max(1, forecast_horizon)

    for s in series_names:
        df_series = df_mapped[df_mapped["series"] == s].sort_values("ds").reset_index(drop=True)
        y = df_series["y"].to_numpy()
        params = best_params_per_series.get(s, {"p": 1, "d": 1, "q": 0, "P": 1, "D": 1, "Q": 0, "m": 7})

        order = (params.get("p", 1), params.get("d", 1), params.get("q", 0))
        seasonal_order = (params.get("P", 1), params.get("D", 1), params.get("Q", 0), params.get("m", 7))

        try:
            model = pm.ARIMA(order=order, seasonal_order=seasonal_order, suppress_warnings=True)
            model.fit(y)
            trained_models[s] = model
            forecast_y, conf_int = model.predict(n_periods=forecast_horizon, return_conf_int=True)
        except Exception:
            try:
                model = pm.ARIMA(order=(1, 1, 1), seasonal_order=(1, 1, 0, 7), suppress_warnings=True)
                model.fit(y)
                trained_models[s] = model
                forecast_y, conf_int = model.predict(n_periods=forecast_horizon, return_conf_int=True)
            except Exception:
                continue

        future_dates = pd.date_range(
            start=last_data_date + pd.Timedelta(days=1),
            periods=forecast_horizon,
            freq="D",
        )

        forecast_df = pd.DataFrame({
            "ds": future_dates,
            "series": s,
            "yhat": np.clip(forecast_y, 0, None),
            "yhat_lower": np.clip(conf_int[:, 0], 0, None),
            "yhat_upper": np.clip(conf_int[:, 1], 0, None),
        })

        start_ts = pd.to_datetime(forecast_start_date)
        end_ts = pd.to_datetime(forecast_end_date)
        forecast_parts.append(
            forecast_df[(forecast_df["ds"] >= start_ts) & (forecast_df["ds"] <= end_ts)].copy()
        )

    if not forecast_parts:
        return pd.DataFrame(columns=["ds", "series", "yhat", "yhat_lower", "yhat_upper"]), trained_models

    return pd.concat(forecast_parts, ignore_index=True), trained_models


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
        # Clear any previous dedup state from a prior run
        st.session_state["_dedup_count"] = 0
        st.session_state["_df_pre_dedup"] = None
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
                    # Check for duplicate dates per series
                    _dup_sizes = slim.groupby(["series", "ds"]).size()
                    _dup_date_count = int((_dup_sizes > 1).sum())
                    if _dup_date_count > 0:
                        st.session_state["_dedup_count"] = _dup_date_count
                        st.session_state["_df_pre_dedup"] = slim
                        st.session_state["df_mapped"] = None
                        st.session_state["schema_error"] = None
                    else:
                        st.session_state["df_mapped"] = slim
                        st.session_state["schema_error"] = None
        except KeyError as e:
            st.session_state["schema_error"] = f"Column missing: {e}"
            st.session_state["df_mapped"] = None

    # --- Duplicate date resolution ---
    _dedup_count = st.session_state.get("_dedup_count", 0)
    _df_pre_dedup = st.session_state.get("_df_pre_dedup")
    if _dedup_count > 0 and _df_pre_dedup is not None:
        _date_label = "date" if _dedup_count == 1 else "dates"
        st.warning(
            f"⚠️ **{_dedup_count} duplicate {_date_label} found** — the same date appears more than once "
            f"for the same metric. Choose how to resolve them before continuing:"
        )
        _dedup_method = st.selectbox(
            "How should duplicate dates be resolved?",
            options=["Sum", "Average", "Keep last"],
            index=0,
            key="dedup_method_select",
            help=(
                "**Sum:** add values together — correct when a day's data was split across multiple rows "
                "(e.g. orders from two systems). "
                "**Average:** take the mean — correct when a metric was recorded twice by mistake. "
                "**Keep last:** use the final entry for each date — correct when a row was later corrected/overwritten."
            ),
        )
        if st.button("Resolve duplicates & continue", type="primary", key="btn_resolve_dedup"):
            _slim = _df_pre_dedup.copy()
            if _dedup_method == "Sum":
                _slim = _slim.groupby(["series", "ds"], as_index=False)["y"].sum()
            elif _dedup_method == "Average":
                _slim = _slim.groupby(["series", "ds"], as_index=False)["y"].mean()
            else:  # Keep last
                _slim = _slim.groupby(["series", "ds"], as_index=False)["y"].last()
            _slim = _slim.sort_values(["series", "ds"]).reset_index(drop=True)
            st.session_state["df_mapped"] = _slim
            st.session_state["_dedup_count"] = 0
            st.session_state["_df_pre_dedup"] = None
            st.rerun()

    if st.session_state["schema_error"]:
        st.error(_friendly_ui_error(st.session_state["schema_error"]))
    elif st.session_state["df_mapped"] is not None:
        df_mapped = st.session_state["df_mapped"]
        series_count = df_mapped["series"].nunique()
        _day_counts = df_mapped.groupby("series")["ds"].nunique()
        min_days = int(_day_counts.min())
        max_days = int(_day_counts.max())
        _metric_label = "metric" if series_count == 1 else "metrics"
        _history_str = (
            f"{min_days} days of history"
            if min_days == max_days
            else f"{min_days}\u2013{max_days} days of history"
        )
        if min_days < 180:
            st.warning(
                f"\u26A0\uFE0F Data loaded \u2014 {series_count} {_metric_label} detected with {_history_str}. "
                "We recommend at least 180 days for reliable forecasts."
            )
        else:
            st.success(
                f"\u2705 Data loaded successfully \u2014 {series_count} {_metric_label} detected with {_history_str}. "
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
    "You can fill in one, both, or neither.\n\n"
    "⚠️ **These signals are only used by Prophet.** "
    "ARIMA, SARIMA, ETS, Linear Regression, and XGBoost do not use campaign or holiday data."
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
        if start_date < last_data_date:
            st.session_state["forecast_date_error"] = (
                f"Forecast start date must be on or after your last data point ({last_data_date})."
            )
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
    "Prophet": "Best for seasonal data with holidays & campaigns",
    "ARIMA": "Best for stable trends without strong seasonality",
    "SARIMA": "Best for data with repeating seasonal cycles",
    "ETS": "Best for data where recent values matter most",
    "Linear Regression": "Best for simple, steady growth patterns",
    "XGBoost": "Best for complex data with many influencing factors",
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
        "- **Prophet:** Best for data with strong weekly or seasonal patterns. Handles holidays and campaigns natively.\n"
        "- **ARIMA:** Good for stable time series without strong seasonality. Works purely from past values and trends.\n"
        "- **SARIMA:** Like ARIMA but also models repeating seasonal cycles (e.g. higher sales every Friday). Slower to train.\n"
        "- **ETS:** Focuses on weighted averages — gives more importance to recent data, less to older history.\n"
        "- **Linear Regression:** Uses time-based features like day of week and recent values to find patterns. Needs 45+ days.\n"
        "- **XGBoost:** Most powerful for complex patterns. Same features as Linear Regression but uses hundreds of decision trees."
    )

st.session_state["selected_metric"] = st.selectbox(
    "Accuracy metric",
    options=METRIC_OPTIONS,
    index=METRIC_OPTIONS.index(st.session_state.get("selected_metric", "MAPE")),
    key="select_metric",
    format_func=lambda metric: METRIC_DISPLAY_NAMES.get(metric, metric),
)
with st.expander("Which metric should I use?"):
    st.markdown(
        "**MAPE** — Average % error. Easy to compare across datasets. Avoid if data contains values near zero.\n\n"
        "**MAE** — Average absolute error, in the same unit as your data. Good all-rounder.\n\n"
        "**RMSE** — Like MAE but penalises large errors more. Use when big misses matter.\n\n"
        "**MSE** — Squared errors. Use when you want to heavily penalise outliers."
    )
    st.markdown("**MAPE score guide:**")
    st.caption(
        "✅ Excellent (< 10%) · ✅ Good (< 20%) · ⚠️ Fair (< 35%) · ❌ Poor (above 35%)"
    )
    st.caption(
        "Note: If your data contains percentage values or numbers near zero, use MAE or RMSE"
        " — small denominators can inflate MAPE scores."
    )
_ml_selected_pre = any(m in selected_models for m in ["Linear Regression", "XGBoost"])
_pre_fs = st.session_state.get("forecast_start_date")
_pre_fe = st.session_state.get("forecast_end_date")
if (
    _ml_selected_pre
    and _pre_fs is not None
    and _pre_fe is not None
    and not st.session_state.get("forecast_date_error")
    and (pd.to_datetime(_pre_fe) - pd.to_datetime(_pre_fs)).days > 30
):
    _pre_horizon = (pd.to_datetime(_pre_fe) - pd.to_datetime(_pre_fs)).days
    st.warning(
        f"⚠️ Your forecast horizon is {_pre_horizon} days. Linear Regression and XGBoost are most "
        "accurate for short-term forecasts (up to 30 days) — consider switching to Prophet or ARIMA "
        "for longer horizons, or reduce your forecast end date."
    )

st.divider()

# --------------------------------------------------------------------------- 5
st.header("5. Run forecast")
st.markdown("We'll train each selected model, compare their accuracy, and automatically pick the best one for your data.")
st.caption("\u23F1 Estimated run time: Prophet + XGBoost may take 2\u20135 minutes on large datasets or many series. SARIMA is slower than ARIMA due to its larger search space. ARIMA, ETS, and Linear Regression are typically faster.")

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
                    
                    # Single pass: collect actuals/predictions for all metrics and per-series scores
                    all_y_true_p: list = []
                    all_y_pred_p: list = []
                    scores_per_series_prophet = {}
                    for s in mapped["series"].unique():
                        s_mapped = mapped[mapped["series"] == s]
                        n = len(s_mapped)
                        vd = int(min(90, max(7, n * 0.2)))
                        val_y = s_mapped.iloc[n - vd:]["y"].to_numpy()
                        s_fc = validation_forecast[validation_forecast["series"] == s]
                        if not s_fc.empty:
                            s_pred = s_fc["yhat"].to_numpy()
                            all_y_true_p.extend(val_y)
                            all_y_pred_p.extend(s_pred)
                            s_score = _compute_metric(primary_metric, val_y, s_pred)
                            scores_per_series_prophet[s] = float(s_score) if np.isfinite(s_score) else float("inf")

                    all_metric_results = {}
                    if all_y_true_p and all_y_pred_p:
                        for m in METRIC_OPTIONS:
                            all_metric_results[m] = _compute_metric(m, np.array(all_y_true_p), np.array(all_y_pred_p))

                    model_results["Prophet"] = {
                        "score": best_score,
                        "scores_per_series": scores_per_series_prophet,
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
                    text=f"Completed {completed}/{total_steps} — Last finished: Prophet",
                )

        # 2. ARIMA
        if "ARIMA" in selected_models:
            try:
                with st.spinner("Running ARIMA (grid search p,d,q)..."):
                    result = _run_model_validation(
                        "ARIMA",
                        _fit_predict_arima_validation,
                        mapped,
                        primary_metric,
                        lambda model: (
                            {"p": model.order[0], "d": model.order[1], "q": model.order[2]}
                            if hasattr(model, "order")
                            else None
                        ),
                        lambda n: f"ARIMA grid-searched (p,d,q) across {n} series.",
                    )
                    if result:
                        model_results["ARIMA"] = result
            except Exception as e:  # noqa: BLE001
                st.warning(f"ARIMA training failed: {e}")
            finally:
                completed += 1
                progress_bar.progress(
                    completed / total_steps,
                    text=f"Completed {completed}/{total_steps} — Last finished: ARIMA",
                )

        # 3. ETS
        if "ETS" in selected_models:
            try:
                with st.spinner("Running ETS (ExponentialSmoothing grid search)..."):
                    result = _run_model_validation(
                        "ETS",
                        _fit_predict_ets_validation,
                        mapped,
                        primary_metric,
                        lambda _, params: params,
                        lambda n: f"ETS tuned via grid search across {n} series.",
                    )
                    if result:
                        model_results["ETS"] = result
            except Exception as e:  # noqa: BLE001
                st.warning(f"ETS training failed: {e}")
            finally:
                completed += 1
                progress_bar.progress(
                    completed / total_steps,
                    text=f"Completed {completed}/{total_steps} — Last finished: ETS",
                )

        # 4. Linear Regression
        if "Linear Regression" in selected_models:
            try:
                with st.spinner("Running Linear Regression (feature engineering)..."):
                    result = _run_model_validation(
                        "Linear Regression",
                        _fit_predict_lr_validation,
                        mapped,
                        primary_metric,
                        lambda model: model,
                        lambda n: f"Linear Regression trained across {n} series.",
                    )
                    if result:
                        model_results["Linear Regression"] = result
            except Exception as e:  # noqa: BLE001
                st.warning(f"Linear Regression training failed: {e}")
            finally:
                completed += 1
                progress_bar.progress(
                    completed / total_steps,
                    text=f"Completed {completed}/{total_steps} — Last finished: Linear Regression",
                )

        # 5. XGBoost
        if "XGBoost" in selected_models:
            try:
                with st.spinner("Running XGBoost (gradient boosting)..."):
                    result = _run_model_validation(
                        "XGBoost",
                        _fit_predict_xgboost_validation,
                        mapped,
                        primary_metric,
                        lambda model: model,
                        lambda n: f"XGBoost trained across {n} series.",
                    )
                    if result:
                        model_results["XGBoost"] = result
            except Exception as e:  # noqa: BLE001
                st.warning(f"XGBoost training failed: {e}")
            finally:
                completed += 1
                progress_bar.progress(
                    completed / total_steps,
                    text=f"Completed {completed}/{total_steps} — Last finished: XGBoost",
                )

        
        # 6. SARIMA
        if "SARIMA" in selected_models:
            try:
                with st.spinner("Running SARIMA (seasonal ARIMA, m=7)..."):
                    result = _run_model_validation(
                        "SARIMA",
                        _fit_predict_sarima_validation,
                        mapped,
                        primary_metric,
                        lambda model: (
                            {
                                "p": model.order[0], "d": model.order[1], "q": model.order[2],
                                "P": model.seasonal_order[0], "D": model.seasonal_order[1],
                                "Q": model.seasonal_order[2], "m": model.seasonal_order[3],
                            }
                            if hasattr(model, "order") and hasattr(model, "seasonal_order")
                            else None
                        ),
                        lambda n: f"SARIMA (m=7) auto-tuned (p,d,q)(P,D,Q) across {n} series.",
                    )
                    if result:
                        model_results["SARIMA"] = result
            except Exception as e:  # noqa: BLE001
                st.warning(f"SARIMA training failed: {e}")
            finally:
                completed += 1
                progress_bar.progress(
                    completed / total_steps,
                    text=f"Completed {completed}/{total_steps} — Last finished: SARIMA",
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

        # --- Per-series model selection ---
        all_series_for_selection = sorted(mapped["series"].dropna().unique().tolist())

        # Global best (used as fallback and for the ranking/metric display)
        global_best_model = min(model_results, key=lambda k: model_results[k]["score"])
        best_res = model_results[global_best_model]

        # Independently pick the best model for each series
        best_model_per_series = {}
        for _s in all_series_for_selection:
            _best_for_s = None
            _best_score_for_s = float("inf")
            for _mname, _mres in model_results.items():
                _s_score = _mres.get("scores_per_series", {}).get(_s, float("inf"))
                if _s_score < _best_score_for_s:
                    _best_score_for_s = _s_score
                    _best_for_s = _mname
            best_model_per_series[_s] = _best_for_s if _best_for_s is not None else global_best_model

        # Consolidation: collapse to a single name when all series agree
        _unique_best = set(best_model_per_series.values())
        best_model_name = next(iter(_unique_best)) if len(_unique_best) == 1 else global_best_model

        # Build combined validation forecast from each series' own best model
        val_parts = []
        for _s, _mname in best_model_per_series.items():
            _s_val = model_results[_mname].get("validation_forecast", pd.DataFrame())
            if not _s_val.empty and "series" in _s_val.columns:
                _s_part = _s_val[_s_val["series"] == _s].copy()
            elif not _s_val.empty:
                _s_part = _s_val.copy()
                _s_part["series"] = _s
            else:
                _s_part = pd.DataFrame()
            if not _s_part.empty:
                val_parts.append(_s_part)
        validation_forecast = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame()

        # --- Retrain each model on full data for its assigned series ---
        with st.spinner("Retraining model(s) on full data..."):
            future_parts = []
            f_start = st.session_state["forecast_start_date"]
            f_end = st.session_state["forecast_end_date"]

            # Campaign/holiday context (Prophet only, prepared once)
            _cp_state = st.session_state.get("campaign_periods")
            if _cp_state is None:
                _cp_state = pd.DataFrame()
            _cl_state = st.session_state.get("selected_campaign_labels") or []
            _mc_state = st.session_state.get("campaign_manual")
            if _mc_state is None:
                _mc_state = pd.DataFrame()
            _sel_csv = _cp_state[_cp_state["label"].isin(_cl_state)]
            campaign_days = _build_campaign_days(pd.concat([_sel_csv, _mc_state], ignore_index=True), None)
            holidays_df = st.session_state.get("ecommerce_holidays")

            for _mname in _unique_best:
                _series_for_model = [s for s, m in best_model_per_series.items() if m == _mname]
                _mapped_sub = mapped[mapped["series"].isin(_series_for_model)].copy()
                try:
                    if _mname == "Prophet":
                        _params = model_results["Prophet"]["best_params"]
                        _subset_fc, _ = _fit_prophet_full(_mapped_sub, campaign_days, holidays_df, _params, f_start, f_end)
                    elif _mname == "ARIMA":
                        _all_p = model_results["ARIMA"]["best_params"]
                        _subset_p = {s: _all_p[s] for s in _series_for_model if s in _all_p}
                        _subset_fc, _ = _fit_arima_full(_mapped_sub, _subset_p, f_start, f_end)
                    elif _mname == "ETS":
                        _all_p = model_results["ETS"]["best_params"]
                        _subset_p = {s: _all_p[s] for s in _series_for_model if s in _all_p}
                        _subset_fc, _ = _fit_ets_full(_mapped_sub, _subset_p, f_start, f_end)
                    elif _mname == "SARIMA":
                        _all_p = model_results["SARIMA"]["best_params"]
                        _subset_p = {s: _all_p[s] for s in _series_for_model if s in _all_p}
                        _subset_fc, _ = _fit_sarima_full(_mapped_sub, _subset_p, f_start, f_end)
                    elif _mname in ["Linear Regression", "XGBoost"]:
                        _ml_horizon = (pd.to_datetime(f_end) - pd.to_datetime(f_start)).days
                        if _ml_horizon > 30:
                            st.warning(
                                f"⚠️ {_mname} is most accurate for short-term forecasts (up to 30 days). "
                                f"Your selected horizon is {_ml_horizon} days — consider using Prophet or ARIMA for "
                                f"longer forecasts, or reduce your forecast end date."
                            )
                        _all_p = model_results[_mname]["best_params"]
                        _subset_p = {s: _all_p[s] for s in _series_for_model if s in _all_p}
                        _subset_fc = _fit_ml_full(_mapped_sub, _subset_p, f_start, f_end)
                    else:
                        _subset_fc = pd.DataFrame()
                    if _subset_fc is not None and not _subset_fc.empty:
                        future_parts.append(_subset_fc)
                except Exception as _e:
                    st.warning(f"Retraining failed for {_mname} ({', '.join(_series_for_model)}): {_e}")

            future_forecast = pd.concat(future_parts, ignore_index=True) if future_parts else pd.DataFrame()

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
            "best_model_per_series": best_model_per_series,
            "metric_scores": best_res["all_metrics"],
            "all_model_results": model_results,
            "primary_metric": primary_metric,
            "series_names": series_names,
            "validation_df": validation_forecast,
            "forecast_df": future_forecast,
            "message": msg,
        }


st.divider()

# --------------------------------------------------------------------------- 6
import plotly.graph_objects as go

st.header("6. Results")

with st.expander("ℹ️ Known limitations", expanded=False):
    st.markdown(
        """
- **Confidence intervals are approximate** — they use bootstrapped/analytical estimates, not full simulation, so the bands may understate true uncertainty.
- **Long-horizon accuracy degrades** — the models use recursive (iterative) prediction, meaning each step's error compounds. Forecasts beyond 30–60 days should be treated as directional, not precise.
- **Linear Regression and XGBoost are less reliable past 30 days** — these models extrapolate from engineered lag/rolling features that grow stale over time. A dashed reliability line is shown on the chart at the 30-day mark.
- **180-day history is recommended, not required** — shorter datasets will still run but may produce noisier models with wider error ranges.
- **Campaigns and holidays are only used by Prophet** — ARIMA, SARIMA, ETS, Linear Regression, and XGBoost do not incorporate these signals.
- **SARIMA is slower and needs enough data to detect seasonal cycles** — it requires at least 2× the seasonal period (14+ days for weekly patterns) in the training window. On large datasets or many series it can add 1–2 minutes to training time.
- **Per-series model selection** — when multiple metrics are forecast, each series independently picks its best model. A single global winner is shown only when all series agree.
        """
    )

if not st.session_state.get("forecast_run_done"):
    st.info("Run the forecast in step 5 to see the best model, scores, and chart here.")
else:
    res = st.session_state.get("results_placeholder") or {}
    series_names = res.get("series_names") or []
    st.subheader("Best model evaluation")

    primary_m = res.get("primary_metric") or "MAPE"
    primary_val = (res.get("metric_scores") or {}).get(primary_m, 0)
    best_per_series = res.get("best_model_per_series") or {}
    all_model_results = res.get("all_model_results") or {}
    _df_mapped = st.session_state.get("df_mapped")

    if len(best_per_series) <= 1:
        # Single series — simple two-metric display
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
                f"**{primary_m} = {score:,.2f}** — measured in the same unit as your data; lower is better."
            )
    else:
        # Multiple series — one card per series with its own individually computed score
        cols = st.columns(len(best_per_series))
        for i, (series_name, model_name) in enumerate(best_per_series.items()):
            model_info = all_model_results.get(model_name, {})
            val_df = model_info.get("validation_forecast", pd.DataFrame())

            series_score = None
            if not val_df.empty and "series" in val_df.columns and _df_mapped is not None:
                s_val = val_df[val_df["series"] == series_name]
                s_mapped = _df_mapped[_df_mapped["series"] == series_name]
                if not s_val.empty and not s_mapped.empty:
                    s_actual = s_mapped[s_mapped["ds"].isin(s_val["ds"])]["y"].to_numpy()
                    s_pred = s_val["yhat"].to_numpy()
                    if len(s_actual) > 0 and len(s_pred) > 0:
                        min_len = min(len(s_actual), len(s_pred))
                        series_score = _compute_metric(primary_m, s_actual[:min_len], s_pred[:min_len])

            with cols[i]:
                st.markdown(f"**{series_name}**")
                st.metric(label="Best model", value=model_name)
                if series_score is not None:
                    score_display = (
                        f"{series_score:.1%}" if primary_m == "MAPE" else f"{series_score:,.2f}"
                    )
                    st.metric(label=primary_m, value=score_display)
                    if primary_m == "MAPE":
                        if series_score < 0.10:
                            st.success(f"✅ Excellent ({series_score:.1%})")
                        elif series_score < 0.20:
                            st.success(f"✅ Good ({series_score:.1%})")
                        elif series_score < 0.35:
                            st.warning(f"⚠️ Fair ({series_score:.1%})")
                        else:
                            st.error(f"❌ Poor ({series_score:.1%})")
                else:
                    st.caption("Score unavailable")

    if all_model_results and len(all_model_results) > 1 and best_per_series:
        with st.expander("See full model ranking", expanded=False):
            for series_name in best_per_series.keys():
                st.markdown(f"**{series_name}**")

                series_scores: dict = {}
                for _mn, _mi in all_model_results.items():
                    _vdf = _mi.get("validation_forecast", pd.DataFrame())
                    if _vdf.empty or "series" not in _vdf.columns:
                        continue
                    _sv = _vdf[_vdf["series"] == series_name]
                    if _df_mapped is None or _sv.empty:
                        continue
                    _sm = _df_mapped[_df_mapped["series"] == series_name]
                    if _sm.empty:
                        continue
                    _sa = _sm[_sm["ds"].isin(_sv["ds"])]["y"].to_numpy()
                    _sp = _sv["yhat"].to_numpy()
                    if len(_sa) > 0 and len(_sp) > 0:
                        _ml = min(len(_sa), len(_sp))
                        _sc = _compute_metric(primary_m, _sa[:_ml], _sp[:_ml])
                        if np.isfinite(_sc):
                            series_scores[_mn] = _sc

                if not series_scores:
                    st.caption("No valid scores available for this series.")
                    continue

                sorted_series_models = sorted(series_scores.items(), key=lambda x: x[1])
                best_series_score = sorted_series_models[0][1]

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

                for rank, (mdl_name, mdl_score) in enumerate(sorted_series_models, 1):
                    medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"{rank}.")
                    is_winner = mdl_name == best_per_series.get(series_name)
                    if mdl_score == best_series_score:
                        diff_str, diff_color = "← best", "green"
                    else:
                        pct_worse = ((mdl_score - best_series_score) / best_series_score) * 100
                        diff_str, diff_color = f"+{pct_worse:.0f}% worse", "red"

                    col_rank, col_name, col_score, col_diff = st.columns([0.5, 2, 1.5, 1.5])
                    with col_rank:
                        st.write(medal)
                    with col_name:
                        st.write(f"**{mdl_name}**" if is_winner else mdl_name)
                    with col_score:
                        st.write(f"{mdl_score:.1%}" if primary_m == "MAPE" else f"{mdl_score:,.2f}")
                    with col_diff:
                        st.markdown(f":{diff_color}[{diff_str}]")

                if len(sorted_series_models) > 1:
                    best_name, best_score_val = sorted_series_models[0]
                    runner_name, runner_score = sorted_series_models[1]
                    if runner_score > 0:
                        improvement = ((runner_score - best_score_val) / runner_score) * 100
                        st.caption(
                            f"🏆 {best_name} is {improvement:.0f}% more accurate than {runner_name} for {series_name}."
                        )

                st.markdown("---")
    elif all_model_results and len(all_model_results) <= 1:
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
    _bm_ps_warn = res.get("best_model_per_series") or {}
    _uses_ml = (
        best_model_name in ["Linear Regression", "XGBoost"]
        or any(m in ["Linear Regression", "XGBoost"] for m in _bm_ps_warn.values())
    )
    forecast_start_date = st.session_state.get("forecast_start_date")
    forecast_end_date = st.session_state.get("forecast_end_date")
    if (
        _uses_ml
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
                    table_df["Error %"] = (table_df["Error"] / table_df["Actual"].replace(0, np.nan)).abs() * 100

                    def get_status(error_pct):
                        if pd.isna(error_pct):
                            return ""
                        if error_pct <= 10:
                            return "✅ Good"
                        if error_pct <= 20:
                            return "⚠️ Fair"
                        return "❌ High"

                    table_df["Status"] = table_df["Error %"].apply(get_status)

                    st.markdown("### Validation Data (Actual vs Predicted)")
                    st.dataframe(
                        table_df.style.format(
                            {
                                "Actual": "{:,.2f}",
                                "Forecast": "{:,.2f}",
                                "Error": "{:,.2f}",
                                "Error %": "{:.2f}%",
                            },
                            na_rep="-",
                        ),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No validation data available for table display.")

