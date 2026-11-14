
"""
Time-Series Model Comparison
============================

Models:
    1. ARIMAX (SARIMAX with exogenous technical features)
    2. VAR (joint daily-return forecasting)
    3. GARCH(1,1) (conditional volatility forecasting)

Run from backend:
    python -m validation.compare_time_series_models

MLflow experiment:
    Portfolio_TimeSeries_Models

Outputs:
    outputs/time_series_comparison/
"""

from pathlib import Path
import json
import tempfile
import time
import warnings

import numpy as np
import pandas as pd
import mlflow

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR
from arch import arch_model

from ml.indian_market_data import get_synchronized_market_data
from ml.ml_engine import (
    extract_asset_features,
    extract_forward_targets,
)
from ml.tracking import setup_mlflow


# =========================================================
# CONFIGURATION
# =========================================================

BACKEND_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR = BACKEND_DIR / "outputs" / "time_series_comparison"

EXPERIMENT_NAME = "Portfolio_TimeSeries_Models"

HORIZON = 21
N_SPLITS = 5
VALIDATION_GAP = 21

ARIMAX_ORDER = (1, 0, 1)
VAR_LAGS = 2

GARCH_P = 1
GARCH_Q = 1

RANDOM_STATE = 42


# =========================================================
# COMMON METRICS
# =========================================================

def calculate_return_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "directional_accuracy": float(
            np.mean(np.sign(actual) == np.sign(predicted))
        ),
        "prediction_bias": float(np.mean(predicted - actual)),
    }


def calculate_volatility_metrics(actual_variance, predicted_variance):
    """
    Metrics for conditional variance forecasts.

    QLIKE:
        log(forecast variance)
        + actual variance / forecast variance

    A constant term is omitted because it does not affect
    model comparisons.
    """

    actual = np.asarray(actual_variance, dtype=float)
    predicted = np.asarray(predicted_variance, dtype=float)

    epsilon = 1e-12

    predicted = np.maximum(predicted, epsilon)
    actual = np.maximum(actual, epsilon)

    qlike = np.mean(
        np.log(predicted) + actual / predicted
    )

    return {
        "volatility_mae": float(
            mean_absolute_error(
                np.sqrt(actual),
                np.sqrt(predicted),
            )
        ),
        "volatility_rmse": float(
            np.sqrt(
                mean_squared_error(
                    np.sqrt(actual),
                    np.sqrt(predicted),
                )
            )
        ),
        "qlike": float(qlike),
    }


# =========================================================
# DATA PREPARATION
# =========================================================

def prepare_asset_data(market, asset):
    """
    Prepare technical features and 21-observation
    forward compounded returns for one asset.
    """

    daily_returns = market[asset]

    features = extract_asset_features(daily_returns)

    targets = extract_forward_targets(
        daily_returns,
        horizon_days=HORIZON,
    )

    target_name = targets.name

    data = pd.concat(
        [features, targets],
        axis=1,
    ).dropna()

    feature_cols = list(features.columns)

    X = data[feature_cols]
    y = data[target_name]

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite features found for {asset}")

    if not np.isfinite(y.to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite targets found for {asset}")

    return X, y


def prepare_all_targets(market):
    """
    Build a common-date forward-return target DataFrame
    for all four assets.
    """

    targets = {}

    for asset in market.columns:
        targets[asset] = extract_forward_targets(
            market[asset],
            horizon_days=HORIZON,
        )

    target_df = pd.DataFrame(targets).dropna()

    return target_df


# =========================================================
# 1. ARIMAX
# =========================================================

def run_arimax_fold(
    X,
    y,
    train_idx,
    valid_idx,
):
    """
    ARIMAX using a positional index.

    Statsmodels cannot reliably generate future dates from
    irregular trading-date indexes without frequency metadata.
    We therefore use NumPy arrays and a RangeIndex internally.

    Exogenous features are scaled using training data only.
    """

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    train_end = train_idx[-1]
    valid_end = valid_idx[-1]

    # Forecast across the purge gap and validation period.
    future_idx = np.arange(train_end + 1, valid_end + 1)

    X_future = X.iloc[future_idx]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_future_scaled = scaler.transform(X_future)

    # Use positional arrays instead of irregular DatetimeIndex.
    y_train_array = y_train.to_numpy(dtype=float)

    model = SARIMAX(
        endog=y_train_array,
        exog=X_train_scaled,
        order=ARIMAX_ORDER,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    start = time.perf_counter()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        fitted = model.fit(
            disp=False,
            maxiter=200,
        )

    training_seconds = time.perf_counter() - start

    forecast = fitted.get_forecast(
        steps=len(future_idx),
        exog=X_future_scaled,
    )

    all_predictions = np.asarray(
        forecast.predicted_mean,
        dtype=float,
    )

    # Remove predictions belonging to the 21-observation gap.
    validation_offset = valid_idx[0] - train_end - 1

    predictions = all_predictions[
        validation_offset:
        validation_offset + len(valid_idx)
    ]

    actual = y.iloc[valid_idx].to_numpy(dtype=float)

    if len(predictions) != len(actual):
        raise ValueError(
            "ARIMAX prediction and validation lengths do not match."
        )

    return actual, predictions, training_seconds


# =========================================================
# 2. VAR
# =========================================================

def run_var_fold(
    market,
    target_df,
    train_idx,
    valid_idx,
):
    """
    Fit VAR on daily returns through the training endpoint.

    For each validation forecast origin:
        - Use observed daily returns through that date.
        - Forecast the next 21 daily returns.
        - Compound forecasts into a 21-day return.

    VAR parameters remain fixed within the fold.
    """

    train_end_date = target_df.index[train_idx[-1]]

    training_returns = market.loc[:train_end_date].copy()

    if len(training_returns) <= VAR_LAGS + 5:
        raise ValueError("Insufficient data to fit VAR.")

    model = VAR(training_returns)

    start = time.perf_counter()

    fitted = model.fit(
        maxlags=VAR_LAGS,
        ic=None,
        trend="c",
    )

    training_seconds = time.perf_counter() - start

    predictions = []
    actuals = []
    dates = []

    for idx in valid_idx:

        forecast_date = target_df.index[idx]

        # Observed information available at this forecast origin.
        observed_history = market.loc[:forecast_date].to_numpy(
            dtype=float
        )

        if len(observed_history) < fitted.k_ar:
            raise ValueError("Insufficient VAR history.")

        daily_forecasts = fitted.forecast(
            y=observed_history[-fitted.k_ar:],
            steps=HORIZON,
        )

        # Convert daily return forecasts into compounded
        # 21-observation forward return forecasts.
        compounded = np.prod(
            1.0 + daily_forecasts,
            axis=0,
        ) - 1.0

        predictions.append(compounded)

        actuals.append(
            target_df.loc[forecast_date].to_numpy(dtype=float)
        )

        dates.append(forecast_date)

    predictions = np.asarray(predictions)
    actuals = np.asarray(actuals)

    return actuals, predictions, dates, training_seconds


# =========================================================
# 3. GARCH(1,1)
# =========================================================

def forecast_garch_variance(
    omega,
    alpha,
    beta,
    last_variance,
    last_residual,
    horizon,
):
    """
    Expected future conditional variance under GARCH(1,1).

    sigma²(t+1) = omega + alpha * residual(t)²
                        + beta * sigma²(t)

    For subsequent steps, expected squared residuals
    equal conditional variance.
    """

    variance = (
        omega
        + alpha * last_residual ** 2
        + beta * last_variance
    )

    total_variance = variance

    for _ in range(1, horizon):
        variance = (
            omega
            + (alpha + beta) * variance
        )

        total_variance += variance

    return total_variance


def run_garch_fold(
    daily_returns,
    train_idx,
    valid_idx,
):
    """
    Fit GARCH(1,1) on training returns.

    Parameters remain fixed within the fold.

    Conditional variance is updated using observed daily
    returns through each validation forecast origin.

    The predicted target is cumulative 21-day variance.
    """

    train_end_date = daily_returns.index[train_idx[-1]]

    training_returns = (
        daily_returns.loc[:train_end_date] * 100.0
    )

    # GARCH requires a reasonably sized training sample.
    if len(training_returns) < 100:
        raise ValueError("Insufficient GARCH training observations.")

    model = arch_model(
        training_returns,
        mean="Zero",
        vol="GARCH",
        p=GARCH_P,
        q=GARCH_Q,
        dist="normal",
        rescale=False,
    )

    start = time.perf_counter()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        fitted = model.fit(
            disp="off",
            show_warning=False,
        )

    training_seconds = time.perf_counter() - start

    params = fitted.params

    omega = float(params["omega"])
    alpha = float(params["alpha[1]"])
    beta = float(params["beta[1]"])

    # Conditional variance at the end of the training period.
    last_variance = float(
        fitted.conditional_volatility.iloc[-1] ** 2
    )

    # Zero-mean GARCH: residual equals observed return.
    last_residual = float(training_returns.iloc[-1])

    # Process the observations between training end and
    # the final validation origin.
    all_dates = daily_returns.index

    first_position = all_dates.get_loc(train_end_date)

    final_validation_position = all_dates.get_loc(
        daily_returns.index[valid_idx[-1]]
    )

    validation_positions = set(valid_idx.tolist())

    predicted_variances = []
    realized_variances = []
    prediction_dates = []

    for position in range(
        first_position + 1,
        final_validation_position + 1,
    ):

        current_date = all_dates[position]

        current_return = float(
            daily_returns.iloc[position] * 100.0
        )

        # Update conditional variance after observing
        # the return at the current date.
        next_variance = (
            omega
            + alpha * last_residual ** 2
            + beta * last_variance
        )

        last_variance = next_variance
        last_residual = current_return

        if position not in validation_positions:
            continue

        # Forecast cumulative variance over the next 21 days.
        forecast_variance_scaled = forecast_garch_variance(
            omega=omega,
            alpha=alpha,
            beta=beta,
            last_variance=last_variance,
            last_residual=last_residual,
            horizon=HORIZON,
        )

        # Convert percentage-return variance to decimal units.
        forecast_variance = (
            forecast_variance_scaled / (100.0 ** 2)
        )

        # Realized cumulative variance over the next 21 days.
        future_returns = daily_returns.iloc[
            position + 1:
            position + HORIZON + 1
        ].to_numpy(dtype=float)

        if len(future_returns) != HORIZON:
            continue

        realized_variance = float(
            np.sum(future_returns ** 2)
        )

        predicted_variances.append(forecast_variance)
        realized_variances.append(realized_variance)
        prediction_dates.append(current_date)

    return (
        np.asarray(realized_variances),
        np.asarray(predicted_variances),
        prediction_dates,
        training_seconds,
    )


# =========================================================
# MLflow LOGGING HELPERS
# =========================================================

def log_dataframe_artifact(df, filename):
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / filename

        df.to_csv(path, index=False)

        mlflow.log_artifact(str(path))


def log_model_run(
    run_name,
    params,
    metrics,
    fold_df,
    prediction_df,
):
    with mlflow.start_run(
        run_name=run_name,
        nested=True,
    ):

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        log_dataframe_artifact(
            fold_df,
            "fold_metrics.csv",
        )

        log_dataframe_artifact(
            prediction_df,
            "out_of_fold_predictions.csv",
        )


# =========================================================
# MAIN EXPERIMENT
# =========================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    setup_mlflow()

    mlflow.set_experiment(EXPERIMENT_NAME)

    market = get_synchronized_market_data().sort_index()

    if market.index.has_duplicates:
        raise ValueError("Duplicate market dates detected.")

    if not market.index.is_monotonic_increasing:
        raise ValueError("Market dates are not sorted.")

    if market.isna().any().any():
        raise ValueError("Missing market returns detected.")

    if not np.isfinite(market.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite market returns detected.")

    assets = list(market.columns)

    print("\nTIME-SERIES MODEL COMPARISON")
    print("--------------------------------")
    print("Assets:", assets)
    print("Observations:", len(market))
    print("Horizon:", HORIZON)
    print("Folds:", N_SPLITS)
    print("Gap:", VALIDATION_GAP)

    all_fold_rows = []
    all_prediction_rows = []
    all_summary_rows = []

    # -----------------------------------------------------
    # PREPARE COMMON RETURN TARGETS
    # -----------------------------------------------------

    target_df = prepare_all_targets(market)

    # One common split definition for all assets and VAR.
    splitter = TimeSeriesSplit(
        n_splits=N_SPLITS,
        gap=VALIDATION_GAP,
    )

    splits = list(splitter.split(target_df))

    # -----------------------------------------------------
    # PARENT MLFLOW RUN
    # -----------------------------------------------------

    with mlflow.start_run(
        run_name="arimax_var_garch_comparison"
    ) as parent_run:

        mlflow.log_params({
            "forecast_horizon": HORIZON,
            "n_splits": N_SPLITS,
            "validation_gap": VALIDATION_GAP,
            "assets": ",".join(assets),
            "arimax_order": str(ARIMAX_ORDER),
            "var_lags": VAR_LAGS,
            "garch_order": f"({GARCH_P},{GARCH_Q})",
            "validation": "Expanding TimeSeriesSplit",
        })

        # =================================================
        # ARIMAX: FOUR ASSETS
        # =================================================

        for asset in assets:

            print(f"\nARIMAX: {asset}")

            X, y = prepare_asset_data(
                market,
                asset,
            )

            # Generate folds using this asset's actual feature dates.
            # Technical indicators have rolling warm-up periods.
            asset_splits = list(
                TimeSeriesSplit(
                    n_splits=N_SPLITS,
                    gap=VALIDATION_GAP,
                ).split(X)
            )

            asset_fold_rows = []
            asset_prediction_rows = []

            for fold, (train_idx, valid_idx) in enumerate(
                asset_splits,
                start=1,
            ):

                actual, predicted, train_seconds = (
                    run_arimax_fold(
                        X,
                        y,
                        train_idx,
                        valid_idx,
                    )
                )

                metrics = calculate_return_metrics(
                    actual,
                    predicted,
                )

                validation_dates = X.index[valid_idx]

                row = {
                    "model": "ARIMAX",
                    "asset": asset,
                    "fold": fold,
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "directional_accuracy":
                        metrics["directional_accuracy"],
                    "prediction_bias": metrics["prediction_bias"],
                    "training_seconds": train_seconds,
                    "validation_start": str(
                        validation_dates[0].date()
                    ),
                    "validation_end": str(
                        validation_dates[-1].date()
                    ),
                }

                asset_fold_rows.append(row)
                all_fold_rows.append(row)

                for date, a, p in zip(
                    validation_dates,
                    actual,
                    predicted,
                ):
                    prediction_row = {
                        "model": "ARIMAX",
                        "asset": asset,
                        "fold": fold,
                        "date": str(date.date()),
                        "actual": float(a),
                        "predicted": float(p),
                        "residual": float(a - p),
                    }

                    asset_prediction_rows.append(
                        prediction_row
                    )

                    all_prediction_rows.append(
                        prediction_row
                    )

                print(
                    f"Fold {fold}: "
                    f"MAE={metrics['mae']:.5f}, "
                    f"RMSE={metrics['rmse']:.5f}"
                )

            asset_fold_df = pd.DataFrame(asset_fold_rows)
            asset_prediction_df = pd.DataFrame(
                asset_prediction_rows
            )

            pooled_actual = asset_prediction_df["actual"].to_numpy()
            pooled_predicted = asset_prediction_df[
                "predicted"
            ].to_numpy()

            pooled = calculate_return_metrics(
                pooled_actual,
                pooled_predicted,
            )

            summary = {
                "model": "ARIMAX",
                "asset": asset,
                "fold_mae_mean": float(
                    asset_fold_df["mae"].mean()
                ),
                "fold_rmse_mean": float(
                    asset_fold_df["rmse"].mean()
                ),
                "pooled_mae": pooled["mae"],
                "pooled_rmse": pooled["rmse"],
                "directional_accuracy_mean": float(
                    asset_fold_df[
                        "directional_accuracy"
                    ].mean()
                ),
                "pooled_directional_accuracy":
                    pooled["directional_accuracy"],
                "mean_training_seconds": float(
                    asset_fold_df["training_seconds"].mean()
                ),
            }

            all_summary_rows.append(summary)

            log_model_run(
                run_name=f"ARIMAX_{asset}",
                params={
                    "model": "SARIMAX",
                    "asset": asset,
                    "order": str(ARIMAX_ORDER),
                    "forecast_horizon": HORIZON,
                    "feature_scaling": "StandardScaler",
                },
                metrics={
                    "fold_mae_mean": summary["fold_mae_mean"],
                    "fold_rmse_mean": summary["fold_rmse_mean"],
                    "pooled_mae": pooled["mae"],
                    "pooled_rmse": pooled["rmse"],
                    "directional_accuracy_mean":
                        summary["directional_accuracy_mean"],
                    "pooled_directional_accuracy":
                        pooled["directional_accuracy"],
                    "mean_training_seconds":
                        summary["mean_training_seconds"],
                },
                fold_df=asset_fold_df,
                prediction_df=asset_prediction_df,
            )

        # =================================================
        # VAR: FOUR ASSETS JOINTLY
        # =================================================

        print("\nVAR: JOINT FOUR-ASSET MODEL")

        var_fold_rows = []
        var_prediction_rows = []

        for fold, (train_idx, valid_idx) in enumerate(
            splits,
            start=1,
        ):

            actual, predicted, dates, train_seconds = (
                run_var_fold(
                    market=market,
                    target_df=target_df,
                    train_idx=train_idx,
                    valid_idx=valid_idx,
                )
            )

            for asset_idx, asset in enumerate(assets):

                asset_actual = actual[:, asset_idx]
                asset_predicted = predicted[:, asset_idx]

                metrics = calculate_return_metrics(
                    asset_actual,
                    asset_predicted,
                )

                row = {
                    "model": "VAR",
                    "asset": asset,
                    "fold": fold,
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "directional_accuracy":
                        metrics["directional_accuracy"],
                    "prediction_bias": metrics["prediction_bias"],
                    "training_seconds": train_seconds,
                    "validation_start": str(dates[0].date()),
                    "validation_end": str(dates[-1].date()),
                }

                var_fold_rows.append(row)
                all_fold_rows.append(row)

                for date, a, p in zip(
                    dates,
                    asset_actual,
                    asset_predicted,
                ):

                    prediction_row = {
                        "model": "VAR",
                        "asset": asset,
                        "fold": fold,
                        "date": str(date.date()),
                        "actual": float(a),
                        "predicted": float(p),
                        "residual": float(a - p),
                    }

                    var_prediction_rows.append(
                        prediction_row
                    )

                    all_prediction_rows.append(
                        prediction_row
                    )

                print(
                    f"Fold {fold} | {asset}: "
                    f"MAE={metrics['mae']:.5f}"
                )

        var_fold_df = pd.DataFrame(var_fold_rows)
        var_prediction_df = pd.DataFrame(
            var_prediction_rows
        )

        for asset in assets:

            asset_folds = var_fold_df[
                var_fold_df["asset"] == asset
            ].copy()

            asset_predictions = var_prediction_df[
                var_prediction_df["asset"] == asset
            ].copy()

            pooled = calculate_return_metrics(
                asset_predictions["actual"],
                asset_predictions["predicted"],
            )

            summary = {
                "model": "VAR",
                "asset": asset,
                "fold_mae_mean": float(
                    asset_folds["mae"].mean()
                ),
                "fold_rmse_mean": float(
                    asset_folds["rmse"].mean()
                ),
                "pooled_mae": pooled["mae"],
                "pooled_rmse": pooled["rmse"],
                "directional_accuracy_mean": float(
                    asset_folds[
                        "directional_accuracy"
                    ].mean()
                ),
                "pooled_directional_accuracy":
                    pooled["directional_accuracy"],
                "mean_training_seconds": float(
                    asset_folds["training_seconds"].mean()
                ),
            }

            all_summary_rows.append(summary)

            log_model_run(
                run_name=f"VAR_{asset}",
                params={
                    "model": "VAR",
                    "asset": asset,
                    "var_lags": VAR_LAGS,
                    "assets_in_joint_model": ",".join(assets),
                    "forecast_horizon": HORIZON,
                },
                metrics={
                    "fold_mae_mean": summary["fold_mae_mean"],
                    "fold_rmse_mean": summary["fold_rmse_mean"],
                    "pooled_mae": pooled["mae"],
                    "pooled_rmse": pooled["rmse"],
                    "directional_accuracy_mean":
                        summary["directional_accuracy_mean"],
                    "pooled_directional_accuracy":
                        pooled["directional_accuracy"],
                    "mean_training_seconds":
                        summary["mean_training_seconds"],
                },
                fold_df=asset_folds,
                prediction_df=asset_predictions,
            )

        # =================================================
        # GARCH: FOUR ASSETS
        # =================================================

        for asset in assets:

            print(f"\nGARCH: {asset}")

            daily_returns = market[asset]

            asset_fold_rows = []
            asset_prediction_rows = []

            for fold, (train_idx, valid_idx) in enumerate(
                splits,
                start=1,
            ):

                realized, predicted, dates, train_seconds = (
                    run_garch_fold(
                        daily_returns=daily_returns,
                        train_idx=train_idx,
                        valid_idx=valid_idx,
                    )
                )

                metrics = calculate_volatility_metrics(
                    realized,
                    predicted,
                )

                row = {
                    "model": "GARCH",
                    "asset": asset,
                    "fold": fold,
                    "volatility_mae": metrics["volatility_mae"],
                    "volatility_rmse": metrics["volatility_rmse"],
                    "qlike": metrics["qlike"],
                    "training_seconds": train_seconds,
                    "validation_start": str(dates[0].date()),
                    "validation_end": str(dates[-1].date()),
                }

                asset_fold_rows.append(row)
                all_fold_rows.append(row)

                for date, actual_var, predicted_var in zip(
                    dates,
                    realized,
                    predicted,
                ):

                    prediction_row = {
                        "model": "GARCH",
                        "asset": asset,
                        "fold": fold,
                        "date": str(date.date()),
                        "actual_variance": float(actual_var),
                        "predicted_variance": float(predicted_var),
                        "actual_volatility": float(
                            np.sqrt(actual_var)
                        ),
                        "predicted_volatility": float(
                            np.sqrt(predicted_var)
                        ),
                    }

                    asset_prediction_rows.append(
                        prediction_row
                    )

                    all_prediction_rows.append(
                        prediction_row
                    )

                print(
                    f"Fold {fold}: "
                    f"Vol MAE={metrics['volatility_mae']:.6f}, "
                    f"QLIKE={metrics['qlike']:.5f}"
                )

            asset_fold_df = pd.DataFrame(asset_fold_rows)
            asset_prediction_df = pd.DataFrame(
                asset_prediction_rows
            )

            pooled_vol_metrics = calculate_volatility_metrics(
                asset_prediction_df["actual_variance"],
                asset_prediction_df["predicted_variance"],
            )

            summary = {
                "model": "GARCH",
                "asset": asset,
                "fold_volatility_mae_mean": float(
                    asset_fold_df["volatility_mae"].mean()
                ),
                "fold_volatility_rmse_mean": float(
                    asset_fold_df["volatility_rmse"].mean()
                ),
                "pooled_volatility_mae":
                    pooled_vol_metrics["volatility_mae"],
                "pooled_volatility_rmse":
                    pooled_vol_metrics["volatility_rmse"],
                "fold_qlike_mean": float(
                    asset_fold_df["qlike"].mean()
                ),
                "pooled_qlike": pooled_vol_metrics["qlike"],
                "mean_training_seconds": float(
                    asset_fold_df["training_seconds"].mean()
                ),
            }

            all_summary_rows.append(summary)

            log_model_run(
                run_name=f"GARCH_{asset}",
                params={
                    "model": "GARCH",
                    "asset": asset,
                    "p": GARCH_P,
                    "q": GARCH_Q,
                    "distribution": "normal",
                    "forecast_horizon": HORIZON,
                },
                metrics={
                    "fold_volatility_mae_mean":
                        summary["fold_volatility_mae_mean"],
                    "fold_volatility_rmse_mean":
                        summary["fold_volatility_rmse_mean"],
                    "pooled_volatility_mae":
                        pooled_vol_metrics["volatility_mae"],
                    "pooled_volatility_rmse":
                        pooled_vol_metrics["volatility_rmse"],
                    "fold_qlike_mean": summary["fold_qlike_mean"],
                    "pooled_qlike": pooled_vol_metrics["qlike"],
                    "mean_training_seconds":
                        summary["mean_training_seconds"],
                },
                fold_df=asset_fold_df,
                prediction_df=asset_prediction_df,
            )

        # =================================================
        # SAVE CONSOLIDATED RESULTS
        # =================================================

        summary_df = pd.DataFrame(all_summary_rows)
        fold_df = pd.DataFrame(all_fold_rows)
        prediction_df = pd.DataFrame(all_prediction_rows)

        summary_path = (
            OUTPUT_DIR / "time_series_summary.csv"
        )

        fold_path = (
            OUTPUT_DIR / "time_series_fold_metrics.csv"
        )

        prediction_path = (
            OUTPUT_DIR / "time_series_predictions.csv"
        )

        metadata_path = (
            OUTPUT_DIR / "time_series_metadata.json"
        )

        summary_df.to_csv(summary_path, index=False)
        fold_df.to_csv(fold_path, index=False)
        prediction_df.to_csv(prediction_path, index=False)

        metadata = {
            "experiment": EXPERIMENT_NAME,
            "parent_run_id": parent_run.info.run_id,
            "assets": assets,
            "models": [
                "ARIMAX",
                "VAR",
                "GARCH",
            ],
            "forecast_horizon": HORIZON,
            "n_splits": N_SPLITS,
            "validation_gap": VALIDATION_GAP,
            "arimax_order": ARIMAX_ORDER,
            "var_lags": VAR_LAGS,
            "garch_order": [GARCH_P, GARCH_Q],
            "market_start": str(market.index.min().date()),
            "market_end": str(market.index.max().date()),
            "production_code_modified": False,
        }

        metadata_path.write_text(
            json.dumps(metadata, indent=4),
            encoding="utf-8",
        )

        mlflow.log_artifact(str(summary_path))
        mlflow.log_artifact(str(fold_path))
        mlflow.log_artifact(str(prediction_path))
        mlflow.log_artifact(str(metadata_path))

    print("\n===================================")
    print("TIME-SERIES EXPERIMENT COMPLETED")
    print("===================================")

    print("\nSummary:")
    print(summary_df.to_string(index=False))

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    print("\nMLflow experiment:")
    print(EXPERIMENT_NAME)


if __name__ == "__main__":
    main()
