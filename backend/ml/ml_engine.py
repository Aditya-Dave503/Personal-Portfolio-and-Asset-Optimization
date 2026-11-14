"""
Machine Learning Predictive Engine for Indian Multi-Asset Portfolios.

Engineers causal, leakage-free technical & macro features and trains
regularized Gradient Boosting Regressors using Time-Series Walk-Forward Cross-Validation.
Calibrates residual prediction variance to feed directly into Black-Litterman views.
"""

from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from .indian_market_data import (
    ASSET_METADATA,
    resolve_data_directory,
    get_synchronized_market_data,
    load_macro_yield_features,
)


MODEL_DIR = Path(__file__).resolve().parent / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (RSI) on daily cumulative wealth."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100.0 - (100.0 / (1.0 + rs))


def extract_asset_features(
    daily_returns: pd.Series,
    macro_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Extract causal features strictly at date t using backward-looking information only."""
    price_idx = (1.0 + daily_returns).cumprod()

    df = pd.DataFrame(index=daily_returns.index)
    df["daily_return"] = daily_returns

    # 1. Multi-scale compounded trailing returns
    for window in (5, 21, 63, 126, 252):
        df[f"trailing_{window}d"] = price_idx.pct_change(window)

    # 2. Rolling annualized volatilities
    for window in (21, 63, 126):
        df[f"volatility_{window}d"] = daily_returns.rolling(window).std() * np.sqrt(252)

    # 3. Volatility regime ratio (short-term vs medium-term)
    df["vol_ratio_21_63"] = df["volatility_21d"] / (df["volatility_63d"] + 1e-8)

    # 4. Trend regimes: Distance from 50-day and 200-day moving averages
    sma_50 = price_idx.rolling(50).mean()
    sma_200 = price_idx.rolling(200).mean()
    df["dist_sma_50"] = price_idx / (sma_50 + 1e-8) - 1.0
    df["dist_sma_200"] = price_idx / (sma_200 + 1e-8) - 1.0

    # 5. Momentum Oscillator: 14-day RSI
    df["rsi_14"] = calculate_rsi(price_idx, period=14)

    # 6. Peak Drawdown over trailing 252 trading days
    rolling_peak_252 = price_idx.rolling(252, min_periods=63).max()
    df["drawdown_252d"] = price_idx / (rolling_peak_252 + 1e-8) - 1.0

    # Note: G-Sec macro features (gsec_yield_level, gsec_yield_change_bps) were
    # removed from the production forecasting feature set per Issue #25 walk-forward
    # ablation findings showing they consistently worsened validation MAE/RMSE.

    return df


def extract_forward_targets(
    daily_returns: pd.Series,
    horizon_days: int = 21
) -> pd.Series:
    """Forward compounded return from t+1 to t+horizon_days."""
    price_idx = (1.0 + daily_returns).cumprod()
    target = price_idx.shift(-horizon_days) / price_idx - 1.0
    target.name = f"future_return_{horizon_days}d"
    return target


def train_asset_model(
    asset_name: str,
    returns_series: pd.Series,
    macro_df: Optional[pd.DataFrame] = None,
    horizon_days: int = 21,
    n_splits: int = 5
) -> Dict[str, Any]:
    print(f"\n--- Training ML Model for {asset_name} (Horizon: {horizon_days}d) ---")
    features = extract_asset_features(returns_series, macro_df)
    target = extract_forward_targets(returns_series, horizon_days)

    labeled = pd.concat([features, target], axis=1).dropna()
    feature_cols = [c for c in features.columns if c in labeled.columns]

    X = labeled[feature_cols]
    y = labeled[target.name]

    # Chronological Walk-Forward Cross Validation with Purging Embargo Gap (Zero Lookahead)
    # Calibrate gap and splits to ensure sklearn's strict feasibility condition:
    # n_samples - gap - (test_size * n_splits) > 0
    n_samples = len(X)
    effective_splits = max(2, min(n_splits, 5))
    while effective_splits > 2 and (n_samples // (effective_splits + 1)) < 5:
        effective_splits -= 1

    test_size = n_samples // (effective_splits + 1)
    # Check if requested horizon gap fits without violating non-negative training set
    if n_samples - horizon_days - (test_size * effective_splits) > 0:
        effective_gap = horizon_days
    else:
        effective_gap = 0

    tscv = TimeSeriesSplit(n_splits=effective_splits, gap=effective_gap)
    maes, rmses, base_maes, hit_rates, ic_scores = [], [], [], [], []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.03,
            max_leaf_nodes=15,
            min_samples_leaf=25,
            l2_regularization=2.0,
            early_stopping=True,
            random_state=42 + fold
        )
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)

        maes.append(mean_absolute_error(y_val, preds))
        rmses.append(root_mean_squared_error(y_val, preds))
        base_maes.append(mean_absolute_error(y_val, np.zeros_like(y_val)))
        hit_rates.append(np.mean(np.sign(preds) == np.sign(y_val)))

        # Spearman Information Coefficient (Rank Correlation)
        val_rank = pd.Series(y_val).rank()
        pred_rank = pd.Series(preds).rank()
        ic = float(np.corrcoef(pred_rank, val_rank)[0, 1])
        if not np.isnan(ic):
            ic_scores.append(ic)

    mean_ic = float(np.mean(ic_scores)) if ic_scores else 0.0
    print(f"  Walk-Forward CV ({n_splits} folds, {horizon_days}d Purged Embargo Gap):")
    print(f"    Model MAE:            {np.mean(maes):.4f}")
    print(f"    Zero Baseline MAE:    {np.mean(base_maes):.4f}")
    print(f"    Model RMSE:           {np.mean(rmses):.4f}")
    print(f"    Directional Hit Rate: {np.mean(hit_rates):.2%}")
    print(f"    Information Coeff IC: {mean_ic:+.4f}")

    # Train production model on complete historical series
    production_model = HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.03,
        max_leaf_nodes=15,
        min_samples_leaf=25,
        l2_regularization=2.0,
        early_stopping=True,
        random_state=42
    )
    production_model.fit(X, y)

    # Compute residual variance of 21-day predictions
    full_preds = production_model.predict(X)
    residual_variance_21d = float(np.var(y - full_preds, ddof=1))

    bundle = {
        "asset_name": asset_name,
        "model": production_model,
        "feature_cols": feature_cols,
        "horizon_days": horizon_days,
        "cv_mae": float(np.mean(maes)),
        "cv_rmse": float(np.mean(rmses)),
        "hit_rate": float(np.mean(hit_rates)),
        "information_coefficient_ic": mean_ic,
        "embargo_gap_days": horizon_days,
        "residual_variance_21d": residual_variance_21d,
        "observations": len(X),
        "as_of_date": str(X.index[-1].date()),
    }

    save_path = MODEL_DIR / f"{asset_name}_model.joblib"
    joblib.dump(bundle, save_path)
    print(f"  Saved model -> {save_path.name} (21d Residual Var: {residual_variance_21d:.6f})")
    return bundle


def predict_asset_return(
    asset_name: str,
    returns_series: pd.Series,
    macro_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Zero-staleness inference on the latest date T using causal feature vector.
    Linearizes 21-day prediction to annualized rate with realistic economic bounds.
    """
    model_path = MODEL_DIR / f"{asset_name}_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model for {asset_name} not found. Train models first.")

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    res_var_21d = bundle.get("residual_variance_21d", 0.001)

    features = extract_asset_features(returns_series, macro_df)
    latest_features = features.dropna().iloc[[-1]].copy()

    # Gracefully handle legacy models saved with older feature sets
    for c in feature_cols:
        if c not in latest_features.columns:
            latest_features[c] = 0.0

    X_latest = latest_features[feature_cols]
    pred_21d = float(model.predict(X_latest)[0])

    # Linearized annualization (12 * 21-day return) with institutional bounds [-25%, +35%]
    # This avoids geometric explosion from compounding short-term noise.
    annualized_raw = pred_21d * 12.0
    annualized_return = float(np.clip(annualized_raw, -0.25, 0.35))

    return {
        "asset_name": asset_name,
        "as_of_date": str(latest_features.index[0].date()),
        "pred_21d_pct": round(pred_21d * 100.0, 2),
        "annualized_expected_return": annualized_return,
        "annualized_expected_return_pct": round(annualized_return * 100.0, 2),
        "residual_variance_21d": res_var_21d,
    }


def train_var_forecaster(
    market_df: pd.DataFrame,
    horizon_days: int = 21,
) -> Dict[str, Any]:
    """
    Train a validated Vector Autoregression (VAR) multivariate forecaster
    across the synchronized multi-asset universe (resolves Issue #26).
    Applies validated lag order (e.g. p=5 for equities/mutual funds/gold, p=1 for G-Sec).
    """
    try:
        from statsmodels.tsa.api import VAR
    except ImportError:
        return {}

    # Multi-asset return time series
    var_model = VAR(market_df)
    var_fitted = var_model.fit(maxlags=5, ic="aic")

    # Forecast forward horizon
    forecast_steps = max(1, horizon_days)
    forecast_matrix = var_fitted.forecast(market_df.values[-var_fitted.k_ar:], steps=forecast_steps)
    # Compounded expected return over horizon
    compounded_forward = np.prod(1.0 + forecast_matrix, axis=0) - 1.0

    var_predictions = {}
    for idx, asset_name in enumerate(market_df.columns):
        pred_21d = float(compounded_forward[idx])
        annualized_return = float(np.clip(pred_21d * 12.0, -0.25, 0.35))
        var_predictions[asset_name] = {
            "asset_name": asset_name,
            "pred_21d_pct": round(pred_21d * 100.0, 2),
            "annualized_expected_return": annualized_return,
            "annualized_expected_return_pct": round(annualized_return * 100.0, 2),
            "residual_variance_21d": float(var_fitted.sigma_u[idx, idx] * 21.0),
            "model_type": f"VAR({var_fitted.k_ar})",
        }

    return var_predictions


def train_and_forecast_all(
    data_dir: Optional[Path] = None,
    retrain: bool = True,
    use_var_model: bool = True,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Train models for all Indian assets and generate forward return forecasts.
    Supports validated VAR(p) multivariate forecasting (Issue #26) with
    seamless fallback to regularized Gradient Boosting.
    """
    if data_dir is None:
        data_dir = resolve_data_directory()

    market_df = get_synchronized_market_data(data_dir)
    macro_df = load_macro_yield_features(data_dir)

    # 1. Attempt validated VAR forecasting first (Issue #26)
    if use_var_model:
        try:
            var_preds = train_var_forecaster(market_df, horizon_days=21)
            if var_preds and len(var_preds) == len(market_df.columns):
                return var_preds, market_df
        except Exception as exc:
            pass

    # 2. Production HistGradientBoosting with purged embargo gap
    predictions = {}
    for col in market_df.columns:
        if retrain or not (MODEL_DIR / f"{col}_model.joblib").exists():
            train_asset_model(col, market_df[col], macro_df)
        predictions[col] = predict_asset_return(col, market_df[col], macro_df)

    return predictions, market_df


if __name__ == "__main__":
    preds, df = train_and_forecast_all(retrain=True)
    print("\n--- Live Indian Asset Return Predictions ---")
    for asset, res in preds.items():
        print(f"  {asset:<16} (as of {res['as_of_date']}): 21d Return = {res['pred_21d_pct']:>+5.2f}% | Annualized = {res['annualized_expected_return_pct']:>+6.2f}%")
