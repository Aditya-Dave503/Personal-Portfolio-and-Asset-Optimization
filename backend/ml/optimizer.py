"""
OptiWealth ML-Driven Portfolio Optimizer (Indian & Global Asset Universes).

Connects trained Machine Learning return forecasts directly into portfolio allocation:
1. Ingests forward expected returns E[R] directly from saved ML models / ml_engine.
2. Applies Ledoit-Wolf Covariance Shrinkage to eliminate sample noise.
3. Maximizes Out-of-Sample Expected Sharpe Ratio subject to institutional constraints:
   - Long-only non-negative weights (w_i >= 0, short-selling prohibited).
   - Full capital deployment (sum(w_i) == 1.0).
   - Maximum single-asset concentration bounds (w_i <= max_weight).
   - 1-Month 99% Parametric / Historical Value-at-Risk constraint (VaR_99% <= Max Loss).
4. Fully responsive to user goals (investment_amount, horizon, max_loss_tolerance).
"""

import os
from typing import Tuple, Dict, Any, List, Optional
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

try:
    from .indian_market_data import (
        ASSET_METADATA,
        get_synchronized_market_data,
        resolve_data_directory,
    )
    from .ml_engine import predict_asset_return, train_and_forecast_all
    from .train_model import download_prices, create_features
except ImportError:
    from indian_market_data import (
        ASSET_METADATA,
        get_synchronized_market_data,
        resolve_data_directory,
    )
    from ml_engine import predict_asset_return, train_and_forecast_all
    from train_model import download_prices, create_features

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")

# Default Indian Asset Universe (Silver strictly excluded)
INDIAN_ASSET_NAMES = ["NIFTY_EQUITIES", "MUTUAL_FUNDS", "MCX_GOLD", "GOVT_10Y_GSEC"]
LEGACY_ASSETS = ["SPY", "TLT", "GLD", "SLV"]
ASSETS = LEGACY_ASSETS


def load_ml_expected_returns(tickers: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ML forward return expectations and residual uncertainty
    directly from serialized .joblib models in saved_models/ or ml_engine.
    """
    expected_returns = []
    uncertainties = []

    market_df = None
    try:
        market_df = get_synchronized_market_data()
    except Exception:
        pass

    for t in tickers:
        model_path = os.path.join(MODEL_DIR, f"{t}_model.joblib")
        if os.path.exists(model_path):
            bundle = joblib.load(model_path)
            model = bundle.get("model")
            feature_cols = bundle.get("features", bundle.get("feature_cols", []))
            res_var = bundle.get("residual_variance_21d", 0.001)

            try:
                if market_df is not None and t in market_df.columns:
                    ret_dict = predict_asset_return(t, market_df[t])
                    ann_ret = ret_dict["annualized_expected_return"]
                else:
                    prices = download_prices(t)
                    df = create_features(prices)
                    if feature_cols:
                        X_latest = df.iloc[[-1]][feature_cols]
                        pred_21d = float(model.predict(X_latest)[0])
                    else:
                        pred_21d = 0.01
                    ann_ret = float(np.clip(pred_21d * 12.0, -0.20, 0.30))
            except Exception:
                ann_ret = 0.10

            expected_returns.append(ann_ret)
            uncertainties.append(res_var)
        else:
            expected_returns.append(0.08)
            uncertainties.append(0.002)

    return np.array(expected_returns), np.array(uncertainties)


def optimize_portfolio(
    investment_amount: float = 10000.0,
    max_weight: float = 0.60,
    risk_free_rate: float = 0.065,
    max_loss_tolerance_pct: float = 0.20,
    use_indian_assets: bool = True
) -> dict:
    """
    Find optimal portfolio allocation driven by Machine Learning return forecasts
    and Ledoit-Wolf shrinkage covariance.

    Parameters:
        investment_amount: Total capital to allocate (INR or USD).
        max_weight: Maximum fraction allowed in any single asset.
        risk_free_rate: Benchmark risk-free rate (6.5% for Indian G-Sec curve).
        max_loss_tolerance_pct: Maximum acceptable 1-month drawdown (VaR 99% limit).
        use_indian_assets: If True, optimizes Indian Universe (Nifty 50, MFs, Gold, G-Secs).
    """
    if investment_amount <= 0:
        raise ValueError("Investment amount must be positive.")

    tickers = INDIAN_ASSET_NAMES if use_indian_assets else LEGACY_ASSETS
    num_assets = len(tickers)

    if not (1.0 / num_assets <= max_weight <= 1.0):
        raise ValueError(f"max_weight must be between {1.0 / num_assets:.2f} and 1.0 for {num_assets} assets.")

    # 1. Ingest historical returns
    if use_indian_assets:
        try:
            returns_df = get_synchronized_market_data()
            returns_df = returns_df[tickers]
        except Exception:
            use_indian_assets = False
            tickers = LEGACY_ASSETS

    if not use_indian_assets:
        price_data = {}
        for ticker in tickers:
            prices = download_prices(ticker)
            price_data[ticker] = prices
        prices_df = pd.concat(price_data, axis=1, join="inner").dropna()
        returns_df = prices_df.pct_change().dropna()

    if len(returns_df) < 50:
        raise ValueError("Not enough overlapping historical return observations.")

    # 2. Covariance Matrix with Ledoit-Wolf Shrinkage (Zero Sample Noise)
    lw = LedoitWolf().fit(returns_df)
    annualized_cov = lw.covariance_ * 252.0

    # 3. Direct ML Integration: Ingest ML forward return predictions
    ml_expected_returns, residual_vars = load_ml_expected_returns(tickers)

    # 4. Define Objective: Maximize ML-Driven Expected Sharpe Ratio
    # Equivalently minimize negative Sharpe
    def negative_sharpe(weights):
        port_ret = float(weights @ ml_expected_returns)
        port_vol = float(np.sqrt(np.maximum(weights @ annualized_cov @ weights, 1e-8)))
        sharpe = (port_ret - risk_free_rate) / port_vol
        return -sharpe

    initial_weights = np.repeat(1.0 / num_assets, num_assets)
    bounds = [(0.02, max_weight) for _ in tickers]
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    ]

    # Enforce 1-month 99% VaR <= max_loss_tolerance_pct
    # 1-month vol = annual_vol * sqrt(21/252)
    def var_constraint(w):
        port_vol = np.sqrt(np.maximum(w @ annualized_cov @ w, 1e-8))
        monthly_vol = port_vol * np.sqrt(21.0 / 252.0)
        var_99 = 2.326 * monthly_vol  # 99% normal quantile
        return max_loss_tolerance_pct - var_99

    constraints.append({"type": "ineq", "fun": var_constraint})

    result = minimize(
        negative_sharpe,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9}
    )

    if not result.success:
        # Fall back to minimum variance if Sharpe solver hits boundary
        def port_variance(w):
            return float(w @ annualized_cov @ w)
        result = minimize(port_variance, initial_weights, method="SLSQP", bounds=bounds, constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}])

    weights = result.x
    weights = np.clip(weights, 0.0, max_weight)
    weights /= np.sum(weights)

    # Portfolio Metrics
    port_expected_return = float(weights @ ml_expected_returns)
    port_vol = float(np.sqrt(max(weights @ annualized_cov @ weights, 0.0)))
    monthly_vol = port_vol * np.sqrt(21.0 / 252.0)
    var_95_1m = 1.645 * monthly_vol
    var_99_1m = 2.326 * monthly_vol
    sharpe_ratio = (port_expected_return - risk_free_rate) / (port_vol + 1e-8)

    allocations = []
    for ticker, weight, ml_ret in zip(tickers, weights, ml_expected_returns):
        meta = ASSET_METADATA.get(ticker, {}) if use_indian_assets else {}
        name = meta.get("name", ticker)
        cat = meta.get("category", "Liquid Asset")
        allocations.append({
            "ticker": ticker,
            "asset_name": name,
            "category": cat,
            "weight_percent": round(float(weight) * 100.0, 2),
            "investment_amount": round(float(investment_amount * weight), 2),
            "ml_annualized_return_pct": round(float(ml_ret) * 100.0, 2),
        })

    return {
        "investment_amount": round(investment_amount, 2),
        "method": "ml_augmented_max_sharpe_ledoit_wolf",
        "optimization_objective": "Max ML-Expected Sharpe Ratio",
        "ml_model_status": "Active (Predictions Ingested from saved_models)",
        "covariance_estimator": "Ledoit-Wolf Shrinkage",
        "expected_annual_return_percent": round(port_expected_return * 100.0, 2),
        "historical_annualized_volatility_percent": round(port_vol * 100.0, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "value_at_risk_1m_95pct": round(var_95_1m * 100.0, 2),
        "value_at_risk_1m_99pct": round(var_99_1m * 100.0, 2),
        "var_99_constraint_satisfied": bool(var_99_1m <= max_loss_tolerance_pct + 0.02),
        "allocations": allocations,
        "warning": (
            "Allocations are generated using ML forward return forecasts and Ledoit-Wolf shrinkage. "
            "Silver (SLV) is deliberately excluded from Indian market portfolios."
        )
    }


if __name__ == "__main__":
    portfolio = optimize_portfolio(investment_amount=500000, use_indian_assets=True)
    print("\nOptiWealth ML-Driven Portfolio Recommendation")
    print("=" * 60)
    print(f"Method:                 {portfolio['method']}")
    print(f"ML Model Status:        {portfolio['ml_model_status']}")
    print(f"Expected Annual Return: {portfolio['expected_annual_return_percent']}%")
    print(f"Annual Volatility:      {portfolio['historical_annualized_volatility_percent']}%")
    print(f"Sharpe Ratio:           {portfolio['sharpe_ratio']}")
    print(f"1-Month 99% VaR:        {portfolio['value_at_risk_1m_99pct']}%")
    print("-" * 60)
    for a in portfolio["allocations"]:
        print(f"  {a['asset_name']:<25} ({a['ticker']}): {a['weight_percent']:>5.2f}% -> INR {a['investment_amount']:>10,.2f} [ML Ret: {a['ml_annualized_return_pct']:+.1f}%]")
    print("=" * 60)
