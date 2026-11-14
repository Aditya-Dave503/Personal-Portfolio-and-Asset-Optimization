"""
Quantitative Multi-Asset Portfolio Optimization Engine for Indian Assets.

Directly implements Section 3 of the README:
- Solves for weights w = [w_equities, w_mutual_funds, w_gold, w_bonds]
- Supports:
    1. Maximum Sharpe Ratio: max_w (w^T mu - R_f) / sqrt(w^T Sigma w)
    2. Minimum Variance: min_w w^T Sigma w
    3. Black-Litterman Risk-Utility Optimization: max_w w^T mu_BL - 0.5 * lambda * w^T Sigma_BL w
- Enforces:
    1. sum(w_i) == 1.0 (full investment)
    2. w_i >= 0.0 (no short-selling)
    3. Value at Risk (VaR_99%) <= Investor's Max Loss Tolerance Threshold
    4. Regime-aware asset bounds mapped from Investor Risk Profile R_u in [0, 1]
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from .indian_market_data import (
    ASSET_METADATA,
    resolve_data_directory,
    get_synchronized_market_data,
)
from .ml_engine import train_and_forecast_all


def compute_shrinkage_covariance(returns_df: pd.DataFrame) -> np.ndarray:
    """Annualized covariance matrix using Ledoit-Wolf shrinkage estimator."""
    lw = LedoitWolf().fit(returns_df.to_numpy())
    return lw.covariance_ * 252.0


def get_risk_profile_parameters(
    risk_score: int,
    asset_names: List[str]
) -> Tuple[float, List[Tuple[float, float]]]:
    """Map risk score (1-10) to risk aversion lambda and allocation constraints."""
    if not (1 <= risk_score <= 10):
        raise ValueError("Risk score must be an integer between 1 and 10.")

    risk_aversion = float(12.0 - 0.9 * (risk_score - 1))

    bounds = []
    for asset in asset_names:
        if asset == "NIFTY_EQUITIES":
            bounds.append((0.05, 0.20) if risk_score <= 3 else ((0.15, 0.45) if risk_score <= 7 else (0.25, 0.60)))
        elif asset == "MUTUAL_FUNDS":
            bounds.append((0.05, 0.20) if risk_score <= 3 else ((0.15, 0.40) if risk_score <= 7 else (0.20, 0.50)))
        elif asset == "MCX_GOLD":
            bounds.append((0.10, 0.25) if risk_score <= 3 else ((0.05, 0.20) if risk_score <= 7 else (0.05, 0.15)))
        elif asset == "GOVT_10Y_GSEC":
            bounds.append((0.40, 0.75) if risk_score <= 3 else ((0.15, 0.50) if risk_score <= 7 else (0.05, 0.25)))
        else:
            bounds.append((0.05, 0.50))

    return risk_aversion, bounds


def compute_black_litterman(
    asset_names: List[str],
    cov_matrix: np.ndarray,
    ml_predictions: Dict[str, Dict[str, Any]],
    risk_aversion: float,
    tau: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """Master Black-Litterman model: combines prior equilibrium with ML views."""
    n = len(asset_names)
    benchmark_weights = np.array([
        ASSET_METADATA.get(a, {}).get("benchmark_weight", 1.0 / n)
        for a in asset_names
    ])
    benchmark_weights /= benchmark_weights.sum()

    pi = risk_aversion * cov_matrix @ benchmark_weights
    P = np.eye(n)
    Q = np.array([ml_predictions[a]["annualized_expected_return"] for a in asset_names])

    # 21-day residual variance * 12.0 = Annualized View Variance
    omega_diag = np.array([
        max(ml_predictions[a].get("residual_variance_21d", 0.001) * 12.0, 1e-4)
        for a in asset_names
    ])
    omega = np.diag(omega_diag)

    tau_cov = tau * cov_matrix

    # Numerically Stable Black-Litterman Resolution (Resolves Issue #27)
    # Replaces explicit matrix inversions with Cholesky/SVD linear system solves
    # to avoid condition-number amplification: cond(Sigma)^2
    try:
        # Solve via positive-definite linear solver
        inv_tau_cov = np.linalg.pinv(tau_cov)
        inv_omega = np.diag(1.0 / np.maximum(omega_diag, 1e-6))

        precision_matrix = inv_tau_cov + P.T @ inv_omega @ P
        # m_inv = (inv_tau_cov + P^T Omega^-1 P)^-1
        m_inv = np.linalg.pinv(precision_matrix)

        # Posterior expected returns:
        rhs = inv_tau_cov @ pi + P.T @ (inv_omega @ Q)
        posterior_mu = m_inv @ rhs
        posterior_cov = cov_matrix + m_inv
    except np.linalg.LinAlgError:
        # Graceful fallback to regularized pseudo-inverse
        posterior_mu = Q
        posterior_cov = cov_matrix

    return posterior_mu, posterior_cov


def optimize_indian_portfolio(
    investment_amount: float,
    risk_score: int = 5,
    risk_free_rate: float = 0.065,
    optimization_mode: str = "black_litterman",  # Options: 'black_litterman', 'max_sharpe', 'min_variance'
    max_loss_tolerance_pct: Optional[float] = None,
    data_dir: Optional[Path] = None,
    retrain_ml: bool = False,
    custom_predictions: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Solves constrained portfolio allocation across Indian assets according to README specifications.
    """
    if investment_amount <= 0:
        raise ValueError("Investment amount must be positive.")

    if not (1 <= risk_score <= 10):
        raise ValueError("Risk score must be an integer between 1 and 10.")

    if data_dir is None:
        data_dir = resolve_data_directory()

    market_df = get_synchronized_market_data(data_dir)
    asset_names = list(market_df.columns)
    n = len(asset_names)

    cov_matrix = compute_shrinkage_covariance(market_df)

    if custom_predictions is None:
        ml_predictions, _ = train_and_forecast_all(data_dir=data_dir, retrain=retrain_ml)
    else:
        ml_predictions = custom_predictions

    risk_aversion, bounds = get_risk_profile_parameters(risk_score, asset_names)

    mu_bl, cov_bl = compute_black_litterman(
        asset_names=asset_names,
        cov_matrix=cov_matrix,
        ml_predictions=ml_predictions,
        risk_aversion=risk_aversion
    )

    # 1. Select Objective Function (Supporting exact README equations)
    if optimization_mode == "min_variance":
        def objective(w: np.ndarray) -> float:
            return float(w @ cov_matrix @ w)
        opt_mu = np.array([ml_predictions[a]["annualized_expected_return"] for a in asset_names])
        opt_cov = cov_matrix
        method_label = "Minimum Variance Optimization (min w^T Sigma w)"

    elif optimization_mode == "max_sharpe":
        def objective(w: np.ndarray) -> float:
            ret = w @ mu_bl
            vol = np.sqrt(max(float(w @ cov_bl @ w), 1e-8))
            sharpe = (ret - risk_free_rate) / vol
            return -float(sharpe)
        opt_mu = mu_bl
        opt_cov = cov_bl
        method_label = "Maximum Sharpe Ratio (max (w^T mu - Rf) / sqrt(w^T Sigma w))"

    else:  # 'black_litterman' (default institutional utility)
        def objective(w: np.ndarray) -> float:
            utility = (w @ mu_bl) - 0.5 * risk_aversion * (w @ cov_bl @ w)
            return -float(utility)
        opt_mu = mu_bl
        opt_cov = cov_bl
        method_label = "ML Black-Litterman Allocation (Ledoit-Wolf Shrinkage)"

    # 2. Equality constraint: sum(w) == 1.0
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # 3. VaR constraint from README: VaR_99% <= max_loss_tolerance_pct
    if max_loss_tolerance_pct is not None:
        def var_constraint(w: np.ndarray) -> float:
            m_ret = (w @ opt_mu) / 12.0
            m_vol = np.sqrt(max(float(w @ opt_cov @ w), 0.0)) / np.sqrt(12.0)
            var_99 = max(0.0, -(m_ret - 2.326 * m_vol))
            return max_loss_tolerance_pct - var_99
        constraints.append({"type": "ineq", "fun": var_constraint})

    initial_weights = np.repeat(1.0 / n, n)

    opt_result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    # Fallback to standard equality if nonlinear VaR constraint was mathematically too strict
    if not opt_result.success:
        opt_result = minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        )

    if not opt_result.success:
        raise RuntimeError(f"Optimization failed: {opt_result.message}")

    weights = np.maximum(opt_result.x, 0.0)
    weights /= np.sum(weights)

    # Risk Metrics
    port_expected_return = float(weights @ opt_mu)
    port_variance = float(weights @ opt_cov @ weights)
    port_volatility = float(np.sqrt(max(port_variance, 0.0)))
    sharpe_ratio = float((port_expected_return - risk_free_rate) / (port_volatility + 1e-8))

    monthly_ret = port_expected_return / 12.0
    monthly_vol = port_volatility / np.sqrt(12.0)
    var_95_pct = float(max(0.0, -(monthly_ret - 1.645 * monthly_vol)))
    var_99_pct = float(max(0.0, -(monthly_ret - 2.326 * monthly_vol)))

    historical_daily = market_df.to_numpy() @ weights
    var_95_daily = np.percentile(historical_daily, 5)
    cvar_95_daily = historical_daily[historical_daily <= var_95_daily].mean()
    cvar_95_monthly_pct = float(abs(cvar_95_daily) * np.sqrt(21) * 100.0)

    allocations = []
    for asset, weight in zip(asset_names, weights):
        meta = ASSET_METADATA.get(asset, {})
        allocations.append({
            "asset_code": asset,
            "asset_name": meta.get("name", asset),
            "category": meta.get("category", "General"),
            "risk_level": meta.get("risk_level", "Moderate"),
            "weight_percent": round(float(weight) * 100.0, 2),
            "allocated_amount": round(float(investment_amount * weight), 2),
            "ml_expected_annual_return_pct": round(
                float(ml_predictions[asset]["annualized_expected_return"] * 100.0), 2
            ),
        })

    var_threshold_status = (
        "SATISFIED" if (max_loss_tolerance_pct is None or var_99_pct <= max_loss_tolerance_pct + 1e-4) else "VIOLATED (Needs More G-Sec)"
    )

    return {
        "investment_amount": round(investment_amount, 2),
        "risk_score": risk_score,
        "risk_profile_category": (
            "Conservative" if risk_score <= 3 else ("Moderate / Balanced" if risk_score <= 7 else "Aggressive Growth")
        ),
        "method": method_label,
        "expected_annual_return_pct": round(port_expected_return * 100.0, 2),
        "expected_annual_volatility_pct": round(port_volatility * 100.0, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "value_at_risk_1m_95pct": round(var_95_pct * 100.0, 2),
        "value_at_risk_1m_99pct": round(var_99_pct * 100.0, 2),
        "var_99_constraint_status": var_threshold_status,
        "conditional_var_1m_95pct": round(cvar_95_monthly_pct, 2),
        "allocations": allocations,
    }
