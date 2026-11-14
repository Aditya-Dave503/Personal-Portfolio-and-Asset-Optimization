"""
Monte Carlo Goal Simulator.

Implements continuous-time stochastic simulation using Geometric Brownian Motion (GBM)
with discrete monthly compounding and ongoing systematic investment plan (SIP) cash injections:

Dynamics:
    dW_t = (mu * W_t + SIP_annual) * dt + sigma * W_t * dW_t

Discrete Monthly Path:
    W_{m+1} = (W_m + SIP) * exp( (mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z_m )
    where dt = 1 / 12, Z_m ~ N(0, 1).

Key Analytics:
1. Generates N forward wealth trajectory paths (default: 1,000).
2. Calculates exact Probability of Goal Achievement: P(W_T >= Target).
3. Evaluates 10th, 50th (median), and 90th percentiles of the final wealth distribution.
4. Identifies shortfall distributions and expected terminal capital.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


def simulate_goal_paths(
    initial_capital: float,
    target_amount: float,
    horizon_years: int,
    expected_annual_return: float,
    annual_volatility: float,
    monthly_sip: float = 0.0,
    n_simulations: int = 1000,
    random_seed: Optional[int] = 42
) -> Dict[str, Any]:
    """
    Simulate wealth paths via Geometric Brownian Motion with monthly SIP contributions.

    Parameters:
        initial_capital: Starting lump-sum investment W_0 (INR). Must be >= 0.
        target_amount: User's financial goal target capital (INR). Must be > 0.
        horizon_years: Investment runway in years (T). Must be >= 1.
        expected_annual_return: Annual portfolio drift mu (e.g. 0.12 for 12%).
        annual_volatility: Annual portfolio standard deviation sigma (e.g. 0.15 for 15%).
        monthly_sip: Periodic monthly contribution added at start of each month (INR).
        n_simulations: Number of Monte Carlo path iterations (default: 1,000).
        random_seed: Random seed for deterministic simulation.

    Returns:
        Dictionary with:
            - probability_of_success: P(W_T >= Target) in [0, 1]
            - probability_of_success_pct: P(W_T >= Target) * 100
            - final_wealth_percentiles: { 'p10': ..., 'p50': ..., 'p90': ... }
            - median_final_wealth: float
            - expected_final_wealth: float
            - total_invested_capital: initial_capital + monthly_sip * 12 * horizon_years
            - sample_trajectories: Array of 10 sampled representative paths for UI plotting
    """
    if initial_capital < 0:
        raise ValueError("Initial capital cannot be negative.")
    if target_amount <= 0:
        raise ValueError("Target amount must be strictly positive.")
    if horizon_years <= 0:
        raise ValueError("Investment horizon must be at least 1 year.")
    if annual_volatility < 0:
        raise ValueError("Volatility cannot be negative.")

    rng = np.random.default_rng(random_seed)
    n_months = int(horizon_years * 12)
    dt = 1.0 / 12.0

    # Itô drift adjustment: (mu - 0.5 * sigma^2) * dt
    drift = (expected_annual_return - 0.5 * (annual_volatility ** 2)) * dt
    vol_scaled = annual_volatility * np.sqrt(dt)

    # Standard normal increments: shape (n_simulations, n_months)
    shocks = rng.normal(loc=0.0, scale=1.0, size=(n_simulations, n_months))

    # Initialize paths: shape (n_simulations, n_months + 1)
    wealth_paths = np.zeros((n_simulations, n_months + 1), dtype=float)
    wealth_paths[:, 0] = initial_capital

    for m in range(n_months):
        current_w = wealth_paths[:, m] + monthly_sip
        growth_factor = np.exp(drift + vol_scaled * shocks[:, m])
        wealth_paths[:, m + 1] = current_w * growth_factor

    final_wealth = wealth_paths[:, -1]

    # Analytics
    success_count = np.sum(final_wealth >= target_amount)
    prob_success = float(success_count / n_simulations)

    p10 = float(np.percentile(final_wealth, 10))
    p50 = float(np.percentile(final_wealth, 50))
    p90 = float(np.percentile(final_wealth, 90))

    total_invested = float(initial_capital + monthly_sip * n_months)

    # Pick 10 representative paths for frontend sparkline / fan chart visualization
    sample_indices = np.linspace(0, n_simulations - 1, 10, dtype=int)
    time_points = [round(i / 12.0, 2) for i in range(n_months + 1)]
    sample_trajectories = [
        {"path_id": int(idx), "values": [round(float(v), 2) for v in wealth_paths[idx]]}
        for idx in sample_indices
    ]

    return {
        "target_amount": float(target_amount),
        "initial_capital": float(initial_capital),
        "monthly_sip": float(monthly_sip),
        "horizon_years": horizon_years,
        "total_invested_capital": total_invested,
        "n_simulations": n_simulations,
        "probability_of_success": round(prob_success, 4),
        "probability_of_success_pct": round(prob_success * 100.0, 2),
        "percentile_10_wealth": round(p10, 2),
        "percentile_50_wealth": round(p50, 2),
        "percentile_90_wealth": round(p90, 2),
        "mean_final_wealth": round(float(np.mean(final_wealth)), 2),
        "time_grid_years": time_points,
        "sample_trajectories": sample_trajectories,
    }

