"""
Suite 4: Portfolio Optimization & Mathematical Invariants
Verifies weight sums, no short-selling, covariance properties, and Black-Litterman convergence.
"""
import pytest
import numpy as np
from ml.indian_market_data import resolve_data_directory, get_synchronized_market_data, ASSET_METADATA
from ml.portfolio_optimizer import (
    compute_shrinkage_covariance,
    compute_black_litterman,
    optimize_indian_portfolio,
)

@pytest.fixture(scope="module")
def market_df():
    data_dir = resolve_data_directory()
    return get_synchronized_market_data(data_dir)

def test_covariance_positive_definite(market_df):
    """Ledoit-Wolf shrinkage covariance matrix must be strictly positive definite."""
    cov = compute_shrinkage_covariance(market_df)
    assert cov.shape == (4, 4)
    np.testing.assert_allclose(cov, cov.T, atol=1e-10)
    eigenvalues = np.linalg.eigvalsh(cov)
    assert (eigenvalues > 0).all(), f"Found non-positive eigenvalues: {eigenvalues}"

def test_black_litterman_asymptotic_prior_convergence(market_df):
    """Theorem: When view uncertainty Omega -> infinity, posterior return mu_BL converges to market equilibrium prior Pi."""
    asset_names = list(market_df.columns)
    cov = compute_shrinkage_covariance(market_df)
    risk_aversion = 5.0

    infinite_uncertainty_views = {
        a: {"annualized_expected_return": 0.80, "residual_variance_21d": 1e9}
        for a in asset_names
    }

    mu_bl, _ = compute_black_litterman(
        asset_names=asset_names,
        cov_matrix=cov,
        ml_predictions=infinite_uncertainty_views,
        risk_aversion=risk_aversion
    )

    bench_w = np.array([ASSET_METADATA[a]["benchmark_weight"] for a in asset_names])
    bench_w /= bench_w.sum()
    pi = risk_aversion * cov @ bench_w

    np.testing.assert_allclose(mu_bl, pi, rtol=1e-3, atol=1e-3)

@pytest.mark.parametrize("mode", ["black_litterman", "max_sharpe", "min_variance"])
def test_all_optimization_modes_satisfy_invariants(mode):
    """Fundamental Financial Invariants: sum(w) == 1.0 and w_i >= 0 across all modes."""
    res = optimize_indian_portfolio(investment_amount=500000, risk_score=5, optimization_mode=mode)
    weights = [alloc["weight_percent"] / 100.0 for alloc in res["allocations"]]

    assert np.isclose(sum(weights), 1.0, atol=1e-4), f"Weights do not sum to 1 in {mode}!"
    assert all(w >= -1e-6 for w in weights), f"Negative weights detected in {mode}!"
    assert res["expected_annual_volatility_pct"] > 0
    assert res["sharpe_ratio"] is not None

def test_var_99_constraint_satisfaction():
    """Verify that 99% 1-month Value-at-Risk does not exceed user's loss tolerance."""
    loss_tolerance = 0.15  # 15% max loss
    res = optimize_indian_portfolio(
        investment_amount=1000000,
        risk_score=5,
        max_loss_tolerance_pct=loss_tolerance
    )
    assert res["value_at_risk_1m_99pct"] <= (loss_tolerance * 100.0) + 0.10
    assert "SATISFIED" in res["var_99_constraint_status"]

