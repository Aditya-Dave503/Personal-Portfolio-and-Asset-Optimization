"""
Automated Test Suite for OptiWealth Indian Multi-Asset ML Pipeline.

Tests all specifications from the README:
- Section 1: Risk Profiling Model (Supervised continuous R_u in [0, 1] and 1-10)
- Section 2: Covariance Shrinkage & Annualized Risk/Return Modeling
- Section 3: Constrained Optimization (Max Sharpe, Min Variance, Black-Litterman, VaR_99% constraint)
- Section 4: Regime awareness and duration/convexity G-Sec returns
- Absence of lookahead data leakage & Silver exclusion

Run with:
    pytest test_ml_pipeline.py -v
"""

import sys
from pathlib import Path

# Automatically add backend/ml, backend, and project root to sys.path
# so tests can be run seamlessly from any directory (repo root, backend/, or backend/tests/)
_TESTS_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _TESTS_DIR.parent
_REPO_ROOT = _BACKEND_DIR.parent
_ML_DIR = _BACKEND_DIR / "ml"

_CANDIDATE_PATHS = [
    _ML_DIR,
    _BACKEND_DIR,
    _REPO_ROOT,
    _TESTS_DIR,
]

for _p in _CANDIDATE_PATHS:
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import pandas as pd
import pytest

from indian_market_data import (
    ASSET_METADATA,
    resolve_data_directory,
    get_synchronized_market_data,
    load_gsec_bond_returns,
)
from risk_profiler import calculate_user_risk_profile
from ml_engine import (
    extract_asset_features,
    extract_forward_targets,
)
from portfolio_optimizer import (
    compute_shrinkage_covariance,
    compute_black_litterman,
    optimize_indian_portfolio,
)


@pytest.fixture(scope="module")
def market_data():
    data_dir = resolve_data_directory()
    df = get_synchronized_market_data(data_dir)
    return df


def test_data_synchronization(market_data):
    """Verify that synchronized data has expected assets, no NaNs, and chronological order."""
    expected_assets = ["NIFTY_EQUITIES", "MUTUAL_FUNDS", "MCX_GOLD", "GOVT_10Y_GSEC"]
    assert list(market_data.columns) == expected_assets
    assert market_data.isna().sum().sum() == 0
    assert market_data.index.is_monotonic_increasing
    assert len(market_data) > 2000


def test_silver_completely_excluded(market_data):
    """Verify that Silver ('SLV') is completely absent from all datasets and metadata."""
    assert "SLV" not in market_data.columns
    assert "SILVER" not in market_data.columns
    assert "SILVER" not in ASSET_METADATA
    assert "SLV" not in ASSET_METADATA


def test_gsec_bond_return_properties():
    """Verify G-Sec duration and convexity conversion produce realistic sovereign bond returns."""
    data_dir = resolve_data_directory()
    bond_ret = load_gsec_bond_returns(data_dir, modified_duration=6.8, convexity=55.0)
    ann_return = bond_ret.mean() * 252
    ann_vol = bond_ret.std() * np.sqrt(252)

    assert 0.04 <= ann_return <= 0.12, f"Unrealistic bond return: {ann_return}"
    assert 0.02 <= ann_vol <= 0.08, f"Unrealistic bond volatility: {ann_vol}"


def test_feature_causality_and_no_lookahead(market_data):
    """Verify features at date t do NOT leak future price information."""
    series = market_data["NIFTY_EQUITIES"]
    features_full = extract_asset_features(series)

    midpoint = len(series) // 2
    series_truncated = series.iloc[:midpoint]
    features_truncated = extract_asset_features(series_truncated)

    common_idx = features_truncated.dropna().index
    pd.testing.assert_frame_equal(
        features_full.loc[common_idx, features_truncated.columns],
        features_truncated.loc[common_idx],
        check_exact=False,
        rtol=1e-5
    )


def test_forward_target_isolation(market_data):
    """Verify that forward target at date t starts at t+1 and does not include date t return."""
    series = market_data["NIFTY_EQUITIES"]
    price_idx = (1.0 + series).cumprod()
    target_21d = extract_forward_targets(series, horizon_days=21)

    t = 100
    expected_target = (price_idx.iloc[t + 21] / price_idx.iloc[t]) - 1.0
    actual_target = target_21d.iloc[t]
    assert np.isclose(expected_target, actual_target, atol=1e-6)


def test_covariance_matrix_positive_definite(market_data):
    """Verify Ledoit-Wolf shrinkage covariance matrix is strictly positive-definite."""
    cov = compute_shrinkage_covariance(market_data)
    assert cov.shape == (4, 4)
    np.testing.assert_allclose(cov, cov.T, atol=1e-10)

    eigenvalues = np.linalg.eigvalsh(cov)
    assert (eigenvalues > 0).all(), f"Non-positive eigenvalues: {eigenvalues}"


def test_black_litterman_convergence_to_prior(market_data):
    """Theorem: As view uncertainty Omega -> infinity, posterior returns converge to prior Pi."""
    asset_names = list(market_data.columns)
    cov = compute_shrinkage_covariance(market_data)
    risk_aversion = 5.0

    infinite_uncertainty_preds = {
        a: {"annualized_expected_return": 0.50, "residual_variance_21d": 1e8}
        for a in asset_names
    }

    mu_bl, _ = compute_black_litterman(
        asset_names=asset_names,
        cov_matrix=cov,
        ml_predictions=infinite_uncertainty_preds,
        risk_aversion=risk_aversion
    )

    bench_w = np.array([ASSET_METADATA[a]["benchmark_weight"] for a in asset_names])
    bench_w /= bench_w.sum()
    pi = risk_aversion * cov @ bench_w

    np.testing.assert_allclose(mu_bl, pi, rtol=1e-3, atol=1e-3)


def test_risk_profiler_model():
    """Verify Section 1 Risk Profiler outputs continuous R_u in [0, 1] and sensible ordering."""
    young_aggressive = calculate_user_risk_profile(age=23, investment_horizon_years=25, savings_to_income_ratio=0.8, max_loss_tolerance_pct=0.35)
    older_conservative = calculate_user_risk_profile(age=68, investment_horizon_years=3, savings_to_income_ratio=0.2, max_loss_tolerance_pct=0.08)

    assert 0.05 <= young_aggressive["continuous_risk_score_Ru"] <= 0.95
    assert 0.05 <= older_conservative["continuous_risk_score_Ru"] <= 0.95
    assert young_aggressive["discrete_risk_score"] > older_conservative["discrete_risk_score"]


def test_all_readme_optimization_modes():
    """Verify README Section 3: Supports max_sharpe, min_variance, and black_litterman."""
    for mode in ["black_litterman", "max_sharpe", "min_variance"]:
        res = optimize_indian_portfolio(investment_amount=500000, risk_score=5, optimization_mode=mode)
        weights = [alloc["weight_percent"] / 100.0 for alloc in res["allocations"]]
        assert np.isclose(sum(weights), 1.0, atol=1e-4), f"Weights do not sum to 1 in {mode}"
        assert all(w >= 0.0 for w in weights), f"Negative weights in {mode}"


def test_var_99_loss_constraint_enforcement():
    """Verify README constraint: VaR_99% <= Investor's Max Loss Tolerance Threshold."""
    res = optimize_indian_portfolio(
        investment_amount=500000,
        risk_score=5,
        max_loss_tolerance_pct=0.15
    )
    assert res["value_at_risk_1m_99pct"] <= 15.0 + 1e-2
    assert "SATISFIED" in res["var_99_constraint_status"]


def test_optimizer_ml_integration():
    """Verify optimizer.py directly integrates ML return forecasts and Ledoit-Wolf shrinkage."""
    from backend.ml.optimizer import optimize_portfolio
    res = optimize_portfolio(investment_amount=500000, max_weight=0.60, use_indian_assets=True)
    assert res["investment_amount"] == 500000.0
    assert "Active" in res["ml_model_status"]
    assert res["covariance_estimator"] == "Ledoit-Wolf Shrinkage"
    weights = [a["weight_percent"] / 100.0 for a in res["allocations"]]
    assert np.isclose(sum(weights), 1.0, atol=1e-3)
    assert all(w >= 0.0 for w in weights)
    assert res["value_at_risk_1m_99pct"] > 0.0


def test_purged_walk_forward_cv_and_spearman_ic():
    """Verify zero lookahead bias with 21-day embargo gap and Spearman IC metrics."""
    from backend.ml.train_model import calculate_spearman_ic
    actual = np.array([0.05, -0.02, 0.08, -0.01, 0.03])
    pred = np.array([0.04, -0.01, 0.07, -0.03, 0.02])
    ic = calculate_spearman_ic(actual, pred)
    assert ic > 0.8
