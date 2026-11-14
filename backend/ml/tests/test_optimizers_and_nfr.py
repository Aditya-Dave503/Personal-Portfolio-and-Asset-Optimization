"""
Phase 5, Phase 6 & Phase 7 Portfolio Optimizers, Goal Simulator, Pipeline & NFR Suite:
- Phase 5: Ledoit-Wolf Shrinkage Covariance SPD, Black-Litterman Asymptotic Theorems 1 & 2,
           SLSQP Simplex & Long-Only Invariants across 3 Modes, Risk-Tier Box Constraints,
           and Parametric VaR_99% / CVaR_95% Enforcement.
- Phase 6: Monte Carlo Goal-Based Optimizer (Stock Weight Caps, Dirichlet Capped Random Weights,
           Mixed Stock/Fund Candidate Generation, and Zero-Return Exact Wealth Identity).
- Phase 7: End-to-End ML Pipeline Orchestration and NFR-01 Latency Benchmark (< 1.8s).
"""

import sys
import time
from pathlib import Path

# Bootstrap sys.path so imports work cleanly from repo root, backend/, or backend/tests/
_TESTS_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _TESTS_DIR.parent
_REPO_ROOT = _BACKEND_DIR.parent
_ML_DIR = _BACKEND_DIR / "ml"

for _p in (_ML_DIR, _BACKEND_DIR, _REPO_ROOT, _TESTS_DIR):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import pandas as pd
import pytest

from ml.indian_market_data import (
    ASSET_METADATA,
    resolve_data_directory,
    get_synchronized_market_data,
)
from ml.portfolio_optimizer import (
    compute_shrinkage_covariance,
    get_risk_profile_parameters,
    compute_black_litterman,
    optimize_indian_portfolio,
)
from ml.goal_optimizer import (
    MIN_PORTFOLIO_ASSETS,
    MAX_PORTFOLIO_ASSETS,
    MIN_FUND_ALLOCATION,
    MAX_FUND_ALLOCATION,
    MAX_FUND_WEIGHT,
    money,
    get_stock_weight_limit,
    capped_random_weights,
    create_candidate_weights,
    simulate_portfolio,
    load_data as load_goal_data,
    get_gsec_context,
    evaluate_portfolios,
)
from ml.pipeline import run_pipeline


# ============================================================================
# SHARED FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def market_df() -> pd.DataFrame:
    data_dir = resolve_data_directory()
    return get_synchronized_market_data(data_dir)


# ============================================================================
# PHASE 5: MULTI-ASSET PORTFOLIO OPTIMIZER TESTS
# ============================================================================

def test_phase5_ledoit_wolf_covariance_symmetry_and_spd(market_df: pd.DataFrame):
    """Verify annualized Ledoit-Wolf shrinkage covariance is symmetric and strictly positive-definite."""
    cov = compute_shrinkage_covariance(market_df)
    assert cov.shape == (4, 4)
    np.testing.assert_allclose(cov, cov.T, atol=1e-12)

    eigenvalues = np.linalg.eigvalsh(cov)
    assert (eigenvalues > 0.0).all(), f"Covariance matrix has non-positive eigenvalues: {eigenvalues}"
    assert (np.diag(cov) > 0.0).all()


def test_phase5_black_litterman_theorem1_infinite_uncertainty_converges_to_prior(market_df: pd.DataFrame):
    """
    Theorem 1: As ML view uncertainty Omega -> infinity, Black-Litterman posterior
    expected returns mu_BL must converge to the CAPM market equilibrium prior Pi.
    """
    asset_names = list(market_df.columns)
    cov = compute_shrinkage_covariance(market_df)
    risk_aversion = 6.0

    infinite_uncertainty_views = {
        a: {"annualized_expected_return": 0.75, "residual_variance_21d": 1e9}
        for a in asset_names
    }
    mu_bl, cov_bl = compute_black_litterman(
        asset_names=asset_names,
        cov_matrix=cov,
        ml_predictions=infinite_uncertainty_views,
        risk_aversion=risk_aversion,
    )

    bench_w = np.array([ASSET_METADATA[a]["benchmark_weight"] for a in asset_names])
    bench_w /= bench_w.sum()
    pi = risk_aversion * cov @ bench_w

    np.testing.assert_allclose(mu_bl, pi, rtol=1e-3, atol=1e-3)
    assert cov_bl.shape == (4, 4)


def test_phase5_black_litterman_theorem2_high_confidence_pulls_toward_ml_views(market_df: pd.DataFrame):
    """
    Theorem 2: When ML view uncertainty is minimal (Omega -> 1e-4 floor),
    posterior returns mu_BL are pulled strongly toward the ML view vector Q.
    """
    asset_names = list(market_df.columns)
    cov = compute_shrinkage_covariance(market_df)
    risk_aversion = 5.0

    target_views = {
        "NIFTY_EQUITIES": {"annualized_expected_return": 0.22, "residual_variance_21d": 1e-9},
        "MUTUAL_FUNDS": {"annualized_expected_return": 0.18, "residual_variance_21d": 1e-9},
        "MCX_GOLD": {"annualized_expected_return": 0.14, "residual_variance_21d": 1e-9},
        "GOVT_10Y_GSEC": {"annualized_expected_return": 0.08, "residual_variance_21d": 1e-9},
    }
    q_vec = np.array([target_views[a]["annualized_expected_return"] for a in asset_names])

    bench_w = np.array([ASSET_METADATA[a]["benchmark_weight"] for a in asset_names])
    bench_w /= bench_w.sum()
    pi = risk_aversion * cov @ bench_w

    mu_bl, _ = compute_black_litterman(
        asset_names=asset_names,
        cov_matrix=cov,
        ml_predictions=target_views,
        risk_aversion=risk_aversion,
    )

    dist_to_q = np.linalg.norm(mu_bl - q_vec)
    dist_to_pi = np.linalg.norm(mu_bl - pi)
    assert dist_to_q < dist_to_pi, f"Expected posterior closer to Q ({dist_to_q:.4f}) than Pi ({dist_to_pi:.4f})"


@pytest.mark.parametrize("mode", ["black_litterman", "max_sharpe", "min_variance"])
def test_phase5_optimization_modes_simplex_and_long_only_invariants(mode: str):
    """Verify sum(w) == 1.0, w_i >= 0, and valid risk metrics across all 3 optimization modes."""
    res = optimize_indian_portfolio(
        investment_amount=500000.0,
        risk_score=5,
        optimization_mode=mode,
    )
    weights = [alloc["weight_percent"] / 100.0 for alloc in res["allocations"]]
    amounts = [alloc["allocated_amount"] for alloc in res["allocations"]]

    assert len(res["allocations"]) == 4
    assert np.isclose(sum(weights), 1.0, atol=1e-3), f"Simplex violated in {mode}"
    assert all(w >= 0.0 for w in weights), f"Short-selling detected in {mode}"
    assert np.isclose(sum(amounts), 500000.0, atol=50.0)
    assert res["expected_annual_volatility_pct"] > 0.0
    assert res["value_at_risk_1m_99pct"] >= res["value_at_risk_1m_95pct"] >= 0.0
    assert res["conditional_var_1m_95pct"] > 0.0


@pytest.mark.parametrize(
    "risk_score,expected_bounds",
    [
        (2, {"NIFTY_EQUITIES": (5.0, 20.0), "MUTUAL_FUNDS": (5.0, 20.0), "MCX_GOLD": (10.0, 25.0), "GOVT_10Y_GSEC": (40.0, 75.0)}),
        (5, {"NIFTY_EQUITIES": (15.0, 45.0), "MUTUAL_FUNDS": (15.0, 40.0), "MCX_GOLD": (5.0, 20.0), "GOVT_10Y_GSEC": (15.0, 50.0)}),
        (9, {"NIFTY_EQUITIES": (25.0, 60.0), "MUTUAL_FUNDS": (20.0, 50.0), "MCX_GOLD": (5.0, 15.0), "GOVT_10Y_GSEC": (5.0, 25.0)}),
    ],
)
def test_phase5_regime_box_constraints_by_risk_tier(risk_score: int, expected_bounds: dict):
    """Verify Conservative (1-3), Moderate (4-7), and Aggressive (8-10) box constraints hold."""
    res = optimize_indian_portfolio(investment_amount=1000000.0, risk_score=risk_score)
    for alloc in res["allocations"]:
        code = alloc["asset_code"]
        low_pct, high_pct = expected_bounds[code]
        w_pct = alloc["weight_percent"]
        assert low_pct - 0.05 <= w_pct <= high_pct + 0.05, (
            f"Asset {code} weight {w_pct}% outside [{low_pct}%, {high_pct}%] for risk_score={risk_score}"
        )


def test_phase5_risk_aversion_monotonic_schedule():
    """Verify risk aversion lambda decreases strictly from 12.0 (score 1) to 3.9 (score 10)."""
    assets = ["NIFTY_EQUITIES", "MUTUAL_FUNDS", "MCX_GOLD", "GOVT_10Y_GSEC"]
    lambdas = [get_risk_profile_parameters(score, assets)[0] for score in range(1, 11)]
    assert np.isclose(lambdas[0], 12.0)
    assert np.isclose(lambdas[-1], 3.9)
    assert all(lambdas[i] > lambdas[i + 1] for i in range(len(lambdas) - 1))


def test_phase5_var_99_constraint_enforcement_and_tight_fallback():
    """Verify VaR_99% <= max_loss_tolerance_pct when feasible, and graceful fallback when ultra-tight."""
    res_feasible = optimize_indian_portfolio(
        investment_amount=500000.0,
        risk_score=5,
        max_loss_tolerance_pct=0.15,
    )
    assert res_feasible["value_at_risk_1m_99pct"] <= 15.0 + 0.05
    assert res_feasible["var_99_constraint_status"] == "SATISFIED"

    # Impossibly tight 0.1% monthly 99% VaR triggers fallback without crashing and flags VIOLATED
    res_tight = optimize_indian_portfolio(
        investment_amount=500000.0,
        risk_score=9,
        max_loss_tolerance_pct=0.001,
    )
    assert "VIOLATED" in res_tight["var_99_constraint_status"]
    weights = [a["weight_percent"] / 100.0 for a in res_tight["allocations"]]
    assert np.isclose(sum(weights), 1.0, atol=1e-3)


def test_phase5_optimizer_input_validation_errors():
    """Verify negative/zero investment amounts and out-of-range risk scores raise ValueError."""
    for bad_amount in (0.0, -100.0, -500000.0):
        with pytest.raises(ValueError, match="Investment amount must be positive"):
            optimize_indian_portfolio(investment_amount=bad_amount, risk_score=5)

    for bad_score in (0, -3, 11, 99):
        with pytest.raises(ValueError, match="between 1 and 10"):
            optimize_indian_portfolio(investment_amount=100000.0, risk_score=bad_score)


# ============================================================================
# PHASE 6: MONTE CARLO GOAL-BASED OPTIMIZER UNIT TESTS
# ============================================================================

def test_phase6_stock_weight_limits_across_risk_scores():
    """Verify get_stock_weight_limit maps scores 1..10 to 10%, 15%, 20%, 25% and rejects invalid scores."""
    expected = {
        1: 0.10, 2: 0.10, 3: 0.10,
        4: 0.15, 5: 0.15, 6: 0.15,
        7: 0.20, 8: 0.20,
        9: 0.25, 10: 0.25,
    }
    for score, limit in expected.items():
        assert np.isclose(get_stock_weight_limit(score), limit)

    for invalid_score in (0, 11, -1):
        with pytest.raises(ValueError, match="between 1 and 10"):
            get_stock_weight_limit(invalid_score)

    assert money(125000.5) == "₹125,000.50"


def test_phase6_capped_random_weights_simplex_and_cap_invariants():
    """Verify capped_random_weights satisfies sum == total and w_i <= cap across 50 random draws."""
    rng = np.random.default_rng(42)
    for _ in range(50):
        w = capped_random_weights(count=6, total=0.70, cap=0.15, rng=rng)
        assert len(w) == 6
        assert np.isclose(w.sum(), 0.70, atol=1e-6)
        assert (w >= 0.0).all()
        assert (w <= 0.15 + 1e-6).all()

    with pytest.raises(ValueError, match="At least one asset"):
        capped_random_weights(count=0, total=1.0, cap=0.5, rng=rng)

    with pytest.raises(ValueError, match="Not enough assets"):
        capped_random_weights(count=3, total=0.80, cap=0.20, rng=rng)


def test_phase6_create_candidate_weights_diversification_rules():
    """
    Verify create_candidate_weights enforces:
    - 5 to 12 total holdings
    - 20% to 60% mutual fund share
    - Individual stock weight <= max_stock_weight
    - Individual fund weight <= MAX_FUND_WEIGHT (0.40)
    """
    stock_names = [f"STOCK_{i}" for i in range(1, 21)]
    fund_names = [f"FUND_{i}" for i in range(1, 9)]
    rng = np.random.default_rng(123)
    max_stock_w = get_stock_weight_limit(5)  # 0.15

    for _ in range(30):
        alloc = create_candidate_weights(stock_names, fund_names, rng, max_stock_w)
        assert MIN_PORTFOLIO_ASSETS <= len(alloc) <= MAX_PORTFOLIO_ASSETS
        assert np.isclose(sum(alloc.values()), 1.0, atol=1e-6)

        stock_weights = [w for k, w in alloc.items() if k.startswith("STOCK_")]
        fund_weights = [w for k, w in alloc.items() if k.startswith("FUND_")]

        assert len(stock_weights) >= 1 and len(fund_weights) >= 1
        assert all(0.0 <= w <= max_stock_w + 1e-6 for w in stock_weights)
        assert all(0.0 <= w <= MAX_FUND_WEIGHT + 1e-6 for w in fund_weights)
        assert MIN_FUND_ALLOCATION - 1e-6 <= sum(fund_weights) <= MAX_FUND_ALLOCATION + 1e-6


def test_phase6_simulate_portfolio_zero_return_exact_identity():
    """
    Verify simulate_portfolio satisfies exact accounting identity:
    Under 0% monthly returns, terminal_wealth == current_savings + monthly_contribution * months.
    Under positive constant returns, terminal_wealth > current_savings + monthly_contribution * months.
    """
    n_sims, months, n_assets = 250, 36, 4
    zero_returns = np.zeros((60, n_assets), dtype=float)
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    rng = np.random.default_rng(42)
    sampled_idx = rng.integers(0, 60, size=(n_sims, months))

    savings = 200000.0
    monthly = 15000.0
    terminal_zero = simulate_portfolio(zero_returns, weights, sampled_idx, savings, monthly)
    expected_exact = savings + monthly * months
    np.testing.assert_allclose(terminal_zero, expected_exact, atol=1e-9)

    pos_returns = np.full((60, n_assets), 0.01, dtype=float)  # +1% per month
    terminal_pos = simulate_portfolio(pos_returns, weights, sampled_idx, savings, monthly)
    assert (terminal_pos > expected_exact).all()


def test_phase6_goal_optimizer_load_and_evaluate_pipeline(monkeypatch, tmp_path: Path):
    """Verify goal_optimizer load_data, get_gsec_context, and evaluate_portfolios end-to-end."""
    import ml.goal_optimizer as go_mod

    returns_df, stock_names, fund_names = load_goal_data()
    assert len(returns_df) >= 60
    assert len(stock_names) >= 1 and len(fund_names) >= 1
    get_gsec_context()

    # Run lightweight evaluation (25 candidates x 100 simulations) writing into tmp_path
    monkeypatch.setattr(go_mod, "NUM_CANDIDATES", 25)
    monkeypatch.setattr(go_mod, "NUM_SIMULATIONS", 100)
    monkeypatch.setattr(go_mod, "PROCESSED_DIR", tmp_path)

    evaluate_portfolios(
        returns=returns_df,
        stock_names=stock_names,
        fund_names=fund_names,
        target=1500000.0,
        savings=250000.0,
        monthly=15000.0,
        years=5,
        max_loss=300000.0,
        risk_score=6,
    )
    assert (tmp_path / "goal_optimizer_candidates.csv").exists()
    assert (tmp_path / "goal_optimizer_allocation.csv").exists()
    alloc_out = pd.read_csv(tmp_path / "goal_optimizer_allocation.csv")
    assert np.isclose(alloc_out["weight"].sum(), 1.0, atol=1e-4)

    # Validation error checks on evaluate_portfolios
    with pytest.raises(ValueError, match="Target must be greater than zero"):
        evaluate_portfolios(returns_df, stock_names, fund_names, 0, 1000, 500, 5, 100, 5)
    with pytest.raises(ValueError, match="cannot be negative"):
        evaluate_portfolios(returns_df, stock_names, fund_names, 10000, -1, 500, 5, 100, 5)
    with pytest.raises(ValueError, match="Years must be greater than zero"):
        evaluate_portfolios(returns_df, stock_names, fund_names, 10000, 1000, 500, 0, 100, 5)
    with pytest.raises(ValueError, match="shortfall cannot be negative"):
        evaluate_portfolios(returns_df, stock_names, fund_names, 10000, 1000, 500, 5, -10, 5)


# ============================================================================
# PHASE 7: END-TO-END PIPELINE & NFR-01 LATENCY BENCHMARK TESTS
# ============================================================================

def test_phase7_end_to_end_run_pipeline_orchestration():
    """Verify pipeline.run_pipeline executes end-to-end with demographic scoring and target_amount precedence."""
    res_demo = run_pipeline(
        investment_amount=500000.0,
        age=28,
        investment_horizon_years=15,
        savings_to_income_ratio=0.7,
        max_loss_tolerance_pct=0.20,
        optimization_mode="black_litterman",
        retrain=False,
    )
    assert res_demo["investment_amount"] == 500000.0
    assert 1 <= res_demo["risk_score"] <= 10
    assert len(res_demo["allocations"]) == 4
    assert res_demo["var_99_constraint_status"] == "SATISFIED"

    # Verify target_amount overrides investment_amount and explicit_risk_score overrides profiler
    res_override = run_pipeline(
        investment_amount=500000.0,
        target_amount=1200000.0,
        explicit_risk_score=8,
        optimization_mode="max_sharpe",
        retrain=False,
    )
    assert res_override["investment_amount"] == 1200000.0
    assert res_override["risk_score"] == 8


@pytest.mark.parametrize("mode", ["black_litterman", "max_sharpe", "min_variance"])
def test_phase7_nfr01_optimization_latency_under_1800ms(mode: str):
    """
    NFR-01 Benchmark: Single-run portfolio optimization (with cached trained models)
    must complete in strictly less than 1.8 seconds (1800 ms).
    """
    t0 = time.perf_counter()
    res = optimize_indian_portfolio(
        investment_amount=1000000.0,
        risk_score=6,
        optimization_mode=mode,
        retrain_ml=False,
    )
    elapsed_sec = time.perf_counter() - t0
    assert elapsed_sec < 1.8, f"NFR-01 latency violated for {mode}: {elapsed_sec:.3f}s >= 1.800s"
    assert len(res["allocations"]) == 4
