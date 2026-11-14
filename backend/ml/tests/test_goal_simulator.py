"""Unit tests for ml.goal_simulator (Monte Carlo Geometric Brownian Motion)."""

import pytest

from ml import goal_simulator as gs


def test_simulate_goal_paths_deterministic_growth_zero_vol():
    # If volatility is 0, wealth grows deterministically: W_T = W_0 * exp(mu * T)
    res = gs.simulate_goal_paths(
        initial_capital=100000.0,
        target_amount=120000.0,
        horizon_years=2,
        expected_annual_return=0.10,
        annual_volatility=0.0,
        monthly_sip=0.0,
        n_simulations=100,
        random_seed=42,
    )

    # W_2 = 100,000 * exp(0.10 * 2) = 100,000 * exp(0.20) ~= 122,140.28
    expected_terminal = 100000.0 * (2.718281828459045 ** 0.20)
    assert res["percentile_50_wealth"] == pytest.approx(expected_terminal, rel=1e-3)
    assert res["probability_of_success"] == 1.0  # 122k > 120k


def test_simulate_goal_paths_sip_accumulates_capital():
    res = gs.simulate_goal_paths(
        initial_capital=0.0,
        target_amount=500000.0,
        horizon_years=5,
        expected_annual_return=0.12,
        annual_volatility=0.15,
        monthly_sip=10000.0,
        n_simulations=1000,
        random_seed=42,
    )

    assert res["total_invested_capital"] == 10000.0 * 60  # 600,000 invested
    # Investing 600k with positive expected return should achieve 500k target with very high probability
    assert res["probability_of_success"] > 0.90
    assert res["percentile_10_wealth"] < res["percentile_50_wealth"] < res["percentile_90_wealth"]


def test_simulate_goal_paths_unreachable_target():
    res = gs.simulate_goal_paths(
        initial_capital=10000.0,
        target_amount=5000000.0,  # 50 Lakhs from 10k in 1 year
        horizon_years=1,
        expected_annual_return=0.08,
        annual_volatility=0.12,
        monthly_sip=0.0,
        n_simulations=500,
        random_seed=42,
    )
    assert res["probability_of_success"] == 0.0


def test_simulate_goal_paths_invalid_inputs():
    with pytest.raises(ValueError, match="Initial capital cannot be negative"):
        gs.simulate_goal_paths(initial_capital=-100, target_amount=1000, horizon_years=1, expected_annual_return=0.1, annual_volatility=0.1)

    with pytest.raises(ValueError, match="Target amount must be strictly positive"):
        gs.simulate_goal_paths(initial_capital=100, target_amount=0, horizon_years=1, expected_annual_return=0.1, annual_volatility=0.1)

    with pytest.raises(ValueError, match="Investment horizon must be at least 1 year"):
        gs.simulate_goal_paths(initial_capital=100, target_amount=1000, horizon_years=0, expected_annual_return=0.1, annual_volatility=0.1)

