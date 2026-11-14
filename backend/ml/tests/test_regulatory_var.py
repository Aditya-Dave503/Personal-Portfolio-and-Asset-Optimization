"""
Regulatory Value-at-Risk Backtesting Tests (Basel-III Compliance)
Verifies Kupiec POF Likelihood-Ratio, Christoffersen Independence, and Traffic Lights.
"""
import pytest
import numpy as np
import pandas as pd

from ml.regulatory_backtester import (
    kupiec_pof_test,
    christoffersen_independence_test,
    classify_basel_traffic_light,
    run_full_regulatory_backtest,
    compute_var_violation_series,
)


def test_kupiec_pof_ideal_model_passes():
    """For 1000 observations at 99% VaR, 10 violations represents exact 1% coverage."""
    lr_stat, p_val, passed = kupiec_pof_test(total_obs=1000, violations=10, alpha=0.01)
    assert passed is True
    assert lr_stat < 3.841  # 95% critical value
    assert p_val > 0.05


def test_kupiec_pof_underestimating_model_fails():
    """For 1000 observations, 35 violations (3.5% failure rate vs 1% target) must be rejected."""
    lr_stat, p_val, passed = kupiec_pof_test(total_obs=1000, violations=35, alpha=0.01)
    assert passed is False
    assert lr_stat > 3.841
    assert p_val < 0.001


def test_christoffersen_independent_failures():
    """Independent evenly spaced violations must pass the independence hypothesis."""
    # 200 days with no consecutive violations (e.g. violation every 50 days)
    v = np.zeros(200, dtype=int)
    v[[30, 80, 130, 180]] = 1
    s = pd.Series(v)

    lr_stat, p_val, passed = christoffersen_independence_test(s)
    assert passed is True
    assert p_val > 0.05


def test_christoffersen_clustered_failures_rejected():
    """Clustered consecutive failures (violations happening on consecutive days) must fail."""
    # 4 violations occurring back-to-back
    v = np.zeros(200, dtype=int)
    v[50:54] = 1
    s = pd.Series(v)

    lr_stat, p_val, passed = christoffersen_independence_test(s)
    assert passed is False
    assert p_val < 0.05


def test_basel_traffic_light_zones():
    """Basel Committee traffic light zones for 250 trading days."""
    # Green zone: <= 4 exceptions
    zone_g, mult_g, _ = classify_basel_traffic_light(total_obs=250, violations=3, alpha=0.01)
    assert zone_g == "GREEN"
    assert mult_g == 3.00

    # Yellow zone: 5 to 9 exceptions
    zone_y, mult_y, _ = classify_basel_traffic_light(total_obs=250, violations=7, alpha=0.01)
    assert zone_y == "YELLOW"
    assert 3.40 <= mult_y <= 3.85

    # Red zone: >= 10 exceptions
    zone_r, mult_r, _ = classify_basel_traffic_light(total_obs=250, violations=12, alpha=0.01)
    assert zone_r == "RED"
    assert mult_r == 4.00


def test_full_regulatory_backtest_pipeline():
    """End-to-end regulatory verification on synthetic time series."""
    rng = np.random.default_rng(42)
    # 500 trading days
    returns = pd.Series(rng.normal(0.0005, 0.01, size=500))
    # 99% VaR = 2.326 * std = 0.02326
    var_series = pd.Series(np.full(500, 0.0233))

    res = run_full_regulatory_backtest(returns, var_series, alpha=0.01)
    assert res.total_observations == 500
    assert res.basel_zone in ["GREEN", "YELLOW"]
    assert res.kupiec_lr_stat >= 0.0
    assert 0.0 <= res.kupiec_p_value <= 1.0

