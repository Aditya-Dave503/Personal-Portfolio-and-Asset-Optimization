"""Unit tests for ml.tail_risk (Non-Gaussian Tail Risk & Cornish-Fisher Expansion)."""

import numpy as np
import pytest
from scipy import stats

from ml import tail_risk


def test_calculate_higher_moments_gaussian():
    rng = np.random.default_rng(42)
    # 50,000 standard normal draws -> skewness near 0, excess kurtosis near 0
    norm_samples = rng.normal(0.0, 1.0, size=50000)
    moments = tail_risk.calculate_higher_moments(norm_samples)

    assert moments["mean"] == pytest.approx(0.0, abs=0.03)
    assert moments["std"] == pytest.approx(1.0, abs=0.03)
    assert moments["skewness"] == pytest.approx(0.0, abs=0.05)
    assert moments["excess_kurtosis"] == pytest.approx(0.0, abs=0.08)
    assert moments["raw_kurtosis"] == pytest.approx(3.0, abs=0.08)


def test_cornish_fisher_quantile_reduces_to_gaussian():
    # When S = 0 and K - 3 = 0, z_CF must exactly equal standard normal z_alpha
    for alpha in [0.05, 0.01, 0.10]:
        z_cf = tail_risk.cornish_fisher_quantile(alpha=alpha, skewness=0.0, excess_kurtosis=0.0)
        z_gauss = stats.norm.ppf(alpha)
        assert z_cf == pytest.approx(z_gauss, rel=1e-6)


def test_cornish_fisher_penalizes_negative_skew_and_fat_tails():
    alpha = 0.05
    z_normal = stats.norm.ppf(alpha)  # -1.64485

    # Negative skewness (left tail risk) shifts z_cf to be more negative
    z_neg_skew = tail_risk.cornish_fisher_quantile(alpha=alpha, skewness=-0.8, excess_kurtosis=0.0)
    assert z_neg_skew < z_normal

    # In deep tails (alpha = 0.01 where |z| > sqrt(3) ~ 1.732), high excess kurtosis (fat tails)
    # makes z_cf strictly more negative, increasing estimated tail losses
    alpha_deep = 0.01
    z_norm_deep = stats.norm.ppf(alpha_deep)
    z_fat_tails_deep = tail_risk.cornish_fisher_quantile(alpha=alpha_deep, skewness=0.0, excess_kurtosis=4.0)
    assert z_fat_tails_deep < z_norm_deep


def test_modified_var_and_cvar_ordering():
    rng = np.random.default_rng(123)
    # Simulate realistic fat-tailed, negatively skewed market returns (Student-t with negative jump)
    base = rng.standard_t(df=4, size=10000) * 0.01
    jumps = rng.binomial(1, 0.02, size=10000) * (-0.05)
    returns = base + jumps

    mvar_95 = tail_risk.calculate_modified_var(returns, alpha=0.05)
    cvar_95 = tail_risk.calculate_cvar(returns, var_threshold=mvar_95, alpha=0.05)

    # 1. VaR must be positive
    assert mvar_95 > 0.0
    # 2. CVaR (Expected Shortfall) must be strictly greater than or equal to VaR
    assert cvar_95 >= mvar_95


def test_evaluate_tail_risk_suite():
    returns = np.array([-0.05, -0.04, -0.03, -0.01, 0.00, 0.01, 0.015, 0.02, 0.025, 0.03])
    report = tail_risk.evaluate_tail_risk(returns, alpha=0.05)

    assert "modified_var" in report
    assert "cvar_expected_shortfall" in report
    assert "z_cornish_fisher" in report
    assert report["cvar_expected_shortfall"] >= report["modified_var"]
    assert report["confidence_level_pct"] == 95.0


def test_tail_risk_insufficient_data():
    with pytest.raises(ValueError, match="At least 4 observations required"):
        tail_risk.calculate_higher_moments([0.01, 0.02])

    with pytest.raises(ValueError, match="Alpha must be in"):
        tail_risk.cornish_fisher_quantile(alpha=0.6, skewness=0.0, excess_kurtosis=0.0)

