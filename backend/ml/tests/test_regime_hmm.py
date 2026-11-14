"""
Macroeconomic Regime-Switching Model Tests (Gaussian HMM)
Verifies Markov transition matrix stochastic properties, state ordering, and risk scaling.
"""
import pytest
import numpy as np
import pandas as pd

from ml.regime_model import (
    GaussianHMM2State,
    detect_market_regimes,
)


def test_hmm_transition_matrix_stochastic_property():
    """Transition matrix rows must strictly sum to 1.0 (Markov transition property)."""
    rng = np.random.default_rng(42)
    # 200 days: first 100 calm, next 100 turbulent
    calm = rng.normal(0.0008, 0.007, 100)
    turbulent = rng.normal(-0.002, 0.025, 100)
    returns = np.concatenate([calm, turbulent])

    hmm = GaussianHMM2State(max_iter=30)
    hmm.fit(returns)

    # Each row of transition matrix A must sum to 1.0
    row_sums = np.sum(hmm.A, axis=1)
    np.testing.assert_allclose(row_sums, [1.0, 1.0], atol=1e-4)
    assert (hmm.A >= 0.0).all() and (hmm.A <= 1.0).all()


def test_hmm_volatility_state_ordering():
    """State 0 must strictly represent the lower-volatility calm regime."""
    rng = np.random.default_rng(101)
    calm = rng.normal(0.001, 0.006, 150)
    panic = rng.normal(-0.001, 0.022, 150)
    returns = np.concatenate([calm, panic])

    hmm = GaussianHMM2State(max_iter=30)
    hmm.fit(returns)

    assert hmm.stds[0] < hmm.stds[1], (
        f"State 0 volatility ({hmm.stds[0]:.4f}) must be less than State 1 ({hmm.stds[1]:.4f})"
    )


def test_filtered_probabilities_bounds():
    """Filtered probabilities P(S_t | x_1...x_t) must be bounded in [0, 1] and sum to 1."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0005, 0.012, 100)

    hmm = GaussianHMM2State(max_iter=20)
    hmm.fit(returns)
    probs = hmm.predict_filtered_probabilities(returns)

    assert probs.shape == (100, 2)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(100), atol=1e-5)


def test_detect_market_regimes_pipeline():
    """End-to-end regime output formatting and risk aversion scaling."""
    dates = pd.date_range("2025-01-01", periods=120, freq="B")
    rng = np.random.default_rng(99)
    s = pd.Series(rng.normal(0.0004, 0.01, 120), index=dates)

    out = detect_market_regimes(s, base_risk_aversion=5.0)

    assert len(out.states) == 2
    assert 0.0 <= out.panic_probability <= 1.0
    # Panic probability scales risk aversion higher
    assert out.effective_risk_aversion >= 5.0
    assert out.active_regime_id in [0, 1]
    assert len(out.historical_regimes) == 120

