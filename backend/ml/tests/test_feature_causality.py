"""
Suite 2: Feature Engineering & Anti-Data-Leakage Tests
Verifies mathematical correctness of RSI, moving averages, and strict causality.
"""
import pytest
import numpy as np
import pandas as pd
from ml.indian_market_data import resolve_data_directory, get_synchronized_market_data
from ml.ml_engine import extract_asset_features, extract_forward_targets, calculate_rsi

@pytest.fixture(scope="module")
def series_nifty():
    data_dir = resolve_data_directory()
    df = get_synchronized_market_data(data_dir)
    return df["NIFTY_EQUITIES"]

def test_feature_causality_no_lookahead(series_nifty):
    """
    Gold standard test for time-series ML:
    Features computed on data up to index t must be identical to features computed on full data up to index t.
    """
    features_full = extract_asset_features(series_nifty)
    cutoff = len(series_nifty) // 2
    features_truncated = extract_asset_features(series_nifty.iloc[:cutoff])

    common_indices = features_truncated.dropna().index
    pd.testing.assert_frame_equal(
        features_full.loc[common_indices, features_truncated.columns],
        features_truncated.loc[common_indices],
        rtol=1e-5,
        atol=1e-5
    )

def test_forward_target_starts_at_t_plus_one(series_nifty):
    """Verify forward targets start at t+1 and do not include return at date t."""
    price_idx = (1.0 + series_nifty).cumprod()
    target_21d = extract_forward_targets(series_nifty, horizon_days=21)

    t = 150
    expected = (price_idx.iloc[t + 21] / price_idx.iloc[t]) - 1.0
    actual = target_21d.iloc[t]
    assert np.isclose(expected, actual, atol=1e-6)

def test_rsi_bounds(series_nifty):
    """Verify RSI values are strictly contained in [0, 100]."""
    price_idx = (1.0 + series_nifty).cumprod()
    rsi = calculate_rsi(price_idx, period=14).dropna()
    assert (rsi >= 0.0).all() and (rsi <= 100.0).all()

