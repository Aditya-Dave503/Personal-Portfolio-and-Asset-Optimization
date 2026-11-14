"""
Suite 1: Data Ingestion & Market Data Integrity Tests
Verifies cleanliness, synchrony, and sovereign bond duration math.
"""
import pytest
import numpy as np
import pandas as pd
from ml.indian_market_data import (
    ASSET_METADATA,
    resolve_data_directory,
    get_synchronized_market_data,
    load_gsec_bond_returns,
)

@pytest.fixture(scope="module")
def market_data():
    data_dir = resolve_data_directory()
    return get_synchronized_market_data(data_dir)

def test_data_synchronization(market_data):
    """Ensure exactly 4 core Indian assets are present, chronologically ordered with 0 NaNs."""
    expected_assets = ["NIFTY_EQUITIES", "MUTUAL_FUNDS", "MCX_GOLD", "GOVT_10Y_GSEC"]
    assert list(market_data.columns) == expected_assets
    assert market_data.isna().sum().sum() == 0, "Null values detected in synchronized returns!"
    assert market_data.index.is_monotonic_increasing, "Market dates are not chronologically sorted!"
    assert len(market_data) >= 2000, f"Expected >2000 trading days, found {len(market_data)}"

def test_silver_exclusion_policy(market_data):
    """Verify that Silver ('SLV' / 'SILVER') is strictly excluded from all datasets."""
    assert "SLV" not in market_data.columns
    assert "SILVER" not in market_data.columns
    assert "SLV" not in ASSET_METADATA
    assert "SILVER" not in ASSET_METADATA

def test_gsec_bond_duration_convexity():
    """Verify duration and convexity transformation produces realistic sovereign yield returns."""
    data_dir = resolve_data_directory()
    bond_ret = load_gsec_bond_returns(data_dir, modified_duration=6.8, convexity=55.0)
    ann_return = bond_ret.mean() * 252
    ann_vol = bond_ret.std() * np.sqrt(252)

    assert 0.04 <= ann_return <= 0.12, f"Unrealistic annualized sovereign bond return: {ann_return:.2%}"
    assert 0.02 <= ann_vol <= 0.09, f"Unrealistic annualized sovereign bond volatility: {ann_vol:.2%}"

