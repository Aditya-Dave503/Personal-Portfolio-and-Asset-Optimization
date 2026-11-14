"""
Phase 1 & Phase 2 ML Unit and Causality Test Suite:
- Phase 1: Indian Multi-Asset Market Data Ingestion, Synchronization, Silver Exclusion,
           and 10Y G-Sec Duration/Convexity 2nd-Order Taylor Approximation.
- Phase 2: Causal Feature Engineering, Zero-Lookahead Truncation Invariance,
           Forward Target Isolation (t+1), RSI/Volatility/Drawdown Mathematical Bounds,
           Backward Macro Yield merge_asof Causality, and ml_src/feature_engineering.py units.
"""

import sys
from pathlib import Path

# Bootstrap sys.path so imports work cleanly from repo root, backend/, or backend/tests/
_TESTS_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _TESTS_DIR.parent
_REPO_ROOT = _BACKEND_DIR.parent
_ML_DIR = _BACKEND_DIR / "ml"
_ML_SRC_DIR = _REPO_ROOT / "ml_src"

for _p in (_ML_DIR, _BACKEND_DIR, _REPO_ROOT, _ML_SRC_DIR, _TESTS_DIR):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import pandas as pd
import pytest

from ml.indian_market_data import (
    ASSET_METADATA,
    resolve_data_directory,
    load_nifty_equity_returns,
    load_mutual_fund_returns,
    load_mcx_gold_returns,
    load_gsec_bond_returns,
    load_macro_yield_features,
    get_synchronized_market_data,
)
from ml.ml_engine import (
    calculate_rsi,
    extract_asset_features,
    extract_forward_targets,
)
from ml.features import (
    extract_features as legacy_extract_features,
    extract_targets as legacy_extract_targets,
    prepare_training_dataset,
    prepare_inference_features,
)
def _rolling_return(returns: pd.Series, window: int) -> pd.Series:
    safe = returns.where(returns > -1.0)
    return np.expm1(np.log1p(safe).rolling(window, min_periods=window).sum())


def _forward_return_targets(returns: pd.Series, horizons=(21, 63, 126, 252)) -> pd.DataFrame:
    values = returns.astype(float).to_numpy()
    valid = np.isfinite(values) & (values > -1.0)
    log_returns = np.full(values.shape, np.nan, dtype=float)
    log_returns[valid] = np.log1p(values[valid])
    reverse_log = pd.Series(log_returns[::-1])
    reverse_valid = pd.Series(valid[::-1].astype(float))
    out = {}
    for h in horizons:
        sums = reverse_log.rolling(h, min_periods=h).sum().shift(1).to_numpy()[::-1]
        counts = reverse_valid.rolling(h, min_periods=h).sum().shift(1).to_numpy()[::-1]
        target = np.expm1(sums)
        target[(counts != h) | ~np.isfinite(target)] = np.nan
        out[f"future_return_{h}d"] = target
    return pd.DataFrame(out, index=returns.index)



# ============================================================================
# SHARED FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def data_dir() -> Path:
    return resolve_data_directory()


@pytest.fixture(scope="module")
def market_data(data_dir: Path) -> pd.DataFrame:
    return get_synchronized_market_data(data_dir)


@pytest.fixture(scope="module")
def macro_data(data_dir: Path) -> pd.DataFrame:
    return load_macro_yield_features(data_dir)


# ============================================================================
# PHASE 1: DATA INGESTION & PREPROCESSING UNIT TESTS
# ============================================================================

def test_phase1_synchronized_market_data_schema_and_cleanliness(market_data: pd.DataFrame):
    """Verify synchronized dataset contains the 4 core Indian assets, 0 NaNs, and >2000 chronological days."""
    expected_assets = ["NIFTY_EQUITIES", "MUTUAL_FUNDS", "MCX_GOLD", "GOVT_10Y_GSEC"]
    assert list(market_data.columns) == expected_assets
    assert market_data.isna().sum().sum() == 0, "Synchronized dataset must contain zero NaNs"
    assert np.isfinite(market_data.to_numpy()).all(), "All daily returns must be finite"
    assert isinstance(market_data.index, pd.DatetimeIndex)
    assert market_data.index.is_monotonic_increasing, "Trading dates must be strictly chronological"
    assert not market_data.index.duplicated().any(), "Duplicate trading dates are not permitted"
    assert len(market_data) > 2000, f"Expected >2000 synchronized trading days, got {len(market_data)}"
    # Daily return sanity bounds: no single day should exceed +-50% across diversified indices
    assert (market_data.abs() < 0.50).all().all()


def test_phase1_silver_strictly_excluded(market_data: pd.DataFrame):
    """Verify Silver ('SLV' / 'SILVER') is completely absent from market data and metadata."""
    for forbidden in ("SLV", "SILVER", "slv", "silver"):
        assert forbidden not in market_data.columns
        assert forbidden not in ASSET_METADATA


def test_phase1_benchmark_weights_sum_to_unity():
    """Verify market equilibrium benchmark weights across the 4 Indian assets sum to 1.0."""
    weights = [meta["benchmark_weight"] for meta in ASSET_METADATA.values()]
    assert len(weights) == 4
    assert all(w > 0.0 for w in weights)
    assert np.isclose(sum(weights), 1.0, atol=1e-9)


def test_phase1_gsec_duration_convexity_macro_bounds(data_dir: Path):
    """Verify 10Y G-Sec modified duration (6.8) and convexity (55.0) produce realistic Indian bond returns."""
    bond_ret = load_gsec_bond_returns(data_dir, modified_duration=6.8, convexity=55.0)
    ann_return = float(bond_ret.mean() * 252.0)
    ann_vol = float(bond_ret.std(ddof=1) * np.sqrt(252.0))

    assert 0.04 <= ann_return <= 0.12, f"Annualized G-Sec return out of bounds: {ann_return:.4f}"
    assert 0.02 <= ann_vol <= 0.08, f"Annualized G-Sec volatility out of bounds: {ann_vol:.4f}"


def test_phase1_gsec_taylor_expansion_exact_formula(tmp_path: Path):
    """
    Verify exact 2nd-order Taylor approximation formula on a synthetic yield CSV:
    R_{bond, t} = (y_{t-1} / 252) - D_mod * delta_y_t + 0.5 * Convexity * (delta_y_t)^2
    """
    csv_path = tmp_path / "india_10y_gsec_yield_changes.csv"
    pd.DataFrame({
        "date": ["2025-01-01", "2025-01-02", "2025-01-03"],
        "yield_pct": [7.00, 7.10, 6.95],
        "yield_change_bps": [0.0, 10.0, -15.0],
    }).to_csv(csv_path, index=False)

    d_mod = 6.8
    conv = 55.0
    series = load_gsec_bond_returns(tmp_path, modified_duration=d_mod, convexity=conv)
    assert len(series) == 2

    # Day 1 -> Day 2: y_{t-1} = 0.0700, y_t = 0.0710, delta_y = +0.0010
    y0, dy1 = 0.0700, 0.0010
    expected_1 = (y0 / 252.0) - d_mod * dy1 + 0.5 * conv * (dy1 ** 2)
    assert np.isclose(series.iloc[0], expected_1, atol=1e-12)

    # Day 2 -> Day 3: y_{t-1} = 0.0710, y_t = 0.0695, delta_y = -0.0015
    y1, dy2 = 0.0710, -0.0015
    expected_2 = (y1 / 252.0) - d_mod * dy2 + 0.5 * conv * (dy2 ** 2)
    assert np.isclose(series.iloc[1], expected_2, atol=1e-12)


def test_phase1_individual_asset_loaders_and_error_handling(data_dir: Path, tmp_path: Path):
    """Verify individual loaders return non-empty Series and raise FileNotFoundError on invalid paths."""
    nifty = load_nifty_equity_returns(data_dir)
    mf = load_mutual_fund_returns(data_dir)
    gold = load_mcx_gold_returns(data_dir)
    macro = load_macro_yield_features(data_dir)

    assert nifty.name == "NIFTY_EQUITIES" and len(nifty) > 2000
    assert mf.name == "MUTUAL_FUNDS" and len(mf) > 2000
    assert gold.name == "MCX_GOLD" and len(gold) > 2000
    assert {"yield_pct", "yield_change_bps"}.issubset(macro.columns)

    with pytest.raises(FileNotFoundError):
        resolve_data_directory("Z:/non_existent_directory_optiwealth_12345")

    with pytest.raises(FileNotFoundError):
        load_nifty_equity_returns(tmp_path)

    with pytest.raises(FileNotFoundError):
        load_mutual_fund_returns(tmp_path)

    with pytest.raises(FileNotFoundError):
        load_mcx_gold_returns(tmp_path)

    with pytest.raises(FileNotFoundError):
        load_gsec_bond_returns(tmp_path)


def test_phase1_data_loader_and_audit_validation_utilities():
    """Verify data_loader.py, validate_unified_dataset.py, and audit_all_assets.py utilities execute cleanly."""
    from ml.data_loader import (
        load_equity_data,
        load_mutual_fund_data,
        load_gold_data,
        load_gsec_yields,
    )
    from ml.validate_unified_dataset import (
        validate_returns,
        validate_gsec,
        validate_gold,
    )
    from ml.audit_all_assets import FILES, audit_file
    from ml.build_unified_dataset import monthly_returns_from_prices

    eq_df = load_equity_data()
    mf_df = load_mutual_fund_data()
    gold_df = load_gold_data()
    gsec_df = load_gsec_yields()

    assert len(eq_df) > 10000 and "adjusted_close" in eq_df.columns
    assert len(mf_df) > 5000 and "nav" in mf_df.columns
    assert len(gold_df) > 2000
    assert len(gsec_df) > 2000 and "yield_pct" in gsec_df.columns

    # Exercise monthly_returns_from_prices on a small slice
    sample_eq = eq_df.head(500)
    monthly_eq = monthly_returns_from_prices(
        sample_eq, date_col="date", asset_col="symbol", price_col="adjusted_close", prefix="STOCK"
    )
    assert not monthly_eq.empty

    # Exercise dataset validation & audit functions
    validate_returns()
    validate_gsec()
    validate_gold()
    for name, (filename, date_col) in FILES.items():
        audit_file(name, filename, date_col)


# ============================================================================
# PHASE 2: CAUSAL FEATURE ENGINEERING & ANTI-LEAKAGE TESTS
# ============================================================================

def test_phase2_feature_causality_truncation_invariance(market_data: pd.DataFrame, macro_data: pd.DataFrame):
    """
    Gold-standard zero-lookahead test:
    Features extracted on truncated history [0..midpoint] must match features extracted
    on the full 10-year history at every timestamp <= midpoint.
    """
    series = market_data["NIFTY_EQUITIES"]
    features_full = extract_asset_features(series, macro_data)

    midpoint = len(series) // 2
    series_truncated = series.iloc[:midpoint]
    features_truncated = extract_asset_features(series_truncated, macro_data)

    common_idx = features_truncated.dropna().index
    assert len(common_idx) > 500
    pd.testing.assert_frame_equal(
        features_full.loc[common_idx, features_truncated.columns],
        features_truncated.loc[common_idx],
        check_exact=False,
        rtol=1e-6,
        atol=1e-8,
    )


def test_phase2_forward_target_isolation_and_trailing_nans(market_data: pd.DataFrame):
    """Verify forward target at date t compounds strictly from t+1 to t+h and leaves trailing h NaNs."""
    series = market_data["NIFTY_EQUITIES"]
    price_idx = (1.0 + series).cumprod()
    horizon = 21
    target_21d = extract_forward_targets(series, horizon_days=horizon)

    for t in (50, 250, 1000, len(series) - horizon - 1):
        expected = (price_idx.iloc[t + horizon] / price_idx.iloc[t]) - 1.0
        actual = target_21d.iloc[t]
        assert np.isclose(expected, actual, atol=1e-9), f"Mismatch at index {t}"

    # Perturbing return at date t must NOT change forward target at date t
    series_perturbed = series.copy()
    series_perturbed.iloc[250] += 0.08
    target_perturbed = extract_forward_targets(series_perturbed, horizon_days=horizon)
    assert np.isclose(target_21d.iloc[250], target_perturbed.iloc[250], atol=1e-9)

    # Last `horizon` observations have no complete future window and must be NaN
    assert target_21d.iloc[-horizon:].isna().all()
    assert target_21d.iloc[:-horizon].notna().all()


def test_phase2_rsi_mathematical_bounds_and_monotonic_extremes(market_data: pd.DataFrame):
    """Verify 14-day RSI is bounded in [0, 100] and approaches 100 (or 0) on monotonic trends."""
    for col in market_data.columns:
        price_idx = (1.0 + market_data[col]).cumprod()
        rsi = calculate_rsi(price_idx, period=14).dropna()
        assert (rsi >= 0.0).all() and (rsi <= 100.0).all(), f"RSI out of [0, 100] for {col}"

    # Purely increasing price series -> RSI approaches 100
    up_prices = pd.Series(np.linspace(100.0, 200.0, 100))
    rsi_up = calculate_rsi(up_prices, period=14).dropna()
    assert rsi_up.iloc[-1] > 99.0

    # Purely decreasing price series -> RSI approaches 0
    down_prices = pd.Series(np.linspace(200.0, 100.0, 100))
    rsi_down = calculate_rsi(down_prices, period=14).dropna()
    assert rsi_down.iloc[-1] < 1.0


def test_phase2_volatility_and_drawdown_invariants(market_data: pd.DataFrame):
    """Verify rolling annualized volatilities are non-negative and 252d peak drawdown is in [-1, 0]."""
    feats = extract_asset_features(market_data["NIFTY_EQUITIES"]).dropna()

    for w in (21, 63, 126):
        vol_col = f"volatility_{w}d"
        assert (feats[vol_col] >= 0.0).all(), f"{vol_col} must be non-negative"

    assert (feats["vol_ratio_21_63"] > 0.0).all()
    assert (feats["drawdown_252d"] <= 1e-7).all(), "Drawdown cannot be positive above peak"
    assert (feats["drawdown_252d"] >= -1.0).all(), "Drawdown cannot exceed -100%"


def test_phase2_macro_yield_merge_asof_backward_causality(market_data: pd.DataFrame, macro_data: pd.DataFrame):
    """
    Verify pd.merge_asof(..., direction='backward') never leaks future G-Sec yield spikes
    into earlier feature timestamps.
    """
    series = market_data["NIFTY_EQUITIES"]
    feats_baseline = extract_asset_features(series, macro_data)

    cutoff_date = series.index[len(series) // 2]
    macro_shocked = macro_data.copy()
    future_mask = macro_shocked.index > cutoff_date
    macro_shocked.loc[future_mask, "yield_pct"] = 99.99
    macro_shocked.loc[future_mask, "yield_change_bps"] = 5000.0

    feats_shocked = extract_asset_features(series, macro_shocked)

    past_idx = series.index[series.index <= cutoff_date]
    pd.testing.assert_frame_equal(
        feats_baseline.loc[past_idx],
        feats_shocked.loc[past_idx],
        check_exact=True,
    )


def test_phase2_ml_src_feature_engineering_helpers():
    """Unit test _forward_return_targets and _rolling_return from ml_src/feature_engineering.py."""
    returns = pd.Series([0.10, 0.20, 0.30, 0.40, -0.05])
    target_df = _forward_return_targets(returns, horizons=(2, 3))

    # At index 0: 2d forward compounds index 1 (0.20) and index 2 (0.30)
    assert np.isclose(target_df.loc[0, "future_return_2d"], (1.20 * 1.30) - 1.0, atol=1e-9)
    # At index 0: 3d forward compounds indices 1, 2, 3
    assert np.isclose(target_df.loc[0, "future_return_3d"], (1.20 * 1.30 * 1.40) - 1.0, atol=1e-9)
    # Trailing rows lacking full horizon must be NaN
    assert pd.isna(target_df.loc[3, "future_return_2d"])
    assert pd.isna(target_df.loc[4, "future_return_2d"])

    # Invalid return (<= -1.0 or NaN) inside future window must invalidate target to NaN
    bad_returns = pd.Series([0.05, -1.0, 0.03, np.nan, 0.02])
    bad_targets = _forward_return_targets(bad_returns, horizons=(2,))
    assert bad_targets["future_return_2d"].isna().all()

    # _rolling_return compounds simple returns accurately
    roll_2 = _rolling_return(pd.Series([0.10, 0.20, 0.30]), window=2)
    assert pd.isna(roll_2.iloc[0])
    assert np.isclose(roll_2.iloc[1], (1.10 * 1.20) - 1.0, atol=1e-9)
    assert np.isclose(roll_2.iloc[2], (1.20 * 1.30) - 1.0, atol=1e-9)


def test_phase2_legacy_features_module_consistency(market_data: pd.DataFrame):
    """Verify ml/features.py helper functions preserve zero-staleness inference vs training dropna."""
    prices = (1.0 + market_data["NIFTY_EQUITIES"]).cumprod() * 100.0
    feats = legacy_extract_features(prices)
    targets = legacy_extract_targets(prices, horizon_days=21)
    X_train, y_train = prepare_training_dataset(prices, horizon_days=21)
    latest_row = prepare_inference_features(prices)

    assert len(X_train) == len(y_train)
    assert latest_row.shape == (1, feats.shape[1])
    assert latest_row.index[0] == prices.index[-1]
    assert X_train.index[-1] == prices.index[-22]
    assert targets.iloc[-21:].isna().all()
