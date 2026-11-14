"""Unit tests for ml.indian_market_data."""

from pathlib import Path
import pandas as pd
import pytest

from ml import indian_market_data as imd


def test_resolve_data_directory_custom_valid(synthetic_data_dir: Path):
    resolved = imd.resolve_data_directory(str(synthetic_data_dir))
    assert resolved == synthetic_data_dir


def test_resolve_data_directory_custom_missing(tmp_path: Path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError, match="Custom data directory does not exist"):
        imd.resolve_data_directory(str(missing))


def test_resolve_data_directory_auto_discovery_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(imd, "POSSIBLE_DATA_DIRS", [tmp_path / "empty_a", tmp_path / "empty_b"])
    with pytest.raises(FileNotFoundError, match="Could not automatically locate"):
        imd.resolve_data_directory()


def test_resolve_data_directory_auto_discovery_succeeds(synthetic_data_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(imd, "POSSIBLE_DATA_DIRS", [synthetic_data_dir])
    assert imd.resolve_data_directory() == synthetic_data_dir


def test_load_nifty_equity_returns_cleans_and_equal_weights(synthetic_data_dir: Path):
    series = imd.load_nifty_equity_returns(synthetic_data_dir)
    assert series.name == "NIFTY_EQUITIES"
    assert not series.isna().any()
    assert len(series) == 319  # 320 dates -> 319 returns after pct_change
    assert series.index.is_monotonic_increasing


def test_load_mutual_fund_returns_drops_non_positive_nav(synthetic_data_dir: Path):
    series = imd.load_mutual_fund_returns(synthetic_data_dir)
    assert series.name == "MUTUAL_FUNDS"
    assert not series.isna().any()
    assert len(series) == 319


def test_load_mcx_gold_returns_supports_lowercase_close_columns(tmp_path: Path):
    df = pd.DataFrame({
        "date": ["2024-01-03", "2024-01-01", "2024-01-02"],
        "Close": [110.0, 100.0, 105.0],
    })
    df.to_csv(tmp_path / "gold_data_unvalidated.csv", index=False)

    series = imd.load_mcx_gold_returns(tmp_path)
    assert series.name == "MCX_GOLD"
    assert list(series.round(6)) == [0.05, round((110.0 / 105.0) - 1.0, 6)]


def test_load_gsec_bond_returns_taylor_approximation(tmp_path: Path):
    # Day 1: 7.00%, Day 2: 6.90% (delta_y = -0.001)
    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "yield_pct": [7.00, 6.90],
        "yield_change_bps": [0.0, -10.0],
    })
    df.to_csv(tmp_path / "india_10y_gsec_yield_changes.csv", index=False)

    d_mod = 6.8
    conv = 55.0
    series = imd.load_gsec_bond_returns(tmp_path, modified_duration=d_mod, convexity=conv)

    expected = (0.07 / 252.0) - d_mod * (-0.001) + 0.5 * conv * ((-0.001) ** 2)
    assert len(series) == 1
    assert series.iloc[0] == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    "loader_fn",
    [
        imd.load_nifty_equity_returns,
        imd.load_mutual_fund_returns,
        imd.load_mcx_gold_returns,
        imd.load_gsec_bond_returns,
    ],
)
def test_asset_loaders_raise_when_csv_missing(tmp_path: Path, loader_fn):
    with pytest.raises(FileNotFoundError):
        loader_fn(tmp_path)


def test_load_macro_yield_features_and_synchronized_market_data(synthetic_data_dir: Path):
    macro = imd.load_macro_yield_features(synthetic_data_dir)
    assert {"yield_pct", "yield_change_bps"}.issubset(macro.columns)
    assert macro.index.is_monotonic_increasing

    synced = imd.get_synchronized_market_data(synthetic_data_dir)
    assert list(synced.columns) == ["NIFTY_EQUITIES", "MUTUAL_FUNDS", "MCX_GOLD", "GOVT_10Y_GSEC"]
    assert not synced.isna().any().any()
    assert len(synced) == 319


def test_get_synchronized_market_data_uses_auto_discovery_when_none(
    synthetic_data_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(imd, "POSSIBLE_DATA_DIRS", [synthetic_data_dir])
    synced = imd.get_synchronized_market_data(data_dir=None)
    assert len(synced) == 319

