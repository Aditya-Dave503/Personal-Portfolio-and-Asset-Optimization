"""Unit tests for ml.ml_engine."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ml import indian_market_data as imd
from ml import ml_engine


def test_calculate_rsi_extremes():
    dates = pd.bdate_range("2024-01-01", periods=40)
    up_prices = pd.Series(np.linspace(100, 200, 40), index=dates)
    down_prices = pd.Series(np.linspace(200, 100, 40), index=dates)

    rsi_up = ml_engine.calculate_rsi(up_prices, period=14).dropna()
    rsi_down = ml_engine.calculate_rsi(down_prices, period=14).dropna()

    assert rsi_up.iloc[-1] > 95.0
    assert rsi_down.iloc[-1] < 5.0


def test_extract_asset_features_is_causal_and_merges_macro(synthetic_data_dir: Path):
    market_df = imd.get_synchronized_market_data(synthetic_data_dir)
    macro_df = imd.load_macro_yield_features(synthetic_data_dir)
    nifty = market_df["NIFTY_EQUITIES"]

    feats = ml_engine.extract_asset_features(nifty, macro_df)
    expected_cols = {
        "daily_return",
        "trailing_5d",
        "trailing_252d",
        "volatility_21d",
        "vol_ratio_21_63",
        "dist_sma_50",
        "dist_sma_200",
        "rsi_14",
        "drawdown_252d",
    }
    assert expected_cols.issubset(feats.columns)

    # Verify strict causality: perturbing the last 10 days must not alter earlier features
    perturbed = nifty.copy()
    perturbed.iloc[-10:] = 0.05
    feats_perturbed = ml_engine.extract_asset_features(perturbed, macro_df)

    cutoff = -15
    pd.testing.assert_frame_equal(
        feats.iloc[:cutoff].dropna(),
        feats_perturbed.iloc[:cutoff].dropna(),
    )


def test_extract_forward_targets_exact_compounding():
    dates = pd.bdate_range("2024-01-01", periods=30)
    # Constant +1% daily return
    rets = pd.Series(0.01, index=dates)
    target = ml_engine.extract_forward_targets(rets, horizon_days=5)

    assert target.isna().sum() == 5
    assert target.iloc[0] == pytest.approx((1.01 ** 5) - 1.0, rel=1e-6)


def test_train_and_predict_asset_model_end_to_end(synthetic_data_dir: Path, isolated_model_dir: Path):
    market_df = imd.get_synchronized_market_data(synthetic_data_dir)
    macro_df = imd.load_macro_yield_features(synthetic_data_dir)

    bundle = ml_engine.train_asset_model(
        asset_name="NIFTY_EQUITIES",
        returns_series=market_df["NIFTY_EQUITIES"],
        macro_df=macro_df,
        horizon_days=21,
        n_splits=3,
    )
    assert (isolated_model_dir / "NIFTY_EQUITIES_model.joblib").exists()
    assert bundle["residual_variance_21d"] >= 0.0
    assert 0.0 <= bundle["hit_rate"] <= 1.0

    pred = ml_engine.predict_asset_return("NIFTY_EQUITIES", market_df["NIFTY_EQUITIES"], macro_df)
    assert -0.25 <= pred["annualized_expected_return"] <= 0.35
    assert pred["asset_name"] == "NIFTY_EQUITIES"


def test_predict_asset_return_missing_model_raises(synthetic_data_dir: Path, isolated_model_dir: Path):
    market_df = imd.get_synchronized_market_data(synthetic_data_dir)
    with pytest.raises(FileNotFoundError, match="Train models first"):
        ml_engine.predict_asset_return("NIFTY_EQUITIES", market_df["NIFTY_EQUITIES"])


def test_train_and_forecast_all_caches_models(synthetic_data_dir: Path, isolated_model_dir: Path):
    preds1, df1 = ml_engine.train_and_forecast_all(data_dir=synthetic_data_dir, retrain=True)
    assert set(preds1.keys()) == {"NIFTY_EQUITIES", "MUTUAL_FUNDS", "MCX_GOLD", "GOVT_10Y_GSEC"}
    assert len(df1) == 319

    # Second call with retrain=False should reuse saved .joblib artifacts and yield identical forecasts
    preds2, _ = ml_engine.train_and_forecast_all(data_dir=synthetic_data_dir, retrain=False)
    for asset in preds1:
        assert preds2[asset]["pred_21d_pct"] == preds1[asset]["pred_21d_pct"]


def test_extract_asset_features_without_macro_and_auto_dir_forecast(
    synthetic_data_dir: Path, isolated_model_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    market_df = imd.get_synchronized_market_data(synthetic_data_dir)
    feats_no_macro = ml_engine.extract_asset_features(market_df["NIFTY_EQUITIES"], macro_df=None)
    assert "gsec_yield_level" not in feats_no_macro.columns

    monkeypatch.setattr(ml_engine, "resolve_data_directory", lambda: synthetic_data_dir)
    preds, _ = ml_engine.train_and_forecast_all(data_dir=None, retrain=True)
    assert len(preds) == 4

