"""Unit tests for ml.predict."""

from pathlib import Path
import joblib
import pytest

from ml import predict


def test_predict_return_excludes_silver():
    with pytest.raises(ValueError, match="deliberately excluded"):
        predict.predict_return(" slv ")


def test_predict_return_missing_model_raises(isolated_model_dir: Path):
    with pytest.raises(FileNotFoundError, match="No trained model found"):
        predict.predict_return("UNKNOWN_ASSET")


def test_predict_return_modern_bundle_and_legacy_ticker_mapping(isolated_model_dir: Path):
    bundle = {
        "asset_name": "NIFTY_EQUITIES",
        "cv_mae": 0.025,
        "cv_rmse": 0.035,
        "annualized_expected_return": 0.144,
        "as_of_date": "2025-12-31",
    }
    joblib.dump(bundle, isolated_model_dir / "NIFTY_EQUITIES_model.joblib")

    res_direct = predict.predict_return("NIFTY_EQUITIES")
    res_legacy = predict.predict_return("SPY")

    assert res_direct["asset_code"] == "NIFTY_EQUITIES"
    assert res_direct["predicted_annual_return_percent"] == pytest.approx(14.4)
    assert res_direct["predicted_return_percent"] == pytest.approx(1.2)
    assert res_legacy["ticker"] == "SPY"
    assert res_legacy["asset_code"] == "NIFTY_EQUITIES"


def test_predict_return_legacy_bundle_format(isolated_model_dir: Path):
    legacy_bundle = {"model": object(), "features": ["f1", "f2"]}
    joblib.dump(legacy_bundle, isolated_model_dir / "CUSTOM_model.joblib")

    res = predict.predict_return("CUSTOM")
    assert res["predicted_return"] == 0.015
    assert res["predicted_return_percent"] == 1.5


def test_predict_all_returns_all_four_core_assets(isolated_model_dir: Path):
    for asset in ["NIFTY_EQUITIES", "MUTUAL_FUNDS", "MCX_GOLD", "GOVT_10Y_GSEC"]:
        joblib.dump(
            {"cv_mae": 0.02, "cv_rmse": 0.03, "annualized_expected_return": 0.12},
            isolated_model_dir / f"{asset}_model.joblib",
        )

    all_preds = predict.predict_all()
    assert set(all_preds.keys()) == {"NIFTY_EQUITIES", "MUTUAL_FUNDS", "MCX_GOLD", "GOVT_10Y_GSEC"}

