"""Unit tests for ml.xai_explainer (Permutation Importance & Narrative Attribution)."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from ml import xai_explainer as xai


def test_compute_feature_importances_ranks_signal_above_noise():
    rng = np.random.default_rng(42)
    n = 200
    # X1 is strong signal; X2 is moderate signal; X3 is pure noise
    x1 = rng.normal(0, 1, size=n)
    x2 = rng.normal(0, 1, size=n)
    x3 = rng.normal(0, 1, size=n)
    y = 3.0 * x1 + 1.2 * x2 + rng.normal(0, 0.1, size=n)

    X = pd.DataFrame({"signal_dominant": x1, "signal_secondary": x2, "noise": x3})
    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(X, y)

    imp = xai.compute_feature_importances(model, X, pd.Series(y), n_repeats=5, random_state=42)
    assert imp["feature"].iloc[0] == "signal_dominant"
    assert imp["feature"].iloc[1] == "signal_secondary"
    assert imp["feature"].iloc[2] == "noise"


def test_explain_asset_prediction_top_k_and_narrative():
    rng = np.random.default_rng(101)
    n = 150
    # Simulate features matching our real production column names
    rsi = rng.uniform(40, 75, size=n)
    sma_dist = rng.normal(0.02, 0.03, size=n)
    vol = rng.uniform(0.10, 0.25, size=n)
    y = pd.Series(0.05 * sma_dist + 0.001 * rsi + rng.normal(0, 0.01, size=n))

    X = pd.DataFrame({
        "dist_sma_50": sma_dist,
        "rsi_14": rsi,
        "volatility_21d": vol,
    })

    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(X, y)

    bundle = {
        "model": model,
        "feature_cols": list(X.columns),
    }

    latest = pd.DataFrame([{
        "dist_sma_50": 0.04,
        "rsi_14": 65.0,
        "volatility_21d": 0.12,
    }])

    res = xai.explain_asset_prediction(
        asset_name="NIFTY_EQUITIES",
        model_bundle=bundle,
        latest_feature_vector=latest,
        historical_X=X,
        historical_y=y,
        top_k=3,
    )

    assert res["asset_name"] == "NIFTY_EQUITIES"
    assert len(res["top_drivers"]) == 3
    assert "explanation" in res
    assert "NIFTY_EQUITIES" in res["explanation"]

    # Verify human readable mapping exists
    driver_labels = [d["display_name"] for d in res["top_drivers"]]
    assert any("Moving Average" in lab or "RSI" in lab for lab in driver_labels)


def test_explain_asset_prediction_missing_model_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        xai.explain_asset_prediction("NON_EXISTENT_ASSET", model_dir=tmp_path)

