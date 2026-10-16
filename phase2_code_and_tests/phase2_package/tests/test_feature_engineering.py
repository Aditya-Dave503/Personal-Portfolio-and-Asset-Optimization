import numpy as np
import pandas as pd

from features.feature_engineering import _forward_return_targets, _rolling_return


def test_forward_targets_start_after_feature_date():
    returns = pd.Series([0.10, 0.20, 0.30, 0.40])
    target = _forward_return_targets(returns, horizons=(2,))
    # At index 0, compound returns at indices 1 and 2, not index 0.
    assert np.isclose(target.loc[0, "future_return_2d"], (1.20 * 1.30) - 1)
    assert pd.isna(target.loc[2, "future_return_2d"])
    assert pd.isna(target.loc[3, "future_return_2d"])


def test_forward_target_requires_complete_valid_future_window():
    returns = pd.Series([0.01, 0.02, np.nan, 0.04, 0.05])
    target = _forward_return_targets(returns, horizons=(2,))
    assert pd.isna(target.loc[0, "future_return_2d"])
    assert pd.isna(target.loc[1, "future_return_2d"])
    assert pd.isna(target.loc[3, "future_return_2d"])
    assert pd.isna(target.loc[4, "future_return_2d"])


def test_rolling_return_compounds_simple_returns():
    values = pd.Series([0.10, 0.20, 0.30])
    result = _rolling_return(values, 2)
    assert pd.isna(result.iloc[0])
    assert np.isclose(result.iloc[1], 1.10 * 1.20 - 1)
