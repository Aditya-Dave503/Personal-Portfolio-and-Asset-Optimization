"""Unit tests for ml.crisis_simulator (Historical Crisis Stress-Testing)."""

import pandas as pd
import pytest

from ml import crisis_simulator as cs


def test_calculate_drawdown_series_peak_to_trough():
    # Price path: 100 -> 120 (+20%) -> 90 (-25%) -> 100
    rets = pd.Series([0.20, -0.25, 0.1111])
    dd_series, max_dd = cs.calculate_drawdown_series(rets)

    # Peak was at 120. At price 90, drawdown = (90 - 120) / 120 = -0.25 (-25%)
    assert max_dd == pytest.approx(25.0, abs=0.1)
    assert dd_series.min() == pytest.approx(-0.25, abs=0.01)


def test_simulate_crisis_event_2020_covid_uses_local_data():
    dates = pd.date_range("2020-01-01", "2020-06-01", freq="B")
    # Synthetic crash: Equities plunge 30%, Gold and G-Sec act as safe haven cushions
    n_days = len(dates)
    equities = [-0.02 if i % 2 == 0 else 0.005 for i in range(n_days)]
    gold = [0.003 for _ in range(n_days)]
    gsec = [0.0003 for _ in range(n_days)]

    local_df = pd.DataFrame({
        "NIFTY_EQUITIES": equities,
        "MUTUAL_FUNDS": equities,
        "MCX_GOLD": gold,
        "GOVT_10Y_GSEC": gsec,
    }, index=dates)

    # Diversified portfolio: 30% Equities, 35% Gold, 35% G-Sec
    weights = {
        "NIFTY_EQUITIES": 0.30,
        "MUTUAL_FUNDS": 0.00,
        "MCX_GOLD": 0.35,
        "GOVT_10Y_GSEC": 0.35,
    }

    res = cs.simulate_crisis_event(
        portfolio_weights=weights,
        crisis_key="2020_COVID",
        local_market_df=local_df,
    )

    assert res["crisis_key"] == "2020_COVID"
    assert res["data_source_used"] == "local_csv"
    # Diversified portfolio must exhibit lower maximum drawdown than 100% equity benchmark
    assert res["portfolio_max_drawdown_pct"] < res["benchmark_max_drawdown_pct"]
    assert res["capital_preserved_drawdown_reduction_pct"] > 0.0
    assert "PASS" in res["stress_test_result"]


def test_simulate_crisis_event_2008_gfc_yfinance_fallback(monkeypatch: pytest.MonkeyPatch):
    # Mock yfinance response to make unit tests deterministic and independent of network
    dates = pd.bdate_range("2008-01-02", "2008-06-30")
    mock_rets = pd.DataFrame({
        "NIFTY_EQUITIES": [-0.01] * len(dates),
        "MUTUAL_FUNDS": [-0.01] * len(dates),
        "MCX_GOLD": [0.002] * len(dates),
        "GOVT_10Y_GSEC": [0.0003] * len(dates),
    }, index=dates)

    monkeypatch.setattr(cs, "fetch_crisis_data_via_yfinance", lambda s, e, a: mock_rets)

    weights = {"NIFTY_EQUITIES": 0.40, "MUTUAL_FUNDS": 0.20, "MCX_GOLD": 0.20, "GOVT_10Y_GSEC": 0.20}
    res = cs.simulate_crisis_event(
        portfolio_weights=weights,
        crisis_key="2008_GFC",
        local_market_df=None,  # Forces yfinance fallback
    )

    assert res["crisis_key"] == "2008_GFC"
    assert res["data_source_used"] == "yfinance_api"
    assert res["portfolio_max_drawdown_pct"] < res["benchmark_max_drawdown_pct"]
    assert res["capital_preserved_drawdown_reduction_pct"] > 0.0


def test_simulate_all_crises(monkeypatch: pytest.MonkeyPatch):
    dates_2020 = pd.date_range("2019-12-01", "2020-06-01", freq="B")
    mock_df_2020 = pd.DataFrame({
        "NIFTY_EQUITIES": [0.001] * len(dates_2020),
        "MUTUAL_FUNDS": [0.001] * len(dates_2020),
        "MCX_GOLD": [0.001] * len(dates_2020),
        "GOVT_10Y_GSEC": [0.0003] * len(dates_2020),
    }, index=dates_2020)

    dates_2008 = pd.date_range("2008-01-01", "2009-04-01", freq="B")
    mock_df_2008 = pd.DataFrame({
        "NIFTY_EQUITIES": [-0.005] * len(dates_2008),
        "MUTUAL_FUNDS": [-0.005] * len(dates_2008),
        "MCX_GOLD": [0.002] * len(dates_2008),
        "GOVT_10Y_GSEC": [0.0003] * len(dates_2008),
    }, index=dates_2008)

    # Mock yfinance call for offline testing isolation
    monkeypatch.setattr(cs, "fetch_crisis_data_via_yfinance", lambda s, e, a: mock_df_2008)

    weights = {"NIFTY_EQUITIES": 0.5, "MUTUAL_FUNDS": 0.2, "MCX_GOLD": 0.15, "GOVT_10Y_GSEC": 0.15}
    results = cs.simulate_all_crises(portfolio_weights=weights, local_market_df=mock_df_2020)

    assert "2020_COVID" in results
    assert "2008_GFC" in results
    assert results["2020_COVID"]["data_source_used"] == "local_csv"
    assert results["2008_GFC"]["data_source_used"] == "yfinance_api"

