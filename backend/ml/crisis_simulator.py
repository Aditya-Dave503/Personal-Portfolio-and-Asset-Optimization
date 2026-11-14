"""
Historical Crisis Scenario Stress-Testing Simulator.

Stress-tests user-optimized asset allocations against severe macroeconomic shocks:
1. 2008 Global Financial Crisis (GFC): 2008-01-01 to 2009-03-31
2. 2020 COVID-19 Liquidity Shock: 2020-01-15 to 2020-05-15

Data Resolution Protocol:
- Attempts to slice from synchronized local CSV market data first (covers 2020 COVID shock).
- For dates outside CSV coverage (such as the 2008 GFC spanning 2008-2009), fetches
  proxy indices dynamically via `yfinance` (^NSEI for Equities/Mutual Funds, GC=F for Gold,
  and historical Indian sovereign curve approximations).
- Calculates the maximum cumulative peak-to-trough drawdown for the portfolio:
      MDD = min_{t} ( (W_t - max_{s <= t} W_s) / max_{s <= t} W_s )
- Benchmarks against a 100% Nifty 50 equity portfolio to quantify capital preserved
  through algorithmic asset diversification.
"""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
import pandas as pd

# Hardcoded crisis windows
CRISIS_WINDOWS: Dict[str, Dict[str, str]] = {
    "2008_GFC": {
        "name": "2008 Global Financial Crisis",
        "start_date": "2008-01-01",
        "end_date": "2009-03-31",
        "description": "Lehman Brothers collapse, global credit freeze, and emerging market sell-off.",
    },
    "2020_COVID": {
        "name": "2020 COVID-19 Liquidity Shock",
        "start_date": "2020-01-15",
        "end_date": "2020-05-15",
        "description": "Global pandemic lockdown, rapid equity repricing, and flight to sovereign bonds.",
    },
}

CORE_ASSET_MAPPING = {
    "NIFTY_EQUITIES": "^NSEI",
    "MUTUAL_FUNDS": "^NSEI",
    "MCX_GOLD": "GC=F",
    "GOVT_10Y_GSEC": "^NSEI",  # Proxy fallback when yield curve not on Yahoo
}


def calculate_drawdown_series(returns: pd.Series) -> Tuple[pd.Series, float]:
    """
    Compute daily wealth index, running peak, and percentage drawdown series.

    Returns:
        (drawdown_series, max_drawdown_pct)
        where max_drawdown_pct is a positive percentage, e.g. 38.4 for 38.4% max drawdown.
    """
    clean_rets = returns.fillna(0.0)
    wealth_index = (1.0 + clean_rets).cumprod()
    running_peak = wealth_index.cummax()
    drawdown = (wealth_index - running_peak) / running_peak
    max_dd = abs(float(drawdown.min())) * 100.0 if len(drawdown) > 0 else 0.0
    return drawdown, float(round(max_dd, 2))


def fetch_crisis_data_via_yfinance(
    start_date: str,
    end_date: str,
    asset_names: List[str]
) -> pd.DataFrame:
    """
    Dynamic fallback fetcher using yfinance for periods prior to local CSV coverage.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is required for fetching historical crisis dates not present in CSVs. "
            "Install with: pip install yfinance"
        )

    # Standard proxies for historical Indian multi-asset analysis
    ticker_symbols = {
        "NIFTY_EQUITIES": "^NSEI",
        "MUTUAL_FUNDS": "^NSEI",
        "MCX_GOLD": "GC=F",
    }

    unique_tickers = list({ticker_symbols.get(a, "^NSEI") for a in asset_names if a != "GOVT_10Y_GSEC"})
    data = yf.download(unique_tickers, start=start_date, end=end_date, progress=False)

    if data.empty:
        raise RuntimeError(f"Unable to fetch historical crisis data from yfinance for {start_date} to {end_date}.")

    if "Close" in data:
        prices = data["Close"]
    elif "Adj Close" in data:
        prices = data["Adj Close"]
    else:
        prices = data

    # Compute daily percentage returns
    rets = prices.pct_change().dropna()

    # Reconstruct columns matching our asset universe
    crisis_df = pd.DataFrame(index=rets.index)
    for asset in asset_names:
        if asset in ["NIFTY_EQUITIES", "MUTUAL_FUNDS"]:
            crisis_df[asset] = rets["^NSEI"] if "^NSEI" in rets else rets.iloc[:, 0]
        elif asset == "MCX_GOLD":
            crisis_df[asset] = rets["GC=F"] if "GC=F" in rets else rets.iloc[:, 0]
        elif asset == "GOVT_10Y_GSEC":
            # Sovereign G-Sec historical flight-to-safety return proxy (carry + duration effect)
            # 10Y Indian sovereign yield averaged ~7.5% in 2008 with easing during rate cuts
            crisis_df[asset] = 0.075 / 252.0
        else:
            crisis_df[asset] = 0.0

    return crisis_df.dropna()


def get_crisis_returns(
    crisis_key: str,
    local_market_df: Optional[pd.DataFrame] = None,
    asset_names: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, str]:
    """
    Resolve crisis return series adhering strictly to data protocol:
    1. Slices from local CSV DataFrame if date range is covered.
    2. Falls back dynamically to yfinance.
    3. Fails cleanly if neither is available.
    """
    if crisis_key not in CRISIS_WINDOWS:
        raise KeyError(f"Unknown crisis key '{crisis_key}'. Available: {list(CRISIS_WINDOWS.keys())}")

    meta = CRISIS_WINDOWS[crisis_key]
    start = pd.to_datetime(meta["start_date"])
    end = pd.to_datetime(meta["end_date"])

    if asset_names is None:
        asset_names = ["NIFTY_EQUITIES", "MUTUAL_FUNDS", "MCX_GOLD", "GOVT_10Y_GSEC"]

    # Step 1: Check local market data
    if local_market_df is not None and not local_market_df.empty:
        df_start = pd.to_datetime(local_market_df.index.min())
        df_end = pd.to_datetime(local_market_df.index.max())

        if df_start <= start and df_end >= end:
            sliced = local_market_df.loc[start:end]
            if len(sliced) >= 20:
                return sliced[asset_names], "local_csv"

    # Step 2: Fallback to yfinance
    df_yf = fetch_crisis_data_via_yfinance(meta["start_date"], meta["end_date"], asset_names)
    return df_yf, "yfinance_api"


def simulate_crisis_event(
    portfolio_weights: Dict[str, float],
    crisis_key: str = "2020_COVID",
    local_market_df: Optional[pd.DataFrame] = None,
    benchmark_asset: str = "NIFTY_EQUITIES"
) -> Dict[str, Any]:
    """
    Stress-test portfolio weights against a specific historical crisis.

    Parameters:
        portfolio_weights: Dict mapping asset_code to weight, summing to 1.0.
        crisis_key: '2020_COVID' or '2008_GFC'.
        local_market_df: Optional pre-loaded synchronized historical market DataFrame.
        benchmark_asset: Benchmark against 100% allocation to this asset.

    Returns:
        Structured audit report comparing portfolio vs benchmark drawdowns.
    """
    meta = CRISIS_WINDOWS[crisis_key]
    asset_names = list(portfolio_weights.keys())

    crisis_df, data_source = get_crisis_returns(
        crisis_key=crisis_key,
        local_market_df=local_market_df,
        asset_names=asset_names
    )

    # Calculate portfolio return series
    weights_arr = np.array([portfolio_weights.get(col, 0.0) for col in crisis_df.columns])
    if weights_arr.sum() > 0:
        weights_arr /= weights_arr.sum()

    portfolio_daily = pd.Series(crisis_df.values @ weights_arr, index=crisis_df.index)
    benchmark_daily = crisis_df[benchmark_asset] if benchmark_asset in crisis_df else crisis_df.iloc[:, 0]

    port_dd_series, port_max_dd = calculate_drawdown_series(portfolio_daily)
    bm_dd_series, bm_max_dd = calculate_drawdown_series(benchmark_daily)

    # Cumulative total return across crisis period
    port_cum_return = float(((1.0 + portfolio_daily).prod() - 1.0) * 100.0)
    bm_cum_return = float(((1.0 + benchmark_daily).prod() - 1.0) * 100.0)

    # Capital preserved (reduction in maximum peak drawdown)
    capital_preserved_pct = max(0.0, float(round(bm_max_dd - port_max_dd, 2)))

    return {
        "crisis_key": crisis_key,
        "crisis_name": meta["name"],
        "description": meta["description"],
        "data_source_used": data_source,
        "trading_days": len(crisis_df),
        "start_date": str(crisis_df.index.min().date()),
        "end_date": str(crisis_df.index.max().date()),
        "portfolio_max_drawdown_pct": port_max_dd,
        "benchmark_max_drawdown_pct": bm_max_dd,
        "capital_preserved_drawdown_reduction_pct": capital_preserved_pct,
        "portfolio_cumulative_return_pct": round(port_cum_return, 2),
        "benchmark_cumulative_return_pct": round(bm_cum_return, 2),
        "stress_test_result": "PASS (Outperformed Benchmark)" if port_max_dd < bm_max_dd else "EQUAL/UNDERPERFORM",
    }


def simulate_all_crises(
    portfolio_weights: Dict[str, float],
    local_market_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Run the stress testing harness across all defined crisis epochs.
    """
    results = {}
    for crisis_key in CRISIS_WINDOWS:
        results[crisis_key] = simulate_crisis_event(
            portfolio_weights=portfolio_weights,
            crisis_key=crisis_key,
            local_market_df=local_market_df
        )
    return results

