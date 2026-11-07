"""
Feature Engineering Engine for Multi-Asset Portfolio Optimization.

Strictly separates feature computation (available at date t) from forward targets
(realized between t+1 and t+h), guaranteeing ZERO lookahead bias and allowing
real-time inference on the most recent market date.
"""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
import yfinance as yf


# Default 3-asset macro universe (Silver 'SLV' is deliberately excluded).
DEFAULT_ASSETS = {
    "SPY": "Equities (US Large Cap / S&P 500)",
    "TLT": "Long-Term Treasury Bonds (20+ Year)",
    "GLD": "Gold (SPDR Gold Shares)",
}

# Optional Indian proxy mapping if running on NSE:
# {"NIFTYBEES.NS": "Equities", "GOLDBEES.NS": "Gold", "SETFNIF50.NS": "Equities"}


def download_historical_prices(
    ticker: str,
    period: str = "10y",
    interval: str = "1d"
) -> pd.Series:
    """Download clean, split-and-dividend adjusted closing prices."""
    data = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False
    )
    if data.empty:
        raise ValueError(f"No price data retrieved for ticker: {ticker}")

    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = close.dropna().astype(float)
    close.name = ticker
    return close


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI) without lookahead."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def extract_features(prices: pd.Series) -> pd.DataFrame:
    """
    Extract causal, leakage-free technical & volatility features at date t.
    All calculations use strictly trailing historical information.
    """
    df = pd.DataFrame({"price": prices}, index=prices.index)

    # 1. Multi-scale percentage returns
    df["return_1d"] = df["price"].pct_change(1)
    df["return_5d"] = df["price"].pct_change(5)
    df["return_21d"] = df["price"].pct_change(21)
    df["return_63d"] = df["price"].pct_change(63)
    df["return_126d"] = df["price"].pct_change(126)

    # 2. Multi-scale annualized rolling volatility
    df["volatility_21d"] = df["return_1d"].rolling(21).std() * np.sqrt(252)
    df["volatility_63d"] = df["return_1d"].rolling(63).std() * np.sqrt(252)

    # 3. Volatility regime ratio (expansion vs contraction)
    df["vol_ratio_21_63"] = df["volatility_21d"] / (df["volatility_63d"] + 1e-8)

    # 4. Trend & moving average regimes
    sma_50 = df["price"].rolling(50).mean()
    sma_200 = df["price"].rolling(200).mean()
    df["dist_sma_50"] = df["price"] / (sma_50 + 1e-8) - 1.0
    df["dist_sma_200"] = df["price"] / (sma_200 + 1e-8) - 1.0
    df["trend_spread_50_200"] = sma_50 / (sma_200 + 1e-8) - 1.0

    # 5. Momentum Oscillator: 14-day RSI
    df["rsi_14"] = calculate_rsi(df["price"], period=14)

    # 6. Peak Drawdown over the trailing 1-year window (252 trading days)
    rolling_max_252 = df["price"].rolling(252, min_periods=63).max()
    df["drawdown_252d"] = df["price"] / (rolling_max_252 + 1e-8) - 1.0

    feature_cols = [
        "return_1d",
        "return_5d",
        "return_21d",
        "return_63d",
        "return_126d",
        "volatility_21d",
        "volatility_63d",
        "vol_ratio_21_63",
        "dist_sma_50",
        "dist_sma_200",
        "trend_spread_50_200",
        "rsi_14",
        "drawdown_252d",
    ]

    return df[feature_cols]


def extract_targets(prices: pd.Series, horizon_days: int = 21) -> pd.Series:
    """
    Compute forward return target over the next `horizon_days` trading days:
    R_{t -> t+h} = (Price_{t+h} / Price_t) - 1.
    """
    target = prices.shift(-horizon_days) / prices - 1.0
    target.name = f"target_{horizon_days}d"
    return target


def prepare_training_dataset(
    prices: pd.Series,
    horizon_days: int = 21
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare labeled dataset for model training. Drops trailing rows where
    the forward target is not yet observed.
    """
    features = extract_features(prices)
    target = extract_targets(prices, horizon_days=horizon_days)

    combined = pd.concat([features, target], axis=1).dropna()
    X = combined[features.columns]
    y = combined[target.name]
    return X, y


def prepare_inference_features(prices: pd.Series) -> pd.DataFrame:
    """
    Extract the single latest feature row available at the present date T.
    Unlike training, this DOES NOT drop recent rows, giving zero-staleness inference.
    """
    features = extract_features(prices)
    # The last row contains today's market features (as long as rolling windows are met).
    latest = features.dropna().iloc[[-1]]
    return latest
