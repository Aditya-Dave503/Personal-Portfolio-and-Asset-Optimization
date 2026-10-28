
import os
import joblib
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# Assets used for this starter demonstration.
# These are US-listed ETF proxies, not Indian securities.
ASSETS = {
    "SPY": "US stocks",
    "TLT": "US long-term government bonds",
    "GLD": "Gold",
    "SLV": "Silver",
}

MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "saved_models"
)

os.makedirs(MODEL_DIR, exist_ok=True)


def download_prices(ticker):
    """Download historical daily prices for one asset."""

    data = yf.download(
        ticker,
        period="10y",
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise ValueError(f"No historical data received for {ticker}")

    # Handle the column format returned by yfinance.
    close = data["Close"]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    return close.dropna().astype(float)


def create_features(prices):
    """Create features using only information available at each date."""

    df = pd.DataFrame({"price": prices})

    # Daily percentage return.
    df["return_1d"] = df["price"].pct_change()

    # Recent returns and momentum.
    df["return_5d"] = df["price"].pct_change(5)
    df["return_21d"] = df["price"].pct_change(21)
    df["return_63d"] = df["price"].pct_change(63)

    # Recent historical volatility.
    df["volatility_21d"] = (
        df["return_1d"]
        .rolling(21)
        .std()
        * np.sqrt(252)
    )

    # Target: return over the NEXT 21 trading days.
    df["target"] = (
        df["price"].shift(-21) / df["price"] - 1
    )

    return df.dropna()


def train_one_asset(ticker):
    """Train and evaluate a Random Forest for one asset."""

    print(f"\nTraining model for {ticker}...")

    prices = download_prices(ticker)
    df = create_features(prices)

    feature_columns = [
        "return_1d",
        "return_5d",
        "return_21d",
        "return_63d",
        "volatility_21d",
    ]

    # Split chronologically, not randomly.
    # The most recent 20% is reserved for testing.
    split = int(len(df) * 0.8)

    train = df.iloc[:split]
    test = df.iloc[split:]

    if len(train) < 252 or len(test) < 50:
        raise ValueError(
            f"Insufficient historical observations for {ticker}"
        )

    X_train = train[feature_columns]
    y_train = train["target"]

    X_test = test[feature_columns]
    y_test = test["target"]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    # Compare predictions against a zero-return baseline.
    baseline_mae = mean_absolute_error(
        y_test,
        np.zeros(len(y_test))
    )

    print(f"Test observations: {len(test)}")
    print(f"Model MAE: {mae:.4f}")
    print(f"Zero-return baseline MAE: {baseline_mae:.4f}")

    model_path = os.path.join(
        MODEL_DIR,
        f"{ticker}_model.joblib"
    )

    joblib.dump(
        {
            "model": model,
            "features": feature_columns,
            "ticker": ticker,
        },
        model_path
    )

    print(f"Saved model to: {model_path}")


def main():
    for ticker in ASSETS:
        try:
            train_one_asset(ticker)
        except Exception as exc:
            print(f"Could not train {ticker}: {exc}")


if __name__ == "__main__":
    main()
