
import os
import joblib

from train_model import download_prices, create_features


MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "saved_models"
)


def predict_return(ticker):
    """Predict an asset's return over the next 21 trading days."""

    model_path = os.path.join(
        MODEL_DIR,
        f"{ticker}_model.joblib"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found for {ticker}. "
            "Run train_model.py first."
        )

    saved = joblib.load(model_path)

    prices = download_prices(ticker)
    df = create_features(prices)

    # The latest row has features available as of the
    # latest downloaded trading date.
    latest = df.iloc[[-1]]

    model = saved["model"]
    features = saved["features"]

    predicted_return = float(
        model.predict(latest[features])[0]
    )

    return {
        "ticker": ticker,
        "forecast_horizon_trading_days": 21,
        "predicted_return": predicted_return,
        "predicted_return_percent": predicted_return * 100,
        "as_of_date": str(latest.index[0].date()),
    }


if __name__ == "__main__":
    for ticker in ["SPY", "TLT", "GLD", "SLV"]:
        try:
            print(predict_return(ticker))
        except Exception as exc:
            print(f"{ticker}: {exc}")
