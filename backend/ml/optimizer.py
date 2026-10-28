
import numpy as np
import pandas as pd

from scipy.optimize import minimize
from train_model import download_prices


ASSETS = ["SPY", "TLT", "GLD", "SLV"]


def optimize_portfolio(
    investment_amount,
    max_weight=0.60
):
    """
    Find a long-only, minimum-variance allocation.

    investment_amount: total money to allocate.
    max_weight: maximum fraction assigned to one asset.
 
    This is a research demonstration, not financial advice.
    """

    if investment_amount <= 0:
        raise ValueError("Investment amount must be positive.")

    if not 0.25 <= max_weight <= 1.0:
        raise ValueError(
            "max_weight must be between 0.25 and 1.0 "
            "for four assets."
        )

    price_data = {}

    for ticker in ASSETS:
        prices = download_prices(ticker)
        price_data[ticker] = prices

    # Align dates so each row represents the same trading day.
    prices_df = pd.concat(
        price_data,
        axis=1,
        join="inner"
    ).dropna()

    if len(prices_df) < 100:
        raise ValueError(
            "Not enough overlapping historical price data."
        )

    # Calculate daily asset returns.
    returns = prices_df.pct_change().dropna()

    # Daily covariance matrix.
    covariance = returns.cov().to_numpy()

    number_of_assets = len(ASSETS)

    # Every asset starts with an equal allocation.
    initial_weights = np.repeat(
        1.0 / number_of_assets,
        number_of_assets
    )

    # Minimize portfolio variance: w.T @ covariance @ w
    def portfolio_variance(weights):
        return float(
            weights @ covariance @ weights
        )

    # All weights must sum to 1.
    constraints = [
        {
            "type": "eq",
            "fun": lambda weights: np.sum(weights) - 1.0
        }
    ]

    # No short selling; limit concentration in one asset.
    bounds = [
        (0.0, max_weight)
        for _ in ASSETS
    ]

    result = minimize(
        portfolio_variance,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        raise RuntimeError(
            f"Optimization failed: {result.message}"
        )

    weights = result.x

    # Annualized historical volatility estimate.
    daily_volatility = np.sqrt(
        max(portfolio_variance(weights), 0.0)
    )

    annualized_volatility = daily_volatility * np.sqrt(252)

    allocations = []

    for ticker, weight in zip(ASSETS, weights):
        allocations.append({
            "ticker": ticker,
            "weight_percent": round(weight * 100, 2),
            "investment_amount": round(
                investment_amount * weight, 2
            )
        })

    return {
        "investment_amount": round(investment_amount, 2),
        "method": "minimum_variance",
        "historical_annualized_volatility_percent": round(
            annualized_volatility * 100, 2
        ),
        "allocations": allocations,
        "warning": (
            "Historical risk estimates do not guarantee "
            "future performance. Assets are US-listed ETFs."
        )
    }


if __name__ == "__main__":
    portfolio = optimize_portfolio(
        investment_amount=10000
    )

    print("\nPortfolio recommendation (research demo)")
    print("-----------------------------------------")

    for item in portfolio["allocations"]:
        print(
            f"{item['ticker']}: "
            f"{item['weight_percent']}% "
            f"(${item['investment_amount']})"
        )

    print(
        "\nEstimated annualized volatility: "
        f"{portfolio['historical_annualized_volatility_percent']}%"
    )

    print("\n", portfolio["warning"])
