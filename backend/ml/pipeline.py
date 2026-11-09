"""
OptiWealth Production Machine Learning Pipeline.

Seamless End-to-End Orchestrator:
1. Ingests user demographics (age, runway, savings, loss tolerance) -> computes objective risk score R_u in [0, 1].
2. Ingests 10-year Indian datasets (Nifty 50, Mutual Funds, MCX Gold, 10Y G-Secs).
3. Forecasts forward returns using regularized Gradient Boosting.
4. Executes constrained optimization: Max Sharpe, Min Variance, or Black-Litterman.
5. Enforces VaR_99% <= Max Loss Threshold and returns institutional advisory report.

Usage:
    python pipeline.py --amount 500000 --age 30 --horizon 10 --loss-tolerance 0.20
    python pipeline.py --amount 1000000 --risk-score 8 --mode max_sharpe
"""

import argparse
from pathlib import Path
from .indian_market_data import (
    ASSET_METADATA,
    resolve_data_directory,
    get_synchronized_market_data,
)

from .risk_profiler import calculate_user_risk_profile
from .ml_engine import train_and_forecast_all
from .portfolio_optimizer import optimize_indian_portfolio


def run_pipeline(
    investment_amount: float = 500000.0,
    age: int = 30,
    investment_horizon_years: int = 10,
    savings_to_income_ratio: float = 0.5,
    max_loss_tolerance_pct: float = 0.20,
    explicit_risk_score: int = None,
    optimization_mode: str = "black_litterman",
    retrain: bool = False,
    risk_free_rate: float = 0.065,
    data_dir_str: str = None
):
    data_dir = resolve_data_directory(data_dir_str)

    # 1. Determine Risk Profile R_u (Section 1 of README)
    if explicit_risk_score is not None:
        risk_score = explicit_risk_score
        profile_meta = {
            "discrete_risk_score": risk_score,
            "continuous_risk_score_Ru": round(risk_score / 10.0, 2),
            "category": "Direct User Input",
            "max_loss_tolerance_pct": max_loss_tolerance_pct,
        }
    else:
        profile_meta = calculate_user_risk_profile(
            age=age,
            investment_horizon_years=investment_horizon_years,
            savings_to_income_ratio=savings_to_income_ratio,
            max_loss_tolerance_pct=max_loss_tolerance_pct
        )
        risk_score = profile_meta["discrete_risk_score"]

    print("=" * 75)
    print("      OPTIWEALTH ML ASSET ALLOCATION ENGINE (INDIAN MARKETS)")
    print("=" * 75)
    print(f"Data Directory:             {data_dir}")
    print(f"Total Investment Capital:   INR {investment_amount:,.2f}")
    print(f"Investor Risk Budget R_u:   {profile_meta['continuous_risk_score_Ru']:.4f} (Score: {risk_score}/10)")
    print(f"Investor Profile Category:  {profile_meta['category']}")
    print(f"Max Acceptable Drawdown:    {max_loss_tolerance_pct:.1%}")
    print(f"Optimization Methodology:   {optimization_mode.upper()}")
    print(f"Excluded Assets:            Silver (Deliberately Removed)")
    print(f"Core Indian Asset Universe: Nifty 50, Mutual Funds, MCX Gold, 10Y G-Secs")
    print(f"Risk-Free Benchmark:        {risk_free_rate * 100:.2f}% (RBI Sovereign Curve)")
    print("-" * 75)

    print("\n[Step 1/3] Ingesting 10-Year Indian Market Time Series...")
    market_df = get_synchronized_market_data(data_dir)
    print(f"  Loaded {len(market_df):,} daily observations across {market_df.shape[1]} core asset classes.")
    print(f"  Historical Period: {market_df.index.min().date()} to {market_df.index.max().date()}")

    print("\n[Step 2/3] Machine Learning Feature Extraction & Forward Return Modeling...")
    predictions, _ = train_and_forecast_all(data_dir=data_dir, retrain=retrain)

    print("\n  Live Asset Forecasts (Zero-Staleness):")
    for asset, p in predictions.items():
        meta = ASSET_METADATA.get(asset, {})
        print(
            f"    - {meta.get('name', asset):<40} "
            f"21d Return: {p['pred_21d_pct']:>+5.2f}% | "
            f"Ann. Expected: {p['annualized_expected_return_pct']:>+6.2f}% | "
            f"21d Uncertainty: {p['residual_variance_21d']:.6f}"
        )

    print(f"\n[Step 3/3] Solving Constrained Optimization ({optimization_mode})...")
    portfolio = optimize_indian_portfolio(
        investment_amount=investment_amount,
        risk_score=risk_score,
        risk_free_rate=risk_free_rate,
        optimization_mode=optimization_mode,
        max_loss_tolerance_pct=max_loss_tolerance_pct,
        data_dir=data_dir,
        custom_predictions=predictions
    )

    print("\n" + "=" * 75)
    print("                OPTIMAL INVESTMENT ADVISORY ALLOCATION")
    print("=" * 75)
    print(f"Investor Profile:         {portfolio['risk_profile_category']} (Score: {risk_score}/10, R_u: {profile_meta['continuous_risk_score_Ru']})")
    print(f"Optimization Objective:   {portfolio['method']}")
    print(f"Expected Annual Return:   {portfolio['expected_annual_return_pct']:>+6.2f}%")
    print(f"Portfolio Volatility:     {portfolio['expected_annual_volatility_pct']:>6.2f}%")
    print(f"Sharpe Ratio (Rf=6.5%):   {portfolio['sharpe_ratio']:>6.2f}")
    print(f"1-Month 95% VaR:          {portfolio['value_at_risk_1m_95pct']:>6.2f}%")
    print(f"1-Month 99% VaR:          {portfolio['value_at_risk_1m_99pct']:>6.2f}% (Limit: <= {max_loss_tolerance_pct:.1%}) -> {portfolio['var_99_constraint_status']}")
    print(f"1-Month 95% CVaR:         {portfolio['conditional_var_1m_95pct']:>6.2f}%")
    print("-" * 75)
    print(f"{'Asset Class':<35} {'Category':<24} {'Weight':<10} {'Allocated (INR)'}")
    print("-" * 75)
    for alloc in portfolio["allocations"]:
        print(
            f"{alloc['asset_name']:<35} "
            f"{alloc['category']:<24} "
            f"{alloc['weight_percent']:>5.2f}%    "
            f"INR {alloc['allocated_amount']:>12,.2f}"
        )
    print("=" * 75)
    return portfolio


def main():
    parser = argparse.ArgumentParser(description="OptiWealth Indian Market ML Allocation Engine")
    parser.add_argument("--amount", type=float, default=500000.0, help="Total investment amount in INR")
    parser.add_argument("--age", type=int, default=30, help="Investor age")
    parser.add_argument("--horizon", type=int, default=10, help="Investment runway in years")
    parser.add_argument("--savings-ratio", type=float, default=0.5, help="Savings to income ratio")
    parser.add_argument("--loss-tolerance", type=float, default=0.20, help="Max loss tolerance (e.g. 0.20 for 20%)")
    parser.add_argument("--risk-score", type=int, default=None, help="Explicit risk score (1-10) to override profiling")
    parser.add_argument("--mode", type=str, default="black_litterman", choices=["black_litterman", "max_sharpe", "min_variance"], help="Optimization mode")
    parser.add_argument("--train", action="store_true", help="Retrain ML models using walk-forward CV")
    parser.add_argument("--rf", type=float, default=0.065, help="Risk-free rate (default: 0.065 for India)")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to raw CSV data directory")

    args = parser.parse_args()
    run_pipeline(
        investment_amount=args.amount,
        age=args.age,
        investment_horizon_years=args.horizon,
        savings_to_income_ratio=args.savings_ratio,
        max_loss_tolerance_pct=args.loss_tolerance,
        explicit_risk_score=args.risk_score,
        optimization_mode=args.mode,
        retrain=args.train,
        risk_free_rate=args.rf,
        data_dir_str=args.data_dir
    )


if __name__ == "__main__":
    main()
