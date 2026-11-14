"""
Interactive Demonstration Harness for Portfolio Intelligence & Simulation Modules:
1. Hierarchical Risk Parity (HRP) Optimizer
2. Non-Gaussian Tail Risk (Cornish-Fisher VaR & CVaR)
3. Explainable AI (XAI) Attribution & Natural Language Narrative
4. Monte Carlo Goal Simulator (Geometric Brownian Motion + SIP)
5. Historical Crisis Scenario Stress-Testing (2008 GFC & 2020 COVID)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure backend root is on sys.path
_ML_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _ML_DIR.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))

from ml.indian_market_data import resolve_data_directory, get_synchronized_market_data, load_macro_yield_features
from ml.ml_engine import train_and_forecast_all, extract_asset_features, extract_forward_targets
from ml.hrp_optimizer import hrp_optimize
from ml.tail_risk import evaluate_tail_risk
from ml.xai_explainer import explain_asset_prediction
from ml.goal_simulator import simulate_goal_paths
from ml.crisis_simulator import simulate_all_crises


def run_full_demo():
    print("=" * 80)
    print("      OPTIWEALTH PORTFOLIO OPTIMIZATION & STOCHASTIC SIMULATION HARNESS")
    print("=" * 80)

    # 0. Load Market Data
    print("\n[Loading Indian Multi-Asset Market Data...]")
    data_dir = resolve_data_directory()
    market_df = get_synchronized_market_data(data_dir)
    macro_df = load_macro_yield_features(data_dir)
    print(f"  Loaded {len(market_df):,} daily observations across {list(market_df.columns)}")
    print(f"  Historical Coverage: {market_df.index.min().date()} to {market_df.index.max().date()}")

    # =========================================================================
    # MODULE 1: Hierarchical Risk Parity (HRP)
    # =========================================================================
    print("\n" + "=" * 80)
    print("  MODULE 1: HIERARCHICAL RISK PARITY (HRP) OPTIMIZER")
    print("=" * 80)
    print("  Computing tree clustering and recursive bisection without inverting Sigma...")
    hrp_weights = hrp_optimize(market_df)
    
    print("\n  Allocated HRP Portfolio Weights:")
    for asset, w in hrp_weights.items():
        print(f"    - {asset:<20}: {w * 100:>6.2f}%")

    # =========================================================================
    # MODULE 2: Non-Gaussian Tail Risk & Cornish-Fisher Expansion
    # =========================================================================
    print("\n" + "=" * 80)
    print("  MODULE 2: NON-GAUSSIAN TAIL RISK & CORNISH-FISHER EXPANSION")
    print("=" * 80)
    
    # Portfolio daily return series
    weights_vec = np.array([hrp_weights[c] for c in market_df.columns])
    port_returns = market_df.to_numpy() @ weights_vec
    
    tail_metrics = evaluate_tail_risk(port_returns, alpha=0.05)
    print(f"  Confidence Level:         {tail_metrics['confidence_level_pct']:.0f}%")
    print(f"  Sample Skewness (S):       {tail_metrics['skewness']:>+7.4f} (Negative = fat left crash tail)")
    print(f"  Excess Kurtosis (K - 3):   {tail_metrics['excess_kurtosis']:>+7.4f} (Leptokurtic fat tails)")
    print(f"  Standard Normal Quantile:  {tail_metrics['z_gaussian']:>+7.4f}")
    print(f"  Cornish-Fisher Quantile:   {tail_metrics['z_cornish_fisher']:>+7.4f}")
    print("-" * 80)
    print(f"  Standard Gaussian VaR_95%: {tail_metrics['gaussian_var'] * 100:>6.2f}% (Daily)")
    print(f"  Cornish-Fisher mVaR_95%:   {tail_metrics['modified_var'] * 100:>6.2f}% (Daily - Skew/Kurtosis adjusted)")
    print(f"  Conditional VaR (CVaR):    {tail_metrics['cvar_expected_shortfall'] * 100:>6.2f}% (Expected Shortfall in tail)")

    # =========================================================================
    # MODULE 3: Explainable AI (XAI) Attribution
    # =========================================================================
    print("\n" + "=" * 80)
    print("  MODULE 3: EXPLAINABLE AI (XAI) ATTRIBUTION & NATURAL LANGUAGE NARRATIVE")
    print("=" * 80)
    
    # Run ML engine to get predictions and feature matrices
    predictions, _ = train_and_forecast_all(data_dir=data_dir, retrain=False)
    
    for asset in ["NIFTY_EQUITIES", "MCX_GOLD"]:
        feats = extract_asset_features(market_df[asset], macro_df)
        targets = extract_forward_targets(market_df[asset], horizon_days=21)
        labeled = pd.concat([feats, targets], axis=1).dropna()
        feature_cols = [c for c in feats.columns if c in labeled.columns]
        
        explanation = explain_asset_prediction(
            asset_name=asset,
            latest_feature_vector=feats.dropna().iloc[[-1]],
            historical_X=labeled[feature_cols].tail(252),
            historical_y=labeled[targets.name].tail(252),
            top_k=3
        )
        
        print(f"\n  Asset: {asset} (Expected Return: {explanation['predicted_annual_return_pct']:+.2f}% p.a.)")
        print(f"  Institutional Narrative:")
        print(f"    \"{explanation['explanation']}\"")
        print("  Top Contributing Feature Drivers:")
        for d in explanation["top_drivers"]:
            print(f"    - {d['display_name']:<42} | Share: {d['weight_share_pct']:>5.1f}% | Signal: {d['signal_direction'].upper()}")

    # =========================================================================
    # MODULE 4: Monte Carlo Goal Simulator (GBM + SIP)
    # =========================================================================
    print("\n" + "=" * 80)
    print("  MODULE 4: MONTE CARLO GOAL SIMULATOR (GBM + MONTHLY SIP)")
    print("=" * 80)
    
    # Portfolio expected return & volatility
    port_ann_return = float(np.sum([hrp_weights[a] * predictions[a]["annualized_expected_return"] for a in market_df.columns]))
    port_ann_vol = float(np.std(port_returns) * np.sqrt(252))
    
    goal_res = simulate_goal_paths(
        initial_capital=200000.0,       # Rs. 2,00,000 lump sum
        target_amount=1500000.0,        # Rs. 15,00,000 goal
        horizon_years=5,                # 5 years runway
        expected_annual_return=port_ann_return,
        annual_volatility=port_ann_vol,
        monthly_sip=12000.0,            # Rs. 12,000 / month SIP
        n_simulations=1000,
        random_seed=42
    )
    
    print(f"  Target Goal Capital:      INR {goal_res['target_amount']:>12,.2f}")
    print(f"  Total Capital Invested:   INR {goal_res['total_invested_capital']:>12,.2f} (Rs. 2L initial + Rs. 12k/mo for 5y)")
    print(f"  Portfolio Return Drift:   {port_ann_return * 100:>6.2f}% p.a.")
    print(f"  Portfolio Volatility:     {port_ann_vol * 100:>6.2f}% p.a.")
    print("-" * 80)
    print(f"  PROBABILITY OF SUCCESS:   {goal_res['probability_of_success_pct']:>6.2f}%")
    print(f"  10th Percentile (Bear):   INR {goal_res['percentile_10_wealth']:>12,.2f}")
    print(f"  50th Percentile (Median): INR {goal_res['percentile_50_wealth']:>12,.2f}")
    print(f"  90th Percentile (Bull):   INR {goal_res['percentile_90_wealth']:>12,.2f}")

    # =========================================================================
    # MODULE 5: Historical Crisis Stress-Testing Simulator
    # =========================================================================
    print("\n" + "=" * 80)
    print("  MODULE 5: HISTORICAL CRISIS SCENARIO STRESS-TESTING")
    print("=" * 80)
    print("  Simulating portfolio resilience during historic market crashes:")
    
    crisis_results = simulate_all_crises(portfolio_weights=hrp_weights, local_market_df=market_df)
    
    for c_key, c_res in crisis_results.items():
        print(f"\n  Crisis: {c_res['crisis_name']} ({c_res['start_date']} to {c_res['end_date']})")
        print(f"    Source Data:            {c_res['data_source_used']}")
        print(f"    HRP Portfolio Max DD:   {c_res['portfolio_max_drawdown_pct']:>6.2f}%")
        print(f"    100% Nifty Benchmark:   {c_res['benchmark_max_drawdown_pct']:>6.2f}%")
        print(f"    Capital Preserved:      {c_res['capital_preserved_drawdown_reduction_pct']:>6.2f}% drawdown reduction")
        print(f"    Portfolio Total Return: {c_res['portfolio_cumulative_return_pct']:>+6.2f}%")
        print(f"    Benchmark Total Return: {c_res['benchmark_cumulative_return_pct']:>+6.2f}%")
        print(f"    Outcome:                {c_res['stress_test_result']}")
    
    print("\n" + "=" * 80)
    print("                    ALL 5 MODULE DEMONSTRATIONS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_full_demo()

