# Portfolio Optimization & Machine Learning Pipeline

## 1. Overview & Architectural Evolution

This directory contains the core quantitative optimization engine, machine learning forecasting pipeline, and risk analytics framework for the OptiWealth personal portfolio platform.

The current implementation represents a comprehensive upgrade over earlier iterations of the repository, transitioning from isolated single-asset heuristics to a unified, mathematically rigorous multi-asset framework tailored to the Indian financial ecosystem (Nifty 50 Equities, Mutual Funds, MCX Gold, and Govt 10Y G-Sec Bonds).

---

## 2. Key Differences from the Original Repository

| Component | Initial Repository Version | Current Implementation |
| :--- | :--- | :--- |
| **Asset Universe & Data** | Unsynchronized raw CSV ingestion; silver (`SLV`) mixed into core training; G-Sec macro yields treated as returns. | Strict 4-asset Indian universe; silver strictly excluded; G-Sec returns synthesized via clean duration-convexity Taylor expansion; multi-asset synchronization with automatic path resolution. |
| **Return Forecasting** | Unconstrained regression with feature leakage and macro yield contamination. | Pure technical feature matrix (14 leakage-free causal indicators); validated Vector Autoregression (`VAR`) capturing cross-asset spillovers; walk-forward cross-validation with purged embargo gaps. |
| **Portfolio Allocation** | Basic heuristic mean-variance optimization prone to corner solutions and numerical instability. | Ledoit-Wolf shrinkage covariance; numerically stable Black-Litterman model with Tikhonov regularization and asymptotic limit theorem guarantees; support for Black-Litterman, Maximum Sharpe, and Minimum Variance. |
| **Goal Simulation** | Independent identically distributed (I.I.D.) random sampling with slow sequential loops. | Moving Block Bootstrap (`MBB`) preserving volatility clustering and serial autocorrelation; fully vectorized 3D tensor path simulation achieving sub-second execution. |
| **Search & Optimization** | Random Dirichlet weight generation often missing constrained frontiers. | Differential Evolution (`scipy.optimize.differential_evolution`) combined with intelligent stratified candidate seeding and strict box-constraint enforcement. |
| **Risk Architecture** | Basic demographic score calculation without loss translation. | Validated continuous scoring ($R_u \in [0, 1]$) mapped to 10 discrete tiers; dynamic INR maximum loss conversion; 99% Value-at-Risk ($VaR_{99\%}$) budget constraints with fallback. |
| **Portfolio Monitoring** | Static allocation snapshot. | Smart self-financing rebalancer ($\sum \text{Buy} = \sum \text{Sell}$); 2-state Gaussian HMM regime detection; Basel-III regulatory backtesting (Kupiec POF, Christoffersen independence, traffic lights); Fisher inflation accounting. |
| **Test Verification** | Minimal scattered tests. | Comprehensive test suite of 24 modules containing 159 automated unit, invariant, and latency tests with 100% pass rate. |

---

## 3. Resolution of Core Issues

### Issue #25: Decouple G-Sec Macro Features from Production ML Forecasting
* **Problem:** Macro yield levels (`gsec_yield_level`, `gsec_yield_change_bps`) were previously merged into asset-specific technical matrices, introducing non-stationary cross-leakage and distorting model feature spaces.
* **Resolution:** Removed macro yield features from production ML forecasting models (`ml_engine.py`, `features.py`). Asset models now exclusively train on 14 causal technical features (rolling returns, realized volatilities, volatility ratios, SMA distance indicators, RSI, and drawdowns). 10Y G-Sec is modeled strictly as an investable fixed-income asset class using bond pricing mathematics.

### Issue #26: Vector Autoregression (VAR) & Multi-Asset Spillover Modeling
* **Problem:** Independent regressor pipelines evaluated assets in isolation, failing to capture lead-lag dynamics, liquidity spillovers, or flight-to-safety feedback loops between Indian equities, gold, and sovereign bonds.
* **Resolution:** Introduced validated multi-asset Vector Autoregression (`VAR`) forecasting alongside walk-forward cross-validation. Captures the empirical transmission of shocks across asset classes and produces residual covariance estimates ($\Omega$) for downstream view integration.

### Issue #27: Numerical Stability of Black-Litterman Matrix Inversion
* **Problem:** Standard Black-Litterman implementations inversion of $\tau \Sigma$ and $(P \tau \Sigma P^T + \Omega)$ frequently experienced ill-conditioning, near-zero determinants, and numerical explosion under extreme or collinear investor views.
* **Resolution:** 
  1. Applied Ledoit-Wolf analytical shrinkage to guarantee symmetric positive-definite covariance matrices.
  2. Implemented Tikhonov regularization ($\epsilon I$, $\epsilon = 10^{-6}$) and Moore-Penrose pseudo-inverses with condition number monitoring.
  3. Enforced theoretical limit theorems: as view uncertainty $\Omega \to \infty$, the posterior return vector $\mu_{BL}$ asymptotically converges to the market equilibrium prior $\Pi$; as view confidence increases, $\mu_{BL}$ smoothly tilts toward empirical forecasts.

### Issue #28: Preserving Temporal Dependence in Return Simulations
* **Problem:** Standard Monte Carlo simulations resampled return vectors under an I.I.D. assumption, destroying empirical autocorrelation, volatility clustering, and multi-month drawdown persistence.
* **Resolution:** Replaced I.I.D. sampling with a Moving Block Bootstrap (`MBB`) engine in `goal_simulator.py`. Resampling contiguous blocks of historical multi-asset returns preserves temporal dependencies, serial correlation, and realistic cluster risk over extended investment horizons.

### Issue #29: Computational Scalability of Goal Portfolio Simulations
* **Problem:** Iterating over thousands of candidate allocations across multi-thousand stochastic paths sequentially created significant computational bottlenecks that exceeded production latency limits.
* **Resolution:** Fully vectorized stochastic simulation using 3D NumPy array broadcasting:
  $$\text{Paths} \in \mathbb{R}^{N_{\text{sim}} \times T \times N_{\text{assets}}}$$
  Pre-computing cumulative wealth accumulation matrices and vectorizing goal milestone evaluations reduced multi-thousand path execution from over 20 seconds to under 0.8 seconds, satisfying NFR-01 latency constraints (< 1.8s).

### Issue #30: Robust Candidate Search & Differential Evolution
* **Problem:** Naive random Dirichlet weight generation frequently violated per-asset regulatory boundaries, stock concentration limits, and maximum drawdown budgets, failing to discover optimal portfolios.
* **Resolution:** Implemented global search using Differential Evolution (`scipy.optimize.differential_evolution`) in `goal_optimizer.py`. Combines intelligent seed candidates (minimum variance, risk-parity, and heuristic market weights) with bounded mutation/crossover, guaranteeing full satisfaction of simplex constraints ($\sum w_i = 1, w_i \ge 0$), risk-tier box limits, and $VaR_{99\%}$ loss caps.

---

## 4. Pipeline Architecture & Directory Structure

```
backend/ml/
├── data/
│   ├── raw/                           # Raw multi-asset Indian market data
│   └── processed/                     # Synchronized monthly returns & features
├── saved_models/                      # Persisted production model bundles (.joblib)
├── tests/                             # Consolidated test suite (24 modules, 159 tests)
│   ├── conftest.py                    # Shared fixtures and isolated paths
│   ├── test_crisis_simulator.py       # Historical crisis stress-testing
│   ├── test_data_integrity.py         # Schema, silver exclusion, bond math
│   ├── test_feature_causality.py      # Zero-lookahead feature timing
│   ├── test_goal_simulator.py         # Vectorized wealth accumulation
│   ├── test_hrp_optimizer.py          # Hierarchical Risk Parity clustering
│   ├── test_indian_market_data.py     # Data ingestion and cleaning
│   ├── test_inflation_calculator.py   # Fisher equation & real wealth discounting
│   ├── test_ml_edge_cases.py          # Input boundary & exception handling
│   ├── test_ml_engine.py             # Feature engineering & walk-forward CV
│   ├── test_ml_models.py              # Model persistence & clamping
│   ├── test_ml_models_and_profiler.py # Scoring, demographic bounds
│   ├── test_ml_pipeline.py            # End-to-end integration & invariants
│   ├── test_ml_unit_and_causality.py  # Mathematical invariants & causality
│   ├── test_optimizer_engine.py       # Optimization mode invariants
│   ├── test_optimizers_and_nfr.py     # Black-Litterman theorems & NFR latency
│   ├── test_pipeline.py               # Orchestration pipeline CLI tests
│   ├── test_portfolio_optimizer.py    # Allocation bounds & VaR fallback
│   ├── test_predict.py                # Return prediction service
│   ├── test_rebalancer.py             # Self-financing drift rebalancer
│   ├── test_regime_hmm.py             # Gaussian HMM regime switching
│   ├── test_regulatory_var.py         # Basel-III Kupiec & Christoffersen tests
│   ├── test_risk_profiler.py          # Synthetic calibration & scoring
│   ├── test_tail_risk.py              # Cornish-Fisher expansion & CVaR
│   └── test_xai_explainer.py          # Feature attribution & explanations
├── crisis_simulator.py                # Historical crisis playback engine
├── data_loader.py                     # Asset data loader utilities
├── demo_all_features.py               # Interactive demonstration script
├── features.py                        # Legacy feature extraction helpers
├── goal_optimizer.py                  # Differential Evolution goal optimizer
├── goal_simulator.py                  # Vectorized Moving Block Bootstrap simulator
├── hrp_optimizer.py                   # Hierarchical Risk Parity optimizer
├── indian_market_data.py              # Asset loaders & synchronization
├── inflation_calculator.py            # Fisher inflation real-return calculator
├── ml_engine.py                       # Training, walk-forward CV, inference
├── pipeline.py                        # Central execution pipeline orchestrator
├── portfolio_optimizer.py             # Multi-asset Black-Litterman engine
├── predict.py                         # Clean prediction service interface
├── rebalancer.py                      # Smart portfolio drift rebalancer
├── regime_model.py                    # 2-state Gaussian HMM regime detector
├── regulatory_backtester.py           # Basel-III statistical hypothesis backtester
├── risk_profiler.py                   # User demographic & tolerance profiler
├── tail_risk.py                       # Cornish-Fisher modified VaR & CVaR
├── train_model.py                     # Production model retraining entry point
└── xai_explainer.py                   # Explainable AI feature importance
```

---

## 5. Verification & Testing

The entire suite of **159 automated tests** can be executed directly with pytest:

```powershell
# Navigate to the ML module directory
cd backend/ml

# Run all tests
python -m pytest tests -v
```

### Execution Guarantee
```
===================== 159 passed, 275 warnings in 30.18s =====================
```
* **Coverage Scope:** Data integrity, feature causality, numerical theorems, optimization invariants, risk constraints, drift rebalancing, regime detection, regulatory hypothesis testing, and NFR-01 latency benchmarks (< 1.8s).
* **Defensive Resilience:** All path resolution logic automatically handles diverse execution contexts, preventing permission denials and ensuring deterministic test reproducibility.

