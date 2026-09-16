# Personal-Portfolio-and-Asset-Optimization
Agile-built platform that profiles investor risk and recommends optimized stock, gold, bond, and silver allocations — backed by both rule-based logic and ML-driven suggestions.

---

## ✨ Key Features & Modules

### 1. Subscription & Monetization Module
- Tiered subscription model (Monthly / Annual / Institutional).
- Integrated payment gateway (Razorpay / Stripe) with webhook handling, automated invoicing, and entitlement enforcement.

### 2. Intelligent Onboarding & Psychometric Risk Assessment
- Multi-dimensional questionnaire capturing demographic profile (age, dependents, income bracket).
- **Hard Drawdown Calibration:** Evaluates risk appetite in concrete monetary terms (maximum acceptable percentage and absolute loss tolerance $\\Delta V_{\\max}$).
- Goal-based tagging (e.g., retirement, house down-payment, higher education) with target date and liquidity constraints.

### 3. Current Portfolio & Savings Ingestion
- Unified input mechanism for tracking existing cash savings, fixed deposits, gold holdings, equities, and active SIPs.
- Net-worth aggregation and overlap detection across mutual fund holdings.

### 4. Quantitative Multi-Asset Recommendation Engine
- Generates optimal portfolio weights across uncorrelated asset classes.
- Delivers granular breakdowns down to underlying indices, funds, and sovereign instruments.

### 5. Portfolio Rebalancing & Simulation Dashboard
- Real-time simulation of forward returns based on historical backtesting and Monte Carlo simulations.
- Drift alerts when asset weights diverge from target boundaries beyond predefined thresholds.

---

## 📊 Asset Universe

The optimization pipeline encompasses core instruments traded in the Indian financial ecosystem:

| Asset Class | Instrument / Underlying Proxy | Risk Category | Role in Portfolio |
| :--- | :--- | :--- | :--- |
| **Large-Cap Equities** | Nifty 50 Index / Index ETFs / Blue-chip Equities | High Risk / High Growth | Long-term capital appreciation |
| **Mutual Funds** | Diversified Flexi-cap, Mid-cap, & Hybrid Mutual Funds | Moderate to High Risk | Active alpha generation & systematic compounding |
| **Precious Metals** | Digital Gold, Sovereign Gold Bonds (SGB), 24K Physical Gold | Moderate Risk | Inflation hedge & safe-haven liquidity |
| **Precious Metals** | Silver ETFs / Silver Commodities | Moderate to High Risk | Industrial demand hedge & tactical growth |
| **Sovereign Debt** | Indian Government Treasury Bills (91/182/364-day), 10-Year G-Secs | Low Risk / Risk-Free Benchmark | Capital preservation & steady sovereign-backed yields |

---

## 🧠 Machine Learning & Quantitative Optimization Pipeline

The core allocation engine blends statistical finance with machine learning:

1. **Investor Risk Profiling (Supervised / Clustering):**
   - User responses (age, investment runway, savings-to-income ratio, maximum loss tolerance) are mapped into an objective risk-budget score $R_u \\in [0, 1]$ using a trained Gradient Boosting Classifier / Random Forest Regressor calibrated against historical investor behavior.
2. **Covariance & Volatility Modeling:**
   - Real-time fetching of historical daily price series ($P_{t}$) over a rolling 5-to-10 year window.
   - Calculation of annualized expected returns $\\mu$, standard deviation $\\sigma$, and cross-asset covariance matrix $\\Sigma$.
3. **Constrained Portfolio Optimization:**
   - Solves for asset weights $w = [w_{\\text{stocks}}, w_{\\text{MF}}, w_{\\text{gold}}, w_{\\text{silver}}, w_{\\text{bonds}}]$ subject to:
     $$\\max_{w} \\; \\frac{w^T \\mu - R_f}{\\sqrt{w^T \\Sigma w}} \\quad \\text{or} \\quad \\min_{w} \\; w^T \\Sigma w$$
     Subject to:
     $$\\sum_{i=1}^n w_i = 1, \\quad w_i \\ge 0 \\quad (\\text{No Short-Selling})$$
     $$\\text{Value at Risk (VaR)}_{99\\%} \\le \\text{Investor's Max Loss Tolerance Threshold}$$
4. **Regime-Aware Asset Tilting:**
   - Adjusts macro weights dynamically depending on interest-rate cycles (RBI repo-rate trajectories) and equity market valuation percentiles.

---

## ⚙️ Software Engineering Specifications

### Functional Requirements (FRs)
- **FR-01 (Authentication):** Users must be able to register, log in via OAuth2/JWT, and undergo KYC verification.
- **FR-02 (Paywall Gate):** Platform features remain locked until an active subscription status is validated via payment webhooks.
- **FR-03 (Questionnaire Engine):** Multi-step dynamic form collecting financial horizon, liquidity needs, age, and loss tolerance.
- **FR-04 (Asset Aggregation):** Users can add, update, and delete current holdings (stocks, savings, gold, fixed deposits).
- **FR-05 (Prediction & Allocation):** The ML engine calculates and displays asset allocation with expected annual return, Sharpe ratio, and Value at Risk (VaR).
- **FR-06 (Export & Reporting):** System generates downloadable PDF investment advisory reports summarizing the allocation rationale.

### Non-Functional Requirements (NFRs)
- **NFR-01 (Latency):** Portfolio optimization queries must resolve within $< 1.8$ seconds for single-run estimations.
- **NFR-02 (Security & Compliance):** All sensitive financial records encrypted at rest using AES-256; communication enforced via TLS 1.3.
- **NFR-03 (Availability):** Core API target availability of 99.5%.
- **NFR-04 (Maintainability):** Adherence to clean architecture principles, PEP 8 / Clean Code standards, and $> 80\\%$ unit test coverage.

### Database Schema (High-Level ERD)
