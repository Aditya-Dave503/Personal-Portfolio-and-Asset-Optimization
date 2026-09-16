# ML-Driven Personal Portfolio & Asset Allocation Engine

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()
[![Academic Track](https://img.shields.io/badge/Course-Software%20Engineering%20Capstone-orange.svg)]()

An end-to-end automated personal portfolio management and multi-asset optimization platform. The system leverages machine learning and modern portfolio theory to evaluate an investor's comprehensive risk profile, investment horizon, and target goals—synthesizing personalized, risk-adjusted asset allocations across Indian Equities (Nifty 50), Mutual Funds, Sovereign Gold, Silver, and Government Treasury Bonds (G-Secs).

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [System Architecture](#-system-architecture)
- [Key Features & Modules](#-key-features--modules)
- [Asset Universe](#-asset-universe)
- [Machine Learning & Quantitative Optimization Pipeline](#-machine-learning--quantitative-optimization-pipeline)
- [Software Engineering Specifications](#-software-engineering-specifications)
  - [Functional Requirements (FRs)](#functional-requirements-frs)
  - [Non-Functional Requirements (NFRs)](#non-functional-requirements-nfrs)
  - [Database Schema (High-Level ERD)](#database-schema-high-level-erd)
- [Tech Stack](#-tech-stack)
- [Directory Structure](#-directory-structure)
- [Installation & Setup](#-installation--setup)
- [API Documentation](#-api-documentation)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [Academic Evaluation Metrics](#-academic-evaluation-metrics)
- [Future Roadmap](#-future-roadmap)
- [Authors & Acknowledgments](#-authors--acknowledgments)

---

## 📖 Project Overview

Retail investors in emerging markets often struggle with multi-asset diversification, balancing risk-adjusted returns against specific drawdown tolerances, liquidity requirements, and milestone goals. 

**OptiWealth** solves this through a guided, rule-and-ML-driven software lifecycle:
1. **Subscription & Paywall Tier:** Secures premium services via recurring payment integration.
2. **Dynamic Risk Profiling:** Quantifies loss aversion by evaluating maximum acceptable capital drawdown, investor age, existing net worth, liquid reserves, and specific temporal milestones.
3. **Automated Asset Optimization:** Employs predictive modeling and algorithmic asset weighting to generate optimal allocations tailored to domestic macroeconomic cycles and individual risk budgets.
4. **Tracking & Rebalancing Engine:** Continuously tracks existing assets and triggers alert-based portfolio rebalancing.

---

## 🏗 System Architecture

The project follows a decoupled, modular service-oriented architecture (SOA):

```
+-------------------------------------------------------------+
|                      Client Layer                           |
|       (React / Next.js / Flutter Interactive Dashboards)     |
+------------------------------+------------------------------+
                               | HTTPS / REST / WSS
                               v
+-------------------------------------------------------------+
|                    API Gateway & Auth                       |
|       (JWT Authentication, Rate Limiting, RBAC Middleware)  |
+------------------------------+------------------------------+
                               |
       +-----------------------+-----------------------+
       |                                               |
       v                                               v
+-------------------------------+       +-------------------------------+
|     User & Billing Service    |       |   Risk Profiling & Goals      |
|  - Payment Gateway Webhooks   |       |  - Onboarding Questionnaire   |
|  - Subscription Tier Control  |       |  - Max Drawdown Quantification|
|  - KYC / Account Lifecycle    |       |  - Horizon & Goal Mapping     |
+---------------+---------------+       +---------------+---------------+
                |                                       |
                +-------------------+-------------------+
                                    |
                                    v
+---------------------------------------------------------------+
|            Optimization & Machine Learning Engine             |
|  - Feature Extraction (Returns, Volatility, Covariance Matrix) |
|  - Clustering & Risk Scoring Model                            |
|  - Multi-Asset Allocation (Mean-Variance / Black-Litterman)   |
|  - Indian Market Data Ingestion Pipeline (NSE, RBI G-Secs)    |
+-------------------------------+-------------------------------+
                                |
                                v
+---------------------------------------------------------------+
|                      Persistence Layer                        |
|  - PostgreSQL: Relational user data, subscriptions, holdings  |
|  - Redis: In-memory market price cache & session tokens       |
|  - TimescaleDB / Time-series: Historical tick/daily returns   |
+---------------------------------------------------------------+
```

---

## ✨ Key Features & Modules

### 1. Subscription & Monetization Module
- Tiered subscription model (Monthly / Annual / Institutional).
- Integrated payment gateway (Razorpay / Stripe) with webhook handling, automated invoicing, and entitlement enforcement.

### 2. Intelligent Onboarding & Psychometric Risk Assessment
- Multi-dimensional questionnaire capturing demographic profile (age, dependents, income bracket).
- **Hard Drawdown Calibration:** Evaluates risk appetite in concrete monetary terms (maximum acceptable percentage and absolute loss tolerance $\Delta V_{\max}$).
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
   - User responses (age, investment runway, savings-to-income ratio, maximum loss tolerance) are mapped into an objective risk-budget score $R_u \in [0, 1]$ using a trained Gradient Boosting Classifier / Random Forest Regressor calibrated against historical investor behavior.
2. **Covariance & Volatility Modeling:**
   - Real-time fetching of historical daily price series ($P_{t}$) over a rolling 5-to-10 year window.
   - Calculation of annualized expected returns $\mu$, standard deviation $\sigma$, and cross-asset covariance matrix $\Sigma$.
3. **Constrained Portfolio Optimization:**
   - Solves for asset weights $w = [w_{\text{stocks}}, w_{\text{MF}}, w_{\text{gold}}, w_{\text{silver}}, w_{\text{bonds}}]$ subject to:
     $$\max_{w} \; \frac{w^T \mu - R_f}{\sqrt{w^T \Sigma w}} \quad \text{or} \quad \min_{w} \; w^T \Sigma w$$
     Subject to:
     $$\sum_{i=1}^n w_i = 1, \quad w_i \ge 0 \quad (\text{No Short-Selling})$$
     $$\text{Value at Risk (VaR)}_{99\%} \le \text{Investor's Max Loss Tolerance Threshold}$$
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
- **NFR-03 (Availability):** Core API target availability of 99.9%.
- **NFR-04 (Maintainability):** Adherence to clean architecture principles, PEP 8 / Clean Code standards, and $> 80\%$ unit test coverage.

### Database Schema (High-Level ERD)

```
[Users]
  ├── id (UUID, PK)
  ├── email (VARCHAR, Unique)
  ├── password_hash (VARCHAR)
  └── created_at (TIMESTAMP)
        |
        +-- 1:1 --> [Subscriptions]
        |             ├── id (UUID, PK)
        |             ├── plan_type (ENUM)
        |             ├── status (ENUM: ACTIVE, EXPIRED)
        |             └── valid_until (TIMESTAMP)
        |
        +-- 1:1 --> [RiskProfiles]
        |             ├── id (UUID, PK)
        |             ├── user_id (FK)
        |             ├── calculated_risk_score (FLOAT)
        |             ├── max_loss_amount (NUMERIC)
        |             └── investment_horizon_years (INT)
        |
        +-- 1:N --> [CurrentHoldings]
        |             ├── id (UUID, PK)
        |             ├── asset_type (ENUM: STOCK, MF, GOLD, SILVER, BOND)
        |             ├── units (NUMERIC)
        |             └── invested_value (NUMERIC)
        |
        +-- 1:N --> [Allocations]
                      ├── id (UUID, PK)
                      ├── allocation_matrix (JSONB)
                      ├── expected_return (FLOAT)
                      └── generated_at (TIMESTAMP)
```

---

## 🛠 Tech Stack

| Layer | Technology | Justification |
| :--- | :--- | :--- |
| **Frontend** | React / Next.js, Tailwind CSS, Recharts | Interactive dynamic questionnaire, responsive charts, and component reusability. |
| **Backend API** | FastAPI / Python | High asynchronous throughput, native Python ML framework compatibility, automatic Swagger docs. |
| **Machine Learning & Stats** | Scikit-learn, NumPy, Pandas, SciPy Optimize, PyPortfolioOpt | Numerical modeling, constraint optimization, and statistical distribution scoring. |
| **Market Data Ingestion** | `yfinance`, NSEpy, RBI Open Data | Ingestion of Nifty 50, Precious Metals spot prices, and Sovereign Yield Curves. |
| **Database** | PostgreSQL + SQLAlchemy ORM | Relational consistency for payments, user profiles, and portfolio state storage. |
| **Caching & Queue** | Redis + Celery | Fast retrieval of latest asset ticks and asynchronous background model recalculations. |
| **Payments** | Razorpay / Stripe API | Secure recurring subscription billing and automated webhook verification. |
| **Deployment / DevOps** | Docker, Docker Compose, GitHub Actions | Containerized micro-services and automated CI/CD pipeline. |

---

## 📂 Directory Structure

```
optiwealth-backend/
├── api/
│   ├── v1/
│   │   ├── endpoints/
│   │   │   ├── auth.py          # Authentication & user sessions
│   │   │   ├── billing.py       # Payment webhooks & subscriptions
│   │   │   ├── profile.py       # Risk questionnaire & goals
│   │   │   └── portfolio.py     # Allocation & optimization trigger
│   │   └── router.py
├── core/
│   ├── config.py                # Environment variables & constants
│   ├── security.py              # Password hashing & JWT verification
│   └── database.py              # PostgreSQL session connection
├── models/                      # SQLAlchemy DB entities
│   ├── user.py
│   ├── subscription.py
│   └── portfolio.py
├── schemas/                     # Pydantic data validation schemas
│   ├── profile_schema.py
│   └── portfolio_schema.py
├── ml_engine/                   # Quantitative & ML modules
│   ├── data_fetcher.py          # Data ingestion from NSE / RBI sources
│   ├── risk_scorer.py           # ML model predicting risk category
│   ├── optimizer.py             # Mean-variance / Sharpe optimizer
│   └── backtester.py            # Historical validation & VaR calculator
├── tests/                       # Unit & integration test suites
│   ├── test_optimizer.py
│   ├── test_risk_scorer.py
│   └── test_api.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- PostgreSQL (v14+)
- Redis (v6+)
- Node.js (v18+) for client application

### Step-by-Step Backend Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/optiwealth.git
   cd optiwealth
   ```

2. **Set up virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   Create a `.env` file in the root directory:
   ```ini
   APP_ENV=development
   SECRET_KEY=your_super_secret_jwt_key
   DATABASE_URL=postgresql://user:password@localhost:5432/optiwealth_db
   REDIS_URL=redis://localhost:6379/0
   RAZORPAY_KEY_ID=rzp_test_xxxxxxx
   RAZORPAY_KEY_SECRET=xxxxxxx
   ```

5. **Execute Database Migrations:**
   ```bash
   alembic upgrade head
   ```

6. **Run the Development Server:**
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   Access API documentation interactively at `http://localhost:8000/docs`.

---

## 🔌 API Documentation

| Method | Endpoint | Description | Auth Required |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/auth/register` | Register a new user account | No |
| `POST` | `/api/v1/auth/login` | Authenticate and obtain JWT token | No |
| `POST` | `/api/v1/billing/create-checkout` | Initialize subscription checkout session | Yes |
| `POST` | `/api/v1/billing/webhook` | Process payment confirmations from gateway | Signature |
| `POST` | `/api/v1/profile/questionnaire` | Submit risk profile and drawdown answers | Yes (Paid) |
| `POST` | `/api/v1/portfolio/holdings` | Ingest existing assets and liquid savings | Yes (Paid) |
| `POST` | `/api/v1/portfolio/optimize` | Run ML engine and get asset allocation | Yes (Paid) |
| `GET` | `/api/v1/portfolio/report` | Export detailed PDF allocation report | Yes (Paid) |

---

## 🧪 Testing & Quality Assurance

The codebase includes automated unit, integration, and stress tests:

```bash
# Run complete test suite with coverage report
pytest --cov=ml_engine --cov=api tests/

# Execute type checking and linting
flake8 api/ ml_engine/
mypy api/ ml_engine/
```

- **Optimizer Invariance Tests:** Verifies all resulting asset weights $\sum w_i = 1.0$ and no allocation is negative.
- **Drawdown Cap Verification:** Asserts simulated worst-case VaR does not violate the maximum user-defined capital loss.
- **Payment Verification Tests:** Mocked webhooks to confirm state transitions from unverified to paid.

---

## 🎓 Academic Evaluation Metrics

For faculty and viva evaluation panels, this project demonstrates competencies across several core Software Engineering areas:
- **Requirements Engineering:** Clear mapping from problem statement to formal FR/NFR matrices and IEEE 830 SRS guidelines.
- **Object-Oriented & Modular Design:** Separation of concerns between the transaction tier, domain logic, and ML pipeline.
- **Data Engineering:** Automated time-series ETL pipelines fetching volatility and yield curves for domestic Indian benchmarks.
- **Mathematical Modeling:** Application of Modern Portfolio Theory (MPT) constrained by real-world risk metrics.
- **DevOps & Verification:** Containerized execution with Docker and reproducible test suites.

---

## 🗺 Future Roadmap
- [ ] Direct broker integration (Zerodha Kite, Groww API) for 1-click execution.
- [ ] Integration of corporate bonds (AAA / AA+ rated debt) into the fixed-income basket.
- [ ] Deep Reinforcement Learning (DRL) agent for continuous tactical asset reallocation.
- [ ] Multi-lingual onboarding questionnaire for regional accessibility.

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.
