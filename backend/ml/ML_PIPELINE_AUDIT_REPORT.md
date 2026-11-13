# ML Pipeline Technical Audit & Defect Analysis

**Target Directory:** `backend/ml/`  
**Purpose:** Technical assessment of the current machine learning models, validation pipeline, and portfolio integration.

---

## Executive Summary

The machine learning implementation in `backend/ml/` suffers from fundamental flaws across **time-series data handling**, **model selection**, **evaluation metrics**, and **software architecture**. 

Crucially, **the portfolio optimizer does not even use the trained ML models**, and the training code suffers from **look-ahead data leakage**, ignores macroeconomic and cross-asset correlations, and uses metrics (MAE) that yield near-zero predictive value for portfolio construction.

---

## 1. Look-Ahead Data Leakage in Validation Split

### The Flaw
In `train_model.py` (lines 74–102), the target is defined as the forward return over the next 21 trading days, and the dataset is split sequentially at the 80% mark:

```python
# Line 74-76: Target looks ahead 21 days
df["target"] = (
    df["price"].shift(-21) / df["price"] - 1
)

# Line 99-102: Simple 80/20 chronological split
split = int(len(df) * 0.8)
train = df.iloc[:split]
test = df.iloc[split:]
```

### Why It Is Catastrophic for Time Series
For the last 20 rows of the `train` set (`split - 20` to `split - 1`), the target label `price[t + 21] / price[t] - 1` incorporates prices that occur **inside the test partition**.

```text
TRAIN DATA: [---------------------------------] (ends at index 'split - 1')
TARGET WINDOW FOR LAST TRAIN ROWS:     [==== 21 DAYS FORWARD ====]
                                              ▲
TEST DATA BEGINS HERE:                        [-------------------------]
                                              ▲ (20 days of direct overlap!)
```

* **Impact:** The model trains on ground-truth price movements that belong to the out-of-sample evaluation period. This creates artificial out-of-sample performance and look-ahead bias.
* **The Correction:** A temporal **purging gap (embargo)** of at least 21 trading days must be placed between the training cut-off and the testing start date.

---

## 2. Architectural Disconnect: The Optimizer Ignores the ML Model

### The Flaw
The codebase contains a trained Random Forest model (`train_model.py`) and an inference script (`predict.py`), but the portfolio optimization module (`optimizer.py`) **never calls either of them**:

```python
# In optimizer.py (Lines 52-70):
# Calculates historical return sample covariance
returns = prices_df.pct_change().dropna()
covariance = returns.cov().to_numpy()

def portfolio_variance(weights):
    return float(weights @ covariance @ weights)

# Standard quadratic programming to minimize variance
result = minimize(portfolio_variance, initial_weights, ...)
```

* **Impact:** The trained `.joblib` files in `backend/ml/saved_models/` are completely orphaned artifacts. The output from `train_model.py` has zero influence on the recommended asset allocation.
* **The Correction:** Machine learning outputs (e.g., expected return rankings or covariance forecasts) must feed directly into the optimization objective (e.g., via a Black-Litterman matrix or expected Sharpe maximization).

---

## 3. Methodological Flaw: Random Forest on Non-Stationary Time Series

### The Flaw
`RandomForestRegressor` is an ensemble of decision trees trained with bootstrap aggregation (bagging) under the assumption that sample observations are **Independent and Identically Distributed (I.I.D.)**.

Financial time series violate this assumption in multiple ways:
1. **Autocorrelation & Regime Shifts:** Market volatility clusters (periods of high volatility followed by high volatility), and returns exhibit cyclical momentum and mean-reversion. Decision trees treat each daily observation as a disconnected, independent lottery draw.
2. **Inability to Extrapolate:** A decision tree partitions feature space into orthogonal bounding boxes and predicts the mean value of the terminal leaf. It **cannot extrapolate** outside the historical range seen in training. In extreme bull runs or market crashes, tree-based models predict bounded horizontal step-functions.
3. **No Temporal Sequence:** Tree splits do not preserve the directional arrow of time (`t-1 -> t -> t+1`), discarding sequential trends.

---

## 4. Inappropriate Metric: MAE on Zero-Centered Noisy Data

### The Flaw
In `train_model.py` (lines 127–137), the model is evaluated using **Mean Absolute Error (MAE)** against a zero-return baseline:

```python
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

# Compare predictions against a zero-return baseline
baseline_mae = mean_absolute_error(y_test, np.zeros(len(y_test)))
```

### Why MAE is the Wrong Objective
1. **Minimizing MAE drives predictions to the median (~0% return):** Asset returns have very low signal-to-noise ratios. A model minimizing L1/MAE discovers that predicting a number hovering near zero minimizes error across noisy fluctuations.
2. **Portfolio optimization requires relative ranking, not absolute return levels:** An asset allocator does not need to know whether Reliance will rise by 2.15% vs 2.30%. The allocator needs to know: **Is Asset A expected to outperform Asset B?**
3. **Missing Metrics:** The pipeline fails to evaluate:
   * **Spearman Rank Correlation (Information Coefficient):** Does the predicted rank match actual future rank?
   * **Directional Accuracy:** Did the model get the sign (+/-) right?
   * **Portfolio Sharpe / Sortino:** Does an ML-weighted basket beat an equal-weighted benchmark?

---

## 5. Isolated Single-Asset Modeling (No Macro or Cross-Asset Signals)

### The Flaw
`train_model.py` loops over assets and trains a completely isolated model for each one:

```python
def main():
    for ticker in ASSETS:
        train_one_asset(ticker)
```

Each asset's feature matrix consists strictly of its own local lags:
* `return_1d`, `return_5d`, `return_21d`, `return_63d`, `volatility_21d`

### What Is Missing
* **Cross-Asset Interactions:** In real markets, equities move in response to bond yields, currency movements, and sector rotations.
* **Macroeconomic Yield Drivers:** The project already possesses 10-year Indian Government Bond yield changes (`india_10y_gsec_yield_changes.csv`), but the training pipeline ignores them completely.
* **Market-Wide Beta:** Individual asset movements are largely driven by overall market direction (Nifty 50 beta). A model with no market benchmark feature cannot distinguish between idiosyncratic asset strength and a rising market tide.

---

## 6. Disconnect from Project Goals and Pre-Engineered Datasets

1. **Bypassing Phase 2 Feature Engineering:**
   The project contains a dedicated feature engineering package (`phase2_package/outputs/phase2/training_dataset_21d.csv.gz`) containing **123,252 rows across 58 assets** with 18 leakage-aware features (trailing returns, drawdown, macro yield features). `train_model.py` does not touch this file.
2. **Ignoring User Investment Profiles:**
   The core user requirements (investment horizon in years, target amount in ₹, and maximum risk/loss tolerance) are captured by the database (`backend/models/goal.py` and `backend/models/risk_profile.py`), but the ML training script operates in a vacuum with a hardcoded 21-day horizon and zero risk-profile awareness.

---

## Summary Matrix

| Defect Category | Root Cause in Code | Consequence |
| :--- | :--- | :--- |
| **Data Leakage** | `shift(-21)` target + contiguous 80/20 train/test split | Out-of-sample test data contaminates the training set. |
| **Pipeline Integration** | `optimizer.py` does not import or call `saved_models/` | The trained ML models have no impact on the final allocations. |
| **Model Fit** | `RandomForestRegressor` on raw time-series prices | Inability to capture time sequences, autocorrelation, or extreme market tail risks. |
| **Metric Selection** | MAE against zero-return baseline | Model minimizes error by predicting near-zero returns, providing no ranking power. |
| **Feature Scope** | Univariate lags only; loop over individual tickers | Ignores macroeconomic indicators (G-Sec yields) and cross-asset correlations. |
| **Data Siloing** | `phase2` generated training tables are ignored | 123K engineered rows with macro features remain completely unused. |

---

## Recommended Remediation Steps

1. **Unify the Target Generation with a Purged Walk-Forward Split:**
   Implement an expanding-window walk-forward validation scheme with a 21-day purging embargo to eliminate look-ahead bias.
2. **Train a Cross-Asset Ranker:**
   Train a single gradient-boosted ranker (e.g., LightGBM) across all assets using the `training_dataset_21d.csv.gz` table, evaluating with **Spearman Rank Correlation (IC)**.
3. **Bridge the Model to the Optimizer:**
   Feed the model's return rankings and a shrinkage covariance matrix (Ledoit-Wolf) into `goal_optimizer.py`, allowing the optimizer to recommend portfolios tailored to the user's target wealth, horizon, and risk tolerance.

