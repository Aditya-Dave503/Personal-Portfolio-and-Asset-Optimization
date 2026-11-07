"""
Investor Psychometric Risk Profiling Model (Section 1 of README).

Calibrates user demographic and financial risk tolerance parameters:
- age (years)
- investment_horizon_years (runway)
- savings_to_income_ratio (financial cushion)
- max_loss_tolerance_pct (maximum acceptable portfolio drawdown)

Maps these inputs into an objective risk-budget score R_u in [0, 1] (and 1-10 integer scale)
using a regularized Gradient Boosting Regressor calibrated against financial planning lifecycle rules.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold


MODEL_DIR = Path(__file__).resolve().parent / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "risk_profiling_model.joblib"


def generate_synthetic_calibration_profiles(n_samples: int = 5000, seed: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Synthesize diverse financial profiles based on behavioral economics & lifecycle theory:
    - Higher age -> lower risk capacity (traditional '100 - age' rule of thumb).
    - Longer runway -> higher capacity to absorb drawdowns.
    - Higher savings-to-income -> higher liquidity buffer.
    - Higher explicit loss tolerance -> lower psychological loss aversion.
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 75, size=n_samples)
    horizon = rng.integers(1, 35, size=n_samples)
    savings_ratio = rng.uniform(0.1, 2.5, size=n_samples)
    max_loss_pct = rng.uniform(0.05, 0.50, size=n_samples)

    # Lifecycle scoring formula
    # 1. Age factor: younger -> higher score (weight: 0.25)
    age_score = (75.0 - age) / (75.0 - 21.0)

    # 2. Horizon factor: longer horizon -> higher score (weight: 0.30)
    horizon_score = np.clip(horizon / 25.0, 0.0, 1.0)

    # 3. Financial cushion factor: higher savings ratio -> higher score (weight: 0.20)
    cushion_score = np.clip(savings_ratio / 1.5, 0.0, 1.0)

    # 4. Drawdown tolerance factor: higher loss acceptance -> higher score (weight: 0.25)
    tolerance_score = np.clip((max_loss_pct - 0.05) / 0.40, 0.0, 1.0)

    # Synthetic latent true risk score with minor behavioral noise
    raw_score = (
        0.25 * age_score +
        0.30 * horizon_score +
        0.20 * cushion_score +
        0.25 * tolerance_score +
        rng.normal(0, 0.02, size=n_samples)
    )

    true_score = np.clip(raw_score, 0.05, 0.95)

    df = pd.DataFrame({
        "age": age,
        "investment_horizon_years": horizon,
        "savings_to_income_ratio": savings_ratio,
        "max_loss_tolerance_pct": max_loss_pct,
    })
    y = pd.Series(true_score, name="risk_score_continuous")
    return df, y


def train_risk_profiling_model(retrain: bool = False) -> HistGradientBoostingRegressor:
    """Train and persist the risk scoring model."""
    if MODEL_PATH.exists() and not retrain:
        bundle = joblib.load(MODEL_PATH)
        return bundle["model"]

    X, y = generate_synthetic_calibration_profiles(n_samples=5000)

    model = HistGradientBoostingRegressor(
        max_iter=100,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42
    )
    model.fit(X, y)

    bundle = {
        "model": model,
        "features": list(X.columns),
        "r2_score": float(model.score(X, y)),
    }
    joblib.dump(bundle, MODEL_PATH)
    return model


def calculate_user_risk_profile(
    age: int = 30,
    investment_horizon_years: int = 10,
    savings_to_income_ratio: float = 0.5,
    max_loss_tolerance_pct: float = 0.20
) -> Dict[str, Any]:
    """
    Quantifies user inputs into continuous R_u in [0, 1] and discrete scale 1-10.
    """
    if not (18 <= age <= 100):
        raise ValueError("Age must be between 18 and 100.")
    if investment_horizon_years < 1:
        raise ValueError("Investment horizon must be at least 1 year.")
    if savings_to_income_ratio < 0:
        raise ValueError("Savings-to-income ratio cannot be negative.")
    if not (0.01 <= max_loss_tolerance_pct <= 0.90):
        raise ValueError("Loss tolerance must be between 1% and 90%.")

    model = train_risk_profiling_model(retrain=False)

    user_features = pd.DataFrame([{
        "age": age,
        "investment_horizon_years": investment_horizon_years,
        "savings_to_income_ratio": savings_to_income_ratio,
        "max_loss_tolerance_pct": max_loss_tolerance_pct,
    }])

    continuous_score = float(model.predict(user_features)[0])
    continuous_score = float(np.clip(continuous_score, 0.05, 0.95))

    # Scale to integer 1-10
    discrete_score = int(np.clip(np.round(continuous_score * 10), 1, 10))

    if discrete_score <= 3:
        category = "Conservative (Capital Preservation)"
    elif discrete_score <= 7:
        category = "Moderate (Balanced Growth & Income)"
    else:
        category = "Aggressive (Capital Appreciation)"

    return {
        "continuous_risk_score_Ru": round(continuous_score, 4),
        "discrete_risk_score": discrete_score,
        "category": category,
        "max_loss_tolerance_pct": max_loss_tolerance_pct,
        "investment_horizon_years": investment_horizon_years,
        "age": age,
    }


if __name__ == "__main__":
    train_risk_profiling_model(retrain=True)
    profiles = [
        {"age": 25, "investment_horizon_years": 20, "savings_to_income_ratio": 0.8, "max_loss_tolerance_pct": 0.35},
        {"age": 45, "investment_horizon_years": 10, "savings_to_income_ratio": 0.5, "max_loss_tolerance_pct": 0.20},
        {"age": 65, "investment_horizon_years": 3, "savings_to_income_ratio": 0.3, "max_loss_tolerance_pct": 0.08},
    ]
    for p in profiles:
        res = calculate_user_risk_profile(**p)
        print(f"Age {p['age']}, Horizon {p['investment_horizon_years']}y, Loss {p['max_loss_tolerance_pct']:.0%} -> Score {res['discrete_risk_score']}/10 ({res['continuous_risk_score_Ru']}) [{res['category']}]")
