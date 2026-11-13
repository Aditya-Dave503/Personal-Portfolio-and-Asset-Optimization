
"""
Investor Risk Profiling Engine.

Calculates:
- Objective financial risk capacity.
- Behavioural risk tolerance.
- Combined continuous risk score.
- Discrete risk score from 1 to 10.

Backward compatibility:
- Preserves original function names and parameters.
- Preserves all original return dictionary keys.
- get_calculated_risk_score() still returns an integer.

This is a deterministic, rule-based scoring engine.
It is not a trained or clinically validated ML model.
"""

from typing import Dict, Any, Optional, List
import math


QUESTIONNAIRE_VERSION = "1.0"
MODEL_VERSION = "2.0"

ANSWER_SCORES = {
    "A": 0.0,
    "B": 1.0 / 3.0,
    "C": 2.0 / 3.0,
    "D": 1.0,
}

BEHAVIORAL_WEIGHTS = {
    1: 0.30,
    2: 0.25,
    3: 0.25,
    4: 0.20,
}

OBJECTIVE_WEIGHTS = {
    "age": 0.20,
    "horizon": 0.30,
    "savings": 0.20,
    "loss_capacity": 0.30,
}


def _validate_finite_number(name: str, value: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a valid number.")

    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")

    return result


def calculate_objective_capacity(
    age: int,
    investment_horizon_years: int,
    annual_income: float,
    annual_savings: float,
    portfolio_value: float,
    max_loss_amount: float,
) -> Dict[str, Any]:

    age = int(age)
    horizon = int(investment_horizon_years)

    annual_income = _validate_finite_number(
        "annual_income", annual_income
    )
    annual_savings = _validate_finite_number(
        "annual_savings", annual_savings
    )
    portfolio_value = _validate_finite_number(
        "portfolio_value", portfolio_value
    )
    max_loss_amount = _validate_finite_number(
        "max_loss_amount", max_loss_amount
    )

    if not 18 <= age <= 100:
        raise ValueError("Age must be between 18 and 100.")

    if not 1 <= horizon <= 60:
        raise ValueError(
            "Investment horizon must be between 1 and 60 years."
        )

    if annual_income <= 0:
        raise ValueError("Annual income must be greater than zero.")

    if annual_savings < 0:
        raise ValueError("Annual savings cannot be negative.")

    if portfolio_value <= 0:
        raise ValueError("Portfolio value must be greater than zero.")

    if max_loss_amount <= 0 or max_loss_amount > portfolio_value:
        raise ValueError(
            "Maximum loss must be positive and cannot exceed portfolio value."
        )

    savings_ratio = annual_savings / annual_income
    max_loss_pct = max_loss_amount / portfolio_value

    age_score = (100.0 - age) / 82.0
    horizon_score = min(horizon / 30.0, 1.0)
    savings_score = min(savings_ratio / 0.50, 1.0)
    loss_capacity_score = min(max_loss_pct / 0.50, 1.0)

    objective_score = (
        OBJECTIVE_WEIGHTS["age"] * age_score
        + OBJECTIVE_WEIGHTS["horizon"] * horizon_score
        + OBJECTIVE_WEIGHTS["savings"] * savings_score
        + OBJECTIVE_WEIGHTS["loss_capacity"] * loss_capacity_score
    )

    return {
        "objective_risk_score": round(objective_score, 4),
        "savings_to_income_ratio": round(savings_ratio, 4),
        "max_loss_tolerance_pct": round(max_loss_pct, 4),
        "component_scores": {
            "age": round(age_score, 4),
            "horizon": round(horizon_score, 4),
            "savings": round(savings_score, 4),
            "loss_capacity": round(loss_capacity_score, 4),
        },
    }


def calculate_behavioral_tolerance(
    answers: List[Dict[str, Any]],
) -> Dict[str, Any]:

    if not isinstance(answers, list):
        raise ValueError("Answers must be a list.")

    answer_map = {}

    for answer in answers:
        question_id = int(answer["question_id"])
        selected_option = str(
            answer["selected_option"]
        ).upper()

        if question_id in answer_map:
            raise ValueError(
                f"Duplicate answer for question {question_id}."
            )

        if question_id not in range(1, 6):
            raise ValueError("Question ID must be between 1 and 5.")

        if selected_option not in ANSWER_SCORES:
            raise ValueError(
                "Selected option must be A, B, C, or D."
            )

        answer_map[question_id] = selected_option

    if set(answer_map) != {1, 2, 3, 4, 5}:
        raise ValueError(
            "Exactly five answers are required, one per question."
        )

    weighted_score = 0.0
    answer_details = []

    for question_id, weight in BEHAVIORAL_WEIGHTS.items():
        option = answer_map[question_id]
        score = ANSWER_SCORES[option]

        weighted_score += weight * score

        answer_details.append({
            "question_id": question_id,
            "selected_option": option,
            "behavioral_score": round(score, 4),
        })

    return {
        "behavioral_risk_score": round(weighted_score, 4),
        "liquidity_response": answer_map[5],
        "answer_details": answer_details,
    }


def _get_risk_category(score: int) -> str:

    if score <= 3:
        return "Conservative (Capital Preservation)"

    if score <= 7:
        return "Moderate (Balanced Growth & Income)"

    return "Aggressive (Capital Appreciation)"


def calculate_user_risk_profile(
    age: int = 30,
    investment_horizon_years: int = 10,
    savings_to_income_ratio: float = 0.5,
    max_loss_tolerance_pct: Optional[float] = None,
    max_loss_amount: Optional[float] = None,
    portfolio_value: float = 500000.0,
    target_amount: Optional[float] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Calculate complete investor risk profile.

    Original parameters and return keys are preserved.

    New optional keyword arguments:
    - annual_income
    - annual_savings
    - answers

    Existing callers can continue using the original interface.
    """

    # Preserve legacy target_amount behaviour.
    if target_amount is not None:
        target_amount = _validate_finite_number(
            "target_amount", target_amount
        )

        if target_amount > 0:
            portfolio_value = target_amount

    portfolio_value = _validate_finite_number(
        "portfolio_value", portfolio_value
    )

    if not 18 <= int(age) <= 100:
        age = 30

    if int(investment_horizon_years) < 1:
        investment_horizon_years = 1

    # Preserve legacy savings_to_income_ratio input.
    savings_to_income_ratio = _validate_finite_number(
        "savings_to_income_ratio",
        savings_to_income_ratio
    )

    if savings_to_income_ratio < 0:
        raise ValueError(
            "Savings-to-income ratio cannot be negative."
        )

    # Derive loss tolerance using the old input conventions.
    if max_loss_tolerance_pct is None:

        if max_loss_amount is not None:
            max_loss_amount = _validate_finite_number(
                "max_loss_amount", max_loss_amount
            )

            if max_loss_amount > 1.0:
                max_loss_tolerance_pct = (
                    max_loss_amount / portfolio_value
                )
            else:
                max_loss_tolerance_pct = max_loss_amount

        else:
            max_loss_tolerance_pct = 0.20

    max_loss_tolerance_pct = _validate_finite_number(
        "max_loss_tolerance_pct",
        max_loss_tolerance_pct
    )

    if not 0 < max_loss_tolerance_pct <= 1:
        raise ValueError(
            "Maximum loss tolerance percentage must be greater than 0 and at most 1."
        )

    # Derive loss amount if the caller supplied only a percentage.
    if max_loss_amount is None or max_loss_amount <= 1.0:
        effective_max_loss_amount = (
            max_loss_tolerance_pct * portfolio_value
        )
    else:
        effective_max_loss_amount = max_loss_amount

    if effective_max_loss_amount > portfolio_value:
        raise ValueError(
            "Maximum loss amount cannot exceed portfolio value."
        )

    # New optional financial inputs.
    annual_income = kwargs.get(
        "annual_income", 600000.0
    )
    annual_savings = kwargs.get(
        "annual_savings",
        float(savings_to_income_ratio) * float(annual_income)
    )

    # Calculate actual savings ratio from financial inputs.
    annual_income = _validate_finite_number(
        "annual_income", annual_income
    )
    annual_savings = _validate_finite_number(
        "annual_savings", annual_savings
    )

    if annual_income <= 0 or annual_savings < 0:
        raise ValueError("Invalid income or savings values.")

    actual_savings_ratio = annual_savings / annual_income

    # New optional questionnaire answers.
    answers = kwargs.get("answers")

    if answers is None:
        # Compatibility fallback for old callers.
        # New assessments should always supply real answers.
        answers = [
            {"question_id": 1, "selected_option": "B"},
            {"question_id": 2, "selected_option": "B"},
            {"question_id": 3, "selected_option": "B"},
            {"question_id": 4, "selected_option": "B"},
            {"question_id": 5, "selected_option": "B"},
        ]

    objective = calculate_objective_capacity(
        age=age,
        investment_horizon_years=investment_horizon_years,
        annual_income=annual_income,
        annual_savings=annual_savings,
        portfolio_value=portfolio_value,
        max_loss_amount=effective_max_loss_amount,
    )

    behavioral = calculate_behavioral_tolerance(answers)

    capacity_score = objective["objective_risk_score"]
    behavioral_score = behavioral["behavioral_risk_score"]

    raw_combined_score = (
        0.60 * capacity_score
        + 0.40 * behavioral_score
    )

    combined_score = min(
        capacity_score,
        raw_combined_score
    )

    combined_score = max(
        0.0,
        min(1.0, combined_score)
    )

    discrete_score = int(
        max(1, min(10, round(combined_score * 9 + 1)))
    )

    category = _get_risk_category(discrete_score)

    # IMPORTANT:
    # Original return keys are preserved.
    return {
        # ORIGINAL KEYS
        "continuous_risk_score_Ru": round(combined_score, 4),
        "discrete_risk_score": discrete_score,
        "category": category,
        "max_loss_tolerance_pct": round(
            max_loss_tolerance_pct, 4
        ),
        "max_loss_amount": max_loss_amount,
        "investment_horizon_years": investment_horizon_years,
        "portfolio_value": portfolio_value,
        "age": age,

        # NEW KEYS
        "objective_risk_score": round(capacity_score, 4),
        "behavioral_risk_score": round(behavioral_score, 4),
        "combined_risk_score": round(combined_score, 4),
        "calculated_risk_score": discrete_score,
        "risk_category": category,
        "savings_to_income_ratio": round(
            actual_savings_ratio, 4
        ),
        "liquidity_response": behavioral["liquidity_response"],
        "objective_components": objective["component_scores"],
        "answer_details": behavioral["answer_details"],
        "model_version": MODEL_VERSION,
        "questionnaire_version": QUESTIONNAIRE_VERSION,
    }


def get_calculated_risk_score(
    investment_horizon_years: int = 5,
    max_loss_amount: float = 50000.0,
    target_amount: Optional[float] = None,
    portfolio_value: float = 500000.0,
    **kwargs
) -> int:
    """
    Backward-compatible entry point.

    Always returns an integer from 1 to 10.
    """

    profile = calculate_user_risk_profile(
        investment_horizon_years=investment_horizon_years,
        max_loss_amount=max_loss_amount,
        target_amount=target_amount,
        portfolio_value=portfolio_value,
        **kwargs
    )

    return profile["discrete_risk_score"]
