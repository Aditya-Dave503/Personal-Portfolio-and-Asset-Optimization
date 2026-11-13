
import math

from sqlalchemy.orm import Session

from models.risk_profile import RiskProfile, RiskProfileAnswer
from schemas.risk_profile import RiskProfileCreate
from ml.risk_profiler import calculate_user_risk_profile


class RiskProfileValidationError(Exception):
    """Raised when risk profile inputs are invalid."""


class RiskScoringError(Exception):
    """Raised when risk scoring fails or returns an invalid result."""


def create_risk_profile(
    db: Session,
    user_id: int,
    profile_data: RiskProfileCreate
):
    # 1. Validate user ID
    if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id <= 0:
        raise RiskProfileValidationError("Invalid user ID")

    # 2. Extract financial inputs
    age = profile_data.age
    horizon = profile_data.investment_horizon_years
    annual_income = profile_data.annual_income
    annual_savings = profile_data.annual_savings
    portfolio_value = profile_data.portfolio_value
    max_loss = profile_data.max_loss_amount

    # 3. Validate numerical inputs
    numeric_inputs = {
        "annual_income": annual_income,
        "annual_savings": annual_savings,
        "portfolio_value": portfolio_value,
        "max_loss_amount": max_loss,
    }

    for name, value in numeric_inputs.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise RiskProfileValidationError(
                f"{name} must be a valid finite number"
            )

    if not 18 <= age <= 100:
        raise RiskProfileValidationError("Invalid age")

    if not 1 <= horizon <= 60:
        raise RiskProfileValidationError(
            "Investment horizon must be between 1 and 60 years"
        )

    if annual_income <= 0:
        raise RiskProfileValidationError(
            "Annual income must be greater than zero"
        )

    if annual_savings < 0:
        raise RiskProfileValidationError(
            "Annual savings cannot be negative"
        )

    if portfolio_value <= 0:
        raise RiskProfileValidationError(
            "Portfolio value must be greater than zero"
        )

    if max_loss <= 0 or max_loss > portfolio_value:
        raise RiskProfileValidationError(
            "Maximum loss must be positive and cannot exceed portfolio value"
        )

    # 4. Prepare questionnaire answers
    answers = [
        {
            "question_id": answer.question_id,
            "selected_option": answer.selected_option,
        }
        for answer in profile_data.answers
    ]

    # 5. Calculate risk profile
    try:
        result = calculate_user_risk_profile(
            age=age,
            investment_horizon_years=horizon,
            annual_income=annual_income,
            annual_savings=annual_savings,
            portfolio_value=portfolio_value,
            max_loss_amount=max_loss,
            answers=answers,
        )

    except (ValueError, TypeError, KeyError) as exc:
        raise RiskProfileValidationError(
            f"Invalid risk assessment inputs: {exc}"
        ) from exc

    except Exception as exc:
        raise RiskScoringError(
            "Unable to calculate risk profile"
        ) from exc

    # 6. Validate risk score
    risk_score = result["discrete_risk_score"]

    if (
        isinstance(risk_score, bool)
        or not isinstance(risk_score, int)
        or not 1 <= risk_score <= 10
    ):
        raise RiskScoringError(
            "Risk scoring engine returned an invalid score"
        )

    # 7. Map behavioural scores to their question IDs.
    # Question 5 is intentionally absent and will have None.
    answer_score_map = {
        item["question_id"]: item["behavioral_score"]
        for item in result["answer_details"]
    }

    # 8. Create answer database records
    saved_answers = [
        RiskProfileAnswer(
            question_id=answer.question_id,
            selected_option=answer.selected_option,
            behavioral_score=answer_score_map.get(
                answer.question_id
            ),
        )
        for answer in profile_data.answers
    ]

    # 9. Create the risk profile
    profile = RiskProfile(
        user_id=user_id,

        age=age,
        annual_income=annual_income,
        annual_savings=annual_savings,
        portfolio_value=portfolio_value,

        max_loss_amount=max_loss,
        investment_horizon_years=horizon,

        savings_to_income_ratio=result[
            "savings_to_income_ratio"
        ],
        max_loss_tolerance_pct=result[
            "max_loss_tolerance_pct"
        ],

        objective_risk_score=result[
            "objective_risk_score"
        ],
        behavioral_risk_score=result[
            "behavioral_risk_score"
        ],
        combined_risk_score=result[
            "combined_risk_score"
        ],

        calculated_risk_score=risk_score,
        risk_category=result["risk_category"],

        model_version=result["model_version"],
        questionnaire_version=result["questionnaire_version"],

        answers=saved_answers,
    )

    # 10. Save everything in one transaction
    try:
        db.add(profile)
        db.commit()
        db.refresh(profile)

    except Exception:
        db.rollback()
        raise

    return profile


def get_risk_profile(db: Session, user_id: int):
    """Return the latest risk assessment for a user."""

    return (
        db.query(RiskProfile)
        .filter(RiskProfile.user_id == user_id)
        .order_by(RiskProfile.created_at.desc())
        .first()
    )


def get_risk_profile_history(db: Session, user_id: int):
    """Return all risk assessments for a user."""

    return (
        db.query(RiskProfile)
        .filter(RiskProfile.user_id == user_id)
        .order_by(RiskProfile.created_at.desc())
        .all()
    )
