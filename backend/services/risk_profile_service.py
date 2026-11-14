
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
    profile_data: RiskProfileCreate,
):
    """Create the user's first profile or update their existing profile."""

    if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id <= 0:
        raise RiskProfileValidationError("Invalid user ID")

    age = profile_data.age
    horizon = profile_data.investment_horizon_years
    annual_income = profile_data.annual_income
    annual_savings = profile_data.annual_savings
    portfolio_value = profile_data.portfolio_value
    max_loss = profile_data.max_loss_amount

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

    if isinstance(age, bool) or not isinstance(age, int) or not 18 <= age <= 100:
        raise RiskProfileValidationError("Invalid age")

    if (
        isinstance(horizon, bool)
        or not isinstance(horizon, int)
        or not 1 <= horizon <= 60
    ):
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

    answers = [
        {
            "question_id": answer.question_id,
            "selected_option": answer.selected_option,
        }
        for answer in profile_data.answers
    ]

    question_ids = [answer["question_id"] for answer in answers]

    if len(question_ids) != len(set(question_ids)):
        raise RiskProfileValidationError(
            "Each questionnaire question must be answered only once"
        )

    # Calculate the new assessment before modifying the database.
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

    risk_score = result["discrete_risk_score"]

    if (
        isinstance(risk_score, bool)
        or not isinstance(risk_score, int)
        or not 1 <= risk_score <= 10
    ):
        raise RiskScoringError(
            "Risk scoring engine returned an invalid score"
        )

    answer_score_map = {
        item["question_id"]: item["behavioral_score"]
        for item in result["answer_details"]
    }

    saved_answers = [
        RiskProfileAnswer(
            question_id=answer["question_id"],
            selected_option=answer["selected_option"],
            behavioral_score=answer_score_map.get(
                answer["question_id"]
            ),
        )
        for answer in answers
    ]

    # Prepare all calculated fields together.
    profile_values = {
        "age": age,
        "annual_income": annual_income,
        "annual_savings": annual_savings,
        "portfolio_value": portfolio_value,
        "max_loss_amount": max_loss,
        "investment_horizon_years": horizon,
        "savings_to_income_ratio": result["savings_to_income_ratio"],
        "max_loss_tolerance_pct": result["max_loss_tolerance_pct"],
        "objective_risk_score": result["objective_risk_score"],
        "behavioral_risk_score": result["behavioral_risk_score"],
        "combined_risk_score": result["combined_risk_score"],
        "calculated_risk_score": risk_score,
        "risk_category": result["risk_category"],
        "model_version": result["model_version"],
        "questionnaire_version": result["questionnaire_version"],
    }

    try:
        # Lock an existing profile while updating it.
        profile = (
            db.query(RiskProfile)
            .filter(RiskProfile.user_id == user_id)
            .with_for_update()
            .first()
        )

        if profile is None:
            profile = RiskProfile(
                user_id=user_id,
                **profile_values,
                answers=saved_answers,
            )
            db.add(profile)
            created = True

        else:
            # Preserve profile ID and created_at.
            for field, value in profile_values.items():
                setattr(profile, field, value)

            # Replace the previous questionnaire answers.
            # delete-orphan removes the old answer records.
            profile.answers = saved_answers
            created = False

        db.commit()
        db.refresh(profile)

    except Exception:
        db.rollback()
        raise

    return profile, created


def get_risk_profile(db: Session, user_id: int):
    """Return the user's single active risk profile."""

    return (
        db.query(RiskProfile)
        .filter(RiskProfile.user_id == user_id)
        .first()
    )
