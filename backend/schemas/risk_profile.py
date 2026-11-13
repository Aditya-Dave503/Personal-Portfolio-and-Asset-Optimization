
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


# --------------------------------------------------
# Behavioural questionnaire
# --------------------------------------------------

class RiskAssessmentAnswerCreate(BaseModel):
    question_id: int = Field(ge=1, le=5)

    selected_option: Literal["A", "B", "C", "D"]


# --------------------------------------------------
# Create risk assessment
# --------------------------------------------------

class RiskProfileCreate(BaseModel):

    # Financial inputs
    age: int = Field(ge=18, le=100)

    investment_horizon_years: int = Field(ge=1, le=60)

    annual_income: float = Field(gt=0)

    annual_savings: float = Field(ge=0)

    portfolio_value: float = Field(gt=0)

    max_loss_amount: float = Field(gt=0)

    # Five behavioural answers
    answers: list[RiskAssessmentAnswerCreate] = Field(
        min_length=5,
        max_length=5
    )

    @model_validator(mode="after")
    def validate_assessment(self):

        # Maximum acceptable loss cannot exceed portfolio value
        if self.max_loss_amount > self.portfolio_value:
            raise ValueError(
                "Maximum loss amount cannot exceed portfolio value"
            )

        # Ensure all five questions are answered exactly once
        question_ids = [answer.question_id for answer in self.answers]

        if len(set(question_ids)) != 5:
            raise ValueError(
                "Each behavioural question must be answered exactly once"
            )

        if set(question_ids) != {1, 2, 3, 4, 5}:
            raise ValueError(
                "Question IDs must be 1, 2, 3, 4, and 5"
            )

        return self


# --------------------------------------------------
# Behavioural answer response
# --------------------------------------------------

class RiskAssessmentAnswerResponse(BaseModel):
    id: int
    question_id: int
    selected_option: str
    behavioral_score: float | None


    model_config = {
        "from_attributes": True
    }


# --------------------------------------------------
# Risk assessment response
# --------------------------------------------------

class RiskProfileResponse(BaseModel):

    id: int
    user_id: int

    # Original financial inputs
    age: int
    investment_horizon_years: int
    annual_income: float
    annual_savings: float
    portfolio_value: float
    max_loss_amount: float

    # Derived financial features
    savings_to_income_ratio: float
    max_loss_tolerance_pct: float

    # Calculated risk scores
    objective_risk_score: float | None
    behavioral_risk_score: float | None
    combined_risk_score: float | None

    # Existing optimizer-compatible score
    calculated_risk_score: int | None

    risk_category: str | None

    # Assessment versioning
    model_version: str
    questionnaire_version: str

    # Stored answers
    answers: list[RiskAssessmentAnswerResponse]

    created_at: datetime

    model_config = {
        "from_attributes": True
    }
