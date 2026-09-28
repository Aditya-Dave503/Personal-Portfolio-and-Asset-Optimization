from pydantic import BaseModel, Field


class RiskProfileCreate(BaseModel):
    max_loss_amount: float = Field(gt=0)
    investment_horizon_years: int = Field(gt=0)


class RiskProfileResponse(BaseModel):
    id: int
    user_id: int
    calculated_risk_score: float | None
    max_loss_amount: float
    investment_horizon_years: int

    model_config = {
        "from_attributes": True
    }