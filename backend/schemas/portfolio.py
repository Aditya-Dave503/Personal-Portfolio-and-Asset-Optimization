from datetime import datetime
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints


OptimizationMode = Literal["black_litterman", "max_sharpe", "min_variance"]
AssetType = Literal["NIFTY_EQUITIES", "MUTUAL_FUNDS", "MCX_GOLD", "GOVT_10Y_GSEC"]


class PortfolioOptimizeRequest(BaseModel):
    investment_amount: Decimal = Field(gt=0, max_digits=14, decimal_places=2)
    optimization_mode: OptimizationMode = Field(
        json_schema_extra={"examples": ["max_sharpe"]}
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"investment_amount": "100000.00", "optimization_mode": "max_sharpe"}
            ]
        }
    }


class PortfolioAllocationRead(BaseModel):
    id: int
    portfolio_id: int
    asset_type: AssetType
    weight: Decimal = Field(ge=0, le=1, max_digits=8, decimal_places=6)
    amount: Decimal = Field(ge=0, max_digits=14, decimal_places=2)

    model_config = {"from_attributes": True}


class PortfolioRead(BaseModel):
    id: int
    user_id: int
    investment_amount: Decimal = Field(gt=0)
    optimization_mode: OptimizationMode
    expected_return: Decimal | None = None
    sharpe_ratio: Decimal | None = None
    value_at_risk: Decimal | None = None
    generated_at: datetime

    model_config = {"from_attributes": True}


class PortfolioDetailRead(PortfolioRead):
    allocations: list[PortfolioAllocationRead] = Field(default_factory=list)


class PortfolioHistoryRead(BaseModel):
    items: list[PortfolioRead]
    total: int


class DashboardRead(BaseModel):
    total_invested: Decimal
    holdings_count: int
    portfolio_count: int


class PortfolioComparisonRead(BaseModel):
    portfolios: list[PortfolioRead]


class ReportRead(BaseModel):
    portfolio: PortfolioDetailRead
    summary: Annotated[str, StringConstraints(min_length=1)]
