from datetime import date
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, model_validator


GoalCategory = Literal[
    "retirement",
    "house_down_payment",
    "higher_education",
    "other"
]
GoalName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=100)
]


class GoalCreate(BaseModel):
    name: GoalName
    category: GoalCategory
    target_amount: Decimal = Field(gt=0, max_digits=12, decimal_places=2)
    target_date: date

    @model_validator(mode="after")
    def validate_target_date(self):
        if self.target_date < date.today():
            raise ValueError("Target date cannot be in the past")
        return self


class GoalUpdate(BaseModel):
    name: GoalName | None = None
    category: GoalCategory | None = None
    target_amount: Decimal | None = Field(
        default=None,
        gt=0,
        max_digits=12,
        decimal_places=2
    )
    target_date: date | None = None

    @model_validator(mode="after")
    def validate_update(self):
        if not self.model_fields_set:
            raise ValueError("At least one field must be provided")

        if any(getattr(self, field) is None for field in self.model_fields_set):
            raise ValueError("Goal fields cannot be null")

        if self.target_date is not None and self.target_date < date.today():
            raise ValueError("Target date cannot be in the past")

        return self


class GoalResponse(BaseModel):
    id: int
    user_id: int
    name: str
    category: GoalCategory
    target_amount: Decimal
    target_date: date

    model_config = {
        "from_attributes": True
    }
