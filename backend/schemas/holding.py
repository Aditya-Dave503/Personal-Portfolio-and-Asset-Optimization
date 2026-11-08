from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, model_validator


AssetType = Literal["STOCK", "MF", "GOLD", "SILVER", "BOND"]
HoldingName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=120),
]


class HoldingCreate(BaseModel):
    asset_type: AssetType
    name: HoldingName
    units: Decimal = Field(gt=0, max_digits=16, decimal_places=6)
    invested_value: Decimal = Field(gt=0, max_digits=14, decimal_places=2)

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "asset_type": "STOCK",
                "name": "Nifty 50 ETF",
                "units": "4.5",
                "invested_value": "12000.00",
            }]
        }
    }


class HoldingUpdate(BaseModel):
    asset_type: AssetType | None = None
    name: HoldingName | None = None
    units: Decimal | None = Field(default=None, gt=0, max_digits=16, decimal_places=6)
    invested_value: Decimal | None = Field(default=None, gt=0, max_digits=14, decimal_places=2)

    @model_validator(mode="after")
    def validate_update(self):
        if not self.model_fields_set or any(
            getattr(self, field) is None for field in self.model_fields_set
        ):
            raise ValueError("Provide at least one non-null holding field")
        return self


class HoldingRead(BaseModel):
    id: int
    user_id: int
    asset_type: AssetType
    name: str
    units: Decimal
    invested_value: Decimal

    model_config = {"from_attributes": True}
