
from decimal import Decimal, InvalidOperation
import math
import re

from sqlalchemy.orm import Session

from models.holding import Holding


# Assets accepted by the current Indian portfolio optimizer
OPTIMIZER_SUPPORTED_ASSETS = {
    "NIFTY_EQUITIES",
    "MUTUAL_FUNDS",
    "MCX_GOLD",
    "GOVT_10Y_GSEC",
}


class HoldingValidationError(Exception):
    """Raised when holding data is invalid."""


class HoldingNotFoundError(Exception):
    """Raised when a holding does not exist for the specified user."""


def is_optimizer_supported_asset(asset_code: str) -> bool:
    """Check whether an asset is supported by the current optimizer."""
    return asset_code in OPTIMIZER_SUPPORTED_ASSETS


def _validate_holding_data(data: dict) -> dict:
    """Validate and normalize holding fields."""

    required_fields = {
        "asset_code",
        "asset_class",
        "instrument_name",
        "quantity",
        "purchase_value",
        "current_value",
    }

    if set(data) != required_fields:
        raise HoldingValidationError("Invalid holding fields")

    asset_code = data["asset_code"]
    asset_class = data["asset_class"]
    instrument_name = data["instrument_name"]

    if not isinstance(asset_code, str):
        raise HoldingValidationError("Invalid asset code")

    asset_code = asset_code.strip().upper()

    if not re.fullmatch(r"[A-Z0-9_]{1,50}", asset_code):
        raise HoldingValidationError("Invalid asset code")

    for field_name, value, max_length in [
        ("asset_class", asset_class, 100),
        ("instrument_name", instrument_name, 150),
    ]:
        if (
            not isinstance(value, str)
            or not value.strip()
            or len(value.strip()) > max_length
        ):
            raise HoldingValidationError(
                f"Invalid {field_name.replace('_', ' ')}"
            )

    normalized = {
        "asset_code": asset_code,
        "asset_class": asset_class.strip(),
        "instrument_name": instrument_name.strip(),
    }

    for field_name in ("quantity", "purchase_value", "current_value"):
        value = data[field_name]

        if isinstance(value, bool):
            raise HoldingValidationError(f"Invalid {field_name}")

        try:
            number = Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError):
            raise HoldingValidationError(f"Invalid {field_name}")

        if not number.is_finite():
            raise HoldingValidationError(f"Invalid {field_name}")

        if field_name == "quantity" and number <= 0:
            raise HoldingValidationError("Quantity must be positive")

        if field_name != "quantity" and number < 0:
            raise HoldingValidationError(
                f"{field_name} cannot be negative"
            )

        normalized[field_name] = number

    return normalized


def create_holding(
    db: Session,
    user_id: int,
    *,
    asset_code: str,
    asset_class: str,
    instrument_name: str,
    quantity,
    purchase_value,
    current_value,
) -> Holding:
    """Create a holding belonging to a specific user."""

    if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id <= 0:
        raise HoldingValidationError("Invalid user ID")

    data = _validate_holding_data({
        "asset_code": asset_code,
        "asset_class": asset_class,
        "instrument_name": instrument_name,
        "quantity": quantity,
        "purchase_value": purchase_value,
        "current_value": current_value,
    })

    holding = Holding(user_id=user_id, **data)

    try:
        db.add(holding)
        db.commit()
        db.refresh(holding)
    except Exception:
        db.rollback()
        raise

    return holding


def get_holdings(db: Session, user_id: int) -> list[Holding]:
    """Get all holdings belonging to a user."""

    return (
        db.query(Holding)
        .filter(Holding.user_id == user_id)
        .order_by(Holding.id)
        .all()
    )


def get_holding(
    db: Session,
    user_id: int,
    holding_id: int,
) -> Holding | None:
    """Get a specific holding only if it belongs to the user."""

    return (
        db.query(Holding)
        .filter(
            Holding.user_id == user_id,
            Holding.id == holding_id,
        )
        .first()
    )


def update_holding(
    db: Session,
    user_id: int,
    holding_id: int,
    updates: dict,
) -> Holding:
    """Update a holding only if it belongs to the user."""

    holding = get_holding(db, user_id, holding_id)

    if holding is None:
        raise HoldingNotFoundError("Holding not found")

    allowed_fields = {
        "asset_code",
        "asset_class",
        "instrument_name",
        "quantity",
        "purchase_value",
        "current_value",
    }

    if not isinstance(updates, dict) or not updates:
        raise HoldingValidationError("No valid update data provided")

    if not set(updates).issubset(allowed_fields):
        raise HoldingValidationError("Unsupported update fields")

    current_data = {
        field: getattr(holding, field)
        for field in allowed_fields
    }

    current_data.update(updates)

    validated_data = _validate_holding_data(current_data)

    try:
        for field, value in validated_data.items():
            setattr(holding, field, value)

        db.commit()
        db.refresh(holding)
    except Exception:
        db.rollback()
        raise

    return holding


def delete_holding(
    db: Session,
    user_id: int,
    holding_id: int,
) -> None:
    """Delete a holding only if it belongs to the user."""

    holding = get_holding(db, user_id, holding_id)

    if holding is None:
        raise HoldingNotFoundError("Holding not found")

    try:
        db.delete(holding)
        db.commit()
    except Exception:
        db.rollback()
        raise
