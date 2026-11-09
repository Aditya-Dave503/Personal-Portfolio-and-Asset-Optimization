from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column,relationship

from core.database import Base


class Holding(Base):
    __tablename__ = "holdings"

    __table_args__ = (
        CheckConstraint(
            "quantity > 0",
            name="ck_holdings_quantity_positive",
        ),
        CheckConstraint(
            "purchase_value >= 0",
            name="ck_holdings_purchase_value_nonnegative",
        ),
        CheckConstraint(
            "current_value >= 0",
            name="ck_holdings_current_value_nonnegative",
        ),
        Index("ix_holdings_user_id", "user_id"),
    )

    id: Mapped[int] = mapped_column(
        primary_key=True, autoincrement=True
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    asset_code: Mapped[str] = mapped_column(
        String(50), nullable=False
    )

    asset_class: Mapped[str] = mapped_column(
        String(100), nullable=False
    )

    instrument_name: Mapped[str] = mapped_column(
        String(150), nullable=False
    )

    quantity: Mapped[Decimal] = mapped_column(
        Numeric(18, 6), nullable=False
    )

    purchase_value: Mapped[Decimal] = mapped_column(
        Numeric(14, 2), nullable=False
    )

    current_value: Mapped[Decimal] = mapped_column(
        Numeric(14, 2), nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    user: Mapped["User"] = relationship(
        back_populates="holdings"
    )
