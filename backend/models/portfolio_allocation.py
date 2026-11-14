from decimal import Decimal

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Index,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column,relationship

from core.database import Base


class PortfolioAllocation(Base):
    __tablename__ = "portfolio_allocations"

    __table_args__ = (
        CheckConstraint(
            "weight_percent >= 0 AND weight_percent <= 100",
            name="ck_allocations_weight_range",
        ),
        CheckConstraint(
            "allocated_amount >= 0",
            name="ck_allocations_amount_nonnegative",
        ),
        UniqueConstraint(
            "portfolio_id",
            "asset_code",
            name="uq_allocations_portfolio_asset",
        ),
        Index("ix_portfolio_allocations_portfolio_id", "portfolio_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    portfolio_id: Mapped[int] = mapped_column(
        ForeignKey("portfolios.id", ondelete="CASCADE"),
        nullable=False,
    )

    asset_code: Mapped[str] = mapped_column(
        String(50), nullable=False
    )

    asset_name: Mapped[str] = mapped_column(
        String(150), nullable=False
    )

    category: Mapped[str] = mapped_column(
        String(100), nullable=False
    )

    risk_level: Mapped[str] = mapped_column(
        String(100), nullable=False
    )

    weight_percent: Mapped[Decimal] = mapped_column(
        Numeric(10, 4), nullable=False
    )

    allocated_amount: Mapped[Decimal] = mapped_column(
        Numeric(14, 2), nullable=False
    )

    ml_expected_annual_return_pct: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 4), nullable=True
    )
    portfolio: Mapped["Portfolio"] = relationship(
        back_populates="allocations"
    )
