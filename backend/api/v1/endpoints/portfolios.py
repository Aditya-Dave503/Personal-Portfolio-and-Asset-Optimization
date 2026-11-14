from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from core.database import get_db
from core.security import get_current_user_id
from schemas.portfolio import PortfolioAllocationRead, PortfolioOptimizeRequest, PortfolioRead


router = APIRouter(prefix="/portfolios", tags=["Portfolios"])


def _service_function(name):
    try:
        from services import portfolio_service

        return getattr(portfolio_service, name)
    except (ImportError, AttributeError) as exc:
        raise HTTPException(status_code=503, detail="Portfolio service is unavailable") from exc


def _not_found_as_http(fn, *args):
    try:
        result = fn(*args)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="Portfolio not found") from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    return result


@router.post("/optimize", response_model=PortfolioRead, status_code=201,
             summary="Optimize a portfolio", description="Run the selected optimizer for the authenticated user.",
             responses={201: {"description": "Generated portfolio"}})
def optimize_portfolio(data: PortfolioOptimizeRequest, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    try:
        return _service_function("optimize_portfolio")(
            db, user_id, data.investment_amount, data.optimization_mode
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/", response_model=list[PortfolioRead], summary="List portfolios")
def list_portfolios(user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return _service_function("list_portfolios")(db, user_id)


@router.get("/{portfolio_id}", response_model=PortfolioRead, summary="Get a portfolio")
def get_portfolio(portfolio_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return _not_found_as_http(_service_function("get_portfolio"), db, user_id, portfolio_id)


@router.get("/{portfolio_id}/allocations", response_model=list[PortfolioAllocationRead], summary="Get portfolio allocations")
def get_portfolio_allocations(portfolio_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return _not_found_as_http(
        _service_function("get_portfolio_allocations"), db, user_id, portfolio_id
    )
