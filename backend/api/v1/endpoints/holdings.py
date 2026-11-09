from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from core.database import get_db
from core.security import get_current_user_id
from schemas.holding import HoldingCreate, HoldingRead, HoldingUpdate


router = APIRouter(prefix="/holdings", tags=["Holdings"])


def _service_function(name):
    try:
        from services import holding_service

        return getattr(holding_service, name)
    except (ImportError, AttributeError) as exc:
        raise HTTPException(status_code=503, detail="Holdings service is unavailable") from exc


@router.post("/", response_model=HoldingRead, status_code=status.HTTP_201_CREATED,
             summary="Add a holding", description="Create a holding for the authenticated user.",
             responses={201: {"description": "Created holding"}})
def create_holding(data: HoldingCreate, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return _service_function("create_holding")(db, user_id, data)


@router.get("/", response_model=list[HoldingRead], summary="List holdings")
def list_holdings(user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return _service_function("list_holdings")(db, user_id)


@router.put("/{holding_id}", response_model=HoldingRead, summary="Update a holding")
def update_holding(holding_id: int, data: HoldingUpdate, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    try:
        result = _service_function("update_holding")(db, user_id, holding_id, data)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="Holding not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Holding not found")
    return result


@router.delete("/{holding_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a holding")
def delete_holding(holding_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    try:
        _service_function("delete_holding")(db, user_id, holding_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="Holding not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)
