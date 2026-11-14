from datetime import datetime, timezone
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from api.v1.endpoints import portfolios
from core.database import get_db
from core.security import get_current_user_id
from main import app


@pytest.fixture
def client():
    app.dependency_overrides[get_db] = lambda: object()
    app.dependency_overrides[get_current_user_id] = lambda: 23
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def portfolio():
    return {
        "id": 8,
        "user_id": 23,
        "investment_amount": "50000.00",
        "optimization_mode": "max_sharpe",
        "expected_return": "0.08",
        "sharpe_ratio": "1.2",
        "value_at_risk": "3000.00",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def test_optimize_and_list_portfolios_use_authenticated_owner(client, monkeypatch):
    calls = []

    def resolve(name):
        def service(*args):
            calls.append((name, args))
            return [portfolio()] if name == "list_portfolios" else portfolio()
        return service

    monkeypatch.setattr(portfolios, "_service_function", resolve)
    result = client.post(
        "/api/v1/portfolios/optimize",
        json={"investment_amount": "50000.00", "optimization_mode": "max_sharpe"},
    )
    assert result.status_code == 201
    assert result.json()["optimization_mode"] == "max_sharpe"
    assert client.get("/api/v1/portfolios/").status_code == 200
    assert calls[0][1][1] == 23
    assert calls[0][1][2:] == (Decimal("50000.00"), "max_sharpe")


@pytest.mark.parametrize("payload_data", [
    {"investment_amount": "0", "optimization_mode": "max_sharpe"},
    {"investment_amount": "-50", "optimization_mode": "max_sharpe"},
    {"investment_amount": "100", "optimization_mode": "unknown"},
    {"investment_amount": "100"},
])
def test_invalid_optimization_request_is_rejected(client, payload_data):
    assert client.post("/api/v1/portfolios/optimize", json=payload_data).status_code == 422


def test_portfolio_and_allocation_not_found(client, monkeypatch):
    monkeypatch.setattr(portfolios, "_service_function", lambda name: lambda *args: None)
    assert client.get("/api/v1/portfolios/999").status_code == 404
    assert client.get("/api/v1/portfolios/999/allocations").status_code == 404


def test_portfolio_routes_require_authentication(monkeypatch):
    app.dependency_overrides.clear()
    monkeypatch.setattr(portfolios, "_service_function", lambda name: lambda *args: [])
    with TestClient(app) as client:
        response = client.get("/api/v1/portfolios/")
        assert response.status_code in (401, 403)


def test_missing_portfolio_service_returns_service_unavailable(client):
    assert client.get("/api/v1/portfolios/").status_code == 503


def test_optimizer_service_failure_is_not_returned_as_success(client, monkeypatch):
    monkeypatch.setattr(
        portfolios,
        "_service_function",
        lambda name: lambda *args: (_ for _ in ()).throw(RuntimeError("optimizer failed")),
    )
    with TestClient(app, raise_server_exceptions=False) as failing_client:
        response = failing_client.post(
            "/api/v1/portfolios/optimize",
            json={"investment_amount": "25000", "optimization_mode": "min_variance"},
        )
    assert response.status_code == 500
