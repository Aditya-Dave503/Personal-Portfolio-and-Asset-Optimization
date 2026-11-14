from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from api.v1.endpoints import holdings
from core.database import get_db
from core.security import get_current_user_id
from main import app


@pytest.fixture
def client():
    app.dependency_overrides[get_db] = lambda: object()
    app.dependency_overrides[get_current_user_id] = lambda: 17
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def payload(**updates):
    return {
        "asset_type": "STOCK",
        "name": "Nifty 50 ETF",
        "units": "4.5",
        "invested_value": "12000.00",
        **updates,
    }


def test_create_list_update_delete_holdings_use_authenticated_owner(client, monkeypatch):
    calls = []
    holding = {"id": 3, "user_id": 17, **payload()}

    def resolve(name):
        def service(*args):
            calls.append((name, args))
            return [holding] if name == "list_holdings" else holding
        return service

    monkeypatch.setattr(holdings, "_service_function", resolve)
    assert client.post("/api/v1/holdings/", json=payload()).status_code == 201
    assert client.get("/api/v1/holdings/").json()[0]["user_id"] == 17
    assert client.put("/api/v1/holdings/3", json={"name": "Updated"}).status_code == 200
    assert client.delete("/api/v1/holdings/3").status_code == 204
    assert all(args[1] == 17 for _, args in calls)


@pytest.mark.parametrize("invalid", [
    {"invested_value": "1.001"},
    {"asset_type": "UNKNOWN"},
    {"units": "0"},
    {"invested_value": "-1"},
])
def test_invalid_holding_payload_is_rejected(client, invalid):
    assert client.post("/api/v1/holdings/", json=payload(**invalid)).status_code == 422


def test_empty_update_is_rejected(client):
    assert client.put("/api/v1/holdings/3", json={}).status_code == 422


def test_holding_routes_require_authentication(monkeypatch):
    app.dependency_overrides.clear()
    monkeypatch.setattr(holdings, "_service_function", lambda name: lambda *args: [])
    with TestClient(app) as client:
        assert client.get("/api/v1/holdings/").status_code in (401, 403)


def test_missing_holding_service_returns_service_unavailable(client):
    assert client.get("/api/v1/holdings/").status_code == 503
