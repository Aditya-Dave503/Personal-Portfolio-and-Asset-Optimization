from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root():
    response = client.get("/")

    assert response.status_code == 200

def test_app_loads():
    assert app is not None
    assert app.title == "OptiWealth API"

def test_user_routes_registered():
    routes = app.openapi()["paths"]

    assert "/api/v1/users/" in routes