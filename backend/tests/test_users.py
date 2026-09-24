from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_create_user():
    response = client.post(
        "/api/v1/users/",
        json={
            "email": "test_new@example.com",
            "password": "StrongPassword123"
        }
    )

    assert response.status_code == 201

    data = response.json()

    assert "id" in data
    assert data["email"] == "test_new@example.com"
    assert "password_hash" not in data


def test_get_existing_user():
    response = client.get("/api/v1/users/1")

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == 1
    assert "email" in data
    assert "password_hash" not in data


def test_duplicate_email():
    response = client.post(
        "/api/v1/users/",
        json={
            "email": "test_new@example.com",
            "password": "StrongPassword123"
        }
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Email already registered"


def test_invalid_email():
    response = client.post(
        "/api/v1/users/",
        json={
            "email": "not-an-email",
            "password": "StrongPassword123"
        }
    )

    assert response.status_code == 422


def test_missing_user():
    response = client.get("/api/v1/users/999999")

    assert response.status_code == 404