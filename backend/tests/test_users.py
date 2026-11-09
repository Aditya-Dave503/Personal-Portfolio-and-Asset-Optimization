from fastapi.testclient import TestClient
from uuid import uuid4
from main import app


client = TestClient(app)

def test_create_user():
    email = f"test_{uuid4().hex}@example.com"

    response = client.post(
        "/api/v1/users/",
        json={
            "email": email,
            "password": "StrongPassword123"
        }
    )

    assert response.status_code == 201

    data = response.json()

    assert "id" in data
    assert data["email"] == email
    assert "password_hash" not in data

def test_get_user_requires_authentication():
    response = client.get("/api/v1/users/1")
    assert response.status_code in (401, 403)


def test_duplicate_email():
    email = f"duplicate_{uuid4().hex}@example.com"

    first_response = client.post(
        "/api/v1/users/",
        json={
            "email": email,
            "password": "StrongPassword123"
        }
    )

    assert first_response.status_code == 201

    second_response = client.post(
        "/api/v1/users/",
        json={
            "email": email,
            "password": "StrongPassword123"
        }
    )

    assert second_response.status_code == 409
    assert second_response.json()["detail"] == "Email already registered"

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
    assert response.status_code in (401, 403)
