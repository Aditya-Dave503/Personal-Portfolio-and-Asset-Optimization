from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_register_user():
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "autotest@example.com",
            "password": "TestPassword123"
        }
    )

    assert response.status_code == 201

    data = response.json()

    assert data["email"] == "autotest@example.com"
    assert "password" not in data
    assert "password_hash" not in data


def test_duplicate_email():
    client.post(
        "/api/v1/auth/register",
        json={
            "email": "duplicate@example.com",
            "password": "TestPassword123"
        }
    )

    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "duplicate@example.com",
            "password": "TestPassword123"
        }
    )

    assert response.status_code == 409


def test_login_success():
    client.post(
        "/api/v1/auth/register",
        json={
            "email": "login@example.com",
            "password": "TestPassword123"
        }
    )

    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "login@example.com",
            "password": "TestPassword123"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_login_wrong_password():
    client.post(
        "/api/v1/auth/register",
        json={
            "email": "wrongpass@example.com",
            "password": "TestPassword123"
        }
    )

    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "wrongpass@example.com",
            "password": "WrongPassword123"
        }
    )

    assert response.status_code == 401


def test_me_without_token():
    response = client.get(
        "/api/v1/auth/me"
    )

    assert response.status_code == 401


def test_me_with_invalid_token():
    response = client.get(
        "/api/v1/auth/me",
        headers={
            "Authorization": "Bearer invalid-token"
        }
    )

    assert response.status_code == 401


def test_me_with_valid_token():
    client.post(
        "/api/v1/auth/register",
        json={
            "email": "me@example.com",
            "password": "TestPassword123"
        }
    )

    login_response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "me@example.com",
            "password": "TestPassword123"
        }
    )

    token = login_response.json()["access_token"]

    response = client.get(
        "/api/v1/auth/me",
        headers={
            "Authorization": f"Bearer {token}"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["email"] == "me@example.com"
    assert "password" not in data
    assert "password_hash" not in data
