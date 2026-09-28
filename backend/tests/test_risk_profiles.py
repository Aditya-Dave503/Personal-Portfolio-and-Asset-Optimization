from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def get_auth_token():
    client.post(
        "/api/v1/auth/register",
        json={
            "email": "riskprofile@example.com",
            "password": "TestPassword123"
        }
    )

    login_response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "riskprofile@example.com",
            "password": "TestPassword123"
        }
    )

    return login_response.json()["access_token"]


def test_create_risk_profile():
    token = get_auth_token()

    response = client.post(
        "/api/v1/risk-profiles/",
        headers={
            "Authorization": f"Bearer {token}"
        },
        json={
            "max_loss_amount": 50000,
            "investment_horizon_years": 10
        }
    )

    assert response.status_code == 201

    data = response.json()

    assert data["max_loss_amount"] == 50000
    assert data["investment_horizon_years"] == 10
    assert data["calculated_risk_score"] is None


def test_get_risk_profile():
    token = get_auth_token()

    client.post(
        "/api/v1/risk-profiles/",
        headers={
            "Authorization": f"Bearer {token}"
        },
        json={
            "max_loss_amount": 50000,
            "investment_horizon_years": 10
        }
    )

    response = client.get(
        "/api/v1/risk-profiles/me",
        headers={
            "Authorization": f"Bearer {token}"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["max_loss_amount"] == 50000
    assert data["investment_horizon_years"] == 10


def test_risk_profile_without_token():
    response = client.get(
        "/api/v1/risk-profiles/me"
    )

    assert response.status_code == 401


def test_duplicate_risk_profile():
    token = get_auth_token()

    headers = {
        "Authorization": f"Bearer {token}"
    }

    client.post(
        "/api/v1/risk-profiles/",
        headers=headers,
        json={
            "max_loss_amount": 50000,
            "investment_horizon_years": 10
        }
    )

    response = client.post(
        "/api/v1/risk-profiles/",
        headers=headers,
        json={
            "max_loss_amount": 60000,
            "investment_horizon_years": 15
        }
    )

    assert response.status_code == 409


def test_invalid_max_loss_amount():
    token = get_auth_token()

    response = client.post(
        "/api/v1/risk-profiles/",
        headers={
            "Authorization": f"Bearer {token}"
        },
        json={
            "max_loss_amount": -5000,
            "investment_horizon_years": 10
        }
    )

    assert response.status_code == 422


def test_invalid_investment_horizon():
    token = get_auth_token()

    response = client.post(
        "/api/v1/risk-profiles/",
        headers={
            "Authorization": f"Bearer {token}"
        },
        json={
            "max_loss_amount": 50000,
            "investment_horizon_years": 0
        }
    )

    assert response.status_code == 422