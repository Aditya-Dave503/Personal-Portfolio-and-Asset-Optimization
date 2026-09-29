from datetime import date, timedelta
from uuid import uuid4

from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def get_auth_headers():
    email = f"goals_{uuid4().hex}@example.com"
    client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": "TestPassword123"
        }
    )
    login_response = client.post(
        "/api/v1/auth/login",
        json={
            "email": email,
            "password": "TestPassword123"
        }
    )

    return {
        "Authorization": f"Bearer {login_response.json()['access_token']}"
    }


def goal_payload(**overrides):
    payload = {
        "name": "First home",
        "category": "house_down_payment",
        "target_amount": "2500000.00",
        "target_date": (date.today() + timedelta(days=365)).isoformat()
    }
    payload.update(overrides)
    return payload


def test_create_and_list_goals():
    headers = get_auth_headers()
    create_response = client.post(
        "/api/v1/goals/",
        headers=headers,
        json=goal_payload()
    )

    assert create_response.status_code == 201
    created = create_response.json()
    assert created["name"] == "First home"
    assert created["category"] == "house_down_payment"
    assert created["target_amount"] == "2500000.00"

    list_response = client.get("/api/v1/goals/", headers=headers)

    assert list_response.status_code == 200
    assert [goal["id"] for goal in list_response.json()] == [created["id"]]


def test_goal_can_be_read_updated_and_deleted():
    headers = get_auth_headers()
    create_response = client.post(
        "/api/v1/goals/",
        headers=headers,
        json=goal_payload()
    )
    goal_id = create_response.json()["id"]

    get_response = client.get(f"/api/v1/goals/{goal_id}", headers=headers)
    assert get_response.status_code == 200

    update_response = client.patch(
        f"/api/v1/goals/{goal_id}",
        headers=headers,
        json={"name": "Updated home fund"}
    )
    assert update_response.status_code == 200
    assert update_response.json()["name"] == "Updated home fund"

    delete_response = client.delete(
        f"/api/v1/goals/{goal_id}",
        headers=headers
    )
    assert delete_response.status_code == 204
    assert client.get(
        f"/api/v1/goals/{goal_id}",
        headers=headers
    ).status_code == 404


def test_goals_are_private_to_each_user():
    owner_headers = get_auth_headers()
    other_user_headers = get_auth_headers()
    create_response = client.post(
        "/api/v1/goals/",
        headers=owner_headers,
        json=goal_payload()
    )
    goal_id = create_response.json()["id"]

    assert client.get(
        f"/api/v1/goals/{goal_id}",
        headers=other_user_headers
    ).status_code == 404
    assert client.patch(
        f"/api/v1/goals/{goal_id}",
        headers=other_user_headers,
        json={"name": "Not yours"}
    ).status_code == 404
    assert client.delete(
        f"/api/v1/goals/{goal_id}",
        headers=other_user_headers
    ).status_code == 404
    assert client.get(
        "/api/v1/goals/",
        headers=other_user_headers
    ).json() == []


def test_goal_routes_require_authentication():
    assert client.get("/api/v1/goals/").status_code == 401

    response = client.post(
        "/api/v1/goals/",
        json=goal_payload()
    )
    assert response.status_code == 401


def test_invalid_goal_data_is_rejected():
    headers = get_auth_headers()

    invalid_amount = client.post(
        "/api/v1/goals/",
        headers=headers,
        json=goal_payload(target_amount="0")
    )
    assert invalid_amount.status_code == 422

    invalid_category = client.post(
        "/api/v1/goals/",
        headers=headers,
        json=goal_payload(category="vacation")
    )
    assert invalid_category.status_code == 422

    past_date = client.post(
        "/api/v1/goals/",
        headers=headers,
        json=goal_payload(target_date=(date.today() - timedelta(days=1)).isoformat())
    )
    assert past_date.status_code == 422

    empty_patch = client.patch(
        "/api/v1/goals/1",
        headers=headers,
        json={}
    )
    assert empty_patch.status_code == 422
