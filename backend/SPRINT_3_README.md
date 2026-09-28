# Sprint 3 — Financial Goal Management

## 1. Sprint Overview

Sprint 3 extends the OptiWealth backend with authenticated financial goal
management. Users can record and maintain savings or investment goals, including
their category, target amount, and target date.

## 2. Sprint Goal

The goal of this sprint is to provide API support for:

- Creating a financial goal for the authenticated user
- Listing that user's goals in target-date order
- Retrieving, updating, and deleting an individual goal
- Validating goal data and protecting records from access by other users
- Persisting goals with an Alembic database migration

## 3. Scope

This sprint implements goal management only. Portfolio holdings, investment
recommendations, risk scoring, machine-learning models, and payment processing
are not part of this backend feature.

## 4. Features Implemented

### 4.1 Goal Data

Each goal stores:

| Field | Description |
| --- | --- |
| `id` | Database-generated goal identifier |
| `user_id` | Owner of the goal |
| `name` | Required name, up to 100 characters |
| `category` | `retirement`, `house_down_payment`, `higher_education`, or `other` |
| `target_amount` | Positive monetary amount, with up to 2 decimal places |
| `target_date` | Date today or later |

Goal records are stored in the `goals` table and reference `users.id`.

### 4.2 Authenticated API

All goal routes require the bearer access token obtained from
`POST /api/v1/auth/login`. Goal reads and mutations are scoped to the token's
user. Requests for another user's goal return `404 Goal not found`.

| Method | Endpoint | Description | Success |
| --- | --- | --- | --- |
| `POST` | `/api/v1/goals/` | Create a goal | `201 Created` |
| `GET` | `/api/v1/goals/` | List the authenticated user's goals, ordered by target date | `200 OK` |
| `GET` | `/api/v1/goals/{goal_id}` | Retrieve one of the user's goals | `200 OK` |
| `PATCH` | `/api/v1/goals/{goal_id}` | Update one or more goal fields | `200 OK` |
| `DELETE` | `/api/v1/goals/{goal_id}` | Delete one of the user's goals | `204 No Content` |

Invalid request data returns `422 Unprocessable Entity`. An empty update or an
update containing null goal fields is rejected. Requests without a valid token
are rejected by the existing authentication dependency.

### 4.3 Example

Create a goal using a bearer token:

```http
POST /api/v1/goals/
Authorization: Bearer <access_token>
Content-Type: application/json
```

```json
{
  "name": "First home",
  "category": "house_down_payment",
  "target_amount": "2500000.00",
  "target_date": "2030-12-31"
}
```

Partially update a goal:

```http
PATCH /api/v1/goals/1
Authorization: Bearer <access_token>
Content-Type: application/json
```

```json
{
  "target_amount": "2750000.00"
}
```

## 5. Database Migration

The migration `c6a2e8b4731f_create_goals_table` follows the existing risk-profile
migration. From the `backend` directory, apply it with:

```bash
alembic upgrade head
```

To roll back this migration:

```bash
alembic downgrade 39b1c6c6896c
```

## 6. Testing

From the `backend` directory, run the goal API tests with:

```bash
python -m pytest -q tests/test_goals.py
```

The tests cover goal creation and listing, read/update/delete behavior,
authentication, invalid goal data, and preventing users from accessing or
modifying another user's goals.
