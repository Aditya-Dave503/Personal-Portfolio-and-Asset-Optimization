# OptiWealth — Backend Integration & Frontend Handoff Guide

**Author**: Frontend Engineering (Marm)  
**Target Audience**: Backend Development Team & Project Lead  
**Scope**: Backend configuration updates, API contract alignment, and future enhancement roadmap.

---

## 1. Summary of Changes Made to Backend Files

Only two non-breaking configuration changes were introduced to support frontend communication:

### 1.1 CORS Middleware in `backend/main.py`
* **Why it was changed**: When the React application runs on `http://localhost:5173`, modern web browsers enforce Cross-Origin Resource Sharing (CORS) security. Without explicit CORS headers from FastAPI, all browser-based `fetch()` requests are rejected.
* **Implementation added**:
  ```python
  from fastapi.middleware.cors import CORSMiddleware

  app.add_middleware(
      CORSMiddleware,
      allow_origins=[
          "http://localhost:5173",
          "http://127.0.0.1:5173",
          "http://localhost:3000",
          "http://127.0.0.1:3000",
          "*"
      ],
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"],
  )
  ```

### 1.2 Environment Template in `backend/.env.example`
* **Why it was changed**: Previously `backend/.env.example` was empty (0 bytes). Since `backend/core/config.py` uses `pydantic-settings` requiring `DATABASE_URL` and `JWT_SECRET_KEY`, missing environment files cause Pydantic validation crashes on startup.
* **Content added**:
  ```env
  DATABASE_URL=postgresql://postgres:postgres@localhost:5432/optiwealth
  JWT_SECRET_KEY=dev_secret_key_change_in_production
  JWT_ALGORITHM=HS256
  JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
  ```

---

## 2. API Endpoints Consumed by the Frontend

The React frontend integrates with the existing FastAPI endpoints without requiring any route changes:

### 2.1 Authentication & Identity (`/api/v1/auth`)
| Method | Endpoint | Headers | Request Body | Expected Success |
| :--- | :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/auth/register` | `Content-Type: application/json` | `{ "email": "...", "password": "..." }` | `201 Created` → User object |
| `POST` | `/api/v1/auth/login` | `Content-Type: application/json` | `{ "email": "...", "password": "..." }` | `200 OK` → `{ "access_token": "...", "token_type": "bearer" }` |
| `GET` | `/api/v1/auth/me` | `Authorization: Bearer <token>` | *None* | `200 OK` → `{ "id": 1, "email": "...", "created_at": "..." }` |

### 2.2 Financial Goals CRUD (`/api/v1/goals`)
| Method | Endpoint | Headers | Request Body | Expected Success |
| :--- | :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/goals/` | `Authorization: Bearer <token>` | `{ "name": "...", "category": "...", "target_amount": 2500000.00, "target_date": "2028-12-31" }` | `201 Created` → Goal object |
| `GET` | `/api/v1/goals/` | `Authorization: Bearer <token>` | *None* | `200 OK` → `list[GoalResponse]` (sorted by target date) |
| `GET` | `/api/v1/goals/{id}` | `Authorization: Bearer <token>` | *None* | `200 OK` → Single goal object |
| `PATCH` | `/api/v1/goals/{id}` | `Authorization: Bearer <token>` | `{ "name": "...", "target_amount": "..." }` (partial fields) | `200 OK` → Updated goal object |
| `DELETE` | `/api/v1/goals/{id}` | `Authorization: Bearer <token>` | *None* | `204 No Content` |

*Category choices*: `"retirement"`, `"house_down_payment"`, `"higher_education"`, `"other"`.

### 2.3 Risk Profiles (`/api/v1/risk-profiles`)
| Method | Endpoint | Headers | Request Body | Expected Success |
| :--- | :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/risk-profiles/` | `Authorization: Bearer <token>` | `{ "max_loss_amount": 50000.0, "investment_horizon_years": 5 }` | `201 Created` → Risk profile object |
| `GET` | `/api/v1/risk-profiles/me` | `Authorization: Bearer <token>` | *None* | `200 OK` → `{ "id": 1, "user_id": 1, "calculated_risk_score": float, "max_loss_amount": float, "investment_horizon_years": int }` *(or 404 if not yet created)* |

---

## 3. Potential Issues & Recommended Backend Enhancements

Below are minor suggestions to help the backend team streamline the integration:

### 3.1 Support Updating an Existing Risk Profile (`PATCH /api/v1/risk-profiles/me`)
* **Current Behavior**: If a user submits `POST /api/v1/risk-profiles/` when a profile already exists, the endpoint returns `409 Conflict: Risk profile already exists`.
* **Recommendation**: Add a `PATCH /api/v1/risk-profiles/me` endpoint allowing users to update their `max_loss_amount` and `investment_horizon_years` as their life circumstances change.

### 3.2 Connecting the ML Risk Scoring Model to the Database
* **Current Behavior**: In `backend/services/risk_profile_service.py`, `calculated_risk_score` is stored as `None`.
* **Opportunity**: The ML team already implemented `get_calculated_risk_score()` in `backend/ml/risk_profiler.py`. You can import and invoke it inside `create_risk_profile()`:
  ```python
  from ml.risk_profiler import get_calculated_risk_score

  score = get_calculated_risk_score(
      investment_horizon_years=profile_data.investment_horizon_years,
      max_loss_amount=float(profile_data.max_loss_amount)
  )
  profile = RiskProfile(
      user_id=user_id,
      max_loss_amount=profile_data.max_loss_amount,
      investment_horizon_years=profile_data.investment_horizon_years,
      calculated_risk_score=score
  )
  ```
  This will automatically populate the risk score in PostgreSQL for the user!

### 3.3 Root Directory Uvicorn Execution
* In `backend/core/config.py`, `env_file=".env"` searches for `.env` relative to the current working directory. If a developer runs `uvicorn backend.main:app` from the project root rather than `cd backend`, Pydantic may not find `backend/.env`.
* Consider pointing `env_file` to the absolute directory of `config.py` using `Path(__file__).resolve().parent.parent / ".env"`.

---

## 4. Verification & Testing

To test integration between frontend and backend:
1. Ensure PostgreSQL is running locally and database `optiwealth` exists.
2. In `backend/.env`: set your database password.
3. Run migrations: `alembic upgrade head`.
4. Start backend: `uvicorn main:app --reload --port 8000`.
5. Start frontend: `npm run dev` in `frontend/`.
6. Open `http://localhost:5173`, create a user, log in, create a goal, and submit risk preferences.
