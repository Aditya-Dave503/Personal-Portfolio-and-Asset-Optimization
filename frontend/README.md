# OptiWealth — React.js Frontend

A modern React.js + Vite frontend for **OptiWealth**, designed to interface with the FastAPI backend and PostgreSQL database.

---

## 🚀 Features Implemented (Aligned with Backend Current State)

### 1. Authentication & Session Security (Sprint 2)
- **User Registration** (`POST /api/v1/auth/register`): Email validation, minimum 8 characters password check, password confirmation.
- **User Login** (`POST /api/v1/auth/login`): JWT Bearer token generation.
- **Session Verification** (`GET /api/v1/auth/me`): Validates session, loads user identity, automatic token expiration handling.
- **Real-Time Backend Status**: Live pill indicator in the topbar showing API connectivity (`API Live` / `API Offline`) with health checks against `GET /health`.

### 2. Financial Goal Management CRUD (Sprint 3)
- **Create Goal** (`POST /api/v1/goals/`): Target monetary amount (₹ INR), horizon date validation (must be today or future), category assignment (`retirement`, `house_down_payment`, `higher_education`, `other`).
- **List Goals** (`GET /api/v1/goals/`): Fetches goals in target-date order from database.
- **Edit Goal** (`PATCH /api/v1/goals/{goal_id}`): Dynamic form pre-filling, validation, and update.
- **Delete Goal** (`DELETE /api/v1/goals/{goal_id}`): Confirmation modal and instant deletion.
- **Search, Filter & Sort**: Category filter tabs, keyword search bar, and multi-criteria sorting (target date, target amount, alphabetical).
- **Runway & Milestones**: Automatic relative runway countdown tags (e.g. `2y 4m runway`, `45 days left`).

### 3. Psychometric Risk Profiling
- **Record Risk Preferences** (`POST /api/v1/risk-profiles/`): Maximum acceptable drawdown amount and investment horizon in years.
- **Retrieve Profile** (`GET /api/v1/risk-profiles/me`): Displays active saved risk profile parameters.
- **Behavioral Risk Simulator**: Interactive simulator demonstrating how age, runway, and drawdown tolerance map to continuous risk budget score ($R_u \in [0, 1]$) and discrete scale (1-10) using Gradient Boosting lifecycle rules.

### 4. Interactive Dashboard
- Stat metric cards: Total Goals count, Total Target Amount in ₹ INR, Nearest Milestone, Risk Profile status.
- Recent active goals preview with category avatars and target completion dates.
- Direct quick actions for goal creation and risk configuration.

---

## 🛠 Prerequisites

- **Node.js**: v18+ (tested on v24)
- **Python**: 3.10+
- **PostgreSQL**: Local instance running on port 5432 with database `optiwealth`

---

## 💻 Running the Application

### 1. Start the FastAPI Backend

From the repository root:

```bash
cd backend

# Ensure dependencies are installed
pip install -r requirements.txt

# Configure your .env file (copy from .env.example)
# DATABASE_URL=postgresql://postgres:postgres@localhost:5432/optiwealth
# JWT_SECRET_KEY=dev_secret_key_change_in_production

# Apply database migrations
alembic upgrade head

# Launch FastAPI server with live reload
uvicorn main:app --reload --port 8000
```

Verify backend is healthy at: [http://localhost:8000/health](http://localhost:8000/health) or check Swagger docs at [http://localhost:8000/docs](http://localhost:8000/docs).

### 2. Start the React Frontend

Open a new terminal window:

```bash
cd frontend

# Install dependencies (already prepared)
npm install

# Start Vite development server
npm run dev
```

Open your browser at: **[http://localhost:5173](http://localhost:5173)**

---

## 🏗 Build Check

To verify client build compilation:

```bash
cd frontend
npm run build
```
