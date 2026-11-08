# OptiWealth — Frontend Documentation

## 1. System Overview

OptiWealth's frontend is a single-page web application (SPA) built with **React.js** and bundled using **Vite**. Its objective is to provide a clean, modern, and trustworthy wealth management interface through which users can manage life milestones, establish risk boundaries, and explore multi-asset allocation strategies aligned with the existing FastAPI backend.

The user interface is designed from an **end-user perspective**, focusing on personal finance terminology, portfolio clarity, and ease of use.

---

## 2. Technology Stack

| Layer / Concern | Technology | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **UI Library** | React.js | `^18.x` / latest | Component-driven declarative UI |
| **Build & Dev Tool** | Vite | `^6.x` / latest | Fast development server & optimized production rollup |
| **Routing** | React Router DOM | `^7.x` / latest | Client-side routing, protected routes, and navigation |
| **Styling** | Vanilla CSS3 (Custom Design System) | — | DM Sans + Manrope typography, responsive design, no bloated UI framework dependencies |
| **Data Fetching** | Native Fetch API | Native | Asynchronous HTTP communication with backend API |
| **State Management** | React Context (`AuthContext`) | Native | Centralized JWT session management and user identity |

---

## 3. Directory Structure

```
frontend/
├── index.html                   # HTML entry point with font preloads & meta tags
├── package.json                 # Project dependencies & npm scripts
├── package-lock.json            # Deterministic dependency tree
├── vite.config.js               # Vite configuration with @vitejs/plugin-react
├── .env.example                 # Environment variable template
├── .env                         # Local environment settings (gitignored)
├── README.md                    # Quickstart guide
└── src/
    ├── main.jsx                 # Application bootstrapping with Router & AuthProvider
    ├── App.jsx                  # Route definitions and layout encapsulation
    ├── styles.css               # Unified design system & responsive media queries
    ├── components/
    │   ├── Layout.jsx           # Application shell (Header, Navigation, Footer)
    │   ├── Notice.jsx           # Dismissible alert banners (success, error, info)
    │   └── ProtectedRoute.jsx   # Route guard enforcing active user session
    ├── context/
    │   └── AuthContext.jsx      # Authentication state, login/register/logout actions
    ├── pages/
    │   ├── Landing.jsx          # Public introduction, methodology, and benefits
    │   ├── Login.jsx            # Account authentication with password visibility toggle
    │   ├── Register.jsx         # Account registration with client validations
    │   ├── Dashboard.jsx        # Portfolio summary metrics, recent goals, and guidance
    │   ├── Goals.jsx            # Financial goals CRUD, filters, search, and runway countdown
    │   └── RiskProfile.jsx      # Risk parameters view/create & interactive strategy explorer
    └── services/
        └── api.js               # Centralized API request helper & endpoint methods
```

---

## 4. Architectural Modules & Pages

### 4.1 Authentication & Security (`AuthContext.jsx`, `Login.jsx`, `Register.jsx`, `ProtectedRoute.jsx`)
- **JWT Storage**: The bearer access token returned from `/api/v1/auth/login` is kept in `localStorage` under the key `optiwealth_access_token`.
- **Session Hydration**: On app startup, `AuthContext` calls `GET /api/v1/auth/me` with the stored token. If the token is valid, user metadata (`id`, `email`, `created_at`) is loaded into context. If invalid (401), the token is cleared and the user is redirected to `/login`.
- **Route Guard**: Any route wrapped in `<ProtectedRoute>` checks whether a valid user object exists. Unauthenticated visits redirect to `/login` preserving the intended destination in `location.state.from`.

### 4.2 Financial Goals Management (`Goals.jsx`)
- **Full CRUD Support**:
  - **Create**: Submits `{ name, category, target_amount, target_date }` to `POST /api/v1/goals/`. Validates that target amount is $> 0$ and target date is today or in the future.
  - **List**: Fetches user-scoped records from `GET /api/v1/goals/`.
  - **Edit (Update)**: Switches the form dynamically into edit mode, pre-fills fields, and issues a `PATCH /api/v1/goals/{id}` request.
  - **Delete**: Prompts for confirmation and calls `DELETE /api/v1/goals/{id}`.
- **Client Search & Filters**:
  - Filter pills by category: `All`, `Retirement`, `Home Down Payment`, `Higher Education`, `Other Milestone`.
  - Search box with immediate keyword matching.
  - Multi-criteria sorting: Target date (soonest/farthest), target amount (high/low), and alphabetical.
- **Runway Countdown**: Computes remaining time until the target date (e.g. `2y 4m runway`, `45 days left`).

### 4.3 Risk Profiling (`RiskProfile.jsx`)
- **Preference Persistence**: Communicates with `POST /api/v1/risk-profiles/` and `GET /api/v1/risk-profiles/me`.
- **Single-Profile Handling**: The current backend only permits one risk profile per user. If a profile already exists, the UI gracefully presents the saved maximum loss amount and investment horizon in a structured card.
- **Interactive Investor Style Explorer**: An educational interactive tool letting investors adjust Age, Runway, and Drawdown tolerance to visualize their investor category (Conservative, Balanced Growth, Aggressive Growth) and a proportional asset allocation bar (Equities, Bonds, Gold).

### 4.4 Dashboard Overview (`Dashboard.jsx`)
- Combines goals and risk profile data into an executive summary:
  - **Active Goals Count**
  - **Total Target Sum** (formatted in Indian Rupee currency `₹`)
  - **Next Milestone** (nearest target date with relative runway)
  - **Risk Profile Status**
- Displays recent goals with custom category avatars and timestamps.
- Summarizes the core asset classes of the OptiWealth portfolio.

---

## 5. Environment Variables & Setup

Create a `.env` file in `frontend/` (based on `.env.example`):

```env
VITE_API_URL=http://localhost:8000
```

### Installation & Execution

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```
The dev server launches on `http://localhost:5173`.
