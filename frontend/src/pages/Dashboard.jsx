import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { useAuth } from "../context/AuthContext.jsx";
import { goalsApi, riskApi, holdingsApi, portfoliosApi } from "../services/api.js";
import Notice from "../components/Notice.jsx";

const CATEGORY_NAMES = {
  retirement: "Retirement",
  house_down_payment: "Home Down Payment",
  higher_education: "Higher Education",
  other: "Other Milestone",
};

const CATEGORY_ICONS = {
  retirement: "◷",
  house_down_payment: "⌂",
  higher_education: "🎓",
  other: "↗",
};

function formatMoney(amount) {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(Number(amount || 0));
}

function getRelativeTimeDescription(targetDateStr) {
  if (!targetDateStr) return "";
  const now = new Date();
  const target = new Date(`${targetDateStr}T00:00:00`);
  const diffMs = target.getTime() - now.getTime();
  const diffDays = Math.ceil(diffMs / (1000 * 60 * 60 * 24));

  if (diffDays <= 0) return "Due today or past target";
  if (diffDays < 30) return `${diffDays} day${diffDays === 1 ? "" : "s"} left`;

  const months = Math.floor(diffDays / 30.4375);
  const years = Math.floor(months / 12);
  const remMonths = months % 12;

  if (years > 0 && remMonths > 0) {
    return `${years}y ${remMonths}m runway`;
  }
  if (years > 0) {
    return `${years} year${years === 1 ? "" : "s"} runway`;
  }
  return `${months} month${months === 1 ? "" : "s"} runway`;
}

export default function Dashboard() {
  const { token, user } = useAuth();
  const [goals, setGoals] = useState([]);
  const [risk, setRisk] = useState(null);
  const [holdings, setHoldings] = useState([]);
  const [latestPortfolio, setLatestPortfolio] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError("");

    Promise.all([
      goalsApi.list(token).catch((e) => {
        throw new Error(`Unable to load goals: ${e.message}`);
      }),
      riskApi.getMine(token).catch((e) => {
        if (e.status === 404) return null;
        throw new Error(`Unable to load risk profile: ${e.message}`);
      }),
      holdingsApi.list(token).catch(() => []),
      portfoliosApi.list(token).catch(() => []),
    ])
      .then(([g, r, h, p]) => {
        if (active) {
          setGoals(Array.isArray(g) ? g : []);
          setRisk(r);
          setHoldings(Array.isArray(h) ? h : []);
          if (Array.isArray(p) && p.length > 0) {
            setLatestPortfolio(p[0]);
          }
        }
      })
      .catch((e) => {
        if (active) setError(e.message);
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
    };
  }, [token]);

  const totalTarget = goals.reduce((sum, goal) => sum + Number(goal.target_amount || 0), 0);
  const totalHoldingsInvested = holdings.reduce((sum, h) => sum + Number(h.invested_value || 0), 0);
  const userName = user?.email ? user.email.split("@")[0] : "Investor";

  // Find nearest upcoming goal
  const upcomingGoals = [...goals].sort(
    (a, b) => new Date(a.target_date) - new Date(b.target_date)
  );
  const nearestGoal = upcomingGoals[0];

  // Category breakdown
  const categoryCounts = goals.reduce((acc, goal) => {
    acc[goal.category] = (acc[goal.category] || 0) + 1;
    return acc;
  }, {});

  // Determine risk profile style
  let riskStyleName = "Not Set";
  if (risk) {
    const horizon = Number(risk.investment_horizon_years || 5);
    if (horizon <= 3) riskStyleName = "Conservative (Capital Preservation)";
    else if (horizon <= 8) riskStyleName = "Moderate (Balanced Growth)";
    else riskStyleName = "Aggressive (High Growth)";
  }

  return (
    <div className="dashboard-page section-wrap">
      <div className="dashboard-welcome">
        <div>
          <span className="eyebrow">YOUR FINANCIAL OVERVIEW</span>
          <h1>
            Good to see you, <em>{userName}.</em>
          </h1>
          <p className="muted">
            Here is a snapshot of your active financial goals and current investment preferences.
          </p>
        </div>
        <div className="dashboard-date-chip">
          <span className="date-dot" /> Portfolio Overview
        </div>
      </div>

      <Notice type="error" onClose={() => setError("")}>
        {error}
      </Notice>

      {loading ? (
        <div className="loading-card">
          <span className="spinner" /> Loading your financial overview…
        </div>
      ) : (
        <>
          {/* Main 4-Card Stats Grid */}
          <div className="stats-grid">
            <article className="stat-card stat-primary">
              <span className="stat-label">RECORDED HOLDINGS</span>
              <strong>{formatMoney(totalHoldingsInvested)}</strong>
              <span className="stat-caption">{holdings.length} asset(s) tracked in portfolio</span>
              <span className="stat-watermark">₹</span>
            </article>

            <article className="stat-card">
              <span className="stat-label">FINANCIAL GOALS</span>
              <strong>{goals.length.toString().padStart(2, "0")}</strong>
              <span className="stat-caption">Total Target: {formatMoney(totalTarget)}</span>
              <span className="stat-watermark">◎</span>
            </article>

            <article className="stat-card">
              <span className="stat-label">ML OPTIMIZED PORTFOLIO</span>
              <strong className="stat-word">
                {latestPortfolio ? formatMoney(latestPortfolio.investment_amount) : "Not Generated"}
              </strong>
              <span className="stat-caption">
                {latestPortfolio
                  ? `${latestPortfolio.expected_annual_return_pct}% Exp. Return · Sharpe ${Number(latestPortfolio.sharpe_ratio || 0).toFixed(2)}`
                  : "Run optimizer to get asset allocation"}
              </span>
              <span className="stat-watermark">↗</span>
            </article>

            <article className="stat-card">
              <span className="stat-label">RISK PROFILE</span>
              <strong className="stat-word">
                {risk ? (risk.calculated_risk_score ? `Score ${risk.calculated_risk_score}/10` : "Configured") : "Not Set"}
              </strong>
              <span className="stat-caption">
                {risk
                  ? `${risk.investment_horizon_years}-Yr Horizon · Max Loss ${formatMoney(risk.max_loss_amount)}`
                  : "Required to personalize asset allocation"}
              </span>
              <span className="stat-watermark">⌁</span>
            </article>
          </div>

          {/* Quick Actions Bar */}
          <div className="dashboard-actions-bar">
            <div className="action-buttons-group">
              <Link to="/portfolio" className="button button-small">
                Optimize Portfolio <span>↗</span>
              </Link>
              <Link to="/holdings" className="button button-small button-quiet">
                + Record Holding <span>+</span>
              </Link>
              <Link to="/goals" className="button button-small button-quiet">
                + Add Financial Goal <span>↗</span>
              </Link>
              <Link to="/risk-profile" className="button button-small button-quiet">
                {risk ? "Risk Preferences" : "Set Up Risk"} <span>→</span>
              </Link>
            </div>
            {goals.length > 0 && (
              <div className="quick-category-pills">
                {Object.entries(categoryCounts).map(([cat, count]) => (
                  <span key={cat} className="category-counter-pill">
                    {CATEGORY_ICONS[cat] || "•"} {CATEGORY_NAMES[cat] || cat}: <strong>{count}</strong>
                  </span>
                ))}
              </div>
            )}
          </div>

          <div className="dashboard-columns">
            {/* Left Column: Goals Overview */}
            <section className="content-card">
              <div className="card-heading">
                <div>
                  <span className="eyebrow">YOUR MILESTONES</span>
                  <h2>Financial goals in progress</h2>
                </div>
                <Link to="/goals" className="small-text-link">
                  Manage all goals ({goals.length}) →
                </Link>
              </div>

              {goals.length > 0 ? (
                <div className="goal-preview-list">
                  {goals.slice(0, 5).map((goal) => (
                    <div className="goal-preview" key={goal.id}>
                      <span className={`goal-avatar goal-avatar-${goal.category}`}>
                        {CATEGORY_ICONS[goal.category] || "↗"}
                      </span>
                      <div className="goal-preview-info">
                        <strong>{goal.name}</strong>
                        <span>
                          {CATEGORY_NAMES[goal.category] || goal.category.replaceAll("_", " ")} ·{" "}
                          <em>{getRelativeTimeDescription(goal.target_date)}</em>
                        </span>
                      </div>
                      <div className="goal-preview-amount-block">
                        <strong className="goal-amount">{formatMoney(goal.target_amount)}</strong>
                        <small className="goal-target-date">
                          Target:{" "}
                          {new Date(`${goal.target_date}T00:00:00`).toLocaleDateString("en-IN", {
                            day: "numeric",
                            month: "short",
                            year: "numeric",
                          })}
                        </small>
                      </div>
                    </div>
                  ))}
                  {goals.length > 5 && (
                    <div className="more-goals-note">
                      <Link to="/goals">+ {goals.length - 5} more goal(s) in your full plan →</Link>
                    </div>
                  )}
                </div>
              ) : (
                <div className="empty-state">
                  <span className="empty-icon">◎</span>
                  <h3>No goals created yet</h3>
                  <p>
                    Add your first milestone (like a home downpayment or retirement nest egg) to begin structuring your wealth plan.
                  </p>
                  <Link className="button button-small" to="/goals">
                    Add your first goal <span>↗</span>
                  </Link>
                </div>
              )}
            </section>

            {/* Right Column: Risk & Asset Diversification Guidance */}
            <div className="dashboard-side-stack">
              <section className="content-card next-step-card">
                <span className="eyebrow">YOUR RISK PROFILE</span>
                <span className="next-step-icon">{risk ? "✳" : "⌁"}</span>
                <h2>{risk ? "Risk preferences in place" : "Set your comfort zone"}</h2>
                <p>
                  {risk
                    ? `Configured for an investment runway of ${risk.investment_horizon_years} years with a maximum loss tolerance of ${formatMoney(risk.max_loss_amount)}.`
                    : "Specify how much market volatility you can comfortably tolerate so your portfolio asset allocation stays aligned with your goals."}
                </p>
                <Link className="text-link" to="/risk-profile">
                  {risk ? "Review risk profile" : "Set up risk profile"} <span>→</span>
                </Link>
                <div className="next-step-foot">
                  {risk ? riskStyleName : "Takes just 1 minute to complete"}
                </div>
              </section>

              {/* Multi-Asset Diversification Guidance */}
              <section className="content-card info-subcard">
                <span className="eyebrow">PORTFOLIO DIVERSIFICATION</span>
                <h3>Target Asset Classes</h3>
                <p className="tiny-muted">
                  Your goals are optimized across 4 balanced asset classes:
                </p>
                <div className="asset-class-chips">
                  <span className="asset-chip">📈 Equities (Nifty 50) — Long-term wealth generation</span>
                  <span className="asset-chip">💼 Mutual Funds — Broad sector diversification</span>
                  <span className="asset-chip">🪙 Sovereign Gold — Inflation hedge &amp; crisis store</span>
                  <span className="asset-chip">🏛️ Government Bonds — Capital security &amp; stable yield</span>
                </div>
                <div style={{ marginTop: "16px" }}>
                  <Link to="/portfolio" className="button button-small button-full">
                    Launch ML Optimizer for These Assets <span>↗</span>
                  </Link>
                </div>
              </section>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
