import { useEffect, useState, useMemo } from "react";
import { Link } from "react-router-dom";
import { riskApi } from "../services/api.js";
import { useAuth } from "../context/AuthContext.jsx";
import Notice from "../components/Notice.jsx";

function formatMoney(amount) {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(Number(amount || 0));
}

// Interactive investor style calculation
function calculateInvestorStyle(age, horizon, lossTolerancePct) {
  const ageFactor = Math.max(0, Math.min(1, (70 - age) / 45));
  const horizonFactor = Math.max(0, Math.min(1, horizon / 20));
  const lossFactor = Math.max(0, Math.min(1, (lossTolerancePct - 0.05) / 0.35));

  const compositeScore = 0.3 * ageFactor + 0.4 * horizonFactor + 0.3 * lossFactor;

  if (compositeScore < 0.38) {
    return {
      styleName: "Conservative",
      tagline: "Capital Preservation & Income",
      color: "#35627a",
      equityPct: 20,
      bondsPct: 60,
      goldPct: 20,
      description:
        "Prioritizes protecting your accumulated capital with a strong emphasis on sovereign bonds and gold, with a modest allocation to large-cap equities for inflation defense.",
    };
  } else if (compositeScore < 0.7) {
    return {
      styleName: "Balanced Growth",
      tagline: "Steady Compounding & Stability",
      color: "#2e6945",
      equityPct: 55,
      bondsPct: 30,
      goldPct: 15,
      description:
        "Strikes an optimal balance between equity growth and bond stability, creating reliable long-term wealth while cushioning against sudden market corrections.",
    };
  } else {
    return {
      styleName: "Aggressive Growth",
      tagline: "Maximum Long-Term Compounding",
      color: "#834426",
      equityPct: 75,
      bondsPct: 15,
      goldPct: 10,
      description:
        "Maximizes compounding by giving heavy weight to broad-market equities and growth mutual funds, utilizing your longer runway to ride out temporary volatility.",
    };
  }
}

const LOSS_PRESETS = [25000, 50000, 100000, 250000, 500000];
const HORIZON_PRESETS = [1, 3, 5, 7, 10, 15, 20, 25];

export default function RiskProfile() {
  const { token } = useAuth();
  const [profile, setProfile] = useState(null);
  const [maxLoss, setMaxLoss] = useState("");
  const [horizon, setHorizon] = useState("5");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  // Interactive guide state
  const [guideAge, setGuideAge] = useState(30);
  const [guideHorizon, setGuideHorizon] = useState(10);
  const [guideLossPct, setGuideLossPct] = useState(0.2);

  useEffect(() => {
    let active = true;
    setLoading(true);
    riskApi
      .getMine(token)
      .then((data) => {
        if (active) {
          setProfile(data);
          setMaxLoss(data.max_loss_amount ?? "");
          setHorizon(String(data.investment_horizon_years ?? "5"));
        }
      })
      .catch((e) => {
        if (active && e.status !== 404) {
          setError(e.message);
        }
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [token]);

  async function submit(event) {
    event.preventDefault();
    setError("");
    setSuccess("");

    const lossAmountNum = Number(maxLoss);
    const horizonNum = Number(horizon);

    if (!lossAmountNum || lossAmountNum <= 0) {
      setError("Please enter a valid loss tolerance amount greater than zero.");
      return;
    }
    if (!horizonNum || horizonNum < 1) {
      setError("Investment horizon must be at least 1 year.");
      return;
    }

    setBusy(true);
    try {
      const result = await riskApi.create(token, {
        max_loss_amount: lossAmountNum,
        investment_horizon_years: horizonNum,
      });
      setProfile(result);
      setSuccess("Your risk profile preferences have been successfully saved.");
    } catch (e) {
      if (e.status === 409) {
        setError(
          "Your risk profile is already configured. Your current active preferences are displayed below."
        );
      } else {
        setError(e.message);
      }
    } finally {
      setBusy(false);
    }
  }

  const guidedResult = useMemo(
    () => calculateInvestorStyle(guideAge, guideHorizon, guideLossPct),
    [guideAge, guideHorizon, guideLossPct]
  );

  if (loading) {
    return (
      <div className="center-state">
        <span className="spinner" /> Loading your risk preferences…
      </div>
    );
  }

  return (
    <div className="section-wrap inner-page risk-page">
      <div className="page-heading">
        <span className="eyebrow">PORTFOLIO COMFORT ZONE</span>
        <h1>
          Risk <em>profile.</em>
        </h1>
        <p>
          Investing with clarity starts by defining your boundaries. Tell us how much market volatility
          you can comfortably tolerate and your intended investment timeframe.
        </p>
      </div>

      <Notice type="error" onClose={() => setError("")}>
        {error}
      </Notice>
      <Notice type="success" onClose={() => setSuccess("")}>
        {success}
      </Notice>

      <div className="risk-content-grid">
        {/* Main Profile View / Creation Card */}
        {profile ? (
          <div className="profile-result content-card">
            <div className="result-top">
              <span className="result-icon">⌁</span>
              <div>
                <span className="eyebrow">YOUR PREFERENCES</span>
                <h2>Your profile is active</h2>
                <span className="badge-record-id">Active Configuration</span>
              </div>
            </div>

            <div className="profile-values">
              <div className="profile-stat-box">
                <span className="box-label">MAXIMUM COMFORTABLE LOSS</span>
                <strong>{formatMoney(profile.max_loss_amount)}</strong>
                <span className="box-caption">Maximum tolerable drawdown</span>
              </div>
              <div className="profile-stat-box">
                <span className="box-label">INVESTMENT TIME HORIZON</span>
                <strong>
                  {profile.investment_horizon_years} year{profile.investment_horizon_years === 1 ? "" : "s"}
                </strong>
                <span className="box-caption">Liquidity runway</span>
              </div>
            </div>

            {/* Investor Style Card */}
            <div className="score-explanation-card">
              <div className="score-header">
                <span className="score-badge">Strategy Tailored</span>
                <span className="score-tier-label">
                  {profile.investment_horizon_years <= 3
                    ? "Conservative Wealth Preservation"
                    : profile.investment_horizon_years <= 8
                    ? "Balanced Growth Portfolio"
                    : "Long-Term Appreciation"}
                </span>
              </div>
              <p className="score-text">
                Your portfolio allocations are personalized according to these limits.
                Assets will be diversified across Indian Equities (Nifty 50), Mutual Funds,
                Sovereign Gold, and Government Treasury Bonds to optimize growth within your risk budget.
              </p>
            </div>

            <div className="profile-actions-row">
              <Link to="/dashboard" className="button">
                Back to Dashboard <span>→</span>
              </Link>
              <Link to="/goals" className="button button-quiet">
                Manage Financial Goals <span>↗</span>
              </Link>
            </div>
          </div>
        ) : (
          <div className="form-card content-card">
            <div className="form-progress">
              <span className="progress-active" />
              <span className="progress-active" />
              <span />
            </div>

            <span className="eyebrow">RISK BOUNDARIES</span>
            <h2>Set your comfort zone</h2>
            <p className="muted">
              Specify your maximum acceptable loss and how long you plan to stay invested.
            </p>

            <form className="form-stack" onSubmit={submit}>
              <label>
                Maximum Acceptable Loss Amount
                <span className="label-help">
                  What rupee amount of temporary drawdown would you feel comfortable absorbing?
                </span>
                <div className="input-with-prefix">
                  <span>₹</span>
                  <input
                    type="number"
                    min="1"
                    step="0.01"
                    value={maxLoss}
                    onChange={(e) => setMaxLoss(e.target.value)}
                    placeholder="e.g. 50000"
                    required
                  />
                </div>
              </label>

              {/* Quick Presets */}
              <div className="presets-container">
                <span className="presets-label">Quick select:</span>
                <div className="preset-chips">
                  {LOSS_PRESETS.map((amt) => (
                    <button
                      type="button"
                      key={amt}
                      className={`preset-chip ${Number(maxLoss) === amt ? "preset-chip-active" : ""}`}
                      onClick={() => setMaxLoss(String(amt))}
                    >
                      {formatMoney(amt)}
                    </button>
                  ))}
                </div>
              </div>

              <label>
                Investment Time Horizon
                <span className="label-help">
                  How many years do you plan to remain invested before withdrawing?
                </span>
                <div className="select-wrap">
                  <select value={horizon} onChange={(e) => setHorizon(e.target.value)} required>
                    {HORIZON_PRESETS.map((yr) => (
                      <option key={yr} value={yr}>
                        {yr} {yr === 1 ? "year" : "years"} ({yr <= 3 ? "Short-term" : yr <= 7 ? "Medium-term" : "Long-term"})
                      </option>
                    ))}
                  </select>
                </div>
              </label>

              <div className="form-note">
                <span>ⓘ</span>
                <p>
                  <strong>Why this matters:</strong> Setting clear risk boundaries ensures you never
                  experience sleepless nights during normal market pullbacks, keeping your wealth plan
                  sustainable and stress-free.
                </p>
              </div>

              <button className="button button-full" disabled={busy} type="submit">
                {busy ? "Saving Profile…" : "Save Risk Profile"} <span>↗</span>
              </button>
            </form>
          </div>
        )}

        {/* Side Column: Interactive Strategy Explorer */}
        <section className="content-card simulator-card">
          <div className="simulator-header">
            <span className="eyebrow">PORTFOLIO EXPLORER</span>
            <h3>Explore Your Investor Style</h3>
            <p className="tiny-muted">
              Adjust the sliders below to see how different time horizons and loss tolerances shape
              your suggested asset allocation.
            </p>
          </div>

          <div className="simulator-controls">
            <div className="sim-slider-row">
              <div className="sim-slider-label">
                <span>Your Current Age:</span>
                <strong>{guideAge} years</strong>
              </div>
              <input
                type="range"
                min="21"
                max="70"
                value={guideAge}
                onChange={(e) => setGuideAge(Number(e.target.value))}
              />
              <span className="sim-slider-hint">Younger investors have longer earnings runways</span>
            </div>

            <div className="sim-slider-row">
              <div className="sim-slider-label">
                <span>Investment Horizon:</span>
                <strong>{guideHorizon} years</strong>
              </div>
              <input
                type="range"
                min="1"
                max="30"
                value={guideHorizon}
                onChange={(e) => setGuideHorizon(Number(e.target.value))}
              />
              <span className="sim-slider-hint">Longer horizons absorb market fluctuations better</span>
            </div>

            <div className="sim-slider-row">
              <div className="sim-slider-label">
                <span>Loss Tolerance:</span>
                <strong>{Math.round(guideLossPct * 100)}% of portfolio</strong>
              </div>
              <input
                type="range"
                min="5"
                max="40"
                step="5"
                value={Math.round(guideLossPct * 100)}
                onChange={(e) => setGuideLossPct(Number(e.target.value) / 100)}
              />
              <span className="sim-slider-hint">Your comfort level with temporary market dips</span>
            </div>
          </div>

          {/* Result Card */}
          <div className="simulator-result-box" style={{ borderColor: guidedResult.color }}>
            <div className="sim-score-circle" style={{ backgroundColor: guidedResult.color }}>
              <span>✓</span>
              <small>Match</small>
            </div>
            <div className="sim-score-info">
              <span className="sim-category" style={{ color: guidedResult.color }}>
                {guidedResult.styleName}
              </span>
              <span className="sim-continuous">{guidedResult.tagline}</span>
            </div>
          </div>

          {/* Asset Mix Breakdown */}
          <div className="asset-allocation-hint">
            <strong>Suggested Asset Allocation Mix:</strong>
            <div className="allocation-bar-visual">
              <div
                className="alloc-segment alloc-equity"
                style={{ width: `${guidedResult.equityPct}%` }}
                title={`Equities & Funds: ${guidedResult.equityPct}%`}
              >
                {guidedResult.equityPct}% Equities
              </div>
              <div
                className="alloc-segment alloc-bonds"
                style={{ width: `${guidedResult.bondsPct}%` }}
                title={`Government Bonds: ${guidedResult.bondsPct}%`}
              >
                {guidedResult.bondsPct}% Bonds
              </div>
              <div
                className="alloc-segment alloc-gold"
                style={{ width: `${guidedResult.goldPct}%` }}
                title={`Gold: ${guidedResult.goldPct}%`}
              >
                {guidedResult.goldPct}% Gold
              </div>
            </div>
            <p>{guidedResult.description}</p>
          </div>
        </section>
      </div>
    </div>
  );
}
