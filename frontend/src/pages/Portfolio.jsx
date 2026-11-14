import { useEffect, useState, useMemo } from "react";
import { Link, useSearchParams } from "react-router-dom";
import Notice from "../components/Notice.jsx";
import { portfoliosApi, riskApi } from "../services/api.js";
import { useAuth } from "../context/AuthContext.jsx";

const OPTIMIZATION_MODES = [
  {
    value: "max_sharpe",
    label: "Maximum Sharpe Ratio",
    badge: "Recommended",
    desc: "Optimizes risk-adjusted excess returns. Maximizes expected return per unit of volatility.",
  },
  {
    value: "min_variance",
    label: "Minimum Volatility",
    badge: "Defensive",
    desc: "Minimizes portfolio return variance. Prioritizes stability and drawdown defense.",
  },
  {
    value: "black_litterman",
    label: "Black-Litterman Model",
    badge: "Bayesian ML",
    desc: "Blends market equilibrium distributions with Machine Learning return forecasts.",
  },
];

const ASSET_META = {
  NIFTY_EQUITIES: {
    name: "Nifty 50 Large-Cap Equities",
    category: "Equity",
    icon: "📈",
    color: "#28557d",
    desc: "Core growth engine targeting India's top 50 enterprises.",
  },
  MUTUAL_FUNDS: {
    name: "Multi-Cap Active Mutual Funds",
    category: "Equities & Debt",
    icon: "💼",
    color: "#2d6b3c",
    desc: "Broad multi-sector diversification across market caps.",
  },
  MCX_GOLD: {
    name: "MCX Gold Commodity",
    category: "Commodities",
    icon: "🪙",
    color: "#b58231",
    desc: "Inflation defense and non-correlated crisis store of value.",
  },
  GOVT_10Y_GSEC: {
    name: "10-Year Government Securities",
    category: "Sovereign Debt",
    icon: "🏛️",
    color: "#5e3a75",
    desc: "Zero credit risk sovereign bonds providing baseline coupon yield.",
  },
};

function formatMoney(amount) {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(Number(amount || 0));
}

function formatPercent(val) {
  if (val === null || val === undefined || isNaN(val)) return "—";
  return `${Number(val).toFixed(2)}%`;
}

export default function Portfolio() {
  const { token } = useAuth();
  const [searchParams] = useSearchParams();

  // URL query params (e.g. from Goal "Optimize for Goal")
  const initialAmount = searchParams.get("amount") || "250000";
  const goalName = searchParams.get("goalName") || "";

  const [investmentAmount, setInvestmentAmount] = useState(initialAmount);
  const [optimizationMode, setOptimizationMode] = useState("max_sharpe");
  const [riskProfile, setRiskProfile] = useState(null);
  const [loadingProfile, setLoadingProfile] = useState(true);

  // Active Portfolio Result
  const [activePortfolio, setActivePortfolio] = useState(null);
  const [allocations, setAllocations] = useState([]);
  const [history, setHistory] = useState([]);

  // States
  const [optimizing, setOptimizing] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [serviceNotice, setServiceNotice] = useState("");

  // Load Risk Profile and Portfolio History
  useEffect(() => {
    let active = true;
    setLoadingProfile(true);

    riskApi
      .getMine(token)
      .then((profile) => {
        if (active) setRiskProfile(profile);
      })
      .catch((e) => {
        if (active && e.status !== 404) {
          setError(e.message);
        }
      })
      .finally(() => {
        if (active) setLoadingProfile(false);
      });

    loadHistory();

    return () => {
      active = false;
    };
  }, [token]);

  async function loadHistory() {
    setLoadingHistory(true);
    try {
      const data = await portfoliosApi.list(token);
      if (Array.isArray(data) && data.length > 0) {
        setHistory(data);
        // Select newest portfolio by default
        loadPortfolioDetails(data[0].id);
      }
    } catch (e) {
      if (e.status === 503) {
        setServiceNotice(
          "Portfolio backend service is in the process of being connected. You can still test parameters and preview model allocation mechanics below."
        );
      }
    } finally {
      setLoadingHistory(false);
    }
  }

  async function loadPortfolioDetails(portfolioId) {
    try {
      const [port, allocs] = await Promise.all([
        portfoliosApi.get(token, portfolioId),
        portfoliosApi.getAllocations(token, portfolioId).catch(() => []),
      ]);
      setActivePortfolio(port);
      setAllocations(Array.isArray(allocs) ? allocs : []);
    } catch (e) {
      // Fallback if detail call fails
    }
  }

  async function handleOptimize(e) {
    e.preventDefault();
    setError("");
    setSuccess("");

    const amountNum = Number(investmentAmount);
    if (!amountNum || amountNum <= 0) {
      setError("Please specify a valid positive investment amount.");
      return;
    }

    if (!riskProfile) {
      setError("You must have a configured Risk Profile before running the portfolio optimizer.");
      return;
    }

    const maxLoss = Number(riskProfile.max_loss_amount || 0);
    if (maxLoss > 0 && amountNum < maxLoss) {
      setError(
        `Investment amount (${formatMoney(amountNum)}) cannot be less than your maximum loss tolerance (${formatMoney(maxLoss)}). Please increase the amount.`
      );
      return;
    }

    setOptimizing(true);
    try {
      const result = await portfoliosApi.optimize(token, {
        investment_amount: amountNum,
        optimization_mode: optimizationMode,
      });

      setActivePortfolio(result);
      setSuccess("Portfolio optimization generated and persisted successfully!");

      // Fetch newly generated allocations
      if (result && result.id) {
        const allocData = await portfoliosApi.getAllocations(token, result.id).catch(() => []);
        setAllocations(Array.isArray(allocData) ? allocData : []);
      }
      await loadHistory();
    } catch (err) {
      if (err.status === 503) {
        setServiceNotice(
          "The backend portfolio service is currently returning HTTP 503 (service function names being bridged with endpoints). Below is the simulated allocation model based on your inputs."
        );
        // Provide mock demonstration view for immediate frontend validation
        generateSimulatedAllocations(amountNum, optimizationMode);
      } else {
        setError(err.message);
      }
    } finally {
      setOptimizing(false);
    }
  }

  function generateSimulatedAllocations(amount, mode) {
    const riskScore = riskProfile?.calculated_risk_score || 6;
    let weights = {
      NIFTY_EQUITIES: 35,
      MUTUAL_FUNDS: 30,
      MCX_GOLD: 15,
      GOVT_10Y_GSEC: 20,
    };

    if (mode === "min_variance" || riskScore <= 4) {
      weights = { NIFTY_EQUITIES: 15, MUTUAL_FUNDS: 20, MCX_GOLD: 25, GOVT_10Y_GSEC: 40 };
    } else if (riskScore >= 8) {
      weights = { NIFTY_EQUITIES: 50, MUTUAL_FUNDS: 30, MCX_GOLD: 10, GOVT_10Y_GSEC: 10 };
    }

    const simulatedAllocs = Object.entries(weights).map(([code, wt], idx) => ({
      id: idx + 1,
      asset_code: code,
      asset_name: ASSET_META[code].name,
      category: ASSET_META[code].category,
      risk_level: wt >= 35 ? "High" : wt >= 20 ? "Moderate" : "Low",
      weight_percent: wt,
      allocated_amount: (amount * wt) / 100,
      ml_expected_annual_return_pct: code === "NIFTY_EQUITIES" ? 14.5 : code === "MUTUAL_FUNDS" ? 13.2 : code === "MCX_GOLD" ? 9.8 : 7.1,
    }));

    setActivePortfolio({
      id: "preview",
      investment_amount: amount,
      optimization_mode: mode,
      risk_score: riskScore,
      expected_annual_return_pct: 12.84,
      expected_annual_volatility_pct: 10.15,
      sharpe_ratio: 1.26,
      value_at_risk_1m_95pct: 4.82,
      value_at_risk_1m_99pct: 6.91,
      conditional_var_1m_95pct: 5.74,
      var_99_constraint_status: "SATISFIED",
      generated_at: new Date().toISOString(),
    });
    setAllocations(simulatedAllocs);
  }

  return (
    <div className="section-wrap inner-page portfolio-page">
      <div className="page-heading">
        <span className="eyebrow">ASSET ALLOCATION ENGINE</span>
        <h1>
          Portfolio <em>optimization.</em>
        </h1>
        <p>
          Harness Machine Learning to allocate capital across Indian Equities, Mutual Funds,
          Gold, and Sovereign Debt tailored to your risk boundaries and investment horizon.
        </p>
      </div>

      <Notice type="error" onClose={() => setError("")}>
        {error}
      </Notice>
      <Notice type="success" onClose={() => setSuccess("")}>
        {success}
      </Notice>

      {serviceNotice && (
        <div className="warning-banner-card">
          <span className="warning-icon">ℹ</span>
          <div>
            <strong>Service Notice:</strong> {serviceNotice}
          </div>
        </div>
      )}

      {/* Pre-flight Risk Profile Warning */}
      {!loadingProfile && !riskProfile && (
        <div className="warning-banner-card">
          <span className="warning-icon">⚠️</span>
          <div>
            <strong>Risk Profile Required:</strong> You haven't set up your investment risk profile yet.
            The ML optimizer requires your horizon and loss tolerance to compute customized constraints.
            <div style={{ marginTop: "8px" }}>
              <Link to="/risk-profile" className="button button-small">
                Configure Risk Profile Now →
              </Link>
            </div>
          </div>
        </div>
      )}

      {goalName && (
        <div className="goal-target-banner">
          <span>🎯 Targeting Financial Goal:</span>
          <strong>{goalName}</strong>
          <small>({formatMoney(investmentAmount)})</small>
        </div>
      )}

      <div className="portfolio-grid-layout">
        {/* Left Column: Optimization Launcher Form */}
        <div className="content-card portfolio-launcher-card">
          <span className="eyebrow">CONFIGURATION</span>
          <h2>Optimize Capital</h2>
          <p className="muted">
            Choose your investment sum and select the mathematical optimization model.
          </p>

          <form className="form-stack" onSubmit={handleOptimize}>
            <label>
              Investment Amount (₹)
              <div className="input-with-prefix">
                <span>₹</span>
                <input
                  type="number"
                  step="1000"
                  min="1000"
                  value={investmentAmount}
                  onChange={(e) => setInvestmentAmount(e.target.value)}
                  placeholder="e.g. 500000"
                  required
                />
              </div>
              {riskProfile && (
                <span className="label-help">
                  Minimum recommended: {formatMoney(riskProfile.max_loss_amount)} (your maximum loss tolerance)
                </span>
              )}
            </label>

            <label>
              Optimization Methodology
              <div className="optimization-mode-picker">
                {OPTIMIZATION_MODES.map((mode) => (
                  <div
                    key={mode.value}
                    className={`mode-option-card ${optimizationMode === mode.value ? "mode-option-active" : ""}`}
                    onClick={() => setOptimizationMode(mode.value)}
                  >
                    <div className="mode-option-header">
                      <strong>{mode.label}</strong>
                      <span className="mode-badge">{mode.badge}</span>
                    </div>
                    <p className="tiny-muted">{mode.desc}</p>
                  </div>
                ))}
              </div>
            </label>

            <button
              className="button button-full"
              type="submit"
              disabled={optimizing || !riskProfile}
            >
              {optimizing ? "Running ML Models…" : "Generate Optimal Portfolio"} <span>↗</span>
            </button>
          </form>

          {riskProfile && (
            <div className="active-risk-mini-card">
              <span className="eyebrow">BOUNDARIES APPLIED</span>
              <div className="mini-risk-stats">
                <div>
                  <small>Horizon:</small>
                  <strong>{riskProfile.investment_horizon_years} Yrs</strong>
                </div>
                <div>
                  <small>Loss Limit:</small>
                  <strong>{formatMoney(riskProfile.max_loss_amount)}</strong>
                </div>
                <div>
                  <small>ML Risk Score:</small>
                  <strong>{riskProfile.calculated_risk_score || "Calibrated"}/10</strong>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Optimization Results & Allocations */}
        <div className="portfolio-results-stack">
          {activePortfolio ? (
            <>
              {/* Key ML Performance Metrics Banner */}
              <div className="portfolio-metrics-card content-card">
                <div className="card-heading">
                  <div>
                    <span className="eyebrow">ML OPTIMIZATION METRICS</span>
                    <h2>Risk &amp; Return Projections</h2>
                  </div>
                  <span className="opt-mode-tag">
                    {OPTIMIZATION_MODES.find((m) => m.value === activePortfolio.optimization_mode)?.label || activePortfolio.optimization_mode}
                  </span>
                </div>

                <div className="ml-metrics-grid">
                  <div className="metric-box">
                    <span className="metric-label">EXPECTED ANNUAL RETURN</span>
                    <strong className="metric-primary">
                      {formatPercent(activePortfolio.expected_annual_return_pct)}
                    </strong>
                    <span className="metric-sub">Forecasted aggregate yield</span>
                  </div>

                  <div className="metric-box">
                    <span className="metric-label">ANNUAL VOLATILITY</span>
                    <strong>{formatPercent(activePortfolio.expected_annual_volatility_pct)}</strong>
                    <span className="metric-sub">Standard deviation ($\sigma$)</span>
                  </div>

                  <div className="metric-box">
                    <span className="metric-label">SHARPE RATIO</span>
                    <strong className="metric-highlight">
                      {Number(activePortfolio.sharpe_ratio || 0).toFixed(2)}
                    </strong>
                    <span className="metric-sub">Excess return / risk</span>
                  </div>

                  <div className="metric-box">
                    <span className="metric-label">1-MONTH 95% VaR</span>
                    <strong>{formatPercent(activePortfolio.value_at_risk_1m_95pct)}</strong>
                    <span className="metric-sub">Max monthly loss bound</span>
                  </div>
                </div>

                {activePortfolio.var_99_constraint_status && (
                  <div className="var-status-pill">
                    <span>🛡️ VaR 99% Constraint:</span>
                    <strong>{activePortfolio.var_99_constraint_status}</strong>
                  </div>
                )}
              </div>

              {/* Asset Allocation Distribution Visualizer */}
              <div className="content-card">
                <div className="card-heading">
                  <div>
                    <span className="eyebrow">RECOMMENDED DISTRIBUTION</span>
                    <h2>Asset Class Weights</h2>
                  </div>
                  <span className="total-cap-badge">
                    Total: {formatMoney(activePortfolio.investment_amount)}
                  </span>
                </div>

                {/* Visual Stacked Multi-Color Progress Bar */}
                <div className="allocation-bar-visual-large">
                  {allocations.map((alloc) => {
                    const meta = ASSET_META[alloc.asset_code] || {
                      name: alloc.asset_code,
                      color: "#477451",
                    };
                    const wt = Number(alloc.weight_percent || 0);
                    if (wt <= 0) return null;
                    return (
                      <div
                        key={alloc.asset_code}
                        className="alloc-bar-segment"
                        style={{
                          width: `${wt}%`,
                          backgroundColor: meta.color,
                        }}
                        title={`${meta.name}: ${wt.toFixed(1)}% (${formatMoney(alloc.allocated_amount)})`}
                      >
                        {wt >= 12 && `${wt.toFixed(0)}%`}
                      </div>
                    );
                  })}
                </div>

                {/* Allocation Table Breakdown */}
                <div className="holdings-table-wrap" style={{ marginTop: "24px" }}>
                  <table className="holdings-table">
                    <thead>
                      <tr>
                        <th>Asset Class</th>
                        <th>Category</th>
                        <th>Weight</th>
                        <th>Target Allocation</th>
                        <th>Forecast Return</th>
                      </tr>
                    </thead>
                    <tbody>
                      {allocations.map((alloc) => {
                        const meta = ASSET_META[alloc.asset_code] || {
                          name: alloc.asset_code,
                          category: alloc.category || "Asset",
                          icon: "•",
                          color: "#477451",
                        };
                        const wt = Number(alloc.weight_percent || 0);
                        const amt = Number(alloc.allocated_amount || 0);
                        const ret = Number(alloc.ml_expected_annual_return_pct || 0);

                        return (
                          <tr key={alloc.asset_code}>
                            <td>
                              <div className="holding-name-cell">
                                <span className="holding-icon" style={{ color: meta.color }}>
                                  {meta.icon}
                                </span>
                                <div>
                                  <strong>{meta.name}</strong>
                                  <small className="tiny-muted">{alloc.asset_code}</small>
                                </div>
                              </div>
                            </td>
                            <td>
                              <span className="asset-badge badge-default">
                                {meta.category}
                              </span>
                            </td>
                            <td>
                              <div className="weight-display">
                                <span
                                  className="weight-color-dot"
                                  style={{ backgroundColor: meta.color }}
                                />
                                <strong>{wt.toFixed(1)}%</strong>
                              </div>
                            </td>
                            <td>
                              <strong className="holding-val-text">{formatMoney(amt)}</strong>
                            </td>
                            <td>
                              <span className="ret-badge">{formatPercent(ret)}</span>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          ) : (
            <div className="empty-state content-card">
              <span className="empty-icon">📈</span>
              <h3>No portfolio generated yet</h3>
              <p>
                Configure your capital amount and select an optimization mode on the left to generate
                your personalized asset distribution.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
