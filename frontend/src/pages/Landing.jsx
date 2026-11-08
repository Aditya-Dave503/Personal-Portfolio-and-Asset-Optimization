import { Link } from "react-router-dom";
import { useAuth } from "../context/AuthContext.jsx";

const steps = [
  {
    step: "01",
    icon: "🎯",
    title: "Define Your Financial Milestones",
    text: "Set tangible targets for the moments that matter most—whether it is buying a home, funding higher education, or securing a comfortable retirement.",
  },
  {
    step: "02",
    icon: "⚖️",
    title: "Discover Your Risk Comfort Zone",
    text: "Quantify your real risk tolerance and investment runway. We help you invest with confidence without taking uncompensated market risks.",
  },
  {
    step: "03",
    icon: "📈",
    title: "Intelligent Multi-Asset Balance",
    text: "Achieve true diversification by balancing high-growth equities and mutual funds with defensive anchors like Sovereign Gold and Government Treasury Bonds.",
  },
];

const pillars = [
  {
    icon: "🛡️",
    title: "Capital Protection",
    text: "Anchor your wealth during turbulent market cycles with sovereign debt securities and inflation-hedging precious metals.",
  },
  {
    icon: "🚀",
    title: "Long-Term Growth",
    text: "Harness the compounding power of leading Indian companies and diversified equity funds tailored to your time horizon.",
  },
  {
    icon: "⏳",
    title: "Timeline Clarity",
    text: "Every goal receives a dedicated runway countdown, helping you stay disciplined and on track for every life milestone.",
  },
];

export default function Landing() {
  const { user } = useAuth();

  return (
    <div className="landing">
      {/* Hero Section */}
      <section className="hero section-wrap">
        <div className="hero-copy">
          <div className="eyebrow">
            <span className="eyebrow-dot" /> Intelligent Wealth &amp; Goal Planning
          </div>
          <h1>
            Make your money
            <br />
            mean <em>more.</em>
          </h1>
          <p className="hero-lede">
            Bring your financial future into sharp focus. OptiWealth helps you articulate meaningful goals,
            measure your comfort with risk, and align your savings across a balanced multi-asset portfolio.
          </p>
          <div className="hero-actions">
            {user ? (
              <Link className="button" to="/dashboard">
                Go to Dashboard <span>→</span>
              </Link>
            ) : (
              <>
                <Link className="button" to="/register">
                  Start Planning Today <span>↗</span>
                </Link>
                <Link className="text-link" to="/login">
                  Sign In to Account <span>→</span>
                </Link>
              </>
            )}
          </div>
          <div className="trust-line">
            <span className="trust-icon">✓</span> Bank-Grade Data Protection · Goal-Oriented Investing · Multi-Asset Balance
          </div>
        </div>

        <div className="hero-art" aria-label="Illustration of portfolio growth">
          <div className="art-orbit orbit-one" />
          <div className="art-orbit orbit-two" />
          <div className="art-card art-card-top">
            <span className="mini-label">PORTFOLIO DIVERSIFICATION</span>
            <span className="art-number">
              4 <small>Core Asset Classes</small>
            </span>
            <span className="mini-line">
              <i />
            </span>
          </div>
          <div className="art-chart">
            <div className="chart-heading">
              <span>Goal Growth Projection</span>
              <span className="chart-pill">Optimized Path ↗</span>
            </div>
            <svg viewBox="0 0 360 180" role="img" aria-label="Decorative rising wealth trajectory">
              <path
                d="M8 145 H352 M8 105 H352 M8 65 H352 M8 25 H352"
                stroke="#dfe7db"
                strokeDasharray="3 6"
                fill="none"
              />
              <path
                d="M8 148 C48 138 52 118 91 125 S142 95 174 103 S230 63 260 75 S315 28 352 18 L352 170 L8 170 Z"
                fill="url(#chartFill)"
                opacity=".8"
              />
              <path
                d="M8 148 C48 138 52 118 91 125 S142 95 174 103 S230 63 260 75 S315 28 352 18"
                stroke="#3e795a"
                strokeWidth="3.5"
                strokeLinecap="round"
                fill="none"
              />
              <circle cx="352" cy="18" r="6" fill="#3e795a" stroke="#fff" strokeWidth="3" />
              <defs>
                <linearGradient id="chartFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#a9d6a5" />
                  <stop offset="100%" stopColor="#eff6e9" stopOpacity=".15" />
                </linearGradient>
              </defs>
            </svg>
            <div className="chart-foot">
              <span>SAFETY ANCHOR (BONDS &amp; GOLD)</span>
              <span>GROWTH ENGINE (EQUITIES &amp; FUNDS)</span>
            </div>
          </div>
          <div className="art-card art-card-bottom">
            <span className="art-spark">✳</span>
            <div>
              <strong>Personalized Allocation</strong>
              <small>Calibrated to your horizon</small>
            </div>
          </div>
          <span className="art-star star-one">✳</span>
          <span className="art-star star-two">✳</span>
        </div>
      </section>

      {/* How it Works Section */}
      <section id="how-it-works" className="features-section">
        <div className="section-wrap">
          <div className="section-intro">
            <span className="eyebrow">A SIMPLE, PROVEN METHODOLOGY</span>
            <h2>
              Take the stress out of investing.
              <br />
              <em>Focus on what you want to achieve.</em>
            </h2>
            <p>
              Successful investing is not about predicting market news. It is about aligning your portfolio
              with your personal life goals and risk boundaries.
            </p>
          </div>
          <div className="feature-grid">
            {steps.map((s) => (
              <article className="feature-card" key={s.title}>
                <span className="feature-index">{s.step}</span>
                <span className="feature-icon">{s.icon}</span>
                <h3>{s.title}</h3>
                <p>{s.text}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      {/* Benefits / Pillars Section */}
      <section id="benefits" className="benefits-section section-wrap">
        <div className="section-intro">
          <span className="eyebrow">DESIGNED FOR DISCIPLINED INVESTORS</span>
          <h2>
            Built around your life.
            <br />
            <em>Not market noise.</em>
          </h2>
          <p>
            Whether planning 3 years or 30 years ahead, OptiWealth provides the structure and clarity
            you need to build and preserve wealth.
          </p>
        </div>
        <div className="benefits-grid">
          {pillars.map((p) => (
            <div className="benefit-card content-card" key={p.title}>
              <span className="benefit-icon">{p.icon}</span>
              <h3>{p.title}</h3>
              <p>{p.text}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Closing CTA */}
      <section className="closing-cta section-wrap">
        <div>
          <span className="eyebrow">START BUILDING TODAY</span>
          <h2>
            Give your wealth direction.
            <br />
            <em>Create your plan now.</em>
          </h2>
        </div>
        {user ? (
          <Link to="/dashboard" className="button">
            Go to Your Dashboard <span>→</span>
          </Link>
        ) : (
          <Link to="/register" className="button">
            Get Started Free <span>↗</span>
          </Link>
        )}
      </section>
    </div>
  );
}
