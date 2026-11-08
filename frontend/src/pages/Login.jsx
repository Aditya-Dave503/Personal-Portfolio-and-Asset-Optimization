import { useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import Notice from "../components/Notice.jsx";
import { useAuth } from "../context/AuthContext.jsx";

export default function Login() {
  const location = useLocation();
  const navigate = useNavigate();
  const [email, setEmail] = useState(location.state?.email || "");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const { login } = useAuth();

  async function submit(event) {
    event.preventDefault();
    setError("");
    setBusy(true);

    try {
      await login(email.trim(), password);
      navigate(location.state?.from || "/dashboard", { replace: true });
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="auth-page">
      <div className="auth-aside login-aside">
        <span className="eyebrow light-eyebrow">Welcome Back</span>
        <h1>
          Pick up where
          <br />
          <em>you left off.</em>
        </h1>
        <p>
          Your financial goals and portfolio preferences are ready. Sign in to review your progress
          and keep your investments on track.
        </p>
        <div className="aside-quote">
          “An investment in knowledge pays the best interest.” — Benjamin Franklin
        </div>
        <span className="aside-decoration">✳</span>
      </div>

      <div className="auth-panel">
        <div className="auth-form-wrap">
          <span className="eyebrow">ACCOUNT LOGIN</span>
          <h2>Welcome back</h2>
          <p className="muted">Enter your email and password to access your dashboard.</p>

          <Notice type="error" onClose={() => setError("")}>
            {error}
          </Notice>

          <form onSubmit={submit} className="form-stack">
            <label>
              Email address
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                autoComplete="email"
                required
              />
            </label>

            <label>
              Password
              <div className="password-input-wrap">
                <input
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Your account password"
                  autoComplete="current-password"
                  required
                />
                <button
                  type="button"
                  className="password-toggle-btn"
                  onClick={() => setShowPassword(!showPassword)}
                  title={showPassword ? "Hide password" : "Show password"}
                >
                  {showPassword ? "👁️" : "🙈"}
                </button>
              </div>
            </label>

            <button className="button button-full" disabled={busy} type="submit">
              {busy ? "Signing in…" : "Log in"} <span>↗</span>
            </button>
          </form>

          <p className="auth-switch">
            New to OptiWealth? <Link to="/register">Create an account</Link>
          </p>
          <p className="tiny-note">
            🔒 Bank-grade encryption. Your personal credentials are safe and secure.
          </p>
        </div>
      </div>
    </section>
  );
}
