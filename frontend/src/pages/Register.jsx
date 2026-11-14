
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import Notice from "../components/Notice.jsx";
import { useAuth } from "../context/AuthContext.jsx";

export default function Register() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const { register } = useAuth();
  const navigate = useNavigate();

  async function submit(event) {
    event.preventDefault();
    setError("");
    setSuccess("");

    if (password !== confirm) {
      setError("Your passwords do not match.");
      return;
    }

    if (password.length < 8) {
      setError("For your security, your password must be at least 8 characters long.");
      return;
    }

    setBusy(true);

    try {
      await register(email.trim(), password);
      setSuccess("Account successfully created! Taking you to login…");

      setTimeout(() => {
        navigate("/login");
      }, 1000);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="auth-page">
      <div className="auth-aside">
        <span className="eyebrow light-eyebrow">Smart Wealth Planning</span>
        <h1>
          Your financial goals
          <br />
          <em>deserve a plan.</em>
        </h1>
        <p>
          Create your OptiWealth account to organize life milestones, establish risk boundaries,
          and build a balanced personal portfolio.
        </p>
        <div className="aside-quote">
          “Small, disciplined steps accumulate into life-changing financial security.”
        </div>
        <span className="aside-decoration">✳</span>
      </div>

      <div className="auth-panel">
        <div className="auth-form-wrap">
          <span className="eyebrow">GET STARTED FREE</span>
          <h2>Create your account</h2>
          <p className="muted">
            Enter your email and create a password to begin planning.
          </p>

          <Notice type="error" onClose={() => setError("")}>
            {error}
          </Notice>

          <Notice type="success" onClose={() => setSuccess("")}>
            {success}
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
                  placeholder="At least 8 characters"
                  autoComplete="new-password"
                  required
                  minLength={8}
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

            <label>
              Confirm Password
              <input
                type={showPassword ? "text" : "password"}
                value={confirm}
                onChange={(e) => setConfirm(e.target.value)}
                placeholder="Re-enter your password"
                autoComplete="new-password"
                required
                minLength={8}
              />
            </label>

            <button className="button button-full" disabled={busy} type="submit">
              {busy ? "Creating account…" : "Create Account"} <span>↗</span>
            </button>
          </form>

          <p className="auth-switch">
            Already have an account? <Link to="/login">Log in</Link>
          </p>

          <p className="tiny-note">
            🔒 Bank-grade security. Your data is private, protected, and never shared.
          </p>
        </div>
      </div>
    </section>
  );
}
