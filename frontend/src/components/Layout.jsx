import { Link, NavLink, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext.jsx";

export default function Layout({ children }) {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  return (
    <div className="app-shell">
      <header className="topbar">
        <Link to="/" className="brand">
          <span className="brand-mark">o</span> optiwealth
        </Link>

        <nav className="nav-links" aria-label="Main navigation">
          {user ? (
            <>
              <NavLink to="/dashboard" className={({ isActive }) => (isActive ? "active" : "")}>
                Overview
              </NavLink>
              <NavLink to="/goals" className={({ isActive }) => (isActive ? "active" : "")}>
                Goals
              </NavLink>
              <NavLink to="/risk-profile" className={({ isActive }) => (isActive ? "active" : "")}>
                Risk Profile
              </NavLink>
              <div className="user-menu">
                <span className="user-email-chip" title={`Signed in as ${user.email}`}>
                  <span className="user-avatar-circle">{user.email.charAt(0).toUpperCase()}</span>
                  <span className="user-email-text">{user.email}</span>
                </span>
                <button
                  className="nav-logout"
                  onClick={() => {
                    logout();
                    navigate("/");
                  }}
                >
                  Log out
                </button>
              </div>
            </>
          ) : (
            <>
              <a href="#how-it-works">How It Works</a>
              <a href="#benefits">Benefits</a>
              <Link to="/login" className="nav-login">
                Log In
              </Link>
              <Link to="/register" className="button button-small">
                Get Started <span>↗</span>
              </Link>
            </>
          )}
        </nav>
      </header>

      <main>{children}</main>

      <footer className="footer">
        <div className="footer-left">
          <span className="brand footer-brand">
            <span className="brand-mark">o</span> optiwealth
          </span>
          <span className="footer-tagline">Smart Personal Portfolio &amp; Asset Optimization</span>
        </div>
        <div className="footer-center">
          <span className="footer-note">
            Investments in securities are subject to market risks. Always align allocations with your personal financial objectives.
          </span>
        </div>
        <div className="footer-right">
          <span className="footer-copy">© 2026 OptiWealth. All rights reserved.</span>
        </div>
      </footer>
    </div>
  );
}
