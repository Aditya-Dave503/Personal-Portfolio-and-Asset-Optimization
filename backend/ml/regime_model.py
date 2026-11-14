"""
Macroeconomic Regime-Switching Model (Gaussian HMM / Markov State Transitions)

Detects latent financial market states (e.g. Low-Volatility Bull vs High-Volatility Panic)
using Gaussian Expectation-Maximization and Baum-Welch forward-backward filtering.
Dynamically tilts portfolio risk aversion and asset bounds based on regime probabilities.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class RegimeState:
    state_id: int
    name: str
    mean_return_annualized: float
    volatility_annualized: float
    current_probability: float


@dataclass
class RegimeModelOutput:
    states: List[RegimeState]
    transition_matrix: np.ndarray
    active_regime_id: int
    active_regime_name: str
    panic_probability: float
    effective_risk_aversion: float
    historical_regimes: pd.Series


class GaussianHMM2State:
    """
    Robust 2-State Gaussian Hidden Markov Model for financial returns and volatility.
    State 0: Low-Volatility Bull (Calm Growth)
    State 1: High-Volatility Crisis (Turbulence / Panic)
    """

    def __init__(self, max_iter: int = 50, tol: float = 1e-4):
        self.max_iter = max_iter
        self.tol = tol
        self.n_states = 2
        # Transition matrix A: A[i, j] = P(S_t = j | S_{t-1} = i)
        self.A = np.array([[0.95, 0.05], [0.15, 0.85]], dtype=float)
        # Initial state distribution
        self.pi = np.array([0.80, 0.20], dtype=float)
        # Emission parameters (mean and std of daily returns)
        self.means = np.array([0.0008, -0.0005], dtype=float)
        self.stds = np.array([0.008, 0.022], dtype=float)

    def _emission_probs(self, x: np.ndarray) -> np.ndarray:
        """Compute Gaussian emission probability density matrix (T x n_states)."""
        T = len(x)
        B = np.zeros((T, self.n_states), dtype=float)
        for k in range(self.n_states):
            # Evaluate Gaussian pdf with variance floor to avoid division by zero
            var = max(self.stds[k] ** 2, 1e-8)
            diff = x - self.means[k]
            B[:, k] = np.exp(-0.5 * (diff ** 2) / var) / np.sqrt(2.0 * np.pi * var)
        # Floor emission density
        return np.maximum(B, 1e-12)

    def fit(self, returns: np.ndarray) -> "GaussianHMM2State":
        """Fit HMM parameters using Baum-Welch Expectation-Maximization."""
        x = np.asarray(returns, dtype=float).ravel()
        T = len(x)
        if T < 50:
            raise ValueError("At least 50 observations required to fit Regime HMM.")

        # Heuristic initial clustering based on 21-day rolling volatility
        vol_threshold = np.percentile(np.abs(x), 70)
        calm_mask = np.abs(x) <= vol_threshold
        panic_mask = ~calm_mask

        self.means[0] = float(np.mean(x[calm_mask])) if np.any(calm_mask) else 0.0005
        self.means[1] = float(np.mean(x[panic_mask])) if np.any(panic_mask) else -0.0010
        self.stds[0] = float(max(np.std(x[calm_mask]), 1e-4)) if np.any(calm_mask) else 0.008
        self.stds[1] = float(max(np.std(x[panic_mask]), 1e-4)) if np.any(panic_mask) else 0.020

        # Ensure State 0 is strictly lower volatility than State 1
        if self.stds[0] > self.stds[1]:
            self.means = self.means[::-1]
            self.stds = self.stds[::-1]

        prev_ll = -np.inf

        for iteration in range(self.max_iter):
            B = self._emission_probs(x)

            # 1. Forward Pass with scaling to prevent underflow
            alpha = np.zeros((T, self.n_states), dtype=float)
            c = np.zeros(T, dtype=float)

            alpha[0] = self.pi * B[0]
            c[0] = np.sum(alpha[0]) or 1e-12
            alpha[0] /= c[0]

            for t in range(1, T):
                alpha[t] = (alpha[t - 1] @ self.A) * B[t]
                c[t] = np.sum(alpha[t]) or 1e-12
                alpha[t] /= c[t]

            # 2. Backward Pass
            beta = np.zeros((T, self.n_states), dtype=float)
            beta[-1] = 1.0 / c[-1]

            for t in range(T - 2, -1, -1):
                beta[t] = (self.A @ (B[t + 1] * beta[t + 1])) / c[t]

            # 3. Smoothed Posterior Probabilities (E-Step)
            gamma = alpha * beta
            gamma_sum = np.sum(gamma, axis=1, keepdims=True)
            gamma = np.where(gamma_sum > 0, gamma / gamma_sum, 0.5)

            # 4. Two-slice probabilities xi
            xi = np.zeros((T - 1, self.n_states, self.n_states), dtype=float)
            for t in range(T - 1):
                num = np.outer(alpha[t], B[t + 1] * beta[t + 1]) * self.A
                denom = np.sum(num) or 1e-12
                xi[t] = num / denom

            # Log-Likelihood
            log_ll = float(np.sum(np.log(c + 1e-12)))
            if abs(log_ll - prev_ll) < self.tol:
                break
            prev_ll = log_ll

            # 5. M-Step: Update Parameters
            self.pi = np.clip(gamma[0], 1e-4, 1.0)
            self.pi /= np.sum(self.pi)

            xi_sum = np.sum(xi, axis=0)
            gamma_sum_t = np.sum(gamma[:-1], axis=0, keepdims=True).T
            gamma_sum_t = np.maximum(gamma_sum_t, 1e-12)
            self.A = np.clip(xi_sum / gamma_sum_t, 1e-4, 1.0)
            self.A /= np.sum(self.A, axis=1, keepdims=True)

            gamma_all = np.maximum(np.sum(gamma, axis=0), 1e-12)
            for k in range(self.n_states):
                self.means[k] = float(np.sum(gamma[:, k] * x) / gamma_all[k])
                diff_sq = (x - self.means[k]) ** 2
                self.stds[k] = float(max(np.sqrt(np.sum(gamma[:, k] * diff_sq) / gamma_all[k]), 1e-4))

            # Maintain State 0 = Low Vol, State 1 = High Vol ordering
            if self.stds[0] > self.stds[1]:
                self.means = self.means[::-1]
                self.stds = self.stds[::-1]
                self.A = self.A[::-1, ::-1]
                self.pi = self.pi[::-1]

        return self

    def predict_filtered_probabilities(self, returns: np.ndarray) -> np.ndarray:
        """
        Compute real-time filtered forward probabilities P(S_t = k | x_1...x_t).
        Guarantees zero-lookahead (only uses past and present returns).
        """
        x = np.asarray(returns, dtype=float).ravel()
        T = len(x)
        B = self._emission_probs(x)

        alpha = np.zeros((T, self.n_states), dtype=float)
        alpha[0] = self.pi * B[0]
        alpha[0] /= (np.sum(alpha[0]) or 1e-12)

        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A) * B[t]
            alpha[t] /= (np.sum(alpha[t]) or 1e-12)

        return alpha


def detect_market_regimes(
    market_returns: pd.Series,
    base_risk_aversion: float = 5.0,
) -> RegimeModelOutput:
    """
    Fits Gaussian HMM on asset returns and extracts filtered regime states,
    panic probability, and regime-adjusted risk aversion for portfolio allocation.
    """
    clean_series = market_returns.dropna()
    arr = clean_series.to_numpy(dtype=float)

    hmm = GaussianHMM2State(max_iter=35)
    hmm.fit(arr)
    filtered_probs = hmm.predict_filtered_probabilities(arr)

    # Latest filtered probability vector at date T
    current_probs = filtered_probs[-1]
    prob_bull = float(current_probs[0])
    prob_panic = float(current_probs[1])

    # Dynamic risk aversion scaling: increases by up to 2.0x during peak crisis regimes
    effective_lambda = float(base_risk_aversion * (1.0 + 1.25 * prob_panic))

    states = [
        RegimeState(
            state_id=0,
            name="Low-Volatility Bull (Calm Growth)",
            mean_return_annualized=round(float(hmm.means[0] * 252 * 100.0), 2),
            volatility_annualized=round(float(hmm.stds[0] * np.sqrt(252) * 100.0), 2),
            current_probability=round(prob_bull, 4),
        ),
        RegimeState(
            state_id=1,
            name="High-Volatility Panic (Crisis / Stress)",
            mean_return_annualized=round(float(hmm.means[1] * 252 * 100.0), 2),
            volatility_annualized=round(float(hmm.stds[1] * np.sqrt(252) * 100.0), 2),
            current_probability=round(prob_panic, 4),
        ),
    ]

    active_id = 0 if prob_bull >= 0.50 else 1
    active_name = states[active_id].name

    viterbi_states = np.argmax(filtered_probs, axis=1)
    hist_series = pd.Series(viterbi_states, index=clean_series.index, name="regime_state")

    return RegimeModelOutput(
        states=states,
        transition_matrix=np.round(hmm.A, 4),
        active_regime_id=active_id,
        active_regime_name=active_name,
        panic_probability=round(prob_panic, 4),
        effective_risk_aversion=round(effective_lambda, 2),
        historical_regimes=hist_series,
    )


def fit_regime_model(
    returns: Union[np.ndarray, pd.Series, pd.DataFrame],
    n_regimes: int = 2,
    random_state: int = 42
) -> Tuple[Any, pd.DataFrame]:
    """Fit a 2-state Gaussian HMM to asset returns."""
    if isinstance(returns, (pd.Series, pd.DataFrame)):
        clean_s = returns.dropna().squeeze()
        dates = clean_s.index
        r_arr = clean_s.to_numpy(dtype=float)
    else:
        r_arr = np.asarray(returns, dtype=float)
        r_arr = r_arr[np.isfinite(r_arr)]
        dates = pd.RangeIndex(len(r_arr))

    hmm = GaussianHMM2State(max_iter=50)
    hmm.fit(r_arr)
    s_ret = pd.Series(r_arr, index=dates)
    s_vol = s_ret.rolling(21, min_periods=5).std().bfill() * np.sqrt(252)
    feature_df = pd.DataFrame({"return": s_ret, "volatility": s_vol}, index=dates)
    return hmm, feature_df


def predict_regime_probabilities(model: Any, feature_df: pd.DataFrame) -> pd.DataFrame:
    """Compute filtered posterior probabilities for each market regime across time."""
    arr = feature_df["return"].to_numpy(dtype=float)
    if hasattr(model, "predict_filtered_probabilities"):
        posteriors = model.predict_filtered_probabilities(arr)
    else:
        posteriors = np.tile([0.5, 0.5], (len(arr), 1))

    result_df = pd.DataFrame(index=feature_df.index)
    result_df["low_vol_bull_prob"] = posteriors[:, 0]
    result_df["high_vol_bear_prob"] = posteriors[:, 1]
    states = np.argmax(posteriors, axis=1)
    result_df["current_regime"] = np.where(states == 0, "Low-Vol Bull", "High-Vol Bear")
    return result_df


def get_current_macro_regime(returns_series: pd.Series) -> Dict[str, Any]:
    """Real-time zero-staleness regime diagnostic at latest date T."""
    res = detect_market_regimes(returns_series)
    return {
        "as_of_date": str(returns_series.index[-1].date()) if hasattr(returns_series.index[-1], "date") else str(returns_series.index[-1]),
        "active_regime": "Low-Vol Bull" if res.active_regime_id == 0 else "High-Vol Bear",
        "low_vol_bull_probability_pct": round(res.states[0].current_probability * 100.0, 2),
        "high_vol_bear_probability_pct": round(res.panic_probability * 100.0, 2),
        "transition_matrix": res.transition_matrix.tolist(),
        "recommended_action": "Aggressive / Max Sharpe" if res.active_regime_id == 0 else "Defensive / Flight-to-Safety",
    }
