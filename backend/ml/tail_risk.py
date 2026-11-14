"""
Non-Gaussian Tail Risk Analytics.

Implements rigorous tail-risk measurements accounting for skewness and fat-tailed excess kurtosis:
1. Sample Skewness (S) and Fisher-Pearson Excess Kurtosis (K = Kurtosis - 3).
2. Cornish-Fisher Expansion adjusted critical quantile:
       z_{CF} = z_alpha + (S/6) * (z_alpha^2 - 1)
                        + ((K - 3)/24) * (z_alpha^3 - 3*z_alpha)
                        - (S^2 / 36) * (2*z_alpha^3 - 5*z_alpha)
   *Note*: When excess kurtosis K_excess = K - 3 is already passed, the 2nd term is (K_excess / 24).
3. Modified Value at Risk (mVaR):
       mVaR_alpha = -(mu + z_{CF} * sigma)
4. Conditional Value at Risk (CVaR / Expected Shortfall):
       CVaR_alpha = -E[R | R <= -mVaR_alpha]
"""

from typing import Dict, Union, Optional
import numpy as np
import pandas as pd
from scipy import stats


def calculate_higher_moments(
    returns: Union[np.ndarray, pd.Series]
) -> Dict[str, float]:
    """
    Compute mean, standard deviation, skewness, and excess kurtosis.

    Parameters:
        returns: 1D array or Series of historical periodic returns.

    Returns:
        Dictionary with:
            - mean: sample mean
            - std: sample standard deviation (ddof=1)
            - skewness: sample skewness S
            - excess_kurtosis: Fisher excess kurtosis (K - 3, normal = 0.0)
            - raw_kurtosis: Pearson kurtosis (normal = 3.0)
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 4:
        raise ValueError("At least 4 observations required to estimate skewness and kurtosis.")

    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1))

    # Fisher-Pearson skewness and excess kurtosis (bias=False for sample unbiasing)
    skew_val = float(stats.skew(arr, bias=False))
    kurt_excess = float(stats.kurtosis(arr, fisher=True, bias=False))  # Excess: normal = 0.0
    kurt_raw = kurt_excess + 3.0  # Pearson raw kurtosis: normal = 3.0

    return {
        "mean": mean_val,
        "std": std_val,
        "skewness": skew_val,
        "excess_kurtosis": kurt_excess,
        "raw_kurtosis": kurt_raw,
    }


def cornish_fisher_quantile(
    alpha: float,
    skewness: float,
    excess_kurtosis: float,
    raw_kurtosis: Optional[float] = None
) -> float:
    """
    Compute the Cornish-Fisher adjusted critical value z_{CF} for confidence level (1 - alpha).

    Specification:
        z_{CF} = z_alpha + (S/6) * (z_alpha^2 - 1)
                        + ((K_raw - 3) / 24) * (z_alpha^3 - 3*z_alpha)
                        - (S^2 / 36) * (2*z_alpha^3 - 5*z_alpha)

    where z_alpha is the standard normal critical quantile, e.g., stats.norm.ppf(alpha).
    For alpha = 0.05, z_alpha ~= -1.64485.
    For alpha = 0.01, z_alpha ~= -2.32635.
    """
    if not (0.0 < alpha < 0.5):
        raise ValueError("Alpha must be in (0, 0.5), e.g. 0.05 for 95% VaR or 0.01 for 99% VaR.")

    z_alpha = float(stats.norm.ppf(alpha))

    # K - 3 is precisely excess_kurtosis
    k_minus_3 = excess_kurtosis if raw_kurtosis is None else (raw_kurtosis - 3.0)
    s = skewness

    z2 = z_alpha ** 2
    z3 = z_alpha ** 3

    term1 = z_alpha
    term2 = (s / 6.0) * (z2 - 1.0)
    term3 = (k_minus_3 / 24.0) * (z3 - 3.0 * z_alpha)
    term4 = -(s ** 2 / 36.0) * (2.0 * z3 - 5.0 * z_alpha)

    z_cf = term1 + term2 + term3 + term4
    return float(z_cf)


def calculate_modified_var(
    returns: Union[np.ndarray, pd.Series],
    alpha: float = 0.05
) -> float:
    """
    Calculate Modified Value at Risk (mVaR) using Cornish-Fisher expansion.
    Expressed as a positive percentage/decimal representing the loss threshold.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    moments = calculate_higher_moments(arr)

    z_cf = cornish_fisher_quantile(
        alpha=alpha,
        skewness=moments["skewness"],
        excess_kurtosis=moments["excess_kurtosis"],
        raw_kurtosis=moments["raw_kurtosis"],
    )

    # mVaR = -(mu + z_cf * sigma)
    # A negative return at quantile z_cf yields a positive VaR loss.
    mvar = -(moments["mean"] + z_cf * moments["std"])
    return float(max(0.0, mvar))


def calculate_cvar(
    returns: Union[np.ndarray, pd.Series],
    var_threshold: Optional[float] = None,
    alpha: float = 0.05
) -> float:
    """
    Calculate Conditional Value at Risk (CVaR / Expected Shortfall).

    Definition:
        Expected loss given that the loss exceeds the VaR threshold:
        CVaR = -E[R | R <= -VaR]

    Parameters:
        returns: 1D array of historical returns.
        var_threshold: Positive loss threshold (if None, computes modified VaR).
        alpha: Confidence significance level (default: 0.05).
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0

    if var_threshold is None:
        var_threshold = calculate_modified_var(arr, alpha=alpha)

    # Losses exceeding VaR correspond to returns R <= -var_threshold
    tail_losses = arr[arr <= -var_threshold]

    if len(tail_losses) == 0:
        # If no historical return breaches the parametric Cornish-Fisher mVaR,
        # CVaR represents at least the VaR threshold itself (or worst loss if more extreme)
        return float(max(var_threshold, abs(np.min(arr))))

    cvar = float(-np.mean(tail_losses))
    return float(max(var_threshold, cvar))


def evaluate_tail_risk(
    returns: Union[np.ndarray, pd.Series],
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Complete non-Gaussian tail risk diagnostic suite.

    Returns:
        Dictionary with higher moments, Gaussian vs Modified VaR, and CVaR.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    moments = calculate_higher_moments(arr)

    z_gaussian = float(stats.norm.ppf(alpha))
    gaussian_var = max(0.0, -(moments["mean"] + z_gaussian * moments["std"]))

    z_cf = cornish_fisher_quantile(
        alpha=alpha,
        skewness=moments["skewness"],
        excess_kurtosis=moments["excess_kurtosis"],
        raw_kurtosis=moments["raw_kurtosis"],
    )
    modified_var = max(0.0, -(moments["mean"] + z_cf * moments["std"]))
    cvar = calculate_cvar(arr, var_threshold=modified_var, alpha=alpha)

    # Historical empirical percentile VaR for cross-validation
    empirical_var = max(0.0, -float(np.percentile(arr, alpha * 100.0)))

    return {
        "mean_return": round(moments["mean"], 6),
        "volatility": round(moments["std"], 6),
        "skewness": round(moments["skewness"], 4),
        "excess_kurtosis": round(moments["excess_kurtosis"], 4),
        "raw_kurtosis": round(moments["raw_kurtosis"], 4),
        "z_gaussian": round(z_gaussian, 4),
        "z_cornish_fisher": round(z_cf, 4),
        "gaussian_var": round(gaussian_var, 6),
        "modified_var": round(modified_var, 6),
        "empirical_var": round(empirical_var, 6),
        "cvar_expected_shortfall": round(cvar, 6),
        "confidence_level_pct": (1.0 - alpha) * 100.0,
    }

