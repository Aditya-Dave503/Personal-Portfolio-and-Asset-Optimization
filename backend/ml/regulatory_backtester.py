"""
Basel-III Regulatory Value-at-Risk (VaR) Backtesting Engine

Implements regulatory statistical hypothesis tests mandated by the Basel Committee
on Banking Supervision (BCBS) for internal quantitative risk models:
1. Kupiec's Proportion of Failures (POF) Likelihood-Ratio Test
2. Christoffersen's Independence Likelihood-Ratio Test
3. Christoffersen's Conditional Coverage Joint Test
4. Basel Regulatory Traffic Light Classification (Green / Yellow / Red)
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import chi2


@dataclass
class BacktestResult:
    total_observations: int
    total_violations: int
    expected_violations: float
    observed_failure_rate_pct: float
    target_alpha_pct: float
    # Kupiec POF
    kupiec_lr_stat: float
    kupiec_p_value: float
    kupiec_passed: bool
    # Christoffersen Independence
    independence_lr_stat: float
    independence_p_value: float
    independence_passed: bool
    # Combined Conditional Coverage
    conditional_coverage_lr: float
    conditional_coverage_p_value: float
    conditional_coverage_passed: bool
    # Basel Traffic Light
    basel_zone: str
    regulatory_multiplier: float
    regulatory_status: str


def compute_var_violation_series(
    returns: pd.Series,
    var_series: pd.Series,
) -> pd.Series:
    """
    Generate binary indicator series I_t where 1 represents an exception (loss > VaR).
    Note: Returns and VaR are expressed in positive loss magnitude.
    """
    aligned = pd.concat([returns.rename("return"), var_series.rename("var")], axis=1).dropna()
    # A violation occurs when return is more negative than -VaR (i.e. loss exceeds VaR)
    violations = (aligned["return"] < -aligned["var"]).astype(int)
    violations.name = "violation"
    return violations


def kupiec_pof_test(
    total_obs: int,
    violations: int,
    alpha: float = 0.01,
    conf_level: float = 0.95,
) -> Tuple[float, float, bool]:
    """
    Kupiec's Proportion of Failures (POF) Test (Unconditional Coverage).
    Tests H0: p == alpha against H1: p != alpha.
    Test Statistic: LR_POF = -2 * ln(L(alpha) / L(p_hat)) ~ Chi2(1)
    """
    N = int(total_obs)
    x = int(violations)

    if N <= 0:
        raise ValueError("Total observations must be strictly positive.")
    if x < 0 or x > N:
        raise ValueError("Violations must be between 0 and total observations.")

    p = float(alpha)
    p_hat = x / N

    if x == 0:
        # Limit as x -> 0: LR = -2 * N * ln(1 - p)
        lr_stat = float(-2.0 * N * np.log(1.0 - p))
    elif x == N:
        lr_stat = float(-2.0 * N * np.log(p))
    else:
        log_l_null = (N - x) * np.log(1.0 - p) + x * np.log(p)
        log_l_alt = (N - x) * np.log(1.0 - p_hat) + x * np.log(p_hat)
        lr_stat = float(-2.0 * (log_l_null - log_l_alt))

    lr_stat = max(0.0, lr_stat)
    p_val = float(chi2.sf(lr_stat, df=1))
    crit_val = float(chi2.ppf(conf_level, df=1))
    passed = bool(lr_stat <= crit_val)

    return lr_stat, p_val, passed


def christoffersen_independence_test(
    violation_series: Union[pd.Series, np.ndarray, List[int]],
    conf_level: float = 0.95,
) -> Tuple[float, float, bool]:
    """
    Christoffersen's Interval Forecast Test of Independence.
    Tests whether VaR violations cluster consecutively (autocorrelation of failures).
    Test Statistic: LR_ind ~ Chi2(1).
    """
    if isinstance(violation_series, pd.Series):
        v = violation_series.to_numpy(dtype=int)
    else:
        v = np.asarray(violation_series, dtype=int)

    if len(v) < 2:
        return 0.0, 1.0, True

    # Build 2x2 transition count matrix
    # n00: 0 -> 0, n01: 0 -> 1, n10: 1 -> 0, n11: 1 -> 1
    t0 = v[:-1]
    t1 = v[1:]

    n00 = int(np.sum((t0 == 0) & (t1 == 0)))
    n01 = int(np.sum((t0 == 0) & (t1 == 1)))
    n10 = int(np.sum((t0 == 1) & (t1 == 0)))
    n11 = int(np.sum((t0 == 1) & (t1 == 1)))

    # If no violations occurred, independence cannot be rejected
    if (n01 + n11) == 0:
        return 0.0, 1.0, True

    # Transition probabilities
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    # Log-likelihood under null (independent)
    log_l_null = (
        (n00 + n10) * np.log(max(1.0 - pi, 1e-12)) +
        (n01 + n11) * np.log(max(pi, 1e-12))
    )

    # Log-likelihood under alternative (Markov 1st order)
    log_l_alt = 0.0
    if n00 > 0:
        log_l_alt += n00 * np.log(max(1.0 - pi01, 1e-12))
    if n01 > 0:
        log_l_alt += n01 * np.log(max(pi01, 1e-12))
    if n10 > 0:
        log_l_alt += n10 * np.log(max(1.0 - pi11, 1e-12))
    if n11 > 0:
        log_l_alt += n11 * np.log(max(pi11, 1e-12))

    lr_stat = float(max(0.0, -2.0 * (log_l_null - log_l_alt)))
    p_val = float(chi2.sf(lr_stat, df=1))
    crit_val = float(chi2.ppf(conf_level, df=1))
    passed = bool(lr_stat <= crit_val)

    return lr_stat, p_val, passed


def classify_basel_traffic_light(
    total_obs: int,
    violations: int,
    alpha: float = 0.01,
) -> Tuple[str, float, str]:
    """
    Basel Committee on Banking Supervision (BCBS) Traffic Light System.
    Returns: (Zone: 'GREEN'|'YELLOW'|'RED', Capital Multiplier, Status Description)
    """
    # Scale regulatory boundaries proportionally for non-250 sample sizes
    scale = total_obs / 250.0
    green_max = int(np.round(4 * scale))
    yellow_max = int(np.round(9 * scale))

    if violations <= green_max:
        return "GREEN", 3.00, "Approved: Model accuracy is statistically acceptable."
    elif violations <= yellow_max:
        # Linear interpolation between 3.40 and 3.85
        factor = (violations - green_max) / max(1, (yellow_max - green_max))
        multiplier = round(3.40 + 0.45 * factor, 2)
        return "YELLOW", multiplier, "Supervisory Monitoring: Increased capital multiplier required."
    else:
        return "RED", 4.00, "Rejected: Model systematically underestimates tail risk."


# Alias for backwards compatibility
basel_traffic_light_classification = classify_basel_traffic_light


def run_full_regulatory_backtest(
    returns: pd.Series,
    var_series: pd.Series,
    alpha: float = 0.01,
    conf_level: float = 0.95,
) -> BacktestResult:
    """
    Executes complete Basel-III regulatory risk model verification:
    Kupiec POF + Christoffersen Independence + Conditional Coverage + Traffic Light.
    """
    violation_series = compute_var_violation_series(returns, var_series)
    N = len(violation_series)
    x = int(violation_series.sum())

    lr_kupiec, p_kupiec, pass_kupiec = kupiec_pof_test(N, x, alpha=alpha, conf_level=conf_level)
    lr_ind, p_ind, pass_ind = christoffersen_independence_test(violation_series, conf_level=conf_level)

    # Combined Christoffersen Conditional Coverage LR_cc = LR_pof + LR_ind ~ Chi2(2)
    lr_cc = float(lr_kupiec + lr_ind)
    p_cc = float(chi2.sf(lr_cc, df=2))
    crit_cc = float(chi2.ppf(conf_level, df=2))
    pass_cc = bool(lr_cc <= crit_cc)

    zone, multiplier, status_desc = classify_basel_traffic_light(N, x, alpha=alpha)

    return BacktestResult(
        total_observations=N,
        total_violations=x,
        expected_violations=round(N * alpha, 2),
        observed_failure_rate_pct=round((x / N) * 100.0, 3) if N > 0 else 0.0,
        target_alpha_pct=round(alpha * 100.0, 2),
        kupiec_lr_stat=round(lr_kupiec, 4),
        kupiec_p_value=round(p_kupiec, 4),
        kupiec_passed=pass_kupiec,
        independence_lr_stat=round(lr_ind, 4),
        independence_p_value=round(p_ind, 4),
        independence_passed=pass_ind,
        conditional_coverage_lr=round(lr_cc, 4),
        conditional_coverage_p_value=round(p_cc, 4),
        conditional_coverage_passed=pass_cc,
        basel_zone=zone,
        regulatory_multiplier=multiplier,
        regulatory_status=status_desc,
    )


run_full_var_backtest = run_full_regulatory_backtest
