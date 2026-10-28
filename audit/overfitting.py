"""Overfitting statistics.

Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014, "The Deflated Sharpe
Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality").

All Sharpe ratios here are *per period* (daily), not annualised.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

_N = NormalDist()
EULER_GAMMA = 0.5772156649015329


def sharpe_variance(sr: float, n_obs: int, skew: float, kurt: float) -> float:
    """Variance of the SR estimator under non-normal returns (kurt is non-excess)."""
    return max((1 - skew * sr + (kurt - 1) / 4 * sr ** 2) / max(n_obs - 1, 1), 1e-12)


def expected_max_sharpe(n_trials: int, sr_var: float) -> float:
    """E[max SR] of n_trials unskilled strategies (false strategy theorem)."""
    if n_trials <= 1:
        return 0.0
    a = _N.inv_cdf(1 - 1 / n_trials)
    b = _N.inv_cdf(1 - 1 / (n_trials * math.e))
    return math.sqrt(sr_var) * ((1 - EULER_GAMMA) * a + EULER_GAMMA * b)


def probabilistic_sharpe(sr: float, sr_benchmark: float, n_obs: int, skew: float, kurt: float) -> float:
    denom = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    return _N.cdf((sr - sr_benchmark) * math.sqrt(max(n_obs - 1, 1)) / denom)


def deflated_sharpe(returns, n_trials: int, trial_sharpes: list[float] | None = None) -> dict:
    """DSR = PSR evaluated against the expected maximum Sharpe of n_trials.

    If ``trial_sharpes`` (daily SRs of the other variants tried) has >= 2 entries
    their cross-sectional variance is used; otherwise the estimator variance.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 30 or r.std(ddof=1) == 0:
        return {"dsr": 0.0, "psr": 0.0, "sr_daily": 0.0, "sr0_daily": 0.0, "n_trials": n_trials, "n_obs": n}
    sr = float(r.mean() / r.std(ddof=1))
    m = r - r.mean()
    skew = float((m ** 3).mean() / (m ** 2).mean() ** 1.5)
    kurt = float((m ** 4).mean() / (m ** 2).mean() ** 2)
    if trial_sharpes is not None and len(trial_sharpes) >= 2:
        var = float(np.var(trial_sharpes, ddof=1))
    else:
        var = sharpe_variance(0.0, n, skew, kurt)
    sr0 = expected_max_sharpe(max(int(n_trials), 1), var)
    return {
        "dsr": probabilistic_sharpe(sr, sr0, n, skew, kurt),
        "psr": probabilistic_sharpe(sr, 0.0, n, skew, kurt),
        "sr_daily": sr,
        "sr_annual": sr * math.sqrt(252),
        "sr0_daily": sr0,
        "sr0_annual": sr0 * math.sqrt(252),
        "n_trials": int(n_trials),
        "n_obs": n,
        "skew": skew,
        "kurtosis": kurt,
    }


def split_sharpes(returns, train_frac: float = 0.7) -> tuple[float, float]:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    k = int(len(r) * train_frac)

    def ann(x):
        return float(x.mean() / x.std(ddof=1) * math.sqrt(252)) if len(x) > 2 and x.std(ddof=1) > 0 else 0.0

    return ann(r[:k]), ann(r[k:])
