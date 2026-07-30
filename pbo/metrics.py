"""
metrics.py — Performance metrics for CSCV ranking.

Each function takes a 1-D numpy array of PnL values and returns a scalar.
New metrics can be added to REGISTRY to make them available in compute_cscv().
"""
from __future__ import annotations

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Standard metrics
# ──────────────────────────────────────────────────────────────────────────────

def sharpe(series: np.ndarray) -> float:
    """Mean / std.  Returns 0.0 when std == 0."""
    sigma = series.std()
    return float(series.mean() / sigma) if sigma > 0 else 0.0


def sortino(series: np.ndarray) -> float:
    """Mean / downside-std.  Returns 0.0 when downside-std == 0."""
    neg = series[series < 0]
    ds = neg.std() if len(neg) > 0 else 0.0
    return float(series.mean() / ds) if ds > 0 else 0.0


def calmar(series: np.ndarray) -> float:
    """Mean / max-drawdown (absolute).  Returns 0.0 when drawdown == 0."""
    cumulative = np.cumsum(series)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    mdd = drawdown.max()
    return float(series.mean() / mdd) if mdd > 0 else 0.0


def total_pnl(series: np.ndarray) -> float:
    """Sum of the PnL series (raw absolute profit)."""
    return float(series.sum())


# ──────────────────────────────────────────────────────────────────────────────
# GA-aligned fitness (mirrors Fitness.py)
# ──────────────────────────────────────────────────────────────────────────────

# Capital normalisation constant matching the GA's evaluate_fitness():
#   pnl_norm = final_pnl / (3 * 1e6)
#   fitness  = 0.005 * sharpe + 0.995 * pnl_norm
_GA_CAPITAL = 3_000_000.0
_GA_W_SHARPE = 0.005
_GA_W_PNL    = 0.995


def ga_fitness(series: np.ndarray) -> float:
    """
    Replicates the GA fitness from Fitness.py:

        mu    = mean(series)
        sigma = std(series)
        sr    = mu / sigma          (Sharpe)
        pnl   = sum(series) / (3e6) (normalised final PnL)
        score = 0.005 * sr + 0.995 * pnl

    Returns -1e9 when sigma <= 0 (degenerate series).
    """
    sigma = series.std()
    if sigma <= 0:
        return -1e9
    sr  = float(series.mean() / sigma)
    pnl = float(series.sum() / _GA_CAPITAL)
    return _GA_W_SHARPE * sr + _GA_W_PNL * pnl


# ──────────────────────────────────────────────────────────────────────────────
# Registry — add custom metrics here
# ──────────────────────────────────────────────────────────────────────────────

REGISTRY: dict[str, callable] = {
    "sharpe":     sharpe,
    "sortino":    sortino,
    "calmar":     calmar,
    "total_pnl":  total_pnl,
    "ga_fitness": ga_fitness,
}
