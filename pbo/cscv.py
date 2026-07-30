"""
cscv.py — Core CSCV algorithm.

Implements Algorithm 2.3 from:
    Bailey, Borwein, López de Prado, Zhu
    "The Probability of Backtest Overfitting" (2015)
    https://ssrn.com/abstract=2326253

Public functions
----------------
compute_cscv(M, S, metric, max_S, min_partition_size)  → result dict
stochastic_dominance(result)       → SD analysis dict
aggregate_series(series, window)   → aggregated np.ndarray
"""
from __future__ import annotations

import math
from itertools import combinations
from typing import Sequence

import numpy as np
from scipy import stats

from .metrics import REGISTRY


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_series(series: Sequence[float], window: int = 1) -> np.ndarray:
    """
    Sum-aggregate a PnL series into non-overlapping windows.

    Parameters
    ----------
    series : array-like  — raw PnL series (e.g. daily).
    window : int         — observations to sum per output slice.
                           window=1 → no change.  window=5 → 5-day sums.

    Returns
    -------
    np.ndarray  length = floor(len(series) / window)
    """
    s = np.asarray(series, dtype=float)
    if window <= 1:
        return s
    n = len(s) // window
    return s[: n * window].reshape(n, window).sum(axis=1)


def _auto_S(T: int, max_S: int, min_partition_size: int = 2) -> int:
    """
    Largest even S such that:
      - S <= max_S
      - every partition has >= min_partition_size rows
        (np.array_split gives floor(T/S) or ceil(T/S) rows per partition,
         so the constraint is floor(T/S) >= min_partition_size, i.e. S <= T//min_partition_size)
      - S >= 4  (minimum for any useful cross-validation)

    For weekly data (T ~ 10-30):  min_partition_size=2  → S = T//2 rounded down to even
    For daily data  (T ~ 200+):   min_partition_size=4  → aligns with paper recommendation

    Examples (max_S=16):
        T=17, min=2  → S=8   (C(8,4)=70  combinations, IS≈9 weeks)
        T=17, min=1  → S=16  (C(16,8)=12870 combinations, IS≈9 weeks)
        T=80, min=4  → S=16  (C(16,8)=12870 combinations)
        T=40, min=4  → S=10  (C(10,5)=252 combinations)
    """
    S_from_T = T // min_partition_size        # largest S where each partition >= min_partition_size
    if S_from_T % 2 != 0:
        S_from_T -= 1                          # must be even
    S = min(max_S if max_S % 2 == 0 else max_S - 1, S_from_T)
    return max(S, 4)                           # floor at 4


def _empty(S, N, T, metric, warning):
    nan = float("nan")
    return {
        "pbo": nan, "prob_loss": nan,
        "logits": [], "is_perfs": [], "oos_perfs": [], "oos_all_perfs": [],
        "degradation_slope": nan, "degradation_intercept": nan, "degradation_r2": nan,
        "S": S, "N": N, "T": T, "n_combinations": 0,
        "metric": metric, "warning": warning,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Core
# ──────────────────────────────────────────────────────────────────────────────

def compute_cscv(
    M: np.ndarray,
    S: int | None = None,
    metric: str = "sharpe",
    max_S: int = 16,
    min_partition_size: int = 1,
) -> dict:
    """
    Combinatorially Symmetric Cross-Validation (Algorithm 2.3).

    Parameters
    ----------
    M : np.ndarray, shape (T, N)
        Matrix of PnL time-series.  Rows = time, columns = strategy configs.
    S : int or None
        Even number of time-partitions (>= 4).
        None → auto-selected; pass an explicit int to override.
    metric : str
        Performance measure used to rank strategies.
        Options: "sharpe" (default), "sortino", "calmar", "total_pnl", "ga_fitness".
    max_S : int
        Upper cap on auto-selected S.
        Recommended defaults: 16 for daily data, 8 for weekly.
    min_partition_size : int
        Minimum rows per partition when auto-selecting S.
        S is chosen as the largest even S where floor(T/S) >= min_partition_size.
        - 2  (default) → maximises combinations for short series (e.g. weekly)
        - 4            → paper recommendation for daily data

    Returns
    -------
    dict
        pbo                  float  — Probability of Backtest Overfitting ∈ [0,1]
        prob_loss            float  — P(best-IS strategy OOS < 0)
        logits               list   — one logit per IS/OOS combination
        is_perfs             list   — IS performance of selected strategy per combo
        oos_perfs            list   — OOS performance of selected strategy per combo
        oos_all_perfs        list   — OOS performance of ALL N strategies per combo
        degradation_slope    float  — β  from  OOS = α + β·IS  regression
        degradation_intercept float
        degradation_r2       float
        S, N, T, n_combinations, metric
        warning              str | None
    """
    M = np.asarray(M, dtype=float)
    if M.ndim != 2:
        raise ValueError("M must be a 2-D array of shape (T, N)")

    T, N = M.shape

    perf_fn = REGISTRY.get(metric)
    if perf_fn is None:
        raise ValueError(f"Unknown metric '{metric}'. Available: {list(REGISTRY)}")

    # ── Validate / auto-select S ──────────────────────────────────────────────
    if S is None:
        S = _auto_S(T, max_S, min_partition_size)
    S = int(S)
    if S % 2 != 0:
        S -= 1
    # Safety guard: ensure no partition is empty (floor(T/S) >= 1)
    while S > 4 and T // S < 1:
        S -= 2

    if N < 2:
        return _empty(S, N, T, metric, "Need N >= 2 strategy configurations.")
    if T < S:
        return _empty(S, N, T, metric, f"T={T} < S={S}: cannot partition.")

    # ── Step 1: partition M into S equal sub-matrices ─────────────────────────
    partitions = np.array_split(M, S, axis=0)

    # ── Steps 2–4: iterate over all C(S, S/2) IS/OOS splits ──────────────────
    all_combos = list(combinations(range(S), S // 2))

    logits        = []
    is_perfs      = []
    oos_perfs     = []
    oos_all_perfs = []

    for is_idx in all_combos:
        oos_idx = [i for i in range(S) if i not in is_idx]

        IS  = np.vstack([partitions[i] for i in is_idx])
        OOS = np.vstack([partitions[i] for i in oos_idx])

        IS_perf  = np.array([perf_fn(IS[:, n])  for n in range(N)])
        OOS_perf = np.array([perf_fn(OOS[:, n]) for n in range(N)])

        n_star = int(np.argmax(IS_perf))

        # ω_c: relative OOS rank of the IS-best strategy, normalised to (0, 1)
        oos_rank = int(np.sum(OOS_perf <= OOS_perf[n_star]))
        omega_c  = np.clip(oos_rank / (N + 1), 1e-9, 1 - 1e-9)

        # λ_c = logit(ω_c) — positive → IS-best outperforms OOS median
        lambda_c = float(np.log(omega_c / (1.0 - omega_c)))

        logits.append(lambda_c)
        is_perfs.append(float(IS_perf[n_star]))
        oos_perfs.append(float(OOS_perf[n_star]))
        oos_all_perfs.append(OOS_perf.tolist())

    logits_arr = np.array(logits)
    is_arr     = np.array(is_perfs)
    oos_arr    = np.array(oos_perfs)

    pbo       = float(np.mean(logits_arr < 0))
    prob_loss = float(np.mean(oos_arr < 0))

    # ── Performance degradation regression ───────────────────────────────────
    if len(logits) >= 2:
        slope, intercept, r, *_ = stats.linregress(is_arr, oos_arr)
        r2 = float(r ** 2)
    else:
        slope = intercept = r2 = float("nan")

    return {
        "pbo":                   pbo,
        "prob_loss":             prob_loss,
        "logits":                logits,
        "is_perfs":              is_perfs,
        "oos_perfs":             oos_perfs,
        "oos_all_perfs":         oos_all_perfs,
        "degradation_slope":     float(slope),
        "degradation_intercept": float(intercept),
        "degradation_r2":        r2,
        "S":                     S,
        "N":                     N,
        "T":                     T,
        "n_combinations":        len(logits),
        "metric":                metric,
        "warning":               None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Stochastic dominance
# ──────────────────────────────────────────────────────────────────────────────

def stochastic_dominance(result: dict) -> dict:
    """
    First- and second-order stochastic dominance of the IS-optimal OOS
    distribution over the full non-optimized OOS distribution.

    Returns
    -------
    dict
        x_grid   np.ndarray  evaluation grid
        cdf_opt  np.ndarray  CDF of IS-selected strategy OOS performance
        cdf_all  np.ndarray  CDF of mean OOS performance across all N strategies
        sd2      np.ndarray  running integral of (cdf_all − cdf_opt)
        sd1_dom  bool        True if 1st-order dominance holds (opt >> all)
        sd2_dom  bool        True if 2nd-order dominance holds
    """
    oos_opt = np.array(result["oos_perfs"])
    oos_all = np.array([np.mean(row) for row in result["oos_all_perfs"]])

    combined = np.concatenate([oos_opt, oos_all])
    x_grid   = np.linspace(combined.min(), combined.max(), 500)

    def _ecdf(data, x):
        return np.array([np.mean(data <= xi) for xi in x])

    cdf_opt = _ecdf(oos_opt, x_grid)
    cdf_all = _ecdf(oos_all, x_grid)

    dx  = x_grid[1] - x_grid[0]
    sd2 = np.cumsum((cdf_all - cdf_opt) * dx)

    sd1_dom = bool(np.all(cdf_opt <= cdf_all))
    sd2_dom = bool(sd1_dom or np.all(sd2 >= 0))

    return {
        "x_grid":  x_grid,
        "cdf_opt": cdf_opt,
        "cdf_all": cdf_all,
        "sd2":     sd2,
        "sd1_dom": sd1_dom,
        "sd2_dom": sd2_dom,
    }
