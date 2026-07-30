"""
Walk-forward simulation of a dynamic (rolling) BPV exclusion threshold, and
the lookback x n_buckets grid built from it.

No lookahead: for day t, only days strictly before t are ever used to derive
that day's bucket edges / exclusion set. day t's own pnl is only ever used
to *score* the decision made about day t, never to make it.

Two exclusion rules are supported (both operate on the trailing window only):
    worst_pnl_dynamic — rank trailing buckets by trailing total PnL, exclude
                        the worst n_exclude (mirrors calibrate_thresholds.py,
                        just recomputed on a rolling window)
    highest_vol       — always exclude the top n_exclude buckets by value
                        (no PnL ranking, no small-sample estimation)

Two bucketing methods (see bucketing.py): equal_count / equal_width.

Four IV-combination modes (IV_MODES) — chosen to be non-redundant. Note that
"both filters must agree to trade" and "no-trade if either filter says no"
are the same boolean formula (trade = harcj_trade AND iv_trade); the live
dashboard's own flag_combine_method names them "or" and "and" respectively
(named after how the NO-TRADE conditions combine, not the trade signals —
confusing, and exactly why this module uses direct names instead):
    harcj_only — ignore IV entirely, decide on HARCJ alone
    iv_only    — ignore HARCJ entirely, decide on the static IV cutoff alone
                 (matches SPXW's actual live config, flag_combine_method="ignore_harcj")
    and_both   — trade = harcj_trade AND iv_trade (both must pass)
                 (matches live flag_combine_method="or")
    or_either  — trade = harcj_trade OR iv_trade (either passing is enough)
                 (matches live flag_combine_method="and")
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from bucketing import assign_buckets, BucketMethod

_TRADING_DAYS_PER_YEAR = 252

IV_MODES = ["harcj_only", "iv_only", "and_both", "or_either"]


def _combine(harcj_trade: bool, iv_trade: bool | None, iv_mode: str) -> bool:
    """Apply one of the four IV_MODES to a single day's two filter outcomes."""
    if iv_mode == "harcj_only":
        return harcj_trade
    if iv_mode == "iv_only":
        return bool(iv_trade)
    if iv_mode == "and_both":
        return harcj_trade and bool(iv_trade)
    if iv_mode == "or_either":
        return harcj_trade or bool(iv_trade)
    raise ValueError(f"Unknown iv_mode: {iv_mode!r}. Expected one of {IV_MODES}")


# ── Metrics ────────────────────────────────────────────────────────────────────

def _sortino(daily_pnl: np.ndarray) -> float:
    if len(daily_pnl) == 0:
        return np.nan
    mean = daily_pnl.mean()
    downside = daily_pnl[daily_pnl < 0]
    if len(downside) == 0:
        return np.nan
    dd = downside.std(ddof=0)
    if dd == 0:
        return np.nan
    return float(mean / dd * np.sqrt(_TRADING_DAYS_PER_YEAR))


def _sharpe(daily_pnl: np.ndarray) -> float:
    if len(daily_pnl) == 0:
        return np.nan
    sd = daily_pnl.std(ddof=0)
    if sd == 0:
        return np.nan
    return float(daily_pnl.mean() / sd * np.sqrt(_TRADING_DAYS_PER_YEAR))


def _max_drawdown(daily_pnl: pd.Series) -> float:
    """Worst peak-to-trough decline of the cumulative PnL curve (dollars, <= 0)."""
    if daily_pnl.empty:
        return np.nan
    cum = daily_pnl.cumsum()
    running_peak = cum.cummax()
    drawdown = cum - running_peak
    return float(drawdown.min())


def summarize(daily_pnl: pd.Series, traded_mask: pd.Series) -> dict:
    """
    daily_pnl   — realized pnl per day, 0 on no-trade days (full period, same
                  index as the walk-forward run)
    traded_mask — bool series, True where a trade was actually taken

    Includes "_daily_pnl"/"_traded_mask" (underscore-prefixed, for equity-curve
    plotting) alongside the plain numeric summary keys used for display.
    """
    traded_pnl = daily_pnl[traded_mask]
    return {
        "total_pnl":      float(daily_pnl.sum()),
        "avg_daily_pnl":  float(daily_pnl.mean()) if len(daily_pnl) else np.nan,
        "sortino":        _sortino(daily_pnl.values),
        "sharpe":         _sharpe(daily_pnl.values),
        "win_rate":       float((traded_pnl > 0).mean()) if len(traded_pnl) else np.nan,
        "drawdown":       _max_drawdown(daily_pnl),
        "max_daily_loss": float(traded_pnl.min()) if len(traded_pnl) else np.nan,
        "n_days":         int(len(daily_pnl)),
        "n_traded":       int(traded_mask.sum()),
        "n_skipped":      int((~traded_mask).sum()),
        "_daily_pnl":     daily_pnl,
        "_traded_mask":   traded_mask,
    }


# ── Baselines ──────────────────────────────────────────────────────────────────

def always_trade_baseline(df: pd.DataFrame, pnl_col: str = "pnl") -> dict:
    daily_pnl = df[pnl_col]
    traded = pd.Series(True, index=df.index)
    return summarize(daily_pnl, traded)


def static_threshold_baseline(
    df: pd.DataFrame,
    exclusion_ranges: list[list[float]],
    iv_cutoff: float | None = None,
    value_col: str = "pred_vol_pct",
    pnl_col: str = "pnl",
    iv_col: str = "mean_iv",
    iv_mode: str = "harcj_only",
) -> dict:
    """
    Apply today's *static* harcj_exclusion_ranges (+ iv_cutoff, per iv_mode)
    as fixed constants across the whole period — the "what the live config
    would have done" reference line. Not walk-forward; these numbers are
    already fixed today, so there's nothing to walk forward over.
    """
    excluded = pd.Series(False, index=df.index)
    for lo, hi in exclusion_ranges:
        excluded |= df[value_col].between(lo, hi)
    harcj_trade = ~excluded
    iv_trade = (df[iv_col] <= iv_cutoff) if iv_cutoff is not None else pd.Series(True, index=df.index)

    trade = pd.Series(
        [_combine(bool(h), bool(i), iv_mode) for h, i in zip(harcj_trade, iv_trade)],
        index=df.index,
    )

    daily_pnl = df[pnl_col].where(trade, 0.0)
    return summarize(daily_pnl, trade)


# ── Shared per-day bucket-exclusion decision ──────────────────────────────────

@dataclass
class _BucketDecision:
    trade: bool
    bucket: int
    n_actual: int
    excluded_bucket_idx: list
    edges: np.ndarray


def _bucket_decision(
    train_values: np.ndarray,
    train_pnls: np.ndarray,
    today_val: float,
    n_buckets: int,
    method: BucketMethod,
    exclusion_rule: str,
    n_exclude: int,
) -> _BucketDecision | None:
    """
    One day's bucket-exclusion decision, given only the trailing window
    (no lookahead). Shared by BPV and IV — identical mechanism, different
    input series. Returns None if the trailing window can't form any
    bucket (not enough distinct values).
    """
    labels, edges = assign_buckets(pd.Series(train_values), n_buckets, method)
    n_actual = len(edges) - 1
    if n_actual < 1:
        return None

    today_bucket = np.searchsorted(edges, today_val, side="right") - 1
    today_bucket = int(np.clip(today_bucket, 0, n_actual - 1))

    if exclusion_rule == "worst_pnl_dynamic":
        train_bucket_pnl = pd.Series(train_pnls).groupby(labels.astype(float)).sum()
        worst = train_bucket_pnl.nsmallest(min(n_exclude, len(train_bucket_pnl))).index
        excluded_buckets = set(int(b) for b in worst)
    elif exclusion_rule == "highest_vol":
        excluded_buckets = set(range(max(0, n_actual - n_exclude), n_actual))
    else:
        raise ValueError(f"Unknown exclusion_rule: {exclusion_rule!r}")

    trade = today_bucket not in excluded_buckets
    return _BucketDecision(
        trade=trade, bucket=today_bucket, n_actual=n_actual,
        excluded_bucket_idx=sorted(excluded_buckets), edges=edges,
    )


def _excluded_ranges_for_display(decision: _BucketDecision, today_val: float) -> list:
    """[lo, hi] pairs for each excluded bucket, extended to include today_val
    if it was clipped into range (see walk_forward_run's original comment)."""
    ranges = []
    for b in decision.excluded_bucket_idx:
        lo, hi = float(decision.edges[b]), float(decision.edges[b + 1])
        if b == decision.bucket:
            lo, hi = min(lo, today_val), max(hi, today_val)
        ranges.append([lo, hi])
    return ranges


# ── Walk-forward single run ───────────────────────────────────────────────────

@dataclass
class RunResult:
    metrics: dict
    reliable: bool
    avg_days_per_bucket: float
    daily_pnl: pd.Series = field(repr=False)
    traded_mask: pd.Series = field(repr=False)
    diagnostics: pd.DataFrame | None = field(default=None, repr=False)


def walk_forward_run(
    df: pd.DataFrame,
    lookback: int,
    n_buckets: int,
    method: BucketMethod = "equal_count",
    exclusion_rule: str = "worst_pnl_dynamic",
    n_exclude: int = 1,
    value_col: str = "pred_vol_pct",
    pnl_col: str = "pnl",
    iv_col: str = "mean_iv",
    iv_cutoff: float | None = None,
    iv_mode: str = "harcj_only",         # one of IV_MODES
    min_days_per_bucket: int = 10,
    return_diagnostics: bool = False,
) -> RunResult:
    """
    Walk forward one day at a time. At day t, use the trailing `lookback`
    days strictly before t to derive bucket edges + the excluded set;
    apply that to t's own value to decide trade/no-trade; day t's own pnl
    is only used afterwards, to score that decision.

    return_diagnostics=True additionally records, per scored day: the
    day's own value, its bucket, the excluded-range(s) in force that day
    (a list of [lo, hi] pairs — possibly non-contiguous for
    worst_pnl_dynamic), the IV value, and each filter's individual
    trade/no-trade outcome. Off by default since run_grid doesn't need it.
    """
    if iv_mode in ("iv_only", "and_both", "or_either") and iv_cutoff is None:
        raise ValueError(f"iv_mode={iv_mode!r} requires iv_cutoff")

    df = df.sort_index()
    n = len(df)
    decisions = pd.Series(False, index=df.index)
    realized  = pd.Series(0.0, index=df.index)
    diag_rows = [] if return_diagnostics else None

    values = df[value_col].values
    pnls   = df[pnl_col].values
    ivs    = df[iv_col].values if iv_col in df.columns else np.full(n, np.nan)

    for t in range(lookback, n):
        train_values = values[t - lookback: t]
        train_pnls   = pnls[t - lookback: t]
        today_val    = values[t]

        decision = _bucket_decision(train_values, train_pnls, today_val,
                                     n_buckets, method, exclusion_rule, n_exclude)
        if decision is None:
            continue
        harcj_trade = decision.trade

        iv_today = ivs[t]
        iv_trade = (not np.isnan(iv_today)) and (iv_today <= iv_cutoff) if iv_cutoff is not None else None

        trade = _combine(harcj_trade, iv_trade, iv_mode)

        decisions.iloc[t] = trade
        if trade:
            realized.iloc[t] = pnls[t]

        if return_diagnostics:
            diag_rows.append({
                "date":                   df.index[t],
                "value":                  float(today_val),
                "bucket":                 decision.bucket,
                "n_actual_buckets":       decision.n_actual,
                "excluded_bucket_idx":    decision.excluded_bucket_idx,
                "excluded_ranges":        _excluded_ranges_for_display(decision, today_val),
                "harcj_trade":            harcj_trade,
                "iv_value":               None if np.isnan(iv_today) else float(iv_today),
                "iv_trade":               iv_trade,
                "trade":                  trade,
                "pnl":                    float(pnls[t]) if trade else 0.0,
            })

    scored = decisions.index[lookback:] if n > lookback else decisions.index[:0]
    daily_pnl   = realized.loc[scored]
    traded_mask = decisions.loc[scored]

    metrics = summarize(daily_pnl, traded_mask)
    avg_days_per_bucket = lookback / n_buckets if n_buckets else np.nan
    reliable = (
        exclusion_rule != "worst_pnl_dynamic"
        or avg_days_per_bucket >= min_days_per_bucket
    )

    diagnostics = pd.DataFrame(diag_rows).set_index("date") if return_diagnostics and diag_rows else None

    return RunResult(
        metrics=metrics,
        reliable=reliable,
        avg_days_per_bucket=avg_days_per_bucket,
        daily_pnl=daily_pnl,
        traded_mask=traded_mask,
        diagnostics=diagnostics,
    )


# ── Dual walk-forward (independent BPV + IV dynamic thresholds) ────────────────

@dataclass
class DualParams:
    """One side's (BPV's or IV's) dynamic-threshold configuration."""
    lookback: int
    n_buckets: int
    method: BucketMethod = "equal_count"
    exclusion_rule: str = "worst_pnl_dynamic"
    n_exclude: int = 1


def walk_forward_run_dual(
    df: pd.DataFrame,
    harcj: DualParams,
    iv: DualParams,
    harcj_value_col: str = "pred_vol_pct",
    iv_value_col: str = "iv_dynamic",
    pnl_col: str = "pnl",
    iv_mode: str = "and_both",       # "and_both" | "or_either" only
    min_days_per_bucket: int = 10,
    return_diagnostics: bool = False,
) -> RunResult:
    """
    Walk forward one day at a time, running TWO independent bucket-exclusion
    decisions (BPV via `harcj`, IV via `iv` — each with its own lookback,
    n_buckets, method, exclusion_rule, n_exclude) and combining them via
    `iv_mode`. Same no-lookahead discipline as walk_forward_run: each side's
    edges/exclusion set come only from that side's own trailing window,
    ending strictly before the day being judged.

    harcj_only/iv_only don't need this — use walk_forward_run directly,
    pointed at whichever single series matters (this function is only for
    genuinely combining two independently-parameterized dynamic decisions).

    df must already have both value columns and pnl with no NaN gaps in the
    scored range (dropna beforehand, same discipline as load_joined()).
    """
    if iv_mode not in ("and_both", "or_either"):
        raise ValueError(
            f"walk_forward_run_dual is only for and_both/or_either, got {iv_mode!r} — "
            "use walk_forward_run directly for harcj_only/iv_only"
        )

    df = df.sort_index()
    n = len(df)
    max_lookback = max(harcj.lookback, iv.lookback)

    decisions = pd.Series(False, index=df.index)
    realized  = pd.Series(0.0, index=df.index)
    diag_rows = [] if return_diagnostics else None

    harcj_values = df[harcj_value_col].values
    iv_values    = df[iv_value_col].values
    pnls         = df[pnl_col].values

    for t in range(max_lookback, n):
        harcj_decision = _bucket_decision(
            harcj_values[t - harcj.lookback: t], pnls[t - harcj.lookback: t], harcj_values[t],
            harcj.n_buckets, harcj.method, harcj.exclusion_rule, harcj.n_exclude,
        )
        iv_decision = _bucket_decision(
            iv_values[t - iv.lookback: t], pnls[t - iv.lookback: t], iv_values[t],
            iv.n_buckets, iv.method, iv.exclusion_rule, iv.n_exclude,
        )
        if harcj_decision is None or iv_decision is None:
            continue

        trade = _combine(harcj_decision.trade, iv_decision.trade, iv_mode)
        decisions.iloc[t] = trade
        if trade:
            realized.iloc[t] = pnls[t]

        if return_diagnostics:
            diag_rows.append({
                "date":                  df.index[t],
                "harcj_value":           float(harcj_values[t]),
                "harcj_bucket":          harcj_decision.bucket,
                "harcj_excluded_ranges": _excluded_ranges_for_display(harcj_decision, harcj_values[t]),
                "harcj_trade":           harcj_decision.trade,
                "iv_value":              float(iv_values[t]),
                "iv_bucket":             iv_decision.bucket,
                "iv_excluded_ranges":    _excluded_ranges_for_display(iv_decision, iv_values[t]),
                "iv_trade":              iv_decision.trade,
                "trade":                 trade,
                "pnl":                   float(pnls[t]) if trade else 0.0,
            })

    scored = decisions.index[max_lookback:] if n > max_lookback else decisions.index[:0]
    daily_pnl   = realized.loc[scored]
    traded_mask = decisions.loc[scored]

    metrics = summarize(daily_pnl, traded_mask)

    harcj_days_per_bucket = harcj.lookback / harcj.n_buckets if harcj.n_buckets else np.nan
    iv_days_per_bucket    = iv.lookback / iv.n_buckets if iv.n_buckets else np.nan
    harcj_reliable = harcj.exclusion_rule != "worst_pnl_dynamic" or harcj_days_per_bucket >= min_days_per_bucket
    iv_reliable    = iv.exclusion_rule != "worst_pnl_dynamic" or iv_days_per_bucket >= min_days_per_bucket
    reliable = harcj_reliable and iv_reliable

    diagnostics = pd.DataFrame(diag_rows).set_index("date") if return_diagnostics and diag_rows else None

    return RunResult(
        metrics=metrics,
        reliable=reliable,
        avg_days_per_bucket=min(harcj_days_per_bucket, iv_days_per_bucket),
        daily_pnl=daily_pnl,
        traded_mask=traded_mask,
        diagnostics=diagnostics,
    )


# ── Grid ───────────────────────────────────────────────────────────────────────

def run_grid(
    df: pd.DataFrame,
    lookbacks: list[int],
    bucket_counts: list[int],
    method: BucketMethod = "equal_count",
    exclusion_rule: str = "worst_pnl_dynamic",
    n_exclude: int = 1,
    value_col: str = "pred_vol_pct",
    iv_cutoff: float | None = None,
    iv_mode: str = "harcj_only",
    min_days_per_bucket: int = 10,
) -> dict:
    """
    Returns:
        {
          "metrics": {metric_name: DataFrame[index=bucket_counts, columns=lookbacks]},
          "reliable": DataFrame[index=bucket_counts, columns=lookbacks] (bool),
          "n_traded": DataFrame[...] (int, for context),
        }

    value_col defaults to BPV's "pred_vol_pct"; pass "iv_dynamic" (with
    iv_mode="harcj_only") to sweep the IV-only dynamic threshold instead —
    same grid mechanism, just bucketing a different series.
    """
    metric_names = ["total_pnl", "avg_daily_pnl", "sortino", "sharpe", "win_rate", "drawdown", "max_daily_loss"]
    grids = {m: pd.DataFrame(index=bucket_counts, columns=lookbacks, dtype=float)
              for m in metric_names}
    reliable = pd.DataFrame(index=bucket_counts, columns=lookbacks, dtype=bool)
    n_traded = pd.DataFrame(index=bucket_counts, columns=lookbacks, dtype=float)

    for L in lookbacks:
        for B in bucket_counts:
            if L >= len(df):
                for m in metric_names:
                    grids[m].loc[B, L] = np.nan
                reliable.loc[B, L] = False
                n_traded.loc[B, L] = np.nan
                continue

            result = walk_forward_run(
                df, lookback=L, n_buckets=B, method=method,
                exclusion_rule=exclusion_rule, n_exclude=n_exclude,
                value_col=value_col, iv_cutoff=iv_cutoff, iv_mode=iv_mode,
                min_days_per_bucket=min_days_per_bucket,
            )
            for m in metric_names:
                grids[m].loc[B, L] = result.metrics[m]
            reliable.loc[B, L] = result.reliable
            n_traded.loc[B, L] = result.metrics["n_traded"]

    return {"metrics": grids, "reliable": reliable, "n_traded": n_traded}


# ── PBO grid ───────────────────────────────────────────────────────────────────

def run_pbo_grid(
    df: pd.DataFrame,
    lookbacks: list[int],
    bucket_counts: list[int],
    static_daily_pnl: pd.Series,
    method: BucketMethod = "equal_count",
    exclusion_rule: str = "worst_pnl_dynamic",
    n_exclude: int = 1,
    value_col: str = "pred_vol_pct",
    iv_cutoff: float | None = None,
    iv_mode: str = "harcj_only",
    min_days_per_bucket: int = 10,
    cscv_metric: str = "sortino",
    max_S: int = 16,
    min_partition_size: int = 4,
) -> dict:
    """
    Probability of Backtest Overfitting (CSCV, Bailey et al. 2015), per grid
    cell, vs. today's static threshold.

    For each (L, B) cell: build a 2-column PnL matrix [this cell's dynamic
    daily PnL, the static-threshold baseline's daily PnL], both restricted to
    the cell's own scored date range (the shorter of the two — the dynamic
    cell always starts later, after its own lookback warm-up). Run CSCV on
    just those two "strategies" and report the resulting PBO: the
    probability that, choosing whichever of {this cell, static threshold}
    looks best using only in-sample data, that choice turns out to be the
    worse one out-of-sample.

    PBO close to 0 → this cell's apparent edge over the static threshold is
    likely real. PBO > 0.5 → likely noise (the "better" one in-sample is a
    coin flip against the static baseline out-of-sample).

    Returns the same shape as run_grid() (single metric "pbo") so the
    existing heatmap renderer works unchanged. "n_traded" is repurposed to
    hold n_combinations (context on how many IS/OOS splits the PBO is based
    on) rather than a trade count.
    """
    from pbo import compute_cscv

    pbo_grid = pd.DataFrame(index=bucket_counts, columns=lookbacks, dtype=float)
    reliable = pd.DataFrame(index=bucket_counts, columns=lookbacks, dtype=bool)
    n_combos = pd.DataFrame(index=bucket_counts, columns=lookbacks, dtype=float)

    for L in lookbacks:
        for B in bucket_counts:
            if L >= len(df):
                pbo_grid.loc[B, L] = np.nan
                reliable.loc[B, L]  = False
                n_combos.loc[B, L]  = np.nan
                continue

            result = walk_forward_run(
                df, lookback=L, n_buckets=B, method=method,
                exclusion_rule=exclusion_rule, n_exclude=n_exclude,
                value_col=value_col, iv_cutoff=iv_cutoff, iv_mode=iv_mode,
                min_days_per_bucket=min_days_per_bucket,
            )
            common_idx = result.daily_pnl.index
            static_aligned = static_daily_pnl.reindex(common_idx).fillna(0.0)
            M = np.column_stack([result.daily_pnl.values, static_aligned.values])

            cscv_result = compute_cscv(
                M, S=None, metric=cscv_metric,
                max_S=max_S, min_partition_size=min_partition_size,
            )
            pbo_grid.loc[B, L] = cscv_result["pbo"]
            reliable.loc[B, L] = result.reliable and cscv_result["warning"] is None
            n_combos.loc[B, L] = cscv_result["n_combinations"]

    return {"metrics": {"pbo": pbo_grid}, "reliable": reliable, "n_traded": n_combos}


# ── Composite score ────────────────────────────────────────────────────────────

# Direction each metric is ranked in (rank 1 = best):
#   avg_daily_pnl, sortino        — higher raw value = better
#   max_daily_loss                — higher raw value = better (stored <= 0;
#                                   "minimize the loss" means closest to zero,
#                                   i.e. the LESS negative / higher number —
#                                   ranking ascending on the raw signed value
#                                   would backwards-penalize a shallow loss
#                                   versus a deep one)
#   pbo                           — lower raw value = better
_RANK_ASCENDING = {"avg_daily_pnl": False, "sortino": False, "max_daily_loss": False, "pbo": True}


def composite_score_table(
    grid_result: dict,
    pbo_grid_result: dict,
    weights: dict,
) -> pd.DataFrame:
    """
    Combine avg_daily_pnl, sortino, max_daily_loss (from grid_result) and pbo
    (from pbo_grid_result, vs. the static threshold) into one weighted-rank
    composite score per (lookback, n_buckets) cell.

    Each metric is ranked independently across all cells (rank 1 = best, per
    _RANK_ASCENDING's direction for that metric — see module comment).
    weights (dict with keys "avg_daily_pnl", "sortino", "max_daily_loss", "pbo")
    are normalized to sum to 1 regardless of what's passed in, so the
    composite score stays a weighted AVERAGE of ranks (same scale as an
    individual rank), not something that drifts with arbitrary weight
    magnitudes.

    NaN cells (no data at that lookback/bucket-count) are dropped before
    ranking — they were never real variants. "reliable" is the AND of the
    underlying dynamic-grid and PBO-grid reliability flags for that cell,
    carried through for display, never used to silently drop a cell from
    the ranking itself.

    Returns a DataFrame sorted by composite score ascending (best first),
    or empty if no cell has data for all four metrics.
    """
    pnl_df = grid_result["metrics"]["avg_daily_pnl"]
    srt_df = grid_result["metrics"]["sortino"]
    mdl_df = grid_result["metrics"]["max_daily_loss"]
    pbo_df = pbo_grid_result["metrics"]["pbo"]
    rel_grid = grid_result["reliable"]
    rel_pbo  = pbo_grid_result["reliable"]

    rows = []
    for B in pnl_df.index:
        for L in pnl_df.columns:
            rows.append({
                "lookback":       L,
                "n_buckets":      B,
                "avg_daily_pnl":  pnl_df.loc[B, L],
                "sortino":        srt_df.loc[B, L],
                "max_daily_loss": mdl_df.loc[B, L],
                "pbo":            pbo_df.loc[B, L],
                "reliable":       bool(rel_grid.loc[B, L]) and bool(rel_pbo.loc[B, L]),
            })

    table = pd.DataFrame(rows).dropna(subset=["avg_daily_pnl", "sortino", "max_daily_loss", "pbo"])
    if table.empty:
        return table

    for m, ascending in _RANK_ASCENDING.items():
        table[f"rank_{m}"] = table[m].rank(ascending=ascending, method="average")

    w_sum = sum(weights.get(m, 0.0) for m in _RANK_ASCENDING) or 1.0
    table["composite"] = sum(
        (weights.get(m, 0.0) / w_sum) * table[f"rank_{m}"] for m in _RANK_ASCENDING
    )

    return table.sort_values("composite", ascending=True).reset_index(drop=True)


# ── Paired grid (and_both / or_either) ──────────────────────────────────────────
#
# For and_both/or_either, a "variant" is really a 4-tuple
# (harcj_lookback, harcj_buckets, iv_lookback, iv_buckets) — too many
# dimensions for a 2D heatmap. The paired grid below sweeps a single (L, B)
# pair applied to BOTH sides at once (harcj_L=iv_L=L, harcj_B=iv_B=B),
# keeping it a normal 2D grid. For a fully independent combination — any
# harcj(L,B) against any iv(L,B) — see the single non-swept lookup in
# run_paired_cell(), used by the "Specific combination" inspector.

def run_paired_grid(
    df: pd.DataFrame,
    lookbacks: list[int],
    bucket_counts: list[int],
    harcj_value_col: str = "pred_vol_pct",
    iv_value_col: str = "iv_dynamic",
    harcj_method: BucketMethod = "equal_count",
    harcj_exclusion_rule: str = "worst_pnl_dynamic",
    harcj_n_exclude: int = 1,
    iv_method: BucketMethod = "equal_count",
    iv_exclusion_rule: str = "worst_pnl_dynamic",
    iv_n_exclude: int = 1,
    iv_mode: str = "and_both",
    min_days_per_bucket: int = 10,
) -> dict:
    """
    Same (L, B) drives both BPV and IV simultaneously at each cell — but
    each side keeps its OWN bucketing method / exclusion rule / n_exclude
    (only the lookback and bucket count are forced equal, per the paired-
    grid design). Same return shape as run_grid().
    """
    metric_names = ["total_pnl", "avg_daily_pnl", "sortino", "sharpe", "win_rate", "drawdown", "max_daily_loss"]
    grids = {m: pd.DataFrame(index=bucket_counts, columns=lookbacks, dtype=float)
              for m in metric_names}
    reliable = pd.DataFrame(index=bucket_counts, columns=lookbacks, dtype=bool)
    n_traded = pd.DataFrame(index=bucket_counts, columns=lookbacks, dtype=float)

    for L in lookbacks:
        for B in bucket_counts:
            if L >= len(df):
                for m in metric_names:
                    grids[m].loc[B, L] = np.nan
                reliable.loc[B, L] = False
                n_traded.loc[B, L] = np.nan
                continue

            harcj_params = DualParams(lookback=L, n_buckets=B, method=harcj_method,
                                       exclusion_rule=harcj_exclusion_rule, n_exclude=harcj_n_exclude)
            iv_params = DualParams(lookback=L, n_buckets=B, method=iv_method,
                                    exclusion_rule=iv_exclusion_rule, n_exclude=iv_n_exclude)
            result = run_paired_cell(
                df, harcj_params, iv_params, harcj_value_col=harcj_value_col, iv_value_col=iv_value_col,
                iv_mode=iv_mode, min_days_per_bucket=min_days_per_bucket,
            )
            for m in metric_names:
                grids[m].loc[B, L] = result.metrics[m]
            reliable.loc[B, L] = result.reliable
            n_traded.loc[B, L] = result.metrics["n_traded"]

    return {"metrics": grids, "reliable": reliable, "n_traded": n_traded}


def run_paired_cell(
    df: pd.DataFrame,
    harcj_params: DualParams,
    iv_params: DualParams,
    harcj_value_col: str = "pred_vol_pct",
    iv_value_col: str = "iv_dynamic",
    iv_mode: str = "and_both",
    min_days_per_bucket: int = 10,
    return_diagnostics: bool = False,
) -> RunResult:
    """
    One non-swept variant: harcj(L,B) and iv(L,B) fully independent. Thin
    wrapper over walk_forward_run_dual — used both by run_paired_grid
    (called with harcj_params is iv_params) and directly by the "Specific
    combination" inspector (called with genuinely different params).
    """
    return walk_forward_run_dual(
        df, harcj=harcj_params, iv=iv_params,
        harcj_value_col=harcj_value_col, iv_value_col=iv_value_col,
        iv_mode=iv_mode, min_days_per_bucket=min_days_per_bucket,
        return_diagnostics=return_diagnostics,
    )


def run_paired_pbo_grid(
    df: pd.DataFrame,
    lookbacks: list[int],
    bucket_counts: list[int],
    static_daily_pnl: pd.Series,
    harcj_value_col: str = "pred_vol_pct",
    iv_value_col: str = "iv_dynamic",
    harcj_method: BucketMethod = "equal_count",
    harcj_exclusion_rule: str = "worst_pnl_dynamic",
    harcj_n_exclude: int = 1,
    iv_method: BucketMethod = "equal_count",
    iv_exclusion_rule: str = "worst_pnl_dynamic",
    iv_n_exclude: int = 1,
    iv_mode: str = "and_both",
    min_days_per_bucket: int = 10,
    cscv_metric: str = "sortino",
    max_S: int = 16,
    min_partition_size: int = 4,
) -> dict:
    """PBO counterpart to run_paired_grid — same paired (L, B) sweep (each
    side keeping its own method/exclusion_rule/n_exclude), each cell's
    dynamic PnL (from the combined BPV+IV decision) vs. the static
    threshold. Same return shape as run_pbo_grid()."""
    from pbo import compute_cscv

    pbo_grid = pd.DataFrame(index=bucket_counts, columns=lookbacks, dtype=float)
    reliable = pd.DataFrame(index=bucket_counts, columns=lookbacks, dtype=bool)
    n_combos = pd.DataFrame(index=bucket_counts, columns=lookbacks, dtype=float)

    for L in lookbacks:
        for B in bucket_counts:
            if L >= len(df):
                pbo_grid.loc[B, L] = np.nan
                reliable.loc[B, L]  = False
                n_combos.loc[B, L]  = np.nan
                continue

            harcj_params = DualParams(lookback=L, n_buckets=B, method=harcj_method,
                                       exclusion_rule=harcj_exclusion_rule, n_exclude=harcj_n_exclude)
            iv_params = DualParams(lookback=L, n_buckets=B, method=iv_method,
                                    exclusion_rule=iv_exclusion_rule, n_exclude=iv_n_exclude)
            result = run_paired_cell(
                df, harcj_params, iv_params, harcj_value_col=harcj_value_col, iv_value_col=iv_value_col,
                iv_mode=iv_mode, min_days_per_bucket=min_days_per_bucket,
            )
            common_idx = result.daily_pnl.index
            static_aligned = static_daily_pnl.reindex(common_idx).fillna(0.0)
            M = np.column_stack([result.daily_pnl.values, static_aligned.values])

            cscv_result = compute_cscv(
                M, S=None, metric=cscv_metric,
                max_S=max_S, min_partition_size=min_partition_size,
            )
            pbo_grid.loc[B, L] = cscv_result["pbo"]
            reliable.loc[B, L] = result.reliable and cscv_result["warning"] is None
            n_combos.loc[B, L] = cscv_result["n_combinations"]

    return {"metrics": {"pbo": pbo_grid}, "reliable": reliable, "n_traded": n_combos}


def pbo_for_result(
    result: RunResult,
    static_daily_pnl: pd.Series,
    cscv_metric: str = "sortino",
    max_S: int = 16,
    min_partition_size: int = 4,
) -> dict:
    """
    PBO for one already-computed RunResult vs. the static threshold — the
    single-cell counterpart to run_pbo_grid/run_paired_pbo_grid, used by the
    "Specific combination" inspector where harcj(L,B) and iv(L,B) are chosen
    independently rather than swept as a grid.
    """
    from pbo import compute_cscv

    common_idx = result.daily_pnl.index
    static_aligned = static_daily_pnl.reindex(common_idx).fillna(0.0)
    M = np.column_stack([result.daily_pnl.values, static_aligned.values])
    return compute_cscv(M, S=None, metric=cscv_metric, max_S=max_S, min_partition_size=min_partition_size)
