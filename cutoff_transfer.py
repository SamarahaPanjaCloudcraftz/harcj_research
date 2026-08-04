"""
Backend for the "Cutoff Transfer" page — testing whether the set of
calendar dates one source's best cutoff skips is, applied verbatim to a
DIFFERENT source's own PnL, better than that other source's own best
cutoff criteria.

"Applying a cutoff from A to B" here means exactly what the user specified:
NOT re-running A's (mode, L, B) recipe on B's own signals (that would just
be B's own dynamic-threshold grid, already covered elsewhere). It means:
take the literal set of calendar dates A's walk-forward decision skipped,
and skip those SAME dates in B — a pure date-mask transplant, testing
whether cross-market/cross-variant "bad days" agree, not whether a
strategy-shape generalizes.

Every source's own "best" cutoff = the #1 row of its Global Ranking (same
composite-score mechanism as the Summary page, PBO measured against that
source's own live-configured baseline), computed with the SAME default
settings used everywhere else in the dashboard (method=equal_count,
n_exclude=1, paired grid — not independent BPV/IV — for and_both/or_either,
default IV source/window). These defaults match warm_cache.py's, so
looking up a source's best cutoff is typically an instant cache hit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import backtest_source
import data_loader
import dynamic_threshold as dt
import grid_cache as gc

LOOKBACKS = (60, 90, 120, 180)
BUCKETS   = (4, 5, 8, 10)
METHOD    = "equal_count"
RULE      = "highest_vol"
N_EXCLUDE = 1
MIN_DAYS_PER_BUCKET = 10

CSCV_METRIC = "sortino"
CSCV_MAX_S  = 16
CSCV_MIN_PARTITION = 4

WEIGHTS = {"pbo": 0.49, "avg_daily_pnl": 0.17, "sortino": 0.17, "max_daily_loss": 0.17}

_LIVE_MODE_MAP = {"ignore_harcj": "iv_only", "or": "and_both", "and": "or_either"}

MODE_LABELS = {
    "harcj_only": "HARCJ only",
    "iv_only":    "IV only",
    "and_both":   "AND (both must pass)",
    "or_either":  "OR (either passing is enough)",
}


def default_iv_window(source: str, profile: str) -> tuple[str, str]:
    """Same per-source default IV window logic used throughout the dashboard."""
    loader = gc._loader(source)
    if isinstance(loader, backtest_source.BacktestSource):
        start, _end = loader.derive_session_window()
        window_start = (
            pd.Timestamp(start) - pd.Timedelta(hours=backtest_source.DEFAULT_IV_WINDOW_HOURS_BEFORE_START)
        ).strftime("%H:%M")
        return window_start, start
    return "07:30", "08:30"


def canonical_mode_and_baselines(source: str, profile: str):
    """(canonical_mode, always_baseline, static_baseline) — the source's own live-equivalent reference."""
    loader = gc._loader(source)
    cfg = loader.load_static_config(profile)
    canonical_mode = _LIVE_MODE_MAP.get(cfg["flag_combine_method"], "harcj_only")
    always, static, _ = gc.cached_baselines(source, profile, canonical_mode)
    return canonical_mode, always, static


def global_ranking(source: str, profile: str) -> pd.DataFrame:
    """
    Same computation as the Summary page's Global ranking, using the
    dashboard-wide default settings — every (mode, L, B) cell across all 4
    modes, PBO measured against this source's own live-equivalent baseline.
    """
    loader = gc._loader(source)
    cfg = loader.load_static_config(profile)
    canonical_mode = _LIVE_MODE_MAP.get(cfg["flag_combine_method"], "harcj_only")
    iv_start, iv_end = default_iv_window(source, profile)

    grid_h = gc.cached_grid(source, profile, LOOKBACKS, BUCKETS, METHOD, RULE, N_EXCLUDE,
                             cfg["iv_cutoff"], "harcj_only", MIN_DAYS_PER_BUCKET)
    pbo_h = gc.cached_pbo_grid_vs_mode(source, profile, LOOKBACKS, BUCKETS, METHOD, RULE, N_EXCLUDE,
                                       cfg["iv_cutoff"], "harcj_only", MIN_DAYS_PER_BUCKET,
                                       canonical_mode, CSCV_METRIC, CSCV_MAX_S, CSCV_MIN_PARTITION)

    grid_iv = gc.cached_iv_grid(source, profile, "mean", None, iv_start, iv_end,
                                LOOKBACKS, BUCKETS, METHOD, RULE, N_EXCLUDE, MIN_DAYS_PER_BUCKET)
    pbo_iv = gc.cached_iv_pbo_grid_vs_mode(source, profile, "mean", None, iv_start, iv_end,
                                           LOOKBACKS, BUCKETS, METHOD, RULE, N_EXCLUDE, MIN_DAYS_PER_BUCKET,
                                           canonical_mode, CSCV_METRIC, CSCV_MAX_S, CSCV_MIN_PARTITION)

    mode_grids = {"harcj_only": (grid_h, pbo_h), "iv_only": (grid_iv, pbo_iv)}
    for mode in ("and_both", "or_either"):
        grid_p = gc.cached_paired_grid(source, profile, "mean", None, iv_start, iv_end,
                                        LOOKBACKS, BUCKETS, METHOD, RULE, N_EXCLUDE, METHOD, RULE, N_EXCLUDE,
                                        mode, MIN_DAYS_PER_BUCKET)
        pbo_p = gc.cached_paired_pbo_grid_vs_mode(source, profile, "mean", None, iv_start, iv_end,
                                                   LOOKBACKS, BUCKETS, METHOD, RULE, N_EXCLUDE, METHOD, RULE, N_EXCLUDE,
                                                   mode, MIN_DAYS_PER_BUCKET, canonical_mode,
                                                   CSCV_METRIC, CSCV_MAX_S, CSCV_MIN_PARTITION)
        mode_grids[mode] = (grid_p, pbo_p)

    return dt.global_composite_score_table(mode_grids, WEIGHTS)


def best_cutoff(source: str, profile: str) -> dict | None:
    """{"mode", "lookback", "n_buckets"} for this source's #1 Global Ranking row, or None if empty."""
    table = global_ranking(source, profile)
    if table.empty:
        return None
    top = table.iloc[0]
    return {"mode": top["mode"], "lookback": int(top["lookback"]), "n_buckets": int(top["n_buckets"])}


def cell_result_for(source: str, profile: str, mode: str, L: int, B: int,
                     iv_source_mode: str = "mean", iv_timestamp=None,
                     iv_start_time: str | None = None, iv_end_time: str | None = None) -> dt.RunResult:
    """
    This (mode, L, B) cell's own walk-forward result on `source`'s own data,
    at default method/n_exclude (or caller-supplied IV window) — used to
    read off .traded_mask (the actual cutoff decision).
    """
    loader = gc._loader(source)
    cfg = loader.load_static_config(profile)
    if iv_start_time is None or iv_end_time is None:
        iv_start_time, iv_end_time = default_iv_window(source, profile)

    if mode == "harcj_only":
        return gc.cached_cell(source, profile, L, B, METHOD, RULE, N_EXCLUDE,
                               cfg["iv_cutoff"], "harcj_only", MIN_DAYS_PER_BUCKET)
    if mode == "iv_only":
        return gc.cached_iv_cell(source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
                                  L, B, METHOD, RULE, N_EXCLUDE, MIN_DAYS_PER_BUCKET)
    return gc.cached_paired_cell(source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
                                  L, B, L, B, METHOD, RULE, N_EXCLUDE, METHOD, RULE, N_EXCLUDE,
                                  mode, MIN_DAYS_PER_BUCKET)


def cutoff_dates_for(source: str, profile: str, mode: str, L: int, B: int, **iv_kwargs) -> set:
    """The set of calendar dates this (source, mode, L, B) cell decided to skip."""
    result = cell_result_for(source, profile, mode, L, B, **iv_kwargs)
    mask = result.traded_mask
    return set(mask.index[~mask])


def apply_transplanted_cutoff(target_source: str, target_profile: str, cutoff_dates: set) -> dt.RunResult:
    """
    Zero out `target`'s own raw PnL on any date in `cutoff_dates` that's
    within target's own scored range; trade every other day normally
    (including any date target has that origin never scored at all).
    """
    loader = gc._loader(target_source)
    df = loader.load_joined(target_profile)
    raw_pnl = df["pnl"]
    traded_mask = pd.Series(~df.index.isin(cutoff_dates), index=df.index)
    daily_pnl = raw_pnl.where(traded_mask, 0.0)
    metrics = dt.summarize(daily_pnl, traded_mask)
    return dt.RunResult(
        metrics=metrics, reliable=True, avg_days_per_bucket=float("nan"),
        daily_pnl=metrics["_daily_pnl"], traded_mask=metrics["_traded_mask"],
    )


def transplant_pbo(target_source: str, target_profile: str, transplanted_result: dt.RunResult) -> dict:
    """PBO of the transplanted result vs. target's own live-equivalent static baseline."""
    _canonical_mode, _always, static = canonical_mode_and_baselines(target_source, target_profile)
    return dt.pbo_for_result(transplanted_result, static["_daily_pnl"], CSCV_METRIC, CSCV_MAX_S, CSCV_MIN_PARTITION)


def restrict_result_to_mask(result: dt.RunResult, mask_fn) -> dt.RunResult:
    """
    Re-summarize a RunResult over a subset of its own dates (e.g. only
    economic-release event days, or only non-event days) — used by the
    Cutoff Transfer page's event-day-only / non-event-day-only sections.
    `mask_fn(index) -> bool array` decides which of the result's OWN dates
    survive; PBO for the restricted result should be recomputed separately
    via transplant_pbo (it reindexes the static baseline to whatever index
    the passed-in result has, so restricting first is enough).
    """
    idx = result.daily_pnl.index
    mask = mask_fn(idx)
    pnl = result.daily_pnl[mask]
    traded = result.traded_mask[mask]
    metrics = dt.summarize(pnl, traded)
    return dt.RunResult(
        metrics=metrics, reliable=True, avg_days_per_bucket=float("nan"),
        daily_pnl=metrics["_daily_pnl"], traded_mask=metrics["_traded_mask"],
    )


def restrict_metrics_to_mask(metrics: dict, mask_fn) -> dict:
    """Same as restrict_result_to_mask but for a plain metrics dict (always-trade / live-baseline)."""
    idx = metrics["_daily_pnl"].index
    mask = mask_fn(idx)
    pnl = metrics["_daily_pnl"][mask]
    traded = metrics["_traded_mask"][mask]
    return dt.summarize(pnl, traded)
