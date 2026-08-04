"""
Economic-release event days — CME_events_2026.csv-driven — and the blended
"default variant, except on event days use the 08:30-12:00 variant instead"
final PnL curve.

Mechanism (see conversation history): NDAQ trades from 06:00 by default,
SPX from 05:00 by default; on days with a high-impact economic release
(CPI, PPI, NFP, PCE, Job Openings/JOLTS, FOMC), the live system switches to
the 08:30-12:00 window instead. This module:
  1. Loads the event-day date set from the events CSV (economic_release
     rows only — holiday/expiry/dst_change rows don't represent a session
     shift, and are excluded).
  2. Ranks (mode, L, B) cells for the 08:30-12:00 variant using ONLY event
     days for scoring (not the walk-forward mechanics — those are unchanged,
     no lookahead either way; only which days count toward the metrics/PBO
     used to pick "best" changes).
  3. Builds the final blended daily PnL: for each date in the default
     variant's own scored range, if it's an event day AND the event variant
     has a decision for that date, use the event variant's outcome for that
     date instead of the default's.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import cutoff_transfer as ct
import dynamic_threshold as dt
import grid_cache as gc
from disk_cache import load_or_compute

_HERE = os.path.dirname(os.path.abspath(__file__))
_EVENTS_CSV = os.path.join(_HERE, "exchange_events", "CME_events_2026.csv")

# underlying -> (default source/profile, event-window source/profile)
SOURCE_PAIRS = {
    "spxw": {"default": "spxw_gamma_hedge_false", "event": "spxw_delta_condor_0830_1200"},
    "ndaq": {"default": "ndaq_delta_condor_0600_1000", "event": "ndaq_delta_condor_0830_1200"},
}


def load_event_dates() -> set:
    """Unique calendar dates of genuine economic-release rows (CPI/PPI/NFP/PCE/JOLTS/FOMC)."""
    df = pd.read_csv(_EVENTS_CSV)
    er = df[df["event_type"] == "economic_release"]
    return set(pd.to_datetime(er["date"]).dt.normalize().unique())


def event_day_ranking(source: str, profile: str, event_dates: set) -> pd.DataFrame:
    """
    Same composite-score global ranking as cutoff_transfer.global_ranking,
    but every cell's metrics/PBO are computed using ONLY the event-day subset
    of its own scored range — finding the (mode, L, B) that's best
    SPECIFICALLY on event days, not overall.

    Disk-cached: the per-cell walk-forward runs underneath are already
    cached (cutoff_transfer.cell_result_for), but CSCV itself isn't — its
    cost is dominated by the partition-combination count (fixed by max_S),
    not by how few event days there are, so it's genuinely expensive
    (~5-10 min for a full 112-cell sweep) and worth caching at this level
    too. event_dates itself isn't part of the cache key — it's a fixed
    function of exchange_events/CME_events_2026.csv, not a per-call choice.
    ct.LOOKBACKS/ct.BUCKETS ARE part of the key (even though they're
    currently module-level constants, not call args) — changing either
    default must invalidate this cache, not silently serve a stale ranking
    computed over a different grid.
    """
    args = (profile, ct.LOOKBACKS, ct.BUCKETS)
    return load_or_compute("event_day_ranking", args, lambda: _compute_event_day_ranking(source, profile, event_dates),
                            profile=profile, source=source)


def _compute_event_day_ranking(source: str, profile: str, event_dates: set) -> pd.DataFrame:
    from pbo import compute_cscv

    loader = gc._loader(source)
    cfg = loader.load_static_config(profile)
    canonical_mode, _always, static = ct.canonical_mode_and_baselines(source, profile)
    static_pnl = static["_daily_pnl"]

    rows = []
    for mode in dt.IV_MODES:
        for L in ct.LOOKBACKS:
            for B in ct.BUCKETS:
                if mode == "harcj_only" and L >= len(loader.load_joined(profile)):
                    continue
                result = ct.cell_result_for(source, profile, mode, L, B)
                idx = result.daily_pnl.index
                event_mask = idx.isin(event_dates)
                if event_mask.sum() < 4:
                    continue

                ev_pnl = result.daily_pnl[event_mask]
                ev_traded = result.traded_mask[event_mask]
                metrics = dt.summarize(ev_pnl, ev_traded)

                static_aligned = static_pnl.reindex(ev_pnl.index).fillna(0.0)
                M = np.column_stack([ev_pnl.values, static_aligned.values])
                cscv = compute_cscv(M, S=None, metric=ct.CSCV_METRIC,
                                     max_S=min(ct.CSCV_MAX_S, len(ev_pnl) // 2 or 1),
                                     min_partition_size=2)

                rows.append({
                    "mode": mode, "lookback": L, "n_buckets": B,
                    "avg_daily_pnl": metrics["avg_daily_pnl"],
                    "sortino": metrics["sortino"],
                    "max_daily_loss": metrics["max_daily_loss"],
                    "pbo": cscv["pbo"],
                    "reliable": cscv["warning"] is None,
                    "n_event_days": int(event_mask.sum()),
                })

    table = pd.DataFrame(rows).dropna(subset=["avg_daily_pnl", "sortino", "max_daily_loss", "pbo"])
    return dt._rank_and_weight(table, ct.WEIGHTS)


def best_event_cutoff(source: str, profile: str, event_dates: set) -> dict | None:
    table = event_day_ranking(source, profile, event_dates)
    if table.empty:
        return None
    top = table.iloc[0]
    return {"mode": top["mode"], "lookback": int(top["lookback"]), "n_buckets": int(top["n_buckets"])}


def blended_result(
    default_source: str, default_profile: str, default_config: dict,
    event_source: str, event_profile: str, event_config: dict,
    event_dates: set,
) -> dt.RunResult:
    """
    Default variant's own (mode, L, B) result everywhere, EXCEPT on event
    days where the event variant's own (mode, L, B) result is spliced in
    instead (only for event dates the event variant actually has a
    decision for — otherwise falls back to the default's own outcome for
    that date, e.g. if the event variant's own lookback warm-up hadn't
    started yet).
    """
    default_result = ct.cell_result_for(default_source, default_profile,
                                         default_config["mode"], default_config["lookback"], default_config["n_buckets"])
    event_result = ct.cell_result_for(event_source, event_profile,
                                       event_config["mode"], event_config["lookback"], event_config["n_buckets"])

    idx = default_result.daily_pnl.index
    is_event_day = pd.Series(idx.isin(event_dates), index=idx)
    overlap = idx[is_event_day & idx.isin(event_result.daily_pnl.index)]

    combined_pnl = default_result.daily_pnl.copy()
    combined_traded = default_result.traded_mask.copy()
    combined_pnl.loc[overlap] = event_result.daily_pnl.loc[overlap]
    combined_traded.loc[overlap] = event_result.traded_mask.loc[overlap]

    metrics = dt.summarize(combined_pnl, combined_traded)
    return dt.RunResult(
        metrics=metrics, reliable=True, avg_days_per_bucket=float("nan"),
        daily_pnl=metrics["_daily_pnl"], traded_mask=metrics["_traded_mask"],
    ), len(overlap)


def live_baseline_result(underlying: str, event_dates: set) -> dt.RunResult:
    """
    "Current live baseline" — NOT a walk-forward dynamic cutoff, a FIXED
    rule using this underlying's actual deployed thresholds
    (dashboard_new/params/<...>_persistence.json's harcj_exclusion_ranges/
    iv_cutoff, read via load_static_config — real numbers, not approximated):

      non-event days: exclude if yesterday's BPV (pred_vol_pct, computed on
        the 08:30-window event variant's own series) falls in the live
        exclusion range — BPV alone.
      event days: exclude if BPV is excluded OR mean-IV (07:30-08:30, same
        event variant's own series) exceeds the live iv_cutoff — i.e. trade
        only if BOTH pass (and_both).

    Same rule STRUCTURE applied to both SPX and NDAQ, each with its own
    numbers — a deliberate hypothesis to test, not a literal reproduction
    of each underlying's actual current flag_combine_method (SPX's real
    live config is actually iv_only/ignore_harcj, not this BPV+IV mix).

    PnL sourcing is the same blend as blended_result(): the event variant's
    PnL on event days, the default variant's PnL otherwise — just gated by
    this fixed rule instead of a walk-forward "best cutoff".
    """
    pair = SOURCE_PAIRS[underlying]
    default_source = pair["default"]
    event_source = pair["event"]

    loader = gc._loader(event_source)
    cfg = loader.load_static_config(event_source)
    exclusion_ranges = cfg["exclusion_ranges"]
    iv_cutoff = cfg["iv_cutoff"]

    event_df = loader.load_joined(event_source)
    excluded = pd.Series(False, index=event_df.index)
    for lo, hi in exclusion_ranges:
        excluded |= event_df["pred_vol_pct"].between(lo, hi)
    harcj_trade = ~excluded
    iv_trade = event_df["mean_iv"] <= iv_cutoff

    is_event_on_event_df = pd.Series(event_df.index.isin(event_dates), index=event_df.index)
    rule_trade = harcj_trade.where(~is_event_on_event_df, harcj_trade & iv_trade)

    default_df = gc._loader(default_source).load_joined(default_source)
    idx = default_df.index
    is_event = pd.Series(idx.isin(event_dates), index=idx)

    combined_traded = pd.Series(True, index=idx)
    common_idx = idx[idx.isin(rule_trade.index)]
    combined_traded.loc[common_idx] = rule_trade.loc[common_idx].values

    combined_pnl = default_df["pnl"].copy()
    overlap = idx[is_event & idx.isin(event_df.index)]
    combined_pnl.loc[overlap] = event_df.loc[overlap, "pnl"]
    combined_pnl = combined_pnl.where(combined_traded, 0.0)

    metrics = dt.summarize(combined_pnl, combined_traded)
    return dt.RunResult(
        metrics=metrics, reliable=True, avg_days_per_bucket=float("nan"),
        daily_pnl=metrics["_daily_pnl"], traded_mask=metrics["_traded_mask"],
    )
