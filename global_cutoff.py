"""
Global Cutoff Search — implements docs/global_cutoff_framework.md.

Stage 0: reuse cutoff_transfer.py's cached cell/transplant machinery — a
candidate's own decision depends only on (origin, mode, L, B), never on
the target, so there are only ~256 distinct walk-forward cells in the
entire framework, shared across every target via cutoff_transfer.py's
existing disk caching.

Stage 1: exhaustive 256x256 blend per index (event-slice choice x
non-event-slice choice), cheap metrics only (avg_daily_pnl, sortino,
max_daily_loss, drawdown) — no CSCV, nothing pre-filtered.

Stage 1.5/2: finalists = union(global top-K, per-pool top-C), CSCV/PBO
computed for the finalist union against both reference baselines
(current live baseline, no-cutoff) — two rankings per index.

Everything expensive is disk-cached via disk_cache.load_or_compute under
source="global_cutoff", profile=<index_key> — a partially-run batch can
be resumed without repeating already-cached stages.
"""

from __future__ import annotations

import time

import pandas as pd

import backtest_source
import cutoff_transfer as ct
import dynamic_threshold as dt
import exchange_events as ee
import grid_cache as gc
from disk_cache import load_or_compute

LOOKBACKS = ct.LOOKBACKS
BUCKETS = ct.BUCKETS
MODES = ("harcj_only", "iv_only", "and_both", "or_either")

TOP_K = 50
POOL_C = 2

# Local to this module — dt._RANK_ASCENDING/_rank_and_weight are shared by
# pages whose tables have no "drawdown" column, so Stage 1's cheap ranking
# (which does include drawdown) gets its own copy rather than mutating the
# shared dict.
_CHEAP_RANK_ASCENDING = {
    "avg_daily_pnl": False, "sortino": False, "max_daily_loss": False, "drawdown": False,
}
CHEAP_WEIGHTS = {"avg_daily_pnl": 0.25, "sortino": 0.25, "max_daily_loss": 0.25, "drawdown": 0.25}

_POOL_COLS = ["event_origin", "event_mode", "non_event_origin", "non_event_mode"]
_KEY_COLS = ["event_origin", "event_mode", "event_L", "event_B",
             "non_event_origin", "non_event_mode", "non_event_L", "non_event_B"]


def _cheap_rank_and_weight(table: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Same mechanism as dt._rank_and_weight, scoped to this module's 4 cheap metrics."""
    if table.empty:
        return table
    table = table.copy()
    for m, ascending in _CHEAP_RANK_ASCENDING.items():
        table[f"rank_{m}"] = table[m].rank(ascending=ascending, method="average")
    w_sum = sum(weights.get(m, 0.0) for m in _CHEAP_RANK_ASCENDING) or 1.0
    table["composite"] = sum(
        (weights.get(m, 0.0) / w_sum) * table[f"rank_{m}"] for m in _CHEAP_RANK_ASCENDING
    )
    return table.sort_values("composite", ascending=True).reset_index(drop=True)


def slice_candidate_keys(lookbacks=LOOKBACKS, buckets=BUCKETS, modes=MODES) -> list[tuple[str, str, int, int]]:
    """
    All valid (origin, mode, L, B) — the same universal candidate pool used
    for BOTH day-type slices of BOTH indices (a candidate's own decision
    never depends on the target it's later transplanted onto).
    """
    keys = []
    for origin in backtest_source.ALL_SOURCES:
        origin_name = origin.DATA_SOURCE_NAME
        n_days = len(origin.load_joined(origin_name))
        for mode in modes:
            for L in lookbacks:
                if mode == "harcj_only" and L >= n_days:
                    continue
                for B in buckets:
                    keys.append((origin_name, mode, L, B))
    return keys


def _slice_result_for(target_source: str, origin_name: str, mode: str, L: int, B: int) -> dt.RunResult:
    """One candidate's cutoff applied to target's own PnL — target's own cell if self-origin, a date-mask transplant otherwise."""
    if origin_name == target_source:
        return ct.cell_result_for(target_source, target_source, mode, L, B)
    cutoff_dates = ct.cutoff_dates_for(origin_name, origin_name, mode, L, B)
    return ct.apply_transplanted_cutoff(target_source, target_source, cutoff_dates)


def slice_results_for_target(target_source: str, lookbacks=LOOKBACKS, buckets=BUCKETS, modes=MODES) -> dict[tuple, dt.RunResult]:
    """{(origin, mode, L, B): RunResult} for every candidate, applied to one target."""
    results = {}
    for key in slice_candidate_keys(lookbacks, buckets, modes):
        origin_name, mode, L, B = key
        results[key] = _slice_result_for(target_source, origin_name, mode, L, B)
    return results


def _blend(event_result: dt.RunResult, non_event_result: dt.RunResult, event_dates: set) -> dt.RunResult:
    """
    Splice event-slice and non-event-slice results into one full-year
    series — event/non-event dates are disjoint, so this is pure
    concatenation, not new computation. Same mechanism as
    exchange_events.blended_result/live_baseline_result.
    """
    idx = non_event_result.daily_pnl.index
    is_event = pd.Series(idx.isin(event_dates), index=idx)
    overlap = idx[is_event & idx.isin(event_result.daily_pnl.index)]
    combined_pnl = non_event_result.daily_pnl.copy()
    combined_traded = non_event_result.traded_mask.copy()
    combined_pnl.loc[overlap] = event_result.daily_pnl.loc[overlap]
    combined_traded.loc[overlap] = event_result.traded_mask.loc[overlap]
    metrics = dt.summarize(combined_pnl, combined_traded)
    return dt.RunResult(
        metrics=metrics, reliable=True, avg_days_per_bucket=float("nan"),
        daily_pnl=metrics["_daily_pnl"], traded_mask=metrics["_traded_mask"],
    )


def always_trade_blend(index_key: str, event_dates: set) -> dt.RunResult:
    """
    "No-cutoff" reference baseline — the index-level blend (event source's
    raw PnL on event days, default source's raw PnL otherwise) with every
    day traded. Same shape/index as every searched cutoff, so it's a valid
    CSCV comparison partner.
    """
    pair = ee.SOURCE_PAIRS[index_key]
    event_df = gc._loader(pair["event"]).load_joined(pair["event"])
    default_df = gc._loader(pair["default"]).load_joined(pair["default"])
    idx = default_df.index
    is_event = pd.Series(idx.isin(event_dates), index=idx)
    combined_pnl = default_df["pnl"].copy()
    overlap = idx[is_event & idx.isin(event_df.index)]
    combined_pnl.loc[overlap] = event_df.loc[overlap, "pnl"]
    combined_traded = pd.Series(True, index=idx)
    metrics = dt.summarize(combined_pnl, combined_traded)
    return dt.RunResult(
        metrics=metrics, reliable=True, avg_days_per_bucket=float("nan"),
        daily_pnl=metrics["_daily_pnl"], traded_mask=metrics["_traded_mask"],
    )


def stage1_table(index_key: str, event_dates: set, lookbacks=LOOKBACKS, buckets=BUCKETS, modes=MODES) -> pd.DataFrame:
    """Exhaustive event-slice x non-event-slice blend for one index — cheap metrics only, no CSCV, nothing pre-filtered."""
    pair = ee.SOURCE_PAIRS[index_key]
    event_slices = slice_results_for_target(pair["event"], lookbacks, buckets, modes)
    non_event_slices = slice_results_for_target(pair["default"], lookbacks, buckets, modes)

    rows = []
    for e_key, e_result in event_slices.items():
        for ne_key, ne_result in non_event_slices.items():
            m = _blend(e_result, ne_result, event_dates).metrics
            rows.append({
                "event_origin": e_key[0], "event_mode": e_key[1], "event_L": e_key[2], "event_B": e_key[3],
                "non_event_origin": ne_key[0], "non_event_mode": ne_key[1], "non_event_L": ne_key[2], "non_event_B": ne_key[3],
                "avg_daily_pnl": m["avg_daily_pnl"], "sortino": m["sortino"],
                "max_daily_loss": m["max_daily_loss"], "drawdown": m["drawdown"],
            })
    return pd.DataFrame(rows)


def select_finalists(table: pd.DataFrame, top_k: int = TOP_K, pool_c: int = POOL_C) -> pd.DataFrame:
    """
    Union of the global top-K by cheap composite score, and the top-C from
    each of the (event_origin,event_mode,non_event_origin,non_event_mode)
    pools — guarantees both the overall best performers AND full pool
    coverage advance to CSCV, deduplicated.
    """
    ranked = _cheap_rank_and_weight(table, CHEAP_WEIGHTS)
    global_top = ranked.head(top_k)
    pool_top = ranked.groupby(_POOL_COLS, group_keys=False).apply(lambda g: g.head(pool_c))
    finalists = pd.concat([global_top, pool_top]).drop_duplicates(subset=_KEY_COLS).reset_index(drop=True)
    return finalists


def stage2_finalist_results(index_key: str, finalists: pd.DataFrame, event_dates: set) -> dict[tuple, dt.RunResult]:
    """Rebuild the full blended RunResult (with daily_pnl/traded_mask) for each finalist — cheap, cached cells + fast splice."""
    pair = ee.SOURCE_PAIRS[index_key]
    results = {}
    for _, row in finalists.iterrows():
        key = tuple(row[c] if not c.endswith(("_L", "_B")) else int(row[c]) for c in _KEY_COLS)
        e_result = _slice_result_for(pair["event"], row["event_origin"], row["event_mode"], int(row["event_L"]), int(row["event_B"]))
        ne_result = _slice_result_for(pair["default"], row["non_event_origin"], row["non_event_mode"], int(row["non_event_L"]), int(row["non_event_B"]))
        results[key] = _blend(e_result, ne_result, event_dates)
    return results


def rank_finalists_vs_baseline(finalist_results: dict[tuple, dt.RunResult], baseline_daily_pnl: pd.Series,
                                progress_label: str = "", progress_every: int = 50) -> pd.DataFrame:
    """CSCV/PBO for every finalist vs. one baseline series, then the standard composite ranking (dt.WEIGHTS, includes pbo)."""
    rows = []
    n = len(finalist_results)
    t0 = time.time()
    for i, (key, result) in enumerate(finalist_results.items()):
        m = result.metrics
        pbo = dt.pbo_for_result(result, baseline_daily_pnl, ct.CSCV_METRIC, ct.CSCV_MAX_S, ct.CSCV_MIN_PARTITION)
        rows.append({
            "event_origin": key[0], "event_mode": key[1], "event_L": key[2], "event_B": key[3],
            "non_event_origin": key[4], "non_event_mode": key[5], "non_event_L": key[6], "non_event_B": key[7],
            "avg_daily_pnl": m["avg_daily_pnl"], "sortino": m["sortino"], "max_daily_loss": m["max_daily_loss"],
            "pbo": pbo["pbo"], "reliable": pbo["warning"] is None,
        })
        if progress_label and (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            print(f"  [{progress_label}] {i+1}/{n} CSCV calls, {elapsed:.1f}s elapsed, "
                  f"~{elapsed/(i+1)*(n-i-1):.1f}s remaining")
    table = pd.DataFrame(rows)
    return dt._rank_and_weight(table, ct.WEIGHTS)


def run_full_search(index_key: str, top_k: int = TOP_K, pool_c: int = POOL_C,
                     lookbacks=LOOKBACKS, buckets=BUCKETS, modes=MODES) -> dict:
    """Runs the full Stage 0-2 pipeline for one index, disk-cached at every stage."""
    event_dates = ee.load_event_dates()

    t0 = time.time()
    stage1 = load_or_compute("stage1_table", (lookbacks, buckets, modes),
                              lambda: stage1_table(index_key, event_dates, lookbacks, buckets, modes),
                              profile=index_key, source="global_cutoff")
    print(f"[{index_key}] Stage 1: {len(stage1)} blended cutoffs, {time.time()-t0:.1f}s")

    t0 = time.time()
    finalists = load_or_compute("finalists", (lookbacks, buckets, modes, top_k, pool_c),
                                 lambda: select_finalists(stage1, top_k, pool_c),
                                 profile=index_key, source="global_cutoff")
    print(f"[{index_key}] Finalists: {len(finalists)} (upper bound {top_k + 16*16*pool_c}), {time.time()-t0:.1f}s")

    t0 = time.time()
    finalist_results = load_or_compute("finalist_results", (lookbacks, buckets, modes, top_k, pool_c),
                                        lambda: stage2_finalist_results(index_key, finalists, event_dates),
                                        profile=index_key, source="global_cutoff")
    print(f"[{index_key}] Finalist results rebuilt: {time.time()-t0:.1f}s")

    live_baseline = ee.live_baseline_result(index_key, event_dates)
    no_cutoff_baseline = load_or_compute("no_cutoff_baseline", (), lambda: always_trade_blend(index_key, event_dates),
                                          profile=index_key, source="global_cutoff")

    t0 = time.time()
    ranking_vs_live = load_or_compute(
        "ranking_vs_live", (lookbacks, buckets, modes, top_k, pool_c),
        lambda: rank_finalists_vs_baseline(finalist_results, live_baseline.metrics["_daily_pnl"],
                                            progress_label=f"{index_key} vs live"),
        profile=index_key, source="global_cutoff",
    )
    print(f"[{index_key}] Ranking vs current live baseline: {time.time()-t0:.1f}s")

    t0 = time.time()
    ranking_vs_no_cutoff = load_or_compute(
        "ranking_vs_no_cutoff", (lookbacks, buckets, modes, top_k, pool_c),
        lambda: rank_finalists_vs_baseline(finalist_results, no_cutoff_baseline.metrics["_daily_pnl"],
                                            progress_label=f"{index_key} vs no-cutoff"),
        profile=index_key, source="global_cutoff",
    )
    print(f"[{index_key}] Ranking vs no-cutoff baseline: {time.time()-t0:.1f}s")

    return {
        "stage1": stage1, "finalists": finalists, "finalist_results": finalist_results,
        "live_baseline": live_baseline, "no_cutoff_baseline": no_cutoff_baseline,
        "ranking_vs_live": ranking_vs_live, "ranking_vs_no_cutoff": ranking_vs_no_cutoff,
    }
