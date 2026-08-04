"""
Pre-computes the exact default sidebar view for both pages (Threshold Grid
and Summary), for EVERY (source, profile) pair — not just whichever one is
index=0 in the selectboxes — switching either selectbox is a one-click
sidebar action, so it should be just as fast as the default. Run this
manually after changing defaults or clearing dashboard_data/:

    conda run -n harcj_research python3 warm_cache.py

Not run automatically by the app — this is a one-off/occasional warm-up
script, not part of the request path.
"""

from __future__ import annotations

import time

import pandas as pd

import data_loader
import backtest_source
import grid_cache as gc

# ── Exact sidebar defaults (must match pages/2_threshold_grid.py and
#    pages/3_summary.py — see their st.text_input/st.radio default values) ──

LOOKBACKS = (60, 90, 120, 180)
BUCKETS   = (4, 5, 8, 10)
METHOD    = "equal_count"          # first option in the bucketing-method radio
RULE      = "highest_vol"          # first option, now the default (per user request)
N_EXCLUDE = 1
MIN_DAYS_PER_BUCKET = 10

IV_LOOKBACKS   = LOOKBACKS         # same default string as BPV's
IV_BUCKETS     = BUCKETS
IV_METHOD      = "equal_count"
IV_RULE        = "highest_vol"
IV_N_EXCLUDE   = 1

CSCV_METRIC = "sortino"
CSCV_MAX_S  = 16
CSCV_MIN_PARTITION = 4

_LIVE_MODE_MAP = {"ignore_harcj": "iv_only", "or": "and_both", "and": "or_either"}

_LOADERS = {data_loader.DATA_SOURCE_NAME: data_loader}
_LOADERS.update({src.DATA_SOURCE_NAME: src for src in backtest_source.ALL_SOURCES})

# Per-source IV defaults (must match the sidebar's per-source default logic
# in both pages): live_dumps has a fixed 07:30-08:30 dump window, every
# backtest source derives its own session and defaults to the hour before it.
_IV_SOURCE_MODE = {name: "mean" for name in _LOADERS}       # first option in the IV-source radio


def _iv_mean_window(source: str) -> tuple[str, str]:
    loader = _LOADERS[source]
    if isinstance(loader, backtest_source.BacktestSource):
        start, _end = loader.derive_session_window()
        window_start = (
            pd.Timestamp(start) - pd.Timedelta(hours=backtest_source.DEFAULT_IV_WINDOW_HOURS_BEFORE_START)
        ).strftime("%H:%M")
        return window_start, start
    return "07:30", "08:30"


def _timed(label, fn):
    t0 = time.time()
    fn()
    print(f"  {label}: {time.time() - t0:.1f}s")


def _warm_profile(source: str, profile: str):
    loader = _LOADERS[source]
    cfg = loader.load_static_config(profile)
    iv_source_mode = _IV_SOURCE_MODE[source]
    iv_timestamp = None
    iv_start_time, iv_end_time = _iv_mean_window(source)

    print(f"Warming cache for source={source!r} profile={profile!r}")

    # ── Threshold Grid page default view: harcj_only only ──────────────────
    print("Threshold Grid — harcj_only:")
    _timed("baselines", lambda: gc.cached_baselines(source, profile, "harcj_only"))
    _timed("grid", lambda: gc.cached_grid(
        source, profile, LOOKBACKS, BUCKETS, METHOD, RULE, N_EXCLUDE,
        cfg["iv_cutoff"], "harcj_only", MIN_DAYS_PER_BUCKET,
    ))
    _timed("pbo_grid", lambda: gc.cached_pbo_grid(
        source, profile, LOOKBACKS, BUCKETS, METHOD, RULE, N_EXCLUDE,
        cfg["iv_cutoff"], "harcj_only", MIN_DAYS_PER_BUCKET,
        CSCV_METRIC, CSCV_MAX_S, CSCV_MIN_PARTITION,
    ))
    _timed(f"default cell L={LOOKBACKS[0]},B={BUCKETS[0]}", lambda: gc.cached_cell(
        source, profile, LOOKBACKS[0], BUCKETS[0], METHOD, RULE, N_EXCLUDE,
        cfg["iv_cutoff"], "harcj_only", MIN_DAYS_PER_BUCKET,
    ))

    canonical_mode = _LIVE_MODE_MAP.get(cfg["flag_combine_method"], "harcj_only")

    # ── Summary page default view: all 4 modes, PBO vs. the ONE live baseline ──
    print("Summary — harcj_only (PBO vs. live baseline):")
    _timed("pbo_grid_vs_mode", lambda: gc.cached_pbo_grid_vs_mode(
        source, profile, LOOKBACKS, BUCKETS, METHOD, RULE, N_EXCLUDE,
        cfg["iv_cutoff"], "harcj_only", MIN_DAYS_PER_BUCKET,
        canonical_mode, CSCV_METRIC, CSCV_MAX_S, CSCV_MIN_PARTITION,
    ))

    print("Summary — iv_only:")
    _timed("baselines", lambda: gc.cached_iv_baselines(
        source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
    ))
    _timed("grid", lambda: gc.cached_iv_grid(
        source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
        IV_LOOKBACKS, IV_BUCKETS, IV_METHOD, IV_RULE, IV_N_EXCLUDE, MIN_DAYS_PER_BUCKET,
    ))
    _timed("pbo_grid_vs_mode", lambda: gc.cached_iv_pbo_grid_vs_mode(
        source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
        IV_LOOKBACKS, IV_BUCKETS, IV_METHOD, IV_RULE, IV_N_EXCLUDE, MIN_DAYS_PER_BUCKET,
        canonical_mode, CSCV_METRIC, CSCV_MAX_S, CSCV_MIN_PARTITION,
    ))
    _timed(f"default cell L={LOOKBACKS[0]},B={BUCKETS[0]}", lambda: gc.cached_iv_cell(
        source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
        IV_LOOKBACKS[0], IV_BUCKETS[0], IV_METHOD, IV_RULE, IV_N_EXCLUDE, MIN_DAYS_PER_BUCKET,
    ))

    for mode in ("and_both", "or_either"):
        print(f"Summary — {mode}:")
        _timed("baselines", lambda: gc.cached_paired_baselines(
            source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time, mode,
        ))
        _timed("paired grid", lambda: gc.cached_paired_grid(
            source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
            LOOKBACKS, BUCKETS, METHOD, RULE, N_EXCLUDE, IV_METHOD, IV_RULE, IV_N_EXCLUDE,
            mode, MIN_DAYS_PER_BUCKET,
        ))
        _timed("paired pbo_grid_vs_mode", lambda: gc.cached_paired_pbo_grid_vs_mode(
            source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
            LOOKBACKS, BUCKETS, METHOD, RULE, N_EXCLUDE, IV_METHOD, IV_RULE, IV_N_EXCLUDE,
            mode, MIN_DAYS_PER_BUCKET, canonical_mode, CSCV_METRIC, CSCV_MAX_S, CSCV_MIN_PARTITION,
        ))
        _timed(f"default paired cell L={LOOKBACKS[0]},B={BUCKETS[0]}", lambda: gc.cached_paired_cell(
            source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
            LOOKBACKS[0], BUCKETS[0], LOOKBACKS[0], BUCKETS[0],
            METHOD, RULE, N_EXCLUDE, IV_METHOD, IV_RULE, IV_N_EXCLUDE,
            mode, MIN_DAYS_PER_BUCKET,
        ))

    print(f"Summary — canonical-mode baselines ({canonical_mode}, for reference lines):")
    if canonical_mode == "harcj_only":
        _timed("baselines", lambda: gc.cached_baselines(source, profile, canonical_mode))
    elif canonical_mode == "iv_only":
        _timed("baselines", lambda: gc.cached_iv_baselines(
            source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
        ))
    else:
        _timed("baselines", lambda: gc.cached_paired_baselines(
            source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time, canonical_mode,
        ))


def main():
    for source, loader in _LOADERS.items():
        for profile in loader.available_profiles():
            _warm_profile(source, profile)
            print()
    print("Done — all sources/profiles warmed.")


if __name__ == "__main__":
    main()
