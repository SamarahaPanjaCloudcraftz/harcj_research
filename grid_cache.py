"""
Cached grid/cell computation wrappers, shared by pages/2_threshold_grid.py
and pages/3_summary.py — extracted so both pages hit the SAME st.cache_data
cache (same function objects) rather than recomputing independently when
both are configured the same way.

Two cache layers stack here:
  1. st.cache_data — fast, in-memory, but empty on every process restart.
  2. disk_cache.load_or_compute — persists to dashboard_data/, survives
     restarts. A parameter combination computed once (e.g. the sidebar's
     defaults) loads instantly next time the app starts, from any page;
     any DIFFERENT combination just computes fresh (and is then saved for
     its own next time) — nothing here hard-codes "defaults" specially.

Every function here takes a `source` argument (a DATA_SOURCE_NAME string)
selecting which loader (module or BacktestSource instance) to read from —
it's part of the cache key so sources never collide, on disk or in
st.cache_data. Otherwise: just loads data + calls into dynamic_threshold.py;
no rendering, no widgets.
"""

from __future__ import annotations

import streamlit as st

import data_loader
import backtest_source
import dynamic_threshold as dt
from disk_cache import load_or_compute

_LOADERS = {data_loader.DATA_SOURCE_NAME: data_loader}
_LOADERS.update({src.DATA_SOURCE_NAME: src for src in backtest_source.ALL_SOURCES})


def _loader(source: str):
    return _LOADERS[source]


def _extra_history(source: str, profile: str):
    """
    BPV/harcj_only pre-inception warm-up history (see backtest_source.py),
    if this source's loader exposes one — None for live_dumps and any future
    source that doesn't have one, in which case run_grid/walk_forward_run
    fall back to their normal (no-extension) behavior automatically.
    """
    loader = _loader(source)
    fn = getattr(loader, "extended_pred_vol_pct_history", None)
    return fn(profile) if fn is not None else None


def _iv_extra_history(source: str, profile: str, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time):
    """IV-side counterpart of _extra_history — see backtest_source.extended_iv_series."""
    loader = _loader(source)
    fn = getattr(loader, "extended_iv_series", None)
    if fn is None:
        return None
    return fn(profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time)


# ── BPV (harcj_only) ─────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Running walk-forward grid…")
def cached_grid(source, profile, lookbacks, buckets, method, exclusion_rule,
                 n_exclude, iv_cutoff, iv_mode, min_days_per_bucket):
    args = (profile, lookbacks, buckets, method, exclusion_rule,
            n_exclude, iv_cutoff, iv_mode, min_days_per_bucket)

    def _compute():
        df = _loader(source).load_joined(profile)
        return dt.run_grid(
            df, list(lookbacks), list(buckets), method=method,
            exclusion_rule=exclusion_rule, n_exclude=n_exclude,
            iv_cutoff=iv_cutoff, iv_mode=iv_mode,
            min_days_per_bucket=min_days_per_bucket,
            extra_history=_extra_history(source, profile),
        )
    return load_or_compute("cached_grid", args, _compute, profile=profile, source=source)


@st.cache_data(show_spinner=False)
def cached_baselines(source, profile, iv_mode):
    args = (profile, iv_mode)

    def _compute():
        df = _loader(source).load_joined(profile)
        cfg = _loader(source).load_static_config(profile)
        always = dt.always_trade_baseline(df)
        static = dt.static_threshold_baseline(
            df, cfg["exclusion_ranges"], cfg["iv_cutoff"], iv_mode=iv_mode,
        )
        return always, static, cfg
    return load_or_compute("cached_baselines", args, _compute, profile=profile, source=source)


@st.cache_data(show_spinner="Computing cell diagnostics…")
def cached_cell(source, profile, L, B, method, exclusion_rule, n_exclude, iv_cutoff,
                 iv_mode, min_days_per_bucket):
    args = (profile, L, B, method, exclusion_rule, n_exclude, iv_cutoff, iv_mode, min_days_per_bucket)

    def _compute():
        df = _loader(source).load_joined(profile)
        return dt.walk_forward_run(
            df, lookback=L, n_buckets=B, method=method,
            exclusion_rule=exclusion_rule, n_exclude=n_exclude,
            iv_cutoff=iv_cutoff, iv_mode=iv_mode,
            min_days_per_bucket=min_days_per_bucket, return_diagnostics=True,
            extra_history=_extra_history(source, profile),
        )
    return load_or_compute("cached_cell", args, _compute, profile=profile, source=source)


@st.cache_data(show_spinner="Computing PBO grid — this can take up to a minute…")
def cached_pbo_grid(source, profile, lookbacks, buckets, method, exclusion_rule, n_exclude,
                     iv_cutoff, iv_mode, min_days_per_bucket,
                     cscv_metric, max_S, min_partition_size):
    args = (profile, lookbacks, buckets, method, exclusion_rule, n_exclude,
            iv_cutoff, iv_mode, min_days_per_bucket, cscv_metric, max_S, min_partition_size)

    def _compute():
        df = _loader(source).load_joined(profile)
        cfg = _loader(source).load_static_config(profile)
        static = dt.static_threshold_baseline(df, cfg["exclusion_ranges"], cfg["iv_cutoff"], iv_mode=iv_mode)
        return dt.run_pbo_grid(
            df, list(lookbacks), list(buckets), static_daily_pnl=static["_daily_pnl"],
            method=method, exclusion_rule=exclusion_rule, n_exclude=n_exclude,
            iv_cutoff=iv_cutoff, iv_mode=iv_mode, min_days_per_bucket=min_days_per_bucket,
            cscv_metric=cscv_metric, max_S=max_S, min_partition_size=min_partition_size,
            extra_history=_extra_history(source, profile),
        )
    return load_or_compute("cached_pbo_grid", args, _compute, profile=profile, source=source)


# ── IV-only ──────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_iv_df(source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time):
    return _loader(source).load_joined_with_iv(profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time)


@st.cache_data(show_spinner="Running IV walk-forward grid…")
def cached_iv_grid(source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
                    lookbacks, buckets, method, exclusion_rule, n_exclude, min_days_per_bucket):
    args = (profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
            lookbacks, buckets, method, exclusion_rule, n_exclude, min_days_per_bucket)

    def _compute():
        df = _loader(source).load_joined_with_iv(profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time)
        return dt.run_grid(
            df, list(lookbacks), list(buckets), method=method,
            exclusion_rule=exclusion_rule, n_exclude=n_exclude,
            value_col="iv_dynamic", iv_mode="harcj_only",
            min_days_per_bucket=min_days_per_bucket,
            extra_history=_iv_extra_history(source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time),
        )
    return load_or_compute("cached_iv_grid", args, _compute, profile=profile, source=source)


@st.cache_data(show_spinner="Computing IV PBO grid — this can take up to a minute…")
def cached_iv_pbo_grid(source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
                        lookbacks, buckets, method, exclusion_rule, n_exclude, min_days_per_bucket,
                        cscv_metric, max_S, min_partition_size):
    args = (profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
            lookbacks, buckets, method, exclusion_rule, n_exclude, min_days_per_bucket,
            cscv_metric, max_S, min_partition_size)

    def _compute():
        df = _loader(source).load_joined_with_iv(profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time)
        cfg = _loader(source).load_static_config(profile)
        static = dt.static_threshold_baseline(df, cfg["exclusion_ranges"], cfg["iv_cutoff"], iv_mode="iv_only")
        return dt.run_pbo_grid(
            df, list(lookbacks), list(buckets), static_daily_pnl=static["_daily_pnl"],
            method=method, exclusion_rule=exclusion_rule, n_exclude=n_exclude,
            value_col="iv_dynamic", iv_mode="harcj_only", min_days_per_bucket=min_days_per_bucket,
            cscv_metric=cscv_metric, max_S=max_S, min_partition_size=min_partition_size,
            extra_history=_iv_extra_history(source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time),
        )
    return load_or_compute("cached_iv_pbo_grid", args, _compute, profile=profile, source=source)


@st.cache_data(show_spinner="Computing IV cell diagnostics…")
def cached_iv_cell(source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
                    L, B, method, exclusion_rule, n_exclude, min_days_per_bucket):
    args = (profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
            L, B, method, exclusion_rule, n_exclude, min_days_per_bucket)

    def _compute():
        df = _loader(source).load_joined_with_iv(profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time)
        return dt.walk_forward_run(
            df, lookback=L, n_buckets=B, method=method, exclusion_rule=exclusion_rule,
            n_exclude=n_exclude, value_col="iv_dynamic", iv_mode="harcj_only",
            min_days_per_bucket=min_days_per_bucket, return_diagnostics=True,
            extra_history=_iv_extra_history(source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time),
        )
    return load_or_compute("cached_iv_cell", args, _compute, profile=profile, source=source)


@st.cache_data(show_spinner=False)
def cached_iv_baselines(source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time):
    args = (profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time)

    def _compute():
        df = _loader(source).load_joined_with_iv(profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time)
        cfg = _loader(source).load_static_config(profile)
        always = dt.always_trade_baseline(df)
        static = dt.static_threshold_baseline(df, cfg["exclusion_ranges"], cfg["iv_cutoff"], iv_mode="iv_only")
        return always, static
    return load_or_compute("cached_iv_baselines", args, _compute, profile=profile, source=source)


# ── Paired (and_both / or_either) ───────────────────────────────────────────────

@st.cache_data(show_spinner="Running paired walk-forward grid…")
def cached_paired_grid(source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
                        lookbacks, buckets,
                        harcj_method, harcj_exclusion_rule, harcj_n_exclude,
                        iv_method, iv_exclusion_rule, iv_n_exclude,
                        iv_mode, min_days_per_bucket):
    args = (profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
            lookbacks, buckets, harcj_method, harcj_exclusion_rule, harcj_n_exclude,
            iv_method, iv_exclusion_rule, iv_n_exclude, iv_mode, min_days_per_bucket)

    def _compute():
        df = _loader(source).load_joined_with_iv(profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time)
        return dt.run_paired_grid(
            df, list(lookbacks), list(buckets),
            harcj_method=harcj_method, harcj_exclusion_rule=harcj_exclusion_rule, harcj_n_exclude=harcj_n_exclude,
            iv_method=iv_method, iv_exclusion_rule=iv_exclusion_rule, iv_n_exclude=iv_n_exclude,
            iv_mode=iv_mode, min_days_per_bucket=min_days_per_bucket,
            harcj_extra_history=_extra_history(source, profile),
            iv_extra_history=_iv_extra_history(source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time),
        )
    return load_or_compute("cached_paired_grid", args, _compute, profile=profile, source=source)


@st.cache_data(show_spinner="Computing paired PBO grid — this can take a few minutes…")
def cached_paired_pbo_grid(source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
                            lookbacks, buckets,
                            harcj_method, harcj_exclusion_rule, harcj_n_exclude,
                            iv_method, iv_exclusion_rule, iv_n_exclude,
                            iv_mode, min_days_per_bucket, cscv_metric, max_S, min_partition_size):
    args = (profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
            lookbacks, buckets, harcj_method, harcj_exclusion_rule, harcj_n_exclude,
            iv_method, iv_exclusion_rule, iv_n_exclude, iv_mode, min_days_per_bucket,
            cscv_metric, max_S, min_partition_size)

    def _compute():
        df = _loader(source).load_joined_with_iv(profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time)
        cfg = _loader(source).load_static_config(profile)
        static = dt.static_threshold_baseline(df, cfg["exclusion_ranges"], cfg["iv_cutoff"], iv_mode=iv_mode)
        return dt.run_paired_pbo_grid(
            df, list(lookbacks), list(buckets), static_daily_pnl=static["_daily_pnl"],
            harcj_method=harcj_method, harcj_exclusion_rule=harcj_exclusion_rule, harcj_n_exclude=harcj_n_exclude,
            iv_method=iv_method, iv_exclusion_rule=iv_exclusion_rule, iv_n_exclude=iv_n_exclude,
            iv_mode=iv_mode, min_days_per_bucket=min_days_per_bucket,
            cscv_metric=cscv_metric, max_S=max_S, min_partition_size=min_partition_size,
            harcj_extra_history=_extra_history(source, profile),
            iv_extra_history=_iv_extra_history(source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time),
        )
    return load_or_compute("cached_paired_pbo_grid", args, _compute, profile=profile, source=source)


@st.cache_data(show_spinner="Computing paired cell diagnostics…")
def cached_paired_cell(source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
                        harcj_L, harcj_B, iv_L, iv_B,
                        harcj_method, harcj_exclusion_rule, harcj_n_exclude,
                        iv_method, iv_exclusion_rule, iv_n_exclude,
                        iv_mode, min_days_per_bucket):
    args = (profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
            harcj_L, harcj_B, iv_L, iv_B, harcj_method, harcj_exclusion_rule, harcj_n_exclude,
            iv_method, iv_exclusion_rule, iv_n_exclude, iv_mode, min_days_per_bucket)

    def _compute():
        df = _loader(source).load_joined_with_iv(profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time)
        harcj_params = dt.DualParams(lookback=harcj_L, n_buckets=harcj_B, method=harcj_method,
                                      exclusion_rule=harcj_exclusion_rule, n_exclude=harcj_n_exclude)
        iv_params = dt.DualParams(lookback=iv_L, n_buckets=iv_B, method=iv_method,
                                   exclusion_rule=iv_exclusion_rule, n_exclude=iv_n_exclude)
        return dt.run_paired_cell(
            df, harcj_params, iv_params, iv_mode=iv_mode,
            min_days_per_bucket=min_days_per_bucket, return_diagnostics=True,
            harcj_extra_history=_extra_history(source, profile),
            iv_extra_history=_iv_extra_history(source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time),
        )
    return load_or_compute("cached_paired_cell", args, _compute, profile=profile, source=source)


@st.cache_data(show_spinner=False)
def cached_paired_baselines(source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time, iv_mode):
    args = (profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time, iv_mode)

    def _compute():
        df = _loader(source).load_joined_with_iv(profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time)
        cfg = _loader(source).load_static_config(profile)
        always = dt.always_trade_baseline(df)
        static = dt.static_threshold_baseline(df, cfg["exclusion_ranges"], cfg["iv_cutoff"], iv_mode=iv_mode)
        return always, static
    return load_or_compute("cached_paired_baselines", args, _compute, profile=profile, source=source)


# ── PBO vs. a FIXED baseline mode (Summary page's global cross-mode ranking) ────
# Unlike cached_pbo_grid/cached_iv_pbo_grid/cached_paired_pbo_grid (each cell
# measured against ITS OWN mode's static baseline), these measure every
# cell — regardless of which mode's grid it comes from — against ONE shared
# baseline (`baseline_mode`, typically the profile's actual live-configured
# combination). That's what makes a single ranking pooling all 4 modes
# together apples-to-apples: everything is racing the same opponent.

@st.cache_data(show_spinner="Running walk-forward grid — PBO vs. live baseline…")
def cached_pbo_grid_vs_mode(source, profile, lookbacks, buckets, method, exclusion_rule, n_exclude,
                             iv_cutoff, iv_mode, min_days_per_bucket, baseline_mode,
                             cscv_metric, max_S, min_partition_size):
    args = (profile, lookbacks, buckets, method, exclusion_rule, n_exclude,
            iv_cutoff, iv_mode, min_days_per_bucket, baseline_mode, cscv_metric, max_S, min_partition_size)

    def _compute():
        df = _loader(source).load_joined(profile)
        _, baseline_static, _ = cached_baselines(source, profile, baseline_mode)
        return dt.run_pbo_grid(
            df, list(lookbacks), list(buckets), static_daily_pnl=baseline_static["_daily_pnl"],
            method=method, exclusion_rule=exclusion_rule, n_exclude=n_exclude,
            iv_cutoff=iv_cutoff, iv_mode=iv_mode, min_days_per_bucket=min_days_per_bucket,
            cscv_metric=cscv_metric, max_S=max_S, min_partition_size=min_partition_size,
            extra_history=_extra_history(source, profile),
        )
    return load_or_compute("cached_pbo_grid_vs_mode", args, _compute, profile=profile, source=source)


@st.cache_data(show_spinner="Running IV walk-forward grid — PBO vs. live baseline…")
def cached_iv_pbo_grid_vs_mode(source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
                                lookbacks, buckets, method, exclusion_rule, n_exclude, min_days_per_bucket,
                                baseline_mode, cscv_metric, max_S, min_partition_size):
    args = (profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
            lookbacks, buckets, method, exclusion_rule, n_exclude, min_days_per_bucket,
            baseline_mode, cscv_metric, max_S, min_partition_size)

    def _compute():
        df = _loader(source).load_joined_with_iv(profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time)
        _, baseline_static, _ = cached_baselines(source, profile, baseline_mode)
        return dt.run_pbo_grid(
            df, list(lookbacks), list(buckets), static_daily_pnl=baseline_static["_daily_pnl"],
            method=method, exclusion_rule=exclusion_rule, n_exclude=n_exclude,
            value_col="iv_dynamic", iv_mode="harcj_only", min_days_per_bucket=min_days_per_bucket,
            cscv_metric=cscv_metric, max_S=max_S, min_partition_size=min_partition_size,
            extra_history=_iv_extra_history(source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time),
        )
    return load_or_compute("cached_iv_pbo_grid_vs_mode", args, _compute, profile=profile, source=source)


@st.cache_data(show_spinner="Running paired walk-forward grid — PBO vs. live baseline…")
def cached_paired_pbo_grid_vs_mode(source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
                                    lookbacks, buckets,
                                    harcj_method, harcj_exclusion_rule, harcj_n_exclude,
                                    iv_method, iv_exclusion_rule, iv_n_exclude,
                                    iv_mode, min_days_per_bucket, baseline_mode,
                                    cscv_metric, max_S, min_partition_size):
    args = (profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
            lookbacks, buckets, harcj_method, harcj_exclusion_rule, harcj_n_exclude,
            iv_method, iv_exclusion_rule, iv_n_exclude, iv_mode, min_days_per_bucket,
            baseline_mode, cscv_metric, max_S, min_partition_size)

    def _compute():
        df = _loader(source).load_joined_with_iv(profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time)
        _, baseline_static, _ = cached_baselines(source, profile, baseline_mode)
        return dt.run_paired_pbo_grid(
            df, list(lookbacks), list(buckets), static_daily_pnl=baseline_static["_daily_pnl"],
            harcj_method=harcj_method, harcj_exclusion_rule=harcj_exclusion_rule, harcj_n_exclude=harcj_n_exclude,
            iv_method=iv_method, iv_exclusion_rule=iv_exclusion_rule, iv_n_exclude=iv_n_exclude,
            iv_mode=iv_mode, min_days_per_bucket=min_days_per_bucket,
            cscv_metric=cscv_metric, max_S=max_S, min_partition_size=min_partition_size,
            harcj_extra_history=_extra_history(source, profile),
            iv_extra_history=_iv_extra_history(source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time),
        )
    return load_or_compute("cached_paired_pbo_grid_vs_mode", args, _compute, profile=profile, source=source)
