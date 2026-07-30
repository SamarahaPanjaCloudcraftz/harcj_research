"""
Cached grid/cell computation wrappers, shared by pages/2_threshold_grid.py
and pages/3_summary.py — extracted so both pages hit the SAME st.cache_data
cache (same function objects) rather than recomputing independently when
both are configured the same way.

Every function here just loads data + calls into dynamic_threshold.py;
no rendering, no widgets.
"""

from __future__ import annotations

import streamlit as st

from data_loader import load_joined, load_static_config, load_joined_with_iv
import dynamic_threshold as dt


# ── BPV (harcj_only) ─────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Running walk-forward grid…")
def cached_grid(profile, lookbacks, buckets, method, exclusion_rule,
                 n_exclude, iv_cutoff, iv_mode, min_days_per_bucket):
    df = load_joined(profile)
    return dt.run_grid(
        df, list(lookbacks), list(buckets), method=method,
        exclusion_rule=exclusion_rule, n_exclude=n_exclude,
        iv_cutoff=iv_cutoff, iv_mode=iv_mode,
        min_days_per_bucket=min_days_per_bucket,
    )


@st.cache_data(show_spinner=False)
def cached_baselines(profile, iv_mode):
    df = load_joined(profile)
    cfg = load_static_config(profile)
    always = dt.always_trade_baseline(df)
    static = dt.static_threshold_baseline(
        df, cfg["exclusion_ranges"], cfg["iv_cutoff"], iv_mode=iv_mode,
    )
    return always, static, cfg


@st.cache_data(show_spinner="Computing cell diagnostics…")
def cached_cell(profile, L, B, method, exclusion_rule, n_exclude, iv_cutoff,
                 iv_mode, min_days_per_bucket):
    df = load_joined(profile)
    return dt.walk_forward_run(
        df, lookback=L, n_buckets=B, method=method,
        exclusion_rule=exclusion_rule, n_exclude=n_exclude,
        iv_cutoff=iv_cutoff, iv_mode=iv_mode,
        min_days_per_bucket=min_days_per_bucket, return_diagnostics=True,
    )


@st.cache_data(show_spinner="Computing PBO grid — this can take up to a minute…")
def cached_pbo_grid(profile, lookbacks, buckets, method, exclusion_rule, n_exclude,
                     iv_cutoff, iv_mode, min_days_per_bucket,
                     cscv_metric, max_S, min_partition_size):
    df = load_joined(profile)
    cfg = load_static_config(profile)
    static = dt.static_threshold_baseline(df, cfg["exclusion_ranges"], cfg["iv_cutoff"], iv_mode=iv_mode)
    return dt.run_pbo_grid(
        df, list(lookbacks), list(buckets), static_daily_pnl=static["_daily_pnl"],
        method=method, exclusion_rule=exclusion_rule, n_exclude=n_exclude,
        iv_cutoff=iv_cutoff, iv_mode=iv_mode, min_days_per_bucket=min_days_per_bucket,
        cscv_metric=cscv_metric, max_S=max_S, min_partition_size=min_partition_size,
    )


# ── IV-only ──────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_iv_df(profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time):
    return load_joined_with_iv(profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time)


@st.cache_data(show_spinner="Running IV walk-forward grid…")
def cached_iv_grid(profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
                    lookbacks, buckets, method, exclusion_rule, n_exclude, min_days_per_bucket):
    df = load_joined_with_iv(profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time)
    return dt.run_grid(
        df, list(lookbacks), list(buckets), method=method,
        exclusion_rule=exclusion_rule, n_exclude=n_exclude,
        value_col="iv_dynamic", iv_mode="harcj_only",
        min_days_per_bucket=min_days_per_bucket,
    )


@st.cache_data(show_spinner="Computing IV PBO grid — this can take up to a minute…")
def cached_iv_pbo_grid(profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
                        lookbacks, buckets, method, exclusion_rule, n_exclude, min_days_per_bucket,
                        cscv_metric, max_S, min_partition_size):
    df = load_joined_with_iv(profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time)
    cfg = load_static_config(profile)
    static = dt.static_threshold_baseline(df, cfg["exclusion_ranges"], cfg["iv_cutoff"], iv_mode="iv_only")
    return dt.run_pbo_grid(
        df, list(lookbacks), list(buckets), static_daily_pnl=static["_daily_pnl"],
        method=method, exclusion_rule=exclusion_rule, n_exclude=n_exclude,
        value_col="iv_dynamic", iv_mode="harcj_only", min_days_per_bucket=min_days_per_bucket,
        cscv_metric=cscv_metric, max_S=max_S, min_partition_size=min_partition_size,
    )


@st.cache_data(show_spinner="Computing IV cell diagnostics…")
def cached_iv_cell(profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
                    L, B, method, exclusion_rule, n_exclude, min_days_per_bucket):
    df = load_joined_with_iv(profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time)
    return dt.walk_forward_run(
        df, lookback=L, n_buckets=B, method=method, exclusion_rule=exclusion_rule,
        n_exclude=n_exclude, value_col="iv_dynamic", iv_mode="harcj_only",
        min_days_per_bucket=min_days_per_bucket, return_diagnostics=True,
    )


@st.cache_data(show_spinner=False)
def cached_iv_baselines(profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time):
    df = load_joined_with_iv(profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time)
    cfg = load_static_config(profile)
    always = dt.always_trade_baseline(df)
    static = dt.static_threshold_baseline(df, cfg["exclusion_ranges"], cfg["iv_cutoff"], iv_mode="iv_only")
    return always, static


# ── Paired (and_both / or_either) ───────────────────────────────────────────────

@st.cache_data(show_spinner="Running paired walk-forward grid…")
def cached_paired_grid(profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
                        lookbacks, buckets,
                        harcj_method, harcj_exclusion_rule, harcj_n_exclude,
                        iv_method, iv_exclusion_rule, iv_n_exclude,
                        iv_mode, min_days_per_bucket):
    df = load_joined_with_iv(profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time)
    return dt.run_paired_grid(
        df, list(lookbacks), list(buckets),
        harcj_method=harcj_method, harcj_exclusion_rule=harcj_exclusion_rule, harcj_n_exclude=harcj_n_exclude,
        iv_method=iv_method, iv_exclusion_rule=iv_exclusion_rule, iv_n_exclude=iv_n_exclude,
        iv_mode=iv_mode, min_days_per_bucket=min_days_per_bucket,
    )


@st.cache_data(show_spinner="Computing paired PBO grid — this can take a few minutes…")
def cached_paired_pbo_grid(profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
                            lookbacks, buckets,
                            harcj_method, harcj_exclusion_rule, harcj_n_exclude,
                            iv_method, iv_exclusion_rule, iv_n_exclude,
                            iv_mode, min_days_per_bucket, cscv_metric, max_S, min_partition_size):
    df = load_joined_with_iv(profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time)
    cfg = load_static_config(profile)
    static = dt.static_threshold_baseline(df, cfg["exclusion_ranges"], cfg["iv_cutoff"], iv_mode=iv_mode)
    return dt.run_paired_pbo_grid(
        df, list(lookbacks), list(buckets), static_daily_pnl=static["_daily_pnl"],
        harcj_method=harcj_method, harcj_exclusion_rule=harcj_exclusion_rule, harcj_n_exclude=harcj_n_exclude,
        iv_method=iv_method, iv_exclusion_rule=iv_exclusion_rule, iv_n_exclude=iv_n_exclude,
        iv_mode=iv_mode, min_days_per_bucket=min_days_per_bucket,
        cscv_metric=cscv_metric, max_S=max_S, min_partition_size=min_partition_size,
    )


@st.cache_data(show_spinner="Computing paired cell diagnostics…")
def cached_paired_cell(profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
                        harcj_L, harcj_B, iv_L, iv_B,
                        harcj_method, harcj_exclusion_rule, harcj_n_exclude,
                        iv_method, iv_exclusion_rule, iv_n_exclude,
                        iv_mode, min_days_per_bucket):
    df = load_joined_with_iv(profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time)
    harcj_params = dt.DualParams(lookback=harcj_L, n_buckets=harcj_B, method=harcj_method,
                                  exclusion_rule=harcj_exclusion_rule, n_exclude=harcj_n_exclude)
    iv_params = dt.DualParams(lookback=iv_L, n_buckets=iv_B, method=iv_method,
                               exclusion_rule=iv_exclusion_rule, n_exclude=iv_n_exclude)
    return dt.run_paired_cell(
        df, harcj_params, iv_params, iv_mode=iv_mode,
        min_days_per_bucket=min_days_per_bucket, return_diagnostics=True,
    )


@st.cache_data(show_spinner=False)
def cached_paired_baselines(profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time, iv_mode):
    df = load_joined_with_iv(profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time)
    cfg = load_static_config(profile)
    always = dt.always_trade_baseline(df)
    static = dt.static_threshold_baseline(df, cfg["exclusion_ranges"], cfg["iv_cutoff"], iv_mode=iv_mode)
    return always, static
