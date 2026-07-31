"""
Threshold Grid — lookback x n_buckets, walk-forward, no lookahead.

Dynamic (rolling) version of today's static harcj_exclusion_ranges: instead
of one threshold calibrated once on the full history, this recomputes bucket
edges + exclusion set from only the trailing `lookback` days, at every day
in the backtest. Two reference values (always-trade, today's static
threshold) are shown alongside so the grid has something to beat.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_loader
import backtest_source
import dynamic_threshold as dt
from chart_helpers import (
    render_heatmap, render_multi_equity_curve, render_pnl_bars, render_diff_bars,
    render_cutoff_timeline, render_reference,
    METRIC_LABELS, _DOLLAR_METRICS, _BAR_FREQ,
    _BLUE, _RED, _MID, _GRID, _MUTED, _INK, _CATEGORICAL,
)
from grid_cache import (
    cached_grid, cached_baselines, cached_cell, cached_pbo_grid,
    cached_iv_grid, cached_iv_pbo_grid, cached_iv_cell, cached_iv_baselines,
    cached_paired_grid, cached_paired_pbo_grid, cached_paired_cell, cached_paired_baselines,
)

_SOURCES = {
    data_loader.DATA_SOURCE_NAME: data_loader,
    backtest_source.DATA_SOURCE_NAME: backtest_source,
}
_SOURCE_LABELS = {
    data_loader.DATA_SOURCE_NAME: "Live dumps (dashboard_new/dumps/)",
    backtest_source.DATA_SOURCE_NAME: "Backtest — spxw_gamma_hedge_false",
}

st.set_page_config(page_title="Threshold Grid — HARCJ Research", layout="wide")
st.title("Threshold Grid")
st.caption(
    "Walk-forward (no lookahead): at each day, bucket edges and the excluded "
    "set are derived only from the trailing `lookback` days, then applied to "
    "that day's own prediction. Compare against the two reference lines "
    "(always-trade, today's static threshold) to see whether a dynamic "
    "cutoff actually helps."
)


def _parse_int_list(raw: str) -> list[int]:
    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return sorted(set(out))


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Controls")

    source = st.selectbox(
        "Data source", list(_SOURCES.keys()), index=0, format_func=lambda s: _SOURCE_LABELS[s],
    )
    loader = _SOURCES[source]
    profile = st.selectbox("Profile", loader.available_profiles(), index=0)
    df = loader.load_joined(profile)
    st.caption(f"Usable days: {len(df)} ({df.index.min().date()} → {df.index.max().date()})")
    if source == backtest_source.DATA_SOURCE_NAME:
        _sess_start, _sess_end = backtest_source.derive_session_window()
        st.caption(
            f"Session window auto-derived from blotter majority vote: "
            f"{_sess_start}–{_sess_end} CT. Open-IV window defaults to the hour "
            f"immediately before session start."
        )

    st.subheader("BPV settings")
    lookback_raw = st.text_input("BPV lookback grid (trading days, comma-separated)",
                                  value="20,40,60,90,120,180,252")
    bucket_raw   = st.text_input("BPV bucket-count grid (comma-separated)",
                                  value="4,5,8,10")
    lookbacks = _parse_int_list(lookback_raw)
    buckets   = _parse_int_list(bucket_raw)

    _warmup_days_fn = getattr(loader, "available_warmup_days", None)
    if _warmup_days_fn is not None:
        _warmup_days = _warmup_days_fn(profile)
        _short_lookbacks = [L for L in lookbacks if L > _warmup_days]
        if _short_lookbacks:
            st.warning(
                f"Only {_warmup_days} days of pre-backtest BPV history available — "
                f"lookback(s) {', '.join(map(str, _short_lookbacks))} exceed this, so "
                f"those variants still start a few days into the backtest instead of "
                f"at its first day."
            )

    method = st.radio(
        "BPV bucketing method",
        options=["equal_count", "equal_width"],
        format_func=lambda m: "Equal-count (quantile)" if m == "equal_count" else "Equal-width (range)",
        key="bpv_method",
    )

    # Only rule-based exclusion is exposed: it needs no trailing PnL history
    # (unlike the removed worst_pnl_dynamic rule), so it can be "ready" as
    # soon as there's a trailing lookback window of values — see conversation
    # history re: extending the backtest source's price history back of its
    # PnL-scored range.
    exclusion_rule = "highest_vol"

    n_exclude = st.number_input("BPV — number of buckets to exclude", min_value=1,
                                 max_value=10, value=1, step=1, key="bpv_n_exclude")

    IV_MODE_LABELS = {
        "harcj_only": "HARCJ only (ignore IV)",
        "iv_only":    "IV only (ignore HARCJ)",
        "and_both":   "AND — both must pass to trade",
        "or_either":  "OR — either passing is enough",
    }
    iv_modes_selected = st.multiselect(
        "IV-combination modes to compare",
        options=list(dt.IV_MODES),
        default=["harcj_only"],
        format_func=lambda m: IV_MODE_LABELS[m],
        help="AND and OR are the only two genuinely different ways to combine "
             "the two filters; harcj_only/iv_only isolate one filter entirely. "
             "Select more than one to render them side by side.",
    )

    show_iv_settings = any(m in iv_modes_selected for m in ("iv_only", "and_both", "or_either"))
    if show_iv_settings:
        st.divider()
        st.subheader("IV settings")
        if source == backtest_source.DATA_SOURCE_NAME:
            _sess_start, _ = backtest_source.derive_session_window()
            _default_iv_end = _sess_start
            _default_iv_start = (
                pd.Timestamp(_sess_start) - pd.Timedelta(hours=backtest_source.DEFAULT_IV_WINDOW_HOURS_BEFORE_START)
            ).strftime("%H:%M")
            st.caption(
                "IV gets the identical dynamic-threshold machinery as BPV, computed "
                "from research/atm_iv/SPWX/ (weekday-matched, 00:00–23:59 CT "
                f"available, 2025 warm-up + this backtest's own 2026 days). Defaults "
                f"to the hour before session start ({_default_iv_start}–{_default_iv_end})."
            )
        else:
            _default_iv_start, _default_iv_end = "07:30", "08:30"
            st.caption(
                "IV gets the identical dynamic-threshold machinery as BPV, computed "
                "from the raw 1-min IV dump (dumps/<underlying>_stage1/atm_iv/) — the "
                "trustworthy source, not roll.csv's mean_iv (which can be stale for "
                "recent days — see conversation history). Only the 07:30–08:30 CT "
                "window has data; timestamps/ranges outside that will be empty."
            )

        iv_source_mode = st.radio(
            "IV source",
            options=["mean", "point"],
            format_func=lambda m: "Open IV (single timestamp)" if m == "point" else "Open IV mean (window average)",
        )
        if iv_source_mode == "point":
            iv_timestamp  = st.text_input("Timestamp (HH:MM)", value=_default_iv_end)
            iv_start_time = None
            iv_end_time   = None
        else:
            iv_timestamp = None
            c1, c2 = st.columns(2)
            with c1:
                iv_start_time = st.text_input("Window start (HH:MM)", value=_default_iv_start)
            with c2:
                iv_end_time = st.text_input("Window end (HH:MM)", value=_default_iv_end)

        iv_lookback_raw = st.text_input("IV lookback grid (trading days, comma-separated)",
                                         value="20,40,60,90,120,180,252")
        iv_bucket_raw   = st.text_input("IV bucket-count grid (comma-separated)",
                                         value="4,5,8,10")
        iv_lookbacks = _parse_int_list(iv_lookback_raw)
        iv_buckets   = _parse_int_list(iv_bucket_raw)

        _iv_warmup_days_fn = getattr(loader, "available_iv_warmup_days", None)
        if _iv_warmup_days_fn is not None:
            _iv_warmup_days = _iv_warmup_days_fn(profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time)
            _short_iv_lookbacks = [L for L in iv_lookbacks if L > _iv_warmup_days]
            if _short_iv_lookbacks:
                st.warning(
                    f"Only {_iv_warmup_days} days of pre-backtest IV history available — "
                    f"lookback(s) {', '.join(map(str, _short_iv_lookbacks))} exceed this, so "
                    f"those variants still start a few days into the backtest instead of "
                    f"at its first day."
                )

        iv_method = st.radio(
            "IV bucketing method",
            options=["equal_count", "equal_width"],
            format_func=lambda m: "Equal-count (quantile)" if m == "equal_count" else "Equal-width (range)",
            key="iv_method",
        )
        iv_exclusion_rule = "highest_vol"
        iv_n_exclude = st.number_input("IV — number of buckets to exclude", min_value=1,
                                        max_value=10, value=1, step=1, key="iv_n_exclude")
    else:
        # Defaults so the paired-grid code path below has something even if
        # the user hasn't opened the IV settings yet.
        iv_source_mode = "mean"
        iv_timestamp = None
        iv_start_time, iv_end_time = "07:30", "08:30"
        iv_lookbacks, iv_buckets = lookbacks, buckets
        iv_method, iv_exclusion_rule, iv_n_exclude = method, exclusion_rule, n_exclude

    # Was configurable for the worst-PnL dynamic rule's "reliable" flag (now
    # removed) — the highest-vol/IV rule has no such reliability concern, so
    # this is just the fixed value passed through to the grid functions.
    min_days_per_bucket = 10

    st.divider()
    metric = st.selectbox("Metric", list(METRIC_LABELS.keys()),
                           format_func=lambda m: METRIC_LABELS[m])

    st.divider()
    st.subheader("PBO (vs static threshold)")
    st.caption(
        "Probability of Backtest Overfitting (Bailey et al. 2015, CSCV) — "
        "per cell, is this cell's edge over today's static threshold likely "
        "real, or would picking the in-sample winner between the two have "
        "been a coin flip OOS? PBO < 0.05 ≈ unlikely overfit, > 0.50 ≈ high risk."
    )
    cscv_metric = st.selectbox(
        "CSCV ranking metric", ["sortino", "sharpe", "calmar", "total_pnl"], index=0,
    )
    cscv_max_S = st.number_input("Max S (time partitions)", min_value=4, max_value=32,
                                  value=16, step=2)
    cscv_min_partition = st.number_input("Min days per partition", min_value=1, max_value=50,
                                          value=4, step=1)

    st.divider()
    st.subheader("Composite score")
    st.caption(
        "Ranks every (lookback, n_buckets) cell on each of the 4 metrics below "
        "(rank 1 = best), then takes a weighted average of those ranks. "
        "Weights are normalized to sum to 1 regardless of what you enter here."
    )
    weight_pbo     = st.number_input("Weight — PBO",            min_value=0.0, value=0.49, step=0.01)
    weight_avg_pnl = st.number_input("Weight — Avg daily PnL",  min_value=0.0, value=0.17, step=0.01)
    weight_sortino = st.number_input("Weight — Sortino",        min_value=0.0, value=0.17, step=0.01)
    weight_mdl     = st.number_input("Weight — Max daily loss", min_value=0.0, value=0.17, step=0.01)
    top_n = st.number_input("Top N variants to show", min_value=1, max_value=50, value=5, step=1)

if not lookbacks or not buckets:
    st.warning("Enter at least one lookback and one bucket count.")
    st.stop()

cfg = loader.load_static_config(profile)


# ── Main ─────────────────────────────────────────────────────────────────────

if not iv_modes_selected:
    st.warning("Select at least one IV-combination mode.")
    st.stop()

for mode in iv_modes_selected:
    iv_label = IV_MODE_LABELS[mode]
    st.subheader(iv_label)

    # ── Per-mode setup: grid_result / pbo_grid_result / combo lookup / get_cell ─
    if mode == "harcj_only":
        always, static, _ = cached_baselines(source, profile, mode)
        this_lookbacks, this_buckets = lookbacks, buckets
        grid_result = cached_grid(
            source, profile, tuple(lookbacks), tuple(buckets), method, exclusion_rule,
            int(n_exclude), cfg["iv_cutoff"], mode, int(min_days_per_bucket),
        )
        pbo_grid_result = cached_pbo_grid(
            source, profile, tuple(lookbacks), tuple(buckets), method, exclusion_rule,
            int(n_exclude), cfg["iv_cutoff"], mode, int(min_days_per_bucket),
            cscv_metric, int(cscv_max_S), int(cscv_min_partition),
        )
        combo_labels = [f"L={L}, B={B}" for L in lookbacks for B in buckets]
        combo_lookup = {f"L={L}, B={B}": (L, B) for L in lookbacks for B in buckets}

        def get_cell(L, B):
            return cached_cell(source, profile, L, B, method, exclusion_rule,
                                 int(n_exclude), cfg["iv_cutoff"], mode, int(min_days_per_bucket))
        cutoff_rule = exclusion_rule

    elif mode == "iv_only":
        if not iv_lookbacks or not iv_buckets:
            st.warning("Enter at least one IV lookback and bucket count in the sidebar's IV settings.")
            st.divider()
            continue
        always, static = cached_iv_baselines(source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time)
        this_lookbacks, this_buckets = iv_lookbacks, iv_buckets
        grid_result = cached_iv_grid(
            source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
            tuple(iv_lookbacks), tuple(iv_buckets), iv_method, iv_exclusion_rule,
            int(iv_n_exclude), int(min_days_per_bucket),
        )
        pbo_grid_result = cached_iv_pbo_grid(
            source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
            tuple(iv_lookbacks), tuple(iv_buckets), iv_method, iv_exclusion_rule,
            int(iv_n_exclude), int(min_days_per_bucket),
            cscv_metric, int(cscv_max_S), int(cscv_min_partition),
        )
        combo_labels = [f"L={L}, B={B}" for L in iv_lookbacks for B in iv_buckets]
        combo_lookup = {f"L={L}, B={B}": (L, B) for L in iv_lookbacks for B in iv_buckets}

        def get_cell(L, B):
            return cached_iv_cell(source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
                                    L, B, iv_method, iv_exclusion_rule, int(iv_n_exclude), int(min_days_per_bucket))
        cutoff_rule = iv_exclusion_rule

    else:  # and_both / or_either — paired grid, same (L, B) for both sides
        always, static = cached_paired_baselines(
            source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time, mode,
        )
        this_lookbacks, this_buckets = lookbacks, buckets
        st.caption(
            "Paired grid: the same (lookback, n_buckets) drives both the BPV and "
            "IV dynamic thresholds at each cell — each side keeps its own bucketing "
            "method / exclusion rule / n_exclude (from the sidebar's BPV and IV "
            "settings), only L and B are forced equal. For fully independent "
            "BPV vs. IV (L, B), use \"Specific combination\" below."
        )
        grid_result = cached_paired_grid(
            source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
            tuple(lookbacks), tuple(buckets),
            method, exclusion_rule, int(n_exclude),
            iv_method, iv_exclusion_rule, int(iv_n_exclude),
            mode, int(min_days_per_bucket),
        )
        pbo_grid_result = cached_paired_pbo_grid(
            source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
            tuple(lookbacks), tuple(buckets),
            method, exclusion_rule, int(n_exclude),
            iv_method, iv_exclusion_rule, int(iv_n_exclude),
            mode, int(min_days_per_bucket), cscv_metric, int(cscv_max_S), int(cscv_min_partition),
        )
        combo_labels = [f"L={L}, B={B}" for L in lookbacks for B in buckets]
        combo_lookup = {f"L={L}, B={B}": (L, B) for L in lookbacks for B in buckets}

        def get_cell(L, B):
            return cached_paired_cell(
                source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
                L, B, L, B,
                method, exclusion_rule, int(n_exclude),
                iv_method, iv_exclusion_rule, int(iv_n_exclude),
                mode, int(min_days_per_bucket),
            )
        cutoff_rule = exclusion_rule

    render_reference(always, static, cfg, metric, iv_label)

    render_heatmap(grid_result, metric, f"{METRIC_LABELS[metric]} — {iv_label}",
                   min_days_per_bucket=int(min_days_per_bucket))
    with st.expander(f"Raw grid values ({iv_label})"):
        st.dataframe(grid_result["metrics"][metric], use_container_width=True)
        st.caption("Days traded per cell:")
        st.dataframe(grid_result["n_traded"], use_container_width=True)

    render_heatmap(pbo_grid_result, "pbo", f"PBO vs static threshold — {iv_label}",
                   label="PBO", min_days_per_bucket=int(min_days_per_bucket))
    with st.expander(f"Raw PBO values ({iv_label})"):
        st.dataframe(pbo_grid_result["metrics"]["pbo"], use_container_width=True)
        st.caption("CSCV combinations per cell:")
        st.dataframe(pbo_grid_result["n_traded"], use_container_width=True)

    st.markdown(f"**Composite score leaderboard — {iv_label}**")
    weights = {
        "pbo": weight_pbo, "avg_daily_pnl": weight_avg_pnl,
        "sortino": weight_sortino, "max_daily_loss": weight_mdl,
    }
    composite_table = dt.composite_score_table(grid_result, pbo_grid_result, weights)
    if composite_table.empty:
        st.info("No cells have data for all four metrics — nothing to rank.")
    else:
        display = composite_table.head(int(top_n)).copy()
        display["avg_daily_pnl"]  = display["avg_daily_pnl"].map(lambda v: f"{v:,.0f}")
        display["sortino"]        = display["sortino"].map(lambda v: f"{v:.2f}")
        display["max_daily_loss"] = display["max_daily_loss"].map(lambda v: f"{v:,.0f}")
        display["pbo"]            = display["pbo"].map(lambda v: f"{v:.3f}")
        display["composite"]      = display["composite"].map(lambda v: f"{v:.2f}")
        for c in ["rank_avg_daily_pnl", "rank_sortino", "rank_max_daily_loss", "rank_pbo"]:
            display[c] = display[c].map(lambda v: f"{v:.1f}")
        st.dataframe(
            display[["lookback", "n_buckets", "composite", "avg_daily_pnl", "sortino",
                     "max_daily_loss", "pbo", "rank_avg_daily_pnl", "rank_sortino",
                     "rank_max_daily_loss", "rank_pbo", "reliable"]],
            use_container_width=True, hide_index=True,
        )
        if not display["reliable"].all():
            st.caption(
                "`reliable = False` rows use a worst-PnL-dynamic cell that's below "
                "the min-days-per-bucket threshold, or hit a PBO warning — still "
                "ranked, but treat the composite score there with more caution."
            )
        with st.expander(f"Full ranked table ({iv_label})"):
            st.dataframe(composite_table, use_container_width=True, hide_index=True)

    st.markdown(f"**Compare combos — {iv_label}**")
    st.caption(
        "Pick any number of (lookback, n_buckets) combos from this grid to "
        "overlay on the equity curve and daily/weekly/monthly PnL bars."
    )
    combos_selected = st.multiselect(
        "Combos to compare", combo_labels, default=combo_labels[:1],
        key=f"combos_{mode}",
    )
    cell_results = {label: get_cell(*combo_lookup[label]) for label in combos_selected}
    render_multi_equity_curve(cell_results, always, static, f"Equity curve — {iv_label}")

    freqs_selected = st.multiselect(
        "PnL bar frequency to show", list(_BAR_FREQ.keys()), default=["Weekly"],
        key=f"freqs_{mode}",
        help="Showing all three at once gets cluttered — pick whichever you want to look at.",
    )
    for freq_label in freqs_selected:
        render_pnl_bars(cell_results, always, static, freq_label, iv_label)

    st.markdown(f"**Difference bars — {iv_label}**")
    st.caption(
        "PnL(A) − PnL(B) per period. A and B can each be any (lookback, n_buckets) "
        "cell from this grid, always-trade (\"no cutoff\"), or the static threshold."
    )
    diff_options = ["Always-trade (no cutoff)", "Static threshold"] + combo_labels
    dc1, dc2 = st.columns(2)
    with dc1:
        variant_A = st.selectbox("Variant A", diff_options, index=0, key=f"diffA_{mode}")
    with dc2:
        variant_B = st.selectbox("Variant B", diff_options, index=1, key=f"diffB_{mode}")

    def _variant_daily_pnl(label: str) -> pd.Series:
        if label == "Always-trade (no cutoff)":
            return always["_daily_pnl"]
        if label == "Static threshold":
            return static["_daily_pnl"]
        return get_cell(*combo_lookup[label]).daily_pnl

    pnl_A = _variant_daily_pnl(variant_A)
    pnl_B = _variant_daily_pnl(variant_B)
    diff_freqs_selected = st.multiselect(
        "Difference bar frequency to show", list(_BAR_FREQ.keys()), default=["Weekly"],
        key=f"diff_freqs_{mode}",
    )
    for freq_label in diff_freqs_selected:
        render_diff_bars(pnl_A, variant_A, pnl_B, variant_B, freq_label, iv_label)

    st.markdown(f"**Inspect a single cell — {iv_label}**")
    st.caption(
        "Pick one (lookback, n_buckets) to see its day-by-day cutoff — "
        "overlaying multiple here wouldn't be readable."
    )
    c1, c2 = st.columns(2)
    with c1:
        inspect_L = st.selectbox("Lookback to inspect", this_lookbacks, key=f"inspect_L_{mode}")
    with c2:
        inspect_B = st.selectbox("Buckets to inspect", this_buckets, key=f"inspect_B_{mode}")

    cell_result = get_cell(inspect_L, inspect_B)
    render_cutoff_timeline(cell_result, inspect_B, f"L={inspect_L}, B={inspect_B} — {iv_label}",
                           rule=cutoff_rule)

    # ── Specific combination — independent BPV(L,B) vs IV(L,B), and_both/or_either only
    if mode in ("and_both", "or_either"):
        st.markdown(f"**Specific combination — {iv_label}**")
        st.caption(
            "Independent BPV(L,B) and IV(L,B) — a single non-swept lookup, not a "
            "sweep. No composite/rank shown here: ranking is only meaningful "
            "against a population of cells to compare against, which a freely "
            "chosen independent pair generally isn't part of."
        )
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            spec_harcj_L = st.selectbox("BPV lookback", lookbacks, key=f"spec_harcj_L_{mode}")
        with sc2:
            spec_harcj_B = st.selectbox("BPV buckets", buckets, key=f"spec_harcj_B_{mode}")
        with sc3:
            spec_iv_L = st.selectbox("IV lookback", iv_lookbacks or lookbacks, key=f"spec_iv_L_{mode}")
        with sc4:
            spec_iv_B = st.selectbox("IV buckets", iv_buckets or buckets, key=f"spec_iv_B_{mode}")

        spec_result = cached_paired_cell(
            source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
            spec_harcj_L, spec_harcj_B, spec_iv_L, spec_iv_B,
            method, exclusion_rule, int(n_exclude),
            iv_method, iv_exclusion_rule, int(iv_n_exclude),
            mode, int(min_days_per_bucket),
        )
        spec_pbo = dt.pbo_for_result(spec_result, static["_daily_pnl"], cscv_metric,
                                      int(cscv_max_S), int(cscv_min_partition))

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Avg daily PnL", f"{spec_result.metrics['avg_daily_pnl']:,.0f}")
        m2.metric("Total PnL", f"{spec_result.metrics['total_pnl']:,.0f}")
        m3.metric("Sortino", f"{spec_result.metrics['sortino']:.2f}")
        m4.metric("Max drawdown", f"{spec_result.metrics['drawdown']:,.0f}")
        m5.metric("Max daily loss", f"{spec_result.metrics['max_daily_loss']:,.0f}")
        m6.metric("PBO vs static", f"{spec_pbo['pbo']:.3f}")
        st.caption(
            f"Win rate: {spec_result.metrics['win_rate']:.1%} — "
            f"{spec_result.metrics['n_traded']}/{spec_result.metrics['n_days']} days traded — "
            f"reliable: {spec_result.reliable} — "
            f"PBO combinations: {spec_pbo['n_combinations']}"
        )

        spec_label = f"BPV L={spec_harcj_L},B={spec_harcj_B} / IV L={spec_iv_L},B={spec_iv_B}"

        render_multi_equity_curve(
            {spec_label: spec_result}, always, static,
            f"Equity curve — Specific combination ({iv_label})",
        )

        spec_freqs_selected = st.multiselect(
            "PnL bar frequency to show", list(_BAR_FREQ.keys()), default=["Weekly"],
            key=f"spec_freqs_{mode}",
        )
        for freq_label in spec_freqs_selected:
            render_pnl_bars({spec_label: spec_result}, always, static, freq_label,
                             f"Specific combination ({iv_label})")

        st.markdown(f"**Difference bars — Specific combination ({iv_label})**")
        st.caption(
            "This specific combination vs. any variant B — always-trade, the "
            "static threshold, or any (lookback, n_buckets) from this grid."
        )
        spec_diff_options = ["Always-trade (no cutoff)", "Static threshold"] + combo_labels
        spec_variant_B = st.selectbox(
            "Variant B", spec_diff_options, index=0, key=f"spec_diffB_{mode}",
        )
        spec_pnl_B = _variant_daily_pnl(spec_variant_B)
        spec_diff_freqs_selected = st.multiselect(
            "Difference bar frequency to show", list(_BAR_FREQ.keys()), default=["Weekly"],
            key=f"spec_diff_freqs_{mode}",
        )
        for freq_label in spec_diff_freqs_selected:
            render_diff_bars(spec_result.daily_pnl, spec_label, spec_pnl_B, spec_variant_B,
                              freq_label, iv_label)

        render_cutoff_timeline(
            spec_result, spec_iv_B, f"{spec_label} — Specific combination ({iv_label})",
            rule=exclusion_rule,
        )

    st.divider()
