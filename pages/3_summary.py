"""
Summary — cross-mode leaderboards + unified variant comparison.

Computes all 4 IV-combination modes (harcj_only, iv_only, and_both,
or_either) from ONE set of sidebar settings, shows each mode's own
composite-score leaderboard separately (never merged into one ranking —
each mode's pool of variants is its own population), then lets you pick
any variants from ANY of those pools (plus the always-trade and static-
threshold baselines) to overlay on the same equity curve / PnL bars /
difference bars already used on the Threshold Grid page.

"Pool" = every (lookback, n_buckets) cell produced by the current sidebar
settings for that mode — for and_both/or_either that's the paired grid
(same L, B on both sides), not the independent "Specific combination"
one-offs from the Threshold Grid page.
"""

from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import available_profiles, load_joined, load_static_config
import dynamic_threshold as dt
from chart_helpers import (
    render_multi_equity_curve, render_pnl_bars, render_diff_bars, _BAR_FREQ,
)
from grid_cache import (
    cached_grid, cached_baselines, cached_cell, cached_pbo_grid,
    cached_iv_grid, cached_iv_pbo_grid, cached_iv_cell, cached_iv_baselines,
    cached_paired_grid, cached_paired_pbo_grid, cached_paired_cell,
)

st.set_page_config(page_title="Summary — HARCJ Research", layout="wide")
st.title("Summary")
st.caption(
    "One leaderboard per IV-combination mode (harcj_only, iv_only, and_both, "
    "or_either) — never merged, since each mode's pool of variants is its own "
    "population to rank within. Below that: pick variants from any mode's pool "
    "to compare on the same charts."
)

# Maps the live params.json flag_combine_method to this app's IV_MODES naming
# — used to pick ONE canonical static-threshold reference line for the whole
# page (see conversation history: "or"→and_both, "and"→or_either, "ignore_harcj"→iv_only).
_LIVE_MODE_MAP = {"ignore_harcj": "iv_only", "or": "and_both", "and": "or_either"}

MODE_LABELS = {
    "harcj_only": "HARCJ only",
    "iv_only":    "IV only",
    "and_both":   "AND (both must pass)",
    "or_either":  "OR (either passing is enough)",
}


def _parse_int_list(raw: str) -> list[int]:
    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return sorted(set(out))


# ── Sidebar (self-contained — mirrors Threshold Grid's controls) ──────────────

with st.sidebar:
    st.header("Controls")

    profile = st.selectbox("Profile", available_profiles(), index=0)
    df = load_joined(profile)
    st.caption(f"Usable days: {len(df)} ({df.index.min().date()} → {df.index.max().date()})")

    st.subheader("BPV settings")
    lookback_raw = st.text_input("BPV lookback grid (trading days, comma-separated)",
                                  value="20,40,60,90,120,180,252")
    bucket_raw   = st.text_input("BPV bucket-count grid (comma-separated)", value="4,5,8,10")
    lookbacks = _parse_int_list(lookback_raw)
    buckets   = _parse_int_list(bucket_raw)

    method = st.radio(
        "BPV bucketing method", options=["equal_count", "equal_width"],
        format_func=lambda m: "Equal-count (quantile)" if m == "equal_count" else "Equal-width (range)",
        key="bpv_method",
    )
    exclusion_rule = st.radio(
        "BPV exclusion rule", options=["worst_pnl_dynamic", "highest_vol"],
        format_func=lambda r: "Worst-PnL bucket(s), dynamic" if r == "worst_pnl_dynamic" else "Highest-vol bucket(s)",
        key="bpv_rule",
    )
    n_exclude = st.number_input("BPV — number of buckets to exclude", min_value=1,
                                 max_value=10, value=1, step=1, key="bpv_n_exclude")

    st.divider()
    st.subheader("IV settings")
    st.caption(
        "From the raw 1-min IV dump (the trustworthy source — see conversation "
        "history on roll.csv's mean_iv staleness). Only 07:30–08:30 CT has data."
    )
    iv_source_mode = st.radio(
        "IV source", options=["point", "mean"],
        format_func=lambda m: "Open IV (single timestamp)" if m == "point" else "Open IV mean (window average)",
    )
    if iv_source_mode == "point":
        iv_timestamp  = st.text_input("Timestamp (HH:MM, within 07:30–08:30)", value="08:30")
        iv_start_time = None
        iv_end_time   = None
    else:
        iv_timestamp = None
        c1, c2 = st.columns(2)
        with c1:
            iv_start_time = st.text_input("Window start (HH:MM)", value="07:30")
        with c2:
            iv_end_time = st.text_input("Window end (HH:MM)", value="08:30")

    iv_lookback_raw = st.text_input("IV lookback grid (trading days, comma-separated)",
                                     value="20,40,60,90,120,180,252")
    iv_bucket_raw   = st.text_input("IV bucket-count grid (comma-separated)", value="4,5,8,10")
    iv_lookbacks = _parse_int_list(iv_lookback_raw)
    iv_buckets   = _parse_int_list(iv_bucket_raw)

    iv_method = st.radio(
        "IV bucketing method", options=["equal_count", "equal_width"],
        format_func=lambda m: "Equal-count (quantile)" if m == "equal_count" else "Equal-width (range)",
        key="iv_method",
    )
    iv_exclusion_rule = st.radio(
        "IV exclusion rule", options=["worst_pnl_dynamic", "highest_vol"],
        format_func=lambda r: "Worst-PnL bucket(s), dynamic" if r == "worst_pnl_dynamic" else "Highest-IV bucket(s)",
        key="iv_rule",
    )
    iv_n_exclude = st.number_input("IV — number of buckets to exclude", min_value=1,
                                    max_value=10, value=1, step=1, key="iv_n_exclude")

    st.divider()
    min_days_per_bucket = st.number_input(
        "Min days/bucket for 'reliable'", min_value=1, max_value=100, value=10, step=1,
    )

    st.divider()
    st.subheader("PBO")
    cscv_metric = st.selectbox("CSCV ranking metric", ["sortino", "sharpe", "calmar", "total_pnl"], index=0)
    cscv_max_S = st.number_input("Max S (time partitions)", min_value=4, max_value=32, value=16, step=2)
    cscv_min_partition = st.number_input("Min days per partition", min_value=1, max_value=50, value=4, step=1)

    st.divider()
    st.subheader("Composite score")
    weight_pbo     = st.number_input("Weight — PBO",            min_value=0.0, value=0.49, step=0.01)
    weight_avg_pnl = st.number_input("Weight — Avg daily PnL",  min_value=0.0, value=0.17, step=0.01)
    weight_sortino = st.number_input("Weight — Sortino",        min_value=0.0, value=0.17, step=0.01)
    weight_mdl     = st.number_input("Weight — Max daily loss", min_value=0.0, value=0.17, step=0.01)
    top_n = st.number_input("Top N per leaderboard", min_value=1, max_value=50, value=5, step=1)

if not lookbacks or not buckets or not iv_lookbacks or not iv_buckets:
    st.warning("Enter at least one lookback and one bucket count for both BPV and IV.")
    st.stop()

cfg = load_static_config(profile)
weights = {"pbo": weight_pbo, "avg_daily_pnl": weight_avg_pnl,
           "sortino": weight_sortino, "max_daily_loss": weight_mdl}


# ── Compute all 4 modes' grids + composite tables ───────────────────────────────

@st.cache_data(show_spinner=False)
def _all_composites(profile, lookbacks, buckets, method, exclusion_rule, n_exclude,
                     iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
                     iv_lookbacks, iv_buckets, iv_method, iv_exclusion_rule, iv_n_exclude,
                     min_days_per_bucket, cscv_metric, cscv_max_S, cscv_min_partition, weights):
    cfg = load_static_config(profile)
    tables = {}

    grid = cached_grid(profile, tuple(lookbacks), tuple(buckets), method, exclusion_rule,
                        int(n_exclude), cfg["iv_cutoff"], "harcj_only", int(min_days_per_bucket))
    pbo = cached_pbo_grid(profile, tuple(lookbacks), tuple(buckets), method, exclusion_rule,
                           int(n_exclude), cfg["iv_cutoff"], "harcj_only", int(min_days_per_bucket),
                           cscv_metric, int(cscv_max_S), int(cscv_min_partition))
    tables["harcj_only"] = dt.composite_score_table(grid, pbo, weights)

    grid = cached_iv_grid(profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
                           tuple(iv_lookbacks), tuple(iv_buckets), iv_method, iv_exclusion_rule,
                           int(iv_n_exclude), int(min_days_per_bucket))
    pbo = cached_iv_pbo_grid(profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
                              tuple(iv_lookbacks), tuple(iv_buckets), iv_method, iv_exclusion_rule,
                              int(iv_n_exclude), int(min_days_per_bucket),
                              cscv_metric, int(cscv_max_S), int(cscv_min_partition))
    tables["iv_only"] = dt.composite_score_table(grid, pbo, weights)

    for mode in ("and_both", "or_either"):
        grid = cached_paired_grid(profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
                                   tuple(lookbacks), tuple(buckets),
                                   method, exclusion_rule, int(n_exclude),
                                   iv_method, iv_exclusion_rule, int(iv_n_exclude),
                                   mode, int(min_days_per_bucket))
        pbo = cached_paired_pbo_grid(profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
                                      tuple(lookbacks), tuple(buckets),
                                      method, exclusion_rule, int(n_exclude),
                                      iv_method, iv_exclusion_rule, int(iv_n_exclude),
                                      mode, int(min_days_per_bucket),
                                      cscv_metric, int(cscv_max_S), int(cscv_min_partition))
        tables[mode] = dt.composite_score_table(grid, pbo, weights)

    return tables


composite_tables = _all_composites(
    profile, lookbacks, buckets, method, exclusion_rule, n_exclude,
    iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
    iv_lookbacks, iv_buckets, iv_method, iv_exclusion_rule, iv_n_exclude,
    min_days_per_bucket, cscv_metric, cscv_max_S, cscv_min_partition, weights,
)

# ── 4 separate leaderboards ──────────────────────────────────────────────────

st.header("Leaderboards")
for mode in ("harcj_only", "iv_only", "and_both", "or_either"):
    st.subheader(MODE_LABELS[mode])
    table = composite_tables[mode]
    if table.empty:
        st.info("No cells have data for all four metrics — nothing to rank.")
        continue
    display = table.head(int(top_n)).copy()
    display["avg_daily_pnl"]  = display["avg_daily_pnl"].map(lambda v: f"{v:,.0f}")
    display["sortino"]        = display["sortino"].map(lambda v: f"{v:.2f}")
    display["max_daily_loss"] = display["max_daily_loss"].map(lambda v: f"{v:,.0f}")
    display["pbo"]            = display["pbo"].map(lambda v: f"{v:.3f}")
    display["composite"]      = display["composite"].map(lambda v: f"{v:.2f}")
    st.dataframe(
        display[["lookback", "n_buckets", "composite", "avg_daily_pnl", "sortino",
                 "max_daily_loss", "pbo", "reliable"]],
        use_container_width=True, hide_index=True,
    )

st.divider()

# ── Unified variant pool ────────────────────────────────────────────────────────

st.header("Compare variants across all modes")

canonical_mode = _LIVE_MODE_MAP.get(cfg["flag_combine_method"], "harcj_only")
always, static, _ = cached_baselines(profile, canonical_mode)
st.caption(
    f"Reference lines use the live-configured combination mode for this profile "
    f"(`flag_combine_method={cfg['flag_combine_method']}` → {MODE_LABELS[canonical_mode]}), "
    "so \"Static threshold\" below matches what the live system actually does today."
)


def get_cell(mode: str, L: int, B: int):
    if mode == "harcj_only":
        return cached_cell(profile, L, B, method, exclusion_rule,
                            int(n_exclude), cfg["iv_cutoff"], "harcj_only", int(min_days_per_bucket))
    if mode == "iv_only":
        return cached_iv_cell(profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
                               L, B, iv_method, iv_exclusion_rule, int(iv_n_exclude), int(min_days_per_bucket))
    return cached_paired_cell(
        profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
        L, B, L, B, method, exclusion_rule, int(n_exclude),
        iv_method, iv_exclusion_rule, int(iv_n_exclude), mode, int(min_days_per_bucket),
    )


pool_labels = []
pool_lookup = {}
for L in lookbacks:
    for B in buckets:
        label = f"harcj_only: L={L}, B={B}"
        pool_labels.append(label)
        pool_lookup[label] = ("harcj_only", L, B)
for L in iv_lookbacks:
    for B in iv_buckets:
        label = f"iv_only: L={L}, B={B}"
        pool_labels.append(label)
        pool_lookup[label] = ("iv_only", L, B)
for mode in ("and_both", "or_either"):
    for L in lookbacks:
        for B in buckets:
            label = f"{mode}: L={L}, B={B}"
            pool_labels.append(label)
            pool_lookup[label] = (mode, L, B)

st.caption(f"Pool size: {len(pool_labels)} variants across all 4 modes.")

combos_selected = st.multiselect("Variants to compare", pool_labels, default=pool_labels[:1])
cell_results = {label: get_cell(*pool_lookup[label]) for label in combos_selected}
render_multi_equity_curve(cell_results, always, static, "Equity curve — across all modes")

freqs_selected = st.multiselect(
    "PnL bar frequency to show", list(_BAR_FREQ.keys()), default=["Weekly"],
    help="Showing all three at once gets cluttered — pick whichever you want to look at.",
)
for freq_label in freqs_selected:
    render_pnl_bars(cell_results, always, static, freq_label, "Across all modes")

st.subheader("Difference bars")
st.caption(
    "PnL(A) − PnL(B) per period. A and B can each be any variant from any "
    "mode's pool, always-trade (\"no cutoff\"), or the static threshold."
)
diff_options = ["Always-trade (no cutoff)", "Static threshold"] + pool_labels
dc1, dc2 = st.columns(2)
with dc1:
    variant_A = st.selectbox("Variant A", diff_options, index=0)
with dc2:
    variant_B = st.selectbox("Variant B", diff_options, index=1)


def _variant_daily_pnl(label: str):
    if label == "Always-trade (no cutoff)":
        return always["_daily_pnl"]
    if label == "Static threshold":
        return static["_daily_pnl"]
    return get_cell(*pool_lookup[label]).daily_pnl


pnl_A = _variant_daily_pnl(variant_A)
pnl_B = _variant_daily_pnl(variant_B)
diff_freqs_selected = st.multiselect(
    "Difference bar frequency to show", list(_BAR_FREQ.keys()), default=["Weekly"],
)
for freq_label in diff_freqs_selected:
    render_diff_bars(pnl_A, variant_A, pnl_B, variant_B, freq_label, "Across all modes")
