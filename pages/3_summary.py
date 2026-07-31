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

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_loader
import backtest_source
import dynamic_threshold as dt
from chart_helpers import (
    render_multi_equity_curve, render_pnl_bars, render_diff_bars, _BAR_FREQ,
)
from grid_cache import (
    cached_grid, cached_baselines, cached_cell, cached_pbo_grid,
    cached_iv_grid, cached_iv_pbo_grid, cached_iv_cell, cached_iv_baselines,
    cached_paired_grid, cached_paired_pbo_grid, cached_paired_cell,
)

_SOURCES = {
    data_loader.DATA_SOURCE_NAME: data_loader,
    backtest_source.DATA_SOURCE_NAME: backtest_source,
}
_SOURCE_LABELS = {
    data_loader.DATA_SOURCE_NAME: "Live dumps (dashboard_new/dumps/)",
    backtest_source.DATA_SOURCE_NAME: "Backtest — spxw_gamma_hedge_false",
}

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
    bucket_raw   = st.text_input("BPV bucket-count grid (comma-separated)", value="4,5,8,10")
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
        "BPV bucketing method", options=["equal_count", "equal_width"],
        format_func=lambda m: "Equal-count (quantile)" if m == "equal_count" else "Equal-width (range)",
        key="bpv_method",
    )
    # Only rule-based exclusion is exposed — see Threshold Grid page for why
    # the worst-PnL dynamic rule was removed.
    exclusion_rule = "highest_vol"
    n_exclude = st.number_input("BPV — number of buckets to exclude", min_value=1,
                                 max_value=10, value=1, step=1, key="bpv_n_exclude")

    st.divider()
    st.subheader("IV settings")
    if source == backtest_source.DATA_SOURCE_NAME:
        _sess_start, _ = backtest_source.derive_session_window()
        _default_iv_end = _sess_start
        _default_iv_start = (
            pd.Timestamp(_sess_start) - pd.Timedelta(hours=backtest_source.DEFAULT_IV_WINDOW_HOURS_BEFORE_START)
        ).strftime("%H:%M")
        st.caption(
            "From research/atm_iv/SPWX/ (weekday-matched, 00:00–23:59 CT "
            "available, 2025 warm-up + this backtest's own 2026 days). Defaults "
            f"to the hour before session start ({_default_iv_start}–{_default_iv_end})."
        )
    else:
        _default_iv_start, _default_iv_end = "07:30", "08:30"
        st.caption(
            "From the raw 1-min IV dump (the trustworthy source — see conversation "
            "history on roll.csv's mean_iv staleness). Only 07:30–08:30 CT has data."
        )
    iv_source_mode = st.radio(
        "IV source", options=["mean", "point"],
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
    iv_bucket_raw   = st.text_input("IV bucket-count grid (comma-separated)", value="4,5,8,10")
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
        "IV bucketing method", options=["equal_count", "equal_width"],
        format_func=lambda m: "Equal-count (quantile)" if m == "equal_count" else "Equal-width (range)",
        key="iv_method",
    )
    iv_exclusion_rule = "highest_vol"
    iv_n_exclude = st.number_input("IV — number of buckets to exclude", min_value=1,
                                    max_value=10, value=1, step=1, key="iv_n_exclude")

    # Was configurable for the removed worst-PnL dynamic rule's "reliable"
    # flag — highest-vol/IV has no such reliability concern, so this is just
    # the fixed value passed through to the grid functions.
    min_days_per_bucket = 10

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

cfg = loader.load_static_config(profile)
weights = {"pbo": weight_pbo, "avg_daily_pnl": weight_avg_pnl,
           "sortino": weight_sortino, "max_daily_loss": weight_mdl}


# ── Compute all 4 modes' grids + composite tables ───────────────────────────────
# Each cached_* call below is individually cached (grid_cache.py) and already
# carries its own spinner text — deliberately NOT wrapped in one outer cached
# function (that would hide all 8 grid/PBO computations behind a single
# opaque, silent call). Instead a visible status block updates step by step,
# so a slow run (PBO especially) shows *which* of the 4 modes it's on rather
# than going blank for however long the whole thing takes.

composite_tables = {}
with st.status("Computing all 4 mode leaderboards…", expanded=True) as status:
    st.write("HARCJ only — grid…")
    grid = cached_grid(source, profile, tuple(lookbacks), tuple(buckets), method, exclusion_rule,
                        int(n_exclude), cfg["iv_cutoff"], "harcj_only", int(min_days_per_bucket))
    st.write("HARCJ only — PBO grid (this is usually the slow part)…")
    pbo = cached_pbo_grid(source, profile, tuple(lookbacks), tuple(buckets), method, exclusion_rule,
                           int(n_exclude), cfg["iv_cutoff"], "harcj_only", int(min_days_per_bucket),
                           cscv_metric, int(cscv_max_S), int(cscv_min_partition))
    composite_tables["harcj_only"] = dt.composite_score_table(grid, pbo, weights)
    st.write("✓ HARCJ only done (1/4)")

    st.write("IV only — grid…")
    grid = cached_iv_grid(source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
                           tuple(iv_lookbacks), tuple(iv_buckets), iv_method, iv_exclusion_rule,
                           int(iv_n_exclude), int(min_days_per_bucket))
    st.write("IV only — PBO grid…")
    pbo = cached_iv_pbo_grid(source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
                              tuple(iv_lookbacks), tuple(iv_buckets), iv_method, iv_exclusion_rule,
                              int(iv_n_exclude), int(min_days_per_bucket),
                              cscv_metric, int(cscv_max_S), int(cscv_min_partition))
    composite_tables["iv_only"] = dt.composite_score_table(grid, pbo, weights)
    st.write("✓ IV only done (2/4)")

    for i, mode in enumerate(("and_both", "or_either"), start=3):
        st.write(f"{MODE_LABELS[mode]} — paired grid…")
        grid = cached_paired_grid(source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
                                   tuple(lookbacks), tuple(buckets),
                                   method, exclusion_rule, int(n_exclude),
                                   iv_method, iv_exclusion_rule, int(iv_n_exclude),
                                   mode, int(min_days_per_bucket))
        st.write(f"{MODE_LABELS[mode]} — paired PBO grid…")
        pbo = cached_paired_pbo_grid(source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
                                      tuple(lookbacks), tuple(buckets),
                                      method, exclusion_rule, int(n_exclude),
                                      iv_method, iv_exclusion_rule, int(iv_n_exclude),
                                      mode, int(min_days_per_bucket),
                                      cscv_metric, int(cscv_max_S), int(cscv_min_partition))
        composite_tables[mode] = dt.composite_score_table(grid, pbo, weights)
        st.write(f"✓ {MODE_LABELS[mode]} done ({i}/4)")

    status.update(label="All 4 mode leaderboards computed", state="complete", expanded=False)

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
    for c in ["rank_avg_daily_pnl", "rank_sortino", "rank_max_daily_loss", "rank_pbo"]:
        display[c] = display[c].map(lambda v: f"{v:.1f}")
    st.dataframe(
        display[["lookback", "n_buckets", "composite", "avg_daily_pnl", "sortino",
                 "max_daily_loss", "pbo", "rank_avg_daily_pnl", "rank_sortino",
                 "rank_max_daily_loss", "rank_pbo", "reliable"]],
        use_container_width=True, hide_index=True,
    )

st.divider()

# ── Unified variant pool ────────────────────────────────────────────────────────

st.header("Compare variants across all modes")

canonical_mode = _LIVE_MODE_MAP.get(cfg["flag_combine_method"], "harcj_only")
always, static, _ = cached_baselines(source, profile, canonical_mode)
st.caption(
    f"Reference lines use the live-configured combination mode for this profile "
    f"(`flag_combine_method={cfg['flag_combine_method']}` → {MODE_LABELS[canonical_mode]}), "
    "so \"Static threshold\" below matches what the live system actually does today."
)


def get_cell(mode: str, L: int, B: int):
    if mode == "harcj_only":
        return cached_cell(source, profile, L, B, method, exclusion_rule,
                            int(n_exclude), cfg["iv_cutoff"], "harcj_only", int(min_days_per_bucket))
    if mode == "iv_only":
        return cached_iv_cell(source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
                               L, B, iv_method, iv_exclusion_rule, int(iv_n_exclude), int(min_days_per_bucket))
    return cached_paired_cell(
        source, profile, iv_source_mode, iv_timestamp, iv_start_time, iv_end_time,
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

st.divider()

# ── Variant day breakdown ────────────────────────────────────────────────────────

st.header("Variant day breakdown")
st.caption(
    "Pick any number of variants — from any mode's pool, or the always-trade / "
    "live static-threshold baselines — to compare side by side. For each: total "
    "scored days, how many were cut off vs. traded, and — for cut-off days "
    "specifically — what would have happened (positive/negative, and average "
    "size) had the cutoff not applied, using the actual raw PnL for those dates "
    "rather than the realized (zeroed-out) PnL."
)

breakdown_options = ["Always-trade (no cutoff)", "Static threshold"] + pool_labels


def _variant_traded_mask(label: str) -> pd.Series:
    if label == "Always-trade (no cutoff)":
        return always["_traded_mask"]
    if label == "Static threshold":
        return static["_traded_mask"]
    return get_cell(*pool_lookup[label]).traded_mask


def _breakdown_row(label: str) -> dict:
    traded_mask = _variant_traded_mask(label)
    scored_dates = traded_mask.index
    raw_pnl = df["pnl"].reindex(scored_dates)

    skipped_mask = ~traded_mask
    skipped_pnl = raw_pnl[skipped_mask]
    traded_pnl  = raw_pnl[traded_mask]

    skipped_pos = skipped_pnl[skipped_pnl > 0]
    skipped_neg = skipped_pnl[skipped_pnl < 0]
    traded_pos  = traded_pnl[traded_pnl > 0]
    traded_neg  = traded_pnl[traded_pnl < 0]

    traded_dates = scored_dates[traded_mask]

    return {
        "Variant":        label,
        "Total days":     len(traded_mask),
        "Traded from":    traded_dates.min().date() if len(traded_dates) else None,
        "Traded to":      traded_dates.max().date() if len(traded_dates) else None,
        "Cut off":        int(skipped_mask.sum()),
        "Cutoff +":       int(len(skipped_pos)),
        "Cutoff avg +":   skipped_pos.mean() if len(skipped_pos) else None,
        "Cutoff -":       int(len(skipped_neg)),
        "Cutoff avg -":   skipped_neg.mean() if len(skipped_neg) else None,
        "Traded":         int(traded_mask.sum()),
        "Traded +":       int(len(traded_pos)),
        "Traded avg +":   traded_pos.mean() if len(traded_pos) else None,
        "Traded -":       int(len(traded_neg)),
        "Traded avg -":   traded_neg.mean() if len(traded_neg) else None,
    }


breakdown_selected = st.multiselect(
    "Variants", breakdown_options, default=["Static threshold"], key="breakdown_variants",
)

if breakdown_selected:
    rows = [_breakdown_row(lbl) for lbl in breakdown_selected]
    breakdown_df = pd.DataFrame(rows).set_index("Variant")
    for c in ["Cutoff avg +", "Cutoff avg -", "Traded avg +", "Traded avg -"]:
        breakdown_df[c] = breakdown_df[c].map(lambda v: f"{v:,.0f}" if pd.notna(v) else "—")
    st.dataframe(breakdown_df, use_container_width=True)
else:
    st.info("Select at least one variant to see its breakdown.")

st.subheader("Compare two variants — cutoff overlap")
st.caption(
    "A and B usually cover different date ranges (different lookbacks start "
    "scoring at different points), so comparing their raw cutoff counts "
    "directly is misleading. This splits each variant's cut-off dates into "
    "three buckets — dates cut off by A only, by B only, and by both — and "
    "shows the same stats (using actual raw PnL) for each bucket, so the "
    "comparison only ever covers dates both variants actually had an opinion on."
)

cc1, cc2 = st.columns(2)
with cc1:
    cmp_A = st.selectbox("Variant A", breakdown_options, index=1, key="cmp_variant_A")  # "Static threshold"
with cc2:
    cmp_B_default = 2 if len(breakdown_options) > 2 else 0  # first pool variant, not "Always-trade"
    cmp_B = st.selectbox("Variant B", breakdown_options, index=cmp_B_default, key="cmp_variant_B")

mask_A = _variant_traded_mask(cmp_A)
mask_B = _variant_traded_mask(cmp_B)
cutoff_A = set(mask_A.index[~mask_A])
cutoff_B = set(mask_B.index[~mask_B])

common    = cutoff_A & cutoff_B
only_A    = cutoff_A - common
only_B    = cutoff_B - common


def _date_set_stats(dates) -> dict:
    idx = pd.DatetimeIndex(sorted(dates))
    raw = df["pnl"].reindex(idx)
    pos = raw[raw > 0]
    neg = raw[raw < 0]
    return {
        "Days":         len(idx),
        "Positive":     int(len(pos)),
        "Avg positive": pos.mean() if len(pos) else None,
        "Negative":     int(len(neg)),
        "Avg negative": neg.mean() if len(neg) else None,
    }


cmp_rows = {
    f"A only ({cmp_A})":          _date_set_stats(only_A),
    f"Common (both cut off)":     _date_set_stats(common),
    f"B only ({cmp_B})":          _date_set_stats(only_B),
}
cmp_df = pd.DataFrame(cmp_rows).T
cmp_df["Avg positive"] = cmp_df["Avg positive"].map(lambda v: f"{v:,.0f}" if pd.notna(v) else "—")
cmp_df["Avg negative"] = cmp_df["Avg negative"].map(lambda v: f"{v:,.0f}" if pd.notna(v) else "—")
st.dataframe(cmp_df, use_container_width=True)
st.caption(
    f"Total cutoff days — A: {len(cutoff_A)}, B: {len(cutoff_B)}, "
    f"overlap: {len(common)} ({len(common) / max(len(cutoff_A | cutoff_B), 1):.0%} of the union)."
)
