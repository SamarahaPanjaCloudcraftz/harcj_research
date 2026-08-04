"""
Cutoff Transfer — does another source's cutoff DATES, applied to this
source's own PnL, beat this source's own best cutoff criteria?

"Applying a cutoff from A to B" here is a pure date-mask transplant, not a
recipe transplant: take the literal set of calendar dates A's best
(mode, L, B) walk-forward decision skipped, and skip those SAME dates in
B's own PnL — B's own signals are never touched. This tests whether
cross-source "bad days" agree, not whether a strategy-shape generalizes
(see conversation history for why recipe-transfer was explicitly ruled out).

Scoped to the 4 backtest sources only (not live_dumps) — this is a
cross-backtest-variant question, and cutoff dates only mean something
where the calendar ranges genuinely overlap.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backtest_source
import cutoff_transfer as ct
import exchange_events as ee
from chart_helpers import render_multi_equity_curve, render_pnl_bars, _BAR_FREQ

_BACKTEST_LABELS = {
    "spxw_gamma_hedge_false": "SPXW gamma-hedge (05:00–15:00)",
    "spxw_delta_condor_0830_1200": "SPXW delta-condor 08:30–12:00",
    "ndaq_delta_condor_0830_1200": "NDAQ delta-condor 08:30–12:00",
    "ndaq_delta_condor_0600_1000": "NDAQ delta-condor 06:00–10:00",
}

_SOURCES = {src.DATA_SOURCE_NAME: src for src in backtest_source.ALL_SOURCES}


def _label(name: str) -> str:
    return _BACKTEST_LABELS.get(name, name)


def _underlying_for_source(source: str) -> str:
    for underlying, pair in ee.SOURCE_PAIRS.items():
        if source in (pair["default"], pair["event"]):
            return underlying
    raise ValueError(f"No underlying mapping for source {source!r}")


st.set_page_config(page_title="Cutoff Transfer — HARCJ Research", layout="wide")
st.title("Cutoff Transfer")
st.caption(
    "Take the set of calendar dates one source's best cutoff skips, and skip "
    "those same dates in a different source's own PnL — no recomputation of "
    "the target's own signals. Tests whether cross-source \"bad days\" agree, "
    "not whether a strategy-shape generalizes."
)

with st.sidebar:
    st.header("Target")
    target_source = st.selectbox(
        "Apply cutoffs to (target)", list(_SOURCES.keys()), index=0, format_func=_label,
    )

target_profile = target_source  # every BacktestSource's profile name == its source name
target_underlying = _underlying_for_source(target_source)
event_dates = ee.load_event_dates()

# 08:30 variants only ever trade on event days live; the two default-window
# variants only ever trade on non-event days live. Every comparison below is
# shown TWICE for the selected target: once over all dates, once restricted
# to that day-type — not a replacement of the all-dates view, an addition.
_EVENT_SOURCES = {"spxw_delta_condor_0830_1200", "ndaq_delta_condor_0830_1200"}
_NON_EVENT_SOURCES = {"spxw_gamma_hedge_false", "ndaq_delta_condor_0600_1000"}
if target_source in _EVENT_SOURCES:
    target_mask_fn = lambda idx: idx.isin(event_dates)
    target_day_type = "event days only"
elif target_source in _NON_EVENT_SOURCES:
    target_mask_fn = lambda idx: ~idx.isin(event_dates)
    target_day_type = "non-event days only"
else:
    target_mask_fn = None
    target_day_type = "all dates"


def _restrict(x, mask_fn=None):
    mask_fn = target_mask_fn if mask_fn is None else mask_fn
    if mask_fn is None:
        return x
    if hasattr(x, "metrics"):
        return ct.restrict_result_to_mask(x, mask_fn)
    return ct.restrict_metrics_to_mask(x, mask_fn)


_, target_always_full, _ = ct.canonical_mode_and_baselines(target_source, target_source)
target_static_full = ee.live_baseline_result(target_underlying, event_dates).metrics
target_always_filtered = _restrict(target_always_full)
target_static_filtered = _restrict(target_static_full)

target_own_best = ct.best_cutoff(target_source, target_profile)
target_own_result_full = ct.cell_result_for(target_source, target_profile,
                                             target_own_best["mode"], target_own_best["lookback"], target_own_best["n_buckets"])
target_own_pbo_full = ct.transplant_pbo(target_source, target_profile, target_own_result_full)
target_own_result_filtered = _restrict(target_own_result_full)
target_own_pbo_filtered = ct.transplant_pbo(target_source, target_profile, target_own_result_filtered)

st.caption(
    f"Target: **{_label(target_source)}** — {target_always_full['n_days']} scored days total, "
    f"{target_always_filtered['n_days']} on **{target_day_type}**. "
    f"Target's own best cutoff: **{ct.MODE_LABELS[target_own_best['mode']]}**, "
    f"L={target_own_best['lookback']}, B={target_own_best['n_buckets']}."
)


def _metrics_row(label: str, result_or_dict, pbo: dict | None = None) -> dict:
    m = result_or_dict.metrics if hasattr(result_or_dict, "metrics") else result_or_dict
    return {
        "Variant":        label,
        "Avg daily PnL":  m["avg_daily_pnl"],
        "Total PnL":      m["total_pnl"],
        "Sortino":        m["sortino"],
        "Max drawdown":   m["drawdown"],
        "Max daily loss": m["max_daily_loss"],
        "PBO":            pbo["pbo"] if pbo is not None else None,
        "Traded":         f"{m['n_traded']}/{m['n_days']}",
    }


def _render_comparison(rows: list[dict], cell_results: dict, always: dict, static: dict, title_suffix: str):
    df = pd.DataFrame(rows).set_index("Variant")
    display = df.copy()
    for c in ["Avg daily PnL", "Total PnL", "Max drawdown", "Max daily loss"]:
        display[c] = display[c].map(lambda v: f"{v:,.0f}")
    display["Sortino"] = display["Sortino"].map(lambda v: f"{v:.2f}")
    display["PBO"] = display["PBO"].map(lambda v: f"{v:.3f}" if v is not None else "—")
    st.dataframe(display, use_container_width=True)

    render_multi_equity_curve(cell_results, always, static, f"Equity curve — {title_suffix}",
                               static_label="Current live baseline")
    freqs_selected = st.multiselect(
        "PnL bar frequency to show", list(_BAR_FREQ.keys()), default=["Weekly"],
        key=f"freqs_{title_suffix}",
    )
    for freq_label in freqs_selected:
        render_pnl_bars(cell_results, always, static, freq_label, title_suffix,
                         static_label="Current live baseline")


st.divider()
st.header("Preloaded — best cutoff from every other source")
st.caption(
    "For each of the other 3 sources: its own best cutoff (by Global Ranking), "
    "the calendar dates that cutoff skipped, applied to the target's own PnL. "
    "Shown twice: over all dates, and restricted to the target's day-type "
    "(event days only for the 08:30 variants, non-event days only for the "
    "default-window variants). \"Current live baseline\" = the fixed, "
    "non-walk-forward rule from Exchange Events (BPV-only on non-event days, "
    "BPV AND mean-IV on event days, using this underlying's real deployed "
    "thresholds) — not a static threshold on the target alone."
)

other_sources = [s for s in _SOURCES if s != target_source]

with st.status("Computing transplants from every other source…", expanded=True) as status:
    rows_full = [
        _metrics_row("Always-trade (target)", target_always_full),
        _metrics_row("Current live baseline (target)", target_static_full),
        _metrics_row(f"Target's own best ({ct.MODE_LABELS[target_own_best['mode']]}, "
                     f"L={target_own_best['lookback']}, B={target_own_best['n_buckets']})",
                     target_own_result_full, target_own_pbo_full),
    ]
    rows_filtered = [
        _metrics_row("Always-trade (target)", target_always_filtered),
        _metrics_row("Current live baseline (target)", target_static_filtered),
        _metrics_row(f"Target's own best ({ct.MODE_LABELS[target_own_best['mode']]}, "
                     f"L={target_own_best['lookback']}, B={target_own_best['n_buckets']})",
                     target_own_result_filtered, target_own_pbo_filtered),
    ]
    cell_results_full = {"Target's own best": target_own_result_full}
    cell_results_filtered = {"Target's own best": target_own_result_filtered}
    for origin_source in other_sources:
        st.write(f"{_label(origin_source)} — finding its best cutoff…")
        origin_best = ct.best_cutoff(origin_source, origin_source)
        cutoff_dates = ct.cutoff_dates_for(origin_source, origin_source,
                                            origin_best["mode"], origin_best["lookback"], origin_best["n_buckets"])
        st.write(f"{_label(origin_source)} — applying its {len(cutoff_dates)} cutoff dates to target…")
        transplanted_full = ct.apply_transplanted_cutoff(target_source, target_profile, cutoff_dates)
        transplanted_filtered = _restrict(transplanted_full)
        pbo_full = ct.transplant_pbo(target_source, target_profile, transplanted_full)
        pbo_filtered = ct.transplant_pbo(target_source, target_profile, transplanted_filtered)
        label = (f"From {_label(origin_source)} ({ct.MODE_LABELS[origin_best['mode']]}, "
                 f"L={origin_best['lookback']}, B={origin_best['n_buckets']})")
        rows_full.append(_metrics_row(label, transplanted_full, pbo_full))
        rows_filtered.append(_metrics_row(label, transplanted_filtered, pbo_filtered))
        cell_results_full[f"From {_label(origin_source)}"] = transplanted_full
        cell_results_filtered[f"From {_label(origin_source)}"] = transplanted_filtered
    status.update(label="All transplants computed", state="complete", expanded=False)

st.subheader("All dates")
_render_comparison(rows_full, cell_results_full, target_always_full, target_static_full,
                    "preloaded transplants — all dates")

st.subheader(f"Restricted to {target_day_type}")
_render_comparison(rows_filtered, cell_results_filtered, target_always_filtered, target_static_filtered,
                    f"preloaded transplants — {target_day_type}")

st.divider()
st.header("Custom")
st.caption(
    "Pick any origin source, mode, lookback, and bucket count — the cutoff dates "
    "that exact (mode, L, B) decision produces on the origin, applied to the target."
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    custom_origin = st.selectbox("Origin source (cutoff comes from)", list(_SOURCES.keys()),
                                  index=0, format_func=_label, key="custom_origin")
with c2:
    custom_mode = st.selectbox("Flag combination mode", list(ct.MODE_LABELS.keys()),
                                format_func=lambda m: ct.MODE_LABELS[m], key="custom_mode")
with c3:
    custom_L = st.number_input("Lookback (L)", min_value=2, max_value=500, value=20, step=1, key="custom_L")
with c4:
    custom_B = st.number_input("Buckets (B)", min_value=2, max_value=20, value=4, step=1, key="custom_B")

iv_kwargs = {}
if custom_mode in ("iv_only", "and_both", "or_either"):
    st.subheader("Origin's IV settings")
    _default_iv_start, _default_iv_end = ct.default_iv_window(custom_origin, custom_origin)
    custom_iv_source_mode = st.radio(
        "IV source", options=["mean", "point"],
        format_func=lambda m: "Open IV (single timestamp)" if m == "point" else "Open IV mean (window average)",
        key="custom_iv_mode",
    )
    if custom_iv_source_mode == "point":
        custom_iv_timestamp = st.text_input("Timestamp (HH:MM)", value=_default_iv_end, key="custom_iv_ts")
        custom_iv_start, custom_iv_end = None, None
    else:
        custom_iv_timestamp = None
        ic1, ic2 = st.columns(2)
        with ic1:
            custom_iv_start = st.text_input("Window start (HH:MM)", value=_default_iv_start, key="custom_iv_start")
        with ic2:
            custom_iv_end = st.text_input("Window end (HH:MM)", value=_default_iv_end, key="custom_iv_end")
    iv_kwargs = {
        "iv_source_mode": custom_iv_source_mode, "iv_timestamp": custom_iv_timestamp,
        "iv_start_time": custom_iv_start, "iv_end_time": custom_iv_end,
    }

if st.button("Compute custom transplant"):
    with st.spinner("Computing…"):
        custom_cutoff_dates = ct.cutoff_dates_for(custom_origin, custom_origin, custom_mode,
                                                   int(custom_L), int(custom_B), **iv_kwargs)
        custom_result_full = ct.apply_transplanted_cutoff(target_source, target_profile, custom_cutoff_dates)
        custom_result_filtered = _restrict(custom_result_full)
        custom_pbo_full = ct.transplant_pbo(target_source, target_profile, custom_result_full)
        custom_pbo_filtered = ct.transplant_pbo(target_source, target_profile, custom_result_filtered)

    custom_label = (f"From {_label(custom_origin)} ({ct.MODE_LABELS[custom_mode]}, "
                     f"L={int(custom_L)}, B={int(custom_B)})")
    st.caption(f"{len(custom_cutoff_dates)} cutoff dates from origin applied to target.")

    custom_rows_full = [
        _metrics_row("Always-trade (target)", target_always_full),
        _metrics_row(f"Target's own best ({ct.MODE_LABELS[target_own_best['mode']]}, "
                     f"L={target_own_best['lookback']}, B={target_own_best['n_buckets']})",
                     target_own_result_full, target_own_pbo_full),
        _metrics_row(custom_label, custom_result_full, custom_pbo_full),
    ]
    custom_cell_results_full = {
        "Target's own best": target_own_result_full,
        custom_label: custom_result_full,
    }
    st.subheader("All dates")
    _render_comparison(custom_rows_full, custom_cell_results_full, target_always_full, target_static_full,
                        "custom transplant — all dates")

    custom_rows_filtered = [
        _metrics_row("Always-trade (target)", target_always_filtered),
        _metrics_row(f"Target's own best ({ct.MODE_LABELS[target_own_best['mode']]}, "
                     f"L={target_own_best['lookback']}, B={target_own_best['n_buckets']})",
                     target_own_result_filtered, target_own_pbo_filtered),
        _metrics_row(custom_label, custom_result_filtered, custom_pbo_filtered),
    ]
    custom_cell_results_filtered = {
        "Target's own best": target_own_result_filtered,
        custom_label: custom_result_filtered,
    }
    st.subheader(f"Restricted to {target_day_type}")
    _render_comparison(custom_rows_filtered, custom_cell_results_filtered,
                        target_always_filtered, target_static_filtered,
                        f"custom transplant — {target_day_type}")
