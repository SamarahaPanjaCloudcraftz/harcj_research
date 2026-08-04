"""
Exchange Events — does switching to the 08:30-12:00 window specifically on
economic-release days (CPI/PPI/NFP/PCE/JOLTS/FOMC) beat each underlying's
own default-window cutoff?

Mechanism (see conversation history): NDAQ trades from 06:00 by default, SPX
from 05:00 by default. On event days, the live system switches to 08:30-12:00
instead. This page: for each underlying, finds the best (mode, L, B) cutoff
for the 08:30-12:00 variant SCORED ONLY on event days, and splices that
variant's outcome into the default variant's own best-cutoff PnL curve on
exactly those dates — a blended final curve, one per underlying.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cutoff_transfer as ct
import exchange_events as ee
from chart_helpers import render_multi_equity_curve, render_pnl_bars, _BAR_FREQ

st.set_page_config(page_title="Exchange Events — HARCJ Research", layout="wide")
st.title("Exchange Events")
st.caption(
    "Each underlying's default-window variant, except on economic-release days "
    "(CPI/PPI/NFP/PCE/JOLTS/FOMC) where the 08:30–12:00 variant's outcome is "
    "spliced in instead — gated by whichever cutoff is best on THAT variant, "
    "scored only over event days."
)

event_dates = ee.load_event_dates()
st.caption(f"{len(event_dates)} economic-release dates loaded from `exchange_events/CME_events_2026.csv` "
           "(CPI, PPI, Total Nonfarm Payrolls, PCE Price Index, Job Openings/JOLTS, FOMC — "
           "holiday/expiry/DST rows excluded).")


def _metrics_row(label: str, result, pbo: dict | None = None, n_event: int | None = None) -> dict:
    m = result.metrics
    row = {
        "Variant":        label,
        "Avg daily PnL":  m["avg_daily_pnl"],
        "Total PnL":      m["total_pnl"],
        "Sortino":        m["sortino"],
        "Max drawdown":   m["drawdown"],
        "Max daily loss": m["max_daily_loss"],
        "Traded":         f"{m['n_traded']}/{m['n_days']}",
    }
    if n_event is not None:
        row["Event days spliced in"] = n_event
    return row


def _render_comparison(rows: list[dict], cell_results: dict, always: dict, static: dict, title_suffix: str):
    df = pd.DataFrame(rows).set_index("Variant")
    display = df.copy()
    for c in ["Avg daily PnL", "Total PnL", "Max drawdown", "Max daily loss"]:
        display[c] = display[c].map(lambda v: f"{v:,.0f}")
    display["Sortino"] = display["Sortino"].map(lambda v: f"{v:.2f}")
    st.dataframe(display, use_container_width=True)

    render_multi_equity_curve(cell_results, always, static, f"Equity curve — {title_suffix}")
    freqs_selected = st.multiselect(
        "PnL bar frequency to show", list(_BAR_FREQ.keys()), default=["Weekly"],
        key=f"freqs_{title_suffix}",
    )
    for freq_label in freqs_selected:
        render_pnl_bars(cell_results, always, static, freq_label, title_suffix)


st.divider()
st.header("Auto — best cutoff for each underlying, blended with its best event-day cutoff")

for underlying, pair in ee.SOURCE_PAIRS.items():
    default_source = pair["default"]
    event_source = pair["event"]
    label = "SPX" if underlying == "spxw" else "NDAQ"

    st.subheader(f"{label}: {default_source} (default) + {event_source} (event days)")

    with st.status(f"Computing {label}…", expanded=False) as status:
        _, always, static = ct.canonical_mode_and_baselines(default_source, default_source)
        default_best = ct.best_cutoff(default_source, default_source)
        default_result = ct.cell_result_for(default_source, default_source,
                                             default_best["mode"], default_best["lookback"], default_best["n_buckets"])

        event_best = ee.best_event_cutoff(event_source, event_source, event_dates)
        event_result_full = ct.cell_result_for(event_source, event_source,
                                                event_best["mode"], event_best["lookback"], event_best["n_buckets"])

        blended, n_overlap = ee.blended_result(
            default_source, default_source, default_best,
            event_source, event_source, event_best,
            event_dates,
        )
        live_baseline = ee.live_baseline_result(underlying, event_dates)
        status.update(label=f"{label} computed", state="complete")

    st.caption(
        f"Default's own best: **{ct.MODE_LABELS[default_best['mode']]}**, L={default_best['lookback']}, B={default_best['n_buckets']}. "
        f"Event-day best (for {event_source}, scored only on event days): "
        f"**{ct.MODE_LABELS[event_best['mode']]}**, L={event_best['lookback']}, B={event_best['n_buckets']}."
    )
    st.caption(
        "**Current live baseline** — fixed rule, not walk-forward-optimized: BPV-only exclusion "
        "(yesterday's BPV from the event variant's own series) on non-event days; BPV AND mean-IV "
        "(07:30–08:30) on event days — using this underlying's real deployed exclusion_ranges/iv_cutoff "
        "(`dashboard_new/params/`), same PnL blend as above."
    )

    rows = [
        _metrics_row("Always-trade (default)", ct.dt.RunResult(
            metrics=always, reliable=True, avg_days_per_bucket=float("nan"),
            daily_pnl=always["_daily_pnl"], traded_mask=always["_traded_mask"],
        )),
        _metrics_row("Current live baseline", live_baseline),
        _metrics_row(f"Default only, own best ({ct.MODE_LABELS[default_best['mode']]}, "
                     f"L={default_best['lookback']}, B={default_best['n_buckets']})", default_result),
        _metrics_row(f"Event variant only, event-day best ({ct.MODE_LABELS[event_best['mode']]}, "
                     f"L={event_best['lookback']}, B={event_best['n_buckets']})", event_result_full),
        _metrics_row("Blended (default + event-day splice)", blended, n_event=n_overlap),
    ]
    cell_results = {
        "Current live baseline": live_baseline,
        "Default only": default_result,
        "Event variant only": event_result_full,
        "Blended": blended,
    }
    _render_comparison(rows, cell_results, always, static, f"{label} blended")
    st.divider()

st.header("Custom")
st.caption(
    "Pick the underlying, and (mode, L, B) for both the default variant and the "
    "event variant independently — instead of each one's auto-best."
)

c0, c1 = st.columns(2)
with c0:
    custom_underlying = st.selectbox(
        "Underlying", list(ee.SOURCE_PAIRS.keys()),
        format_func=lambda u: "SPX" if u == "spxw" else "NDAQ", key="custom_underlying",
    )
custom_pair = ee.SOURCE_PAIRS[custom_underlying]

st.markdown(f"**Default variant** — `{custom_pair['default']}`")
d1, d2, d3 = st.columns(3)
with d1:
    default_mode = st.selectbox("Mode", list(ct.MODE_LABELS.keys()), format_func=lambda m: ct.MODE_LABELS[m], key="default_mode")
with d2:
    default_L = st.number_input("Lookback (L)", min_value=2, max_value=500, value=20, step=1, key="default_L")
with d3:
    default_B = st.number_input("Buckets (B)", min_value=2, max_value=20, value=4, step=1, key="default_B")

st.markdown(f"**Event variant** — `{custom_pair['event']}`")
e1, e2, e3 = st.columns(3)
with e1:
    event_mode = st.selectbox("Mode", list(ct.MODE_LABELS.keys()), format_func=lambda m: ct.MODE_LABELS[m], key="event_mode")
with e2:
    event_L = st.number_input("Lookback (L)", min_value=2, max_value=500, value=20, step=1, key="event_L")
with e3:
    event_B = st.number_input("Buckets (B)", min_value=2, max_value=20, value=4, step=1, key="event_B")

if st.button("Compute custom blend"):
    with st.spinner("Computing…"):
        default_source = custom_pair["default"]
        event_source = custom_pair["event"]
        _, c_always, c_static = ct.canonical_mode_and_baselines(default_source, default_source)

        custom_default_result = ct.cell_result_for(default_source, default_source, default_mode, int(default_L), int(default_B))
        custom_event_result = ct.cell_result_for(event_source, event_source, event_mode, int(event_L), int(event_B))
        custom_blended, custom_n_overlap = ee.blended_result(
            default_source, default_source, {"mode": default_mode, "lookback": int(default_L), "n_buckets": int(default_B)},
            event_source, event_source, {"mode": event_mode, "lookback": int(event_L), "n_buckets": int(event_B)},
            event_dates,
        )

    custom_rows = [
        _metrics_row("Always-trade (default)", ct.dt.RunResult(
            metrics=c_always, reliable=True, avg_days_per_bucket=float("nan"),
            daily_pnl=c_always["_daily_pnl"], traded_mask=c_always["_traded_mask"],
        )),
        _metrics_row(f"Default only ({ct.MODE_LABELS[default_mode]}, L={int(default_L)}, B={int(default_B)})", custom_default_result),
        _metrics_row(f"Event variant only ({ct.MODE_LABELS[event_mode]}, L={int(event_L)}, B={int(event_B)})", custom_event_result),
        _metrics_row("Blended (custom)", custom_blended, n_event=custom_n_overlap),
    ]
    custom_cell_results = {
        "Default only": custom_default_result,
        "Event variant only": custom_event_result,
        "Blended (custom)": custom_blended,
    }
    _render_comparison(custom_rows, custom_cell_results, c_always, c_static, "custom blend")
