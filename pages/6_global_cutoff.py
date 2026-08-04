"""
Global Cutoff Search — results viewer for docs/global_cutoff_framework.md.

Reads the disk-cached batch-run output (built via global_cutoff.run_full_search
— an offline job, not run from this page) — every view here is instant, no
recomputation, since global_cutoff.run_full_search's own load_or_compute
calls simply return what the batch already persisted.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import exchange_events as ee
import global_cutoff as gcut
from chart_helpers import render_multi_equity_curve, render_pnl_bars, _BAR_FREQ

st.set_page_config(page_title="Global Cutoff Search — HARCJ Research", layout="wide")
st.title("Global Cutoff Search")
st.caption(
    "Exhaustive search over every (day-type, origin, mode, L, B) pair, blended "
    "into full-year cutoffs, ranked by PBO against both the current live "
    "baseline and the no-cutoff baseline. See docs/global_cutoff_framework.md "
    "for the full methodology (Stage 0-2, K/C selection, measured timings)."
)

INDEX_LABELS = {"spxw": "SPX", "ndaq": "NDAQ"}

with st.sidebar:
    index_key = st.selectbox("Index", list(INDEX_LABELS.keys()), index=0, format_func=lambda k: INDEX_LABELS[k])


@st.cache_data(show_spinner="Loading cached results…")
def _load(index_key: str):
    return gcut.run_full_search(index_key, top_k=gcut.TOP_K, pool_c=gcut.POOL_C)


result = _load(index_key)

st.caption(
    f"Target: **{INDEX_LABELS[index_key]}** — {len(result['stage1']):,} blended cutoffs searched (Stage 1), "
    f"{len(result['finalists'])} finalists reached CSCV (top-{gcut.TOP_K} global + top-{gcut.POOL_C} per pool, "
    f"deduplicated)."
)

_DISPLAY_COLS = ["Variant", "event_origin", "event_mode", "event_L", "event_B",
                  "non_event_origin", "non_event_mode", "non_event_L", "non_event_B",
                  "avg_daily_pnl", "total_pnl", "sortino", "drawdown", "max_daily_loss",
                  "traded", "pbo", "reliable", "composite"]


def _augment_with_extra_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add total_pnl/drawdown/traded columns — pulled from the already-persisted
    finalist_results (cheap lookup, no recomputation, no cache-key change)."""
    d = df.copy()
    d["Variant"] = ""
    total_pnl, drawdown, traded = [], [], []
    for _, r in d.iterrows():
        key = tuple(r[c] if not c.endswith(("_L", "_B")) else int(r[c]) for c in gcut._KEY_COLS)
        fm = result["finalist_results"][key].metrics
        total_pnl.append(fm["total_pnl"])
        drawdown.append(fm["drawdown"])
        traded.append(f"{fm['n_traded']}/{fm['n_days']}")
    d["total_pnl"] = total_pnl
    d["drawdown"] = drawdown
    d["traded"] = traded
    return d


def _baseline_row(label: str, run_result) -> dict:
    m = run_result.metrics
    return {
        "Variant": label,
        "event_origin": "—", "event_mode": "—", "event_L": "—", "event_B": "—",
        "non_event_origin": "—", "non_event_mode": "—", "non_event_L": "—", "non_event_B": "—",
        "avg_daily_pnl": m["avg_daily_pnl"], "total_pnl": m["total_pnl"], "sortino": m["sortino"],
        "drawdown": m["drawdown"], "max_daily_loss": m["max_daily_loss"],
        "traded": f"{m['n_traded']}/{m['n_days']}",
        "pbo": np.nan, "reliable": "—", "composite": np.nan,
    }


def _fmt(df: pd.DataFrame) -> pd.DataFrame:
    d = df[_DISPLAY_COLS].copy()
    for c in ["avg_daily_pnl", "total_pnl", "drawdown", "max_daily_loss"]:
        d[c] = d[c].map(lambda v: f"{v:,.0f}")
    d["sortino"] = d["sortino"].map(lambda v: f"{v:.2f}")
    d["pbo"] = d["pbo"].map(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
    d["composite"] = d["composite"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
    # baseline rows use "—" (a string) for L/B/reliable while variant rows use
    # int/bool — mixed dtype in one column breaks pyarrow's Arrow serialization,
    # so stringify uniformly.
    for c in ["event_L", "event_B", "non_event_L", "non_event_B"]:
        d[c] = d[c].map(lambda v: v if isinstance(v, str) else str(int(v)))
    d["reliable"] = d["reliable"].map(lambda v: v if isinstance(v, str) else str(bool(v)))
    return d


st.subheader("Reference baselines")
_baselines_df = pd.DataFrame([
    _baseline_row("Current live baseline", result["live_baseline"]),
    _baseline_row("No-cutoff baseline", result["no_cutoff_baseline"]),
])
st.dataframe(_fmt(_baselines_df).drop(columns=[c for c in _DISPLAY_COLS if c.startswith(("event_", "non_event_")) or c in ("pbo", "reliable", "composite")]),
             use_container_width=True, hide_index=True)

st.divider()


def _render_ranking_section(title: str, ranking_key: str):
    with st.expander(title, expanded=True):
        full = _augment_with_extra_metrics(result[ranking_key])
        st.dataframe(_fmt(full.head(10)), use_container_width=True)
        with st.expander(f"Full table ({len(full)} finalists)", expanded=False):
            st.dataframe(_fmt(full), use_container_width=True, height=500)


_render_ranking_section("Ranking vs. current live baseline", "ranking_vs_live")
_render_ranking_section("Ranking vs. no-cutoff baseline", "ranking_vs_no_cutoff")

def _row_label(i: int, row) -> str:
    return (f"#{i+1}: event={row['event_origin']}/{row['event_mode']}/L{row['event_L']}B{row['event_B']}  |  "
            f"non-event={row['non_event_origin']}/{row['non_event_mode']}/L{row['non_event_L']}B{row['non_event_B']}  "
            f"(pbo={row['pbo']:.3f})")


def _candidate_result(ranking_table, i):
    row = ranking_table.iloc[i]
    key = tuple(row[c] if not c.endswith(("_L", "_B")) else int(row[c]) for c in gcut._KEY_COLS)
    return result["finalist_results"][key]


st.divider()
st.header("Compare candidates — equity curves")
st.caption("Select one or more candidates to overlay on the same equity curve / PnL bars.")

chart_ranking_choice = st.radio("Rank by", ["vs. current live baseline", "vs. no-cutoff baseline"],
                                 horizontal=True, key="chart_ranking_choice")
chart_ranking_table = result["ranking_vs_live"] if chart_ranking_choice == "vs. current live baseline" else result["ranking_vs_no_cutoff"]

chart_selected = st.multiselect(
    "Candidates", list(range(len(chart_ranking_table))),
    format_func=lambda i: _row_label(i, chart_ranking_table.iloc[i]),
    default=[0] if len(chart_ranking_table) else [],
    key=f"candidates_multi_{index_key}_{chart_ranking_choice}",
)

if chart_selected:
    cell_results = {f"#{i+1}": _candidate_result(chart_ranking_table, i) for i in chart_selected}
    render_multi_equity_curve(cell_results, result["no_cutoff_baseline"].metrics, result["live_baseline"].metrics,
                               "Equity curve — selected candidates", static_label="Current live baseline")
    freqs_selected = st.multiselect("PnL bar frequency to show", list(_BAR_FREQ.keys()), default=["Weekly"],
                                     key="chart_freqs")
    for freq_label in freqs_selected:
        render_pnl_bars(cell_results, result["no_cutoff_baseline"].metrics, result["live_baseline"].metrics,
                         freq_label, "selected candidates", static_label="Current live baseline")
else:
    st.info("Select at least one candidate to see charts.")

st.divider()
st.header("Inspect a variant — confusion matrix")
st.caption(
    "The cutoff as a binary classifier: predicted class = trade/cutoff, actual "
    "class = whether that day's no-cutoff-baseline PnL was positive or negative. "
    "Traded days use the variant's own PnL (identical to the no-cutoff baseline's "
    "raw PnL on those dates, by construction); cutoff days use the no-cutoff "
    "baseline's raw PnL to show what they would have earned had they traded."
)

inspect_ranking_choice = st.radio("Rank by", ["vs. current live baseline", "vs. no-cutoff baseline"],
                                   horizontal=True, key="inspect_ranking_choice")
inspect_ranking_table = result["ranking_vs_live"] if inspect_ranking_choice == "vs. current live baseline" else result["ranking_vs_no_cutoff"]

inspect_idx = st.selectbox(
    "Candidate", list(range(len(inspect_ranking_table))),
    format_func=lambda i: _row_label(i, inspect_ranking_table.iloc[i]),
    key=f"inspect_candidate_{index_key}_{inspect_ranking_choice}",
)
selected_result = _candidate_result(inspect_ranking_table, inspect_idx)

no_cutoff_pnl = result["no_cutoff_baseline"].daily_pnl.reindex(selected_result.daily_pnl.index)
traded_mask = selected_result.traded_mask
event_dates = ee.load_event_dates()
is_event = pd.Series(selected_result.daily_pnl.index.isin(event_dates), index=selected_result.daily_pnl.index)


def _breakdown(mask: pd.Series) -> dict:
    pnl = no_cutoff_pnl[mask]
    pos, neg = pnl[pnl > 0], pnl[pnl < 0]
    return {
        "n": int(mask.sum()), "n_pos": len(pos), "n_neg": len(neg),
        "avg_pos": float(pos.mean()) if len(pos) else float("nan"),
        "avg_neg": float(neg.mean()) if len(neg) else float("nan"),
        "total_pos": float(pos.sum()) if len(pos) else 0.0,
        "total_neg": float(neg.sum()) if len(neg) else 0.0,
    }


all_mask = pd.Series(True, index=selected_result.daily_pnl.index)
whole_dataset_stats = _breakdown(all_mask)
whole_non_event_stats = _breakdown(~is_event)
whole_event_stats = _breakdown(is_event)

traded_stats = _breakdown(traded_mask)
cutoff_stats = _breakdown(~traded_mask)
cutoff_non_event_stats = _breakdown((~traded_mask) & (~is_event))
cutoff_event_stats = _breakdown((~traded_mask) & is_event)


def _fnum(v):
    return f"{v:,.0f}" if pd.notna(v) else "—"


def _render_block(stats: dict, total_label: str = "Total"):
    k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
    k1.metric(total_label, stats["n"])
    k2.metric("Positive days", stats["n_pos"])
    k3.metric("Negative days", stats["n_neg"])
    k4.metric("Avg positive PnL", _fnum(stats["avg_pos"]))
    k5.metric("Avg negative PnL", _fnum(stats["avg_neg"]))
    k6.metric("Total positive PnL", _fnum(stats["total_pos"]))
    k7.metric("Total negative PnL", _fnum(stats["total_neg"]))


st.subheader("0) No-cutoff baseline — whole dataset (ground truth)")
_render_block(whole_dataset_stats, "Total days")

st.subheader("1) Traded days (predicted: trade)")
t1, t2, t3, t4, t5 = st.columns(5)
t1.metric("Total traded", traded_stats["n"])
t2.metric("Positive days", traded_stats["n_pos"])
t3.metric("Negative days", traded_stats["n_neg"])
t4.metric("Avg positive PnL", _fnum(traded_stats["avg_pos"]))
t5.metric("Avg negative PnL", _fnum(traded_stats["avg_neg"]))

st.subheader("2) Cutoff days (predicted: no-trade) — overall")
_render_block(cutoff_stats, "Total cutoff")

st.subheader("3) Cutoff days — by day-type")
st.markdown("**Non-event days**")
st.caption("Whole dataset — all non-event days, regardless of traded/cutoff:")
_render_block(whole_non_event_stats, "Total non-event days")
st.caption("Cutoff only — non-event days that were skipped:")
_render_block(cutoff_non_event_stats, "Total cutoff")
st.markdown("**Event days**")
st.caption("Whole dataset — all event days, regardless of traded/cutoff:")
_render_block(whole_event_stats, "Total event days")
st.caption("Cutoff only — event days that were skipped:")
_render_block(cutoff_event_stats, "Total cutoff")
