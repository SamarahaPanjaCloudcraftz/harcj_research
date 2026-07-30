"""
Bucket Explorer — interactive diagnostic.

Pick a date range, a bucket count, and a bucketing method; see the actual
PnL-by-predicted-vol-bucket relationship in the data. This is the evidence
to look at before assuming anything about which buckets are "bad" — the
Threshold Grid page's exclusion-rule toggle should be set based on what you
see here, not the other way around.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import available_profiles, load_joined, PROFILE_TO_UNDERLYING
from bucketing import bucket_stats

_BLUE = "#2a78d6"   # positive PnL — diverging pair, blue pole
_RED  = "#e34948"   # negative PnL — diverging pair, red pole
_GRID = "#e1e0d9"
_MUTED = "#898781"

st.set_page_config(page_title="Bucket Explorer — HARCJ Research", layout="wide")
st.title("Bucket Explorer")
st.caption(
    "See the PnL-by-predicted-vol-bucket relationship directly in the data "
    "before deciding which exclusion rule the Threshold Grid page should use."
)

# ── Sidebar controls ──────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Controls")

    profile = st.selectbox("Profile", available_profiles(), index=0)
    df = load_joined(profile)

    if df.empty:
        st.error("No usable joined data (pred_vol_pct + pnl) for this profile.")
        st.stop()

    min_date, max_date = df.index.min().date(), df.index.max().date()
    st.caption(f"Data available: {min_date} → {max_date} ({len(df)} days)")

    date_range = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    n_buckets = st.slider("Number of buckets", min_value=2, max_value=20, value=10)

    method = st.radio(
        "Bucketing method",
        options=["equal_count", "equal_width"],
        format_func=lambda m: (
            "Equal-count (quantile — each bucket has the same # of days)"
            if m == "equal_count"
            else "Equal-width (range — bins evenly spaced in vol%, uneven population)"
        ),
    )

    value_col = st.radio(
        "Bucket on",
        options=["pred_vol_pct", "pred_bpv"],
        format_func=lambda c: "Predicted BPVvol% (annualized)" if c == "pred_vol_pct" else "Raw predicted BPV",
        index=0,
    )

# ── Filter to date range ───────────────────────────────────────────────────────

mask = (df.index.date >= start_date) & (df.index.date <= end_date)
sub = df.loc[mask]

if sub.empty:
    st.warning("No data in the selected date range.")
    st.stop()

st.caption(f"Showing {len(sub)} days in range {start_date} → {end_date}")

# ── Compute bucket stats ───────────────────────────────────────────────────────

stats, edges = bucket_stats(sub, value_col=value_col, pnl_col="pnl",
                            n_buckets=n_buckets, method=method)

if stats.empty:
    st.warning("Not enough distinct values to form buckets in this range.")
    st.stop()

n_actual = len(stats)
if n_actual < n_buckets:
    st.info(f"Requested {n_buckets} buckets, but only {n_actual} distinct bucket(s) "
             "could be formed (not enough unique values / duplicate bin edges).")

# ── Table ──────────────────────────────────────────────────────────────────────

st.subheader("Per-bucket PnL")

display = stats.copy()
display["range"] = display.apply(
    lambda r: f"[{r['range_lo']:.2f}, {r['range_hi']:.2f})"
    if value_col == "pred_vol_pct" else f"[{r['range_lo']:.2e}, {r['range_hi']:.2e})",
    axis=1,
)
display["win_rate"] = (display["win_rate"] * 100).round(1)
display = display[["range", "n", "total_pnl", "mean_pnl", "win_rate"]]
display.columns = ["Vol% range" if value_col == "pred_vol_pct" else "BPV range",
                    "Days", "Total PnL", "Mean PnL", "Win rate %"]

st.dataframe(
    display.style.format({
        "Total PnL": "{:,.0f}",
        "Mean PnL":  "{:,.0f}",
        "Win rate %": "{:.1f}",
    }),
    use_container_width=True,
)

worst_bucket = stats["total_pnl"].idxmin()
best_bucket  = stats["total_pnl"].idxmax()
col1, col2 = st.columns(2)
col1.metric("Worst bucket (by total PnL)", f"Bucket {worst_bucket}",
            f"{stats.loc[worst_bucket, 'total_pnl']:,.0f}")
col2.metric("Best bucket (by total PnL)", f"Bucket {best_bucket}",
            f"{stats.loc[best_bucket, 'total_pnl']:,.0f}")

# ── Chart ──────────────────────────────────────────────────────────────────────

st.subheader("Total PnL by bucket")

colors = [_BLUE if v >= 0 else _RED for v in stats["total_pnl"]]

fig = go.Figure()
fig.add_trace(go.Bar(
    x=[str(b) for b in stats.index],
    y=stats["total_pnl"],
    marker_color=colors,
    hovertemplate=(
        "Bucket %{x}<br>"
        "Total PnL: %{y:,.0f}<br>"
        "Days: %{customdata[0]}<br>"
        "Win rate: %{customdata[1]:.1%}<extra></extra>"
    ),
    customdata=stats[["n", "win_rate"]].values,
))
fig.add_hline(y=0, line_color=_MUTED, line_width=1)
fig.update_layout(
    xaxis_title=f"Bucket (0 = lowest {'vol%' if value_col == 'pred_vol_pct' else 'BPV'})",
    yaxis_title="Total PnL",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(gridcolor=_GRID, type="category"),
    yaxis=dict(gridcolor=_GRID, zeroline=False),
    margin=dict(l=10, r=10, t=20, b=10),
    height=420,
    showlegend=False,
)
st.plotly_chart(fig, use_container_width=True)

# ── Mean PnL chart (secondary view) ────────────────────────────────────────────

st.subheader("Mean PnL per day, by bucket")
st.caption(
    "Total PnL rewards buckets with more days in them; mean PnL per day controls "
    "for that — useful when equal-width bucketing leaves some buckets almost empty."
)

colors_mean = [_BLUE if v >= 0 else _RED for v in stats["mean_pnl"]]
fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=[str(b) for b in stats.index],
    y=stats["mean_pnl"],
    marker_color=colors_mean,
    hovertemplate="Bucket %{x}<br>Mean PnL: %{y:,.0f}<extra></extra>",
))
fig2.add_hline(y=0, line_color=_MUTED, line_width=1)
fig2.update_layout(
    xaxis_title=f"Bucket (0 = lowest {'vol%' if value_col == 'pred_vol_pct' else 'BPV'})",
    yaxis_title="Mean PnL / day",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(gridcolor=_GRID, type="category"),
    yaxis=dict(gridcolor=_GRID, zeroline=False),
    margin=dict(l=10, r=10, t=20, b=10),
    height=380,
    showlegend=False,
)
st.plotly_chart(fig2, use_container_width=True)

with st.expander("Raw bucket stats"):
    st.dataframe(stats, use_container_width=True)
