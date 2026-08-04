"""
Shared chart-rendering functions for the research dashboard's Streamlit
pages — extracted from pages/2_threshold_grid.py so pages/3_summary.py can
reuse the identical rendering logic rather than duplicating it.

Nothing here computes anything — it only renders dicts/RunResults already
produced by dynamic_threshold.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_BLUE = "#2a78d6"
_RED  = "#e34948"
_MID  = "#f0efec"
_GRID = "#e1e0d9"
_MUTED = "#898781"
_INK  = "#0b0b0b"   # skipped-day marker — deliberately not red, so it never
                    # disappears against the red "excluded bucket" background

# Categorical palette, fixed order — for identifying which (L, B) combo a
# line/bar belongs to when comparing several at once.
_CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

METRIC_LABELS = {
    "total_pnl":      "Total PnL",
    "avg_daily_pnl":  "Avg daily PnL",
    "sortino":        "Sortino ratio",
    "sharpe":         "Sharpe ratio",
    "win_rate":       "Win rate",
    "drawdown":       "Max drawdown",
    "max_daily_loss": "Max daily loss",
}

# Metrics expressed in dollars — comma-formatted, same as Total PnL.
_DOLLAR_METRICS = {"total_pnl", "avg_daily_pnl", "drawdown", "max_daily_loss"}

_BAR_FREQ = {"Daily": None, "Weekly": "W-FRI", "Monthly": "ME"}


def render_heatmap(grid_result: dict, metric: str, title: str,
                    label: str | None = None, min_days_per_bucket: int = 10):
    label = label or METRIC_LABELS.get(metric, metric)
    values   = grid_result["metrics"][metric]
    reliable = grid_result["reliable"]
    n_traded = grid_result["n_traded"]

    z = values.values.astype(float)
    finite = z[np.isfinite(z)]
    if finite.size == 0:
        st.warning(f"No finite values for {label} in this grid.")
        return

    if metric == "win_rate":
        colorscale = [[0, _RED], [0.5, _MID], [1, _BLUE]]
        zmid = None
        zmin, zmax = 0.0, 1.0
    elif metric == "pbo":
        # low PBO = good (blue), high PBO = bad (red) — opposite direction to win_rate
        colorscale = [[0, _BLUE], [0.5, _MID], [1, _RED]]
        zmid = None
        zmin, zmax = 0.0, 1.0
    else:
        bound = float(np.nanmax(np.abs(finite))) or 1.0
        colorscale = [[0, _RED], [0.5, _MID], [1, _BLUE]]
        zmid = 0.0
        zmin, zmax = -bound, bound

    text = np.empty(z.shape, dtype=object)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            v = z[i, j]
            if not np.isfinite(v):
                text[i, j] = "—"
            elif metric in _DOLLAR_METRICS:
                text[i, j] = f"{v:,.0f}"
            elif metric == "win_rate":
                text[i, j] = f"{v:.1%}"
            else:
                text[i, j] = f"{v:.2f}"
            if not reliable.values[i, j]:
                text[i, j] += " *"

    hover_extra_label = "Combinations" if metric == "pbo" else "Days traded"
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=[str(c) for c in values.columns],
        y=[str(r) for r in values.index],
        colorscale=colorscale,
        zmid=zmid, zmin=zmin, zmax=zmax,
        text=text,
        texttemplate="%{text}",
        textfont={"size": 12},
        customdata=n_traded.values,
        hovertemplate=(
            "Lookback %{x} / Buckets %{y}<br>"
            + label + ": %{z}<br>"
            + hover_extra_label + ": %{customdata}<extra></extra>"
        ),
        colorbar=dict(title=label),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Lookback (trading days)",
        yaxis_title="Number of buckets",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10),
        height=380 + 20 * len(values.index),
        yaxis=dict(type="category"),
        xaxis=dict(type="category"),
    )
    st.plotly_chart(fig, use_container_width=True)
    if (~reliable).values.any():
        st.caption(
            f"`*` = fewer than {min_days_per_bucket} days/bucket for the worst-PnL "
            "dynamic rule at that (lookback, n_buckets) — ranking buckets by PnL "
            "there is likely noise, not signal."
        )


def _common_start(cell_results: dict) -> pd.Timestamp | None:
    """Latest of each selected cell's own scored-start date, so every line/bar
    covers exactly the same window — no cell gets credit for a longer history
    than the others being compared."""
    starts = [r.daily_pnl.index.min() for r in cell_results.values() if len(r.daily_pnl)]
    return max(starts) if starts else None


def render_multi_equity_curve(cell_results: dict, always: dict, static: dict, title: str,
                               static_label: str = "Static threshold"):
    """
    Cumulative PnL, one line per selected (L, B) combo, plus always-trade
    and today's static threshold as reference lines — all sliced to the
    same common start date and cumulated from 0 there, so lines starting
    at different lookback offsets are still an apples-to-apples comparison.
    """
    if not cell_results:
        st.warning("Select at least one combo to compare.")
        return
    common_start = _common_start(cell_results)
    if common_start is None:
        st.warning("No scored days for the selected combo(s).")
        return

    full_idx = always["_daily_pnl"].index
    common_idx = full_idx[full_idx >= common_start]

    fig = go.Figure()
    for i, (label, result) in enumerate(cell_results.items()):
        cum = result.daily_pnl.reindex(common_idx).fillna(0.0).cumsum()
        fig.add_trace(go.Scatter(x=common_idx, y=cum, mode="lines", name=label,
                                  line=dict(color=_CATEGORICAL[i % len(_CATEGORICAL)], width=2.5)))

    always_cum = always["_daily_pnl"].reindex(common_idx).fillna(0.0).cumsum()
    static_cum = static["_daily_pnl"].reindex(common_idx).fillna(0.0).cumsum()
    fig.add_trace(go.Scatter(x=common_idx, y=always_cum, mode="lines",
                              name="Always-trade", line=dict(color=_MUTED, width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=common_idx, y=static_cum, mode="lines",
                              name=static_label, line=dict(color=_INK, width=1.5, dash="dash")))
    fig.add_hline(y=0, line_color=_MUTED, line_width=1)
    fig.update_layout(
        title=dict(text=title, y=0.97, yanchor="top"),
        yaxis_title="Cumulative PnL", xaxis_title="Date",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor=_GRID), yaxis=dict(gridcolor=_GRID),
        margin=dict(l=10, r=10, t=90, b=10), height=460,
        legend=dict(orientation="h", y=1.15, yanchor="bottom"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"All lines start from 0 at {common_idx[0].date()} — the latest start date "
        "among the selected combos — so every line covers the same window."
    )


def render_pnl_bars(cell_results: dict, always: dict, static: dict, freq_label: str, title: str,
                     static_label: str = "Static threshold"):
    """Grouped bar chart, one color per selected combo plus the two
    reference baselines, aggregated to the given frequency (None = daily,
    no resampling)."""
    if not cell_results:
        return
    common_start = _common_start(cell_results)
    if common_start is None:
        return
    freq = _BAR_FREQ[freq_label]

    def _agg(series: pd.Series) -> pd.Series:
        s = series[series.index >= common_start]
        return s.resample(freq).sum() if freq is not None else s

    fig = go.Figure()
    for i, (label, result) in enumerate(cell_results.items()):
        s = _agg(result.daily_pnl)
        fig.add_trace(go.Bar(x=s.index, y=s.values, name=label,
                              marker_color=_CATEGORICAL[i % len(_CATEGORICAL)]))

    always_s = _agg(always["_daily_pnl"])
    static_s = _agg(static["_daily_pnl"])
    fig.add_trace(go.Bar(x=always_s.index, y=always_s.values, name="Always-trade",
                          marker_color=_MUTED))
    fig.add_trace(go.Bar(x=static_s.index, y=static_s.values, name=static_label,
                          marker_color=_INK))

    fig.add_hline(y=0, line_color=_MUTED, line_width=1)
    fig.update_layout(
        title=dict(text=f"{title} — {freq_label} PnL", y=0.96, yanchor="top"),
        barmode="group",
        yaxis_title="PnL", xaxis_title="Date",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor=_GRID), yaxis=dict(gridcolor=_GRID),
        margin=dict(l=10, r=10, t=90, b=10), height=360,
        legend=dict(orientation="h", y=1.18, yanchor="bottom"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_diff_bars(pnl_A: pd.Series, label_A: str, pnl_B: pd.Series, label_B: str,
                      freq_label: str, title: str):
    """
    Single bar series: (A's PnL − B's PnL) per period, colored by sign
    (blue = A ahead that period, red = B ahead). Aligned to the later of
    A's and B's own start dates, then resampled — resampling after
    differencing gives the same result as differencing after resampling,
    since sum is linear.
    """
    common_start = max(pnl_A.index.min(), pnl_B.index.min())
    full_idx = pnl_A.index.union(pnl_B.index)
    common_idx = full_idx[full_idx >= common_start]

    diff = (pnl_A.reindex(common_idx).fillna(0.0) - pnl_B.reindex(common_idx).fillna(0.0))
    freq = _BAR_FREQ[freq_label]
    if freq is not None:
        diff = diff.resample(freq).sum()

    colors = [_BLUE if v >= 0 else _RED for v in diff.values]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=diff.index, y=diff.values, marker_color=colors, showlegend=False))
    fig.add_hline(y=0, line_color=_MUTED, line_width=1)
    fig.update_layout(
        title=dict(text=f"{title} — {label_A} minus {label_B} ({freq_label})", y=0.96, yanchor="top"),
        yaxis_title=f"PnL(A) − PnL(B)", xaxis_title="Date",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor=_GRID), yaxis=dict(gridcolor=_GRID),
        margin=dict(l=10, r=10, t=70, b=10), height=340,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Blue = A (\"{label_A}\") ahead that period; red = B (\"{label_B}\") ahead.")


def _cutoff_timeline_figure(diag: pd.DataFrame, value_col: str, excluded_col: str,
                             trade_col: str, title: str, y_axis_title: str):
    """Shared figure-builder for one side's (single, or one half of a paired) cutoff timeline."""
    dates = diag.index
    colors = [_BLUE if t else _INK for t in diag[trade_col]]

    if len(dates) > 1:
        half = (dates.to_series().diff().dropna().median()) * 0.4
    else:
        half = pd.Timedelta(days=1) * 0.4

    shapes = []
    for date, excluded_ranges in zip(dates, diag[excluded_col]):
        for lo, hi in excluded_ranges:
            shapes.append(dict(
                type="rect", xref="x", yref="y",
                x0=date - half, x1=date + half, y0=lo, y1=hi,
                fillcolor=_RED, opacity=0.35, line=dict(width=0),
                layer="below",
            ))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=diag[value_col], mode="lines",
        line=dict(color=_MUTED, width=1), hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=diag[value_col], mode="markers",
        marker=dict(color=colors, size=6, line=dict(color="white", width=0.5)),
        customdata=diag[trade_col],
        hovertemplate="%{x|%Y-%m-%d}<br>value: %{y:.2f}<br>traded: %{customdata}<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(shapes=shapes)
    fig.update_layout(
        title=title,
        yaxis_title=y_axis_title,
        xaxis_title="Date",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor=_GRID), yaxis=dict(gridcolor=_GRID),
        margin=dict(l=10, r=10, t=40, b=10), height=420,
    )
    return fig


def _cutoff_stats(diag: pd.DataFrame, excluded_col: str, rule: str, heading: str):
    """Shared descriptive-stats block for one side's cutoff."""
    cutoff_series = diag[excluded_col].map(
        lambda ranges: min(lo for lo, hi in ranges) if ranges else np.nan
    ).dropna()

    st.markdown(f"**{heading}**")
    if rule == "highest_vol":
        st.caption(
            "highest_vol always excludes the top bucket(s), so this is a true "
            "one-sided cutoff: \"skip if value ≥ X\" — these stats describe how X "
            "moved day to day as the trailing window rolled."
        )
    else:
        st.caption(
            "⚠ worst_pnl_dynamic can exclude a middle or non-contiguous bucket, "
            "so this is only the lower edge of the lowest excluded bucket each "
            "day — not necessarily a true one-sided \"skip above this\" cutoff. "
            "Treat these stats as a rough proxy, not a strict threshold."
        )
    if cutoff_series.empty:
        st.info("No excluded days in this window — nothing to summarize.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean", f"{cutoff_series.mean():.2f}")
        c2.metric("Median", f"{cutoff_series.median():.2f}")
        c3.metric("Std dev", f"{cutoff_series.std():.2f}")
        c4.metric("Days w/ exclusion", f"{len(cutoff_series)}")


def render_cutoff_timeline(result, n_buckets_configured: int, title: str, rule: str = "worst_pnl_dynamic"):
    """
    One chart: pred_vol_pct per day (line + points), with that day's own
    excluded value-range(s) shaded directly behind it (translucent red
    rectangles, one per excluded range per day — handles non-contiguous
    exclusions for worst_pnl_dynamic honestly rather than assuming a single
    clean cutoff line). Points are colored by trade outcome using colors
    that don't disappear against the red shading (blue=traded, ink=skipped —
    deliberately not red-on-red).

    Below the chart: descriptive stats (mean/median/std) of the daily
    "cutoff" — the lower edge of the lowest-indexed excluded bucket each
    day. For highest_vol this IS the one-sided cutoff (exclusion is always
    contiguous from the top). For worst_pnl_dynamic it's only a proxy —
    the excluded region can be a middle bucket or non-contiguous, so this
    number doesn't necessarily mean "skip above this value."

    Paired-mode results (and_both/or_either — from walk_forward_run_dual)
    carry TWO independent diagnostics (harcj_* and iv_*, no plain
    value/excluded_ranges) — rendered as two separate timelines/stats
    blocks, one per side, since each side's cutoff is genuinely independent
    that day.
    """
    diag = result.diagnostics
    if diag is None or diag.empty:
        st.info("No diagnostics available for this cell.")
        return

    if "harcj_value" in diag.columns:
        st.caption(
            "Paired mode: BPV and IV each have their own independent daily "
            "cutoff — shown separately below."
        )
        fig_harcj = _cutoff_timeline_figure(
            diag, "harcj_value", "harcj_excluded_ranges", "harcj_trade",
            f"{title} — BPV daily value, with that day's excluded range shaded",
            "pred_vol_pct (or raw BPV)",
        )
        st.plotly_chart(fig_harcj, use_container_width=True)
        fig_iv = _cutoff_timeline_figure(
            diag, "iv_value", "iv_excluded_ranges", "iv_trade",
            f"{title} — IV daily value, with that day's excluded range shaded",
            "iv_dynamic",
        )
        st.plotly_chart(fig_iv, use_container_width=True)
        st.caption(
            "Red shading = the excluded value range(s) in force that day (from the "
            "trailing window, recomputed daily). Blue points = traded, dark points = "
            "skipped — a dark point should always fall inside its own day's red band."
        )
        _cutoff_stats(diag, "harcj_excluded_ranges", rule, "BPV cutoff descriptive stats")
        _cutoff_stats(diag, "iv_excluded_ranges", rule, "IV cutoff descriptive stats")
        return

    fig = _cutoff_timeline_figure(
        diag, "value", "excluded_ranges", "trade",
        f"{title} — daily value, with that day's excluded range shaded",
        "pred_vol_pct (or raw BPV)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Red shading = the excluded value range(s) in force that day (from the "
        "trailing window, recomputed daily). Blue points = traded, dark points = "
        "skipped — a dark point should always fall inside its own day's red band."
    )
    _cutoff_stats(diag, "excluded_ranges", rule, "Cutoff descriptive stats")


def render_reference(always: dict, static: dict, cfg: dict, metric: str, iv_label: str):
    c1, c2 = st.columns(2)
    with c1:
        st.metric(
            f"Always-trade baseline — {METRIC_LABELS[metric]}",
            f"{always[metric]:,.2f}" if metric != "win_rate" else f"{always[metric]:.1%}",
        )
        st.caption(f"{always['n_traded']}/{always['n_days']} days traded")
    with c2:
        st.metric(
            f"Today's static threshold ({iv_label}) — {METRIC_LABELS[metric]}",
            f"{static[metric]:,.2f}" if metric != "win_rate" else f"{static[metric]:.1%}",
        )
        st.caption(f"{static['n_traded']}/{static['n_days']} days traded — "
                   f"exclusion_ranges={cfg['exclusion_ranges']}, "
                   f"iv_cutoff={cfg['iv_cutoff']}")
