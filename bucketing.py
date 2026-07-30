"""
Bucket assignment methods, shared by the Bucket Explorer and the Threshold
Grid pages.

Two interchangeable methods:
    equal_count — pd.qcut: sort values, split so each bucket holds the same
                  number of data points. Bucket edges land wherever the data
                  density puts them. Matches the method calibrate_thresholds.py
                  uses today.
    equal_width — pd.cut: split (min, max) into n_buckets equal-width bins.
                  Bucket edges are evenly spaced in value space; population
                  per bucket can be very uneven.

Both return a bucket label per row (0 = lowest) plus the bin edges used, so
callers can report "bucket 4 = [12.3%, 15.1%)" rather than just an index.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

BucketMethod = Literal["equal_count", "equal_width"]


def assign_buckets(
    values: pd.Series,
    n_buckets: int,
    method: BucketMethod = "equal_count",
) -> tuple[pd.Series, np.ndarray]:
    """
    Assign each value in `values` to one of n_buckets buckets.

    Returns (bucket_labels, bin_edges). bucket_labels is an Int64 series
    aligned to values.index (NaN where the source value was NaN). bin_edges
    has len == n_actual_buckets + 1; n_actual_buckets may be < n_buckets if
    there weren't enough distinct values (duplicates='drop').
    """
    if values.empty:
        return pd.Series(dtype="Int64"), np.array([])

    if method == "equal_count":
        labels, edges = pd.qcut(values, q=n_buckets, retbins=True,
                                 labels=False, duplicates="drop")
    elif method == "equal_width":
        labels, edges = pd.cut(values, bins=n_buckets, retbins=True,
                                labels=False, include_lowest=True,
                                duplicates="drop")
    else:
        raise ValueError(f"Unknown bucket method: {method!r}")

    return labels.astype("Int64"), edges


def bucket_stats(
    df: pd.DataFrame,
    value_col: str,
    pnl_col: str,
    n_buckets: int,
    method: BucketMethod = "equal_count",
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Compute per-bucket PnL stats for df[value_col] vs df[pnl_col].

    Returns (stats_df, bin_edges). stats_df is indexed by bucket (0 = lowest
    value_col bucket) with columns: n, total_pnl, mean_pnl, win_rate,
    range_lo, range_hi.
    """
    work = df[[value_col, pnl_col]].dropna()
    labels, edges = assign_buckets(work[value_col], n_buckets, method)
    work = work.assign(bucket=labels)

    rows = []
    for bk in sorted(work["bucket"].dropna().unique()):
        grp = work[work["bucket"] == bk]
        bk = int(bk)
        rows.append({
            "bucket":    bk,
            "n":         len(grp),
            "total_pnl": grp[pnl_col].sum(),
            "mean_pnl":  grp[pnl_col].mean(),
            "win_rate":  (grp[pnl_col] > 0).mean(),
            "range_lo":  edges[bk],
            "range_hi":  edges[bk + 1],
        })
    stats = pd.DataFrame(rows).set_index("bucket").sort_index()
    return stats, edges


def worst_buckets(stats: pd.DataFrame, n_exclude: int) -> list[int]:
    """Return the n_exclude bucket indices with the lowest total_pnl."""
    return stats["total_pnl"].nsmallest(n_exclude).index.tolist()


def highest_value_buckets(stats: pd.DataFrame, n_exclude: int) -> list[int]:
    """Return the n_exclude highest-index buckets (i.e. highest value_col range)."""
    return sorted(stats.index)[-n_exclude:] if n_exclude > 0 else []
