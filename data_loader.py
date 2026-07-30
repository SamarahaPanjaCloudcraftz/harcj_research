"""
Read-only loaders for the research dashboard.

Everything here reads from dashboard_new/dumps/ — the same on-disk staged
data the live app reads. Nothing here imports scheduler/, writes anywhere
under dashboard_new/, or touches the live scheduler/state.

The daily PnL series currently comes from dashboard_new/dumps/<underlying>_stage2/pnl/.
A dedicated backtest PnL series will replace this later — swap it in `load_daily_pnl()`
only; every caller of load_joined() is unaffected by that change.
"""

from __future__ import annotations

import glob
import json
import os

import pandas as pd

_HERE          = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT     = os.path.dirname(_HERE)
_DASHBOARD_DIR = os.path.join(_REPO_ROOT, "dashboard_new")
_DUMPS_DIR     = os.path.join(_DASHBOARD_DIR, "dumps")
_PARAMS_DIR    = os.path.join(_DASHBOARD_DIR, "params")

PROFILE_TO_UNDERLYING = {
    "spxw_persistence": "spxw",
    "ndaq_persistence": "ndaq",
}


def available_profiles() -> list[str]:
    return sorted(PROFILE_TO_UNDERLYING.keys())


def load_roll(profile: str) -> pd.DataFrame:
    """
    Read <profile>_stage3/roll.csv.

    Returns date-indexed DataFrame with at least: pred_bpv, pred_vol_pct,
    actual_bpv, actual_vol_pct, mean_iv.
    """
    path = os.path.join(_DUMPS_DIR, f"{profile}_stage3", "roll.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"roll.csv not found for profile {profile!r}: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    return df.set_index("date").sort_index()


def load_daily_pnl(underlying: str) -> pd.Series:
    """
    Read <underlying>_stage2/pnl/*.csv → one daily PnL value per date.

    Each file is named YYYYMMDD_pnl.csv and holds a single-column/day PnL
    value written by the scheduler. This is a placeholder series pending a
    dedicated backtest PnL feed — swap the body of this function when that
    lands; callers of load_joined() need no changes.
    """
    pnl_dir = os.path.join(_DUMPS_DIR, f"{underlying}_stage2", "pnl")
    files = sorted(glob.glob(os.path.join(pnl_dir, "????????_pnl.csv")))
    rows: dict[pd.Timestamp, float] = {}
    for path in files:
        date_str = os.path.basename(path)[:8]
        try:
            date = pd.Timestamp(date_str)
        except ValueError:
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        # Single daily value — take the last non-null row's pnl-like column.
        col = "pnl" if "pnl" in df.columns else df.columns[-1]
        series = df[col].dropna()
        if not series.empty:
            rows[date] = float(series.iloc[-1])
    return pd.Series(rows, name="pnl").sort_index()


def load_joined(profile: str) -> pd.DataFrame:
    """
    Join roll_df (pred_vol_pct, mean_iv, ...) with the daily PnL series on date.

    Returns date-indexed DataFrame with columns:
        pred_bpv, pred_vol_pct, actual_bpv, actual_vol_pct, mean_iv, pnl
    Rows with no pred_vol_pct or no pnl are dropped — this is the usable
    backtest window (bounded today by PnL coverage, currently the shorter
    of the two series).
    """
    underlying = PROFILE_TO_UNDERLYING[profile]
    roll = load_roll(profile)
    pnl  = load_daily_pnl(underlying)

    df = roll.copy()
    df["pnl"] = df.index.map(pnl)
    keep_cols = [c for c in ["pred_bpv", "pred_vol_pct", "actual_bpv",
                              "actual_vol_pct", "mean_iv", "pnl"] if c in df.columns]
    df = df[keep_cols].dropna(subset=["pred_vol_pct", "pnl"])
    return df


def load_joined_with_iv(
    profile: str,
    iv_source_mode: str,            # "point" | "mean"
    iv_timestamp: str | None = None,
    iv_start_time: str | None = None,
    iv_end_time: str | None = None,
) -> pd.DataFrame:
    """
    load_joined(profile) plus an "iv_dynamic" column built from the
    configurable IV source (see compute_iv_series — the trustworthy raw
    1-min source, not roll.csv's mean_iv). Rows with no iv_dynamic value
    are dropped too, on top of load_joined's existing pred_vol_pct/pnl
    requirement — needed by walk_forward_run_dual, which requires no gaps
    in either series over the scored range.
    """
    underlying = PROFILE_TO_UNDERLYING[profile]
    df = load_joined(profile)
    iv_series = compute_iv_series(
        underlying, mode=iv_source_mode, timestamp=iv_timestamp,
        start_time=iv_start_time, end_time=iv_end_time,
    )
    df = df.copy()
    df["iv_dynamic"] = df.index.map(iv_series)
    return df.dropna(subset=["iv_dynamic"])


def load_static_config(profile: str) -> dict:
    """
    Read the profile's current static thresholds, for use as a reference
    baseline against dynamic-grid results — read-only, never modified here.

    Returns: {"exclusion_ranges": [[lo, hi], ...], "iv_cutoff": float,
              "flag_combine_method": str}
    """
    path = os.path.join(_PARAMS_DIR, f"{profile}.json")
    with open(path) as f:
        params = json.load(f)
    return {
        "exclusion_ranges":    params.get("harcj_exclusion_ranges", []),
        "iv_cutoff":           params.get("iv_cutoff"),
        "flag_combine_method": params.get("flag_combine_method", "or"),
    }


def load_iv_1m(underlying: str) -> pd.DataFrame:
    """
    Read all raw 1-min ATM IV bars for `underlying` from
    dumps/<underlying>_stage1/atm_iv/*.csv.

    IMPORTANT: only the window the live scheduler was configured to fetch
    is available (07:30-08:30 CT for both SPXW and NDAQ currently, per the
    filename's HHMM-HHMM tag) — not the full trading day. A requested
    timestamp/window outside that range will simply have no data.

    Returns a timestamp-indexed DataFrame, column "atm_iv", spanning every
    dumped day.
    """
    iv_dir = os.path.join(_DUMPS_DIR, f"{underlying}_stage1", "atm_iv")
    files = sorted(glob.glob(os.path.join(iv_dir, "????????_*_atm_iv.csv")))
    chunks = [pd.read_csv(p, index_col=0, parse_dates=True) for p in files]
    if not chunks:
        return pd.DataFrame(columns=["atm_iv"])
    full = pd.concat(chunks).sort_index()
    return full[~full.index.duplicated(keep="first")]


def compute_iv_series(
    underlying: str,
    mode: str,                      # "point" | "mean"
    timestamp: str | None = None,   # "HH:MM" — used when mode="point"
    start_time: str | None = None,  # "HH:MM" — used when mode="mean"
    end_time: str | None = None,    # "HH:MM" — used when mode="mean"
) -> pd.Series:
    """
    Build a daily IV series from the raw 1-min dump, per the configured
    source variant:

        mode="point" — the last available 1-min bar at or before
                        `timestamp`, each day ("open IV" snapshot)
        mode="mean"  — mean of all bars in [start_time, end_time], each day
                        ("open IV mean")

    Only meaningful within the dumped window (see load_iv_1m) — days with
    no bars in range are simply absent from the returned series.
    """
    raw = load_iv_1m(underlying)
    if raw.empty:
        return pd.Series(dtype=float, name="iv")

    dates = raw.index.normalize()
    times = raw.index.time

    result: dict[pd.Timestamp, float] = {}

    if mode == "point":
        t = pd.Timestamp(timestamp).time()
        mask = times <= t
        sub = raw[mask].copy()
        sub["date"] = dates[mask]
        for date, grp in sub.groupby("date"):
            result[date] = float(grp["atm_iv"].iloc[-1])
    elif mode == "mean":
        t_start = pd.Timestamp(start_time).time()
        t_end   = pd.Timestamp(end_time).time()
        mask = (times >= t_start) & (times <= t_end)
        sub = raw[mask].copy()
        sub["date"] = dates[mask]
        for date, grp in sub.groupby("date"):
            result[date] = float(grp["atm_iv"].mean())
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Expected 'point' or 'mean'.")

    return pd.Series(result, name="iv").sort_index()


def data_coverage(profile: str) -> dict:
    """Quick summary of what's available for this profile — used by the sidebar."""
    underlying = PROFILE_TO_UNDERLYING[profile]
    roll = load_roll(profile)
    pnl  = load_daily_pnl(underlying)
    joined = load_joined(profile)
    return {
        "pred_vol_pct_range": (
            roll["pred_vol_pct"].dropna().index.min(),
            roll["pred_vol_pct"].dropna().index.max(),
        ) if "pred_vol_pct" in roll.columns and roll["pred_vol_pct"].notna().any() else (None, None),
        "pnl_range": (pnl.index.min(), pnl.index.max()) if not pnl.empty else (None, None),
        "joined_range": (joined.index.min(), joined.index.max()) if not joined.empty else (None, None),
        "joined_n_days": len(joined),
    }
