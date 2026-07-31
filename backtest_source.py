"""
Loader for the spxw_gamma_hedge_false backtest — a second, additional data
source alongside live_dumps (data_loader.py). Read-only against
backtests/spxw_gamma_hedge_false/; never touches dashboard_new/ or the
live scheduler.

Folder layout: 2026/<DOW>/{backtest,blotter,...}/*.csv — one full backtest
run per weekday-expiry variant (MON..FRI), each file set covering the whole
2026-01-02..2026-07-28 date range. Only the dates whose actual calendar
weekday matches the DOW folder name carry real trading activity (that
folder's 0DTE options only expire on that weekday) — every other date in
that folder is a dormant placeholder (flat price/IV walk, header-only
blotter). list_active_days() combines the DOW-matched slice from each of
the 5 folders, giving one continuous daily series of (date, backtest PnL
path, blotter path) across the whole backtest — used for PnL/session-window
derivation, both inherently specific to this backtest's own output.

spot and atm_iv are NOT read from that raw per-DOW-folder output directly.
Both differ across the 5 folders for the same date — spot is actually
identical (pure market replay), but atm_iv genuinely isn't (each folder
prices a different expiry chain; measured mean relative spread ~38-47%
across dates — see conversation history), so there's no single "right"
in-folder answer in general. Instead, both are pre-extracted (once, via
script/extract_spot.py + extract_spot_2026.py + extract_atm_iv.py, all
using the same weekday-matching rule) into flat per-date CSV stores —
research/spot/SPWX/ and research/atm_iv/SPWX/ — and everything here reads
from those instead. That's also what decouples this module from any one
backtest's folder layout: a future backtest source with a different folder
structure or time range just needs its own extraction script targeting the
same two directories; compute_daily_metrics/compute_iv_series/
extended_bpv_history need no changes.

research/spot/SPWX/ additionally holds 2025 warm-up dates (from
SPXW_baseline, a separate, earlier backtest) alongside this backtest's own
2026 dates — one continuous store, no filename collision — so
extended_bpv_history() draws on both with the same formula, no seam
between "warm-up" and "backtest" data.

Unlike live_dumps (fixed 08:30-15:00 CT session, IV dumped only for a
narrow 07:30-08:30 window), this backtest's raw bars span the full
00:00-23:59 day and its actual trading session start/stop varies by
variant/day (see manifest.json). Per-day BPV and the "open IV mean" window
both need a session start/end time — derive_session_window() finds it by
majority vote across every day's blotter, rather than assuming a fixed
clock time.
"""

from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_BT_DIR = os.path.join(_REPO_ROOT, "backtests", "spxw_gamma_hedge_false")
_LIVE_PARAMS_DIR = os.path.join(_REPO_ROOT, "dashboard_new", "params")

DATA_SOURCE_NAME = "spxw_gamma_hedge_false"
PROFILE_NAME = "spxw_gamma_hedge_false"
PROFILE_TO_UNDERLYING = {PROFILE_NAME: "spxw"}
_LIVE_PROFILE_FOR_UNDERLYING = {"spxw": "spxw_persistence", "ndaq": "ndaq_persistence"}

_DOW_TO_WEEKDAY = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4}
_DOW_FOLDERS = list(_DOW_TO_WEEKDAY.keys())

# Default open-IV window: the hour immediately before the derived session
# start (matches the live dashboard's "open IV" convention of looking at
# the run-up into the trading window).
DEFAULT_IV_WINDOW_HOURS_BEFORE_START = 1.0

# Generic per-date market-data stores (see module docstring) — spot spans
# 2025 warm-up + this backtest's own 2026 days; atm_iv currently covers only
# 2026 (no clean pre-2026 IV source yet — see conversation history).
_SPOT_DIR = os.path.join(_HERE, "spot", "SPWX")
_ATM_IV_DIR = os.path.join(_HERE, "atm_iv", "SPWX")


def available_profiles() -> list[str]:
    return [PROFILE_NAME]


def _list_dow_files(dow: str, subdir: str) -> list[str]:
    d = os.path.join(_BT_DIR, "2026", dow, subdir)
    return sorted(
        f for f in glob.glob(os.path.join(d, "*.csv"))
        if not os.path.basename(f).startswith("._")
    )


def _date_from_backtest_filename(path: str) -> pd.Timestamp:
    return pd.Timestamp(os.path.splitext(os.path.basename(path))[0])


def _date_from_blotter_filename(path: str) -> pd.Timestamp | None:
    m = re.match(r"blotter_(\d{8})", os.path.basename(path))
    return pd.Timestamp(m.group(1)) if m else None


def list_active_days() -> list[tuple[pd.Timestamp, str, str | None]]:
    """
    (date, backtest_csv_path, blotter_csv_path) for every date whose
    calendar weekday matches its DOW folder — i.e. every day this backtest
    actually had a live expiry/trading session, across the whole run.
    """
    out = []
    for dow in _DOW_FOLDERS:
        wd = _DOW_TO_WEEKDAY[dow]
        bt_files = {_date_from_backtest_filename(p): p for p in _list_dow_files(dow, "backtest")}
        bl_files = {
            d: p for p in _list_dow_files(dow, "blotter")
            if (d := _date_from_blotter_filename(p)) is not None
        }
        for date, bt_path in bt_files.items():
            if date.weekday() != wd:
                continue
            out.append((date, bt_path, bl_files.get(date)))
    return sorted(out, key=lambda t: t[0])


def derive_session_window() -> tuple[str, str]:
    """
    Scan every active day's blotter for its first/last trade time-of-day,
    and return the (start, end) "HH:MM" pair that occurs most often across
    the whole backtest — days with zero trades (header-only blotter) are
    skipped since they carry no session-time information.
    """
    starts, ends = [], []
    for _date, _bt_path, blotter_path in list_active_days():
        if blotter_path is None:
            continue
        df = pd.read_csv(blotter_path)
        if df.empty:
            continue
        ts = pd.to_datetime(df["timestamp"])
        starts.append(ts.min().strftime("%H:%M"))
        ends.append(ts.max().strftime("%H:%M"))
    if not starts:
        raise RuntimeError("No trades found in any blotter file — cannot derive session window.")
    start = pd.Series(starts).value_counts().idxmax()
    end = pd.Series(ends).value_counts().idxmax()
    return start, end


def bpv_rv_jump(prices: np.ndarray) -> tuple[float, float, float]:
    """Same formula as dashboard_new/signal_layer/harcj_engine.py::bpv_rv_jump."""
    lr = np.diff(np.log(prices))
    rv = float(np.sum(lr ** 2))
    bpv = float((np.pi / 2) * np.sum(np.abs(lr[:-1]) * np.abs(lr[1:])))
    jump = max(rv - bpv, 0.0)
    return bpv, rv, jump


def bpv_to_vol_pct(bpv: float) -> float:
    return float(np.sign(bpv) * np.sqrt(np.abs(bpv) * 252) * 100)


def _load_day_bars(backtest_path: str) -> pd.DataFrame:
    """
    Each minute appears twice (trade_done False/True pre/post-trade
    snapshot rows) — keep the last (post-trade) row per timestamp. Only
    portfolio_value (PnL) is read from this; spot/atm_iv come from the
    per-date stores instead (see module docstring).
    """
    df = pd.read_csv(backtest_path, parse_dates=["timestamp"])
    df = df.drop_duplicates(subset="timestamp", keep="last").set_index("timestamp").sort_index()
    return df


def _list_daily_files(directory: str) -> dict[pd.Timestamp, str]:
    """date -> path, for a research/<kind>/SPWX/<YYYYMMDD>.csv store."""
    out = {}
    for f in glob.glob(os.path.join(directory, "*.csv")):
        date_str = os.path.splitext(os.path.basename(f))[0]
        try:
            out[pd.Timestamp(date_str)] = f
        except ValueError:
            continue
    return out


def _load_daily_series(directory: str, date: pd.Timestamp, col: str) -> pd.Series | None:
    """timestamp-indexed series for one date from a research/<kind>/SPWX/ store, or None if missing."""
    path = os.path.join(directory, f"{date.strftime('%Y%m%d')}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    return df[col]


def compute_daily_metrics(
    session_start: str,
    session_end: str,
    iv_window_hours_before_start: float = DEFAULT_IV_WINDOW_HOURS_BEFORE_START,
) -> pd.DataFrame:
    """
    For every active day: BPV/jump/vol_pct over [session_start, session_end]
    (from research/spot/SPWX/), mean open IV over the hour before
    session_start (from research/atm_iv/SPWX/ — NaN if that date isn't
    there), and daily PnL (from the backtest's own portfolio_value, still
    read directly — PnL is strategy-specific, no generic store for it).
    Each day's backtest resets portfolio_value to 0 at 00:00, so the last
    bar's portfolio_value *is* that day's PnL.
    """
    start_t = pd.Timestamp(session_start).time()
    end_t = pd.Timestamp(session_end).time()
    iv_start_t = (
        pd.Timestamp(session_start) - pd.Timedelta(hours=iv_window_hours_before_start)
    ).time()

    rows: dict[pd.Timestamp, dict] = {}
    for date, bt_path, _blotter_path in list_active_days():
        spot = _load_daily_series(_SPOT_DIR, date, "spot")
        if spot is None:
            continue

        s = pd.Timestamp.combine(date.date(), start_t)
        e = pd.Timestamp.combine(date.date(), end_t)
        iv_s = pd.Timestamp.combine(date.date(), iv_start_t)

        session = spot.loc[s:e]
        if len(session) < 3:
            continue
        bpv, _rv, jump = bpv_rv_jump(session.values)
        vol_pct = bpv_to_vol_pct(bpv)

        iv_series = _load_daily_series(_ATM_IV_DIR, date, "atm_iv")
        if iv_series is not None:
            iv_window = iv_series.loc[iv_s:s]
            mean_iv = float(iv_window.mean()) if not iv_window.empty else np.nan
        else:
            mean_iv = np.nan

        bars = _load_day_bars(bt_path)
        if bars.empty:
            continue
        pnl = float(bars["portfolio_value"].iloc[-1])

        rows[date] = {
            "actual_bpv": bpv,
            "actual_vol_pct": vol_pct,
            "jump": jump,
            "mean_iv": mean_iv,
            "pnl": pnl,
        }

    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    df.index.name = "date"
    return df


def extended_bpv_history() -> pd.Series:
    """
    Daily BPV over the backtest's derived session window, for every date in
    research/spot/SPWX/ — both the 2025 warm-up (SPXW_baseline) and this
    backtest's own 2026 days live in the same per-date store, so this is
    one loop, one formula, no seam between the two periods. Spans well
    before the backtest's first scored day, for walk-forward grids to draw
    trailing lookback history from without truncating the backtest's own
    scored/PnL range.
    """
    session_start, session_end = derive_session_window()
    start_t = pd.Timestamp(session_start).time()
    end_t = pd.Timestamp(session_end).time()

    rows: dict[pd.Timestamp, float] = {}
    for date in sorted(_list_daily_files(_SPOT_DIR)):
        spot = _load_daily_series(_SPOT_DIR, date, "spot")
        s = pd.Timestamp.combine(date.date(), start_t)
        e = pd.Timestamp.combine(date.date(), end_t)
        session = spot.loc[s:e]
        if len(session) < 3:
            continue
        bpv, _rv, _jump = bpv_rv_jump(session.values)
        rows[date] = bpv

    return pd.Series(rows, name="actual_bpv").sort_index()


def load_joined(profile: str = PROFILE_NAME) -> pd.DataFrame:
    """
    Same shape/contract as data_loader.load_joined(): date-indexed,
    columns pred_bpv, pred_vol_pct, actual_bpv, actual_vol_pct, mean_iv,
    pnl. pred_bpv[t] = actual_bpv[t-1] — the same naive persistence model
    as the live system (dashboard_new/signal_layer/harcj_engine.py).
    """
    session_start, session_end = derive_session_window()
    df = compute_daily_metrics(session_start, session_end)
    df["pred_bpv"] = df["actual_bpv"].shift(1)
    df["pred_vol_pct"] = df["pred_bpv"].apply(lambda b: bpv_to_vol_pct(b) if pd.notna(b) else np.nan)
    df = df.dropna(subset=["pred_vol_pct", "pnl"])
    return df[["pred_bpv", "pred_vol_pct", "actual_bpv", "actual_vol_pct", "mean_iv", "pnl"]]


def extended_pred_vol_pct_history(profile: str = PROFILE_NAME) -> pd.Series:
    """
    pred_vol_pct for every date strictly before the backtest's own first
    scored day (load_joined()'s index) — derived by shifting
    extended_bpv_history() by one row over the full 2025+2026 series, same
    naive-persistence mechanism as load_joined(), just computed over the
    longer combined history. Feed this into dynamic_threshold.walk_forward_run
    /run_grid/run_pbo_grid's `extra_history=` for the BPV/harcj_only side —
    lets a lookback draw its trailing window from 2025 instead of truncating
    the backtest's own scored range.

    IV-combined modes (iv_only/and_both/or_either) have no equivalent yet —
    there's no pre-2026 mean_iv warm-up series (see conversation history:
    pending proper CME preproc IV data, since the backtest output's atm_iv
    isn't a clean single source across DOW folders).
    """
    full_bpv = extended_bpv_history()
    pred_vol_pct = full_bpv.shift(1).dropna().apply(bpv_to_vol_pct)
    backtest_index = load_joined(profile).index
    return pred_vol_pct[~pred_vol_pct.index.isin(backtest_index)]


def available_warmup_days(profile: str = PROFILE_NAME) -> int:
    """How many days of extra_history are available for the BPV/harcj_only side."""
    return len(extended_pred_vol_pct_history(profile))


def compute_iv_series(
    mode: str,                      # "point" | "mean"
    timestamp: str | None = None,   # "HH:MM" — used when mode="point"
    start_time: str | None = None,  # "HH:MM" — used when mode="mean"
    end_time: str | None = None,    # "HH:MM" — used when mode="mean"
) -> pd.Series:
    """
    Same contract as data_loader.compute_iv_series, but sourced from
    research/atm_iv/SPWX/ (full-day bars, 00:00-23:59, unlike live_dumps'
    narrow 07:30-08:30 dump window) — the weekday-matched per-date store,
    not this backtest's raw per-DOW-folder output (see module docstring).

    Iterates every date in research/atm_iv/SPWX/, not just the backtest's
    own active days — that store spans 2025 warm-up + this backtest's own
    2026 days (like research/spot/SPWX/), so this one function serves both
    load_joined_with_iv() (which only keeps the intersection with its own
    scored index) and extended_iv_series() (which keeps everything else).
    """
    result: dict[pd.Timestamp, float] = {}
    for date in sorted(_list_daily_files(_ATM_IV_DIR)):
        iv_series = _load_daily_series(_ATM_IV_DIR, date, "atm_iv")
        if iv_series is None or iv_series.empty:
            continue
        if mode == "point":
            t = pd.Timestamp.combine(date.date(), pd.Timestamp(timestamp).time())
            sub = iv_series.loc[:t]
            if sub.empty:
                continue
            result[date] = float(sub.iloc[-1])
        elif mode == "mean":
            s = pd.Timestamp.combine(date.date(), pd.Timestamp(start_time).time())
            e = pd.Timestamp.combine(date.date(), pd.Timestamp(end_time).time())
            sub = iv_series.loc[s:e]
            if sub.empty:
                continue
            result[date] = float(sub.mean())
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Expected 'point' or 'mean'.")
    return pd.Series(result, name="iv").sort_index()


def extended_iv_series(
    profile: str = PROFILE_NAME,
    mode: str = "mean",
    timestamp: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
) -> pd.Series:
    """
    IV counterpart of extended_pred_vol_pct_history(): the same
    compute_iv_series() values, restricted to dates strictly before the
    backtest's own first scored day — for dynamic_threshold's extra_history=
    on the IV side (iv_only, and the IV half of and_both/or_either).
    """
    full_iv = compute_iv_series(mode, timestamp=timestamp, start_time=start_time, end_time=end_time)
    backtest_index = load_joined(profile).index
    return full_iv[~full_iv.index.isin(backtest_index)]


def available_iv_warmup_days(
    profile: str = PROFILE_NAME,
    mode: str = "mean",
    timestamp: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
) -> int:
    """IV counterpart of available_warmup_days() — how many pre-backtest days extended_iv_series() covers."""
    return len(extended_iv_series(profile, mode, timestamp, start_time, end_time))


def load_joined_with_iv(
    profile: str,
    iv_source_mode: str,
    iv_timestamp: str | None = None,
    iv_start_time: str | None = None,
    iv_end_time: str | None = None,
) -> pd.DataFrame:
    """Same contract as data_loader.load_joined_with_iv."""
    df = load_joined(profile)
    iv_series = compute_iv_series(
        iv_source_mode, timestamp=iv_timestamp, start_time=iv_start_time, end_time=iv_end_time,
    )
    df = df.copy()
    df["iv_dynamic"] = df.index.map(iv_series)
    return df.dropna(subset=["iv_dynamic"])


def load_static_config(profile: str) -> dict:
    """
    Today's actual LIVE-deployed threshold (dashboard_new/params/<live
    profile>.json) for this backtest's underlying — a fair "what would
    today's live cutoff do on our own backtest data" comparison, NOT a
    threshold calibrated on this backtest's data. exclusion_ranges/iv_cutoff
    are applied via static_threshold_baseline() to pred_vol_pct/mean_iv that
    are already computed on THIS backtest's own derived session/IV window
    (see compute_daily_metrics/derive_session_window) rather than the live
    system's fixed 08:30-15:00/07:30-08:30 windows — the cutoff values
    themselves are just fixed numbers, not tied to any particular window, so
    this comparison is well-posed despite the two systems using different
    clock windows to compute their own inputs.
    """
    underlying = PROFILE_TO_UNDERLYING[profile]
    live_profile = _LIVE_PROFILE_FOR_UNDERLYING[underlying]
    path = os.path.join(_LIVE_PARAMS_DIR, f"{live_profile}.json")
    with open(path) as f:
        params = json.load(f)
    return {
        "exclusion_ranges":    params.get("harcj_exclusion_ranges", []),
        "iv_cutoff":           params.get("iv_cutoff"),
        "flag_combine_method": params.get("flag_combine_method", "or"),
    }


def data_coverage(profile: str) -> dict:
    joined = load_joined(profile)
    session_start, session_end = derive_session_window()
    return {
        "session_window": (session_start, session_end),
        "pred_vol_pct_range": (
            (joined.index.min(), joined.index.max()) if not joined.empty else (None, None)
        ),
        "pnl_range": (joined.index.min(), joined.index.max()) if not joined.empty else (None, None),
        "joined_range": (joined.index.min(), joined.index.max()) if not joined.empty else (None, None),
        "joined_n_days": len(joined),
    }
