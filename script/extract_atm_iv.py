"""
Extracts daily 1-min atm_iv series from the current backtest source
(backtests/spxw_gamma_hedge_false/2026/), one CSV per date, into
research/atm_iv/SPWX/ — the IV counterpart of research/spot/SPWX/
(see script/extract_spot.py).

Unlike spot (byte-identical across all 5 DOW folders — pure market replay),
atm_iv genuinely differs across folders for the same date: each DOW folder
prices a DIFFERENT expiry chain (MON's atm_iv is the Monday-expiry chain's,
FRI's is the Friday-expiry chain's), so there's no single folder that's
"right" for a date in general — measured mean relative spread across the 5
folders is ~38% (see conversation history). The one folder that IS
authoritative for a given date is the one whose own weekday matches that
date (that's the chain that's actually 0DTE / actively traded that day) —
the same weekday-matching rule backtest_source.list_active_days() already
uses for spot/PnL. This script applies that same rule to atm_iv and
materializes it into its own per-date store, so the dashboard's
auto-derived-window machinery (BPV session window, IV window) can compute
mean-IV from a clean daily series independent of backtest folder layout —
the same way it already does for spot — regardless of what folder
structure or time range a future backtest source uses.

Scoped to the CURRENT 2026 backtest only (not a 2025 warm-up — the
cross-folder atm_iv discrepancy is being investigated separately before any
IV warm-up history gets built).

Output: one CSV per date, columns "timestamp,atm_iv", full 00:00-23:59 CT
1-min series (each source minute is deduped from its trade_done True/False
pair — atm_iv is identical either way, so the last row per timestamp is
kept).

Usage:
    python3 extract_atm_iv.py
"""

from __future__ import annotations

import glob
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESEARCH_DIR = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_RESEARCH_DIR)
_SOURCE_DIR = os.path.join(_REPO_ROOT, "backtests", "spxw_gamma_hedge_false", "2026")
_OUT_DIR = os.path.join(_RESEARCH_DIR, "atm_iv", "SPWX")

_DOW_TO_WEEKDAY = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4}


def _list_dow_backtest_files(dow: str) -> dict[str, str]:
    """date string (YYYYMMDD) -> file path, for one DOW folder."""
    pattern = os.path.join(_SOURCE_DIR, dow, "backtest", "*.csv")
    out = {}
    for f in glob.glob(pattern):
        name = os.path.basename(f)
        if name.startswith("._"):
            continue
        date_str = os.path.splitext(name)[0]
        out[date_str] = f
    return out


def _resolve_date_to_file() -> dict[str, str]:
    """
    For each date, only the DOW folder whose weekday matches that date's
    actual calendar weekday — same rule as backtest_source.list_active_days().
    """
    resolved: dict[str, str] = {}
    for dow, weekday in _DOW_TO_WEEKDAY.items():
        for date_str, path in _list_dow_backtest_files(dow).items():
            if pd.Timestamp(date_str).weekday() != weekday:
                continue
            resolved[date_str] = path
    return resolved


def _extract_one(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["timestamp", "atm_iv"], parse_dates=["timestamp"])
    df = df.drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp")
    return df


def main():
    os.makedirs(_OUT_DIR, exist_ok=True)
    date_to_file = _resolve_date_to_file()
    print(f"Found {len(date_to_file)} weekday-matched dates in {_SOURCE_DIR}.")

    written = 0
    for date_str in sorted(date_to_file):
        out_path = os.path.join(_OUT_DIR, f"{date_str}.csv")
        df = _extract_one(date_to_file[date_str])
        df.to_csv(out_path, index=False)
        written += 1

    print(f"Wrote {written} daily atm_iv CSVs to {_OUT_DIR}")


if __name__ == "__main__":
    main()
