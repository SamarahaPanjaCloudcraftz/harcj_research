"""
Extracts daily 1-min atm_iv series from SPXW_baseline (the full 2025
backtest at research/Baseline_backtest/SPXW_baseline/), one CSV per date,
into research/atm_iv/SPWX/ — alongside the 2026 atm_iv already extracted
there by extract_atm_iv.py (from spxw_gamma_hedge_false). No filename
collision (2025 vs 2026 dates), so this gives one continuous daily atm_iv
store spanning both — the IV counterpart of research/spot/SPWX/'s
2025+2026 coverage, needed for the same reason: extending the BPV/harcj_only
side's walk-forward lookback already draws on 2025 (extended_bpv_history),
IV-combined modes need the equivalent pre-2026 mean-IV history.

Same weekday-matching rule as extract_atm_iv.py, applied to SPXW_baseline
instead: for each date, only the DOW folder whose weekday matches that
date. atm_iv genuinely differs across the 5 folders for the same date in
SPXW_baseline too (same structural cause as spxw_gamma_hedge_false — each
folder prices a different expiry chain; measured mean relative spread
~47% — see conversation history), so the weekday-matching folder is the
only one that's actually pricing the chain that's live/0DTE that day.

Output: one CSV per date, columns "timestamp,atm_iv", full 00:00-23:59 CT
1-min series (each source minute is deduped from its trade_done True/False
pair — atm_iv is identical either way, so the last row per timestamp is
kept).

Usage:
    python3 extract_atm_iv_2025.py
"""

from __future__ import annotations

import glob
import os

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESEARCH_DIR = os.path.dirname(_HERE)
_SOURCE_DIR = os.path.join(_RESEARCH_DIR, "Baseline_backtest", "SPXW_baseline", "2025")
_OUT_DIR = os.path.join(_RESEARCH_DIR, "atm_iv", "SPWX")

_DOW_TO_WEEKDAY = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4}


def _list_dow_backtest_files(dow: str) -> dict[str, str]:
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
    """For each date, only the DOW folder whose weekday matches that date."""
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
