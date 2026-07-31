"""
Extracts daily 1-min spot price series from the current backtest source
(backtests/spxw_gamma_hedge_false/2026/), one CSV per date, into
research/spot/SPWX/ — alongside the 2025 warm-up spot already extracted
there by extract_spot.py (from SPXW_baseline). No filename collision (2025
vs 2026 dates), so this gives one continuous daily spot store spanning both.

Same weekday-matching rule as extract_atm_iv.py: for each date, only the
DOW folder whose weekday matches that date. Spot happens to be
byte-identical across all 5 folders for the same date anyway (pure market
replay, verified — unlike atm_iv, which is chain/expiry-dependent and
genuinely differs per folder), so this is just for a uniform rule across
both extraction scripts, not a correctness requirement for spot itself.

Output: one CSV per date, columns "timestamp,spot", full 00:00-23:59 CT
1-min series (each source minute is deduped from its trade_done True/False
pair — spot is identical either way, so the last row per timestamp is kept).

Usage:
    python3 extract_spot_2026.py
"""

from __future__ import annotations

import glob
import os

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESEARCH_DIR = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_RESEARCH_DIR)
_SOURCE_DIR = os.path.join(_REPO_ROOT, "backtests", "spxw_gamma_hedge_false", "2026")
_OUT_DIR = os.path.join(_RESEARCH_DIR, "spot", "SPWX")

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
    resolved: dict[str, str] = {}
    for dow, weekday in _DOW_TO_WEEKDAY.items():
        for date_str, path in _list_dow_backtest_files(dow).items():
            if pd.Timestamp(date_str).weekday() != weekday:
                continue
            resolved[date_str] = path
    return resolved


def _extract_one(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["timestamp", "spot"], parse_dates=["timestamp"])
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

    print(f"Wrote {written} daily spot CSVs to {_OUT_DIR}")


if __name__ == "__main__":
    main()
