"""
Extracts daily 1-min atm_iv series from a 2025 warm-up backtest, one CSV
per date, into research/atm_iv/<TAG>/ — alongside the 2026 atm_iv already
extracted there by extract_atm_iv.py. No filename collision (2025 vs 2026
dates), so this gives one continuous daily atm_iv store spanning both — the
IV counterpart of research/spot/<TAG>/'s 2025+2026 coverage, needed for the
same reason: extending the BPV/harcj_only side's walk-forward lookback
already draws on 2025 (extended_bpv_history), IV-combined modes need the
equivalent pre-2026 mean-IV history.

Same weekday-matching rule as extract_atm_iv.py: for each date, only the
DOW folder whose weekday matches that date. atm_iv genuinely differs
across the 5 folders for the same date (each folder prices a different
expiry chain; measured mean relative spread ~38-47% — see conversation
history), so the weekday-matching folder is the only one that's actually
pricing the chain that's live/0DTE that day.

Rows with no atm_iv are dropped.

Output: one CSV per date, columns "timestamp,atm_iv", 1-min series (each
source minute is deduped from its trade_done True/False pair — atm_iv is
identical either way, so the last row per timestamp is kept).

Reusable for other underlyings' 2025 warm-up backtests via --source-dir/
--out-dir, e.g. for NDAQ:
    python3 extract_atm_iv_2025.py \\
        --source-dir ../Baseline_backtest/ndaq_100000const_20delta/2025 \\
        --out-dir ../atm_iv/NDAQ

Usage:
    python3 extract_atm_iv_2025.py
"""

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESEARCH_DIR = os.path.dirname(_HERE)
_DEFAULT_SOURCE_DIR = os.path.join(_RESEARCH_DIR, "Baseline_backtest", "SPXW_baseline", "2025")
_DEFAULT_OUT_DIR = os.path.join(_RESEARCH_DIR, "atm_iv", "SPWX")

_DOW_TO_WEEKDAY = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4}


def _list_dow_backtest_files(source_dir: str, dow: str) -> dict[str, str]:
    pattern = os.path.join(source_dir, dow, "backtest", "*.csv")
    out = {}
    for f in glob.glob(pattern):
        name = os.path.basename(f)
        if name.startswith("._"):
            continue
        date_str = os.path.splitext(name)[0]
        out[date_str] = f
    return out


def _resolve_date_to_file(source_dir: str) -> dict[str, str]:
    """For each date, only the DOW folder whose weekday matches that date."""
    resolved: dict[str, str] = {}
    for dow, weekday in _DOW_TO_WEEKDAY.items():
        for date_str, path in _list_dow_backtest_files(source_dir, dow).items():
            if pd.Timestamp(date_str).weekday() != weekday:
                continue
            resolved[date_str] = path
    return resolved


def _extract_one(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["timestamp", "atm_iv"], parse_dates=["timestamp"])
    df = df.dropna(subset=["atm_iv"])
    df = df.drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default=_DEFAULT_SOURCE_DIR, help="<...>/2025 folder with MON..FRI subfolders")
    parser.add_argument("--out-dir", default=_DEFAULT_OUT_DIR, help="research/atm_iv/<TAG> output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    date_to_file = _resolve_date_to_file(args.source_dir)
    print(f"Found {len(date_to_file)} weekday-matched dates in {args.source_dir}.")

    written = 0
    for date_str in sorted(date_to_file):
        out_path = os.path.join(args.out_dir, f"{date_str}.csv")
        df = _extract_one(date_to_file[date_str])
        if df.empty:
            continue
        df.to_csv(out_path, index=False)
        written += 1

    print(f"Wrote {written} daily atm_iv CSVs to {args.out_dir}")


if __name__ == "__main__":
    main()
