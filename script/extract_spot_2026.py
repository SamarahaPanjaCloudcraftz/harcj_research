"""
Extracts daily 1-min spot price series from a 2026 backtest source, one CSV
per date, into research/spot/<TAG>/ — alongside the 2025 warm-up spot
already extracted there by extract_spot.py. No filename collision (2025 vs
2026 dates), so this gives one continuous daily spot store spanning both.

Same weekday-matching rule as extract_atm_iv.py: for each date, only the
DOW folder whose weekday matches that date. Spot happens to be
byte-identical across all 5 folders for the same date anyway (pure market
replay, verified — unlike atm_iv, which is chain/expiry-dependent and
genuinely differs per folder), so this is just for a uniform rule across
both extraction scripts, not a correctness requirement for spot itself.

Output: one CSV per date, columns "timestamp,spot", full 00:00-23:59 CT
1-min series (each source minute is deduped from its trade_done True/False
pair — spot is identical either way, so the last row per timestamp is kept).

Reusable for other underlyings' 2026 backtests via --source-dir/--out-dir,
e.g. for NDAQ:
    python3 extract_spot_2026.py \\
        --source-dir ../../backtests/ndaq_delta_condor_trade_start_time_08-30_delta_condor_trade_stop_time_12-00_gamma_hedge_false/2026 \\
        --out-dir ../spot/NDAQ

Usage:
    python3 extract_spot_2026.py
"""

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESEARCH_DIR = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_RESEARCH_DIR)
_DEFAULT_SOURCE_DIR = os.path.join(_REPO_ROOT, "backtests", "spxw_gamma_hedge_false", "2026")
_DEFAULT_OUT_DIR = os.path.join(_RESEARCH_DIR, "spot", "SPWX")

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
    resolved: dict[str, str] = {}
    for dow, weekday in _DOW_TO_WEEKDAY.items():
        for date_str, path in _list_dow_backtest_files(source_dir, dow).items():
            if pd.Timestamp(date_str).weekday() != weekday:
                continue
            resolved[date_str] = path
    return resolved


def _extract_one(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["timestamp", "spot"], parse_dates=["timestamp"])
    df = df.drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default=_DEFAULT_SOURCE_DIR, help="<...>/2026 folder with MON..FRI subfolders")
    parser.add_argument("--out-dir", default=_DEFAULT_OUT_DIR, help="research/spot/<TAG> output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    date_to_file = _resolve_date_to_file(args.source_dir)
    print(f"Found {len(date_to_file)} weekday-matched dates in {args.source_dir}.")

    written = 0
    for date_str in sorted(date_to_file):
        out_path = os.path.join(args.out_dir, f"{date_str}.csv")
        df = _extract_one(date_to_file[date_str])
        df.to_csv(out_path, index=False)
        written += 1

    print(f"Wrote {written} daily spot CSVs to {args.out_dir}")


if __name__ == "__main__":
    main()
