"""
Extracts daily 1-min spot price series from SPXW_baseline (the full 2025
backtest at research/Baseline_backtest/SPXW_baseline/), for use as the
pre-2026 BPV warm-up history for the spxw_gamma_hedge_false backtest source
(research/backtest_source.py) — extends the trailing lookback window back
before the backtest's own first day, without touching its scored/PnL range.

SPXW_baseline is split into 5 DOW-named folders (MON/TUE/WED/THU/FRI), same
convention as spxw_gamma_hedge_false — each folder independently spans the
whole year's business days (not just its own weekday), and `spot` is
byte-identical across all 5 for the same date (verified: it's a pure market
data replay, not strategy-dependent — only atm_iv/portfolio state differ
between folders). So this pulls each date's spot column from whichever
folder has it first (MON, TUE, WED, THU, FRI, in that order), unioning
across all 5 to cover the 2 dates missing from MON alone (2025-12-30/31).

Output: one CSV per date, columns "timestamp,spot", full 00:00-23:59 CT
1-min series (each source minute is deduped from its trade_done True/False
pair — spot is identical either way, so the last row per timestamp is kept).

Reusable for other underlyings' 2025 warm-up backtests via --source-dir/
--out-dir, e.g. for NDAQ (research/Baseline_backtest/ndaq_100000const_20delta):
    python3 extract_spot.py \\
        --source-dir ../Baseline_backtest/ndaq_100000const_20delta/2025 \\
        --out-dir ../spot/NDAQ

Usage:
    python3 extract_spot.py
"""

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESEARCH_DIR = os.path.dirname(_HERE)
_DEFAULT_SOURCE_DIR = os.path.join(_RESEARCH_DIR, "Baseline_backtest", "SPXW_baseline", "2025")
_DEFAULT_OUT_DIR = os.path.join(_RESEARCH_DIR, "spot", "SPWX")

_DOW_PRIORITY = ["MON", "TUE", "WED", "THU", "FRI"]


def _list_dow_backtest_files(source_dir: str, dow: str) -> dict[str, str]:
    """date string (YYYYMMDD) -> file path, for one DOW folder."""
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
    """Union across all 5 DOW folders, first-found wins (MON, TUE, WED, THU, FRI)."""
    resolved: dict[str, str] = {}
    for dow in _DOW_PRIORITY:
        for date_str, path in _list_dow_backtest_files(source_dir, dow).items():
            resolved.setdefault(date_str, path)
    return resolved


def _extract_one(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["timestamp", "spot"], parse_dates=["timestamp"])
    df = df.drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default=_DEFAULT_SOURCE_DIR, help="<...>/<year> folder with MON..FRI subfolders")
    parser.add_argument("--out-dir", default=_DEFAULT_OUT_DIR, help="research/spot/<TAG> output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    date_to_file = _resolve_date_to_file(args.source_dir)
    print(f"Found {len(date_to_file)} dates across {args.source_dir}'s 5 DOW folders.")

    written = 0
    for date_str in sorted(date_to_file):
        out_path = os.path.join(args.out_dir, f"{date_str}.csv")
        df = _extract_one(date_to_file[date_str])
        df.to_csv(out_path, index=False)
        written += 1

    print(f"Wrote {written} daily spot CSVs to {args.out_dir}")


if __name__ == "__main__":
    main()
