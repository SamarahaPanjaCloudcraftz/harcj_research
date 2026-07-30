# HARCJ Research Dashboard

Exploratory workspace for questions about the HARCJ / open-IV filters. Fully
separate from the live system in `dashboard_new/`:

- Reads `dashboard_new/dumps/` **read-only** — nothing here writes there
- Own conda env (`harcj_research`), own `requirements.txt`, own port — the live
  app/venv/scheduler are never touched
- Own Streamlit process, run manually, independent of `run_dashboard.sh`

## Setup

```
./setup.sh          # creates the harcj_research conda env, installs deps
./setup.sh --run     # also launches the app after setup
```

Or manually:

```
conda activate harcj_research
streamlit run app.py --server.port 8600
```

## Structure

```
research/
  data_loader.py          # read-only loaders from dashboard_new/dumps/
  bucketing.py            # equal-count / equal-width bucket assignment
  dynamic_threshold.py    # walk-forward sim + metrics (not yet built)
  app.py                  # entrypoint
  pages/
    1_bucket_explorer.py  # interactive bucket-vs-PnL diagnostic (done)
    2_threshold_grid.py   # lookback x n_buckets grid (not yet built)
```

## Data status

- `pred_vol_pct` / `mean_iv`: from `<profile>_stage3/roll.csv`, through 2026-07-22
- Daily PnL: currently from `<underlying>_stage2/pnl/*.csv` (the live scheduler's
  own PnL dump) — **this is a placeholder**. A dedicated backtest PnL series is
  coming later and will replace the body of `data_loader.load_daily_pnl()` only;
  no other file needs to change when that happens.
- Usable joined window today (bounded by PnL coverage, the shorter series):
  - SPXW: 2025-02-06 → 2026-04-24 (~300 days)
  - NDAQ: 2025-02-05 → 2026-04-02 (~290 days)

## Page 1 — Bucket Explorer

Pick a profile, date range, bucket count, and bucketing method (equal-count
vs equal-width), bucket on either `pred_vol_pct` or raw `pred_bpv`. Shows
total PnL and mean PnL per bucket as a table + bar charts, plus win rate.
Use this to see the actual PnL-vs-predicted-vol relationship in the data
before choosing an exclusion rule for the Threshold Grid page — e.g. whether
it's really a monotonic "high vol = bad" relationship, or something else
(only the tail buckets are bad, non-monotonic, etc).

## Page 2 — Threshold Grid (next)

Planned: lookback (x-axis) x n_buckets (y-axis) grid, walk-forward (no
lookahead), scored by PnL / Sortino / Sharpe / win-rate. Toggles: bucketing
method, exclusion rule (worst-PnL dynamic vs highest-vol fixed), n_exclude,
IV filter mode (isolated / combined with existing static iv_cutoff / both).
Reference values shown alongside: always-trade baseline, today's static
threshold result. Cells with too few days-per-bucket for the chosen
(lookback, n_buckets) pair get flagged as statistically unreliable rather
than silently shown.
