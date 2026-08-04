# Global Cutoff Framework

Consolidates the fragmented per-page cutoff searches (Threshold Grid, Cutoff
Transfer, Exchange Events) into one systematic search per index, so nothing
is left unexplored due to which page happened to cover which slice of the
space.

## 1) What uniquely defines a cutoff

The atomic unit is a **slice candidate**:

```
(index, day-type, origin, mode, L, B)
```

- **index** — SPX or NDAQ.
- **day-type** — `default` or `event`, for that index. Not a free per-date
  choice — determined externally by the exchange-events calendar. Each
  (index, day-type) maps to a **window** (start_time, end_time) read off
  that source's own derived session window — not hardcoded. Today's 4
  concrete windows: SPX-default=05:00–15:00, SPX-event=08:30–12:00,
  NDAQ-default=06:00–10:00, NDAQ-event=08:30–12:00 — but the framework
  itself makes no assumption about these specific times; if a
  differently-timed backtest is added later it becomes a new window
  candidate without changing the framework.
- **origin** — which of the 4 (index × day-type) sources the cutoff
  decision is actually computed from. For target (X, Y), the 4 candidates
  are (X,Y) itself, (X,¬Y), (¬X,Y), (¬X,¬Y) — a pure date-mask transplant
  when origin ≠ target (no recomputation of the target's own signals,
  same mechanism as the existing Cutoff Transfer page). **Both day-type
  slices draw from all 4 sources — no restriction.**
- **mode** — `harcj_only` / `iv_only` / `and_both` / `or_either`. IV
  source is fixed to **mean** (window-average) — not selectable.
  Bucketing mechanics fixed to dashboard defaults: `method=equal_count`,
  `rule=highest_vol`, `n_exclude=1` — not part of the search.
- **L, B** — lookback and bucket count, from the standard grid
  `(60,90,120,180) × (4,5,8,10)`.

Per (index, day-type) slice: 4 origins × 4 modes × 16 (L,B) ≈ **256
candidates** (a few less after dropping invalid `harcj_only` cells where L
exceeds the data length).

A single slice candidate only answers "trade or not" for one day-type's
dates — not a complete, deployable answer, since every calendar date is
either an event day or a non-event day and the real system needs an
answer for both.

**A complete cutoff for an index is therefore a pair of slice
candidates** — one for the event day-type, one for the non-event
day-type — spliced into one full-year trade/no-trade series (event and
non-event dates are disjoint, so this is pure concatenation, not
recomputation):

```
cutoff(index) = ( (event_origin, event_mode, event_L, event_B),
                   (non_event_origin, non_event_mode, non_event_L, non_event_B) )
```

Two fixed reference cutoffs exist outside this parameterization, scored
the same way as every searched cutoff:

- **Current live baseline** — the fixed, non-walk-forward rule using the
  real deployed thresholds (`dashboard_new/params/*.json`
  `harcj_exclusion_ranges`/`iv_cutoff`): BPV-only exclusion on non-event
  days, BPV AND mean-IV (07:30–08:30) on event days. Already a genuine
  full-year blend (`exchange_events.live_baseline_result`).
- **No-cutoff (always-trade)** — full year, no exclusions.

## 2) How we search for the best cutoff

### Stage 0 — Base walk-forward cells (one-time, shared)

A slice candidate's decision depends only on `(origin, mode, L, B)` —
never on which index/day-type it's being applied to. So there are only
4×4×16 = **256 distinct walk-forward cells in the entire framework**,
computed once and cached; every slice candidate for every target is a
cached-mask lookup (`apply_transplanted_cutoff`-style), not a new
computation.

### Stage 1 — Blend + cheap metrics, fully exhaustive (no pre-reduction)

Per index: 256 event-slice candidates × 256 non-event-slice candidates =
**65,536 complete cutoffs**. Every one is scored — nothing is dropped
before being scored. For each: splice the two cached daily-PnL series and
compute `avg_daily_pnl`, `sortino`, `max_daily_loss`, `drawdown` — cheap
arithmetic, no CSCV.

Ranked via the existing `_rank_and_weight` composite mechanism
(percentile-rank based, not raw magnitude), weights renormalized across
these 4 metrics (~0.25 each, `pbo` excluded since it isn't computed yet).

### Stage 1.5 — Pool ranking (for finalist selection, not for dropping candidates)

Partition the 65,536 cutoffs by their categorical identity
`(event_origin, event_mode, non_event_origin, non_event_mode)` → **256
pools** (16 event-side origin×mode combos × 16 non-event-side origin×mode
combos), each pool containing the 16×16=256 (L,B)-pair variations. Within
each pool, rank by the same Stage 1 composite score.

### Stage 2 finalists — union, not intersection

```
finalists = (global top-K by composite score, across all 65,536)
            ∪ (top-C from each of the 256 pools)
```

Guarantees the overall best performers are never excluded (the K safety
net), **and** guarantees every distinct (origin, mode)×(origin, mode)
combination is represented by at least C of its own best (L,B) pairs
regardless of whether it also happened to crack the global top-K — so a
combination that's merely "good but not top-ranked overall" still gets
evaluated by CSCV, rather than being silently dropped by a pre-blend
intuition filter.

### Stage 2 — CSCV/PBO on the finalist union

For each finalist, run CSCV against **both** reference baselines
separately (current live baseline, no-cutoff).

**Output per index:** 2 rankings — PBO vs. current live baseline, PBO vs.
no-cutoff. **4 rankings total** across SPX and NDAQ.

## Time complexity, as a function of K and C

- **Stage 0:** O(256), one-time. Initially not negligible in practice: two
  shared `BacktestSource` methods (`derive_session_window()`,
  `load_joined()`) were uncached and got called redundantly on every
  cross-origin transplant (`apply_transplanted_cutoff` has no cache of its
  own) — costing ~0.3s and ~1.2–1.6s per call respectively, discovered via
  the unit smoke test (a tiny 32-candidate slice took 30.7s before the
  fix). Both are now memoized on the `BacktestSource` instance (safe: both
  are pure functions of the source's own on-disk files, and every caller
  treats their results as read-only). After the fix, the same 32-candidate
  slice build dropped to 5.7s cold / 0.04s warm — Stage 0 is genuinely
  negligible now. All 4 existing pages regression-tested clean (zero
  exceptions) after this change, since it touches a function shared
  dashboard-wide.
- **Stage 1:** fixed at O(65,536) per index — independent of C and K,
  since nothing is pre-filtered. Time = `65,536 · t₁`.
  **Measured** (unit timing test, one candidate shape, `and_both` mode,
  L=90/B=5, NDAQ default+event pair): **t₁ ≈ 1.183 ms/call → 65,536
  calls ≈ 77.5s per index**.
- **Stage 1.5:** ranking/grouping already-computed data — negligible
  additional cost.
- **Stage 2:** `|finalists| ≤ K + 256·C` (upper bound before dedup —
  actual union is smaller since some pool-tops are already in the global
  top-K). Time ≈ `2 · |finalists| · t_cscv`.
  **Measured**: **t_cscv ≈ 1.941s/call** (same unit test) — close to,
  slightly above, the earlier ~1.6s estimate from the `event_day_ranking`
  re-warm.

Total per index ≈ `77.5s (Stage 1) + 3.88 · |finalists| (Stage 2)`:

| K  | C | Upper-bound finalists | Total time per index | Both indices |
|----|---|------------------------|------------------------|--------------|
| 20 | 1 | 20 + 256 = 276         | ≈1,149s (~19.1 min) | ~38 min |
| 50 | 1 | 50 + 256 = 306         | ≈1,266s (~21.1 min) | ~42 min |
| 20 | 2 | 20 + 512 = 532         | ≈2,142s (~35.7 min) | ~71 min |
| 50 | 0 (pure top-K, no pool guarantee) | 50 | ≈272s (~4.5 min) | ~9 min |

C dominates Stage 2's cost far more than K does, since 256 pools means
even C=1 adds up to 256 finalists — an order of magnitude more than a
typical K.

**Decision: K=50, C=2** — ~2,259s (~37.7 min) per index, ~75 min for
both SPX+NDAQ (upper bound before dedup — actual will be somewhat lower).
Chosen over C=1 to also guard against a pool's *best* (L,B) being a
fluke and its second-best being the more robust choice, at the cost of
roughly double the runtime — still an acceptable offline batch job.
