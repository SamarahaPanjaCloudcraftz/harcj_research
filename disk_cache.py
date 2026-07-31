"""
Disk-backed cache, on top of (not instead of) st.cache_data.

st.cache_data's in-memory cache is fast but empty on every process restart —
so every restart of the Streamlit process recomputes everything from
scratch, including the slow PBO grids. This adds a second layer: results are
pickled into dashboard_data/, keyed by function name + arguments, and reused
across restarts. A parameter combination that's never been computed before
still triggers a fresh computation (and gets saved for next time); nothing
here special-cases "default settings" — it falls out naturally, since the
default settings are simply the first combination anyone (or the warm-up
script) ever asks for.

Layout — organized so it stays navigable as more data sources get added
(e.g. a future extended-backtest PnL feed alongside today's live_dumps):

    dashboard_data/
      <source>/                e.g. "live_dumps" (data_loader.DATA_SOURCE_NAME)
        <profile>/              e.g. "spxw_persistence"
          <function_name>/      e.g. "cached_paired_pbo_grid"
            <hash>.pkl           the actual result, pickled
            <hash>.json          human-readable args for that hash — so the
                                  directory is inspectable without decoding
                                  hashes; never read back by load_or_compute,
                                  purely for a human to `cat` and understand
                                  what a given cache file corresponds to.

Clearing a subset (e.g. "this profile's cache is stale, the underlying
dumps/ changed") is then just an `rm -rf dashboard_data/<source>/<profile>/`
rather than nuking the whole cache or hand-decoding opaque filenames.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from typing import Callable

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_HERE, "dashboard_data")


def _make_key(name: str, args: tuple) -> str:
    raw = name + "|" + repr(args)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def load_or_compute(name: str, args: tuple, compute_fn: Callable,
                     *, profile: str, source: str):
    """
    Return the pickled result for (source, profile, name, args) if it
    exists on disk; otherwise call compute_fn(), save the result (plus a
    human-readable .json sidecar of the args), and return it.

    profile/source are required and explicit (not inferred from args) so
    the directory structure stays correct even if a function's argument
    order ever changes.
    """
    entry_dir = os.path.join(CACHE_DIR, source, profile, name)
    os.makedirs(entry_dir, exist_ok=True)

    key = _make_key(name, args)
    path = os.path.join(entry_dir, f"{key}.pkl")
    meta_path = os.path.join(entry_dir, f"{key}.json")

    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    result = compute_fn()

    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(result, f)
    os.replace(tmp_path, path)  # atomic — no partially-written cache files

    try:
        with open(meta_path, "w") as f:
            json.dump({"function": name, "args": [repr(a) for a in args]}, f, indent=2)
    except Exception:
        pass  # the .json sidecar is a debugging aid only — never fatal

    return result
