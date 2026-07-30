"""
Vendored core of the CSCV / Probability-of-Backtest-Overfitting algorithm
(Bailey, Borwein, Lopez de Prado, Zhu — 2015), from /home/oem/Downloads/pbo.

Only the algorithm (cscv.py, metrics.py) is vendored here — the GA-specific
notebook plumbing (result.json loading, GAConfig, mapper.csv) is not needed
and not included. Plotting is done separately in the research dashboard
(Plotly, matching its existing style) rather than reusing plots.py's
matplotlib figures.
"""

from .cscv import compute_cscv, stochastic_dominance, aggregate_series
from .metrics import REGISTRY

__all__ = ["compute_cscv", "stochastic_dominance", "aggregate_series", "REGISTRY"]
