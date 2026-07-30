"""
HARCJ Research Dashboard — entrypoint.

Fully separate from dashboard_new/ — read-only against its dumps/ folder,
own conda env (see setup.sh), own port. Nothing here is part of the live
system; run it with:

    conda activate harcj_research
    streamlit run app.py --server.port 8600

Pages live under pages/ (Streamlit's native multipage convention) so new
research questions can be added as new files without touching existing ones.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="HARCJ Research",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("HARCJ Research Dashboard")
st.markdown(
    """
Exploratory workspace for questions about the HARCJ / open-IV filters —
entirely separate from the live dashboard in `dashboard_new/`. Nothing here
writes to, imports from the scheduler, or otherwise touches the live system;
everything reads `dashboard_new/dumps/` read-only.
 
**Pages** (see sidebar):
- **Bucket Explorer** — pick a date range and bucket count, see the actual
  PnL-by-predicted-vol-bucket relationship in the data before assuming
  anything about it.
- **Threshold Grid** *(coming next)* — lookback × bucket-count grid, scored
  by PnL / Sortino / Sharpe / win-rate, for a dynamic (rolling) version of
  today's static BPV exclusion threshold.
"""
)

st.info("Select a page from the sidebar to get started.")
