"""Ouroboros Dashboard — Streamlit entry point.

Launched via ``ouroboros dashboard [--port 8501]``.
Uses ``st.navigation`` (Streamlit ≥ 1.36) for multi-page routing so that
session_state (including the active run id) survives page switches.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from ouroboros.web.data import get_results_dir, list_runs, get_running_jobs

_HERE = Path(__file__).parent

st.set_page_config(
    page_title="Ouroboros Dashboard",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Page definitions ──────────────────────────────────────────────────────────

launch_page  = st.Page(str(_HERE / "pages" / "1_Launch.py"),  title="Launch",  icon="⚡")
monitor_page = st.Page(str(_HERE / "pages" / "2_Monitor.py"), title="Monitor", icon="📡")
results_page = st.Page(str(_HERE / "pages" / "3_Results.py"), title="Results", icon="📊")
compare_page = st.Page(str(_HERE / "pages" / "4_Compare.py"), title="Compare", icon="🔀")

pg = st.navigation([launch_page, monitor_page, results_page, compare_page])

# ── Sidebar: quick stats ──────────────────────────────────────────────────────

with st.sidebar:
    results_dir = get_results_dir()
    st.caption(f"Results dir: `{results_dir}`")
    runs = list_runs(results_dir)
    running = get_running_jobs(results_dir)

    col1, col2 = st.columns(2)
    col1.metric("Total runs", len(runs))
    col2.metric("Running", len(running))

    if running:
        st.caption("🟢 Active job(s):")
        for j in running:
            label = j.get("run_id") or j.get("pending_id", "?")
            st.caption(f"  • `{label}`")

# ── Run the selected page ─────────────────────────────────────────────────────

pg.run()
