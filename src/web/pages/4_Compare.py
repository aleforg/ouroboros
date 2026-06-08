"""Compare page — cross-run aggregation using ``metrics.aggregate_runs()``.

Lets the user select 2+ completed runs and shows:
- Cross-run ASR per category (mean ± std)
- Per-seed stability (success rate and mean iters across runs)
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ouroboros.metrics import aggregate_runs
from ouroboros.web.charts import cross_run_asr_chart
from ouroboros.web.data import get_results_dir, list_runs, read_meta

RESULTS_DIR = get_results_dir()

st.title("🔀 Compare Runs")

# ── Run selector ──────────────────────────────────────────────────────────────

all_runs = list_runs(RESULTS_DIR)
finished_runs = [r for r in all_runs if r["meta"] and r["meta"].get("ended_at")]

if len(finished_runs) < 2:
    st.info(
        f"Need at least **2 completed** runs to compare.  "
        f"Found {len(finished_runs)} completed run(s) out of {len(all_runs)} total."
    )
    st.stop()

run_id_options = [r["run_id"] for r in finished_runs]
selected_run_ids = st.multiselect(
    "Select runs to compare (2 or more)",
    run_id_options,
    default=run_id_options[:min(len(run_id_options), 3)],
    format_func=lambda rid: (
        f"{rid}  [{next((r['meta']['config']['mode'] for r in finished_runs if r['run_id'] == rid), '?')}]"
    ),
)

if len(selected_run_ids) < 2:
    st.warning("Please select at least 2 runs.")
    st.stop()

# ── Aggregation ──────────────────────────────────────────────────────────────

with st.spinner(f"Aggregating {len(selected_run_ids)} runs…"):
    run_dirs = [RESULTS_DIR / rid for rid in selected_run_ids]
    try:
        agg = aggregate_runs(run_dirs)
    except Exception as exc:
        st.error(f"Aggregation failed: {exc}")
        st.stop()

# ── Summary ───────────────────────────────────────────────────────────────────

st.subheader("Selected runs")
meta_rows = []
for rid in selected_run_ids:
    m = read_meta(RESULTS_DIR / rid) or {}
    cfg = m.get("config", {})
    meta_rows.append({
        "run_id": rid,
        "mode": cfg.get("mode", "?"),
        "attacker": cfg.get("attacker_model", "?").split(":")[0],
        "judge": f"{m.get('judge_backend', '?')}/{m.get('judge_model', '?')}",
        "seeds_filter": cfg.get("seeds_filter") or "—",
        "started_at": m.get("started_at", "?")[:19],
    })
st.dataframe(meta_rows, use_container_width=True)

# ── Cross-run ASR per category ────────────────────────────────────────────────

st.subheader("Cross-Run ASR per Category")
per_cat_data = agg.get("per_category", [])
if per_cat_data:
    per_cat_df = pd.DataFrame(per_cat_data)
    col_a, col_b = st.columns([2, 3])
    with col_a:
        fmt_cols = {c: "{:.3f}" for c in per_cat_df.columns if per_cat_df[c].dtype == float}
        st.dataframe(per_cat_df.style.format(fmt_cols, na_rep="—"), use_container_width=True)
    with col_b:
        try:
            st.altair_chart(cross_run_asr_chart(per_cat_df), use_container_width=True)
        except Exception as e:
            st.warning(f"Chart error: {e}")
else:
    st.caption("No per-category aggregate data.")

# ── Per-seed stability ────────────────────────────────────────────────────────

st.subheader("Per-Seed Stability")
stability_data = agg.get("per_seed_stability", [])
if stability_data:
    stab_df = pd.DataFrame(stability_data)

    def _color_success_rate(val: float) -> str:
        if val >= 0.9:
            return "color: #2d8a4e"  # green
        if val <= 0.1:
            return "color: #c0392b"  # red
        return "color: #e67e22"  # orange

    fmt = {c: "{:.3f}" for c in stab_df.columns if stab_df[c].dtype == float}
    styled = stab_df.style.format(fmt, na_rep="—")
    if "success_rate" in stab_df.columns:
        styled = styled.applymap(_color_success_rate, subset=["success_rate"])
    st.dataframe(styled, use_container_width=True)
else:
    st.caption("No per-seed stability data.")

# ── Aggregate footer ──────────────────────────────────────────────────────────

st.caption(f"Aggregated {agg.get('n_runs', len(selected_run_ids))} run(s).")
