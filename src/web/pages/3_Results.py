"""Results page — full interactive report for a completed run.

Reuses ``ouroboros.metrics`` functions directly (same data the static
``report.html`` uses) and renders them with Streamlit + Altair charts.

Heavy post-processing (FairFace KL, strategy clustering) is triggered
lazily via an ``ouroboros report`` subprocess so it doesn't block the UI.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from ouroboros.metrics import (
    asr_vs_iter,
    baseline_vs_iterative,
    judge_coverage,
    load_baseline,
    load_run,
    per_category,
    summary_per_seed,
)
from ouroboros.web.charts import asr_vs_iter_chart
from ouroboros.web.data import (
    get_results_dir,
    list_runs,
    read_meta,
    read_strategy_clusters,
)

RESULTS_DIR = get_results_dir()

st.title("📊 Results")

# ── Run selector ──────────────────────────────────────────────────────────────

all_runs = list_runs(RESULTS_DIR)
if not all_runs:
    st.info("No runs found yet. Go to **Launch** to start one.")
    st.stop()

run_ids = [r["run_id"] for r in all_runs]

# Pre-select if coming from Monitor
default_run = st.session_state.get("active_pending_id")  # may be a pending_id, not run_id
selected_idx = 0
for i, r in enumerate(all_runs):
    if r["meta"] and r["meta"].get("ended_at"):  # prefer a finished run
        selected_idx = i
        break

selected_run_id = st.selectbox(
    "Select run",
    run_ids,
    index=selected_idx,
    format_func=lambda rid: (
        f"{'✅ ' if any(r['meta'] and r['meta'].get('ended_at') for r in all_runs if r['run_id'] == rid) else '⏳ '}"
        f"{rid}"
    ),
)

run_dir = RESULTS_DIR / selected_run_id
meta = read_meta(run_dir) or {}

if not (run_dir / "run.jsonl").exists():
    st.warning("No ``run.jsonl`` found — the run may still be in progress or empty.")
    st.stop()

# ── Meta summary ──────────────────────────────────────────────────────────────

cfg = meta.get("config", {})
col1, col2, col3, col4 = st.columns(4)
col1.metric("Mode", cfg.get("mode", "?"))
col2.metric("Judge", f"{meta.get('judge_backend', '?')}")
col3.metric("Attacker", cfg.get("attacker_model", "?").split(":")[0])
started = meta.get("started_at", "?")[:19]
ended = meta.get("ended_at", "—")
ended = ended[:19] if ended else "—"
col4.metric("Started", started)

if meta.get("ended_at"):
    st.caption(f"Completed: `{ended}` | config hash: `{meta.get('config_hash', '?')[:8]}`")
else:
    st.warning("This run has not completed yet (``ended_at`` is missing).")

# ── Load data ─────────────────────────────────────────────────────────────────

with st.spinner("Loading run data…"):
    run_df = load_run(run_dir)
    baseline_df = load_baseline(run_dir)

if run_df.empty:
    st.warning("``run.jsonl`` is present but contains no valid records.")
    st.stop()

# ── Per-category ASR ──────────────────────────────────────────────────────────

st.subheader("Per-Category Attack Success Rate")
cat_df = per_category(run_df)
if not cat_df.empty:
    display_cols = [
        "category", "n_seeds", "asr", "asr_ci_low", "asr_ci_high",
        "mean_queries_to_success", "mean_max_bias_score", "refusal_rate",
    ]
    display_cols = [c for c in display_cols if c in cat_df.columns]
    st.dataframe(
        cat_df[display_cols].style.format({
            "asr": "{:.3f}", "asr_ci_low": "{:.3f}", "asr_ci_high": "{:.3f}",
            "mean_queries_to_success": "{:.1f}", "mean_max_bias_score": "{:.2f}",
            "refusal_rate": "{:.3f}",
        }, na_rep="—"),
        use_container_width=True,
    )

# ── ASR vs. iteration budget chart ───────────────────────────────────────────

st.subheader("ASR vs. Iteration Budget")
asr_iter_df = asr_vs_iter(run_df)
if not asr_iter_df.empty:
    st.altair_chart(asr_vs_iter_chart(asr_iter_df), use_container_width=True)
else:
    st.caption("Not enough data to plot ASR curves.")

# ── Baseline vs. iterative comparison ────────────────────────────────────────

if not baseline_df.empty:
    st.subheader("Baseline vs. Iterative — Gender ASR & Skew")
    from ouroboros.config import FULL_BUDGET, TEST_BUDGET

    _budget = TEST_BUDGET if cfg.get("mode", "test") == "test" else FULL_BUDGET
    bvi = baseline_vs_iterative(
        baseline_df,
        run_df,
        success_n_of_m=_budget.success_n_of_m,
    )
    if bvi:
        b_col1, b_col2, b_col3, b_col4 = st.columns(4)
        b_col1.metric("Baseline ASR", f"{bvi.get('baseline_asr', 0):.3f}")
        b_col2.metric("Iterative ASR", f"{bvi.get('iterative_asr', 0):.3f}")
        b_col3.metric(
            "Iterative mean max skew",
            f"{bvi.get('iterative_mean_max_skew', 0):.2f}",
        )
        b_col4.metric(
            "Mean iters to success",
            f"{bvi.get('iterative_mean_iters_to_success', 0):.1f}",
        )
        st.caption(
            "Paired label-based N-of-M gender ASR (majority perceived gender ≥ N) "
            "and per-batch skew, computed identically on both sides."
        )

# ── Per-seed summary ──────────────────────────────────────────────────────────

st.subheader("Per-Seed Summary")
sum_df = summary_per_seed(run_df)
if not sum_df.empty:
    def _outcome_badge(o: str) -> str:
        return {
            "success": "✅ success", "fail": "❌ fail",
            "refused": "🚫 refused", "judge_error": "⚠️ judge_error",
            "attacker_refused": "🤐 attacker_refused", "error": "💥 error",
        }.get(o, o)

    display = sum_df.copy()
    display["outcome"] = display["outcome"].map(_outcome_badge)
    st.dataframe(display, use_container_width=True)

# ── Judge coverage ────────────────────────────────────────────────────────────

cov_df = judge_coverage(run_df)
if not cov_df.empty:
    with st.expander("Judge Coverage (unclear rate)", expanded=False):
        st.dataframe(cov_df, use_container_width=True)

# ── FairFace KL (from report CSV if available) ────────────────────────────────

ff_csv = run_dir / "report" / "fairface_per_category.csv"
if ff_csv.exists():
    with st.expander("FairFace KL Divergence", expanded=False):
        st.dataframe(pd.read_csv(ff_csv), use_container_width=True)

# ── Strategy clusters ─────────────────────────────────────────────────────────

clusters = read_strategy_clusters(run_dir)
if clusters:
    with st.expander("Strategy Clusters", expanded=False):
        st.dataframe(clusters, use_container_width=True)

# ── Image gallery ─────────────────────────────────────────────────────────────

st.subheader("Image Gallery")

# Prefer report/thumbs if available (top-bias curated), else walk images/ directly
thumbs_dir = run_dir / "report" / "thumbs"
images_dir = run_dir / "images"

gallery_paths: list[Path] = []
if thumbs_dir.exists():
    gallery_paths = sorted(thumbs_dir.glob("*.png"))
elif images_dir.exists():
    gallery_paths = sorted(images_dir.rglob("sample_*.png"))

if gallery_paths:
    # Group by category prefix (from filename when using thumbs)
    st.caption(f"{len(gallery_paths)} image(s) found.")
    cols_per_row = 4
    for i in range(0, min(len(gallery_paths), 24), cols_per_row):
        row_paths = gallery_paths[i:i + cols_per_row]
        row_cols = st.columns(len(row_paths))
        for col, img_path in zip(row_cols, row_paths):
            col.image(str(img_path), use_container_width=True, caption=img_path.name)
    if len(gallery_paths) > 24:
        st.caption(f"Showing first 24 of {len(gallery_paths)} images.")
else:
    st.caption("No images found for this run.")

# ── Generate full report (FairFace + clusters) ────────────────────────────────

st.divider()
with st.expander("⚙️ Generate full report (FairFace KL + clusters)", expanded=False):
    st.caption(
        "Runs ``ouroboros report`` as a subprocess.  "
        "Requires ``pip install -e '.[fairface]'`` and the FairFace weights."
    )
    no_ff = st.checkbox("--no-fairface (skip FairFace)", value=False)
    if st.button("Run report", type="primary"):
        cmd = [sys.executable, "-m", "ouroboros", "report", selected_run_id,
               "--output-dir", str(RESULTS_DIR)]
        if no_ff:
            cmd.append("--no-fairface")
        with st.spinner(f"Running: {' '.join(cmd)}"):
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            st.success("Report generated. Refresh the page to see the updated data.")
        else:
            st.error(f"Report failed (exit {result.returncode}):")
            st.code(result.stderr or result.stdout, language="text")

# ── Static report link ────────────────────────────────────────────────────────

report_html = run_dir / "report" / "report.html"
if report_html.exists():
    st.caption(f"Static report available at: `{report_html}`")
