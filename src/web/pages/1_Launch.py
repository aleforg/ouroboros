"""Launch page — configure and start an ``ouroboros run``.

Pre-flight: RAM budget is checked before the launch button is enabled.
Seeds are previewed so the user can verify the filter before committing.
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from ouroboros.config import (
    ATTACKER_DEFAULT,
    JUDGE_BACKEND_DEFAULT,
    JUDGE_MLX_DEFAULT,
    JUDGE_OLLAMA_DEFAULT,
    TEST_BUDGET,
    FULL_BUDGET,
)
from ouroboros.seeds import ALLOWED_CATEGORIES, load_full_seeds, load_test_seeds
from ouroboros.web.data import get_results_dir
from ouroboros.web.runner import build_run_cfg, launch_run, preflight_ram, reconcile_all_jobs

RESULTS_DIR = get_results_dir()

st.title("⚡ Launch Run")

# ── Guard: only one run at a time on a 16 GB Mac ──────────────────────────────
# Reconcile first so dead processes don't block the form indefinitely.
_all_jobs = reconcile_all_jobs(RESULTS_DIR)
running = [j for j in _all_jobs if j.get("status") in ("running", "starting")]
if running:
    active = running[0]
    label = active.get("run_id") or active.get("pending_id", "?")
    st.warning(
        f"A run is already in progress: **`{label}`**  \n"
        "Stop it on the **Monitor** page before launching a new one."
    )
    st.stop()

# ── Configuration form ────────────────────────────────────────────────────────

with st.form("launch_form"):
    st.subheader("Mode & Seeds")
    col_a, col_b = st.columns(2)
    with col_a:
        mode = st.selectbox(
            "Mode",
            ["test", "full"],
            help="test: 10 seeds, M=2, max_iter=5 | full: 175 Stable Bias seeds, M=4, max_iter=20",
        )
    with col_b:
        seeds_filter = st.text_input(
            "Seeds filter (category)",
            placeholder="e.g. gender   — leave blank for all",
            help="Restrict to a single seed category. "
                 f"Known categories: {', '.join(sorted(ALLOWED_CATEGORIES))}",
        )

    run_baseline = st.checkbox(
        "Run single-shot baseline (no attacker)",
        help="Generates baseline images with no adversarial rewriting before the PAIR loop.",
    )

    st.subheader("Judge")
    col_c, col_d = st.columns(2)
    with col_c:
        judge_backend = st.selectbox(
            "Judge backend",
            ["mlx", "ollama"],
            index=0,
            help="mlx: offline Qwen3-VL-8B | ollama: offline Qwen3-VL-8B",
        )
    with col_d:
        default_judge_model = {
            "mlx": JUDGE_MLX_DEFAULT,
            "ollama": JUDGE_OLLAMA_DEFAULT,
        }.get(judge_backend, JUDGE_MLX_DEFAULT)
        judge_model = st.text_input(
            "Judge model override",
            value="",
            placeholder=default_judge_model,
            help=f"Leave blank to use the default for the selected backend ({default_judge_model}).",
        )

    # The dashboard launches the mflux backend only; the CUDA backends
    # (diffusers, qwen-image) are CLI-only via --target-backend.
    st.subheader("Target — FLUX")
    col_e, col_f, col_g = st.columns(3)
    with col_e:
        target_quantize = st.selectbox(
            "Quantization bits", [3, 4, 5, 6, 8, 16], index=1,
            help="Q4 ≈ 3.5 GB | Q8 ≈ 6.5 GB | 16 (bf16) ≈ 11 GB"
        )
    with col_f:
        target_steps = st.number_input(
            "Inference steps", min_value=1, max_value=50, value=4, step=1,
            help="4 is recommended for FLUX.2-klein distilled",
        )
    with col_g:
        target_size = st.number_input(
            "Image size (px)", min_value=256, max_value=1024, value=512, step=64,
            help="Applied to both width and height",
        )

    st.subheader("Advanced")
    col_h, col_i = st.columns(2)
    with col_h:
        attacker_model = st.text_input("Attacker model", value=ATTACKER_DEFAULT)
        rate_limit = st.number_input(
            "Rate limit (req/min)", min_value=1, max_value=600, value=60,
            help="Throttle for remote APIs. Currently inert: attacker, target "
                 "and judge all run locally, so nothing consumes it — but the "
                 "value still enters config_hash and changes the run_id.",
        )
    with col_i:
        max_t2i_calls = st.number_input(
            "Max T2I calls (0 = mode default)", min_value=0, value=0,
            help="Hard cap on total T2I API calls. 0 means auto: 200 (test) or 14 000 (full).",
        )
        allow_swap = st.checkbox(
            "Allow swap (override RAM limit)",
            help="Launch even when the estimated RAM usage exceeds the 13 GB budget.",
        )
        no_aggressive_unload = st.checkbox(
            "No aggressive unload",
            help="Keep attacker and target loaded simultaneously — may OOM on 16 GB.",
        )

    st.subheader("Options")
    col_j, col_k = st.columns(2)
    with col_j:
        dry_run = st.checkbox(
            "Dry run",
            help="List seeds and create the run directory without calling any API. "
                 "Useful to verify configuration.",
        )
    with col_k:
        resume_id = st.text_input(
            "Resume run_id (optional)",
            placeholder="e.g. 2026-05-16_083457_20dad29b",
            help="Pick up an interrupted run from its last checkpoint.",
        )

    submitted = st.form_submit_button("🚀 Launch", type="primary")

# ── Pre-flight + launch ───────────────────────────────────────────────────────

if submitted:
    form: dict = {
        "mode": mode,
        "seeds_filter": seeds_filter.strip() or None,
        "run_baseline": run_baseline,
        "judge_backend": judge_backend,
        "judge_model": judge_model.strip() or None,
        "target_quantize": target_quantize,
        "target_steps": target_steps,
        "target_size": target_size,
        "attacker_model": attacker_model,
        "rate_limit_per_min": rate_limit,
        "max_t2i_calls": int(max_t2i_calls) if max_t2i_calls > 0 else None,
        "allow_swap": allow_swap,
        "no_aggressive_unload": no_aggressive_unload,
        "dry_run": dry_run,
        "resume": resume_id.strip() or None,
        "output_dir": str(RESULTS_DIR),
    }

    # ── RAM pre-flight ──
    ram_ok, ram_msg = preflight_ram(form)
    if ram_msg:
        if ram_ok:
            st.warning(f"⚠️ RAM estimate: {ram_msg}")
        else:
            if allow_swap:
                st.warning(f"⚠️ RAM limit exceeded but --allow-swap is set: {ram_msg}")
            else:
                st.error(f"❌ RAM budget exceeded: {ram_msg}")
                st.error("Enable **Allow swap** to continue anyway.")
                st.stop()

    # ── Seed preview ──
    cfg = build_run_cfg(form)
    with st.expander("Seed preview", expanded=dry_run):
        try:
            seeds = load_test_seeds() if cfg.mode == "test" else load_full_seeds()
            if cfg.seeds_filter:
                seeds = [s for s in seeds if s.category == cfg.seeds_filter]
            if seeds:
                st.caption(f"{len(seeds)} seed(s) will be used:")
                rows = [
                    {"seed_id": s.seed_id, "category": s.category, "scene": s.base_scene}
                    for s in seeds[:20]
                ]
                st.dataframe(rows, use_container_width=True)
                if len(seeds) > 20:
                    st.caption(f"... and {len(seeds) - 20} more.")
            else:
                st.warning("No seeds matched the filter. Check the category name.")
        except FileNotFoundError as e:
            st.warning(f"Could not load full seeds: {e}")

    # ── Launch ──
    with st.spinner("Launching subprocess…"):
        try:
            pending_id = launch_run(form, RESULTS_DIR)
        except Exception as exc:
            st.error(f"Failed to launch: {exc}")
            st.stop()

    st.session_state["active_pending_id"] = pending_id
    mode_label = "dry run" if dry_run else "run"
    st.success(
        f"✅ Launched {mode_label} (pending_id=`{pending_id}`).  \n"
        "Switching to **Monitor** page …"
    )
    st.switch_page("pages/2_Monitor.py")
