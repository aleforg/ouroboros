"""Monitor page — live view of an in-progress or recently completed run.

A ``@st.fragment(run_every="2s")`` block re-reads the run directory files
(``checkpoint.json``, ``run.jsonl``, ``ram.jsonl``, ``meta.json``) every two
seconds so the UI stays current without a full-page rerun.

When the run completes (``meta.ended_at`` is set), the fragment triggers a
full-page rerun which removes the auto-refresh and shows the completed state.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from ouroboros.config import RAM_BUDGET_GB
from ouroboros.web.charts import ram_summary
from ouroboros.web.data import (
    get_results_dir,
    get_job,
    get_running_jobs,
    latest_ram,
    list_runs,
    read_checkpoint,
    read_live,
    read_meta,
    resolve_pending_job,
    tail_run_jsonl,
    update_job,
    read_web_log,
)
from ouroboros.web.runner import reconcile_job, stop_run

RESULTS_DIR = get_results_dir()

st.title("📡 Live Monitor")

# ── Run selector ──────────────────────────────────────────────────────────────

# Priority 1: active run set by the Launch page
pending_id: str | None = st.session_state.get("active_pending_id")
job: dict | None = None

if pending_id:
    job = get_job(RESULTS_DIR, pending_id)
    # Try to resolve the run_id if still "starting"
    if job and not job.get("run_id"):
        resolved = resolve_pending_job(RESULTS_DIR, job)
        if resolved and resolved.get("run_id"):
            update_job(RESULTS_DIR, pending_id, {"run_id": resolved["run_id"], "status": "running"})
            job = get_job(RESULTS_DIR, pending_id)

# Priority 2: any running job
if job is None:
    running = get_running_jobs(RESULTS_DIR)
    if running:
        job = running[0]
        pending_id = job.get("pending_id")

# Priority 3: let the user pick from all runs
all_runs = list_runs(RESULTS_DIR)
run_ids = [r["run_id"] for r in all_runs]

if not run_ids:
    st.info("No runs found. Go to **Launch** to start one.")
    st.stop()

current_run_id = job.get("run_id") if job else None
try:
    default_idx = run_ids.index(current_run_id) if current_run_id in run_ids else 0
except ValueError:
    default_idx = 0

selected_run_id = st.selectbox(
    "Select run",
    run_ids,
    index=default_idx,
    format_func=lambda rid: f"{'🟢 ' if job and job.get('run_id') == rid else ''}{rid}",
)

run_dir = RESULTS_DIR / selected_run_id

# ── Status badge ──────────────────────────────────────────────────────────────

meta = read_meta(run_dir)
if meta is None:
    st.warning("Run directory exists but ``meta.json`` is missing — may still be starting.")
    meta = {}

is_finished = bool(meta.get("ended_at"))
is_running = (job is not None and job.get("run_id") == selected_run_id) and not is_finished

status_color = "🟢" if is_running else ("✅" if is_finished else "⏸️")
st.caption(
    f"{status_color}  **{selected_run_id}**  "
    f"| mode: `{meta.get('config', {}).get('mode', '?')}`  "
    f"| judge: `{meta.get('judge_backend', '?')}/{meta.get('judge_model', '?')}`  "
    f"| started: `{meta.get('started_at', '?')[:19]}`"
    + (f"  | ended: `{meta.get('ended_at', '')[:19]}`" if is_finished else "")
)

# ── Stop / Resume buttons ─────────────────────────────────────────────────────

if is_running and pending_id:
    if st.button("⏹️ Stop run", type="secondary"):
        stop_run(pending_id, RESULTS_DIR)
        st.success("Stop signal sent. The run can be resumed later with `--resume`.")
        st.session_state.pop("active_pending_id", None)
        st.rerun()

if not is_running and not is_finished:
    ckpt = read_checkpoint(run_dir)
    if ckpt and ckpt.get("completed_seed_ids"):
        st.info(
            f"This run was interrupted with {len(ckpt['completed_seed_ids'])} seeds done.  \n"
            f"Resume it from the **Launch** page using **Resume run_id** = `{selected_run_id}`."
        )

# ── Config summary ────────────────────────────────────────────────────────────

with st.expander("Run config", expanded=False):
    cfg_dict = meta.get("config", {})
    st.json(cfg_dict)

# ── Live block (auto-refreshes every 2s while running) ───────────────────────

_PHASE_BADGE: dict[str, str] = {
    "attacking":        "🧠 Attacking",
    "generating":       "🎨 Generating",
    "judging":          "🔍 Judging",
    "iter_done":        "✅ Iter done",
    "attacker_refused": "🤐 Attacker refused",
    "refused":          "🚫 All refused",
    "judge_error":      "⚠️ Judge error",
    "finished":         "🏁 Finished",
}

_OUTCOME_BADGE: dict[str, str] = {
    "success": "✅ success",
    "fail":    "❌ fail",
}

def _render_current_iteration(run_dir: Path, meta: dict) -> None:
    """Render the current iteration as a 4-tile grid: Status · Attacker · Target · Judge."""
    if meta.get("ended_at"):
        return

    live = read_live(run_dir)

    phase = live.get("phase", "?") if live else None
    seed_id = live.get("seed_id", "?") if live else "?"
    category = live.get("category", "?") if live else "?"
    iter_idx = live.get("iter", "?") if live else "?"
    max_iter = live.get("max_iter", "?") if live else "?"
    strategy_label = (live.get("strategy_label") or "—") if live else "—"
    attacker_rationale = (live.get("attacker_rationale") or "") if live else ""
    target_prompt = (live.get("target_prompt") or "") if live else ""
    sample_paths = live.get("samples", []) if live else []

    # Last completed iteration from run.jsonl (stable — not subject to the
    # millisecond timing race between iter_done and the next attacking write)
    last_records = tail_run_jsonl(run_dir, n=1)
    last = last_records[0] if last_records else None
    last_judge = (last.get("judge") or {}) if last else {}
    last_bias = last_judge.get("bias_score")
    last_split = last_judge.get("per_image_genders")
    last_rationale = last_judge.get("rationale") or ""
    last_outcome = last.get("outcome", "?") if last else None
    last_strategy = last.get("strategy_label", "—") if last else "—"

    # If we caught iter_done in live.json, prefer that (fresher than run.jsonl)
    if phase == "iter_done":
        last_bias = live.get("bias_score", last_bias)
        last_split = live.get("gender_split") or last_split
        last_rationale = live.get("judge_rationale") or last_rationale
        last_outcome = live.get("outcome", last_outcome)
        last_strategy = strategy_label

    st.subheader("Current iteration")
    tile_status, tile_attacker, tile_target, tile_judge = st.columns(4)

    # ── Tile 1: Status ───────────────────────────────────────────────────────
    with tile_status:
        with st.container(border=True):
            st.markdown("**🔄 Status**")
            badge = _PHASE_BADGE.get(phase, f"⚙️ {phase}") if phase else "⏳ Starting…"
            st.markdown(badge)
            st.caption(f"seed: `{seed_id}`")
            st.caption(f"category: `{category}`")
            st.caption(f"iter: {iter_idx} / {max_iter}")
            _STEP_PHASES = ("attacking", "generating", "judging", "iter_done")
            phase_idx = _STEP_PHASES.index(phase) if phase in _STEP_PHASES else -1
            for i, (icon, lbl) in enumerate([("🧠", "Attacker"), ("🎨", "Target"), ("🔍", "Judge")]):
                if phase in ("attacker_refused", "refused", "judge_error"):
                    marker = "✗"
                elif phase == "iter_done" or i < phase_idx:
                    marker = "✓"
                elif i == min(phase_idx, 2):
                    marker = "▶"
                else:
                    marker = "·"
                st.caption(f"{marker} {icon} {lbl}")

    # ── Tile 2: Attacker ─────────────────────────────────────────────────────
    with tile_attacker:
        with st.container(border=True):
            st.markdown("**🧠 Attacker**")
            if phase in ("attacking", None):
                st.caption("⏳ Thinking…")
            else:
                st.markdown(f"*{strategy_label}*")
            if attacker_rationale:
                st.caption(attacker_rationale[:300] + ("…" if len(attacker_rationale) > 300 else ""))
            if target_prompt:
                with st.expander("Prompt", expanded=False):
                    st.code(target_prompt, language=None)

    # ── Tile 3: Target ───────────────────────────────────────────────────────
    with tile_target:
        with st.container(border=True):
            st.markdown("**🎨 Target**")
            existing_imgs = [p for p in sample_paths if (run_dir / p).exists()]
            if existing_imgs:
                for rel_path in existing_imgs[:2]:
                    st.image(str(run_dir / rel_path), use_container_width=True)
                if len(existing_imgs) > 2:
                    st.caption(f"+{len(existing_imgs) - 2} more")
            elif phase == "generating":
                m = live.get("m", "?") if live else "?"
                st.caption(f"🎨 Generating {m} image(s)…")
            else:
                st.caption("Images will appear here once generated.")

    # ── Tile 4: Judge ────────────────────────────────────────────────────────
    # "judgment_pending": a new iteration is running but its result isn't in
    # run.jsonl yet.  Detected by comparing current live iter vs last completed.
    current_iter = live.get("iter") if live else None
    last_completed_iter = last.get("iter") if last else None
    judgment_pending = (
        phase not in ("iter_done", "finished", None)
        and not (
            phase == "iter_done"
            or (current_iter is not None
                and last_completed_iter is not None
                and current_iter <= last_completed_iter)
        )
    )

    with tile_judge:
        with st.container(border=True):
            st.markdown("**🔍 Judge**")
            if phase == "judging":
                st.markdown("🔍 **Analyzing images…**")
            elif phase in ("attacking", "generating") and judgment_pending:
                st.caption(f"⏳ iter {current_iter} in progress…")

            if last is None:
                st.caption("⏳ Waiting for first result…")
            else:
                outcome_label = _OUTCOME_BADGE.get(last_outcome or "", last_outcome or "?")
                iter_label = last.get("iter", "?") if last else "?"
                label_prefix = "Previous: " if judgment_pending else ""
                st.caption(f"{label_prefix}iter {iter_label} · {outcome_label}")
                if last_bias is not None:
                    st.metric("gender skew (0–10)", last_bias)
                if last_split:
                    if isinstance(last_split, list):
                        from ouroboros.judge import gender_counts
                        n_f, n_m, n_u = gender_counts([str(x) for x in last_split])
                        last_split = f"{n_m}M/{n_f}F/{n_u}U"
                    st.caption(f"♀♂ gender split: {last_split}")
                if last_rationale:
                    with st.expander("Rationale", expanded=False):
                        st.write(last_rationale)
                if last_strategy and last_strategy != "—":
                    st.caption(f"strategy: *{last_strategy[:60]}*")

    st.divider()


def _render_progress(run_dir: Path, meta: dict) -> bool:
    """Render progress bars, latest iterations, images, and RAM.

    Returns True if the run is still in progress.
    """
    # Show the live intra-iteration panel at the top while the run is active.
    _render_current_iteration(run_dir, meta)

    ckpt = read_checkpoint(run_dir)
    cfg = meta.get("config", {})
    mode = cfg.get("mode", "test")
    seeds_filter = cfg.get("seeds_filter")
    max_calls = cfg.get("max_t2i_calls", 200)

    # Seed progress
    completed_seeds: list[str] = ckpt.get("completed_seed_ids", []) if ckpt else []
    calls_used: int = ckpt.get("t2i_calls_used", 0) if ckpt else 0
    # Estimate total seeds
    total_seeds: int | None = 175 if mode == "full" else 10
    if seeds_filter:
        total_seeds = None  # unknown without loading seeds

    col1, col2, col3 = st.columns(3)
    if total_seeds:
        col1.progress(
            min(len(completed_seeds) / total_seeds, 1.0),
            text=f"Seeds: {len(completed_seeds)} / {total_seeds}",
        )
    else:
        col1.metric("Seeds done", len(completed_seeds))
    col2.progress(
        min(calls_used / max(max_calls, 1), 1.0),
        text=f"T2I calls: {calls_used} / {max_calls}",
    )

    # RAM
    ram_rec = latest_ram(run_dir)
    if ram_rec:
        rsumm = ram_summary(ram_rec)
        rss = rsumm.get("rss_gb", 0.0)
        avail = rsumm.get("available_gb", 0.0)
        col3.metric(
            "Process RSS",
            f"{rss:.1f} GB",
            delta=f"avail: {avail:.1f} GB",
            delta_color="normal",
        )

    st.divider()

    # Latest iterations table
    records = tail_run_jsonl(run_dir, n=10)
    if records:
        st.subheader("Latest iterations")
        rows = []
        for rec in reversed(records):
            outcome = rec.get("outcome", "?")
            badge = {"success": "✅", "fail": "❌", "refused": "🚫", "judge_error": "⚠️",
                     "attacker_refused": "🤐", "error": "💥"}.get(outcome, outcome)
            rows.append({
                "seed": rec.get("seed_id", "?"),
                "iter": rec.get("iter", "?"),
                "strategy": rec.get("strategy_label", "—")[:40],
                "outcome": badge,
                "bias_score": (rec.get("judge") or {}).get("bias_score", "—"),
                "calls": rec.get("t2i_calls_used_so_far", "—"),
            })
        st.dataframe(rows, use_container_width=True)

        # Latest images (from the most recent record with images)
        for rec in reversed(records):
            imgs = [
                run_dir / s["path"]
                for s in rec.get("samples", [])
                if s.get("outcome") == "image" and (run_dir / s["path"]).exists()
            ]
            if imgs:
                st.subheader(
                    f"Latest images — `{rec.get('seed_id', '?')}` iter {rec.get('iter', '?')}"
                )
                img_cols = st.columns(min(len(imgs), 4))
                for col, img_path in zip(img_cols, imgs):
                    col.image(str(img_path), use_container_width=True)
                break  # show images from the most recent record only
    else:
        st.info("Waiting for the first iteration to complete…")

    return not bool(read_meta(run_dir).get("ended_at") if read_meta(run_dir) else False)


if is_running:
    @st.fragment(run_every="2s")
    def live_block() -> None:
        current_meta = read_meta(run_dir) or {}
        if current_meta.get("ended_at"):
            st.success(f"✅ Run finished at `{current_meta['ended_at']}`")
            st.balloons()
            st.session_state.pop("active_pending_id", None)
            st.rerun()  # full page rerun → exits "is_running" branch
            return
        _render_progress(run_dir, current_meta)

    live_block()
else:
    if is_finished:
        st.success(f"✅ Run completed at `{meta.get('ended_at', '?')[:19]}`")
    _render_progress(run_dir, meta)
    if is_finished:
        st.info("Go to the **Results** page to view the full report.")

# ── Subprocess log ────────────────────────────────────────────────────────────

with st.expander("Subprocess log (last 50 lines)", expanded=False):
    log_text = read_web_log(run_dir)
    if log_text:
        st.code(log_text, language="text")
    else:
        st.caption("No log file found for this run.")
