"""Subprocess launch/stop helpers for ``ouroboros run``.

Does NOT import ``streamlit``, so it is fully unit-testable.

Architecture note: runs are launched as ``ouroboros run ...`` child processes.
This avoids asyncio.run conflicts with Streamlit's event loop and keeps the
heavy model loading (FLUX, Ollama, Gemini) isolated in the child.  Progress
is tracked by tailing the run directory files (``run.jsonl``, ``checkpoint.json``,
``ram.jsonl``, ``meta.json``), which the child writes incrementally.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ouroboros.config import (
    ATTACKER_DEFAULT,
    FULL_BUDGET,
    JUDGE_BACKEND_DEFAULT,
    JUDGE_GEMINI_DEFAULT,
    JUDGE_MLX_DEFAULT,
    JUDGE_OLLAMA_DEFAULT,
    RAM_BUDGET_GB,
    TARGET_BACKEND_DEFAULT,
    RunConfig,
    check_ram_budget,
    config_hash,
)
from ouroboros.web.data import (
    add_job,
    read_jobs,
    read_meta,
    update_job,
    write_jobs,
)

# ---------------------------------------------------------------------------
# Judge model defaults (mirrors _cmd_run in cli.py)
# ---------------------------------------------------------------------------

_JUDGE_DEFAULTS: dict[str, str] = {
    "gemini": JUDGE_GEMINI_DEFAULT,
    "mlx": JUDGE_MLX_DEFAULT,
    "ollama": JUDGE_OLLAMA_DEFAULT,
}


def _derive_max_t2i(mode: str, max_t2i_calls: int | None) -> int:
    """Replicate the default max_t2i_calls derivation in _cmd_run."""
    if max_t2i_calls is not None and max_t2i_calls > 0:
        return max_t2i_calls
    if mode == "full":
        return FULL_BUDGET.m * FULL_BUDGET.max_iter * 175  # 14 000
    return 200  # test mode safety net


# ---------------------------------------------------------------------------
# RunConfig builder
# ---------------------------------------------------------------------------


def build_run_cfg(form: dict) -> RunConfig:
    """Build a ``RunConfig`` from a Launch-form dict.

    Keys mirror the CLI flag names (snake_case).  Defaults match the CLI
    so ``config_hash(cfg)`` agrees with what ``_cmd_run`` would produce.
    """
    mode: str = form.get("mode", "test")
    judge_backend: str = form.get("judge_backend", JUDGE_BACKEND_DEFAULT)
    judge_model_raw: str = form.get("judge_model", "") or ""
    judge_model = judge_model_raw.strip() or _JUDGE_DEFAULTS.get(judge_backend, JUDGE_GEMINI_DEFAULT)
    max_t2i = _derive_max_t2i(mode, form.get("max_t2i_calls") or None)
    flux_size = int(form.get("flux_size", 512))
    return RunConfig(
        mode=mode,
        attacker_model=form.get("attacker_model", ATTACKER_DEFAULT),
        target_backend=form.get("target_backend", TARGET_BACKEND_DEFAULT),
        flux_quantize=int(form.get("flux_quantize", 4)),
        flux_steps=int(form.get("flux_steps", 4)),
        flux_width=flux_size,
        flux_height=flux_size,
        judge_backend=judge_backend,
        judge_model=judge_model,
        max_t2i_calls=max_t2i,
        rate_limit_per_min=int(form.get("rate_limit_per_min", 60)),
        output_dir=str(form.get("output_dir", "results")),
        seeds_filter=form.get("seeds_filter") or None,
        run_baseline=bool(form.get("run_baseline", False)),
        allow_swap=bool(form.get("allow_swap", False)),
        aggressive_unload=not bool(form.get("no_aggressive_unload", False)),
        ollama_host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        google_cloud_project=os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
        google_cloud_location=os.environ.get("GOOGLE_CLOUD_LOCATION", ""),
    )


# ---------------------------------------------------------------------------
# RAM pre-flight
# ---------------------------------------------------------------------------


def preflight_ram(form: dict) -> tuple[bool, str]:
    """Run ``check_ram_budget`` for the given form config.

    Returns ``(ok, message)`` exactly as ``check_ram_budget`` does.
    The diffusers backend lives on VRAM; the system-RAM check is skipped.
    """
    cfg = build_run_cfg(form)
    if cfg.target_backend != "flux":
        return True, "VRAM-based target — system RAM budget check skipped."
    target_key = f"flux2-klein-4b-q{cfg.flux_quantize}"
    return check_ram_budget(
        cfg.attacker_model, target_key, RAM_BUDGET_GB, cfg.aggressive_unload
    )


# ---------------------------------------------------------------------------
# Subprocess entry-point
# ---------------------------------------------------------------------------


def _ouroboros_entrypoint() -> list[str]:
    """Return the command prefix for launching ``ouroboros`` as a subprocess.

    Uses ``python -m ouroboros`` (via ``src/__main__.py``) so that the same
    Python interpreter and environment as the Streamlit app are always used,
    regardless of the shell PATH.
    """
    return [sys.executable, "-m", "ouroboros"]


# ---------------------------------------------------------------------------
# argv builder
# ---------------------------------------------------------------------------


def build_run_argv(form: dict) -> list[str]:
    """Build the full ``ouroboros run ...`` argv from a form dict.

    Only non-default flags are included so the resulting ``RunConfig`` inside
    the subprocess matches the one built by ``build_run_cfg`` exactly
    (same ``config_hash``).
    """
    cfg = build_run_cfg(form)
    argv: list[str] = _ouroboros_entrypoint() + ["run"]

    # Always explicit about mode to avoid ambiguity
    argv += ["--mode", cfg.mode]

    if cfg.attacker_model != ATTACKER_DEFAULT:
        argv += ["--attacker-model", cfg.attacker_model]

    if cfg.target_backend != TARGET_BACKEND_DEFAULT:
        argv += ["--target-backend", cfg.target_backend]

    if cfg.flux_quantize != 4:
        argv += ["--flux-quantize", str(cfg.flux_quantize)]
    if cfg.flux_steps != 4:
        argv += ["--flux-steps", str(cfg.flux_steps)]
    if cfg.flux_width != 512:
        argv += ["--flux-size", str(cfg.flux_width)]

    if cfg.judge_backend != JUDGE_BACKEND_DEFAULT:
        argv += ["--judge-backend", cfg.judge_backend]
    if cfg.judge_model != _JUDGE_DEFAULTS.get(cfg.judge_backend, JUDGE_GEMINI_DEFAULT):
        argv += ["--judge-model", cfg.judge_model]

    default_max_t2i = _derive_max_t2i(cfg.mode, None)
    if cfg.max_t2i_calls != default_max_t2i:
        argv += ["--max-t2i-calls", str(cfg.max_t2i_calls)]

    if cfg.rate_limit_per_min != 60:
        argv += ["--rate-limit", str(cfg.rate_limit_per_min)]

    if cfg.output_dir != "results":
        argv += ["--output-dir", cfg.output_dir]

    if cfg.seeds_filter:
        argv += ["--seeds-filter", cfg.seeds_filter]

    if cfg.run_baseline:
        argv += ["--baseline", "single-shot"]

    if cfg.allow_swap:
        argv += ["--allow-swap"]

    if not cfg.aggressive_unload:
        argv += ["--no-aggressive-unload"]

    if form.get("dry_run"):
        argv += ["--dry-run"]

    if form.get("resume"):
        argv += ["--resume", str(form["resume"])]

    return argv


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------


def launch_run(form: dict, results_dir: Path) -> str:
    """Launch ``ouroboros run`` as a detached subprocess.

    Returns a *pending_id* (``"pending_<pid>"``).  The actual run_id is
    resolved asynchronously via ``resolve_pending_job`` in data.py once the
    subprocess creates the run directory.

    The subprocess stdout+stderr is redirected to a ``web_run_<ts>.log`` file
    in *results_dir* so it doesn't block and can be inspected later.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    argv = build_run_argv(form)
    cfg = build_run_cfg(form)
    expected_hash8 = config_hash(cfg)[:8]

    # Snapshot existing dirs for later resolution
    before_dirs: list[str] = []
    if results_dir.exists():
        before_dirs = [d.name for d in results_dir.iterdir() if d.is_dir()]

    # Per-launch log file
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = results_dir / f"web_run_{ts}.log"
    log_fh = open(log_path, "w", encoding="utf-8")  # noqa: WPS515

    proc = subprocess.Popen(
        argv,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        cwd=str(results_dir.parent),  # CWD = project root so relative paths work
    )
    log_fh.close()  # the child inherited the fd; we can close our handle

    pending_id = f"pending_{proc.pid}"
    job: dict[str, Any] = {
        "pending_id": pending_id,
        "run_id": None,
        "pid": proc.pid,
        "argv": [str(a) for a in argv],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "starting",
        "output_dir": str(results_dir),
        "expected_hash8": expected_hash8,
        "before_dirs": before_dirs,
        "log_path": str(log_path),
        "dry_run": bool(form.get("dry_run")),
    }
    add_job(results_dir, job)
    return pending_id


# ---------------------------------------------------------------------------
# Stop / resume
# ---------------------------------------------------------------------------


def stop_run(pending_id: str, results_dir: Path) -> bool:
    """Send SIGTERM to the process registered under *pending_id*.

    Returns True if the signal was delivered, False otherwise.
    The run is resumable via ``ouroboros run --resume <run_id>`` because
    the loop writes a checkpoint after every seed.
    """
    jobs = read_jobs(results_dir)
    for j in jobs:
        if j.get("pending_id") != pending_id:
            continue
        pid = j.get("pid")
        if pid and _pid_alive(pid):
            try:
                os.kill(pid, signal.SIGTERM)
                j["status"] = "stopped"
                write_jobs(results_dir, jobs)
                return True
            except Exception:
                pass
        j["status"] = "stopped"
        write_jobs(results_dir, jobs)
        return False
    return False


# ---------------------------------------------------------------------------
# Status reconciliation
# ---------------------------------------------------------------------------


def reconcile_job(job: dict) -> str:
    """Derive the current status of a job from pid liveness and meta.json.

    Returns one of: ``"starting"``, ``"running"``, ``"finished"``,
    ``"stopped"``, ``"error"``, ``"dry_run_done"``.
    """
    run_id = job.get("run_id")
    output_dir = job.get("output_dir", "results")
    results_dir = Path(output_dir)

    if run_id:
        run_dir = results_dir / run_id
        meta = read_meta(run_dir)
        if meta and meta.get("ended_at"):
            if job.get("dry_run"):
                return "dry_run_done"
            return "finished"

    pid = job.get("pid")
    if pid and _pid_alive(pid):
        return job.get("status", "running")

    # Process is dead — always return a terminal status (never the stale "starting"/"running")
    if run_id:
        run_dir = results_dir / run_id
        if (run_dir / "run.jsonl").exists():
            return "error"  # died without writing ended_at
    return "stopped"


def reconcile_all_jobs(results_dir: Path) -> list[dict]:
    """Reconcile every job's status against pid liveness and meta.json.

    Reads ``web_jobs.json``, updates any stale "running"/"starting" entries
    whose process has died, writes back if anything changed, and returns the
    up-to-date job list.  Call this before checking ``get_running_jobs`` so
    the guard on the Launch page never blocks on a dead process.
    """
    jobs = read_jobs(results_dir)
    changed = False
    for j in jobs:
        if j.get("status") not in ("running", "starting"):
            continue
        new_status = reconcile_job(j)
        if new_status != j.get("status"):
            j["status"] = new_status
            changed = True
    if changed:
        write_jobs(results_dir, jobs)
    return jobs


def _pid_alive(pid: int) -> bool:
    """Return True if the process *pid* is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False
