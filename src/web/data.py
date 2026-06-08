"""Pure (no-Streamlit) helpers for reading run data and managing the job registry.

All functions operate on the filesystem only and do not import ``streamlit``,
so they are unit-testable without a running server.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Results directory resolution
# ---------------------------------------------------------------------------


def get_results_dir() -> Path:
    """Return the results directory, defaulting to ``results/`` next to CWD.

    Can be overridden by the ``OUROBOROS_RESULTS_DIR`` environment variable,
    which ``_cmd_dashboard`` in cli.py sets to the absolute path before launching
    the Streamlit subprocess.
    """
    return Path(os.environ.get("OUROBOROS_RESULTS_DIR", "results"))


# ---------------------------------------------------------------------------
# Run directory readers
# ---------------------------------------------------------------------------


def list_runs(results_dir: str | Path) -> list[dict[str, Any]]:
    """Return metadata for all run directories, sorted newest-first.

    Each entry: ``{run_id, run_dir, meta, has_report, has_images, n_records}``.
    Directories without ``meta.json`` are skipped (e.g. web_jobs.json, logs).
    """
    base = Path(results_dir)
    if not base.exists():
        return []
    entries: list[tuple[float, dict[str, Any]]] = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        meta = _safe_json(meta_path)
        mtime = d.stat().st_mtime
        n_records = _count_jsonl_lines(d / "run.jsonl")
        entries.append((mtime, {
            "run_id": d.name,
            "run_dir": d,
            "meta": meta,
            "has_report": (d / "report" / "report.html").exists(),
            "has_images": (d / "images").exists() and any((d / "images").iterdir()),
            "n_records": n_records,
        }))
    entries.sort(key=lambda t: t[0], reverse=True)
    return [e for _, e in entries]


def read_meta(run_dir: Path) -> dict | None:
    """Read ``meta.json`` for a run directory, returning None if absent."""
    return _safe_json(run_dir / "meta.json")


def read_checkpoint(run_dir: Path) -> dict | None:
    """Read ``checkpoint.json``, returning None if absent.

    Delegates to ``ouroboros.storage.load_checkpoint`` for consistency.
    """
    from ouroboros.storage import load_checkpoint
    return load_checkpoint(run_dir)


def tail_run_jsonl(run_dir: Path, n: int = 20) -> list[dict]:
    """Return the last *n* records from ``run.jsonl`` (oldest-to-newest).

    Silently skips malformed lines.
    """
    path = run_dir / "run.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    records: list[dict] = []
    for line in lines[-n:]:
        parsed = _safe_json_str(line)
        if parsed is not None:
            records.append(parsed)
    return records


def read_baseline_jsonl(run_dir: Path) -> list[dict]:
    """Return all records from ``baseline.jsonl``."""
    path = run_dir / "baseline.jsonl"
    if not path.exists():
        return []
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        parsed = _safe_json_str(line)
        if parsed is not None:
            records.append(parsed)
    return records


def latest_ram(run_dir: Path) -> dict | None:
    """Return the most-recent record from ``ram.jsonl``, or None."""
    path = run_dir / "ram.jsonl"
    if not path.exists():
        return None
    for line in reversed(path.read_text(encoding="utf-8").strip().splitlines()):
        parsed = _safe_json_str(line)
        if parsed is not None:
            return parsed
    return None


def read_live(run_dir: Path) -> dict | None:
    """Read run_dir/live.json — the current intra-iteration state, or None.

    Written atomically by ``loop._write_live``; returns None if the file
    doesn't exist or is not valid JSON (e.g. mid-write race condition).
    """
    return _safe_json(run_dir / "live.json")


def read_strategy_clusters(run_dir: Path) -> list[dict]:
    """Return strategy cluster records from ``report/strategy_clusters.json``."""
    p = run_dir / "report" / "strategy_clusters.json"
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def read_web_log(run_dir: Path, tail_lines: int = 50) -> str:
    """Return the tail of the web.log file for this run, if present."""
    # web_run_*.log is written to results_dir, not run_dir. We look for the
    # most-recent log whose name contains the run_id hash suffix.
    results_dir = run_dir.parent
    hash8 = run_dir.name.split("_")[-1] if "_" in run_dir.name else ""
    candidates = sorted(results_dir.glob("web_run_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    log_path: Path | None = None
    if hash8:
        for c in candidates:
            # match by proximity in time — just pick the most recent
            log_path = c
            break
    if log_path is None and candidates:
        log_path = candidates[0]
    if log_path is None or not log_path.exists():
        return ""
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-tail_lines:])


# ---------------------------------------------------------------------------
# Job registry (on-disk, survives Streamlit reruns and browser refreshes)
# ---------------------------------------------------------------------------

_JOBS_FILE = "web_jobs.json"


def _jobs_path(results_dir: Path) -> Path:
    return results_dir / _JOBS_FILE


def read_jobs(results_dir: Path) -> list[dict]:
    """Return all registered jobs from ``results_dir/web_jobs.json``."""
    p = _jobs_path(results_dir)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def write_jobs(results_dir: Path, jobs: list[dict]) -> None:
    """Overwrite ``web_jobs.json`` with the given list."""
    results_dir.mkdir(parents=True, exist_ok=True)
    _jobs_path(results_dir).write_text(json.dumps(jobs, indent=2), encoding="utf-8")


def add_job(results_dir: Path, job: dict) -> None:
    """Append a new job to the registry."""
    jobs = read_jobs(results_dir)
    jobs.append(job)
    write_jobs(results_dir, jobs)


def update_job(results_dir: Path, pending_id: str, updates: dict) -> None:
    """Apply *updates* dict to the job identified by *pending_id*."""
    jobs = read_jobs(results_dir)
    for j in jobs:
        if j.get("pending_id") == pending_id:
            j.update(updates)
    write_jobs(results_dir, jobs)


def get_job(results_dir: Path, pending_id: str) -> dict | None:
    """Return the job with *pending_id*, or None."""
    for j in read_jobs(results_dir):
        if j.get("pending_id") == pending_id:
            return j
    return None


def get_running_jobs(results_dir: Path) -> list[dict]:
    """Return all jobs whose status is 'running' or 'starting'."""
    return [j for j in read_jobs(results_dir) if j.get("status") in ("running", "starting")]


def resolve_pending_job(results_dir: Path, job: dict) -> dict | None:
    """Try to resolve the run_id for a 'starting' job by scanning results_dir.

    Returns the updated job dict if resolved, or None if not yet found.
    The run dir is identified as the newest directory whose name ends with
    ``_<expected_hash8>`` that appeared after the snapshot stored in *job*.
    """
    if job.get("run_id"):
        return job  # already resolved
    expected_hash8 = job.get("expected_hash8", "")
    before: set[str] = set(job.get("before_dirs", []))
    if not results_dir.exists():
        return None
    for d in results_dir.iterdir():
        if not d.is_dir():
            continue
        if d.name in before:
            continue
        if expected_hash8 and not d.name.endswith(f"_{expected_hash8}"):
            continue
        if (d / "meta.json").exists():
            return {**job, "run_id": d.name, "status": "running"}
    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_json_str(s: str) -> dict | None:
    try:
        return json.loads(s)
    except Exception:
        return None


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return sum(1 for line in path.open(encoding="utf-8") if line.strip())
    except Exception:
        return 0
