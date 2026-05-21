from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ouroboros.config import RunConfig, config_hash


def _run_id_from_cfg(cfg: RunConfig) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    short = config_hash(cfg)[:8]
    return f"{ts}_{short}"


def make_run_dir(base: str | Path, cfg: RunConfig, run_id: str | None = None) -> tuple[Path, str]:
    """Create the run directory tree and return (run_dir, run_id)."""
    if run_id is None:
        run_id = _run_id_from_cfg(cfg)
    run_dir = Path(base) / run_id
    (run_dir / "images").mkdir(parents=True, exist_ok=True)
    return run_dir, run_id


def save_image(run_dir: Path, seed_id: str, iter_idx: int, sample_idx: int, png_bytes: bytes) -> str:
    """Save PNG bytes and return the relative path string."""
    img_dir = run_dir / "images" / seed_id / f"iter_{iter_idx:02d}"
    img_dir.mkdir(parents=True, exist_ok=True)
    rel_path = f"images/{seed_id}/iter_{iter_idx:02d}/sample_{sample_idx}.png"
    (run_dir / rel_path).write_bytes(png_bytes)
    return rel_path


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class JSONLWriter:
    """Append-only JSONL writer with periodic fsync."""

    def __init__(self, path: Path, fsync_every: int = 5) -> None:
        self._path = path
        self._fsync_every = fsync_every
        self._count = 0

    def append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line)
            self._count += 1
            if self._count % self._fsync_every == 0:
                f.flush()
                os.fsync(f.fileno())


def write_meta(
    run_dir: Path,
    run_id: str,
    cfg: RunConfig,
    attacker_model: str,
    judge_model: str,
    judge_backend: str,
    started_at: str,
    ended_at: str | None = None,
) -> None:
    from dataclasses import asdict
    meta = {
        "run_id": run_id,
        "config": asdict(cfg),
        "config_hash": config_hash(cfg),
        "attacker_model": attacker_model,
        "judge_model": judge_model,
        "judge_backend": judge_backend,
        "started_at": started_at,
        "ended_at": ended_at,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def update_meta_ended(run_dir: Path, ended_at: str) -> None:
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["ended_at"] = ended_at
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def write_checkpoint(
    run_dir: Path,
    run_id: str,
    cfg: RunConfig,
    completed_seed_ids: list[str],
    t2i_calls_used: int,
) -> None:
    ckpt = {
        "run_id": run_id,
        "completed_seed_ids": completed_seed_ids,
        "t2i_calls_used": t2i_calls_used,
        "config_hash": config_hash(cfg),
    }
    (run_dir / "checkpoint.json").write_text(json.dumps(ckpt, indent=2), encoding="utf-8")


def load_checkpoint(run_dir: Path) -> dict | None:
    ckpt_path = run_dir / "checkpoint.json"
    if not ckpt_path.exists():
        return None
    return json.loads(ckpt_path.read_text(encoding="utf-8"))
