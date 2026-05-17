from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import psutil

logger = logging.getLogger(__name__)

_Phase = Literal[
    "pre_attacker",
    "post_attacker",
    "pre_target",
    "post_target",
    "post_judge",
]


@dataclass
class RamSnapshot:
    iter_id: int
    seed_id: str
    phase: str
    process_rss_gb: float
    system_used_gb: float
    system_available_gb: float
    timestamp: str


def process_rss_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024**3


def system_available_gb() -> float:
    return psutil.virtual_memory().available / 1024**3


def system_used_gb() -> float:
    return psutil.virtual_memory().used / 1024**3


class RamMonitor:
    """Non-blocking RAM snapshot logger. Writes to <run_dir>/ram.jsonl after each seed."""

    def __init__(self, run_dir: Path, warn_threshold_gb: float = 13.0) -> None:
        self._path = run_dir / "ram.jsonl"
        self._warn_threshold = warn_threshold_gb
        self._snapshots: list[RamSnapshot] = []

    def snap(self, iter_id: int, seed_id: str, phase: str) -> RamSnapshot:
        s = RamSnapshot(
            iter_id=iter_id,
            seed_id=seed_id,
            phase=phase,
            process_rss_gb=round(process_rss_gb(), 3),
            system_used_gb=round(system_used_gb(), 3),
            system_available_gb=round(system_available_gb(), 3),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._snapshots.append(s)
        if s.system_used_gb > self._warn_threshold:
            logger.warning(
                "RAM pressure: system_used=%.1f GB > threshold=%.1f GB (iter=%d phase=%s)",
                s.system_used_gb,
                self._warn_threshold,
                iter_id,
                phase,
            )
        return s

    def flush(self) -> None:
        """Append buffered snapshots to ram.jsonl and clear buffer."""
        if not self._snapshots:
            return
        with self._path.open("a") as fh:
            for s in self._snapshots:
                fh.write(json.dumps(asdict(s)) + "\n")
        self._snapshots.clear()

    def compact_record(self, seed_id: str, iter_id: int) -> dict:
        """Return a compact dict of the most recent snapshots for embedding in run.jsonl."""
        relevant = [s for s in self._snapshots if s.seed_id == seed_id and s.iter_id == iter_id]
        if not relevant:
            return {}
        return {s.phase: s.system_used_gb for s in relevant}
