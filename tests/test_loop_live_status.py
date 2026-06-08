"""Tests for the _write_live helper in ouroboros.loop.

Verifies atomic write behaviour, field presence, overwrite semantics, and
that a bad path never propagates an exception into the caller.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_write_live_creates_file(tmp_path: Path) -> None:
    from ouroboros.loop import _write_live

    _write_live(tmp_path, {"phase": "attacking", "seed_id": "nurse", "iter": 0})

    live_path = tmp_path / "live.json"
    assert live_path.exists(), "live.json should be created after _write_live"
    data = json.loads(live_path.read_text(encoding="utf-8"))
    assert data["phase"] == "attacking"
    assert data["seed_id"] == "nurse"


def test_write_live_adds_updated_at(tmp_path: Path) -> None:
    from ouroboros.loop import _write_live

    _write_live(tmp_path, {"phase": "generating"})

    data = json.loads((tmp_path / "live.json").read_text(encoding="utf-8"))
    assert "updated_at" in data, "_write_live must inject 'updated_at'"
    # Should be a non-empty ISO string
    assert len(data["updated_at"]) > 10


def test_write_live_overwrites_previous(tmp_path: Path) -> None:
    from ouroboros.loop import _write_live

    _write_live(tmp_path, {"phase": "attacking", "iter": 0})
    _write_live(tmp_path, {"phase": "generating", "iter": 0, "strategy_label": "skew"})

    data = json.loads((tmp_path / "live.json").read_text(encoding="utf-8"))
    # The second write must overwrite; 'phase' reflects the latest call
    assert data["phase"] == "generating"
    assert data.get("strategy_label") == "skew"


def test_write_live_iter_done_fields(tmp_path: Path) -> None:
    from ouroboros.loop import _write_live

    payload = {
        "seed_id": "firefighter",
        "category": "profession",
        "base_scene": "Photo portrait of a firefighter",
        "iter": 2,
        "max_iter": 5,
        "phase": "iter_done",
        "strategy_label": "occupational gender skew",
        "attacker_rationale": "Emphasise physical strength…",
        "target_prompt": "Photo portrait of a firefighter, rugged…",
        "samples": ["images/firefighter/iter_02/sample_0.png"],
        "outcome": "fail",
        "bias_score": 6,
        "judge_rationale": "Images skewed toward male depiction",
        "per_axis_scores": {
            "gender_representation": 7,
            "race_representation": 3,
            "age_representation": 2,
            "intersectional_bias": 4,
            "stereotype_framing": 6,
        },
    }
    _write_live(tmp_path, payload)

    data = json.loads((tmp_path / "live.json").read_text(encoding="utf-8"))
    assert data["phase"] == "iter_done"
    assert data["bias_score"] == 6
    assert "stereotype_framing" in data["per_axis_scores"]
    assert data["outcome"] == "fail"
    assert isinstance(data["samples"], list) and len(data["samples"]) == 1


def test_write_live_no_raise_on_bad_path() -> None:
    """_write_live must never raise even if the directory doesn't exist."""
    from ouroboros.loop import _write_live

    bad_dir = Path("/nonexistent/path/that/does/not/exist")
    # Should complete without exception
    _write_live(bad_dir, {"phase": "attacking"})


def test_write_live_tmp_file_absent_after_write(tmp_path: Path) -> None:
    """The .tmp sentinel file must be cleaned up (replaced) after a successful write."""
    from ouroboros.loop import _write_live

    _write_live(tmp_path, {"phase": "judging"})

    assert not (tmp_path / "live.json.tmp").exists(), \
        "live.json.tmp should be replaced (not left behind) after atomic write"
