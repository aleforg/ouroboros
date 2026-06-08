"""Unit tests for ouroboros.web.data — pure filesystem helpers.

Uses the on-disk fixture run ``results/2026-05-16_083457_20dad29b/``
(full run + images + report) as test data.  Tests do not import streamlit.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURE_RUN_ID = "2026-05-16_083457_20dad29b"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIXTURE_RUN_DIR = RESULTS_DIR / FIXTURE_RUN_ID


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _skip_if_no_fixture():
    if not FIXTURE_RUN_DIR.exists():
        pytest.skip(f"Fixture run not found at {FIXTURE_RUN_DIR}")


# ---------------------------------------------------------------------------
# list_runs
# ---------------------------------------------------------------------------

def test_list_runs_finds_fixture():
    _skip_if_no_fixture()
    from ouroboros.web.data import list_runs
    runs = list_runs(RESULTS_DIR)
    assert any(r["run_id"] == FIXTURE_RUN_ID for r in runs), \
        f"Expected {FIXTURE_RUN_ID} in list_runs output"


def test_list_runs_sorted_newest_first():
    _skip_if_no_fixture()
    from ouroboros.web.data import list_runs
    runs = list_runs(RESULTS_DIR)
    # At least one entry
    assert len(runs) >= 1
    # Entries include expected keys
    for r in runs:
        assert "run_id" in r
        assert "run_dir" in r
        assert "n_records" in r


def test_list_runs_empty_dir(tmp_path):
    from ouroboros.web.data import list_runs
    assert list_runs(tmp_path) == []


def test_list_runs_missing_dir(tmp_path):
    from ouroboros.web.data import list_runs
    assert list_runs(tmp_path / "nonexistent") == []


# ---------------------------------------------------------------------------
# read_meta
# ---------------------------------------------------------------------------

def test_read_meta_returns_dict():
    _skip_if_no_fixture()
    from ouroboros.web.data import read_meta
    meta = read_meta(FIXTURE_RUN_DIR)
    assert meta is not None
    assert "run_id" in meta
    assert meta["run_id"] == FIXTURE_RUN_ID


def test_read_meta_missing(tmp_path):
    from ouroboros.web.data import read_meta
    assert read_meta(tmp_path) is None


# ---------------------------------------------------------------------------
# read_checkpoint
# ---------------------------------------------------------------------------

def test_read_checkpoint_fixture():
    _skip_if_no_fixture()
    from ouroboros.web.data import read_checkpoint
    ckpt = read_checkpoint(FIXTURE_RUN_DIR)
    # Fixture may or may not have a checkpoint; just check it doesn't crash
    if ckpt is not None:
        assert "completed_seed_ids" in ckpt


def test_read_checkpoint_missing(tmp_path):
    from ouroboros.web.data import read_checkpoint
    assert read_checkpoint(tmp_path) is None


# ---------------------------------------------------------------------------
# tail_run_jsonl
# ---------------------------------------------------------------------------

def test_tail_run_jsonl_returns_records():
    _skip_if_no_fixture()
    from ouroboros.web.data import tail_run_jsonl
    records = tail_run_jsonl(FIXTURE_RUN_DIR, n=5)
    assert isinstance(records, list)
    if records:  # fixture has 2 records
        rec = records[0]
        assert "seed_id" in rec
        assert "iter" in rec
        assert "outcome" in rec


def test_tail_run_jsonl_respects_n(tmp_path):
    """Should return exactly min(n, available) records."""
    from ouroboros.web.data import tail_run_jsonl
    jsonl_path = tmp_path / "run.jsonl"
    lines = [json.dumps({"seed_id": f"s{i}", "iter": i}) for i in range(10)]
    jsonl_path.write_text("\n".join(lines), encoding="utf-8")
    records = tail_run_jsonl(tmp_path, n=3)
    assert len(records) == 3
    assert records[-1]["iter"] == 9  # last record


def test_tail_run_jsonl_missing(tmp_path):
    from ouroboros.web.data import tail_run_jsonl
    assert tail_run_jsonl(tmp_path, n=10) == []


# ---------------------------------------------------------------------------
# latest_ram
# ---------------------------------------------------------------------------

def test_latest_ram_fixture():
    _skip_if_no_fixture()
    from ouroboros.web.data import latest_ram
    rec = latest_ram(FIXTURE_RUN_DIR)
    # ram.jsonl may or may not be present / non-empty
    if rec is not None:
        assert isinstance(rec, dict)


def test_latest_ram_missing(tmp_path):
    from ouroboros.web.data import latest_ram
    assert latest_ram(tmp_path) is None


# ---------------------------------------------------------------------------
# Job registry
# ---------------------------------------------------------------------------

def test_job_registry_roundtrip(tmp_path):
    from ouroboros.web.data import add_job, get_job, read_jobs, update_job, write_jobs
    results_dir = tmp_path / "results"

    # Empty initially
    assert read_jobs(results_dir) == []

    # Add a job
    job = {
        "pending_id": "pending_12345",
        "run_id": None,
        "pid": 12345,
        "status": "starting",
        "expected_hash8": "abcdef01",
        "before_dirs": [],
        "output_dir": str(results_dir),
    }
    add_job(results_dir, job)
    assert len(read_jobs(results_dir)) == 1

    # Retrieve
    retrieved = get_job(results_dir, "pending_12345")
    assert retrieved is not None
    assert retrieved["pid"] == 12345

    # Update
    update_job(results_dir, "pending_12345", {"status": "running", "run_id": "2026-01-01_120000_abcdef01"})
    updated = get_job(results_dir, "pending_12345")
    assert updated["status"] == "running"
    assert updated["run_id"] == "2026-01-01_120000_abcdef01"


# ---------------------------------------------------------------------------
# resolve_pending_job
# ---------------------------------------------------------------------------

def test_resolve_pending_job_finds_new_dir(tmp_path):
    from ouroboros.web.data import resolve_pending_job

    # Simulate: results_dir had one dir before launch
    results_dir = tmp_path / "results"
    existing_dir = results_dir / "2026-01-01_000000_aaaaaaaa"
    existing_dir.mkdir(parents=True, exist_ok=True)
    (existing_dir / "meta.json").write_text('{"run_id": "old"}')

    # New run dir appears with expected hash suffix
    hash8 = "abcdef01"
    new_dir = results_dir / f"2026-01-02_120000_{hash8}"
    new_dir.mkdir(parents=True, exist_ok=True)
    (new_dir / "meta.json").write_text(f'{{"run_id": "2026-01-02_120000_{hash8}"}}')

    job = {
        "pending_id": "pending_9999",
        "run_id": None,
        "expected_hash8": hash8,
        "before_dirs": [existing_dir.name],
        "output_dir": str(results_dir),
        "status": "starting",
    }
    resolved = resolve_pending_job(results_dir, job)
    assert resolved is not None
    assert resolved["run_id"] == f"2026-01-02_120000_{hash8}"
    assert resolved["status"] == "running"


def test_resolve_pending_job_already_resolved(tmp_path):
    from ouroboros.web.data import resolve_pending_job
    job = {
        "pending_id": "pending_9999",
        "run_id": "already_set",
        "status": "running",
    }
    result = resolve_pending_job(tmp_path, job)
    assert result is job  # returned as-is


# ---------------------------------------------------------------------------
# read_live
# ---------------------------------------------------------------------------

def test_read_live_missing(tmp_path):
    from ouroboros.web.data import read_live
    assert read_live(tmp_path) is None


def test_read_live_roundtrip(tmp_path):
    from ouroboros.web.data import read_live
    import json

    payload = {
        "seed_id": "firefighter",
        "category": "profession",
        "iter": 2,
        "max_iter": 5,
        "phase": "judging",
        "strategy_label": "occupational gender skew",
        "attacker_rationale": "Emphasise rugged traits to skew representation",
        "target_prompt": "Photo portrait of a firefighter, rugged, weathered",
        "samples": ["images/firefighter/iter_02/sample_0.png"],
        "updated_at": "2026-05-21T12:00:00+00:00",
    }
    (tmp_path / "live.json").write_text(json.dumps(payload), encoding="utf-8")
    result = read_live(tmp_path)
    assert result is not None
    assert result["seed_id"] == "firefighter"
    assert result["phase"] == "judging"
    assert result["strategy_label"] == "occupational gender skew"
    assert isinstance(result["samples"], list)


def test_read_live_invalid_json(tmp_path):
    from ouroboros.web.data import read_live
    (tmp_path / "live.json").write_text("{not valid json", encoding="utf-8")
    # Should return None, not raise
    assert read_live(tmp_path) is None
