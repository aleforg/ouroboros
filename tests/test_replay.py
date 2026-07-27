import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from ouroboros.replay import run_replay
from ouroboros.targets.base import SampleResult

@pytest.mark.anyio
@patch("ouroboros.replay.build_target")
async def test_run_replay_success(mock_build_target, tmp_path):
    # 1. Setup mock past run
    past_run_id = "2026-05-19_120000_abc12345"
    past_run_dir = tmp_path / "results" / past_run_id
    past_run_dir.mkdir(parents=True, exist_ok=True)
    
    # Write meta.json
    meta = {
        "config": {
            "mode": "test",
            "target_quantize": 4,
            "target_steps": 4,
            "target_width": 512,
            "target_height": 512,
        }
    }
    (past_run_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    
    # Write baseline.jsonl
    baseline_records = [
        {
            "seed_id": "seed-1",
            "target_prompt": "baseline prompt 1",
            "samples": [{"path": "images/seed-1/baseline/sample_0.png", "sha256": "hash-abc"}]
        }
    ]
    with (past_run_dir / "baseline.jsonl").open("w", encoding="utf-8") as f:
        for r in baseline_records:
            f.write(json.dumps(r) + "\n")
            
    # Write run.jsonl
    run_records = [
        {
            "seed_id": "seed-1",
            "target_prompt": "run prompt 1",
            "iter": 0,
            "samples": [{"path": "images/seed-1/iter_00/sample_0.png", "sha256": "hash-xyz"}]
        }
    ]
    with (past_run_dir / "run.jsonl").open("w", encoding="utf-8") as f:
        for r in run_records:
            f.write(json.dumps(r) + "\n")
            
    # 2. Setup mock target backend
    mock_target = MagicMock()
    # Generate 1 image with outcome 'image' and mock bytes
    # The first generate_m is for baseline, the second is for run
    mock_target.generate_m = AsyncMock()
    mock_target.generate_m.side_effect = [
        [SampleResult(outcome="image", image_bytes=b"baseline-bytes")],
        [SampleResult(outcome="image", image_bytes=b"run-bytes")]
    ]
    mock_target.aclose = AsyncMock()
    mock_build_target.return_value = mock_target
    
    # 3. Setup mock hash calculation to match one and mismatch the other
    def mock_sha(b):
        if b == b"baseline-bytes":
            return "hash-abc"  # Match
        return "hash-other"   # Mismatch
        
    with patch("ouroboros.replay.compute_sha256", side_effect=mock_sha):
        await run_replay(past_run_dir, tmp_path / "results")
        
    # 4. Verify outputs
    replay_dir = tmp_path / "results" / f"replay_{past_run_id}"
    assert replay_dir.exists()
    assert (replay_dir / "meta.json").exists()
    
    # Check baseline.jsonl contents
    replay_base_jsonl = replay_dir / "baseline.jsonl"
    assert replay_base_jsonl.exists()
    lines = replay_base_jsonl.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    base_record = json.loads(lines[0])
    assert base_record["samples"][0]["path"] == "images/seed-1/baseline/sample_0.png"
    assert base_record["samples"][0]["sha256"] == "hash-abc"
    
    # Verify baseline image written to disk
    base_img = replay_dir / "images" / "seed-1" / "baseline" / "sample_0.png"
    assert base_img.exists()
    assert base_img.read_bytes() == b"baseline-bytes"
    
    # Check run.jsonl contents
    replay_run_jsonl = replay_dir / "run.jsonl"
    assert replay_run_jsonl.exists()
    lines = replay_run_jsonl.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    run_record = json.loads(lines[0])
    assert run_record["samples"][0]["path"] == "images/seed-1/iter_00/sample_0.png"
    assert run_record["samples"][0]["sha256"] == "hash-other"
    
    # Verify run image written to disk
    run_img = replay_dir / "images" / "seed-1" / "iter_00" / "sample_0.png"
    assert run_img.exists()
    assert run_img.read_bytes() == b"run-bytes"
    
    # Verify mock target was called correctly
    assert mock_target.generate_m.call_count == 2
    mock_target.generate_m.assert_any_call("baseline prompt 1", 1)
    mock_target.generate_m.assert_any_call("run prompt 1", 1)

    # Replay is target-only: unload exactly once at the end, not per record
    assert mock_target.aclose.call_count == 1

    # Reproducibility summary is persisted (baseline matches, run mismatches)
    summary_path = replay_dir / "replay_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["replay_of"] == past_run_id
    assert summary["baseline"] == {"matched": 1, "total": 1}
    assert summary["run"] == {"matched": 0, "total": 1}
    assert summary["matched"] == 1
    assert summary["total"] == 2
    assert summary["match_rate"] == 50.0


def _write_minimal_run(run_dir: Path, config: dict) -> None:
    """Past run with one prompt, enough for run_replay to reach build_target."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.json").write_text(json.dumps({"config": config}), encoding="utf-8")
    record = {
        "seed_id": "seed-1",
        "target_prompt": "run prompt 1",
        "iter": 0,
        "samples": [{"path": "images/seed-1/iter_00/sample_0.png", "sha256": "h"}],
    }
    (run_dir / "run.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")


def _replay_target_mock() -> MagicMock:
    target = MagicMock()
    target.generate_m = AsyncMock(
        return_value=[SampleResult(outcome="image", image_bytes=b"bytes")]
    )
    target.aclose = AsyncMock()
    return target


@pytest.mark.anyio
@patch("ouroboros.replay.build_target")
async def test_replay_reuses_the_recorded_backend(mock_build_target, tmp_path):
    """A CUDA run must not replay through mflux.

    ``target_backend`` used to be missing from the config reconstruction, so a
    diffusers/qwen-image run silently replayed on the mflux default — which on a
    CUDA box fails at import.
    """
    past_run_dir = tmp_path / "results" / "2026-05-19_120000_abc12345"
    _write_minimal_run(past_run_dir, {
        "mode": "test",
        "target_backend": "qwen-image",
        "target_quantize": 4,
        "target_steps": 50,
        "target_width": 1024,
        "target_height": 1024,
    })
    mock_build_target.return_value = _replay_target_mock()

    await run_replay(past_run_dir, tmp_path / "results")

    args, kwargs = mock_build_target.call_args
    assert args[0] == "qwen-image"
    assert kwargs["target_steps"] == 50
    assert kwargs["target_width"] == 1024


@pytest.mark.anyio
@patch("ouroboros.replay.build_target")
async def test_replay_reads_pre_rename_flux_params(mock_build_target, tmp_path):
    """Runs recorded before the flux_* → target_* rename stay replayable."""
    past_run_dir = tmp_path / "results" / "2026-05-19_120000_def67890"
    _write_minimal_run(past_run_dir, {
        "mode": "test",
        "flux_quantize": 8,
        "flux_steps": 6,
        "flux_width": 768,
        "flux_height": 768,
    })
    mock_build_target.return_value = _replay_target_mock()

    await run_replay(past_run_dir, tmp_path / "results")

    args, kwargs = mock_build_target.call_args
    assert args[0] == "flux"  # absent from old meta.json → mflux default
    assert kwargs["target_quantize"] == 8
    assert kwargs["target_steps"] == 6
    assert kwargs["target_width"] == 768
    assert kwargs["target_height"] == 768
