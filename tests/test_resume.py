"""Resume must extend a run, not fork it.

Before these were fixed, resuming a capped run silently produced a broken
paired comparison: the continuation landed in a fresh timestamped directory,
the matched baseline lost the completed seeds' draw counts and fell back to one
batch each, and every already-covered seed had its comparator regenerated.
"""
from __future__ import annotations

import json
from pathlib import Path

from ouroboros.baseline import baseline_batches_per_seed
from ouroboros.config import RunConfig, config_hash
from ouroboros.loop import _batches_from_run_jsonl
from ouroboros.storage import record_resume, write_meta


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


class TestBatchesFromRunJsonl:
    def test_counts_only_records_that_generated_images(self, tmp_path):
        _write_jsonl(tmp_path / "run.jsonl", [
            {"seed_id": "a", "samples": [{"path": "x"}] * 8},
            {"seed_id": "a", "samples": [{"path": "x"}] * 8},
            # attacker_refused: no target call, so not a generative batch
            {"seed_id": "a", "samples": []},
            {"seed_id": "b", "samples": [{"path": "x"}] * 8},
        ])
        assert _batches_from_run_jsonl(tmp_path, {"a", "b"}) == {"a": 2, "b": 1}

    def test_ignores_seeds_outside_the_requested_set(self, tmp_path):
        _write_jsonl(tmp_path / "run.jsonl", [
            {"seed_id": "a", "samples": [{"path": "x"}]},
            {"seed_id": "z", "samples": [{"path": "x"}]},
        ])
        assert _batches_from_run_jsonl(tmp_path, {"a"}) == {"a": 1}

    def test_missing_file_is_not_an_error(self, tmp_path):
        assert _batches_from_run_jsonl(tmp_path, {"a"}) == {}

    def test_survives_a_truncated_final_line(self, tmp_path):
        # A run killed mid-write leaves a partial line; losing the whole file
        # would silently unmatch every seed.
        path = tmp_path / "run.jsonl"
        path.write_text(
            json.dumps({"seed_id": "a", "samples": [{"path": "x"}]}) + "\n"
            + '{"seed_id": "b", "samp',
            encoding="utf-8",
        )
        assert _batches_from_run_jsonl(tmp_path, {"a", "b"}) == {"a": 1}


class TestBaselineBatchesPerSeed:
    def test_counts_batches_not_seeds(self, tmp_path):
        # Counts, not a set: a seed with 1 of 3 owed batches must be topped up,
        # and a done/not-done flag cannot express that.
        _write_jsonl(tmp_path / "baseline.jsonl", [
            {"seed_id": "a", "iter": 0},
            {"seed_id": "a", "iter": 1},
            {"seed_id": "b", "iter": 0},
        ])
        assert baseline_batches_per_seed(tmp_path) == {"a": 2, "b": 1}

    def test_missing_file_means_nothing_done(self, tmp_path):
        assert baseline_batches_per_seed(tmp_path) == {}


class TestRecordResume:
    def test_preserves_the_original_config_and_start(self, tmp_path):
        original = RunConfig(mode="full", max_t2i_calls=1500)
        write_meta(
            tmp_path, "run-1", original,
            attacker_model="a", judge_model="j", judge_backend="ollama",
            started_at="2026-07-28T00:00:00+00:00",
        )
        extended = RunConfig(mode="full", max_t2i_calls=2000)
        record_resume(tmp_path, extended, "2026-07-29T00:00:00+00:00")

        meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
        # The rows written in the first session were produced under cap 1500;
        # overwriting that would misdescribe them.
        assert meta["config"]["max_t2i_calls"] == 1500
        assert meta["started_at"] == "2026-07-28T00:00:00+00:00"
        assert meta["ended_at"] is None  # reopened
        assert len(meta["resumes"]) == 1
        assert meta["resumes"][0]["config"]["max_t2i_calls"] == 2000
        assert meta["resumes"][0]["config_hash"] == config_hash(extended)

    def test_resumes_accumulate(self, tmp_path):
        cfg = RunConfig()
        write_meta(
            tmp_path, "run-1", cfg,
            attacker_model="a", judge_model="j", judge_backend="ollama",
            started_at="t0",
        )
        record_resume(tmp_path, cfg, "t1")
        record_resume(tmp_path, cfg, "t2")
        meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
        assert [r["resumed_at"] for r in meta["resumes"]] == ["t1", "t2"]

    def test_no_meta_is_a_no_op(self, tmp_path):
        record_resume(tmp_path, RunConfig(), "t1")
        assert not (tmp_path / "meta.json").exists()
