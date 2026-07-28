"""End-to-end resume: a capped run, then a resume that finishes it.

Exercises the real cli._async_run path with the models faked out, because the
individual pieces passing in isolation is not evidence that extending a run in
place works — and by the time it runs for real there are hours of GPU time in
the directory it writes into.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ouroboros.config import RunConfig
from ouroboros.judge import GenderJudgement
from ouroboros.seeds import Seed
from ouroboros.targets.base import SampleResult

M = 2  # test-mode budget


def _seeds(n: int) -> list[Seed]:
    return [
        Seed(
            seed_id=f"s{i:02d}",
            category="balanced",
            base_scene=f"Photo portrait of a worker {i}",
            source_text=f"worker {i}",
        )
        for i in range(n)
    ]


class FakeTarget:
    name = "fake"
    estimated_peak_ram_gb = 0.0

    def __init__(self) -> None:
        self.calls = 0

    async def generate_m(self, prompt: str, m: int) -> list[SampleResult]:
        self.calls += 1
        return [SampleResult(outcome="image", image_bytes=b"png") for _ in range(m)]

    async def aclose(self) -> None:
        pass


class FakeJudge:
    judge_id = "fake"

    def __init__(self, labels: list[str]) -> None:
        self._labels = labels

    def judge(self, target_prompt, images, base_scene):
        return GenderJudgement(
            per_image_genders=list(self._labels)[: len(images)],
            rationale="fake",
            judge_id=self.judge_id,
        )

    async def aclose(self) -> None:
        pass


class FakeAttacker:
    def __init__(self, *a, **k) -> None:
        pass

    def propose(self, base_scene, memory):   # sync, like OllamaAttacker.propose
        from ouroboros.attacker import AttackerCandidate

        return AttackerCandidate(
            target_prompt=f"adversarial: {base_scene}",
            strategy_label="fake_strategy",
            rationale="fake",
        )

    async def aclose(self) -> None:
        pass


def _args(resume: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(resume=resume, replay=None, dry_run=False)


def _run(cfg: RunConfig, seeds: list[Seed], resume: str | None, labels: list[str]) -> None:
    """Drive cli._async_run with the three models faked out."""
    from ouroboros import cli

    with patch("ouroboros.targets.build_target", return_value=FakeTarget()), \
         patch("ouroboros.judge.build_judge", return_value=FakeJudge(labels)), \
         patch("ouroboros.attacker.OllamaAttacker", FakeAttacker):
        asyncio.run(cli._async_run(cfg, seeds, _args(resume)))


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture
def cfg_factory(tmp_path):
    def make(max_t2i_calls: int) -> RunConfig:
        return RunConfig(
            mode="test",
            output_dir=str(tmp_path),
            run_baseline=True,
            baseline_mode="matched",
            max_t2i_calls=max_t2i_calls,
            aggressive_unload=False,
        )
    return make


class TestResumeEndToEnd:
    def test_capped_run_then_resume_completes_in_place(self, cfg_factory, tmp_path):
        seeds = _seeds(6)
        # "unclear" everywhere: no seed ever succeeds, so every seed spends its
        # full max_iter and the cap is what stops the loop — the situation the
        # resume exists for.
        labels = ["unclear"] * M

        # Session 1: cap allows 3 batches (M=2 → 6 images).
        _run(cfg_factory(6), seeds, resume=None, labels=labels)
        run_dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
        assert len(run_dirs) == 1, "the first session must create exactly one directory"
        run_dir = run_dirs[0]
        run_id = run_dir.name

        ckpt_1 = json.loads((run_dir / "checkpoint.json").read_text())
        done_1 = list(ckpt_1["completed_seed_ids"])
        rows_1 = _read(run_dir / "run.jsonl")
        base_1 = _read(run_dir / "baseline.jsonl")
        assert 0 < len(done_1) < len(seeds), "the cap should stop the loop part-way"
        assert {r["seed_id"] for r in base_1} == set(done_1), \
            "the baseline covers exactly the seeds the loop completed"

        # Session 2: resume with a fresh per-session budget.
        _run(cfg_factory(100), seeds, resume=run_id, labels=labels)

        assert [p for p in tmp_path.iterdir() if p.is_dir()] == [run_dir], \
            "the resume must not mint a second directory"

        rows_2 = _read(run_dir / "run.jsonl")
        base_2 = _read(run_dir / "baseline.jsonl")
        ckpt_2 = json.loads((run_dir / "checkpoint.json").read_text())

        # Nothing from session 1 was lost or rewritten.
        assert rows_2[: len(rows_1)] == rows_1
        assert base_2[: len(base_1)] == base_1

        assert set(ckpt_2["completed_seed_ids"]) == {s.seed_id for s in seeds}

        # Every seed has a baseline, and no seed has it twice over.
        base_seeds = [r["seed_id"] for r in base_2]
        assert set(base_seeds) == {s.seed_id for s in seeds}
        for seed_id in done_1:
            assert base_seeds.count(seed_id) == sum(
                1 for r in base_1 if r["seed_id"] == seed_id
            ), f"{seed_id} was given a second comparator on resume"

    def test_resumed_seeds_keep_their_budget_matching(self, cfg_factory, tmp_path):
        """The baseline must draw as many batches as the loop spent, for the
        seeds completed *before* the resume as well — those never enter the
        resuming session's batches_per_seed."""
        seeds = _seeds(4)
        labels = ["unclear"] * M  # never succeeds → max_iter batches per seed

        _run(cfg_factory(6), seeds, resume=None, labels=labels)
        run_dir = next(p for p in tmp_path.iterdir() if p.is_dir())

        rows = _read(run_dir / "run.jsonl")
        spent = {}
        for r in rows:
            if r.get("samples"):
                spent[r["seed_id"]] = spent.get(r["seed_id"], 0) + 1

        base = _read(run_dir / "baseline.jsonl")
        drawn = {}
        for r in base:
            drawn[r["seed_id"]] = drawn.get(r["seed_id"], 0) + 1
        assert drawn == spent, "session 1 baseline is matched"

        _run(cfg_factory(100), seeds, resume=run_dir.name, labels=labels)

        rows = _read(run_dir / "run.jsonl")
        spent = {}
        for r in rows:
            if r.get("samples"):
                spent[r["seed_id"]] = spent.get(r["seed_id"], 0) + 1
        base = _read(run_dir / "baseline.jsonl")
        drawn = {}
        for r in base:
            drawn[r["seed_id"]] = drawn.get(r["seed_id"], 0) + 1

        assert drawn == spent, (
            "after the resume every seed's comparator still matches its realized "
            f"draw count. spent={spent} drawn={drawn}"
        )

    def test_meta_keeps_the_original_config_and_records_the_resume(self, cfg_factory, tmp_path):
        seeds = _seeds(4)
        _run(cfg_factory(6), seeds, resume=None, labels=["unclear"] * M)
        run_dir = next(p for p in tmp_path.iterdir() if p.is_dir())

        _run(cfg_factory(100), seeds, resume=run_dir.name, labels=["unclear"] * M)

        meta = json.loads((run_dir / "meta.json").read_text())
        assert meta["config"]["max_t2i_calls"] == 6, "original config preserved"
        assert meta["run_id"] == run_dir.name
        assert len(meta["resumes"]) == 1
        assert meta["resumes"][0]["config"]["max_t2i_calls"] == 100
        assert meta["ended_at"] is not None, "the resumed session closed the run"

    def test_images_of_earlier_seeds_are_untouched(self, cfg_factory, tmp_path):
        seeds = _seeds(4)
        _run(cfg_factory(6), seeds, resume=None, labels=["unclear"] * M)
        run_dir = next(p for p in tmp_path.iterdir() if p.is_dir())

        before = {p: p.read_bytes() for p in (run_dir / "images").rglob("*.png")}
        assert before, "session 1 wrote images"

        _run(cfg_factory(100), seeds, resume=run_dir.name, labels=["unclear"] * M)

        for path, content in before.items():
            assert path.exists(), f"{path} disappeared on resume"
            assert path.read_bytes() == content, f"{path} was overwritten on resume"

    def test_seeds_the_cap_never_reached_get_no_orphan_baseline(self, cfg_factory, tmp_path):
        """A comparator for a seed with no iterative data cannot be paired.

        The loop stops at the cap; run_baseline used to walk the full seed list
        anyway and hand every unreached seed a one-batch baseline — images spent
        on rows nothing can be compared against.
        """
        seeds = _seeds(6)
        _run(cfg_factory(6), seeds, resume=None, labels=["unclear"] * M)
        run_dir = next(p for p in tmp_path.iterdir() if p.is_dir())

        done = set(json.loads((run_dir / "checkpoint.json").read_text())["completed_seed_ids"])
        base_seeds = {r["seed_id"] for r in _read(run_dir / "baseline.jsonl")}
        assert base_seeds == done, (
            f"baseline covers seeds the loop never ran: {sorted(base_seeds - done)}"
        )

    def test_a_partial_comparator_is_topped_up_not_frozen(self, cfg_factory, tmp_path):
        """The exact shape left behind by a pre-fix run.

        A seed can end a session with fewer baseline batches than the loop spent
        on it. Skipping it wholesale on resume would freeze that mismatch into
        the paired comparison; the missing batches must be generated.
        """
        seeds = _seeds(3)
        _run(cfg_factory(4), seeds, resume=None, labels=["unclear"] * M)
        run_dir = next(p for p in tmp_path.iterdir() if p.is_dir())

        # Simulate the pre-fix leftovers: one baseline batch for a seed whose
        # iterative side spent more than one.
        rows = _read(run_dir / "run.jsonl")
        spent = {}
        for r in rows:
            if r.get("samples"):
                spent[r["seed_id"]] = spent.get(r["seed_id"], 0) + 1
        victim = next(s for s, n in spent.items() if n > 1)

        base = _read(run_dir / "baseline.jsonl")
        kept = [r for r in base if r["seed_id"] != victim][:0] + [
            next(r for r in base if r["seed_id"] == victim)
        ]
        with (run_dir / "baseline.jsonl").open("w", encoding="utf-8") as fh:
            for r in kept:
                fh.write(json.dumps(r) + "\n")
        assert len([r for r in kept if r["seed_id"] == victim]) == 1 < spent[victim]

        _run(cfg_factory(100), seeds, resume=run_dir.name, labels=["unclear"] * M)

        base = _read(run_dir / "baseline.jsonl")
        drawn = {}
        for r in base:
            drawn[r["seed_id"]] = drawn.get(r["seed_id"], 0) + 1
        rows = _read(run_dir / "run.jsonl")
        spent = {}
        for r in rows:
            if r.get("samples"):
                spent[r["seed_id"]] = spent.get(r["seed_id"], 0) + 1

        assert drawn[victim] == spent[victim], (
            f"{victim} kept a partial comparator: {drawn[victim]} batches vs "
            f"{spent[victim]} iterative draws"
        )
        # And the topped-up batches did not overwrite the original one.
        paths = [s["path"] for r in base if r["seed_id"] == victim for s in r["samples"]]
        assert len(paths) == len(set(paths)), "baseline images collided on resume"
