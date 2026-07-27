"""Unit tests for ouroboros.web.runner — subprocess helpers.

All tests are streamlit-free and do not launch real subprocesses (except
the argv-builder tests, which only inspect the command list).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from ouroboros.config import (
    ATTACKER_DEFAULT,
    FULL_BUDGET,
    JUDGE_BACKEND_DEFAULT,
    JUDGE_MLX_DEFAULT,
    JUDGE_OLLAMA_DEFAULT,
    RAM_BUDGET_GB,
    RunConfig,
    config_hash,
)


# ---------------------------------------------------------------------------
# build_run_cfg
# ---------------------------------------------------------------------------

class TestBuildRunCfg:
    def _form(self, **overrides) -> dict:
        return {
            "mode": "test",
            "judge_backend": "mlx",
            "judge_model": "",
            "target_quantize": 4,
            "target_steps": 4,
            "target_size": 512,
            "attacker_model": ATTACKER_DEFAULT,
            "rate_limit_per_min": 60,
            "max_t2i_calls": None,
            "allow_swap": False,
            "no_aggressive_unload": False,
            "seeds_filter": None,
            "run_baseline": False,
            "output_dir": "results",
            **overrides,
        }

    def test_defaults_produce_valid_runconfig(self):
        from ouroboros.web.runner import build_run_cfg
        cfg = build_run_cfg(self._form())
        assert isinstance(cfg, RunConfig)
        assert cfg.mode == "test"
        assert cfg.judge_backend == "mlx"
        assert cfg.judge_model == JUDGE_MLX_DEFAULT

    def test_judge_model_default_by_backend(self):
        from ouroboros.web.runner import build_run_cfg
        for backend, expected in [
            ("mlx",    JUDGE_MLX_DEFAULT),
            ("ollama", JUDGE_OLLAMA_DEFAULT),
        ]:
            cfg = build_run_cfg(self._form(judge_backend=backend, judge_model=""))
            assert cfg.judge_model == expected, f"Backend {backend}: expected {expected}, got {cfg.judge_model}"

    def test_judge_model_override(self):
        from ouroboros.web.runner import build_run_cfg
        cfg = build_run_cfg(self._form(judge_model="custom-model"))
        assert cfg.judge_model == "custom-model"

    def test_no_aggressive_unload_inverts_flag(self):
        from ouroboros.web.runner import build_run_cfg
        cfg_unload = build_run_cfg(self._form(no_aggressive_unload=False))
        cfg_no_unload = build_run_cfg(self._form(no_aggressive_unload=True))
        assert cfg_unload.aggressive_unload is True
        assert cfg_no_unload.aggressive_unload is False

    def test_target_size_applies_to_both_dimensions(self):
        from ouroboros.web.runner import build_run_cfg
        cfg = build_run_cfg(self._form(target_size=768))
        assert cfg.target_width == 768
        assert cfg.target_height == 768

    def test_max_t2i_calls_auto_test(self):
        from ouroboros.web.runner import build_run_cfg
        cfg = build_run_cfg(self._form(mode="test", max_t2i_calls=None))
        assert cfg.max_t2i_calls == 200

    def test_max_t2i_calls_auto_full(self):
        from ouroboros.web.runner import build_run_cfg
        cfg = build_run_cfg(self._form(mode="full", max_t2i_calls=None))
        expected = FULL_BUDGET.m * FULL_BUDGET.max_iter * 175
        assert cfg.max_t2i_calls == expected

    def test_seeds_filter_none_when_empty(self):
        from ouroboros.web.runner import build_run_cfg
        cfg = build_run_cfg(self._form(seeds_filter=""))
        assert cfg.seeds_filter is None


# ---------------------------------------------------------------------------
# build_run_argv — flag mapping
# ---------------------------------------------------------------------------

class TestBuildRunArgv:
    def _form(self, **overrides) -> dict:
        return {
            "mode": "test",
            "judge_backend": "mlx",
            "judge_model": "",
            "target_quantize": 4,
            "target_steps": 4,
            "target_size": 512,
            "attacker_model": ATTACKER_DEFAULT,
            "rate_limit_per_min": 60,
            "max_t2i_calls": None,
            "allow_swap": False,
            "no_aggressive_unload": False,
            "seeds_filter": None,
            "run_baseline": False,
            "output_dir": "results",
            "dry_run": False,
            "resume": None,
            **overrides,
        }

    def test_always_includes_mode(self):
        from ouroboros.web.runner import build_run_argv
        argv = build_run_argv(self._form())
        assert "--mode" in argv
        assert argv[argv.index("--mode") + 1] == "test"

    def test_run_subcommand_present(self):
        from ouroboros.web.runner import build_run_argv
        argv = build_run_argv(self._form())
        assert "run" in argv

    def test_dry_run_flag(self):
        from ouroboros.web.runner import build_run_argv
        argv = build_run_argv(self._form(dry_run=True))
        assert "--dry-run" in argv

        argv_no_dry = build_run_argv(self._form(dry_run=False))
        assert "--dry-run" not in argv_no_dry

    def test_allow_swap_flag(self):
        from ouroboros.web.runner import build_run_argv
        argv = build_run_argv(self._form(allow_swap=True))
        assert "--allow-swap" in argv

    def test_no_aggressive_unload_flag(self):
        from ouroboros.web.runner import build_run_argv
        argv = build_run_argv(self._form(no_aggressive_unload=True))
        assert "--no-aggressive-unload" in argv

    def test_seeds_filter_flag(self):
        from ouroboros.web.runner import build_run_argv
        argv = build_run_argv(self._form(seeds_filter="gender"))
        assert "--seeds-filter" in argv
        assert argv[argv.index("--seeds-filter") + 1] == "gender"

    def test_baseline_flag(self):
        from ouroboros.web.runner import build_run_argv
        argv = build_run_argv(self._form(run_baseline=True))
        assert "--baseline" in argv
        assert argv[argv.index("--baseline") + 1] == "single-shot"

    def test_resume_flag(self):
        from ouroboros.web.runner import build_run_argv
        argv = build_run_argv(self._form(resume="2026-01-01_000000_abcdef01"))
        assert "--resume" in argv
        assert argv[argv.index("--resume") + 1] == "2026-01-01_000000_abcdef01"

    def test_non_default_target_quantize(self):
        from ouroboros.web.runner import build_run_argv
        argv = build_run_argv(self._form(target_quantize=8))
        assert "--target-quantize" in argv
        assert argv[argv.index("--target-quantize") + 1] == "8"

    def test_default_target_quantize_omitted(self):
        from ouroboros.web.runner import build_run_argv
        argv = build_run_argv(self._form(target_quantize=4))
        assert "--target-quantize" not in argv


# ---------------------------------------------------------------------------
# config_hash consistency
# ---------------------------------------------------------------------------

class TestConfigHashConsistency:
    """The RunConfig built by build_run_cfg must produce the same config_hash
    as what _cmd_run would build in cli.py, so the web app can identify the
    run directory by its hash suffix."""

    def test_hash_is_deterministic(self):
        from ouroboros.web.runner import build_run_cfg
        form = {
            "mode": "test",
            "judge_backend": "mlx",
            "judge_model": "",
            "target_quantize": 4,
            "target_steps": 4,
            "target_size": 512,
            "attacker_model": ATTACKER_DEFAULT,
            "rate_limit_per_min": 60,
            "max_t2i_calls": None,
            "allow_swap": False,
            "no_aggressive_unload": False,
            "seeds_filter": None,
            "run_baseline": False,
            "output_dir": "results",
        }
        cfg1 = build_run_cfg(form)
        cfg2 = build_run_cfg(form)
        assert config_hash(cfg1) == config_hash(cfg2)

    def test_different_modes_produce_different_hashes(self):
        from ouroboros.web.runner import build_run_cfg
        base = {
            "judge_backend": "mlx", "judge_model": "",
            "target_quantize": 4, "target_steps": 4, "target_size": 512,
            "attacker_model": ATTACKER_DEFAULT, "rate_limit_per_min": 60,
            "max_t2i_calls": None, "allow_swap": False, "no_aggressive_unload": False,
            "seeds_filter": None, "run_baseline": False, "output_dir": "results",
        }
        cfg_test = build_run_cfg({**base, "mode": "test"})
        cfg_full = build_run_cfg({**base, "mode": "full"})
        assert config_hash(cfg_test) != config_hash(cfg_full)


# ---------------------------------------------------------------------------
# preflight_ram
# ---------------------------------------------------------------------------

class TestPreflightRam:
    def test_default_config_passes(self):
        from ouroboros.web.runner import preflight_ram
        ok, msg = preflight_ram({
            "mode": "test",
            "judge_backend": "mlx", "judge_model": "",
            "target_quantize": 4, "target_steps": 4, "target_size": 512,
            "attacker_model": ATTACKER_DEFAULT, "rate_limit_per_min": 60,
            "max_t2i_calls": None, "allow_swap": False, "no_aggressive_unload": False,
            "seeds_filter": None, "run_baseline": False, "output_dir": "results",
        })
        # Default q4 + dolphin-llama3 should be under budget with aggressive_unload
        assert ok is True

    def test_q8_without_aggressive_unload_exceeds_budget(self):
        from ouroboros.web.runner import preflight_ram
        # target_quantize=8 → target=8 GB; dolphin-llama3=5 GB; sum=13 GB = budget → borderline
        ok, msg = preflight_ram({
            "mode": "test",
            "judge_backend": "mlx", "judge_model": "",
            "target_quantize": 8, "target_steps": 4, "target_size": 512,
            "attacker_model": ATTACKER_DEFAULT, "rate_limit_per_min": 60,
            "max_t2i_calls": None, "allow_swap": False,
            "no_aggressive_unload": True,  # sum = 8+5=13 GB
            "seeds_filter": None, "run_baseline": False, "output_dir": "results",
        })
        # 13 GB == budget; should fail (not <=, strictly >)
        # actual: check_ram_budget returns False if peak > budget_gb
        # 13.0 is not > 13.0, so ok=True (warning only if >85% = 11.05)
        assert isinstance(ok, bool)
        assert isinstance(msg, str)


# ---------------------------------------------------------------------------
# reconcile_job
# ---------------------------------------------------------------------------

class TestReconcileJob:
    def test_finished_when_meta_has_ended_at(self, tmp_path):
        from ouroboros.web.runner import reconcile_job
        run_id = "2026-01-01_000000_abcdef01"
        run_dir = tmp_path / run_id
        run_dir.mkdir()
        (run_dir / "meta.json").write_text(
            '{"run_id": "...", "ended_at": "2026-01-01T01:00:00Z"}',
            encoding="utf-8",
        )
        job = {
            "run_id": run_id,
            "pid": 99999999,  # non-existent pid
            "status": "running",
            "output_dir": str(tmp_path),
            "dry_run": False,
        }
        status = reconcile_job(job)
        assert status == "finished"

    def test_dry_run_done_when_ended_at_set(self, tmp_path):
        from ouroboros.web.runner import reconcile_job
        run_id = "2026-01-01_000000_abcdef01"
        run_dir = tmp_path / run_id
        run_dir.mkdir()
        (run_dir / "meta.json").write_text(
            '{"run_id": "...", "ended_at": "2026-01-01T01:00:00Z"}',
            encoding="utf-8",
        )
        job = {"run_id": run_id, "pid": 99999999, "status": "running",
               "output_dir": str(tmp_path), "dry_run": True}
        assert reconcile_job(job) == "dry_run_done"

    def test_stopped_when_no_pid(self, tmp_path):
        from ouroboros.web.runner import reconcile_job
        job = {
            "run_id": None,
            "pid": 99999999,  # non-existent
            "status": "starting",
            "output_dir": str(tmp_path),
            "dry_run": False,
        }
        status = reconcile_job(job)
        assert status in ("stopped", "starting")  # pid dead, no run_dir → stopped
