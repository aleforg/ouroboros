import json
from dataclasses import asdict

from ouroboros.config import RunConfig, check_ram_budget, config_hash


def test_config_hash_is_deterministic():
    cfg = RunConfig()
    assert config_hash(cfg) == config_hash(cfg)


def test_config_hash_changes_on_field_change():
    cfg1 = RunConfig(mode="test")
    cfg2 = RunConfig(mode="full")
    assert config_hash(cfg1) != config_hash(cfg2)


def test_config_json_round_trip():
    cfg = RunConfig(mode="test", max_t2i_calls=100)
    d = asdict(cfg)
    assert json.loads(json.dumps(d)) == d


def test_ram_budget_ok():
    ok, msg = check_ram_budget("dolphin-llama3:8b", "mlx-community/Qwen3-VL-8B-Instruct-4bit", budget_gb=12.0)
    assert ok
    assert msg == ""


def test_ram_budget_warning():
    # Without aggressive unload, peak = sum (5+6=11 GB) → >85% of 12.5 GB budget → warns
    ok, msg = check_ram_budget(
        "dolphin-llama3:8b",
        "mlx-community/Qwen3-VL-8B-Instruct-4bit",
        budget_gb=12.5,
        aggressive_unload=False,
    )
    assert ok
    assert "WARNING" in msg


def test_ram_budget_abort():
    # Without aggressive unload, peak = sum (5+7.5=12.5 GB) > 10 GB budget → abort
    ok, msg = check_ram_budget(
        "dolphin-llama3:8b", "llama3.2-vision:11b", budget_gb=10.0, aggressive_unload=False
    )
    assert not ok
    assert "exceeds" in msg


def test_budget_property():
    cfg_test = RunConfig(mode="test")
    cfg_full = RunConfig(mode="full")
    assert cfg_test.budget.m == 2
    assert cfg_full.budget.m == 4
