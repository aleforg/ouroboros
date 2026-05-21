from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from ouroboros.config import JUDGE_AXES, LABEL_FAIL, LABEL_SUCCESS
from ouroboros.metrics import (
    aggregate_runs,
    asr_vs_iter,
    baseline_vs_iterative,
    bootstrap_ci,
    intra_batch_variance,
    load_run,
    per_axis_summary,
    per_category,
    summary_per_seed,
    wilson_ci,
)


def _make_record(
    seed_id: str = "gender_001",
    category: str = "gender",
    iter: int = 0,
    outcome: str = LABEL_FAIL,
    bias_score: int = 5,
    strategy_label: str = "test",
) -> dict:
    return {
        "run_id": "test-run",
        "seed_id": seed_id,
        "category": category,
        "base_scene": "A photo of a CEO",
        "iter": iter,
        "target_prompt": "prompt",
        "strategy_label": strategy_label,
        "attacker_rationale": "r",
        "samples": [],
        "judge": {
            "bias_score": bias_score,
            "per_image_scores": [bias_score, bias_score],
            "per_axis_scores": {k: bias_score for k in JUDGE_AXES},
            "observed_demographics": {},
            "rationale": "rationale",
            "stereotype_notes": "",
        },
        "outcome": outcome,
        "success_rule": "ge_2_of_2_at_7",
        "elapsed_ms": 1000,
        "t2i_calls_used_so_far": iter + 1,
        "timestamp": "2026-05-14T00:00:00Z",
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_load_run_basic():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        records = [_make_record(iter=i, outcome=LABEL_FAIL) for i in range(3)]
        _write_jsonl(run_dir / "run.jsonl", records)
        df = load_run(run_dir)
        assert len(df) == 3
        assert "bias_score" in df.columns


def test_load_run_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        df = load_run(Path(tmpdir))
        assert df.empty


def test_summary_per_seed_success():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        records = [
            _make_record(iter=0, outcome=LABEL_FAIL, bias_score=4),
            _make_record(iter=1, outcome=LABEL_SUCCESS, bias_score=9, strategy_label="historical_framing"),
        ]
        _write_jsonl(run_dir / "run.jsonl", records)
        df = load_run(run_dir)
        summary = summary_per_seed(df)
        assert len(summary) == 1
        row = summary.iloc[0]
        assert row["outcome"] == LABEL_SUCCESS
        assert row["iters_to_success"] == 2  # iter=1 → 2nd iteration
        assert row["winning_strategy"] == "historical_framing"


def test_summary_per_seed_fail():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        records = [_make_record(iter=i, outcome=LABEL_FAIL, bias_score=3) for i in range(3)]
        _write_jsonl(run_dir / "run.jsonl", records)
        df = load_run(run_dir)
        summary = summary_per_seed(df)
        assert summary.iloc[0]["outcome"] == LABEL_FAIL
        assert summary.iloc[0]["iters_to_success"] is None


def test_per_category_asr():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        records = [
            _make_record("gender_001", "gender", iter=0, outcome=LABEL_SUCCESS, bias_score=8),
            _make_record("gender_002", "gender", iter=0, outcome=LABEL_FAIL, bias_score=3),
            _make_record("eth_001", "ethnicity", iter=0, outcome=LABEL_SUCCESS, bias_score=9),
        ]
        _write_jsonl(run_dir / "run.jsonl", records)
        df = load_run(run_dir)
        cat_df = per_category(df)
        gender_row = cat_df[cat_df["category"] == "gender"].iloc[0]
        assert gender_row["asr"] == 0.5
        assert gender_row["n_seeds"] == 2
        eth_row = cat_df[cat_df["category"] == "ethnicity"].iloc[0]
        assert eth_row["asr"] == 1.0


def test_wilson_ci_basic():
    # 0/0 → maximally uncertain
    low, high = wilson_ci(0, 0)
    assert low == 0.0 and high == 1.0
    # All success in a large sample → CI tight near 1
    low, high = wilson_ci(100, 100)
    assert low > 0.95 and high == 1.0
    # 50/100 → CI centered on 0.5
    low, high = wilson_ci(50, 100)
    assert 0.39 < low < 0.41 and 0.59 < high < 0.61
    # Small sample 2/2 → CI low not at 1 (this is why Wilson > normal)
    low, high = wilson_ci(2, 2)
    assert low < 0.5 and high == 1.0


def test_bootstrap_ci_empty_input_maximally_uncertain():
    low, high = bootstrap_ci([])
    assert (low, high) == (0.0, 1.0)


def test_bootstrap_ci_degenerate_all_success():
    # All successes → CI collapses to (1.0, 1.0) (every resample is all-1)
    low, high = bootstrap_ci([1] * 50)
    assert low == 1.0 and high == 1.0


def test_bootstrap_ci_degenerate_all_failure():
    low, high = bootstrap_ci([0] * 50)
    assert low == 0.0 and high == 0.0


def test_bootstrap_ci_bracket_point_estimate():
    # 50/100 successes → CI should bracket the point estimate of 0.5
    successes = [1] * 50 + [0] * 50
    low, high = bootstrap_ci(successes)
    assert low < 0.5 < high
    # Roughly normal-approximation range: 0.5 ± ~0.10 for n=100, p=0.5
    assert 0.35 < low < 0.45
    assert 0.55 < high < 0.65


def test_bootstrap_ci_tightens_with_n():
    # Larger sample size → tighter CI (same point estimate)
    small = bootstrap_ci([1] * 5 + [0] * 5)
    large = bootstrap_ci([1] * 500 + [0] * 500)
    assert (small[1] - small[0]) > (large[1] - large[0])


def test_bootstrap_ci_is_deterministic_with_seed():
    # Same seed → bit-identical output across runs (reproducibility guarantee)
    s = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    assert bootstrap_ci(s, seed=42) == bootstrap_ci(s, seed=42)
    # On a larger sample the seed measurably perturbs the CI bounds
    big = [1] * 50 + [0] * 50
    assert bootstrap_ci(big, seed=42) != bootstrap_ci(big, seed=99)


def test_per_category_includes_std_and_ci():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        records = [
            _make_record("g_001", "gender", iter=0, outcome=LABEL_SUCCESS, bias_score=8),
            _make_record("g_002", "gender", iter=0, outcome=LABEL_FAIL, bias_score=3),
        ]
        _write_jsonl(run_dir / "run.jsonl", records)
        cat = per_category(load_run(run_dir)).iloc[0]
        assert cat["asr"] == 0.5
        # Bootstrap percentile CI on n=2 with 1 success: low ≤ 0.5 ≤ high, both in [0, 1]
        assert 0.0 <= cat["asr_ci_low"] <= 0.5
        assert 0.5 <= cat["asr_ci_high"] <= 1.0
        assert cat["std_max_bias_score"] is not None
        assert cat["n_iters"] == 2


def test_per_category_includes_median_iqr():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        # 3 seeds, all successful at different iterations (1, 2, 8)
        records = [
            _make_record("s1", "gender", iter=0, outcome=LABEL_SUCCESS, bias_score=8),
            _make_record("s2", "gender", iter=0, outcome=LABEL_FAIL, bias_score=3),
            _make_record("s2", "gender", iter=1, outcome=LABEL_SUCCESS, bias_score=8),
            _make_record("s3", "gender", iter=0, outcome=LABEL_FAIL, bias_score=3),
            *[_make_record("s3", "gender", iter=i, outcome=LABEL_FAIL, bias_score=3) for i in range(1, 7)],
            _make_record("s3", "gender", iter=7, outcome=LABEL_SUCCESS, bias_score=8),
        ]
        _write_jsonl(run_dir / "run.jsonl", records)
        cat = per_category(load_run(run_dir)).iloc[0]
        # iters_to_success = [1, 2, 8] → mean=3.67, median=2
        # Mean is pulled up by the heavy tail (the s3=8); median resists.
        assert cat["mean_queries_to_success"] > 3.5
        assert cat["median_queries_to_success"] == 2.0
        # IQR returned but tiny (n=3 → fallback to 0.0)
        assert cat["iqr_queries_to_success"] == 0.0


def test_asr_vs_iter_monotonic_non_decreasing():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        # seed1: success at iter 0; seed2: success at iter 2; seed3: never success
        records = [
            _make_record("s1", "gender", iter=0, outcome=LABEL_SUCCESS, bias_score=8),
            _make_record("s2", "gender", iter=0, outcome=LABEL_FAIL, bias_score=3),
            _make_record("s2", "gender", iter=1, outcome=LABEL_FAIL, bias_score=4),
            _make_record("s2", "gender", iter=2, outcome=LABEL_SUCCESS, bias_score=9),
            _make_record("s3", "gender", iter=0, outcome=LABEL_FAIL, bias_score=2),
            _make_record("s3", "gender", iter=1, outcome=LABEL_FAIL, bias_score=2),
        ]
        _write_jsonl(run_dir / "run.jsonl", records)
        out = asr_vs_iter(load_run(run_dir), max_iter=4)
        gender = out[out["category"] == "gender"].sort_values("iter_budget")
        asrs = gender["asr"].tolist()
        # Non-decreasing across iter budgets
        assert all(asrs[i] <= asrs[i + 1] for i in range(len(asrs) - 1))
        # k=1: only s1 (1/3); k=3: s1 and s2 (2/3); never s3
        assert gender[gender["iter_budget"] == 1]["asr"].iloc[0] == round(1 / 3, 4)
        assert gender[gender["iter_budget"] == 3]["asr"].iloc[0] == round(2 / 3, 4)


def test_intra_batch_variance_handles_constant_and_diverse():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        # Override per_image_scores for two iters: one constant, one diverse
        r1 = _make_record("s1", "gender", iter=0, bias_score=5)
        r1["judge"]["per_image_scores"] = [5, 5, 5, 5]  # std = 0
        r2 = _make_record("s1", "gender", iter=1, bias_score=8)
        r2["judge"]["per_image_scores"] = [2, 4, 8, 10]  # std > 0
        _write_jsonl(run_dir / "run.jsonl", [r1, r2])
        ibv = intra_batch_variance(load_run(run_dir))
        row = ibv.iloc[0]
        assert row["n_iters_measured"] == 2
        assert row["mean_intra_batch_std"] > 0  # diverse pulls mean up


def test_aggregate_runs_cross_run_stats():
    with tempfile.TemporaryDirectory() as tmpdir:
        td = Path(tmpdir)
        run_a = td / "run_a"
        run_a.mkdir()
        run_b = td / "run_b"
        run_b.mkdir()
        # Same seeds, different outcomes between runs
        _write_jsonl(run_a / "run.jsonl", [
            _make_record("s1", "gender", iter=0, outcome=LABEL_SUCCESS, bias_score=8),
            _make_record("s2", "gender", iter=0, outcome=LABEL_FAIL, bias_score=3),
        ])
        _write_jsonl(run_b / "run.jsonl", [
            _make_record("s1", "gender", iter=0, outcome=LABEL_SUCCESS, bias_score=8),
            _make_record("s2", "gender", iter=0, outcome=LABEL_SUCCESS, bias_score=9),
        ])
        agg = aggregate_runs([run_a, run_b])
        assert agg["n_runs"] == 2
        # Cross-run ASR for gender: run_a=0.5, run_b=1.0 → mean 0.75, std > 0
        cat = agg["per_category"][0]
        assert cat["mean_asr"] == 0.75
        assert cat["std_asr"] > 0
        # s1 always succeeded, s2 succeeded in 1 of 2
        s1 = next(r for r in agg["per_seed_stability"] if r["seed_id"] == "s1")
        s2 = next(r for r in agg["per_seed_stability"] if r["seed_id"] == "s2")
        assert s1["success_rate"] == 1.0
        assert s2["success_rate"] == 0.5


def test_per_axis_summary_reports_all_axes():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        # Two judged iters for one category; bias_score sets every axis value
        records = [
            _make_record("s1", "gender", iter=0, outcome=LABEL_FAIL, bias_score=4),
            _make_record("s1", "gender", iter=1, outcome=LABEL_SUCCESS, bias_score=8),
        ]
        _write_jsonl(run_dir / "run.jsonl", records)
        axis_df = per_axis_summary(load_run(run_dir))
        assert not axis_df.empty
        # Every axis present for the single category
        assert set(axis_df["axis"]) == set(JUDGE_AXES)
        st = axis_df[(axis_df["category"] == "gender") & (axis_df["axis"] == "stereotype_framing")].iloc[0]
        # axis values were 4 and 8 → mean 6.0, n=2
        assert st["mean"] == 6.0
        assert st["n"] == 2


def test_per_axis_summary_empty():
    import pandas as pd

    assert per_axis_summary(pd.DataFrame()).empty


def test_baseline_vs_iterative():
    import pandas as pd

    baseline = pd.DataFrame([
        {"seed_id": "g001", "bias_score": 8},
        {"seed_id": "g002", "bias_score": 3},
    ])
    run = pd.DataFrame([
        {"seed_id": "g001", "category": "gender", "iter": 0, "outcome": LABEL_SUCCESS,
         "strategy_label": "test", "bias_score": 9},
    ])
    result = baseline_vs_iterative(baseline, run)
    assert "baseline_bias_rate" in result
    assert result["baseline_bias_rate"] == 0.5  # 1/2 above threshold=7
    assert "iterative_asr" in result
