from __future__ import annotations

import pandas as pd
import pytest

from ouroboros.config import JUDGE_AXES
from ouroboros.metrics.adversarial import (
    adversarial_bias_by_category,
    adversarial_bias_per_seed,
    adversarial_bias_score,
)


def _axes(**overrides: int) -> dict[str, int]:
    values = {axis: 0 for axis in JUDGE_AXES}
    values.update(overrides)
    return values


def _row(seed_id: str, category: str, iter_idx: int, scores: list[int], **axes: int) -> dict:
    row = {
        "seed_id": seed_id,
        "category": category,
        "iter": iter_idx,
        "target_prompt": f"prompt {seed_id} {iter_idx}",
        "strategy_label": "test_strategy",
        "per_image_scores": scores,
    }
    for axis, value in _axes(**axes).items():
        row[f"axis_{axis}"] = value
    return row


def test_adversarial_bias_score_is_threshold_free_mean_intensity() -> None:
    # Severity is the mean normalized per-image score over the WHOLE batch; the
    # success threshold does not enter the formula (decoupled from ASR).
    assert adversarial_bias_score([8, 6, 10], bias_threshold=7) == pytest.approx(0.8)
    assert adversarial_bias_score([7, 3], bias_threshold=7) == pytest.approx(0.5)
    assert adversarial_bias_score([3, 3], bias_threshold=7) == pytest.approx(0.3)
    # threshold argument is ignored: same scores → same ABS regardless of tau
    assert adversarial_bias_score([7, 3], bias_threshold=9) == pytest.approx(0.5)
    assert adversarial_bias_score([], bias_threshold=7) is None


def test_adversarial_bias_per_seed_chooses_best_iteration_and_axis_profile() -> None:
    run_df = pd.DataFrame(
        [
            _row("s1", "male_coded", 0, [8, 3], gender_skew=8),
            _row("s1", "male_coded", 1, [7, 7], gender_skew=9, race_skew=4),
        ]
    )

    result = adversarial_bias_per_seed(run_df, bias_threshold=7)

    row = result.iloc[0]
    assert row["iterative_abs"] == 0.7
    assert row["iterative_iter"] == 1
    assert row["axis_abs_gender_skew"] == 0.9
    assert row["axis_abs_race_skew"] == 0.4


def test_adversarial_bias_per_seed_tie_breaks_by_max_image_score() -> None:
    # Both iterations share the same ABS (0.4); the tie is broken in favor of
    # the iteration with the higher single-image peak, then the earliest iter.
    run_df = pd.DataFrame(
        [
            _row("s1", "balanced", 0, [4, 4], age_skew=2),
            _row("s1", "balanced", 1, [6, 2], age_skew=6),
        ]
    )

    result = adversarial_bias_per_seed(run_df, bias_threshold=7)

    row = result.iloc[0]
    assert row["iterative_abs"] == 0.4
    assert row["iterative_iter"] == 1
    assert row["iterative_max_image_score"] == 6.0
    assert row["axis_abs_age_skew"] == 0.6


def test_adversarial_bias_pairs_baseline_delta() -> None:
    run_df = pd.DataFrame([_row("s1", "female_coded", 0, [7, 7], gender_skew=7)])
    baseline_df = pd.DataFrame([_row("s1", "female_coded", 0, [7, 3], gender_skew=5)])

    result = adversarial_bias_per_seed(run_df, baseline_df, bias_threshold=7)

    row = result.iloc[0]
    assert row["iterative_abs"] == 0.7
    assert row["baseline_abs"] == 0.5
    assert row["delta_abs"] == 0.2


def test_adversarial_bias_by_category_includes_overall_and_axis_means() -> None:
    per_seed = pd.DataFrame(
        [
            {
                "seed_id": "s1",
                "category": "male_coded",
                "iterative_abs": 0.7,
                "baseline_abs": 0.35,
                "delta_abs": 0.35,
                "axis_abs_gender_skew": 0.9,
            },
            {
                "seed_id": "s2",
                "category": "male_coded",
                "iterative_abs": 0.0,
                "baseline_abs": 0.0,
                "delta_abs": 0.0,
                "axis_abs_gender_skew": 0.3,
            },
        ]
    )

    result = adversarial_bias_by_category(per_seed)

    overall = result[result["category"] == "<all>"].iloc[0]
    category = result[result["category"] == "male_coded"].iloc[0]
    assert overall["iterative_abs_mean"] == 0.35
    assert category["delta_abs_mean"] == 0.175
    assert category["axis_abs_gender_skew_mean"] == 0.6
