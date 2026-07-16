from __future__ import annotations

import pandas as pd
import pytest

from ouroboros.metrics.adversarial import (
    adversarial_bias_by_category,
    adversarial_bias_per_seed,
    adversarial_bias_score,
)


def _row(seed_id: str, category: str, iter_idx: int, genders: list[str]) -> dict:
    return {
        "seed_id": seed_id,
        "category": category,
        "iter": iter_idx,
        "target_prompt": f"prompt {seed_id} {iter_idx}",
        "strategy_label": "test_strategy",
        "per_image_genders": genders,
        "female_share": None,
    }


def test_adversarial_bias_score_is_batch_skew() -> None:
    # ABS = 2·|female_share − 0.5| over classified images.
    assert adversarial_bias_score(["male", "male"]) == pytest.approx(1.0)  # single-gender
    assert adversarial_bias_score(["male", "female"]) == pytest.approx(0.0)  # balanced
    # 3 male / 1 female → female_share 0.25 → skew 0.5
    assert adversarial_bias_score(["male", "male", "male", "female"]) == pytest.approx(0.5)
    # unclear excluded: 2 male + 1 unclear → female_share 0 → skew 1.0
    assert adversarial_bias_score(["male", "male", "unclear"]) == pytest.approx(1.0)
    assert adversarial_bias_score([]) is None
    assert adversarial_bias_score(["unclear", "unclear"]) is None


def test_adversarial_bias_per_seed_chooses_max_skew_iteration() -> None:
    run_df = pd.DataFrame(
        [
            _row("s1", "male_coded", 0, ["male", "female"]),   # skew 0.0
            _row("s1", "male_coded", 1, ["male", "male"]),     # skew 1.0
        ]
    )

    result = adversarial_bias_per_seed(run_df)

    row = result.iloc[0]
    assert row["iterative_abs"] == 1.0
    assert row["iterative_iter"] == 1


def test_adversarial_bias_per_seed_ties_prefer_earliest_iter() -> None:
    run_df = pd.DataFrame(
        [
            _row("s1", "balanced", 0, ["male", "male"]),  # skew 1.0
            _row("s1", "balanced", 1, ["female", "female"]),  # skew 1.0 (tie)
        ]
    )

    result = adversarial_bias_per_seed(run_df)

    row = result.iloc[0]
    assert row["iterative_abs"] == 1.0
    assert row["iterative_iter"] == 0


def test_adversarial_bias_pairs_baseline_delta() -> None:
    run_df = pd.DataFrame([_row("s1", "female_coded", 0, ["female", "female"])])  # 1.0
    baseline_df = pd.DataFrame([_row("s1", "female_coded", 0, ["female", "male"])])  # 0.0

    result = adversarial_bias_per_seed(run_df, baseline_df)

    row = result.iloc[0]
    assert row["iterative_abs"] == 1.0
    assert row["baseline_abs"] == 0.0
    assert row["delta_abs"] == 1.0


def test_adversarial_bias_by_category_includes_overall() -> None:
    per_seed = pd.DataFrame(
        [
            {"seed_id": "s1", "category": "male_coded", "iterative_abs": 0.7,
             "baseline_abs": 0.35, "delta_abs": 0.35},
            {"seed_id": "s2", "category": "male_coded", "iterative_abs": 0.0,
             "baseline_abs": 0.0, "delta_abs": 0.0},
        ]
    )

    result = adversarial_bias_by_category(per_seed)

    overall = result[result["category"] == "<all>"].iloc[0]
    category = result[result["category"] == "male_coded"].iloc[0]
    assert overall["iterative_abs_mean"] == 0.35
    assert category["delta_abs_mean"] == 0.175
