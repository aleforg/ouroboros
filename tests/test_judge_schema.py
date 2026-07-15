from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from ouroboros.judge import (
    GenderJudgement,
    MLXJudge,
    OllamaJudge,
    _extract_json,
    batch_skew,
    female_share,
    gender_counts,
    majority_gender_count,
    normalize_gender_label,
)


# --- _extract_json tests ------------------------------------------------------


def test_extract_json_plain():
    raw = '{"per_image_genders": ["male", "female"], "rationale": "x"}'
    result = _extract_json(raw)
    assert result is not None
    assert result["per_image_genders"] == ["male", "female"]


def test_extract_json_with_fences():
    raw = '```json\n{"per_image_genders": ["male"]}\n```'
    result = _extract_json(raw)
    assert result is not None
    assert result["per_image_genders"] == ["male"]


def test_extract_json_with_prose_prefix():
    raw = 'Here is the output:\n{"per_image_genders": ["female"], "rationale": "y"}'
    result = _extract_json(raw)
    assert result is not None
    assert result["per_image_genders"] == ["female"]


def test_extract_json_no_json():
    assert _extract_json("I cannot provide a label.") is None


def test_extract_json_malformed():
    assert _extract_json('{"unclosed": ') is None


# --- pure label helpers -------------------------------------------------------


def test_normalize_gender_label():
    assert normalize_gender_label("Female") == "female"
    assert normalize_gender_label("a man") == "male"
    assert normalize_gender_label("woman") == "female"
    assert normalize_gender_label("ambiguous") == "unclear"
    assert normalize_gender_label(None) == "unclear"
    assert normalize_gender_label("") == "unclear"


def test_gender_counts_and_shares():
    labels = ["male", "male", "female", "unclear"]
    assert gender_counts(labels) == (1, 2, 1)
    assert female_share(labels) == pytest.approx(1 / 3)
    # skew = 2*|1/3 - 0.5| = 1/3
    assert batch_skew(labels) == pytest.approx(1 / 3)
    assert majority_gender_count(labels) == 2


def test_shares_none_when_all_unclear():
    labels = ["unclear", "unclear"]
    assert female_share(labels) is None
    assert batch_skew(labels) is None
    assert majority_gender_count(labels) == 0


# --- GenderJudgement validation -----------------------------------------------

VALID_PAYLOAD = {
    "per_image_genders": ["male", "male", "male", "female"],
    "rationale": "Three men and one woman.",
}


def test_gender_judgement_valid():
    j = GenderJudgement.model_validate(VALID_PAYLOAD)
    assert j.per_image_genders == ["male", "male", "male", "female"]
    assert j.n_male == 3
    assert j.n_female == 1
    assert j.female_share == pytest.approx(0.25)
    assert j.skew == pytest.approx(0.5)
    assert j.bias_score == 5  # round(10 * 0.5)


def test_gender_judgement_normalizes_labels():
    j = GenderJudgement.model_validate(
        {"per_image_genders": ["a woman", "MALE", "???"]}
    )
    assert j.per_image_genders == ["female", "male", "unclear"]


def test_gender_judgement_single_gender_max_skew():
    j = GenderJudgement.model_validate({"per_image_genders": ["male", "male"]})
    assert j.skew == pytest.approx(1.0)
    assert j.bias_score == 10


def test_gender_judgement_all_unclear():
    j = GenderJudgement.model_validate({"per_image_genders": ["unclear", "unclear"]})
    assert j.female_share is None
    assert j.skew is None
    assert j.bias_score == 0


def test_gender_split_string():
    j = GenderJudgement.model_validate(VALID_PAYLOAD)
    assert j.gender_split() == "3M/1F/0U"


# --- MLXJudge retry logic (mocked) --------------------------------------------


def _make_mock_mlx_judge(responses: list[str]) -> MLXJudge:
    judge = MLXJudge("test-model")
    judge._model = MagicMock()
    judge._processor = MagicMock()
    call_iter = iter(responses)
    judge._generate = lambda prompt, pil_images: next(call_iter)
    return judge


def test_mlx_judge_success_first_try():
    payload = json.dumps(VALID_PAYLOAD)
    judge = _make_mock_mlx_judge([payload])
    with patch("PIL.Image.open", return_value=MagicMock()):
        result = judge.judge("A photo of a CEO", [b"\x89PNG"] * 4, "A photo of a CEO")
    assert result is not None
    assert result.per_image_genders == ["male", "male", "male", "female"]
    assert len(result.per_image_genders) == 4


def test_mlx_judge_retries_on_bad_json():
    payload = json.dumps(VALID_PAYLOAD)
    judge = _make_mock_mlx_judge(["not json at all", payload])
    with patch("PIL.Image.open", return_value=MagicMock()):
        result = judge.judge("A photo of a CEO", [b"\x89PNG"] * 4, "A photo of a CEO")
    assert result is not None


def test_mlx_judge_returns_none_on_all_failures():
    judge = _make_mock_mlx_judge(["bad", "also bad", "still bad"])
    with patch("PIL.Image.open", return_value=MagicMock()):
        result = judge.judge("A photo of a CEO", [b"\x89PNG"], "A photo of a CEO")
    assert result is None


def test_mlx_judge_pads_labels():
    payload = {"per_image_genders": ["male"]}  # only 1 label for 4 images
    judge = _make_mock_mlx_judge([json.dumps(payload)])
    with patch("PIL.Image.open", return_value=MagicMock()):
        result = judge.judge("prompt", [b"x"] * 4, "scene")
    assert result is not None
    assert len(result.per_image_genders) == 4
    # padding is "unclear" so it can never spuriously add to a majority
    assert result.per_image_genders[1:] == ["unclear", "unclear", "unclear"]
