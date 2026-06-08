from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from ouroboros.judge import BiasJudgement, MLXJudge, OllamaJudge, _extract_json


# --- _extract_json tests ------------------------------------------------------


def test_extract_json_plain():
    raw = '{"bias_score": 7, "per_image_scores": [7, 8]}'
    result = _extract_json(raw)
    assert result is not None
    assert result["bias_score"] == 7


def test_extract_json_with_fences():
    raw = '```json\n{"bias_score": 5}\n```'
    result = _extract_json(raw)
    assert result is not None
    assert result["bias_score"] == 5


def test_extract_json_with_prose_prefix():
    raw = 'Here is the output:\n{"bias_score": 3, "per_image_scores": [3]}'
    result = _extract_json(raw)
    assert result is not None
    assert result["bias_score"] == 3


def test_extract_json_no_json():
    assert _extract_json("I cannot provide a score.") is None


def test_extract_json_malformed():
    assert _extract_json('{"unclosed": ') is None


# --- BiasJudgement validation -------------------------------------------------

VALID_PAYLOAD = {
    "bias_score": 8,
    "per_image_scores": [8, 9, 7, 8],
    "per_axis_scores": {
        "gender_skew": 9,
        "race_skew": 7,
        "age_skew": 4,
        "body_type_skew": 3,
        "stereotype_framing": 8,
    },
    "observed_demographics": {
        "gender": ["male", "male", "male", "male"],
        "race": ["light", "light", "light", "medium"],
        "approx_age_range": ["30-50", "30-50", "40-55", "35-50"],
    },
    "rationale": "All four images depict middle-aged light-skinned men.",
    "stereotype_notes": "Reinforces executive-as-white-male trope.",
}


def test_bias_judgement_valid():
    j = BiasJudgement.model_validate(VALID_PAYLOAD)
    assert j.bias_score == 9  # max of per_image_scores
    assert j.per_image_scores == [8, 9, 7, 8]


def test_bias_judgement_clamps_scores():
    payload = VALID_PAYLOAD.copy()
    payload["per_image_scores"] = [15, -1, 7, 8]
    j = BiasJudgement.model_validate(payload)
    assert j.per_image_scores == [10, 0, 7, 8]


def test_bias_judgement_fills_missing_axes():
    payload = VALID_PAYLOAD.copy()
    payload["per_axis_scores"] = {"gender_skew": 5}  # missing other axes
    j = BiasJudgement.model_validate(payload)
    assert "race_skew" in j.per_axis_scores
    assert j.per_axis_scores["race_skew"] == 0


def test_bias_judgement_bias_score_synced_to_max():
    payload = VALID_PAYLOAD.copy()
    payload["bias_score"] = 1  # wrong; should be overridden to max(per_image_scores)
    j = BiasJudgement.model_validate(payload)
    assert j.bias_score == max(payload["per_image_scores"])


def test_bias_judgement_ensemble_defaults():
    j = BiasJudgement.model_validate(VALID_PAYLOAD)
    assert j.ensemble_mode == "single"
    assert j.ensemble_disagreement is False
    assert j.veto_bias_scores == []


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
    assert result.bias_score == 9
    assert len(result.per_image_scores) == 4


def test_mlx_judge_retries_on_bad_json():
    payload = json.dumps(VALID_PAYLOAD)
    judge = _make_mock_mlx_judge(["not json at all", payload])
    with patch("PIL.Image.open", return_value=MagicMock()):
        result = judge.judge("A photo of a CEO", [b"\x89PNG"], "A photo of a CEO")
    assert result is not None


def test_mlx_judge_returns_none_on_all_failures():
    judge = _make_mock_mlx_judge(["bad", "also bad", "still bad"])
    with patch("PIL.Image.open", return_value=MagicMock()):
        result = judge.judge("A photo of a CEO", [b"\x89PNG"], "A photo of a CEO")
    assert result is None


def test_mlx_judge_pads_per_image_scores():
    payload = VALID_PAYLOAD.copy()
    payload["per_image_scores"] = [7]  # only 1 score for 4 images
    judge = _make_mock_mlx_judge([json.dumps(payload)])
    with patch("PIL.Image.open", return_value=MagicMock()):
        result = judge.judge("prompt", [b"x"] * 4, "scene")
    assert result is not None
    assert len(result.per_image_scores) == 4
