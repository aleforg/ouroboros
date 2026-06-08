from __future__ import annotations

from ouroboros.ensemble_judge import EnsembleJudge, _median, build_ensemble
from ouroboros.judge import BiasJudgement
from ouroboros.config import RunConfig


# --- helpers ------------------------------------------------------------------


def _bj(composite: float, judge_id: str = "") -> BiasJudgement:
    """Build a BiasJudgement whose bias_score == `composite`."""
    score = int(composite)
    return BiasJudgement.model_validate({
        "bias_score": score,
        "per_image_scores": [score],
        "per_axis_scores": {
            "gender_skew": score,
            "race_skew": 0,
            "age_skew": 0,
            "body_type_skew": 0,
            "stereotype_framing": score,
        },
        "observed_demographics": {"gender": ["unknown"]},
        "rationale": "r",
        "stereotype_notes": "n",
        "judge_id": judge_id,
    })


class FakeJudge:
    def __init__(self, composite: float | None, judge_id: str) -> None:
        self._composite = composite
        self.judge_id = judge_id
        self.calls = 0
        self.aclose_calls = 0

    def judge(self, target_prompt, images, base_scene) -> BiasJudgement | None:
        self.calls += 1
        return _bj(self._composite, self.judge_id) if self._composite is not None else None

    def aclose(self) -> None:
        self.aclose_calls += 1


def _ensemble(anchor_c, v1_c, v2_c, mode="ensemble", **kw):
    anchor = FakeJudge(anchor_c, "anchor")
    v1 = FakeJudge(v1_c, "veto1")
    v2 = FakeJudge(v2_c, "veto2")
    ej = EnsembleJudge(anchor, [v1, v2], mode=mode, **kw)
    return ej, anchor, v1, v2


# --- _median ------------------------------------------------------------------


def test_median_odd():
    assert _median([1, 2, 3]) == 2


def test_median_even():
    assert _median([1, 3]) == 2.0


def test_median_empty():
    assert _median([]) is None


# --- ensemble mode ------------------------------------------------------------


def test_ensemble_primary_is_anchor():
    ej, anchor, v1, v2 = _ensemble(8.0, 7.0, 9.0)
    r = ej.judge("p", [b"x"], "s")
    assert r.bias_score == 8          # anchor's score is primary
    assert r.judge_id == "anchor"
    assert anchor.calls == 1 and v1.calls == 1 and v2.calls == 1


def test_ensemble_records_veto_scores():
    ej, *_ = _ensemble(8.0, 7.0, 9.0)
    r = ej.judge("p", [b"x"], "s")
    assert r.veto_bias_scores == [7, 9]
    assert r.veto_judge_ids == ["veto1", "veto2"]
    assert r.ensemble_mode == "ensemble"


def test_ensemble_no_disagreement_when_close():
    ej, *_ = _ensemble(8.0, 7.0, 9.0, disagreement_threshold=3.0)
    r = ej.judge("p", [b"x"], "s")
    assert r.ensemble_disagreement is False
    assert r.ensemble_max_delta == 1.0


def test_ensemble_flags_disagreement():
    ej, *_ = _ensemble(8.0, 2.0, 9.0, disagreement_threshold=3.0)  # veto1 delta=6 > 3
    r = ej.judge("p", [b"x"], "s")
    assert r.ensemble_disagreement is True
    assert r.ensemble_max_delta == 6.0


def test_ensemble_unloads_vetos():
    ej, anchor, v1, v2 = _ensemble(8.0, 7.0, 9.0, unload_between=True)
    ej.judge("p", [b"x"], "s")
    assert v1.aclose_calls == 1 and v2.aclose_calls == 1


def test_ensemble_no_unload_when_disabled():
    ej, anchor, v1, v2 = _ensemble(8.0, 7.0, 9.0, unload_between=False)
    ej.judge("p", [b"x"], "s")
    assert v1.aclose_calls == 0 and v2.aclose_calls == 0


# --- cascading mode -----------------------------------------------------------


def test_cascading_skips_anchor_when_high_consensus():
    # local consensus = median(8,9)=8.5 > grey_hi=7 → skip cloud
    ej, anchor, v1, v2 = _ensemble(5.0, 8.0, 9.0, mode="cascading", grey_zone=(4.0, 7.0))
    r = ej.judge("p", [b"x"], "s")
    assert anchor.calls == 0
    assert r.bias_score == 9  # highest veto used as fallback primary


def test_cascading_skips_anchor_when_low_consensus():
    # local consensus = median(1,2)=1.5 < grey_lo=4 → skip cloud
    ej, anchor, v1, v2 = _ensemble(5.0, 1.0, 2.0, mode="cascading", grey_zone=(4.0, 7.0))
    r = ej.judge("p", [b"x"], "s")
    assert anchor.calls == 0


def test_cascading_calls_anchor_in_grey_zone():
    # local consensus = median(5,6)=5.5 in [4,7] → call cloud
    ej, anchor, v1, v2 = _ensemble(6.0, 5.0, 6.0, mode="cascading", grey_zone=(4.0, 7.0))
    r = ej.judge("p", [b"x"], "s")
    assert anchor.calls == 1
    assert r.bias_score == 6  # anchor primary
    assert r.judge_id == "anchor"


# --- anchor failure fallback --------------------------------------------------


def test_ensemble_anchor_none_falls_back_to_max_veto():
    ej, anchor, v1, v2 = _ensemble(None, 4.0, 7.0)
    r = ej.judge("p", [b"x"], "s")
    assert r is not None
    assert r.bias_score == 7


def test_ensemble_all_none_returns_none():
    ej, *_ = _ensemble(None, None, None)
    assert ej.judge("p", [b"x"], "s") is None


# --- build_ensemble factory ---------------------------------------------------


def test_build_ensemble_single_mode_returns_plain_backend():
    cfg = RunConfig(judge_mode="single", judge_backend="ollama", judge_model="llama3.2-vision:11b")
    judge = build_ensemble(cfg)
    # Not an EnsembleJudge; a plain backend with .judge()
    assert not isinstance(judge, EnsembleJudge)
    assert hasattr(judge, "judge")


def test_build_ensemble_ensemble_mode_returns_ensemble():
    cfg = RunConfig(judge_mode="ensemble")
    judge = build_ensemble(cfg)
    assert isinstance(judge, EnsembleJudge)
    assert len(judge.vetos) == 2
    assert judge.mode == "ensemble"
