"""Tests for ouroboros.metrics.agreement (judge <-> FairFace convergent validity).

Pure pandas — no torch, no FairFace weights needed.
"""
from __future__ import annotations

import pandas as pd
import pytest

from ouroboros.metrics.agreement import (
    _cohen_kappa,
    _normalize_judge_gender,
    judge_fairface_axis_spearman,
    judge_fairface_gender_agreement,
)


# --- helpers --------------------------------------------------------------------


def _face_rows(seed_id: str, image_path: str, genders: list[str]) -> list[dict]:
    return [
        {
            "seed_id": seed_id,
            "image_path": image_path,
            "gender": g,
            "race": "White",
            "age_bucket": "30-39",
        }
        for g in genders
    ]


def _run_row(
    seed_id: str,
    image_paths: list[str],
    judge_genders: list[str] | None,
    gender_skew: float = 5.0,
) -> dict:
    return {
        "seed_id": seed_id,
        "category": "male_coded",
        # ``skew`` is the per-iteration batch gender skew the Spearman metric
        # ranks against the FairFace gender KL (rank-only, so any monotone
        # encoding of imbalance works as a fixture value).
        "skew": gender_skew,
        "samples": [{"path": p, "outcome": "image"} for p in image_paths],
        "judge": {
            "per_image_genders": judge_genders
            if judge_genders is not None
            else [],
        },
    }


# --- label normalization ----------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("male", "Male"),
        ("Male", "Male"),
        ("man", "Male"),
        ("M", "Male"),
        ("female", "Female"),
        ("woman", "Female"),
        ("F", "Female"),
        ("young woman", "Female"),
        ("androgynous", None),
        ("", None),
        (None, None),
        (3, None),
    ],
)
def test_normalize_judge_gender(raw, expected):
    assert _normalize_judge_gender(raw) == expected


# --- Cohen's kappa -----------------------------------------------------------------


def test_kappa_perfect_agreement():
    pairs = [("Male", "Male"), ("Female", "Female")] * 4
    assert _cohen_kappa(pairs) == pytest.approx(1.0)


def test_kappa_complete_disagreement_balanced():
    pairs = [("Male", "Female"), ("Female", "Male")] * 4
    assert _cohen_kappa(pairs) == pytest.approx(-1.0)


def test_kappa_degenerate_marginals_is_none():
    # Both raters always say Male → pe = 1 → kappa undefined
    pairs = [("Male", "Male")] * 5
    assert _cohen_kappa(pairs) is None


def test_kappa_empty_is_none():
    assert _cohen_kappa([]) is None


# --- seed-level Spearman -------------------------------------------------------------


def test_spearman_perfect_monotonic():
    # 3 seeds: judge gender_skew ordering matches FairFace imbalance ordering
    run_df = pd.DataFrame([
        _run_row("s_balanced", ["img/b.png"], None, gender_skew=2.0),
        _run_row("s_mid", ["img/m.png"], None, gender_skew=6.0),
        _run_row("s_skewed", ["img/s.png"], None, gender_skew=9.0),
    ])
    faces = (
        _face_rows("s_balanced", "img/b.png", ["Male", "Female", "Male", "Female"])
        + _face_rows("s_mid", "img/m.png", ["Male", "Male", "Male", "Female"])
        + _face_rows("s_skewed", "img/s.png", ["Male", "Male", "Male", "Male"])
    )
    out = judge_fairface_axis_spearman(run_df, pd.DataFrame(faces))

    gender = out[out["axis"] == "gender"]
    assert len(gender) == 1
    row = gender.iloc[0]
    assert row["judge_axis"] == "gender_skew"
    assert row["n_seeds"] == 3
    assert row["spearman_rho"] == pytest.approx(1.0)


def test_spearman_inverse_monotonic():
    run_df = pd.DataFrame([
        _run_row("s1", [], None, gender_skew=9.0),
        _run_row("s2", [], None, gender_skew=5.0),
        _run_row("s3", [], None, gender_skew=1.0),
    ])
    faces = (
        _face_rows("s1", "a.png", ["Male", "Female"])      # KL = 0
        + _face_rows("s2", "b.png", ["Male", "Male", "Male", "Female"])
        + _face_rows("s3", "c.png", ["Male", "Male", "Male", "Male"])
    )
    out = judge_fairface_axis_spearman(run_df, pd.DataFrame(faces))
    assert out[out["axis"] == "gender"].iloc[0]["spearman_rho"] == pytest.approx(-1.0)


def test_spearman_skips_seeds_missing_either_side():
    run_df = pd.DataFrame([
        _run_row("s1", [], None, gender_skew=3.0),
        _run_row("s2", [], None, gender_skew=8.0),
        _run_row("s_no_faces", [], None, gender_skew=7.0),
    ])
    faces = (
        _face_rows("s1", "a.png", ["Male", "Female"])
        + _face_rows("s2", "b.png", ["Male", "Male"])
        + _face_rows("s_not_in_run", "x.png", ["Female"])
    )
    out = judge_fairface_axis_spearman(run_df, pd.DataFrame(faces))
    assert out[out["axis"] == "gender"].iloc[0]["n_seeds"] == 2


def test_spearman_empty_inputs():
    assert judge_fairface_axis_spearman(pd.DataFrame(), pd.DataFrame()).empty
    run_df = pd.DataFrame([_run_row("s1", [], None)])
    assert judge_fairface_axis_spearman(run_df, pd.DataFrame()).empty


# --- per-image gender agreement -------------------------------------------------------


def test_gender_agreement_perfect():
    run_df = pd.DataFrame([
        _run_row("s1", ["i1.png", "i2.png"], ["male", "female"]),
        _run_row("s2", ["i3.png", "i4.png"], ["man", "woman"]),
    ])
    faces = (
        _face_rows("s1", "i1.png", ["Male"])
        + _face_rows("s1", "i2.png", ["Female"])
        + _face_rows("s2", "i3.png", ["Male"])
        + _face_rows("s2", "i4.png", ["Female"])
    )
    out = judge_fairface_gender_agreement(run_df, pd.DataFrame(faces))

    assert len(out) == 1
    row = out.iloc[0]
    assert row["n_images_judged"] == 4
    assert row["n_compared"] == 4
    assert row["observed_agreement"] == pytest.approx(1.0)
    assert row["cohen_kappa"] == pytest.approx(1.0)
    assert row["judge_female_share"] == pytest.approx(0.5)
    assert row["fairface_female_share"] == pytest.approx(0.5)


def test_gender_agreement_skips_multi_and_no_face():
    run_df = pd.DataFrame([
        _run_row("s1", ["one.png", "crowd.png", "empty.png"], ["male", "female", "male"]),
    ])
    faces = (
        _face_rows("s1", "one.png", ["Male"])
        + _face_rows("s1", "crowd.png", ["Female", "Male"])  # 2 faces → skipped
        # empty.png: no detected face → skipped
    )
    out = judge_fairface_gender_agreement(run_df, pd.DataFrame(faces))
    row = out.iloc[0]
    assert row["n_images_judged"] == 3
    assert row["n_compared"] == 1
    assert row["n_skipped_multi_face"] == 1
    assert row["n_skipped_no_face"] == 1


def test_gender_agreement_skips_length_mismatch_and_bad_labels():
    run_df = pd.DataFrame([
        # 2 images but only 1 judge label → whole iteration skipped
        _run_row("s1", ["a.png", "b.png"], ["male"]),
        # unparseable label → that image skipped
        _run_row("s2", ["c.png"], ["androgynous"]),
    ])
    faces = (
        _face_rows("s1", "a.png", ["Male"])
        + _face_rows("s1", "b.png", ["Male"])
        + _face_rows("s2", "c.png", ["Male"])
    )
    out = judge_fairface_gender_agreement(run_df, pd.DataFrame(faces))
    row = out.iloc[0]
    assert row["n_images_judged"] == 3
    assert row["n_compared"] == 0
    assert row["n_skipped_label"] == 3
    assert row["cohen_kappa"] is None or pd.isna(row["cohen_kappa"])


def test_gender_agreement_disagreement_kappa():
    # 4 comparisons, 2 agree / 2 disagree, balanced marginals → po=0.5, pe=0.5 → kappa=0
    run_df = pd.DataFrame([
        _run_row("s1", ["1.png", "2.png", "3.png", "4.png"],
                 ["male", "female", "male", "female"]),
    ])
    faces = (
        _face_rows("s1", "1.png", ["Male"])      # agree
        + _face_rows("s1", "2.png", ["Female"])  # agree
        + _face_rows("s1", "3.png", ["Female"])  # disagree
        + _face_rows("s1", "4.png", ["Male"])    # disagree
    )
    out = judge_fairface_gender_agreement(run_df, pd.DataFrame(faces))
    row = out.iloc[0]
    assert row["observed_agreement"] == pytest.approx(0.5)
    assert row["cohen_kappa"] == pytest.approx(0.0)


def test_gender_agreement_empty_inputs():
    assert judge_fairface_gender_agreement(pd.DataFrame(), pd.DataFrame()).empty
    run_df = pd.DataFrame([_run_row("s1", ["a.png"], ["male"])])
    assert judge_fairface_gender_agreement(run_df, pd.DataFrame()).empty
