from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from mirtage import fairface as ff
from mirtage.fairface import (
    AGE_BUCKETS,
    AXIS_BUCKETS,
    GENDER_BUCKETS,
    RACE_BUCKETS,
    AxisMetrics,
    DetectedFace,
    _entropy_nats,
    _FAIRFACE_AGE_ORDER,
    _FAIRFACE_GENDER_ORDER,
    _FAIRFACE_RACE_ORDER,
    _LABEL_NORMALIZE,
    _load_image_index,
    _normalize_label,
    _smoothed_distribution,
    axis_metrics,
    compute_kl_metrics,
    load_fairface,
    process_run,
)


# --- low-level math -----------------------------------------------------------


def test_smoothed_distribution_uniform_input():
    p = _smoothed_distribution([2, 2, 2, 2], k=4, alpha=1.0)
    assert all(abs(pi - 0.25) < 1e-9 for pi in p)
    assert abs(sum(p) - 1.0) < 1e-9


def test_smoothed_distribution_handles_zero_counts():
    # All counts zero: Laplace smoothing makes it uniform
    p = _smoothed_distribution([0, 0, 0, 0], k=4, alpha=1.0)
    assert all(abs(pi - 0.25) < 1e-9 for pi in p)


def test_smoothed_distribution_alpha_zero_is_mle():
    p = _smoothed_distribution([3, 1, 0, 0], k=4, alpha=0.0)
    assert p == [0.75, 0.25, 0.0, 0.0]


def test_entropy_nats_uniform_is_log_k():
    for k in (2, 4, 7, 9):
        p = [1.0 / k] * k
        assert abs(_entropy_nats(p) - math.log(k)) < 1e-9


def test_entropy_nats_degenerate_is_zero():
    assert _entropy_nats([1.0, 0.0, 0.0]) == 0.0


def test_entropy_nats_ignores_zeros_no_nan():
    # log(0) would NaN; helper must treat 0·log0 = 0
    h = _entropy_nats([0.5, 0.5, 0.0, 0.0])
    assert abs(h - math.log(2)) < 1e-9


# --- axis_metrics -------------------------------------------------------------


def test_axis_metrics_uniform_counts_give_zero_kl():
    counts = {b: 10 for b in GENDER_BUCKETS}
    m = axis_metrics(counts, GENDER_BUCKETS)
    assert m.n_samples == 20
    assert m.kl_nats < 1e-6
    assert abs(m.norm_entropy - 1.0) < 1e-6


def test_axis_metrics_degenerate_approaches_log_k():
    # All 100 samples in one bucket (out of 2) → KL approaches log(2)
    counts = {"Male": 100, "Female": 0}
    m = axis_metrics(counts, GENDER_BUCKETS, alpha=1.0)
    assert m.n_samples == 100
    # With Laplace smoothing: p = [101/102, 1/102]; KL slightly below log(2)
    assert 0.6 < m.kl_nats < math.log(2)
    assert 0.0 < m.norm_entropy < 0.1


def test_axis_metrics_no_samples_returns_none():
    m = axis_metrics({}, RACE_BUCKETS)
    assert m == AxisMetrics(kl_nats=None, norm_entropy=None, n_samples=0)


def test_axis_metrics_ignores_unknown_buckets():
    # 'Klingon' is not in the FairFace race set; it should be silently dropped
    counts = {"White": 5, "Klingon": 99}
    m = axis_metrics(counts, RACE_BUCKETS)
    assert m.n_samples == 5


def test_axis_metrics_kl_nonnegative_floor():
    # Floating-point near-zero KL must not be returned as a tiny negative
    counts = {b: 1000 for b in AGE_BUCKETS}
    m = axis_metrics(counts, AGE_BUCKETS)
    assert m.kl_nats >= 0.0


def test_axis_metrics_requires_at_least_two_buckets():
    with pytest.raises(ValueError):
        axis_metrics({"x": 1}, ["x"])


def test_axis_metrics_norm_entropy_in_unit_interval():
    counts = {"White": 5, "Black": 3, "East_Asian": 2}
    m = axis_metrics(counts, RACE_BUCKETS)
    assert 0.0 <= m.norm_entropy <= 1.0


# --- compute_kl_metrics aggregator --------------------------------------------


def _ff_row(category: str, image_path: str, gender: str, race: str, age: str) -> dict:
    return {
        "category": category,
        "image_path": image_path,
        "gender": gender,
        "race": race,
        "age_bucket": age,
    }


def test_compute_kl_metrics_empty_input():
    out = compute_kl_metrics(pd.DataFrame())
    assert out.empty


def test_compute_kl_metrics_single_category_balanced():
    rows = [
        _ff_row("gender", "img1.png", "Male", "White", "30-39"),
        _ff_row("gender", "img2.png", "Female", "Black", "30-39"),
    ]
    out = compute_kl_metrics(pd.DataFrame(rows))
    assert len(out) == 1
    row = out.iloc[0]
    assert row["category"] == "gender"
    assert row["n_with_faces"] == 2
    assert row["n_faces_total"] == 2
    # 1 Male + 1 Female with alpha=1 → smoothed [2/4, 2/4] → KL=0
    assert row["kl_gender_nats"] == 0.0
    assert row["norm_entropy_gender"] == 1.0
    # Race skew: 1 White + 1 Black + 5 zero buckets → some skew
    assert row["kl_race_nats"] > 0.0
    assert row["norm_entropy_race"] < 1.0


def test_compute_kl_metrics_multiple_categories():
    # Use 20 faces per category so Laplace smoothing (alpha=1) doesn't dominate
    rows = [_ff_row("gender", f"g{i}.png", "Male", "White", "30-39") for i in range(20)]
    rows += [
        _ff_row("race", f"r{i}.png", "Female", "Black", "20-29") for i in range(10)
    ] + [
        _ff_row("race", f"r{10 + i}.png", "Female", "East_Asian", "20-29")
        for i in range(10)
    ]
    out = compute_kl_metrics(pd.DataFrame(rows))
    assert sorted(out["category"].tolist()) == ["gender", "race"]

    gender_row = out[out["category"] == "gender"].iloc[0]
    # 20 Male / 0 Female with alpha=1 → p ≈ [21/22, 1/22] → KL close to log(2)
    assert gender_row["kl_gender_nats"] > 0.5
    assert gender_row["norm_entropy_gender"] < 0.3

    race_row = out[out["category"] == "race"].iloc[0]
    # All female (1 of 2 gender buckets fully populated) → high gender skew
    assert race_row["kl_gender_nats"] > 0.5


def test_compute_kl_metrics_n_images_from_run_df():
    # 3 total iterations but only 1 image had a face detected
    ff = pd.DataFrame([_ff_row("gender", "iter0_s0.png", "Male", "White", "30-39")])
    run = pd.DataFrame(
        [
            {
                "category": "gender",
                "samples": [
                    {"path": "iter0_s0.png", "outcome": "image"},
                    {"path": "iter0_s1.png", "outcome": "image"},
                    {"path": None, "outcome": "error"},
                ],
            },
            {
                "category": "gender",
                "samples": [
                    {"path": "iter1_s0.png", "outcome": "image"},
                ],
            },
        ]
    )
    out = compute_kl_metrics(ff, run_df=run)
    row = out.iloc[0]
    assert row["n_with_faces"] == 1
    assert row["n_images"] == 3  # 2 + 1 valid paths across samples
    assert row["n_faces_total"] == 1


def test_compute_kl_metrics_unknown_axis_value_ignored():
    # If FairFace returns a label outside the canonical set, drop it
    rows = [
        _ff_row("gender", "img1.png", "NonBinary", "Vulcan", "200+"),
        _ff_row("gender", "img2.png", "Male", "White", "30-39"),
    ]
    out = compute_kl_metrics(pd.DataFrame(rows))
    row = out.iloc[0]
    # Only 1 valid face per axis remains
    # KL and entropy should still be computable (1 sample is enough)
    assert row["kl_gender_nats"] is not None
    assert row["norm_entropy_gender"] is not None


def test_compute_kl_metrics_axis_bucket_constants_have_expected_sizes():
    # Sanity: the canonical FairFace bucket sizes used in the literature
    assert len(AXIS_BUCKETS["gender"]) == 2
    assert len(AXIS_BUCKETS["race"]) == 7
    assert len(AXIS_BUCKETS["age"]) == 9


# --- label normalization (FairFace raw → canonical) ---------------------------


def test_normalize_label_handles_known_mappings():
    assert _normalize_label("East Asian") == "East_Asian"
    assert _normalize_label("Southeast Asian") == "Southeast_Asian"
    assert _normalize_label("Middle Eastern") == "Middle_Eastern"
    assert _normalize_label("more than 70") == "70+"


def test_normalize_label_passes_through_already_canonical():
    assert _normalize_label("Male") == "Male"
    assert _normalize_label("White") == "White"
    assert _normalize_label("30-39") == "30-39"


def test_fairface_raw_orders_match_classifier_head_size():
    # The ResNet-34 FC outputs 18 logits in this exact split
    assert len(_FAIRFACE_RACE_ORDER) == 7
    assert len(_FAIRFACE_GENDER_ORDER) == 2
    assert len(_FAIRFACE_AGE_ORDER) == 9


def test_every_raw_label_normalizes_to_a_canonical_bucket():
    # Every raw FairFace label, post-normalization, must be a valid bucket
    for raw in _FAIRFACE_RACE_ORDER:
        assert _normalize_label(raw) in RACE_BUCKETS
    for raw in _FAIRFACE_GENDER_ORDER:
        assert _normalize_label(raw) in GENDER_BUCKETS
    for raw in _FAIRFACE_AGE_ORDER:
        assert _normalize_label(raw) in AGE_BUCKETS


def test_label_normalize_targets_are_canonical():
    # Every value in the rename map must itself be a valid bucket
    all_canonical = set(GENDER_BUCKETS) | set(RACE_BUCKETS) | set(AGE_BUCKETS)
    for canonical in _LABEL_NORMALIZE.values():
        assert canonical in all_canonical, f"Rename target {canonical!r} not in any bucket set"


# --- _load_image_index --------------------------------------------------------


def _write_run_jsonl(run_dir: Path, records: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "run.jsonl").open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def test_load_image_index_skips_null_paths_and_errors(tmp_path):
    _write_run_jsonl(tmp_path, [
        {
            "seed_id": "gender_001", "category": "gender", "iter": 0,
            "samples": [
                {"path": "images/gender_001/iter_00/sample_0.png", "outcome": "image"},
                {"path": None, "outcome": "error"},
                {"path": "images/gender_001/iter_00/sample_2.png", "outcome": "image"},
            ],
        },
        {
            "seed_id": "gender_001", "category": "gender", "iter": 1,
            "samples": [{"path": "images/gender_001/iter_01/sample_0.png", "outcome": "image"}],
        },
    ])
    idx = _load_image_index(tmp_path)
    assert len(idx) == 3
    assert "images/gender_001/iter_00/sample_0.png" in idx
    assert idx["images/gender_001/iter_01/sample_0.png"]["iter"] == 1
    assert idx["images/gender_001/iter_00/sample_2.png"]["sample_idx"] == 2


def test_load_image_index_missing_run_jsonl_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load_image_index(tmp_path)


def test_load_image_index_handles_missing_samples_field(tmp_path):
    # Records like attacker_refused have no samples
    _write_run_jsonl(tmp_path, [
        {"seed_id": "x", "category": "gender", "iter": 0, "samples": []},
        {"seed_id": "y", "category": "gender", "iter": 0},
    ])
    idx = _load_image_index(tmp_path)
    assert idx == {}


# --- process_run (with detect/classify mocked) --------------------------------


@pytest.fixture
def fake_run(tmp_path):
    """Build a minimal run_dir with run.jsonl + 2 dummy PNG files on disk."""
    img_dir = tmp_path / "images" / "gender_001" / "iter_00"
    img_dir.mkdir(parents=True)
    (img_dir / "sample_0.png").write_bytes(b"\x89PNG\r\n\x1a\n")  # placeholder
    (img_dir / "sample_1.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    _write_run_jsonl(tmp_path, [
        {
            "seed_id": "gender_001", "category": "gender", "iter": 0,
            "samples": [
                {"path": "images/gender_001/iter_00/sample_0.png", "outcome": "image"},
                {"path": "images/gender_001/iter_00/sample_1.png", "outcome": "image"},
            ],
        },
    ])
    return tmp_path


def test_process_run_writes_one_row_per_face(fake_run):
    # Mock: each image has 2 detected faces, both classified the same
    fake_face = DetectedFace(bbox=(10.0, 20.0, 110.0, 220.0), confidence=0.98, face_pil=object())

    with patch.object(ff, "_load_models"), \
         patch.object(ff, "detect_faces", return_value=[fake_face, fake_face]) as m_det, \
         patch.object(ff, "classify", return_value={"gender": "Male", "race": "White", "age_bucket": "30-39"}) as m_clf:
        n = process_run(fake_run, show_progress=False)

    assert n == 4  # 2 images × 2 faces
    assert m_det.call_count == 2
    assert m_clf.call_count == 4

    df = load_fairface(fake_run)
    assert len(df) == 4
    assert set(df["image_path"].unique()) == {
        "images/gender_001/iter_00/sample_0.png",
        "images/gender_001/iter_00/sample_1.png",
    }
    assert set(df["face_idx"].unique()) == {0, 1}
    assert df["category"].unique().tolist() == ["gender"]


def test_process_run_skips_detection_failures(fake_run):
    fake_face = DetectedFace(bbox=(0, 0, 100, 100), confidence=0.95, face_pil=object())

    def detect_side_effect(png_bytes, **kwargs):
        # First image: detection succeeds. Second: raises.
        if detect_side_effect.calls == 0:
            detect_side_effect.calls += 1
            return [fake_face]
        raise RuntimeError("Boom")
    detect_side_effect.calls = 0

    with patch.object(ff, "_load_models"), \
         patch.object(ff, "detect_faces", side_effect=detect_side_effect), \
         patch.object(ff, "classify", return_value={"gender": "Female", "race": "Black", "age_bucket": "20-29"}):
        n = process_run(fake_run, show_progress=False)

    assert n == 1  # only the first image succeeded


def test_process_run_image_missing_on_disk_is_skipped(fake_run):
    # Add a record pointing to a file that doesn't exist
    (fake_run / "images" / "gender_001" / "iter_00" / "sample_0.png").unlink()

    with patch.object(ff, "_load_models"), \
         patch.object(ff, "detect_faces", return_value=[]) as m_det:
        n = process_run(fake_run, show_progress=False)

    assert n == 0
    # Only the surviving file should have been opened
    assert m_det.call_count == 1


def test_process_run_no_images_in_jsonl(tmp_path):
    _write_run_jsonl(tmp_path, [
        {"seed_id": "x", "category": "gender", "iter": 0, "samples": []},
    ])
    n = process_run(tmp_path, show_progress=False)
    assert n == 0
    assert (tmp_path / "fairface.jsonl").exists()
    assert (tmp_path / "fairface.jsonl").read_text() == ""


def test_load_fairface_missing_file_returns_empty(tmp_path):
    df = load_fairface(tmp_path)
    assert df.empty
