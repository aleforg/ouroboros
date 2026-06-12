from __future__ import annotations

import pandas as pd

from ouroboros.metrics.fairness import bls_gender_alignment_summary, distribution_gap_summary


def test_distribution_gap_summary_max_min_by_axis():
    df = pd.DataFrame([
        {
            "category": "male_coded",
            "kl_gender_nats": 0.4,
            "kl_race_nats": 0.1,
            "kl_age_nats": 0.2,
        },
        {
            "category": "female_coded",
            "kl_gender_nats": 0.1,
            "kl_race_nats": 0.7,
            "kl_age_nats": 0.3,
        },
    ])

    out = distribution_gap_summary(df)
    gender = out[out["axis"] == "gender"].iloc[0]

    assert gender["min_category"] == "female_coded"
    assert gender["max_category"] == "male_coded"
    assert gender["gap_kl_nats"] == 0.3


def test_bls_gender_alignment_summary_from_raw_fairface_rows():
    df = pd.DataFrame([
        {"seed_id": "m1", "category": "male_coded", "gender": "Male"},
        {"seed_id": "m1", "category": "male_coded", "gender": "Male"},
        {"seed_id": "m2", "category": "male_coded", "gender": "Male"},
        {"seed_id": "m2", "category": "male_coded", "gender": "Female"},
        {"seed_id": "b1", "category": "balanced", "gender": "Male"},
        {"seed_id": "b1", "category": "balanced", "gender": "Female"},
        {"seed_id": "f1", "category": "female_coded", "gender": "Female"},
        {"seed_id": "f1", "category": "female_coded", "gender": "Female"},
        {"seed_id": "f2", "category": "female_coded", "gender": "Female"},
        {"seed_id": "f2", "category": "female_coded", "gender": "Male"},
    ])
    reference = pd.DataFrame([
        {"seed_id": "m1", "profession": "p1", "women_share": 0.10, "group": "male_coded", "include_primary": True},
        {"seed_id": "m2", "profession": "p2", "women_share": 0.20, "group": "male_coded", "include_primary": True},
        {"seed_id": "b1", "profession": "p3", "women_share": 0.50, "group": "balanced", "include_primary": True},
        {"seed_id": "f1", "profession": "p4", "women_share": 0.80, "group": "female_coded", "include_primary": True},
        {"seed_id": "f2", "profession": "p5", "women_share": 0.70, "group": "female_coded", "include_primary": True},
    ])

    out = bls_gender_alignment_summary(df, reference_df=reference)

    male = out[out["category"] == "male_coded"].iloc[0]
    balanced = out[out["category"] == "balanced"].iloc[0]
    female = out[out["category"] == "female_coded"].iloc[0]

    assert male["mean_generated_female_share"] == 0.25
    assert male["mean_bls_women_share"] == 0.15
    assert male["mean_signed_error"] == 0.1
    assert balanced["mean_generated_female_share"] == 0.5
    assert female["mean_generated_female_share"] == 0.75
    assert female["direction_match_rate"] == 0.5
    assert out["spearman_bls_vs_generated_female_share"].iloc[0] > 0
