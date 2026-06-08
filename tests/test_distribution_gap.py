from __future__ import annotations

import pandas as pd

from ouroboros.metrics.fairness import distribution_gap_summary


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
