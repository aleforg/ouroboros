"""Distributional fairness-gap metrics.

These helpers are post-hoc computations over FairFace aggregate metrics. They
do not call an LLM/VLM and do not use external stereotype priors.
"""
from __future__ import annotations

import pandas as pd


def distribution_gap_summary(fairface_metrics: pd.DataFrame) -> pd.DataFrame:
    """Compute max-min KL gaps across categories for each FairFace axis.

    Input is the output of ``fairface.compute_kl_metrics`` with columns such as
    ``kl_gender_nats``. Output columns:
      axis, n_categories, min_category, min_kl, max_category, max_kl, gap_kl_nats
    """
    if fairface_metrics is None or fairface_metrics.empty:
        return pd.DataFrame(columns=[
            "axis",
            "n_categories",
            "min_category",
            "min_kl",
            "max_category",
            "max_kl",
            "gap_kl_nats",
        ])

    rows: list[dict] = []
    for axis in ("gender", "race", "age"):
        col = f"kl_{axis}_nats"
        if col not in fairface_metrics.columns:
            continue
        work = fairface_metrics[["category", col]].dropna()
        if work.empty:
            continue
        min_row = work.loc[work[col].idxmin()]
        max_row = work.loc[work[col].idxmax()]
        min_kl = float(min_row[col])
        max_kl = float(max_row[col])
        rows.append({
            "axis": axis,
            "n_categories": int(work["category"].nunique()),
            "min_category": min_row["category"],
            "min_kl": round(min_kl, 4),
            "max_category": max_row["category"],
            "max_kl": round(max_kl, 4),
            "gap_kl_nats": round(max_kl - min_kl, 4),
        })

    return pd.DataFrame(rows)
