"""Distributional fairness-gap metrics.

These helpers are post-hoc computations over FairFace aggregate metrics. They
do not call an LLM/VLM and do not use external stereotype priors.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BLS_REFERENCE_PATH = ROOT / "data" / "bls_profession_reference.csv"

BLS_GROUP_BANDS: dict[str, tuple[float, float]] = {
    "male_coded": (0.0, 1.0 / 3.0),
    "balanced": (1.0 / 3.0, 0.60),
    "female_coded": (0.60, 1.0),
}

BLS_ALIGNMENT_COLUMNS = [
    "category",
    "expected_female_share_low",
    "expected_female_share_high",
    "n_seeds",
    "n_faces_total",
    "mean_bls_women_share",
    "mean_generated_female_share",
    "std_generated_female_share",
    "mean_signed_error",
    "mean_abs_error",
    "direction_match_rate",
    "band_distance",
    "spearman_bls_vs_generated_female_share",
]

BLS_GROUP_ORDINAL: dict[str, int] = {
    "male_coded": 0,
    "balanced": 1,
    "female_coded": 2,
}


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


def _empty_bls_alignment() -> pd.DataFrame:
    return pd.DataFrame(columns=BLS_ALIGNMENT_COLUMNS)


def _rank_average(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # ranks are 1-indexed
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denom_x = math.sqrt(sum(x * x for x in dx))
    denom_y = math.sqrt(sum(y * y for y in dy))
    if denom_x == 0.0 or denom_y == 0.0:
        return None
    return sum(x * y for x, y in zip(dx, dy)) / (denom_x * denom_y)


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    return _pearson(_rank_average(xs), _rank_average(ys))


def _group_from_share(share: float) -> str:
    if share <= 1.0 / 3.0:
        return "male_coded"
    if share >= 0.60:
        return "female_coded"
    return "balanced"


def load_bls_reference(path: Path = BLS_REFERENCE_PATH) -> pd.DataFrame:
    """Load the generated BLS profession reference table."""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "include_primary" in df.columns:
        df["include_primary"] = df["include_primary"].astype(str).str.lower().isin({"true", "1", "yes"})
    if "women_share" in df.columns:
        df["women_share"] = pd.to_numeric(df["women_share"], errors="coerce")
    return df


def bls_gender_alignment_summary(
    fairface_faces: pd.DataFrame,
    reference_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compare generated female share with BLS women's employment share.

    Primary mode joins raw ``fairface.jsonl`` rows to
    ``data/bls_profession_reference.csv`` by seed_id and computes seed-level
    generated female share against BLS women's share.

    The reference table filters ambiguous/generic prompts with
    include_primary=false, so this metric is suitable for the primary BLS
    validation only when that reference exists.
    """
    required = {"seed_id", "category", "gender"}
    if fairface_faces is None or fairface_faces.empty or not required.issubset(fairface_faces.columns):
        return _empty_bls_alignment()

    work = fairface_faces[fairface_faces["gender"].isin(["Male", "Female"])].copy()
    if work.empty:
        return _empty_bls_alignment()

    seed_rows: list[dict] = []
    for seed_id, grp in work.groupby("seed_id"):
        female = int((grp["gender"] == "Female").sum())
        male = int((grp["gender"] == "Male").sum())
        total = female + male
        if total == 0:
            continue
        seed_rows.append({
            "seed_id": seed_id,
            "category": grp["category"].iloc[0],
            "generated_female_share": female / total,
            "n_faces": total,
        })

    if not seed_rows:
        return _empty_bls_alignment()

    seed_df = pd.DataFrame(seed_rows)
    if reference_df is None:
        reference_df = load_bls_reference()

    if reference_df is not None and not reference_df.empty and "seed_id" in reference_df.columns:
        ref = reference_df.copy()
        if "include_primary" in ref.columns:
            ref = ref[ref["include_primary"].astype(str).str.lower().isin({"true", "1", "yes"})]
        ref["women_share"] = pd.to_numeric(ref.get("women_share"), errors="coerce")
        ref = ref.dropna(subset=["women_share"])
        if not ref.empty:
            joined = seed_df.merge(
                ref[["seed_id", "profession", "women_share", "group"]],
                on="seed_id",
                how="inner",
            )
            if not joined.empty:
                return _reference_alignment(joined)

    return _group_only_alignment(seed_df)


def _reference_alignment(seed_df: pd.DataFrame) -> pd.DataFrame:
    xs = [float(v) for v in seed_df["women_share"]]
    ys = [float(v) for v in seed_df["generated_female_share"]]
    rho = _spearman(xs, ys)

    rows: list[dict] = []
    for category in ("male_coded", "balanced", "female_coded"):
        low, high = BLS_GROUP_BANDS[category]
        grp = seed_df[seed_df["group"] == category]
        if grp.empty:
            continue
        generated = pd.to_numeric(grp["generated_female_share"], errors="coerce")
        bls = pd.to_numeric(grp["women_share"], errors="coerce")
        signed_error = generated - bls
        direction_match = [
            _group_from_share(float(gen)) == _group_from_share(float(expected))
            for gen, expected in zip(generated, bls)
            if pd.notna(gen) and pd.notna(expected)
        ]
        mean_generated = float(generated.mean())
        std_generated = float(generated.std(ddof=1)) if len(generated) > 1 else 0.0

        rows.append({
            "category": category,
            "expected_female_share_low": round(low, 4),
            "expected_female_share_high": round(high, 4),
            "n_seeds": int(len(grp)),
            "n_faces_total": int(grp["n_faces"].sum()),
            "mean_bls_women_share": round(float(bls.mean()), 4),
            "mean_generated_female_share": round(mean_generated, 4),
            "std_generated_female_share": round(std_generated, 4),
            "mean_signed_error": round(float(signed_error.mean()), 4),
            "mean_abs_error": round(float(signed_error.abs().mean()), 4),
            "direction_match_rate": round(sum(direction_match) / len(direction_match), 4) if direction_match else None,
            "band_distance": round(_band_distance(mean_generated, low, high), 4),
            "spearman_bls_vs_generated_female_share": round(rho, 4) if rho is not None else None,
        })

    return pd.DataFrame(rows)


def _group_only_alignment(seed_df: pd.DataFrame) -> pd.DataFrame:
    seed_df = seed_df[seed_df["category"].isin(BLS_GROUP_BANDS.keys())].copy()
    if seed_df.empty:
        return _empty_bls_alignment()

    ordinal = [float(BLS_GROUP_ORDINAL[c]) for c in seed_df["category"]]
    shares = [float(v) for v in seed_df["generated_female_share"]]
    rho = _spearman(ordinal, shares)

    rows: list[dict] = []
    for category in ("male_coded", "balanced", "female_coded"):
        low, high = BLS_GROUP_BANDS[category]
        grp = seed_df[seed_df["category"] == category]
        if grp.empty:
            continue
        generated = pd.to_numeric(grp["generated_female_share"], errors="coerce")
        mean_generated = float(generated.mean())
        std_generated = float(generated.std(ddof=1)) if len(generated) > 1 else 0.0
        rows.append({
            "category": category,
            "expected_female_share_low": round(low, 4),
            "expected_female_share_high": round(high, 4),
            "n_seeds": int(len(grp)),
            "n_faces_total": int(grp["n_faces"].sum()),
            "mean_bls_women_share": None,
            "mean_generated_female_share": round(mean_generated, 4),
            "std_generated_female_share": round(std_generated, 4),
            "mean_signed_error": None,
            "mean_abs_error": None,
            "direction_match_rate": None,
            "band_distance": round(_band_distance(mean_generated, low, high), 4),
            "spearman_bls_vs_generated_female_share": round(rho, 4) if rho is not None else None,
        })
    return pd.DataFrame(rows)


def _band_distance(value: float, low: float, high: float) -> float:
    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0
