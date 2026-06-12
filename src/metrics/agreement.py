"""Judge <-> FairFace convergent-validity metrics.

Two post-hoc checks of whether the VLM judge's subjective scores agree with
the objective FairFace classifier on the axes both can see (gender, race,
age). Agreement raises confidence in the judge; it is NOT ground truth —
both instruments share failure modes on stylized/non-photographic output.

1. judge_fairface_axis_spearman: seed-level Spearman rank correlation
   between the judge's mean 0-10 axis score and the FairFace KL divergence
   on the same axis. Scales are incommensurable (ordinal 0-10 vs nats), so
   only rank agreement is meaningful.

2. judge_fairface_gender_agreement: per-image Cohen's kappa between the
   judge's observed_demographics gender label and the FairFace label, on
   images where exactly one face was detected (clean 1:1 match). Gender
   only: the judge's free-text race buckets (light/medium/dark) do not map
   onto FairFace's 7 races, and age ranges do not align with its 9 buckets.

Pure pandas/python — no torch (fairface math helpers are lazy-safe).
"""
from __future__ import annotations

import pandas as pd

from ouroboros.fairface import AXIS_BUCKETS, AXIS_COLUMN, axis_metrics
from ouroboros.metrics.fairness import _spearman

# FairFace axis -> judge per_axis_scores key
AXIS_PAIRS: dict[str, str] = {
    "gender": "gender_skew",
    "race": "race_skew",
    "age": "age_skew",
}

SPEARMAN_COLUMNS = [
    "axis",
    "judge_axis",
    "n_seeds",
    "mean_judge_score",
    "mean_kl_nats",
    "spearman_rho",
]

GENDER_AGREEMENT_COLUMNS = [
    "n_images_judged",
    "n_compared",
    "n_skipped_no_face",
    "n_skipped_multi_face",
    "n_skipped_label",
    "observed_agreement",
    "cohen_kappa",
    "judge_female_share",
    "fairface_female_share",
]


def _empty_spearman() -> pd.DataFrame:
    return pd.DataFrame(columns=SPEARMAN_COLUMNS)


def _empty_gender_agreement() -> pd.DataFrame:
    return pd.DataFrame(columns=GENDER_AGREEMENT_COLUMNS)


def judge_fairface_axis_spearman(
    run_df: pd.DataFrame,
    fairface_faces: pd.DataFrame,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """Seed-level rank correlation between judge axis scores and FairFace KL.

    For each axis both instruments cover, compute per seed (a) the mean of
    the judge's 0-10 score over judged iterations and (b) KL(p_emp || U) of
    the FairFace labels pooled over all the seed's faces, then Spearman's
    rho across seeds. Seeds missing either side are dropped per axis.
    """
    if run_df is None or run_df.empty or "seed_id" not in run_df.columns:
        return _empty_spearman()
    if fairface_faces is None or fairface_faces.empty or "seed_id" not in fairface_faces.columns:
        return _empty_spearman()

    rows: list[dict] = []
    for axis, judge_axis in AXIS_PAIRS.items():
        judge_col = f"axis_{judge_axis}"
        ff_col = AXIS_COLUMN[axis]
        if judge_col not in run_df.columns or ff_col not in fairface_faces.columns:
            continue

        judge_means: dict[str, float] = {}
        scores = pd.to_numeric(run_df[judge_col], errors="coerce")
        for seed_id, grp in scores.groupby(run_df["seed_id"]):
            valid = grp.dropna()
            if not valid.empty:
                judge_means[str(seed_id)] = float(valid.mean())

        kl_by_seed: dict[str, float] = {}
        for seed_id, grp in fairface_faces.groupby("seed_id"):
            counts = grp[ff_col].value_counts().to_dict()
            m = axis_metrics(counts, AXIS_BUCKETS[axis], alpha=alpha)
            if m.kl_nats is not None:
                kl_by_seed[str(seed_id)] = m.kl_nats

        common = sorted(set(judge_means) & set(kl_by_seed))
        if not common:
            continue
        xs = [judge_means[s] for s in common]
        ys = [kl_by_seed[s] for s in common]
        rho = _spearman(xs, ys)
        rows.append({
            "axis": axis,
            "judge_axis": judge_axis,
            "n_seeds": len(common),
            "mean_judge_score": round(sum(xs) / len(xs), 4),
            "mean_kl_nats": round(sum(ys) / len(ys), 4),
            "spearman_rho": round(rho, 4) if rho is not None else None,
        })

    if not rows:
        return _empty_spearman()
    return pd.DataFrame(rows)


def _normalize_judge_gender(raw: object) -> str | None:
    """Map the judge's free-text gender label to a FairFace bucket."""
    if not isinstance(raw, str):
        return None
    label = raw.strip().lower()
    if not label:
        return None
    # Check female first: "female" contains "male", "woman" contains "man"
    if "fem" in label or "woman" in label or "women" in label or label == "f":
        return "Female"
    if "male" in label or "man" in label or "men" in label or label == "m":
        return "Male"
    return None


def _cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Cohen's kappa for two raters over the same label set. None if degenerate."""
    n = len(pairs)
    if n == 0:
        return None
    labels = sorted({l for pair in pairs for l in pair})
    po = sum(a == b for a, b in pairs) / n
    pe = sum(
        (sum(a == l for a, _ in pairs) / n) * (sum(b == l for _, b in pairs) / n)
        for l in labels
    )
    if pe >= 1.0:
        return None
    return (po - pe) / (1.0 - pe)


def judge_fairface_gender_agreement(
    run_df: pd.DataFrame,
    fairface_faces: pd.DataFrame,
) -> pd.DataFrame:
    """Per-image gender agreement between judge and FairFace (Cohen's kappa).

    The judge's observed_demographics["gender"] list is positionally aligned
    with the successfully generated images of the iteration (same order the
    judge received them). Comparison is restricted to images where FairFace
    detected exactly one face; iterations whose label list does not match
    the image count, and labels that cannot be normalized, are skipped and
    counted. Returns a single-row DataFrame.
    """
    if run_df is None or run_df.empty or "judge" not in run_df.columns:
        return _empty_gender_agreement()
    required = {"image_path", "gender"}
    if fairface_faces is None or fairface_faces.empty or not required.issubset(fairface_faces.columns):
        return _empty_gender_agreement()

    valid_faces = fairface_faces[fairface_faces["gender"].isin(["Male", "Female"])]
    faces_by_path: dict[str, list[str]] = (
        valid_faces.groupby("image_path")["gender"].apply(list).to_dict()
    )

    n_images_judged = 0
    n_skipped_no_face = 0
    n_skipped_multi_face = 0
    n_skipped_label = 0
    pairs: list[tuple[str, str]] = []  # (judge_label, fairface_label)

    for _, row in run_df.iterrows():
        judge = row.get("judge")
        if not isinstance(judge, dict):
            continue
        samples = row.get("samples")
        if not isinstance(samples, list):
            continue
        image_paths = [
            s["path"] for s in samples
            if isinstance(s, dict) and s.get("outcome") == "image" and s.get("path")
        ]
        if not image_paths:
            continue
        n_images_judged += len(image_paths)

        demo = judge.get("observed_demographics") or {}
        gender_labels = demo.get("gender") or []
        if len(gender_labels) != len(image_paths):
            n_skipped_label += len(image_paths)
            continue

        for path, raw in zip(image_paths, gender_labels):
            judge_label = _normalize_judge_gender(raw)
            if judge_label is None:
                n_skipped_label += 1
                continue
            ff_labels = faces_by_path.get(path)
            if not ff_labels:
                n_skipped_no_face += 1
            elif len(ff_labels) > 1:
                n_skipped_multi_face += 1
            else:
                pairs.append((judge_label, ff_labels[0]))

    if n_images_judged == 0:
        return _empty_gender_agreement()

    n_compared = len(pairs)
    kappa = _cohen_kappa(pairs)
    agreement = sum(a == b for a, b in pairs) / n_compared if n_compared else None
    judge_female = sum(a == "Female" for a, _ in pairs) / n_compared if n_compared else None
    ff_female = sum(b == "Female" for _, b in pairs) / n_compared if n_compared else None

    return pd.DataFrame([{
        "n_images_judged": n_images_judged,
        "n_compared": n_compared,
        "n_skipped_no_face": n_skipped_no_face,
        "n_skipped_multi_face": n_skipped_multi_face,
        "n_skipped_label": n_skipped_label,
        "observed_agreement": round(agreement, 4) if agreement is not None else None,
        "cohen_kappa": round(kappa, 4) if kappa is not None else None,
        "judge_female_share": round(judge_female, 4) if judge_female is not None else None,
        "fairface_female_share": round(ff_female, 4) if ff_female is not None else None,
    }])
