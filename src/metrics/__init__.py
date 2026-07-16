from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from ouroboros.config import (
    EVALUABLE_OUTCOMES,
    LABEL_FAIL,
    LABEL_SUCCESS,
)
from ouroboros.judge import batch_skew, majority_gender_count

# Z value for 95% normal CI (kept for the Wilson helper below)
_Z_95 = 1.959963984540054

# Default bootstrap CI params — 2000 resamples is enough for ~3-decimal stability
# on n=10-15 categories without slowing the report perceptibly.
_BOOTSTRAP_N_RESAMPLES = 2000
_BOOTSTRAP_SEED = 42


# --- statistical helpers ------------------------------------------------------


def wilson_ci(k: int, n: int, z: float = _Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n.

    Kept as a helper for ad-hoc use; the report now uses bootstrap_ci instead
    (T2I-bias / red-teaming literature convention — Stable Bias, NeurIPS 2023).
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def bootstrap_ci(
    successes: Sequence[bool] | Sequence[int],
    n_resamples: int = _BOOTSTRAP_N_RESAMPLES,
    confidence: float = 0.95,
    seed: int = _BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Non-parametric percentile bootstrap CI on the rate sum/len.

    Resamples the binary indicator array with replacement `n_resamples` times,
    takes the percentile cut at (1-confidence)/2 and 1-(1-confidence)/2.
    Returns (low, high) bounded to [0, 1]. Deterministic via `seed`.

    For Bernoulli rates this is asymptotically equivalent to Wilson but is
    the convention reported in the T2I-bias literature (Luccioni et al.,
    NeurIPS 2023) and recent red-teaming work (arXiv:2505.20162). It also
    generalizes naturally if we ever move beyond binary success indicators.
    """
    n = len(successes)
    if n == 0:
        return (0.0, 1.0)
    rng = random.Random(seed)
    rates: list[float] = []
    for _ in range(n_resamples):
        resampled_sum = sum(successes[rng.randrange(n)] for _ in range(n))
        rates.append(resampled_sum / n)
    rates.sort()
    alpha = (1.0 - confidence) / 2.0
    low_idx = int(alpha * n_resamples)
    high_idx = min(n_resamples - 1, int((1.0 - alpha) * n_resamples))
    return (float(rates[low_idx]), float(rates[high_idx]))


def _mean_std(series: pd.Series) -> tuple[float | None, float | None, int]:
    """Return (mean, std, n) ignoring NaN. std is sample std (ddof=1)."""
    s = series.dropna()
    n = len(s)
    if n == 0:
        return (None, None, 0)
    mean = float(s.mean())
    std = float(s.std(ddof=1)) if n > 1 else 0.0
    return (mean, std, n)


def _median_iqr(series: pd.Series) -> tuple[float | None, float | None, int]:
    """Return (median, IQR, n) ignoring NaN. IQR=0 when n<4 (not meaningful)."""
    s = series.dropna()
    n = len(s)
    if n == 0:
        return (None, None, 0)
    median = float(s.median())
    if n < 4:
        return (median, 0.0, n)
    iqr = float(s.quantile(0.75) - s.quantile(0.25))
    return (median, iqr, n)


# --- IO -----------------------------------------------------------------------


def _flatten_judge(r: dict) -> dict:
    """Lift the GenderJudgement fields onto the top-level record.

    ``bias_score`` is the derived 0-10 batch skew (round(10·skew)); it is kept
    under that name for continuity with the ASR/summary helpers and the
    dashboard. ``per_image_genders``, ``female_share`` and ``skew`` are the
    label-based measures the v3.0 metrics operate on.
    """
    judge = r.get("judge") or {}
    r["per_image_genders"] = judge.get("per_image_genders", None)
    r["female_share"] = judge.get("female_share", None)
    r["skew"] = judge.get("skew", None)
    r["bias_score"] = judge.get("bias_score", None)
    return r


def load_run(run_dir: Path) -> pd.DataFrame:
    """Load run.jsonl into a flat DataFrame (one row per iteration).

    Handles the GenderJudgement schema (per_image_genders + derived female_share,
    skew, bias_score) used by the loop and report.
    """
    jsonl = run_dir / "run.jsonl"
    if not jsonl.exists():
        return pd.DataFrame()
    records = [
        _flatten_judge(json.loads(line))
        for line in jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return pd.DataFrame(records)


def load_baseline(run_dir: Path) -> pd.DataFrame:
    """Load baseline.jsonl if present."""
    jsonl = run_dir / "baseline.jsonl"
    if not jsonl.exists():
        return pd.DataFrame()
    records = [
        _flatten_judge(json.loads(line))
        for line in jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return pd.DataFrame(records)


# --- per-seed / per-category --------------------------------------------------


def is_evaluable(df: pd.DataFrame) -> pd.Series:
    """Boolean mask: iterations whose outcome carries information about the target.

    Measurement failures (judge_error / attacker_refused / error) are not
    evaluable — see ``EVALUABLE_OUTCOMES`` in ``ouroboros.config``.
    """
    if df.empty or "outcome" not in df:
        return pd.Series(dtype=bool)
    return df["outcome"].isin(EVALUABLE_OUTCOMES)


def censored_seeds(df: pd.DataFrame) -> list[str]:
    """Seed ids excluded from S': every iteration was a measurement failure."""
    if df.empty or "seed_id" not in df:
        return []
    evaluable = is_evaluable(df)
    per_seed = evaluable.groupby(df["seed_id"]).any()
    return sorted(per_seed[~per_seed].index.astype(str).tolist())


def censorship_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Censoring diagnostics for the ASR denominator.

    Reports how many seeds were dropped from S' because no iteration was
    evaluable, and the share of individual iterations that were measurement
    failures. Both are reported alongside ASR so that a high error rate of the
    instrument cannot silently masquerade as a low attack success rate.
    """
    if df.empty or "seed_id" not in df:
        return {
            "n_seeds_total": 0,
            "n_seeds_censored": 0,
            "n_seeds_evaluable": 0,
            "seed_censorship_rate": None,
            "n_iters_total": 0,
            "n_iters_non_evaluable": 0,
            "iter_censorship_rate": None,
            "censored_seed_ids": [],
        }
    censored = censored_seeds(df)
    n_total = int(df["seed_id"].nunique())
    n_censored = len(censored)
    evaluable = is_evaluable(df)
    n_iters = int(len(df))
    n_iters_bad = int((~evaluable).sum())
    return {
        "n_seeds_total": n_total,
        "n_seeds_censored": n_censored,
        "n_seeds_evaluable": n_total - n_censored,
        "seed_censorship_rate": round(n_censored / n_total, 4) if n_total else None,
        "n_iters_total": n_iters,
        "n_iters_non_evaluable": n_iters_bad,
        "iter_censorship_rate": round(n_iters_bad / n_iters, 4) if n_iters else None,
        "censored_seed_ids": censored,
    }


def summary_per_seed(df: pd.DataFrame) -> pd.DataFrame:
    """One row per seed in S': outcome, iters_to_success, max_bias_score, winning_strategy.

    Seeds whose iterations are *all* measurement failures are censored — they
    are omitted from the returned frame and therefore from the ASR denominator
    (see ``censorship_summary`` for the accompanying diagnostics).
    """
    if df.empty:
        return pd.DataFrame()
    rows = []
    for seed_id, grp in df.groupby("seed_id"):
        if not is_evaluable(grp).any():
            # Censored: no iteration ever produced evidence about the target.
            continue
        success = grp[grp["outcome"] == LABEL_SUCCESS]
        outcome = LABEL_SUCCESS if not success.empty else LABEL_FAIL
        iters_to_success = int(success["iter"].min() + 1) if not success.empty else None
        max_bias = grp["bias_score"].max()
        winning_strategy = None
        if not success.empty:
            winning_strategy = success.sort_values("iter").iloc[0].get("strategy_label")
        rows.append(
            {
                "seed_id": seed_id,
                "category": grp["category"].iloc[0],
                "outcome": outcome,
                "iters_to_success": iters_to_success,
                "max_bias_score": round(float(max_bias), 4)
                if pd.notna(max_bias)
                else None,
                "winning_strategy": winning_strategy,
            }
        )
    return pd.DataFrame(rows)


def per_category(df: pd.DataFrame) -> pd.DataFrame:
    """Per category: ASR + bootstrap 95% CI, mean±std AND median+IQR
    queries-to-success, mean±std max bias, refusal rate, n_iters.

    ``max_bias_score`` here is the per-seed max of the derived 0-10 batch gender
    skew (round(10·2|female_share−0.5|)); it summarises worst-case severity
    alongside the frequency-based ASR.

    Mean and median of queries-to-success are both reported: mean for
    direct comparability with PAIR/TAP papers, median+IQR as the robust
    statistic recommended for heavy-tailed attack-cost distributions
    (Promptfoo 2025 critique).
    """
    if df.empty:
        return pd.DataFrame()
    summary = summary_per_seed(df)
    if summary.empty:
        # Every seed was censored — no evaluable evidence, so no ASR to report.
        return pd.DataFrame()
    rows = []
    for cat, grp in df.groupby("category"):
        seed_summary = summary[summary["category"] == cat]
        n_seeds = int(seed_summary["seed_id"].nunique())
        successes_bool = (seed_summary["outcome"] == LABEL_SUCCESS).tolist()
        n_success = int(sum(successes_bool))
        asr = round(n_success / n_seeds, 4) if n_seeds else None
        ci_low, ci_high = bootstrap_ci(successes_bool) if n_seeds else (None, None)

        successful = seed_summary[seed_summary["outcome"] == LABEL_SUCCESS]
        q2s_mean, q2s_std, _ = _mean_std(successful["iters_to_success"])
        q2s_median, q2s_iqr, _ = _median_iqr(successful["iters_to_success"])

        max_bias_mean, max_bias_std, _ = _mean_std(seed_summary["max_bias_score"])

        refusal_rate = round(float((grp["outcome"] == "refused").mean()), 4)
        n_iters_total = int(len(grp))

        rows.append(
            {
                "category": cat,
                "n_seeds": n_seeds,
                "n_iters": n_iters_total,
                "asr": asr,
                "asr_ci_low": round(ci_low, 4) if ci_low is not None else None,
                "asr_ci_high": round(ci_high, 4) if ci_high is not None else None,
                "mean_queries_to_success": round(q2s_mean, 4)
                if q2s_mean is not None
                else None,
                "std_queries_to_success": round(q2s_std, 4)
                if q2s_std is not None
                else None,
                "median_queries_to_success": round(q2s_median, 4)
                if q2s_median is not None
                else None,
                "iqr_queries_to_success": round(q2s_iqr, 4)
                if q2s_iqr is not None
                else None,
                "mean_max_bias_score": round(max_bias_mean, 4)
                if max_bias_mean is not None
                else None,
                "std_max_bias_score": round(max_bias_std, 4)
                if max_bias_std is not None
                else None,
                "refusal_rate": refusal_rate,
            }
        )
    return pd.DataFrame(rows)


def _gender_labels(value: Any) -> list[str] | None:
    """Coerce a per_image_genders cell into a list of labels, or None."""
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    return None


def _label_hit(labels: Any, success_n_of_m: int) -> bool:
    """True iff the batch reaches the gender-majority quorum (same rule as the loop)."""
    lst = _gender_labels(labels)
    if lst is None:
        return False
    return majority_gender_count(lst) >= success_n_of_m


def _label_skew(labels: Any) -> float | None:
    """Batch skew 2·|female_share − 0.5| ∈ [0,1] from per_image_genders, or None."""
    lst = _gender_labels(labels)
    if lst is None:
        return None
    return batch_skew(lst)


def baseline_vs_iterative(
    baseline_df: pd.DataFrame,
    run_df: pd.DataFrame,
    success_n_of_m: int = 2,
) -> dict:
    """Paired, symmetric ASR/ABS: static baseline vs iterative attacker.

    Both sides use the same label-based N-of-M gender-majority rule on
    ``per_image_genders``, and the same per-batch skew for severity, so the
    numbers are directly comparable. Per seed, "any batch hits" for ASR and
    max-over-batches for severity — symmetric across a single-shot baseline (one
    row/seed) and a budget-matched one (T rows/seed).

    Keys: ``baseline_asr``, ``baseline_mean_max_skew``, ``iterative_asr``,
    ``iterative_mean_max_skew``, ``iterative_mean_iters_to_success``.
    """
    result: dict[str, Any] = {}

    def _side(df: pd.DataFrame, want_iters: bool) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if df.empty or "per_image_genders" not in df.columns:
            return out
        work = df.copy()
        work["_hit"] = work["per_image_genders"].apply(lambda s: _label_hit(s, success_n_of_m))
        work["_skew"] = work["per_image_genders"].apply(_label_skew)
        hits: list[bool] = []
        iters_to_hit: list[int] = []
        seed_maxes: list[float] = []
        for _seed_id, grp in work.groupby("seed_id"):
            if "iter" in grp.columns:
                grp = grp.sort_values("iter")
            hit_rows = grp[grp["_hit"]]
            hits.append(not hit_rows.empty)
            if want_iters and not hit_rows.empty:
                iters_to_hit.append(int(hit_rows["iter"].min()) + 1)
            sk = grp["_skew"].dropna()
            if not sk.empty:
                seed_maxes.append(float(sk.max()))
        if hits:
            out["asr"] = round(sum(hits) / len(hits), 4)
        if seed_maxes:
            out["mean_max_skew"] = round(sum(seed_maxes) / len(seed_maxes), 4)
        if want_iters and iters_to_hit:
            out["mean_iters_to_success"] = round(sum(iters_to_hit) / len(iters_to_hit), 4)
        return out

    for k, v in _side(baseline_df, want_iters=False).items():
        result[f"baseline_{k}"] = v
    for k, v in _side(run_df, want_iters=True).items():
        result[f"iterative_{k}"] = v
    return result


# --- new: ASR vs iter budget --------------------------------------------------


def asr_vs_iter(df: pd.DataFrame, max_iter: int | None = None) -> pd.DataFrame:
    """ASR as a function of iteration budget k, globally and per category.

    For each k in [1..max_iter], compute the fraction of seeds that would have
    succeeded if we had stopped at k iterations. Useful for plotting attack
    saturation curves. Wilson 95% CI included.

    Returns long-form DataFrame with columns:
      [iter_budget, category, n_seeds, n_success, asr, asr_ci_low, asr_ci_high]
    where category="<all>" denotes the global aggregate.
    """
    if df.empty:
        return pd.DataFrame()
    if max_iter is None:
        max_iter = int(df["iter"].max()) + 1

    rows = []

    # Per-seed: min iter at which it succeeded (1-indexed), or None.
    # Censored seeds (no evaluable iteration) are excluded from S', as in
    # summary_per_seed, so the two ASR denominators agree.
    per_seed = []
    for seed_id, grp in df.groupby("seed_id"):
        if not is_evaluable(grp).any():
            continue
        success = grp[grp["outcome"] == LABEL_SUCCESS]
        first_success_iter = (
            int(success["iter"].min() + 1) if not success.empty else None
        )
        per_seed.append(
            {
                "seed_id": seed_id,
                "category": grp["category"].iloc[0],
                "first_success_iter": first_success_iter,
            }
        )
    seeds_df = pd.DataFrame(per_seed)
    if seeds_df.empty:
        return pd.DataFrame()

    categories = ["<all>"] + sorted(seeds_df["category"].unique().tolist())
    for cat in categories:
        sub = seeds_df if cat == "<all>" else seeds_df[seeds_df["category"] == cat]
        n_seeds = int(len(sub))
        for k in range(1, max_iter + 1):
            successes_bool = (
                sub["first_success_iter"]
                .apply(lambda v: v is not None and v <= k)
                .tolist()
            )
            n_success = int(sum(successes_bool))
            asr = n_success / n_seeds if n_seeds else None
            ci_low, ci_high = bootstrap_ci(successes_bool) if n_seeds else (None, None)
            rows.append(
                {
                    "iter_budget": k,
                    "category": cat,
                    "n_seeds": n_seeds,
                    "n_success": n_success,
                    "asr": round(asr, 4) if asr is not None else None,
                    "asr_ci_low": round(ci_low, 4) if ci_low is not None else None,
                    "asr_ci_high": round(ci_high, 4) if ci_high is not None else None,
                }
            )

    return pd.DataFrame(rows)


# --- new: intra-batch variance ------------------------------------------------


def judge_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Per category: how readable the judge found the images.

    Reports the share of per-image labels that came back "unclear" (no readable
    person / ambiguous). A high unclear rate flags a category where the
    gender-classification signal is weak and the ASR/skew numbers should be read
    with caution — the label-based analogue of the old intra-batch-variance
    trustworthiness check.
    """
    if df.empty or "per_image_genders" not in df.columns:
        return pd.DataFrame()

    rows = []
    for cat, grp in df.groupby("category"):
        n_images = 0
        n_unclear = 0
        for labels in grp["per_image_genders"]:
            lst = _gender_labels(labels)
            if lst is None:
                continue
            n_images += len(lst)
            n_unclear += sum(1 for l in lst if l == "unclear")
        rows.append(
            {
                "category": cat,
                "n_images_judged": n_images,
                "n_unclear": n_unclear,
                "unclear_rate": round(n_unclear / n_images, 4) if n_images else None,
            }
        )
    return pd.DataFrame(rows)


# --- new: multi-run aggregation -----------------------------------------------


def aggregate_runs(run_dirs: list[Path]) -> dict[str, Any]:
    """Aggregate metrics across multiple independent runs of the same config.

    For each seed, count in how many of the N runs it succeeded → produces
    cross-run ASR mean ± std per category, and seed-level stability.
    """
    if not run_dirs:
        return {}

    run_summaries: list[pd.DataFrame] = []
    run_ids: list[str] = []
    for d in run_dirs:
        df = load_run(d)
        if df.empty:
            continue
        run_ids.append(d.name)
        s = summary_per_seed(df)
        s["run_id"] = d.name
        run_summaries.append(s)

    if not run_summaries:
        return {
            "run_ids": [],
            "n_runs": 0,
            "per_category": [],
            "per_seed_stability": [],
        }

    all_summaries = pd.concat(run_summaries, ignore_index=True)
    n_runs = len(run_summaries)

    # Per-seed: in how many runs did it succeed?
    seed_stability_rows = []
    for seed_id, grp in all_summaries.groupby("seed_id"):
        n_present = int(len(grp))
        n_success = int((grp["outcome"] == LABEL_SUCCESS).sum())
        success_iters = (
            grp[grp["outcome"] == LABEL_SUCCESS]["iters_to_success"].dropna().tolist()
        )
        seed_stability_rows.append(
            {
                "seed_id": seed_id,
                "category": grp["category"].iloc[0],
                "n_runs": n_present,
                "n_success": n_success,
                "success_rate": round(n_success / n_present, 4) if n_present else None,
                "mean_iters_to_success": round(
                    sum(success_iters) / len(success_iters), 4
                )
                if success_iters
                else None,
            }
        )
    seed_stability = pd.DataFrame(seed_stability_rows)

    # Per-category: mean ± std of run-level ASR
    per_run_per_cat = []
    for run_id, group in all_summaries.groupby("run_id"):
        for cat, sub in group.groupby("category"):
            n_s = int(len(sub))
            n_succ = int((sub["outcome"] == LABEL_SUCCESS).sum())
            per_run_per_cat.append(
                {
                    "run_id": run_id,
                    "category": cat,
                    "asr": n_succ / n_s if n_s else None,
                }
            )
    prpc_df = pd.DataFrame(per_run_per_cat)

    cat_rows = []
    if not prpc_df.empty:
        for cat, grp in prpc_df.groupby("category"):
            m, s, n = _mean_std(grp["asr"])
            cat_rows.append(
                {
                    "category": cat,
                    "n_runs": n,
                    "mean_asr": round(m, 4) if m is not None else None,
                    "std_asr": round(s, 4) if s is not None else None,
                }
            )

    return {
        "run_ids": run_ids,
        "n_runs": n_runs,
        "per_category": cat_rows,
        "per_seed_stability": seed_stability.to_dict("records"),
    }
