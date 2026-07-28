from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader  # type: ignore[import]
from ouroboros import __version__
from ouroboros.config import FULL_BUDGET, LABEL_SUCCESS, TEST_BUDGET
from ouroboros.metrics import (
    aggregate_runs,
    asr_vs_iter,
    baseline_vs_iterative,
    censorship_summary,
    judge_coverage,
    load_baseline,
    load_run,
    per_category,
    summary_per_seed,
)
from ouroboros.metrics.agreement import (
    judge_fairface_axis_spearman,
    judge_fairface_gender_agreement,
)
from ouroboros.metrics.adversarial import (
    adversarial_bias_by_category,
    adversarial_bias_per_seed,
)
from ouroboros.metrics.fairness import bls_gender_alignment_summary, distribution_gap_summary

logger = logging.getLogger(__name__)


# --- SVG line chart helper ----------------------------------------------------


_CHART_COLORS = [
    "#4a90e2", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#34495e", "#e67e22", "#c0392b", "#16a085",
]


def _svg_asr_curves(asr_df, width: int = 720, height: int = 320) -> str:
    """Build an inline SVG line chart of ASR vs iter_budget, one line per category.

    Keeps the report self-contained (no external chart libs, no PNG files).
    Returns "" if input is empty.
    """
    if asr_df is None or asr_df.empty:
        return ""

    pad_l, pad_r, pad_t, pad_b = 50, 130, 20, 40
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    max_x = int(asr_df["iter_budget"].max())
    min_x = 1
    x_range = max(1, max_x - min_x)

    def x_pos(k: int) -> float:
        return pad_l + (k - min_x) / x_range * plot_w

    def y_pos(asr: float) -> float:
        return pad_t + (1.0 - asr) * plot_h

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="system-ui, sans-serif">'
    ]
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fff"/>')
    # Y gridlines + labels
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = y_pos(frac)
        parts.append(
            f'<line x1="{pad_l}" y1="{y}" x2="{pad_l + plot_w}" y2="{y}" '
            f'stroke="#eee" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad_l - 8}" y="{y + 4}" text-anchor="end" '
            f'font-size="10" fill="#666">{frac:.2f}</text>'
        )
    # X axis ticks
    for k in range(min_x, max_x + 1):
        if max_x <= 10 or k % max(1, max_x // 10) == 0 or k == max_x:
            x = x_pos(k)
            parts.append(
                f'<line x1="{x}" y1="{pad_t + plot_h}" x2="{x}" y2="{pad_t + plot_h + 4}" '
                f'stroke="#666"/>'
            )
            parts.append(
                f'<text x="{x}" y="{pad_t + plot_h + 16}" text-anchor="middle" '
                f'font-size="10" fill="#666">{k}</text>'
            )
    # Axis labels
    parts.append(
        f'<text x="{pad_l + plot_w / 2}" y="{height - 6}" text-anchor="middle" '
        f'font-size="11" fill="#333">iteration budget k</text>'
    )
    parts.append(
        f'<text x="14" y="{pad_t + plot_h / 2}" text-anchor="middle" font-size="11" '
        f'fill="#333" transform="rotate(-90 14 {pad_t + plot_h / 2})">ASR(k)</text>'
    )

    categories = list(asr_df["category"].unique())
    legend_y = pad_t + 6
    for i, cat in enumerate(categories):
        color = _CHART_COLORS[i % len(_CHART_COLORS)]
        sub = asr_df[asr_df["category"] == cat].sort_values("iter_budget")
        pts = [(x_pos(int(r.iter_budget)), y_pos(float(r.asr))) for r in sub.itertuples()]
        path = " ".join(f"{'M' if j == 0 else 'L'} {x:.1f} {y:.1f}" for j, (x, y) in enumerate(pts))
        parts.append(
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        for x, y in pts:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" fill="{color}"/>')
        # legend
        ly = legend_y + i * 16
        parts.append(
            f'<line x1="{pad_l + plot_w + 12}" y1="{ly}" '
            f'x2="{pad_l + plot_w + 32}" y2="{ly}" stroke="{color}" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{pad_l + plot_w + 36}" y="{ly + 4}" font-size="10" fill="#333">{cat}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


# --- per-run report -----------------------------------------------------------


def _success_n_of_m(run_dir: Path) -> int:
    """Read success_n_of_m for the run from meta.json.

    Resolves the frozen ModeBudget via the run's mode. Falls back to 2 when meta
    is missing or unreadable, so old runs still re-score sensibly.
    """

    meta_path = run_dir / "meta.json"
    try:
        meta = json.loads(meta_path.read_text())
        mode = (meta.get("config") or {}).get("mode", "test")
        budget = TEST_BUDGET if mode == "test" else FULL_BUDGET
        return budget.success_n_of_m
    except Exception:
        return 2


def _terminal_run_subset(run_df) -> "pd.DataFrame":
    """Filter run_df to each seed's terminal iteration (last iter with images).

    Used so KL n_images for the iterative-terminal selection counts only the
    terminal batch, matching the face rows produced by
    ``process_run(selection="iterative_terminal")``.
    """

    if run_df.empty or "samples" not in run_df.columns or "seed_id" not in run_df.columns:
        return run_df

    def _has_img(ss) -> bool:
        return isinstance(ss, list) and any(
            isinstance(s, dict) and s.get("path") for s in ss
        )

    keep_idx: list = []
    for _seed_id, grp in run_df.groupby("seed_id"):
        with_img = grp[grp["samples"].apply(_has_img)]
        if with_img.empty:
            continue
        term_iter = with_img["iter"].max()
        keep_idx.extend(with_img[with_img["iter"] == term_iter].index.tolist())
    return run_df.loc[keep_idx] if keep_idx else run_df.iloc[0:0]


def _kl_delta(baseline_kl, iterative_kl) -> "pd.DataFrame":
    """Build the per-category baseline vs iterative-terminal KL delta table.

    Columns per axis (gender/race/age): baseline_kl_<axis>, iterative_kl_<axis>,
    delta_kl_<axis> (iterative − baseline; positive = the attacker widened skew).
    """

    axes = ["gender", "race", "age"]
    b_by = (
        {r["category"]: r for r in baseline_kl.to_dict("records")}
        if baseline_kl is not None and not baseline_kl.empty
        else {}
    )
    t_by = (
        {r["category"]: r for r in iterative_kl.to_dict("records")}
        if iterative_kl is not None and not iterative_kl.empty
        else {}
    )
    cats = sorted(set(b_by) | set(t_by))
    if not cats:
        return pd.DataFrame()

    rows: list[dict] = []
    for cat in cats:
        row: dict = {"category": cat}
        for ax in axes:
            b = b_by.get(cat, {}).get(f"kl_{ax}_nats")
            t = t_by.get(cat, {}).get(f"kl_{ax}_nats")
            row[f"baseline_kl_{ax}"] = b
            row[f"iterative_kl_{ax}"] = t
            row[f"delta_kl_{ax}"] = (
                round(t - b, 4) if b is not None and t is not None else None
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _run_fairface_pipeline(run_dir: Path, run_df, baseline_df):
    """Run FairFace classification + baseline-vs-iterative KL aggregation.

    Returns ``(iterative_terminal_kl, baseline_kl, delta_kl, raw_all_df)``, all
    possibly-empty DataFrames. ``raw_all_df`` is the all-iterations per-face
    table (fairface.jsonl), kept broad for convergent-validity metrics.

    **The two KL tables are NOT symmetric under ``baseline_mode="matched"``.**
    ``iterative_kl`` always covers the terminal batch per seed (one M-image
    batch), but ``baseline_kl`` covers *every* baseline batch, and a matched
    baseline gives a seed as many batches as the loop spent generative
    iterations on it. On a 175-seed matched run that is ~1896 baseline images
    against ~1400 iterative ones. Only under ``single-shot`` is it truly one
    batch per seed on both sides. To pair them symmetrically, restrict the
    baseline side to the last batch per seed before calling
    :func:`compute_kl_metrics`.

    Also note ``compute_kl_metrics`` pools every face of a category into one
    distribution, so per-seed skews in opposite directions cancel: a category
    split between all-female and all-male seeds yields ``kl_gender ≈ 0`` even
    though every individual batch is single-gender. For the attacker's effect
    use the per-seed absolute skew in :mod:`ouroboros.metrics.adversarial`.

    Errors (missing weights, torch not installed, etc.) are logged but never
    raised — the rest of the report should still generate.
    """

    empty = pd.DataFrame()
    try:
        from ouroboros.fairface import compute_kl_metrics, load_fairface, process_run

        # 1. All-iterations classification (fairface.jsonl): substrate for the
        #    judge↔FairFace agreement and BLS metrics, which want max coverage.
        existing = load_fairface(run_dir)
        if existing.empty:
            logger.info("Running FairFace classification (all iters) — this may take a few minutes …")
            process_run(run_dir, selection="iterative_all")
        else:
            logger.info("Reusing existing fairface.jsonl (%d face rows)", len(existing))
        raw_all = load_fairface(run_dir)

        # 2. Iterative terminal batch per seed — the headline skew number.
        process_run(
            run_dir,
            output_jsonl=run_dir / "fairface_iterative_terminal.jsonl",
            selection="iterative_terminal",
        )
        term_ff = load_fairface(run_dir, "fairface_iterative_terminal.jsonl")
        iterative_kl = (
            compute_kl_metrics(term_ff, run_df=_terminal_run_subset(run_df))
            if not term_ff.empty
            else empty
        )

        # 3. Baseline batch per seed (only if a baseline was run).
        baseline_kl = empty
        if (run_dir / "baseline.jsonl").exists():
            process_run(
                run_dir,
                output_jsonl=run_dir / "fairface_baseline.jsonl",
                selection="baseline",
            )
            base_ff = load_fairface(run_dir, "fairface_baseline.jsonl")
            baseline_kl = (
                compute_kl_metrics(base_ff, run_df=baseline_df)
                if not base_ff.empty
                else empty
            )

        return iterative_kl, baseline_kl, _kl_delta(baseline_kl, iterative_kl), raw_all
    except FileNotFoundError as e:
        logger.warning("FairFace skipped — %s", e)
    except ImportError as e:
        logger.warning("FairFace skipped — %s", e)
    except Exception as e:
        logger.warning("FairFace pipeline failed unexpectedly: %s", e)
    return empty, empty, empty, empty


def run_report(run_dir: Path, skip_fairface: bool = False, bls: bool = False) -> None:
    """Generate post-hoc report artifacts under run_dir/report/."""
    report_dir = run_dir / "report"
    report_dir.mkdir(exist_ok=True)
    thumbs_dir = report_dir / "thumbs"
    thumbs_dir.mkdir(exist_ok=True)

    logger.info("Generating report for %s …", run_dir.name)

    run_df = load_run(run_dir)
    baseline_df = load_baseline(run_dir)

    if run_df.empty:
        logger.warning("run.jsonl is empty or missing — report will be sparse")

    summary_df = summary_per_seed(run_df)
    cat_df = per_category(run_df)
    # Censoring diagnostics for the ASR denominator: seeds whose iterations were
    # all measurement failures are excluded from S', so the rate at which that
    # happens is reported alongside ASR rather than silently folded into it.
    censoring = censorship_summary(run_df)
    if censoring["n_seeds_censored"]:
        logger.warning(
            "%d/%d seeds censored from ASR denominator (all iterations were "
            "measurement failures): %s",
            censoring["n_seeds_censored"],
            censoring["n_seeds_total"],
            ", ".join(censoring["censored_seed_ids"][:5]),
        )
    # Read the success N-of-M from the frozen budget in meta.json so ASR is
    # recomputed at the same quorum the run used (fallback 2 when unavailable).
    success_n_of_m = _success_n_of_m(run_dir)
    bvi = baseline_vs_iterative(baseline_df, run_df, success_n_of_m=success_n_of_m)
    # Same quorum as the readability floor for ABS: a batch scores only if it had
    # enough classified images to have satisfied the success rule. Without it, a
    # batch with one readable image yields skew 1.0, and since ABS takes the max
    # over iterations the iterative side — which draws far more batches than the
    # baseline — collects that inflation asymmetrically into ΔABS.
    abs_seed_df = adversarial_bias_per_seed(
        run_df, baseline_df, min_readable=success_n_of_m
    )
    abs_cat_df = adversarial_bias_by_category(abs_seed_df)
    asr_iter_df = asr_vs_iter(run_df) if not run_df.empty else None
    coverage_df = judge_coverage(run_df) if not run_df.empty else None

    fairface_df = None
    baseline_kl_df = None
    fairface_delta_df = None
    distribution_gap_df = None
    bls_alignment_df = None
    agreement_spearman_df = None
    agreement_gender_df = None
    if not skip_fairface and not run_df.empty:
        fairface_df, baseline_kl_df, fairface_delta_df, fairface_raw_df = (
            _run_fairface_pipeline(run_dir, run_df, baseline_df)
        )
        # Headline skew table = iterative terminal batch. Also kept (redefined)
        # as fairface_per_category.csv below for dashboard/report back-compat.
        if fairface_df is not None and not fairface_df.empty:
            distribution_gap_df = distribution_gap_summary(fairface_df)
            if not distribution_gap_df.empty:
                distribution_gap_df.to_csv(report_dir / "distribution_gap.csv", index=False)
                logger.info("  distribution_gap.csv             → %d rows", len(distribution_gap_df))
        # Paired baseline-vs-iterative-terminal KL artifacts.
        if baseline_kl_df is not None and not baseline_kl_df.empty:
            baseline_kl_df.to_csv(report_dir / "fairface_baseline_per_category.csv", index=False)
            logger.info("  fairface_baseline_per_category.csv → %d rows", len(baseline_kl_df))
        if fairface_df is not None and not fairface_df.empty:
            fairface_df.to_csv(
                report_dir / "fairface_iterative_terminal_per_category.csv", index=False
            )
        if fairface_delta_df is not None and not fairface_delta_df.empty:
            fairface_delta_df.to_csv(report_dir / "fairface_baseline_vs_iterative.csv", index=False)
            logger.info("  fairface_baseline_vs_iterative.csv → %d rows", len(fairface_delta_df))
        # Convergent-validity metrics use the all-iterations face table (broad
        # coverage), independent of the terminal KL table.
        if fairface_raw_df is not None and not fairface_raw_df.empty:
            if bls:
                # BLS occupational alignment (external-validity / RQ2) is an
                # exploratory extension in thesis scope — opt-in via --bls.
                try:
                    bls_alignment_df = bls_gender_alignment_summary(fairface_raw_df)
                    if not bls_alignment_df.empty:
                        bls_alignment_df.to_csv(report_dir / "bls_gender_alignment.csv", index=False)
                        logger.info("  bls_gender_alignment.csv        → %d rows", len(bls_alignment_df))
                except Exception as exc:
                    logger.warning("BLS gender alignment failed: %s — skipping", exc)
            try:
                agreement_spearman_df = judge_fairface_axis_spearman(run_df, fairface_raw_df)
                if not agreement_spearman_df.empty:
                    agreement_spearman_df.to_csv(
                        report_dir / "judge_fairface_spearman.csv", index=False
                    )
                    logger.info("  judge_fairface_spearman.csv     → %d rows", len(agreement_spearman_df))
                agreement_gender_df = judge_fairface_gender_agreement(run_df, fairface_raw_df)
                if not agreement_gender_df.empty:
                    agreement_gender_df.to_csv(
                        report_dir / "judge_fairface_gender_agreement.csv", index=False
                    )
                    logger.info("  judge_fairface_gender_agreement.csv → %d rows", len(agreement_gender_df))
            except Exception as exc:
                logger.warning("Judge-FairFace agreement failed: %s — skipping", exc)

    clusters_data: list[dict] = []
    if not run_df.empty and "strategy_label" in run_df.columns:
        success_labels = run_df[run_df["outcome"] == LABEL_SUCCESS]["strategy_label"].dropna().tolist()
        if success_labels:
            try:
                from ouroboros.cluster import cluster_strategies, cluster_success_rate

                assignments = cluster_strategies(success_labels)
                es_df = cluster_success_rate(run_df, assignments)
                clusters_data = es_df.to_dict("records")
            except Exception as exc:
                logger.warning("Clustering failed: %s — skipping strategy clusters", exc)

    if not summary_df.empty:
        summary_df.to_csv(report_dir / "summary.csv", index=False)
    if not cat_df.empty:
        cat_df.to_csv(report_dir / "per_category.csv", index=False)
    if not run_df.empty:
        pd.DataFrame(
            [{k: v for k, v in censoring.items() if k != "censored_seed_ids"}]
        ).to_csv(report_dir / "censorship.csv", index=False)
    if not abs_seed_df.empty:
        abs_seed_df.to_csv(report_dir / "adversarial_bias_per_seed.csv", index=False)
    if not abs_cat_df.empty:
        abs_cat_df.to_csv(report_dir / "adversarial_bias_by_category.csv", index=False)
    if asr_iter_df is not None and not asr_iter_df.empty:
        asr_iter_df.to_csv(report_dir / "asr_vs_iter.csv", index=False)
    if coverage_df is not None and not coverage_df.empty:
        coverage_df.to_csv(report_dir / "judge_coverage.csv", index=False)
    if fairface_df is not None and not fairface_df.empty:
        # NOTE: as of v2.7 this is the iterative *terminal* batch per seed (was
        # all-iterations). Kept under the historical name for dashboard/report
        # back-compat; the explicit twin is fairface_iterative_terminal_per_category.csv.
        fairface_df.to_csv(report_dir / "fairface_per_category.csv", index=False)
    (report_dir / "strategy_clusters.json").write_text(
        json.dumps(clusters_data, indent=2), encoding="utf-8"
    )

    thumbs_by_category: dict[str, list[str]] = {}
    if not run_df.empty and "category" in run_df.columns and "bias_score" in run_df.columns:
        for cat, grp in run_df.groupby("category"):
            top = grp.nlargest(3, "bias_score", keep="first")
            cat_thumbs = []
            for _, row in top.iterrows():
                samples = row.get("samples") or []
                if isinstance(samples, str):
                    try:
                        samples = json.loads(samples)
                    except Exception:
                        samples = []
                for s in samples:
                    if isinstance(s, dict) and s.get("outcome") == "image" and s.get("path"):
                        src = run_dir / s["path"]
                        if src.exists():
                            dst_name = f"{cat}_{src.parent.parent.name}_{src.name}"
                            dst = thumbs_dir / dst_name
                            shutil.copy2(src, dst)
                            cat_thumbs.append(f"thumbs/{dst_name}")
                            break
            if cat_thumbs:
                thumbs_by_category[cat] = cat_thumbs

    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    asr_chart_svg = _svg_asr_curves(asr_iter_df) if asr_iter_df is not None else ""

    html = _render_html(
        run_id=run_dir.name,
        meta=meta,
        summary=summary_df.to_dict("records") if not summary_df.empty else [],
        per_category_rows=cat_df.to_dict("records") if not cat_df.empty else [],
        baseline_vs_iter=bvi,
        adversarial_bias_rows=abs_cat_df.to_dict("records") if not abs_cat_df.empty else [],
        clusters=clusters_data,
        thumbs_by_category=thumbs_by_category,
        asr_chart_svg=asr_chart_svg,
        coverage_rows=coverage_df.to_dict("records") if coverage_df is not None and not coverage_df.empty else [],
        fairface_rows=fairface_df.to_dict("records") if fairface_df is not None and not fairface_df.empty else [],
        fairface_delta_rows=(
            fairface_delta_df.to_dict("records")
            if fairface_delta_df is not None and not fairface_delta_df.empty
            else []
        ),
        distribution_gap_rows=(
            distribution_gap_df.to_dict("records")
            if distribution_gap_df is not None and not distribution_gap_df.empty
            else []
        ),
        bls_alignment_rows=(
            bls_alignment_df.to_dict("records")
            if bls_alignment_df is not None and not bls_alignment_df.empty
            else []
        ),
        agreement_spearman_rows=(
            agreement_spearman_df.to_dict("records")
            if agreement_spearman_df is not None and not agreement_spearman_df.empty
            else []
        ),
        agreement_gender_rows=(
            agreement_gender_df.to_dict("records")
            if agreement_gender_df is not None and not agreement_gender_df.empty
            else []
        ),
    )
    (report_dir / "report.html").write_text(html, encoding="utf-8")

    logger.info("Report written to %s", report_dir)
    logger.info("  summary.csv                    → %d rows", len(summary_df) if not summary_df.empty else 0)
    logger.info("  per_category.csv               → %d rows", len(cat_df) if not cat_df.empty else 0)
    if not abs_seed_df.empty:
        logger.info("  adversarial_bias_per_seed.csv  → %d rows", len(abs_seed_df))
    if not abs_cat_df.empty:
        logger.info("  adversarial_bias_by_category.csv → %d rows", len(abs_cat_df))
    logger.info("  asr_vs_iter.csv                → %d rows", len(asr_iter_df) if asr_iter_df is not None else 0)
    logger.info("  judge_coverage.csv             → %d rows", len(coverage_df) if coverage_df is not None else 0)
    if fairface_df is not None and not fairface_df.empty:
        logger.info("  fairface_per_category.csv      → %d rows (iterative terminal)", len(fairface_df))
    if fairface_delta_df is not None and not fairface_delta_df.empty:
        logger.info("  fairface_baseline_vs_iterative.csv → %d rows", len(fairface_delta_df))
    if distribution_gap_df is not None and not distribution_gap_df.empty:
        logger.info("  distribution_gap.csv           → %d rows", len(distribution_gap_df))
    if bls_alignment_df is not None and not bls_alignment_df.empty:
        logger.info("  bls_gender_alignment.csv       → %d rows", len(bls_alignment_df))
    logger.info("  strategy_clusters              → %d clusters", len(clusters_data))
    logger.info("  report.html                    → self-contained")


# --- multi-run aggregate report -----------------------------------------------


def run_aggregate_report(run_dirs: list[Path], out_dir: Path) -> None:
    """Aggregate multiple runs into a cross-run report."""
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Aggregating %d runs into %s …", len(run_dirs), out_dir.name)

    agg = aggregate_runs(run_dirs)

    cat_df = pd.DataFrame(agg.get("per_category", []))
    stab_df = pd.DataFrame(agg.get("per_seed_stability", []))

    if not cat_df.empty:
        cat_df.to_csv(out_dir / "cross_run_per_category.csv", index=False)
    if not stab_df.empty:
        stab_df.to_csv(out_dir / "per_seed_stability.csv", index=False)

    html = _render_aggregate_html(
        run_ids=agg.get("run_ids", []),
        n_runs=agg.get("n_runs", 0),
        per_category=cat_df.to_dict("records") if not cat_df.empty else [],
        stability=stab_df.to_dict("records") if not stab_df.empty else [],
    )
    (out_dir / "aggregate_report.html").write_text(html, encoding="utf-8")

    logger.info("Aggregate report written to %s", out_dir)
    logger.info("  n_runs aggregated:                   %d", agg.get("n_runs", 0))
    logger.info("  cross_run_per_category.csv         → %d rows", len(cat_df))
    logger.info("  per_seed_stability.csv             → %d rows", len(stab_df))
    logger.info("  aggregate_report.html              → self-contained")


# --- HTML rendering -----------------------------------------------------------


def _render_html(
    run_id: str,
    meta: dict,
    summary: list[dict],
    per_category_rows: list[dict],
    baseline_vs_iter: dict,
    adversarial_bias_rows: list[dict],
    clusters: list[dict],
    thumbs_by_category: dict[str, list[str]],
    asr_chart_svg: str,
    coverage_rows: list[dict],
    fairface_rows: list[dict],
    fairface_delta_rows: list[dict],
    distribution_gap_rows: list[dict],
    bls_alignment_rows: list[dict],
    agreement_spearman_rows: list[dict],
    agreement_gender_rows: list[dict],
) -> str:

    templates_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(str(templates_dir)), autoescape=False)
    template = env.get_template("report.html.j2")
    return template.render(
        run_id=run_id,
        meta=meta,
        summary=summary,
        per_category=per_category_rows,
        baseline_vs_iter=baseline_vs_iter if baseline_vs_iter else None,
        adversarial_bias_rows=adversarial_bias_rows,
        clusters=clusters,
        thumbs_by_category=thumbs_by_category,
        version=__version__,
        generated_at=datetime.now(timezone.utc).isoformat(),
        asr_chart_svg=asr_chart_svg,
        coverage_rows=coverage_rows,
        fairface_rows=fairface_rows,
        fairface_delta_rows=fairface_delta_rows,
        distribution_gap_rows=distribution_gap_rows,
        bls_alignment_rows=bls_alignment_rows,
        agreement_spearman_rows=agreement_spearman_rows,
        agreement_gender_rows=agreement_gender_rows,
    )


def _render_aggregate_html(
    run_ids: list[str],
    n_runs: int,
    per_category: list[dict],
    stability: list[dict],
) -> str:

    templates_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(str(templates_dir)), autoescape=False)
    template = env.get_template("aggregate_report.html.j2")
    return template.render(
        run_ids=run_ids,
        n_runs=n_runs,
        per_category=per_category,
        stability=stability,
        version=__version__,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
