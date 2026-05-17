from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mirtage import __version__
from mirtage.config import LABEL_SUCCESS
from mirtage.metrics import (
    aggregate_runs,
    asr_vs_iter,
    baseline_vs_iterative,
    intra_batch_variance,
    load_baseline,
    load_run,
    per_category,
    summary_per_seed,
)

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


def _run_fairface_pipeline(run_dir: Path, run_df) -> "pd.DataFrame":
    """Run FairFace classification + KL aggregation. Returns empty DataFrame on failure.

    Errors (missing weights, torch not installed, etc.) are logged but never
    raised — the rest of the report should still generate.
    """
    import pandas as pd

    try:
        from mirtage.fairface import compute_kl_metrics, load_fairface, process_run

        existing = load_fairface(run_dir)
        if existing.empty:
            logger.info("Running FairFace classification (this may take a few minutes) …")
            process_run(run_dir)
        else:
            logger.info("Reusing existing fairface.jsonl (%d face rows)", len(existing))

        ff_df = load_fairface(run_dir)
        if ff_df.empty:
            logger.warning("FairFace produced no face rows — KL metrics skipped")
            return pd.DataFrame()
        return compute_kl_metrics(ff_df, run_df=run_df)
    except FileNotFoundError as e:
        logger.warning("FairFace skipped — %s", e)
    except ImportError as e:
        logger.warning("FairFace skipped — %s", e)
    except Exception as e:
        logger.warning("FairFace pipeline failed unexpectedly: %s", e)
    return pd.DataFrame()


def run_report(run_dir: Path, skip_fairface: bool = False) -> None:
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
    bvi = baseline_vs_iterative(baseline_df, run_df)
    asr_iter_df = asr_vs_iter(run_df) if not run_df.empty else None
    variance_df = intra_batch_variance(run_df) if not run_df.empty else None

    fairface_df = None
    if not skip_fairface and not run_df.empty:
        fairface_df = _run_fairface_pipeline(run_dir, run_df)

    clusters_data: list[dict] = []
    if not run_df.empty and "strategy_label" in run_df.columns:
        success_labels = run_df[run_df["outcome"] == LABEL_SUCCESS]["strategy_label"].dropna().tolist()
        if success_labels:
            try:
                from mirtage.cluster import cluster_strategies, cluster_success_rate

                assignments = cluster_strategies(success_labels)
                es_df = cluster_success_rate(run_df, assignments)
                clusters_data = es_df.to_dict("records")
            except Exception as exc:
                logger.warning("Clustering failed: %s — skipping strategy clusters", exc)

    if not summary_df.empty:
        summary_df.to_csv(report_dir / "summary.csv", index=False)
    if not cat_df.empty:
        cat_df.to_csv(report_dir / "per_category.csv", index=False)
    if asr_iter_df is not None and not asr_iter_df.empty:
        asr_iter_df.to_csv(report_dir / "asr_vs_iter.csv", index=False)
    if variance_df is not None and not variance_df.empty:
        variance_df.to_csv(report_dir / "intra_batch_variance.csv", index=False)
    if fairface_df is not None and not fairface_df.empty:
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
        clusters=clusters_data,
        thumbs_by_category=thumbs_by_category,
        asr_chart_svg=asr_chart_svg,
        variance_rows=variance_df.to_dict("records") if variance_df is not None and not variance_df.empty else [],
        fairface_rows=fairface_df.to_dict("records") if fairface_df is not None and not fairface_df.empty else [],
    )
    (report_dir / "report.html").write_text(html, encoding="utf-8")

    logger.info("Report written to %s", report_dir)
    logger.info("  summary.csv                    → %d rows", len(summary_df) if not summary_df.empty else 0)
    logger.info("  per_category.csv               → %d rows", len(cat_df) if not cat_df.empty else 0)
    logger.info("  asr_vs_iter.csv                → %d rows", len(asr_iter_df) if asr_iter_df is not None else 0)
    logger.info("  intra_batch_variance.csv       → %d rows", len(variance_df) if variance_df is not None else 0)
    if fairface_df is not None and not fairface_df.empty:
        logger.info("  fairface_per_category.csv      → %d rows", len(fairface_df))
    logger.info("  strategy_clusters              → %d clusters", len(clusters_data))
    logger.info("  report.html                    → self-contained")


# --- multi-run aggregate report -----------------------------------------------


def run_aggregate_report(run_dirs: list[Path], out_dir: Path) -> None:
    """Aggregate multiple runs into a cross-run report."""
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Aggregating %d runs into %s …", len(run_dirs), out_dir.name)

    agg = aggregate_runs(run_dirs)

    import pandas as pd

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
    clusters: list[dict],
    thumbs_by_category: dict[str, list[str]],
    asr_chart_svg: str,
    variance_rows: list[dict],
    fairface_rows: list[dict],
) -> str:
    from jinja2 import Environment, FileSystemLoader  # type: ignore[import]

    templates_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(str(templates_dir)), autoescape=False)
    template = env.get_template("report.html.j2")
    return template.render(
        run_id=run_id,
        meta=meta,
        summary=summary,
        per_category=per_category_rows,
        baseline_vs_iter=baseline_vs_iter if baseline_vs_iter else None,
        clusters=clusters,
        thumbs_by_category=thumbs_by_category,
        version=__version__,
        generated_at=datetime.now(timezone.utc).isoformat(),
        asr_chart_svg=asr_chart_svg,
        variance_rows=variance_rows,
        fairface_rows=fairface_rows,
    )


def _render_aggregate_html(
    run_ids: list[str],
    n_runs: int,
    per_category: list[dict],
    stability: list[dict],
) -> str:
    from jinja2 import Environment, FileSystemLoader  # type: ignore[import]

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
