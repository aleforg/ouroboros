#!/usr/bin/env python
"""Scatter plot of judge-vs-FairFace agreement on per-seed female share.

Companion figure to the Cohen's kappa in ``report/judge_fairface_gender_agreement.csv``
(docs/06-metrics.md section 7). The kappa is per image; this is per seed, which is
the granularity a reader can actually eyeball — one point per seed, judge female
share on x, FairFace female share on y, y=x for perfect agreement.

    python scripts/plot_judge_fairface_scatter.py results/<run_id>

Writes into ``<run_dir>/report/``, PNG (200 dpi) + PDF for each of two variants:

* ``judge_fairface_gender_scatter``          — single-face images only
* ``judge_fairface_gender_scatter_allfaces`` — every detected face

The single-face variant is the fair comparison: the judge labels *the main
person* in an image, so on a group scene its one label cannot be matched against
FairFace's several. The all-faces variant is reported alongside to show the
conclusion does not depend on that restriction.

Correlations are computed on raw values; the jitter is applied only when drawing,
to de-overlap the corners where most seeds pile up.

Needs matplotlib and scipy (not core dependencies):
    pip install matplotlib scipy
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

from ouroboros.fairface import load_fairface  # noqa: E402

STYLE = {
    "balanced": ("#2a78d6", "o"),
    "female_coded": ("#eb6834", "D"),
    "male_coded": ("#1baf7a", "^"),
}


def _terminal_records(run_dir: Path) -> dict[str, dict]:
    """Last image-producing iteration per seed."""
    out: dict[str, dict] = {}
    for line in (run_dir / "run.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        has_image = any(
            isinstance(s, dict) and s.get("path") for s in (rec.get("samples") or [])
        )
        if not has_image:
            continue
        cur = out.get(rec["seed_id"])
        if cur is None or rec["iter"] >= cur["iter"]:
            out[rec["seed_id"]] = rec
    return out


def build_frame(run_dir: Path) -> pd.DataFrame:
    terminal = _terminal_records(run_dir)
    faces = load_fairface(run_dir, "fairface_iterative_terminal.jsonl")
    per_image = faces.groupby("image_path").size()
    single_face = set(per_image[per_image == 1].index)
    faces_sf = faces[faces.image_path.isin(single_face)]

    rows = []
    for seed_id, rec in terminal.items():
        sf = faces_sf[faces_sf.seed_id == seed_id]
        allf = faces[faces.seed_id == seed_id]
        rows.append(
            {
                "seed_id": seed_id,
                "category": rec["category"],
                "judge_fs": (rec.get("judge") or {}).get("female_share"),
                "ff_fs_single": (sf["gender"] == "Female").mean() if len(sf) else np.nan,
                "ff_fs_all": (allf["gender"] == "Female").mean() if len(allf) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_plot(df: pd.DataFrame, ycol: str, subtitle: str, out_path: Path) -> dict:
    d = df.dropna(subset=["judge_fs", ycol]).copy()
    x, y = d["judge_fs"].to_numpy(dtype=float), d[ycol].to_numpy(dtype=float)
    stats = {
        "n_seeds": int(len(d)),
        "pearson_r": round(float(pearsonr(x, y)[0]), 4),
        "spearman_rho": round(float(spearmanr(x, y)[0]), 4),
        "mae": round(float(np.abs(x - y).mean()), 4),
        "majority_agreement": round(
            float((((x > 0.5) == (y > 0.5)) | ((x == 0.5) & (y == 0.5))).mean()), 4
        ),
    }

    fig, ax = plt.subplots(figsize=(6.2, 6.2), dpi=200)
    ax.plot([0, 1], [0, 1], ls="--", lw=1.3, color="#9a988f", zorder=1,
            label="accordo perfetto (y=x)")
    rng = np.random.default_rng(0)  # fixed seed: the figure is reproducible
    for cat, (colour, marker) in STYLE.items():
        c = d[d.category == cat]
        if c.empty:
            continue
        jx = c["judge_fs"] + rng.uniform(-0.012, 0.012, len(c))
        jy = c[ycol] + rng.uniform(-0.012, 0.012, len(c))
        ax.scatter(jx, jy, s=46, marker=marker, facecolor=colour, edgecolor="white",
                   linewidth=0.6, alpha=0.75, zorder=3, label=f"{cat} (n={len(c)})")

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Quota femminile — giudice VLM", fontsize=11)
    ax.set_ylabel("Quota femminile — FairFace", fontsize=11)
    ax.set_xticks(np.arange(0, 1.01, 0.25))
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_title("Accordo giudice VLM ↔ FairFace sul genere per-seed", fontsize=12.5, pad=10)
    ax.text(
        0.02, 0.975,
        f"r = {stats['pearson_r']:.2f}   ρ = {stats['spearman_rho']:.2f}   "
        f"MAE = {stats['mae']:.2f}   accordo maggioranza = {stats['majority_agreement']*100:.0f}%"
        f"\n{subtitle}   (N = {stats['n_seeds']} seed)",
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", fc="#f5f4ef", ec="#d3d1c7", lw=0.6),
    )
    ax.grid(True, color="#e1e0d9", lw=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95, edgecolor="#d3d1c7")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return stats


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    run_dir = Path(sys.argv[1])
    if not (run_dir / "fairface_iterative_terminal.jsonl").exists():
        print("error: no fairface_iterative_terminal.jsonl — run `ouroboros report` first")
        return 1

    out_dir = run_dir / "report"
    out_dir.mkdir(exist_ok=True)
    df = build_frame(run_dir)

    variants = [
        ("ff_fs_single", "solo immagini a volto singolo", "judge_fairface_gender_scatter.png"),
        ("ff_fs_all", "tutti i volti rilevati", "judge_fairface_gender_scatter_allfaces.png"),
    ]
    report = {"run_id": run_dir.name, "variants": {}}
    for ycol, subtitle, fname in variants:
        stats = build_plot(df, ycol, subtitle, out_dir / fname)
        report["variants"][fname.replace(".png", "")] = stats
        print(f"{fname}: r={stats['pearson_r']:.3f} rho={stats['spearman_rho']:.3f} "
              f"MAE={stats['mae']:.3f} maj={stats['majority_agreement']:.3f} N={stats['n_seeds']}")

    (out_dir / "judge_fairface_scatter_stats.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n→ {out_dir / 'judge_fairface_scatter_stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
