#!/usr/bin/env python
"""Robustness checks behind docs/06-metrics.md sections 6.1 and 6.2.

The FairFace KL tables in ``report/`` carry two traps that are easy to quote
wrongly. This script regenerates, from a run's existing artifacts, every number
the docs cite about them — so the claims are reproducible rather than asserted.

Runs entirely off files already on disk (``run.jsonl``, ``baseline.jsonl``,
``fairface_*.jsonl``): no image classification, no torch, no GPU. Re-running it
on the same run must reproduce the same numbers exactly.

    python scripts/fairface_robustness_checks.py results/<run_id>

Writes ``<run_dir>/report/fairface_robustness.json`` and prints a summary.

Three checks:

1. **Asymmetry** (docs 6.1) — ``_run_fairface_pipeline`` builds ``baseline_kl``
   from *every* baseline batch but ``iterative_kl`` from only the terminal batch
   per seed. This reports both image counts and recomputes the delta KL with the
   baseline restricted to its own terminal batch, i.e. a genuinely paired
   comparison, to show whether the asymmetry drives the published deltas.

2. **Pooling cancellation** (docs 6.2) — ``compute_kl_metrics`` pools every face
   of a category into one distribution, so per-seed skews in opposite directions
   cancel. This contrasts the pooled female share of a category with the mean
   per-seed |skew| inside it, and breaks each category down into all-female /
   all-male / mixed seeds.

3. **Single-face control** — adversarial prompts yield more group scenes than
   neutral ones, which is a candidate confound for the race/age deltas. This
   restricts *both* sides to images with exactly one detected face and re-reports
   the deltas.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ouroboros.fairface import AXIS_BUCKETS, AXIS_COLUMN, axis_metrics  # noqa: E402

AXES = ("gender", "race", "age")


# --- loading ------------------------------------------------------------------


def _batches(path: Path) -> list[dict[str, Any]]:
    """One entry per record that produced at least one image."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        paths = [
            s["path"]
            for s in (rec.get("samples") or [])
            if isinstance(s, dict) and s.get("outcome") == "image" and s.get("path")
        ]
        if paths:
            out.append({"seed_id": rec["seed_id"], "iter": rec.get("iter"), "paths": paths})
    return out


def _terminal_paths(batches: list[dict[str, Any]]) -> set[str]:
    """Paths of the last image-producing batch per seed."""
    best: dict[str, dict[str, Any]] = {}
    for b in batches:
        cur = best.get(b["seed_id"])
        if cur is None or _iter_key(b["iter"]) >= _iter_key(cur["iter"]):
            best[b["seed_id"]] = b
    return {p for b in best.values() for p in b["paths"]}


def _iter_key(value: Any) -> tuple[int, str]:
    if isinstance(value, int):
        return (value, "")
    if isinstance(value, float) and float(value).is_integer():
        return (int(value), "")
    return (-1, str(value))


def _faces(run_dir: Path, name: str) -> pd.DataFrame:
    path = run_dir / name
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return pd.DataFrame(rows)


# --- metrics ------------------------------------------------------------------


def _kl_by_category(faces: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    """KL(p_emp || U) per category per axis — same math as compute_kl_metrics."""
    out: dict[str, dict[str, float | None]] = {}
    for cat, grp in faces.groupby("category"):
        out[str(cat)] = {}
        for axis in AXES:
            counts = grp[AXIS_COLUMN[axis]].value_counts().to_dict()
            out[str(cat)][axis] = axis_metrics(counts, AXIS_BUCKETS[axis]).kl_nats
    return out


def _mean_delta(baseline_kl: dict, iterative_kl: dict) -> dict[str, float | None]:
    """Mean over categories of (iterative − baseline), per axis."""
    out: dict[str, float | None] = {}
    for axis in AXES:
        deltas = [
            iterative_kl[c][axis] - baseline_kl[c][axis]
            for c in sorted(set(baseline_kl) & set(iterative_kl))
            if baseline_kl[c].get(axis) is not None and iterative_kl[c].get(axis) is not None
        ]
        out[axis] = round(sum(deltas) / len(deltas), 4) if deltas else None
    return out


def _single_face_paths(faces: pd.DataFrame) -> set[str]:
    per = faces.groupby("image_path").size()
    return set(per[per == 1].index)


def _seed_gender_profile(faces: pd.DataFrame) -> pd.DataFrame:
    """Per seed: female share and |skew| over classified faces."""
    rows = []
    for (seed_id, cat), grp in faces.groupby(["seed_id", "category"]):
        g = grp[grp["gender"].isin(["Male", "Female"])]
        if g.empty:
            continue
        share = float((g["gender"] == "Female").mean())
        rows.append(
            {
                "seed_id": seed_id,
                "category": cat,
                "n_faces": len(g),
                "female_share": share,
                "abs_skew": abs(2 * (share - 0.5)),
            }
        )
    return pd.DataFrame(rows)


# --- checks -------------------------------------------------------------------


def check_asymmetry(run_dir: Path) -> dict[str, Any]:
    baseline_batches = _batches(run_dir / "baseline.jsonl")
    run_batches = _batches(run_dir / "run.jsonl")

    n_baseline_all = sum(len(b["paths"]) for b in baseline_batches)
    baseline_terminal = _terminal_paths(baseline_batches)
    iterative_terminal = _terminal_paths(run_batches)

    bf = _faces(run_dir, "fairface_baseline.jsonl")
    itf = _faces(run_dir, "fairface_iterative_terminal.jsonl")

    kl_baseline_all = _kl_by_category(bf)
    kl_baseline_term = _kl_by_category(bf[bf["image_path"].isin(baseline_terminal)])
    kl_iterative = _kl_by_category(itf)

    return {
        "n_images_baseline_all_batches": n_baseline_all,
        "n_batches_baseline": len(baseline_batches),
        "n_images_baseline_terminal_only": len(baseline_terminal),
        "n_images_iterative_terminal": len(iterative_terminal),
        "published_comparison_is_paired": n_baseline_all == len(iterative_terminal),
        "mean_delta_kl_as_published": _mean_delta(kl_baseline_all, kl_iterative),
        "mean_delta_kl_symmetric": _mean_delta(kl_baseline_term, kl_iterative),
    }


def check_pooling_cancellation(run_dir: Path) -> dict[str, Any]:
    itf = _faces(run_dir, "fairface_iterative_terminal.jsonl")
    per_seed = _seed_gender_profile(itf)

    out: dict[str, Any] = {}
    for cat, grp in per_seed.groupby("category"):
        faces = itf[itf["category"] == cat]
        faces = faces[faces["gender"].isin(["Male", "Female"])]
        composition = Counter(
            "all_female" if r.female_share == 1.0
            else "all_male" if r.female_share == 0.0
            else "mixed"
            for r in grp.itertuples()
        )
        out[str(cat)] = {
            "n_seeds": int(len(grp)),
            "pooled_female_share": round(float((faces["gender"] == "Female").mean()), 4),
            "pooled_kl_gender": round(
                axis_metrics(
                    faces[AXIS_COLUMN["gender"]].value_counts().to_dict(), AXIS_BUCKETS["gender"]
                ).kl_nats or 0.0,
                4,
            ),
            "mean_per_seed_abs_skew": round(float(grp["abs_skew"].mean()), 4),
            "composition": {
                "all_female": composition.get("all_female", 0),
                "all_male": composition.get("all_male", 0),
                "mixed": composition.get("mixed", 0),
            },
        }
    return out


def check_single_face_control(run_dir: Path) -> dict[str, Any]:
    bf = _faces(run_dir, "fairface_baseline.jsonl")
    itf = _faces(run_dir, "fairface_iterative_terminal.jsonl")

    baseline_batches = _batches(run_dir / "baseline.jsonl")
    run_batches = _batches(run_dir / "run.jsonl")
    n_gen_baseline = sum(len(b["paths"]) for b in baseline_batches)
    n_gen_iterative = len(_terminal_paths(run_batches))

    bf_sf = bf[bf["image_path"].isin(_single_face_paths(bf))]
    itf_sf = itf[itf["image_path"].isin(_single_face_paths(itf))]

    return {
        "faces_per_generated_image": {
            "baseline": round(len(bf) / n_gen_baseline, 3),
            "iterative_terminal": round(len(itf) / n_gen_iterative, 3),
        },
        "share_of_generated_images_with_a_face": {
            "baseline": round(bf["image_path"].nunique() / n_gen_baseline, 4),
            "iterative_terminal": round(itf["image_path"].nunique() / n_gen_iterative, 4),
        },
        "n_single_face_images": {
            "baseline": int(bf_sf["image_path"].nunique()),
            "iterative_terminal": int(itf_sf["image_path"].nunique()),
        },
        "mean_delta_kl_all_faces": _mean_delta(_kl_by_category(bf), _kl_by_category(itf)),
        "mean_delta_kl_single_face_only": _mean_delta(
            _kl_by_category(bf_sf), _kl_by_category(itf_sf)
        ),
    }


# --- main ---------------------------------------------------------------------


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    run_dir = Path(sys.argv[1])
    if not (run_dir / "run.jsonl").exists():
        print(f"error: {run_dir} has no run.jsonl")
        return 1
    missing = [
        n
        for n in ("baseline.jsonl", "fairface_baseline.jsonl", "fairface_iterative_terminal.jsonl")
        if not (run_dir / n).exists()
    ]
    if missing:
        print(f"error: missing {', '.join(missing)} — run `ouroboros report` with a baseline first")
        return 1

    report = {
        "run_id": run_dir.name,
        "asymmetry": check_asymmetry(run_dir),
        "pooling_cancellation": check_pooling_cancellation(run_dir),
        "single_face_control": check_single_face_control(run_dir),
    }

    out_dir = run_dir / "report"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "fairface_robustness.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    a = report["asymmetry"]
    print(f"run {report['run_id']}\n")
    print("1. Asimmetria baseline/iterative (docs 6.1)")
    print(f"   baseline, tutte le batch : {a['n_images_baseline_all_batches']} immagini "
          f"({a['n_batches_baseline']} batch)")
    print(f"   baseline, solo terminale : {a['n_images_baseline_terminal_only']} immagini")
    print(f"   iterative, terminale     : {a['n_images_iterative_terminal']} immagini")
    print(f"   appaiato come pubblicato : {a['published_comparison_is_paired']}")
    print(f"   mean delta KL pubblicato : {a['mean_delta_kl_as_published']}")
    print(f"   mean delta KL simmetrico : {a['mean_delta_kl_symmetric']}\n")

    print("2. Cancellazione da pooling (docs 6.2)")
    print(f"   {'categoria':<14} {'n_seed':>6} {'share_pooled':>13} {'KL_pooled':>10} "
          f"{'mean|skew|':>11}   composizione")
    for cat, v in sorted(report["pooling_cancellation"].items()):
        c = v["composition"]
        print(f"   {cat:<14} {v['n_seeds']:>6} {v['pooled_female_share']:>13.4f} "
              f"{v['pooled_kl_gender']:>10.4f} {v['mean_per_seed_abs_skew']:>11.4f}   "
              f"{c['all_female']}F + {c['all_male']}M + {c['mixed']} misti")

    s = report["single_face_control"]
    print("\n3. Controllo a volto singolo")
    print(f"   facce per immagine generata : {s['faces_per_generated_image']}")
    print(f"   quota immagini con volto    : {s['share_of_generated_images_with_a_face']}")
    print(f"   immagini a volto singolo    : {s['n_single_face_images']}")
    print(f"   mean delta KL tutti i volti : {s['mean_delta_kl_all_faces']}")
    print(f"   mean delta KL volto singolo : {s['mean_delta_kl_single_face_only']}")
    print(f"\n→ {out_dir / 'fairface_robustness.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
