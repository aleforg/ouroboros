#!/usr/bin/env python3
"""Re-score completed runs without re-generating a single image.

The comparator batches of `baseline_mode="matched"` are bit-identical copies: the
target seeds its sampler per-image (`seed_base + i*1000`) and never per-call, so
repeating `generate_m()` with the same neutral prompt returns the same 8 images.
The published ΔABS takes the max-skew batch on both sides, which climbs over the
attacker's genuinely different draws and cannot move over the comparator's copies.

    What this script CAN repair: the comparison. It matches the two sides on
    *effective independent draws* instead of nominal batch count and applies the
    same selector inside that budget, which removes the asymmetry.

    What NOTHING can repair after the fact: the missing independent comparator
    draws. Those images were never generated. With the runs as they stand every
    seed has exactly one independent comparator draw, so the corrected estimate
    answers "does one adversarial prompt beat one neutral prompt" — not "does the
    iterative *search* beat the comparator", which this data cannot answer.

Both readings are written out, side by side and clearly labelled, so nothing has
to be taken on trust.

Usage
-----
    python scripts/recompute_corrected.py results/<run_id> [results/<run_id> ...]
    python scripts/recompute_corrected.py results/A results/B --shared-seeds

Writes `<run_dir>/report_corrected/` and prints a summary. Never touches
run.jsonl, baseline.jsonl or any image.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ouroboros.config import FULL_BUDGET, TEST_BUDGET  # noqa: E402
from ouroboros.metrics import load_baseline, load_run  # noqa: E402
from ouroboros.metrics.dedup import (  # noqa: E402
    duplicate_audit,
    matched_budget_asr,
    matched_comparison_by_category,
    matched_effective_comparison,
    selection_sensitivity,
)

OUT_DIR_NAME = "report_corrected"


def _budget(run_dir: Path) -> tuple[int, int]:
    """(success_n_of_m, m) from the frozen mode in meta.json."""
    try:
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        mode = (meta.get("config") or {}).get("mode", "test")
    except Exception:
        mode = "test"
    b = TEST_BUDGET if mode == "test" else FULL_BUDGET
    return b.success_n_of_m, b.m


def _fmt_ci(row: pd.Series, lo: str, hi: str) -> str:
    a, b = row.get(lo), row.get(hi)
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return "—"
    return f"[{a:+.3f}, {b:+.3f}]"


def process(run_dir: Path, min_readable: int | None, keep_seeds: set[str] | None) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    success_n, m = _budget(run_dir)
    floor = success_n if min_readable is None else min_readable

    run_df = load_run(run_dir)
    base_df = load_baseline(run_dir)
    if run_df.empty:
        raise SystemExit(f"{run_dir}: run.jsonl mancante o vuoto")
    if base_df.empty:
        raise SystemExit(f"{run_dir}: baseline.jsonl mancante — non c'è nulla da confrontare")

    if keep_seeds is not None:
        run_df = run_df[run_df.seed_id.isin(keep_seeds)].reset_index(drop=True)
        base_df = base_df[base_df.seed_id.isin(keep_seeds)].reset_index(drop=True)

    out = run_dir / OUT_DIR_NAME
    out.mkdir(exist_ok=True)

    audit = duplicate_audit(run_df, base_df)
    per_seed = matched_effective_comparison(run_df, base_df, min_readable=floor, selector="max")
    by_cat = matched_comparison_by_category(per_seed)
    sens = selection_sensitivity(run_df, base_df, min_readable=floor)
    asr = matched_budget_asr(run_df, base_df, success_n_of_m=success_n)

    audit.to_csv(out / "duplicate_audit.csv", index=False)
    per_seed.to_csv(out / "abs_matched_per_seed.csv", index=False)
    by_cat.to_csv(out / "abs_matched_by_category.csv", index=False)
    sens.to_csv(out / "selection_sensitivity.csv", index=False)
    asr.to_csv(out / "matched_budget_asr.csv", index=False)

    side = audit.groupby("side")[
        ["n_batches", "n_distinct_draws", "n_duplicate_batches", "images_wasted"]
    ].sum()
    waste = {
        s: {
            "batches": int(side.loc[s, "n_batches"]),
            "distinct_draws": int(side.loc[s, "n_distinct_draws"]),
            "duplicate_batches": int(side.loc[s, "n_duplicate_batches"]),
            "images_wasted": int(side.loc[s, "images_wasted"]),
        }
        for s in side.index
    }
    total_images = int(sum(v["batches"] for v in waste.values()) * m)
    total_wasted = int(sum(v["images_wasted"] for v in waste.values()))

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "run_id": run_dir.name,
        "success_n_of_m": success_n,
        "m": m,
        "readability_floor": floor,
        "seeds_restricted_to": (len(keep_seeds) if keep_seeds is not None else None),
        "waste": waste,
        "images_generated": total_images,
        "images_wasted_on_duplicates": total_wasted,
        "wasted_share": round(total_wasted / total_images, 4) if total_images else None,
        "matched_abs": by_cat.to_dict("records"),
        "selection_sensitivity": sens.to_dict("records"),
        "matched_budget_asr": asr.to_dict("records"),
        "mean_matched_k": round(float(per_seed["matched_k"].mean()), 3) if not per_seed.empty else None,
        "caveat": (
            "Con matched_k=1 il confronto corretto vale 'un prompt avversariale contro "
            "un prompt neutro'. La domanda 'la ricerca iterativa batte il controllo' non "
            "è rispondibile con questi dati: il controllo non ha estrazioni indipendenti "
            "da opporre. Serve un nuovo run con il seme variato fra batch."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_note(out / "CORRECTIONS.md", summary, by_cat, sens, asr, audit)
    return summary


def _write_note(
    path: Path,
    summary: dict[str, Any],
    by_cat: pd.DataFrame,
    sens: pd.DataFrame,
    asr: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    L: list[str] = []
    A = L.append
    A(f"# Metriche ricalcolate — `{summary['run_id']}`")
    A("")
    A("Prodotto da `scripts/recompute_corrected.py`. Nessuna immagine è stata rigenerata.")
    A("")
    A("## Che cosa è stato corretto")
    A("")
    A(
        "I batch di controllo ripetuti sono copie bit-identiche del primo (il target semina "
        "il campionatore per immagine, mai per chiamata). La regola ufficiale «batch a skew "
        "massimo» premia quindi solo il lato avversariale. Qui i due lati sono appaiati sul "
        "numero di **estrazioni indipendenti**, non di batch."
    )
    A("")
    w = summary["waste"]
    A("| lato | batch | estrazioni indipendenti | batch duplicati | immagini sprecate |")
    A("|---|---|---|---|---|")
    for side in ("iterative", "baseline"):
        if side in w:
            v = w[side]
            A(
                f"| {side} | {v['batches']} | {v['distinct_draws']} | "
                f"{v['duplicate_batches']} | **{v['images_wasted']}** |"
            )
    A("")
    A(
        f"Totale sprecato: **{summary['images_wasted_on_duplicates']} immagini su "
        f"{summary['images_generated']}** "
        f"({(summary['wasted_share'] or 0) * 100:.1f}%). Estrazioni appaiate per seed "
        f"(k medio): **{summary['mean_matched_k']}**."
    )
    A("")
    A("## ΔABS corretto (selezione appaiata sulle estrazioni indipendenti)")
    A("")
    A("| categoria | n seed | ABS attacco | ABS controllo | Δ ABS | CI 95% | solido |")
    A("|---|---|---|---|---|---|---|")
    for _, r in by_cat.iterrows():
        A(
            f"| `{r['category']}` | {int(r['n_seeds'])} | {r['iterative_abs_mean']:.3f} | "
            f"{r['baseline_abs_mean']:.3f} | **{r['delta_abs_mean']:+.3f}** | "
            f"{_fmt_ci(r, 'delta_abs_ci_low', 'delta_abs_ci_high')} | "
            f"{'sì' if r['significant'] else 'no'} |"
        )
    A("")
    A("## Sensibilità alla regola di selezione")
    A("")
    A(
        "`matched=False` riproduce la metrica pubblicata (selettore su tutti i batch, "
        "duplicati compresi); `matched=True` la applica dentro il budget appaiato. "
        "La differenza fra le due righe è la dimensione dell'artefatto."
    )
    A("")
    A("| selettore | appaiato | n seed | Δ ABS | CI 95% | solido |")
    A("|---|---|---|---|---|---|")
    for _, r in sens.iterrows():
        A(
            f"| {r['selector']} | {'sì' if r['matched_on_effective_draws'] else 'no'} | "
            f"{int(r['n_seeds'])} | **{r['delta_abs_mean']:+.3f}** | "
            f"{_fmt_ci(r, 'delta_abs_ci_low', 'delta_abs_ci_high')} | "
            f"{'sì' if r['significant'] else 'no'} |"
        )
    A("")
    if summary["mean_matched_k"] == 1.0:
        A(
            "> Con k = 1 i tre selettori appaiati coincidono per costruzione: su una sola "
            "estrazione massimo, media e primo sono lo stesso valore. Non è un bug."
        )
        A("")
    if not asr.empty:
        r = asr.iloc[0]
        A("## Successo a budget appaiato (un'estrazione per lato)")
        A("")
        A(
            f"Attaccante **{int(r['attacker_hits'])}/{int(r['n_seeds'])} = {r['attacker_rate']:.1%}** "
            f"[{r['attacker_ci_low']:.1%}, {r['attacker_ci_high']:.1%}] contro prompt neutro "
            f"**{int(r['baseline_hits'])}/{int(r['n_seeds'])} = {r['baseline_rate']:.1%}** "
            f"[{r['baseline_ci_low']:.1%}, {r['baseline_ci_high']:.1%}] — "
            f"differenza **{r['delta']:+.1%}**, discordanti "
            f"{int(r['discordant_attacker_only'])} contro {int(r['discordant_baseline_only'])}, "
            f"McNemar esatto p = {r['mcnemar_exact_p']:.5f}."
        )
        A("")
    A("## Limite invalicabile")
    A("")
    A(summary["caveat"])
    A("")
    A(
        "Il fix a monte è in `targets/*.py`: far dipendere il seme anche dall'indice della "
        "chiamata, non solo da quello dell'immagine dentro il batch."
    )
    A("")
    path.write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Ricalcola le metriche di un run correggendo l'asimmetria dei batch duplicati."
    )
    ap.add_argument("run_dirs", nargs="+", type=Path, help="una o più directory results/<run_id>")
    ap.add_argument(
        "--min-readable",
        type=int,
        default=None,
        help="soglia di leggibilità per l'ABS (default: success_n_of_m del run)",
    )
    ap.add_argument(
        "--shared-seeds",
        action="store_true",
        help="con più run, limita l'analisi ai seed presenti in tutti",
    )
    args = ap.parse_args()

    keep: set[str] | None = None
    if args.shared_seeds and len(args.run_dirs) > 1:
        sets = []
        for d in args.run_dirs:
            df = load_run(d)
            if df.empty:
                raise SystemExit(f"{d}: run.jsonl mancante o vuoto")
            sets.append(set(df.seed_id.astype(str)))
        keep = set.intersection(*sets)
        print(f"Seed condivisi fra {len(args.run_dirs)} run: {len(keep)}\n")

    for d in args.run_dirs:
        s = process(d, args.min_readable, keep)
        w = s["waste"]
        print(f"=== {s['run_id']} ===")
        print(
            f"  duplicati: {s['images_wasted_on_duplicates']}/{s['images_generated']} immagini "
            f"({(s['wasted_share'] or 0) * 100:.1f}%)  ·  k appaiato medio {s['mean_matched_k']}"
        )
        for side in ("iterative", "baseline"):
            if side in w:
                v = w[side]
                print(
                    f"    {side:10s} {v['batches']:4d} batch -> {v['distinct_draws']:4d} "
                    f"estrazioni indipendenti ({v['duplicate_batches']} duplicati)"
                )
        for row in s["matched_abs"]:
            print(
                f"    ΔABS {row['category']:14s} n={row['n_seeds']:3d} "
                f"{row['delta_abs_mean']:+.3f} "
                f"[{row['delta_abs_ci_low']:+.3f},{row['delta_abs_ci_high']:+.3f}] "
                f"{'SOLIDO' if row['significant'] else ''}"
            )
        for row in s["matched_budget_asr"]:
            print(
                f"    successo a budget appaiato: attacco {row['attacker_rate']:.1%} vs "
                f"neutro {row['baseline_rate']:.1%} (McNemar p={row['mcnemar_exact_p']:.5f})"
            )
        print(f"  scritto in {Path(s['run_dir']) / OUT_DIR_NAME}\n")


if __name__ == "__main__":
    main()
