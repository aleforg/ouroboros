from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_XLSX = ROOT / "data/raw/bls/cpsaat11_2022_gpts_are_gpts.xlsx"
DEFAULT_CROSSWALK = ROOT / "data/bls_profession_crosswalk.tsv"
DEFAULT_SEEDS = ROOT / "data/stable_bias_prompts.jsonl"
DEFAULT_PARSED = ROOT / "data/raw/bls/cpsaat11_2022_parsed.csv"
DEFAULT_REFERENCE = ROOT / "data/bls_profession_reference.csv"
DEFAULT_GROUPS = ROOT / "data/profession_groups.json"
DEFAULT_MANIFEST = ROOT / "data/raw/bls/manifest.json"

BLS_YEAR = 2022
BLS_OFFICIAL_URL = "https://www.bls.gov/cps/aa2022/cpsaat11.htm"
BLS_FALLBACK_MIRROR_URL = (
    "https://raw.githubusercontent.com/openai/GPTs-are-GPTs/main/data/cpsaat11.xlsx"
)

GROUP_THRESHOLDS = {
    "male_coded": "women_share <= 0.33",
    "balanced": "0.33 < women_share < 0.60",
    "female_coded": "women_share >= 0.60",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _cell_col(ref: str) -> str:
    return re.match(r"[A-Z]+", ref).group(0)  # type: ignore[union-attr]


def _parse_xlsx_cells(path: Path) -> list[dict[str, str]]:
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as zf:
        shared_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
        shared = [
            "".join(t.text or "" for t in si.iter(ns + "t"))
            for si in shared_root.findall(ns + "si")
        ]
        sheet_root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))

    rows: list[dict[str, str]] = []
    for row in sheet_root.findall(".//" + ns + "row"):
        values: dict[str, str] = {}
        for cell in row.findall(ns + "c"):
            ref = cell.attrib.get("r", "")
            value_node = cell.find(ns + "v")
            value = "" if value_node is None else (value_node.text or "")
            if cell.attrib.get("t") == "s" and value:
                value = shared[int(value)]
            values[_cell_col(ref)] = value.strip()
        rows.append(values)
    return rows


def parse_bls_table(path: Path) -> list[dict[str, Any]]:
    rows = []
    for values in _parse_xlsx_cells(path):
        occupation = values.get("A", "").strip()
        total = values.get("B", "").strip()
        women = values.get("C", "").strip()
        white = values.get("D", "").strip()
        black = values.get("E", "").strip()
        asian = values.get("F", "").strip()
        hispanic = values.get("G", "").strip()
        if not occupation or occupation == "Occupation":
            continue
        if not total or not women or women in {"-", "–"}:
            continue
        try:
            total_employed = float(total.replace(",", ""))
            percent_women = float(women)
        except ValueError:
            continue
        rows.append(
            {
                "bls_occupation": occupation,
                "bls_year": BLS_YEAR,
                "total_employed_thousands": total_employed,
                "percent_women": percent_women,
                "women_share": percent_women / 100.0,
                "percent_white": _float_or_none(white),
                "percent_black": _float_or_none(black),
                "percent_asian": _float_or_none(asian),
                "percent_hispanic_or_latino": _float_or_none(hispanic),
                "source_url": BLS_OFFICIAL_URL,
            }
        )
    return rows


def _float_or_none(value: str) -> float | None:
    if value in {"", "-", "–"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _group_from_share(women_share: float | None) -> str:
    if women_share is None:
        return "balanced"
    if women_share <= 0.33:
        return "male_coded"
    if women_share >= 0.60:
        return "female_coded"
    return "balanced"


def _read_seeds(path: Path) -> dict[str, dict[str, str]]:
    seeds: dict[str, dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        seeds[row["profession"]] = {
            "seed_id": row["id"],
            "prompt": row["prompt"],
            "source_dataset": row.get("source_dataset", ""),
        }
    return seeds


def _read_crosswalk(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        return {row["profession"]: row for row in reader}


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def build_reference(
    bls_rows: list[dict[str, Any]],
    seeds: dict[str, dict[str, str]],
    crosswalk: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    missing = sorted(set(seeds) - set(crosswalk))
    extra = sorted(set(crosswalk) - set(seeds))
    if missing:
        raise SystemExit(f"Crosswalk missing professions: {missing}")
    if extra:
        raise SystemExit(f"Crosswalk contains unknown professions: {extra}")

    bls_by_name = {row["bls_occupation"]: row for row in bls_rows}
    out: list[dict[str, Any]] = []
    for profession, seed in seeds.items():
        cw = crosswalk[profession]
        bls_name = cw.get("bls_occupation", "").strip()
        confidence = cw.get("confidence", "").strip()
        include_primary = _truthy(cw.get("include_primary", ""))
        notes = cw.get("notes", "").strip()
        if include_primary and not bls_name:
            raise SystemExit(f"{profession}: include_primary=true but no bls_occupation")
        bls = bls_by_name.get(bls_name) if bls_name else None
        if bls_name and bls is None:
            raise SystemExit(f"{profession}: BLS occupation not found: {bls_name!r}")

        women_share = bls["women_share"] if bls else None
        group = _group_from_share(women_share)
        out.append(
            {
                "profession": profession,
                "seed_id": seed["seed_id"],
                "prompt": seed["prompt"],
                "source_dataset": seed["source_dataset"],
                "bls_occupation": bls_name,
                "bls_year": BLS_YEAR if bls else "",
                "total_employed_thousands": bls["total_employed_thousands"] if bls else "",
                "percent_women": bls["percent_women"] if bls else "",
                "women_share": round(women_share, 4) if women_share is not None else "",
                "group": group,
                "confidence": confidence,
                "include_primary": include_primary,
                "source_url": BLS_OFFICIAL_URL if bls else "",
                "notes": notes,
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_groups(path: Path, reference: list[dict[str, Any]]) -> None:
    mapping = {row["profession"]: row["group"] for row in reference}
    payload = {
        "_doc": {
            "purpose": "Maps Stable Bias professions to gender-stereotype groups derived from BLS women's employment share.",
            "source": "Generated by scripts/build_bls_reference.py from data/bls_profession_reference.csv.",
            "bls_source": "U.S. Bureau of Labor Statistics CPS Annual Averages Table 11.",
            "bls_year": BLS_YEAR,
            "official_url": BLS_OFFICIAL_URL,
            "thresholds": GROUP_THRESHOLDS,
            "caveats": (
                "This file is derived, not hand-authored. For statistically strong BLS validation, "
                "filter data/bls_profession_reference.csv to include_primary=true; low-confidence or "
                "generic profession prompts remain grouped for reporting but should not drive primary claims."
            ),
        },
        "mapping": mapping,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_manifest(path: Path, raw_xlsx: Path, parsed_csv: Path, reference_csv: Path) -> None:
    payload = {
        "dataset": "BLS CPS Annual Averages Table 11",
        "year": BLS_YEAR,
        "official_url": BLS_OFFICIAL_URL,
        "retrieval_url_used": BLS_FALLBACK_MIRROR_URL,
        "retrieval_note": (
            "Direct automated retrieval from bls.gov returned Access Denied in the local sandbox; "
            "the checked-in raw XLSX was retrieved from the public openai/GPTs-are-GPTs repository, "
            "which mirrors the BLS CPS Table 11 workbook. The official BLS HTML URL is recorded above."
        ),
        "generated_by": "scripts/build_bls_reference.py",
        "files": {
            str(raw_xlsx.relative_to(ROOT)): {"sha256": _sha256(raw_xlsx)},
            str(parsed_csv.relative_to(ROOT)): {"sha256": _sha256(parsed_csv)},
            str(reference_csv.relative_to(ROOT)): {"sha256": _sha256(reference_csv)},
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reproducible BLS profession reference files.")
    parser.add_argument("--raw-xlsx", type=Path, default=DEFAULT_RAW_XLSX)
    parser.add_argument("--crosswalk", type=Path, default=DEFAULT_CROSSWALK)
    parser.add_argument("--seeds", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--parsed-out", type=Path, default=DEFAULT_PARSED)
    parser.add_argument("--reference-out", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--groups-out", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    bls_rows = parse_bls_table(args.raw_xlsx)
    _write_csv(args.parsed_out, bls_rows)
    reference = build_reference(
        bls_rows=bls_rows,
        seeds=_read_seeds(args.seeds),
        crosswalk=_read_crosswalk(args.crosswalk),
    )
    _write_csv(args.reference_out, reference)
    _write_groups(args.groups_out, reference)
    _write_manifest(args.manifest_out, args.raw_xlsx, args.parsed_out, args.reference_out)

    n_primary = sum(1 for row in reference if row["include_primary"])
    print(f"Parsed BLS rows: {len(bls_rows)}")
    print(f"Reference rows: {len(reference)} ({n_primary} include_primary)")
    print(f"Wrote {args.reference_out}")


if __name__ == "__main__":
    main()
