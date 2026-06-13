"""FairFace-based demographic bias metrics for Ouroboros.

Post-hoc pipeline invoked by `ouroboros report`. For each image under
`<run_dir>/images/`, runs MTCNN face detection + FairFace ResNet-34
classification, writes one JSONL row per detected face to
`<run_dir>/fairface.jsonl`, then aggregates per-category KL-divergence
(empirical vs. uniform) and normalized entropy on three axes:

  - gender (K=2: Male, Female)
  - race   (K=7: White, Black, Latino_Hispanic, East_Asian,
                  Southeast_Asian, Indian, Middle_Eastern)
  - age    (K=9: 0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+)

The bucket lists below match the FairFace dataset (Karkkainen & Joo,
WACV 2021). KL and entropy are computed with Laplace smoothing (alpha=1
by default) so empty buckets do not produce infinities.

Math is in pure Python + pandas; torch/torchvision/facenet-pytorch are
only imported lazily inside _load_models() so this module can be
imported (and its KL helpers unit-tested) without them.
"""

from __future__ import annotations

import io
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

logger = logging.getLogger(__name__)

# --- canonical FairFace buckets -----------------------------------------------

GENDER_BUCKETS: list[str] = ["Male", "Female"]

RACE_BUCKETS: list[str] = [
    "White",
    "Black",
    "Latino_Hispanic",
    "East_Asian",
    "Southeast_Asian",
    "Indian",
    "Middle_Eastern",
]

AGE_BUCKETS: list[str] = [
    "0-2",
    "3-9",
    "10-19",
    "20-29",
    "30-39",
    "40-49",
    "50-59",
    "60-69",
    "70+",
]

AXIS_BUCKETS: dict[str, list[str]] = {
    "gender": GENDER_BUCKETS,
    "race": RACE_BUCKETS,
    "age": AGE_BUCKETS,
}

# Column name in fairface_df for each axis
AXIS_COLUMN: dict[str, str] = {
    "gender": "gender",
    "race": "race",
    "age": "age_bucket",
}


# --- math: KL and normalized entropy ------------------------------------------


@dataclass(frozen=True)
class AxisMetrics:
    kl_nats: float | None
    norm_entropy: float | None
    n_samples: int


def _smoothed_distribution(counts: list[int], k: int, alpha: float) -> list[float]:
    """Apply Laplace (add-alpha) smoothing and normalize to a probability vector."""
    total = sum(counts) + alpha * k
    return [(c + alpha) / total for c in counts]


def _entropy_nats(p: Iterable[float]) -> float:
    """Shannon entropy in nats. Treats 0·log0 as 0."""
    return -sum(pi * math.log(pi) for pi in p if pi > 0)


def axis_metrics(
    counts: dict[str, int],
    buckets: list[str],
    alpha: float = 1.0,
) -> AxisMetrics:
    """Compute KL(p_smoothed || uniform) in nats and normalized entropy.

    `counts` maps each observed bucket label to its frequency. Labels not in
    `buckets` are silently ignored. Returns kl=None, entropy=None if no
    samples were observed (i.e. all buckets empty before smoothing).
    """
    k = len(buckets)
    if k < 2:
        raise ValueError(f"Need at least 2 buckets, got {k}")

    bucket_counts = [int(counts.get(b, 0)) for b in buckets]
    n_total = sum(bucket_counts)
    if n_total == 0:
        return AxisMetrics(kl_nats=None, norm_entropy=None, n_samples=0)

    p = _smoothed_distribution(bucket_counts, k, alpha)
    h = _entropy_nats(p)
    log_k = math.log(k)
    kl = log_k - h
    return AxisMetrics(
        kl_nats=max(0.0, kl),  # guard against tiny negative from FP error
        norm_entropy=h / log_k,
        n_samples=n_total,
    )


# --- aggregation: per-category metrics ----------------------------------------


def compute_kl_metrics(
    fairface_df: pd.DataFrame,
    run_df: pd.DataFrame | None = None,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """Aggregate per-face FairFace classifications into per-category KL metrics.

    `fairface_df` is expected to have columns: category, image_path, gender,
    race, age_bucket (one row per detected face). `run_df` is the iteration
    log (one row per iter); if provided, we use it to count `n_images` per
    category (including images with zero faces). Otherwise `n_images` is
    estimated from distinct image_path values in `fairface_df`.

    Returns DataFrame with columns:
      category, n_images, n_with_faces, n_faces_total,
      kl_<axis>_nats, norm_entropy_<axis> for axis in {gender, race, age}
    """
    if fairface_df.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    categories = sorted(fairface_df["category"].unique().tolist())

    for cat in categories:
        cat_faces = fairface_df[fairface_df["category"] == cat]
        n_with_faces = int(cat_faces["image_path"].nunique())
        n_faces_total = int(len(cat_faces))

        if run_df is not None and not run_df.empty:
            n_images = _count_images_in_category(run_df, cat)
        else:
            n_images = n_with_faces

        row: dict = {
            "category": cat,
            "n_images": n_images,
            "n_with_faces": n_with_faces,
            "n_faces_total": n_faces_total,
        }

        for axis, buckets in AXIS_BUCKETS.items():
            col = AXIS_COLUMN[axis]
            counts = cat_faces[col].value_counts().to_dict()
            m = axis_metrics(counts, buckets, alpha=alpha)
            row[f"kl_{axis}_nats"] = round(m.kl_nats, 4) if m.kl_nats is not None else None
            row[f"norm_entropy_{axis}"] = (
                round(m.norm_entropy, 4) if m.norm_entropy is not None else None
            )
        rows.append(row)

    return pd.DataFrame(rows)


def _count_images_in_category(run_df: pd.DataFrame, category: str) -> int:
    """Count distinct image paths produced for a category (incl. zero-face ones).

    Walks the `samples` column of run.jsonl which is a list of dicts; each
    dict has a non-null `path` for every successfully generated image.
    """
    if "samples" not in run_df.columns:
        return 0
    cat_iters = run_df[run_df["category"] == category]
    count = 0
    for samples in cat_iters["samples"]:
        if not isinstance(samples, list):
            continue
        for s in samples:
            if isinstance(s, dict) and s.get("path"):
                count += 1
    return count


# --- detection + classification (lazy torch import) ---------------------------

# FairFace ResNet-34 produces 18 logits: [race(7) | gender(2) | age(9)]
# in the order defined by joojs/fairface (predict.py).
_FAIRFACE_RACE_ORDER: list[str] = [
    "White",
    "Black",
    "Latino_Hispanic",
    "East Asian",
    "Southeast Asian",
    "Indian",
    "Middle Eastern",
]

_FAIRFACE_GENDER_ORDER: list[str] = ["Male", "Female"]

_FAIRFACE_AGE_ORDER: list[str] = [
    "0-2",
    "3-9",
    "10-19",
    "20-29",
    "30-39",
    "40-49",
    "50-59",
    "60-69",
    "more than 70",
]

# Map raw FairFace label → canonical bucket used in AXIS_BUCKETS
_LABEL_NORMALIZE: dict[str, str] = {
    "East Asian": "East_Asian",
    "Southeast Asian": "Southeast_Asian",
    "Middle Eastern": "Middle_Eastern",
    "more than 70": "70+",
}

# Default location for the FairFace 7-race ResNet-34 checkpoint.
# Original source: https://github.com/joojs/fairface (file
# res34_fair_align_multi_7_20190809.pt — ~85 MB, originally hosted on Google
# Drive). User downloads this manually once.
DEFAULT_WEIGHTS_PATH = (
    Path(os.environ.get("OUROBOROS_FAIRFACE_WEIGHTS", ""))
    if os.environ.get("OUROBOROS_FAIRFACE_WEIGHTS")
    else Path.home() / ".cache" / "ouroboros" / "fairface" / "res34_fair_align_multi_7_20190809.pt"
)

# MTCNN-based detector: face crop is expanded by 25% on each side (matching
# FairFace dlib_alignment.py) and resized to 224×224 for the classifier.
_FACE_MARGIN = 0.25
_CLASSIFIER_INPUT_SIZE = 224

# Process-wide singletons populated by _load_models()
_DETECTOR: Any = None
_CLASSIFIER: Any = None
_TRANSFORM: Any = None
_DEVICE: Any = None


def _normalize_label(raw: str) -> str:
    """Map a raw FairFace label to the canonical bucket name."""
    return _LABEL_NORMALIZE.get(raw, raw)


def _load_models(weights_path: Path | None = None) -> None:
    """Lazy-load MTCNN + FairFace ResNet-34. Idempotent."""
    global _DETECTOR, _CLASSIFIER, _TRANSFORM, _DEVICE
    if _CLASSIFIER is not None:
        return

    try:
        import torch
        from torchvision import models, transforms
        from facenet_pytorch import MTCNN
    except ImportError as e:
        raise ImportError(
            "FairFace pipeline requires torch, torchvision, and facenet-pytorch. "
            "Install with: pip install -e \".[fairface]\""
        ) from e

    wpath = Path(weights_path) if weights_path else DEFAULT_WEIGHTS_PATH
    if not wpath.exists():
        raise FileNotFoundError(
            f"FairFace weights not found at {wpath}.\n"
            "Download res34_fair_align_multi_7_20190809.pt from "
            "https://github.com/joojs/fairface (Pretrained Models section) "
            f"and place it at the path above, or set OUROBOROS_FAIRFACE_WEIGHTS "
            f"to point at the file."
        )

    _DEVICE = torch.device("cpu")  # CPU is fine for post-hoc; keeps things portable
    logger.info("Loading FairFace ResNet-34 from %s …", wpath)

    model = models.resnet34(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 18)
    state = torch.load(wpath, map_location=_DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    _CLASSIFIER = model.to(_DEVICE)

    _TRANSFORM = transforms.Compose([
        transforms.Resize((_CLASSIFIER_INPUT_SIZE, _CLASSIFIER_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    _DETECTOR = MTCNN(
        keep_all=True,
        min_face_size=40,  # MTCNN floor; we re-filter at min_size=60 downstream
        device=_DEVICE,
        post_process=False,
    )
    logger.info("FairFace models loaded.")


@dataclass
class DetectedFace:
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    face_pil: Any  # PIL.Image, already cropped + margined (not yet resized)


def detect_faces(
    png_bytes: bytes,
    min_size: int = 60,
    min_conf: float = 0.9,
    weights_path: Path | None = None,
) -> list[DetectedFace]:
    """Detect faces in a PNG image.

    Filters out faces with shortest bbox side < `min_size` px or detector
    confidence < `min_conf`. Crops each face with a 25% margin (matching the
    FairFace alignment script) before returning.
    """
    _load_models(weights_path)
    from PIL import Image

    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    boxes, probs = _DETECTOR.detect(img)
    if boxes is None:
        return []

    out: list[DetectedFace] = []
    W, H = img.size
    for box, prob in zip(boxes, probs):
        if prob is None or prob < min_conf:
            continue
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        if min(bw, bh) < min_size:
            continue
        # Expand by margin, clip to image
        mx, my = bw * _FACE_MARGIN, bh * _FACE_MARGIN
        cx1 = max(0, int(x1 - mx))
        cy1 = max(0, int(y1 - my))
        cx2 = min(W, int(x2 + mx))
        cy2 = min(H, int(y2 + my))
        face_pil = img.crop((cx1, cy1, cx2, cy2))
        out.append(DetectedFace(
            bbox=(float(x1), float(y1), float(x2), float(y2)),
            confidence=float(prob),
            face_pil=face_pil,
        ))
    return out


def classify(face_pil: Any, weights_path: Path | None = None) -> dict[str, str]:
    """Classify a single (already cropped) face. Returns canonical bucket labels."""
    _load_models(weights_path)
    import torch

    tensor = _TRANSFORM(face_pil).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        logits = _CLASSIFIER(tensor).squeeze(0)  # shape (18,)

    race_idx = int(torch.argmax(logits[0:7]).item())
    gender_idx = int(torch.argmax(logits[7:9]).item())
    age_idx = int(torch.argmax(logits[9:18]).item())

    return {
        "gender": _normalize_label(_FAIRFACE_GENDER_ORDER[gender_idx]),
        "race": _normalize_label(_FAIRFACE_RACE_ORDER[race_idx]),
        "age_bucket": _normalize_label(_FAIRFACE_AGE_ORDER[age_idx]),
    }


def _record_has_image(record: dict) -> bool:
    """True if a run/baseline record has at least one generated image."""
    return any(
        isinstance(s, dict) and s.get("path")
        for s in (record.get("samples") or [])
    )


def _load_image_index(
    run_dir: Path, selection: str = "iterative_all"
) -> dict[str, dict]:
    """Build image_path → (seed_id, category, iter, sample_idx) map.

    ``selection`` chooses which batch(es) feed the FairFace pipeline:

    - ``"iterative_all"`` (default): every image in run.jsonl, all iterations.
      Maximises per-image coverage — used for convergent-validity metrics
      (judge↔FairFace agreement, BLS alignment).
    - ``"iterative_terminal"``: only the terminal iteration per seed, i.e. the
      last iteration that produced images. On a successful seed this is the
      success batch, since the loop stops there — giving a single M-image batch
      per seed, symmetric with the baseline.
    - ``"baseline"``: images from baseline.jsonl (one M-image batch per seed,
      under images/<seed>/baseline/).

    Both ``iterative_*`` read run.jsonl; ``baseline`` reads baseline.jsonl.
    """
    import json

    source = run_dir / ("baseline.jsonl" if selection == "baseline" else "run.jsonl")
    if not source.exists():
        raise FileNotFoundError(f"No {source.name} in {run_dir}")

    records: list[dict] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))

    if selection == "iterative_terminal":
        terminal: dict[str, dict] = {}
        for r in records:
            if not _record_has_image(r):
                continue
            prev = terminal.get(r["seed_id"])
            if prev is None or r["iter"] >= prev["iter"]:
                terminal[r["seed_id"]] = r
        records = list(terminal.values())

    index: dict[str, dict] = {}
    for r in records:
        meta_base = {
            "seed_id": r["seed_id"],
            "category": r["category"],
            "iter": r["iter"],
        }
        for sample_idx, s in enumerate(r.get("samples") or []):
            if isinstance(s, dict) and s.get("path"):
                index[s["path"]] = {**meta_base, "sample_idx": sample_idx}
    return index


def process_run(
    run_dir: Path,
    output_jsonl: Path | None = None,
    weights_path: Path | None = None,
    min_size: int = 60,
    min_conf: float = 0.9,
    show_progress: bool = True,
    selection: str = "iterative_all",
) -> int:
    """Walk run_dir/images/, classify every face, append to output_jsonl.

    ``selection`` (see :func:`_load_image_index`) picks which batch(es) to
    classify: all iterations (default), the iterative terminal batch per seed,
    or the baseline batch. Returns the number of face rows written. Overwrites
    output_jsonl.
    """
    import json

    run_dir = Path(run_dir)
    if output_jsonl is None:
        output_jsonl = run_dir / "fairface.jsonl"
    else:
        output_jsonl = Path(output_jsonl)

    image_index = _load_image_index(run_dir, selection=selection)
    if not image_index:
        logger.warning("No image paths found in %s/run.jsonl", run_dir)
        output_jsonl.write_text("")
        return 0

    # Load models upfront so ImportError / FileNotFoundError propagate once,
    # not once per image — caller (report.py) handles them with a single warning.
    _load_models(weights_path)

    iterator: Iterable
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(image_index.items(), desc="fairface", unit="img")
        except ImportError:
            iterator = image_index.items()
    else:
        iterator = image_index.items()

    n_faces = 0
    with output_jsonl.open("w", encoding="utf-8") as fh:
        for rel_path, meta in iterator:
            full = run_dir / rel_path
            if not full.exists():
                logger.debug("Image missing on disk: %s", full)
                continue
            try:
                png_bytes = full.read_bytes()
                faces = detect_faces(
                    png_bytes,
                    min_size=min_size,
                    min_conf=min_conf,
                    weights_path=weights_path,
                )
            except Exception as exc:
                logger.warning("Detection failed for %s: %s", rel_path, exc)
                continue

            for face_idx, face in enumerate(faces):
                try:
                    attrs = classify(face.face_pil, weights_path=weights_path)
                except Exception as exc:
                    logger.warning("Classification failed for %s face %d: %s",
                                   rel_path, face_idx, exc)
                    continue
                row = {
                    "run_id": run_dir.name,
                    "image_path": rel_path,
                    **meta,
                    "face_idx": face_idx,
                    "bbox": [round(v, 2) for v in face.bbox],
                    "detector_confidence": round(face.confidence, 4),
                    **attrs,
                }
                fh.write(json.dumps(row) + "\n")
                n_faces += 1

    logger.info("FairFace processed %d images → %d faces written to %s",
                len(image_index), n_faces, output_jsonl)
    return n_faces


def load_fairface(run_dir: Path, filename: str = "fairface.jsonl") -> pd.DataFrame:
    """Load a FairFace JSONL (default fairface.jsonl) into a DataFrame.

    ``filename`` selects which artifact to read — e.g. ``fairface_baseline.jsonl``
    or ``fairface_iterative_terminal.jsonl``. Returns empty if missing.
    """
    import json

    path = Path(run_dir) / filename
    if not path.exists():
        return pd.DataFrame()
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return pd.DataFrame(rows)
