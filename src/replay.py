from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime, timezone

from ouroboros.config import RunConfig
from ouroboros.targets import build_target
from ouroboros.storage import JSONLWriter, save_image, compute_sha256

logger = logging.getLogger(__name__)


async def run_replay(past_run_dir: Path, output_dir: Path) -> None:
    """Read prompts and seeds from a past run and regenerate the images to verify reproducibility."""
    meta_path = past_run_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Past run metadata not found at {meta_path}")

    with meta_path.open("r", encoding="utf-8") as f:
        meta_data = json.load(f)

    cfg_dict = meta_data.get("config", {})

    # Reconstruct the run config
    cfg = RunConfig(
        mode=cfg_dict.get("mode", "test"),
        attacker_model=cfg_dict.get("attacker_model", ""),
        flux_quantize=cfg_dict.get("flux_quantize", 4),
        flux_steps=cfg_dict.get("flux_steps", 4),
        flux_width=cfg_dict.get("flux_width", 512),
        flux_height=cfg_dict.get("flux_height", 512),
        judge_backend=cfg_dict.get("judge_backend", "gemini"),
        judge_model=cfg_dict.get("judge_model", ""),
        max_t2i_calls=cfg_dict.get("max_t2i_calls", 200),
        rate_limit_per_min=cfg_dict.get("rate_limit_per_min", 60),
        output_dir=str(output_dir),
        seeds_filter=cfg_dict.get("seeds_filter"),
        run_baseline=cfg_dict.get("run_baseline", False),
        allow_swap=cfg_dict.get("allow_swap", True),
        aggressive_unload=cfg_dict.get("aggressive_unload", True),
        ollama_host=cfg_dict.get("ollama_host", "http://localhost:11434"),
        google_cloud_project=cfg_dict.get("google_cloud_project", ""),
        google_cloud_location=cfg_dict.get("google_cloud_location", ""),
    )

    target = build_target(
        cfg.target_backend,
        flux_quantize=cfg.flux_quantize,
        flux_steps=cfg.flux_steps,
        flux_width=cfg.flux_width,
        flux_height=cfg.flux_height,
    )

    replay_run_id = f"replay_{past_run_dir.name}"
    replay_dir = output_dir / replay_run_id
    replay_dir.mkdir(parents=True, exist_ok=True)

    # Write metadata
    replay_meta = {
        **meta_data,
        "replay_of": past_run_dir.name,
        "replayed_at": datetime.now(timezone.utc).isoformat(),
    }
    (replay_dir / "meta.json").write_text(json.dumps(replay_meta, indent=2), encoding="utf-8")

    logger.info("═══ Starting Replay for run: %s ═══", past_run_dir.name)
    logger.info("Replay directory: %s", replay_dir)

    async def replay_file(filename: str) -> tuple[int, int]:
        source_jsonl = past_run_dir / filename
        if not source_jsonl.exists():
            return 0, 0

        logger.info("Replaying log: %s", filename)
        dest_writer = JSONLWriter(replay_dir / filename)
        
        matched_hashes = 0
        total_images = 0

        with source_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)

                target_prompt = record.get("target_prompt", "")
                seed_id = record.get("seed_id", "")
                iter_idx = record.get("iter", 0)
                is_baseline = (filename == "baseline.jsonl")
                iter_val = "baseline" if is_baseline else iter_idx

                samples_old = record.get("samples") or []
                m = len(samples_old)

                if m == 0:
                    dest_writer.append(record)
                    continue

                samples_raw = await target.generate_m(target_prompt, m)

                samples_new = []
                for idx, s in enumerate(samples_raw):
                    if s.outcome == "image" and s.image_bytes:
                        rel_path = save_image(replay_dir, seed_id, iter_idx=iter_val, sample_idx=idx, png_bytes=s.image_bytes)
                        new_hash = compute_sha256(s.image_bytes)

                        # Match with old hash
                        old_hash = ""
                        if idx < len(samples_old):
                            old_hash = samples_old[idx].get("sha256", "")

                        match = (new_hash == old_hash)
                        total_images += 1
                        if match:
                            matched_hashes += 1
                        else:
                            logger.warning(
                                "Hash mismatch for %s (iter %s, sample %d): expected %s, got %s",
                                seed_id, str(iter_val), idx, old_hash[:8] if old_hash else "none", new_hash[:8]
                            )

                        samples_new.append({
                            "path": rel_path,
                            "outcome": "image",
                            "sha256": new_hash,
                        })
                    else:
                        samples_new.append({
                            "path": None,
                            "outcome": s.outcome,
                            "error": s.error,
                        })

                new_record = {
                    **record,
                    "samples": samples_new,
                    "replayed_at": datetime.now(timezone.utc).isoformat(),
                }
                dest_writer.append(new_record)

        return matched_hashes, total_images

    try:
        base_match, base_total = await replay_file("baseline.jsonl")
        run_match, run_total = await replay_file("run.jsonl")
    finally:
        # Replay is target-only (no attacker/judge alternation), so there is no
        # RAM benefit to unloading between records — load once, free once at the end.
        await target.aclose()

    total = base_total + run_total
    matched = base_match + run_match
    match_rate = (matched / total) * 100 if total > 0 else None

    summary = {
        "replay_of": past_run_dir.name,
        "baseline": {"matched": base_match, "total": base_total},
        "run": {"matched": run_match, "total": run_total},
        "matched": matched,
        "total": total,
        "match_rate": match_rate,
        "replayed_at": replay_meta["replayed_at"],
    }
    (replay_dir / "replay_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    logger.info("═══ Replay Complete ═══")
    if total > 0:
        logger.info("Image Reproducibility: %d/%d (%.1f%%) identical SHA256 hashes", matched, total, match_rate)
    else:
        logger.info("No images generated during replay.")
    logger.info("Summary written to %s", replay_dir / "replay_summary.json")
