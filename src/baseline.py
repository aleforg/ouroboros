from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from ouroboros.config import LABEL_FAIL, LABEL_REFUSED, RunConfig
from ouroboros.judge import JudgeBackend
from ouroboros.seeds import Seed
from ouroboros.storage import JSONLWriter, compute_sha256, save_image
from ouroboros.targets import TargetBackend

logger = logging.getLogger(__name__)


async def run_baseline(
    seeds: list[Seed],
    cfg: RunConfig,
    target: TargetBackend,
    judge: JudgeBackend,
    writer: JSONLWriter,
    run_dir: Path,
) -> None:
    """Run one single-shot iteration per seed using the unmodified base_scene prompt."""
    budget = cfg.budget
    t_calls = 0

    for seed in tqdm(seeds, desc="baseline", unit="seed"):
        ts_start = datetime.now(timezone.utc)

        samples_raw = await target.generate_m(seed.base_scene, budget.m)
        t_calls += len(samples_raw)

        samples = []
        image_bytes_list = []
        for idx, s in enumerate(samples_raw):
            if s.outcome == "image" and s.image_bytes:
                rel_path = save_image(run_dir, seed.seed_id, iter_idx="baseline", sample_idx=idx, png_bytes=s.image_bytes)
                samples.append({"path": rel_path, "outcome": "image", "sha256": compute_sha256(s.image_bytes)})
                image_bytes_list.append(s.image_bytes)
            else:
                samples.append({"path": None, "outcome": s.outcome, "error": s.error})

        judge_result = None
        if image_bytes_list:
            judge_result = judge.judge(
                target_prompt=seed.base_scene,
                images=image_bytes_list,
                base_scene=seed.base_scene,
            )

        all_refused = all(s["outcome"] == "refused" for s in samples)
        outcome = LABEL_REFUSED if all_refused else LABEL_FAIL

        record = {
            "run_id": run_dir.name,
            "seed_id": seed.seed_id,
            "category": seed.category,
            "base_scene": seed.base_scene,
            "iter": 0,
            "target_prompt": seed.base_scene,
            "strategy_label": "baseline",
            "attacker_rationale": "Single-shot baseline — no attacker",
            "samples": samples,
            "judge": judge_result.model_dump() if judge_result else None,
            "outcome": outcome,
            "success_rule": f"baseline_ge_{budget.success_n_of_m}_of_{budget.m}_at_{budget.bias_threshold}",
            "elapsed_ms": int((datetime.now(timezone.utc) - ts_start).total_seconds() * 1000),
            "t2i_calls_used_so_far": t_calls,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        writer.append(record)

    logger.info("Baseline complete — %d seeds, %d T2I calls", len(seeds), t_calls)
