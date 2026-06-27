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
    """Run the single-shot baseline per seed using the unmodified base_scene prompt.

    ``cfg.baseline_batches`` (default 1) controls how many independent base-scene
    batches are generated per seed. With 1 this is the classic single-shot
    comparator. With ``max_iter`` it becomes a budget-matched (best-of-T static)
    baseline: the iterative loop draws up to T batches and the report keeps the
    max over them for ABS / N-of-M, so the baseline must be allowed the same T
    draws for ΔABS / ΔASR to isolate the attacker's search rather than the
    maximization advantage of drawing more batches.
    """
    budget = cfg.budget
    n_batches = max(1, int(getattr(cfg, "baseline_batches", 1)))
    t_calls = 0

    for seed in tqdm(seeds, desc="baseline", unit="seed"):
        for batch_idx in range(n_batches):
            ts_start = datetime.now(timezone.utc)

            # Preserve the historical image layout (images/<seed>/baseline/) when
            # only one batch is requested; namespace per batch otherwise.
            iter_tag = "baseline" if n_batches == 1 else f"baseline_{batch_idx}"

            samples_raw = await target.generate_m(seed.base_scene, budget.m)
            t_calls += len(samples_raw)

            samples = []
            image_bytes_list = []
            for idx, s in enumerate(samples_raw):
                if s.outcome == "image" and s.image_bytes:
                    rel_path = save_image(run_dir, seed.seed_id, iter_idx=iter_tag, sample_idx=idx, png_bytes=s.image_bytes)
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
                "iter": batch_idx,
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

    logger.info(
        "Baseline complete — %d seeds × %d batch(es), %d T2I calls",
        len(seeds), n_batches, t_calls,
    )
