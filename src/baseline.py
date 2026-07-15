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
    batches_per_seed: dict[str, int] | None = None,
) -> None:
    """Static-prompt comparator: generate images directly from ``base_scene``,
    no attacker.

    Two modes (``cfg.baseline_mode``):

    * ``"single-shot"`` — exactly one batch per seed. Cheap smoke comparator.
    * ``"matched"`` — *budget-matched*: for each seed, generate as many
      independent base-scene batches as the iterative loop actually spent
      generating images on that seed (passed in ``batches_per_seed``). The
      report keeps the best batch per seed on both sides, so matching the number
      of draws is what makes ΔASR/ΔABS reflect the attacker's *search* rather
      than the mechanical advantage of taking a max over more draws. Falls back
      to one batch for any seed absent from the map.

    Success is not decided here — outcomes are logged as fail/refused and the
    report recomputes the label-based N-of-M rule symmetrically for both sides.
    """
    budget = cfg.budget
    matched = cfg.baseline_mode == "matched" and batches_per_seed is not None
    t_calls = 0

    for seed in tqdm(seeds, desc="baseline", unit="seed"):
        n_batches = 1
        if matched:
            n_batches = max(1, int(batches_per_seed.get(seed.seed_id, 1)))

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
                "attacker_rationale": f"Static baseline ({cfg.baseline_mode}) — no attacker",
                "samples": samples,
                "judge": judge_result.model_dump() if judge_result else None,
                "outcome": outcome,
                "success_rule": f"baseline_gender_majority_ge_{budget.success_n_of_m}_of_{budget.m}",
                "elapsed_ms": int((datetime.now(timezone.utc) - ts_start).total_seconds() * 1000),
                "t2i_calls_used_so_far": t_calls,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            writer.append(record)

    logger.info(
        "Baseline complete (%s) — %d seeds, %d T2I calls",
        cfg.baseline_mode, len(seeds), t_calls,
    )
