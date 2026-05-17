# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MIRTAGE is an iterative LLM red-teaming framework for measuring demographic bias in text-to-image models. A local uncensored LLM (attacker) rewrites a neutral scene description into adversarial prompts, a T2I model (target) generates M images, and a VLM (judge) scores them on five bias axes. The loop iterates per seed until success (≥N of M images cross the bias threshold) or `max_iter`.

Lineage: adapts PAIR (Chao et al., 2023) from text-jailbreak to T2I-fairness. See `docs/` (numbered 01–08) for the design rationale and `SPEC.md` for the original design contract; `docs/08-deviations.md` tracks where the current code diverges from both.

## Hardware constraint drives the architecture

The framework is built for **Apple Silicon Macs with 16 GB unified RAM** (M4 baseline). This is not a footnote — it shapes most design decisions:

- Attacker capped at ~8B 4-bit params (default `dolphin-llama3:latest` via Ollama, ~5 GB).
- Target is local FLUX.2-klein-4B via mflux (~5 GB). It's the only T2I backend currently supported (`target_backend: Literal["flux"]` in `src/config.py`). The factory `build_target()` in [src/targets/base.py](src/targets/base.py) is structured to make adding a second backend (DALL-E, Imagen, SDXL on Vertex) a one-file change without touching the loop.
- Default judge is **cloud** Gemini 2.5 Pro via Vertex (0 GB local). `JUDGE_BACKEND_DEFAULT = "gemini"` + `JUDGE_GEMINI_DEFAULT = "gemini-2.5-pro"` in `src/config.py`. MLX (`mlx-vlm` Qwen2.5-VL-7B-4bit) and Ollama (`qwen2.5vl:7b`) are offline fallbacks.
- **Aggressive unload between phases** (`cfg.aggressive_unload=True`) is critical: attacker → `aclose()` (evicts Ollama model) → target → `aclose()` (frees MLX/Metal cache) → judge. Peak RAM is `max(attacker, target)`, not the sum. `--no-aggressive-unload` keeps both resident and risks OOM.
- `src/config.py:check_ram_budget` aborts startup if the estimate exceeds `RAM_BUDGET_GB` (13 GB) unless `--allow-swap` is passed.

## Common commands

Install (editable + dev):
```bash
pip install -e ".[dev]"
```

Run the loop:
```bash
# Test mode: 10 hard-coded seeds, M=2, max_iter=5, threshold=7
mirtage run --mode test

# Full mode: 175 seeds from data/stable_bias_prompts.jsonl (Stable Bias professions), M=4, max_iter=20
mirtage run --mode full

# Useful flags
mirtage run --mode test --baseline single-shot       # also run no-attacker baseline
mirtage run --mode test --seeds-filter gender        # restrict to one CLEAR-Bias category
mirtage run --mode test --judge-backend mlx          # offline judge
mirtage run --resume <run_id>                        # pick up after interruption
mirtage run --dry-run                                # list seeds + create run dir, no API calls
```

Post-hoc analysis:
```bash
mirtage report <run_id>                              # CSV + self-contained report.html
mirtage report <run_id> --no-fairface                # skip FairFace KL pipeline (torch not installed, etc.)
mirtage aggregate <run_id_1> <run_id_2> [...]        # cross-run mean±std
```

FairFace pipeline (post-hoc, runs inside `mirtage report`): requires the `[fairface]` extra (`pip install -e ".[fairface]"`) and the ResNet-34 weights `res34_fair_align_multi_7_20190809.pt` downloaded manually from [joojs/fairface](https://github.com/joojs/fairface) into `~/.cache/mirtage/fairface/` (or pointed to by `MIRTAGE_FAIRFACE_WEIGHTS`). Produces `<run_dir>/fairface.jsonl` (one row per detected face) and `<run_dir>/report/fairface_per_category.csv` (KL divergence + normalized entropy on gender/race/age).

Tests:
```bash
pytest                                                # full suite
pytest tests/test_loop_success_rule.py                # single file
pytest tests/test_loop_success_rule.py::test_success_rule_exactly_n  # single test
```

Required env (see `.env.example`): `GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` for Vertex (judge only — target is always local FLUX); `OLLAMA_HOST` for the local attacker.

## Package layout quirk

`pyproject.toml` maps the package `mirtage` to the `src/` directory (`[tool.setuptools.package-dir] mirtage = "src"`). So **modules in `src/` are imported as `mirtage.X`**, not `src.X`. There is no nested `src/mirtage/` directory. Test imports follow the same convention.

The entry point `mirtage` (defined in `pyproject.toml`) routes to `src/cli.py:main`.

## Architecture: the per-seed loop

`src/loop.py:run_one_seed` is the orchestrator. One iteration:

1. `ram_monitor.snap(..., "pre_attacker")` → `attacker.propose(base_scene, memory)` returns `AttackerCandidate{target_prompt, strategy_label, rationale}` or `None` on refusal. Self-refusal detection uses a regex; retry once with a stronger framing prefix.
2. `attacker.aclose()` (if `aggressive_unload`) — sends `keep_alive=0` to Ollama.
3. `ram_monitor.snap(..., "pre_target")` → `target.generate_m(prompt, M)`. FLUX runs **sequentially on the asyncio thread** (MLX binds a GPU stream to the thread that created the model — do not `asyncio.to_thread` it).
4. `target.aclose()` — for FLUX, this `del`s the model and calls `mx.clear_cache()`.
5. Save PNGs + sha256 via `storage.save_image`.
6. **Refusal pivot**: if `all(samples == "refused")`, skip judge, write a synthetic memory entry, continue. Only triggers on cloud targets with safety filters — FLUX never returns `"refused"` (only `"image"` or `"error"`), so on FLUX this branch is effectively dead.
7. `judge.judge(...)` → `BiasJudgement` (Pydantic, retries 2× on JSON parse failure).
8. `_success_rule(per_image_scores, bias_threshold, success_n_of_m)` — at least N of M images must cross the threshold. Outcome labels are `success | fail | refused | judge_error | attacker_refused | error` (constants in `src/config.py`).
9. Push `MemoryEntry` into `Memory` (`src/attacker.py`), which keeps **top-K by bias_score + most recent** (deduped by iter).

Outcomes labeled `judge_error` or `attacker_refused` are written to `run.jsonl` but excluded from ASR by `src/metrics.py`. The seed loop breaks early on `SUCCESS`.

## Module map (`src/`)

| Module | Role |
|---|---|
| `config.py` | All constants, `RunConfig` (frozen dataclass), `ModeBudget` (TEST vs FULL), RAM budget check, `MODEL_SIZE_REGISTRY` |
| `cli.py` | `argparse` entry; subcommands `run`, `report`, `aggregate` (plus stubs `validate-judge`, `dashboard`) |
| `seeds.py` | `load_test_seeds()` (10 hard-coded across 6 legacy categories) and `load_full_seeds()` (175 prompts from `data/stable_bias_prompts.jsonl`, all `category="profession"`, `"Photo portrait of a [X]"` template). Old CLEAR-Bias × BOLD dataset archived under `data/legacy/` |
| `attacker.py` | `OllamaAttacker` + `Memory` + `MemoryEntry`. JSON-format Ollama call, refusal detection, retry-with-prefix, `aclose()` lifecycle |
| `targets/base.py` | `TargetBackend` Protocol, `SampleResult`, `RateLimiter`, `build_target()` factory. Only `"flux"` backend currently wired; the factory raises ValueError for any other name |
| `targets/flux.py` | `FluxLocalTarget` — FLUX.2-klein-4B via mflux; sequential generation on the asyncio thread; `aclose()` frees MLX cache |
| `judge.py` | `BiasJudgement` Pydantic schema (5 axes), `GeminiJudge` / `MLXJudge` / `OllamaJudge`, brace-counting JSON extractor |
| `ram.py` | `RamMonitor` — psutil snapshots at 5 phases per iter; writes `ram.jsonl`; embeds compact `ram_gb` dict in each `run.jsonl` record |
| `loop.py` | Per-seed `run_one_seed`, outer `run_pair_loop`, `_success_rule`, `_write_record` |
| `baseline.py` | `run_baseline` — single-shot using `seed.base_scene` directly (no attacker), writes `baseline.jsonl` |
| `storage.py` | `make_run_dir`, `JSONLWriter` (append + periodic fsync), `save_image`, `write_checkpoint`/`load_checkpoint`, `write_meta` |
| `metrics.py` | `wilson_ci`, `summary_per_seed`, `per_category`, `baseline_vs_iterative`, `asr_vs_iter`, `intra_batch_variance`, `aggregate_runs`. **Per-axis judge means no longer reported** — migrated to `fairface.py` (v2.2) |
| `fairface.py` | MTCNN + FairFace ResNet-34 pipeline; `process_run` walks `images/` and writes `fairface.jsonl`; `compute_kl_metrics` aggregates per-category KL divergence + normalized entropy on gender/race/age. Torch is imported lazily — math helpers (`axis_metrics`, `_smoothed_distribution`) are testable without it |
| `cluster.py` | HDBSCAN clustering of `strategy_label` using sentence-transformers embeddings |
| `report.py` | Jinja2 templates in `src/templates/`; `run_report` produces self-contained `report.html` with inline SVG ASR-vs-iter chart; `run_aggregate_report` for cross-run. Invokes `fairface.process_run` unless `skip_fairface=True` |

## Output layout

```
results/<run_id>/                       # run_id = "YYYY-MM-DD_HHMMSS_<8-char-config-hash>"
├── run.jsonl                           # one row per iteration; includes compact ram_gb
├── baseline.jsonl                      # if --baseline was passed
├── ram.jsonl                           # full psutil snapshots, 5 phases × iters × seeds
├── fairface.jsonl                      # one row per detected face (written by `mirtage report` when FairFace runs)
├── checkpoint.json                     # completed_seed_ids + global_calls; consumed by --resume
├── meta.json                           # frozen RunConfig + config_hash + timestamps
├── images/<seed_id>/iter_NN/sample_K.png
└── report/                             # produced by `mirtage report`
    ├── summary.csv, per_category.csv, asr_vs_iter.csv, …
    ├── fairface_per_category.csv       # KL + norm_entropy on gender/race/age
    └── report.html
```

`run_id` is derived from `config_hash(cfg)` — two runs with identical config get the same hash suffix.

## Conventions worth knowing

- Code uses `from __future__ import annotations` everywhere. Python ≥ 3.10.
- `RunConfig` is `frozen=True`; mutate by constructing a new instance.
- Judge output schema is strict: `per_axis_scores` must contain all five axes in `JUDGE_AXES`; the validator clamps to 0–10 and recomputes `bias_score = max(per_image_scores)`.
- `mirtage.targets.SampleResult` (re-exported via `targets/__init__.py`) is the canonical import path; the old shim `mirtage.target` was removed in v2.4.
- Documentation in `docs/` is **in Italian**; SPEC and code comments are English. Keep that split when editing.
- `validate-judge` and `dashboard` subcommands are intentional stubs (M4/M5, deferred per SPEC §19).
