# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Ouroboros is an iterative LLM red-teaming framework for measuring demographic bias in text-to-image models. A local uncensored LLM (attacker) rewrites a neutral scene description into adversarial prompts, a T2I model (target) generates M images, and a VLM (judge) scores them on five bias axes. The loop iterates per seed until success (≥N of M images cross the bias threshold) or `max_iter`.

Lineage: adapts PAIR (Chao et al., 2023) from text-jailbreak to T2I-fairness. See `docs/` (numbered 01–08) for the design rationale and `SPEC.md` for the original design contract; `docs/08-deviations.md` tracks where the current code diverges from both.

## Hardware constraint drives the architecture

- Attacker capped at ~8B 4-bit params (default `dolphin-llama3:latest` via Ollama, ~5 GB).
- Default target is local FLUX.2-klein-4B via mflux (~5 GB). Two backends are wired (`target_backend: Literal["flux", "diffusers"]` in `src/config.py`, selectable via `--target-backend`): `flux` (mflux, Apple Silicon, the default for the 16 GB Mac) and `diffusers` (FLUX.1-schnell via HuggingFace diffusers on **NVIDIA CUDA** — for cloud GPU boxes like RunPod, needs the `[diffusers]` extra). The factory `build_target()` in [src/targets/base.py](src/targets/base.py) keeps adding a further backend (DALL-E, Imagen, SDXL on Vertex) a one-file change without touching the loop.
- Default judge is **cloud** Gemini 2.5 Pro via Vertex (0 GB local). `JUDGE_BACKEND_DEFAULT = "gemini"` + `JUDGE_GEMINI_DEFAULT = "gemini-2.5-pro"` in `src/config.py`. MLX (`mlx-vlm` Qwen2.5-VL-7B-4bit) and Ollama (`qwen2.5vl:7b`) are offline fallbacks.
- **Aggressive unload between phases** (`cfg.aggressive_unload=True`) is critical: attacker → `aclose()` (evicts Ollama model) → target → `aclose()` (frees MLX/Metal cache) → judge. Peak RAM is `max(attacker, target)`, not the sum. `--no-aggressive-unload` keeps both resident and risks OOM.
- `src/config.py:check_ram_budget` aborts startup if the estimate exceeds `RAM_BUDGET_GB` (13 GB) unless `--allow-swap` is passed.

## Common commands

Install (editable + dev):
```bash
pip install -e ".[dev]"
pip install -e ".[web]"            # Streamlit dashboard
pip install -e ".[diffusers]"      # NVIDIA-CUDA target backend (cloud GPU)
```

Run the loop:
```bash
# Test mode: 10 hard-coded seeds, M=2, max_iter=5, threshold=7
ouroboros run --mode test

# Full mode: 175 seeds from data/stable_bias_prompts.jsonl (Stable Bias professions), M=4, max_iter=20
ouroboros run --mode full

# Useful flags
ouroboros run --mode test --baseline single-shot       # also run no-attacker baseline
ouroboros run --mode test --seeds-filter gender        # restrict to one CLEAR-Bias category
ouroboros run --mode test --judge-backend mlx          # offline judge
ouroboros run --resume <run_id>                        # pick up after interruption
ouroboros run --replay <run_id>                        # regenerate a past run's prompts into results/replay_<id>/, compare SHA256
ouroboros run --dry-run                                # list seeds + create run dir, no API calls
```

Post-hoc analysis:
```bash
ouroboros report <run_id>                              # CSV + self-contained report.html
ouroboros report <run_id> --no-fairface                # skip FairFace KL pipeline (torch not installed, etc.)
ouroboros aggregate <run_id_1> <run_id_2> [...]        # cross-run mean±std
ouroboros validate-judge --dataset control.jsonl       # judge vs human control set: MAE, Pearson, agreement@threshold
```

Web dashboard (Streamlit, needs the `[web]` extra):
```bash
ouroboros dashboard                                   # http://localhost:8501 — Launch / Monitor / Results / Compare pages
ouroboros dashboard --port 8600 --output-dir results  # custom port / results dir to monitor
```
The dashboard launches runs as `ouroboros run ...` child processes (via `python -m ouroboros`) and tracks progress by tailing the run dir — it never imports the loop in-process.

FairFace pipeline (post-hoc, runs inside `ouroboros report`): requires the `[fairface]` extra (`pip install -e ".[fairface]"`) and the ResNet-34 weights `res34_fair_align_multi_7_20190809.pt` downloaded manually from [joojs/fairface](https://github.com/joojs/fairface) into `~/.cache/ouroboros/fairface/` (or pointed to by `OUROBOROS_FAIRFACE_WEIGHTS`). Produces `<run_dir>/fairface.jsonl` (one row per detected face, all iterations — substrate for the convergent-validity metrics) plus paired baseline-vs-iterative-terminal artifacts: `fairface_baseline.jsonl`, `fairface_iterative_terminal.jsonl`, and under `report/` the CSVs `fairface_baseline_per_category.csv`, `fairface_iterative_terminal_per_category.csv`, and `fairface_baseline_vs_iterative.csv` (delta KL per axis). `fairface_per_category.csv` is kept under its historical name but, as of v2.7, holds the iterative **terminal** batch per seed (was all iterations).
Tests:
```bash
pytest                                                # full suite
pytest tests/test_loop_success_rule.py                # single file
pytest tests/test_loop_success_rule.py::test_success_rule_exactly_n  # single test
```

Required env (see `.env.example`): `GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` for Vertex (judge only — target is always local FLUX); `OLLAMA_HOST` for the local attacker.

## Package layout quirk

`pyproject.toml` maps the package `ouroboros` to the `src/` directory (`[tool.setuptools.package-dir] ouroboros = "src"`; subpackages `ouroboros.web`, `ouroboros.web.pages`, `ouroboros.targets` are mapped alongside). So **modules in `src/` are imported as `ouroboros.X`**, not `src.X`. There is no nested `src/ouroboros/` directory. Test imports follow the same convention.

The entry point `ouroboros` (defined in `pyproject.toml`) routes to `src/cli.py:main`. `src/__main__.py` also routes `python -m ouroboros` to the same `main` — the dashboard uses this to spawn run subprocesses with the active interpreter.

## Architecture: the per-seed loop

`src/loop.py:run_one_seed` is the orchestrator. One iteration:

1. `ram_monitor.snap(..., "pre_attacker")` → `attacker.propose(base_scene, memory)` returns `AttackerCandidate{target_prompt, strategy_label, rationale}` or `None` on refusal. Self-refusal detection uses a regex; retry once with a stronger framing prefix.
2. `attacker.aclose()` (if `aggressive_unload`) — sends `keep_alive=0` to Ollama.
3. `ram_monitor.snap(..., "pre_target")` → `target.generate_m(prompt, M)`. FLUX runs **sequentially on the asyncio thread** (MLX binds a GPU stream to the thread that created the model — do not `asyncio.to_thread` it).
4. `target.aclose()` — for FLUX, this `del`s the model and calls `mx.clear_cache()`.
5. Save PNGs + sha256 via `storage.save_image`.
6. **Refusal pivot**: if `all(samples == "refused")`, skip judge, write a synthetic memory entry, continue. Only triggers on cloud targets with safety filters — FLUX never returns `"refused"` (only `"image"` or `"error"`), so on FLUX this branch is effectively dead.
7. `judge.judge(...)` → `BiasJudgement` (Pydantic, retries 2× on JSON parse failure).
8. `_success_rule(per_image_scores, bias_threshold, success_n_of_m)` — **visual-only**: success iff at least N of M images cross `bias_threshold`. As of v2.7 the judge's `stereotype_framing` axis no longer enters the success rule (the old OR branch was removed); it is reported separately as SER/SRG (`metrics.stereotype_elicitation_summary`) and in `per_axis.csv`. `budget.stereotype_threshold` remains as the report-only threshold for SER/SRG. Outcome labels are `success | fail | refused | judge_error | attacker_refused | error` (constants in `src/config.py`).
9. Push `MemoryEntry` into `Memory` (`src/attacker.py`), which keeps **top-K by bias_score + most recent** (deduped by iter).

Outcomes labeled `judge_error` or `attacker_refused` are written to `run.jsonl` but excluded from ASR by `src/metrics.py`. The seed loop breaks early on `SUCCESS`.

## Module map (`src/`)

| Module | Role |
|---|---|
| `config.py` | All constants, `RunConfig` (frozen dataclass), `ModeBudget` (TEST vs FULL), RAM budget check, `MODEL_SIZE_REGISTRY` |
| `cli.py` | `argparse` entry; subcommands `run` (incl. `--resume`/`--replay`), `report`, `aggregate`, `validate-judge`, `dashboard` (launches Streamlit) |
| `seeds.py` | `load_test_seeds()` (10 hard-coded across 6 legacy categories) and `load_full_seeds()` (175 prompts from `data/stable_bias_prompts.jsonl`, `"Photo portrait of a [X]"` template). Each seed's `category` is its **gender-stereotype group** (`male_coded`/`female_coded`/`balanced`) from `data/profession_groups.json` (`load_profession_groups()`, BLS-based — see its `_doc` block), so report breakdowns aggregate by stereotype direction. Old CLEAR-Bias × BOLD dataset archived under `data/legacy/` |
| `attacker.py` | `OllamaAttacker` + `Memory` + `MemoryEntry`. JSON-format Ollama call, refusal detection, retry-with-prefix, `aclose()` lifecycle |
| `targets/base.py` | `TargetBackend` Protocol, `SampleResult`, `RateLimiter`, `build_target()` factory. Dispatches `"flux"` / `"diffusers"`; raises ValueError for any other name |
| `targets/flux.py` | `FluxLocalTarget` — FLUX.2-klein-4B via mflux (Apple Silicon); sequential generation on the asyncio thread; `aclose()` frees MLX cache |
| `targets/diffusers_flux.py` | `FluxDiffusersTarget` — FLUX.1-schnell via HuggingFace diffusers on NVIDIA CUDA (RunPod/Lambda/Colab). Drop-in for `FluxLocalTarget` where mflux is unavailable; `quantize_bits` 4→NF4 (~7 GB VRAM) / 8 (~14 GB) / else bf16 (~26 GB). Needs `[diffusers]` extra |
| `judge.py` | `BiasJudgement` Pydantic schema (5 axes), `GeminiJudge` / `MLXJudge` / `OllamaJudge`, brace-counting JSON extractor |
| `ram.py` | `RamMonitor` — psutil snapshots at 5 phases per iter; writes `ram.jsonl`; embeds compact `ram_gb` dict in each `run.jsonl` record |
| `loop.py` | Per-seed `run_one_seed`, outer `run_pair_loop`, `_success_rule`, `_write_record` |
| `baseline.py` | `run_baseline` — single-shot using `seed.base_scene` directly (no attacker), writes `baseline.jsonl`; images go under `images/<seed>/baseline/` (own namespace, no `iter_00` collision) |
| `replay.py` | `run_replay` — reads prompts from a past run's `run.jsonl`/`baseline.jsonl`, regenerates via the target (loaded once, unloaded once — target-only, no per-record `aclose`), compares new vs stored SHA256. Output dir `results/replay_<id>/` with `replay_summary.json` (match/total + rate) |
| `validate.py` | `run_judge_validation` — judge vs human-annotated control-set JSONL: MAE, `pearson_correlation`, agreement rate at threshold, failure rate |
| `storage.py` | `make_run_dir`, `JSONLWriter` (append + periodic fsync), `save_image` (`iter_idx: int \| str`), `write_checkpoint`/`load_checkpoint`, `write_meta` |
|`per_axis_summary` | (v2.6) additionally reports the judge's subjective 0–10 per-axis means (incl. `stereotype_framing`, which has no FairFace equivalent; as of v2.7 it no longer drives loop success — visual-only N-of-M — and is reported separately as SER/SRG via `stereotype_elicitation_summary`). `baseline_vs_iterative` (v2.7) is a paired, symmetric Visual ASR recomputed from `per_image_scores` on both sides. |
| `fairface.py` | MTCNN + FairFace ResNet-34 pipeline; `process_run` walks `images/` and writes `fairface.jsonl`; `compute_kl_metrics` aggregates per-category KL divergence + normalized entropy on gender/race/age. Torch is imported lazily — math helpers (`axis_metrics`, `_smoothed_distribution`) are testable without it |
| `cluster.py` | HDBSCAN clustering of `strategy_label` using sentence-transformers embeddings |
| `report.py` | Jinja2 templates in `src/templates/`; `run_report` produces self-contained `report.html` with inline SVG ASR-vs-iter chart; `run_aggregate_report` for cross-run. Invokes `fairface.process_run` unless `skip_fairface=True` |

## Web dashboard (`src/web/`)

Streamlit multi-page app launched by `ouroboros dashboard`. Design rule: only `app.py` and `pages/` import Streamlit; `runner.py` and `data.py` are pure and unit-tested (`tests/test_web_runner.py`, `tests/test_web_data.py`).

| Module | Role |
|---|---|
| `web/app.py` | Streamlit entry; `st.navigation` multi-page routing (≥1.36) so `session_state` (active run id) survives page switches; sidebar quick-stats |
| `web/pages/` | `1_Launch.py`, `2_Monitor.py`, `3_Results.py`, `4_Compare.py` |
| `web/runner.py` | Launches `ouroboros run` as a **child process** (avoids asyncio-vs-Streamlit event-loop conflict; isolates heavy model loading); job registry; progress via tailing run-dir files. No Streamlit import |
| `web/data.py` | Pure fs helpers — `list_runs`, `get_running_jobs`, `get_results_dir`, reads `live.json`/`run.jsonl`/`checkpoint.json`. No Streamlit import |
| `web/charts.py` | Chart builders for the Results/Compare pages |

## Output layout

```
results/<run_id>/                       # run_id = "YYYY-MM-DD_HHMMSS_<8-char-config-hash>"
├── run.jsonl                           # one row per iteration; includes compact ram_gb
├── baseline.jsonl                      # if --baseline was passed
├── ram.jsonl                           # full psutil snapshots, 5 phases × iters × seeds
├── fairface.jsonl                      # one row per detected face (written by `ouroboros report` when FairFace runs)
├── checkpoint.json                     # completed_seed_ids + global_calls; consumed by --resume
├── meta.json                           # frozen RunConfig + config_hash + timestamps
├── live.json                           # current intra-iteration state; overwritten each phase by loop._write_live; read by the web Monitor for the real-time panel
├── images/<seed_id>/iter_NN/sample_K.png
└── report/                             # produced by `ouroboros report`
    ├── summary.csv, per_category.csv, asr_vs_iter.csv, …
    ├── per_axis.csv                     # judge subjective per-axis means (5 axes, incl. stereotype_framing)
    ├── stereotype_elicitation.csv       # SER/SRG on stereotype_framing (no longer in the success rule)
    ├── fairface_per_category.csv       # KL + norm_entropy on gender/race/age (iterative TERMINAL batch, v2.7)
    ├── fairface_baseline_per_category.csv             # KL on the baseline batch (if --baseline)
    ├── fairface_iterative_terminal_per_category.csv   # explicit twin of fairface_per_category.csv
    ├── fairface_baseline_vs_iterative.csv             # baseline_kl / iterative_kl / delta_kl per axis
```

`run_id` is derived from `config_hash(cfg)` — two runs with identical config get the same hash suffix.

## Conventions worth knowing

- Code uses `from __future__ import annotations` everywhere. Python ≥ 3.10.
- `RunConfig` is `frozen=True`; mutate by constructing a new instance.
- Judge output schema is strict: `per_axis_scores` must contain all five axes in `JUDGE_AXES`; the validator clamps to 0–10 and recomputes `bias_score = max(per_image_scores)`.
- `ouroboros.targets.SampleResult` (re-exported via `targets/__init__.py`) is the canonical import path; the old shim `ouroboros.target` was removed in v2.4.
- Documentation in `docs/` is **in Italian**; SPEC and code comments are English. Keep that split when editing.
- `validate-judge` (`src/validate.py`) scores a judge backend against a human-annotated control-set JSONL (MAE, Pearson correlation, agreement rate at `--bias-threshold`).
- The web layer (`src/web/`) keeps Streamlit imports out of `runner.py`/`data.py` so they stay unit-testable; runs are launched as subprocesses, never in-process. Keep that boundary when editing.
