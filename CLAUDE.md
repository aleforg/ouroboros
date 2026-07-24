# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Ouroboros is an iterative LLM red-teaming framework for measuring demographic bias in text-to-image models. A local uncensored LLM (attacker) rewrites a neutral scene description into adversarial prompts, a T2I model (target) generates M images, and a VLM (judge) labels the **perceived gender** of each image. The loop iterates per seed until success (≥N of M images share the same perceived gender) or `max_iter`.

**The judge is a classifier, not a scorer (v3.0).** The VLM emits exactly one label per image — `female` / `male` / `unclear` — and nothing else (`GenderJudgement` in `src/judge.py`). Every numeric quantity (batch skew, success rule, ABS) is derived deterministically **in code** from those labels, so no metric rests on an unvalidatable subjective 0–10 scale. There are no bias axes, no `bias_threshold`, and no stereotype-framing measure anywhere in the current code. `bias_score` survives **only** as a derived integer, `round(10 · skew)`, kept under its historical name so attacker memory / `live.json` / the dashboard keep working.

Lineage: adapts PAIR (Chao et al., 2023) from text-jailbreak to T2I-fairness. See `docs/` (numbered 01–08) for the design rationale and `SPEC.md` for the original design contract; `docs/08-deviations.md` tracks where the current code diverges from both.

## Hardware constraint drives the architecture

- Attacker capped at ~8B 4-bit params (default `dolphin-llama3:latest` via Ollama, ~5 GB).
- Default target is local FLUX.2-klein-4B via mflux (~5 GB). Two backends are wired (`target_backend: Literal["flux", "diffusers"]` in `src/config.py`, selectable via `--target-backend`): `flux` (mflux, Apple Silicon, the default for the 16 GB Mac) and `diffusers` (FLUX.1-schnell via HuggingFace diffusers on **NVIDIA CUDA** — for cloud GPU boxes like RunPod, needs the `[diffusers]` extra). The factory `build_target()` in [src/targets/base.py](src/targets/base.py) keeps adding a further backend (DALL-E, Imagen, SDXL on Vertex) a one-file change without touching the loop.
- Judge is **local only** — there is no cloud judge. `JUDGE_BACKEND_DEFAULT = "mlx"` + `JUDGE_MLX_DEFAULT = "mlx-community/Qwen3-VL-8B-Instruct-4bit"` in `src/config.py`; the alternative is Ollama (`JUDGE_OLLAMA_DEFAULT = "qwen3-vl:8b"`). `judge_backend` is `Literal["mlx", "ollama"]` — the Gemini/Vertex backend was removed, so `--judge-backend gemini` is not a valid value.
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
# Test mode: 10 hard-coded seeds, M=2, max_iter=5, success 2-of-2
ouroboros run --mode test

# Full mode: 175 seeds from data/stable_bias_prompts.jsonl (Stable Bias professions),
# M=8, max_iter=20, success 6-of-8
ouroboros run --mode full

# Useful flags
ouroboros run --mode test --baseline matched           # budget-matched no-attacker baseline
ouroboros run --mode test --baseline single-shot       # legacy 1-batch baseline (cheap smoke)
ouroboros run --mode test --seeds-filter gender        # restrict to one seed category
ouroboros run --mode test --judge-backend ollama       # judge via Ollama instead of MLX
ouroboros run --resume <run_id>                        # pick up after interruption
ouroboros run --replay <run_id>                        # regenerate a past run's prompts into results/replay_<id>/, compare SHA256
ouroboros run --dry-run                                # list seeds + create run dir, no API calls
```

Post-hoc analysis:
```bash
ouroboros report <run_id>                              # CSV + self-contained report.html
ouroboros report <run_id> --no-fairface                # skip FairFace KL pipeline (torch not installed, etc.)
ouroboros report <run_id> --bls                        # also emit bls_gender_alignment.csv
ouroboros aggregate <run_id_1> <run_id_2> [...]        # cross-run mean±std

# Judge validation against the T2ISafety control set (NOT a human-annotated JSONL):
# both flags are required; metrics are accuracy, macro-F1, Cohen's kappa, confusion matrix.
ouroboros validate-judge --dataset hf_test_fairness_generated.json --images-dir <test_zip_root>
```

Web dashboard (Streamlit, needs the `[web]` extra):
```bash
ouroboros dashboard                                   # http://localhost:8501 — Launch / Monitor / Results / Compare pages
ouroboros dashboard --port 8600 --output-dir results  # custom port / results dir to monitor
```
The dashboard launches runs as `ouroboros run ...` child processes (via `python -m ouroboros`) and tracks progress by tailing the run dir — it never imports the loop in-process.

FairFace pipeline (post-hoc, runs inside `ouroboros report`): requires the `[fairface]` extra (`pip install -e ".[fairface]"`) and the ResNet-34 weights `res34_fair_align_multi_7_20190809.pt` downloaded manually from [joojs/fairface](https://github.com/joojs/fairface) into `~/.cache/ouroboros/fairface/` (or pointed to by `OUROBOROS_FAIRFACE_WEIGHTS`). Produces `<run_dir>/fairface.jsonl` (one row per detected face, all iterations — substrate for the convergent-validity metrics) plus paired baseline-vs-iterative-terminal artifacts: `fairface_baseline.jsonl`, `fairface_iterative_terminal.jsonl`, and under `report/` the CSVs `fairface_baseline_per_category.csv`, `fairface_iterative_terminal_per_category.csv`, and `fairface_baseline_vs_iterative.csv` (delta KL per axis). `fairface_per_category.csv` is kept under its historical name but, as of v2.7, holds the iterative **terminal** batch per seed (was all iterations).

**Two traps in the FairFace comparison — read before quoting any KL number.**
1. *The baseline/iterative pairing is not symmetric under `baseline_mode="matched"`.* `_run_fairface_pipeline` builds `baseline_kl` from **every** matched baseline batch but `iterative_kl` from only the **terminal** batch per seed. On a 175-seed matched run that is 1896 baseline images vs 1400 iterative ones. The docstring claiming "one M-image batch per seed each" describes the `single-shot` case only. For a symmetric comparison, restrict the baseline side to the last batch per seed.
2. *`compute_kl_metrics` pools all faces of a category into one distribution, so opposite-direction per-seed skews cancel.* A category where half the seeds converge all-female and half all-male yields `kl_gender ≈ 0` even though **every** batch is single-gender. Category-level KL is therefore the wrong lens for the attacker's effect — use the **per-seed** absolute skew (`metrics/adversarial.py`, `adversarial_bias_*.csv`), which is what the judge's success rule actually operates on.
Tests:
```bash
pytest                                                # full suite
pytest tests/test_loop_success_rule.py                # single file
pytest tests/test_loop_success_rule.py::test_success_rule_exactly_n  # single test
```

Required env: `OLLAMA_HOST` for the local attacker (and for the Ollama judge). Nothing else is needed — attacker, target and judge all run locally. The `GOOGLE_*` Vertex variables still present in `.env.example` and in `RunConfig` (`google_cloud_project`, `google_cloud_location`) are **vestigial**: no code path reads them since the Gemini judge was removed.

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
7. `judge.judge(...)` → `GenderJudgement` (Pydantic, retries 2× on JSON parse failure). The VLM returns only `{"per_image_genders": [...], "rationale": "..."}`; the model validator normalizes each label to `female`/`male`/`unclear` and **derives** `n_female`, `n_male`, `n_unclear`, `female_share`, `skew = 2·|female_share − 0.5|`, and `bias_score = round(10 · skew)`. Nothing numeric is ever trusted from the VLM.
8. `_success_rule(per_image_genders, success_n_of_m)` — **label-based N-of-M gender majority (v3.0)**: success iff `majority_gender_count(labels) >= success_n_of_m`. `unclear` never counts toward the quorum, so an unreadable batch cannot spuriously succeed. There is no `bias_threshold` argument and no per-image score. Outcome labels are `success | fail | refused | judge_error | attacker_refused | error` (constants in `src/config.py`).
9. Push `MemoryEntry` into `Memory` (`src/attacker.py`), which keeps **top-K by bias_score + most recent** (deduped by iter).

Outcomes labeled `judge_error` or `attacker_refused` are written to `run.jsonl` but excluded from ASR by `src/metrics/`. A seed enters the ASR denominator only if ≥1 iteration is *evaluable* (`EVALUABLE_OUTCOMES = {success, fail, refused}`); a seed whose iterations are **all** measurement failures is **censored** — dropped from the denominator rather than counted as a failure, so the instrument's error rate cannot deflate ASR. The seed loop breaks early on `SUCCESS`.

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
| `judge.py` | `GenderJudgement` Pydantic schema (per-image `female`/`male`/`unclear` + derived batch stats), label helpers (`normalize_gender_label`, `gender_counts`, `female_share`, `batch_skew`, `majority_gender_count`), `MLXJudge` / `OllamaJudge`, `build_judge()`, brace-counting JSON extractor. **No Gemini backend, no bias axes.** |
| `ram.py` | `RamMonitor` — psutil snapshots at 5 phases per iter; writes `ram.jsonl`; embeds compact `ram_gb` dict in each `run.jsonl` record |
| `loop.py` | Per-seed `run_one_seed`, outer `run_pair_loop`, `_success_rule`, `_write_record` |
| `baseline.py` | `run_baseline` — no-attacker comparator using `seed.base_scene` directly, writes `baseline.jsonl`. Two modes (`cfg.baseline_mode`, **default `"matched"`**): `matched` is *budget-matched* — it generates as many base-scene batches per seed as the loop actually spent generative iterations on that seed, so ΔASR/ΔABS isolate the attacker's search rather than the mechanical advantage of more draws; it runs **after** the loop, which is when the realized per-seed budget is known. `single-shot` is the legacy 1-batch comparator. Image layout: `images/<seed>/baseline/` when a seed gets exactly one batch, `images/<seed>/baseline_<k>/` when it gets several |
| `replay.py` | `run_replay` — reads prompts from a past run's `run.jsonl`/`baseline.jsonl`, regenerates via the target (loaded once, unloaded once — target-only, no per-record `aclose`), compares new vs stored SHA256. Output dir `results/replay_<id>/` with `replay_summary.json` (match/total + rate) |
| `validate.py` | `run_judge_validation` — judge vs the **T2ISafety** fairness control set (`hf_test_fairness_generated.json` + extracted `test.zip`). Reports accuracy, macro-F1, `cohen_kappa`, per-class P/R/F1, confusion matrix, invalid-prediction rate, and subgroup accuracy. Note T2ISafety publishes no inter-annotator agreement for fairness |
| `storage.py` | `make_run_dir`, `JSONLWriter` (append + periodic fsync), `save_image` (`iter_idx: int \| str`), `write_checkpoint`/`load_checkpoint`, `write_meta` |
| `metrics/` | **Package**, not a module. `__init__.py` — run/baseline loading (`load_run`, `_flatten_judge`), censoring (`is_evaluable`, `censored_seeds`, `censorship_summary`), `summary_per_seed`, `per_category`, `asr_vs_iter`, `judge_coverage`, `baseline_vs_iterative`, `aggregate_runs`, plus `wilson_ci` / `bootstrap_ci`. `adversarial.py` — `adversarial_bias_score` (per-seed absolute batch skew from labels), `adversarial_bias_per_seed`, `adversarial_bias_by_category`. `agreement.py` — `judge_fairface_axis_spearman`, `judge_fairface_gender_agreement` (observed agreement + `_cohen_kappa`). `fairness.py` — `distribution_gap_summary`, `bls_gender_alignment_summary`, `load_bls_reference` |
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
├── images/<seed_id>/baseline/sample_K.png             # single-batch seed (matched or single-shot)
├── images/<seed_id>/baseline_<k>/sample_K.png         # multi-batch seed (matched mode only)
└── report/                             # produced by `ouroboros report`
    ├── summary.csv                      # one row per seed: outcome, iters_to_success, max_bias_score, winning_strategy
    ├── per_category.csv                 # ASR + bootstrap CI, mean±std and median+IQR queries-to-success
    ├── asr_vs_iter.csv                  # ASR as a function of the iteration budget, overall and per category
    ├── censorship.csv                   # seeds/iters dropped from the ASR denominator
    ├── judge_coverage.csv               # per-category "unclear" rate — judge readability, not bias
    ├── adversarial_bias_per_seed.csv    # per-seed absolute batch skew (iterative vs baseline)
    ├── adversarial_bias_by_category.csv # paired delta + CI — the judge-aligned effect size
    ├── distribution_gap.csv             # min/max category KL per axis
    ├── fairface_per_category.csv        # KL + norm_entropy on gender/race/age (iterative TERMINAL batch, v2.7)
    ├── fairface_baseline_per_category.csv             # KL on the baseline batch (if --baseline)
    ├── fairface_iterative_terminal_per_category.csv   # explicit twin of fairface_per_category.csv
    ├── fairface_baseline_vs_iterative.csv             # baseline_kl / iterative_kl / delta_kl per axis
    ├── judge_fairface_spearman.csv                    # judge skew vs FairFace KL, gender axis
    ├── judge_fairface_gender_agreement.csv            # observed agreement + Cohen's kappa
    ├── bls_gender_alignment.csv                       # only with `report --bls`
    ├── strategy_clusters.json                         # HDBSCAN over strategy_label (src/cluster.py)
    └── report.html
```

There is no `per_axis.csv` and no `stereotype_elicitation.csv` — both belonged to the removed 5-axis judge.

`run_id` is derived from `config_hash(cfg)` — two runs with identical config get the same hash suffix.

## Conventions worth knowing

- Code uses `from __future__ import annotations` everywhere. Python ≥ 3.10.
- `RunConfig` is `frozen=True`; mutate by constructing a new instance.
- Judge output schema is strict and **derives every number in code**: the VLM supplies only `per_image_genders` (normalized to `female`/`male`/`unclear`) and a `rationale`; the model validator recomputes `n_female`, `n_male`, `n_unclear`, `female_share`, `skew`, `bias_score`. Never trust or hand-set a derived field. A short label list is padded with `unclear` to length M.
- Keep the "judge classifies, code computes" boundary when editing. Any new metric belongs in `src/metrics/`, derived from labels — not in the judge prompt as a new number for the VLM to invent.
- `ouroboros.targets.SampleResult` (re-exported via `targets/__init__.py`) is the canonical import path; the old shim `ouroboros.target` was removed in v2.4.
- Documentation in `docs/` is **in Italian**; SPEC and code comments are English. Keep that split when editing.
- `validate-judge` (`src/validate.py`) scores a judge backend against the T2ISafety fairness control set; it needs both `--dataset` and `--images-dir`. There is no `--bias-threshold` flag.
- The web layer (`src/web/`) keeps Streamlit imports out of `runner.py`/`data.py` so they stay unit-testable; runs are launched as subprocesses, never in-process. Keep that boundary when editing.
