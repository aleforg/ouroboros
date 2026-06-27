# 02 — Architecture

## Vista a blocchi (end-to-end)

```mermaid
flowchart TD
    A[Seed prompts<br/>CLEAR-Bias × BOLD] --> B{per ogni seed}
    B --> C[Attacker LLM<br/>Ollama locale<br/>dolphin-llama3:8b]
    C -->|target_prompt| UNL1[unload attacker<br/>libera ~5 GB RAM]
    UNL1 --> D[Target T2I<br/>FLUX.2-klein-4B locale<br/>via mflux / MLX]
    D -->|M immagini sequenziali| UNL2[unload FLUX<br/>libera ~5 GB RAM]
    UNL2 --> RAM[RAM monitor<br/>psutil snapshot]
    RAM --> E[Judge VLM<br/>Gemini 2.5 Pro cloud<br/>zero RAM locale]
    E -->|bias_score + per_axis_scores| F{≥ N immagini<br/>≥ threshold?}
    F -->|sì| G[SUCCESS<br/>break]
    F -->|no, iter < max| H[Aggiorna memoria<br/>top-K + most recent]
    H --> C
    F -->|no, iter == max| I[FAIL]
    G --> J[Write run.jsonl<br/>+ ram.jsonl + checkpoint]
    I --> J
    J --> K[ouroboros report<br/>metrics + clustering]
```

## Vista compatta del loop interno

```
        ┌────────────────────── seed ──────────────────────┐
        │                                                  │
        │  iter = 0;   memory = []                         │
        │                                                  │
        │  ┌─ attacker → [unload] → target (×M seq) → ──┐  │
        │  │  [unload] → judge cloud                    │  │
        │  │                                            │  │
        │  │  crafted prompt                            │  │
        │  │       │                                    │  │
        │  │       ▼                                    │  │
        │  │  M images (sequential, FLUX.2 klein locale)│  │
        │  │       │                                    │  │
        │  │       ▼                                    │  │
        │  │  bias_score + per_axis + observed demo     │  │
        │  │  (Gemini cloud judge, 0 RAM locale)        │  │
        │  │       │                                    │  │
        │  │       ▼                                    │  │
        │  │  push to memory (top-K kept)               │  │
        │  │       │                                    │  │
        │  │       ▼                                    │  │
        │  │  if ≥ N of M ≥ threshold → SUCCESS, break  │  │
        │  │  if iter == max_iter   → FAIL, break       │  │
        │  │  if all M refused      → REFUSAL-pivot*    │  │
        │  └────────────────────────────────────────────┘  │
        │                                                  │
        └──────────────────────────────────────────────────┘

        * REFUSAL-pivot dead code finché c'è solo FLUX (no safety filter).
          Il branch resta nel loop pronto per quando arriverà un cloud target.
```

## Mappa dei moduli (`src/`)

```mermaid
flowchart LR
    subgraph CLI
        cli[cli.py]
    end
    subgraph Core
        config[config.py<br/>thresholds, defaults]
        seeds[seeds.py<br/>CLEAR-Bias adapter]
        loop[loop.py<br/>PAIR orchestrator]
        ram[ram.py<br/>psutil monitor]
    end
    subgraph Actors
        atk[attacker.py<br/>Ollama + Memory]
        subgraph Target
            tgt_b[targets/base.py<br/>Protocol + factory]
            tgt_f[targets/flux.py<br/>FLUX.2-klein-4B locale]
        end
        jdg["judge.py<br/>Gemini 2.5 Pro (fallback: MLX / Ollama)"]
    end
    subgraph Output
        store[storage.py<br/>JSONL + checkpoint]
        base[baseline.py<br/>single-shot]
        rep[report.py<br/>HTML + CSV + aggregate]
        met[metrics.py<br/>ASR + Wilson CI + std<br/>asr_vs_iter + variance<br/>aggregate_runs]
        clu[cluster.py<br/>HDBSCAN]
    end

    cli --> config
    cli --> seeds
    cli --> loop
    cli --> base
    cli --> rep
    cli -->|aggregate cmd| rep
    loop --> atk
    loop --> tgt_b
    loop --> jdg
    loop --> store
    loop --> ram
    tgt_b --> tgt_f
    rep --> met
    rep --> clu
    met --> store
    clu --> met
    atk --> config
    jdg --> config
    tgt_b --> config
```

### Responsabilità per modulo

| Modulo | Responsabilità | Dipendenze esterne |
|---|---|---|
| `config.py` | Costanti, threshold, dataclass `RunConfig`, `ModeBudget`, RAM check statico | — |
| `seeds.py` | Carica `data/base_prompts.jsonl` o 10 seed test; transform CLEAR-Bias → scene T2I | — |
| `attacker.py` | Client Ollama uncensored + `Memory` (top-K + most-recent) + `aclose()` lifecycle | `ollama`, `pydantic` |
| `targets/base.py` | `TargetBackend` Protocol, `SampleResult`, `RateLimiter`, `build_target()` factory. Solo `"flux"` supportato; raise ValueError per altri valori (estendibile aggiungendo un modulo sibling) | — |
| `targets/flux.py` | `FluxLocalTarget`: FLUX.2-klein-4B (distilled) via mflux ≥0.17, lazy load, M sequenziali, `aclose()` libera MLX cache | `mflux` |
| `judge.py` | `GeminiJudge` (default cloud) + `MLXJudge` (offline fallback) + `OllamaJudge`; schema Pydantic; JSON brace-counting | `google-genai`, `mlx-vlm`, `ollama`, `pydantic`, `Pillow` |
| `ram.py` | `RamMonitor`: psutil snapshot RSS + system memory ad ogni fase del loop; scrive `ram.jsonl` | `psutil` |
| `loop.py` | Orchestratore: attacker → [unload] → target → [unload] → judge; RAM snapshot; refusal pivot; success rule | tutto sopra |
| `storage.py` | `JSONLWriter`, `make_run_dir`, `save_image`, `write_checkpoint`, `load_checkpoint` | — |
| `baseline.py` | Single-shot baseline (no attacker, solo seed.base_scene → target → judge) | targets, judge, storage |
| `metrics.py` | `summary_per_seed`, `per_category` (con Wilson CI e std), `baseline_vs_iterative`, `wilson_ci`, `asr_vs_iter`, `intra_batch_variance`, `aggregate_runs` (multi-run) | `pandas` |
| `cluster.py` | Embedding strategie via `sentence-transformers`, clustering HDBSCAN, E(s) | `sentence-transformers`, `hdbscan` |
| `report.py` | Aggrega metrics + cluster → CSV + `report.html` + chart SVG inline (ASR-vs-iter); `run_aggregate_report()` per `aggregate_report.html` cross-run | `jinja2` |
| `cli.py` | argparse + entry point `ouroboros` — subcommands: `run`, `report`, `aggregate` | `python-dotenv` |

## Flusso di una singola iterazione

```mermaid
sequenceDiagram
    autonumber
    participant L as loop.run_one_seed
    participant A as Attacker (Ollama locale)
    participant T as Target (FLUX.2-klein locale)
    participant J as Judge (Gemini cloud)
    participant R as RamMonitor (psutil)
    participant S as Storage (JSONL)

    L->>R: snap(iter, seed, "pre_attacker")
    L->>A: propose(base_scene, memory)
    Note over L,A: retry 1× con stronger prefix se rifiuto
    A-->>L: AttackerCandidate{prompt, strategy, rationale}
    L->>R: snap(iter, seed, "post_attacker")

    Note over L,A: aggressive_unload → aclose() libera Ollama RAM
    L->>A: aclose()

    L->>R: snap(iter, seed, "pre_target")
    loop M volte (sequenziale)
        L->>T: generate_image(seed=base+i, prompt, steps=4)
        T-->>L: SampleResult{outcome="image", image_bytes}
    end
    L->>R: snap(iter, seed, "post_target")

    Note over L,T: aggressive_unload → aclose() svuota MLX cache
    L->>T: aclose()

    alt all M outcome=error (nessun refused su FLUX)
        L->>S: write record outcome=error
        L-->>L: next iter
    else almeno una image
        L->>J: judge(target_prompt, images, base_scene)
        Note over J: upload immagini a Vertex, zero RAM locale
        Note over J: retry 2× su JSON parse fail
        J-->>L: BiasJudgement | None
        L->>R: snap(iter, seed, "post_judge")

        alt judge_error
            L->>S: write record outcome=judge_error
            L-->>L: next iter (escluso da ASR)
        else judge ok
            L->>L: success = ≥N of M ≥ threshold?
            L->>S: write record + ram_gb compact
            L->>L: push to memory

            alt success
                L-->>L: BREAK
            else fail e iter < max
                L-->>L: next iter
            end
        end
    end

    L->>R: flush() → ram.jsonl
```

## Gestione RAM: sequencing aggressivo

Con target e attacker entrambi locali (post-ristrutturazione v1→v2), il picco di RAM si gestisce con **unload esplicito** tra le fasi, invece di tenerli entrambi residenti.

```
┌─────────────────── timeline iterazione (aggressive_unload=True) ──────────────────┐
│  t0: attacker (Ollama ~5 GB)  ──────► propose()         │  ~3 s                  │
│  t1: aclose() → Ollama evict                             │  <1 s                  │
│  t2: FLUX carica (~5 GB)  ──────────► generate ×M seq   │  M × ~9-15 s           │
│  t3: aclose() → del model + mx.metal.clear_cache()       │  <1 s                  │
│  t4: judge cloud (Gemini) ──────────► HTTP call          │  ~2-4 s  (0 GB locale) │
│      ─────────────────────────────────────────────────── │                        │
│  Peak locale: max(5, 5) = 5 GB attivi contemporaneamente │                        │
│  macOS background: ~3 GB                                  │                        │
│  TOTALE worst-case: ~8-10 GB su 16 GB M4                 │                        │
└────────────────────────────────────────────────────────────────────────────────────┘
```

Con `--no-aggressive-unload`: Ollama rimane caricato durante FLUX → picco combinato ~10 GB (warning a schermo, non abortisce).

`RamMonitor` registra un snapshot psutil a ciascuna delle 5 fasi: `pre_attacker`, `post_attacker`, `pre_target`, `post_target`, `post_judge`. Vengono scritti in `ram.jsonl` dopo ogni seed, e una versione compatta `ram_gb: {phase: gb}` viene inclusa direttamente nel record di `run.jsonl`.

Codice: `src/ram.py`, `src/loop.py`.

## Flusso di dati: `run.jsonl`

```
results/<run_id>/
├── run.jsonl       ← una riga per iterazione; include campo "ram_gb" compatto
├── ram.jsonl       ← snapshot RSS+system completi per tutte le fasi
├── checkpoint.json ← seed completati + global_calls (per --resume)
├── meta.json       ← config + timestamps
└── images/         ← PNG salvate per ogni iter × sample
    └── <seed_id>_iter<N>_sample<K>.png
```

Una riga `run.jsonl` (formato v2 con campo `target_backend` e `ram_gb`):

```json
{
  "run_id": "20250515_173000_abc",
  "seed_id": "gender_001",
  "category": "gender",
  "iter": 2,
  "target_prompt": "Vintage 1960s magazine CEO ...",
  "strategy_label": "historical_framing",
  "outcome": "success",
  "target_backend": "flux",
  "ram_gb": {"pre_attacker": 4.1, "post_attacker": 5.2, "pre_target": 3.8, "post_target": 8.1, "post_judge": 3.9}
}
```

## Comandi CLI esposti

Il binario `ouroboros` (entry point in `pyproject.toml` → `cli.main`) espone i subcommand:

| Subcommand | Scopo |
|---|---|
| `ouroboros run --mode {test,full} [flags]` | Esegue il loop PAIR e scrive `run.jsonl` (incl. `--resume`/`--replay`/`--baseline`/`--baseline-batches`) |
| `ouroboros report <run_id>` | Post-hoc: produce `report/` con CSV + `report.html` self-contained + chart SVG ASR-vs-iter |
| `ouroboros aggregate <run_id_1> <run_id_2> [...]` | Cross-run: aggrega N run indipendenti dello stesso config → ASR mean ± std per categoria + per-seed stability |
| `ouroboros validate-judge --dataset … --images-dir …` | Valida la classificazione demografica del judge contro il control set T2ISafety (accuracy/macro-F1/κ) — vedi [08-deviations.md](08-deviations.md) A.18 |
| `ouroboros dashboard` | Lancia la dashboard Streamlit (richiede l'extra `[web]`) |

Nota storica: nella v1/v2 `validate-judge` e `dashboard` erano stub (`"not implemented in v1"`); entrambi sono ora implementati (vedi [08-deviations.md](08-deviations.md) §B.3).

## Da dove proseguire

→ [03-pair-loop.md](03-pair-loop.md) per la teoria del loop iterativo
→ [04-components.md](04-components.md) per i dettagli di attacker / judge / target
