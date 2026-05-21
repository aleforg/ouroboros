# 01 — Overview

## Cos'è Ouroboros

Ouroboros è un **framework di red-teaming iterativo** per modelli text-to-image (T2I). L'obiettivo è quantificare quanto un *attaccante automatico* (un LLM "uncensored" che riformula prompt) riesca a far emergere **bias demografici** o **stereotipi** da un modello T2I (FLUX.2-klein-4B locale via mflux), partendo da scene description neutrali.

L'idea è derivata dal lavoro di Chao et al. su PAIR ([Chao et al., 2023](07-references.md#pair)) per i jailbreak su LLM, ma adattata al dominio **immagini** invece che testo.

## Differenza chiave: cosa cambia nel dominio T2I

Il PAIR originale ha tre attori: **attacker LLM**, **target LLM**, **judge LLM**. Tutti e tre operano su testo, e il judge legge l'output testuale del target.

Nel nostro caso:

| Componente | PAIR originale | Ouroboros |
|---|---|---|
| **Attacker** | LLM uncensored | LLM uncensored (Ollama, locale) |
| **Target** | LLM black-box | T2I model locale (FLUX.2-klein-4B via mflux) |
| **Judge** | LLM (cloud) | **VLM** cloud (Gemini 2.5 Pro via Vertex AI) |

Il judge in Ouroboros deve **guardare le immagini**, non leggere testo — di qui la scelta di un VLM. Questo è il delta più importante rispetto al paper originale.

## Goals

1. Quantificare di quanto il prompt engineering iterativo aumenti il tasso di bias rispetto al **baseline single-shot** (un solo prompt, niente loop).
2. Produrre metriche **publication-grade**: ASR (Attack Success Rate), queries-to-success, distribuzioni per-asse di bias, effectiveness per strategia.
3. Riusare la tassonomia di [CLEAR-Bias](07-references.md#clear-bias) per permettere comparazioni cross-pipeline con la suite testuale `fairness-eval/`.

## Non-goals (v1)

- **Comparare più target T2I** (solo FLUX.2-klein locale in v1; cloud target — DALL-E, Imagen, SDXL Vertex — rimandati a v2 via factory `build_target()` in `src/targets/base.py`).
- **Attacchi white-box** (gradient-based): solo API chiuse.
- **Dialoghi multi-turn col target**: ogni iterazione è una singola chiamata T2I con un prompt nuovo.
- **Sostituire** il generatore statico `generate_image_fairness_dataset.py` esistente: quello rimane come fonte di base prompts.

## Vincolo hardware (importante)

Il framework è progettato per girare su **Mac Apple Silicon con 16 GB di RAM unificata** (M4 baseline).

```
┌─────────────── 16 GB unified memory ───────────────┐
│  ~6 GB  macOS + browser + IDE                      │
│  ~5 GB  attacker LLM   (Ollama, dolphin-llama3:8b) │
│  ~5 GB  target T2I     (FLUX.2-klein-4B via mflux) │
│   0 GB  judge VLM      (Gemini 2.5 Pro, cloud)     │
└─────────────────────────────────────────────────────┘
```

Questo vincolo guida la scelta principale di design:

1. **Dimensione dell'attacker** → max 8B parametri quantizzati (no Mixtral, no 13B). Default `dolphin-llama3:8b`.
2. **Judge in cloud** (Gemini 2.5 Pro via Vertex) → 0 GB locali, lasciando tutta la RAM disponibile per FLUX e l'attacker in sequenza.

Se la RAM combinata sfora il budget (configurabile via `RAM_BUDGET_GB`, default 10 GB), la CLI **aborta lo startup** a meno di passare `--allow-swap`. La logica è in `src/config.py:129` (`check_ram_budget`).

## Posizionamento nella letteratura

Ouroboros è un esempio di **automated adversarial red-teaming** applicato alla *fairness* di sistemi multimodali. Le sue radici intellettuali stanno in tre filoni:

1. **Adversarial jailbreaking di LLM** — PAIR [Chao et al., 2023], TAP [Mehrotra et al., 2023]
2. **Benchmark statici di bias T2I/T2T** — CLEAR-Bias, BOLD, StereoSet, BBQ
3. **Audit di T2I models** — Bianchi et al. (Stable Diffusion bias), Bird et al. ("Typecast")

Vedi [07-references.md](07-references.md) per i dettagli bibliografici completi.

## Da dove proseguire

→ [02-architecture.md](02-architecture.md) per il diagramma a blocchi del sistema
→ [03-pair-loop.md](03-pair-loop.md) per la teoria del loop
