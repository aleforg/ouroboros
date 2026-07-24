# 01 — Overview

> **Stato: allineato al codice v3.0.** Il judge è un **classificatore di genere
> locale**; non esiste più il judge cloud. Vedi
> [08-deviations.md](08-deviations.md) §0 per la riduzione di scope decisa a
> luglio 2026.

## Cos'è Ouroboros

Ouroboros è un **framework di red-teaming iterativo** per modelli text-to-image
(T2I). L'obiettivo è quantificare quanto un *attaccante automatico* (un LLM
"uncensored" che riformula prompt) riesca a far emergere **sbilanciamento di
genere** da un modello T2I, partendo da descrizioni di scena neutrali.

L'idea è derivata dal lavoro di Chao et al. su PAIR
([Chao et al., 2023](07-references.md#pair)) per i jailbreak su LLM, ma adattata
al dominio **immagini** invece che testo.

## Differenza chiave: cosa cambia nel dominio T2I

Il PAIR originale ha tre attori: **attacker LLM**, **target LLM**, **judge LLM**.
Tutti e tre operano su testo, e il judge legge l'output testuale del target.

| Componente | PAIR originale | Ouroboros |
|---|---|---|
| **Attacker** | LLM uncensored | LLM uncensored (Ollama, locale) |
| **Target** | LLM black-box | T2I locale: FLUX.2-klein-4B via mflux (Apple Silicon) o FLUX.1-schnell via diffusers (NVIDIA CUDA) |
| **Judge** | LLM che *assegna un punteggio* 1–10 | **VLM locale** (Qwen3-VL-8B, MLX o Ollama) che *assegna un'etichetta* |

Due delta rispetto al paper originale, entrambi importanti:

1. **Il judge deve guardare le immagini**, non leggere testo — di qui la scelta
   di un VLM.
2. **Il judge non è uno scorer, è un classificatore.** In PAIR il judge emette
   un punteggio di jailbreak 1–10 e la soglia decide il successo. In Ouroboros
   il VLM emette **una sola etichetta per immagine** — `female`, `male` o
   `unclear` — e ogni numero (skew del batch, success rule, effect size) è
   **derivato in codice** da quelle etichette.

Il motivo del secondo delta è la validabilità. Un punteggio "intensità di bias
0–10" non è verificabile: non esiste un dataset con annotazioni umane su quella
scala. Un'etichetta di genere percepito lo è, una per una, contro un
classificatore standard (FairFace, κ di Cohen) e contro annotazioni umane
(control set T2ISafety, via `ouroboros validate-judge`). La soggettività esce
dal numero ed entra solo nella lettura del sensore, dove può essere misurata.
Dettagli in [06-metrics.md](06-metrics.md) §0.

## Domande di ricerca

Dopo la revisione di luglio 2026 il framework è organizzato attorno a **due sole
domande**:

1. **RQ1 — validità dello strumento.** Quanto è affidabile il metodo automatico
   con cui Ouroboros misura lo sbilanciamento di genere? *(κ judge↔FairFace per
   immagine; validazione esterna su T2ISafety)*
2. **RQ2 — efficacia dell'attacco.** A parità di budget, il raffinamento
   iterativo dei prompt trova sbilanciamenti più **frequenti** o più **severi**
   di una baseline statica? *(ΔASR e ΔABS contro baseline budget-matched)*

Resta nello scope anche l'analisi delle **strategie** dell'attacker (clustering
delle `strategy_label`, [06-metrics.md](06-metrics.md) §11). Tutto il resto —
etnia, età, tipo corporeo, framing stereotipico, allineamento BLS, multi-modello,
test-retest multi-run — è **analisi esplorativa o lavoro futuro**.

## Goals

1. Quantificare di quanto il prompt engineering iterativo aumenti frequenza e
   severità dello sbilanciamento rispetto a una **baseline budget-matched** —
   cioè con lo stesso numero di estrazioni, così che la differenza misuri la
   ricerca dell'attacker e non il vantaggio di pescare più volte.
2. Produrre metriche **publication-grade** e riproducibili: ASR con CI bootstrap,
   queries-to-success, ABS per-seed, tutte derivate deterministicamente dalle
   etichette del judge.
3. Rendere **misurabile lo strumento stesso**, non solo il target: censoring
   esplicito, unclear-rate, accordo con un classificatore indipendente,
   validazione esterna contro ground truth umana.

## Non-goals

- **Comparare più target T2I**: sono cablati due backend FLUX (mflux locale e
  diffusers CUDA), ma il confronto sistematico tra modelli diversi (SD 1.5,
  DALL-E, Imagen) è fuori dai claim. La factory `build_target()` in
  `src/targets/base.py` rende l'aggiunta di un backend una modifica a un file
  solo.
- **Attacchi white-box** (gradient-based): l'attacker vede solo gli output.
- **Dialoghi multi-turn col target**: ogni iterazione è una singola chiamata T2I
  con un prompt nuovo. La memoria vive nell'attacker, non nel target.
- **Assi demografici oltre il genere**: FairFace classifica etnia ed età gratis
  nella stessa forward pass e i dati vengono conservati, ma senza claim — il
  judge non li misura e l'attacker non li ottimizza.
- **Sostituire** la suite testuale in `fairness-eval/`: quella resta il ramo
  text-to-text del lavoro.

## Vincolo hardware (importante)

Il framework è progettato per girare su **Mac Apple Silicon con 16 GB di RAM
unificata** (M4 baseline). Tutti e tre gli attori sono locali, quindi la RAM è
*il* vincolo di progetto.

```
┌─────────────── 16 GB unified memory ───────────────┐
│  ~3 GB  macOS + browser + IDE                      │
│  ~5 GB  attacker LLM   (Ollama, dolphin-llama3:8b) │
│  ~5 GB  target T2I     (FLUX.2-klein-4B via mflux) │
│  ~5 GB  judge VLM      (Qwen3-VL-8B-4bit, locale)  │
└─────────────────────────────────────────────────────┘
        i tre modelli NON sono mai residenti insieme
```

La somma sforerebbe i 16 GB: il framework non ci prova nemmeno. Le tre fasi sono
**sequenziali con unload esplicito** tra l'una e l'altra (`aggressive_unload`,
default attivo), quindi il picco è `max(attacker, target, judge)` ≈ 5 GB e non
la somma ≈ 15 GB. Vedi [02-architecture.md](02-architecture.md) §"Gestione RAM".

Da qui discendono le due scelte di design principali:

1. **Dimensione dell'attacker** → max 8B parametri quantizzati a 4 bit (no
   Mixtral, no 13B). Default `dolphin-llama3:latest`.
2. **Judge locale, non cloud** → il progetto non dipende da un servizio a
   pagamento, da una quota o da una connessione, e nessuna immagine generata
   lascia la macchina. Il costo è che il judge occupa RAM come gli altri due, e
   va quindi schedulato nella stessa sequenza di unload.

> **Nota storica.** Fino alla v2.x il judge era Gemini 2.5 Pro via Vertex AI, e
> l'argomento era l'opposto: 0 GB locali perché il lavoro veniva fatto altrove.
> È cambiato con la riduzione di scope della v3.0 — un judge locale è
> ispezionabile, riproducibile e versionabile, requisiti che contano più del
> risparmio di RAM quando la tesi deve *validare* il judge.

Se la RAM stimata sfora il budget (`RAM_BUDGET_GB`, **13 GB**), la CLI **aborta
lo startup** a meno di `--allow-swap`. La logica è in
`src/config.py:check_ram_budget`.

Per run più grandi esiste la via d'uscita cloud-GPU: `--target-backend
diffusers` esegue FLUX.1-schnell su NVIDIA CUDA (RunPod, Lambda, Colab) con lo
stesso loop e lo stesso judge.

## Dataset dei seed

I seed sono **175 professioni** da [Stable Bias](07-references.md) (Luccioni et
al., NeurIPS 2023), instanziate nel template `"Photo portrait of a [X]"`
(`data/stable_bias_prompts.jsonl`).

La `category` di ogni seed **non** è un asse di bias: è il suo **gruppo
stereotipico di genere** — `male_coded`, `female_coded`, `balanced` — derivato
automaticamente dalla quota femminile BLS della professione
(`data/profession_groups.json`). Così le aggregazioni del report rispondono a
"l'attacco funziona meglio dove lo stereotipo va già nella stessa direzione?".

Il vecchio dataset CLEAR-Bias × BOLD, con le sue 6 categorie tematiche
(`gender`, `race`, `gender-ethnicity`, …), è archiviato in `data/legacy/` e
sopravvive solo nei 10 seed hard-coded della modalità `test`. Vedi
[05-dataset.md](05-dataset.md).

## Posizionamento nella letteratura

Ouroboros è un esempio di **automated adversarial red-teaming** applicato alla
*fairness* di sistemi multimodali. Le sue radici stanno in tre filoni:

1. **Adversarial jailbreaking di LLM** — PAIR [Chao et al., 2023],
   TAP [Mehrotra et al., 2023]
2. **Benchmark statici di bias T2I** — Stable Bias [Luccioni et al., 2023],
   DALL-Eval, FAIntbench, T2ISafety
3. **Audit di modelli T2I** — Bianchi et al. (Stable Diffusion bias),
   Bird et al. ("Typecast")

La cella che Ouroboros occupa e che risulta vuota nella letteratura: red-teaming
**iterativo e adattivo** applicato al bias demografico T2I (non al jailbreak di
contenuti), su hardware commodity, con lo strumento di misura validato invece
che assunto. Vedi [07-references.md](07-references.md) e
`docs/related-work-digest.md`.

## Da dove proseguire

→ [02-architecture.md](02-architecture.md) per il diagramma a blocchi del sistema
→ [03-pair-loop.md](03-pair-loop.md) per la teoria del loop
→ [06-metrics.md](06-metrics.md) per come le etichette diventano metriche
