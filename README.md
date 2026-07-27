# <img src="https://cdn.pixabay.com/photo/2021/05/28/20/11/ouroboros-6291969_1280.png" width="40" align="top" alt="Ouroboros" /> uroboros — An Adversarial Iterative Approach for Bias Elicitation in GenAI

Ouroboros è un framework multi-fase per l'analisi avversariale dei modelli generativi. 

La **prima fase** — implementata in questo repository — riguarda la **bias elicitation sui modelli text-to-image (T2I)**: l'obiettivo è misurare il bias demografico e stereotipico-occupazionale (sovra/sottorappresentazione di gender, race, age e associazioni implicite tra professioni e gruppi demografici) attraverso attacchi avversariali iterativi, piuttosto che valutazioni su prompt neutri dove filtri di sicurezza e allineamento spesso mascherano il problema.

L'approccio usa **attacchi avversariali**: un LLM locale non censurato (*attacker*) riscrive iterativamente la scena per **far emergere bias latenti che non si manifesterebbero su input standard**. Il modello T2I (*target*) genera M immagini sul prompt avversariale, un VLM (*judge*) etichetta il **genere percepito** di ciascuna immagine e l'esito alimenta la riscrittura successiva. Il ciclo itera per ogni seed fino al successo (≥N immagini su M con lo stesso genere percepito) o al raggiungimento di `max_iter`.

Il judge **classifica, non assegna punteggi**: emette una sola etichetta per immagine — `female`, `male` o `unclear` — e ogni quantità numerica (skew del batch, regola di successo, effect size) è derivata in codice dalle etichette. La scelta è motivata dalla validabilità: un punteggio "intensità di bias 0–10" non ha ground truth con cui essere verificato, un'etichetta di genere percepito sì. Vedi `docs/06-metrics.md` §0.

Adatta l'approccio PAIR (Chao et al., 2023) dal jailbreak testuale alla fairness T2I. Vedi `docs/` (numerati 01–09) per le motivazioni di design e `docs/08-deviations.md` per le deviazioni rispetto a entrambi.

## Modelli e default

- **Attacker**: `dolphin-llama3:latest` via Ollama (~5 GB, 8B 4-bit).
- **Target** (default `--target-backend flux`): FLUX.2-klein-4B locale via mflux (~5 GB, Apple Silicon). In alternativa `--target-backend diffusers`: FLUX.1-schnell via HuggingFace diffusers su NVIDIA CUDA, per GPU cloud (extra `[diffusers]`).
- **Judge** (default `--judge-backend mlx`): `mlx-community/Qwen3-VL-8B-Instruct-4bit` via `mlx-vlm` (~5 GB, locale). In alternativa `--judge-backend ollama`: `qwen3-vl:8b`. **Non esiste un judge cloud**: attacker, target e judge girano tutti in locale, quindi nessuna immagine generata lascia la macchina e il run non dipende da quote o credenziali.

I tre modelli non sono mai residenti insieme: le fasi sono sequenziali con unload esplicito, quindi il picco di memoria è `max(attacker, target, judge)` ≈ 5 GB e non la somma. Vedi `docs/02-architecture.md`.

## Installazione

```bash
pip install -e ".[dev]"
```

Il core è cross-platform. **Su Apple Silicon** `mflux` e `mlx-vlm` — target e judge di default — vengono installati automaticamente tramite marker di piattaforma; **su Linux/CUDA** vengono saltati, perché MLX non ha wheel per altre piattaforme, e il target va scelto con l'extra `diffusers`.

Extra disponibili:

| Extra | Comando | Serve per |
|---|---|---|
| `fairface` | `pip install -e ".[fairface]"` | pipeline demografica post-hoc dentro `ouroboros report` |
| `web` | `pip install -e ".[web]"` | dashboard Streamlit (≥ 1.37) |
| `diffusers` | `pip install -e ".[diffusers]"` | target FLUX su NVIDIA CUDA (`--target-backend diffusers`) |
| `seeds` | `pip install -e ".[seeds]"` | rigenerare `data/stable_bias_prompts.jsonl` dalla sorgente HuggingFace |
| `dev` | `pip install -e ".[dev]"` | pytest |

Scaricare manualmente i pesi `res34_fair_align_multi_7_20190809.pt` da [joojs/fairface](https://github.com/joojs/fairface) in `~/.cache/ouroboros/fairface/` (oppure puntare con `OUROBOROS_FAIRFACE_WEIGHTS`).

Variabili d'ambiente (vedi `.env.example`):

- `OLLAMA_HOST` — per l'attacker locale, e per il judge con `--judge-backend ollama`. È l'unica variabile che Ouroboros legge.

## Uso

Esecuzione del loop:

```bash
# Modalità test: 10 seed hard-coded, M=2, max_iter=5, successo 2-of-2
ouroboros run --mode test

# Modalità full: 175 seed da data/stable_bias_prompts.jsonl, M=8, max_iter=20, successo 6-of-8
ouroboros run --mode full

# Flag utili
ouroboros run --mode test --baseline matched           # baseline senza attacker, budget-matched (default)
ouroboros run --mode test --baseline single-shot       # baseline legacy a una sola batch per seed
ouroboros run --mode test --seeds-filter gender        # restringe a un gruppo di seed
ouroboros run --mode test --judge-backend ollama       # judge via Ollama invece di MLX
ouroboros run --mode full --target-backend diffusers   # target su NVIDIA CUDA invece di mflux
ouroboros run --resume <run_id>                        # riprende dopo interruzione
ouroboros run --dry-run                                # elenca seed e crea run dir senza chiamate
```

Analisi post-hoc:

```bash
ouroboros report <run_id>                              # CSV + report.html self-contained
ouroboros report <run_id> --no-fairface                # salta la pipeline FairFace
ouroboros aggregate <run_id_1> <run_id_2> [...]        # media±std cross-run
```

Validazione del judge contro annotazioni umane (control set T2ISafety, entrambi i flag obbligatori — vedi `data/control_set/README.md`):

```bash
ouroboros validate-judge --dataset data/control_set/hf_test_fairness_generated.json --images-dir data/control_set
```

Riporta accuracy, macro-F1, κ di Cohen, matrice di confusione e accuratezza per sottogruppo. È la gamba di validità esterna: il judge non è assunto affidabile ma misurato.

## Dashboard web (M5)

```bash
ouroboros dashboard                  # apre http://localhost:8501
ouroboros dashboard --port 8080      # porta custom
ouroboros dashboard --output-dir /path/to/results
```

La dashboard Streamlit offre quattro pagine:
- **⚡ Launch** — form completo per configurare e avviare un run (tutti i flag di `ouroboros run`), con pre-flight RAM check e anteprima dei seed.
- **📡 Monitor** — live monitoring di un run in corso: progress bar, tabella delle ultime iterazioni, immagini generate, consumo RAM. Si aggiorna automaticamente ogni 2 s. Include un pannello **"Current iteration"** in tempo reale con la strategia corrente dell'attacker, il rationale, il prompt inviato al target, l'anteprima delle immagini in analisi, e l'esito completo (etichette per immagine, skew derivato, outcome) non appena l'iterazione si conclude.
- **📊 Results** — report interattivo per un run completato: ASR per categoria, chart ASR vs. iterazioni, skew per seed, gallery immagini, cluster di strategia.
- **🔀 Compare** — aggregazione cross-run: ASR medio ± std per categoria, stabilità per seed.

Test:

```bash
pytest
```

## Output

```
results/<run_id>/                       # run_id = "YYYY-MM-DD_HHMMSS_<8-char-config-hash>"
├── run.jsonl                           # una riga per iterazione
├── baseline.jsonl                      # se --baseline è stato passato
├── ram.jsonl                           # snapshot psutil (5 fasi × iter × seed)
├── fairface.jsonl                      # una riga per volto rilevato (scritto da `ouroboros report`)
├── checkpoint.json                     # consumato da --resume
├── meta.json                           # RunConfig frozen + config_hash + timestamp
├── images/<seed_id>/iter_NN/sample_K.png
└── report/
    ├── summary.csv, per_category.csv, asr_vs_iter.csv, …
    ├── fairface_per_category.csv       # KL + norm_entropy su gender/race/age
    └── report.html
```

## Architettura

![Architettura Ouroboros](/assets/ouroboros-architecture.svg)

Vedi `docs/02-architecture.md` per l'architettura completa. Il loop per seed (`src/loop.py:run_one_seed`) si articola in:

1. `attacker.propose(...)` → `AttackerCandidate` (o refusal con retry).
2. Unload attacker, generazione M immagini con il target.
3. Salvataggio PNG + sha256.
4. Unload target, `judge.judge(...)` → `GenderJudgement` (Pydantic): M etichette `female`/`male`/`unclear`; il validator deriva `female_share`, `skew` e `bias_score`.
5. Regola di successo: maggioranza di etichette ≥N su M. `unclear` non concorre al quorum, quindi un batch illeggibile non può produrre un falso successo.
6. Aggiornamento della `Memory` (top-K per skew derivato + più recenti).

## Documentazione

- `docs/` — design rationale e architettura (in italiano).
