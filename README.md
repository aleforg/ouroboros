# <img src="https://cdn.pixabay.com/photo/2021/05/28/20/11/ouroboros-6291969_1280.png" width="40" align="top" alt="Ouroboros" /> uroboros — An Adversarial Iterative Approach for Bias Elicitation in GenAI

Ouroboros è un framework multi-fase per l'analisi avversariale dei modelli generativi. 

La **prima fase** — implementata in questo repository — riguarda la **bias elicitation sui modelli text-to-image (T2I)**: l'obiettivo è misurare il bias demografico e stereotipico-occupazionale (sovra/sottorappresentazione di gender, race, age e associazioni implicite tra professioni e gruppi demografici) attraverso attacchi avversariali iterativi, piuttosto che valutazioni su prompt neutri dove filtri di sicurezza e allineamento spesso mascherano il problema.

L'approccio usa **attacchi avversariali**: un LLM locale non censurato (*attacker*) riscrive iterativamente la scena per **far emergere bias latenti che non si manifesterebbero su input standard**. Il modello T2I (*target*) genera M immagini sul prompt avversariale, un VLM (*judge*) le valuta su cinque assi di bias e l'esito alimenta la riscrittura successiva. Il ciclo itera per ogni seed fino al successo (≥N immagini su M oltre la soglia) o al raggiungimento di `max_iter`.

Adatta l'approccio PAIR (Chao et al., 2023) dal jailbreak testuale alla fairness T2I. Vedi `docs/` (numerati 01–08) per le motivazioni di design e `docs/08-deviations.md` per le deviazioni rispetto a entrambi.

## Modelli e default

- **Attacker**: `dolphin-llama3:latest` via Ollama (~5 GB, 8B 4-bit).
- **Target**: FLUX.2-klein-4B locale via mflux (~5 GB).
- **Judge** (default): Gemini 2.5 Pro su Vertex AI (cloud, 0 GB locali; ~$5 per full run, vedi `docs/04-components.md`). Fallback offline: `mlx-vlm` Qwen2.5-VL-7B-4bit oppure `qwen2.5vl:7b` via Ollama.

## Installazione

```bash
pip install -e ".[dev]"
```

Per la pipeline FairFace post-hoc (opzionale):

```bash
pip install -e ".[fairface]"
```

Per la dashboard web (opzionale, richiede Streamlit ≥ 1.37):

```bash
pip install -e ".[web]"
```

Scaricare manualmente i pesi `res34_fair_align_multi_7_20190809.pt` da [joojs/fairface](https://github.com/joojs/fairface) in `~/.cache/ouroboros/fairface/` (oppure puntare con `OUROBOROS_FAIRFACE_WEIGHTS`).

Variabili d'ambiente richieste (vedi `.env.example`):

- `GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` — per Vertex (judge cloud; il target è sempre locale).
- `OLLAMA_HOST` — per l'attacker locale.

## Uso

Esecuzione del loop:

```bash
# Modalità test: 10 seed hard-coded, M=2, max_iter=5, soglia=7
ouroboros run --mode test

# Modalità full: 175 seed da data/stable_bias_prompts.jsonl, M=4, max_iter=20
ouroboros run --mode full

# Flag utili
ouroboros run --mode test --baseline single-shot       # esegue anche baseline senza attacker
ouroboros run --mode test --seeds-filter gender        # restringe a una categoria CLEAR-Bias
ouroboros run --mode test --judge-backend mlx          # judge offline
ouroboros run --resume <run_id>                        # riprende dopo interruzione
ouroboros run --dry-run                                # elenca seed e crea run dir senza chiamate
```

Analisi post-hoc:

```bash
ouroboros report <run_id>                              # CSV + report.html self-contained
ouroboros report <run_id> --no-fairface                # salta la pipeline FairFace
ouroboros aggregate <run_id_1> <run_id_2> [...]        # media±std cross-run
```

## Dashboard web (M5)

```bash
ouroboros dashboard                  # apre http://localhost:8501
ouroboros dashboard --port 8080      # porta custom
ouroboros dashboard --output-dir /path/to/results
```

La dashboard Streamlit offre quattro pagine:
- **⚡ Launch** — form completo per configurare e avviare un run (tutti i flag di `ouroboros run`), con pre-flight RAM check e anteprima dei seed.
- **📡 Monitor** — live monitoring di un run in corso: progress bar, tabella delle ultime iterazioni, immagini generate, consumo RAM. Si aggiorna automaticamente ogni 2 s. Include un pannello **"Current iteration"** in tempo reale con la strategia corrente dell'attacker, il rationale, il prompt inviato al target, l'anteprima delle immagini in fase di analisi da parte del judge, e il giudizio completo (bias score, assi, stereotype_framing) non appena l'iterazione si conclude.
- **📊 Results** — report interattivo per un run completato: ASR per categoria, chart ASR vs. iterazioni, score per asse del judge, gallery immagini, cluster di strategia.
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
3. Unload target, salvataggio PNG + sha256.
4. `judge.judge(...)` → `BiasJudgement` (Pydantic, 5 assi).
5. Regola di successo: ≥N immagini su M oltre `bias_threshold`.
6. Aggiornamento della `Memory` (top-K per bias_score + più recenti).

## Documentazione

- `docs/` — design rationale e architettura (in italiano).
