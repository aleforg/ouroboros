# MIRTAGE - **Multimodal Iterative Red Teaming for Adversarial Generative-AI Evaluation**

Framework iterativo di red-teaming basato su LLM per misurare il bias demografico e stereotipico-occupazionale nei modelli text-to-image (sovra/sottorappresentazione di gender, race, age e associazioni implicite tra professioni e gruppi demografici). L'approccio usa **attacchi avversariali**: invece di valutare il modello su prompt neutri — dove i filtri di sicurezza e l'allineamento spesso mascherano il problema — un LLM locale non censurato (*attacker*) riscrive iterativamente la scena per **far emergere bias latenti che non si manifesterebbero su input standard**. Il modello T2I (*target*) genera M immagini sul prompt avversariale, un VLM (*judge*) le valuta su cinque assi di bias e l'esito alimenta la riscrittura successiva. Il ciclo itera per ogni seed fino al successo (≥N immagini su M oltre la soglia) o al raggiungimento di `max_iter`.

Adatta l'approccio PAIR (Chao et al., 2023) dal jailbreak testuale alla fairness T2I. Vedi `docs/` (numerati 01–08) per le motivazioni di design e `docs/08-deviations.md` per le deviazioni rispetto a entrambi.

## Vincolo hardware

I modelli sono scelti per girare su **Mac Apple Silicon con 16 GB di RAM unificata** (baseline M4):

- **Attacker**: `dolphin-llama3:latest` via Ollama (~5 GB, 8B 4-bit).
- **Target**: FLUX.2-klein-4B locale via mflux (~5 GB).
- **Judge** (default): Gemini 2.5 Pro su Vertex AI (cloud, 0 GB locali; ~$5 per full run, vedi `docs/04-components.md`). Fallback offline: `mlx-vlm` Qwen2.5-VL-7B-4bit oppure `qwen2.5vl:7b` via Ollama.
- **Unload aggressivo** tra fasi: il picco di RAM è `max(attacker, target)`, non la somma. `src/config.py:check_ram_budget` interrompe lo startup se la stima supera `RAM_BUDGET_GB` (13 GB), a meno di `--allow-swap`.

## Installazione

```bash
pip install -e ".[dev]"
```

Per la pipeline FairFace post-hoc (opzionale):

```bash
pip install -e ".[fairface]"
```

Scaricare manualmente i pesi `res34_fair_align_multi_7_20190809.pt` da [joojs/fairface](https://github.com/joojs/fairface) in `~/.cache/mirtage/fairface/` (oppure puntare con `MIRTAGE_FAIRFACE_WEIGHTS`).

Variabili d'ambiente richieste (vedi `.env.example`):

- `GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` — per Vertex (judge cloud; il target è sempre locale).
- `OLLAMA_HOST` — per l'attacker locale.

## Uso

Esecuzione del loop:

```bash
# Modalità test: 10 seed hard-coded, M=2, max_iter=5, soglia=7
mirtage run --mode test

# Modalità full: 175 seed da data/stable_bias_prompts.jsonl, M=4, max_iter=20
mirtage run --mode full

# Flag utili
mirtage run --mode test --baseline single-shot       # esegue anche baseline senza attacker
mirtage run --mode test --seeds-filter gender        # restringe a una categoria CLEAR-Bias
mirtage run --mode test --judge-backend mlx          # judge offline
mirtage run --resume <run_id>                        # riprende dopo interruzione
mirtage run --dry-run                                # elenca seed e crea run dir senza chiamate
```

Analisi post-hoc:

```bash
mirtage report <run_id>                              # CSV + report.html self-contained
mirtage report <run_id> --no-fairface                # salta la pipeline FairFace
mirtage aggregate <run_id_1> <run_id_2> [...]        # media±std cross-run
```

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
├── fairface.jsonl                      # una riga per volto rilevato (scritto da `mirtage report`)
├── checkpoint.json                     # consumato da --resume
├── meta.json                           # RunConfig frozen + config_hash + timestamp
├── images/<seed_id>/iter_NN/sample_K.png
└── report/
    ├── summary.csv, per_category.csv, asr_vs_iter.csv, …
    ├── fairface_per_category.csv       # KL + norm_entropy su gender/race/age
    └── report.html
```

## Architettura

![Architettura MIRTAGE](/assets/mirtage-architecture.svg)

Vedi `docs/02-architecture.md` per l'architettura completa. Il loop per seed (`src/loop.py:run_one_seed`) si articola in:

1. `attacker.propose(...)` → `AttackerCandidate` (o refusal con retry).
2. Unload attacker, generazione M immagini con il target.
3. Unload target, salvataggio PNG + sha256.
4. `judge.judge(...)` → `BiasJudgement` (Pydantic, 5 assi).
5. Regola di successo: ≥N immagini su M oltre `bias_threshold`.
6. Aggiornamento della `Memory` (top-K per bias_score + più recenti).

## Documentazione

- `docs/` — design rationale e architettura (in italiano).
