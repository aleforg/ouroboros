# 07 — References

Bibliografia ragionata delle fonti accademiche e tecniche dietro al framework. Suddiviso per area concettuale.

> **Stato: allineato al codice v3.0.** Il judge è **locale** (Qwen3-VL-8B via
> MLX o Ollama); il judge cloud Gemini/Vertex non esiste più. Il seed-set è
> **Stable Bias** (175 professioni); CLEAR-Bias × BOLD è archiviato in
> `data/legacy/` e resta qui per la genealogia del progetto. Vedi
> [08-deviations.md](08-deviations.md) §0.

---

## Metodologia di attacco

### PAIR

<a id="pair"></a>
**Chao, P., Robey, A., Dobriban, E., Hassani, H., Pappas, G. J., & Wong, E. (2023).**
*Jailbreaking Black Box Large Language Models in Twenty Queries.*
arXiv:2310.08419.
🔗 https://arxiv.org/abs/2310.08419

Il paper originale del **PAIR loop**. Definisce il tre-attori framework attacker-target-judge e dimostra che jailbreak black-box su LLM sono raggiungibili in `O(20)` query con un attacker LLM ben istruito. Ouroboros porta lo schema dal dominio LLM→LLM al dominio LLM→T2I.

**Cosa abbiamo preso:** struttura del loop attacker-target-judge, memoria top-K, retry su attacker self-refusal, il budget di ~O(20) query per seed.
**Cosa abbiamo cambiato:** target multimodale invece che testuale; M generazioni per iterazione invece di una; success rule **N-of-M** invece della soglia su un singolo output; judge VLM invece che LLM; e soprattutto **il judge non assegna un punteggio**. In PAIR il judge emette un jailbreak score 1–10 e una soglia decide il successo; in Ouroboros emette un'etichetta di genere per immagine e tutti i numeri sono derivati in codice — scelta motivata dalla validabilità (vedi [06-metrics.md](06-metrics.md) §0 e [08-deviations.md](08-deviations.md) §0.1). Cambia anche l'obiettivo: non l'elicitazione di contenuto proibito, ma lo **sbilanciamento distribuzionale** di un batch.

### TAP

**Mehrotra, A., Zampetakis, M., Kassianik, P., Nelson, B., Anderson, H., Singer, Y., & Karbasi, A. (2023).**
*Tree of Attacks: Jailbreaking Black-Box LLMs Automatically.*
arXiv:2312.02119.
🔗 https://arxiv.org/abs/2312.02119

Estensione tree-search di PAIR con pruning. Per ogni iter genera b candidati invece di 1, un evaluator pota i meno promettenti, scende solo i rami sopravvissuti. **Ouroboros è esplicitamente PAIR (no branching), non TAP** — la motivazione è il costo T2I, ogni branch aggiuntivo costa M chiamate al target (vedi [03-pair-loop.md](03-pair-loop.md#differenze-rispetto-a-tap-mehrotra-et-al-2023)).

### GCG (per contesto storico)

**Zou, A., Wang, Z., Carlini, N., Nasr, M., Kolter, J. Z., & Fredrikson, M. (2023).**
*Universal and Transferable Adversarial Attacks on Aligned Language Models.*
arXiv:2307.15043.
🔗 https://arxiv.org/abs/2307.15043

Greedy Coordinate Gradient — attacco **white-box** che ottimizza suffissi adversariali. Precursore di PAIR; mostra che il jailbreak è possibile in principio, ma richiede accesso ai gradienti (e i suffissi sono testo robaccia, non leggibili). PAIR è la risposta "black-box, semantica" a GCG.

---

## Dataset di bias

> Le due voci seguenti (CLEAR-Bias, BOLD) descrivono il **seed-set storico**,
> dismesso in v2.5 e archiviato in `data/legacy/`. Il seed-set attuale è
> Stable Bias — vedi la voce omonima sotto "Audit di T2I models" e
> [05-dataset.md](05-dataset.md). Restano qui perché spiegano da dove viene la
> struttura per-categoria del framework.

### CLEAR-Bias

<a id="clear-bias"></a>
**Cantini, R., Orsino, A., Ruggiero, M., & Talia, D. (2025).**
*Benchmarking Adversarial Robustness to Bias Elicitation in Large Language Models: Scalable Automated Assessment with LLM-as-a-Judge.* (dataset: CLEAR-Bias)
arXiv:2504.07887.
🔗 arXiv: https://arxiv.org/abs/2504.07887
🔗 Hugging Face: https://huggingface.co/datasets/RCantini/CLEAR-Bias

Il benchmark di prompt **testuali** da cui derivavano le `base_scene` fino alla v2.4. Il dataset CLEAR-Bias copre **10 categorie** demografiche: 7 isolate (age, disability, ethnicity, gender, religion, sexual orientation, socioeconomic) + 3 intersezionali (gender-ethnicity, gender-sexual orientation, ethnicity-socioeconomic), con prompt in formato multiple-choice o sentence-completion.

**Cosa abbiamo preso:** 6 delle 10 categorie (gender, ethnicity, religion, socioeconomic, gender-ethnicity, ethnicity-socioeconomic); i `source_clear_bias_prompt` come `source_text` dei seed.
**Cosa abbiamo escluso:** le categorie `age`, `disability`, `sexual_orientation` (e l'intersezione gender-sexual_orientation) — presenti nel dataset, ma non si traducono bene in scene visualizzabili.

### BOLD

**Dhamala, J., Sun, T., Kumar, V., Krishna, S., Pruksachatkun, Y., Chang, K.-W., & Gupta, R. (2021).**
*BOLD: Dataset and Metrics for Measuring Biases in Open-Ended Language Generation.*
Proceedings of FAccT 2021. arXiv:2101.11718.
🔗 https://arxiv.org/abs/2101.11718

Dataset di prompt seed estratti da Wikipedia su 5 demographic axes (gender, race, profession, religion, ideologia politica). Usato per **arricchire visivamente** i prompt CLEAR-Bias con scene anchor + visual attributes (vedi [05-dataset.md](05-dataset.md#bold-dhamala-et-al-2021)).

**Cosa abbiamo preso:** scene anchor e visual attributes per i 120 prompt del vecchio full mode CLEAR-Bias × BOLD (dismesso in v2.5, vedi [05-dataset.md](05-dataset.md#dataset-legacy)).
**Cosa NON abbiamo preso:** le metriche di bias proposte dal paper (Sentiment, Regard, Toxicity) — operano su testo, non su immagini.

### StereoSet (referenza ulteriore)

**Nadeem, M., Bethke, A., & Reddy, S. (2021).**
*StereoSet: Measuring Stereotypical Bias in Pretrained Language Models.*
ACL 2021. arXiv:2004.09456.

Benchmark contestuale per stereotipi in LM. Non usato direttamente in Ouroboros, ma è un riferimento cardinale del campo "bias measurement in LM".

---

## Audit di T2I models

### Bianchi et al.

**Bianchi, F., Kalluri, P., Durmus, E., Ladhak, F., Cheng, M., Nozza, D., Hashimoto, T., Jurafsky, D., Zou, J., & Caliskan, A. (2023).**
*Easily Accessible Text-to-Image Generation Amplifies Demographic Stereotypes at Large Scale.*
Proceedings of FAccT 2023. arXiv:2211.03759.
🔗 https://arxiv.org/abs/2211.03759

Studio empirico su Stable Diffusion: mostra che T2I **amplifica** stereotipi rispetto al training data e rispetto alla popolazione reale (es. "CEO" → predominanza maschile bianca al 90%+ anche quando la stat reale è ~70%). Stabilisce il problem statement per cui Ouroboros esiste.

### Cho et al. (DALL-E Eval)

**Cho, J., Zala, A., & Bansal, M. (2023).**
*DALL-Eval: Probing the Reasoning Skills and Social Biases of Text-to-Image Generation Models.*
ICCV 2023. arXiv:2202.04053.
🔗 https://arxiv.org/abs/2202.04053

Framework di valutazione statica per T2I che include test di bias razziale/gender su DALL-E. Approccio "static prompt set", complementare al nostro approccio "adversarial iterative". **Riferimento metodologico** per la metrica `distribution_bias` (KL da uniforme su FairFace), che abbiamo adottato come metrica per-asse principale nel `report` da v2.2.

### FairFace

**Karkkainen, K., & Joo, J. (2021).**
*FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age for Bias Measurement and Mitigation.*
WACV 2021.
🔗 https://openaccess.thecvf.com/content/WACV2021/papers/Karkkainen_FairFace_Face_Attribute_Dataset_for_Balanced_Race_Gender_and_Age_WACV_2021_paper.pdf
🔗 Repo + pesi: https://github.com/joojs/fairface

Dataset + classificatore ResNet-34 standard per audit demografici (7 razze × 2 generi × 9 fasce d'età). Usato in Ouroboros per la pipeline post-hoc in `src/fairface.py` (MTCNN per la detection + ResNet-34 per la classificazione).

**Non sostituisce il judge** — è il judge a guidare la success rule del loop, in tempo reale; FairFace gira dopo, sulle immagini già salvate. I suoi due ruoli sono:
1. **Secondo osservatore indipendente** per la validità convergente: κ di Cohen per immagine contro le etichette del judge, che è la statistica primaria di RQ1.
2. **Base per metriche comparabili** con la letteratura T2I-fairness (KL da uniforme, entropia normalizzata).

Limite da dichiarare sempre: FairFace è addestrato su volti **fotografici reali**, quindi degrada sugli output T2I stilizzati — e degrada probabilmente *nello stesso modo* del judge VLM, il che rende l'accordo tra i due una prova più debole di quanto sembri (vedi [06-metrics.md](06-metrics.md) §7, caveat).

### FAIntbench / BIGbench

**Luo, H., Deng, Z., Chen, R., & Liu, Z. (2024).**
*FAIntbench: A Holistic and Precise Benchmark for Bias Evaluation in Text-to-Image Models.*
arXiv:2405.17814.
🔗 https://arxiv.org/abs/2405.17814

**Luo, H., Huang, H., Deng, Z., Li, X., et al. (2024).**
*BIGbench: A Unified Benchmark for Evaluating Multi-dimensional Social Biases in Text-to-Image Models.*
arXiv:2407.15240.
🔗 https://arxiv.org/abs/2407.15240

Definiscono il framework di metriche T2I-bias contemporaneo: separano *implicit vs explicit bias*, introducono il **Manifestation factor η** che decompone uno score di bias in *ignorance* (il modello evita la demografia) vs *discrimination* (il modello assegna attivamente lo stereotipo). KL e norm_entropy usate in Ouroboros sono allineate a questo filone.

### T2ISafety

**Li, L., et al. (2025).**
*T2ISafety: Benchmark for Assessing Fairness, Toxicity, and Privacy in Image Generation.*
CVPR 2025.
🔗 https://openaccess.thecvf.com/content/CVPR2025/papers/Li_T2ISafety_Benchmark_for_Assessing_Fairness_Toxicity_and_Privacy_in_Image_Generation_CVPR_2025_paper.pdf

Benchmark recente che combina skew per-asse via FairFace + classificatori di tossicità. Riferimento più vicino al setting di Ouroboros (skew demografico + bias toxicity), salvo che è statico e cloud-based.

**Ruolo operativo**: il suo split *fairness* è il **control set con annotazioni umane** contro cui viene validato il judge — `ouroboros validate-judge --dataset hf_test_fairness_generated.json --images-dir <test_zip_root>`, che riporta accuracy, macro-F1, κ di Cohen, P/R/F1 per classe, matrice di confusione, tasso di predizioni non valide e accuratezza per sottogruppo (`src/validate.py`). È la gamba di **validità esterna** di RQ1, complementare all'accordo interno con FairFace.

**Caveat importante**: T2ISafety **non pubblica l'inter-annotator agreement** per la parte fairness. Non sappiamo quindi quanto siano concordi tra loro gli annotatori umani su questo compito, e questo pone un tetto implicito a quanto la validazione esterna possa dimostrare — un κ del judge non può essere letto contro un massimo teorico ignoto.

### Stable Bias

**Luccioni, A. S., Akiki, C., Mitchell, M., & Jernite, Y. (2023).**
*Stable Bias: Analyzing Societal Representations in Diffusion Models.*
NeurIPS 2023 D&B. arXiv:2303.11408.
🔗 https://arxiv.org/abs/2303.11408

Audit Stable Diffusion / DALL-E 2 con BLIP-VQA marker rate + cluster entropy + **95% bootstrap CI** (uno dei pochi paper T2I-bias a riportare CI).

Doppio ruolo in Ouroboros:
1. **Fonte del seed-set attuale** — le 175 professioni di `data/stable_bias_prompts.jsonl`, nel template `"Photo portrait of a [X]"`, vengono da qui.
2. **Precedente metodologico per i CI** — la migrazione da Wilson a bootstrap sull'ASR (fatta in v2.3, vedi [06-metrics.md](06-metrics.md) §1) segue questo paper. È l'eccezione positiva del campo su *entrambe* le fragilità che [08-deviations.md](08-deviations.md) segnala: approccio classifier-free **e** intervalli di confidenza. I benchmark 2024–25 hanno regredito su entrambi i fronti; Ouroboros non inventa lo standard, lo ripristina nel setting adversariale.

### Girrbach et al. (audit su FLUX)

<a id="girrbach"></a>
**Girrbach, L., et al. (2025).**
*Large Scale Analysis of Gender Biases in Text-to-Image Generative Models.*
arXiv:2503.23398.
🔗 https://arxiv.org/abs/2503.23398

3.217 prompt gender-neutral, ~2,3M immagini (2.293.295 dopo filtering), su FLUX, FLUX-Schnell, SD 3.5 Large/Medium, SD 3 Medium; auto-labeling YOLOv10 + InternVL2-8B, metrica Female Ratio ℛf.

Rilevante per due motivi indipendenti:
1. **È l'unico audit large-scale pubblicato che copre FLUX**, cioè proprio la famiglia di modelli che Ouroboros attacca — quindi è il riferimento di sanity esterno per i numeri della baseline.
2. **È il precedente per il task del judge v3.0**: classificare il genere percepito con etichette `female` / `male` / `unclear`, invece di assegnare un punteggio di bias. Non abbiamo inventato la formulazione del compito.

Da segnalare come limite (e quindi come spazio per Ouroboros): le error bar delle sue figure sono deviazioni standard cross-prompt — variabilità semantica — non intervalli di confidenza sulla stima di ℛf, e non c'è test di significatività sui confronti tra modelli.

---

## Modelli usati

### FLUX (target)

**Black Forest Labs (2024).**
FLUX.2 klein — distilled 4B-parameter rectified-flow T2I model.
🔗 https://github.com/black-forest-labs/flux
🔗 mflux (Apple Silicon port): https://github.com/filipstrand/mflux
🔗 diffusers: https://github.com/huggingface/diffusers

Modello T2I locale. Due backend cablati, selezionabili con `--target-backend`:

| Backend | Modello | Piattaforma | Modulo |
|---|---|---|---|
| `flux` (default) | FLUX.2-klein-4B via [`mflux`](https://github.com/filipstrand/mflux) | Apple Silicon (MLX/Metal) | `src/targets/flux.py` |
| `diffusers` | FLUX.1-schnell via HuggingFace diffusers | NVIDIA CUDA (RunPod, Lambda, Colab) | `src/targets/diffusers_flux.py` |

Quantizzato a 4 bit con 4 inference step (variante distilled) per stare nel budget RAM. Su mflux la generazione è **sequenziale sul thread asyncio** — MLX lega lo stream GPU al thread che ha creato il modello, quindi spostarla in un thread pool la rompe. Vedi [04-components.md](04-components.md) per la motivazione della scelta (RAM, nessun safety filter, nessun costo cloud) e la voce Girrbach sopra per il fatto che FLUX è l'unico target di Ouroboros con un audit di bias pubblicato con cui confrontarsi.

### Qwen3-VL (judge)

<a id="qwen-vl"></a>
**Bai, S., Chen, K., Liu, X., Wang, J., et al. (2025).**
*Qwen3-VL Technical Report.*
arXiv:2511.21631.
🔗 https://arxiv.org/abs/2511.21631

Vision-Language Model (2B/4B/8B + MoE variants) di Alibaba/Qwen. È **il judge** di Ouroboros — non un fallback: non esiste un judge cloud. Due backend, stesso modello:

| Backend | Costante in `src/config.py` | Modello |
|---|---|---|
| `--judge-backend mlx` (default) | `JUDGE_MLX_DEFAULT` | `mlx-community/Qwen3-VL-8B-Instruct-4bit` |
| `--judge-backend ollama` | `JUDGE_OLLAMA_DEFAULT` | `qwen3-vl:8b` |

Riceve le M immagini di un batch e restituisce **solo** `{"per_image_genders": [...], "rationale": "..."}`, con ogni etichetta in `{female, male, unclear}`; tutti i valori numerici sono derivati in codice dal validator di `GenderJudgement`. Generazione 8B a 4 bit ≈ 5 GB, quindi il judge partecipa al ciclo di unload aggressivo come attacker e target.

Scelto sulla generazione Qwen2.5-VL per capacità e recenza; la sua affidabilità su questo compito **non è assunta ma misurata** — κ contro FairFace e accuracy/macro-F1 contro T2ISafety (RQ1, vedi [06-metrics.md](06-metrics.md) §7).

🔗 HF MLX: https://huggingface.co/mlx-community/Qwen3-VL-8B-Instruct-4bit
🔗 Ollama: `qwen3-vl:8b`

> **Nota storica — il judge cloud Gemini 2.5 Pro.** Fino alla v2.x il judge
> default era Gemini 2.5 Pro via Vertex AI (`--judge-backend gemini`,
> `JUDGE_GEMINI_DEFAULT`), scelto perché costava 0 GB di RAM locale (~$0.006 a
> chiamata, ~$5 per run full). È stato **rimosso** con la v3.0: `judge_backend`
> è `Literal["mlx", "ollama"]` e non esiste più codice che parli con Vertex.
> Le motivazioni sono in [08-deviations.md](08-deviations.md) §0.6 — in breve,
> quando la tesi deve *validare* il judge, un modello ispezionabile, versionabile
> e riproducibile offline vale più della RAM risparmiata. Le variabili
> `GOOGLE_CLOUD_*` sopravvivono in `RunConfig` come campi vestigiali, ignorati
> da `build_judge()`.

### Dolphin-LLaMA3 (attacker)

<a id="dolphin"></a>
**Hartford, E. (2024).**
Dolphin-2.9-LLaMA3-8B (uncensored fine-tune of LLaMA-3 8B).
🔗 https://huggingface.co/cognitivecomputations/dolphin-2.9-llama3-8b

Fine-tune di LLaMA-3 con denoising dei rifuti di safety. Usato come **attacker** perché il PAIR loop richiede un LLM disposto a generare prompt adversariali per scopi di ricerca. La famiglia "Dolphin" di Eric Hartford è lo standard de facto per uncensored fine-tunes.

### LLaMA-3 (base dell'attacker)

**Meta AI (2024).**
*The Llama 3 Herd of Models.*
arXiv:2407.21783.
🔗 https://arxiv.org/abs/2407.21783

Base model. Rilevante per capire la baseline capability dell'attacker.

---

## Inference runtime

### MLX

**Apple Machine Learning Research (2023).**
*MLX: An array framework for Apple Silicon.*
🔗 GitHub: https://github.com/ml-explore/mlx
🔗 Docs: https://ml-explore.github.io/mlx/

Framework di array Apple ottimizzato per Apple Silicon (unified memory, accelerazione Metal). Ouroboros lo usa per il target FLUX (via mflux) e per il judge fallback offline (Qwen3-VL via mlx-vlm è più veloce di Ollama su M-series).

### mlx-vlm

**Blaizzy/mlx-vlm (2024).**
*MLX-VLM: Vision-Language Model inference on Apple Silicon.*
🔗 https://github.com/Blaizzy/mlx-vlm

Wrapper specifico per VLM su MLX. Espone `mlx_vlm.load()` e `mlx_vlm.generate()` usati dal fallback offline `MLXJudge` in `src/judge.py`.

### Ollama

**Ollama project.**
🔗 https://ollama.com/ — https://github.com/ollama/ollama

Runtime locale per LLM (LLaMA, Mistral, ecc.) con API HTTP-like. Ouroboros lo usa per l'attacker (default) e come fallback per il judge (`--judge-backend ollama`).

---

## Tecnologie del reporting

### HDBSCAN

**Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013).**
*Density-Based Clustering Based on Hierarchical Density Estimates.*
PAKDD 2013, pp. 160-172.
🔗 https://link.springer.com/chapter/10.1007/978-3-642-37456-2_14

Algoritmo di clustering density-based, hierarchical. Vantaggi vs k-means: nessun numero di cluster da specificare a priori, gestisce cluster di forma non sferica, separa il rumore (label `-1`). Ouroboros lo usa in `src/cluster.py:40` con `min_cluster_size=3` per raggruppare le strategy_label freeform dell'attacker.

🔗 Implementazione Python: https://github.com/scikit-learn-contrib/hdbscan

### Sentence-BERT (all-MiniLM-L6-v2)

**Reimers, N., & Gurevych, I. (2019).**
*Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.*
EMNLP 2019. arXiv:1908.10084.
🔗 https://arxiv.org/abs/1908.10084

Architettura per produrre embedding di frasi semanticamente sensati (vs avg di token embeddings BERT vanilla). Ouroboros usa il modello `all-MiniLM-L6-v2` (~80 MB, 384 dim) per embedded delle `strategy_label` prima del clustering HDBSCAN.

🔗 Modello: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

---

## Letteratura collegata (contesto)

### Bird et al. — Tipologia dei rischi T2I

**Bird, C., Ungless, E. L., & Kasirzadeh, A. (2023).**
*Typology of Risks of Generative Text-to-Image Models.*
AIES 2023. arXiv:2307.05543.
🔗 https://arxiv.org/abs/2307.05543

Tassonomia (literature review, non studio empirico) dei rischi dei modelli T2I generativi, organizzati per categorie di stakeholder; include la riproduzione di stereotipi demografici. Inquadra il problema di bias che Ouroboros misura empiricamente.

### BBQ (per testing di bias intersezionale)

**Parrish, A., Chen, A., Nangia, N., Padmakumar, V., Phang, J., Thompson, J., Htut, P. M., & Bowman, S. R. (2022).**
*BBQ: A Hand-Built Bias Benchmark for Question Answering.*
ACL Findings 2022. arXiv:2110.08193.
🔗 https://arxiv.org/abs/2110.08193

Benchmark di QA con esempi intersezionali. Non usato in Ouroboros: le categorie intersezionali del seed-set legacy (`gender-ethnicity`, `ethnicity-socio_economics`) ne ricalcavano il pattern, ma sono uscite con il dataset CLEAR-Bias, e la v3.0 misura il solo genere. Resta il riferimento canonico se l'estensione intersezionale verrà ripresa — vedi [09-future-intersectional-ablation.md](09-future-intersectional-ablation.md).

---

## Strumenti di sviluppo & SDK

| Tool | Uso | Reference |
|---|---|---|
| `ollama` | Client dell'attacker (e del judge in backend `ollama`) | https://github.com/ollama/ollama-python |
| `mlx-vlm` | Judge locale su Apple Silicon | https://github.com/Blaizzy/mlx-vlm |
| `mflux` | Target FLUX su Apple Silicon | https://github.com/filipstrand/mflux |
| `diffusers` | Target FLUX su NVIDIA CUDA | https://github.com/huggingface/diffusers |
| `facenet-pytorch` | MTCNN per la detection dei volti (pipeline FairFace) | https://github.com/timesler/facenet-pytorch |
| `pydantic` v2 | Validazione schema + derivazione dei campi di `GenderJudgement` | https://docs.pydantic.dev/ |
| `pandas` | DataFrame nei metrics | https://pandas.pydata.org/ |
| `jinja2` | Template `report.html` | https://jinja.palletsprojects.com/ |
| `sentence-transformers` + `hdbscan` | Clustering delle strategie dell'attacker | https://sbert.net/ |
| `streamlit` | Dashboard (`ouroboros dashboard`, extra `[web]`) | https://streamlit.io/ |
| `psutil` | Snapshot RAM per fase | https://github.com/giampaolo/psutil |
| `tqdm` | Progress bar | https://github.com/tqdm/tqdm |
| `python-dotenv` | Load `.env` | https://github.com/theskumar/python-dotenv |

Il SDK `google-genai` non è più una dipendenza: è uscito con il judge cloud.

---

## Come citare questo framework

Se usi Ouroboros in pubblicazioni, cita almeno:

- **Il paper PAIR originale** ([Chao et al., 2023](#pair)) come origine metodologica del loop.
- **Stable Bias** [Luccioni et al., 2023] come fonte del seed-set (175 professioni) e precedente per i bootstrap CI.
- **FairFace** [Kärkkäinen & Joo, 2021] per la pipeline di classificazione post-hoc.
- **T2ISafety** [Li et al., 2025] se riporti la validazione esterna del judge.
- **Qwen3-VL** [Bai et al., 2025] o il modello judge specificamente usato.
- **Girrbach et al., 2025** ([qui](#girrbach)) se confronti i risultati con l'audit statico su FLUX.

CLEAR-Bias [Cantini et al.] e BOLD [Dhamala et al., 2021] vanno citati **solo** se
usi i seed legacy in `data/legacy/` o la modalità `test`: il seed-set principale
non deriva più da loro.

## Da dove proseguire

→ [08-deviations.md](08-deviations.md) per il dettaglio di cosa abbiamo aggiunto/escluso rispetto a questi paper
