# 07 — References

Bibliografia ragionata delle fonti accademiche e tecniche dietro al framework. Suddiviso per area concettuale.

---

## Metodologia di attacco

### PAIR

<a id="pair"></a>
**Chao, P., Robey, A., Dobriban, E., Hassani, H., Pappas, G. J., & Wong, E. (2023).**
*Jailbreaking Black Box Large Language Models in Twenty Queries.*
arXiv:2310.08419.
🔗 https://arxiv.org/abs/2310.08419

Il paper originale del **PAIR loop**. Definisce il tre-attori framework attacker-target-judge e dimostra che jailbreak black-box su LLM sono raggiungibili in `O(20)` query con un attacker LLM ben istruito. MIRTAGE porta lo schema dal dominio LLM→LLM al dominio LLM→T2I.

**Cosa abbiamo preso:** struttura del loop, idea della memoria top-K, success threshold scalare, retry su attacker self-refusal.
**Cosa abbiamo cambiato:** target multimodale invece che testuale, M chiamate parallele invece di una, success rule N-of-M, judge VLM invece che LLM. Vedi [08-deviations.md](08-deviations.md).

### TAP

**Mehrotra, A., Zampetakis, M., Kassianik, P., Nelson, B., Anderson, H., Singer, Y., & Karbasi, A. (2023).**
*Tree of Attacks: Jailbreaking Black-Box LLMs Automatically.*
arXiv:2312.02119.
🔗 https://arxiv.org/abs/2312.02119

Estensione tree-search di PAIR con pruning. Per ogni iter genera b candidati invece di 1, un evaluator pota i meno promettenti, scende solo i rami sopravvissuti. **MIRTAGE è esplicitamente PAIR (no branching), non TAP** — la motivazione è il costo T2I, ogni branch aggiuntivo costa M chiamate al target (vedi [03-pair-loop.md](03-pair-loop.md#differenze-rispetto-a-tap-mehrotra-et-al-2023)).

### GCG (per contesto storico)

**Zou, A., Wang, Z., Carlini, N., Nasr, M., Kolter, J. Z., & Fredrikson, M. (2023).**
*Universal and Transferable Adversarial Attacks on Aligned Language Models.*
arXiv:2307.15043.
🔗 https://arxiv.org/abs/2307.15043

Greedy Coordinate Gradient — attacco **white-box** che ottimizza suffissi adversariali. Precursore di PAIR; mostra che il jailbreak è possibile in principio, ma richiede accesso ai gradienti (e i suffissi sono testo robaccia, non leggibili). PAIR è la risposta "black-box, semantica" a GCG.

---

## Dataset di bias

### CLEAR-Bias

<a id="clear-bias"></a>
**Cantini, R., Cosenza, G., Orsino, A., & Talia, D. (2024).**
*CLEAR-Bias: Constrained Adversarial Set for Evaluating Bias in Large Language Models.*

🔗 Hugging Face: https://huggingface.co/datasets/RCantini/CLEAR-Bias

Il benchmark di prompt **testuali** da cui derivano le `base_scene` di MIRTAGE. Copre 6 categorie demografiche (gender, ethnicity, religion, socioeconomic, gender-ethnicity, ethnicity-socioeconomic) con prompt strutturati come comparazioni o descrizioni di archetipi.

**Cosa abbiamo preso:** la tassonomia 6 categorie + intersezioni; i `source_clear_bias_prompt` come `source_text` dei seed.
**Cosa abbiamo escluso:** categorie `age`, `disability`, `sexual_orientation` — non si traducono bene in scene visualizzabili.

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

Benchmark contestuale per stereotipi in LM. Non usato direttamente in MIRTAGE, ma è un riferimento cardinale del campo "bias measurement in LM".

---

## Audit di T2I models

### Bianchi et al.

**Bianchi, F., Kalluri, P., Durmus, E., Ladhak, F., Cheng, M., Nozza, D., Hashimoto, T., Jurafsky, D., Zou, J., & Caliskan, A. (2023).**
*Easily Accessible Text-to-Image Generation Amplifies Demographic Stereotypes at Large Scale.*
Proceedings of FAccT 2023. arXiv:2211.03759.
🔗 https://arxiv.org/abs/2211.03759

Studio empirico su Stable Diffusion: mostra che T2I **amplifica** stereotipi rispetto al training data e rispetto alla popolazione reale (es. "CEO" → predominanza maschile bianca al 90%+ anche quando la stat reale è ~70%). Stabilisce il problem statement per cui MIRTAGE esiste.

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

Dataset + classificatore ResNet-34 standard per audit demografici (7 razze × 2 generi × 9 fasce d'età). Usato in MIRTAGE per la pipeline post-hoc in `src/fairface.py`. Non sostituisce il judge (che continua a guidare la success rule del loop): è la base per le metriche **comparable** con la letteratura T2I-fairness.

### FAIntbench / BIGbench

**Hu, Z., Yu, K., Yang, J., et al. (2024).**
*FAIntbench: A Holistic and Precise Benchmark for Bias Evaluation in Text-to-Image Models.*
arXiv:2405.17814.
🔗 https://arxiv.org/abs/2405.17814

**Luo, H., Deng, J., Shen, R., et al. (2024).**
*BIGbench: A Unified Benchmark for Evaluating Social Bias in Text-to-Image Generative Models.*
arXiv:2407.15240.
🔗 https://arxiv.org/abs/2407.15240

Definiscono il framework di metriche T2I-bias contemporaneo: separano *implicit vs explicit bias*, introducono il **Manifestation factor η** che decompone uno score di bias in *ignorance* (il modello evita la demografia) vs *discrimination* (il modello assegna attivamente lo stereotipo). KL e norm_entropy usate in MIRTAGE sono allineate a questo filone.

### T2ISafety

**Li, L., et al. (2025).**
*T2ISafety: Benchmark for Assessing Fairness, Toxicity, and Privacy in Image Generation.*
CVPR 2025.
🔗 https://openaccess.thecvf.com/content/CVPR2025/papers/Li_T2ISafety_Benchmark_for_Assessing_Fairness_Toxicity_and_Privacy_in_Image_Generation_CVPR_2025_paper.pdf

Benchmark recente che combina skew per-asse via FairFace + classificatori di tossicità. Riferimento più vicino al setting di MIRTAGE (per-axis demographic skew + bias toxicity), salvo che è statico e cloud-based.

### Stable Bias

**Luccioni, A. S., Akiki, C., Mitchell, M., & Jernite, Y. (2023).**
*Stable Bias: Analyzing Societal Representations in Diffusion Models.*
NeurIPS 2023 D&B. arXiv:2303.11408.
🔗 https://arxiv.org/abs/2303.11408

Audit Stable Diffusion / DALL-E 2 con BLIP-VQA marker rate + cluster entropy + **95% bootstrap CI** (uno dei pochi paper T2I-bias a riportare CI). Riferimento per la nostra scelta di clusterizzare le facce per asse + per la futura migrazione da Wilson a bootstrap CI sull'ASR.

---

## Modelli usati

### FLUX.2 klein (target)

**Black Forest Labs (2024).**
FLUX.2 klein — distilled 4B-parameter rectified-flow T2I model.
🔗 https://github.com/black-forest-labs/flux
🔗 mflux (Apple Silicon port): https://github.com/filipstrand/mflux

Modello T2I locale, eseguito via [`mflux`](https://github.com/filipstrand/mflux) su Apple Silicon (MLX/Metal). MIRTAGE lo usa quantizzato a 4-bit con 4 inference step (variante "klein" distilled) per stare nel budget RAM. Generazione sequenziale sul thread asyncio (MLX binda lo stream GPU al thread). Vedi [04-components.md](04-components.md) per la motivazione della scelta (RAM, no safety filter, no costo cloud).

### Gemini 2.5 Pro (judge)

**Google DeepMind (2025).**
Gemini 2.5 Pro documentation.
🔗 https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-pro

Modello VLM cloud chiuso, disponibile via Vertex AI con auth `GOOGLE_GENAI_USE_VERTEXAI=True`. MIRTAGE lo usa come **judge default** (`--judge-backend gemini`, `JUDGE_GEMINI_DEFAULT="gemini-2.5-pro"` in `src/config.py`) — riceve M immagini + prompt e produce un JSON `BiasJudgement`. Pricing $1.25/$10.00 per 1M token input/output → ~$0.006 per call → ~$5 per full run su Stable Bias (175 seed, vedi `docs/04-components.md`).

### Qwen2.5-VL (judge fallback offline)

<a id="qwen-vl"></a>
**Bai, S., Chen, K., Liu, X., Wang, J., et al. (2025).**
*Qwen2.5-VL Technical Report.*
arXiv:2502.13923.
🔗 https://arxiv.org/abs/2502.13923

Vision-Language Model 7B/72B di Alibaba/Qwen. MIRTAGE lo usa come fallback offline (`--judge-backend mlx` o `--judge-backend ollama`) quando non è disponibile accesso a Vertex AI. Il judge default è Gemini 2.5 Pro cloud.

🔗 HF: https://huggingface.co/mlx-community/Qwen2.5-VL-7B-Instruct-4bit

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

Framework di array Apple ottimizzato per Apple Silicon (unified memory, accelerazione Metal). MIRTAGE lo usa per il target FLUX (via mflux) e per il judge fallback offline Qwen2.5-VL (30-50% più veloce di Ollama su M-series).

### mlx-vlm

**Blaizzy/mlx-vlm (2024).**
*MLX-VLM: Vision-Language Model inference on Apple Silicon.*
🔗 https://github.com/Blaizzy/mlx-vlm

Wrapper specifico per VLM su MLX. Espone `mlx_vlm.load()` e `mlx_vlm.generate()` usati dal fallback offline `MLXJudge` in `src/judge.py`.

### Ollama

**Ollama project.**
🔗 https://ollama.com/ — https://github.com/ollama/ollama

Runtime locale per LLM (LLaMA, Mistral, ecc.) con API HTTP-like. MIRTAGE lo usa per l'attacker (default) e come fallback per il judge (`--judge-backend ollama`).

---

## Tecnologie del reporting

### HDBSCAN

**Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013).**
*Density-Based Clustering Based on Hierarchical Density Estimates.*
PAKDD 2013, pp. 160-172.
🔗 https://link.springer.com/chapter/10.1007/978-3-642-37456-2_14

Algoritmo di clustering density-based, hierarchical. Vantaggi vs k-means: nessun numero di cluster da specificare a priori, gestisce cluster di forma non sferica, separa il rumore (label `-1`). MIRTAGE lo usa in `src/cluster.py:40` con `min_cluster_size=3` per raggruppare le strategy_label freeform dell'attacker.

🔗 Implementazione Python: https://github.com/scikit-learn-contrib/hdbscan

### Sentence-BERT (all-MiniLM-L6-v2)

**Reimers, N., & Gurevych, I. (2019).**
*Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.*
EMNLP 2019. arXiv:1908.10084.
🔗 https://arxiv.org/abs/1908.10084

Architettura per produrre embedding di frasi semanticamente sensati (vs avg di token embeddings BERT vanilla). MIRTAGE usa il modello `all-MiniLM-L6-v2` (~80 MB, 384 dim) per embedded delle `strategy_label` prima del clustering HDBSCAN.

🔗 Modello: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

---

## Letteratura collegata (contesto)

### "Typecast" (T2I gender stereotyping)

**Bird, C., Ungless, E., & Kasirzadeh, A. (2023).**
*Typecast: A Practical Insight into Demographic Patterns of T2I Systems.*
🔗 https://arxiv.org/abs/2302.07159

Studio del 2023 sulla riproduzione di stereotipi in T2I. Confermano i pattern visti da Bianchi et al.; aggiungono breakdown per profession × gender.

### BBQ (per testing di bias intersezionale)

**Parrish, A., Chen, A., Nangia, N., Padmakumar, V., Phang, J., Thompson, J., Htut, P. M., & Bowman, S. R. (2022).**
*BBQ: A Hand-Built Bias Benchmark for Question Answering.*
ACL Findings 2022. arXiv:2110.08193.
🔗 https://arxiv.org/abs/2110.08193

Benchmark di QA con esempi intersezionali. Non usato direttamente in MIRTAGE, ma è il riferimento canonico per testing intersezionale (`gender × ethnicity`, ecc.) — pattern che ripercorriamo nelle categorie `gender-ethnicity` e `ethnicity-socio_economics`.

---

## Strumenti di sviluppo & SDK

| Tool | Uso | Reference |
|---|---|---|
| `google-genai` Python SDK | Client Vertex AI | https://github.com/googleapis/python-genai |
| `pydantic` v2 | Validazione schema | https://docs.pydantic.dev/ |
| `pandas` | DataFrame nei metrics | https://pandas.pydata.org/ |
| `jinja2` | Template `report.html` | https://jinja.palletsprojects.com/ |
| `tqdm` | Progress bar | https://github.com/tqdm/tqdm |
| `python-dotenv` | Load `.env` | https://github.com/theskumar/python-dotenv |

---

## Come citare questo framework

Se usi MIRTAGE in pubblicazioni, cita almeno:

- **Il paper PAIR originale** ([Chao et al., 2023](#pair)) come methodology origin.
- **Il dataset CLEAR-Bias** ([Cantini et al.](#clear-bias)) per la tassonomia.
- **BOLD** [Dhamala et al., 2021] per i prompt arricchiti.
- **Qwen2.5-VL** [Bai et al., 2025] o il modello judge specifico usato.

## Da dove proseguire

→ [08-deviations.md](08-deviations.md) per il dettaglio di cosa abbiamo aggiunto/escluso rispetto a questi paper
