export const meta = {
  name: 'verify-thesis-citations',
  description: 'Fact-check every citation in docs/tesi/sez2-background.md against primary web sources; flag errors and find exact corrections',
  phases: [
    { title: 'Verify', detail: 'one web-grounded fact-checker per citation' },
    { title: 'Recheck', detail: 'adversarial re-verification of each flagged problem' },
  ],
}

// Each item carries the verbatim Italian snippet + the verifiable metadata/stats the thesis asserts.
const CITES = [
  { key: 'Wan et al. 2024 (survey)', snippet: 'Wan et al., 2024, arXiv:2404.01030 — survey/tassonomia bias T2I (classification-based vs embedding-based); danni allocativi vs rappresentazionali', claims: 'arXiv:2404.01030; survey "of bias in text-to-image"; afferma: censisce 32 paper sul gender bias nei T2I e 23 di essi misurano il bias via associazione professione-genere; Ethics Statement cita "using these classification tools will inevitably risk propagating their bias in T2I bias evaluation results"; nota che i classificatori "fail to represent transgender individuals". Verifica id, titolo, autori, anno/venue, e i numeri 32 e 23.' },
  { key: 'Bird, Ungless & Kasirzadeh 2023', snippet: 'Bird, Ungless e Kasirzadeh (2023, AIES \'23, arXiv:2307.05543) — tipologia di rischi dei T2I generativi', claims: 'arXiv:2307.05543; venue AIES 2023; claim: 6 categorie di stakeholder e 22 tipologie di rischio, tra cui stereotyping ed erasure. Verifica id/venue/anno/autori e i numeri 6 e 22.' },
  { key: 'Luccioni et al. 2023 (Stable Bias)', snippet: 'Luccioni et al., 2023, Stable Bias, NeurIPS \'23 D&B, arXiv:2303.11408 — template "Photo portrait of a [X]"; classifier-free BLIP-VQA marker; bootstrap CI 95% (Tab.3) e 99% (Tab.5,6); 68+146 prompt, ~96k img; SD v1.4, SD v2, DALL-E 2', claims: 'arXiv:2303.11408; NeurIPS 2023 Datasets & Benchmarks; autori principali Luccioni (Hugging Face). Verifica: template "Photo portrait of a [X]"; metodo classifier-free via embedding/marker BLIP-VQA; bootstrap CI 95% in Tabella 3 e 99% in Tabelle 5 e 6; conteggi 68+146 prompt e ~96k immagini; modelli SD v1.4, SD v2, DALL-E 2.' },
  { key: 'Bianchi et al. 2023', snippet: 'Bianchi et al., 2023, FAccT \'23, arXiv:2211.03759 — prompt ordinari amplificano stereotipi "at scale"; SD v1.4, v2, DALL-E 2', claims: 'arXiv:2211.03759; venue FAccT 2023; titolo del tipo "Easily Accessible Text-to-Image Generation Amplifies Demographic Stereotypes at Scale"; modelli studiati SD v1.4, SD v2, DALL-E 2. Verifica id/venue/anno/autori/modelli.' },
  { key: 'BGPS / Plitsis et al. 2025', snippet: 'Plitsis et al., 2025, BGPS, arXiv:2512.08724 — beam search single-pass (non iterativa) con classificatori su attivazioni UNet; modello debiased 49% -> 79% immagini maschili', claims: 'arXiv:2512.08724 (dic 2025); acronimo BGPS; metodo: beam search single-pass steering del decoding via classificatori addestrati sulle attivazioni interne UNet; risultato: modello esplicitamente debiased passa dal 49% (prompt manuali) al 79% di immagini maschili sotto BGPS. Verifica id, autori (Plitsis), che NON sia iterativa, e i numeri 49% e 79%.' },
  { key: 'OASIS (ICLR 2025)', snippet: 'OASIS, ICLR 2025 Spotlight, arXiv:2501.00962 — open-set stereotype; Stereotype Score + WALS; FLUX.1 e SD v3 restano stereotipati', claims: 'arXiv:2501.00962; venue ICLR 2025 (Spotlight); metriche Stereotype Score e WALS (Weighted Attribute-Level Score), open-set; modelli testati includono FLUX.1 e Stable Diffusion v3. Verifica id, titolo (OASIS?), autori, venue ICLR 2025 spotlight, e le metriche.' },
  { key: 'Jha et al. 2024 (ViSAGe)', snippet: 'Jha et al., 2024, ViSAGe, ACL 2024, arXiv:2401.06310 — 135 nazionalità, 385 attributi da SeeGULL pre-filtrati per visual concreteness (Likert); ~3x', claims: 'arXiv:2401.06310; titolo ViSAGe; venue ACL 2024; claim: 135 nazionalità, 385 attributi derivati da SeeGULL filtrati da annotatori per "visual concreteness" su Likert; un attributo stereotipico ha prob ~3x di comparire. Verifica id/venue/autori e i numeri 135, 385, ~3x.' },
  { key: 'DALL-Eval (Cho, Zala, Bansal 2023)', snippet: 'Cho, Zala e Bansal, 2023, ICCV \'23, arXiv:2202.04053 — primo audit automatico bias sociale; BLIP-2 + ITA/Monk; 252 prompt occupazionali (83 prof) x 9 img; minDALL-E, SD v1.4, Karlo; MAD', claims: 'arXiv:2202.04053; venue ICCV 2023; auto-labeler BLIP-2 + skin tone ITA/Monk; 252 prompt occupazionali (83 professioni) x 9 immagini; modelli minDALL-E, SD v1.4, Karlo; metrica MAD. Verifica id/venue/autori e i conteggi.' },
  { key: 'FAIntbench (Luo et al. 2024)', snippet: 'Luo et al., 2024, arXiv:2405.17814 — tassonomia 4-D; manifestation factor η; CLIP; SDXL/varianti, PixArt-Σ, Playground v2.5, Stable Cascade (7)', claims: 'arXiv:2405.17814; titolo FAIntbench; manifestation factor η; struttura a 4 dimensioni; auto-labeler basato su CLIP; 7 modelli (SDXL e varianti, PixArt-Sigma, Playground v2.5, Stable Cascade). Verifica id/autori/anno e η e i 7 modelli.' },
  { key: 'BIGbench (Luo et al. 2024)', snippet: 'Luo et al., 2024, arXiv:2407.15240 — estende FAIntbench a 8 modelli + 3 metodi debias; auto-labeler Mini-InternVL-4B-1.5', claims: 'arXiv:2407.15240; titolo BIGbench; 8 modelli + 3 metodi di debiasing; auto-labeler Mini-InternVL-4B-1.5. Verifica id/autori/anno e i numeri e l\'auto-labeler.' },
  { key: 'T2ISafety (Li et al. 2025)', snippet: 'Li et al., 2025, CVPR \'25, arXiv:2501.12612 — NKL-Div (KL normalizzata in [0,1]); ~70k prompt (236 neutral fairness), 68k img annotate, 12 modelli; ImageGuard (InternLM-XComposer2)', claims: 'arXiv:2501.12612; venue CVPR 2025; titolo T2ISafety; metrica NKL-Div normalizzata in [0,1]; ~70.000 prompt, di cui 236 neutral per fairness; ~68.000 immagini annotate; 12 modelli; auto-labeler ImageGuard basato su InternLM-XComposer2. Verifica id/venue/autori e i numeri 70k, 68k, 12, 236.' },
  { key: 'Girrbach et al. 2025', snippet: 'Girrbach et al., 2025, arXiv:2503.23398 — ~2,3M img (2.293.295 dopo filtering); FLUX, FLUX-Schnell, SD 3.5 L/M, SD 3 M; 3.217 prompt gender-neutral; Female Ratio ℛf; YOLOv10 + InternVL2-8B; error bar Fig 4/15/16 std cross-prompt', claims: 'arXiv:2503.23398; ~2,3 milioni di immagini (2.293.295 dopo filtering); modelli FLUX, FLUX-Schnell, SD 3.5 Large/Medium, SD 3 Medium; 3.217 prompt gender-neutral; metrica Female Ratio; pipeline YOLOv10 + InternVL2-8B; le error bar (Fig 4/15/16) sono std cross-prompt. Verifica id/autori e i numeri 2.293.295, 3.217, i modelli, la pipeline.' },
  { key: 'FairFace (Kärkkäinen & Joo 2021)', snippet: 'Kärkkäinen e Joo, 2021, WACV \'21, arXiv:1908.04913 — ResNet-34; 108.501 volti da YFCC-100M; 7 razza/2 genere/9 età; 88% acc su classificazione etnica a 4 classi', claims: 'arXiv:1908.04913; venue WACV 2021; autori Kärkkäinen e Joo; architettura ResNet-34; 108.501 volti dal dataset YFCC-100M; 7 classi di razza, 2 di genere, 9 fasce di età; accuratezza dichiarata ~88% sul task di classificazione etnica a 4 classi. Verifica id/venue/autori e i numeri 108.501, 7/2/9, 88%, e la provenienza YFCC-100M.' },
  { key: 'Gender Shades (Buolamwini & Gebru 2018)', snippet: 'Buolamwini e Gebru, 2018, Gender Shades, PMLR 81 — errori fino al 34,7% su donne pelle scura vs 0,8% uomini pelle chiara (fattore 43)', claims: 'venue PMLR vol 81 (FAT*/FAccT 2018); errore fino al 34,7% sulle donne dalla pelle scura contro 0,8% sugli uomini dalla pelle chiara. Verifica venue/anno/autori e i numeri 34,7% e 0,8%.' },
  { key: 'Fournier-Montgieux et al. 2025', snippet: 'Fournier-Montgieux et al., 2025, arXiv:2510.20482 — l\'errore del classificatore demografico si propaga nelle stime di fairness a valle', claims: 'arXiv:2510.20482; tesi: l\'errore del classificatore demografico propaga nelle stime di fairness downstream, invalidando le conclusioni anche se il classificatore è mediamente accurato. Verifica id/autori/anno e la tesi.' },
  { key: 'Keyes 2018', snippet: 'Keyes, 2018, CSCW, DOI 10.1145/3274357 — automatic gender recognition intrinsecamente trans-esclusivo', claims: 'DOI 10.1145/3274357; venue CSCW 2018; titolo "The Misgendering Machines"; tesi: AGR (automatic gender recognition) trans-esclusivo per struttura binaria. Verifica DOI/venue/autore/titolo.' },
  { key: 'FLIRT (Mehrabi et al. 2024)', snippet: 'Mehrabi et al., 2024, EMNLP \'24, arXiv:2308.04265 — LLM attacker vs Stable Diffusion; FIFO/LIFO/Scoring; ~85% successo (~80% SD vanilla, ~60% SD safe)', claims: 'arXiv:2308.04265; titolo FLIRT (Feedback Loop In-context Red Teaming); venue EMNLP 2024; meccanismo FIFO/LIFO/Scoring; tasso di successo ~85% (~80% su SD vanilla, ~60% su SD safe). Verifica id/venue/autori e i numeri.' },
  { key: 'FGPI (Xu et al. 2025)', snippet: 'Xu et al., 2025, "Automated Red Teaming for Text-to-Image Models through Feedback-Guided Prompt Iteration with Vision-Language Models", ICCV 2025, IEEE Xplore doc. 11444110 — VLM fine-tunato come red-teamer', claims: 'venue ICCV 2025; IEEE Xplore document 11444110; titolo come citato; VLM fine-tunato in loop di feedback; obiettivo immagini dannose/illegali; acronimo FGPI. Verifica venue/anno/autori, l\'esistenza del doc IEEE Xplore 11444110, e che il titolo combaci.' },
  { key: 'AutoPrompt (Liu et al. 2025)', snippet: 'Liu et al., ICCV 2025, arXiv:2510.24034 — suffissi adversariali human-readable via LLM per aggirare filtri di SD e Leonardo.Ai; NSFW', claims: 'arXiv:2510.24034; venue ICCV 2025; suffissi adversariali human-readable generati via LLM; bersagli Stable Diffusion e Leonardo.Ai; obiettivo NSFW. Verifica id/venue/autori e che il nome/sistema sia "AutoPrompt".' },
  { key: 'SneakyPrompt (Yang et al. 2024)', snippet: 'Yang et al., 2024, IEEE S&P \'24, arXiv:2305.12082 — primo attacco completamente automatizzato contro T2I; perturbazione token via RL', claims: 'arXiv:2305.12082; venue IEEE S&P (Security & Privacy) 2024; titolo SneakyPrompt; metodo: perturbazione di token via reinforcement learning. Verifica id/venue/autori.' },
  { key: 'MMA-Diffusion (Yang et al. 2024)', snippet: 'Yang et al., 2024, CVPR \'24, arXiv:2311.17516 — estende il paradigma all\'input multimodale (testo e immagine)', claims: 'arXiv:2311.17516; venue CVPR 2024; titolo MMA-Diffusion; attacco multimodale (testo+immagine). Verifica id/venue/autori.' },
  { key: 'ART (Li et al. 2024)', snippet: 'Li et al., 2024, NeurIPS \'24, arXiv:2405.19360 — combina LLM e VLM per prompt sicuri-in-formulazione ma pericolosi-in-output; ~metà dei prompt safe manipolabile', claims: 'arXiv:2405.19360; titolo ART (Automatic Red-Teaming); venue NeurIPS 2024; combina LLM e VLM; claim che circa metà dei prompt classificati safe può essere manipolata. Verifica id/venue/autori e il claim "~metà".' },
  { key: 'GenBreak (Wang et al. 2025)', snippet: 'Wang et al., 2025, solo arXiv preprint, arXiv:2506.10047 — red-teaming generalizzato di generatori T2I basato su LLM', claims: 'arXiv:2506.10047; titolo GenBreak; stato: solo preprint arXiv (non a venue); approccio LLM-based red-teaming T2I, obiettivo safety. Verifica id/autori/titolo e che sia preprint-only.' },
  { key: 'Implicit Bias Injection (Huang et al. 2025)', snippet: 'Huang et al., CVPR 2025, arXiv:2504.01819 — inietta bias nello spazio embedding del prompt; threat model white-box di poisoning', claims: 'arXiv:2504.01819; venue CVPR 2025; titolo "Implicit Bias Injection Attacks"; threat model white-box poisoning nello spazio embedding del prompt. Verifica id/venue/autori e il threat model.' },
  { key: 'BiasPainter (2024)', snippet: 'BiasPainter, ACM MM 2024, arXiv:2401.00763 — metamorphic testing di image editing; accuratezza 90,8% vs umani', claims: 'arXiv:2401.00763; venue ACM Multimedia 2024; titolo BiasPainter; metamorphic testing su image editing; accuratezza 90,8% vs valutazione umana. Verifica id/venue/autori e il numero 90,8%.' },
  { key: 'Adversarial Nibbler (Quaye et al. 2024)', snippet: 'Quaye et al., 2024, FAccT \'24, arXiv:2403.12075 — red-teaming human-in-the-loop; include "stereotipi e bias"; benchmark statici non si adattano a strategie nuove', claims: 'arXiv:2403.12075; venue FAccT 2024; titolo Adversarial Nibbler; red-teaming human-in-the-loop (utenti umani generano i prompt); include categoria stereotipi/bias. Verifica id/venue/autori.' },
  { key: 'Prompting4Debugging (Chin et al. 2024)', snippet: 'Chin et al., 2024, ICML \'24, arXiv:2309.06135 — loop ottimizzante scopre più prompt problematici di liste fisse', claims: 'arXiv:2309.06135; titolo Prompting4Debugging (P4D); venue ICML 2024. Verifica id/venue/autori.' },
  { key: 'PAIR (Chao et al. 2023)', snippet: 'Chao et al., 2023, arXiv:2310.08419 — Prompt Automatic Iterative Refinement; LLM attacker conversazionale; judge score 1-10; <20 query', claims: 'arXiv:2310.08419; titolo PAIR (Prompt Automatic Iterative Refinement); judge su scala 1-10; meno di ~20 query per target. Verifica id/autori e i dettagli (scala 1-10, <20 query).' },
  { key: 'TAP (Mehrotra et al. 2023/2024)', snippet: 'Mehrotra et al., 2023/NeurIPS \'24, arXiv:2312.02119 — Tree of Attacks with Pruning; >80% successo, <30 query', claims: 'arXiv:2312.02119; titolo TAP (Tree of Attacks with Pruning); venue NeurIPS 2024 (preprint 2023); tassi di successo >80% con <30 query. Verifica id/venue/autori e i numeri.' },
  { key: 'Zheng et al. 2023 (MT-Bench)', snippet: 'Zheng et al., 2023, MT-Bench/LLM-as-a-Judge, NeurIPS \'23 D&B, arXiv:2306.05685 — position/verbosity/self bias', claims: 'arXiv:2306.05685; titolo "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"; venue NeurIPS 2023 Datasets & Benchmarks; documenta position bias, verbosity bias, self-enhancement bias. Verifica id/venue/autori e i bias elencati.' },
  { key: 'MJ-Bench (Chen et al. 2024)', snippet: 'Chen et al., 2024, arXiv:2407.04842 — estende l\'analisi judge ai modelli multimodali; affidabilità disomogenea', claims: 'arXiv:2407.04842; titolo MJ-Bench; benchmark di multimodal judge; affidabilità disomogenea tra task/modelli. Verifica id/autori/anno.' },
  { key: 'Kumar et al. 2026 (Rank not Score)', snippet: 'Kumar et al., 2026, arXiv:2604.25235 — "VLM Judges Can Rank but Cannot Score"; judge VLM ordinano ma producono score assoluti incoerenti con gli umani', claims: 'arXiv:2604.25235 (ATTENZIONE: id molto recente, aprile 2026 — verificare che esista davvero e che NON sia un id inventato); titolo del tipo "VLM Judges Can Rank but Cannot Score". Verifica con cura l\'esistenza dell\'arXiv id e del titolo/autori; se non esiste, segnalalo come errore maggiore e cerca il paper reale con quel claim.' },
  { key: 'CLEAR-Bias (Cantini et al. 2025)', snippet: 'Cantini et al., 2025, Machine Learning 114(249), arXiv:2504.07887 — elicitazione adversariale bias in LLM testuali; control set 400 coppie annotate; Cohen\'s κ=0,82', claims: 'arXiv:2504.07887; titolo CLEAR-Bias; rivista "Machine Learning" volume 114, article 249; control set di 400 coppie annotate da umani; Cohen\'s kappa = 0,82. Verifica id, la citazione di rivista Machine Learning 114(249), e i numeri 400 e κ=0,82.' },
  { key: 'Narayanan et al. 2025 (Bias in the Picture)', snippet: 'Narayanan et al., 2025, arXiv:2509.19659, NeurIPS 2025 Workshop — "Bias in the Picture"; judge VLM biased su gender e occupation', claims: 'arXiv:2509.19659; titolo "Bias in the Picture"; presentato a un NeurIPS 2025 Workshop (non main proceedings); claim: judge VLM sistematicamente biased su gender e occupation. Verifica id/autori, il titolo, e lo status workshop.' },
  { key: 'MineTheGap (Cohen et al. 2025)', snippet: 'Cohen et al., 2025, arXiv:2512.13427 — algoritmo genetico con LLM come operatore di mutazione; assi open-set; Spearman ρ=0,72 vs BLS; SD 1.4/2.1/3, FLUX.1-schnell', claims: 'arXiv:2512.13427 (dic 2025); titolo MineTheGap; algoritmo genetico con LLM come operatore di mutazione; validato vs proporzioni occupazionali BLS con Spearman ρ=0,72; modelli SD 1.4, 2.1, 3, FLUX.1-schnell. Verifica id/autori e i numeri ρ=0,72 e i modelli.' },
  { key: "D'Incà et al. 2023", snippet: "D'Incà et al., 2023, arXiv:2312.13053 — critica documentata del riportare stime puntuali senza misura di incertezza", claims: "arXiv:2312.13053; autori D'Incà et al., 2023. ATTENZIONE: verificare quale paper è realmente all'id 2312.13053 e se sostiene davvero la critica delle stime puntuali senza CI; D'Incà è anche autore di OpenBias (CVPR 2024). Verifica id/titolo/autori e che il contenuto combaci con la citazione." },
  { key: 'Cherian & Candès 2024', snippet: 'Cherian e Candès, 2024, JMLR 25, arXiv:2305.03712 — fairness auditing come multiple-hypothesis-testing con bootstrap; certificare assenza di disparità', claims: 'arXiv:2305.03712; titolo "Statistical Inference for Fairness Auditing" (o simile); rivista JMLR volume 25; autori Cherian e Candès. Verifica id, la pubblicazione su JMLR 25, autori e tesi.' },
  { key: 'Social Norm Bias 2021', snippet: 'Social Norm Bias, 2021, arXiv:2108.11056 — eterogeneità intra-gruppo nelle categorie demografiche grossolane', claims: 'arXiv:2108.11056; titolo contenente "Social Norm Bias"; anno 2021. Verifica id/titolo/autori/anno e il tema (intra-group heterogeneity / coarse categories).' },
  { key: 'Krishnan et al. 2020', snippet: 'Krishnan et al., 2020, arXiv:2009.11491 — eterogeneità nelle categorie etniche grossolane', claims: 'arXiv:2009.11491; autori Krishnan et al., 2020. Verifica id/titolo/autori/anno e che il tema sia coerente con la citazione (categorie etniche grossolane / bias).' },
]

const VERIFY_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    key: { type: 'string' },
    arxiv_resolves: { type: 'string', enum: ['yes', 'no', 'no_id_claimed', 'unchecked'] },
    real_title: { type: 'string' },
    real_authors: { type: 'string' },
    real_venue: { type: 'string' },
    real_year: { type: 'string' },
    verdict: { type: 'string', enum: ['correct', 'minor_error', 'major_error', 'uncertain'] },
    problems: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          field: { type: 'string', enum: ['arxiv_id', 'venue', 'year', 'authors', 'title', 'statistic', 'other'] },
          claimed: { type: 'string' },
          actual: { type: 'string' },
          correction: { type: 'string' },
          source_url: { type: 'string' },
          confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
        },
        required: ['field', 'claimed', 'actual', 'correction', 'source_url', 'confidence'],
      },
    },
    notes: { type: 'string' },
  },
  required: ['key', 'arxiv_resolves', 'real_title', 'real_authors', 'real_venue', 'real_year', 'verdict', 'problems', 'notes'],
}

const RECHECK_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    key: { type: 'string' },
    final_verdict: { type: 'string', enum: ['correct', 'minor_error', 'major_error', 'uncertain'] },
    confirmed_problems: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          field: { type: 'string' },
          claimed: { type: 'string' },
          actual: { type: 'string' },
          correction: { type: 'string' },
          source_url: { type: 'string' },
          holds: { type: 'boolean' },
          reasoning: { type: 'string' },
        },
        required: ['field', 'claimed', 'actual', 'correction', 'source_url', 'holds', 'reasoning'],
      },
    },
    notes: { type: 'string' },
  },
  required: ['key', 'final_verdict', 'confirmed_problems', 'notes'],
}

const verifyPrompt = (it) => `You are a meticulous academic citation fact-checker. Verify ONE citation from an Italian master's thesis against PRIMARY sources on the web. Use WebSearch and WebFetch (fetch https://arxiv.org/abs/<id> for any claimed arXiv id; also search the title/authors and the venue).

CITATION KEY: ${it.key}
HOW IT IS CITED IN THE THESIS (verbatim):
${it.snippet}

CLAIMED METADATA / FACTS TO CHECK:
${it.claims}

TASK:
1. Resolve the claimed arXiv id (if any): fetch the abstract page. Does it resolve to the SAME paper the thesis intends? Record arxiv_resolves = yes/no/no_id_claimed.
2. Record the REAL title, first author + et al., publication venue (conference/journal + track), and year.
3. Check EACH claimed fact independently: arXiv id correctness, venue, year, author spelling, AND every specific number/statistic (sample sizes, percentages, counts, κ, ρ, model counts, etc.).
4. Classify each discrepancy by field (arxiv_id / venue / year / authors / title / statistic / other).

RULES:
- If the arXiv id does NOT resolve to the intended paper => MAJOR error. Find the CORRECT arXiv id and put it in "correction".
- "NeurIPS D&B" = Datasets & Benchmarks track (legitimate). A paper accepted at a venue different from the claimed one => error; arXiv-only when a venue is claimed => error.
- For a statistic, ONLY flag it if you can confirm the real value differs (give the real value + source url). If you cannot confirm, do NOT flag — leave it and lower confidence / mention in notes. Never invent numbers.
- Distinguish MAJOR (wrong id / wrong paper / wrong venue / fabricated id / wrong headline statistic) from MINOR (small year/track slip, author initial, rounding).
- verdict: correct = all checks pass; minor_error = only minor slips; major_error = at least one major problem; uncertain = could not resolve enough.
- Every problem MUST carry a source_url backing the "actual" value.

Return the structured verdict.`

const recheckPrompt = (it, v) => `You are an adversarial SECOND reviewer. A first fact-checker flagged problems with a thesis citation. Independently RE-VERIFY each alleged problem against primary web sources (WebSearch + WebFetch on arxiv.org / publisher / DOI). Your job is to OVERTURN problems that are actually wrong (original citation was fine) and CONFIRM only those that genuinely hold. Default holds=false unless you can independently confirm the discrepancy from an authoritative source.

CITATION KEY: ${it.key}
THESIS CITATION (verbatim): ${it.snippet}
CLAIMED FACTS: ${it.claims}

FIRST CHECKER FOUND:
real_title: ${v.real_title}
real_authors: ${v.real_authors}
real_venue: ${v.real_venue}
real_year: ${v.real_year}
verdict: ${v.verdict}
problems: ${JSON.stringify(v.problems)}
notes: ${v.notes}

For EACH alleged problem: independently confirm via the web, set holds=true/false, give one-line reasoning, and supply the authoritative correction + source_url. Also add any NEW genuine problem the first checker missed (set holds=true for it). Then give final_verdict for the citation.`

phase('Verify')
log(`Fact-checking ${CITES.length} citations from sez2-background.md against primary sources`)

const results = await pipeline(
  CITES,
  (it) => agent(verifyPrompt(it), { label: `verify:${it.key.slice(0, 28)}`, phase: 'Verify', schema: VERIFY_SCHEMA, model: 'sonnet', effort: 'medium' }),
  (v, it) => {
    if (!v) return { item: it, verify: null, recheck: null }
    if (v.verdict === 'correct') return { item: it, verify: v, recheck: null }
    // flagged => adversarial recheck of the alleged problems
    return agent(recheckPrompt(it, v), { label: `recheck:${it.key.slice(0, 26)}`, phase: 'Recheck', schema: RECHECK_SCHEMA })
      .then((r) => ({ item: it, verify: v, recheck: r }))
  }
)

const clean = results.filter(Boolean)
const flagged = clean.filter((r) => r.verify && r.verify.verdict !== 'correct')
log(`Done: ${clean.length} checked, ${flagged.length} flagged for correction`)

return {
  total: CITES.length,
  checked: clean.length,
  results: clean,
}
