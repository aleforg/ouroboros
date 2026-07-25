# Documentazione Ouroboros

Questa cartella raccoglie la documentazione **teorica** del framework: cosa fa, perché funziona, da quali paper attinge e quali sono le scelte di design rispetto alla letteratura.

Per la documentazione **operativa** (come installare, lanciare run, leggere i risultati) vedi `../README.md`.

---

## Indice

| # | Documento | Contenuto |
|---|---|---|
| 01 | [Overview](01-overview.md) | Obiettivi, non-obiettivi, vincoli hardware |
| 02 | [Architecture](02-architecture.md) | Diagrammi del sistema e mappa dei moduli |
| 03 | [PAIR loop](03-pair-loop.md) | Teoria del loop iterativo, success rule, memoria |
| 04 | [Components](04-components.md) | Attacker, Judge, Target — i tre attori |
| 05 | [Dataset](05-dataset.md) | Seeds Stable Bias, gruppi stereotipici BLS, dataset legacy |
| 06 | [Metrics](06-metrics.md) | ASR, queries-to-success, E(s), strategy clustering |
| 07 | [References](07-references.md) | Bibliografia: paper e fonti |
| 08 | [Deviations](08-deviations.md) | Cosa abbiamo aggiunto/escluso rispetto ai paper e al design contract v1 |
| 09 | [Future: intersectional ablation](09-future-intersectional-ablation.md) | Piano deferred per estendere il dataset con probe intersezionali (option C) |

---

## Convenzioni

- **Citazioni inline** in formato `[Cognome et al., anno]` → risolte in [07-references.md](07-references.md)
- **Diagrammi** in [Mermaid](https://mermaid.js.org/) (renderizzati nativamente da GitHub) + ASCII-art per le viste compatte
- **Riferimenti al codice** in formato `path/file.py:linea` per navigazione diretta dall'IDE

## Percorsi di lettura

| Obiettivo | Sequenza |
|---|---|
| Primo approccio al framework | 01 → 03 → 02 → 04 (overview → loop → architettura → componenti) |
| Comprensione delle metriche | 03 → 06 |
| Confronto con la letteratura | 07 → 08 |
| Aggiunta di un componente | 02 → 04 |
