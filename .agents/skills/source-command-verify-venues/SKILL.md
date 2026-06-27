---
name: "source-command-verify-venues"
description: "Verifica con literature-alignment le venue non ancora confermate (senza ✓) nelle bibliografie di docs/"
---

# source-command-verify-venues

Use this skill when the user asks to run the migrated source command `verify-venues`.

## Command Template

Verifica le venue di pubblicazione non ancora confermate nei documenti bibliografici della tesi.

Procedura:

1. Cerca nei file `docs/lit-review-stereotipo-rappresentazionale.md`, `docs/related-work-digest.md` e
   `docs/tesi-skeleton.md` tutte le voci bibliografiche **senza il marker ✓** o con nota esplicita
   "verificare venue" / "in stampa" / "track non ancora verificabile".
   Residui noti al 2026-06-11:
   - **VIGNETTE** (arXiv:2505.22897) — accettata ad ACL 2026, ma track main vs Findings ignoto finché non escono gli atti;
   - **GenBreak** (arXiv:2506.10047) — solo preprint all'ultima verifica: ricontrollare se nel frattempo è stata accettata;
   - **Wan et al. 2024** (arXiv:2404.01030) — verificato 11/06/2026: solo preprint (nessuna voce ACL
     Anthology); ricontrollare solo se si sospetta una pubblicazione successiva;
   - eventuali voci nuove aggiunte dopo l'11/06/2026.

2. Lancia l'agente **literature-alignment** con l'elenco delle voci da verificare. Istruzioni per l'agente:
   per ciascun paper trovare la venue ufficiale (conferenza/journal + anno, distinguendo **main track vs
   Findings vs workshop**) oppure confermare che è solo preprint. Fonti ammesse: pagina arXiv (campo
   Comments / journal-ref), ACL Anthology, openaccess.thecvf.com, papers.nips.cc / OpenReview, dl.acm.org.
   Niente blog o aggregatori. Output: tabella titolo | venue verificata | fonte URL | confidenza, con le
   correzioni segnalate esplicitamente. L'agente è read-only: non modifica file.

3. Applica le correzioni nei file docs/ interessati: aggiungi il marker ✓ con il link alla fonte ufficiale,
   correggi le venue sbagliate, segna esplicitamente "solo arXiv preprint" dove non c'è accettazione.
   Mantieni la convenzione: workshop ≠ main proceedings va sempre indicato (pesa sulla citabilità in tesi).

4. Aggiorna la sezione "Limiti" della lit-review (o la nota corrispondente nello skeleton) con la data
   della verifica e i residui rimasti.
