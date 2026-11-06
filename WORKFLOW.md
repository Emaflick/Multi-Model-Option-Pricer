# Come usare la pipeline (guida per Emanuele)

## Setup (15 min)
```bash
cd qpricer
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
git init && git add -A && git commit -m "pipeline scaffold"
claude                                                  # avvia Claude Code nella cartella
```
Gli hook usano `python3`: su Windows, se non esiste, sostituisci `python3` con `python` in `.claude/settings.json`.

## Cosa c'è dentro
| File | Ruolo |
|---|---|
| `CLAUDE.md` | memoria di progetto: missione, convenzioni, regole anti-barare |
| `docs/ROADMAP.md` | fasi 0–7 con **gate numerici** di accettazione |
| `.claude/agents/` | `quant-researcher` (matematica) → `implementer` (codice) → `validator` (verifica indipendente) → `interview-coach` (cheat-sheet / mock) → `reporter` |
| `.claude/commands/` | `/build-all`, `/phase N`, `/research`, `/validate`, `/report`, `/cheatsheet`, `/grill`, `/explain` |
| `.claude/hooks/` | blocca modifiche ai valori golden; dopo ogni edit lancia ruff + test veloci e rimanda gli errori a Claude |

## Piano consigliato
| Quando | Comando | Il tuo ruolo |
|---|---|---|
| Gio 5 sera | `/phase 0` | controlla che i dati SPX siano scaricati e il forward implicito abbia senso |
| Ven 6 | `/phase 1` … `/phase 4` (o `/build-all`) | leggi ogni `docs/validation/*.md`: se un gate è FAIL, chiedi perché |
| Sab 7 | `/phase 5`, `/phase 6`, `/phase 7` | controlla il report: ogni numero deve avere una fonte |
| Gio 5 → Sab 7, in parallelo | `docs/HANDS_ON.md` | scrivi tu il file `handson/mini_pricer.py` |
| Dom 8 – Lun 10 | `/explain src/qpricer/engines/mc.py` ecc., studia `docs/interview/` | teoria + codice chiave |
| Mar 11 | `/grill all`, poi `/grill autocallable`, `/grill heston` | mock interview |

Usa `/build-all` solo se vuoi lasciarlo andare da solo. `/phase N` uno alla volta ti dà più controllo, ed è la
"supervisione" che dichiari a colloquio: farla davvero è ciò che la rende credibile.

## Regole per te
- **Non accettare un gate FAIL "perché tanto è una tolleranza".** Chiedi a Claude la diagnosi: le domande migliori
  del colloquio nascono da lì ("cosa è andato storto e come l'hai trovato?").
- Ogni commit ha la data reale. Non riscrivere la history.
- In `report/REPORT.md` la sezione "How this was built" deve essere vera: rileggila tu.

## Cosa dire a colloquio (bozza)
> "Il progetto l'ho sviluppato con Claude Code, organizzato come una pipeline: un agente deriva la matematica,
> uno implementa, uno valida in modo indipendente contro gate numerici (parità put-call, COS vs Lewis, MC entro 3
> errori standard, limiti al modello BS). Io ho definito specifiche e test di validazione e ho verificato la
> matematica. Il core BS e Monte Carlo l'ho riscritto a mano per capirlo fino in fondo."

Dillo così solo se avrai fatto davvero `docs/HANDS_ON.md`.
