---
description: Run the full research → implement → validate → cheat-sheet loop for one roadmap phase
argument-hint: <phase number 0-7>
---
Run phase $ARGUMENTS of `docs/ROADMAP.md`. Read `CLAUDE.md` first.

1. **Research** — delegate to the `quant-researcher` subagent: derivation note(s) for this phase's topics in
   `docs/derivations/`. (Skip for phase 0 and 6.)
2. **Implement** — delegate to the `implementer` subagent with the derivation note paths and the phase gates.
3. **Validate** — delegate to the `validator` subagent. It must not see the implementer's reasoning — pass only
   the phase number and file paths.
4. **Fix loop** — if the validator reports blockers, send them back to `implementer` (max 3 rounds). Then
   anything still failing goes to `docs/OPEN_ISSUES.md`.
5. **Cheat-sheet** — delegate to `interview-coach` (Mode A) for each topic of the phase.
6. Run `pytest -q` and `ruff check src tests scripts`. Commit: `git add -A && git commit -m "phase $ARGUMENTS: <summary>"`.
7. Reply with: gate table (measured / tolerance / status), open issues, files created. Max 25 lines.
