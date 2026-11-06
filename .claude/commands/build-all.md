---
description: Build the whole project: phases 0 to 7 in order, stopping only on a blocker
---
Execute the `/phase` procedure (see `.claude/commands/phase.md`) for phases 0, 1, 2, 3, 4, 5, 6, 7 in order.

- Before starting, create a task list with one task per phase and keep it updated.
- Phase 0 requires live SPX data: run `scripts/fetch_spx.py`. If the download fails, STOP and tell the owner —
  do not substitute synthetic data for phase 5.
- Do not start phase N+1 while a **blocker** from phase N is open. Minor issues may be deferred to
  `docs/OPEN_ISSUES.md`.
- After phase 7, print: headline numbers from `report/REPORT.md`, list of open issues, and the total line count
  of `src/`.
