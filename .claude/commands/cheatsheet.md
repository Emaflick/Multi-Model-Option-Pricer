---
description: Write or refresh the interview cheat-sheet for a topic
argument-hint: <topic, e.g. bsm | mc | heston | sabr | barrier | asian | autocallable | calibration>
---
Delegate to the `interview-coach` subagent, Mode A, for topic: $ARGUMENTS. Output `docs/interview/$ARGUMENTS.md`.
If all topic cheat-sheets exist, also (re)build `docs/interview/00_MASTER.md` with the 40 most likely questions
across topics, each with a 2–3 line answer and a link to the detailed sheet.
