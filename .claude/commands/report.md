---
description: Regenerate figures and the final report from measured outputs
---
1. Run `python scripts/run_calibration.py`, `python scripts/run_benchmarks.py`, `python scripts/make_figures.py`
   (skip any whose outputs are newer than the code they depend on).
2. Delegate to the `reporter` subagent to write `report/REPORT.md` (and PDF via pandoc if installed).
3. Check every number in the report against its source file; fix any mismatch. List the headline numbers.
