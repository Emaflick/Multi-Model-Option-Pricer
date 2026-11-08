# Phase 3 validation: SABR β = 1 (Hagan lognormal vol, log-Euler and conditional MC) (independent)

Validator: `validator` agent, 2026-11-07. Under test (public API only): `models/sabr.py` (`z_over_x`, `implied_vol`,
`alpha_from_atm`, `simulate`, `mc_price`). Also read: `docs/derivations/sabr.md`, `scripts/phase3_gates.py`,
`scripts/sabr_degradation.py`, `report/figures/sabr_hagan_vs_mc.{png,csv}`, `sabr_hagan_vs_mc_summary.csv`, D-022..D-024,
the Phase 3 entry of `docs/OPEN_ISSUES.md`, `tests/test_sabr.py` (to see what was and was not asserted).

Tests: `tests/validation/test_phase3_independent.py`, 35 tests: 26 fast (2.2 s), 9 `slow` (about 45 s). Every number
below is asserted by those tests or printed by `.venv/bin/python -W ignore tests/validation/test_phase3_independent.py
--seeds` (about 10 minutes; without `--seeds` about 4). New golden file: `tests/golden/sabr_prices.json` (34 cases, see
"Golden file created").

## Re-validation after fix round 1 (2026-11-08)

**Scope.** I re-checked findings F1–F5 and F7 against the current code and docs. F6 is deferred to Phase 5 by the
coordinator. I read the gate configuration from the code itself:
- `tests/test_sabr.py`: `M_GATE = 250`, and `n_steps(T)` returns 250 at every T. N = 2e5; seeds 20261107 + i.
- `scripts/phase3_gates.py`: `steps(T)` returns `M_GATE = 250`.
- `scripts/sabr_degradation.py`: the Euler overlay uses 250 steps (all 164 Euler rows of the CSV have `n_steps = 250`).
- `mc_price` now rejects a non-integer `n_steps` or `n_paths`, and its docstring has a ρ > 0 paragraph.

**Changes on my side.**
- *Mirror helpers.* `n_steps(T)` now returns 250, mirroring the code. `n_steps_old` is kept only to reprint F1's pre-fix
  numbers. `_gate_se` uses the new mirror.
- *F1 test.* It now computes V_Δ − V_Δ/2 on **common** RQMC paths (`rqmc_halving`). Two separate RQMC runs at 250 and
  500 dimensions missed the test's own precision guard (0.028 > 0.02 SE_gate at T = 0.25, k = −0.2). The criterion
  (≤ 0.1 SE_gate) and the guard (0.02 SE_gate) are both unchanged.
- *F7 check added.* `n_steps = 2.7` and `n_paths = 100.5` must raise ValueError.

**Reproduction.** Numbers below are printed by `.venv/bin/python -W ignore tests/validation/test_phase3_independent.py
--seeds` (895 s) at the post-fix mirror. The pre-fix figures in the later sections came from the same script with the
old mirror (`n_steps_old`), which reproduces them.

**Gate table (post-fix).**

| Gate (ROADMAP) | Set / method | Measured | Tolerance | Result |
|---|---|---|---|---|
| ν → 0, β = 1 → flat vol α | ν = 0 / ν = 1e-12 | 0 (bitwise) / 3.0e-12 | < 1e-10 | **PASS** |
| Hagan vs MC within 3SE, T ≤ 1, \|k\| ≤ 0.2 | implementer's estimator, declared seeds 20261107+i, Δ = T/250, set G | max \|z\| = 1.68 | 3 | **PASS** |
| | own MC (OMC), validator seeds | max \|z\| = 2.42 | 3 | **PASS** |
| | seed-free offset (E[V_(14)] − Black(σ_H))/SE_gate, exact (SCH) | max 1.145 (T = 1, k = −0.2). T = 0.25: 0.365 (pre-fix 0.843). T = 0.5: 0.640 (pre-fix 0.794) | 3 | **PASS** |
| | 100 fresh validator seeds × 21 | **5.0%** of seeds fail; 0.67% per comparison; mean z −0.025, sd 1.045 | documented ≈ 4.0% / 0.29% | consistent (binomial ±2.2%) |
| Figure where Hagan degrades | regenerated `sabr_hagan_vs_mc.{png,csv}` (Euler overlay at M = 250) | mixing-panel spot checks unchanged, ≤ 2.6 SE_csv | qualitative | **PASS** |

**Recomputed with my own RQMC on the scheme** (set G, SE_gate = the implementer's SE at N = 2e5, k = −0.2 unless stated):

| T | \|V_Δ − V_Δ/2\|/SE_gate, max over k (exact) | implementer, D-023 (CRN, one seed) | scheme bias / SE_gate | scheme bias, vol pt (exact) | D-023 Richardson, vol pt |
|---|---|---|---|---|---|
| 0.25 | 0.081 (pre-fix 0.321) | 0.080 | −0.164 (pre-fix −0.641) | −0.0107 | −0.0104 |
| 0.5 | 0.083 (pre-fix 0.160) | 0.063 | −0.169 (pre-fix −0.328) | −0.0098 | −0.0073 |
| 1 | 0.068 | 0.073 | −0.139 | −0.0081 | −0.0084 |

The slow test `test_scheme14_bias_meets_d023_criterion_at_every_gate_maturity` (common paths, 2^15 × 16) **now
passes**, with the criterion unchanged.

The fresh-seed rate predicted from the exact offsets (unit-variance z) is 0.52% per comparison, against 0.62% pre-fix.
The union bound for any of the 21 failing is 10.8%; because the comparisons are correlated, the realised rate is lower.
That rate is 5.0% on my seeds and 4.0% on the implementer's, which agree within the binomial error.

**Status of each finding.**

| Finding | Status | Evidence (current code and docs) |
|---|---|---|
| F1 Δ bias / halving criterion only checked at T = 1 | **CLOSED** | Exact halving ≤ 0.083 at every maturity (criterion 0.1); bias ≤ 0.169 SE_gate (criterion ≲ 0.2). `test_halving_dt` is parametrised over all three maturities. The reproduction test passes unchanged in strictness. *Nit, non-blocking:* D-023 (ii) and the sabr.md (18) erratum say "Richardson bias ≤ 0.16 SE_gate", but the exact bias is 0.164 (T = 0.25) and 0.169 (T = 0.5). D-023's T = 0.5 values (0.063, −0.0073 vol pt) are single-seed CRN estimates about 25% below the exact 0.083 and −0.0098. Suggest "≤ 0.17 SE_gate" and quoting the exact values. |
| F2 false-fail rate | **CLOSED** | The iid 5.5% claim is withdrawn in D-023 and sabr.md §5. The documented 4.0% matches my independent 5.0% within binomial error. *Nit:* the documented 0.29% per comparison is at the low end; the exact offsets predict 0.52%, and my seeds give 0.67%. |
| F3 gate domain | **CLOSED** (documentation) | "Domain: set G, ν²T ≤ 0.16" appears in D-023 (ii), sabr.md §5, the `test_gate_hagan_vs_mc_within_3se` docstring and the gate printout of `phase3_gates.py`. The F3 example (ν = 1, T = 0.5: +0.40 vol pt, 5.4 SE) matches my output (+0.403, 5.42 SE). The ROADMAP wording itself is unchanged (owner's file); the Phase 7 report must carry the domain. |
| F4 ρ > 0 parity | **CLOSED** | The `mc_price` docstring, the sabr.md §3.4 erratum, pitfall 8 and D-023 (v) all say parity with F and Euler calls are the model's only for ρ ≤ 0. The −0.167 they quote matches my −0.1668 ± 0.0003 (the second method gives −0.1620 ± 0.0016). |
| F5 CHK row 5 | **CLOSED** | The sabr.md erratum says ν = 0.6, two-method value 0.0781203, published off by ≈ +1.0e-5. This matches my RQMC 0.0781203 and PDE 0.0781204. |
| F6 `alpha_from_atm` ceiling / b < 0 cancellation | **OPEN — deferred to Phase 5** (coordinator) | Code unchanged; my two tests document the current behaviour and pass. |
| F7 cosmetics | **CLOSED** (actionable items) | Non-integer `n_steps`/`n_paths` now raise ValueError (asserted in my test). Summary CSV columns are renamed `*_incl_mc_noise`, with D-023 explaining that `max_abs_err_volpts_significant` is the resolved Hagan error. The \|z\| > 1e154 overflow, long-T SE tails and z_c remain as accepted cosmetics, with no action requested. |
| N1 (new nit) | open, non-blocking | sabr.md §5 still says "Wing Hagan errors are unknown". Measured: +0.058 vol pt at T = 1, k = −0.2 (D-023 cites it; the note does not). |

**Suite.** `.venv/bin/pytest -q`: **423 passed, 0 failed** (248 s; the one warning is the pre-existing Phase 1 IV
warning). `ruff check src tests scripts` and `ruff format --check src tests scripts` are clean. No test is left failing.

## Verdict (original validation, 2026-11-07, pre-fix code)

* **The three ROADMAP Phase 3 gates PASS** on the parameter set the implementer declared in advance
  (set G: α = 0.2, ρ = −0.75, ν = 0.4; T ∈ {0.25, 0.5, 1}; k ∈ {0, ±0.05, ±0.1, ±0.2}). Gate 2 also passes with my own,
  independent MC, and in its seed-free form (expected offset ≤ 1.15 SE).
* **No error found in the code.**
  - `implied_vol` equals a 60-digit decimal transcription of Hagan (2.17) to 5.0e-14 relative on 1500 random points
    covering α 1e-3..5, ν 1e-4..5, ρ ±0.999, T 1e-3..30, |k| ≤ 3.
  - `mc_price` (both methods) reproduces the exact expectation of its own discrete scheme. Over 300 seeds at coarse Δ
    the mean z is ~0 and the sd ~1.
  - `alpha_from_atm` returns the correct (small) root to 4.4e-16 wherever it matters.
* **F1: the implementer's replacement for note oracle (e) was checked at T = 1 only, and it fails at the other two
  gate maturities.** Severity: **minor** (the gate verdict is unchanged), **but must be fixed before the Phase 3
  commit.** `test_scheme14_bias_meets_d023_criterion_at_every_gate_maturity` is **left failing on purpose** as the
  reproduction.
  - The criterion is |V_Δ − V_Δ/2| ≤ 0.1 SE_gate. I compute both expectations exactly. Measured: 0.321 at T = 0.25 and
    0.160 at T = 0.5 (k = −0.2). At T = 1 I get 0.068, which reproduces the implementer's 0.073.
  - D-023 (ii) says "the Richardson bias of (14) at Δ = 1/250 is ≤ 0.0084 vol pt". That holds only at T = 1. At T = 0.25
    the bias is 0.0415 vol pt, i.e. 0.64 SE_gate.
  - One-line fix (verified): use 250 steps at every maturity (Δ = T/250).
* **F2: the gate's statistical design is optimistic.** Severity: minor.
  - Claimed: P(any false fail) ≈ 5.5%. Measured on 100 fresh seeds: **9.0%** of seeds fail. Per comparison, 0.86% fail,
    against 0.27% nominal.
  - Cause: deterministic offsets. Hagan's own wing error is +1.0 SE_gate at T = 1, k = −0.2, and the F1 scheme bias is
    −0.64 SE_gate at T = 0.25, k = −0.2.
* **F3: the gate is narrower than its ROADMAP wording.** Severity: minor for Phase 3. It **must** be stated in the
  Phase 5/7 report.
  - It holds for set G (ν²T ≤ 0.16). It **fails inside the same box (T ≤ 1, |k| ≤ 0.2)** for SPX-plausible vol-of-vol.
  - Hagan − model at k = −0.2: +0.40 vol pt (5.4 SE) at ν = 1, T = 0.5. +0.82 vol pt (10 SE) at ν = 1, T = 1, which is
    the implementer's own figure set.
  - This is the expansion error of Hagan's formula, not a code bug.
* **F4: put-call parity with F is wrong in the model for ρ > 0.** β = 1 SABR is a strict local martingale for ρ > 0. At
  α = 0.3, ρ = +0.5, ν = 1, T = 5, E F_T / F − 1 = −0.167. Two independent methods agree on this. The note (§3.4,
  pitfall 8) presents parity with F as exact. The `mc_price` docstring does not warn that, for ρ > 0, its "euler" and
  "mixing" calls have different expectations. Severity: minor (SPX has ρ < 0), but a
  likely interview question.
* **F5: CHK Table 1 row 5.** It is ν = 0.6, not 0.8. The published 0.07811 misses my two-method value 0.0781203 by
  1.0e-5, which is outside its own rounding. Rows 1–4 are reproduced within rounding (≤ 3.6e-6).
* Full suite (`.venv/bin/pytest -q`): **420 passed, 1 failed** (the F1 reproduction only). `ruff check` and `ruff
  format --check` are clean on the new file.

## Method (independent of the implementation)

| tag | reference | how it differs from the implementation |
|---|---|---|
| HAGd | Hagan (2.17a–c) at β = 1, transcribed from the paper. Evaluated in stdlib `decimal` at 60 digits on the exact binary inputs, with x(z) = ln[(√(1−2ρz+z²)+z−ρ)/(1−ρ)]. | No series, no sign split, no log1p rewrite: correctly rounded for every z, including z → 0 and z = ρ. |
| HAGa | The same formula in double precision, with x(z) = asinh((z−ρ)/s) + asinh(ρ/s), s = √(1−ρ²) (closed-form antiderivative of ∫du/D). | A different closed form. Used on wide vectorised grids with \|z\| ≥ 0.05. |
| BLK | Own Black-76 in ln F (`scipy.special.ndtr`) and own IV by `scipy.optimize.brentq`. | `bsm` and `iv` are not used for any reference. |
| RQMC | Model price from the exact conditional representation: given the α path, ln F_T ~ N(ln F + ρ(α_T−α)/ν − I/2, (1−ρ²)I). Built from **scrambled Sobol** points through a **Brownian bridge** (R independent scrambles give the error bar), with I by **Simpson** at M = 512. | The implementation uses pseudo-random incremental draws and a trapezoid I. |
| SCH | **Exact expectation of the implementer's discrete scheme (14)** at the same Δ, by RQMC. Conditional on the α path, the scheme's ln F_T is Gaussian with mean ρĴ − Î/2 and variance (1−ρ²)Î, where Î = Δ Σ α_n² and Ĵ = Σ α_n ΔW_n. | Isolates code errors from both the Δ bias and the Hagan error. |
| PDE | 2-D backward PDE in (ln F, ln α), Modified Craig–Sneyd ADI (θ = 1/3), cell-averaged payoff, Richardson over (401×161×200, 801×321×400) grids. | No mixing identity, no MC. |
| OMC | Own pseudo-random payoff MC: exact conditional sampling of ln F_T given the α path. Layout: Z⊥ first, then an (N, M) α block. Simpson I, Δ = 1/500. | Used for the literal gate with an independent MC. |

Cross-checks of the references:
* **RQMC vs PDE.** Agree to ≤ 3.4e-6 (T = 0.25), ≤ 2.9e-6 (T = 0.5) and ≤ 1.6e-5 (T = 1) on the 21 gate prices
  (F = 100). At F = K = 1 they agree to ≤ 1e-7 for ν ≤ 0.6 (7e-7 at ν = 0.8).
* **SCH vs the implementer's own CRN measurement.** At T = 1, SCH reproduces |V_Δ − V_Δ/2| = 0.068 SE_gate; the
  implementer measured 0.073 with CRN. SCH also reproduces the bias trend in Δ (first order; it halves with Δ).

## Gate table

| Gate (ROADMAP) | Set / method | Measured | Tolerance | Result |
|---|---|---|---|---|
| ν → 0, β = 1 → flat vol α | ν = 0 on \|k\| ≤ 5, T 1e-4..30, α 1e-4..5, ρ ∈ {±0.999, ±0.5, 0} | 0 (σ_H ≡ α bitwise) | < 1e-10 | **PASS** |
| | ν = 1e-12 on \|k\| ≤ 1, T ≤ 10, α ≤ 1 | 3.0e-12 | < 1e-10 | **PASS** |
| | approach: (σ_H − α)/ν at ν = 1e-7 vs my expansion ρ(k/2 + α²T/4) | < 1e-5 abs | 1e-5 | PASS |
| Hagan vs MC within 3SE, T ≤ 1, \|k\| ≤ 0.2 | implementer's estimator (14), declared seeds 20261107+i, set G, 21 puts | max \|z\| = 2.64 | 3 | **PASS** |
| | own MC (OMC), validator seeds, set G | max \|z\| = 2.42 | 3 | **PASS** |
| | seed-free: (E[V_(14)] − Black(σ_H)) / SE_gate, exact via SCH | max 1.15 (T = 1, k = −0.2) | 3 | **PASS** |
| | robustness: 100 fresh seeds × 21 | 9.0% of seeds fail; per-comparison 0.86% | claimed 5.5% / 0.27% | see F2 |
| | SPX-plausible sets in the same box (not declared by the implementer) | up to 10 SE off (ν = 1, T = 1) | 3 | **FAIL** (Hagan expansion error, F3) |
| Figure where Hagan degrades | `report/figures/sabr_hagan_vs_mc.png` + CSV | exists; long-T / wing degradation shown; CSV errors and negative-density minima re-derived independently (below) | qualitative | **PASS** |

Gate-2 decomposition on set G (my references; SE_gate = the implementer's SE at N = 2e5 for its seed; k = −0.2 … +0.2):

| T | Hagan − truth (vol pt) | Hagan − truth (SE_gate) | scheme-(14) bias (SE_gate) | offset E[MC] − Hagan (SE_gate) |
|---|---|---|---|---|
| 0.25 | +0.013 … −0.013 | +0.20 … −0.01 | −0.64 … +0.16 | −0.84, −0.51, −0.22, +0.04, +0.19, +0.19, +0.04 |
| 0.5 | +0.027 … −0.024 | +0.47 … −0.06 | −0.33 … +0.13 | −0.79, −0.42, −0.19, +0.02, +0.16, +0.22, +0.13 |
| 1 | +0.058, +0.034, +0.021, **+0.008**, −0.005, −0.018, −0.043 | +1.00 … −0.23 | −0.14 … +0.10 | −1.15, −0.62, −0.34, −0.08, +0.14, +0.28, +0.31 |

The ATM Hagan error at T = 1 (0.0082 vol pt) confirms the note's estimate of "≈ 0.008 vol pt". The wing error (0.058 vol pt
at k = −0.2), which the note listed as "unknown", is 7x larger and ≈ 1 SE_gate.

## Findings (with numbers)

**F1. Scheme-(14) Δ bias at the gate Δ: the implementer's replacement for oracle (e) fails at T = 0.25 and 0.5.** *Minor;
must be fixed before commit; reproduction test left failing.*
OPEN_ISSUES and `tests/test_sabr.py::test_halving_dt` replace note oracle (e), for scheme (14), by
|V_Δ − V_Δ/2| ≤ 0.1 SE_gate (|bias| ≲ 0.2 SE). They justify it by a CRN measurement **at T = 1 only**.

| T (M = ⌈250T⌉) | \|V_Δ − V_Δ/2\| / SE_gate, k = −0.2 … +0.2 (exact, SCH) | bias in vol pt at k = −0.2 | bias / SE_gate at k = −0.2 |
|---|---|---|---|
| 0.25 (63) | **0.321, 0.190**, 0.079, 0.022, 0.078, 0.079, 0.016 | −0.0415 | −0.64 |
| 0.5 (125) | **0.160**, 0.074, 0.022, 0.024, 0.055, 0.065, 0.035 | −0.0191 | −0.33 |
| 1 (250) | 0.068, 0.023, 0.001, 0.023, 0.039, 0.048, 0.040 | −0.0081 | −0.14 |

Consequences:
- The criterion is the right *kind* of check: the note's 3SE-of-a-CRN-difference was indeed ill-posed for a first-order
  scheme, and I confirm that diagnosis.
- But applying it only at the maturity where it passes is a loosening by omission.
- D-023 (ii) ("Richardson bias ≤ 0.0084 vol pt, ≤ 0.073 gate-SE") and the OPEN_ISSUES text are true at T = 1 only.
- The note's own step-size target, eq. (18) |δσ| ≤ 0.1 SE_σ, is missed by 6.4x at T = 0.25, k = −0.2. Eq. (18) was
  derived ATM, and the wing bias comes from the ρ-term (Ĵ − J), which the note flagged as "measure it".

*Fix (verified):* use n_steps = 250 at every gate maturity (Δ = T/250). Then the bias is ≤ 0.170 SE_gate and
|V_Δ − V_Δ/2| ≤ 0.082 SE_gate (< 0.1) at T = 0.25 and 0.5 (printed by `__main__`). The cost is negligible. Then restate D-023 (ii)
per maturity. Alternatively, gate with the conditional estimator (17): its measured bias is ≤ 0.004 SE_gate at every gate
maturity. As F2 shows, though, this does not remove the Hagan part of the offsets.

**F2. Fresh-seed robustness of gate 2.** *Minor.* 100 fresh seeds × 21 comparisons, implementer's settings:

| estimator vs reference | mean z | sd z | P(\|z\| > 3) per comparison | seeds failing the 21-test gate |
|---|---|---|---|---|
| (14) vs Hagan (the gate) | −0.083 | 1.128 | 0.0086 | **9.0%** |
| (14) vs SCH (code only) | +0.081 | 1.056 | 0.0052 | 3.0% |
| (17) vs Hagan | −0.222 | 1.066 | 0.0057 | 8.0% |
| (17) vs model truth | −0.052 | 1.000 | 0.0014 | 2.0% |

- The per-comparison mean z of the gate reaches −1.02 (T = 0.25, k = −0.2) and −0.86 (T = 1, k = −0.2), as the
  offsets in the decomposition table predict.
- Against the exact references the code behaves as pure noise:
  - 300 seeds at coarse Δ (set (0.2, −0.75, 1.0), T = 1, M = 10): mean z within ±0.02, sd 1.01–1.04.
  - 150 more seeds at the T = 1 gate settings: mean z within ±0.09, sd ≈ 0.92.
  - The +0.29 mean seen at T = 1 in the first 100 seeds did not recur: it was a fluctuation, not a code bias.
- The gate is still passed in expectation; it is just less robust than advertised.
- *Fix:* state the false-fail rate as ≈ 9% (or apply F1's Δ and re-measure). Never re-seed to pass (pitfall 12 is
  respected: the implementer's seeds were fixed a priori).

**F3. Gate scope: Hagan ≠ SABR inside the gate box once ν²T ≳ 0.25.** *Minor for Phase 3; mandatory caveat for
Phase 5/7.* Model truth by RQMC, OTM options, own IV:

| set | ν²T | Hagan − truth (vol pt), k = −0.2 … +0.2 | (Hagan − truth)/SE at N = 2e5 | gate (one seed) |
|---|---|---|---|---|
| S1 α .15, ρ −.7, ν 1.5, T 0.1 | 0.225 | +0.210, +0.125, +0.069, +0.001, −0.073, −0.116, −0.091 | up to 1.9 | FAIL (z = 3.43) |
| S4 α .2, ρ −.75, ν 1, T .25 (figure set) | 0.25 | +0.178 … −0.160 | up to 2.1 | pass (fragile) |
| S5 α .2, ρ −.9, ν .6, T 1 | 0.36 | +0.095 … −0.093 | up to 1.4 | pass |
| S2 α .18, ρ −.7, ν 1, T .5 | 0.5 | +0.403, +0.230, +0.135, … | 5.4 | **FAIL** (z = −6.5) |
| S3 α .2, ρ −.75, ν 1, T 1 (figure set) | 1.0 | +0.821, +0.508, +0.340, +0.161, … | 10.1 | **FAIL** (z = −9.7) |

- The note's pitfall 5 ("gate only where ν²T ≲ 0.2") anticipates this. But the ROADMAP gate text and D-023 do not
  carry the restriction.
- Short-dated SPX smiles fitted with β = 1 typically sit at ν²T ≈ 0.2–0.7. That is exactly where Hagan's in-box error is
  0.1–0.4 vol pt and resolvable by MC.
- So any statement in Phase 5/7 that the calibrated "SABR model" reproduces the smile must refer to the *Hagan formula*
  as a smile parametrisation, not to the SABR dynamics (e.g. for pricing exotics by SABR MC).
- *Fix:* in the report's validation table, write the gate as "set G, ν²T ≤ 0.16". Cite F3 or the figure for larger ν²T.

**F4. ρ > 0: loss of martingality ([LM07]); parity with F is not the model's.** *Minor (SPX ρ < 0); note/docstring
correction.*

| α = 0.3, ν = 1, T = 5 | RQMC: C − P − (F − K) = F(Eξ − 1) | share-measure explosion probability (Euler on ln α, independent) |
|---|---|---|
| ρ = +0.5 | **−0.1668 ± 0.0003** (F = 1) | −0.1620 ± 0.0016 |
| ρ = −0.5 | +0.0000 ± 0.0001 | 0.0000 |

- The log-Euler scheme (14) is an exact martingale for every Δ and ρ. So its call, or "calls by parity with F" (sabr.md
  §3.4 "Parity Ĉ = P̂ + DF(F − K) is unbiased", pitfall 8), targets P_model + DF(F − K), not C_model, when ρ > 0.
- `mc_price(method="mixing")` prices the model call directly. The two methods of the same function therefore have
  different expectations for ρ > 0 calls.
- In practice the Euler sample mean of F̂_T lands below F, because the missing mass sits in paths that are never
  sampled. So the error shows up as an unreliable SE rather than a visible bias.
- *Fix:* restrict "parity with F" and raw Euler calls to ρ ≤ 0 in the note (§3.4, pitfall 8), and say so in the
  `mc_price` docstring. Add the interview line: "for ρ > 0, β = 1 SABR calls ≠ puts + F − K; implied vols from calls and puts differ".

**F5. Published [CHK] Table 1 values.** *Minor (oracle provenance).*

| ρ, ν | RQMC (F = K = 1) | PDE | FDM published | RQMC − FDM | Hagan − RQMC |
|---|---|---|---|---|---|
| −0.75, 0.2 | 0.0790990 | 0.0790989 | 0.07910 | −1.0e-6 | +2.6e-6 |
| −0.50, 0.2 | 0.0794164 | 0.0794163 | 0.07942 | −3.6e-6 | +7.7e-6 |
| −0.25, 0.2 | 0.0796883 | 0.0796882 | 0.07969 | −1.7e-6 | +8.8e-6 |
| −0.75, 0.4 | 0.0785977 | 0.0785976 | 0.07860 | −2.3e-6 | +3.2e-5 |
| −0.75, **0.6** | 0.0781203 | 0.0781204 | 0.07811 | **+1.0e-5** | +1.2e-4 |
| −0.75, 0.8 | 0.0776285 | 0.0776292 | 0.07811 | −4.8e-4 | +3.1e-4 |

- The "UNVERIFIED" row resolves to ν = 0.6 (the table, not the text).
- Its published value is 2x outside its 5-decimal rounding against two independent methods that agree to 1e-7.
- *Fix (note owner):* mark row 5 as ν = 0.6 with a published error of ≈ 1e-5.
- The note's hand evaluations of Hagan (0.079101, 0.079424, 0.079698, 0.078630) are confirmed.

**F6. `alpha_from_atm`.** *Minor; Phase 5 advisory.*
- Round trip σ_ATM(α̂) = s:
  - 4.4e-16 wherever b > 0, which includes every ρ ≤ 0.
  - Where b < 0 < a (ρ > 0.82 and ν²T ≳ 25), the "stable" form 2s/(b + √(b² + 4as)) cancels. The error follows my
    model 8ε(1 + b²/(2as)) (asserted): 4.6e-14 on the report sample, against the 1e-14 claimed in D-024. Cosmetic fix:
    use (−b + √disc)/(2a) when b < 0.
- Root choice is correct: it is the smaller root (bracket ≥ b/2), and matches my own brentq on [0, vertex] to < 1e-13.
- **Feasibility ceiling.** For ρ < 0, Hagan's σ_ATM(α) = α(b + aα) has a maximum b²/(4|a|). Above it the function
  returns NaN, with a warning, exactly when no positive root exists (asserted on 20 000 points).
  - Example: ρ = −0.9, ν = 2, T = 5 gives a ceiling of 4.57%. ρ = −0.9, ν = 5, T = 2 (inside the eq. (11) bounds) gives
    ≈ 0.12%.
  - The Phase 5 σ_ATM parametrisation will therefore produce NaN residuals in part of the TRF box. The loss must map
    NaN to a penalty, or ν must be bounded per expiry. It must not silently drop points.

**F7. Cosmetic / documentation.**
- `mc_price(n_steps=2.7)` silently runs 2 steps (`int()` truncation). Require an integer.
- |z| > ~1e154 (α < ~1e-150) overflows inside `z_over_x`. The result is NaN plus a numpy overflow warning rather than
  the module's own message. Irrelevant in practice.
- `sabr_hagan_vs_mc_summary.csv`, column `max_abs_err_volpts_gate_box`, reports 0.129 vol pt for ν = 0.4, T = 1. That
  number is the MC noise of one seed (the panel sits 2.6 SE off; the true Hagan error there is 0.008). The report should
  quote the `_significant` column or the golden truth, not this column.
- At ν²T ≥ 5 (figure panels T = 5, 10 at ν = 1) the integrated variance is extremely heavy-tailed. Log-sd of α_T² is
  2ν√T ≈ 4.5. In an exploratory run (not in the script) the conditional-MC SE looked unreliable there. The error bars of
  those panels may understate the uncertainty. This is immaterial to the figure's message, because the errors there are
  1.7–21 vol pt.
- The cubic series branch of `z_over_x` is the least accurate part (4.7e-14 at z_c). The log1p branch is accurate to
  8.9e-16 right down to z_c, so a smaller z_c (e.g. 1e-5) would make the series error negligible. Not needed.

**F8. Validator-side note (my reference, not the code).** With R = 8 scrambles × 2^13 points, the scramble-to-scramble
spread of RQMC for a deep-OTM put (k = −0.2, T = 0.25) is skewed. Its 7-dof standard error once understated the true error
about 4x. The golden self-check therefore uses R = 16 × 2^14. The golden values themselves (R = 16 × 2^16) are certified
by the independent PDE. In every MC-vs-reference test the reference error is ≥ 50x below the MC SE, so this does not
affect any other verdict.

## Edge cases probed (all PASS unless listed above)

| probe | result |
|---|---|
| z → 0 and the switch \|z\| = z_c (±1 ulp, ±1e-9 rel) | vs decimal: series branch ≤ 4.7e-14, log1p branch ≤ 8.9e-16; no jump beyond that |
| ρ = ±0.999, z = ρ ± 1e-12 rel | ≤ 1e-14 rel (log1p branch) |
| ν large (up to 5), T up to 30, \|k\| up to 10, α 1e-4..5 | finite, positive, warning-free wherever my own 1 + c₁T > 0. NaN plus exactly one RuntimeWarning wherever it is ≤ 0 (eq. 22), and NaN only there |
| input validation | ρ = ±1, \|ρ\| > 0.999, NaN/inf, α ≤ 0, ν < 0, F/K/T ≤ 0 raise ValueError; ρ = ±0.999 is accepted |
| shapes | (3,1,1)×(1,4,1)×(1,1,5) broadcast equals scalar calls bitwise; scalar in gives 0-d out |
| ATM | σ_H(F, F) = α(1 + c₁T) to 4ε |
| `simulate` | equals my hand recursion of eqs. (13)–(14) on 2 given steps (rel 1e-14) |
| `mc_price` contract | returns (price, stderr, n_paths: int, seed: int); 0-d for scalar K, K's shape otherwise ((2,3) with cp broadcast); bitwise reproducible; different seed gives different prices; strikes share paths (a single-strike call equals the vector element bitwise); DF scales price and SE exactly; (λF, λK) → λV to 1e-12; `seed=None` gives TypeError (no hidden global RNG); bad method/n_paths/n_steps/cp give ValueError |
| SE ∝ N^(−1/2) | SE(4N)/SE(N) = 0.5 ± 0.05 at all 7 strikes |
| ν = 0 MC | both methods equal Black(α) within 3 SE (ρ = −0.999, DF = 0.9, 3 steps) |
| coarse Δ, extreme sets (ρ = ±0.999, ν 1–2) | (14) equals SCH within 3 SE, while SCH − model > 3 SE (power check); (17) equals its trapezoid expectation within 3 SE |
| figure CSV | Max-error strike and ATM of every mixing panel vs my truth: \|csv − mine\| ≤ 3(SE_csv + SE_mine). All 8 panels are printed by `__main__`; the slow test asserts the 6 panels with T ≥ 1. Largest deviation 2.6 SE_csv (ν = 0.4, T = 1, the seed noted in F7). My Breeden–Litzenberger density of Hagan reproduces the CSV's negative minima (−6.14e-2 at ν = 1, T = 10; −7.95e-3 at ν = 1, T = 5). |

## Assessment of the OPEN_ISSUES Phase 3 replacements (as requested)

1. **Oracle (ii), K = F continuity.** The second-order bound is mathematically correct. The note's first-order bound
   misses exactly ½∂²_kσ·h² = ½ × 0.041127 × 1e-8 = 2.06e-10, with ∂²_kσ from eq. (8). This is not a loosening. It is
   also superseded: σ_H matches the 60-digit decimal reference to ≤ 5e-14 relative at every z, including the K = F
   neighbourhood.
2. **Oracle (e), halving for (14).**
   - The diagnosis is correct: CRN makes the SE of V_Δ − V_Δ/2 far smaller than the O(Δ) bias. My exact SCH confirms
     the bias is first order and reproduces the implementer's 0.073 at T = 1.
   - The replacement criterion is sound in form. But it was **evaluated at T = 1 only**, where it passes by a 1.4x
     margin (0.073 vs 0.1). At T = 0.25 and 0.5 it fails (0.32, 0.16).
   - It is also 2x looser than the note's own design target, eq. (18) (bias ≤ 0.1 SE rather than ≤ 0.2 SE).
   - Verdict: **a loosening by selective application, not by intent** (F1). The fix is cheap and verified.

## Recommended fixes (owner: implementer unless stated)

1. (F1) Use n_steps = 250 at every gate maturity in `tests/test_sabr.py`, `scripts/phase3_gates.py` and (for the Euler
   overlay) the figure. Extend `test_halving_dt` to all three maturities. Restate D-023 (ii) and OPEN_ISSUES per
   maturity. Re-run the gate on the declared seeds.
2. (F2, F3) In D-023 / sabr.md §5 / the report, state the measured fresh-seed false-fail rate (≈ 9%). State the gate
   domain as "set G, ν²T ≤ 0.16", and add the F3 table, or a pointer to it, next to the figure.
3. (F4) Note §3.4 and pitfall 8, plus the `mc_price` docstring: "parity with F is the model's only for ρ ≤ 0".
4. (F5, note owner) CHK row 5: ν = 0.6, published value off by ≈ 1e-5.
5. (F6) Phase 5 loss: handle the NaN that `alpha_from_atm` returns above the ceiling b²/(4|a|). Optional: use the
   b < 0 root form.
6. (F7) Optional cosmetics: integer check on n_steps; quote the `_significant` summary column.

## Golden file created

`tests/golden/sabr_prices.json` contains 34 European puts (DF = 1). Each case stores the RQMC value, its SE, the PDE
value and a tolerance tol = 3 se + |RQMC − PDE| (rounded up, minimum 1e-7 F). The cases are:
- Set G: T ∈ {0.25, 0.5, 1} × 7 strikes, tol 2.9e-5 … 6.1e-5 (F = 100).
- NU1 (α 0.2, ρ −0.75, ν 1, T 1; k ∈ {−0.2, 0, 0.2}): tol up to 3.1e-3, because the PDE is less converged at ν = 1.
- SPX_SHORT (α 0.15, ρ −0.7, ν 1.5, T 0.1; 5 strikes): tol ≤ 4.5e-5.
- CHK: the 5 rows of [CHK] Table 1 at F = K = 1, with the published FDM value stored alongside (tol ≤ 4.3e-7).

Source: agreement of two independent methods (RQMC conditional representation; MCS-ADI PDE), neither using qpricer
code, as stated in the file. It was generated by `--golden-out` into a scratch path and copied verbatim (the guard hook
blocks shell writes into `tests/golden/`). The tests `test_golden_file_two_methods_agree` and `test_impl_mixing_vs_golden`
use it.
