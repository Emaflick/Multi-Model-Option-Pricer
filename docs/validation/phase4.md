# Phase 4 validation: exotics (barrier, Asian, mono-asset autocallable) (independent)

Validator: `validator` agent, 2026-11-08/09. Under test (public API only): `engines/analytic.py`, `engines/exotic_mc.py`,
`products.py`, `market.forward_curve`, `scripts/autocall_compare.py`, and the three figures with their CSVs. Also read:
`docs/derivations/{barrier,asian,autocallable}.md` (incl. the §8 errata), ROADMAP Phase 4, D-025…D-031, the Phase 4
OPEN_ISSUES entry, `tests/test_{barrier,asian,autocallable}.py`, `scripts/phase4_gates.py`, `scripts/exotics_figures.py`.

Tests: `tests/validation/test_phase4_independent.py`, 36 tests: 25 fast (about 15 s), 11 `slow` (about 3 min). One of them
fails **on purpose**: it reproduces finding F2. New golden file: `tests/golden/barrier_haug.json`.

Reproduction:
- `.venv/bin/python tests/validation/test_phase4_independent.py` prints every number below (33 s).
- Add `--scan` for the fresh-seed z distributions (19 min).
- `--golden-out PATH` regenerates the golden file (it never overwrites).

No reference uses qpricer code. The methods:
- **RRS:** Reiner–Rubinstein in Haug's spot form, my own transcription.
- **PDE:** Crank–Nicolson with a Rannacher start on y = η ln(S/H), plus one Richardson step. No reflection principle.
- **DQ:** discrete-monitoring barrier by backward Gauss–Legendre quadrature on splines.
- **GEO:** discrete geometric Asian from the covariance double sum.
- **KV:** Kemna–Vorst as printed in Haug §4.20.1. The KV paper itself was not accessible.
- **ARQ:** arithmetic Asian by density recursion of ln Y_k, Y_k = R_k(1 + Y_{k+1}). It matches 1-D quad at n = 2 to 1e-11 and is grid-converged to 7e-11.
- **ACX:** exact autocallable under deterministic variance, by backward induction with the memory state.
- **LOOP:** the term sheet transcribed path by path.
- **GP:** my own Heston CF with Gil-Pelaez inversion. No COS.
- **OQEM:** my own QE-M, with a different RNG layout.

## Re-validation after fix round 1 (2026-11-09)

**Scope.** I re-checked F1–F8 and F11 directly against the current code and docs, using my own references (not the
implementer's descriptions). F9 and F10 were skipped by the coordinator.

Files re-read:
- `engines/exotic_mc.py`, `products.py`;
- `tests/test_{barrier,asian,autocallable}.py`, `scripts/phase4_gates.py`, `scripts/autocall_compare.py`;
- D-025…D-031, the OPEN_ISSUES Phase 4 entries, the errata of barrier.md and autocallable.md, BUILD_STATUS;
- the regenerated `autocall_attribution_illustrative.{png,csv}` and `gate_outputs/phase4_gates.txt`.

**Changes on my side** (`tests/validation/test_phase4_independent.py`, now 38 tests: 26 fast, 12 slow):
- The F2 defect test became `test_f2_heston_asian_default_grid_unbiased`. It is the same comparison on the same seeds,
  plus the library default against my own QE-M at Δ = 1/64.
- New `test_vector_coupon_schedule_vs_own_references` (fast) and `test_deng_case2_qpricer_mc_vs_exact` (slow) check
  the new per-period-coupon path.
- The report script now prints the library default, which it previously did not: it passed `max_dt=None` explicitly.

No criterion was changed.

**No loosening (item 1).** I compared every numeric tolerance in the three test files and in `phase4_gates.py` with
the round-0 versions and with the notes. Only two changed:
- The Haug table is now split per cell. 66 correctly rounded cells are held at 5e-5 (max 4.96e-5 at σ = 0.25, 4.59e-5
  at σ = 0.30). The six E1 cells are held against the golden file at 1e-10 (measured 3.5e-14), and all 72 cells against
  it at 1e-10 (4.8e-14).
- Branch continuity at K = H went from 1e-11 to 1e-12 (± 1 ulp), i.e. tighter.

All other tolerances are as in the notes: 3 SE, 3 SE + 1e-3, 5e-4, 3e-4, 1e-12. Nothing is skipped or marked xfail.

**Status of each finding.**

| Finding | Status | Evidence (current code and docs, own references) |
|---|---|---|
| F1 Haug σ = 0.30 column at 1e-4 | **CLOSED** | `test_rr_haug_table` and the gate script read the golden flags. Results: 66 cells ≤ 5e-5, E1 cells 3.5e-14 vs golden. Erratum E1 and D-027 are rewritten, and the 1e-4 version is explicitly withdrawn. My per-cell test passes. |
| F2 Heston Asian default grid | **CLOSED** | `MAX_DT = 1/32` is now the default in `obs_steps`, `fixings`, `asian_averages` and `asian_mc`, as in `autocall_mc`. With 2e6 paths, default vs 1/32: z = −1.08 / −1.25 / −1.14. My regression test passes (3e6 paths, plus my own QE-M at 1/64). An explicit `max_dt=None` still gives z = +5.36 at K = 110, but it is now opt-in and documented (D-030). |
| F2 corollary: `barrier_mc` passes `None` | **justified** | Its uniform `n_steps` grid *is* the simulation grid. The bridge (9) uses each step's left-point variance with Δ = T/n_steps; hidden sub-steps would pair a coarse-node variance with that Δ. Under BSM the steps are exact for any M; under Heston, Δ is set by `n_steps` (D-028, gate at 1/128). Output is unchanged bit for bit: G2m seed 20261148 still gives +0.01826, G2h still 2.97. |
| F3 single-seed bias table | **CLOSED** | D-028 and `phase4_gates.py` quote the 25-seed pooled +0.095 / +0.055 / +0.020 / +0.012 and say that G2m passes because 0.020 is below the gate SE. |
| F4 G2h seed (info) | **CLOSED** | D-028 cites the 300-seed scan and withdraws the 20-seed "pui 0.36". |
| F5 deferred gate | **CLOSED** | `phase4_gates.py` prints `ROADMAP: autocall BSM (ATM) vs CALIBRATED Heston … DEFERRED (Phase 5)` and the summary "24 PASS, 0 FAIL, 1 DEFERRED". The same appears in `gate_outputs/phase4_gates.txt` and BUILD_STATUS. Every GS line is labelled "(illustrative)" or "(illustrative GS)"; the LABEL line, CSV `label` column (both rows) and figure suptitle carry "illustrative, pre-calibration". |
| F6 "term structure" label | **CLOSED** | Relabelled "flat σ_ATM vs Heston expected-variance curve (level + term structure)" in the code, CSV column, figure and D-031; autocallable.md E3 added. With v0 = θ the script prints the pure level gap (√θ 0.1881 vs σ_ATM 0.1687). Values are unchanged and still match my exact term1 within 0.7 CRN SE (0.59 AC-Q, 0.68 AC-A). |
| F7a (19) upper bounds | **CLOSED** | `test_eq19_bounds_pathwise_and_in_expectation` covers the pathwise and closed-form bounds; my test passes too. |
| F7b Deng et al. | **CLOSED** (code); oracle stays an OPEN_ISSUE | `Autocall.c` accepts n per-period coupons, and the scalar path is unchanged (c τ_i). Checks: flows vs LOOP with random non-proportional schedules and ties are bit-equal; eq. (11)/(12) with vector coupons vs ACX ≤ 1.1e-10; Deng sheet `autocall_mc` vs ACX 97.5068 over 10 validator seeds × 4e5 paths, \|z\| ≤ 1.51 (gate script: 97.508 ± 0.009, z +0.09); first-call p_i vs Table 1 max \|z\| 2.11. Published 98.39 recorded as not reproduced (OPEN_ISSUES, nothing tuned). `test_deng_first_call_probability_p1` now calls `ratio_marginals`. |
| F7c CRN σ-grid | **CLOSED** | `autocall_compare.sigma_grid` prints the BS full-product σ-grid with per-step CRN SEs (diagnostic). |
| F7d H2 vs H1 | unchanged (disclosed in D-030) | — |
| F8 continuity 1e-11 | **CLOSED** | `np.nextafter` at 1e-12. |
| F9 CV raises on an all-zero control | **OPEN** (nit, skipped by coordinator) | Documented behaviour; unchanged. |
| F10 size | **OPEN** (minor, skipped by coordinator) | `wc -l src`: **1900** (target ≲ 1500; +9 lines this round). |
| F11 extrapolation wording | **CLOSED** | autocallable.md E2. |

**New nits (non-blocking).**
- **N1:** `autocall_compare._stats` builds b by `replace(sheet, c=1.0)`. For a vector-coupon sheet, eq. (17) would
  therefore return the par *simple rate* on c τ_i, not a multiplier on the given schedule. Only scalar sheets reach it
  today. Suggest raising, or scaling the schedule.
- **N2:** `phase4_gates.py` hard-codes two validator numbers as citations (pooled G2m biases; ACX 97.507). They are
  reproducible with `test_phase4_independent.py --scan` and the report script.

**Gate status after fix round 1.** Measured values are unchanged, since no engine path used by a gate changed:
- G1–G4 and the CV report: **PASS**, independent figures as in the gate table below.
- Autocallable vs calibrated Heston: **DEFERRED to Phase 5**, labelled as such.

**Suite.** See the end of this file.

## Gate table (ROADMAP Phase 4)

| Gate | Implementer (declared seeds) | Validator (independent reference, fresh seeds) | Tolerance | Result |
|---|---|---|---|---|
| G1 in + out = vanilla, same paths | 3.55e-15 (8 types, BSM + Heston QE-M, bridge + discrete) | 1.07e-14 on new configs: Euler and QE schemes, b < 0, DF > 1, M = 7 / 15, monitor = 5, K = H | < 1e-12 | **PASS** |
| G2 BB-MC vs RR, BSM | max \|z\| 1.99 (56 cells, 8 zero cells exactly 0) | vs RRS, 50 validator seeds × 56 = 2800 z: mean +0.026, sd 0.998, max 3.63, 0.36% beyond 3 (0.27% expected); zero cells exactly 0 | 3 SE | **PASS** |
| G2h Heston ξ = 1e-4, ρ = 0 vs RR(√θ) | max \|z\| 2.97 (seed 20261138) | Declared seed reproduced: 2.97, with the same-path vanilla at −3.15 / +3.13. 300 validator seeds: per-cell mean z −0.031…+0.011, sd 0.96–1.07, 0.25% per comparison beyond 3. 1.0% of seeds have max \|z\| > 3 and 1.3% have ≥ 2.97. corr(z_cdo, z_vanilla) = 0.905 | 3 SE | **PASS** (borderline seed; unbiased in distribution) |
| G2m model-free cdo(K = H = 90) = 10 DF, FO, v⁺ bridge | z +0.75 at Δ = 1/128 | Pooled bias, 25 seeds per Δ: +0.0947 (1/32), +0.0545 (1/64), **+0.0201 ± 0.0049 (1/128)**, +0.0118 ± 0.0049 (1/256). Single-run 3 SE failures: 88% / 16% / 0% / 0% | 3 SE at the finest Δ | **PASS** at 1/128 (the bias is real: 0.8 gate-SE; F3) |
| G3 (i) RR and BGK-shifted RR vs BGK tables | 4.97e-4 | BGK-shifted RR = RRS at the shifted barrier to 1e-11 | 5e-4 | **PASS** |
| G3 (ii) discrete MC vs BGK "True" | max \|d\|/(3SE + 1e-3) = 0.54 | DQ reproduces all 13 "True" values to ≤ 6.3e-4 (BGK accuracy ≈ 1e-3). Discrete MC vs DQ: max \|z\| 1.90 (26 runs) | 3 SE + 1e-3 | **PASS** |
| G4 geometric-Asian MC vs Kemna–Vorst | max \|z\| 1.90 (39 cells) | Closed form = GEO to 1e-12 on 20 random date sets and curves (t = 0 fixings and puts included). MC vs exact on 5 cases × 3 K × 2 seeds: geometric max \|z\| 1.88, plain arithmetic 2.18, CV 2.05. No +3e-4 allowance | 3 SE | **PASS** |
| CV variance-reduction factor reported | FM08: VR 1924 / 1608 / 612 (n = 12). ATM n = 12: 5793 / 1416 / 632 / 211 (σ = 0.1 / 0.2 / 0.3 / 0.5) | CSV: VR = (se_plain/se_cv)² exactly and VR = 1/(1 − ρ̂²) to 1e-6. Own VR on own paths within 15% at 9 (σ, K/F) points. CSV CV prices within 3 se_cv of ARQ | report | **PASS** |
| Autocallable: BSM (ATM vol) vs **calibrated** Heston, report and explain | Illustrative GS set: AC-Q −0.01436 (CRN SE 0.00017), AC-A −0.02464 (0.00023) | σ_ATM 0.168652 (GP agrees to 1e-7). AC-Q exact values: V_BS 1.016654 (MC 1.016721 ± 0.000065); V_TS 1.014543; term1 −0.002110 (MC −0.002146 ± 0.000061); term2 −0.015971 (GP, < 1e-6). Own QE-M V_H 1.002512 ± 0.000084 (impl 1.002359 ± 0.000188), so gap −0.01414. AC-A gap −0.02462 | report | **Mechanism PASS. Gate as worded DEFERRED to Phase 5** (no calibrated Heston yet; F5) |

**Oracle rows (notes §5).**

| Oracle | Measured | Tolerance | Result |
|---|---|---|---|
| Haug table σ = 0.25 (2 sources) | 4.96e-5 | 5e-5 (note) | PASS |
| Haug σ = 0.30, implementer's test | 6.25e-5 | **1e-4** (note: 5e-5) | passes a loosened test (F1) |
| Haug σ = 0.30, per cell (validator) | 30 correctly rounded cells: max 4.59e-5 against 5e-5. 6 E1 cells: equal to golden (RRS = PDE) to < 1e-10 | 5e-5 / 1e-10 | PASS |
| E1 claim: six entries mis-rounded | PDE = RRS = RR to ≤ 5.5e-10; e.g. puo(105, 90) = 4.22923747 against printed 4.2293. No small input perturbation (σ, T, r, b) fits the column better (rms 3.8e-5 either way) | — | **E1 confirmed** |
| RR vs RRS, wide grid | 16 cells × b ∈ {−0.05, 0, 0.04, 0.1} × σ ∈ {0.05, 0.25, 0.8} × T ∈ {0.01, 0.5, 3} × DF × R | 1e-11 | PASS |
| RR vs PDE, extreme inputs | ≤ 4.4e-9. Cases: σ = 0.03 with b = ±0.05, T = 0.005, H 0.1% from spot, K = 60 / 200 | 2e-8 | PASS |
| Branch continuity at K = H (note: 1e-12) | < 1e-12 at K = H ± 1 ulp (implementer tests 1e-11) | 1e-12 | PASS (F8) |
| E2 slope | exact discrete prices (DQ): fitted slope 0.4295; local slopes 0.364 → 0.478, rising. Figure CSV: uncorrected MC within 1.54 SE of DQ; bridge MC within 1.43 SE of RRS | — | **E2 confirmed** |
| [FM08] Table 4, geometric | 1.28e-4. One unrounded σ = 0.1780163 fits all 12 values to 3.6e-6 | 3e-4 | PASS; tolerance justified |
| [FM08] Table 6, arithmetic | table − ARQ at stated inputs: +5.4e-5 … +1.73e-4. At the fitted σ: ≤ 9e-6 (n = 12), 1.9e-5 (n = 50), 5.4e-5 (n = 250) | 3 SE + 3e-4 | PASS; tolerance justified |
| Autocallable eq. (12) and (11) vs ACX | ≤ 1.5e-10 (n = 2, both memory flags; no-call Black and term structure) | 1e-9 | PASS |
| Heston marginals: COS vs GP | put ≤ 3.5e-15, digital ≤ 3.9e-8; no-call Heston value < 1e-7 | 1e-10 / 1e-6 | PASS |
| Full autocallable (BS) MC vs ACX | 18 runs (AC-Q-like and AC-A-like, σ = 0.15 / 0.25 / 0.4): max \|z\| 2.36 | 4 (3 SE each) | PASS |
| Deng et al. (2011) | Case 1: ACX 99.976, "≈ par" as the note expects. Case 2: ACX 97.507 (own MC 97.509 ± 0.062) against published 98.39. First-call p_i match Table 1 within 4 SE | not gated (note) | **published V not reproduced** (F7) |

## Findings

**Blockers: none.** The suite is red only because of the test I left failing for F2.

1. **F1 — minor (tolerance loosening; fix before the phase commit).** Erratum E1 is correct on substance: two
   independent methods (PDE and RRS, agreeing to ≤ 5.5e-10) confirm the six σ = 0.30 cells are not correctly rounded. The
   "correction", however, moves the **whole** σ = 0.30 column from 5e-5 to 1e-4. That loosens 30 cells that pass at 5e-5
   (max 4.59e-5). No error is hidden today, but it is a tolerance change the hard rules forbid when a strictly tighter
   treatment exists. That treatment is the Phase 2 precedent (AMST): reclassify the bad entries and keep the good ones
   strict.
   - **Fix:** test the 66 correctly rounded cells at 5e-5, and the six E1 cells against `tests/golden/barrier_haug.json`
     at 1e-10. This is `test_haug_table_per_cell_without_loosening`, which passes.
   - **Also update:** E1's "Correction", D-027, and the gate line "≤ 1e-4 (note: 5e-5)" in `scripts/phase4_gates.py`.
2. **F2 — minor DEFECT (failing test left).** Under Heston, `asian_mc` and `asian_averages` default to
   `max_dt=None`. That means one QE-M step per fixing period, which contradicts asian.md eq. (4).
   - **Measured:** quarterly fixings, GS, 3e6 paths, independent seeds. The default minus `max_dt=1/32` gives
     z = −2.6 / −3.6 / **+5.9** at K = 90 / 100 / 110 (+0.0133, or 1.3%, at K = 110).
   - **Cross-check:** my OQEM at Δ = 1/64 agrees with 1/32 (|z| ≤ 1.7) and rejects the default.
   - **Test:** `test_defect_heston_asian_default_grid_is_biased`.
   - **Fix:** default `max_dt=1/32`, as in `autocall_mc`, or raise under Heston when `max_dt is None`.
3. **F3 — minor (documentation of a real bias).** D-028's bias-vs-Δ sequence (+0.159, +0.044, +0.048, +0.018) is
   single-seed, and it understates the v⁺-bridge bias: the 25-seed pooled values are +0.095, +0.055, +0.020, +0.012 at
   Δ = 1/32, 1/64, 1/128, 1/256. The bias is O(Δ) and still resolved at the gate step (4.1 pooled SE). G2m passes at 1/128
   because the bias (0.020) is below the gate's resolution (SE 0.024), not because it is zero. At 1/64, 16% of seeds would
   fail. **Fix:** quote the pooled numbers in D-028 and the report.
4. **F4 — info.** The G2h declared seed is a 1-in-75 draw (1.3% of seeds give ≥ 2.97). Its cause is confirmed: the barrier z
   tracks the path set's vanilla error (corr 0.905). No bias over 300 seeds. D-028's "pui mean 0.36" over 20 seeds is
   noise; the 300-seed value is −0.031.
5. **F5 — minor (gate status).** The ROADMAP gate needs a **calibrated** Heston, which arrives in Phase 5.
   `scripts/phase4_gates.py` prints "PASS (report, illustrative)". BUILD_STATUS and the commit message should record this
   gate as **deferred**, not passed. The OPEN_ISSUES entry exists and is accurate.
6. **F6 — minor (interpretation; interview-relevant).** For GS, v0 = θ, so w(t) = θt is flat at 18.82%. Eq. (19)'s
   "ATM term structure" term (−0.0021 AC-Q, −0.0082 AC-A) is therefore **not** a term-structure effect. It is the level gap
   between √θ and σ_ATM = 16.87%, i.e. the vol-of-vol convexity of ATM implied vol. **Fix:** either relabel the term
   (figure and D-031), or rescale w so the TS model matches σ_ATM at T, so that term1 isolates the shape.
7. **F7 — minor (oracles not exercised).** None of these is a loosening, but each note oracle below was dropped without
   a D-entry, except (d).
   - (a) asian §5: the pathwise **upper** forms of (19) are not tested (only Y ≥ X). My test covers them and passes.
   - (b) autocallable §5, Deng et al.: neither V = 98.39 nor the p_i comparison is run. `test_deng_p1_by_hand` calls no
     qpricer code, and `Autocall` cannot represent Deng's coupon schedule (only c_i = cτ_i). My exact pricer reproduces
     Case 1 and the Table 1 p_i but **not** Case 2's 98.39 (97.507). Record this in OPEN_ISSUES, as was done for AMST.
   - (c) autocallable §5: the CRN σ-grid diagnostic for the full product is absent.
   - (d) asian §5 Heston (iii), H2 vs H1: dropped, but disclosed in D-030.
8. **F8 — nit.** The branch-continuity test uses 1e-11 at K = H(1 ± 1e-13), where the K shift alone is about 2e-11. At ±1 ulp
   the note's 1e-12 holds; use `np.nextafter`.
9. **F9 — nit (robustness).** `asian_mc(kind="cv")` raises for the whole strike vector if one strike has an all-zero
   control (e.g. K = [100, 400], σ = 0.2). `kind="arithmetic"` returns 0 for that strike. This behaviour is documented.
   Suggest falling back to ĉ = 0 per strike.
10. **F10 — minor (size).** `src/` has 1891 lines against the ≲ 1500 target (+26%). Phase 4 added 531 of them: products 76,
    analytic 208, exotic_mc 225, market +22. Report to the owner.
11. **F11 — nit.** autocallable.md §1 says "constant zero rate and carry" beyond T_max. D-025 implements a flat forward
    rate anchored at T_max, which is continuous and better. Add a one-line erratum to the note.

## Verified with no issue

- **Autocallable cash flows** (memory, "≥" tests, each flow discounted at its own date, European protection at T only):
  bit-equal (≤ 1e-15) to LOOP on 20,000 paths × 7 sheet variants. The variants include exact ties at every barrier,
  B_c = B_ac, B_p > B_c (autocallable E1) and B_ac ∈ {0, ∞}.
- **CRN eq. (15):** z rebuilt from the raw generator stream equals `fixings()` exactly (difference 0.0), and the
  sub-step grid ends on every date.
- **Eq. (19):** all three terms are computed as the note defines them, and the CRN SEs come from per-path differences.
- **Labelling:** every GS-based output is labelled "illustrative, pre-calibration": CSV `label` column, figure title,
  stdout, D-026, D-029, D-031, OPEN_ISSUES and BUILD_STATUS.
- **`forward_curve`:** equals my own log-linear interpolation and extrapolation to 1e-13 on 200 dates, is continuous at
  T_max, and raises before T_min. S_ref = 7374.4255 matches.
- **Phase 0 market tests** (`tests/test_market.py`, `tests/validation/test_phase0_independent.py`) pass after the
  `market.py` change. The diff only adds `forward_curve`.
- **Barrier edge cases** match the PDE: barrier at or near spot (branch (4); linear convergence of ρ → 1 down to 1e-12),
  T → 0, small σ with large μ (pitfall 8), deep ITM and OTM.

## Suite

**After fix round 1 (2026-11-09):**
- `.venv/bin/pytest -q`: **517 passed, 0 failed**, 1 warning (430 s). The warning is the pre-existing Phase 1 IV
  warning.
- `ruff check src tests scripts`: all checks passed.
- `ruff format --check src tests scripts`: 41 files already formatted.
- `wc -l src`: 1900.

*Original validation (pre-fix):* 511 passed, 1 failed (572 s). The failure was the intentional F2 defect test, which
now passes as `test_f2_heston_asian_default_grid_unbiased`.
