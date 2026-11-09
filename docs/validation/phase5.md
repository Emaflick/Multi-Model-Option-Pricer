# Phase 5 validation: calibration to SPX (independent), plus the calibrated Phase 4 autocall gate

Validator: `validator` agent, 2026-11-09.

**Under test:**
- `src/qpricer/calibration.py`, `tests/test_calibration.py`;
- `scripts/run_calibration.py`, `scripts/phase5_gates.py`, and `calibrated()` in `scripts/phase4_gates.py`;
- the outputs `data/processed/heston_fit_20261106_2213.json`, `report/calibration_*_20261106_2213.csv`,
  `report/figures/calibration_*_20261106_2213.png`, `report/figures/autocall_attribution_heston_fit_20261106_2213.{csv,png}`
  and `docs/validation/gate_outputs/phase5_gates.txt`.

**Also read:**
- `docs/derivations/calibration.md`, including the §8 errata;
- ROADMAP Phases 4 and 5;
- D-031 and D-032…D-038;
- OPEN_ISSUES (F5, F6, F3, the Phase 4 entry and the new S_ref entry).

**Tests:** `tests/validation/test_phase5_independent.py` has 19 tests: 12 fast (about 5 s) and 7 `slow` (about 3 min on
8 workers). All pass, so no test is left failing: I found no defect in the code. No golden file was created.

**Reproduction:** `.venv/bin/python tests/validation/test_phase5_independent.py` prints the fit, fresh-seed and autocall
numbers (about 3 min).

**Independent methods.** No reference number comes from qpricer numerics. qpricer is used only to load the snapshot rows
(`build_targets`, `build_surface`) and as the system under test. The methods:
- **CF:** Heston little-trap CF. It matches my own RK4 integration of the Riccati ODEs to < 1e-12.
- **LEWIS:** Lewis (2000) single integral by composite 10-node Gauss–Legendre, with a cutoff where
  |φ(u − i/2)|/(u² + ¼) < 1e-16. It is converged to < 1e-14 under cutoff doubling and panel halving. The implementation
  uses COS pinned at (16384, 64), and its own Lewis uses `quad_vec`.
- **IV:** 100-step bisection on ln σ plus 2 Newton polish steps, on my own Black.
- **LS:** my own residuals (eq. 8 with fallback 9, or eq. 10), my own central-difference Jacobian for the SEs and the
  sandwich (17), and my own Sobol starts. scipy `least_squares` is used only as the optimiser. Fits run on the **note's
  original bounds box (11)**, not the D-034 box.
- **HAGAN:** β = 1, transcribed from Hagan et al. (2002).
- **F5:** parity pairs rebuilt from the Phase 0 clean chain, my own OLS (it reproduces the table's F and DF to 1e-9), my
  own WLS zero curve and my own forward refit.
- **ARB:** my own calendar and butterfly code.
- **AC:**
  - exact V_BS, V_TS and no-call values by backward induction (ACX);
  - Heston marginals by Gil-Pelaez;
  - Heston paths by the Phase 4 validator's QE-M (OQEM), at Δt = 1/64 (10⁶ paths) and Δt = 1/256 (4·10⁵ paths).

  These come from `tests/validation/test_phase4_independent.py`, which is validator code.

Validator seeds start at 5 050 000 and were never changed after a result was seen.

## Gate table

| # | Gate (source) | Measured (implementer) | Measured (validator, independent) | Tolerance | Result |
|---|---|---|---|---|---|
| 1 | ROADMAP synthetic round trip, Heston S1 (G1–G4, eq. 19) | G1 4.1e-14; z = [1.10, 0.01, −0.74, 0.57, 0.37]; D² 2.52; G3, G4 pass | Pre-check SEs equal D-035 (rel < 2e-3, my own Jacobian). 100 fresh seeds: G2 fails 1/100; sd(z) = [0.95, 0.85, 1.00, 0.95, 0.97]; mean D² 4.84 (χ²₅: 5); G3 100/100; G4b 99/100 | 3 SE; χ²₅(0.999) = 20.52; G1 1e-6 | PASS |
| 2 | Same, Heston S2 | G1 2.1e-14; z = [1.70, −1.94, 2.42, −1.10, −0.20]; D² 12.49 | SEs equal D-035. 100 fresh seeds: G2 fails 0/100; sd(z) = [0.87, 0.95, 1.03, 0.94, 0.96]; mean D² 4.60; G4b 99/100 | same | PASS |
| 3 | SABR synthetic, 4 expiries (eq. 20) | all PASS; max D² 5.41 | Own Hagan = `sabr_iv` to < 1e-13; SEs equal D-035 (rel 1e-2). 200 fresh seeds per set: G2 fails 4, 1, 0, 1 /200; mean D² 3.11, 3.04, 2.83, 2.74 (χ²₃: 3) | 3 SE; χ²₃(0.999) = 16.27 | PASS |
| 4 | Linearisation (note §5): sd(p̂_j)/SE_j, 64 seeds | [1.10, 1.20, 0.98, 1.13, 1.00] | The implementer's 64 seeds refitted with my pricer give the same numbers (max \|Δp̂\| 4e-6 SE). 100 fresh seeds: [0.95, 0.85, 1.00, 0.95, 0.97] | [0.73, 1.27] | PASS |
| 5 | ROADMAP real fit completes; ≥ 8 starts | 16/16 converge, one basin, spread 1.1e-5 SE | **Note's original box**: 16 own Sobol starts + 4 adversarial corner starts → 20/20 reach p̂ (\|Δp\| < 1e-3 SE, cost rel < 1e-9) | ≥ 8 | PASS |
| 6 | p̂, SE, cost, Feller, cond (eq. 12) | p̂ = (0.052372, 4.5751, 0.052925, 1.6477, −0.75180); cost 2062.8978; ν_F 0.178; cond 25.6 | Own cost at p̂ rel 7e-14; own GN SEs = JSON (rel < 1e-4: 3.8189e-4, 0.10704, 2.5424e-4, 0.024266, 0.0017497); cond 25.57; GN step < 1e-6 SE (stationary); ≥ 38 SE from every D-034 bound | — | PASS |
| 7 | ROADMAP RMSE table, no NaN | 39 rows, 0 NaN; in 0.151–0.766 vp, out 0.832–2.093 vp | Own per-expiry RMSE = CSV to 3.6e-13 vp; in sample ⇔ T ≥ 0.1 (21 expiries), out of sample 18; pooled 0.328 / 1.379 vp | 0 NaN | PASS |
| 8 | Note §5: COS vs Lewis at p̂ | 4.9e-13 (own Lewis, I_H) | Pinned COS vs **my** Lewis on all 1029 thinned quotes (in and out of sample): 5.6e-14 | 1e-5 | PASS |
| 9 | Note §5: objective (10) within 1 SE of p̂ | real 0.83 SE; S1 0.48 SE | Real: −0.83, −0.46, 0.15, −0.79, −0.79 SE (identical) | ≤ 1 SE | PASS |
| 10 | D-034 COS budget over the D-034 box | e_vega 9.21e-6 | Worst corner (0.01, 0.5, 0.01, 3, −0.99) vs my Lewis: e_vega 9.21e-6, direct \|Δσ\| 2.5e-3. Erratum E1 claim (N, L) = (1024, 12) → IV error 9.9e-5: reproduced | ≤ 1e-5 (a priori) | PASS (see finding M3) |
| 11 | Sensitivity (D-037) | T ≥ 2 wk: κ +12.97, ξ +12.90 SE; n_t = 50 ≤ 1.06 SE | Own thinning and pricer on the original box: identical to 0.02 SE (cost to 1e-12 relative) | report | PASS |
| 12 | SABR per expiry, β = 1 | 39/39; RMSE 0.057–0.307 vp; ν²T 0.352–0.890, 39/39 flagged | Own Hagan fits, 9 own starts, no eq. (9): params to 1e-6, RMSE to 2e-8 vp, no active bound; ν²T ∈ [0.3516, 0.8900], all 39 > 0.16 | report and flag | PASS |
| 13 | Calendar (eq. 21) | 11054 points, mid 0, strict 0 | 11054 / 0 / 0 | diagnostic | PASS |
| 14 | Butterfly (eq. 22) | 7547 triples, mid 2126, strict 0, max z 0.427 | 7547 / 2126 / 0 / 0.4268 | diagnostic | PASS |
| 15 | F5 closure (eqs. 2–3, D-033) | r̂ = 4.527% + 0.304% T; max \|res\|/h 0.834; max z_F 3.56 | Own refit: F̃ to 3e-12, D̃F exact; max z_F 3.562 (2026-12-03); 4 expiries > 2 F_se; max \|res\|/h 0.8339; DF rises 10 → 4 | \|res\|/h ≤ 1 | PASS |
| 16 | Targets (eqs. 1, 5, 6) | 5806 in band, 1029 thinned, n_H 565; h_min binds 60.9% | Own IVs = implementation's to 8e-15; band, thinning masks identical; OTM type consistent with F̃ | identities | PASS |
| 17 | ROADMAP Phase 4: autocall, BSM (ATM) vs calibrated Heston, AC-Q | gap −0.018688 (CRN SE 0.000217) = −0.008206 − 0.020445 + 0.009963 | Exact V_BS 1.014493 (impl MC 1.014530, z 0.40); OQEM V_H 0.995882 (0.000102), impl z −0.16; gap −0.018611 (z 0.32); term1 exact −0.008136 (z −0.58); term2 −0.0204452 (Δ 4e-10); term3 +0.009970; Δt = 1/256: V_H 0.995905 (no step bias) | 4 SE | PASS |
| 18 | Same, AC-A (extrapolated beyond 1.61 y) | gap −0.025137 (0.000342) = −0.018722 − 0.014800 + 0.008386 | Exact V_BS 1.021103 (impl MC z 1.57); OQEM V_H 0.996395 (0.000163), impl z −0.13; gap −0.024708 (z 1.13); term1 −0.018733; term2 Δ 5e-10; term3 +0.008825 | 4 SE | PASS |

## Findings

No blockers. The code reproduces every number I could recompute independently. The findings below are minor: they
concern documentation, auditability or the strength of a gate.

**M1 — The Phase 4 additivity gate line is a tautology (minor).** `phase4_gates.calibrated()` passes when
`|term1 + term2 + term3 − gap| < 1e-12`. But `autocall_compare.attribution` defines `term3 = V_H − V_TS − term2`, so
the sum equals the gap by construction, and this half of the gate cannot fail. Only the no-call "MC vs exact
|z| ≤ 3" half carries information. The calibrated gap and all three terms are now independently confirmed (rows
17–18), so no number is wrong.
*Fix:* replace the identity with a real check, for example V_BS(MC) against an exact deterministic-variance value
within 3 SE, or label the line "plumbing".

**M2 — "Fixed before the fit" (D-034, D-035) cannot be audited from the repository (minor, process).** These files are
untracked or uncommitted:
- `calibration.md`, including its errata;
- the D-032…D-038 entries;
- `calibration.py` and the tests.

The D-034 pilot fit on the original box has no script and no saved output, so the disclosure is not re-runnable.
CLAUDE.md requires every number to come from a script.

*Impact: none on the results.*
- My fit with my own pricer on the **original** note box (11) reaches the reported optimum from all 20 starts. These are
  16 Sobol starts on the original start box plus 4 adversarial starts: v0 = θ = 1e-3 with ξ = 4.5, κ = 0.01, and others.
- The optimum is ≥ 38 SE from every narrowed bound: κ is 38 SE above 0.5, ξ is 56 SE below 3, v0 is 111 SE above 0.01.
- The n_t = 50, T ≥ 2-week and objective-(10) fits are also interior and are reproduced on the original box.

So narrowing the box does not bias any reported fit, and the narrowing is legitimate: the original box cannot be
priced accurately by any fixed COS, which I confirmed with my own Lewis. The synthetic tolerances are also confirmed to
be a-priori quantities: my own eq. (17) SEs at p* match D-035, and they depend only on (p*, grid, w, σ_ε). The gates pass
on 99–100% of fresh seeds, so no seed selection was needed. The seeds in the code match D-035.
*Fix:* in D-034, cite `test_real_fit_global_on_note_original_box` as the re-runnable replacement for the pilot. Commit
the phase.

**M3 — The D-034 budget measures price error in market-vega units, not IV accuracy over the box (minor,
documentation).** e_vega = max |Δv|/𝒱(σ^m) ≤ 1e-5 is met (9.21e-6 at the worst corner, reproduced with my Lewis). At
that same corner the direct model-IV error is 2.5e-3; the implementation reports 6.4e-3 over the whole box. The metric
is still sensible: it is the first-order IV error wherever the model is near the market, which is the only place an
IV error affects the optimum. The post-fit check confirms it, with COS = Lewis to 5.6e-14 on all 1029 quotes at p̂ and
4.9e-13 at every start endpoint.
*Fix:* the report must not describe 1e-5 as "IV accuracy over the bounds box".

**M4 — The stated false-fail rate covers only the Heston G2 (minor, documentation).** Note eq. (19) states P(false
fail) ≈ 2.9%. That is the union bound for the two Heston G2 gates only. The full synthetic suite also contains:
- 4 SABR G2 gates, about 0.9% each;
- 6 G4b gates, about 0.27% each.

The union bound for the whole suite is therefore about 8%.

Measured on fresh seeds:

| Set | Seeds | G2 fails | G4b fails |
|---|---|---|---|
| Heston S1 | 100 | 1 | 1 |
| Heston S2 | 100 | 0 | 1 |
| SABR 2026-11-13 | 200 | 4 | n/a |
| SABR 2026-12-18 | 200 | 1 | n/a |
| SABR 2027-05-21 | 200 | 0 | n/a |
| SABR 2028-06-16 | 200 | 1 | n/a |

The SABR 2026-11-13 set fails at 2% against a nominal 0.9% (binomial p ≈ 0.1). Its ν = 2 sits at 3SE/ν = 0.27, where
the linearisation starts to bend (max D² 17.9).
*Fix:* state the suite-level rate in D-035.

**M5 — The κ ratio of 1.20 in the linearisation test belongs to that seed set (informational).** Refitting the
implementer's 64 seeds with my pricer reproduces [1.10, 1.20, 0.98, 1.13, 1.00] exactly (|Δp̂| ≤ 4e-6 SE). On 100
fresh seeds the ratios are [0.95, 0.85, 1.00, 0.95, 0.97]. The 1.20 is sampling noise, not a code effect. No action.

**M6 — The real-data SEs are not parameter uncertainty (minor, documentation for the report).** s² = 7.37, so
residuals are about 2.7 half-widths. The T ≥ 2-week window moves κ and ξ by +13.0 and +12.9 SE, which I reproduced to
0.02 SE. κ = 4.58 ± 0.11 (curvature) understates the window and model risk by an order of magnitude. D-037 says this,
and the report must quote parameter uncertainty as the sensitivity range.

**M7 — A note §5 diagnostic is not implemented (minor).** "SABR fitted to the S1 Heston smiles: RMSE should grow with T"
appears nowhere in the scripts or tests. I ran it on the 39 real grids:
- the RMSE grows from 0.010 vp (T = 0.019) to 0.18 vp;
- the Spearman correlation ρ(T, RMSE) is 0.91, so it is consistent with the note.

*Fix:* add it to `run_calibration.py`, or drop it from the note.

**M8 — S_ref (OPEN_ISSUES) is stated correctly and is minor (confirmed).**
- S_ref uses the F5-biased parity DF of the first expiry. Own refit: F̃·D̃F / (F·DF) − 1 = +0.0745% (7374.425 →
  7379.918).
- The AC dates interpolate only between T ≥ 0.1 expiries, so F_0(t_i) and DF(t_i) are unchanged.
- Measured effect of the fix (CRN, OQEM): ΔV_H ≈ −4e-5 and ΔV_BS = +3.7e-5. On the AC-Q gap this is −8e-5 (0.4 CRN
  SE); on AC-A it is ≈ −1e-5.

**M9 — The BSM leg is priced at Heston's ATM vol, not at the market's (minor, documented in D-031).**
- σ_ATM = 18.85% is Heston's ATM IV at 1.5 y. The interpolated market ATM vol there is 18.29%.
- The fit is +0.65 vp too high ATM on the last expiry (2028-06-16, market 18.29% vs Heston 18.94%).
- V_BS at the market ATM vol is 1.015253, against 1.014530: +0.0007, about 3.3 CRN SE of the gap.

The report must say "BSM at the calibrated-Heston ATM vol". The ROADMAP's "explain the difference" is satisfied by the
eq. (19) decomposition, which I confirmed term by term.

**M10 — Size (minor, owner decision).** `src/` has 2468 lines (`calibration.py` 568), against the target of ≲ 1500.
The Phase 4 report (F10) already flagged 1900.

**M11 — JSON provenance nit (trivial).** `heston_fit_*.json` lists "D-032..D-037" and omits D-038.

## Confirmations requested by the brief

**Optimum and SEs.**
- The optimum is global within the note's full original box, as far as 20 diverse starts can show (row 5).
- It is interior (row 6), and the SEs, Feller ratio (0.178) and condition number (25.6) are reproduced to 4–6 digits.

**RMSE table and figures.**
- The RMSE table is correct and NaN-free (row 7).
- In the smile figures, the "Heston OUT of sample (T < 0.1)" panel holds exactly the 18 short expiries and the in-sample
  panel holds the 21 others.
- Each panel title RMSE equals the CSV value. The figure shows the in-band quotes with ±h bars, the thinned targets,
  Heston and SABR.
- Visually, the Heston curve is too flat on the weeklies, the expected failure of eq. (14). It is also above the market
  ATM at 1.61 y.

**F5 erratum (E3).** It is correct.
- The note's 1.5 F_se estimate looked only at 2026-12-07. There my refit gives ΔF = +0.79 pt = 1.96 F_se, against the
  note's ≈ +0.6 pt.
- The maximum z_F = 3.56 is on 2026-12-03 (+0.654 pt), where F_se is smaller.
- The mechanism ΔF = (F − K̄)(DF − D̃F)/D̃F holds to 1e-8 on all 18 expiries, with parity DFs low by up to 0.44%
  (2026-11-19).
- The ATM put/call split this implies is at most 1.2533·0.654/(7395·√0.0738) ≈ 0.041 vp, below the median IV half-width
  of 0.078 vp.

**F6 handling.**
- The SABR loss uses native (α, ρ, ν). Eq. (9) appears only in the starts.
- My fits use no eq. (9) at all and reach the same optimum on all 39 expiries, so restricting the starts does not bias
  the fit.

**ν²T flags.** 39/39 lie above 0.16, in [0.352, 0.890]. Every SABR smile is therefore a Hagan parametrisation outside the
Phase 3 validated domain, as OPEN_ISSUES F3 states.

## `pytest -q` and lint

- `pytest -q` (full suite, including all `slow` tests): 556 passed, 1 warning (a Phase 1 validator test's expected
  RuntimeWarning), 1170 s.
- `ruff check src tests scripts`: All checks passed.
- `ruff format --check`: 46 files already formatted.
