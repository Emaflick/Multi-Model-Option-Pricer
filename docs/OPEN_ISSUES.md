# Open issues

Gates that failed after 3 fix attempts. Format: phase · gate · observed vs expected · diagnosis · next step.

- **Phase 0 · DF in (0.8, 1] (passes; deferred minor item, validator finding F5, not a failed gate)**
  **Observed vs expected:** for T < 0.1 the parity-implied r spans 2.64%–16.89% (se 0.4%–4.3%), where a
  smooth zero curve near the long-end level of 4.5%–5.1% is expected. DF(T) rises between consecutive
  kept expiries 10 times out of 38 (max z = 3.87, 2026-12-03 → 2026-12-04). Going from 20 to 40 pairs moves
  short-end DF by up to 9.9 DF_se, so the residual-based DF_se understates the window uncertainty.
  **Diagnosis:** the slope signal (1 − DF)·ΔK is below quote resolution at short T, and the bias grows with
  window width (ITM-quote asymmetry, D-007). F is unaffected (se ≤ 0.4 pt). The (0.8, 1] gate is loose
  at the short end. Impact ≲ 0.06 vol points ATM.
  (Figures from docs/validation/phase0.md and data/processed/market_20261106_2213.csv.)
  **Next step (Phase 5):** replace the T < 0.1 parity DFs with a smooth or monotone DF(T) fitted to the
  identified long end (or an OIS curve) before IV inversion. Record the choice in DECISIONS.md.
  **Resolved 2026-11-09 (Phase 5, D-033; calibration.md eqs. 2–3; numbers from `scripts/run_calibration.py`,
  `report/calibration_f5_20261106_2213.csv`).**
  - WLS zero curve on the 21 expiries with T ≥ 0.1: r̂(T) = 4.527% + 0.304%·T.
  - Short-end r goes from 2.64%–16.89% to 4.533%–4.556%. DF rises between consecutive kept expiries 4 times
    instead of 10. Three of the remaining rises lie inside T ≥ 0.1, where the parity DFs are kept by design:
    02-19 → 02-26, 06-30 → 07-16 and 09-30 → 10-15. The fourth is the step across T = 0.1 that the note
    anticipates: 2026-12-11 (curve DF 0.99565) → 2026-12-18 (parity DF 0.99609). Below 0.1 y the DF sequence
    is monotone.
  - F refit at fixed D̃F: max |F̃ − F|/F_se = 3.56 (2026-12-03, +0.654 pt). Every used pair stays inside its own
    band: max |res|/h = 0.834, 0 pairs above 1.
  - Put/call IV split at the 347 used strikes with T < 0.1: median |σ_C − σ_P| 0.0275 → 0.0260 vol pt; pairs with
    |σ_C − σ_P| > h_C + h_P: 0 → 0.
  - The re-cut OTM surface is identical: 7625 quotes, no strike changes type.

- **Phase 1 · IV from deep-ITM quotes beyond |z| ≈ 3 (minor, deferred; validator finding F3, not a
  failed gate)**
  **Observed vs expected:** outside the gate domain, an ITM quote with 3 < |z| < 7.5 returns status 0
  with a round-trip error up to 2.0e-4 absolute (sigma = 1, T = 5, z = 6.5) and no flag. From
  z ≈ 6.75–7.5 on, the result is NaN with status 1. The expectation was either an accurate vol or a
  flag.
  **Diagnosis:** the solver is at the input conditioning limit eps·max(v, F, K)/(vega/DF), within
  10x of it (validator `test_iv_deep_itm_is_conditioning_limited`). An ITM quote carries its time value
  only to eps·max(F, K) (note pitfall 3), and the pitfall-7 threshold (time value > 10 eps v)
  accepts quotes that keep only 1–3 significant digits of time value.
  **Impact:** none on the gates. None on calibration either, because Phase 0 keeps OTM quotes only
  (`data.build_surface`), where the error is <= 1e-14.
  **Next step (if ITM quotes are ever inverted):** return an attainable-error estimate
  `info["sigma_err"] = eps·max(v, F, K)·DF/vega`, or add a status for an estimated error > 1e-8.

- **Phase 2 · AMST (2007) Table 1/2 published values (heston.md §5, A1/A2) not reproducible at
  stated precision (oracle issue, not a ROADMAP gate; not hard-tested, D-020)**
  **Observed vs expected:** Lewis and COS, which agree to ≤ 1e-7 there, against the published ATM calls
  (K = S0 = 100, r = 2.5%, q = 0). A1 (FO set): +0.0043, +0.0073, −0.0007, +0.0052, +0.0019 at T = 1, 2,
  3, 5, 10, against a stated precision of 0.005. T = 2 and T = 5 miss. A2 (AT2 set): +6.9e-5 at T = 1
  (precision 5e-5, miss), +6.7e-5 at T = 2 and −2.5e-3 at T = 3 (precision 0.005, pass).
  **Diagnosis:** (i) It is not an input convention: no single r fits all A1 values (T = 2 needs
  r ≤ 0.02498, T = 10 needs r ≥ 0.02499). (ii) The FO parameters are printed to 4 digits. Half-unit
  rounding of the five parameters moves the A1 prices by ±0.0057, 0.0073, 0.0083, 0.0098 and 0.0117
  (T = 1, 2, 3, 5, 10, first-order bound). That exceeds the 2-decimal precision, and every A1 deviation
  lies inside it, so AMST most likely priced with unrounded calibrated parameters. (iii) A2 uses round
  parameters; its 1.9e-5 excess at T = 1 is consistent with AMST's quadrature accuracy. The same CF
  reproduces FO1/FO2 (≤ 1.6e-8), QuantLib's Lewis values (3e-14), BK-1/BK-2 (≤ 4.2e-5 at 4 decimals)
  and Choi–Kwok's A-I (3e-9). (Numbers: `scripts/phase2_gates.py`.)
  **Next step:** the note's author should reclassify A1/A2 as qualitative (the "trap" itself is tested
  by the form-(4) negative control in `tests/test_heston.py`) or find AMST's unrounded parameters.

- **Phase 2 · note corrections requested (heston.md §2.2 eq. 8 and §5; derivation-note author)**
  L = 12 COS fails the 1e-6 gate on sets FO (3.95e-5) and GS (2.58e-6). The error is the left-tail
  reflection term of §3.3, not the series. The code now uses a self-checking (N, L) by default
  (D-017, fix round 1). The §5 diagnostic "L = 16,
  N = 512 vs L = 12, N = 256 < 1e-8" does not hold for FO and GS. The eq. (13) estimate is about 20x
  pessimistic.

- **Phase 3 · note corrections requested (sabr.md §5 oracles (ii) and (e); derivation-note author; no ROADMAP gate
  affected)**
  (1) Oracle (ii), K = F continuity: the bound |σ_H(±h) − σ_ATM| ≤ |∂_kσ|h + 1e-12σ is first order only. At h = 1e-4
  (set G, ρ = −0.75, T = 1), the −h side exceeds it by 2.06e-10 = ½∂²_kσ·h², which is exactly the eq. (8) curvature
  term. The test uses the second-order bound (D-024).
  (2) Oracle (e), halving with common random numbers for scheme (14): CRN shrinks the SE of V_Δ − V_{Δ/2} far below
  the real O(Δ) weak bias. So "|V_Δ − V_{Δ/2}| ≤ 3 SE of the difference" fails for correct code.
  - Measured at the gate step Δ = T/250 (fix round 1, validator F1), set G (`scripts/phase3_gates.py`):
    - T = 0.25, seed 20261128: z = −7.96 … +2.46.
    - T = 0.5, seed 20261129: z = −5.13 … +3.77.
    - T = 1, seed 20261127: z = −4.79 … +1.48.
  - M·(V_{1/M} − V_{1/2M}) is flat in M within noise. At T = 1, k = −0.2 it is −0.25, −0.28, −0.31 and −0.26 ± 0.02–0.05 at
    M = 25, 50, 125 and 250. So the bias is order 1, as eq. (16) predicts, and it is not a code error.
  - The test applies (e) as written to (17), where z ≤ 1.60 at every gate maturity. For (14) it requires
    |V_Δ − V_{Δ/2}| ≤ 0.1 SE_gate at **every** gate maturity, i.e. |bias| ≲ 0.2 SE. That keeps the per-comparison
    false-fail rate ≤ 0.33% (heston-mc.md eq. 3).
  - Measured max ratio: 0.080 (T = 0.25), 0.063 (T = 0.5), 0.073 (T = 1). The Richardson bias is ≤ 0.17 SE_gate everywhere (validator exact: 0.164 / 0.169 / 0.139).
  - The first version checked T = 1 only and used Δ = 1/250 at every T. At T = 0.25 that criterion fails: 0.321,
    validator F1. This was fixed by M = 250 steps at every maturity (D-023).
  **Resolved 2026-11-08 (orchestrator):** errata for (ii) and (e) added in sabr.md §5. The validator re-checked both
  replacement criteria and found no loosening (`docs/validation/phase3.md`).
- **Phase 3 · F6, deferred to Phase 5 (minor).** For ρ < 0, `sabr.alpha_from_atm` returns NaN above the feasibility ceiling
  b²/(4|a|) on the ATM vol. For example, at ρ = −0.9, ν = 5, T = 2 the ceiling is 0.12%, which lies inside plausible
  calibration bounds. A per-expiry SABR loss must treat these NaNs, either as a penalty or by restricting the bounds.
  Where b < 0 < a the round-trip is 4.6e-14, not 1e-14 (cosmetic).
  **Resolved 2026-11-09 (Phase 5, D-036):** the SABR loss uses native (α, ρ, ν), so eq. (9) is used only for the
  starts. Starts without a positive root are dropped and counted: 0 of 195 on the 39 expiries.
- **Phase 3 · F3, carry to Phase 7 (scope).** The Hagan-vs-MC gate holds on set G with ν²T ≤ 0.16. At SPX-like ν ≈ 1,
  Hagan is off by +0.40 vol pt (T = 0.5) and +0.82 vol pt (T = 1) at k = −0.2. That is an expansion error, not a code
  error, and the report must state this domain.
  **Phase 5 update:** every one of the 39 per-expiry SABR fits has ν²T > 0.16 (0.352–0.890; ν from 4.30 at
  T = 0.019 to 0.69 at T = 1.61), so every fitted SABR smile is outside the validated domain
  (`report/calibration_expiry_20261106_2213.csv`). The report must present them as Hagan parametrisations,
  not as SABR dynamics.
- **Phase 4 · autocallable BSM-vs-Heston gap is illustrative until Phase 5 (deferred; no gate failed).** The ROADMAP
  asks for BSM vs *calibrated* Heston. Calibrated parameters do not exist yet, so `scripts/autocall_compare.py` runs on the
  GS set (Gatheral 2006 Table 3.2 SPX fit, v0 := θ; D-029). Every output is labelled "illustrative, pre-calibration" and
  written to `report/figures/autocall_attribution_illustrative.{png,csv}`. With GS, AC-Q gives V_H − V_BS = −0.014362
  (CRN SE 0.000171). GS's ATM vol at 1.5 y is 16.9% against the market's 18.3%, so these numbers must not enter the
  report. **Next step (Phase 5):** write the fitted parameters to JSON, run
  `scripts/autocall_compare.py --heston-params <fit>.json`, and use only that output (tag = JSON stem) in the report. AC-A
  (3 y) stays an extrapolation beyond the last kept expiry (1.61 y) in carry and in Heston.
  **Resolved 2026-11-09 (Phase 5, D-038):** the gate is now run with the calibrated set
  (`data/processed/heston_fit_20261106_2213.json`; `report/figures/autocall_attribution_heston_fit_20261106_2213.csv`).
  AC-Q: V_H − V_BS = −0.018688 (CRN SE 0.000217) = −0.008206 (flat σ_ATM vs expected-variance curve) − 0.020445
  (marginal smile, exact) + 0.009963 (path). AC-A: −0.025137 (0.000342) = −0.018722 − 0.014800 + 0.008386. Both gate
  lines PASS in `scripts/phase4_gates.py` (`calibrated()`).
- **Phase 5 · S_ref still uses the first expiry's parity DF (minor; no gate affected).** `autocall_compare.market_inputs`
  sets S_ref = F·DF at T = 0.019 from `market_20261106_2213.csv`, i.e. the F5-biased parity DF 0.998395. Under the
  smoothed curve (D-033), F̃·D̃F = 7379.918 instead of 7374.425, i.e. +0.074%
  (`report/calibration_f5_20261106_2213.csv`, first row). The AC dates (≥ 0.25 y) interpolate only between T ≥ 0.1
  expiries, so F_0(t_i) and DF(t_i) are unaffected; only the reference level, and hence every barrier relative to
  the forwards, moves by 0.074%. **Next step:** let `market_inputs` take the smoothed table, or define S_ref from
  the curve DF; this needs a Phase 4 D-entry, and the autocall numbers must be re-run.
- **Phase 4 · Deng–Mallett–McCann (2011) Case 2 published value not reproduced (oracle issue; autocallable.md §5, not a
  gate; validator F7b).** Inputs as the note maps them:
  - monthly t_i = i/12;
  - B_ac = B_c = 1.02, memory, accrual coupons c_j = e^{0.092 t_j} − e^{0.092 t_{j−1}};
  - B_p = 0.8 (European), σ = 20%, F_0(t) = 100 e^{0.04t}, DF(t) = e^{−0.06t}.
  **Observed vs expected:** two implementations agree, and neither matches the paper.
  - The validator's exact backward induction gives 97.507 (its own MC: 97.509 ± 0.062).
  - qpricer MC gives 97.508 (SE 0.009, 1e6 paths; `scripts/phase4_gates.py`).
  - The published value is 98.39 (explicit FD), about 100 SE away.
  Case 1 (≈ par) and the Table 1 first-call probabilities are reproduced (qpricer max |z| 2.11; p_1 = 0.3767 is
  tested). **Diagnosis:** unresolved. Candidate causes are a different coupon or protection convention in the paper
  (e.g. coupon paid only at call versus the accrual mapping, or the protection terms) and FD error (the paper states
  no FD accuracy). Nothing was tuned to match. **Next step:** read the journal version of the paper, or obtain the
  exact term sheet; until then the value is not used as an oracle.
- **Phase 4 · minor nits from the validator's re-validation (N1, N2; no gate affected).**
  - **N1.** `autocall_compare._stats` builds the per-unit-coupon leg as `replace(sheet, c=1.0)`. On a vector-coupon sheet,
    the eq. (17) par coupon would then be a simple rate on c·τ_i instead of a multiplier on the given schedule. Only scalar
    sheets reach this code today.
  - **N2.** `scripts/phase4_gates.py` quotes two validator numbers as fixed citations: the pooled G2m biases and the
    exact Deng Case 2 value 97.507. Both are re-runnable from `tests/validation/test_phase4_independent.py`.
  - **F9 (validator round 0).** `kind="cv"` raises for the whole strike vector if any strike has an all-zero control
    payoff. This is documented behaviour.
  - **F10 (size).** `src/` is 1900 lines, above the ≲ 1500 target. Validated modules are not being refactored; this is
    reported to the author.
