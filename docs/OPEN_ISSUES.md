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

- **Phase 2 · note corrections requested (heston.md §2.2 eq. 8 and §5; derivation-note owner)**
  L = 12 COS fails the 1e-6 gate on sets FO (3.95e-5) and GS (2.58e-6). The error is the left-tail
  reflection term of §3.3, not the series. The code now uses a self-checking (N, L) by default
  (D-017, fix round 1). The §5 diagnostic "L = 16,
  N = 512 vs L = 12, N = 256 < 1e-8" does not hold for FO and GS. The eq. (13) estimate is about 20x
  pessimistic.

- **Phase 3 · note corrections requested (sabr.md §5 oracles (ii) and (e); derivation-note owner; no ROADMAP gate
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
- **Phase 3 · F3, carry to Phase 7 (scope).** The Hagan-vs-MC gate holds on set G with ν²T ≤ 0.16. At SPX-like ν ≈ 1,
  Hagan is off by +0.40 vol pt (T = 0.5) and +0.82 vol pt (T = 1) at k = −0.2. That is an expansion error, not a code
  error, and the report must state this domain.
