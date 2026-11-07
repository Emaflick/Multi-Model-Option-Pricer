# Phase 2 validation: Heston (CF, COS, Lewis, Euler-FT / QE / QE-M) (independent)

Validator: `validator` agent, 2026-11-07. Under test (public API only): `models/heston.py` (`HestonParams`, `cf`,
`cumulants`), `engines/fourier.py` (`cos_price`, `lewis_price`, `cos_series_error`), `engines/mc.py`
(`european_price`, `simulate`, `uniform_grid`). Also read: `docs/derivations/heston.md`, `heston-mc.md`,
`scripts/phase2_gates.py`, `scripts/heston_euler_bias.py`, `report/figures/heston_euler_bias.{png,csv}`,
D-015, D-017..D-020, `docs/OPEN_ISSUES.md`.

Tests: `tests/validation/test_phase2_independent.py`. It has 94 tests: 68 fast (5.5 s) and 26 marked `slow`
(about 170 s). Every number below is either asserted by those tests or printed by
`.venv/bin/python -W ignore tests/validation/test_phase2_independent.py [--mc]`. The `--mc` flag adds the
multi-seed MC studies (about 3 minutes). New golden file: `tests/golden/heston_prices.json` (18 cases; see
"Golden file created").

## Verdict

* **Every ROADMAP Phase 2 gate PASSES on the parameter sets the implementer declared in advance**, against
  references that use no qpricer code. The ξ → 0 gate passes under the D-015 reading, and I confirm D-015 independently.
* **The COS gate does not hold beyond those four sets (F1).** On random SPX-plausible parameters, the default
  COS (N = 256, L = 18) misses Lewis by more than 1e-6 in 22.9% of (set, T) cells, by up to 1e-1. The worst
  cells are flagged only by a `RuntimeWarning`. Five cells fail **silently** (max 2.61e-6). The cause is a
  truncation range that is too narrow, and the warning does not cover range error.
  **Severity:**
  - Not a blocker for closing Phase 2's literal gates.
  - **A blocker for Phase 5** if `cos_price` with default settings is the calibration engine.
  - `test_cos_default_meets_gate_or_warns_on_plausible_set` is **left failing on purpose** as the reproduction.
* No error was found in the CF, cumulants (main branch), Lewis, Euler-FT, QE or QE-M. The schemes match
  my own implementations in distribution, including the scheme-specific bias at Δ = 1.
* Full suite (`.venv/bin/pytest -q`): **352 passed, 1 failed** (the F1 reproduction only). `ruff check` is
  clean on the new file.

## Method (independent of the implementation)

| tag | reference | how it differs from the implementation |
|---|---|---|
| RIC | CF from my own fixed-step RK4 integration of the Riccati ODEs D' = ½ξ²D² − (κ − iρξu)D − ½(u² + iu), C' = κθD. I derived them from the pricing PDE. One Richardson step (n, 2n); the step count per u is set from the stiffness \|d\| + \|β\|. | No closed form, no branch choice. |
| GP | Gil-Pelaez / Heston put P = DF[(K−F)/2 − (1/π)∫₀^U Re(e^{−iuz}(Kφ(u) − Fφ(u−i))/(iu))du], with composite 16-point Gauss–Legendre (panels 0.25 on [0, 8], 2 beyond). The cut-off U is probed on RIC (\|φ\| < 1e-15·u). | Lewis uses the Im u = −½ contour and quad_vec. COS uses a cosine series. GP uses the real axis plus the share measure. |
| H93 | Heston's 1993 P1/P2 with his e^{+dT} CF. The log is continued by phase unwrapping along u. | Second closed-form route. Used only to certify the golden values. |
| MOM | c1 and c2 from the order-w and order-w² coefficient ODEs of the CGF, integrated with DOP853. | No closed form; stable as κ → 0. |
| BLK | Black-76 from `scipy.stats.norm`. ∫v̄ dt by `scipy.integrate.quad`. | Neither `bsm.price` nor `total_variance` is used. |
| OQE / OEU | My own Andersen QE / QE-M (ψc = 1.5, γ1 = γ2 = ½) and LKD Euler full truncation. Random layout: U_v uniform first, then Z_v = Φ⁻¹(U_v), then a separate Z⊥. | Different code and random stream, same scheme. |

Sanity checks on the references:
* RIC+GP vs H93: ≤ 3.3e-11 on all 18 golden cases.
* RIC+GP vs Lewis: ≤ 1.62e-9 on the gate lattice.
* Published values:

| Case | Own RIC+GP | Published |
|---|---|---|
| FO1 | 5.7851554344 | 5.785155450; MPRA preprint 5.785155435 |
| FO2 | 22.3189457912 | 22.318945791 |
| BK-1 | 6.8061133135 | 6.8061 |
| BK-2 | 34.9997583512 | 34.9998 |
| A-I | 13.0846701370 | 13.08467014 |

## Gate table

| Gate (ROADMAP) | Set / method | Measured | Tolerance | Result |
|---|---|---|---|---|
| COS vs Lewis | FO / GS / LW / AT2. Full lattice: K = 60..150 step 5, T ∈ {0.1, 0.25, 0.5, 1, 2, 3}, DF ∈ {1, 0.95}, C and P. | 1.01e-7 / 2.48e-9 / 1.10e-13 / 9.82e-9 | 1e-6 | **PASS** |
| COS vs Lewis | Same sets, off-lattice: 25 random K (with ends), 10 random T (with ends), DF ~ U[0.8, 1] | 8.33e-8 / 2.75e-9 / 1.35e-13 / 9.68e-9 | 1e-6 | **PASS** |
| COS, Lewis vs RIC+GP (true error) | Full lattice, 4 sets | COS ≤ 1.01e-7 (FO). Lewis ≤ 1.62e-9. | 1e-6 / 1e-8 (validator) | PASS |
| COS vs Lewis, **generalised** | 300 random sets: κ ∈ [0.3, 8] (log), θ ∈ [0.01, 0.12], ξ ∈ [0.1, 1.2], ρ ∈ [−0.95, −0.2], v0 ∈ [0.005, 0.12]. × 6 T, puts, DF = 1. | > 1e-6 in 413 / 1800 cells (22.9%), worst 9.95e-2. 480 cells warned. **5 silent failures, max 2.61e-6.** | 1e-6 | **FAIL (F1)** |
| ξ → 0, ρ = 0 | κ, θ, v0 = (1.5, .04, .04), (1.5, .04, .09), (1.5, .09, .01), (5, .02, .2), ξ = 1e-4. Full lattice, both engines, C/P, DF {1, .95}. Reference: Black at σ² = (1/T)∫v̄ (quad). | 4.36e-8 / 4.18e-8 / 3.02e-8 / 7.27e-9 | 1e-5 | **PASS** (230×) |
| ξ → 0, ρ = −0.7 (D-015) | Black + ξ·a1, where a1 = ∂P/∂ξ at 0 is a 4th-order central difference of RIC+GP at ξ = ±1e-3, ±2e-3 (RIC accepts ξ < 0). | \|engine − Black − ξ a1\| ≤ 5.6e-8. Eq. (16) term vs own a1: ≤ 1.6e-10 abs (max \|a1\| = 6.05). | 1e-5 | **PASS** |
| ξ → 0, ρ = −0.7, literal | Plain Black | 4.06e-5 / 3.27e-4 / 6.05e-4 at T = 0.1 / 1 / 3. My own pricer shows the same gap (to 1e-9), so the gap is in the model, not the code. | 1e-5 | not attainable (spec, F2) |
| QE MC vs COS | Implementer's gate (`test_gate_qe_vs_cos`, fixed seeds 20261107+i) | all 24 comparisons pass (suite run) | 3 SE | **PASS** |
| QE MC vs COS, cherry-pick check | 20 fresh seeds (777000+s, declared before the run) × {FO, GS, BK2, A-I} × {qe, qe-m} × K ∈ {70, 100, 140} puts, 2e5 paths, Δ = 1/32. Reference: Lewis. | 480 z-scores: max \|z\| 2.86, **none > 3**. Pooled mean −0.115, sd 0.937. Per-cell mean z within 3 SE of 0 (worst BK2 −0.32 with sd 0.84). | 3 SE | **PASS** |
| QE MC, independent implementation | OQE (QE-M) at the gate settings, seed 31337 | max \|z\| 1.35 | 3 SE | PASS |
| Euler bias shrinks with Δt (figure) | CSV of `heston_euler_bias.py` (1e6 paths) | A-I: +6.368 → +0.127 over Δ = 1 … 1/64, monotone, all > 3 SE. FO: +0.692 → +0.041 (5.0 SE) → +0.021 (2.6 SE, inside the noise). Fitted order 0.945 (A-I) and 1.054 (FO). | monotone until noise | **PASS** |
| Euler figure, independent | OEU at 1e6 paths vs CSV `mc_price`: FO Δ = ½ … 1/16, A-I Δ = 1, ¼ | \|z\| ≤ 0.91 (6 points) | 3 SE | PASS |
| One Feller-violating set | ν_F: FO 0.38 and GS 0.62 in the COS gate; FO, GS, BK2 0.36 and A-I 0.04 in the MC gate | all < 1 | — | **PASS** |

Other independent checks:

| Check | Measured | Result |
|---|---|---|
| `cf` vs RIC on the real axis, Im u = −½ and Im u = −1. Sets FO (T = 0.02, 1, 10), KJ (T = 1, 10), A-I (T = 10), ξ = 1.5, ρ = +0.5, ρ = −0.99. | ≤ 1.1e-12 abs | PASS. The little-trap branch handling is right at T = 10 for KJ (ν_F = 0.08). |
| `cumulants` vs MOM, main branch: 9 sets incl. κ = 0.05 and 20, T ∈ {0.02, 1, 10} | c1 ≤ 1.8e-15, c2 ≤ 2.7e-15 rel | PASS |
| `cumulants`, κT < 1e-3 branch | c2 rel. error −2.25e-2 (θ = 0.5, v0 = 0.01); +6.2e-5 (θ = v0) | F3 (cosmetic) |
| Golden file: Lewis everywhere; COS on the gate sets | Lewis < 1e-8, COS < 1e-6 | PASS |
| Short T and wings: T = 0.02 and T = 0.1, K = 60..150, FO and GS, vs RIC+GP | COS < 1e-6, Lewis < 1e-8 | PASS |

## Conventions

| Check | Result |
|---|---|
| Forward-based; discounting applied once | COS and Lewis: V(DF = 0.9) = 0.9·V(1) to 1e-12, and V(λF, λK) = λV to 1e-9. MC: price and SE scale exactly with DF (rtol 1e-15), and are homogeneous in (F, K) to 1e-14. No r or q appears in any kernel. |
| Calls vs puts | Cross-engine parity Lewis C − COS P = DF(F − K) < 1e-6 (asserted). COS calls come from parity (by design). On the gate lattice my own parity-built calls match both engines to the table values. |
| Annualisation | The ξ → 0 limit reproduces Black with σ² = (1/T)∫₀ᵀ v̄ dt, including v0 ≠ θ (time-averaging is exercised). The implementer's gate grid uses v0 = θ only, but `test_xi_to_zero_v0_neq_theta` covers v0 ≠ θ. |
| MC return contract | `(price, stderr, n_paths, seed)`. For a vector K, price is an ndarray; for a scalar K it is np.float64. n_paths and seed are int. The same seed gives bit-identical output. A strike subset gives identical prices (common paths). Mixed cp works. A CSV row of the figure (FO, Euler, Δ = ½, seed 20262107, 1e6 paths) re-runs bit-for-bit. |
| Martingale E[F_T] = F | QE-M and Euler-FT pass within 3 SE on four sets the implementer did not use: ρ = +0.5, ξ = 1.5, ν_F = 0.004, ρ = −0.99. |
| QE scheme identity | At Δ = 1 on A-I (5e6 paths each), implementation vs OQE: QE bias +1.0121 (0.0058) vs +1.0124 (0.0058); QE-M +0.2245 vs +0.2254. Andersen's Table 2, as quoted in heston-mc.md: +1.022 (0.013) and +0.233. Euler at 2e5 paths: +6.299 vs own +6.337 vs published +6.394 (0.029). |

## Edge cases

| Probe | COS (default) | Lewis | MC (QE-M, Δ = 1/32, 2e5) |
|---|---|---|---|
| T = 0.02 (FO, GS) | < 1e-6 vs RIC+GP | < 1e-8 | — |
| T = 0.005 / 0.001 / 1e-4, FO, K = 90..110 | \|COS − Lewis\| 1.2e-14 / 5.3e-14 / 4.9e-13, no warning; \|COS − Black(√v0)\| 9.9e-3 / 5.0e-4 / 1.4e-6 (correct T → 0 limit) | OK (no quad failure) | — |
| ξ = 1.5 (ν_F = 0.071) | 3.3e-10 at T = 0.02. 2.7e-6 … 2.4e-4 at T ≥ 0.1 (warned). | < 1e-8 | z = −1.03, −0.22, −0.27. Euler: +11.5, +15.8 (scheme bias). |
| ρ = −0.99 | 3.0e-5 … 9.7e-4 for T ≥ 0.1 (warned) | < 1e-8 | z ≤ 1.45 |
| ρ = +0.99 (κ − ρξ = 1.005) | 5.5e-11 at T = 0.02; 1.1e-5 … 4.0e-4 for T ≥ 0.1 (all warned) | < 1e-8 | — |
| ν_F = 0.0044 (κ = .5, θ = .01, ξ = 1.5) | 2.1e-7 at T = 0.02; 3.5e-4 … 4.4e-2 for T ≥ 0.1 (all warned) | no quad failure (not compared with RIC+GP: cut-off too large for a test) | z = −0.62, 0.31, −0.20 |
| κ = 1e-4 | ≤ 7.2e-12 at T ≤ 0.1; 3.5e-6 … 8.5e-3 for T ≥ 0.5 (warned) | no quad failure | — |
| v0 = 0 | ≤ 4.5e-7 vs Lewis (warned at T ≤ 0.1 although below 1e-6) | < 1e-8 (T = 0.5) | — |
| ρ = +0.5 | — | — | z ≤ 1.38. QE-M domain check not triggered at Δ = 1/32. |
| Deep-OTM put, T = 0.02, K = 70 | 0 | −0e0 / −7e-14 (round-off) | price 0 and SE 0, so z is undefined (no paths reach K) |

Lewis never failed: no quad_vec status ≠ 0 in 1800 random cells, on all edge sets, or at T down to 1e-4. Where it was
compared, it agrees with RIC+GP to ≤ 1.62e-9 on the gate lattice and < 1e-8 on the edge sets tested (ξ = 1.5, ρ = ±0.99,
v0 = 0 at T = 0.5; ξ = 1.5 at T = 0.02; SILENT at T = 1).

## Findings

**F1 — BLOCKER for Phase 5 (not for closing the literal Phase 2 gates): the default COS (N = 256, L = 18)
is not accurate to the gate tolerance outside the four designated sets, and range-truncation errors are
silent.**
* *Reproduction (left failing):* `test_cos_default_meets_gate_or_warns_on_plausible_set`.
  - Set SILENT = (κ, θ, ξ, ρ, v0) = (3.062, 0.01013, 0.6537, −0.6225, 0.02837), from the random scan (seed 5). ν_F ≈ 0.15.
  - At T = 1, K ∈ [60, 150]: |COS − Lewis| = 2.61e-6, with **no warning** (the eq. 13 estimate is 6.1e-6 < 1e-5).
  - Lewis, RIC+GP and H93 agree to ≤ 3.3e-12 on this set (golden case `SILENT`).
* *Diagnosis:* this is range error, not series error.

  | N, L | 256, 18 | 1024, 18 | 4096, 18 | 4096, 24 | 4096, 30 |
  |---|---|---|---|---|---|
  | \|COS − own\| | 2.61e-6 | 2.51e-6 | 2.51e-6 | 2.09e-8 | 9.28e-10 |

  It is D-017's pitfall 11 again: the left-tail reflection grows into the heavy ρ < 0 tail, and a c2-only range
  ignores kurtosis. The `RuntimeWarning` models only the series error (eq. 13), so range error is never flagged.
* *Breadth:*
  - In 1800 random plausible (set, T) cells, 413 (22.9%) exceed 1e-6. The worst is 9.95e-2 at
    (κ = 0.316, θ = 0.072, ξ = 1.14, ρ = −0.92, v0 = 0.058, T = 3), which needs N = 4096 and L ≥ 30.
  - Most of these cells are warned (480 warnings in total). In a calibration loop the warnings would be
    constant noise, and the prices behind them are wrong by up to 0.1 (roughly 0.14 vol pt at T = 3, hand estimate from
    ATM vega ≈ 69).
  - D-017 says "outside the gate domain L = 18 can be too narrow". The scan shows it is also too narrow
    **inside** the gate's (K, T) domain.
* *Recommended fix (in `src/`, by the implementer):*
  - (a) Make (N, L) adaptive and self-checking, e.g. compare (N, L) with (2N, L + 6) and double until they
    agree to 1e-7. COS stays cheap (≤ 4× the cost).
  - (b) Alternatively, widen the range with c4 (F&O eq. 49), after cross-checking c4 against MOM-type
    numerics.
  - (c) For Phase 5, calibrate with `lewis_price`, which is robust everywhere tested and needs one quad_vec
    call per expiry. Or use COS with a guard: a Lewis re-price at the fitted parameters, as D-017 already
    proposes. Turn RuntimeWarnings into errors inside the optimiser.
  - Record the choice as a D-entry. Until then, the report must say "COS < 1e-6 on sets FO, GS, LW, AT2",
    not "COS < 1e-6".

**F2 — minor (spec): the literal ξ → 0 gate is unattainable for ρ ≠ 0. D-015 is confirmed independently.**
* My own pricer (RIC+GP, no qpricer code) gives the same gap to plain Black: 3.27e-4 at T = 1, ρ = −0.7.
* Its numerically differentiated first-order coefficient matches eq. (16)'s ρξJ∂x∂ΣV_B to 1.6e-10.
* So the D-015 reference is correct, and the gap comes from the model.
* *Recommended:* the owner amends the ROADMAP wording to "ρ = 0 vs Black(σ_eff); ρ ≠ 0 vs Black + first-order
  term", and adds v0 ≠ θ to the gate grid in `scripts/phase2_gates.py`. With v0 = θ the time-average in the
  gate is trivial; I tested v0 ≠ θ and it passes at 4.2e-8.

**F3 — minor (cosmetic): the κT < 1e-3 branch of `cumulants` drops the κθ terms of c2.**
* The relative error is −2.25e-2 at κT = 0.999e-3, θ = 0.5, v0 = 0.01, and grows with θ/v0. c1 is exact.
* Impact: the COS range only (L = 18 becomes about 17.8). No price effect was seen, because COS at small κ is
  dominated by series error, which is warned.
* *Fix:* keep the first-order κ terms, e.g. add κ(θ − v0)T²/2 to the leading part, or evaluate the exact
  formula in a stable form.

**F4 — minor (API hazard): default option type differs across modules.**
* `bsm.price(..., cp=1)` defaults to a **call**.
* `cos_price`, `lewis_price` and `mc.european_price` default to `cp=-1`, a **put**.
* A call site that omits cp silently prices the other option.
* *Recommended:* make cp a required keyword in every pricing kernel.

**F5 — informational: CSV points at about 3 SE in the bias figure.** I re-ran both with 4e6 paths per
implementation:
* A-I, QE, Δ = 1/8: the CSV shows −3.1 SE. The re-run gives −0.0031 (0.0067), own −0.0093 (0.0067). This
  is noise, consistent with Andersen ("insignificant from 1/8").
* FO, QE-M, Δ = ½: the CSV shows −3.0 SE. The re-run gives −0.0236 (0.0039), own −0.0243 (0.0039). This is a
  **real** coarse-step bias of the scheme, not a code defect.

Neither point affects the gate (Δ = 1/32). Readers should not take the hollow or filled markers at coarse Δ
as proof that QE-M is unbiased there.

**F6 — informational: Lewis returns round-off-level negative deep-OTM prices** (about −7e-14 at T = 0.02,
K = 60). This is per note pitfall 13 (no clipping). Downstream IV inversion must keep treating them as
NaN plus a flag.

## Golden file created

`tests/golden/heston_prices.json` is new; no existing golden file was modified.
* *Content:* 18 cases of Heston put prices at K ∈ {60, 80, 100, 120, 150}, F = 100, DF = 1:
  - sets FO, GS, LW, AT2 at T ∈ {0.1, 1, 3};
  - FO at T = 0.02;
  - ξ = 1.5 at T ∈ {0.1, 1};
  - ρ = −0.99 at T ∈ {0.1, 1};
  - SILENT at T = 1.
* *Stored value:* method A, RIC+GP (RK4 step parameter 0.1, panels 0.125 / 1.0).
* *Second method:* H93 (Heston 1993 closed form with phase unwrapping, panels 0.25, parity). The two agree to
  ≤ 3.3e-11; the per-case agreement is stored.
* *Cross-check:* FO T = 1 ATM = 5.78515543437 vs Fang–Oosterlee Table 4, 5.785155450 (MPRA 5.785155435).
* *Generator:* `.venv/bin/python tests/validation/test_phase2_independent.py --golden-out <path>`. It refuses
  to overwrite. The file in `tests/golden/` is byte-for-byte the JSON content it produced.
* *Use:* Lewis must match to < 1e-8. COS is held to < 1e-6 on the gate sets only (F1).

## Reproduce

```
.venv/bin/pytest -q tests/validation/test_phase2_independent.py          # 93 pass, 1 fails (F1, on purpose)
.venv/bin/python tests/validation/test_phase2_independent.py              # prints every number above (~40 s)
.venv/bin/python tests/validation/test_phase2_independent.py --mc         # + MC studies (~3 min)
.venv/bin/python scripts/phase2_gates.py                                  # implementer's gate script (agrees)
```

## Post-fix status (orchestrator note, after implementer fix round 1)
- **F1 closed.** The default `cos_price` is now self-checking (D-017). This validator's own reproduction test
  `test_cos_default_meets_gate_or_warns_on_plausible_set` (written before the fix, unchanged) now passes. The
  implementer's re-run of the 300-set × 6-T scan (`scripts/phase2_gates.py`, output saved in
  `docs/validation/gate_outputs/phase2_gates.txt`) gives 0/1800 cells > 1e-6, worst 3.05e-9 — that number is the
  implementer's measurement, confirmed independently only through the passing test above.
- **F2** v0 ≠ θ cases added to the ξ→0 gate; **F3** small-κT cumulant branch fixed (D-021); **F4** `cp` required in
  the Heston engines. **F5/F6** informational.
- Full suite after the fix: 368 passed (333 fast + 35 slow), including all 94 tests of this validator's file.
