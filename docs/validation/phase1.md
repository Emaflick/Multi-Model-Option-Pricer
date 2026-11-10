# Phase 1 validation: BSM / Black-76 (independent)

Validator: `validator` agent, 2026-11-07. Under test: `src/qpricer/models/bsm.py` and `src/qpricer/iv.py`,
public API only. Also read: `docs/derivations/bsm.md`, `scripts/phase1_gates.py`, D-011..D-014.

Tests: `tests/validation/test_phase1_independent.py` (54 tests, 2.2 s, none marked slow). Every number
below is either printed by `.venv/bin/python tests/validation/test_phase1_independent.py` or asserted by
those tests. New golden file: `tests/golden/bsm_black76.json` (12 cases, created by this validator; see §5).

## Verdict

* **All three ROADMAP Phase 1 gates PASS** on references that do not depend on the implementation.
  Margins (tolerance / worst measured): parity 1e-12 / 1.14e-13 ≈ 9×; Greeks vs FD 1e-5 / 1.24e-6 ≈ 8×;
  IV round-trip 1e-10 / 6.54e-13 ≈ 150×. *(Orchestrator correction: an earlier draft said "5 or more orders
  of magnitude", which the measured values do not support.)*
* *(Orchestrator note, after fix round 1: F1 fixed in `src/qpricer/iv.py` (status 5, NaN + warning), F4 fixed,
  F2 recorded as D-016, F3 deferred to `docs/OPEN_ISSUES.md`; full suite 156 passed, including all 54 tests of
  this file.)*
* **No blocker.**
* One contract violation outside the gate domain (F1, minor): a silent wrong implied vol, including
  sigma = 0 with status 0, for OTM prices below about 1e-280. This shows up as **8 failing tests**
  (`test_iv_deep_otm_never_silent_wrong[...]`), left failing on purpose as the reproduction.
* Full suite (`.venv/bin/pytest -q`): **146 passed, 8 failed** (only those 8). `ruff check src tests scripts`
  is clean.

## Method (independent of the implementation)

| tag | reference | differs from the implementation by |
|---|---|---|
| HP | Black-76 in stdlib `decimal` at 60 digits. The normal tail comes from my own series erf = 2/sqrt(pi) e^{-z^2} sum 2^n z^{2n+1}/(2n+1)!!, which has only positive terms and runs at a working precision raised by z^2/ln 10. Pi comes from Machin's formula. | No scipy and no float64. It is the exact Black price of the exact double inputs, about 50 digits. |
| HP-FD | Central differences of the HP price, computed in decimal. Steps are 1e-12 relative (1e-9 for gamma). Theta uses a joint (T, DF = e^{-rT}) bump at fixed r. Theta_0 uses a bump in T only. | There is no cancellation, and no analytic Greek formula is involved. |
| CS | Complex step Im f(u + ih)/h with h = 1e-30 u, applied to my own complex-valued Black-76. `scipy.special.ndtr` on complex input uses the Faddeeva algorithm. Gamma is the complex step of my own textbook delta. | Different code path for Phi, and no subtraction. |
| RFD | My own double-precision central differences plus one Richardson step. Steps: 1e-3·(F s, sigma, T) for first derivatives, 3e-3 F s for gamma. Run on ITM and OTM points for every Greek. | Different scheme and steps from `phase1_gates.py`, and off-lattice points. |
| QUAD | `scipy.integrate.quad` of K·expm1(s(u − u*)) against phi(u). | A second method for the golden prices. |
| BIS | My own vectorised bisection in sigma (64 halvings) on my own `scipy.stats.norm` pricer. ITM quotes are mapped to the OTM side by parity. | A second root-finder. |

Sample sets:
* The 41-z lattice of the note (147,600 points). It is rebuilt in my own code; the implementer's fixture is not reused.
* Off-lattice random samples: sigma ~ U[0.05, 1], T ~ logU[0.02, 5], z ~ U[−3, 3], DF ~ U[0.8, 1.02], with cp
  random. All 64 domain corners are included (sigma, T, z ∈ {−3, 0, 3}, DF, cp).
* An HP sample of 376 points (188 per cp) with corners.

Sanity checks on my own references:
* HP tail vs scipy `ndtr`: agreement within 3e-14. HP is the correct one. At y = 20 it matches the asymptotic
  Mills series to 30 digits, while scipy `ndtr(-20)` is off by 2.9e-14 relative.
* HP closed-form Greeks vs HP-FD: within 1e-15 in the test. The golden cases show 1e-18 to 1e-28.
* HP vs QUAD: within 4.4e-15 on all golden cases.

## Gate table

| Gate (ROADMAP) | Set / method | Measured | Tolerance | Result |
|---|---|---|---|---|
| Put-call parity | 200k random points, K/F ∈ [0.25, 4], sigma ∈ [0.05, 1], T ∈ [0.02, 5], DF ∈ [0.8, 1.02], C and P evaluated independently | 1.137e-13 | 1e-12 abs | **PASS** |
| Put-call parity | lattice restricted to K/F ∈ [0.25, 4] | 8.53e-14 | 1e-12 abs | **PASS** |
| Put-call parity | whole IV domain \|ln K/F\| ≤ 3σ√T (K up to 81,910) | 1.455e-11. 124 of 147,600 points are above 1e-12, all at K ≥ 8.3e3. Max 2.25 ulp of DF·max(F,K). | 1e-12 abs (literal) / 4 ulp (floor) | literal: not attainable in float64 (F2); floor: **PASS** |
| IV round trip | input = HP price rounded to double, 376 points, both cp, ITM and OTM | 6.54e-13 | 1e-10 | **PASS** |
| IV round trip | 300k random off-lattice points + corners, own pricer | 6.54e-13 (OTM 1.05e-14, ITM 6.54e-13); max 12 Newton evaluations, 0 Brent | 1e-10 | **PASS** |
| IV round trip | full lattice, own pricer, cp = +1 / −1 | 8.23e-13 / 7.67e-13 | 1e-10 | **PASS** |
| IV, second root-finder | 40k random, vs BIS | \|IV − BIS\| ≤ 2.21e-13 | 1e-11 (validator) | PASS |
| Greeks vs central FD | RFD, 100k random, both cp, **ITM and OTM** | Δ 1.86e-10; Γ 1.24e-6 (OTM 2.29e-8, ITM 1.24e-6); vega 2.52e-9; Θ 1.10e-7 † | 1e-5 rel | **PASS** |
| Greeks vs CS | full lattice, both cp | Δ ≤ 9.84e-13; Γ 8.90e-14; vega 9.86e-13; Θ ≤ 1.57e-12 | 1e-10 (validator) | PASS |
| Greeks vs HP-FD (true error) | 376 HP points, both cp, ITM and OTM | Δ 7.8e-15; Γ 8.9e-15; vega 8.7e-15; Θ 5.1e-14 ‡ | 1e-11 (validator) | PASS |
| Price vs HP (true error) | 376 HP points | 1.40e-12 rel. The worst point is the σ = 0.05, T = 0.02, z = ±3 corner (cancellation, note pitfall 6). | 1e-11 (validator) | PASS |

† Near a theta zero (deep-ITM with r > 0, \|Θ\|/\|Θ0\| goes down to 2e-5 in the sample), the relative
error uses the denominator max(\|Θ\|, 1e-3 \|Θ0\|). ‡ The theta error is scaled by \|Θ0\| + \|rV\|. Plain
relative error is 5.5e-14.

Remarks on the gates:
* **D-014 is confirmed independently.** The analytic gamma is exact to 9e-15 (HP-FD). The 1.77e-5 seen at
  the note's step 1e-4·F s is noise in the double-precision FD reference, not an error in gamma. With my
  own Richardson scheme, ITM gamma (which the implementer did not FD-test) passes at 1.24e-6. The step
  change in D-014 is a choice about the reference, not a loosened tolerance. I accept it.
* **The error model of the note holds.**
  - OTM IV round trip: 1.05e-14 measured vs "≲ 1e-14" estimated.
  - ITM IV round trip: 6.5e-13 measured vs "~1e-12" estimated.
  - Max rV/\|Θ0\| on OTM options: 0.688 measured vs the 0.7 estimate in pitfall 11.
  - D-011 Newton statistics: max 12 evaluations, 0 Brent calls, reproduced on 300k off-lattice points.

## Conventions

| Check | Result |
|---|---|
| Forward-based kernel; discounting applied once | V, Δ, Γ and vega are linear in DF to ≤ 4 eps. IV(DF·V₁, DF) = σ to < 1e-10. No r or q inside the kernels. |
| Theta definition | Θ = ∂V/∂t in calendar time, **per year**. F (the forward to the same expiry), K, σ and r = −ln DF/T are held fixed, so DF accretes. This matches HP-FD with the joint (T, DF) bump to 5e-14. Θ − rV = Θ0 = −∂V/∂T at fixed DF matches the HP-FD T-only bump to < 1e-11. The Black PDE Θ + ½σ²F²Γ − rV = 0 holds on the HP values to 1e-12. |
| Theta sign | Negative on every OTM lattice point (max −1.42e-3/yr), and negative ATM with r = 0. **Positive at 50.6% of ITM lattice points** (DF < 1: carry on the premium). The rule "long options have negative theta" therefore fails for ITM options under this convention. |
| Theta vs Hull's spot theta | Different by construction. On Hull's example the kernel Θ is −3.027/yr and Hull's spot Θ is −4.305/yr. Eq. (10) Θ_S = Θ − (r−q)FΔ_F reproduces Hull (see §5). |
| Vega units | Per unit of vol. Hull's 12.1 is reproduced (12.105); divide by 100 for per vol point. |
| Call/put swaps | cp = −1 matches my own put to 1e-13 abs + 1e-11 rel. Γ and vega are **bit-identical** across cp. Δ_C − Δ_P = DF to ≤ 4 eps. Θ_C − Θ_P = rDF(F−K) to 1e-12 rel. IV(C) = IV(P) to < 1e-10. b₋₁(x,s) = b₊₁(−x,s) exactly. V = DF√(FK)·b matches HP to 1e-11. |
| Vectorisation | NumPy broadcasting gives (2,1) × (3,) × (2,1,1) → (2,2,3), with mixed cp. Vectorised results are bit-identical to element-wise calls. Scalars return `float` (np.float64). List inputs work. `implied_vol` with `full_output=True` returns `status` with the broadcast shape. |
| Scale invariance of IV | IV(λV, λF, λK) = IV to < 1e-11 for λ = 60 (F = 6000, SPX level) and λ = 1e-4. |

## Published values

| Source | Inputs | Published | Implementation | Result |
|---|---|---|---|---|
| Hull OFOD 5th ed. pp. 287–288, via MathWorks `blkprice` (secondary, as cited in the note) | F = K = 20, T = 4/12, r = 9%, σ = 25% | C = P = 1.1166 | 1.1166414565589438 (both) | PASS (±0.5e-4) |
| Hull OFOD "The Greek Letters" running example (ch. 17 in the 7th ed. / ch. 18 in the 8th ed.). S = 49, K = 50, r = 5%, q = 0, σ = 20%, T = 20/52. **The values are quoted from the validator's knowledge of the text; the page was not re-read in this session.** | spot Greeks via eq. (10) | price 2.40, Δ 0.522, Γ 0.066, vega 12.1, Θ −4.31/yr = −0.0118/day | 2.4005, 0.52160, 0.065544, 12.1055, −4.30533/yr = −0.0117954/day | PASS (half a unit in the last digit) |
| Haug (2007) §1.1, marked UNVERIFIED in the note | F = K = 19, T = 0.75, r = 10%, σ = 28% | 1.7011 | 1.701050725236268 (HP 1.7010507252362672) | consistent; the source is still unverified (no test) |

## Edge cases (outside the gate domain unless stated)

| Probe | Behaviour | Assessment |
|---|---|---|
| σ = 0 | Price = DF·(cp(F−K))⁺ exactly. Δ = cp·DF·1{cp x>0}. Γ = vega = 0 off-ATM. At x = 0: Γ = +inf and vega = DF F√T/√(2π) (right-derivative, D-012). Θ = rV. | explicit, correct |
| σ√T → 0, ATM | Never negative. Absolute error ≤ 2.5e-15. Relative error ≈ eps/s: 9e-13 at s = 1e-4, 6e-7 at s = 1e-10, 2.6e-3 at s = 1e-14. Cause: F[Φ(s/2) − Φ(−s/2)]. | F5, informational |
| T → 0 at fixed σ | T = 1e-14 → intrinsic to ≤ 1e-13. IV round trip at T = 1e-4, \|z\| ≤ 3 is < 1e-10. | correct |
| T subnormal (1e-310) with DF ≠ 1 | r = −ln DF/T = inf, so for OTM `theta` = inf·0 = **NaN**, with only a NumPy RuntimeWarning. | F4, cosmetic (inconsistent inputs) |
| \|d\| > 1.3e154 (e.g. σ = 1e-300 off-ATM) | The values are correct (φ = 0), but `_phi` emits spurious NumPy "overflow encountered in multiply" warnings. | F4, cosmetic |
| Puts whose two terms cancel exactly | Return −0.0 (e.g. ATM put at T = 1e-300, deep-OTM put after underflow). | F4, cosmetic |
| Deep OTM, \|z\| ≤ 37 | IV accurate to ≤ 4e-14. For \|z\| ≳ 20, Newton hits max_iter = 100 and the Python-loop Brent fallback runs. | correct, but slow on such inputs |
| **Deep OTM, V ≲ 1e-280 (\|z\| ≈ 37–39)** | **Silent wrong IV, status 0, no warning.** σ = 0.05, T = 0.02, z = −38: σ̂ = 0.05042 (err 4.2e-4). σ = 0.3, T = 1, z = −38.5: **σ̂ = 0.0**. σ = 1, T = 5, z = −37: σ̂ = 0.99793 (err 2.1e-3). σ = 1, T = 5, z = −38.5 / −39: **σ̂ = 0.0**. Puts on the mirror side behave the same. `bsm.price` there is up to 17× the true value (2.92e-280 vs 1.71e-281) because `ndtr(d2)` flushes to 0 below d ≈ −37.5 while `ndtr(d1)` does not. | **F1, minor** (violates the stated contract) |
| Deep ITM quotes, 3 < \|z\| < 7.5 | Status 0. Error is at the conditioning limit eps·max(V, DF F, DF K)/vega (all within 10× of it), but it reaches **2.0e-4 absolute** (σ = 1, T = 5, z = 6.5) with no warning. NaN + status 1 from z ≈ 6.75–7.5 on. | F3, minor (design) |
| No-arbitrage bounds | V ≤ DF·ι, V < 0, V = DF·ι + 1 ulp → status 1. V ≥ DF·F (call) / DF·K (put) → status 2. 1 ulp below DF·F or DF·K → status 3 (vol > cap). ±inf / NaN → status 4. Each gives NaN and exactly one RuntimeWarning. | explicit, correct |
| Vol cap | σ = 9.5 is recovered to < 1e-10; σ = 12 → NaN, status 3. | correct |
| Invalid market inputs | F ≤ 0, K < 0, T = 0, DF = NaN, cp = 0, T < 0 and σ < 0 all raise ValueError. | explicit, correct |
| DF > 1 (negative rate) | Price, Greeks and IV round trip work (DF up to 1.02 is in the random sets). | correct |

## Findings

**F1 — minor (contract violation outside the gate domain): `implied_vol` returns a wrong σ, including
σ = 0, with status 0 and no warning, for OTM prices below about 1e-280.**
* *Reproduction:* `test_iv_deep_otm_never_silent_wrong` (8 failing cases). For example, V = 2.162e-305,
  F = 100, K = 100·e^{38.5·√5}, T = 5, DF = 1, cp = +1 gives σ̂ = 0.0 and status 0. The true σ is 1.0.
* *Diagnosis:* `scipy.special.ndtr` returns 0 for d ≲ −37.5, although Φ is representable down to
  d ≈ −38.5. The normalised price e^{−a/2}Φ(d1) − e^{a/2}Φ(d2) then loses its second term in the band
  d2 < −37.5 < d1, so the forward function is wrong by up to ~17×. Newton and Brent then converge, or fail
  towards the bracket end s = 0, on the wrong function. No post-check rejects s = 0.
* *Impact:* none on the gates or on SPX data (prices of 1e-280 are never quoted). It does break the
  documented rule "never σ = 0" (note pitfall 7, D-013) and the "never a silent wrong number" requirement.
* *Recommended fix (in `src/`, by the implementer):*
  - (a) Return NaN with a new status ("price below float64 tail resolution") when the OTM time value
    v − ι < ~1e-280·√(FK). A more principled trigger: Φ(d2) at the solution is below 1e3·`finfo.tiny`.
  - (b) Never return status 0 with s ≤ 0.
  - (c) Optionally, evaluate the OTM normalised price in log space (`log_ndtr`, or the `erfcx` form) so
    that Φ never underflows; this would also fix `bsm.price` there.

**F2 — minor (gate specification): the parity gate does not state its strike range.** On the full IV-gate
domain, K reaches 8.2e4 and ulp(8.2e4) = 1.5e-11. An absolute 1e-12 is then not representable in float64.
The implementation is at the floor (≤ 2.25 ulp of DF·max(F,K)), and it passes at 1e-12 on K/F ∈ [0.25, 4]
(the note's §2.5 domain) and up to K/F ≤ 40. *Recommended:* amend the ROADMAP gate to
"< 1e-12 abs for K/F ∈ [0.25, 4]; ≤ 4 ulp of DF·max(F,K) elsewhere". This is a gate wording change for the
author; the validator did not change it.

**F3 — minor (design): an ITM quote beyond |z| ≈ 3 returns an IV whose round-trip error can reach 2e-4
without any flag.** The implementation is optimal here: the error equals the input conditioning
eps·max(v, F, K)/(vega/DF). But the pitfall-7 threshold (time value > 10 eps v) accepts quotes that carry
only 1–3 significant digits of time value. *Recommended:* return an attainable-error estimate in `info`
(e.g. `info["sigma_err"] = eps·max(v,F,K)·DF/vega`), or add a status for an estimated error > 1e-8. Phase 0
keeps OTM quotes only, so the calibration path is not affected.

**F4 — cosmetic.**
* Spurious NumPy overflow warnings in `_phi` for |d| > 1.3e154 (fix: `np.errstate(over="ignore")` or clip
  d² before `exp`).
* `theta` gives NaN for subnormal T with DF ≠ 1 (inf·0).
* Some puts return −0.0.

**F5 — informational.** Price relative accuracy degrades as ~eps·max(1, |z|)/s when s → 0. It is
1.4e-12 at the gate corner s = 0.00707, 1e-10 to 4e-9 at s = 2e-4 (T = 1e-6, |z| = 4–30), and 6e-7 ATM
at s = 1e-10. The
absolute error stays ≤ ~eps·F. No action is needed for this project. For the ATM case, DF·F·erf(s/2√2)
would remove the loss.

**F6 — documentation.**
* The note's §2.4 gamma row (step 1e-4, "~1e-8") is wrong. D-014's diagnosis is confirmed independently
  (analytic Γ exact to 9e-15). The note author should correct the row, as D-014 already requests.
* The Hull Greek-letter example is now checked through the spot mapping (10). It is a useful
  check, because it shows that the kernel theta (−3.03/yr) and Hull's theta (−4.31/yr) differ by r F Δ_F.
* The Haug value stays unverified as a source, but it is consistent with the closed form (1.70105).

## Golden file created

`tests/golden/bsm_black76.json` (new; no existing golden file was modified) has 12 cases:
* Hull 5th-ed. call and put;
* ATM with DF = 1;
* four gate corners plus one ITM corner;
* an ITM put with positive theta;
* negative rate (DF = 1.02);
* z = ±8 deep OTM.

Each case stores price, Δ, Γ, vega and Θ. The stored values are **Method A**: closed forms (1), (5)–(8)
in 60-digit decimal. They are cross-checked by **Method B**: QUAD for the price (rel. agreement ≤ 4.4e-15)
and 60-digit FD for the Greeks (≤ 6.1e-18). The file records the per-case agreement. Generator:
`.venv/bin/python tests/validation/test_phase1_independent.py --write-golden` (it refuses to overwrite).
The implementation matches the golden file to < 1e-11 rel., and the IV of each golden price to < 1e-10.

## Reproduce

```
.venv/bin/pytest -q tests/validation/test_phase1_independent.py
.venv/bin/python tests/validation/test_phase1_independent.py      # prints every number above
.venv/bin/python scripts/phase1_gates.py                          # implementer's gate script (agrees)
```
