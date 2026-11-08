# Phase 3: SABR β = 1 (Hagan lognormal vol, ATM α-inversion, log-Euler and conditional MC)
Sources: `src/qpricer/models/sabr.py`, `tests/test_sabr.py`, `scripts/phase3_gates.py`, whose post-fix run (100 fresh seeds) is `docs/validation/gate_outputs/phase3_gates.txt` (cited as **gates.txt**), `report/figures/sabr_hagan_vs_mc{.png,_summary.csv}`, `docs/derivations/sabr.md` (eq. numbers, errata F1–F5), D-022..D-024, `docs/OPEN_ISSUES.md` (Phase 3), `docs/validation/phase3.md` (validator; written before the fixes).

## 1. Pitch (30 s)
SABR is Black-76 with a random vol: α_t is a driftless lognormal process with vol-of-vol ν, correlated (ρ) with the forward. Each expiry gets three nearly orthogonal knobs: α sets the level, ρν the skew and ν²/α the curvature. Hagan et al.'s (2002) closed-form σ_H(K) feeds the Phase 1 Black kernel, so Phase 5 can fit (σ_ATM, ρ, ν) one expiry at a time.
β is fixed at 1 (D-022). One smile cannot separate β from ρ, and SPX is quoted in lognormal vols. With β = 1 the model is scale-free, α is a vol, the MC stays positive and the ATM inversion is a quadratic. Hagan is an expansion in ν²T, so it is checked against two Monte Carlos of the true dynamics: log-Euler with exact α, and a conditional ("mixing") estimator.
Measured (gates.txt): at ν = 1e-12, |σ_H − α| = 3.0e-12 (gate 1e-10), and Hagan vs MC on set G (ν²T ≤ 0.16) gives max |z| = 1.68 over 21 puts (gate: 3 SE). Beyond that domain Hagan breaks: at ν = 1, k = −0.2 it is +0.40 to +0.82 vol pt too high (validator), and at ν²T = 10 it is off by 21 vol pt with a negative density.

## 2. Key formulas
1. **Dynamics (§1), T-forward measure.** `dF = α_t F dW`, `dα = να dZ`, `d⟨W,Z⟩ = ρ dt`. At β = 1 this gives `d ln F = −½α²dt + α dW` and, exactly, `α_t = α e^{νZ_t − ν²t/2}`. Terms: α is the lognormal vol, ν the vol-of-vol (per √yr), ρ the spot–vol correlation (< 0 on SPX). [LM07]: F is a true martingale iff ρ ≤ 0. For m > 1, E F_T^m < ∞ iff ρ ≤ −√((m−1)/m), so Var F_T < ∞ iff ρ ≤ −1/√2.
2. **Hagan, β = 1 (2).** `σ_H = α · z/x(z) · (1 + c1·T)`, with `z = (ν/α) ln(F/K)` and `c1 = ρνα/4 + (2 − 3ρ²)ν²/24`. z/x is the leading short-time term (the exact T → 0 limit); c1·T is the first time correction ([H02] App. B, small-vol expansion).
3. **x(z), eqs. (3) and (5).** `x = ln[(D + z − ρ)/(1 − ρ)] = ∫₀^z du/D(u)`, `D = √(1 − 2ρz + z²)`. For |z| < 1e-3: `z/x = 1 − ρz/2 + (2 − 3ρ²)z²/12 + ρ(5 − 6ρ²)z³/24`. The code splits at z = ρ using `(D + z − ρ)(D − z + ρ) = 1 − ρ²`.
4. **Shape (8) and ATM (4).** `σ_H ≈ (1 + c1T)[α + ½ρνk + (2 − 3ρ²)ν²k²/(12α)]`, with k = ln(K/F), and `σ_ATM = α(1 + c1T)`. With the parameters held fixed, the ATM skew ½ρν(1 + c1T) does not decay as T grows.
5. **ATM inversion (9).** `σ_ATM = α(b + aα)`, with `a = ρνT/4` and `b = 1 + (2 − 3ρ²)ν²T/24`, so `α = 2σ_ATM/(b + √(b² + 4aσ_ATM))`. For general β, eq. (4) is a cubic in α; β = 1 removes the α³ term. For ρ < 0, σ_ATM(α) peaks at `b²/(4|a|)`, and above that ceiling no α exists.
6. **MC, eqs. (13)–(17).** Exact vol step `α_{n+1} = α_n e^{ν√ΔZ_α − ν²Δ/2}`. Log-Euler `ln F_{n+1} = ln F_n − ½α_n²Δ + α_n√Δ Z_F`, with `Z_F = ρZ_α + √(1−ρ²)Z⊥`. Mixing identity: `ln F_T = ln F + (ρ/ν)(α_T − α) − ½I + √((1−ρ²)I)·Z`, with `I = ∫α²dt`. This gives `V̂ = DF·mean Black(Fξ, K, √((1−ρ²)Î/T))`, where `ξ = e^{(ρ/ν)(α_T−α) − ½ρ²Î}`.

## 3. The code that matters
**3a. z/x(z): sign split, log1p and series. `src/qpricer/models/sabr.py:37-44`**
```python
    z, rho = np.broadcast_arrays(np.asarray(z, dtype=np.float64), np.asarray(rho, dtype=np.float64))
    small = np.abs(z) < Z_C
    zs = np.where(small, 1.0, z)  # both np.where branches are evaluated (pitfall 1)
    d1 = np.sqrt((zs - rho) ** 2 + (1.0 - rho) * (1.0 + rho)) + 1.0  # D + 1
    up = np.log1p(zs * (d1 + (zs - 2.0 * rho)) / (d1 * (1.0 - rho)))
    dn = -np.log1p(-zs * (d1 + (2.0 * rho - zs)) / (d1 * (1.0 + rho)))
    series = 1.0 - rho * z / 2 + (2 - 3 * rho**2) * z**2 / 12 + rho * (5 - 6 * rho**2) * z**3 / 24
    return np.where(small, series, zs / np.where(zs >= rho, up, dn))
```
- **Lines 38–39, 43.** At K = F or ν = 0, z/x is 0/0. `np.where` evaluates both branches, so any |z| < z_c = 1e-3 is given a dummy value of 1 and the cubic series (5) is returned instead. The jump at the switch is 4.7e-14 (gates.txt). **Line 40:** D + 1, with D² written as (z − ρ)² + (1 − ρ)(1 + ρ), a sum of non-negative terms, so nothing cancels as ρ → ±1.
- **Line 41: why log1p.** For small z the log's argument is 1 + O(z), so the direct log has relative error of about 4ε/|z| (≈ 1e-12 at z_c). The identity D − 1 = z(z − 2ρ)/(D + 1) gives "argument − 1" exactly, which is passed to `log1p`. The error is then O(ε) at every z: 8.9e-16 down to z_c (validator, D-024).
- **Line 42.** For z < ρ, D ≈ ρ − z, so D + z − ρ cancels; the code uses the conjugate form ln[(1 + ρ)/(D + ρ − z)] instead. The two branches mirror each other bitwise, so the §3.3 identities hold to 1.3e-15, and x(z) matches quadrature to 2.9e-15 (gates.txt).

**3b. Hagan vol and ATM α-inversion. `src/qpricer/models/sabr.py:61-65, 79-82, 85`**
```python
    bracket = 1.0 + (rho * nu * alpha / 4 + (2 - 3 * rho**2) * nu**2 / 24) * T
    sigma = alpha * z_over_x(nu / alpha * np.log(F / K), rho) * bracket
    if np.any(bracket <= 0):
        warnings.warn("SABR: 1 + c1 T <= 0 (eq. 22), sigma_H set to NaN", RuntimeWarning, 2)
        sigma = np.where(bracket > 0, sigma, np.nan)
    # ... alpha_from_atm (:83-84 warn if any element has no root):
    a, b = rho * nu * T / 4, 1.0 + (2 - 3 * rho**2) * nu**2 * T / 24
    disc = b * b + 4.0 * a * s
    den = b + np.sqrt(np.maximum(disc, 0.0))  # max only feeds the flagged branch
    ok = (disc > 0) & (den > 0)
    return np.where(ok, 2.0 * s / np.where(ok, den, 1.0), np.nan)[()]
```
- **Lines 61–62: eq. (2).** Only K/F enters, so the formula has no F scale (tested to 1e-12 for λ ∈ {1e-3, 37.5, 1e4}), and r and q never appear. **Lines 63–65:** when c1 < 0, 1 + c1T becomes ≤ 0 at long T (eq. 22). The code then returns NaN with a warning; it never clips. Test case: ρ = −0.9, α = 0.5, ν = 1 gives NaN at T = 10 and a finite vol at T = 5.
- **Lines 79–85: the root of aα² + bα − s = 0.** The form 2s/(b + √disc) equals (−b + √disc)/(2a) but stays finite as a → 0, where it tends to s/b. It is the smaller root, where the bracket is ≥ b/2; on the other root the "correction" would dominate. The validator matched it against brentq on [0, vertex] to < 1e-13. For ρ > 0 the product of the roots is −s/a < 0, so there is exactly one positive root.
- **ρ < 0.** If s > b²/(4|a|), then disc < 0 and the code returns NaN. The validator confirmed this happens exactly when no positive root exists (20 000 points). At ρ = −0.9 the ceiling is 4.57% for ν = 2, T = 5, and ≈ 0.12% for ν = 5, T = 2, which is inside the TRF bounds.

**3c. MC loop (14) and conditional estimator (17). `src/qpricer/models/sabr.py:98-102, 134-138`**
```python
    for n, z in enumerate(draws, start=1):
        x = x + a * (sq * (rho * z[0] + rb * z[1]) - 0.5 * a * dt)  # eq. (14), left point
        i_left = i_left + a * a
        w = w + sq * z[0]
        a = alpha * np.exp(nu * w - 0.5 * nu * nu * n * dt)  # eq. (13), exact in law
    # ... mc_price, method="mixing":
        iv = i_left + 0.5 * dt * (aT * aT - alpha * alpha)  # trapezoid I, eq. (16)
        J = alpha * w if nu == 0 else alpha * np.expm1(nu * w - 0.5 * nu * nu * T) / nu
        lnf, s = np.log(F) + rho * J - 0.5 * rho * rho * iv, np.sqrt((1.0 - rho * rho) * iv)
        d1 = (lnf - np.log(Kc)) / s + 0.5 * s  # Black (bsm.md eq. 1) in ln F: F xi may underflow
        g = c * (np.exp(lnf) * ndtr(c * d1) - Kc * ndtr(c * (d1 - s)))
```
- **Lines 98–99.** Each step takes one (2, N) block: row 0 is Z_α, row 1 is Z⊥. α_n is known at t_n and independent of Z_F, so E[e^{Δx} | F_{t_n}] = 1 for every Δ and ρ. F̂ is therefore positive and an exact martingale. The O(Δ) weak bias comes from Î (left point, low by ≈ ½ν²Δ, eq. 16) and from the ρ-term Ĵ − J. **Lines 101–102:** α is rebuilt in closed form from the running W_n, so it is exact on the grid.
- **Lines 134–138.** Given the α path, ln F_T is Gaussian, so the payoff is replaced by its conditional Black value and the Z⊥ noise is integrated out. J is computed with `expm1`, which gives αW_T at ν = 0 without a threshold. Black is evaluated in ln F because Fξ underflows at ν = 1, T = 10. The trapezoid Î makes the bias O(Δ²). For ρ ≤ 0, ln ξ ≤ |ρ|α/ν, so the conditional call is bounded too (D-023 iv).
- **Line 141.** `mc_estimate(g, DF)` returns DF·mean and DF·s/√N. All strikes share the same paths, so their errors are correlated, which is why the gate's false-fail rate is not the iid figure.

## 4. Validation story
The validator (`phase3.md`, written before the fixes) wrote 35 tests whose references use no qpricer code: Hagan in 60-digit `decimal`, RQMC of the conditional representation, a 2-D ADI PDE in (ln F, ln α), and SCH, the exact expectation of scheme (14). It also created the golden file `tests/golden/sabr_prices.json` (34 puts, RQMC + PDE). All post-fix numbers below come from gates.txt.

| Gate / check | Measured | Tolerance |
|---|---|---|
| ν → 0, β = 1 → flat α (\|k\| ≤ 1, T 0.01..10, 6 values of ρ) | ν = 0: 0.0; ν = 1e-12: 3.0e-12 | 1e-10 |
| σ_H vs 60-digit Hagan, 1500 random points (α, ν ≤ 5; \|ρ\| ≤ 0.999; T ≤ 30; \|k\| ≤ 3) | 5.0e-14 rel (validator) | — |
| Hagan vs MC (14), set G (α 0.2, ρ −0.75, ν 0.4; T 0.25/0.5/1; k 0, ±0.05, ±0.1, ±0.2), puts, N = 2e5, Δ = T/250 | max \|z\| 1.68; (17) reported alongside, max 2.77. Validator before fix: 2.64; its own MC 2.42; seed-free offset ≤ 1.15 SE | 3 SE |
| Same gate, 100 fresh seeds | 4.0% of seeds fail; 0.29% per comparison; mean z −0.074, sd 1.042 (before fix: 9.0% / 0.86%) | 5.5% / 0.27% if iid |
| (14) step bias: CRN halving, Δ = T/250 vs T/500 | max \|V_Δ − V_{Δ/2}\|/SE_gate 0.080 / 0.063 / 0.073 (T = 0.25 / 0.5 / 1); Richardson bias ≤ 0.17 SE_gate (validator exact 0.164 / 0.169 / 0.139) | 0.1 SE_gate |
| (17) step bias, same CRN test; (14) vs (17), independent seeds, T = 1 | \|z\| ≤ 1.60; \|z\| ≤ 1.23 | 3 SE |
| [CHK] Table 1, FDM, ATM, T = 1 (diagnostic only) | Hagan − FDM: +1.6e-6 to +7.0e-6 at ν = 0.2, +3.0e-5 at ν = 0.4, +1.3e-4 at ν = 0.6; MC (17) \|z\| ≤ 1.55 against FDM | rounding |

- **Figure** (gates.txt, Gate 3, the same data as the summary CSV; α 0.2, ρ −0.75; largest |σ_H − σ_MC| among points with |z| > 3, in vol pt). ν = 0.4: at ν²T = 0.04 nothing is resolved (max |z| 2.2); 0.16 → 0.065 at k = +0.40; 0.8 → 1.74 at k = −1.34; 1.6 → 3.94 at k = −1.90.
  ν = 1: 0.25 → 0.33 at k = −0.30; 1 → 1.78 at k = −0.60; 5 → 14.2 at k = −1.34; 10 → 21.0 at k = −1.90. The density K·p(K) goes negative only at ν = 1, at low strikes: −8.0e-3 (ν²T = 5) and −6.1e-2 (ν²T = 10).
- **True Hagan error at k = −0.2** (validator RQMC, F3). Set G, T = 1: +0.058 vol pt (+0.008 ATM) = 1.0 SE_gate, hence the fresh-seed mean z of −1.05 there (gates.txt). At ν = 1: +0.40 (T = 0.5, α 0.18, ρ −0.7; 5.4 SE) and +0.82 (T = 1; 10 SE). Hagan is too high in the left wing.
- **Failed first: oracle (e).** A 3 SE test of a CRN difference fails even for correct code: (14) gave z = −7.96 … +2.46 at T = 0.25. CRN removes the noise from V_Δ − V_{Δ/2} but not the O(Δ) bias. M·(V_{1/M} − V_{1/2M}) at T = 1, k = −0.2 is −0.25, −0.28, −0.31, −0.26 for M = 25..250: flat, so the scheme is order 1 by design, not buggy. For (14) the oracle became a size check, ≤ 0.1 SE_gate.
- **F1: the size check was run at T = 1 only, with Δ = 1/250 at every T.** At T = 0.25 it read 0.321, a bias of −0.0415 vol pt = −0.64 SE_gate at k = −0.2. Eq. (18) was an ATM estimate, and the wing bias comes from the ρ-term. Fix: M = 250 at every T, giving −0.0104 / −0.0073 / −0.0084 vol pt at k = −0.2. The tolerance did not change; only the step did.
- **F2–F5.** F2: the iid figure 1 − 0.9973²¹ ≈ 5.5% ignored shared paths and Hagan's wing offset; measured 9.0% before F1, 4.0% after. F3: gate scope (above). F4: ρ > 0 parity (Q9). F5: [CHK] row 5 is ν = 0.6, and its 0.07811 is off by 1.0e-5. Oracle (ii) missed ½∂²σh² = 2.06e-10 and now uses a second-order bound (D-024).

## 5. Ten likely questions (basic → hard)
1. **What is SABR, and why β = 1?** Black with a lognormal stochastic vol correlated with F: α sets the level, ρν the skew, ν²/α the curvature (eq. 8). β and ρ both create skew (ATM slope −½(1−β)σ_ATM + ½ρν), and α trades off against β in the level, so fitting a free β means "fitting the noise" ([H02] §3.1).
   β is a dynamics choice: it sets the backbone, so it changes deltas, not today's prices. β = 1 matches lognormal SPX quotes and is scale-free; α is a vol, all spot–vol co-movement sits in ρ, the MC is a positive exact martingale, the ATM inversion is quadratic, and Hagan = Obłój (D-022).
2. **Where does z/x(z) come from, and how do you evaluate it?** Measure the distance to the strike in units of the current vol: u = ln(F/K)/α has diffusion D(νu)², so the short-time distance is x(z)/ν. Setting it equal to Black's ln(F/K)/σ gives σ = αz/x(z) (§3.1).
   Numerics: a series for |z| < 1e-3, a sign split at z = ρ, log1p, and NaN if 1 + c1T ≤ 0. The result matches a 60-digit Hagan to 5.0e-14. QuantLib switches at z² = 10ε, which leaves ~1e-8 noise just above the switch (note's estimate).
3. **How do you get α from the ATM vol? Which root? When does it fail?** Calibrate (σ_ATM, ρ, ν) and solve (9) for α ([H02] p. 13), so the level is always matched. Take 2s/(b + √disc): it is stable as ρν → 0 and is the smaller root (bracket ≥ b/2).
   For ρ < 0 there is a ceiling b²/(4|a|): 4.57% at ρ = −0.9, ν = 2, T = 5. Phase 5 must map the resulting NaN to a penalty, or bound ν per expiry; it must never silently drop points (F6).
4. **SABR or Heston: when does a desk use each?** SABR is a per-expiry closed-form smile, the standard for rates (swaptions and caps, per expiry × tenor) and for vanilla interpolation. It has no mean reversion (E α_T² = α²e^{ν²T}) and a skew that does not decay with T, so (ρ, ν) are refit per expiry ([H02] p. 14: ν falls with T); there is no single dynamics across expiries.
   Heston uses five parameters for the whole surface, mean-reverting variance and exact Fourier prices, giving consistent dynamics for exotics (Phase 4). Its short-dated skew is too flat: it tends to a finite ρξ/(4√v0) as T → 0 (`heston.md`).
5. **(Hedging) What delta do you trade, and what do SPX vols do when spot falls?** At fixed α (eq. 12): Δ = Δ_Black − vega·∂_kσ_H/F, i.e. sticky moneyness, which lies above Black's delta for ρ < 0. But E[dα | dF] = ρν dF/F, so Bartlett's delta adds ∂V/∂α·ρν/F, which is < 0.
   A fixed-strike vol then moves by ≈ ½ρν dF/F: vols rise when SPX falls, and the skew-stickiness ratio is ≈ 2, as in Heston. This is a hand derivation; no SABR Greek is implemented or gated.
6. **(Risk) How do you risk-manage a SABR-marked book?** For each expiry, σ_ATM is the vega, ρ the risk-reversal exposure and ν the butterfly (wing) exposure; hedge them with that expiry's straddles, risk reversals and butterflies. Compute vega by bumping σ_ATM with (ρ, ν) fixed and re-solving α from (9).
   The buckets are independent: there is no common factor like Heston's v0. Caveat: for ν²T > 0.16, an MC SABR exotic priced on Hagan-fitted parameters misprices its own hedges, by 0.40–0.82 vol pt at k = −0.2 when ν = 1.
7. **Where does Hagan break, and does your gate prove it right?** At β = 1, prices depend only on (k, α√T, ν√T, ρ), so the expansion is in ν²T and the dropped terms are O((ν²T)²). In the wings σ_H ~ ν|k|/ln|k|, so σ²T/|k| → ∞, which violates Lee's bound of 2: butterfly arbitrage appears far out and moves inward as ν²T grows (negative densities at ν²T = 5 and 10).
   So the gate covers set G (ν²T ≤ 0.16) only. The validator *estimates* that short-dated SPX fits sit at ν²T ≈ 0.2–0.7 (Phase 5 will measure this); there, Hagan is a smile parametrisation, not SABR.
8. **Walk me through the MC. Why 250 steps at every maturity?** (14) is exact α plus log-Euler: positive, an exact martingale, with O(Δ) bias. (17) is conditional, with O(Δ²) bias and less noise, and serves as an independent second MC: (14) vs (17) |z| ≤ 1.23.
   Δ = 1/250 left a −0.64 SE_gate bias at T = 0.25 (F1); Δ = T/250 brings it to ≤ 0.17 SE_gate everywhere. A CRN halving test cannot be a 3 SE test for an order-1 scheme, because CRN shrinks the SE below the true bias; use a size check plus an order check.
9. **(Hard) Why puts in price space? What changes for ρ > 0?** [LM07]: E F_T² = ∞ for ρ > −1/√2, so call payoffs have infinite variance, the CLT fails and SEs jump from seed to seed; puts are bounded by K. Price space is used because the Hagan side is deterministic, while inverting a noisy MC price to an IV is nonlinear and fails in the wings.
   For ρ > 0, F is a strict local martingale: E F_T/F − 1 = −0.167 (α 0.3, ρ 0.5, ν 1, T 5; two methods agree). (14) is still an exact martingale, so its calls target P + DF(F − K) while (17) prices the model call. Puts agree, call- and put-implied vols differ, and parity with F holds only for ρ ≤ 0.
10. **What would you change for production?** Arbitrage-free wings (Hagan 2014 PDE SABR), or MC/PDE pricing beyond ν²T ≈ 0.2. Make (17) the default MC and add a control variate (Phase 6). Gate the Greeks, including Bartlett's delta.
    Use an analytic σ_H Jacobian and turn the ceiling NaN into a penalty. Use the (−b + √disc)/(2a) root when b < 0 (F6). For rates, use shifted or normal SABR with β < 1. Report the gate as "set G, ν²T ≤ 0.16".

## 6. Traps: do NOT say
1. **"Hagan vs MC passes within 3 SE, so Hagan is SABR", or "our calibrated SABR prices exotics consistently."** The gate covers set G (ν²T ≤ 0.16) only. At ν = 1, k = −0.2, Hagan is off by +0.40 vol pt (T = 0.5, 5.4 SE) and +0.82 (T = 1, 10 SE), and by 21 vol pt with a negative density at ν²T = 10.
2. **"Parity always holds, so I price the put and take the call", "I gate on calls" or "I gate in vol space".** Parity with F is the model's only for ρ ≤ 0 (−0.167 at ρ = +0.5). Raw calls have infinite variance for ρ > −1/√2. The gate uses puts in price space.
3. **"Δ = 1/250 was checked", "the gate fails 5.5% of seeds by design", or quoting 0.129 vol pt as Hagan's error at ν = 0.4, T = 1.** Before F1 the wing bias was 0.64 SE_gate at T = 0.25. The measured false-fail rate is 4.0% (9.0% before F1). The 0.129 is one seed's MC noise (F7); the true ATM error is 0.008, and the error the MC resolves there is 0.065 at k = +0.4.
