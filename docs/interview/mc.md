# Phase 2: Heston Monte Carlo (estimator and SE, Euler full truncation, Andersen QE / QE-M)
Sources: `src/qpricer/engines/mc.py`, `tests/test_mc.py`, `scripts/heston_euler_bias.py`, `report/figures/heston_euler_bias.{png,csv}`, `docs/derivations/heston-mc.md` (eq. numbers), D-018, D-020, `docs/validation/phase2.md`. The CF, COS and Lewis (the MC references) are in `heston.md`. Variance reduction and N-convergence are Phase 6 (forward pointers only).

## 1. Pitch (30 s)
`engines/mc.py` simulates Heston under the T-forward measure on the state (x = ln F_t/F, v). The forward is driftless apart from the −½v convexity term, and DF is applied once. **Euler with full truncation** is the baseline: simple and an exact martingale, but with an O(Δ) bias that is large for Feller-violating (SPX-like) parameters.
**Andersen's QE** samples the next variance from a law with the exact CIR conditional mean and variance: a squared Gaussian away from 0, and an atom plus an exponential near 0. It builds the log-forward from the realised variance move, so the skew survives large steps. **QE-M** adds a per-path drift so that E[F_{n+1} | v_n] = F_n exactly.
Andersen reports QE at Δ = 1 or ½ about as accurate as Euler at 1/32 (ATM/OTM), at 1.21× (QE) or 1.38× (QE-M) Euler's cost (published, not measured here). Every price returns `(price, stderr, n_paths, seed)`. Measured: QE/QE-M vs COS 24/24 within 3 SE; 480 fresh-seed z-scores, max |z| 2.86; Euler bias order 0.95/1.05; Andersen's Table 2 reproduced.

## 2. Key formulas (heston-mc.md eq. numbers)
1. **Estimator and gate (1)–(3).** `V̂ = DF·ḡ`, `SE = DF·s_g/√N` (`ddof=1`); pass if `|V̂ − V_ref| ≤ 3 SE`. For an unbiased scheme the false-fail rate is 0.27% per comparison and `1 − 0.9973^m` over m comparisons. Power against a bias b is ≈ Φ(|b|/SE − 3), only 50% at a 3 SE bias. SE ∝ N^{−1/2}, so halving it costs 4× the paths.
2. **Euler-FT (4)–(6).** `v_{n+1} = v_n + κ(θ − v_n⁺)Δ + ξ√(v_n⁺Δ) Z_v`, `x_{n+1} = x_n − ½v_n⁺Δ + √(v_n⁺Δ) Z_F`, `Z_F = ρZ_v + √(1−ρ²) Z⊥`. The stored v may be negative; only v⁺ enters. v_n⁺ is known at t_n, so E[e^{Δx} | F_{t_n}] = 1 exactly, for every Δ.
3. **QE moments (7).** `E = e^{−κΔ}`, `m = θ + (v_n − θ)E`, `s² = v_nξ²E(1−E)/κ + θξ²(1−E)²/(2κ)`, `ψ = s²/m²`. These are the exact CIR conditional mean and variance. ψ is the relative variance; its maximum is 1/ν_F at v = 0, with ν_F = 2κθ/ξ².
4. **QE branches (8)–(9), ψc = 1.5.** If ψ ≤ ψc: `v' = a(b + Z_v)²`, `b² = 2/ψ − 1 + √(2/ψ)√(2/ψ − 1)`, `a = m/(1 + b²)`. If ψ > ψc: `p = (ψ−1)/(ψ+1)`, `β = (1−p)/m`, `v' = 0` if U ≤ p, else `β⁻¹ ln[(1−p)/(1−U)]`, with U = Φ(Z_v). Both branches match m and s² exactly. The quadratic branch needs ψ ≤ 2 and the exponential needs ψ ≥ 1, hence ψc ∈ [1, 2]. If ν_F ≥ 2/3, the exponential branch never fires.
5. **Log-forward (10).** `x_{n+1} = x_n + K0 + K1 v_n + K2 v_{n+1} + √(K3 v_n + K4 v_{n+1}) Z⊥`, `K0 = −ρκθΔ/ξ`, `K1,2 = ½Δ(κρ/ξ − ½) ∓ ρ/ξ`, `K3 = K4 = ½Δ(1−ρ²)`. It follows from ∫√v dW^v = (v' − v − κθΔ + κ∫v)/ξ with a trapezoid for ∫v. Leverage enters through (ρ/ξ)(v_{n+1} − v_n), and Z⊥ is independent of Z_v.
6. **QE-M (11)–(12).** `A = K2 + ½K4`, `M = E[e^{A v_{n+1}} | v_n]`. Quadratic branch: `ln M = A b² a/(1 − 2Aa) − ½ ln(1 − 2Aa)` (needs 2Aa < 1). Exponential branch: `M = p + β(1−p)/(β − A)` (needs A < β). Per path, `K0* = −ln M − (K1 + ½K3) v_n` gives E[e^{Δx} | v_n] = 1. If ρ ≤ 0 then A ≤ 0, and M is always defined.

## 3. The code that matters
**3a. QE variance step: `src/qpricer/engines/mc.py:44-58`**
```python
    k, th, xi, rho = p.kappa, p.theta, p.xi, p.rho
    E = np.exp(-k * dt)
    m = th + (v - th) * E  # eq. (7)
    s2 = v * xi**2 * E * (1.0 - E) / k + th * xi**2 * (1.0 - E) ** 2 / (2.0 * k)
    psi = s2 / (m * m)
    quad = psi <= PSI_C
    iq = 2.0 / np.minimum(psi, PSI_C)  # eq. (8)
    b2 = iq - 1.0 + np.sqrt(iq * (iq - 1.0))
    a = m / (1.0 + b2)
    pe = np.maximum(psi, PSI_C)  # eq. (9)
    pp = (pe - 1.0) / (pe + 1.0)
    beta = (1.0 - pp) / m
    one_minus_u = ndtr(-zv)
    v_exp = np.where(one_minus_u >= 1.0 - pp, 0.0, np.log((1.0 - pp) / one_minus_u) / beta)
    v_new = np.where(quad, a * (np.sqrt(b2) + zv) ** 2, v_exp)
```
- 45–49: the exact CIR conditional mean and variance (7). E = e^{−κΔ} is exact, so unlike Euler there is no κΔ < 1 limit. Line 49 is the branch mask.
- 50–55: `np.where` evaluates both branches on every path. The quadratic branch uses min(ψ, ψc), so the √ never sees ψ > 2. The exponential branch uses max(ψ, ψc), so p ≥ 0. Without this, the unused branch would produce NaNs (pitfall 3).
- 56–58: inverse-cdf sampling. U ≤ p is the same as 1−U ≥ 1−p, which gives the atom at 0. The code uses `ndtr(-zv)` because `1 - ndtr(zv)` is exactly 0 for Z ≳ 8.3, which would give v = ∞ (pitfall 4). One Z_v drives both branches, so the random streams stay aligned under parameter bumps (common random numbers).

**3b. Log-forward step with the martingale correction: `mc.py:60-71`**
```python
    c = dt * (k * rho / xi - 0.5)  # eq. (10)
    K0, K1, K2 = -rho * k * th * dt / xi, 0.5 * c - rho / xi, 0.5 * c + rho / xi
    K3 = K4 = 0.5 * dt * (1.0 - rho**2)
    if martingale:  # eqs. (11)-(12): E[e^{x_{n+1} - x_n} | v_n] = 1
        A = K2 + 0.5 * K4
        if A > 0 and np.any(np.where(quad, 2.0 * A * a >= 1.0, A >= beta)):
            raise ValueError("QE-M: M(v_n) infinite (A >= 1/(2a) or A >= beta), eq. (13)")
        with np.errstate(invalid="ignore", divide="ignore"):  # unselected branch only (checked)
            ln_m_quad = A * b2 * a / (1.0 - 2.0 * A * a) - 0.5 * np.log1p(-2.0 * A * a)
            ln_m_exp = np.log(pp + beta * (1.0 - pp) / (beta - A))
        K0 = -np.where(quad, ln_m_quad, ln_m_exp) - (K1 + 0.5 * K3) * v
    x_new = x + K0 + K1 * v + K2 * v_new + np.sqrt(K3 * v + K4 * v_new) * zp
```
- 60–62: K0..K4 with γ1 = γ2 = ½ (the trapezoid for ∫v). The ∓ρ/ξ terms turn the realised variance move into the correlated forward shock. 64: A is the coefficient of v_{n+1} once Z⊥ is integrated out; ½K4 comes from the Gaussian MGF.
- 65–66: M can be infinite only if A > 0, i.e. ρ > 0. The code checks this per path and raises; it never falls back silently (eq. 13).
- 67–71: ln M is computed in log form (`log1p`), because |A| ~ |ρ|/ξ can be large (pitfall 6). `errstate` silences only the unselected branch, and `heston_steps` raises on any non-finite state (`:122-123`). K0 becomes a per-path array K0*(v_n). Line 71 has no Z_F: `zp` is Z⊥.

**3c. Euler-FT step and the estimator: `mc.py:30-34`, `:156-157`, `:182-187`**
```python
    vp = np.maximum(v, 0.0)
    sq = np.sqrt(vp * dt)
    zf = p.rho * zv + np.sqrt(1.0 - p.rho**2) * zp  # eq. (4)
    v_new = v + p.kappa * (p.theta - vp) * dt + p.xi * sq * zv  # eq. (5), v may go negative
    x_new = x - 0.5 * vp * dt + sq * zf  # eq. (6)
    # ... mc_estimate()
    n = g.shape[-1]
    return DF * g.mean(axis=-1), DF * g.std(axis=-1, ddof=1) / np.sqrt(n)
    # ... european_price()
    rng = np.random.default_rng(seed)
    x, _ = simulate(p, uniform_grid(T, n_steps), n_paths, rng, scheme)
    FT = F * np.exp(x[0])
    g = np.maximum(cp[..., None] * (FT - K[..., None]), 0.0)
    price, se = mc_estimate(g, DF)
    return price[()], se[()], int(n_paths), int(seed)
```
- 30–34: full truncation. v⁺ is used in the drift, the diffusion and the x-step, while the stored v keeps its negative value as the state. Z_F is built by Cholesky (row 0 = Z_v, row 1 = Z⊥). The left-point v⁺ makes the forward an exact martingale.
- 156–157: the sample mean and the `ddof=1` SE along the path axis. DF multiplies once, since it is deterministic under Q^T.
- 182–187: a fresh `default_rng(seed)`: no global state, bit-reproducible, and the seed is returned. F_T = F·e^{x_T}, and x does not depend on F. `cp[..., None]` prices all strikes on one path set (common random numbers), so on-path parity holds to 1e-12 and calls are monotone and convex in K.

## 4. Validation story (independent validator: own QE/QE-M/Euler with a different random layout; COS references checked against Lewis)
| Check | Measured | Tolerance |
|---|---|---|
| QE and QE-M vs COS: FO, GS, BK-2, A-I (ν_F = 0.38, 0.62, 0.36, 0.04). Puts at K = 70/100/140, 2e5 paths, Δ = 1/32, seeds 20261107+i fixed a priori | **24/24 pass**. COS refs within 1e-6 of Lewis inside the test (SE ~ 1e-2) | 3 SE |
| Cherry-pick check: 20 fresh seeds declared before the run × 4 sets × {qe, qe-m} × 3 K, ref = Lewis. Plus the validator's own QE-M (U first, Z_v = Φ⁻¹(U)) | 480 z: max \|z\| **2.86**, none > 3; pooled mean −0.115, sd 0.937. Own QE-M: max \|z\| 1.35 | 3 SE |
| Euler bias vs Δ, ATM call, 1e6 paths (figure: bars ±3 SE, hollow markers < 3 SE) | A-I: +6.368 (Δ = 1) → +0.268 (1/32, 19.6 SE) → +0.127 (1/64). FO: +0.692 (½) → +0.041 (1/32, 5.0 SE) → +0.021 (1/64, 2.6 SE, noise). Fitted order **0.945 / 1.054** | monotone until noise |
| Andersen Table 2 (A-I, K = 100): ours vs published | Euler: +6.368 / +6.394 (Δ = 1), +1.042 / +1.051 (1/8), +0.268 / +0.243 (1/32). Validator at 5e6 paths, Δ = 1: QE +1.0121 (0.0058) vs +1.022 (0.013); QE-M +0.2245 vs +0.233 | qualitative |
| E F_T = F for Euler-FT and QE-M (coarse Δ, plus ρ = +0.5, ξ = 1.5, ν_F = 0.004, ρ = −0.99). Power check: plain QE must fail (FO, one step Δ = 1, 2e5 paths; A-I Δ = 1 from the CSV) | All pass. Plain QE: F̄_T − F = **+12.8 SE**. A-I: QE +0.385 (SE 0.035) vs QE-M +0.009 (0.034) | 3 SE |
| Same paths: C − P = DF(F̄_T − K), calls non-increasing and convex in K. QE E v_T and E v_T² at 4–10 steps; ξ = 1e-4 discrete limits vs Black | < 1e-12 / pass | 1e-12 / 3 SE |

- **Euler fails the 3 SE gate by design.** At gate settings it measures +4.4 and +5.1 SE on A-I (D-018). The note predicted this before any run (published bias 0.243 vs an estimated 3 SE ≈ 0.09). That is scheme bias, so Euler is judged by the Δ-figure; the gate was not loosened.
- **Two typos in the Andersen (2006) working paper were caught in the derivation.** Eq. (42) prints M = β(1−p)/(β−A), dropping the atom p, and the K0* display omits v_n. Both were cross-checked with QuantLib and Choi–Kwok (2024). The sanity check A = 0 ⇒ M = 1 exposes the first typo, which gives 1 − p.
- **Two ~3 SE points in the figure (validator F5), re-run at 4e6 paths.** A-I QE at Δ = 1/8 (−3.1 SE) was noise: −0.0031 (0.0067). FO QE-M at Δ = ½ (−3.0 SE) is a real coarse-step bias: −0.0236 (0.0039). No MC code defect was found.

## 5. Ten likely questions (basic → hard)
1. **What does `(price, stderr, n_paths, seed)` give you? Why are the gate options puts?** The SE comes from the CLT, and the seed makes every result bit-reproducible: a failure is a reproducible event to diagnose, never something to re-roll. The CLT needs Var g < ∞. Puts are bounded, but a Heston call
   needs E F_T² < ∞, which can fail (moment explosion), and then the SE itself is unreliable. Price the OTM side and get ITM options by parity, using a martingale scheme only.
2. **Why simulate x = ln(F_t/F) under Q^T instead of S with r and q?** With deterministic rates, Q^T gives F the same Heston dynamics, with −½v as its only drift. The kernel needs no r or q, DF is applied once, and E F_T = F becomes a test.
   x does not depend on F, so all strikes and any F-bump reuse the same paths. Phase 4 path observations are S_t = F_0(t)e^{x_t} (eq. 17).
3. **Why does Euler go negative, and why full truncation rather than reflection or absorption?** For finite Δ, the Gaussian increment ξ√(vΔ)Z can exceed v. FT keeps the untruncated state and uses v⁺ in the coefficients. "Below zero" there is no vol and κθ pulls the path back up, like the true boundary.
   Absorption and reflection both add variance and over-price. Lord–Koekkoek–van Dijk, BK-2 at 20 steps per year: FT 0.050, absorption 2.102, reflection 4.360.
4. **Explain QE. What is ψ, and why ψc = 1.5?** QE matches the exact CIR conditional mean and variance with a two-parameter law. For small ψ (away from 0) a scaled χ²₁ fits. Near 0 the true law piles up at zero, so QE uses an atom plus an exponential.
   The quadratic branch needs ψ ≤ 2 and the exponential needs ψ ≥ 1, so any ψc in [1, 2] works; 1.5 is Andersen's choice. Tested: QE reproduces E v_T and E v_T² at 4–10 steps.
5. **Why not log-Euler for x on top of QE variance, with corr(Z_F, Z_v) = ρ?** v_{n+1} is non-linear in Z_v, so a Gaussian Z_F correlated with Z_v leaks correlation and makes the wings too thin (Andersen §4.1). Instead, integrate dv: the correlated shock is
   (ρ/ξ)(v' − v − κθΔ + κ∫v), so leverage enters through K2 v_{n+1}, plus an independent Z⊥. Each step draws one Z_v and one Z⊥, in the same order in every scheme.
6. **(Hedging) How would you get delta and vega from this engine?** `mc.py` has no Greeks yet. (i) Bump-and-revalue with CRN (same seed, same draw order): the variance is ~1/(Nh²) with independent seeds and ~1/N with CRN for Lipschitz payoffs. Delta is clean, because a bumped F reuses identical x-paths (prices are homogeneous in (F, K) to 1e-14).
   (ii) Pathwise: Δ_F = DF·mean(e^{x_T}1{F_T > K}), unbiased for Lipschitz payoffs but wrong for digitals and barriers. (iii) Likelihood ratio: handles discontinuities, but has higher variance and needs a density. Vega (bump v0 or θ) is noisy, because QE branch switches and the atom at 0 are discontinuous (pitfall 5).
7. **(Risk) Plain QE gives E F_T ≠ F. Why does a desk care, and why is Euler immune?** The forward is the hedge instrument. A model that misprices it breaks parity against the market, makes ITM prices drift, and leaks its bias into an F_T control variate. Measured at Δ = 1 on A-I: F̄_T − F = +0.385 (SE 0.035)
   for QE vs +0.009 (0.034) for QE-M; FO with one step: +12.8 SE. Euler is a left-point v⁺ times an independent Gaussian, so the mean is exactly 1. QE uses a trapezoid for ∫v and a two-moment law for v', and an exponential moment at A ≈ ρ/ξ depends on the whole law.
8. **(Hard) Derive QE-M. When does it break?** Integrating out Z⊥ gives E[e^{Δx} | v_n] = e^{K0 + (K1 + ½K3)v_n}·E[e^{A v'} | v_n], with A = K2 + ½K4; setting this to 1 gives K0*. M comes from the χ²₁ MGF (needs 2Aa < 1) or from the mixture (needs A < β).
   If ρ ≤ 0 then A ≤ 0, which covers SPX fits. If ρ > 0 the step size is limited (roughly ρξΔ < 2), and the code raises (`mc.py:65-66`). Do not copy the working paper's eq. (42) or its K0* display (§4).
9. **(Hard) What is "24/24 within 3 SE" worth?** P(at least one false fail over 24 | correct code) ≈ 6%, and the power to detect a 3 SE bias is 50%. So the gate bounds the bias to a few SE at 2e5 paths; it does not prove unbiasedness (F5: −0.0236 at Δ = ½ was visible only at 4e6 paths).
   Against cherry-picking: seeds were fixed a priori and never re-seeded, and 480 fresh-seed z-scores gave max 2.86 (480 × 0.27% ≈ 1.3 expected beyond 3). Duffie–Glynn: with an order-1 bias, halving Δ should come with 4× the paths.
10. **What would you change for production?** Make QE-M the default with a per-product Δ-refinement check, and keep Euler as a cross-check only. Add variance reduction (Phase 6: antithetics, F_T or Black control variates on martingale schemes, conditional MC given the v-path, Sobol plus a Brownian bridge).
    Compute Greeks pathwise or by AAD in C++, with smoothing for digital and barrier features. Use counter-based or `SeedSequence.spawn` streams, independent of thread count. For ρ > 0, sub-step and log instead of raising. Pure Heston misses short-dated SPX smiles, so exotics need stochastic-local vol.

## 6. Traps: do NOT say
1. **"It passed 3 SE, so the MC is unbiased / proven correct."** A pass bounds the bias only at that N (50% power at 3 SE), and 24 gates false-fail ~6% of the time. FO QE-M at Δ = ½ has a real −0.0236 (0.0039) bias, while A-I QE at −3.1 SE was noise. Say "statistical evidence at this resolution, seeds fixed a priori".
2. **"Negative variance? Take |v| (reflection) or floor the state at 0 (absorption)."** Both inflate variance and over-price: absorption 2.102 and reflection 4.360, against 0.050 for FT (LKD, BK-2). FT keeps the untruncated state and uses v⁺ only inside the coefficients (`mc.py:30-34`).
3. **"Euler is fine because Feller holds", or "QE is exact".** SPX-like fits violate Feller (FO 0.38, A-I 0.04), and a Gaussian Euler step can go negative at finite Δ even when Feller holds. Euler-FT on A-I is +0.268 (19.6 SE) at Δ = 1/32. QE matches only two moments of v: +0.986 bias at Δ = 1 on A-I.
