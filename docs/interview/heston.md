# Phase 2: Heston characteristic function, COS and Lewis Fourier pricing
Sources: `src/qpricer/models/heston.py`, `src/qpricer/engines/fourier.py`, `scripts/phase2_gates.py`, `docs/derivations/heston.md` (eq. numbers), D-015, D-017, D-019..D-021, `docs/validation/phase2.md`, `docs/OPEN_ISSUES.md`. MC (Euler, QE): `mc.md`.

## 1. Pitch (30 s)
Phase 2 prices European options under Heston, the affine stochastic-variance model, from one characteristic function and two independent Fourier engines. The CF is in the Albrecher et al. (2007) "little trap" form,
written without the β − d subtraction and with a real log1p, so it stays on a continuous branch at any maturity and keeps full accuracy as ξ → 0 (D-019). COS (Fang–Oosterlee 2008, forward-centred as in Le Floc'h 2020) is the fast
engine for Phase 5 calibration: it prices puts on a cumulant range, gets calls by parity, evaluates φ once per expiry and checks its own (N, L) (D-017), in 0.30–0.62 ms for 25 strikes. Lewis (2000), a single integral on Im u = −½,
has no range and no series, so it is the independent check, and it exposed COS's truncation error. Measured: COS vs Lewis ≤ 1.8e-10 on the gate sets (gate 1e-6), ξ → 0 ≤ 5.6e-8 (gate 1e-5), F&O's ATM benchmark to 1.6e-8.

## 2. Key formulas
1. **Dynamics (§1), T-forward measure** `dF = √v F dW^F`, `dv = κ(θ − v)dt + ξ√v dW^v`, `d⟨W^F, W^v⟩ = ρ dt`: v0 spot variance, θ long-run variance, κ mean reversion (1/yr), ξ vol-of-vol, ρ spot–vol correlation (< 0 on SPX); API rejects κ − ρξ ≤ 0.
2. **CF, little trap (1)–(2)** `φ(u) = E[e^{iuX}] = exp(C + v0·D)`, X = ln(F_T/F), `β = κ − iρξu`, `A = u² + iu`, `d = √(β² + ξ²A)`, `g = (β − d)/(β + d)`, `D = (β − d)/ξ² · (1 − e^{−dT})/(1 − g e^{−dT})`,
   `C = (κθ/ξ²)[(β − d)T − 2 Log((1 − g e^{−dT})/(1 − g))]`. Only e^{−dT} (|·| < 1) appears and the Log argument never winds around 0. Heston 1993 (g → 1/g, e^{+dT}) is algebraically equal, but its Log jumps by 2πi unless ν_F ∈ ℕ.
3. **COS put (8)–(11)** `[a, b] = c1 ± L√c2`, `u_k = kπ/(b − a)`, `P = DF Σ'_{k<N} Re[φ(u_k)e^{−iu_k a}]·V_k`, `V_k = 2/(b − a)[Kψ_k(z̄) − Fχ_k(z̄)]`, `C = P + DF(F − K)`: c1 = −Σ̄/2, c2 = Var X (eq. 7, not F&O's misprint),
   Σ' halves k = 0, z̄ = ln(K/F) clipped to [a, b], and χ_k, ψ_k are the closed-form integrals of e^y and 1 against cos(u_k(y − a)).
4. **Lewis (14)** `C = DF[F − (√(FK)/π) ∫₀^∞ Re[e^{iux} φ(u − i/2)] du/(u² + ¼)]`, x = ln(F/K), `P = DF[K − same]`: the contour Im u = −½ needs only E[F_T^{1/2}] < ∞, which always holds.
5. **ξ → 0 (16)** `σ_eff² = Σ̄/T = θ + (v0 − θ)(1 − e^{−κT})/(κT)`; `V_H = V_B(σ_eff) + ρξ·J·∂x∂Σ V_B + O(ξ²)`, `J = (θ/κ)(T − τ_κ) + ((v0 − θ)/κ)(τ_κ − Te^{−κT})`, τ_κ = (1 − e^{−κT})/κ,
   `∂x∂Σ V_B = −DF·F·n(d1)·d2/(2Σ̄)`. For ρ ≠ 0 the gap to plain Black is linear in ξ. Short-T consequence: `δσ ≈ ρξ ln(K/F)/(4√v0)` (⚑, single source).
6. **Feller ratio (§1)** `ν_F = 2κθ/ξ² ≥ 1` ⇒ v never reaches 0. It is a path property: the CF is the same either way, so only MC cares. FO has 0.38 and GS 0.62, both gated. Eq. (17): E[(F_T/F)^ω] < ∞ iff T < T*(ω), and no ω ∈ [0, 1] explodes.

## 3. The code that matters
**3a. CF: `src/qpricer/models/heston.py:69-77`. Lewis integrand: `src/qpricer/engines/fourier.py:140-141`**
```python
    beta = k - 1j * rho * xi * u
    A = u * u + 1j * u
    d = np.sqrt(beta * beta + xi * xi * A)
    s = beta + d
    e1 = -np.expm1(-d * T)
    q = xi * xi * A * e1 / (2.0 * d * s)
    D = -A * e1 / (2.0 * d * (1.0 - q))
    C = -k * th * A * T / s - (2.0 * k * th / (xi * xi)) * _log1p(-q)
    return np.exp(C + v0 * D)
    # ... fourier.py, lewis_price:
    def integrand(u: float) -> NDArray:
        return np.real(np.exp(1j * u * x) * cf(u - 0.5j, T, p)) / (u * u + 0.25)
```
- 69–71: eq. (1). Principal √ gives Re d ≥ 0, and Re d² > 0 on both contours, so d is continuous. 73: only e^{−dT}, via `expm1`, so no overflow (Heston's e^{+dT} overflows). 74: β − d = −ξ²A/(β + d) removes the cancelling
  subtraction, and 1 − q = (1 − ge^{−dT})/(1 − g) is the little-trap Log argument, a ratio rather than a difference of logs. 75–76: eq. (3). `_log1p` (:49-52) is ½ log1p(2Re w + |w|²) + i atan2(Im w, 1 + Re w), so the O(ξ²) Log
  keeps relative accuracy under the 2κθ/ξ² factor (D-019: at ξ = 1e-6 the CF equals Black's to < 1e-10). Lewis: the same `cf` at u − i/2, where A = u² + ¼ is real. There is no 1/u at 0, unlike Gil-Pelaez. One `quad_vec` call covers all strikes at 1e-11 and raises unless status 0 (:143-147).

**3b. COS put kernel: `src/qpricer/engines/fourier.py:58-68`**
```python
    a, b = cos_range(T, p, L)
    u = np.arange(N) * np.pi / (b - a)
    w = np.real(cf(u, T, p) * np.exp(-1j * u * a))
    w[0] = 0.5
    z = np.minimum(np.maximum(np.log(K / F), a), b)[..., None]
    arg = u * (z - a)
    ez = np.exp(z)
    chi = (ez * np.cos(arg) + u * ez * np.sin(arg) - np.exp(a)) / (1.0 + u * u)
    u1 = np.where(u > 0, u, 1.0)
    psi = np.where(u > 0, np.sin(arg) / u1, z - a)
    return DF * (2.0 / (b - a)) * ((K[..., None] * psi - F * chi) @ w)
```
- 58–61: the range is for X = ln(F_T/F), so it does not depend on the strike and φ(u_k) is computed once per expiry. w_k are the density's cosine coefficients, and w_0 = ½ is the halved first term (φ(0) = 1). 62: the put pays K − Fe^y for y < z = ln(K/F).
  Clipping z sets the integration limit: z ≤ a gives P = 0 exactly, z ≥ b gives ≈ DF(K − F), and without it deep strikes go wrong or even negative. 63–67: χ, ψ in closed form; `u1` is a dummy because `np.where` evaluates both branches. 68: one (M × N) @ N product.

**3c. Self-checking (N, L), then calls by parity: `src/qpricer/engines/fourier.py:97-109, 117`**
```python
    if N is None and L is None:
        N, L = 256, 18.0
        put = _cos_put(F, K, T, DF, p, N, L)
        while True:
            N, L, prev = 2 * N, L + 6.0, put
            put = _cos_put(F, K, T, DF, p, N, L)
            gap = float(np.max(np.abs(put - prev), initial=0.0))
            if gap <= tol * F:
                break
            if N >= N_MAX:
                msg = f"COS not converged: |P(N) - P(N/2)| = {gap:.1e} > tol F at N = {N}"
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
                break
    # ... :110-116 explicit N or L: one evaluation, eq. (13) series warning only
    return np.where(cp > 0, put + DF * (F - K), put)[()]
```
- 98–105: start at (256, 18), then double N and add 6 to L. Doubling N squares the series error and +6 on L cuts the range error 20–500× (measured), so the max-over-strikes gap estimates the coarser price's error. It accepts at 1e-9·F (1e-7 at F = 100,
  10× inside the gate) and returns the finer price. 106–109: at N_MAX = 2^14 it warns, so it never misses silently. 117: calls by parity. The put payoff is bounded by K, while a direct COS call has V_k ∝ e^b, which amplifies every error (F&O Remark 5.2).

## 4. Validation story
The independent validator wrote 94 tests, with references that use no qpricer code: its own RK4 Riccati CF plus Gil-Pelaez quadrature ("RIC+GP"), Heston 1993 with phase unwrapping, and its own Black. **Its report predates fix round 1.** The "default" numbers
below are the implementer's (`scripts/phase2_gates.py`, D-017).

| Gate / check | Measured | Tolerance |
|---|---|---|
| COS vs Lewis, FO / GS / LW / AT2, K 60..150 × T {0.1..3} × DF {1, 0.95} × C, P | default 1.8e-10 / 1.4e-12 / 1.1e-13 / 5.1e-12; fixed (256, 18), validator 1.01e-7 / 2.48e-9 / 1.10e-13 / 9.82e-9; true error vs RIC+GP: COS ≤ 1.01e-7, Lewis ≤ 1.62e-9 | 1e-6 |
| Random scan: 300 SPX-plausible sets × 6 T, puts | fixed (256, 18): 413/1800 cells (22.9%) > 1e-6, worst 9.95e-2, 5 silent (max 2.61e-6); default: 0/1800, 0 warned, max 3.05e-9 | 1e-6 |
| ξ = 1e-4: ρ = 0 vs Black(σ_eff), incl. v0 ≠ θ / ρ = −0.7 vs Black + ρξJ term (D-015) | ≤ 4.36e-8 / ≤ 5.6e-8 (gap to plain Black 4.06e-5 / 3.27e-4 / 6.05e-4 at T = 0.1 / 1 / 3) | 1e-5 |
| `cf` vs own Riccati ODE (FO, KJ, A-I to T = 10; ξ = 1.5; ρ = +0.5, −0.99) | ≤ 1.1e-12 abs | — |
| Published: F&O FO1/FO2; QuantLib–Lewis L1–5; Broadie–Kaya BK-1/2 | ≤ 1.6e-8; 3e-14; ≤ 4.2e-5 | 5e-8; 1e-9; 4 dp |
| QE MC vs COS (`mc.md`) | 24/24 within 3 SE; 480 fresh-seed z-scores, max 2.86 | 3 SE |

- **L = 12 was range-limited, not series-limited (D-017).** It gave 3.95e-5 (FO) and 2.58e-6 (GS). The error did not move from N = 256 to 4096, but fell about 8× per +2 in L. At FO, T = 1, ATM it is exactly the even-reflection tail of §3.3: computed 3.948e-5,
  measured −3.947e-5. Q(X < a) is only 7.7e-6 at "12 sd", but the reflected weight grows like e^{−y} into the heavy ρ < 0 left tail. First fix: a fixed L = 18 (FO 1.01e-7).
- **Fixed L = 18 was not robust (validator F1).** On the random scan, 22.9% of cells were > 1e-6, and 5 were silent because the eq. (13) warning models series error only. SILENT at T = 1: 2.61e-6 (N = 256), 2.51e-6 (N = 4096), 2.09e-8 (L = 24), 9.28e-10 (L = 30).
  Self-check: 0/1800, max 3.05e-9. Median cost 768 CF evaluations (p90 3840); 0.30 / 0.62 / 8.0 ms on GS / FO / worst cell, vs 0.12 ms fixed and 4.9 / 4.6 / 9.6 ms for Lewis.
- **The ξ → 0 gate was mis-specified for ρ ≠ 0 (D-015).** The gap to plain Black is linear in ξ: 3.27e-3 / 3.27e-4 / 3.25e-5 at ξ = 1e-3 / 1e-4 / 1e-5 (ρ = −0.7, T = 1), but 2.6e-8 at ρ = 0. The tolerance stayed at 1e-5 and the reference gained the
  eq. (16) term. The validator's own pricer shows the same gap (to 1e-9), and its numerical first-order coefficient matches ρξJ∂x∂ΣV_B to 1.6e-10.
- **F&O Table 11 c2 is misprinted.** ū(6e^{−λT} − 7) should read ū(4e^{−λT} − 5); the printed version is too small by ξ²θ(1 − E)/(4κ³) and can go negative, hence F&O's |c2|. This is tested to 1e-13, and c2 matches the moment ODEs to ≤ 2.7e-15 rel.
- **AMST Tables 1/2 are not reproducible at stated precision (OPEN_ISSUES, D-020).** A1: +0.0043, +0.0073, −0.0007, +0.0052, +0.0019 at T = 1, 2, 3, 5, 10 vs 0.005; A2: +6.9e-5 at T = 1 vs 5e-5. No single r fits, and rounding the 4-digit parameters
  moves prices by ±0.0057…0.0117, so AMST most likely used unrounded parameters. These values are not hard-tested; the trap is tested by the form-(4) negative control. Also (D-021): small-κ c2 −2.25e-2 → ≤ 8.1e-6, and `cp` is now required.

## 5. Ten likely questions (basic → hard)
1. **What does each parameter do to the smile?** v0 sets the short ATM level and θ the long one. κ sets how fast vol moves from √v0 to √θ (v0's weight in σ_eff² is (1 − e^{−κT})/(κT)) and how fast skew decays. ρ < 0 (leverage) tilts the smile into a skew at
   first order in ρξ. ξ adds curvature (both wings) at second order and scales the skew. As ξ → 0 you get Black at σ_eff, with an O(ξ) correction unless ρ = 0.
2. **Why Fourier, and why two engines?** The model is affine: φ is closed form (Riccati ODEs) and the density is not. COS reads the density's cosine coefficients off φ, and the payoff integrals are closed form, so a price is one mat-vec, with exponential convergence (|φ| ~ e^{−cu}).
   Lewis uses another contour plus adaptive quadrature, with no range and no series, so its errors are unrelated to COS's. Cross-engine parity (Lewis C − COS P = DF(F − K) < 1e-6) is tested; parity inside one engine is tautological.
3. **What is the little Heston trap, and how do you know your CF is right?** The forms are algebraically equal, but in Heston's the Log argument spirals around 0 as u grows. The principal Log then jumps by 2πi, multiplying φ by e^{∓4πiκθ/ξ²}: AMST report 29.83 instead of 33.84
   at FO ATM, T = 10. Evidence: independent RK4 Riccati ≤ 1.1e-12 (incl. KJ, ν_F = 0.08, T = 10), form (4) as a negative control that must fail at FO T = 10, and FO1/FO2 to 1.6e-8.
4. **Why not compare the ξ → 0 limit with plain Black?** Expand the Riccati in ξ: ln φ = −½AΣ̄ + ρξ(iu)(−½A)J + O(ξ²), where −½A acts as ∂_Σ and iu as ∂_x, which is eq. (16). At ρ = −0.7, ξ = 1e-4 the gap is 3.27e-4 ≫ 1e-5: a spec error, not a tolerance problem.
   ρ = 0 vs Black gives 4.36e-8 and ρ = −0.7 vs Black + ρξJ gives 5.6e-8. Shrinking ξ to 1e-7 would only test round-off in κθ/ξ².
5. **FO and GS violate Feller (0.38, 0.62). A problem?** Not for Fourier: the CIR transform is the same whether 0 is attainable or not (it is instantaneously reflecting). It matters for simulation, where v can hit 0, hence QE: 24/24 within 3 SE vs COS and max |z| 2.86
   over 480 fresh seeds (`mc.md`). Never impose Feller in calibration: SPX fits violate it, and forcing it wrecks the short-dated skew. Report ν_F.
6. **(Hedging) Spot drops: what does the smile do, and what delta do you trade?** Prices are homogeneous in (F, K) (checked to 1e-9), so at fixed v the smile slides with the forward. But the regression of dv on dF/F is ρξ < 0, so vols rise. The minimum-variance hedge is
   Δ_MV = ∂V/∂F + (ρξ/F)∂V/∂v, below the model delta for calls. At short T: ATM skew ρξ/(4√v0), and ATM vol moves by ρξ/(2√v0) per unit ln F, so the skew-stickiness ratio is about 2 (hand derivation, not gated).
7. **(Risk) How do you vega-hedge a Heston book?** There is no single vol: the risk sits in (v0, θ, κ, ξ, ρ), with one factor v driving the whole surface. A v0 shock moves σ_eff² by (1 − e^{−κT})/(κT), most at short T, so short and long vols cannot move independently in the model.
   Practice: bucketed vega (bump market vols per bucket, recalibrate, re-price) hedged with vanillas, roughly v0 with short ATM, θ with long ATM, ρ with risk reversals, ξ with butterflies. ∂P/∂v0 has a COS form (eq. 12), not yet gated.
8. **(Hard) Why is Heston's short-dated skew too flat for SPX?** As T → 0 the skew tends to the finite ρξ/(4√v0) (⚑): diffusive vol, no jumps; at long T it decays like 1/T through mean reversion. Market short-end skew keeps steepening (stylised fact; Phase 5 will measure).
   A whole-surface fit pushes ξ and |ρ| up to buy short skew, ν_F falls far below 1, and the fit compromises across T. Fixes: jumps (Bates), time-dependent parameters, rough vol, SLV for exotics.
9. **(Hard) How do you know a COS price is converged? Is the self-check a proof?** Tell the L = 12 story (error flat in N, falling with L: the reflection tail), then F1. The self-check is an a posteriori estimate that assumes each refinement cuts the error a lot (20–500× measured);
   it is not a bound, and a Chernoff tail bound via eq. (17) was rejected on cost. Adaptive (N, L) makes the price piecewise at ~1e-9·F, which can corrupt FD Jacobians, so pin (N, L) per expiry in Phase 5 and re-check Lewis at the optimum.
10. **What would you change for production?** A C++ CF kernel with analytic parameter gradients (C, D explicit; e.g. Cui–del Baño Rollin–Germano 2017) or AAD, for an exact calibration Jacobian and Greeks. (N, L) pinned per fit and guarded by Lewis. A rigorous range (cross-checked c4
    or an eq. 17 bound). Gated Greeks (eq. 12 is ⚑). Bates/SLV for the short end. A validator re-run after fix round 1.

## 6. Traps: do NOT say
1. "COS converges exponentially, so F&O's N = 256, L = 12 gives 1e-6." L = 12 gave 3.95e-5 on FO whatever N was, and fixed (256, 18) missed on 22.9% of random cells (5 silently). Say: "< 1e-6 on the 4 gate sets; self-checking default 0/1800, max 3.05e-9".
2. "Heston with ξ → 0 is Black at the average variance" (only at ρ = 0), or "I loosened the ξ gate". At ρ = −0.7, ξ = 1e-4, T = 1 the gap is 3.27e-4, linear in ξ; the tolerance stayed at 1e-5 and the reference gained the ρξJ term (D-015).
3. "Feller violation invalidates the formula", "Heston's 1993 formula is wrong" or "we reproduced AMST". The CF is blind to Feller, and Heston's form is correct algebra (only its principal-branch evaluation jumps). AMST misses its stated precision on 3 of 8 values.
