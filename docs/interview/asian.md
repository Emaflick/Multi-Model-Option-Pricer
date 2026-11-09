# Phase 4: Arithmetic Asian options (discrete Kemna–Vorst geometric closed form, geometric control variate)
Sources: `src/qpricer/engines/analytic.py`, `src/qpricer/engines/exotic_mc.py`, `tests/test_asian.py`, `docs/validation/gate_outputs/phase4_gates.txt` (post-fix gate run, **gates.txt**), `docs/validation/phase4.md` (validator, incl. re-validation after fix round 1), `report/figures/asian_cv_vr.{png,csv}` (**CSV**), `docs/derivations/asian.md` (eq. numbers; [FM08] = Fusai–Meucci 2008), D-025, D-030.
Tags: **[GK]** = general market knowledge or literature, not a result of this repo (no Asian Greek is implemented). **(arith.)** = simple arithmetic on repo numbers, done for this sheet.

## 1. Pitch (30 s)
An arithmetic Asian pays (A − K)⁺ on the average A of n fixings. The average of lognormals has no closed form, so I price it by MC. The one closed form I do have, the discrete geometric Asian (Kemna–Vorst on the actual dates), is used twice: as the oracle for the MC and as a control variate. ln G is a sum of Gaussians, so G is lognormal: a Black call with forward F_G and total vol σ_G. AM and GM differ only at second order in the spread of the fixings, so the two payoffs are almost perfectly correlated. On [FM08], ρ̂ = 0.99918–0.99977 (≈ 0.9997 at and below the money), which cuts the variance by 612× to 2202× on the same paths. Measured (gates.txt): geometric MC vs closed form, max |z| 1.90 over 39 cells; closed form vs [FM08] Table 4 to 1.28e-4; CV prices vs [FM08] Table 6 within 3 SE without the 3e-4 allowance. The CV is BSM-only by design (D-030). Under Heston, the default time step was too coarse (F2); that was found and fixed.

## 2. Key formulas
1. **Fixing forwards (1)–(2)** `E^T[S_{t_i}] = F_i := F(0, t_i)`, so `E A = Ā_F = (1/n)Σ F_i`. The price depends on the whole forward curve (dividend and carry timing), not just F(0, T). With constant carry b, ∂F_i/∂b = −(T − t_i)F_i.
2. **Geometric moments (7)–(9)** `σ_G² = (1/n²)Σ_{i,j} V(min(t_i,t_j)) = Σ_k w_k²ΔV_k`, with w_k = (n−k+1)/n and V the total variance. `F_G = Ḡ_F·exp(−½(V̄ − σ_G²))`, where Ḡ_F = (ΠF_i)^{1/n} and V̄ = mean V(t_i). So F_G ≤ Ḡ_F ≤ Ā_F: the forward of G carries a convexity discount.
3. **Price (10)** `V_G = DF·θ[F_G·Φ(θd1) − K·Φ(θd2)]`, `d_{1,2} = ln(F_G/K)/σ_G ± σ_G/2`. This is Black-76 with F → F_G and σ√T → σ_G.
4. **Equal spacing (11)–(13).** With t_0 excluded, `σ_G² = σ²T(n+1)(2n+1)/(6n²)`; with t_0 included, `σ²T(2n+1)/(6(n+1))`. Both tend to σ²T/3, i.e. vol ≈ σ/√3. The convention matters: the two σ_G differ by a factor n/(n+1), 8.3% at n = 12.
5. **CV (14)–(16)** `e_i = Y_i − ĉ(X_i − V_G)`, with ĉ the OLS slope of Y on X. `VR = Var Y / Var e = 1/(1 − ρ̂²)`, exact on the same paths. Fitting ĉ on the same paths gives an O(1/N) bias; c = 1 is exactly unbiased.
6. **Bounds (19), model-free given the F_i.** Calls: `max{DF(Ā_F − K)⁺, C_G} ≤ C_A ≤ min{C_G + DF(Ā_F − F_G), (DF/n)Σ_i c_i}`. Puts: `P_G − DF(Ā_F − F_G) ≤ P_A ≤ P_G`. The lower bounds come from Jensen and AM ≥ GM. The upper bounds come from (A−K)⁺ ≤ (G−K)⁺ + (A−G), and from convexity: a strip of vanillas dominates the Asian.

## 3. The code that matters
**3a. Kemna–Vorst on arbitrary dates: `src/qpricer/engines/analytic.py:112-117, 127-128`**
```python
    dV = np.diff(V, prepend=0.0)
    if not (np.all(np.isfinite(V)) and np.all(dV >= 0)):
        raise ValueError("V must be finite, >= 0 and non-decreasing (sorted dates)")
    n = V.size
    s2 = float(np.sum(((n - np.arange(n)) / n) ** 2 * dV))
    return float(np.exp(np.mean(np.log(F_fix)) - 0.5 * (V.mean() - s2))), s2
    # ... geometric_asian_price:
    FG, s2 = geometric_asian_moments(F_fix, V)
    return bsm.price(FG, K, 1.0, DF, np.sqrt(s2), cp)
```
- **112–114.** Variance increments ΔV_k from the total-variance vector. The dates must be sorted, and a t = 0 fixing has ΔV = 0.
- **116: the Abel form of (8).** Each increment ΔV_k is shared by every later fixing, so its weight is ((n−k+1)/n)². It is tested equal to the double sum and to the (2n−2i+1) form to 1e-12 on 17 random dates.
- **117.** F_G = Ḡ_F·e^{−½(V̄ − σ_G²)}, with Ḡ_F computed as exp(mean log F), so no product can overflow. **127–128.** The Black kernel is called with σ = σ_G and T = 1, since only σ√T matters (its theta is then meaningless). At n = 1 this is Black-76 to 1e-12.

**3b. Running sums and the F2 fix: `src/qpricer/engines/exotic_mc.py:24, 35-38, 165-170`**
```python
MAX_DT = 1 / 32  # Heston sub-step cap between dates, asian eq. (4), autocallable eq. (14) (D-030)
    # ... fixing_grid:
    edges = np.concatenate(([0.0], t))
    m = np.ceil(np.diff(edges) / max_dt * (1.0 - 1e-12)).astype(int)
    ab = zip(edges[:-1], edges[1:], m, strict=True)
    return np.concatenate([np.linspace(a, b, k + 1)[1:] for a, b, k in ab])  # ends exactly on t_i
    # ... asian_averages:
    later = t > 0
    SA, SL = np.full(int(n_paths), F_fix[~later].sum()), 0.0
    steps = obs_steps(model, t[later], n_paths, rng, max_dt, scheme)
    for Fk, (_, x, _, _) in zip(F_fix[later], steps, strict=True):
        SA, SL = SA + Fk * np.exp(x), SL + x
    return SA / t.size, np.exp(np.mean(np.log(F_fix)) + SL / t.size)
```
- **24, 35–38: the Heston grid.** Each fixing interval is cut into ⌈τ_i/Δ_max⌉ equal steps that end exactly on t_i; `heston_steps` recomputes the QE constants for each step size. The factor (1 − 1e-12) avoids a spurious extra step when τ_i/Δ_max is an integer. Under BSM max_dt is ignored, because steps between dates are exact.
- **165–166.** A t = 0 fixing (today's S_0, supplied by the caller) enters as a constant. **168–169.** Only running sums are kept: Σ F_k e^{x_k} and Σ x_k. Memory is O(N), not O(nN); storing paths would take 2 GB per array at n = 250, N = 1e6 (pitfall 7).
- **170.** G = Ḡ_F·exp(Σx/n) is computed as a log-sum and the product ΠS is never formed: at n = 250 it would be about 10^500 and overflow (pitfall 6).

**3c. Control variate and the BSM-only guard: `src/qpricer/engines/exotic_mc.py:180-188, 203-204, 209`**
```python
    y, x = np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64)
    xc, yc = x - x.mean(-1, keepdims=True), y - y.mean(-1, keepdims=True)
    sxx = (xc * xc).sum(-1)
    if np.any(sxx == 0):
        raise ValueError("control variate has zero sample variance")
    c = (xc * yc).sum(-1) / sxx
    e = y - c[..., None] * (x - np.asarray(mean_x)[..., None])
    est, se = mc.mc_estimate(e)
    return est, se, c, y.var(-1, ddof=1) / e.var(-1, ddof=1)
    # ... asian_mc:
    if kind == "cv" and isinstance(model, HestonParams):
        raise ValueError("geometric CV mean is Black-only; use plain MC under Heston (asian §3.5)")
        mean_x = geometric_asian_price(F_fix, float(model) ** 2 * np.asarray(t_fix), K, DF, cp)
```
- **181–185.** ĉ = Cov/Var by OLS, one per strike, on the same paths (paths on the last axis, all strikes at once). If the control never pays (zero sample variance), the call raises. F9, an open nit, suggests falling back to ĉ = 0 per strike instead.
- **186–188.** e is the residual, SE = std(e)/√N, and VR = s_Y²/s_e². VR equals 1/(1 − ρ̂²) to 1e-10 (tested) because ĉ is the OLS slope computed with the same ddof.
- **203–204: the guard.** The control's mean must be known *under the law that generated it*. The Black V_G is not the Heston mean, so the code raises rather than return a biased price with a tiny SE. **209.** The exact mean uses V(t_i) = σ²t_i.

## 4. Validation story
| Gate / check | Measured (gates.txt unless noted) | Tolerance |
|---|---|---|
| G4 geometric MC vs KV: σ {0.2, 0.5} × T {1, 3} × n {1, 12, 52} × K/F {0.8, 1, 1.2}, plus the [FM08] curve, 2e5 paths | max \|z\| 1.90 (39 cells); mean G vs F_G and mean A vs Ā_F: max \|z\| 1.45 | 3 SE |
| KV closed form vs [FM08] Table 4 (n = 12, 50, 250) | 1.28e-4. Validator: one unrounded σ = 0.1780163 fits all 12 values to 3.6e-6 | 3e-4 |
| CV MC vs [FM08] Table 6 (arithmetic) | max (\|d\| − 3SE)/3e-4 = −1.21, i.e. every value inside 3 SE alone. Example n = 12, K = 100: 4.88132 (SE 3.8e-4) vs 4.8821 | 3 SE + 3e-4 |
| CV vs plain arithmetic, independent seeds (σ 0.3, n 12) | max \|z\| 0.77; SE ratio 28.6 / 24.8 / 15.1 at K = 80 / 100 / 120 | 3 SE |
| Heston QE-M geometric, ξ = 1e-4, vs (10) at QE's discrete variance Σ_Q | max \|z\| 0.55 | 3 SE |
| Closed form vs the validator's own double-sum geometric (GEO) | 1e-12 on 20 random date sets and curves | — |

- **Variance reduction (CSV, D-030).** ATM, n = 12, T = 1, for σ = 0.1 / 0.2 / 0.3 / 0.5: VR = 5793 / 1416 / 632 / 211, ρ̂ = 0.99991 / 0.99965 / 0.99921 / 0.99763, ĉ = 1.014 / 1.030 / 1.047 / 1.086. On [FM08] (σ 0.178, n = 12): VR = 1924 / 1608 / 612 at K = 90 / 100 / 110 (ρ̂ 0.99974 / 0.99969 / 0.99918). Plain SE 1.5e-2 becomes CV SE 3.8e-4 at K = 100.
- **What failed first: F2, Heston sub-stepping.** `asian_mc` and `asian_averages` defaulted to `max_dt=None`, which means one QE-M step per fixing period and contradicts eq. (4). The validator tested GS with quarterly fixings and 3e6 paths: default minus Δ = 1/32 gave z = −2.6 / −3.6 / +5.9 at K = 90 / 100 / 110 (+0.0133, i.e. 1.3%, at K = 110). Its own QE-M at Δ = 1/64 agreed with 1/32.
- **F2: why no gate caught it, and the fix.** The Heston Asian tests pass `max_dt` explicitly (`tests/test_asian.py:296-300`), so they never exercised the default. Fix: `MAX_DT = 1/32` is now the default everywhere. After the fix (re-validation, 2e6 paths), default vs 1/32 gives z = −1.08 / −1.25 / −1.14. An explicit `None` still gives +5.36 at K = 110; it is opt-in and documented. Regression test: `test_heston_default_grid_is_substepped`.
- **F7a.** The pathwise *upper* bounds of (19) were untested in round 0 (only Y ≥ X was). They are now tested pathwise at 1e-12 and in expectation at 3 SE (`test_eq19_bounds_pathwise_and_in_expectation`).
- **Tolerances justified, not tuned.** [FM08] prints σ = 0.17801, a rounded value. The 3e-4 tolerance comes from the note's hand check, and the validator's fitted σ = 0.1780163 confirms that the residual is rounding.

## 5. Ten likely questions (basic → hard)
1. **Why is an Asian cheaper than a vanilla, and which input matters most?** Averaging lowers the effective vol: for continuous averaging from inception σ_G² → σ²T/3, slightly more for coarse fixings when t_0 is excluded (eq. 13). The expected average is the mean of the fixing forwards F(0, t_i), so the whole curve matters (carry, dividend timing). The MC and the closed form share F_fix, so no MC-vs-formula gate can catch a wrong curve. The curve builder is therefore tested on its own (`test_forward_curve_builder`: exact at the expiries, log-linear midpoints, n = 1 equals Black on F(0, t_1)).
2. **Derive the geometric closed form.** ln S_{t_i} = ln F_i − ½V(t_i) + M_{t_i}, with M Gaussian and Cov(M_s, M_t) = V(s ∧ t). ln G is their mean, hence Gaussian, with mean ln Ḡ_F − ½V̄ and variance σ_G². Its lognormal mean is F_G, then apply Black. The convexity factor matters: the forward of G is *not* the geometric mean of the forwards. Use the discrete form on the real dates. The continuous Kemna–Vorst formula overstates the n = 12 price by 0.0879 at K = 100 ([FM08] 4.791541 vs 4.703634).
3. **Why does the geometric control work so well?** A/G = 1 + (1/2n)Σ(y_i − ȳ)² + …, so A − G is second order in the spread of the fixings, O(σ²T)F, while the payoffs move at first order, O(σ√T)F. Hence ĉ ≈ 1 and ρ → 1. On [FM08] ATM, ρ̂ = 0.999689 gives VR = 1/(1 − ρ̂²) = 1608. The cost is one extra log-sum per path.
4. **How does VR depend on σ and strike?** ATM, n = 12: 5793 / 1416 / 632 / 211 for σ = 0.1 / 0.2 / 0.3 / 0.5. My heuristic, not in the notes: 1 − ρ² scales like the squared ratio of the second-order to the first-order term, ~σ²T, so VR ~ 1/σ². The CSV ratios are 4.1, 2.2 and 3.0, against (σ ratio)² = 4, 2.25 and 2.8 (arith.). Out of the money VR collapses: at σ = 0.1, K/F = 1.2, VR = 85.5 (ρ̂ 0.9941, ĉ 1.12). Both payoffs are then mostly zero, and the exercise sets {A > K} ⊇ {G > K} differ in relative terms.
5. **Is the CV estimator unbiased?** With ĉ fitted on the same paths, the bias is −Cov(ĉ, X̄) = O(1/N), against an SE of O(N^{−1/2}), so it is negligible. For exact unbiasedness, set c = 1 (e = Y − X + V_G) or fit ĉ on an independent pilot run, as [FM08] does. Empirical check: CV vs plain MC on independent seeds, max |z| 0.77.
6. **What do the eq. (19) bounds buy you?** Model-free brackets given the F_i: C_G ≤ C_A by AM ≥ GM; C_A ≤ C_G + DF(Ā_F − F_G) because the call payoff is 1-Lipschitz; C_A ≤ the average of the fixing vanillas by convexity. They are tested pathwise (1e-12) and in expectation (3 SE) under BSM, and the vanilla-strip bound under Heston with COS vanillas. They catch convention errors (t_0, wrong F_i) that a tight SE would hide. [GK] The strip of n vanillas with notional 1/n each is a static super-hedge.
7. **Why is the CV BSM-only (D-030)?** A CV needs E[X] under the law that generated X. On Heston paths G is not lognormal, so E^H(G − K)⁺ ≠ V_G(σ). Plugging V_G in shifts the estimate by c(E^H X − V_G); that shift does not shrink with N, and the SE stays tiny, so the result is a confident wrong number. The code raises. Valid alternatives: plain MC (the default); a shadow BSM path on the same normals, whose mean is exact (H2, eq. 17, about 10 lines, not implemented because no gate needs it); or a control with a known Heston mean (A itself, or a COS vanilla). A CV reduces noise; it never removes discretisation bias.
8. **Tell me about the Heston Asian defect.** The default ran one QE-M step per quarter, which biased the price by up to 1.3% (z +5.9 at K = 110); fixed by Δ ≤ 1/32. The error flips sign across strikes, which suggests the shape of the distribution was wrong rather than its mean (my reading). QE-M's martingale correction keeps E[S_t] = F_t exactly, so the mean of A was right. One quarter-long step mis-states the joint law of (x, v). Lesson: test the defaults that users actually call.
9. **(Hedging/risk) How do Asian risks evolve over the life?** asian.md §6 states that delta and vega roll off as fixings set: a set fixing is a known number, so sensitivity shrinks with the fraction still to fix. Nothing here is computed; Greeks are out of Phase 4 scope. [GK] At inception it behaves roughly like a vanilla with vol about σ/√3, so its vega is smaller than the vanilla's. Late in the life one fixing moves A by about S/n, so gamma is small and a single bad settlement print matters little. The main repo-visible risk is the forward curve: a change in carry moves each F_i by −(T − t_i)F_i per unit b, which is dividend-timing risk.
10. **What would you change for production?** For Heston or LSV: the H2 shadow-BSM control, or a Heston geometric Asian by Fourier (H4). [GK] Pathwise Greeks, which work for Asians because the payoff is Lipschitz, unlike digitals. Seasoned deals: rescale the strike to K' = (nK − Σ_past)/m (pitfall 12); the API currently rejects past fixings. A per-strike ĉ fallback (F9). [GK] QMC with a Brownian-bridge construction; a C++ kernel. A timing adjustment if rates are stochastic and correlated with S (§3.1, out of scope).

## 6. Traps: do NOT say
1. **"Under Heston I used the geometric CV with the Black geometric price."** That gives a biased price with a tiny SE, which is why the code raises (D-030).
2. **"VR is about 1000"** (with no σ or strike), or **"Kemna–Vorst continuous is the control for monthly fixings".** VR falls from 5793 to 211 ATM as σ goes from 0.1 to 0.5, and to 85.5 deep OTM at σ = 0.1. The control is the *discrete* geometric on the actual dates; the continuous formula is 0.0879 too high at n = 12. Including or excluding t_0 changes σ_G by n/(n+1).
3. **"The Heston Asian was right from the start."** The default ran one QE-M step per quarter and was biased by +1.3% at K = 110 (z +5.9). The validator found it (F2), and it was fixed by setting Δ ≤ 1/32 as the default.
