# Phase 4: Barrier options (Reiner–Rubinstein, Brownian-bridge MC, BGK shift, Heston bridge)
Sources: `src/qpricer/engines/analytic.py`, `src/qpricer/engines/exotic_mc.py`, `tests/test_barrier.py`, `docs/validation/gate_outputs/phase4_gates.txt` (post-fix gate run, cited as **gates.txt**), `docs/validation/phase4.md` (validator, incl. the re-validation after fix round 1), `report/figures/barrier_mc_vs_rr.{png,csv}` (**CSV**), `docs/derivations/barrier.md` (eq. numbers, errata E1–E2), D-027, D-028.
Tags: **[GK]** = general market knowledge or literature, not a result of this repo (no barrier Greek is implemented or gated). **(arith.)** = simple arithmetic on repo numbers, done for this sheet.

## 1. Pitch (30 s)
A barrier option is a vanilla that dies (knock-out) or is born (knock-in) when spot touches a level H. Under BSM with continuous monitoring it is closed form (Reiner–Rubinstein): the reflection principle maps every path that touches H onto a Black-76 price at an image forward F* = ρ²F, weighted by ρ^{2μ}. The kernel is forward-based, with the barrier on spot S_t = F_t e^{−b(T−t)} (flat carry b, D-028), and an MC that shares only the log-distance to H checks it. A naive MC that tests the barrier on its grid prices a *discretely* monitored contract and overprices knock-outs by O(√Δt). The Brownian-bridge weight removes that bias exactly under BSM, and the BGK shift H·e^{∓0.5826σ√Δt} maps the continuous price to the discrete one. Measured (gates.txt): in + out = vanilla to 3.55e-15 on the same paths; bridge MC vs RR max |z| 1.99 over 56 cells; BGK tables to 4.97e-4; 66 Haug cells within 5e-5; the 6 mis-rounded ones match a two-method golden file to 3.5e-14.

## 2. Key formulas
1. **RR blocks (1)–(3)** `ρ = H/S0 = (H/F)e^{bT}`, `μ = b/σ² − ½`, `F* = ρ²F`, `G(X,L;κ) = θ·DF[X·Φ(κd1(X,L)) − K·Φ(κd2(X,L))]`. A = vanilla, B = vanilla restricted to S_T beyond H, and C, D = ρ^{2μ}× the same legs on F*. Each of the 8 types is a sum from table (3); e.g. with R = 0, cdi(K ≥ H) = C and cdo(K ≥ H) = A − C.
2. **Reflection at zero carry (6), K ≥ H** `cdi = (F/H)·Bl_+(H²/F, K) = (K/H)·Bl_−(F, H²/K)`: a down-and-in call equals K/H puts struck at H²/K. This is a static hedge via put–call symmetry, tested to 1e-12.
3. **In–out parity (5)** `in + out = A + E + 𝓕`, where E and 𝓕 are the rebate legs. With rebate R = 0, in + out = vanilla. It holds path by path, in any model.
4. **Bridge (9)–(10)** `p_i = exp(−2(ηy_i)⁺(ηy_{i+1})⁺ / (σ_i²Δ))`, `w = Π_i(1 − p_i)`, with y = ln(S/H). p_i is the probability that the bridge between two grid points touched H. Out pays g·w, in pays g·(1 − w). σ_i² = σ² is exact for any Δ and any carry; σ_i² = v_i⁺ (Heston) is approximate.
5. **BGK (12)–(13)** `V_m(H) = V(H·e^{−ηβσ√Δt}) + o(m^{−1/2})`, `β = −ζ(½)/√(2π) = 0.5825971579`. The discrete price equals the continuous price with the barrier moved away from spot. Without the shift, the error is ≈ −ηβσ√Δt·H·∂_H V = O(m^{−1/2}).
6. **Model-free (7)** `cdo(K = H) = DF·(F − H)` (b = 0) for any continuous martingale, by optional stopping (F_τ = H on a hit). This gates the Heston engine.

## 3. The code that matters
**3a. Bridge weight and MC loop: `src/qpricer/engines/exotic_mc.py:111-114, 139-149`**
```python
    a, c = np.maximum(ey0, 0.0), np.maximum(ey1, 0.0)
    pos = np.asarray(s2dt) > 0
    p = np.exp(-2.0 * a * c / np.where(pos, s2dt, 1.0))
    return 1.0 - np.where(pos, p, np.where(a * c > 0, 0.0, 1.0))
    # ... barrier_mc:
    ey, w = eta * (np.log(F / H) - b * T), 1.0
    steps = obs_steps(model, t, n_paths, np.random.default_rng(seed), None, scheme)
    for i, (ti, x, vi, _) in enumerate(steps):
        ey_new = eta * (np.log(F / H) - b * (T - ti) + x)
        if every is None:
            w = w * bridge_survival(ey, ey_new, np.maximum(v, 0.0) * dt)
        elif (i + 1) % every == 0:
            w = w * (ey_new > 0)
        ey, v = ey_new, vi
    g = vanilla(F * np.exp(x), K[..., None], cp)
    weight = {"out": w, "in": 1.0 - w, "vanilla": 1.0}[knock]
```
- **111–114: eq. (9) on signed log-distances ηy.** The positive parts give p = 1 once an endpoint is at or beyond H, so the path is dead. When σ²Δ = 0 (Heston v⁺ = 0), p is set to 0 or 1 instead of 0/0 (pitfall 3). For far-away paths `exp` underflows to 0, which is the correct value.
- **139–142.** y_i = ln(F/H) − b(T − t_i) + x_i is the log-distance of *spot* to H (eq. 8), the same quantity the closed form uses, so the two cannot disagree on carry (pitfall 5). `None` means no hidden sub-steps: the `n_steps` grid is the bridge grid (the validator calls this "justified").
- **143–146.** Continuous monitoring multiplies conditional survival probabilities. That is Rao–Blackwell: the variance is lower than with a sampled uniform, and the weight is smooth in the parameters. Discrete monitoring uses a hard indicator at the m dates, with no bridge; t = 0 is not monitored.
- **147.** v is the **left-point** variance (line 138 sets it to σ² or v0). This is exact under BSM and an O(Δ) approximation under Heston (D-028). **149:** in, out and vanilla come from one path set and one w, in separate calls, so in + out − vanilla is round-off (3.55e-15), not O(SE).

**3b. Reiner–Rubinstein core: `src/qpricer/engines/analytic.py:51-53, 61-69, 85, 88`**
```python
    s, lnr = sigma * np.sqrt(T), np.log(H / F) + b * T
    mu = b / sigma**2 - 0.5
    lnFs = np.log(F) + 2.0 * lnr  # ln F*
    # ... :55-58 G(lnw, lnX, L, k) = e^{lnw} theta DF [X Phi(k d1) - K Phi(k d2)] via log_ndtr
        A, B = G(0.0, np.log(F), K, cp), G(0.0, np.log(F), H, cp)
        C, D = G(2 * mu * lnr, lnFs, K, eta), G(2 * mu * lnr, lnFs, H, eta)
        far = cp * (K - H) >= 0
        if cp * eta > 0:
            core = np.where(far, C, A - B + D) if knock == "in" else np.where(far, A - C, B - D)
        else:
            core = (
                np.where(far, A, B - C + D) if knock == "in" else np.where(far, 0.0, A - B + C - D)
            )
    # ... :70-84 rebate legs E, F (only if R > 0)
    breached = eta * lnr >= 0
    return (np.where(breached, A if knock == "in" else R, core) + 0.0)[()]
```
- **51–53.** lnr = ln ρ and ln F* = ln F + 2 ln ρ. No r or q appears; r = −ln DF/T enters only the rebate leg 𝓕. **61–62.** A and B are Black legs on the real forward; C and D are the same legs on F*, weighted by ρ^{2μ} = e^{2μ ln ρ}. The weight is applied as exp(log-weight + log_ndtr) so that ρ^{2μ}·Φ never becomes inf·0 at small σ (pitfall 8).
- **63–69: the 16 cells of table (3), compressed.** `far` means the strike is beyond the barrier, θ(K − H) ≥ 0. θη = +1 covers cdi, cdo, pui and puo; θη = −1 covers cui, cuo, pdi and pdo. Example: cuo with K ≥ H is identically 0, because S_T > K ≥ H requires a hit. **85–88: eq. (4).** If the barrier is already breached, out = rebate and in = vanilla (continuous monitoring only).

**3c. The Haug erratum, handled without loosening: `tests/test_barrier.py:99-111`**
```python
    cells = json.loads(GOLDEN_HAUG.read_text())["cells"]
    assert len(cells) == 72 and sum(not c["published_correctly_rounded"] for c in cells) == 6
    err_pub, err_e1, err_gold = 0.0, 0.0, 0.0
    for c in cells:
        knock, cp, eta, H, sig, K = (c[k] for k in ("knock", "cp", "eta", "H", "sigma", "K"))
        assert c["published"] == HAUG[(knock, cp, eta)][H][int(sig == 0.3)][list(KH).index(K)]
        v = float(barrier_price(HF, K, H, HT, HDF, sig, cp, eta, knock, b=HB, R=HR))
        err_gold = max(err_gold, abs(v - c["value"]))
        if c["published_correctly_rounded"]:
            err_pub = max(err_pub, abs(v - c["published"]))
        else:
            err_e1 = max(err_e1, abs(v - c["value"]))
    assert err_pub <= 5e-5 and err_e1 <= 1e-10 and err_gold <= 1e-10, (err_pub, err_e1, err_gold)
```
- The validator wrote the golden file from two methods independent of the A–F algebra: a Haug spot-form transcription and a Crank–Nicolson PDE, which agree to ≤ 5.5e-10. The 66 correctly rounded cells keep the note's 5e-5, and the 6 flagged cells are held at 1e-10 against the golden value. Line 104 checks that the golden "published" column equals the note's table, so the reference cannot drift.

## 4. Validation story
| Gate / check | Measured (gates.txt unless noted) | Tolerance |
|---|---|---|
| G1 in + out = vanilla, same paths (8 types; BSM and Heston QE-M; bridge and discrete) | 3.55e-15; validator 1.07e-14 on new configs (b < 0, DF > 1, K = H) | < 1e-12 |
| G2 bridge MC vs RR, BSM: 14 cells × b ∈ {0, 0.04} × M ∈ {1, 50}, 2e5 paths | max \|z\| 1.01 / 1.70 / 1.06 / 1.99; zero cells exactly 0. Validator, 2800 fresh z: mean +0.026, sd 0.998, 0.36% beyond 3 (0.27% expected) | 3 SE |
| G2h Heston ξ = 1e-4, ρ = 0 vs RR at √θ / G2m model-free cdo(K = H = 90) = 10·DF, FO set, Δ = 1/128 | max \|z\| 2.97 / z +0.75 | 3 SE |
| G3 RR and BGK-shifted RR vs BGK Tabs 2.1, 2.2, 2.6 / discrete MC vs BGK "True" (13 cells) | 4.97e-4 / max \|d\|/(3SE + 1e-3) = 0.54 | 5e-4 / ≤ 1 |
| Haug: 66 correctly rounded cells / 6 E1 cells vs golden / all 72 vs golden | 4.96e-5 (σ .25), 4.59e-5 (σ .30) / 3.46e-14 / 4.80e-14 | 5e-5 / 1e-10 / 1e-10 |
| RR vs own quadrature of the killed and first-passage densities (no A–F algebra) | ≤ 3.2e-14 (D-027) | 1e-12 |

- **What failed first: six Haug σ = 0.30 cells.** They were cuo(105, K 90), puo(105, K 90/100/110) and pdi(95, K 100/110), missing 5e-5 by 5.2e-5 to 6.25e-5. Example: puo K = 90, where RR = quadrature = 4.22923747 but the table prints 4.2293. Three methods agree, and no small change of (σ, T, r, b) fits the column better, so the table is mis-rounded (erratum E1). **The first fix was wrong.** It moved the whole σ = 0.30 column to 1e-4, loosening 30 good cells. The validator flagged it (F1), and it was withdrawn in favour of the per-cell test in 3c (D-027).
- **Slope 0.42, not 0.5 (E2; figure).** Setup: DOC with BGK Tab. 2.1 inputs (H 95, σ 0.3, T 0.2, b 0.1), where RR = 4.3975. Uncorrected grid MC falls from 5.785 (m = 4) to 4.582 (m = 512); the bridge MC reads 4.404 / 4.403 at those m (CSV). The fitted slope is 0.422. The noise-free BGK prediction on the same m gives 0.426, and its local slope rises from 0.32 (m 4 → 8) to 0.50 (m ≈ 10⁶). The validator's exact discrete prices give 0.4295. This is pre-asymptotic curvature, not a bug.
- **BGK fails near the barrier (diagnostic).** BGK vs discrete MC: at m = 50 the error is ≤ 0.63% in magnitude for H ≤ 98, but −3.10% at H = 99 (z −4.62). At m = 5, H = 99 it is −9.23% (z −21.04). For the reverse UOC at H = 115 it is +1.56%.
- **The Heston bridge bias is real (F3).** Pooled over 25 seeds (validator), cdo − 10·DF = +0.095 / +0.055 / +0.020 ± 0.005 / +0.012 ± 0.005 at Δ = 1/32, 1/64, 1/128, 1/256, i.e. O(Δ). G2m passes because 0.020 is below the gate SE (0.024), not because the bias is zero; at 1/64, 16% of seeds would fail. The single-seed numbers first quoted understated the bias.
- **G2h's 2.97 is a 1-in-75 seed (F4).** On that path set the vanilla itself is off by −3.15 / +3.13 SE, and barrier z tracks vanilla z with correlation 0.905. Over 300 validator seeds the per-cell mean z is −0.031 to +0.011, so there is no bias. The seed was not changed.

## 5. Ten likely questions (basic → hard)
1. **Why does in + out = vanilla hold, and why test it at 1e-12 rather than 3 SE?** On every path exactly one of the pair pays: g·w + g(1 − w) = g. With the same paths and weights it must hold to round-off (3.55e-15). A failure at the SE level means the two legs saw different paths or weights, a different tie rule at y = 0, or a re-drawn uniform. They are priced in separate calls; in := vanilla − out would make the test tautological.
2. **Derive the down-and-in call by reflection. Is it a hedge?** For K ≥ H the payoff needs S_T > K ≥ H, so the path hit H and came back. Reflecting after τ maps {hit, end at y} to {end at 2ℓ − y}, with Girsanov factor e^{2νℓ/σ²} = ρ^{2μ}. That gives cdi = ρ^{2μ}·Bl(F*, K). At zero carry, cdi = (K/H) puts struck at H²/K (tested to 1e-12, σ ∈ {0.1, 0.25, 0.6}): buy the puts, and at the hit swap them for the call at no cost. [GK] This breaks with carry (the ρ^{2μ} tilt) and with skew, because put–call symmetry needs a smile symmetric in log-moneyness.
3. **Your MC checks the barrier on 50 dates. What is wrong?** It prices the 50-date discrete contract. Paths dip through H between dates unseen, so knock-outs are too high. The error is O(√Δt): halving Δ removes only about 29% of it. BGK Tab. 2.1, H = 97: continuous 3.060 vs 50 dates 3.834 (+25%). In the figure, even m = 512 is 0.185 too high (4.582 vs 4.3975).
4. **Why is the bridge exact under BSM?** Given the endpoints, the log-price in a step is a Brownian bridge and P(touch) = exp(−2ab/(σ²Δ)). The drift cancels (§3.5), so this is exact for any Δ, any carry, even M = 1 (gated at M = 1 and 50). I multiply survival probabilities instead of sampling a uniform: same mean, variance lower by E[g²w(1−w)], and a smooth weight usable for CRN Greeks.
5. **Where does 0.5826 come from?** A discretely observed walk first breaches at a date where it has already overshot H. The expected overshoot, in units of σ√Δt, tends to β = −ζ(½)/√(2π), Siegmund's constant (checked to 1e-15). Reflecting at H plus the overshoot gives the continuous price at a barrier moved away from spot by βσ√Δt (BGK Thm 1.1; Kou 2003 for all 8 types). At m = 50, σ√Δt = 0.018974, so the shift is ≈ 1.1% in log (arith.). It is a BSM result, and it degrades within about one step-sd of H.
6. **Theory says slope ½; you measured 0.42. Bug?** No. ½ is the leading order; at practical m the shift is not small relative to the curvature of V in H. The deterministic BGK curve on the same m gives 0.426, with local slope rising to 0.50 at m ≈ 10⁶, and the validator's exact prices give 0.4295. The MC matches the deterministic prediction, and I did not tune it toward 0.5.
7. **Six published values fail your tolerance. What do you do?** Suspect my maths first. RR matched my own density quadrature to 3e-14, and the validator's independent RR and PDE matched each other to 5.5e-10. The misses are one unit in the 4th decimal (4.22923747 printed as 4.2293), so the published values are mis-rounded. My first erratum loosened the whole column to 1e-4. That was wrong, it was caught (F1) and withdrawn: 66 cells stay at 5e-5, and 6 are tested vs the golden file at 1e-10. A tolerance is never widened when a tighter treatment exists.
8. **(Risk) How do you price a barrier under Heston, and how good is it?** The same bridge, with v_i⁺ frozen over the step. Given the variance path, ln S is Gaussian with time-varying variance, so freezing v is exact only as Δ → 0, and a step with v_i⁺ = 0 cannot cross. Gates: ξ = 1e-4 vs RR (2.97, a lucky seed, explained), and the model-free cdo(K = H) on the Feller-violating FO set, where the bias is +0.020 at Δ = 1/128 (0.8 gate SE) and +0.012 at 1/256. A positive bias (knock-out too high) is consistent with missed crossings when v_i⁺ is small. There is no BGK under Heston; discrete contracts get discrete MC.
9. **(Hedging) Why are barriers painful to hedge?** [GK; barrier.md pitfall 7 and §6; nothing computed here] Close to the barrier, a knock-out's vega turns negative: more vol means more chance to die. A reverse knock-out (cuo K < H, pdo K > H) jumps by \|H − K\| at H. Its delta flips sign, and gamma goes from positive near K to large and negative near H, scaling like 1/(σ²τ) near expiry (note's estimate). Hedging a discontinuity needs unbounded trades, so desks shift the barrier or smooth it and charge for that. The price depends on vol at the barrier and on forward skew, not ATM vol.
10. **What would you change for production?** Real fixing calendars (most contracts monitor closes): BGK only far from H, discrete MC or PDE near it. [GK] LSV, to fit vanillas while controlling forward skew. For Heston, Δ ≤ 1/256 or a bridge conditioned on integrated variance. MC rebates (D-028 (v): closed form only). Gated Greeks: CRN on the smooth w, bumps smaller than half the distance to H, and a PDE cross-check. A C++ kernel, and the barrier-shift policy as a desk parameter.

## 6. Traps: do NOT say
1. **"A fine grid converges to the continuous price"** or **"the slope is 0.5".** A grid MC converges at O(√Δt) to a different contract: m = 512 is still 0.185 too high. The gate uses the bridge, which is unbiased under BSM for every M, including M = 1. The measured slope is 0.422 (pre-asymptotic).
2. **"We never touched a tolerance"** or **"we loosened Haug because the table is wrong".** The accurate version: the first erratum loosened the σ = 0.30 column to 1e-4; the validator flagged it (F1) and it was withdrawn. Now 66 cells are at 5e-5 and 6 are tested vs a two-method golden file at 1e-10.
3. **"The bridge (or BGK) is exact under Heston."** The v⁺ bridge has an O(Δ) bias: +0.020 at Δ = 1/128, below the gate SE but real, and 16% of seeds would fail at 1/64. BGK is a BSM result.
