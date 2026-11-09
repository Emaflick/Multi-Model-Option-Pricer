# Phase 5: calibration to the SPX implied-vol surface (Heston on the surface, SABR per expiry, arbitrage checks)
Sources, cited by tag: **g5** `docs/validation/gate_outputs/phase5_gates.txt` (primary), **g4** `docs/validation/gate_outputs/phase4_gates.txt` (calibrated autocall block), **V5** `docs/validation/phase5.md` (validator, findings M1–M11), **N** `docs/derivations/calibration.md` (eq. numbers, errata E1–E4), D-032..D-038 in `docs/DECISIONS.md`, **OI** `docs/OPEN_ISSUES.md` (F5, F6, F3), `report/calibration_{expiry,heston,sensitivity,f5,arbitrage,starts}_20261106_2213.csv` (**EXP, HES, SEN, F5c, ARB, ST**), **NB** `report/calibration_notebox_20261106_2213.txt`, **AC** `report/figures/autocall_attribution_heston_fit_20261106_2213.csv`, **JSON** `data/processed/heston_fit_20261106_2213.json`, figures `report/figures/calibration_{smiles_short,smiles_long,df_r}_20261106_2213.png`. Code: `src/qpricer/calibration.py` (**cal**), `tests/test_calibration.py`. [GK] = general market knowledge, not from the repo. (der.) = simple arithmetic on repo numbers, done here. (est.) = a hand estimate in N, not measured.

## 1. Pitch (30 s)
We calibrate one Heston parameter set to the SPX surface of the 2026-11-06 22:13 snapshot, and SABR β = 1 separately on each expiry. The targets are OTM bid/ask implied vols inside the 5Δ band, thinned to at most 27 quotes per expiry. Heston is fitted on the 21 expiries with T ≥ 0.1 (565 quotes); the 18 shorter expiries are kept out of sample. The loss is the vol error in units of the quote's IV half-width. Inside the optimiser COS runs at one pinned (N, L) = (16384, 64), checked against Lewis. 16 Sobol starts all reach one basin: p̂ = (v0 0.0524, κ 4.58, θ 0.0529, ξ 1.65, ρ −0.752), with Feller violated (ν_F = 0.178). The in-sample RMSE has a median of 0.253 vol pt; out of sample the median is 1.299. That gap is Heston's single clock 1/κ, which cannot produce the steep short-dated SPX skew. A synthetic round trip with tolerances fixed in advance shows the pipeline recovers known parameters. Repriced with the calibrated set, the Phase 4 autocallable comes out 1.87% of notional below BSM, and its par coupon is 3.33 points higher (g5, JSON, D-037, D-038).

## 2. Key formulas
1. **Objective (8)–(9).** `min_p ½Σ r_i²`, `r_i = w_i(σ^H_i(p) − σ^m_i)`, `w_i = 1/max(h_i, 0.001)`, `h_i = ½(σ^a_i − σ^b_i)`. Here σ^H is the Black IV of the COS price at F = DF = 1 with strike e^k, k = ln(K/F_j), so |r_i| ≤ 1 means the model is inside the quote. If the model IV cannot be inverted (status ≠ 0), the fallback is `r_i = w_i(v^H − v^m)/𝒱^m`; the fallback count is 0 at the optimum (g5).
2. **Vol vs price residuals (10).** `(v^H − v^m)/𝒱^m = Δ[1 + d1d2Δ/(2σ^m)] + O(Δ³)`, with Δ = σ^H − σ^m and 𝒱 = φ(d1)√T. To first order, Σw²Δ² = Σw²(Δv)²/𝒱², i.e. price residuals weighted by 1/vega². The volga term d1d2/σ is large in the wings, so the two losses differ far from the optimum. The COS budget uses the same first-order map: `e_vega = max|v_COS − v_Lewis|/𝒱(σ^m) ≤ 1e-5`, a price error expressed in market-vega units.
3. **Short-end DF (2)–(3), F5.** `r̂(T) = a + bT`, fitted by WLS on T ≥ 0.1 with weights 1/(r_se² + (10 bp)²). For T < 0.1, `D̃F = e^{−r̂T}` and `F̃ = K̄_u + ȳ_u/D̃F`, an OLS of C − P on the Phase 0 pairs with D̃F held fixed. IV sensitivity (4): `δσ = −σE·δDF/DF` with E ≈ 1 ATM (every vol moves by the same fraction), and `δσ ≈ ∓1.2533·δF/(F√T)` ATM (puts and calls split apart).
4. **Error bars (12), (17).** Real data: `s² = 2cost/(n − 5)`, `C = s²(JᵀJ)⁺` by SVD, SE = √diag C, cond(J·diag|p̂|). Synthetic data, computed in advance at the true p*: `Σ_p = σ_ε²A⁻¹(J_σᵀW²J_σ)A⁻¹` with A = J_σᵀWJ_σ. G2 passes iff `|δp_j| ≤ 3SE_j` for every j **and** `D² = δpᵀΣ_p⁻¹δp ≤ χ²₅(0.999) = 20.52`.
5. **Heston ATM skew (14).** `ψ_H(T) = ρξJ(T)/(2v̄^{3/2}T²)`. For v0 = θ, `ψ_H(T)/ψ_H(0) = g(κT)` with `g(x) = 2(x − 1 + e^{−x})/x²`. The skew is flat, at ρξ/(4√v0) ⚑, for κT ≪ 1 and decays like ρξ/(2κ√θT) for κT ≫ 1, so Heston has one clock, 1/κ. The SPX ATM skew follows roughly T^−0.4 (Bayer–Friz–Gatheral exponent 0.407 on a 2013 surface: literature ⚑, **not measured on our snapshot**).
6. **Static no-arbitrage (21)–(22).** Calendar: total variance `w(k,T) = σ²T` must not decrease in T at fixed k = ln(K/F_T). Butterfly: `B_i = λc_{i−1} + (1 − λ)c_{i+1} − c_i ≥ 0`, with λ = (K_{i+1} − K_i)/(K_{i+1} − K_{i−1}) on undiscounted calls. A "strict" violation is one that survives at tradeable prices (buy the wings at ask, sell the body at bid). Its size is z = −B^m/h^B, the mid shortfall in units of the butterfly's half-spread.

## 3. The code that matters
**3a. Weights, pinned COS and residuals: `cal:34-35, 175, 216, 275-281`**
```python
H_MIN = 1e-3  # half-width floor of the weights (0.1 vol pt), eq. (8)
COS_NL = (16384, 64.0)  # pinned COS (N, L) for every fit (D-034)
    q["w"] = 1.0 / np.maximum(q.h, h_min)
    g["vega"] = np.exp(-0.5 * d1 * d1) / np.sqrt(2.0 * np.pi) * np.sqrt(T)
    v, _ = heston_prices(x, g, nl)
    pr = g["w"] * (v - g["v"]) / g["vega"]
    if kind == "price":
        return pr, 0
    s, st = model_iv(v, g)
    ok = st == 0
    return np.where(ok, g["w"] * (s - g["sigma"]), pr), int((~ok).sum())
```
- **34–35, 175.** The weight is 1/half-width, floored at 0.1 vol pt so that ultra-tight quotes cannot dominate. The floor binds on 60.9% of the thinned quotes (median half-width 0.078 vp, D-032). One (N, L) pair is used for every evaluation, so r(p) is a smooth function of p.
- **216.** The vega is taken at the **market** vol (normalised, F = DF = 1). It is fixed per quote, which keeps the fallback and the budget well defined even where the model IV does not exist.
- **275–276.** COS prices are computed once and turned into vega-scaled price residuals, which are eq. (10) and also fallback (9). **277–278.** `kind="price"` returns eq. (10) unchanged; it is the cross-check objective. **279–281.** The model price is inverted to an IV. Where the inversion succeeds the residual is the IV residual (8); elsewhere the fallback is used, and the fallbacks are counted. NaN is never passed to TRF.

**3b. Starts, solver and basin: `cal:287, 313-315, 350, 353-355`**
```python
    u = qmc.Sobol(d=5, scramble=True, rng=np.random.default_rng(seed)).random_base2(m)
        res = least_squares(fun, x0, jac="2-point", diff_step=diff_step, x_scale="jac",
                            bounds=bounds, method="trf", ftol=tol, xtol=tol, gtol=tol,
                            max_nfev=2000)  # fmt: skip
    conv = tab.status.isin(CONVERGED).to_numpy()
    cost = np.where(conv, tab.cost.to_numpy(float), np.inf)
    best = int(np.argmin(cost))
    tab["basin"] = conv & (cost <= cost[best] * (1.0 + 1e-4))
```
- **287.** 2⁴ = 16 scrambled Sobol points with a fixed seed (20261109); a power of 2 preserves Sobol balance. They are mapped to √v0, √θ ∈ [0.10, 0.40], ln κ ∈ [ln 0.5, ln 8], ln ξ ∈ [ln 0.2, ln 3], ρ ∈ [−0.95, −0.2], and the start set does not depend on the data.
- **313–315.** TRF is a trust-region method that respects the box bounds. The Jacobian is a forward difference with relative step 1e-6: SciPy's default √ε would hit the 1e-13 IV-inversion floor (N pitfall 3). `x_scale='jac'` rescales the parameters, whose sizes differ by about 100× (v0 vs κ).
- **350–355.** Only starts that converged (status 1–4) count. p̂ is the lowest-cost start. The "basin" is every start within a relative 1e-4 of the best cost, and its spread is reported in SE units.

**3c. Curvature SEs and conditioning: `cal:363-370, 374`**
```python
    J, x = res.jac, res.x
    n, m = J.shape
    s2 = 2.0 * res.cost / (n - m)
    _, S, Vt = np.linalg.svd(J, full_matrices=False)
    keep = S > n * np.finfo(float).eps * S[0]
    C = s2 * (Vt[keep].T / S[keep] ** 2) @ Vt[keep]
    se = np.sqrt(np.diag(C))
    _, Ss, Vts = np.linalg.svd(J * np.abs(x), full_matrices=False)
    # :374  "cond": Ss[0] / Ss[-1], "weakest": Vts[-1]
```
- **363–365.** J is TRF's weighted Jacobian at p̂. s² is the residual variance per degree of freedom (`cost` is ½Σr²); here it is 7.37, so the residuals are about 2.7 weighted half-widths, not quote noise (V5 M6). **366–369.** C = s²(JᵀJ)⁻¹ is computed via the SVD, with tiny singular values dropped and counted (0 dropped, HES). This is the Gauss–Newton covariance, valid **only if the model is right**. **370, 374.** The SVD of J·diag|p| gives the condition number in relative units and the least-identified direction.

## 4. Validation story
| Gate (source) | Measured | Tolerance |
|---|---|---|
| Pinned COS vs Lewis over the bounds box, 32 corners + 128 Sobol points (g5) | e_vega 9.21e-6 (worst corner 0.01, 0.5, 0.01, 3, −0.99); direct IV error 2.48e-3 there, 6.41e-3 over the box | ≤ 1e-5, a priori |
| COS vs Lewis IV at p̂ (g5; V5 row 8) | 4.9e-13 on I_H; validator 5.6e-14 on all 1029 thinned quotes; 7.0e-13 out of sample (D-034) | 1e-5 |
| F5: used pairs inside their band; eq. (3) identity; WLS recovery (g5) | max \|res\|/h 0.834 (0 pairs > 1); 0.0; 6.9e-18 | ≤ 1; 1e-12; 1e-14 |
| Synthetic S1 / S2: G1 noise-free; G2 (g5) | 4.1e-14 / 2.1e-14; max \|z\| 1.10 / 2.42, D² 2.516 / 12.491; G3, G4 pass | 1e-6; 3 SE, 20.52 |
| Linearisation, 64 seeds on S1: sd(p̂_j)/SE_j (g5) | [1.10, 1.20, 0.98, 1.13, 1.00]; validator on 100 fresh seeds [0.95, 0.85, 1.00, 0.95, 0.97] | [0.73, 1.27] |
| SABR synthetic, 4 expiries (g5) | all pass, max D² 5.41 | 3 SE; χ²₃(0.999) = 16.27 |
| Real fit (g5, HES, ST) | 16/16 converge, one basin, spread 1.1e-5 SE; 0 fallbacks; 0 active bounds | ≥ 8 starts |
| Objective (10) vs (8) from the same starts (g5) | real max \|Δp\| 0.83 SE; S1 0.48 SE | ≤ 1 SE |
| Market calendar / butterfly (g5, ARB) | 11054 points: 0 mid, 0 strict / 7547 triples: 2126 mid, **0 strict**, max z 0.427 | diagnostic |

- **Arbitrage, read correctly.** Mid butterflies are often negative at 5-pt strikes, but z ≤ 0.427 means each shortfall is under half the butterfly's own half-spread. All of them sit inside the spreads, so none is tradeable (0 strict). The checks are diagnostics only (D-010 remains the only filter). The fitted SABR smiles show 0 calendar and 0 butterfly violations on the same points (ARB). A +3h bump and a pair of swapped smiles are both flagged, so the checks have power (g5).
- **What failed first (errata E1–E4).** (i) The note's COS (1024, 12) failed its own post-fit check: an IV error of 9.9e-5 at a plausible SPX point (L = 18 gives 9.8e-7). On the note's box no fixed (N, L) up to (4096, 30) was accurate, and Lewis `quad_vec` failed at 4 corners. The fix was the narrowed box plus (16384, 64); the rungs (8192, 48) at 3.7e-5 and (16384, 80) at 2.0e-5 fail the budget (D-034). (ii) The note's "≲ 1.5 F_se" forward shift was exceeded: 3.56 F_se on 2026-12-03 (+0.654 pt), yet every used pair still sits inside its band (E3). (iii) The stated false-fail rate of 2.9% covers only the Heston G2 gates; the whole suite is about 8% (E4, M4).
- **Validator (V5).** 19 independent tests (own CF, Lewis, IV, Jacobian, Hagan) found no code defect; the full `pytest -q` run gives 556 passed. Documentation findings fixed in round 1: M1 (the additivity line was a tautology; it is replaced by checks of each leg against exact values), M2 (the pilot fit was not re-runnable; `--box note` now reproduces it), M3 (budget wording), M6 (what the SEs mean), M7 (the SABR-on-Heston diagnostic, now g5 line 48), M9 (the BSM-leg label).

## 5. Ten likely questions (basic → hard)
1. **Why fit vols, and why weight by 1/half-width?** Vol is the market's normalised price, so errors compare across strikes and maturities, and the deliverable is an IV RMSE. By (10), Δv/𝒱 ≈ Δ, so price errors weighted by 1/vega² give the same optimum near the fit: objective (10) lands within 0.83 SE of p̂ (g5). Far from the optimum, price residuals over-weight the convex wings.
   Since h_IV ≈ (price half-spread)/vega, w·Δ ≈ Δv/(price half-spread) (der.): the residual counts half-spreads. The 0.1 vp floor binds on 60.9% of quotes.
2. **What exactly do you fit, and what was F5?** OTM quotes with mid ≥ 0.50 (10 ticks) and |d1| ≤ 1.6449 (5Δ put to 5Δ call): 7625 → 5806 in band → 1029 thinned (25 d1 targets per expiry plus the ATM put and call) → n_H = 565 on T ≥ 0.1. Thinning removes near-duplicate 5-pt strikes and stops the weeklies from dominating (D-032).
   F5: the short-end parity DFs implied r of 2.64%–16.89%. The fix is a WLS zero curve, r̂ = 4.527% + 0.304%·T, plus an F refit. Short r becomes 4.533–4.556%, and DF rises between consecutive expiries go from 10 to 4, none below 0.1 y (D-033, OI).
3. **How do you know the optimiser recovers the truth, and how often does the gate false-fail?** Synthetic data, labelled as such, on the real grid with 0.3 vp noise. Σ_p (17) at p* depends only on (p*, grid, w, σ_ε), so every tolerance and seed was fixed before any noisy fit (D-035); pre-check max 3SE/|p| is 0.112 for S1 and 0.094 for S2.
   By the union bound the whole suite false-fails ≈ 8%. On fresh seeds, S1 G2 fails 1/100 and S2 0/100; SABR 2026-11-13 fails 4/200 (nominal 0.9%), because the linearisation bends at ν = 2 (M4). Never re-roll the seed.
4. **Why pin COS inside the optimiser, and isn't narrowing the box cheating?** The self-checking COS changes (N, L) as p moves, so r(p) is piecewise. A 1e-6 finite-difference step that crosses a switch measures the jump, not the slope (≈ 1e-9/1e-6, est.), and the Jacobian is corrupted. The pin must be accurate wherever TRF can go; on the note's box no fixed (N, L) was, hence the narrower box (per-expiry pins would save < 10%, D-034).
   Audit: `--box note` sends 20/20 starts to the same p̂ (cost relative −4.7e-15), ≥ 38 SE inside every narrowed bound (NB); the validator reproduced this with its own pricer.
5. **Read me the fit.** p̂ = (0.052372, 4.5751, 0.052925, 1.6477, −0.75180), SE (0.00038, 0.107, 0.00025, 0.024, 0.0017), cost 2062.90 (JSON). All 16 starts end with status 2 after 8–19 evaluations, 29.4 s wall on 8 workers (ST, HES). ν_F = 0.178 violates Feller: it is reported, not imposed. The CF does not care, only MC does (QE), and imposing it ruins the skew.
   cond = 25.6 is benign because the window [0.114, 1.61] y straddles 1/κ̂ ≈ 0.22 y (der.), so κ and θ separate (13). The largest correlation is κ–ξ +0.915, and the weakest direction is mostly κ and ξ (−0.83, −0.50). ξ–ρ is only +0.27: the skew pins ρξ and the 5Δ curvature splits them (HES).
6. **(Risk) "κ = 4.58 ± 0.11": what do you report to risk?** Not that. The SEs are conditional on the model, the window and iid errors, and s² = 7.37 says the residuals are not noise. Moving to T ≥ 2 weeks shifts κ to 5.96 (+12.97 SE) and ξ to 1.96 (+12.90 SE); n_t = 50 shifts every parameter by ≤ 1.06 SE (SEN).
   Report ranges, κ ∈ [4.49, 5.96] and ξ ∈ [1.62, 1.96]: the SEs understate the uncertainty about 10× (D-037, M6). [GK] Reserve by repricing the exotic under each variant. **Not done in the repo for the autocall**, so say so.
7. **Why does Heston fail at the short end?** RMSE in sample 0.151–0.766 vp (median 0.253), out of sample 0.832–2.093 (median 1.299) (g5); worst 2026-11-13, 2.09 vp RMSE and 6.29 vp max (EXP). The short-smile figure shows Heston too flat on every weekly. By (14) its skew is flat for T ≪ 1/κ, while the market's keeps steepening: in N's estimate (est.) the market skew roughly doubles from 0.019 to 0.1 y while Heston's falls ≈ 8% at κ = 3.
   Adding the weeklies pulls κ and ξ up by 13 SE and hurts the long end. Inside the window the error is U-shaped (0.77 vp at 0.114 y, 0.51 at 1.61 y): a compromise. [GK] Fixes: Bates jumps, rough vol, time-dependent parameters, or SVI/SSVI for vanillas with SLV for exotics.
8. **SABR per expiry fits beautifully. Can you use it?** 39/39 fits, RMSE 0.057–0.307 vp, 0/195 starts dropped (F6), ρ −0.81…−0.74, corr(ρ, ν) 0.66–0.89 (EXP). ν falls from 4.30 (T = 0.019) to 0.69 (1.61 y): SABR's ATM skew ≈ ½ρν does not decay at fixed parameters, so ν must shrink to follow the market.
   But ν²T is 0.352–0.890, above 0.16 on all 39 expiries, and at ν ≈ 1 Phase 3 measured Hagan +0.40/+0.82 vp off at k = −0.2. These are Hagan parametrisations, not SABR dynamics (OI F3). Fitted to Heston smiles, SABR's RMSE grows from 0.010 to 0.123 vp (Spearman 0.90), because SABR has no mean reversion. It is an interpolator per expiry and says nothing about how smiles link across T.
9. **(Hedging) What does calibration change for the autocallable, and what do you hedge?** AC-Q: V_BS 1.014530, V_H 0.995842, gap −0.018688 (CRN SE 0.000217) = level −0.008206 (flat 18.85% vs the Heston expected-variance curve at 22.99%) + smile at the 60%/70% barriers −0.020445 (exact) + path +0.009963 (g4, AC). The par coupon moves from 5.60% to 8.93%. The validator independently gets V_BS 1.014493 (exact) and a gap of −0.018611.
   The dominant exposure is static: the skew at the 60–70% strikes, which [GK] is hedged with 1.5 y puts and put spreads. The path term (forward skew, vol-of-vol) is not locked by today's vanillas: it is model risk.
10. **What would you change for production?** [GK] Analytic CF gradients or AAD (Cui et al. 2017), giving an exact Jacobian with no step tuning and no pinned-COS dilemma. [GK] Arbitrage-free SVI/SSVI as the vanilla mark, with Heston/Bates/SLV for exotics. [GK] Daily recalibration regularised towards yesterday's parameters, a model reserve from the window variants, and an OIS curve to cross-check the parity DFs.
    From the repo: fix S_ref (+0.074%, OI), commit the phase (M2), and bring `src/` from 2468 lines back towards 1500 (M10).

## 6. Traps: do NOT say
1. **"Our COS is IV-accurate to 1e-5 over the box", or "we narrowed the bounds because the fit needed it."** The 1e-5 budget is a price error in market-vega units (9.21e-6). The direct IV error is 2.48e-3 at the worst corner, where the model vega is 1e-4 to 1e-11 of the market's. The IV accuracy that matters is at p̂: 4.9e-13. The box was narrowed because no fixed COS prices the note's box, and the audit gives 20/20 starts at the same p̂, ≥ 38 SE inside (M3, M2).
2. **"κ = 4.58 ± 0.11", "Heston fits SPX to a quarter of a vol point", or "SABR fits to 0.06 vp, so SABR is validated."** Say instead: the sensitivity range is κ ∈ [4.49, 5.96]; 0.253 vp is the in-sample median only (out of sample it is 1.299, and only 3.7–55.6% of the in-sample quotes per expiry lie inside the bid–ask, EXP); every SABR fit has ν²T > 0.16.
3. **"Heston is 1.87% below BSM at the market ATM vol", "the three-term attribution is a passed gate", or "the synthetic suite false-fails 2.9%."** Say instead: the BSM leg is at the **calibrated-Heston** ATM vol, 18.85%. At the market's 18.29%, V_BS = 1.015253 (+0.000723, ≈ 3.3 CRN SE) and V_H minus it = −0.019411 (M9). The additivity holds by construction, because term 3 is the residual (M1); the real checks are the legs against exact values (max |z| 0.20, 0.87). The suite false-fails ≈ 8%.
