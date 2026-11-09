# Phase 4: Mono-asset autocallable (cash flows, decomposition, closed-form oracles, CRN, BSM vs Heston)
Sources: `src/qpricer/products.py`, `src/qpricer/engines/analytic.py`, `src/qpricer/engines/exotic_mc.py`, `scripts/autocall_compare.py`, `tests/test_autocallable.py`, `docs/validation/gate_outputs/phase4_gates.txt` (post-fix gate run, **gates.txt**), `docs/validation/phase4.md` (validator, incl. re-validation), `report/figures/autocall_attribution_illustrative.{png,csv}`, `docs/derivations/autocallable.md` (eq. numbers, errata E1–E3), D-025, D-026, D-029, D-031, `docs/OPEN_ISSUES.md` (Phase 4).
Tags: **[GK]** = general market knowledge or literature, not a result of this repo (no autocallable Greek is implemented). **(arith.)** = simple arithmetic on repo numbers, done for this sheet.

> **Every Heston number on this sheet is ILLUSTRATIVE.** It uses GS, Gatheral (2006) Table 3.2, a published SPX fit (κ 1.3253, θ 0.0354, ξ 0.3877, ρ −0.7165), with v0 := θ by our choice. It is **not** a calibration to our snapshot (D-029). GS's ATM vol at 1.5 y is 16.9%, while the market's is 18.3%. The ROADMAP gate (BSM vs *calibrated* Heston) is **DEFERRED to Phase 5**. Phase 5 re-runs `scripts/autocall_compare.py --heston-params <fit>.json`, and only that output may enter the report. The market inputs F_0(t_i), DF(t_i) and S_ref = 7374.4255 are real (D-025).

## 1. Pitch (30 s)
The study note (AC-Q) has quarterly observations over 1.5 y. On each date it pays an 8% p.a. coupon (2% per quarter) if the index is at or above 70%, with memory for missed coupons. The first time the index is at or above 100% before maturity, the note redeems at par plus the coupon. At maturity it repays 1, unless the index is below 60%, in which case the investor takes the full loss (D-026). Decomposed, it is a bond with a random maturity plus coupon digitals, financed by a down-and-in put that the investor sells to the issuer. Two closed forms validate the MC: the no-call sheet (a sum of marginals) and the two-date sheet. The full product is then priced by MC under BSM and Heston on common random numbers. Eq. (19) splits the model gap into three parts: vol level, the marginal smile at the barriers, and a path term. On the illustrative set, Heston comes out 1.44% of notional below flat-ATM BSM. The smile at 60%/70% alone accounts for −1.60%, and the par coupon rises from 5.23% to 7.55%.

## 2. Key formulas
1. **Cash flows (1)–(3)**, with ratios X_i = S_{t_i}/S_ref: `χ_i = A_i·1{X_i ≥ B_ac}` (i < n only), `π_i = A_i·1{X_i ≥ B_c}`, `C_i = π_i(c_i + mU_i)`, `U_{i+1} = (1 − π_i)(c_i + mU_i)`, `R_n = A_n[1 − (1 − X_n)·1{X_n < B_p}]`. Here A is the alive flag, m the memory flag and U the unpaid coupons. Every test is "≥", and the protection is observed at T only.
2. **Decomposition (6)–(7)** `V = E[DF(τ*)] + Σ_i DF_i E[C_i] − DF_n E[A_n(1 − X_n)1{X_n < B_p}]`, with `DIP_E = DF_n[p_n(B_p) + (1 − B_p)·D_n⁻(B_p)]`. The terms are: a bond to the random call date τ*; the coupon digitals; and minus a put at 60% plus 0.4 digital puts there (static, given the T-smile).
3. **Digital under a smile (9)** `D⁻(k) = Φ(−d_−) + k·φ(d_−)·√t·σ'(k)`, evaluated at σ = σ(k). A digital needs the smile's *slope*, not just its level, so no single flat vol prices both the put and the digital.
4. **CRN (15)** `Z_i = Σ_k √(Δ_k/τ_i)·(ρZ_{v,k} + √(1−ρ²)Z_{⊥,k})`: Heston's own spot Brownian increment over period i. It is exactly iid N(0,1), and it drives the BS paths.
5. **Par coupon (17)** PV = a + c·b is affine in c, because no indicator depends on c. Hence `ĉ = (1 − ā)/b̄`, with `SE = s(a_j + ĉ·b_j)/(√N·b̄)` by the delta method.
6. **Attribution (19)** `V_H − V_BS = (V_TS − V_BS) + (V_H^nc − V_TS^nc) + [(V_H − V_TS) − (V_H^nc − V_TS^nc)]`. In order: flat σ_ATM vs Heston's expected-variance curve; the marginal smile at B_p and B_c (exact, on the no-call sheet); and the path term (forward skew, vol-of-vol). σ_ATM is Heston's implied vol at (F_0(T), T).

## 3. The code that matters
**3a. Cash flows on all paths: `src/qpricer/products.py:70-80`**
```python
    C, R, chi = np.zeros(X.shape), np.zeros(X.shape), np.zeros(X.shape, dtype=bool)
    alive, U = np.ones(X.shape[1], dtype=bool), np.zeros(X.shape[1])
    for i, ci in enumerate(sheet.coupons):  # loop over dates; vectorised over paths
        G = ci + sheet.memory * U
        pi = alive & (X[i] >= sheet.B_c)
        C[i], U = pi * G, np.where(pi, 0.0, G)
        if i < n - 1:
            chi[i] = alive & (X[i] >= sheet.B_ac)
            R[i], alive = chi[i], alive & ~chi[i]
    R[-1] = alive * (1.0 - (1.0 - X[-1]) * (X[-1] < sheet.B_p))
    return C, R, chi, alive
```
- **73–75.** The coupon is decided before the call. The entitlement is G = c_i plus, with memory, the unpaid coupons. It is paid if the path is alive and X ≥ B_c; otherwise it is carried forward (pitfall 3 order). A call date has X ≥ B_ac ≥ B_c, so a call always pays its coupon; the code validates 0 ≤ B_c ≤ B_ac (E1).
- **76–78.** There is no call test at T. Flows are masked by `alive` *before* it is updated, so nothing is paid twice. **79.** European protection: 1 if X_n ≥ B_p, else X_n, and only on paths never called. The validator found these flows bit-equal (≤ 1e-15) to a path-by-path transcription on 20,000 paths × 7 sheets, with exact ties at every barrier.

**3b. CRN, Heston spot shocks driving BS: `src/qpricer/engines/exotic_mc.py:76-83`, `scripts/autocall_compare.py:117-122`**
```python
def _aggregate(steps, is_obs: NDArray, rho: float):
    zacc, t_prev, t_last = 0.0, 0.0, 0.0
    for (tn, x, v, z), obs in zip(steps, is_obs, strict=True):
        zacc = zacc + np.sqrt(tn - t_prev) * (rho * z[0] + np.sqrt(1.0 - rho**2) * z[1])
        t_prev = tn
        if obs:
            yield tn, x, v, zacc / np.sqrt(tn - t_last)
            zacc, t_last = 0.0, tn
    # ... attribution():
    xH, z = fixings(p, t, N, np.random.default_rng(seed), max_dt)
    sig = heston_iv(p, f[-1], f[-1], T)
    var = {"BS": sig**2 * t, "TS": total_variance(t, p)}
    X = {k: f[:, None] * np.exp(gbm_paths(w, z)) for k, w in var.items()}
    X["H"] = f[:, None] * np.exp(xH)
    pv = {k: autocall_pv(x, sheet, DF) for k, x in X.items()}
```
- **79–82.** Each QE-M sub-step (Δ ≤ 1/32) draws (Z_v, Z_⊥). The spot shock ρZ_v + √(1−ρ²)Z_⊥ is scaled to a Brownian increment and summed over the period, then divided by √τ_i. A sum of independent N(0, Δ_k) variables is N(0, τ_i), so z_i is exactly N(0,1) and independent across periods, and the BS path on z is an exact GBM. The validator rebuilt z from the raw generator stream and matched it exactly.
- **117–122.** One Heston path set drives everything; the BS (flat σ_ATM) and TS (Heston expected variance w(t)) paths reuse its z. σ_ATM is Heston's own IV at the forward, so both models price the ATM-forward T-vanilla identically (asserted to 1e-10). Every gap and SE comes from per-path differences (eq. 16).

**3c. Exact no-call oracle and the attribution: `src/qpricer/engines/analytic.py:167-173`, `scripts/autocall_compare.py:132-135, 145`**
```python
    t, DF = np.asarray(sheet.t), np.asarray(DF, dtype=np.float64)
    _, Dc = ratio_marginals(f, t, sheet.B_c, model) if sheet.B_c > 0 else (0, np.zeros(t.size))
    dip = 0.0
    if sheet.B_p > 0:
        p, Dp = ratio_marginals(f, t, sheet.B_p, model)
        dip = DF[-1] * (p[-1] + (1.0 - sheet.B_p) * Dp[-1])
    return float(np.sum(DF * sheet.coupons * (1.0 - Dc)) + DF[-1] - dip)
    # ... attribution():
    out |= {"gap": V["H"] - V["BS"], "se_gap": se(pv["H"] - pv["BS"])}
    out |= {"term1": V["TS"] - V["BS"], "se_term1": se(pv["TS"] - pv["BS"])}
    out |= {"term2": nc["H"] - nc["TS"], "se_term2": 0.0}
    out |= {"term3": V["H"] - V["TS"] - (nc["H"] - nc["TS"]), "se_term3": se(pv["H"] - pv["TS"])}
        c = (1.0 - a.mean()) / b.mean()
```
- **167–173: eq. (11).** Without autocall and memory, each flow depends on a single fixing, so the value is a sum of marginals: coupon digitals D⁺(B_c), plus the static DIP (a put plus 0.4 digital puts at 60%). `ratio_marginals` is Black, or Heston COS. The COS digital is a central difference on one COS call at [k−h, k, k+h], so it is the derivative of a single series (D-031).
- **134.** term2 is the difference of two closed forms, so it has no SE. Terms 1 and 3 use CRN per-path differences, and the three terms sum to the gap with 0.0 residual. **145.** The par coupon is solved from PV = a + c·b on one path set; there is no root search on noisy prices (pitfall 9).

## 4. Validation story
| Check | Measured (gates.txt unless noted) | Tolerance |
|---|---|---|
| BS MC vs (11), (11) with B_c = 0, n = 1, and (12) with m = 0/1 (2e5 paths) | max \|z\| 1.79 | 3 SE |
| Heston: mean M_t = 1; D⁺_i(L) vs COS for L ∈ {B_p, B_c, B_ac}; no-call sheet vs (11) + COS | max \|z\| 1.32 | 3 SE |
| Heston ξ = 1e-4 vs BS at QE's discrete variance Σ_Q, same draws | \|z\| 0.18 | 3 SE |
| Full product, BS MC vs the validator's exact backward induction (ACX), 18 runs / (11), (12) vs ACX / COS vs validator Gil-Pelaez | max \|z\| 2.36 / ≤ 1.5e-10 / put ≤ 3.5e-15, digital ≤ 3.9e-8 (phase4.md) | validator's (phase4.md) / 1e-9 / 1e-10, 1e-6 |
| Eq. (19) adds up; V^nc on the paths vs exact | \|sum − gap\| 0.0; max \|z\| 0.53 (AC-Q), 0.84 (AC-A) | 1e-12; 3 SE |
| Expected sign V_H < V_BS (illustrative) / BSM vs **calibrated** Heston | gap −0.01436 (CRN SE 0.00017) / needs a Phase 5 fit | < −3 SE / **DEFERRED** |

| Illustrative GS (gates.txt) | AC-Q: t = 0.25…1.5, n = 6 | AC-A: t = 1, 2, 3 (carry and Heston extrapolated beyond 1.61 y) |
|---|---|---|
| σ_ATM / √(w(T)/T) = √θ | 16.865% / 18.815% | 16.902% / 18.815% |
| V_BS / V_TS / V_H | 1.016721 / 1.014575 / 1.002359 | 1.034036 / 1.025874 / 1.009398 |
| gap V_H − V_BS (CRN SE) | −0.014362 (0.000171) | −0.024638 (0.000232) |
| term1: level, √θ vs σ_ATM (v0 = θ: no shape part) | −0.002146 (0.000061) | −0.008163 (0.000108) |
| term2: marginal smile at B_p, B_c (exact) | −0.015971 | −0.020257 |
| term3: path, forward skew, vol-of-vol | +0.003755 (0.000153) | +0.003782 (0.000212) |
| par coupon BS / H (difference, SE) | 5.228% / 7.547% (+2.319%, 0.032%) | 5.784% / 7.297% (+1.513%, 0.017%) |
| P(call at t_1) BS / H; P(never called); expected life | 0.5289 / 0.592; 0.2069 / 0.1696; 0.633 / 0.570 y | 0.5578 / 0.6457; 0.3057 / 0.2393; 1.748 / 1.594 y |
| BS at σ_H(S_ref) / at σ_H(B_p·S_ref) | 18.02% → 1.015565 / 26.54% → 0.996034 | 18.37% → 1.027910 / 24.17% → 0.995275 |
| CRN σ-grid, BS full product, σ = 0.10, 0.15, 0.20, …, 0.40 | 1.01730, **1.01773** (dV SE 0.00004), 1.01271, …, 0.94767 | 1.04433, 1.04008, 1.01997, …, 0.88936 (monotone) |

- **Deng–Mallett–McCann (2011) Case 2 is not reproduced.** Inputs: monthly dates, B_ac = B_c = 1.02, memory, accrual coupons, B_p = 0.8, σ 20%, F = 100e^{0.04t}, DF = e^{−0.06t}. qpricer MC gives 97.508 (SE 0.009, 1e6 paths) and the validator's exact pricer 97.507 (qpricer vs exact: z +0.09), against a published 98.39 (z −100.6). Case 1 (≈ par; ACX 99.976) and the Table 1 first-call p_i (max |z| 2.11) do match. The diagnosis is open: a different coupon or protection convention, or FD error the paper does not state. Nothing was tuned (OPEN_ISSUES). To represent Deng's schedule, `Autocall.c` now accepts per-period coupons (F7b).
- **Fixed in round 1.** F5: the gate used to print "PASS (report, illustrative)"; it now prints DEFERRED. F6: term1 was labelled "ATM term structure", but with v0 = θ, w(t) = θt is flat, so it is a pure level gap, √θ 18.82% vs σ_ATM 16.87%; it was relabelled (D-031, E3). F7c: the σ-grid diagnostic was added. Validator cross-checks: exact V_BS 1.016654 (MC 1.016721 ± 0.000065); its own QE-M gives V_H 1.002512 ± 0.000084.

## 5. Ten likely questions (basic → hard)
1. **Walk me through one path.** Use hand path (b) of `test_hand_paths` (quarterly, c_i = 0.02). X_1 = 0.65 < 0.70: no coupon, which goes to memory. X_2 = 0.80: pays c_1 + c_2 = 0.04. X_3 = 1.01 ≥ 1: called, pays 1 + 0.02. At T a fixing X = 0.65 (≥ 0.6, < 0.7) repays 1 with no coupon; X = 0.40 repays 0.40. Ties count as "above", and the protection is European: only X_n matters.
2. **Who is long and short what?** The investor is long a bond with random maturity and long the coupon digitals, and is **short** a European down-and-in put at 60%, i.e. a put plus 0.4 digital puts. The issuer holds the mirror position: long the DIP, short the digitals. On the no-call sheet V is strictly decreasing in σ (tested), so the investor is short vega and the issuer long vega at inception (eq. 10). [GK] Issuers hedge by selling vol, so autocall issuance is a steady supply of vol.
3. **How do you know the MC is right?** Closed forms: (11) for the no-call sheet; (12) for n = 2 by 1-D quadrature, split at the memory jump; σ = 0 priced by hand. Exact pathwise identities: Σp_i + P(alive at T) = 1; PV is affine in c; PV is monotone in B_c, B_p and memory. Then MC vs closed form (max |z| 1.79) and the validator's exact full-product pricer (max |z| 2.36 over 18 runs). The published Deng Case 2 is *not* matched: two independent implementations agree at 97.507, about 100 SE from 98.39, so it is not used as an oracle.
4. **How do you get the par coupon?** PV = a + c·b path by path, so on one path set ĉ = (1 − ā)/b̄, with a delta-method SE. AC-Q: 5.23% under BS, 7.55% under illustrative Heston. At the 8% input, V_BS > 1 on both sheets. There is no fee and no issuer spread in the discounting (D-025). [GK] Real notes target 1 − fee, and the issuer's funding spread raises the coupon.
5. **Why CRN, and what did it buy you?** The model gap is about 1% of notional. With CRN the gap SE is 0.000171; independent seeds would give about 0.000199 (arith. from the two printed SEs). The gain is modest because digitals cap the correlation: paths near a barrier flip (pitfall 8). For term1, where both legs are lognormal, CRN gives 0.000061 against about 0.000113 (arith.). The real benefit: each term is a per-path difference driven by Heston's own spot Brownian motion, so the decomposition is coherent and reports model differences, not noise.
6. **Give me the BSM vs Heston numbers.** Illustrative only, and the gate is deferred. AC-Q: V_BS 1.016721, V_H 1.002359, gap −0.014362 (0.000171). Split: term1 −0.0021 is level (Heston's ATM IV 16.87% is below √θ = 18.82%, the vol-of-vol convexity of ATM vol); term2 −0.0160 is the smile at the barriers; term3 +0.0038 is the path term. The par coupon rises by 2.32 points. AC-A shows the same pattern with a −0.0246 gap, but extrapolates beyond 1.61 y.
7. **Why is Heston below flat-ATM BSM?** term2 dominates. The put sits at 60%, where Heston's IV is 26.54% against σ_ATM 16.87%. The fatter left tail makes the issuer's DIP worth more and the 70% coupon digitals less likely to pay; the test asserts D⁻_{n,H}(B_p) > its flat-ATM value. BS at the *barrier* vol overshoots (0.996034 < V_H 1.002359): that vol prices the put, but the coupons and calls sit at other strikes, and the digital also needs the slope (eq. 9). Heston also calls more often at t_1 (59.2% vs 52.9%): with σ' < 0, D⁺(1) is above its flat value (§3.5 iii).
8. **(Hedging) Does the issuer have one vega sign?** On the no-call sheet, yes: the investor is short vega. On the full product there is no general sign. The AC-Q σ-grid rises from 1.01730 (σ 0.10) to 1.01773 (σ 0.15; +0.00043, dV SE 0.00004), then falls to 0.94767 at 0.40. My reading of §3.4 is a timing effect. As σ → 0 the note is called at t_1 for sure, because f_1 = 1.0099 ≥ 1 (eq. 13). Adding vol delays the call, so the investor collects more of the above-par coupon before the put matters. AC-A's grid is monotone, so this is not universal. [GK, trade press via §3.4; not computed here] Dynamically, the issuer's vega grows as spot falls, peaks above the barrier, and flips sign near expiry as the put becomes an in-the-money digital ("peak vega").
9. **(Risk) What is the model risk, and what can vanillas hedge?** Vanillas pin the marginals: the DIP is a static put plus digitals from the T-smile (eq. 7), and that is term2. They do not pin term3. Coupons and the put pay only on paths that stayed below 100%, so the price depends on the *conditional* (forward) smile, driven by vol-of-vol and spot–vol correlation. In Heston, ξ and ρξ are fixed by the vanilla fit, so this effect cannot be moved on its own. My reading: term3 (+0.0038, a quarter of the gap) is a residual. It also contains the fact that the no-call sheet bears the smile penalty on all paths, while the callable note bears it only on surviving ones. [GK; Farkas et al. 2022, abstract only] LV and LSV fitted to the same vanillas disagree here. Greeks: indicators break pathwise Greeks and bump noise grows like 1/h; the fixes are likelihood ratio, call-spread smoothing or Fries–Joshi (pitfall 1; not implemented).
10. **What would you change for production?** Price with the Phase 5 calibrated Heston, then LSV for control of the forward skew. Discount with the issuer's credit or funding spread, as Deng et al. do (none modelled, D-025). Add Greeks (smoothing, likelihood ratio or AAD), and replace the last period by its conditional expectation (pitfall 2: exact under BS, lowers variance). Use discrete dividends and a proper long end, since AC-A extrapolates carry and Heston. Fix N1: the par-coupon ratio does not yet handle vector coupon schedules. A C++ kernel.

## 6. Traps: do NOT say
1. **"On SPX, Heston prices the note 1.4% below BSM and the fair coupon is 7.55%."** Those numbers come from the illustrative GS set, not a calibration (ATM 16.9% vs the market's 18.3%). The gate is DEFERRED, and Phase 5 re-runs the script.
2. **"The investor is long the put / long vol"** or **"an autocallable is short vega, full stop".** The investor is short the 60% DIP. Only the no-call sheet has a clean sign (investor short vega); the full AC-Q product's value rises from σ 0.10 to 0.15 and then falls.
3. **"We reproduced Deng et al."** or **"term1 is the term-structure effect".** Only Case 1 and the Table 1 p_i match; Case 2 is 97.507 vs 98.39, not reproduced and not tuned. With v0 = θ, term1 is a pure level gap (√θ vs σ_ATM).
