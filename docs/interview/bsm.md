# Phase 1: Black-76 pricing, Greeks and implied-vol inversion
Sources: `src/qpricer/models/bsm.py`, `src/qpricer/iv.py`, `scripts/phase1_gates.py`, `docs/derivations/bsm.md` (eq. numbers), D-011..D-014, D-016, `docs/validation/phase1.md`.

## 1. Pitch (30 s)
Phase 1 is the Black-76 kernel that every later phase reuses: price, forward Greeks and implied vol, vectorised, with inputs (F, K, T, DF, σ) only. Under the T-forward
measure the forward is driftless, so rates, dividends and repo sit in F and DF (implied from parity in Phase 0). The price then depends only on x = ln F/K and total vol
s = σ√T. Implied vol is solved in normalised form on the OTM side, by Newton started at the point of maximum vega (Manaster–Koehler), which converges monotonically;
a bracket and a Brent fallback guard it. Measured: parity 1.137e-13 (gate 1e-12), IV round trip 6.54e-13 (gate 1e-10), Greeks within 5.5e-14 of a 60-digit reference.

## 2. Key formulas
1. **Price (1)** `V = DF·θ[F Φ(θd₁) − K Φ(θd₂)]`, `d₁,₂ = ln(F/K)/(σ√T) ± σ√T/2`: θ = +1 call / −1 put, DF = P(0,T), Φ(d₂) = Q^T(F_T > K).
2. **Parity (2)** `C − P = DF·(F − K)`: payoff identity plus E^T[F_T] = F. No model.
3. **Greeks (5)–(7)** `Δ_F = θ·DF·Φ(θd₁)`, `Γ_F = DF·φ(d₁)/(F σ√T)`, `vega = DF·F·φ(d₁)√T = σTF²Γ_F` (per unit vol; /100 per vol point). Γ and vega are the same for C and P.
4. **Theta (8), Black PDE (9)** `Θ = ∂V/∂t = Θ₀ + rV`, `Θ₀ = −DF·F·φ(d₁)σ/(2√T)`, `r = −ln(DF)/T`; `Θ + ½σ²F²Γ_F − rV = 0`.
5. **IV problem (11)–(12)** `β = (v − ι)/√(FK) = b(s)` (v = V/DF, ι = intrinsic, b = OTM normalised call); `b'(s) = e^{−x²/2s² − s²/8}/√(2π) > 0` ⇒ unique root; max vega at `s_c = √(2|x|)`.
6. **Delta-hedged P&L (from 4)** long the option, short Δ_F futures, premium financed at r: `dΠ = ½Γ_F(ΔF)² + Θ₀Δt = ½Γ_F F²[(ΔF/F)² − σ²Δt]`.

## 3. The code that matters
**3a. Price kernel: `src/qpricer/models/bsm.py:33-36` (`_d12`) and `:67-68` (`price`)**
```python
    pos = s > 0
    s1 = np.where(pos, s, 1.0)
    lim = np.where(x > 0, np.inf, np.where(x < 0, -np.inf, 0.0))
    return np.where(pos, x / s1 + 0.5 * s, lim), np.where(pos, x / s1 - 0.5 * s, lim)
    # ... price():
    F, K, T, DF, sigma, cp, x, s, d1, d2 = _inputs(F, K, T, DF, sigma, cp)
    return (DF * cp * (F * ndtr(cp * d1) - K * ndtr(cp * d2)) + 0.0)[()]  # +0.0: no -0.0 puts
```
- 33–34: `np.where` evaluates both branches, so s = 0 is swapped for a dummy 1 (no 0/0). 35: the σ = 0 limits d = +∞ / −∞ / 0 (ITM / OTM / ATM) make `ndtr` return 1 / 0 / ½,
  so price, delta, vega and theta reach their continuous limits with no special branch (D-012). 36: eq. (1). 67: broadcast and validate (otherwise ValueError).
- 68: the θ-form `ndtr(cp·d)` computes the small tail directly. `1 − ndtr(d)` would be ~10% wrong at d = 8, because Φ(−8) ≈ 6e-16 is only 5 ulps of 1. `+0.0` removes −0.0; `[()]` returns scalars.

**3b. Safeguarded Newton on the active set: `src/qpricer/iv.py:118-130`** (start at :108, Brent fallback at :135-137, F1 post-check at :141-145)
```python
        f = _b(a_, s) - be
        bp = _db(a_, s)
        lo = np.where(f < 0, s, lo)
        h = np.where(f > 0, s, h)
        ds = f / np.where(bp > 0, bp, np.nan)  # b' underflow -> NaN step -> bisection
        s_nt = s - ds
        s_new = np.where((s_nt > lo) & (s_nt < h), s_nt, 0.5 * (lo + h))
        res_ok = np.abs(f) <= 4.0 * EPS * be  # eq. (15), residual test
        nt_ok = np.abs(ds) <= tol  # Newton correction below tol (may land on a bracket end)
        done = res_ok | nt_ok | (np.abs(s_new - s) <= tol)  # eq. (15), step test
        fin = np.where(res_ok, s, np.where(nt_ok, s_nt, s_new))
        out[idx[done]] = fin[done] / sqT[idx[done]]
        n_iter[idx[done]] = n
```
- The unknown is s = σ√T; `a_` = |ln F/K| and `be` = β. 108: start at s_c = √(2a), or at the exact ATM inverse 2√2·erfinv(β). 118–121: residual and normalised vega (12).
  b increases in s, so f < 0 raises `lo` and f > 0 lowers `h` (the bracket starts at [0, 10√T]). 122–124: Newton step; if it leaves (lo, h), or b' underflowed (NaN fails both tests), bisect (rtsafe).
- 125–127: stop when the residual is ≤ 4εβ, the Newton correction is ≤ 1e-13√T (the D-011 fix) or the step is ≤ 1e-13√T. 128–130: keep the best iterate and store σ = s/√T. 131–132 shrink the active set.

**3c. Theta convention: `src/qpricer/models/bsm.py:125-127`, and its FD check `scripts/phase1_gates.py:62-65`**
```python
    V = DF * cp * (F * ndtr(cp * d1) - K * ndtr(cp * d2))
    theta0 = -DF * F * _phi(d1) * sigma / (2.0 * np.sqrt(T))
    return (theta0 - np.log(DF) * V / T)[()]  # r V as (-ln DF) V / T: no inf * 0 at tiny T
    # ... phase1_gates.py:
    r, h = -np.log(DF) / T, (T + 1e-5 * T) - T
    fd = (
        P(F, K, T - h, DF * np.exp(r * h), sig, otm) - P(F, K, T + h, DF * np.exp(-r * h), sig, otm)
    ) / (2 * h)
```
- 125–127: Θ = Θ₀ + rV per year (/365 per day). It is ∂V/∂t with F (same expiry), K, σ and the zero rate r held fixed, so DF = e^{−rτ} accretes towards 1. It is the dt term
  of the Black PDE: in P&L attribution, rV cancels the funding of the premium and Θ₀ = −½σ²F²Γ_F remains. 62: r is held fixed; `(T + 1e-5T) − T` makes the step exactly representable (Press §5.7).
- 63–65: moving calendar time by ±h changes τ and DF (by e^{±rh}) together. Bumping T alone measures Θ₀ and misses rV (pitfall 10). The FD runs on OTM options, whose Θ never crosses 0 (pitfall 11).

## 4. Validation story (independent validator, 54 tests; references: 60-digit `decimal` Black, complex step, own bisection)
| Gate (ROADMAP) | Measured | Tolerance |
|---|---|---|
| Parity, C and P priced independently, 200k random points, K/F ∈ [0.25, 4] | 1.137e-13 | 1e-12 abs |
| Parity, whole IV domain (K up to 8.2e4) | 1.455e-11 = 2.25 ulp of DF·max(F, K); 124/147,600 points > 1e-12, all at K ≥ 8.3e3 | float64 floor, 4 ulp (D-016) |
| IV round trip, input = 60-digit price rounded to double (376 points) | 6.54e-13 | 1e-10 |
| IV round trip, 300k random points + corners | OTM 1.05e-14 (note's estimate ≲ 1e-14), ITM 6.54e-13 (est. ~1e-12); max 12 Newton evaluations, 0 Brent | 1e-10 |
| Greeks vs central FD: implementer steps, lattice (D-014) / validator Richardson, 100k, ITM + OTM | Δ 2.0e-8 / 1.86e-10, Γ 9.96e-7 / 1.24e-6, vega 3.0e-7 / 2.52e-9, Θ 1.7e-8 / 1.10e-7 | 1e-5 rel |
| Greeks vs 60-digit FD (true error) | Δ 7.8e-15, Γ 8.9e-15, vega 8.7e-15, Θ 5.1e-14 (scaled; plain rel. 5.5e-14) | 1e-11 (validator) |
| Hull 5th ed. Black-76 example (via MathWorks `blkprice`) | C = P = 1.1166414565589438 vs 1.1166 | ±0.5e-4 |

- **Failed first: gamma FD (D-014).** At the note's step 1e-4·Fσ√T, OTM D₂ gamma reached 1.773e-5 > 1e-5 on 78 of 147,600 points, all at σ√T ≈ 0.007. The analytic Γ was right
  (vega = σTF²Γ holds to < 1e-12). The FD reference was noisy: price noise was 5.2e-15 = 600εV, from the cancellation FΦ(d₁) − KΦ(d₂), so the optimal step grows by 600^{1/4} ≈ 5.
  The minimax step 4.5e-4 predicted 1.8e-6 and measured 9.96e-7. Tolerance, grid and OTM evaluation are unchanged.
- **IV solver bisecting away from a converged root (D-011).** In the tail (|z| ≥ 2.7), noise in b exceeds 4εβ, so the residual test never fired. The Newton correction fell below 1 ulp
  of s, so the iterate landed on a bracket end, failed the strict (lo, h) test and was bisected (39 elements, 16–46 evaluations). Fix: also stop when |ds| ≤ 1e-13√T. Result on
  2 × 147,600: max 12 evaluations, mean 6.95, Brent 0 times. The Brent path is tested with `max_iter=1` (3.0e-13).
- **Deep-OTM silent wrong IV (validator F1, 8 failing tests).** For OTM prices ≲ 1e-280, `ndtr(d₂)` flushes to 0 below d ≈ −37.5 while Φ(d₁) does not. So b(s) was up to ~17× too large
  and Newton converged on the wrong function: V = 2.162e-305, T = 5, true σ = 1 gave σ̂ = 0 with status 0. Fix round 1: accept a root only if s > 0 and Φ(d₂) ≥ PHI_MIN ≈ 2.2e-305,
  otherwise return NaN + status 5 + a warning. The validator's 33-case test now passes (D-013).
- **Open:** F3 (ITM quote with 3 < |z| < 7.5: status 0, error up to 2.0e-4, at the conditioning limit; calibration uses OTM only), and the parity-gate wording (D-016; the owner must amend the ROADMAP).

## 5. Ten likely questions (basic → hard)
1. **Why Black-76 on the forward, not Black–Scholes with S, r, q?** Under the T-forward measure F is a martingale (dF = σF dW^T), so carry sits in F and DF and the price depends only on x and s.
   F and DF come from parity (D-001), so no assumed q leaks into the vols. Spot Greeks follow from the chain rule (10): F = Se^{(r−q)T} gives Δ_S = e^{−qT}Φ(d₁).
2. **What are Φ(d₁) and Φ(d₂)? Why price C and P separately in the parity test?** Φ(d₂) = Q^T(F_T > K) is an exercise probability under the forward measure, not the real world;
   DF·Φ(d₂) = −∂C/∂K is the digital. Φ(d₁) is the same event under the share measure, and DF·Φ(d₁) is the forward delta. Building P = C − DF(F − K) would make the test
   tautological and destroy OTM-put precision (pitfall 4).
3. **Why always invert the OTM option?** By parity the ITM time value equals the OTM price, with the same vega. But an ITM quote carries rounding ε·max(F, K), all of it on the time value:
   |δσ| ≈ |δv_tv|/(√(FK)√T b'). Measured: OTM 1.05e-14 vs ITM 6.54e-13.
4. **Walk me through the solver.** b' > 0, so there is one root inside the bounds (4); outside them it returns NaN + status. b is convex below s_c and concave above, so Newton from s_c (max vega)
   moves monotonically to the root without overshooting; ATM uses the exact erfinv inverse. The bracket is [0, 10√T]; it bisects if Newton leaves it, and Brent takes any leftovers.
   Measured: max 12 evaluations, mean 6.95, Brent never called on the gate grid.
5. **(Hedging) You are long an ATM call, delta-hedged daily. What is your P&L?** Short Δ_F = DF·Φ(d₁) futures, or Φ(d₁) forwards, since a forward's value moves by DF·ΔF (deterministic rates).
   dΠ = ½Γ_F(ΔF)² + Θ₀Δt = ½Γ_F F²[(ΔF/F)² − σ²Δt]: you earn realised minus implied variance, weighted by gamma. So the P&L is path-dependent: moves far from K pay little.
   Theta is the rent for gamma. For short-dated ATM options both scale like 1/√T.
6. **(Risk) Your theta is positive for an ITM call. Bug? Why is it not Hull's number?** Not a bug: Θ = Θ₀ + rV, and deep ITM rV > |Θ₀| (carry on the premium as DF accretes). Θ is positive at 50.6% of
   ITM lattice points; on OTM points max rV/|Θ₀| = 0.688. Hull holds S, r, q fixed: Θ_S = Θ − (r−q)FΔ_F (10). On Hull's Greek-letters example: kernel −3.027/yr, Hull −4.305/yr, and (10) gives −4.30533/yr.
   (The validator recalled Hull's values without re-reading the page, so cite them as recalled.)
7. **(Risk) Vega: units, link to gamma, bucketing?** Per unit vol: Hull's 12.1 (recalled) is reproduced as 12.1055, i.e. 0.121 per vol point. vega = σTF²Γ_F, so one option is long gamma and long vega together.
   Across expiries they split (Γ ~ 1/√T, vega ~ √T): long 1y / short 1m is short gamma and long vega. Each expiry has its own (F, DF, σ), so bump one (expiry, strike) node by 1 vol point,
   re-price, and hedge with options in that bucket. A single parallel vega hides term-structure and skew risk.
8. **Your gamma failed the 1e-5 FD gate. Did you loosen it?** No. Tolerance, grid and OTM evaluation are unchanged; only the step of the FD reference moved. Price noise was 600εV, not εV, so the
   minimiser of h²|f''''|/12 + 4ε|f|/h² grows by ≈ 5 (D-014), and the measured error is 9.96e-7. The validator confirmed the analytic Γ to 8.9e-15 against a 60-digit FD,
   and passed ITM gamma, which the implementer had not FD-tested, at 1.24e-6.
9. **Hard: can your solver return a wrong vol silently?** It could (F1). Below ~1e-280, `ndtr` lost Φ(d₂), so Newton converged on a function up to ~17× too large and returned σ̂ = 0 with status 0.
   Fix: post-check Φ(d₂) ≥ PHI_MIN at the root, and never accept s ≤ 0 (status 5). A threshold on β alone fails, because β can be ~1e-280 after Φ(d₂) has already underflowed.
   Still open: deep-ITM quotes can be off by 2.0e-4 with status 0 (F3). The fix is to return σ_err = ε·max(v, F, K)·DF/vega in `info`.
10. **What would you change for production?** Jäckel (2015), "Let's be rational": rational initial guesses plus two Householder steps reach machine precision (D-011), with asymptotic/log-space
    branches for the wings behind F1 (pitfall 6) and no Python Brent loop. C++ kernels with AAD Greeks. A smile-consistent delta (vanna·∂σ/∂F; choose sticky strike or sticky delta) instead of the
    flat-vol delta. An OIS/collateral DF curve instead of short-end parity DFs (Phase 0, F5). Expose σ_err (F3).

## 6. Traps: do NOT say
1. "Long options always have negative theta", or "my theta is −∂V/∂T, like Hull's". The kernel Θ (F and r fixed, DF accretes) is positive at 50.6% of ITM lattice points, and gives
   −3.027/yr vs Hull's −4.305/yr on the same option. Only Θ₀ (fixed DF) is always negative.
2. "Parity holds to 1e-12 everywhere", or "I got the put from the call". 1e-12 is gated on K/F ∈ [0.25, 4]. On the full IV domain the float64 floor gives 1.455e-11
   (2.25 ulp, D-016). C and P are always priced independently.
3. "Brent makes the solver bulletproof", or "Φ(d₂) is the probability that the option is exercised". Brent never ran on the gate grid. F1 is fixed by flagging (NaN + status 5),
   not by resolving the tail, and F3 is still open. Φ(d₂) is a Q^T probability, not a real-world forecast.
