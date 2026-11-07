# Phase 0: SPX option data and the parity-implied forward / discount factor
Sources: `src/qpricer/{data,market}.py`, `scripts/{fetch_spx,build_market}.py`, D-001..D-010, `docs/validation/phase0.md`; math in docstrings.

## 1. Pitch (30 s)
Phase 0 turns one raw Yahoo ^SPX snapshot (2026-11-06, 15843 rows, 40 expiries 7 d–2 y) into what every kernel needs per expiry:
forward F, discount factor DF, time T, and a clean OTM surface for calibration. I assume no rate curve and no dividend yield.
SPX options are European, so put–call parity C − P = DF·(F − K) is model-free. A robust regression of C − P on K over the 20
near-ATM pairs gives DF from the slope and F from the zero crossing. This sidesteps the discrete dividends and borrow, keeps F consistent
with the quotes, and doubles as a data-quality check. Around it: one settlement convention per expiry (AM SPX vs PM SPXW), quote filters,
stale-quote rejection and a static-arbitrage filter. Result: 39/40 expiries kept, min R² 0.999941, DF ∈ [0.923895, 0.998395], F_se 0.03–0.40 pt.

## 2. Key formulas
1. **Parity (P1)** `C(K) − P(K) = DF·(F − K)`: call − put = forward struck at K; DF = PV of 1 paid at T. No r, q, spot or model.
2. **Estimator (P2)** OLS `y = a + b·K` on y = C − P gives `DF = −b`, `F = a/DF = K̄ − ȳ/b`, `r = −ln(DF)/T`.
3. **Errors (P3)** `se(DF) = s/√Sxx`, `se(F)² = s²/(n·b²) + ȳ²·s²/(Sxx·b⁴)`, s² = SS_res/(n−2); ȳ ≈ 0 on an ATM window, so F is robust.
4. **What F contains** `F = (S₀ − PV(divs))/DF = S₀·e^{(r−q−b)T}` (q = dividend yield, b = borrow/repo): parity gives the net carry.
5. **Static arbitrage (D-010)**, K_i < K_j, on bid/ask: calls `bid_j ≤ ask_i`, `bid_i − ask_j ≤ K_j − K_i` (puts mirrored); using DF ≤ 1 makes them model-free.
6. **Clock (D-003)** `T = (t_settle − t_snap)/(365·86400 s)`; settle at 09:30 ET for SPX (AM), 16:00 ET for SPXW (PM), 13:00 on early-close days.

## 3. The code that matters
**3a. Parity OLS: `src/qpricer/market.py:76-82`** (71–73 centre the strikes: `Kc = K − K̄`, `Sxx = Kc·Kc`)
```python
    b = (Kc @ (y - ybar)) / Sxx
    res = y - ybar - b * Kc
    ss_res, ss_tot = res @ res, (y - ybar) @ (y - ybar)
    s2 = ss_res / (n - 2)
    DF = -b
    F = Kbar - ybar / b
    F_se = np.sqrt(s2 / (n * b**2) + ybar**2 * s2 / (Sxx * b**4))
```
- Centring is needed because K ≈ 7400 makes the raw normal equations ill-conditioned; it also decorrelates ȳ and b. 76: closed-form
  slope. 77–79: residuals, sums of squares for R², unbiased s² (n − 2 dof). 80–81: invert P2 (DF = −slope; F = zero crossing).
  82: delta-method se of F. The slope term carries ȳ² ≈ 0, which is why F is identified even where DF is not.

**3b. Window and stale-quote rejection: `market.py:121-129`** (OLS on the kept pairs follows; in-band gate at 142–144)
```python
    y = C - P
    i0 = np.argmin(np.abs(y))
    F0 = K[i0] + y[i0]
    sel = np.sort(np.argsort(np.abs(K - F0), kind="stable")[:n_near])
    Ks, ys, hs = K[sel], y[sel], h[sel]
    a, b = theil_sen(Ks, ys)
    e = ys - a - b * Ks
    mad = 1.4826 * np.median(np.abs(e - np.median(e)))
    keep = np.abs(e) <= np.maximum(z * mad, hs)
```
- 121–125: first-guess F at the C − P sign change (DF ≈ 1), then the 20 nearest pairs (D-004), because deep-ITM legs are wide and stale.
- 126–128: Theil–Sen line (median pairwise slope, breakdown 1 − 1/√2 ≈ 29.3%); 1.4826·MAD is a robust σ. 129: reject a pair only
  if it is > 4 robust σ off **and** outside its own C − P half-spread h_i (D-005). Both legs then leave the surface.
- An expiry is kept (`market.py:176-187`) iff: n_pairs, n_used ≥ 10; n_rej/n_sel < 0.293; R² > 0.999; max|res_i|/h_i ≤ 1
  (line inside every used quote band); DF ∈ (0.8, 1]. Otherwise it is dropped, with a WARNING and a `reason`.

**3c. Static-arbitrage pairs: `src/qpricer/data.py:175-182`**
```python
    lo_b, hi_b = bid[:, None], bid[None, :]
    lo_a, hi_a = ask[:, None], ask[None, :]
    dK = K[None, :] - K[:, None]
    if put:
        v = (lo_b > hi_a) | (hi_b - lo_a > dK)
    else:
        v = (hi_b > lo_a) | (lo_b - hi_a > dK)
    return np.triu(v, k=1)
```
- 175–177: n×n matrices by broadcasting (row i = low strike, column j = high strike). 181, calls: `bid_j > ask_i` means you buy the
  low-K call and sell the high-K call for a credit, payoff ≥ 0. `bid_i − ask_j > dK` means you sell a call spread above its max payoff. 179: puts. 182: all i < j pairs.
- Driver `data.py:202-209`: greedily drop the quote with the most violations (ties: oldest lastTradeDate, then lowest K) until none remain.

## 4. Validation story (independent validator, 26 tests; full suite 52 passed, 0 failed)
| Gate | Measured | Tolerance |
|---|---|---|
| G1 parity R², each kept expiry | min 0.999941 (2027-02-19); 39/39 | > 0.999 |
| G1b failing expiries dropped + logged | only 2026-12-08 (7 clean pairs < 10) | drop + log |
| G2 implied DF | [0.923895, 0.998395] | (0.8, 1] |
| C6 rebuilt from raw, other primitives | \|ΔF\| ≤ 9.1e-13, \|ΔDF\| ≤ 7.8e-16 | 1e-10 |
| C7 second method: LAD (L1), no rejection | max \|ΔF\|/F_se 1.87 (0.61 pt), \|ΔDF\|/DF_se 2.43 | < 3 SE |
| C8 no hidden r/q | spot / Yahoo IV perturbed: bit-identical; synthetic DF = 0.83 recovered to 1e-12 | none |
| A1 calibration surface static-arb free | 0 violations (plain OTM cut: 468 flags) | 0 |

Funnel: 15843 → 14707 (one root/date) → 14236 (bid > 0) → 14228 (ask ≥ bid) → 13608 (spread/mid ≤ 0.5) → 12956 (OI > 0)
→ 7650 (OTM cut, 39 expiries) → 7625 (static arbitrage).
- **Why rejection exists:** plain OLS R² was 0.12 / 0.39 / 0.93 on 2027-03-31 / 2027-06-30 / 2028-01-21; rejecting 9 stale strikes gives ≈ 0.99998.
- **F1, early close (failed first):** 2026-11-27 SPXW settles at 13:00, so T was 3 h too long (0.0573937 → 0.0570512); r 5.643% → 5.677%.
- **F2, rejection rescuing noise:** with 20 points the MAD is itself noisy, so trimming lifted 4/500 synthetic pure-noise chains
  (σ = 2 pt, h = 0.5) over R² > 0.999. Fix: in-band + breakdown rules, now 0/500 (R² alone passes 61% at σ = 0.9). Snapshot headroom: 0.814/1, 0.20/0.293.
- **F4, stale quotes on the OTM surface:** 25 strict adjacent violations on 5 expiries. Example: the 7615 call bid 434.4 vs the 7610 call
  offered at 258.5 (2027-03-31). D-010 removes 25 quotes (last traded 2026-07-29..09-24), each necessary, 14.4–179.9 pt off their neighbours.
- **Open:** F5. For T < 0.1, r spans 2.64–16.89% (se 0.4–4.3%) vs 4.5–5.1% for T > 0.3, and DF rises with T 10/38 times. Impact
  ≲ 0.06 vol pt ATM; deferred to Phase 5. F7: the in-band gate keeps 96.5% / 55.6% of *valid* synthetic chains at σ/h = 0.35 / 0.5 (conservative).

## 5. Ten likely questions (basic → hard)
1. **Why imply F and DF instead of using r and q?** Discrete, uncertain dividends and unobserved borrow. An assumed q mis-states F,
   which shifts every k = ln K/F and splits call/put IVs. Spot is asynchronous: the 7-day F is 0.13% above spot 7376.86, which would mean a ~6.7% carry (D-008).
2. **Is parity exact here? On SPY?** Yes for SPX: European, cash-settled, a static replication. SPY options are American on a
   dividend-paying ETF, so you only get bounds (S − D − K ≤ C − P ≤ S − K·DF), not a line.
3. **Why does SPX vs SPXW matter?** SPX monthlies settle AM on the special opening quotation; SPXW settle PM on the close. Different
   prices and a 6.5 h T gap (≈ 7.4e-4 y). One root per date, SPX preferred (e.g. 139 vs 44 clean pairs); forwards agree within 0.27 pt.
4. **Why OTM only?** Liquid side, tight relative spreads. An ITM price is mostly intrinsic, so its spread turns into a big IV error.
   With parity-consistent F and DF, call and put IVs at the same K coincide, so nothing is lost. The switch is at K = F.
5. **Which arbitrage, and why bid/ask?** Monotonicity and the vertical spread on all pairs, on tradeable quotes, so only executable
   arbitrage is flagged. Butterfly and calendar checks are Phase 5. There is no lastTradeDate cutoff: good quotes often have old trades.
6. **(Risk) Kernels take F: how do you get spot delta and dividend risk?** Chain rule through a carry model: ∂V/∂S = ∂V/∂F·F/S. With
   F = S/DF − FV(divs), each forward dividend point moves V by −∂V/∂F. Hedge with ES (SOQ-settled, like the AM forward). Autocall sellers
   are long the client's down-and-in put, hence long dividends. Parity gives the implied dividend, not a hedgeable curve.
7. **(Risk) What does a forward or DF error cost?** ATM ∂C/∂F = DF·N(d₁) ≈ ½, so F_se ≤ 0.40 pt is ≲ 0.2 pt of premium. Worse is the
   k shift and the call/put IV gap, which calibration absorbs as fake skew (ρ). DF is second order: ΔDF/DF ≲ 0.4% gives ≲ 0.06 vol pt ATM (D-007).
8. **Why is F identified at the short end but DF not? Is r = 16.9% real?** No. The slope signal (1 − DF)·ΔK is ≈ 0.05–0.1 pt, below
   quote resolution, and it grows with window width (ITM-quote bias): 20 → 40 pairs moves short DF by up to 9.9 DF_se. F has ȳ ≈ 0, so it is
   robust. The long-end 4.5–5.1% is the box-spread rate, [C−P](K₁) − [C−P](K₂) = DF·(K₂−K₁), not OIS.
9. **Hard: R² passes, so why the in-band rule? Why not trim the worst pair until it passes?** R² scales with the strike range (F3), so it is
   an uneven gate. "Line inside every used quote band" has the economic content. Iterative trimming re-opens F2 (it eats noise until R² passes).
   The rule is fixed a priori and failing expiries are dropped. The cost is F7: 2026-11-17 is lost at n_near = 40 (max|res|/h = 1.057).
10. **What would you change for production?** Synchronous OPRA/vendor NBBO at a fixed time, with quote timestamps for staleness.
    An OIS (SOFR) discount curve for CSA trades, with the box rate as a funding spread and a smooth monotone DF(T) (F5). A joint F(T) from
    parity + ES + dividend futures, split into dividend and repo curves for hedging. 1/h²-weighted fits and a trading-time clock for weeklies.

## 6. Traps: do NOT say
1. "The parity DF is the risk-free / OIS rate", or quoting 16.9% as a short rate. The short end is not identified (D-007, F5); the long end is the box rate.
2. "I built F from spot and a dividend yield", or "the 0.13% F-to-spot gap is carry". No r, q or spot enters any kernel (C8), and spot is asynchronous (D-008).
3. "The surface is arbitrage-free" or "R² proves the forwards". Only strict monotonicity / vertical-spread arbitrage in K is enforced; convexity
   and calendar are Phase 5. `["otm"]` still holds 468 flags, so calibrate on `["surface"]`. The in-band rule and F_se carry the evidence.
