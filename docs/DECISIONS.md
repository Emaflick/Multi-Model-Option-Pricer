# Decision log

Format: `## D-NNN — title` · Context · Decision · Alternatives · Consequences.

## D-001 — Forward-based pricing kernels
**Context:** SPX has a discrete, uncertain dividend stream; r and q are not directly observed.
**Decision:** kernels take (F, DF, T); F and DF per expiry are implied from put-call parity on market quotes.
**Alternatives:** OIS curve + dividend-yield fit; futures-implied forward.
**Consequences:** carry assumptions do not leak into the vol surface; the parity fit doubles as a data-quality check.

## D-002 — One contract root per expiry date; SPX (AM) preferred over SPXW (PM) on overlap dates
**Context:** yfinance's ^SPX chain mixes `SPX…` (AM-settled standard monthlies/LEAPS) and `SPXW…`
(PM-settled weeklies, EOM, quarterlies) contracts, identified by the `contractSymbol` root. In the
2026-11-06 snapshot the five 3rd-Friday dates 2026-11-20 … 2027-03-19 list both roots on the same date and
strike. AM and PM contracts settle on different prices (Friday opening SOQ vs Friday close), so mixing them
in one parity regression or one smile is wrong.
**Decision:** keep exactly one root per expiry date (`data.preferred_root_mask`): SPX where it is listed,
SPXW otherwise. Each root keeps its own settlement time in T (D-003).
**Alternatives:** (a) SPXW only — would discard all 11 SPX-only expiries 2027-04-16 … 2028-06-16 (no 2y
surface). (b) both roots as separate slices — the SPXW slices on monthly dates are thin and add a
same-date AM/PM pair to every calendar-arbitrage check.
**Consequences:** on overlap dates the SPX root has far more clean put-call pairs (77/46, 139/44, 129/39,
159/11, 77/9 for SPX/SPXW; `scripts/build_market.py --compare-roots`). The two roots' forwards agree within
0.1–0.4 index points (2–3 standard errors), so the choice does not move F. 1136 raw rows are dropped by this
filter.

## D-003 — Valuation instant, settlement instants, ACT/365F
**Context:** the snapshot was downloaded after the US close (2026-11-06 22:13:53 UTC = 17:13 ET; latest
`lastTradeDate` 16:17:53 ET). Short expiries are sensitive to the hour convention.
**Decision:** `snapshot_utc` = start of the download (also the file stamp `spx_<YYYYMMDD_HHMM>` in UTC,
and the `qpricer` key of the parquet schema metadata). Settlement instant = expiry date at 16:00 ET for
SPXW (PM, closing print) and 09:30 ET for SPX (AM, settled on opening prints), with zoneinfo
America/New_York for EST/EDT. On NYSE early-close days the PM close, and hence the SPXW settlement, is at
13:00 ET. `data.EARLY_CLOSE` holds 2026-11-27, 2026-12-24, 2027-11-26, 2028-07-03 and 2028-11-24, taken
from the 2026–2028 table at nyse.com/markets/hours-calendars (retrieved 2026-11-06). 2027-12-24 is a full
holiday (Christmas observed), not an early close. NYSE publishes no dates after 2028, so
`settlement_utc` warns for PM expiries after 2028-12-31 and assumes 16:00 there. AM settlement (09:30)
is unaffected. T = seconds(settle − snapshot) / (365·86400), calendar time, no trading-day or holiday
adjustment. Holiday-shifted expiries are used as listed, e.g. Thursday 2027-06-17.
**Alternatives:** valuation at the 16:15 ET option close; whole-day ACT/365 counts; a trading-time clock.
**Consequences:** the ~1 h between the end-of-session quotes and the snapshot is ignored (≈ 0.6% of T for
the 7-day expiry, and it is non-trading time). AM vs PM on the same date differ by 6.5 h ≈ 7.4e-4 y.
The early-close rule (validator finding F1) shortens T for 2026-11-27 SPXW from 0.0573937 to 0.0570512
(−3 h). F, DF and R² do not depend on T; implied r on that expiry moves from 5.643% to 5.677%.

## D-004 — Near-ATM window for the parity regression: N_NEAR = 20 pairs, N_MIN = 10
**Context:** C − P = DF·(F − K) holds at every strike. However, deep-ITM legs carry wide absolute spreads
and stale quotes, and an R² gate on very few points is meaningless.
**Decision:** pairs = strikes with both a clean call and a clean put on the expiry. Preliminary
F0 = K* + (C−P)(K*) at K* = argmin|C − P| (the sign change). Use the 20 pairs nearest F0 (all of them if
fewer). Drop the expiry if fewer than 10 pairs exist or remain after D-005. R² is computed on ≥ 10 points.
With 5-pt strike spacing that means SS_tot ≈ 2·10³, so R² > 0.999 requires a residual RMS below ≈ 0.45 index
points, comparable to quoted half-spreads. That tolerance grows with the window's strike range (validator
F3), so the in-band condition of D-005 is the gate with economic content. The table records `logm_max`
(widest |ln K/F| used), `n_used` and `bracketed`. `bracketed` = F inside [min K, max K] of the used pairs
(validator F6): it is a flag plus a WARNING, not a gate, because parity is exactly linear and extrapolation
shows up in F_se. It is False only for 2026-12-07: 13 pairs at K ∈ [7425, 7800], F = 7396.42,
F_se = 0.40.
**Alternatives:** a fixed log-moneyness band (strike density varies 5–100 pts across expiries); 10 or 40
pairs. `scripts/build_market.py --n-near 10` keeps 34/40 expiries: short-end noise pushes DF above 1 on
3 weeklies, and 2 quarterlies fall below N_MIN. `--n-near 40` keeps 38/40 with tighter long-end r but a
larger short-end bias (D-007); 2026-11-17 is dropped by the in-band rule (max|res|/h = 1.057, driven by
ITM-put pairs at K = 7490/7500 — validator F7). The in-band rule caps the single worst pair, so a valid but
noisy expiry can be lost: on valid synthetic chains it keeps 100% / 99.9% / 96.5% / 55.6% at noise =
0.2 / 0.3 / 0.35 / 0.5 × half-spread (`docs/validation/phase0.md`, F7). This is conservative (an expiry is
lost, never a bad F accepted) and is kept deliberately. Against the 20-pair fit, F moves by a median 0.03–0.05 index points
(max 1.5 pt = 2e-4 relative, 2026-12-10 with 14 pairs) for 10 or 40 pairs.
**Consequences:** 2026-12-08 (7 clean pairs) is dropped. F's standard error is 0.03–0.40 index points on
every kept expiry.

## D-005 — Stale-quote rejection inside the parity fit, with in-band and breakdown-point acceptance
**Context:** the Yahoo chain contains stale quotes hundreds of points away from their neighbours. Example,
2027-03-31 SPXW: put K=7455 is bid/ask 212.9/214.9 while the 7450 and 7460 puts are 317.8/319.3 and
321.3/322.9 (last trade 2026-07-29); call K=7465 is 543.0/544.1 between 344.4/346.3 and 338.2/339.9.
Plain OLS on the 20-pair window then gives R² = 0.12 (2027-03-31), 0.39 (2027-06-30) and 0.93 (2028-01-21).
**Decision:** fit a Theil–Sen line (median pairwise slope, intercept = median(y − bK)) to the window and
reject pair i if |e_i| > max(4·1.4826·MAD(e), h_i), where h_i = half-width of the synthetic C − P quote.
A pair is never rejected while the robust line lies inside its own bid-ask band. OLS (eqs. P1–P3 in
`market.py`) is then run on the rest. The rule is fixed a priori and applied to every expiry. `R2_all`
(pre-rejection OLS), `n_rej` and `rejected_K` are written to the table. Both legs of a rejected strike are
removed from the OTM surface, because parity cannot tell which leg is stale.
Two acceptance conditions are added on top of the ROADMAP gates (validator finding F2: with 20 points the
MAD scale is itself noisy, so the rejection alone could remove ordinary noise and lift a noisy expiry over
R² > 0.999; that happened in 4 of 500 synthetic σ = 2 chains):
(i) **in band:** max_i |res_i| / h_i ≤ 1 over the used pairs, with res the final OLS residual. The fitted
parity line must lie inside every used pair's own C − P bid-ask band, so rejection cannot rescue noise
wider than the quotes. Measured on the snapshot: max 0.814 (2026-11-18).
(ii) **breakdown:** n_rej / n_sel < 1 − 1/√2 ≈ 0.293, the Theil–Sen breakdown point. Above it the robust
line, and so the rejection, is not trustworthy. Measured: max 4/20 = 0.20.
Neither condition changes the R² or DF gates. Both only add ways to drop an expiry, and no expiry of this
snapshot is lost.
**Alternatives:** OLS only, dropping the three expiries (loses two quarterlies whose other 16 pairs fit with
R² ≈ 0.99998); a lastTradeDate staleness filter (bid/ask are live market-maker quotes and many good
quotes have old last trades); raising the rejection floor to k·h_i with k ≥ 3 (also suggested by the
validator; not needed once (i) holds, and it would move away from the replicated rule).
**Consequences:** 9 strikes are rejected in 3 expiries (4 + 4 + 1). No pair is rejected on the other 36
kept expiries. Stale quotes on wing strikes that have no partner leg are not detectable here; D-010
removes them from the surface.

## D-006 — Cleaning filters and the after-close snapshot
**Context:** after the close Yahoo may show zero bid/ask.
**Decision/observation:** bid/ask are populated: only 471 of 14707 rows (3.2%, after D-002) have zero or
NaN bid. The filters are applied in this order, with rows in/out saved to
`data/processed/cleaning_<stamp>.csv`: bid > 0; finite ask ≥ bid (added to the roadmap list: the relative
spread is undefined otherwise, 8 rows); (ask − bid)/mid ≤ 0.5; open interest > 0. NaN fails a filter and
is counted, never masked. Contract symbols are cross-checked against the strike/type/expiry columns
(mismatch raises).
**Consequences:** 15843 raw → 12956 clean rows. OTM cut (puts K < F, calls K ≥ F, parity F): 7650
quotes on 39 expiries. After the static-arbitrage filter (D-010): 7625. `cleaning_<stamp>.csv` keeps
exactly the chain filters above; the surface stages are in `surface_<stamp>.csv`.

## D-007 — Short-end DF is not identified by parity on mid quotes; F is (known limitation)
**Context:** the slope signal is (1 − DF)·ΔK ≈ 0.05–0.1 index points across a 1–2 week window, below the
quote resolution.
**Observation:** for T < 0.1 the implied r = −ln DF / T spans 2.6%–16.9% (se 0.4%–4.3%). For T > 0.3 it is
4.5%–5.1% (se ≤ 0.36%). Widening the window raises the short-end r (D-004), which is consistent with a
systematic ITM-quote bias rather than noise. F is unaffected (F = K̄ − ȳ/b with ȳ ≈ 0 on a centred
window; se ≤ 0.4 pt).
**Decision:** Phase 0 keeps the per-expiry parity DF (gates pass; no term-structure model is assumed).
Rough impact: ΔDF/DF ≲ 0.4% changes undiscounted prices by the same relative amount, i.e. ≲ 0.06 vol
points at the money for σ ≈ 15%.
**Alternatives (Phase 5 if needed):** smooth DF(T) by a fit of r(T) on the long end; use an OIS curve.

## D-008 — Spot is informational only
**Context:** Yahoo's `regularMarketPrice` (7376.86, stamped 21:37 UTC) is not synchronous with the option
quotes. The first parity forward (7-day) is 7386.28, 0.13% above spot, which would imply an
annualised carry of ~6.7%.
**Decision:** spot is stored (column + metadata) for sanity checks only. No kernel and no carry estimate
uses it (D-001). A likely cause is that the option quotes are end-of-session (16:15 ET) while the index
closes at 16:00 ET. This is unverified.

## D-009 — Real-data gate test fails (never skips) without the raw snapshot
**Context:** `data/raw/*.parquet` is gitignored; `data/processed/market_<stamp>.csv` is committed.
**Decision:** `tests/test_market.py::test_real_snapshot_parity_gates` takes the latest committed
`market_*.csv`, recomputes the pipeline from the matching `data/raw/spx_<stamp>.parquet`, asserts the
gates on every kept expiry (R² > 0.999, DF ∈ (0.8, 1], n_used ≥ 10), checks non-vacuity (≥ 75% of
expiries kept, kept T reaches ≥ 1 y) and the OTM cut, and checks that the CSV is reproduced to rtol 1e-12. If the raw
file is missing the test calls `pytest.fail`.
**Consequences:** on a fresh clone, run `scripts/fetch_spx.py` then `scripts/build_market.py` (new stamp)
before the full suite passes.

## D-010 — Static-arbitrage filter on the OTM surface (stale quotes in K)
**Context:** the ROADMAP filters and the parity rejection (D-005, which only sees the 20-pair window) leave
stale quotes on the OTM surface. The validator (finding F4) counted 25 strict adjacent-strike violations
on 5 expiries. Example: the 2027-03-31 call K=7615 is bid 434.4 while the 7610 call is offered at 258.5
(IV 26.7% vs about 17% for its neighbours).
**Decision:** a separate stage after the OTM cut, `data.remove_static_arbitrage`. For each
(expiry, option type) slice, a pair K_i < K_j is a strict static arbitrage on tradeable quotes, using
DF ≤ 1 and no model, if:
- calls: bid_j > ask_i (C non-increasing in K) or bid_i − ask_j > K_j − K_i (vertical spread);
- puts: bid_i > ask_j (P non-decreasing in K) or bid_j − ask_i > K_j − K_i.

All pairs are checked, not only adjacent ones. Removal is greedy and deterministic: drop the quote in
the most violating pairs; break ties by the oldest `lastTradeDate`, then the lowest strike; repeat until
no pair violates. A stale quote conflicts with every good quote it crosses, so it has by far the most
violations. Output: `build_surface()["surface"]` is the calibration input. Rows in/out go to
`data/processed/surface_<stamp>.csv` and the removed quotes to `arb_removed_<stamp>.csv`.
`cleaning_<stamp>.csv` and `build_surface()["otm"]` are deliberately unchanged: they are the ROADMAP
filter list and plain OTM cut that the independent validator reproduces.
**Alternatives:** adjacent-only checks (miss stale pairs that cross non-adjacent strikes); a lastTradeDate
cutoff (good quotes also have old trades); butterfly/convexity and calendar checks (left to Phase 5, where
they are diagnostics on the surface).
**Consequences:** 25 quotes are removed on 5 expiries: 2026-12-31 (1 P), 2027-03-31 (1 C, 4 P),
2027-06-30 (6 C, 9 P), 2028-01-21 (2 P), 2028-06-16 (2 P). All of them last traded between 2026-07-29
and 2026-09-24, and their mids sit 14–177 points from the nearer neighbour. No strict violation remains,
so the surface goes from 7650 to 7625 quotes.

## D-011 — IV solver: vectorised safeguarded Newton, non-vectorised Brent fallback
**Context:** eq. (15) of `derivations/bsm.md` sends non-converged elements to `scipy.optimize.brentq`,
which is scalar.
**Decision:** `iv.implied_vol` runs the safeguarded Newton (eq. 14) vectorised on a shrinking active
set. Whatever remains after `max_iter = 100` goes through a Python loop of `brentq(xtol=1e-15,
rtol=4*eps)`, one call per element, on that element's final Newton bracket. The bracket encloses the
root by construction (b(lo) < beta <= b(hi)). An element has converged when one of three tests holds:
the residual |b - beta| <= 4 eps beta; the Newton correction |ds| <= 1e-13 sqrt(T); or the
safeguarded step |s_new - s| <= 1e-13 sqrt(T) (eq. 15).
The Newton-correction test was added during implementation. In the deep-OTM tail (|z| >= 2.7), the
rounding noise in b exceeds 4 eps beta, so the residual test never fires, and the correction falls
below one ulp of s. The iterate then landed exactly on a bracket end, failed the strict (lo, h) test
and was bisected away. That cost 39 elements 16–46 evaluations. With the test, the iteration
stops (rtsafe semantics, Press et al. §9.4).
**Alternatives:** a vectorised Brent or bisection loop (more code, for a path that is never taken on
the gate domain); Jäckel (2015) "Let's be rational" (two Householder steps to machine precision,
much more code).
**Consequences:** on the gate grid (2 × 147,600 elements, `scripts/phase1_gates.py`) Newton takes at
most 12 evaluations (mean 6.95) and the Brent fallback runs 0 times. The fallback path is tested
with `max_iter=1`. It sends 4200 of 4620 elements to brentq at 0.14 ms per element, and the
round-trip error is 3.0e-13.

## D-012 — sigma = 0 limits of the Black-76 kernels
**Context:** pitfall 5 of `derivations/bsm.md`: at s = sigma sqrt(T) = 0, d_{1,2} = x/s is 0/0 at F = K.
**Decision:** sigma = 0 is accepted; T <= 0 and sigma < 0 raise. At s = 0, d_{1,2} are set to their
limits: +inf for x > 0, -inf for x < 0, 0 for x = 0. Eqs. (1), (5), (7) and (8) are then evaluated
unchanged and return the continuous limits. Gamma (6) is set to 0 for x != 0 and +inf for x = 0.
One limit differs from the note: at x = 0, vega is the right-derivative DF F sqrt(T)/sqrt(2 pi),
because C ≈ DF F s/sqrt(2 pi) (Brenner–Subrahmanyam). The note lists 0. Away from x = 0, vega is 0,
as in the note. Kernels also accept DF > 1 (negative rates), while the note states DF in (0, 1].
**Consequences:** there is no special-case branch in the price, delta, vega or theta code.
`test_sigma_zero_limits` checks the exact limit values.

## D-013 — Implied vol: invalid prices give NaN + warning + status, invalid market inputs raise
**Context:** CLAUDE.md forbids silent clipping and silent NaN. Real chains contain stale or crossed
quotes.
**Decision:** if F, K, T or DF is <= 0 or non-finite, or cp is not +-1, the call raises ValueError
(a programming error). Price-level failures give NaN. Each call emits one RuntimeWarning that
counts the failures per type, and `full_output=True` returns a per-element status:
1 = v <= intrinsic, or time value <= 10 eps v (pitfall 7; for OTM quotes this is just v > 0);
2 = v >= F (call) or K (put), eq. (4); 3 = vol above sigma_max = 10 (b(sigma_max sqrt T) < beta,
eq. 14); 4 = non-finite price; 5 = OTM tail not resolved in float64 (added in fix round 1,
validator finding F1, below). If the start s_c = sqrt(2|x|) lies above the cap, Newton starts from
the bracket midpoint and the safeguard handles convergence.
**Update (fix round 1, validator F1):** `scipy.special.ndtr` returns subnormals below d ≈ −37.4 and
exactly 0 from d ≈ −38. Phi itself is representable down to d ≈ −38.5. When Phi(d2) is lost but
Phi(d1) is not (d2 < −37.5 < d1), b(s) = e^{−a/2} Phi(d1) − e^{a/2} Phi(d2) misses its second term
and comes out too large, by up to ~17x. Newton and Brent then converged on the wrong function, and
the solver returned a wrong sigma, or sigma = 0 when beta itself underflowed, with status 0 and no
warning. Example: V = 2.162e-305, F = 100, K = 100 e^{38.5 sqrt 5}, T = 5, call, true sigma = 1
gave sigma_hat = 0. The fix is a post-check at the root. A root s is accepted only if s > 0 and
ndtr(d2(s)) >= PHI_MIN = 1e3·`finfo.tiny` ≈ 2.2e-305 (d2 ≳ −37.3, i.e. Phi(d2) is still a normal
float). Otherwise the result is NaN with status 5 and the warning. The check is sufficient on the
convex branch, where every root of the wrong function lies at a smaller s, hence at a smaller d2,
so it is also flagged. A threshold on beta alone would not be sufficient: beta ≈ b' s/(|d1||d2|)
with b' = e^{a/2} phi(d2), so at large s and |x| beta can be ~1e-280 while Phi(d2) has already
underflowed. Rule "never sigma = 0" holds: s <= 0 always gives status 5. Prices exactly 0 (both
tails flushed) are status 1.
`bsm.price` is not changed. Its contract is absolute accuracy (~eps DF max(F, K)); relative
accuracy is lost once Phi(cp d2) underflows, i.e. for OTM prices of order 1e-280 and below
(absolute error ≲ 1e-279). Evaluating the wings in log space (`log_ndtr`) would restore relative
accuracy there, but it costs a second code path, no quoted price is ever in that range, and
exp(L) at |L| ~ 600 would not improve the gate-domain digits.
Measured: the validator's `test_iv_deep_otm_never_silent_wrong` (33 cases, |z| from 12 to 40, both
cp) passes, and so does `tests/test_iv.py::test_deep_otm_unresolved_tail_is_nan`. On the gate grid
no element is affected (status 0 everywhere, round trip unchanged).
**Alternatives:** raise on any bad price (then one stale quote fails a whole chain); clip to the
bounds (forbidden); log-space b and Newton on ln b (more code, and only for prices nobody quotes).

## D-014 — Gamma FD step 4.5e-4 F sigma sqrt T (not the 1e-4 of the §2.4 table); identity tolerance
**Context:** at the §2.4 table step 1e-4 F sigma sqrt T, OTM gamma by D_2 gives a worst relative error
of 1.773e-5 against the gate of 1e-5. 78 of 147,600 points exceed it, all at sigma sqrt T ≈ 0.007–0.008
(sigma = 0.05, T ≈ 0.02). Diagnosis:
(i) the step is not the problem: with h = k·ulp(F), exact on both sides, the error is still 1.773e-5;
(ii) analytic gamma is right: larger steps converge to it, and vega = sigma T F^2 Gamma (rel < 1e-12)
and ∫ F^2 Gamma / K^2 dK = DF (< 1e-10) both hold;
(iii) the price noise is larger than the table assumes. At the worst point (sigma = 0.05, T = 0.02,
K = 100.852, OTM call V = 0.0398), sweeping F over ±50 ulps shows absolute noise of 5.2e-15 in
eq. (1). That is 600 eps V, while the table assumes eps V. It matches eps F Phi(d1) = 2.6e-15: the
cancellation between F Phi(d1) and K Phi(d2), which pitfall 6 quantifies. The rounding term
4 eps |f| / h^2 of eq. (17) is therefore ~600x larger, and the optimal step grows by about
600^(1/4) ≈ 5.
**Decision:** take the gamma step from the minimax of eq. (17) over the grid. Model the worst-case
error as A c^2 + B / c^2. A = 4.47 comes from c = 2e-3 (truncation corner sigma = 1, T = 5, z = 3)
and B = 1.77e-13 from c = 1e-4 (rounding corner). Then c* = (B/A)^(1/4) = 4.46e-4, and the
predicted worst case is 2 sqrt(AB) = 1.8e-6. Test step c = 4.5e-4, measured worst case 9.96e-7.
Tolerance (1e-5), OTM evaluation and grid are unchanged. Delta, vega and theta keep the table steps
(measured 2.0e-8, 3.0e-7, 1.7e-8). §5 identities: homogeneity V(lam F, lam K) = lam V, the swap
C(F, K) = P(K, F) and put-call symmetry use the note's per-element tolerance (abs <= 1e-12 at
F = 100, or rel <= 1e-12). Under pure rel 1e-12 they fail at 1.1–1.3e-12 on small OTM prices at
s ≈ 0.007: scaling F and K moves x = ln(F/K) by one ulp, and x/s amplifies that. All other
identities pass at pure rel 1e-12 and are tested that way.
**Consequences:** the §2.4 table entry for gamma (step 1e-4, predicted error ~1e-8) needs a
correction by the note's author, because it ignores pitfall 6. `scripts/phase1_gates.py` prints the
1e-4 value as a diagnostic.

## D-015 — Heston ξ→0 gate: exact at ρ = 0; at ρ ≠ 0 compare against Black + first-order ρξ term
**Context:** ROADMAP Phase 2 gate "ξ→1e-4 limit equals BSM with σ² = (1/T)∫v(t)dt, < 1e-5" does not state ρ.
For ρ ≠ 0 the Heston–Black gap is first order in ξ (correlation term ρξ·J·∂²V_B/∂x∂Σ, eq. 16 in
`derivations/heston.md`; Benhamou–Gobet–Miri 2010 a₁). Independent orchestrator check (Lewis integral vs
Black, κ = 1.5, θ = 0.04, T = 1, K ∈ {60,…,150}): max gap 3.27e-4 at ρ = −0.7, ξ = 1e-4 (linear in ξ:
3.27e-3 / 3.27e-4 / 3.25e-5 for ξ = 1e-3 / 1e-4 / 1e-5), but 2.6e-8 at ρ = 0, ξ = 1e-4. So the gate as
literally written is unattainable for ρ ≠ 0 — a specification error, not an implementation tolerance.
**Decision:** the gate is tested (i) at ρ = 0 against plain Black with σ_eff, tolerance 1e-5 (unchanged), and
(ii) additionally at ρ = −0.7 against Black **plus** the eq. 16 correction, tolerance 1e-5 (stricter than
skipping the case). No tolerance is loosened.
**Alternatives:** drive ξ to 1e-7 so the linear term drops below 1e-5 — rejected: the CF's κθ/ξ² terms then
cancel catastrophically (pitfall in heston.md), so it would test round-off, not the model.
**Consequences:** the owner should be aware the ROADMAP wording is ambiguous; the interview answer is
"the ξ→0 limit is Black with integrated variance only for ρ = 0; with correlation the leading correction is
the ρξ skew term, linear in ξ".

## D-016 — Strike domain of the put-call parity gate
**Context:** validator finding F2. ROADMAP Phase 1 states "put-call parity < 1e-12" without a strike
range. On the full IV-gate domain |ln(K/F)| <= 3 sigma sqrt T, K reaches 8.2e4 at F = 100, where
ulp(8.2e4) = 1.5e-11, so an absolute 1e-12 cannot be represented in float64 (note pitfall 4: the
parity floor is ~c eps DF (F + K)).
**Decision:** the gate is tested at < 1e-12 absolute for K/F in [0.25, 4] (F = 100), the domain the
note recommends (§2.5, pitfall 4). Elsewhere the requirement is the float64 floor, <= 4 ulp of
DF max(F, K). Calls and puts are always evaluated independently, never P = C − DF(F − K).
**Consequences:** measured (`scripts/phase1_gates.py`; validator `docs/validation/phase1.md`):
1.137e-13 on K/F in [0.25, 4] (40 sigma × 30 T × 61 K × 3 DF, and 200k random points). On the
full IV-gate domain the maximum is 1.455e-11 = 2.25 ulp of DF max(F, K). 124 of 147,600 points are
above 1e-12, all at K >= 8.3e3. Amending the ROADMAP wording is left to the owner.
