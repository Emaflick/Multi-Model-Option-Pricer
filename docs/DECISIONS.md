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

## D-017 — COS range and size: self-checking (N, L) by default (updated in fix round 1, validator F1)
**Context:** with the eq. (8) range at L = 12 the COS–Lewis gap does not fall with N (256 to 4096
give the same error) but falls by about 8x for every 2 added to L. That is pitfall 11 (range too
narrow), not a coding error. At L = 12, N = 256 the gate (< 1e-6) fails: max |COS − Lewis| = 3.95e-5
(FO) and 2.58e-6 (GS, the SPX-like set). It passes on LW (8.4e-9) and AT2 (1.4e-7)
(`scripts/phase2_gates.py`). At FO, T = 1, ATM put the error is exactly the even-reflection tail term of
note §3.3, ∫_{X<a} F(e^{2a−y} − e^y) f(y) dy. It was computed numerically from a wide-range COS density
as 3.948e-5, against a measured −3.947e-5. Q(X < a) is only 7.7e-6 at "12 sd", but the reflection weight
grows like e^{−y} into the heavy left tail (ρ < 0). Lewis is independent of [a, b] and reproduces FO1
to 1.6e-8 and the QuantLib/Lewis values to 3e-14.
The first decision was a fixed L = 18, N = 256. It passes the four gate sets (max 1.01e-7 FO,
2.5e-9 GS, 9.8e-9 AT2, 1.1e-13 LW). The validator (F1, `docs/validation/phase2.md`) then showed it is
not robust: on 300 random SPX-plausible sets × 6 maturities (seed 5; κ ∈ [0.3, 8] log, θ ∈ [0.01, 0.12],
ξ ∈ [0.1, 1.2], ρ ∈ [−0.95, −0.2], v0 ∈ [0.005, 0.12]; K 60..150; puts):
- 413/1800 cells (22.9%) exceed 1e-6, up to 9.95e-2;
- 480 cells are warned;
- 5 cells fail silently, up to 2.61e-6. The eq. (13) warning models series error only, never range error.
Phase 5 calibrates with COS inside least_squares, so a fixed (N, L) is unacceptable there.
**Decision:** `cos_price` with N = L = None (the default) is self-checking:
- Start at (N, L) = (256, 18).
- While max_K |P(N, L) − P(2N, L + 6)| > tol·F (tol = 1e-9, i.e. 1e-7 at F = 100, 10x below the gate),
  set (N, L) ← (2N, L + 6). Return the finer price.
- Adding 6 to L shrinks the range error by about 20–500x on the sets measured. Doubling N squares the
  series term. So the difference is a reliable estimate of the coarser error, and the returned finer
  price is better than tol·F.
- If N reaches N_MAX = 2^14 without agreement it warns ("not converged"). It never returns a silent miss.
- An explicit N and/or L gives a single evaluation with only the eq. (13) series warning. Use it in tests
  and studies only.
Measured (`scripts/phase2_gates.py`):
- Random scan: 0/1800 cells > 1e-6, 0 warned, 0 silent, max error 3.05e-9. Final N: 512 in 1248 cells,
  1024 in 274, 2048 in 189, 4096 in 69, 8192 in 17, 16384 in 3. Median cost 768 CF evaluations, p90 3840.
- Gate grid: max |COS − Lewis| 1.8e-10 (FO), 1.4e-12 (GS), 5.1e-12 (AT2), 1.1e-13 (LW).
- FO1: −1.6e-8, now tested with the default.
- Time per call, 25 strikes:

  | Set | Fixed (256, 18) | Default | Lewis |
  |---|---|---|---|
  | GS, T = 1 | 0.12 ms | 0.30 ms | 4.9 ms |
  | FO, T = 1 | 0.12 ms | 0.62 ms | 4.6 ms |
  | Worst scan cell, T = 3 | 0.12 ms | 8.0 ms | 9.6 ms |

  The calibration note's fixed N = 1024 would cost about 0.45 ms per call and would still be wrong
  in range-limited cells.
**Alternatives:**
- L = 12 plus √c4 (F&O eq. 49): the c4 closed form is not cross-checked, and it gives no error estimate,
  so it cannot guarantee "accurate or warned".
- A Chernoff-type bound on the left-tail reflection term from E[F_T^{−ω}]: rigorous, but it needs the
  moment-explosion map (eq. 17) per call.
- Lewis inside the optimiser: 15–40x slower.
- Fixed larger (N, L): still wrong somewhere, and slower on easy sets.
**Consequences:** COS cost now depends on the parameters, at 2.5–5x the fixed cost on typical sets.
`cos_series_error` remains as the eq. (13) diagnostic. MC gate references (`tests/test_mc.py`) still use
explicit (N, L), checked against Lewis inside each test. The note's L = 12 claim (eq. 8 and §5
diagnostic "L = 16/N = 512 vs L = 12/N = 256 < 1e-8") needs correcting by the note's author (OPEN_ISSUES).

## D-018 — Heston MC path API (Phase 4 reuse) and the 3SE gate design
**Context:** Phase 4 needs per-step (x, v⁺) for the Brownian-bridge barrier, fixings for Asian and
autocallable, and the raw draws for common random numbers (autocallable eq. 15). CLAUDE.md requires
(price, stderr, n_paths, seed).
**Decision:** `mc.heston_steps(p, times, n_paths, rng, scheme)` is a generator over any strictly
increasing grid (non-uniform allowed; QE and Euler coefficients are computed per step from Δ_n).
It yields (t_n, x_n, v_n, z_n) with x = ln(F_t/F), fresh arrays every step (never mutated, so
consumers can keep references), and z_n = [Z_v; Z_⊥] of shape (2, N). Memory is O(N) per step
(pitfall 12). `mc.simulate(..., observe=dates)` stacks the snapshots and raises if a date is off the
grid. `mc.european_price(..., seed=int)` builds `default_rng(seed)` itself, so it can return the
seed. All strikes share one path set. The state x is independent of F, and S_t = F_0(t) e^{x_t} (eq. 17).
Euler with κΔ ≥ 1 warns rather than raises (the scheme is defined, pitfall 2). The QE-M domain
(eq. 13) raises. `np.where` evaluates both QE branches; `errstate` silences only the unselected one,
and a per-step finiteness check raises otherwise.
**Gate settings:** Δ = 1/32 (n_steps = 32T), 2e5 paths, puts at K ∈ {70, 100, 140} (bounded payoff,
finite variance, heston-mc §3.1). Sets FO (T = 1), GS (T = 1), BK-2 (T = 5) and A-I (T = 10) are all
Feller-violating. Both "qe" and "qe-m" are gated: 24 comparisons, so P(at least one false fail)
≈ 6% for correct code (eq. 2). Seeds are fixed a priori (20261107 + set index) and never re-seeded.
COS references are checked against Lewis (< 1e-6) inside the test. Euler-FT is not 3SE-gated (it
measures +4.4 and +5.1 SE on A-I, as the note predicts); it enters through the bias figure and
`test_gate_euler_bias_shrinks_with_dt`.
**Alternatives:** returning full (M, N) path arrays (0.5 GB at 2e5 × 320); a class-based path object.

## D-019 — CF evaluated as Log(1 − q) with log1p (equivalent to eq. 3, no ξ⁻² amplification)
**Context:** eq. (3) avoids β − d but still takes Log(Υ/2d) = Log(1 + O(ξ²)), amplified by 2κθ/ξ²
(pitfall 3).
**Decision:** use d − β = ξ²A/(β + d), Υ/(2d) = 1 − q with q = ξ²A(1 − e^{−dT})/(2d(β + d)),
D = −A(1 − e^{−dT})/(2d(1 − q)), and Log(1 − q) = ½ log1p(2p + p² + s²) + i atan2(s, 1 + p) for
−q = p + is. This is algebraically identical to (3), with the same principal branch, and has no subtraction.
**Consequences:** at ξ = 1e-6 the CF equals Black's to < 1e-10. The ξ = 1e-4 gate residual is 4.4e-8 (ρ = 0).
Against the DOP853 Riccati ODE the relative error is < 1e-9 on FO and KJ at T up to 10.

## D-020 — Phase 2 test oracles: what is and is not hard-tested
**Decision:** (i) Published values hard-tested: FO1 and FO2 (5e-8, Lewis and default COS), QuantLib/Lewis L1–5
(1e-9, both engines), and Broadie–Kaya BK-1 and BK-2 (two sources each, 4 decimals). (ii) Not hard-tested: AMST Table 1/2
(A1/A2) miss their stated precision on 3 of 8 values. The parameter-rounding diagnosis is in OPEN_ISSUES.
A-I = 13.08467014 (Choi–Kwok only, ⚑) is not tested, although Lewis matches it to 3e-9. KJ = 4.95212
(QuantLib only) is not tested. KJ is still used in the CF-vs-ODE and cumulant checks. (iii) The cumulant
check uses one Richardson step on the h = 1e-3 central difference. The plain stencil has an O(h² c4/c2)
truncation error of 1.27e-5 on KJ at T = 10 (it scales exactly with h²), which tests the stencil, not eq. (7).
(iv) Shape checks (monotone and convex in K, 0 ≤ ∂_K P ≤ DF, calendar) run on the ROADMAP domain
K ∈ [60, 150] with a 1e-9 numerical allowance, 1000x below the gate. The measured worst violation is 2.1e-10.
On [40, 200] the COS tail noise reaches 3.5e-9 deep ITM.

## D-021 — Fix round 1: required option type (F4) and the small-κ c2 branch (F3)
**F4:** `cos_price`, `lewis_price` and `mc.european_price` take `cp` as a required argument with no
default. They previously defaulted to a put, while `bsm.price` defaults to a call. It is required but
positional-or-keyword, not keyword-only, because the frozen validator tests pass it positionally
(`cos_price(F, K, T, DF, p, -1)`). Omitting it raises TypeError (`test_cp_is_required`, and
`tests/test_mc.py::test_input_validation`). `bsm.price` is unchanged: it is a validated module, and
making its cp required would break backward compatibility.
**F3:** the κT < 1e-3 branch of `cumulants` is the first-order expansion in κ of the §3.2 moment ODEs:
c2 = v0(T − ρξT²/2 + ξ²T³/12) + κ[v0(−T²/2 + ρξT³/3 − ξ²T⁴/12) + θ(T²/2 − ρξT³/6 + ξ²T⁴/48)].
At ξ = 0 it reduces to the expansion of Σ̄, v0T + κ(θ − v0)T²/2. Measured against Richardson CGF
differences: rel. error ≤ 8.1e-6 at θ/v0 = 50, where the old branch was −2.25e-2. The 1e-3
threshold is kept, because at the switch the main formula is still accurate to 1.5e-7.

## D-022 — SABR: β fixed = 1 for SPX
**Context:** β cannot be identified from one smile. β and ρ both make the skew: [H02] (3.1a) gives the ATM slope as
−½(1−β)σ_ATM + ½ρν. [H02] §3.1, Fig. 3.3 says that "β cannot be determined by fitting a market smile since this would
clearly amount to 'fitting the noise'". The level σ_ATM ≈ αF^{β−1} makes α trade off against β, so a free β gives an
ill-conditioned (α, β, ρ) fit (sabr.md §1).
**Decision:** β = 1. (i) Equity convention: a positive index quoted in Black (lognormal) vols. With β = 1, σ_H depends on
K/F only, so the model is scale-free (no length scale in F, [H02] p. 7). α is then directly a vol, the backbone is flat
([H02] p. 10), and all spot–vol co-movement sits in ρ. [H02] p. 12 calls β = 1 the usual a-priori choice of desks that
prefer lognormal models or a horizontal backbone. (ii) Numerics: no boundary at F = 0, a log-Euler MC that is positive
and an exact martingale, a quadratic ATM α-inversion (sabr.md eq. 9), and Hagan = Obłój at β = 1 (eq. 6), so there is no
formula choice to defend. `models/sabr.py` implements eq. (2) only; eq. (1) (general β) is written out independently
in `tests/test_sabr.py` as a cross-check (equal at β = 1 to 1e-12; differs by O(1e-6) at β = 1 − 1e-6).
**Alternatives:** β = 0.5; β from a log-log regression of historical (F, σ_ATM) ([H02] (3.3); noisy).
**Consequences:** the model delta is sticky-moneyness (eq. 12). Heavy right tail: E F_T² < ∞ only for ρ ≤ −1/√2
([LM07]), which drives the put-based MC design (D-023).

## D-023 — SABR Monte Carlo: scheme, step, comparison space, conditional estimator
**Context:** Phase 3 gate "Hagan vs MC within 3SE for T ≤ 1, |ln K/F| ≤ 0.2" and the degradation figure (sabr.md §2.3, §5).
**Decision:** (i) `sabr.simulate(alpha, rho, nu, dt, draws)` consumes one (2, N) block [Z_α; Z_⊥] per step (as
heston-mc.md eq. 4). The vol is exact (eq. 13), evaluated in closed form α_n = α exp(νW_n − ν²t_n/2) from the running
Brownian W_n. ln F uses log-Euler (14), which is positive and an exact martingale. Taking the draws as an iterable lets
the halving test feed common random numbers ((Z₁+Z₂)/√2 on the coarse grid). `sabr.mc_price(..., seed=int)` builds
`default_rng(seed)` and returns (price, stderr, n_paths, seed), as `mc.european_price` does (D-018); cp is required
(D-021). The code stays in `models/sabr.py`, so validated `engines/mc.py` is untouched; it reuses `mc.mc_estimate`.
(ii) Gate (fix round 1): set G (α = 0.2, ρ = −0.75, ν = 0.4), **domain ν²T ≤ 0.16**, T ∈ {0.25, 0.5, 1},
k ∈ {0, ±0.05, ±0.1, ±0.2}, **puts** in **price space** (eq. 19, pitfalls 8 and 11), N = 2e5. The step is **M = 250 steps
at every maturity (Δ = T/250)**, which satisfies Δ ≤ min(1/250, 0.5/(ν²√N)) (eq. 18). Seeds 20261107 + i are fixed a
priori. *History:* the first version used Δ = 1/250 (M = ⌈250T⌉). The validator (F1) showed that the (14) wing bias was
then 0.64 SE_gate at T = 0.25, k = −0.2 (0.0415 vol pt), because the halving check had been run at T = 1 only.
*Measured now* (`scripts/phase3_gates.py`, CRN halving Δ vs Δ/2, seeds 20261127/28/29):
max |V_Δ − V_{Δ/2}|/SE_gate = 0.080 (T = 0.25), 0.063 (T = 0.5) and 0.073 (T = 1), all ≤ 0.1. The Richardson bias
2(V_Δ − V_{Δ/2}) is ≤ 0.17 SE_gate at every gate point: −0.0104, −0.0073 and −0.0084 vol pt at k = −0.2 for T = 0.25, 0.5
and 1 (one-seed CRN estimates; the validator's exact RQMC values are −0.0107 / −0.0098 / −0.0081 vol pt, i.e.
−0.164 / −0.169 / −0.139 SE_gate, and halving ratios 0.081 / 0.083 / 0.068, `docs/validation/phase3.md`). The largest in vol pt is −0.021 at T = 0.25, k = +0.2, where it is −0.01 SE_gate because the ITM-put SE is large.
*False-fail rate:* the 21 comparisons share paths, and they carry small deterministic offsets. These are mainly Hagan's
own wing error, −1.05 mean z at T = 1, k = −0.2, which is +0.058 vol pt per the validator. So the rate is not the
iid 1 − 0.9973²¹ ≈ 5.5%. Over 100 fresh seeds (`phase3_gates.py --seeds 100`): 4.0% of seeds fail, the per-comparison
rate is 0.29%, mean z −0.074, sd 1.04 (validator, its own 100 seeds: 5.0% and 0.67%; the exact offsets predict 0.52%
per comparison). At the old Δ the validator measured 9.0% of seeds and 0.86% per comparison (F2).
*Domain (validator F3):* at larger ν²T the gate fails inside the same box because of Hagan's expansion error, not the
code. Example: ν = 1, T = 0.5 (ν²T = 0.5) gives +0.40 vol pt at k = −0.2, i.e. 5.4 SE. So "Hagan reproduces SABR" holds
only for set G with ν²T ≤ 0.16. For SPX-calibrated smiles (ν²T ≈ 0.2–0.7), Hagan is a smile parametrisation, not the
SABR dynamics (see the figure).
(iii) Conditional estimator (17) (`method="mixing"`): the second, independent MC, with O(Δ²) bias. (α_T − α)/ν is
computed as α·expm1(νW_T − ν²T/2)/ν, which is exact for every ν > 0 and → αW_T at ν = 0, so pitfall 10 needs no
threshold. Black is evaluated in ln(Fξ), because Fξ underflows to 0 at large ν²T (ν = 1, T = 10), and `bsm.price` rightly
rejects F = 0. (iv) For ρ ≤ 0, ln ξ = ρ(α_T − α)/ν − ½ρ²Î ≤ |ρ|α/ν, so Fξ ≤ F e^{|ρ|α/ν}, and the *conditional*
call is a bounded payoff (der.). The figure therefore prices OTM options with (17) (puts for k ≤ 0, calls for k > 0).
That cuts the right-wing SE 10–100x relative to deep-ITM puts, whose SE carries F·sd(ξ). The raw log-Euler call stays
excluded (pitfall 8); the (14) overlay uses puts.
**Alternatives:** Euler on F or α (not positive, pitfall 9); a trapezoid drift in (14) (breaks the martingale,
pitfall 7); a vol-space gate (pitfall 11); a ξ control variate (deferred to Phase 6).
(v) ρ > 0 (validator F4): β = 1 SABR is a strict local martingale, E F_T < F ([LM07]). Scheme (14) is an exact
martingale, so for ρ > 0 its calls and parity with F target P + DF(F − K), while "mixing" prices the model call. Puts
agree. Calls and parity with F are the model's only for ρ ≤ 0 (`mc_price` docstring, sabr.md §3.4).
**Consequences:** the gate measures the total (bias + noise) error at ~0.07–0.1 vol-pt resolution (SE_σ). Hagan's
smaller in-box errors are resolved by the figure's (17) OTM estimator instead: 0.065 vol pt at k = +0.4 for set G at T = 1.
The figure's log-Euler overlay also uses M = 250 steps. In the figure summary CSV, the `*_diff_*_incl_mc_noise`
columns are σ_H − σ_MC from a single seed, noise included; `max_abs_err_volpts_significant` (|z| > 3) is the Hagan
error that the MC resolves (validator F7).

## D-024 — Hagan β = 1 numerics: log1p form of x(z), NaN policy, oracle bounds
**Decision:** (i) x(z) keeps the sign split of sabr.md eq. (5) (z ≥ ρ / z < ρ). Each log is evaluated as log1p of a
sum of non-negative terms, using D − 1 = z(z − 2ρ)/(D + 1):
x = log1p(z(D+1+z−2ρ)/((D+1)(1−ρ))) or −log1p(−z(D+1+2ρ−z)/((D+1)(1+ρ))). This is algebraically identical to (5), with
relative error O(ε) for every z instead of ~4ε/|z|. The z_c = 1e-3 cubic series is kept for z → 0 (0/0). The two
branches mirror each other bitwise, so z_over_x(−z, −ρ) == z_over_x(z, ρ), and the §3.3 identities hold to ≤ 1.3e-15
(measured; tol 1e-14). The jump across |z| = z_c, net of the true slope, is 4.7e-14 relative (tol 1e-12). x(z) agrees
with quadrature of ∫du/D(u) to 2.9e-15, including ρ = ±0.999. (ii) |ρ| > 0.999, α ≤ 0, ν < 0 or non-finite inputs raise
ValueError. 1 + c₁T ≤ 0 (eq. 22) and "no positive α" in eq. (9) return NaN with a RuntimeWarning; nothing is clipped.
(iii) Test oracle (ii) uses the second-order Taylor bound |σ(±h) − σ_ATM| ≤ |∂_kσ|h + ½|∂²_kσ|h² + 1e-12σ. The note's
first-order bound misses the curvature term (2.06e-10 at h = 1e-4, OPEN_ISSUES). Oracle (e) (halving) is applied as
written to (17); for (14), see OPEN_ISSUES.
**Consequences:** eq. (9) `alpha_from_atm` is ready for the Phase 5 σ_ATM parameterisation (round-trip 1e-14).

## D-025 — Exotics market inputs: fixing forwards, discount curve and S_ref from the parity table
**Context:** Asians and autocallables need F_0(t) = F(0, t) and DF(t) at arbitrary dates (asian.md §1 (a),
autocallable.md §1), and the autocallable needs a reference level S_ref. Spot is informational only (D-008).
**Decision:** `market.forward_curve(table, t)` (an added function; the validated Phase 0 code is unchanged):
- between kept expiries, linear interpolation of ln F and ln DF in T (piecewise-flat forward rate and carry);
- beyond T_max = 1.61 y, flat extrapolation from the last expiry with the constant zero rate r (OLS of −ln DF on
  T through the origin) and carry b (OLS slope of ln F on T), both fitted on T > 0.3 (the identified part, D-007);
  the curve is continuous at T_max;
- dates before the first kept expiry (T = 0.019) raise, because they would need S_0 (D-008).
S_ref := F_0(T_(1)) · DF(T_(1)) = 7374.4255 (first kept expiry, spot minus the PV of dividends to it). Deterministic
rates and carry (all forward measures coincide); every cash flow is discounted with the parity DF at its own date,
with no issuer credit or funding spread (autocallable.md §1).
**Alternatives:** Yahoo spot as S_ref (non-synchronous, D-008); linear interpolation of F (pitfall 3 of asian.md);
discounting with a CDS spread as in Deng et al. (no issuer is modelled).
**Consequences:** the AC-Q dates (0.25–1.5 y) are interpolated; their DF_i are non-increasing (checked, raises
otherwise; short-end DF noise of D-007 lies below 0.25 y). AC-A (3 y) is extrapolated and labelled as such. Tests:
exact at the expiries, log-linear midpoints, continuity at T_max (`tests/test_asian.py::test_forward_curve_builder`).

## D-026 — Autocallable term sheets AC-Q / AC-A and the sheet validation
**Context:** ROADMAP Phase 4: annual/quarterly observations, autocall 100%, coupon barrier, 60% European
protection, optional memory. autocallable.md §1 leaves the coupon and B_c open.
**Decision:** `products.Autocall`. Study sheets (`scripts/autocall_compare.py`): **AC-Q** t_i = i/4, n = 6,
T = 1.5 y (inside the last kept expiry); **AC-A** t_i = 1, 2, 3 (carry and Heston extrapolated). Both B_ac = 1,
B_c = 0.7, B_p = 0.6, memory on, c = 8% p.a. simple (c_i = c τ_i), notional 1, flows paid at t_i without lag.
Par coupons are reported by the ratio estimator (eq. 17), so c = 8% is only the pricing input. Validation is
0 ≤ B_c ≤ B_ac (a call always pays its coupon) and B_p ≥ 0. The note's §1 ordering B_p ≤ B_c is a term-sheet
convention that eqs. (1)–(11) do not need, and its own §5 sub-case "B_c = 0" (with B_p = 0.6) violates it, so the
code does not enforce it (erratum added to autocallable.md).
`c` may also be a tuple of n per-period coupons (fix round 1, validator F7b). That makes accrual schedules such as
Deng et al. (2011) Case 2 (c_j = e^{0.092 t_j} − e^{0.092 t_{j−1}}) representable. `replace(sheet, c=1.0)` for the
par-coupon ratio (17) applies to scalar-c sheets only.
**Alternatives:** B_c = 0.8; no memory; a 6-month sheet.
**Consequences:** with memory and c = 8%, V_BS > 1 for both sheets; par coupons are 5.2–5.8% (BS) and 7.3–7.5%
(illustrative Heston). Deng Case 2 by qpricer MC: 97.508 (SE 0.009, 1e6 paths). That equals the validator's exact
97.507, not the published 98.39 (OPEN_ISSUES). The first-call p_i match Deng Table 1 (max |z| 2.11).

## D-027 — Erratum E1: six Haug σ = 0.30 barrier values are not correctly rounded (per-cell treatment)
**Context:** barrier.md §5 tests the Haug (1998) table at 5e-5. Six σ = 0.30 cells (QuantLib only, ⚑) miss by
5.2e-5 to 6.25e-5: cuo(105, K = 90), puo(105, K = 90/100/110), pdi(95, K = 100/110). The other 66 cells pass at 5e-5
(max 4.96e-5 at σ = 0.25, 4.59e-5 at σ = 0.30).
**Diagnosis:** three independent methods agree on the six cells.
- Direct quadrature of the killed and first-passage densities, in `tests/test_barrier.py` (no A..F algebra): within
  ≤ 3.2e-14 of RR (printed by `scripts/phase4_gates.py`).
- The validator's Haug spot-form RR (RRS) and a Crank–Nicolson PDE: they agree to ≤ 5.5e-10
  (`docs/validation/phase4.md`).
The printed entries differ from the correctly rounded value by one unit in the 4th decimal; e.g. puo K = 90 is
4.22923747, printed 4.2293. The validator found no small perturbation of (σ, T, r, b) that fits the column better.
**Decision (fix round 1, validator F1):** no tolerance is loosened.
- The 66 correctly rounded cells are tested against the published values at the note's 5e-5.
- The six E1 cells are flagged `published_correctly_rounded = false` in the validator's golden file
  `tests/golden/barrier_haug.json`. That file (RRS = PDE) was created independently and is read-only here. These
  cells are tested against the golden value at 1e-10.
- All 72 cells are also tested against the golden value at 1e-10 (measured max 4.8e-14), and RR vs quadrature at
  1e-12.
- `scripts/phase4_gates.py` prints the two groups separately.
The first version moved the whole σ = 0.30 column to 1e-4, which loosened 30 cells that pass at 5e-5; that version is
withdrawn. Erratum E1 is in barrier.md §8.
**Consequences:** no formula change. As oracles, the six published entries are replaced by a two-method golden value.

## D-028 — Barrier engine: spot barrier with flat carry, bridge weight, Heston v⁺ bridge, gate design
**Context:** barrier.md §1 asks for D-entries on the barrier convention and the Heston bridge (pitfall 6).
**Decision:** (i) The barrier is on spot, S_t = F_t e^{−b(T−t)}, with a flat-carry kernel input b (default 0, i.e. a
barrier on the T-forward). The closed form and the MC share the log-distance of eq. (8). r = −ln DF / T enters only the
rebate term. (ii) MC: conditional bridge weight w = Π(1 − p_i) (eq. 10; Rao–Blackwell, smooth), not uniform
sampling; in := g(1 − w) and out := g·w come from **separate calls** on one seed, so in + out = vanilla holds to
round-off (3.6e-15 measured). (iii) Discrete monitoring: indicator at t_j = jT/m, no bridge, t = 0 not monitored.
(iv) Under Heston the bridge uses the left-point variance v_i⁺ (approximate; there is no BGK under Heston).
Its bias is O(Δ): on the model-free cdo(K = H = 90) = 10·DF on FO, the 25-seed pooled bias is +0.095, +0.055,
+0.020 and +0.012 at Δ = 1/32, 1/64, 1/128, 1/256 (validator; see Consequences). The gate uses Δ = 1/128, where
the bias is below the gate SE. (v) MC rebates are not implemented (R = 0 in the gates; the
closed form has them). (vi) Gate design: Haug market, all 8 types, K ∈ {90, 110} (both branches of table 3 for
H = 95 down and 105 up), b ∈ {0, 0.04}, M ∈ {1, 50} (the BSM bridge is exact for every M), 2e5 paths, one seed
per configuration (SEED + 10 + j). That is 56 non-zero 3SE comparisons, plus 8 cells that must be exactly 0
(cuo with K ≥ H, pdo with K ≤ H). (vii) Code lives in the new `engines/exotic_mc.py`, so the validated `mc.py` is
untouched; it reuses `mc.heston_steps`, `mc.uniform_grid`, `mc.mc_estimate`.
**Consequences:** gate max |z| 1.99 (BSM) and 2.97 (Heston ξ = 1e-4).
- *G2h seed (validator F4).* The declared seed is about a 1-in-75 draw: 1.3% of seeds give max |z| ≥ 2.97. Its path
  set has a vanilla error of −3.15 SE (call) and +3.13 SE (put), and the barrier z tracks the vanilla error with
  correlation 0.905. Over 300 validator seeds the per-cell mean z is −0.031 to +0.011 (sd 0.96–1.07, 0.25% beyond 3 per
  comparison): no bias. An earlier 20-seed scan here (`phase4_gates.py --seeds 20`) gave a pui mean of 0.36; at
  20 seeds that is noise (sd/√20 ≈ 0.2).
- *v⁺-bridge bias under Heston (validator F3).* The per-Δ numbers printed by the gate script use one seed per Δ
  and understate the bias. Pooled over 25 seeds (validator), the model-free cdo(K = H = 90) − 10·DF on FO is
  +0.095, +0.055, +0.020 ± 0.005 and +0.012 ± 0.005 at Δ = 1/32, 1/64, 1/128, 1/256. That is an O(Δ) bias.
  G2m passes at Δ = 1/128 because 0.020 is below the gate SE (0.024), not because the bias is zero. At Δ = 1/64,
  about 16% of seeds would fail the 3SE gate. Heston barrier prices from this engine therefore carry a bias of
  about 0.8 SE at 2e5 paths and Δ = 1/128; use Δ ≤ 1/256 when tighter accuracy is needed.
- *Figure.* The uncorrected grid-monitoring bias has fitted log-log slope 0.422 over m = 4…512. That equals the
  noise-free BGK prediction on the same m (0.426), whose local slope rises from 0.32 (m = 4 → 8) to 0.50 (m ≈ 10⁶)
  (`scripts/exotics_figures.py`; erratum E2, barrier.md §8). The validator's exact discrete prices give a fitted
  slope of 0.4295, with local slopes rising from 0.364 to 0.478.

## D-029 — Autocallable model gap uses an illustrative, pre-calibration Heston set until Phase 5
**Context:** the ROADMAP gate compares BSM (ATM vol) with *calibrated* Heston, but calibration is Phase 5.
**Decision:** `scripts/autocall_compare.py --heston-params <json>` (keys kappa, theta, xi, rho, v0, optional
label). Its default is **GS**: κ = 1.3253, θ = 0.0354, ξ = 0.3877, ρ = −0.7165 from Gatheral (2006) Table 3.2, a
published SPX fit (via Keller-Ressel 2011, as in heston.md §5), with v0 := θ (our choice). It is not fitted to the
2026-11-06 snapshot. Every output made with it carries the label "illustrative, pre-calibration" (stdout, CSV
`label` column, figure title). The files are named `report/figures/autocall_attribution_<tag>.{png,csv}`, with
tag = "illustrative" for the default and the JSON stem otherwise, so Phase 5 output cannot overwrite or be confused
with it. Phase 5 re-runs the script with the fitted parameters, and only those numbers go into the report.
**Alternatives:** the FO (Eurostoxx) set; waiting for Phase 5 (it would leave the gate untested).
**Consequences:** GS's ATM implied vol at T = 1.5 is 16.9%, against a market ATM vol of 18.3%. The illustrative
gap therefore shows the mechanism, not a market number.

## D-030 — Asian engine: running sums, BSM-only geometric control variate, OLS ĉ on the same paths
**Decision:** (i) `asian_averages` keeps Σ_A and Σ_ln (eq. 5), so memory is O(N) at n = 250 and there is no
product overflow. A t = 0 fixing is an ordinary entry with F(0, 0) supplied by the caller. Heston uses
`fixing_grid` (⌈τ_i/Δ_max⌉ equal sub-steps ending on each date; `heston_steps` already recomputes the QE constants
per step). **Fix round 1 (validator F2):** the Heston default is now `max_dt = MAX_DT = 1/32` in `asian_averages`,
`asian_mc`, `obs_steps` and `fixings`, as `autocall_mc` already had. It was `None`, i.e. one QE-M step per fixing
period, against eq. (4). The validator measured this with quarterly fixings, GS and 3e6 paths: the one-step default
minus Δ = 1/32 gave z = −2.6 / −3.6 / +5.9 at K = 90 / 100 / 110 (+1.3% at K = 110), and its own QE-M at Δ = 1/64
agrees with 1/32. Under BSM `max_dt` is ignored (exact steps). `None` remains available for grids that are already
fine (`barrier_mc` passes it explicitly). (ii) The CV (eq. 14) is **BSM only**. `asian_mc(kind="cv")` raises under Heston, because the Black V_G is
not the Heston mean (§3.5). The Heston default is plain MC (H1); the shadow-BSM control H2 is not implemented
(not needed by any gate). (iii) ĉ is the OLS fit on the same paths. Its O(1/N) bias is negligible (eq. 16), and
VR̂ = s_Y²/s_e² = 1/(1 − ρ̂²) is reported from the same paths (tested to 1e-10). (iv) Gate grid as asian.md §5:
σ ∈ {0.2, 0.5} × T ∈ {1, 3} × n ∈ {1, 12, 52} × K/F ∈ {0.8, 1, 1.2} plus the [FM08] curve, 39 comparisons.
**Consequences (measured, `scripts/phase4_gates.py`, `scripts/exotics_figures.py`):** geometric MC vs KV max |z| 1.90.
CV variance-reduction factor on [FM08] (σ = 0.178, n = 12): 1924 / 1608 / 612 at K = 90 / 100 / 110 (ρ̂ = 0.99974 /
0.99969 / 0.99918). At the money with n = 12, T = 1: VR = 5793, 1416, 632, 211 for σ = 0.1, 0.2, 0.3, 0.5.

## D-031 — Autocallable attribution (eq. 19): σ_ATM, term-structure model, exact no-call legs, CRN
**Decision:** (i) V_BS uses the flat σ_ATM = the Heston implied vol at (K = F_0(T), T), so both models price the
ATM-forward T-vanilla alike. Sensitivity rows: BS at σ_H(S_ref), at σ_H(B_p·S_ref), and at the market ATM vol (OTM
quote IVs at the two strikes adjacent to F interpolated to k = 0, then total variance linear in T; n/a beyond the
last expiry). (ii) V_TS is Black with Heston's expected variance w(t) (heston.md eq. 6). Term 1, V_TS − V_BS, is labelled
"flat σ_ATM vs Heston expected-variance curve (level + term structure)" (fix round 1, validator F6; the note calls
it "ATM term structure"). It has a level part, √(w(T)/T) vs σ_ATM (the vol-of-vol convexity of ATM implied vol), and
a shape part. With v0 = θ, as in the illustrative GS set, w(t) = θt is flat, so term 1 is a pure level gap:
√θ = 18.82% vs σ_ATM = 16.87%. (iii) V^nc_H and V^nc_TS are
exact (eq. 11): Black, or Heston COS for the put and the digital D⁻ = ∂_k p. The digital is a central difference
(h = 1e-4·k) on **one** self-checking COS call at [k − h, k, k + h], so the three strikes share (N, L) and the
difference is the derivative of a single series. At ξ = 1e-4 it matches Black's digital to < 1e-6. The no-call legs
are also re-priced on the MC paths as a 3SE check. (iv) CRN: one Heston QE-M path set (Δ ≤ 1/32); the BS and TS
paths are driven by the period-aggregated Heston spot normals (eq. 15). Gap and term SEs come from per-path
differences (eq. 16); par coupons and their gap use eq. (17).
**Consequences (illustrative GS set, AC-Q):** V_BS 1.016721, V_TS 1.014575, V_H 1.002359. Gap −0.014362 (CRN SE
0.000171) = flat σ_ATM vs expected-variance curve −0.002146 (0.000061; pure level here) + marginal smile −0.015971
(exact) + path +0.003755 (0.000153).
Par coupon 5.23% (BS) vs 7.55% (Heston), +2.32% (0.03%). The sign is the expected V_H < V_BS (§3.5).
- The validator's exact Black values confirm V_BS (1.016654) and term 1 (−0.002110) within MC error.
- The note's CRN σ-grid diagnostic for the full BS product (validator F7c) is printed by `autocall_compare.py` and by
  the gate script. On AC-Q, V rises from σ = 0.10 to 0.15 (+0.00043, dV SE 0.00004) and then falls, so the full
  product has no single vega sign (§3.4). The no-call sheet is strictly decreasing in σ (tested).

## D-032 — Calibration targets, thinning and the Heston maturity window (calibration.md §2.2, note text "D-016")
**Context:** 7625 OTM quotes; 18 of 39 expiries below 0.1 y, with 5-pt strikes. Heston's ATM skew is flat for
T ≪ 1/κ (eq. 14), while the SPX skew keeps steepening; weekly T counts weekends as variance time (D-003).
**Decision:** eqs. (5)–(7) and weights (8): in-band = IV status 0, mid ≥ 0.50, |d1(σ^m)| ≤ 1.6449; thinning to 25 d1
targets per expiry, taken in order, each the nearest unused in-band quote, **plus** the put and the call nearest k = 0,
so up to 27 quotes per expiry (the note's "≤ 525" counts only the 25 targets). Heston on T ≥ 0.1 (21 expiries), out of
sample below; SABR on every expiry passing (7). Targets come from `build_surface()["surface"]` re-cut with the smoothed
forwards (D-033), never from `otm`.
**Alternatives:** all 39 expiries; T ≥ 2 weeks (33 expiries, run as a sensitivity); a 10Δ band; no thinning.
**Consequences (measured, `scripts/run_calibration.py`):** 5806 in-band quotes, 1029 thinned (25–27 per expiry),
n_H = 565 Heston residuals. The h_min = 0.1 vol-pt weight floor binds on 60.9% of the thinned quotes (the median IV
half-width is 0.078 vol pt).

## D-033 — Short-end DF from a parity-fitted zero curve (closes OPEN_ISSUES F5; note text "D-017")
**Decision:** eqs. (2)–(3). WLS r̂(T) = a + bT on the 21 expiries with T ≥ 0.1, weights 1/(r_se² + (10 bp)²);
D̃F = exp(−r̂T) for T < 0.1, parity DF kept for T ≥ 0.1; F̃ = K̄_u + ȳ_u/D̃F on the Phase 0 used pairs, which
`calibration.parity_pairs` rebuilds and checks against the table (F and DF to 1e-10). The OTM cut and the static-arbitrage
filter are re-run with F̃ (`data.select_otm`, `data.remove_static_arbitrage`), because a strike between F and F̃ changes type.
**Alternatives:** external OIS curve (breaks D-001); curve DF on every expiry; flat r.
**Consequences (measured):** r̂(T) = 4.527% + 0.304%·T. Short-end r moves from 2.64%–16.89% to 4.533%–4.556%, and the
DF sequence becomes monotone below 0.1 y. |F̃ − F| ≤ 3.56 F_se (2026-12-03, +0.65 pt). The note estimated ≲ 1.5 F_se;
four expiries exceed 2 F_se (11-18, 11-19, 12-02, 12-03: 2.21, 2.37, 2.82, 3.56). Every used pair stays inside its own
band under (F̃, D̃F): max |res|/h is 0.834, and 0 pairs exceed 1. Eq. (3) at T ≥ 0.1 reproduces the parity F exactly.
The re-cut surface is identical to the original, 7625 quotes and no type change, because no listed strike lies between
F and F̃. At the 347 used strikes with T < 0.1, the median put/call IV split goes from 0.0275 to 0.0260 vol pt, and 0 → 0
pairs exceed h_C + h_P. DF rises between consecutive expiries 4 times instead of 10, and none of them lies below 0.1 y
(one is the step across T = 0.1).

## D-034 — Heston objective, restricted bounds box, pinned COS (N, L) = (16384, 64) and its a-priori IV budget (note text "D-018"; orchestrator carry-forward 2)
**Context:** the self-checking COS (D-017) is piecewise in p, which injects finite-difference noise into the Jacobian,
so (N, L) must be frozen for the whole fit. The note's choice (N = 1024, L = 12) and its bounds box (11) were checked
against Lewis first, on the real Heston grid (565 quotes, 21 expiries):
- L = 12 at the plausible point (v0, κ, θ, ξ, ρ) = (0.03, 2, 0.04, 1, −0.7) gives an IV error of 9.9e-5. L = 18 gives 9.8e-7,
  and (2048, 24) gives 9.9e-9.
- At the 32 corners of box (11), Lewis's `quad_vec` fails at 4 corners (v0 = θ = 1e-4, ξ = 5). No fixed (N, L) up to
  (4096, 30) is accurate. Examples: an IV error of 6e-2 at (1e-4, 1e-3, 1, 5, 0), and a price error of 3.8e-3 at
  (1, 1e-3, ·, 5, −0.99).
- The hard region is small v0 and θ, small κ (ν_F ≈ 0) and large ξ. X is then extremely leptokurtic: a spike plus heavy
  tails. At (0.005, 1e-3, 0.005, 3, 0), COS needs L ≈ 80 and N = 16384 to match Lewis (1e-14 at (65536, 120)). Lewis is
  the correct reference there, and the self-checking COS warns "not converged".
**Budget (stated before the production fit and the synthetic gates):** a **price error in market-vega units**,
e_vega = max_i |v_COS,i − v_Lewis,i| / 𝒱_i ≤ 1e-5. It is not a claim of IV accuracy over the box (fix round 1, M3).
with 𝒱_i the normalised vega at the target vol. This is the first-order IV error at the market vol, the scale of
fallback (9), and it is defined even where the model IV does not exist. 1e-5 = 0.001 vol pt is 37x below the smallest
quote IV half-width (0.0375 vol pt), 100x below h_min and 300x below the 0.3 vol-pt synthetic noise. It is checked on
I_H at the 32 corners of the bounds box and at 128 scrambled Sobol points (seed 20261109; log-uniform in v0, κ, θ, ξ,
uniform in ρ). The direct |σ_COS − σ_Lewis| is printed alongside where both inversions succeed. After each fit, the
note's §5 oracle |σ_COS − σ_Lewis| ≤ 1e-5 is re-checked at p̂.
**Decision:**
- **Bounds:** v0, θ ∈ [0.01, 0.25] (10%–50% vol); κ ∈ [0.5, 20]; ξ ∈ [0.01, 3]; ρ ∈ [−0.99, 0]. The note had
  v0, θ ∈ [1e-4, 1], κ ∈ [1e-3, 20], ξ ∈ [0.01, 5]. κ ≥ 0.5 is also the edge of the identifiable region of eq. (13):
  κ·T_max = 0.8 on T_max = 1.61.
- **Pinned COS:** global (N, L) = (16384, 64), the smallest rung measured that meets the budget at every corner.
  Corner maxima: (8192, 48) 3.7e-5 and (16384, 80) 2.0e-5 fail; (16384, 64) gives 9.2e-6, worst corner
  (0.01, 0.5, 0.01, 3, −0.99).
- **Why not per expiry:** pinning (N, L) per expiry was measured and rejected. Long expiries need the most terms:
  T = 1.61 gives 3.7e-5 at (8192, 48). Only T ≤ 0.19 passes at (8192, 56), so per-expiry pins would save under 10% of the
  cost.
- **Start box (11), adapted to the bounds:** √v0, √θ ∈ [0.10, 0.40] (note: [0.08, 0.40], whose lower edge lies below
  v0 = 0.01), ln κ ∈ [ln 0.5, ln 8], ln ξ ∈ [ln 0.2, ln 3], ρ ∈ [−0.95, −0.2]. 16 starts.
- **Objective and solver:** IV residuals (8) with fallback (9); TRF settings (12): `diff_step = 1e-6`, `x_scale = 'jac'`,
  ftol = xtol = gtol = 1e-10. Feller is reported, never imposed.
**Disclosure and audit:** before these bounds were fixed, one pilot fit was run on the note's box at (2048, 24) from
4 starts. It found an interior optimum near (0.052, 4.6, 0.053, 1.65, −0.75), far from every restricted edge. No gate
tolerance depends on the bounds except pre-check (18), which uses them.
- The pilot is now re-runnable (fix round 1, M2): `scripts/run_calibration.py --box note`, output saved in
  `report/calibration_notebox_20261106_2213.{txt,csv}`.
- That run refits on the note's original box (11) with the same pinned COS, from the note's 16 Sobol starts
  (√v ∈ [0.08, 0.40], κ ∈ [0.3, 8]) plus 4 adversarial corner starts. **20/20** starts reach the production p̂
  (max |Δp| 1.9e-5 SE, cost relative −4.7e-15). COS vs Lewis agrees to ≤ 4.9e-13 at every endpoint.
- The production p̂ is **≥ 38 SE inside the narrowed bounds**: 111, 38.1, 169, 55.7 and 136 SE for v0, κ, θ, ξ, ρ.
- The validator independently found the same: 20/20 starts, ≥ 38 SE (`docs/validation/phase5.md`).
- So the narrowing biases no reported fit.
**Measured on the box (`scripts/phase5_gates.py`):** price error in market-vega units e_vega = 9.21e-6 ≤ 1e-5, worst at
the corner (0.01, 0.5, 0.01, 3, −0.99). The direct model-IV error at that worst-budget corner is 2.5e-3 (validator
reproduced it). Over the whole box it reaches 6.4e-3, but only at ξ = 3, ρ = −0.99 corners and only on
OTM calls. There the model IV is 2–5% against a market 13–15%, so the model vega is 1e-4 to 1e-11 of the market vega:
an absolute price error of 1e-15–1e-7 becomes a large IV error, on a residual that is itself about 10 vol pts.
This is why the budget is stated with the market vega. The IV accuracy that matters is the one at the optimum.
After the fit, COS and Lewis agree to 4.9e-13 in IV at p̂ on I_H, to ≤ 4.9e-13 at every one of the 16 start endpoints,
and to 7.0e-13 on the 464 out-of-sample quotes (`scripts/run_calibration.py`).
**Cost:** 87 ms per residual evaluation (COS 99%). One 16-start fit takes about 30 s on 8 worker processes
(`fit_heston(..., workers=8)`; deterministic per start, so the result does not depend on the number of workers).
**Alternatives:** Lewis inside the optimiser (about 0.17 s per evaluation, 2x slower, and `quad_vec` fails at box
corners); the note's box with a looser budget (the IV error reaches 1e-2 there); reparametrisation (excluded by the note).

## D-035 — Synthetic round-trip gate: tolerances and seeds fixed before any noisy fit (eqs. 16–20)
**Decision:** grid = the real I_H (k_i, T_j, w_i; 565 quotes); data labelled `synthetic`, σ^obs = σ^H(p*) + ε with
σ_ε = 0.003, COS at the pinned (16384, 64) for both data and model. Seeds: S1 `default_rng(20261107)`,
S2 `default_rng(20261108)`, SABR expiry j (0..3, in the order of eq. 20) `default_rng(20261117 + j)`, and the slow 64-seed
linearisation test `default_rng(20261207 + s)`, s = 0..63. They are never changed after a result is seen.
**False-fail rate of the whole synthetic suite (fix round 1, M4).** The union bound is:
- 2 Heston G2: 2 × (1 − 0.9973⁵ + 0.001) ≈ 2.9%;
- 4 SABR G2: 4 × (1 − 0.9973³ + 0.001) ≈ 3.6%;
- 6 G4b at 0.27% each: 1.6%;
- total about **8%**; with the 64-seed linearisation test (5 ratios at 3σ, about 1.3%), about 9.5%.
The note's 2.9% covers only the two Heston G2 gates.
**Pre-check (18), computed from eq. (17) (central differences, step 1e-5|p_j|) before any noisy fit:**
- S1 (GS, v0 = θ = 0.0354, κ = 1.3253, ξ = 0.3877, ρ = −0.7165): SE (v0, κ, θ, ξ, ρ) = (1.269e-4, 4.959e-2, 2.947e-4,
  4.435e-3, 4.432e-3). 3SE/|p| = (0.011, 0.112, 0.025, 0.034, 0.019). Inside the bounds. PASS.
- S2 (0.015, 3, 0.04, 0.4, −0.7): SE = (1.632e-4, 9.416e-2, 3.283e-4, 6.555e-3, 6.989e-3).
  3SE/|p| = (0.033, 0.094, 0.025, 0.049, 0.030). PASS.
- SABR (α, ρ, ν) SEs on the real I_j:
  - 2026-11-13 (26 quotes): (9.9e-4, 6.22e-2, 1.577e-1), 3SE/|q| ≤ 0.266;
  - 2026-12-18 (26): (1.06e-3, 4.19e-2, 5.92e-2), ≤ 0.179;
  - 2027-05-21 (27): (1.09e-3, 3.11e-2, 2.66e-2), ≤ 0.133;
  - 2028-06-16 (27): (1.22e-3, 2.78e-2, 1.72e-2), ≤ 0.119.
  All PASS.
**Gates (19)–(20), as the note states them:** G1 noise-free, 16 starts (SABR: seed + Sobol starts), ftol = xtol = gtol =
1e-14, max|p̂/p* − 1| ≤ 1e-6 (SABR 1e-8). G2: |δp_j| ≤ 3 SE_j for every j, and D² ≤ χ²_5(0.999) = 20.515 (SABR χ²_3(0.999)
= 16.266). G3: cost(p̂) ≤ cost(p*). G4: RMSE(p*) = RMS(ε) to 1e-12, and |RMSE(p̂) − σ_ε√((n−m)/n)| ≤ 3σ_ε/√(2n).
**Linearisation test (slow; note §5):** on S1, 64 noise seeds, each fitted from a single start at p*. This tests the
sampling distribution of the estimator; G1 and G2 already test the multi-start. Pass: sd(p̂_j)/SE_j ∈ [0.73, 1.27].

## D-036 — SABR per expiry: native (α, ρ, ν) loss, F6 handled by restricting the start set, ν²T flag
**Context:** Phase 3 F6: `alpha_from_atm` (eq. 9) has no positive root for ρ < 0 when σ_ATM > b²/(4|a|). Also, Hagan is
NaN where 1 + c1T ≤ 0 (sabr.md 22).
**Decision:**
- The loss (15) is in native (α, ρ, ν) with bounds sabr.md (11), so eq. (9) never appears inside it. NaN from
  1 + c1T ≤ 0 at a trial point is rejected by TRF (scipy `trf.py` shrinks the radius on non-finite residuals).
  The optimum is asserted finite.
- Eq. (9) is used only for the starts: the seed (10), with α = s0 inside the ν0 formula and ν_min = 0.1 (the lower
  Sobol edge), and 4 Sobol points (ρ ∈ [−0.95, 0.3], ln ν ∈ [ln 0.1, ln 4], seed 20261109).
- A start whose (9) has no positive root, or whose smile is non-finite, is dropped and counted (restriction, not penalty).
- ν²T is reported per expiry and flagged above 0.16, the Hagan-vs-MC validated domain (Phase 3 F3, D-023).
**Consequences (measured):** 0 starts dropped on all 39 expiries. **Every** fitted expiry has ν²T > 0.16 (0.35–0.89),
so every SABR smile in this report is a Hagan parametrisation outside its validated domain. At SPX-like ν ≈ 1 the
expansion is known to be off by +0.4 to +0.8 vol pt in the left wing at k = −0.2 (Phase 3).

## D-037 — Start design, sensitivity runs and the objective-(10) cross-check
**Decision:**
- **Starts:** the 16 Heston starts are `qmc.Sobol(d=5, scramble=True, rng=default_rng(20261109)).random_base2(4)`,
  mapped to the D-034 start box. They are data-independent and shared by the real fit, S1, S2 and the sensitivity runs.
- **Sensitivity runs** (from the same 16 starts): n_t = 50, and T ≥ 2 weeks (33 expiries). Δp is reported in units of
  the main fit's SE.
- **Objective (10)** (vega-scaled price residuals) is run from the same starts on S1 (noisy) and on real data;
  §5 asks for p̂^(10) within 1 SE of p̂.
- **SE meaning:** on real data the SEs are curvature SEs scaled by s² = 2·cost/(n − 5). They are conditional on the
  model and window and say nothing about misspecification.
  - **They understate the window and model risk by about an order of magnitude** (fix round 1, M6): moving the window
    to T ≥ 2 weeks shifts κ by +13.0 SE and ξ by +12.9 SE.
  - Parameter uncertainty is therefore quoted as the sensitivity range. `scripts/run_calibration.py` prints it next to
    the SEs: κ ∈ [4.49, 5.96] against SE 0.107, and ξ ∈ [1.62, 1.96] against SE 0.024.
**Consequences (measured, `scripts/run_calibration.py`):**
- **Real fit:** p̂ = (v0, κ, θ, ξ, ρ) = (0.052372, 4.5751, 0.052925, 1.6477, −0.75180), SE (0.00038, 0.107, 0.00025,
  0.024, 0.0017). cost 2062.90, s² = 7.37, ν_F = 0.178 (Feller violated), cond(J·diag|p̂|) = 25.6.
- **Starts:** 16/16 converge to one minimum, basin spread 1.1e-5 SE.
- **Sensitivity runs:** n_t = 50 moves p by at most 1.06 SE. The T ≥ 2-week fit moves κ and ξ by +13.0 and +12.9 SE
  (κ = 5.96, ξ = 1.96): the short end pulls Heston towards faster mean reversion and higher vol-of-vol (eq. 14).
  This is misspecification, not noise, and the curvature SEs do not cover it.
- **Objective (10):** within 0.83 SE of p̂ on real data.

## D-038 — Phase 4 ROADMAP gate closed with the calibrated Heston set
**Decision:** `scripts/run_calibration.py` writes `data/processed/heston_fit_20261106_2213.json` (p̂, SE, cost,
provenance). `scripts/autocall_compare.py --heston-params <json>` writes
`report/figures/autocall_attribution_heston_fit_20261106_2213.{csv,png}`; the `_illustrative` files are untouched.
The DEFERRED line of `scripts/phase4_gates.py` becomes two lines per sheet (`calibrated()`):
- the gap and its eq. (19) terms, reported;
- (fix round 1, M1) the BS and TS legs on the attribution's CRN paths, restricted to the sheet's first two dates,
  against the exact eq. (12) within 3 SE, and the no-call MC legs against (11) within 3 SE.
Eq. (19) additivity holds by construction (term3 is the residual), so it is not gated.
The BSM leg is labelled everywhere as **"BSM at the calibrated-Heston ATM vol (18.85%)"** (AC-Q; 19.72% on AC-A):
in the CSV column `bs_leg`, the figure, stdout and the gate lines (M9). An extra row prices V_BS at the market ATM
vol. `autocall_compare.py` is unchanged. In particular, S_ref and
F_0(t_i) still come from the Phase 0 parity table (`market_20261106_2213.csv`), see OPEN_ISSUES.
**Consequences (200k paths, seed 20261308):**
- **AC-Q:** V_BS 1.014530, V_H 0.995842, gap −0.018688 (CRN SE 0.000217). Attribution:
  - term1 (flat σ_ATM = 18.85% vs the Heston expected-variance curve, √(w/T) = 22.99%): −0.008206 (0.000120);
  - term2 (marginal smile, exact): −0.020445;
  - term3 (path: forward skew, vol-of-vol): +0.009963 (0.000216).
  - Par coupon 5.60% (BS) vs 8.93% (H), +3.33% (0.05%).
  - Extra row: V_BS at the market ATM vol (18.29%) = 1.015253, i.e. +0.000723 above the BS leg (about 3.3 CRN SE of
    the gap). V_H minus it is −0.019411.
  - Exact checks: max |z| 0.20 (2-date legs) and 0.87 (no-call).
- **AC-A** (extrapolated beyond 1.61 y): gap −0.025137 (0.000342) = −0.018722 − 0.014800 + 0.008386. Exact checks:
  max |z| 0.72 (2-date legs) and 1.19 (no-call).
