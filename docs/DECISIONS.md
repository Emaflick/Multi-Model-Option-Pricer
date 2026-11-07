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
