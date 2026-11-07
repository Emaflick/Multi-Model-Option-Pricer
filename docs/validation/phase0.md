# Phase 0 validation: setup and SPX data (independent)

Validator: `validator` agent, 2026-11-06. Snapshot: `data/raw/spx_20261106_2213.parquet`
(15843 rows, 40 expiries, snapshot 2026-11-06T22:13:53Z). Outputs checked:
`data/processed/{market,cleaning,surface,arb_removed}_20261106_2213.csv`, `src/qpricer/data.py`,
`src/qpricer/market.py`, `scripts/fetch_spx.py`, `scripts/build_market.py`.

Tests: `tests/validation/test_phase0_independent.py` (26 tests). Every number below is printed by
`.venv/bin/python tests/validation/test_phase0_independent.py` or asserted by the tests.

## Status after fix round 1 (re-validation)

Full suite (`.venv/bin/pytest -q`): **52 passed, 0 failed**. `ruff check src tests scripts` is clean.

**Both ROADMAP gates PASS, and no blocker remains.**
- F1, F2, F4 and F6 are closed, and I re-verified each one with independent code.
- F3 is answered in the documentation.
- F5 remains open, as a minor item deferred to Phase 5 and logged in `OPEN_ISSUES.md`.
- One new minor item, F7, came out of the fix itself. It is a documentation and design note, not a
  blocker.

Changes made to the validation tests in this round:
* My reference clock now applies the early-close rule (13:00 ET for SPXW on NYSE early-close days)
  to every expiry. It is my own finding (F1), so it is the correct reference. The T comparison
  therefore now covers **all 40 expiries at atol 1e-12, with no exclusion**. Before, it compared
  39 expiries and excluded 2026-11-27. The early-close test tolerance went from 1e-9 to 1e-12.
* My independent acceptance rule now includes the two new D-005 conditions, coded on my side.
* New tests:
  - the arbitrage-filtered surface (D-010);
  - local minimality and staleness of the removed quotes;
  - module vs independent kept sets at n_near = 10 / 20 / 40;
  - the new rules drop nothing on the snapshot;
  - the size of the in-band gate on valid synthetic chains.

No tolerance was loosened.

## Method (independent of the implementation)

* Cleaning re-implemented from the ROADMAP text: root parsed from the OCC symbol, one root per date,
  then the filters bid > 0, finite ask ≥ bid, (ask−bid)/mid ≤ 0.5 and OI > 0, then mid = (bid+ask)/2.
* T computed with `datetime` + `zoneinfo` for each expiry.
  - The ET offset is checked against a hand-coded US DST rule.
  - Early-close dates come from the NYSE rule: the day after Thanksgiving, plus July 3 and Dec 24
    when they are weekday sessions. This gives 2026-11-27, 2026-12-24, 2027-11-26, 2028-07-03 and
    2028-11-24, the same set as `data.EARLY_CLOSE`.
* Parity fit, **replication** of the documented rule (D-004/D-005), built from different primitives:
  `scipy.stats.theilslopes(method="joint")`, `median_abs_deviation(scale="normal")`, `np.polyfit`,
  R² = corr², and standard errors from the full uncentred QR covariance + delta method.
* Parity fit, **second method**: L1 regression (LAD, `scipy.optimize.linprog`) on the 20 nearest
  pairs with no outlier rejection at all.
* Static arbitrage, tested in two ways:
  - at single strikes: adjacent bid/ask monotonicity and vertical-spread checks;
  - on the whole surface: an O(n) running-extremum test. It is a different method from the
    module's pairwise matrix, and I cross-checked it against brute force: 468 = 468 flags on the
    OTM cut, 0 = 0 on the surface.
  - plus strict adjacent-triple butterflies.
* Gate power and size: synthetic chains (labelled `synthetic`) run through the public
  `implied_forwards`.

## Gate table (re-measured)

| # | Check | Measured | Tolerance | Result |
|---|---|---|---|---|
| G1 | Parity R² on each kept expiry (my recomputation) | min 0.999941 (2027-02-19); 39/39 kept > 0.999 | > 0.999 | **PASS** |
| G1b | Failing expiries dropped and logged | only 2026-12-08 (7 clean pairs < 10). WARNING logged, `reason` in CSV, independent count agrees | drop + log | **PASS** |
| G2 | Implied DF on each kept expiry | [0.923895, 0.998395] | (0.8, 1] | **PASS** |
| C1 | Cleaning row counts vs `cleaning_*.csv` | 15843 → 14707 → 14236 → 14228 → 13608 → 12956, identical | exact | **PASS** |
| C2 | Clean set and mid vs module | same contract set; mid = (bid+ask)/2 exactly; one root per date | exact | **PASS** |
| C3 | Plain OTM cut (`build_surface()["otm"]`) | 7650 quotes, set equal to my own cut; k = ln(K/F) to 1e-14 | exact | **PASS** |
| C4 | T: ACT/365F, AM 09:30 / PM 16:00 / early close 13:00 ET, DST | max \|ΔT\| = 0 over all 40 expiries; ET offsets match the DST rule | 1e-12 | **PASS** |
| C5 | Early close (F1): 2026-11-27 SPXW in the CSV; `settlement_utc` on all 5 early-close dates of 2026–2028; AM settlement on an early-close date stays 09:30 | T_csv = 0.0570512 = my value; all settlement instants equal | 1e-12 / exact | **PASS** |
| C6 | F, DF, R², SEs, `max_res_h`, `bracketed` reproduced from raw | \|ΔF\| ≤ 9.1e-13, \|ΔDF\| ≤ 7.8e-16, \|ΔR²\| ≤ 6.7e-16; n_pairs, n_used, rejected_K, kept identical; SEs and max_res_h rtol 1e-8 / 1e-9; r = −ln DF / T at rtol 1e-9 | 1e-10 | **PASS** |
| C7 | Second method (LAD, no rejection) vs CSV | max \|ΔF\|/F_se = 1.87 (2028-01-21; 0.61 pt); max \|ΔDF\|/DF_se = 2.43 | < 3 SE | **PASS** |
| C8 | No hard-coded r / q | AST scan of the 4 Phase-0 files: no float in (0, 0.2] and no rate/yield/dividend name. Perturbing spot, Yahoo IV, lastPrice and inTheMoney leaves the table bit-identical. DF = 0.83 at T = 0.1 recovered to 1e-12 | none | **PASS** |
| R1 | Fitted line inside every used pair's C−P band (now also a module gate, D-005 (i)) | max \|res\|/h = 0.814 (2026-11-18) | ≤ 1 | **PASS** |
| R2 | Each rejected strike is a strict static arbitrage; rejected fraction below the Theil–Sen breakdown point (now also a module gate, D-005 (ii)) | 9/9 strikes; max n_rej/n_sel = 0.20 | < 0.293 | **PASS** |
| R3 | Synthetic: rejection must not rescue pure-noise chains (σ = 2 pt, h = 0.5; population R² ≈ 0.995) | 0/500 kept (was 4/500); plain OLS 0/500 | 0 | **PASS** |
| R3b | New rules drop no expiry on the snapshot (production n_near = 20) | the only drop reason is `n_pairs<10` (2026-12-08) | none | **PASS** |
| R3c | In-band gate size: valid synthetic chains at σ = 0.2·h | kept 1000/1000 | ≥ 99 % | **PASS** |
| R4 | Window sensitivity: module and independent kept sets equal at n_near = 10/20/40 | 34 / 39 / 38 kept, sets equal; max \|ΔF\|/F = 2.09e-4 | equal; < 5e-4 | **PASS** |
| R5 | Window placed around F0 equals the 20 pairs nearest the fitted F | identical on all 39 | exact | **PASS** |
| A1 | Calibration surface `build_surface()["surface"]` free of strict static arbitrage (F4) | all-pairs monotonicity + vertical spread: 0 (OTM cut: 468 flags); adjacent butterflies: 0; 7650 → 7625 quotes | 0 | **PASS** |
| A2 | Removed quotes necessary and stale | restoring any one of the 25 re-creates an arbitrage; mids are 14.4–179.9 pt (≥ 6.8 spreads) from the interpolation of their surviving neighbours | > 1 spread | **PASS** |
| A3 | Artefacts `surface_*.csv` / `arb_removed_*.csv` consistent with `build_surface` | rows 7650 → 7625; 25 symbols equal | exact | **PASS** |
| D1 | AM vs PM forwards on the 5 overlap dates (D-002) | max \|F_SPX − F_SPXW\| = 0.27 pt | < 0.4 | **PASS** |
| D2 | Long-end DF (T > 0.3) monotone within noise | max increase z = 0.42 | < 3 | **PASS** |
| S1 | Raw snapshot integrity | single stamp = file stamp; spot = metadata; no duplicate symbols; expiries = listed ∩ [7, 730] d; no failures; last trade before the snapshot | exact | **PASS** |

## Findings: status

| ID | Finding | Status | Severity |
|---|---|---|---|
| F1 | SPXW early-close settlement ignored | **closed** | (was minor) |
| F2 | MAD rejection could lift a noisy expiry over R² | **closed** | (was minor) |
| F3 | R² is an uneven gate on its own | **closed** (D-004/D-005 say the in-band rule carries the economic content) | design note |
| F4 | Stale quotes break static arbitrage on the OTM surface | **closed for `["surface"]`**. `["otm"]` intentionally still has them | (was a forward risk) |
| F5 | Short-end DF not identified; DF_se understates its uncertainty | **open**, deferred to Phase 5 (`OPEN_ISSUES.md`) | minor |
| F6 | F extrapolated below the strike range on 2026-12-07 | **closed** (flag + WARNING) | (was minor) |
| F7 | In-band gate size on noisy short-dated expiries; D-004 text stale | **open** (new) | minor |

### F1 (closed)
`data.settlement_utc` now uses 13:00 ET for SPXW on `EARLY_CLOSE` dates and warns for PM expiries
after 2028-12-31. All of the following agree with my reference clock:
- the CSV T for 2026-11-27 (0.0570512);
- the 4 other early-close dates of 2026–2028;
- an AM expiry on an early-close date (stays 09:30).

My early-close set was derived independently from the NYSE rule and is identical to the
implementer's.

### F2 (closed)
Two acceptance rules were added: (i) max |res|/h ≤ 1 and (ii) n_rej/n_sel < 1 − 1/√2. At h = 0.5
the module now keeps 0.000 of 1000 chains at every σ from 0.9 to 3.0. Plain-OLS R² alone would
pass 0.61 of them at σ = 0.9 and 0.38 at σ = 1.0. So the rescue route is gone, and the module is
now stricter than R² alone. On the real snapshot the rules bind nowhere: headroom is 0.814 vs 1
and 0.20 vs 0.293.

### F4 (closed for the calibration input)
I verified D-010 with a different algorithm, cross-checked against brute force:
- the calibration surface `build_surface()["surface"]` has no strict monotonicity, vertical-spread
  or adjacent butterfly arbitrage on bid/ask;
- the greedy removal is locally minimal: each of the 25 removed quotes is needed;
- every removed quote sits far outside the level its neighbours imply.

`["otm"]` is the plain ROADMAP cut on purpose and still contains them (468 flags).
**Phase 5 must read `["surface"]`, not `["otm"]`.** I recommend that the Phase 5 validation
include a check that the calibrated quote set is a subset of `["surface"]`.

### F5 (open, minor, deferred)
Not re-measured as changed: short-end r spans 2.64 %–16.89 %, DF rises with T 10 times out of 38
(max z = 3.87), and LAD on 40 pairs moves short-end DF by up to 9.9 DF_se. The open issue is
correctly logged. Resolve it before IV inversion in Phase 5.

### F6 (closed)
`bracketed` is False only for 2026-12-07, matching my independent Kmin ≤ F ≤ Kmax check. The
expiry is kept with a WARNING. Its F_se of 0.40 reflects the extrapolation.

### F7 (new, minor): size of the in-band gate; stale D-004 sensitivity text
The in-band rule is a hard cap of 1 on the single worst pair, so it can drop a valid but noisy
expiry because of one marginal quote:
* **Synthetic valid chains** (exact parity + Gaussian noise; 1000 each):
  - kept 1.000 at σ/h = 0.2, 0.999 at 0.3, 0.965 at 0.35 and 0.556 at 0.5.
* **On this snapshot**, rms(res)/median(h) is 0.044–0.372 (median 0.111):
  - three short weeklies sit above 0.3: 2026-11-19 (0.372), 2026-11-18 (0.353) and
    2026-11-17 (0.323);
  - for expiries like these, a future snapshot carries a risk of a few percent per expiry of
    being dropped.
* **Already visible at n_near = 40**: 2026-11-17 is dropped at max |res|/h = 1.057.
  - The two out-of-band pairs are K = 7490/7500 (ITM puts).
  - Their direction is consistent with the ITM-quote bias of D-007.
  - D-004 still says "`--n-near 40` keeps 39/40". It is now 38/40.

The failure mode is conservative: an expiry is lost, never a bad F accepted. So this is not a
blocker.
**Recommendation:**
- update the D-004 sensitivity sentence to 38/40 at n_near = 40;
- in DECISIONS.md, state the in-band gate's measured size and that dropping a whole expiry for
  one marginal pair is accepted on purpose.
- Do not respond by trimming out-of-band pairs one at a time. That would reopen F2.

## Edge cases probed, no issue found
* DST: EST snapshot; settlements after 2027-03-14 use EDT (UTC−4). The holiday-shifted expiry
  2027-06-17 is used as listed.
* AM (SPX) vs PM (SPXW) on overlap dates: one root kept per date, SPX preferred, forwards agree
  within 0.27 pt.
* Expiry window: 49 listed, 40 kept within [7, 730] days.
* Threshold edges:
  - "drop (ask−bid)/mid > 0.5" is implemented as keep ≤ 0.5;
  - zero, NaN and crossed bids are dropped and counted;
  - the clean set has no locked quotes (bid = ask). A locked quote with a non-zero residual would
    give |res|/h = ∞ and drop its expiry; it cannot occur on this snapshot.
* Spot (7376.86) is not used anywhere (C8). The 7-day F is 9.4 pt above it; D-008 covers this.

## Golden files
None created. F and DF from this snapshot have no closed form. My two methods agree within 3 SE,
not to golden precision. The committed CSVs, together with the implementer's rtol-1e-12
reproduction test, already freeze them.

## Remaining actions (none blocking Phase 0)
1. F7: correct the D-004 sensitivity sentence (n_near = 40 keeps 38/40) and record the in-band
   gate's size in DECISIONS.md.
2. F5: resolve in Phase 5 before IV inversion (already in `OPEN_ISSUES.md`).
3. Phase 5: calibrate on `build_surface()["surface"]` only.
