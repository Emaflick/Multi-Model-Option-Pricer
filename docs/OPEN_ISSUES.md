# Open issues

Gates that failed after 3 fix attempts. Format: phase · gate · observed vs expected · diagnosis · next step.

- **Phase 0 · DF in (0.8, 1] (passes; deferred minor item, validator finding F5, not a failed gate)**
  **Observed vs expected:** for T < 0.1 the parity-implied r spans 2.64%–16.89% (se 0.4%–4.3%), where a
  smooth zero curve near the long-end level of 4.5%–5.1% is expected. DF(T) rises between consecutive
  kept expiries 10 times out of 38 (max z = 3.87, 2026-12-03 → 2026-12-04). Going from 20 to 40 pairs moves
  short-end DF by up to 9.9 DF_se, so the residual-based DF_se understates the window uncertainty.
  **Diagnosis:** the slope signal (1 − DF)·ΔK is below quote resolution at short T, and the bias grows with
  window width (ITM-quote asymmetry, D-007). F is unaffected (se ≤ 0.4 pt). The (0.8, 1] gate is loose
  at the short end. Impact ≲ 0.06 vol points ATM.
  (Figures from docs/validation/phase0.md and data/processed/market_20261106_2213.csv.)
  **Next step (Phase 5):** replace the T < 0.1 parity DFs with a smooth or monotone DF(T) fitted to the
  identified long end (or an OIS curve) before IV inversion. Record the choice in DECISIONS.md.
