# Build status (`/build-all`, started 2026-11-06)

| Phase | Topic | Status | Notes |
|---|---|---|---|
| 0 | Setup & SPX data | **done** (commit 5a0a19f) | snapshot 2026-11-06 22:13Z; 39/40 expiries kept, min R² 0.999941, DF ∈ [0.9239, 0.9984]; 1 fix round (F1, F2, F4, F6); F5 deferred |
| 1 | BSM (Black-76) | **done** (commit 1656279) | parity 1.14e-13, IV round-trip 8.2e-13, greeks vs FD ≤ 1.24e-6 (gamma FD step D-014); 1 fix round (deep-OTM IV F1); F3 deferred |
| 2 | Heston (COS, Lewis, MC) | **done** (commit a4d5561) | COS vs Lewis ≤ 1.8e-10 (self-checking COS, D-017); ξ→0 ≤ 5.6e-8 (D-015); QE vs COS 24/24 within 3SE, 480 fresh-seed z max 2.86; 1 fix round (F1 COS truncation on random sets 22.9% → 0%) |
| 3 | SABR | **done** (commit ac7e63c) | ν→0 3.0e-12; Hagan vs MC max abs(z) 1.68 (validator own MC 2.42), domain ν²T ≤ 0.16; step bias ≤ 0.17 SE at Δ = T/250; 1 fix round (F1 step bias, F2–F5, F7); F6 deferred to Phase 5 |
| 4 | Exotics (barrier, Asian, autocallable) | **done** (phase 4 commit) | 24 PASS / 0 FAIL / 1 DEFERRED; in+out 3.6e-15; BB MC vs RR max abs(z) 1.99; KV max abs(z) 1.90; CV VR 612–1924 (FM08); Haug per cell, no loosening; autocall vs calibrated Heston DEFERRED to Phase 5 (illustrative gap −0.0144 AC-Q); 1 fix round (F1 Haug, F2 Heston Asian grid); src 1900 lines |
| 5 | Calibration to SPX | research done | `calibration.md` |
| 6 | Benchmark MC vs closed form | pending | |
| 7 | Report & interview material | pending | |

## Carry-forward notes (orchestrator)
- **Phase 5:** the self-checking COS default (D-017) refines (N, L) adaptively, so the price is piecewise in the
  parameters at the ~1e-9·F level. Inside `least_squares` with finite-difference Jacobians this can inject noise into
  the Jacobian. Pin (N, L) per expiry for the duration of a fit (choose via the self-check at the start / at
  bounds-wide extremes, then pass explicit N, L), or set `diff_step` large enough, and re-check COS vs Lewis at the
  fitted parameters (D-017 note).
- **Phase 5:** calibration note's D-016/017/018 texts collide with existing numbers — use the next free D-numbers.
- **Phase 5:** if SABR is fitted, `alpha_from_atm` returns NaN above the feasibility ceiling b²/(4|a|) for ρ < 0
  (e.g. ρ=−0.9, ν=5, T=2 → 0.12%); the loss must handle NaN (Phase 3 validator F6).
- **Phase 5/7:** state SABR Hagan gate domain ν²T ≤ 0.16; at SPX-like ν≈1 Hagan errs by 0.4–0.8 vol pt in the left wing (Phase 3 F3).
- **Phase 5:** close OPEN_ISSUES F5 (short-end DF smoothing) per `calibration.md` §2.7.
- **Size:** core `src/` at 1891 lines after Phase 4 implementation (1360 after Phase 3) (target ≲ 1500); data.py + market.py = 481. Phases 4–5 will likely exceed the target — report to owner, don't refactor validated code.
