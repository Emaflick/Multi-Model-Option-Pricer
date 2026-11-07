# Build status (`/build-all`, started 2026-11-06)

| Phase | Topic | Status | Notes |
|---|---|---|---|
| 0 | Setup & SPX data | **done** (commit 5a0a19f) | snapshot 2026-11-06 22:13Z; 39/40 expiries kept, min R² 0.999941, DF ∈ [0.9239, 0.9984]; 1 fix round (F1, F2, F4, F6); F5 deferred |
| 1 | BSM (Black-76) | **done** (commit 1656279) | parity 1.14e-13, IV round-trip 8.2e-13, greeks vs FD ≤ 1.24e-6 (gamma FD step D-014); 1 fix round (deep-OTM IV F1); F3 deferred |
| 2 | Heston (COS, Lewis, MC) | **done** | COS vs Lewis ≤ 1.8e-10 (self-checking COS, D-017); ξ→0 ≤ 5.6e-8 (D-015); QE vs COS 24/24 within 3SE, 480 fresh-seed z max 2.86; 1 fix round (F1 COS truncation on random sets 22.9% → 0%) |
| 3 | SABR | implementing | `sabr.md` |
| 4 | Exotics (barrier, Asian, autocallable) | research done | `barrier.md`, `asian.md`, `autocallable.md` |
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
- **Phase 5:** close OPEN_ISSUES F5 (short-end DF smoothing) per `calibration.md` §2.7.
- **Size:** core `src/` at 1218 lines after Phase 2 (target ≲ 1500).
