# Build status (`/build-all`, started 2026-11-06)

| Phase | Topic | Status | Notes |
|---|---|---|---|
| 0 | Setup & SPX data | **done** (commit 5a0a19f) | snapshot 2026-11-06 22:13Z; 39/40 expiries kept, min R² 0.999941, DF ∈ [0.9239, 0.9984]; 1 fix round (F1, F2, F4, F6); F5 deferred |
| 1 | BSM (Black-76) | **done** | parity 1.14e-13, IV round-trip 8.2e-13, greeks vs FD ≤ 1.24e-6 (gamma FD step D-014); 1 fix round (deep-OTM IV F1); F3 deferred |
| 2 | Heston (COS, Lewis, MC) | implementing | `heston.md`, `heston-mc.md`; ξ→0 gate clarified in D-015 |
| 3 | SABR | research done | `sabr.md` |
| 4 | Exotics (barrier, Asian, autocallable) | research done | `barrier.md`, `asian.md`, `autocallable.md` |
| 5 | Calibration to SPX | research | |
| 6 | Benchmark MC vs closed form | pending | |
| 7 | Report & interview material | pending | |
