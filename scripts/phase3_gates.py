"""Print the measured Phase 3 (SABR, beta = 1) gate values (docs/ROADMAP.md) on the test grids.

Same grids, sets, seeds and settings as tests/test_sabr.py (derivations/sabr.md §5; D-022 to
D-024). Gate 3 reads the summary CSV written by scripts/sabr_degradation.py. With --seeds n the
gate is re-run on n fresh seeds (20261107 + 1000 + s) to measure its false-fail rate (D-023).
Usage: .venv/bin/python scripts/phase3_gates.py [--seeds 100]
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from scipy.integrate import quad

from qpricer.engines.mc import mc_estimate
from qpricer.models import bsm, sabr

SEED = 20261107
F0, ALPHA, RHO, NU = 100.0, 0.2, -0.75, 0.4  # gate set G
K_GATE = F0 * np.exp(np.array([-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2]))
T_GATE, M_GATE, N = (0.25, 0.5, 1.0), 250, 200_000  # dt = T/250 at every T (validator F1)
HALVING_SEEDS = {0.25: SEED + 21, 0.5: SEED + 22, 1.0: SEED + 20}  # as tests/test_sabr.py
RHOS = (-0.999, -0.75, -0.3, 0.0, 0.4, 0.999)
FIG = Path(__file__).resolve().parents[1] / "report" / "figures"


def steps(T: float) -> int:
    return M_GATE


def fmt(a) -> str:
    return "[" + ", ".join(f"{v:+.2f}" for v in np.atleast_1d(a)) + "]"


def gate_nu_zero() -> None:
    k, T = np.linspace(-1.0, 1.0, 201), np.array([0.01, 0.25, 1.0, 5.0, 10.0])[:, None]
    e0 = e12 = 0.0
    for rho in RHOS:
        for a in (0.05, 0.2, 1.0):
            e0 = max(e0, np.abs(sabr.implied_vol(1.0, np.exp(k), T, a, rho, 0.0) - a).max())
            e12 = max(e12, np.abs(sabr.implied_vol(1.0, np.exp(k), T, a, rho, 1e-12) - a).max())
    print("Gate 1: nu -> 0, beta = 1 -> flat vol alpha (tol 1e-10); |k| <= 1, T 0.01..10, 6 rho")
    print(
        f"  max |sigma_H - alpha|: nu = 0 -> {e0:.1e}, nu = 1e-12 -> {e12:.1e}   PASS={e12 < 1e-10}"
    )


def gate_hagan_vs_mc() -> None:
    print(
        f"Gate 2: Hagan vs MC within 3 SE, set G (alpha={ALPHA}, rho={RHO}, nu={NU}), puts, "
        f"k = {fmt(np.log(K_GATE / F0))}, N = {N:.0e}, dt = T/250, nu^2 T <= {NU**2:.2f}"
    )
    worst = 0.0
    for i, T in enumerate(T_GATE):
        ref = bsm.price(F0, K_GATE, T, 1.0, sabr.implied_vol(F0, K_GATE, T, ALPHA, RHO, NU), -1)
        for method, seed in (("euler", SEED + i), ("mixing", SEED + 10 + i)):
            t0 = time.perf_counter()
            v, se, n, sd = sabr.mc_price(
                F0, K_GATE, T, 1.0, ALPHA, RHO, NU, -1, n_steps=steps(T), n_paths=N, seed=seed,
                method=method,
            )  # fmt: skip
            z = (v - ref) / se
            if method == "euler":
                worst = max(worst, np.abs(z).max())
            tag = "gated" if method == "euler" else "reported (17)"
            print(
                f"  T={T:<4g} {method:6s} seed={sd} M={steps(T)}: z = {fmt(z)}  "
                f"SE = [{', '.join(f'{s:.4f}' for s in se)}]  {time.perf_counter() - t0:.1f}s {tag}"
            )
    print(f"  max |z| (14) over 21 comparisons = {worst:.2f}   PASS={worst <= 3}")
    kw = dict(n_steps=steps(1.0), n_paths=N)
    v1, s1, *_ = sabr.mc_price(F0, K_GATE, 1.0, 1.0, ALPHA, RHO, NU, -1, seed=SEED + 2, **kw)
    v2, s2, *_ = sabr.mc_price(
        F0, K_GATE, 1.0, 1.0, ALPHA, RHO, NU, -1, seed=SEED + 12, method="mixing", **kw
    )
    z = (v1 - v2) / np.hypot(s1, s2)
    print(f"  oracle (f) (14) vs (17), T = 1, independent seeds: z = {fmt(z)}")


def crn_pair(T: float, M: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per-path puts of (14) and (17) at dt = T/M and T/(2M), common random numbers: the coarse
    increments are (Z_1 + Z_2) / sqrt 2 of the fine pair (tests/test_sabr.py::halving_payoffs)."""

    def fine():
        rng = np.random.default_rng(seed)
        for _ in range(2 * M):
            yield rng.standard_normal((2, N))

    def coarse():
        rng = np.random.default_rng(seed)
        for _ in range(M):
            yield (rng.standard_normal((2, N)) + rng.standard_normal((2, N))) / np.sqrt(2)

    out = []
    for dt, draws in ((T / M, coarse()), (T / (2 * M), fine())):
        x, aT, w, il = sabr.simulate(ALPHA, RHO, NU, dt, draws)
        iv = il + 0.5 * dt * (aT**2 - ALPHA**2)
        lnf = np.log(F0) + RHO * (aT - ALPHA) / NU - 0.5 * RHO**2 * iv
        sig = np.sqrt((1 - RHO**2) * iv / T)
        mix = bsm.price(np.exp(lnf), K_GATE[:, None], T, 1.0, sig, -1)
        out.append((np.maximum(K_GATE[:, None] - F0 * np.exp(x), 0.0), mix))
    return out


def halving() -> None:
    """CRN halving at the gate dt = T/250, every gate T (tests/test_sabr.py::test_halving_dt)."""
    print("Oracle (e): dt = T/250 vs T/500 with common random numbers, k = -0.2 ... +0.2")
    for T, seed in sorted(HALVING_SEEDS.items()):
        (e_c, m_c), (e_f, m_f) = crn_pair(T, steps(T), seed)
        dm, sm = mc_estimate(m_c - m_f)
        de, se = mc_estimate(e_c - e_f)
        _, s_gate = mc_estimate(e_f)
        vega = bsm.vega(F0, K_GATE, T, 1.0, sabr.implied_vol(F0, K_GATE, T, ALPHA, RHO, NU))
        r = abs(de) / s_gate
        print(f"  T={T:<4g} seed={seed}")
        print(f"    (17): d / SE_diff = {fmt(dm / sm)}   PASS(<=3)={np.all(abs(dm) <= 3 * sm)}")
        print(f"    (14): d / SE_diff = {fmt(de / se)}   (CRN resolves the O(dt) bias)")
        print(
            f"    (14): |d| / SE_gate = [{', '.join(f'{v:.3f}' for v in r)}]  max {r.max():.3f}"
            f"   PASS(<=0.1)={np.all(r <= 0.1)}"
        )
        print(
            f"    (14): Richardson bias 2d in vol pts = "
            f"[{', '.join(f'{v:+.4f}' for v in 200 * de / vega)}]; "
            f"in SE_gate = [{', '.join(f'{v:+.2f}' for v in 2 * de / s_gate)}]"
        )
    print("  (14) order check, T = 1: M (V_{1/M} - V_{1/2M}) at k = -0.2, 0, +0.2 (flat: order 1)")
    for M in (25, 50, 125, 250):
        (e_c, _), (e_f, _) = crn_pair(1.0, M, HALVING_SEEDS[1.0])
        d, s = mc_estimate(e_c - e_f)
        sd = ", ".join(f"{v:.2f}" for v in M * s[[0, 3, 6]])
        print(f"    M={M:<4d} {fmt(M * d[[0, 3, 6]])} +- [{sd}]")


def false_fail_rate(n_seeds: int) -> None:
    """Gate 2 re-run on fresh seeds SEED + 1000 + s (never used for the declared gate)."""
    refs = [bsm.price(F0, K_GATE, T, 1.0, sabr.implied_vol(F0, K_GATE, T, ALPHA, RHO, NU), -1)
            for T in T_GATE]  # fmt: skip
    z = np.empty((n_seeds, len(T_GATE), K_GATE.size))
    for s in range(n_seeds):
        for i, T in enumerate(T_GATE):
            v, se, *_ = sabr.mc_price(F0, K_GATE, T, 1.0, ALPHA, RHO, NU, -1, n_steps=steps(T),
                                      n_paths=N, seed=SEED + 1000 + 10 * s + i)  # fmt: skip
            z[s, i] = (v - refs[i]) / se
    fail = np.abs(z) > 3
    print(f"Gate 2 robustness: {n_seeds} fresh seeds x 21 comparisons, dt = T/250")
    print(
        f"  seeds failing the gate: {fail.any(axis=(1, 2)).mean():.1%}; per comparison "
        f"{fail.mean():.2%} (pure noise: 0.27%); mean z {z.mean():+.3f}, sd z {z.std():.3f}"
    )
    for i, T in enumerate(T_GATE):
        print(f"  T={T:<4g} mean z per strike {fmt(z[:, i].mean(axis=0))}")


def oracles() -> None:
    T, kc = 1.0, sabr.Z_C * ALPHA / NU
    jump = 0.0
    for rho in (-0.75, 0.0, 0.5):
        b = 1 + (rho * NU * ALPHA / 4 + (2 - 3 * rho**2) * NU**2 / 24) * T
        for sgn in (1, -1):
            k = sgn * kc * np.array([1 - 1e-9, 1 + 1e-9])
            s = sabr.implied_vol(F0, F0 * np.exp(k), T, ALPHA, rho, NU)
            slope = b * (0.5 * rho * NU + (2 - 3 * rho**2) * NU**2 / (6 * ALPHA) * k[0])  # eq. (8)
            jump = max(jump, abs(s[1] - s[0] - slope * (k[1] - k[0])) / s[0])
    k, Tc = np.linspace(-1.5, 1.5, 61), np.array([0.1, 1.0, 5.0])[:, None]
    sym = off = 0.0
    s0 = sabr.implied_vol(F0, F0 * np.exp(k), Tc, ALPHA, 0.0, NU)
    sym = np.max(np.abs(s0 / sabr.implied_vol(F0, F0 * np.exp(-k), Tc, ALPHA, 0.0, NU) - 1))
    for rho in (-0.999, -0.75, 0.3, 0.999):
        s = sabr.implied_vol(F0, F0 * np.exp(k), Tc, ALPHA, rho, NU)
        s_m = sabr.implied_vol(F0, F0 * np.exp(-k), Tc, ALPHA, -rho, NU)
        o = ALPHA * sabr.z_over_x(-NU * k / ALPHA, rho) * rho * NU * ALPHA * Tc / 2
        off = max(off, np.max(np.abs(s - s_m - o) / s))
    xq = 0.0
    for rho in RHOS:
        for z in (-8.0, -1.5, -0.999, -0.3, -2e-3, -5e-4, 5e-4, 2e-3, 0.3, 0.999, 1.5, 8.0):
            pts = [rho] if min(0, z) < rho < max(0, z) else None
            x, _ = quad(lambda u, r=rho: 1 / np.sqrt(1 - 2 * r * u + u * u), 0, z, points=pts,
                        epsabs=0, epsrel=1e-13, limit=200)  # fmt: skip
            xq = max(xq, abs(sabr.z_over_x(z, rho) / (z / x) - 1))
    print("Formula oracles (sabr.md §5 (ii), (iii), eq. 3)")
    print(f"  jump across |z| = z_c net of the eq. (8) slope (rel): {jump:.1e} (tol 1e-12)")
    print(f"  rho = 0 symmetry sigma(k) = sigma(-k) (rel): {sym:.1e} (tol 1e-14)")
    print(f"  rho != 0 offset identity of §3.3 (rel to sigma): {off:.1e} (tol 1e-14)")
    print(f"  z/x(z) vs quadrature of int du/D(u), rho incl. +-0.999 (rel): {xq:.1e} (tol 1e-11)")


def chk_diagnostic() -> None:
    """[CHK] Table 1 (single source, not gated): F = K = 1, T = 1, alpha = 0.2, ATM call = put."""
    print("Published [CHK] Table 1 (FDM, single source, diagnostic only): ATM, T = 1, alpha = 0.2")
    for i, (rho, nu, ref) in enumerate(
        (
            (-0.75, 0.2, 0.07910),
            (-0.5, 0.2, 0.07942),
            (-0.25, 0.2, 0.07969),
            (-0.75, 0.4, 0.07860),
            (-0.75, 0.6, 0.07811),
        )  # row 5: nu = 0.6 (erratum, validator F5)
    ):
        h = bsm.price(1.0, 1.0, 1.0, 1.0, sabr.implied_vol(1.0, 1.0, 1.0, ALPHA, rho, nu), -1)
        v, se, *_ = sabr.mc_price(1.0, 1.0, 1.0, 1.0, ALPHA, rho, nu, -1, n_steps=250, n_paths=N,
                                  seed=SEED + 30 + i, method="mixing")  # fmt: skip
        print(
            f"  rho={rho:+.2f} nu={nu}: FDM {ref:.5f}  Hagan {h:.6f} ({h - ref:+.1e})  "
            f"MC (17) {v:.5f} +- {se:.5f} (z vs FDM {(v - ref) / se:+.2f})"
        )


def gate_figure() -> None:
    path = FIG / "sabr_hagan_vs_mc_summary.csv"
    print(f"Gate 3: figure {FIG / 'sabr_hagan_vs_mc.png'} (scripts/sabr_degradation.py)")
    if not path.exists():
        print("  summary CSV missing: run scripts/sabr_degradation.py")
        return
    with open(path) as f:
        for r in csv.DictReader(f):
            print(
                f"  nu={r['nu']:<4} T={r['T']:<5} nu^2T={float(r['nu2T']):<5.2f} "
                f"max|err| (|z|>3) = {float(r['max_abs_err_volpts_significant']):6.3f} vol pt "
                f"at k={float(r['k_at_max_significant']):+.2f}; "
                f"max|z| = {float(r['max_abs_z_price']):7.1f}; "
                f"min K p(K) = {float(r['min_hagan_density']):+.1e}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=0, help="fresh seeds for the false-fail rate")
    n_seeds = ap.parse_args().seeds
    gate_nu_zero()
    gate_hagan_vs_mc()
    halving()
    oracles()
    chk_diagnostic()
    gate_figure()
    if n_seeds:
        false_fail_rate(n_seeds)


if __name__ == "__main__":
    main()
