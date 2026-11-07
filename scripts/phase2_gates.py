"""Print the measured Phase 2 (Heston) gate values (docs/ROADMAP.md) on the test grids.

Same grids, sets, seeds and settings as tests/test_heston.py and tests/test_mc.py
(derivations/heston.md §5, heston-mc.md §5; D-015, D-017, D-018).
Usage: .venv/bin/python scripts/phase2_gates.py
"""

import time
import timeit
import warnings
from functools import partial

import numpy as np
from scipy.stats import norm

from qpricer.engines import mc
from qpricer.engines.fourier import cos_price, lewis_price
from qpricer.models import bsm
from qpricer.models.heston import HestonParams, total_variance

FO = HestonParams(1.5768, 0.0398, 0.5751, -0.5711, 0.0175)
GS = HestonParams(1.3253, 0.0354, 0.3877, -0.7165, 0.0354)
LW = HestonParams(4.0, 0.25, 1.0, -0.5, 0.04)
AT2 = HestonParams(1.5, 0.04, 0.3, -0.9, 0.04)
AI = HestonParams(0.5, 0.04, 1.0, -0.9, 0.04)
BK2 = HestonParams(2.0, 0.09, 1.0, -0.3, 0.09)
K_GRID, T_GRID = np.arange(60.0, 150.0 + 1e-9, 5.0), (0.1, 0.25, 0.5, 1.0, 2.0, 3.0)
SEED = 20261107


def quiet_cos(*a, **kw) -> tuple[np.ndarray, bool]:
    """cos_price plus whether it emitted a RuntimeWarning."""
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        v = cos_price(*a, **kw)
    return v, any(issubclass(w.category, RuntimeWarning) for w in wl)


def cos_vs_lewis(p: HestonParams, **fixed) -> float:
    """Max |COS - Lewis| on the gate grid; fixed = {} is the self-checking default (D-017)."""
    worst = 0.0
    for T in T_GRID:
        for DF in (1.0, 0.95):
            for cp in (1, -1):
                c, _ = quiet_cos(100.0, K_GRID, T, DF, p, cp, **fixed)
                worst = max(worst, np.abs(c - lewis_price(100.0, K_GRID, T, DF, p, cp)).max())
    return worst


def random_scan(**fixed) -> tuple[int, int, int, float, float]:
    """Validator F1 experiment: seed 5, 300 random SPX-plausible sets x 6 T, puts, DF = 1.
    Returns (cells > 1e-6, warned, silent failures, max unwarned error, max error)."""
    rng, errs, warned = np.random.default_rng(5), [], []
    for _ in range(300):
        prm = (
            float(np.exp(rng.uniform(np.log(0.3), np.log(8.0)))),
            rng.uniform(0.01, 0.12),
            rng.uniform(0.1, 1.2),
            rng.uniform(-0.95, -0.2),
            rng.uniform(0.005, 0.12),
        )
        p = HestonParams(*prm)
        for T in T_GRID:
            c, w = quiet_cos(100.0, K_GRID, T, 1.0, p, -1, **fixed)
            errs.append(np.abs(c - lewis_price(100.0, K_GRID, T, 1.0, p, -1)).max())
            warned.append(w)
    e, w = np.array(errs), np.array(warned)
    bad = e > 1e-6
    return bad.sum(), w.sum(), (bad & ~w).sum(), np.max(e[~w], initial=0.0), e.max()


def xi_limit(rho: float, theta: float = 0.04, v0: float = 0.04) -> tuple[float, float]:
    p, worst, worst_plain = HestonParams(1.5, theta, 1e-4, rho, v0), 0.0, 0.0
    for T in T_GRID:
        S, k = float(total_variance(T, p)), p.kappa
        tau, E = (1 - np.exp(-k * T)) / k, np.exp(-k * T)
        J = p.theta / k * (T - tau) + (p.v0 - p.theta) / k * (tau - T * E)
        d1 = (np.log(100.0 / K_GRID) + 0.5 * S) / np.sqrt(S)
        for DF in (1.0, 0.95):
            corr = rho * p.xi * J * (-DF * 100.0 * norm.pdf(d1) * (d1 - np.sqrt(S)) / (2 * S))
            for cp in (1, -1):
                black = bsm.price(100.0, K_GRID, T, DF, np.sqrt(S / T), cp)
                for eng in (cos_price, lewis_price):
                    v = eng(100.0, K_GRID, T, DF, p, cp)
                    worst = max(worst, np.abs(v - black - corr).max())
                    worst_plain = max(worst_plain, np.abs(v - black).max())
    return worst, worst_plain


def main() -> None:
    print("Gate 1: |COS - Lewis| on K 60..150 x T 0.1..3 x DF {1, .95} x {C, P} (tol 1e-6)")
    for name, p in (("FO", FO), ("GS", GS), ("LW", LW), ("AT2", AT2)):
        e = cos_vs_lewis(p)
        e18, e12 = cos_vs_lewis(p, N=256, L=18.0), cos_vs_lewis(p, N=256, L=12.0)
        print(
            f"  {name:4s} nu_F={p.feller:.2f}  default (self-checking): {e:.2e}   "
            f"[fixed N=256: L=18 {e18:.2e}, L=12 {e12:.2e}]"
        )

    print("COS robustness (validator F1): 300 random SPX-plausible sets x 6 T, K 60..150, puts")
    for label, fixed in (("fixed N=256, L=18", dict(N=256, L=18.0)), ("default", {})):
        nbad, nw, nsil, mu, mx = random_scan(**fixed)
        print(
            f"  {label:17s}: > 1e-6 in {nbad}/1800 ({nbad / 1800:.1%}), warned {nw}, "
            f"silent {nsil}, max unwarned {mu:.2e}, max {mx:.2e}"
        )
    K25 = np.linspace(70.0, 140.0, 25)
    worst = HestonParams(0.316, 0.072, 1.14, -0.92, 0.058)  # worst cell of the scan, T = 3
    for name, p, T in (("GS", GS, 1.0), ("FO", FO, 1.0), ("scan-worst", worst, 3.0)):
        calls = (
            partial(quiet_cos, 100.0, K25, T, 1.0, p, -1, N=256, L=18.0),
            partial(cos_price, 100.0, K25, T, 1.0, p, -1),
            partial(lewis_price, 100.0, K25, T, 1.0, p, -1),
        )
        ms = [min(timeit.repeat(f, number=5, repeat=5)) / 5 * 1e3 for f in calls]
        print(
            f"  time per call, 25 strikes, {name} T={T:g}: fixed(256,18) {ms[0]:.2f} ms, "
            f"default {ms[1]:.2f} ms, Lewis {ms[2]:.2f} ms"
        )

    print("Gate 2: xi = 1e-4 limit (D-015), COS and Lewis, tol 1e-5")
    for theta, v0 in ((0.04, 0.04), (0.04, 0.09), (0.09, 0.01)):
        w0, _ = xi_limit(0.0, theta, v0)
        w7, w7_plain = xi_limit(-0.7, theta, v0)
        print(
            f"  theta={theta}, v0={v0}: rho=0 vs Black(sigma_eff) {w0:.2e}; rho=-0.7 vs Black + "
            f"eq. 16 term {w7:.2e} (vs plain Black {w7_plain:.2e})"
        )

    print("Published oracles")
    fo1 = [lewis_price(100, 100, 1, 1, FO, 1), cos_price(100, 100, 1, 1, FO, 1, N=256, L=18)]
    print(
        f"  FO1 5.785155450: Lewis {fo1[0] - 5.785155450:+.1e}, COS default "
        f"{cos_price(100, 100, 1, 1, FO, 1) - 5.785155450:+.1e}, COS fixed (256, 18) "
        f"{fo1[1] - 5.785155450:+.1e}"
    )
    print(
        f"  FO2 22.318945791: Lewis {lewis_price(100, 100, 10, 1, FO, 1) - 22.318945791:+.1e}, "
        f"COS {cos_price(100, 100, 10, 1, FO, 1) - 22.318945791:+.1e}"
    )
    print(
        f"  A-I K=100 13.08467014 (Choi-Kwok, single source): Lewis "
        f"{lewis_price(100, 100, 10, 1, AI, 1) - 13.08467014:+.1e}"
    )
    for T, v in ((1, 7.27), (2, 11.73), (3, 15.48), (5, 21.75), (10, 33.84)):
        F, DF = 100 * np.exp(0.025 * T), np.exp(-0.025 * T)
        print(
            f"  AMST A1 T={T:<2d} {v:6.2f}: Lewis dev {lewis_price(F, 100, T, DF, FO, 1) - v:+.4f}"
            f" (stated precision 0.005; OPEN_ISSUES)"
        )
    for T, v, tol in ((1, 8.8948, 5e-5), (2, 13.20, 5e-3), (3, 16.79, 5e-3)):
        F, DF = 100 * np.exp(0.025 * T), np.exp(-0.025 * T)
        print(
            f"  AMST A2 T={T} {v}: Lewis dev {lewis_price(F, 100, T, DF, AT2, 1) - v:+.1e}"
            f" (stated precision {tol:g})"
        )

    print("Gate 3: QE / QE-M MC vs COS, 2e5 paths, dt = 1/32, puts K = 70/100/140, (MC-COS)/SE")
    cases = {
        "FO": (FO, 1.0, 100.0, 1.0, 256, 18.0),
        "GS": (GS, 1.0, 100.0, 1.0, 256, 18.0),
        "BK2": (BK2, 5.0, 100 * np.exp(0.25), np.exp(-0.25), 256, 18.0),
        "AI": (AI, 10.0, 100.0, 1.0, 4096, 40.0),
    }
    K = np.array([70.0, 100.0, 140.0])
    for i, (name, (p, T, F, DF, N, L)) in enumerate(cases.items()):
        ref = cos_price(F, K, T, DF, p, -1, N=N, L=L)
        for scheme in ("qe", "qe-m", "euler"):
            t0 = time.perf_counter()
            v, se, n, seed = mc.european_price(
                F,
                K,
                T,
                DF,
                p,
                -1,
                n_steps=int(32 * T),
                n_paths=200_000,
                seed=SEED + i,
                scheme=scheme,
            )
            z = ", ".join(f"{q:+.2f}" for q in (v - ref) / se)
            tag = "" if scheme != "euler" else "  (diagnostic, not gated)"
            print(
                f"  {name:4s} nu_F={p.feller:.2f} {scheme:5s} seed={seed}: z = [{z}]  "
                f"SE = [{', '.join(f'{s:.4f}' for s in se)}]  {time.perf_counter() - t0:.1f}s{tag}"
            )


if __name__ == "__main__":
    main()
