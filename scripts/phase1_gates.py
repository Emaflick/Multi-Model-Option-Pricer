"""Print the measured Phase 1 gate values (docs/ROADMAP.md) on the dense test grids.

Same grids and FD steps as tests/test_bsm.py and tests/test_iv.py (derivations/bsm.md §2.4-2.5,
gamma step D-014). Usage: .venv/bin/python scripts/phase1_gates.py
"""

import numpy as np

from qpricer.iv import implied_vol
from qpricer.models import bsm

F = 100.0
SIG = np.linspace(0.05, 1.0, 40)[:, None, None, None]
TT = np.geomspace(0.02, 5.0, 30)[None, :, None, None]
DFS = np.array([0.8, 0.9, 1.0])[None, None, None, :]
Z = np.linspace(-3.0, 3.0, 41)[None, None, :, None]


def main() -> None:
    sig, T, K, DF = np.broadcast_arrays(SIG, TT, F * np.exp(-Z * SIG * np.sqrt(TT)), DFS)
    s, x = sig * np.sqrt(T), np.log(F / K)
    otm = np.where(x > 0, -1.0, 1.0)
    rel = lambda a, b: np.abs(a / b - 1.0).max()  # noqa: E731
    P = bsm.price
    rows = []

    Kp = F * np.geomspace(0.25, 4.0, 61)[None, None, :, None]
    Fp, Kp, Tp, DFp, sp = np.broadcast_arrays(F, Kp, TT, DFS, SIG)
    par = P(Fp, Kp, Tp, DFp, sp, 1) - P(Fp, Kp, Tp, DFp, sp, -1) - DFp * (Fp - Kp)
    rows.append(("put-call parity, K/F in [0.25, 4]", np.abs(par).max(), 1e-12))

    for cp in (1, -1):
        V = P(F, K, T, DF, sig, cp)
        sh, info = implied_vol(V, F, K, T, DF, cp, full_output=True)
        e, o = np.abs(sh - sig), cp * x <= 0
        rows.append((f"IV round trip cp={cp:+d} (OTM inputs)", e[o].max(), 1e-10))
        rows.append((f"IV round trip cp={cp:+d} (ITM inputs)", e[~o].max(), 1e-10))
        n = info["n_iter"]
        print(
            f"IV cp={cp:+d}: {n.size} elements, Newton evaluations max {n.max()}, "
            f"mean {n.mean():.2f}, Brent fallback {int(info['brent'].sum())}"
        )

        h = 1e-5 * F * s
        fd = (P(F + h, K, T, DF, sig, cp) - P(F - h, K, T, DF, sig, cp)) / ((F + h) - (F - h))
        rows.append(
            (f"delta cp={cp:+d}, h = 1e-5 F s", rel(fd, bsm.delta(F, K, T, DF, sig, cp)), 1e-5)
        )
        up, dn = sig * (1 + 1e-5), sig * (1 - 1e-5)
        fd = (P(F, K, T, DF, up, cp) - P(F, K, T, DF, dn, cp)) / (up - dn)
        rows.append((f"vega cp={cp:+d}, h = 1e-5 sigma", rel(fd, bsm.vega(F, K, T, DF, sig)), 1e-5))

    G = bsm.gamma(F, K, T, DF, sig)
    for c in (4.5e-4, 1e-4):  # 4.5e-4: minimax step (D-014); 1e-4: §2.4 table, diagnostic only
        h = (F + c * F * s) - F
        f = [P(F + k * h, K, T, DF, sig, otm) for k in (-1, 0, 1)]
        err = rel((f[2] - 2 * f[1] + f[0]) / h**2, G)
        if c == 1e-4:
            print(f"diagnostic (not a gate): gamma OTM at the §2.4 step 1e-4 F s: {err:.3e}")
        else:
            rows.append((f"gamma OTM, h = {c:g} F s (D-014)", err, 1e-5))
    r, h = -np.log(DF) / T, (T + 1e-5 * T) - T
    fd = (
        P(F, K, T - h, DF * np.exp(r * h), sig, otm) - P(F, K, T + h, DF * np.exp(-r * h), sig, otm)
    ) / (2 * h)
    rows.append(
        ("theta OTM, joint (T, DF) bump (18)", rel(fd, bsm.theta(F, K, T, DF, sig, otm)), 1e-5)
    )

    print(f"\n{'gate':48s} {'measured':>10s} {'tol':>8s}  pass")
    for name, val, tol in rows:
        print(f"{name:48s} {val:10.3e} {tol:8.0e}  {val < tol}")


if __name__ == "__main__":
    main()
