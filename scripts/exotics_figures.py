"""Phase 4 figures: barrier MC vs Reiner-Rubinstein, and the Asian control-variate reduction.

1. report/figures/barrier_mc_vs_rr.{png,csv}: BGK (1997) Table 2.1 down-and-out call (S0 = 100,
   b = r = 10%, T = 0.2, sigma = 0.3, K = 100, H = 95). For m = 4..512 grid steps (dt = T/m):
   uncorrected MC (barrier checked at the grid dates only, i.e. the m-date discrete contract),
   Brownian-bridge MC on the same grid (continuous contract, eqs. 8-10), the BGK-shifted RR price
   (eq. 12) and the continuous RR price (eq. 3). Right panel: |uncorrected - RR| vs dt (slope 1/2,
   eq. 13) and |BGK - uncorrected MC|. barrier.md §5 (iv).
2. report/figures/asian_cv_vr.{png,csv}: variance-reduction factor VR = s_Y^2/s_e^2 and rho_hat of
   the geometric control variate (asian.md eqs. 14-15) vs K/F, monthly fixings (n = 12, T = 1,
   F_i = 100, DF = 1, BSM), sigma in {0.1, 0.2, 0.3, 0.5}, 2e5 paths.
Seeds are fixed (SEED + offset, recorded in the CSVs).
Usage: .venv/bin/python scripts/exotics_figures.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from qpricer.engines.analytic import barrier_price, bgk_barrier, geometric_asian_price  # noqa: E402
from qpricer.engines.exotic_mc import asian_averages, barrier_mc, control_variate  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "report" / "figures"
SEED, N = 20261508, 200_000
F, DF, T, B, SIG, K, H = 100 * np.exp(0.02), np.exp(-0.02), 0.2, 0.1, 0.3, 100.0, 95.0
MS = [4, 8, 16, 32, 64, 128, 256, 512]


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def barrier_figure() -> list[dict]:
    rr = float(barrier_price(F, K, H, T, DF, SIG, 1, 1, "out", b=B))
    rows = []
    for i, m in enumerate(MS):
        kw = dict(n_steps=m, n_paths=N, seed=SEED + i, b=B)
        disc, se_d, _, _ = barrier_mc(F, K, H, T, DF, SIG, 1, 1, "out", monitor=m, **kw)
        bb, se_b, _, _ = barrier_mc(F, K, H, T, DF, SIG, 1, 1, "out", **kw)
        bgk = float(
            barrier_price(F, K, bgk_barrier(H, SIG, T / m, 1), T, DF, SIG, 1, 1, "out", b=B)
        )
        row = {"m": m, "dt": T / m, "seed": SEED + i, "n_paths": N, "rr_continuous": rr}
        row |= {"mc_uncorrected": float(disc), "se_uncorrected": float(se_d)}
        rows.append(row | {"mc_bridge": float(bb), "se_bridge": float(se_b), "bgk_shifted_rr": bgk})
    write_csv(OUT / "barrier_mc_vs_rr.csv", rows)
    dt = np.array([r["dt"] for r in rows])
    get = lambda k: np.array([r[k] for r in rows])  # noqa: E731
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.axhline(rr, color="k", lw=1, label="RR continuous (eq. 3)")
    a1.errorbar(dt, get("mc_uncorrected"), 3 * get("se_uncorrected"), fmt="o-", capsize=3,
                label="MC uncorrected (grid monitoring = m-date discrete)")  # fmt: skip
    a1.errorbar(dt, get("mc_bridge"), 3 * get("se_bridge"), fmt="s-", capsize=3,
                label="MC Brownian bridge (continuous, eqs. 8-10)")  # fmt: skip
    a1.plot(
        dt, get("bgk_shifted_rr"), "^--", label="BGK: RR at H e^{-0.5826 sigma sqrt(dt)} (eq. 12)"
    )
    a1.set_xscale("log")
    a1.set_xlabel("dt = T/m")
    a1.set_ylabel("down-and-out call price")
    a1.legend(fontsize=7)
    a1.set_title(f"DOC: F={F:.2f}, K={K:g}, H={H:g}, sigma={SIG}, T={T}, b={B}; +-3 SE", fontsize=9)
    bias = np.abs(get("mc_uncorrected") - rr)
    a2.loglog(dt, bias, "o-", label="|MC uncorrected - RR continuous|")
    a2.loglog(dt, bias[0] * np.sqrt(dt / dt[0]), "k:", label="slope 1/2 (eq. 13)")
    a2.loglog(
        dt,
        np.abs(get("bgk_shifted_rr") - get("mc_uncorrected")),
        "^--",
        label="|BGK - MC uncorrected|",
    )
    a2.loglog(dt, 3 * get("se_uncorrected"), color="gray", lw=0.8, label="3 SE (noise floor)")
    a2.set_xlabel("dt = T/m")
    a2.legend(fontsize=7)
    slope = np.polyfit(np.log(dt), np.log(bias), 1)[0]
    s_bgk = np.polyfit(np.log(dt), np.log(get("bgk_shifted_rr") - rr), 1)[0]  # noise-free
    a2.set_title(f"monitoring bias: fitted slope MC {slope:.3f}, BGK-predicted {s_bgk:.3f} "
                 "(-> 0.5 as dt -> 0, eq. 13)", fontsize=9)  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "barrier_mc_vs_rr.png", dpi=130)
    print(f"barrier figure: RR = {rr:.6f}; slope of |uncorrected - RR| vs dt: MC {slope:.3f},")
    print(
        f"  BGK-predicted (noise-free) {s_bgk:.3f} on the same m (pre-asymptotic; 0.5 as m -> inf)"
    )
    m = 4 * 2 ** np.arange(19)  # local slopes of the noise-free BGK bias, m = 4 .. 2^20
    bias_bgk = barrier_price(F, K, bgk_barrier(H, SIG, T / m, 1), T, DF, SIG, 1, 1, "out", b=B) - rr
    loc = np.diff(np.log(bias_bgk)) / np.diff(np.log(T / m))
    print(f"  local BGK slopes, m = 4 -> 8, ..., 2^19 -> 2^20: {np.round(loc, 3).tolist()}")
    for r in rows:
        zb = (r["mc_bridge"] - rr) / r["se_bridge"]
        zg = (r["bgk_shifted_rr"] - r["mc_uncorrected"]) / r["se_uncorrected"]
        print(
            f"  m={r['m']:4d}: uncorrected {r['mc_uncorrected']:.5f} ({r['se_uncorrected']:.5f})",
            end="",
        )
        print(f", bridge z = {zb:+.2f}, BGK - uncorrected MC = {zg:+.2f} SE")
    return rows


def asian_figure() -> list[dict]:
    t, Kf = np.arange(1, 13) / 12, np.linspace(0.8, 1.2, 9)
    rows = []
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for j, s in enumerate((0.1, 0.2, 0.3, 0.5)):
        A, G = asian_averages(t, np.full(12, 100.0), s, N, np.random.default_rng(SEED + 100 + j))
        Y, X = np.maximum(A - 100 * Kf[:, None], 0), np.maximum(G - 100 * Kf[:, None], 0)
        mean_x = geometric_asian_price(np.full(12, 100.0), s**2 * t, 100 * Kf, 1.0, 1)
        est, se, c, vr = control_variate(Y, X, mean_x)
        rho = np.array([np.corrcoef(y, x)[0, 1] for y, x in zip(Y, X, strict=True)])
        for k in range(Kf.size):
            row = {"sigma": s, "K_over_F": Kf[k], "n_fix": 12, "T": 1.0, "seed": SEED + 100 + j}
            row |= {"n_paths": N, "price_cv": est[k], "se_cv": se[k]}
            row |= {"se_plain": Y[k].std(ddof=1) / np.sqrt(N), "c_hat": c[k], "rho_hat": rho[k]}
            rows.append(row | {"VR": vr[k]})
        a1.semilogy(Kf, vr, "o-", label=f"sigma = {s}")
        a2.plot(Kf, rho, "o-", label=f"sigma = {s}")
    write_csv(OUT / "asian_cv_vr.csv", rows)
    a1.set_xlabel("K / F")
    a1.set_ylabel("VR = Var(Y) / Var(Y - c X)")
    a1.set_title(
        "geometric control variate on the arithmetic Asian call (n = 12, T = 1)", fontsize=9
    )
    a1.legend(fontsize=8)
    a2.set_xlabel("K / F")
    a2.set_ylabel("rho_hat(Y, X)")
    a2.set_title("VR = 1 / (1 - rho^2), eq. (15)", fontsize=9)
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "asian_cv_vr.png", dpi=130)
    atm = [r for r in rows if abs(r["K_over_F"] - 1.0) < 1e-12]
    print("asian figure, ATM: " + ", ".join(f"sigma {r['sigma']}: VR {r['VR']:.0f}" for r in atm))
    return rows


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    barrier_figure()
    asian_figure()
