"""Phase 2 gate figure: Heston MC bias vs time step, Euler-FT / QE / QE-M, Feller-violating sets.

Bias = MC - COS for an ATM call (K = F = 100, DF = 1), with +-3 SE error bars (heston-mc.md §5,
"Bias figure"). Sets: A-I (T = 10, nu_F = 0.04; Andersen 2006 Table 1) and FO (T = 1,
nu_F = 0.38). FO starts at dt = 1/2 because Euler needs kappa dt < 1 (pitfall 2). COS references
use a widened range (D-017) and are checked against Lewis. Each (set, scheme, dt) point has its
own fixed seed, recorded in the CSV together with the forward diagnostic mean F_T - F (§5).

Outputs report/figures/heston_euler_bias.png and .csv.
Usage: .venv/bin/python scripts/heston_euler_bias.py [--paths 1000000]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from qpricer.engines import mc  # noqa: E402
from qpricer.engines.fourier import cos_price, lewis_price  # noqa: E402
from qpricer.models.heston import HestonParams  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "report" / "figures"
SEED = 20261107
SETS = {  # name: (params, T, dts, COS N, COS L)
    "A-I": (HestonParams(0.5, 0.04, 1.0, -0.9, 0.04), 10.0, [2.0**-j for j in range(7)], 4096, 40),
    "FO": (
        HestonParams(1.5768, 0.0398, 0.5751, -0.5711, 0.0175),
        1.0,
        [2.0**-j for j in range(1, 7)],
        512,
        24,
    ),
}
SCHEMES = {"euler": "Euler full truncation", "qe": "QE (psi_c = 1.5)", "qe-m": "QE-M (martingale)"}
STYLE = {"euler": ("tab:red", "o"), "qe": ("tab:blue", "s"), "qe-m": ("tab:green", "^")}


def run(n_paths: int) -> list[dict]:
    rows = []
    for i, (name, (p, T, dts, N, L)) in enumerate(SETS.items()):
        ref = float(cos_price(100.0, 100.0, T, 1.0, p, 1, N=N, L=L))
        lew = float(lewis_price(100.0, 100.0, T, 1.0, p, 1))
        print(
            f"{name}: COS ref {ref:.8f}, Lewis {lew:.8f}, diff {ref - lew:.1e}, nu_F {p.feller:.3f}"
        )
        for j, scheme in enumerate(SCHEMES):
            for k, dt in enumerate(dts):
                seed = SEED + 1000 * i + 10 * j + k
                M = int(round(T / dt))
                x, _ = mc.simulate(
                    p, mc.uniform_grid(T, M), n_paths, np.random.default_rng(seed), scheme
                )
                FT = 100.0 * np.exp(x[0])
                price, se = mc.mc_estimate(np.maximum(FT - 100.0, 0.0))
                fwd, fse = mc.mc_estimate(FT)
                row = dict(
                    set=name,
                    scheme=scheme,
                    T=T,
                    K=100.0,
                    dt=dt,
                    n_steps=M,
                    n_paths=n_paths,
                    seed=seed,
                    mc_price=float(price),
                    stderr=float(se),
                    cos_ref=ref,
                    bias=float(price - ref),
                    bias_over_se=float((price - ref) / se),
                    mean_FT_minus_F=float(fwd - 100.0),
                    se_FT=float(fse),
                )
                rows.append(row)
                z_fwd = row["mean_FT_minus_F"] / fse
                print(
                    f"  {scheme:6s} dt={dt:<9.6g} bias={row['bias']:+.5f} "
                    f"({row['bias_over_se']:+6.2f} SE)  F_T-F={row['mean_FT_minus_F']:+.4f} "
                    f"({z_fwd:+.2f} SE)"
                )
    return rows


def euler_slope(rows: list[dict], name: str) -> float:
    """Least-squares slope of log|bias| on log dt over Euler points with |bias| > 3 SE."""
    pts = [
        (r["dt"], abs(r["bias"]))
        for r in rows
        if r["set"] == name and r["scheme"] == "euler" and abs(r["bias_over_se"]) > 3
    ]
    if len(pts) < 2:
        return float("nan")
    x, y = np.log(np.array(pts)).T
    return float(np.polyfit(x, y, 1)[0])


def plot(rows: list[dict], n_paths: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for ax, name in zip(axes, SETS, strict=True):
        p, T = SETS[name][0], SETS[name][1]
        for scheme, label in SCHEMES.items():
            r = [q for q in rows if q["set"] == name and q["scheme"] == scheme]
            dt = np.array([q["dt"] for q in r])
            b = np.abs([q["bias"] for q in r])
            se3 = 3 * np.array([q["stderr"] for q in r])
            sig = b > se3
            color, mk = STYLE[scheme]
            lo = np.where(b - se3 > 0, se3, b * (1 - 1e-9))  # lower bar stops at the axis floor
            ax.errorbar(dt, b, yerr=[lo, se3], color=color, lw=1, capsize=3, ls="-", marker="none")
            ax.plot(dt[sig], b[sig], mk, color=color, label=label)
            ax.plot(dt[~sig], b[~sig], mk, mfc="none", color=color)
        noise = 3 * np.median([q["stderr"] for q in rows if q["set"] == name])
        ax.axhline(
            noise, color="grey", ls=":", lw=1, label=f"3 SE noise level ({n_paths:.0e} paths)"
        )
        s = euler_slope(rows, name)
        e = [q for q in rows if q["set"] == name and q["scheme"] == "euler"]
        d0, b0 = e[len(e) // 2]["dt"], abs(e[len(e) // 2]["bias"])
        grid = np.array([min(q["dt"] for q in e), max(q["dt"] for q in e)])
        ax.plot(
            grid,
            b0 * grid / d0,
            color="tab:red",
            ls="--",
            lw=0.8,
            label=f"order 1 guide (Euler fitted slope {s:.2f})",
        )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        floor = min(noise / 20, 0.5 * min(abs(q["bias"]) for q in rows if q["set"] == name))
        ax.set_ylim(bottom=floor)  # bars of insignificant points run to the floor (~0)
        ax.set_xlabel("time step dt (years)")
        ax.set_ylabel("|MC - COS|, ATM call (F = K = 100, DF = 1)")
        ax.set_title(f"{name}: T = {T:g}, Feller ratio {p.feller:.2f} (< 1)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Heston MC bias vs time step; bars = +-3 SE, hollow = not significant (< 3 SE)")
    fig.savefig(OUT / "heston_euler_bias.png", dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paths", type=int, default=1_000_000)
    n_paths = ap.parse_args().paths
    OUT.mkdir(parents=True, exist_ok=True)
    rows = run(n_paths)
    with open(OUT / "heston_euler_bias.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot(rows, n_paths)
    for name in SETS:
        e = [r for r in rows if r["set"] == name and r["scheme"] == "euler"]
        mono = all(
            abs(a["bias"]) > abs(b["bias"]) or abs(b["bias_over_se"]) <= 3
            for a, b in zip(e, e[1:], strict=False)
        )
        print(
            f"{name}: Euler bias > 3SE at coarsest dt: {abs(e[0]['bias_over_se']) > 3}; "
            f"decreasing until inside noise: {mono}; fitted slope {euler_slope(rows, name):.2f}"
        )


if __name__ == "__main__":
    main()
