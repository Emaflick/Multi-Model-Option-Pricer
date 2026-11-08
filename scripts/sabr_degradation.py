"""Phase 3 gate figure: where the Hagan (beta = 1) approximation degrades (long T / wings).

Recipe of docs/derivations/sabr.md §5 ("Figure where Hagan degrades"): alpha = 0.2, rho = -0.75,
nu in {0.4, 1.0}, T in {0.25, 1, 5, 10} (nu^2 T from 0.04 to 10), 41 strikes on
k / (alpha sqrt T) in [-3, 2], F = 100, DF = 1. Reference: conditional (mixing) MC, eq. (17),
N = 2e5, dt = 1/100 (trapezoid, bias O(dt^2)); log-Euler (14) at the gate dt = T/250 overlaid for
T <= 1. Mixing MC prices OTM options (puts k <= 0, calls k > 0): for rho <= 0 the conditional
forward F xi <= F e^{|rho| alpha / nu} is bounded, so the conditional call is a bounded payoff
(D-023); log-Euler prices puts at every strike (pitfall 8). The MC vol is IV(V_MC) with
SE_sigma = SE_V / vega (eq. 19, delta method). Top row: sigma_H - sigma_MC in vol points with
+-3 SE_sigma bars and the gate box (T <= 1, |k| <= 0.2). Bottom row: Hagan-implied density of
ln(F_T/F), K p(K), from second differences of Black(sigma_H) with step h = 1e-3 F
(Breeden-Litzenberger); p < 0 shaded.
Seeds are fixed per panel and recorded in the CSV.

Outputs report/figures/sabr_hagan_vs_mc.{png,csv} and sabr_hagan_vs_mc_summary.csv (one row per
panel). Summary columns *_diff_*_incl_mc_noise are sigma_H - sigma_MC of one seed (MC noise
included, not a Hagan error); the Hagan error resolved by the MC is max_abs_err_volpts_significant
(points with |z| > 3 in price space).
Usage: .venv/bin/python scripts/sabr_degradation.py [--paths 200000]
"""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from qpricer import iv  # noqa: E402
from qpricer.models import bsm, sabr  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "report" / "figures"
SEED = 20261107
F, DF, ALPHA, RHO = 100.0, 1.0, 0.2, -0.75
NUS, TS = (0.4, 1.0), (0.25, 1.0, 5.0, 10.0)
STEPS_PER_YEAR_MIXING, STEPS_EULER = 100, 250  # (17): dt = 1/100; (14): gate dt = T/250 (D-023)
COLORS = {0.4: "tab:blue", 1.0: "tab:red"}


def mc_vols(K, T, nu, method, seed, n_paths, cp):
    """MC prices -> (price, SE, sigma_MC, SE_sigma, IV status, n_steps); NaN where IV fails."""
    m = STEPS_EULER if method == "euler" else int(np.ceil(STEPS_PER_YEAR_MIXING * T - 1e-9))
    p, se, _, _ = sabr.mc_price(
        F, K, T, DF, ALPHA, RHO, nu, cp, n_steps=m, n_paths=n_paths, seed=seed, method=method
    )
    with warnings.catch_warnings():  # failed inversions are kept as NaN with their status code
        warnings.simplefilter("ignore", RuntimeWarning)
        sig, info = iv.implied_vol(p, F, K, T, DF, cp, full_output=True)
    ok = info["status"] == 0
    vega = bsm.vega(F, K, T, DF, np.where(ok, sig, ALPHA))
    return p, se, sig, np.where(ok, se / vega, np.nan), info["status"], m


def hagan_density(k_lo, k_hi, T, nu, n=801):
    """K p(K) on a strike grid, p = d^2 C / dK^2 by central differences with h = 1e-3 F."""
    K, h = F * np.exp(np.linspace(k_lo, k_hi, n)), 1e-3 * F
    Ks = K[:, None] + h * np.array([-1.0, 0.0, 1.0])
    C = bsm.price(F, Ks, T, DF, sabr.implied_vol(F, Ks, T, ALPHA, RHO, nu), 1)
    return np.log(K / F), K * (C[:, 0] - 2 * C[:, 1] + C[:, 2]) / (h * h * DF)


def run(n_paths: int) -> tuple[list[dict], list[dict], dict]:
    rows, summary, dens = [], [], {}
    for i, nu in enumerate(NUS):
        for j, T in enumerate(TS):
            k = ALPHA * np.sqrt(T) * np.linspace(-3.0, 2.0, 41)
            K = F * np.exp(k)
            s_h = sabr.implied_vol(F, K, T, ALPHA, RHO, nu)
            for method in ("mixing", "euler") if T <= 1 else ("mixing",):
                seed = SEED + 100 * i + 10 * j + (method == "euler")
                cp = np.where((k > 0) & (method == "mixing"), 1.0, -1.0)
                p_h = bsm.price(F, K, T, DF, s_h, cp)
                p, se, sig, se_s, st, m = mc_vols(K, T, nu, method, seed, n_paths, cp)
                for q in range(K.size):
                    rows.append(
                        dict(
                            nu=nu,
                            T=T,
                            nu2T=nu * nu * T,
                            method=method,
                            k=k[q],
                            K=K[q],
                            sigma_hagan=s_h[q],
                            cp=int(cp[q]),
                            price_hagan=p_h[q],
                            price_mc=p[q],
                            se_price=se[q],
                            z_price=(p[q] - p_h[q]) / se[q],
                            sigma_mc=sig[q],
                            se_sigma=se_s[q],
                            err_volpts=100 * (s_h[q] - sig[q]),
                            iv_status=int(st[q]),
                            n_paths=n_paths,
                            n_steps=m,
                            seed=seed,
                        )  # fmt: skip
                    )
            kd, pd = hagan_density(k[0], k[-1], T, nu)
            dens[(nu, T)] = (kd, pd)
            r = [q for q in rows if q["nu"] == nu and q["T"] == T and q["method"] == "mixing"]
            err = np.array([q["err_volpts"] for q in r])
            z = np.array([q["z_price"] for q in r])
            box = np.abs(k) <= 0.2
            q = int(np.nanargmax(np.abs(err)))
            sig = np.abs(z) > 3  # Hagan error resolved by the MC (price-space z)
            qs = int(np.nanargmax(np.where(sig, np.abs(err), -1.0))) if sig.any() else None
            neg = kd[pd < 0]
            summary.append(
                dict(
                    nu=nu,
                    T=T,
                    nu2T=nu * nu * T,
                    max_abs_diff_volpts_incl_mc_noise=abs(err[q]),
                    k_at_max_diff=k[q],
                    se_volpts_at_max_diff=100 * r[q]["se_sigma"],
                    max_abs_z_price=np.nanmax(np.abs(z)),
                    max_abs_err_volpts_significant=abs(err[qs]) if qs is not None else 0.0,
                    k_at_max_significant=k[qs] if qs is not None else np.nan,
                    gate_box_max_abs_diff_volpts_incl_mc_noise=(
                        np.nanmax(np.abs(err[box])) if T <= 1 else np.nan
                    ),
                    gate_box_max_abs_z_price=np.nanmax(np.abs(z[box])) if T <= 1 else np.nan,
                    n_iv_failed=int(np.sum(np.isnan(err))),
                    min_hagan_density=pd.min(),
                    neg_density_k_min=neg.min() if neg.size else np.nan,
                    neg_density_k_max=neg.max() if neg.size else np.nan,
                    seed=r[0]["seed"],
                )  # fmt: skip
            )
            s = summary[-1]
            print(
                f"nu={nu} T={T:<5g} nu2T={nu * nu * T:<5.2f} "
                f"max|diff|={s['max_abs_diff_volpts_incl_mc_noise']:.3f} vol pt "
                f"at k={s['k_at_max_diff']:+.3f} (3SE={3 * s['se_volpts_at_max_diff']:.3f}); "
                f"max|z|={s['max_abs_z_price']:.1f}; max |err| with |z|>3: "
                f"{s['max_abs_err_volpts_significant']:.3f}; "
                f"min K p(K)={s['min_hagan_density']:+.2e}"
            )
    return rows, summary, dens


def plot(rows: list[dict], dens: dict, n_paths: int) -> None:
    fig, axes = plt.subplots(2, len(TS), figsize=(17, 7.8), constrained_layout=True)
    for j, T in enumerate(TS):
        top, bot = axes[0, j], axes[1, j]
        if T <= 1:
            top.axvspan(-0.2, 0.2, color="tab:green", alpha=0.1, label="gate box |k| <= 0.2")
        lims = []
        for nu in NUS:
            c = COLORS[nu]
            for method, mk, mfc in (("mixing", "o", c), ("euler", "s", "none")):
                r = [q for q in rows if q["nu"] == nu and q["T"] == T and q["method"] == method]
                if not r:
                    continue
                k, e = np.array([q["k"] for q in r]), np.array([q["err_volpts"] for q in r])
                bar = 300 * np.array([q["se_sigma"] for q in r])
                lab = f"nu={nu}, " + (
                    "mixing MC (17)" if method == "mixing" else "log-Euler (14), puts"
                )
                if method == "mixing":
                    lims += [np.nanmin(e - bar), np.nanmax(e + bar)]
                faint = method == "euler"  # overlay: faint bars, ITM-put SEs are large
                top.errorbar(k, e, yerr=bar, fmt=mk, ms=3.5, mfc=mfc, color=c, lw=0.8,
                             capsize=0 if faint else 2, elinewidth=0.5 if faint else 0.8,
                             alpha=0.45 if faint else 1.0, label=lab)  # fmt: skip
            kd, pd = dens[(nu, T)]
            bot.plot(kd, pd, color=c, lw=1.2, label=f"nu={nu} (nu^2 T={nu * nu * T:g})")
            bot.fill_between(kd, pd, 0, where=pd < 0, color=c, alpha=0.35, lw=0)
            if np.any(pd < 0):
                kn = kd[pd < 0]
                bot.axvspan(kn.min(), kn.max(), color=c, alpha=0.08)
        pad = 0.08 * (max(lims) - min(lims))  # y-range from the mixing points (Euler ITM-put bars
        top.set_ylim(min(lims) - pad, max(lims) + pad)  # in the right wing run off-axis; CSV)
        top.axhline(0, color="k", lw=0.6)
        top.set_title(f"T = {T:g} y")
        top.set_xlabel("k = ln(K/F)")
        top.set_ylabel("sigma_Hagan - sigma_MC (vol pts)")
        top.grid(alpha=0.3)
        top.legend(fontsize=7, loc="best")
        bot.axhline(0, color="k", lw=0.6)
        bot.set_yscale("symlog", linthresh=1e-2)
        bot.set_xlabel("k = ln(K/F)")
        bot.set_ylabel("Hagan density of ln(F_T/F), K p(K) (symlog)")
        bot.grid(alpha=0.3)
        bot.legend(fontsize=7, loc="upper right", title="shaded: p(K) < 0", title_fontsize=7)
    fig.suptitle(
        f"SABR beta = 1 (alpha = {ALPHA}, rho = {RHO}): Hagan (2002) vs MC, OTM options, "
        f"N = {n_paths:.0e}, bars = +-3 SE_sigma (SE_V / vega); strikes k/(alpha sqrt T) in [-3, 2]"
    )
    fig.savefig(OUT / "sabr_hagan_vs_mc.png", dpi=150)


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paths", type=int, default=200_000)
    n_paths = ap.parse_args().paths
    OUT.mkdir(parents=True, exist_ok=True)
    rows, summary, dens = run(n_paths)
    write_csv(OUT / "sabr_hagan_vs_mc.csv", rows)
    write_csv(OUT / "sabr_hagan_vs_mc_summary.csv", summary)
    plot(rows, dens, n_paths)


if __name__ == "__main__":
    main()
