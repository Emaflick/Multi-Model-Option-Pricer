"""Autocallable under BSM (ATM vol) vs Heston, with the attribution of autocallable.md eq. (19).

Term sheets (D-026): AC-Q quarterly t_i = i/4, n = 6, T = 1.5 y (inside the last kept expiry,
1.61 y) and AC-A annual t_i = 1, 2, 3 (carry and Heston extrapolated beyond 1.61 y); both with
B_ac = 1, B_c = 0.7, B_p = 0.6, memory coupon, c = 8% p.a. F_0(t_i), DF(t_i) come from
data/processed/market_20261106_2213.csv via market.forward_curve; S_ref = F_0(T_(1)) DF(T_(1))
(D-025). V_BS, V_TS and V_H share one common-random-number path set (eqs. 14-16); the no-call
values V^nc are exact (eq. 11; Black, or Heston COS) and also re-priced on the paths (3 SE check).

Default Heston parameters (D-029): GS = Gatheral (2006) Table 3.2 SPX fit (kappa, theta, xi, rho)
with v0 := theta (our choice), as in heston.md §5. They are NOT fitted to this snapshot: every
output made with them is labelled "illustrative, pre-calibration". Phase 5 re-runs this script
with --heston-params <fitted.json> (keys kappa, theta, xi, rho, v0, optional label); only those
numbers go into the report. Outputs: report/figures/autocall_attribution_<tag>.{png,csv}.
Usage: .venv/bin/python scripts/autocall_compare.py [--heston-params p.json] [--paths 200000]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from qpricer.engines.analytic import autocall_nocall
from qpricer.engines.exotic_mc import autocall_pv, fixings, gbm_paths
from qpricer.engines.fourier import cos_price
from qpricer.iv import implied_vol
from qpricer.market import forward_curve
from qpricer.models.heston import HestonParams, total_variance
from qpricer.products import Autocall, autocall_flows

ROOT = Path(__file__).resolve().parents[1]
MARKET = ROOT / "data" / "processed" / "market_20261106_2213.csv"
OUT = ROOT / "report" / "figures"
SEED, N_PATHS, MAX_DT = 20261308, 200_000, 1 / 32
GS = HestonParams(kappa=1.3253, theta=0.0354, xi=0.3877, rho=-0.7165, v0=0.0354)
GS_LABEL = "illustrative, pre-calibration (GS: Gatheral 2006 Table 3.2 SPX fit, v0 := theta)"
TERMS = {  # eq. (19) labels; with v0 = theta, w(t) = theta t and term1 is a pure level gap (D-031)
    "term1": "flat sigma_ATM vs Heston expected-variance curve (level + term structure)",
    "term2": "marginal smile at B_p, B_c (exact)",
    "term3": "path: forward skew, vol-of-vol",
}
SHEETS = {
    "AC-Q": Autocall(tuple(np.arange(1, 7) / 4), 0.08, 1.0, 0.7, 0.6, True),
    "AC-A": Autocall((1.0, 2.0, 3.0), 0.08, 1.0, 0.7, 0.6, True),
}


def market_inputs(sheet: Autocall, path: Path = MARKET) -> tuple[np.ndarray, np.ndarray, float]:
    """(f_i, DF_i, S_ref): f_i = F_0(t_i)/S_ref, S_ref = F_0(T_(1)) DF(T_(1)) (autocallable §1)."""
    tab = pd.read_csv(path)
    first = tab[tab.kept.astype(bool)].sort_values("T").iloc[0]
    s_ref = float(first.F * first.DF)
    F0, DF = forward_curve(tab, sheet.t)
    if np.any(np.diff(DF) > 0):
        raise ValueError(f"interpolated DF_i not non-increasing: {DF} (pitfall 5)")
    return F0 / s_ref, DF, s_ref


def heston_iv(p: HestonParams, f: float, k: float, T: float) -> float:
    """Heston implied vol at ratio strike k, forward f, expiry T (COS call, iv.implied_vol)."""
    return float(implied_vol(cos_price(f, k, T, 1.0, p, 1), f, k, T, 1.0, 1))


def market_atm_vol(T: float) -> float:
    """SPX ATM vol at T: per bracketing expiry, the OTM implied vols at the two strikes next to F
    interpolated linearly in k to k = 0; total variance linear in T. NaN if the raw snapshot is
    missing or T lies beyond the last kept expiry."""
    from qpricer.data import build_surface

    logging.getLogger("qpricer").setLevel(logging.ERROR)  # Phase 0 drop log, already in D-004
    try:
        surf = build_surface()["surface"]
    except (FileNotFoundError, ValueError):
        return float("nan")
    rows = []
    for _, g in surf.groupby("expiry"):
        lo, hi = g[g.k < 0].nlargest(1, "k"), g[g.k >= 0].nsmallest(1, "k")
        if len(lo) and len(hi):
            q = pd.concat([lo, hi])
            cp = np.where(q.option_type == "C", 1, -1)
            cols = (q.mid, q.F, q.strike, q["T"], q.DF)
            s, k = implied_vol(*(c.to_numpy() for c in cols), cp), q.k.to_numpy()
            rows.append((float(q["T"].iloc[0]), np.interp(0.0, k, s)))
    Ts, s0 = np.array(rows).T
    j = np.searchsorted(Ts, T)
    if not 0 < j < Ts.size:
        return float("nan")
    w = np.interp(T, Ts[j - 1 : j + 1], (s0[j - 1 : j + 1] ** 2) * Ts[j - 1 : j + 1])
    return float(np.sqrt(w / T))


def _stats(sheet, X, DF):
    """Per-path a_j (c = 0) and b_j (coupon PV per unit c), first-call probabilities, A_n."""
    a = autocall_pv(X, replace(sheet, c=0.0), DF)
    b = autocall_pv(X, replace(sheet, c=1.0), DF) - a
    _, _, chi, alive = autocall_flows(X, sheet)
    return a, b, chi.mean(1), alive.mean()


def attribution(p: HestonParams, sheet: Autocall, f, DF, n_paths=N_PATHS, seed=SEED, max_dt=MAX_DT):
    """Eq. (19) on one CRN path set; returns a dict of values, CRN standard errors and diagnostics.

    V_BS: flat sigma_ATM = Heston IV at (K = F_0(T), T); V_TS: Black with Heston's expected
    variance w(t) (heston.md eq. 6); V_H: Heston QE-M. term1 = V_TS - V_BS (flat sigma_ATM vs the
    Heston expected-variance curve: level + term structure; pure level when v0 = theta),
    term2 = V^nc_H - V^nc_TS (marginal smile at B_p, B_c; exact), term3 = (V_H - V_TS) - term2
    (path: forward skew, vol-of-vol). SEs from per-path differences (eq. 16); par coupons (17).
    """
    t, T, N = np.asarray(sheet.t), sheet.t[-1], int(n_paths)
    xH, z = fixings(p, t, N, np.random.default_rng(seed), max_dt)
    sig = heston_iv(p, f[-1], f[-1], T)
    var = {"BS": sig**2 * t, "TS": total_variance(t, p)}
    X = {k: f[:, None] * np.exp(gbm_paths(w, z)) for k, w in var.items()}
    X["H"] = f[:, None] * np.exp(xH)
    pv = {k: autocall_pv(x, sheet, DF) for k, x in X.items()}
    nc_sheet = replace(sheet, B_ac=np.inf, memory=False)
    nc_mc = {k: autocall_pv(x, nc_sheet, DF) for k, x in X.items()}
    nc = {"H": autocall_nocall(sheet, f, DF, p), "TS": autocall_nocall(sheet, f, DF, var["TS"])}
    nc["BS"] = autocall_nocall(sheet, f, DF, var["BS"])
    se = lambda d: float(np.std(d, ddof=1) / np.sqrt(N))  # noqa: E731
    V = {k: float(v.mean()) for k, v in pv.items()}
    out = {"sigma_atm": sig, "sqrt_wT_over_T": float(np.sqrt(var["TS"][-1] / T))}
    out["ev_curve_flat"] = bool(p.v0 == p.theta)  # w(t) = theta t: term1 is a pure level gap
    out |= {f"V_{k}": V[k] for k in V} | {f"se_V_{k}": se(pv[k]) for k in V}
    out |= {"gap": V["H"] - V["BS"], "se_gap": se(pv["H"] - pv["BS"])}
    out |= {"term1": V["TS"] - V["BS"], "se_term1": se(pv["TS"] - pv["BS"])}
    out |= {"term2": nc["H"] - nc["TS"], "se_term2": 0.0}
    out |= {"term3": V["H"] - V["TS"] - (nc["H"] - nc["TS"]), "se_term3": se(pv["H"] - pv["TS"])}
    for k in V:
        out |= {
            f"nc_{k}": nc[k],
            f"nc_mc_{k}": float(nc_mc[k].mean()),
            f"se_nc_mc_{k}": se(nc_mc[k]),
        }
    psi = []
    for k in ("H", "BS"):
        a, b, pcall, alive = _stats(sheet, X[k], DF)
        c = (1.0 - a.mean()) / b.mean()
        psi.append((a + c * b) / b.mean())
        out |= {f"par_c_{k}": c, f"se_par_c_{k}": se(a + c * b) / b.mean()}
        out |= {f"p_call_{k}": pcall[:-1], f"alive_T_{k}": alive}
        out[f"life_{k}"] = float(np.sum(t[:-1] * pcall[:-1]) + T * alive)
    out |= {"par_gap": out["par_c_H"] - out["par_c_BS"], "se_par_gap": se(psi[0] - psi[1])}
    for name, k in (("S_ref", 1.0), ("B_p", sheet.B_p)):  # sensitivity rows (§2.5, pitfall 10)
        s_k = heston_iv(p, f[-1], k, T)
        out[f"sigma_{name}"] = s_k
        Xk = f[:, None] * np.exp(gbm_paths(s_k**2 * t, z))
        out[f"V_BS_at_{name}"] = float(autocall_pv(Xk, sheet, DF).mean())
    out["_z"] = z  # kept for the market-vol row (not printed)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--heston-params", type=Path, default=None, help="JSON with fitted params")
    ap.add_argument("--paths", type=int, default=N_PATHS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)
    if args.heston_params is None:
        p, label, tag = GS, GS_LABEL, "illustrative"
    else:
        d = json.loads(args.heston_params.read_text())
        p = HestonParams(**{k: float(d[k]) for k in ("kappa", "theta", "xi", "rho", "v0")})
        label = d.get("label", f"Heston from {args.heston_params.name}")
        tag = args.heston_params.stem
    print(f"Heston: {p}\nLABEL: {label}\nN = {args.paths}, seed = {args.seed}, QE-M dt <= 1/32\n")
    rows, raw = [], []
    for name, sheet in SHEETS.items():
        f, DF, s_ref = market_inputs(sheet)
        r = attribution(p, sheet, f, DF, args.paths, args.seed)
        s_mkt = market_atm_vol(sheet.t[-1])
        v_mkt = np.nan
        if np.isfinite(s_mkt):
            x = gbm_paths(s_mkt**2 * np.asarray(sheet.t), r["_z"])
            v_mkt = float(autocall_pv(f[:, None] * np.exp(x), sheet, DF).mean())
        report(name, sheet, r, s_ref, f, DF, s_mkt, v_mkt, label)
        sigma_grid(sheet, f, DF, r["_z"])
        keep = {csv_name(k): v for k, v in r.items() if not k.startswith(("_", "p_call"))}
        rows.append({"sheet": name, "label": label, "bs_leg": bs_leg(r, label), **keep})
        rows[-1] |= {"sigma_mkt_atm": s_mkt, "V_BS_at_mkt_atm": v_mkt}
        rows[-1]["gap_H_minus_BS_at_mkt_atm"] = r["V_H"] - v_mkt
        raw.append({"sheet": name, **r})
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"autocall_attribution_{tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    figure(raw, label, OUT / f"autocall_attribution_{tag}.png")
    print(f"wrote {OUT / f'autocall_attribution_{tag}.csv'} and .png")
    return 0


def bs_leg(r: dict, label: str) -> str:
    """Label of the BSM leg (D-031, D-038): flat vol = the Heston set's own ATM implied vol."""
    who = "calibrated-Heston" if label.startswith("calibrated") else "illustrative-Heston"
    return f"BSM at the {who} ATM vol ({100 * r['sigma_atm']:.2f}%)"


def csv_name(key: str) -> str:
    """CSV header: eq. (19) term keys get their descriptive labels (TERMS)."""
    base = key.removeprefix("se_")
    return key if base not in TERMS else f"{key}: {TERMS[base]}"


def sigma_grid(sheet, f, DF, z, sigmas=(0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)) -> list:
    """Diagnostic (autocallable.md §3.4, §5; not a gate): BS value of the full sheet on one set of
    common normals for a grid of flat vols; SE of each step from the per-path differences."""
    t, N = np.asarray(sheet.t), z.shape[1]
    pv = [autocall_pv(f[:, None] * np.exp(gbm_paths(s**2 * t, z)), sheet, DF) for s in sigmas]
    print("   CRN sigma-grid (BS, full product; diagnostic):", end="")
    out = []
    for i, s in enumerate(sigmas):
        se = np.nan if i == 0 else float(np.std(pv[i] - pv[i - 1], ddof=1) / np.sqrt(N))
        out.append((s, float(pv[i].mean()), se))
        print(f" {s:.2f}: {pv[i].mean():.5f}" + ("" if i == 0 else f" (dV SE {se:.5f})"), end=";")
    print()
    return out


def report(name, sheet, r, s_ref, f, DF, s_mkt, v_mkt, label: str = "") -> None:
    """Print one sheet's block: inputs, values, eq. (19) terms, par coupons, call probabilities."""
    ext = " (beyond last expiry 1.61 y: carry, Heston extrapolated)" if sheet.t[-1] > 1.61 else ""
    print(f"== {name}: t = {np.round(sheet.t, 4).tolist()}{ext}")
    print(f"   S_ref = {s_ref:.4f}, f_i = {np.round(f, 5).tolist()}")
    print(f"   DF_i = {np.round(DF, 5).tolist()}")
    print(f"   sigma_ATM (Heston IV at F_0(T)) = {r['sigma_atm']:.5f}", end="")
    print(f"; sqrt(w(T)/T) = {r['sqrt_wT_over_T']:.5f}")
    print(f"   BS leg: {bs_leg(r, label)}")
    for k in ("BS", "TS", "H"):
        print(f"   V_{k:<2} = {r[f'V_{k}']:.6f} (SE {r[f'se_V_{k}']:.6f});", end="")
        print(f" V^nc_{k}: exact {r[f'nc_{k}']:.6f}, MC {r[f'nc_mc_{k}']:.6f}", end="")
        print(f" (SE {r[f'se_nc_mc_{k}']:.6f})")
    print(f"   gap V_H - V_BS = {r['gap']:+.6f} (CRN SE {r['se_gap']:.6f})")
    print(f"     term1 {TERMS['term1']}: {r['term1']:+.6f} (CRN SE {r['se_term1']:.6f})")
    if r["ev_curve_flat"]:
        print(
            "           (v0 = theta: w(t) = theta t is flat, so term1 is a pure level gap,", end=""
        )
        print(f" sqrt(theta) {r['sqrt_wT_over_T']:.4f} vs sigma_ATM {r['sigma_atm']:.4f})")
    print(f"     term2 {TERMS['term2']}: {r['term2']:+.6f}")
    print(f"     term3 {TERMS['term3']}: {r['term3']:+.6f} (CRN SE {r['se_term3']:.6f})")
    for k in ("BS", "H"):
        print(f"   par coupon {k:<2} {r[f'par_c_{k}']:.5f} (SE {r[f'se_par_c_{k}']:.5f});", end="")
        print(f" first-call p_i {np.round(r[f'p_call_{k}'], 4).tolist()},", end="")
        print(f" not called {r[f'alive_T_{k}']:.4f}, life {r[f'life_{k}']:.4f} y")
    print(f"   par coupon H - BS {r['par_gap']:+.5f} (CRN SE {r['se_par_gap']:.5f})")
    print(
        f"   BS sensitivity: sigma_H(S_ref) {r['sigma_S_ref']:.4f} -> {r['V_BS_at_S_ref']:.6f};",
        end="",
    )
    print(f" sigma_H(B_p S_ref) {r['sigma_B_p']:.4f} -> {r['V_BS_at_B_p']:.6f};", end="")
    mkt = (
        f"{s_mkt:.4f} -> {v_mkt:.6f}"
        if np.isfinite(s_mkt)
        else "n/a (no bracketing expiry / not run)"
    )
    print(f" market ATM {mkt}")
    if np.isfinite(v_mkt):
        print(
            f"   extra row: V_BS at the market ATM vol {s_mkt:.4f} = {v_mkt:.6f} "
            f"({v_mkt - r['V_BS']:+.6f} vs the BS leg; V_H minus it {r['V_H'] - v_mkt:+.6f})"
        )
    print()


def figure(rows, label, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(rows), figsize=(6 * len(rows), 4.5))
    for ax, r in zip(np.atleast_1d(axes), rows, strict=True):
        names = [f"V_BS (BSM at Heston\nATM vol {r['sigma_atm']:.2%})",
                 "flat ATM vs Heston\nEV curve (level+TS)", "marginal smile",
                 "path (fwd skew, vvol)", "V_H"]  # fmt: skip
        steps, ses = [r["term1"], r["term2"], r["term3"]], [r["se_term1"], 0.0, r["se_term3"]]
        base = np.concatenate(([0.0], r["V_BS"] + np.cumsum([0.0] + steps[:-1])))
        hts = [r["V_BS"], *steps, r["V_H"]]
        bottoms = [0.0, *base[1:4], 0.0]
        lo = min(r["V_H"], r["V_BS"]) - 0.03
        cols = ["tab:blue"] + ["tab:red" if s < 0 else "tab:green" for s in steps] + ["tab:purple"]
        for i, (h, b0, c) in enumerate(zip(hts, bottoms, cols, strict=True)):
            if i in (0, 4):
                ax.bar(i, h - lo, bottom=lo, color=c)
            else:
                ax.bar(i, h, bottom=b0, color=c, yerr=3 * ses[i - 1], capsize=4)
        ax.set_xticks(range(5), names, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(lo, max(r["V_H"], r["V_BS"]) + 0.01)
        ax.set_title(f"{r['sheet']}: V_H - V_BS = {r['gap']:+.4f} ({r['se_gap']:.4f})", fontsize=9)
        ax.set_ylabel("PV per unit notional (error bars: 3 CRN SE)")
    fig.suptitle(f"Autocallable BSM vs Heston, eq. (19) - {label}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    raise SystemExit(main())
