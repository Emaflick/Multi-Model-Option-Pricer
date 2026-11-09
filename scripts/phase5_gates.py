"""Print every Phase 5 gate (ROADMAP, calibration.md §2.5 and §5): value, tolerance, PASS/FAIL.

Seeds, synthetic sets and helpers are imported from tests/test_calibration.py (D-034, D-035), so the
numbers are those the tests assert. The real-data RMSE table is read from the CSV written by
scripts/run_calibration.py (run it first); the Heston fit itself is repeated here.
Usage: .venv/bin/python scripts/phase5_gates.py
"""

from __future__ import annotations

import importlib.util
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from qpricer import calibration as cal
from qpricer.models import bsm

ROOT = Path(__file__).resolve().parents[1]
RESULTS: list[tuple[str, str]] = []


def load(name: str, folder: str = "tests"):
    spec = importlib.util.spec_from_file_location(name, ROOT / folder / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TC = load("test_calibration")


def gate(name: str, value: str, tol: str, ok: bool) -> None:
    RESULTS.append((name, "PASS" if ok else "FAIL"))
    print(f"  {name:<56} {value:<44} tol {tol:<26} {RESULTS[-1][1]}", flush=True)


def fmt(a, f=".2f") -> str:
    return "[" + ", ".join(format(v, f) for v in np.atleast_1d(a)) + "]"


def cos_box() -> None:
    print("\n== Pinned COS (N, L) = %s vs Lewis over the bounds box (D-034)" % (cal.COS_NL,))
    print(
        "   budget metric: price error in market-vega units, max |v_COS - v_Lewis| / vega(sigma^m)"
    )
    g = TC.heston_grid()
    pts = TC.box_points()
    gaps = [cal.cos_lewis_gap(x, g) for x in pts]
    e = np.array([d["e_vega"] for d in gaps])
    iv = np.array([d["e_iv"] for d in gaps])
    i = int(np.argmax(e))
    print(f"  bounds {[b.tolist() for b in cal.BOUNDS]}; worst point {fmt(pts[i], '.4g')}")
    gate(
        "box: price error in market-vega units (160 points)",
        f"{e.max():.2e} (corners {e[:32].max():.2e})",
        f"<= {TC.BOX_BUDGET:g} (a priori)",
        e.max() <= TC.BOX_BUDGET,
    )
    print(
        f"  (not an IV-accuracy claim: the direct model-IV error |sigma_COS - sigma_Lewis| is "
        f"{iv[i]:.2e} at the worst-budget corner and {iv.max():.2e} over the box, where the model "
        f"vega is 1e-4..1e-11 of the market vega (D-034); COS eq.-13 warnings "
        f"{sum(d['cos_warn'] for d in gaps)})"
    )


def identities() -> None:
    print("\n== F5 closure and target identities (eqs. 1-3, §5)")
    t = TC.targets()
    f5, q = t["f5"], t["q"]
    lg = f5[~f5.smoothed]
    rel = float(np.max(np.abs(lg.F_tilde / lg.F - 1)))
    gate(
        "eq. (3) returns F, DF unchanged for T >= 0.1",
        f"max |F~/F - 1| {rel:.1e}",
        "1e-12",
        rel <= 1e-12 and np.array_equal(lg.DF_tilde, lg.DF),
    )
    T = np.linspace(0.1, 1.6, 21)
    err = np.abs(cal.zero_curve(T, 0.045 + 0.003 * T, np.full(21, 1e-3)) - [0.045, 0.003]).max()
    gate("WLS (2) recovers a linear r(T) exactly", f"{err:.1e}", "1e-14", err <= 1e-14)
    sh = f5[f5.smoothed]
    gate(
        "used pairs inside their band under (F~, DF~)",
        f"max |res|/h {sh.max_res_h_tilde.max():.3f}; n > 1: {int(f5.n_res_gt_h.sum())}",
        "<= 1 (D-005 rule)",
        f5.max_res_h_tilde.max() <= 1,
    )
    mono = int((np.diff(f5.sort_values("T").DF_tilde) > 0).sum())
    print(
        f"  (DF rises between consecutive expiries: {int((np.diff(f5.DF) > 0).sum())} -> {mono}; "
        f"max |F~ - F|/F_se {sh.z_F.abs().max():.2f})"
    )
    back = bsm.price(q.F, q.strike, q["T"], q.DF, q.sigma_m, q.cp)
    e = float(np.max(np.abs(back / q.mid - 1)))
    gate("Black(sigma^m) reproduces V^m (7625 quotes)", f"max rel {e:.1e}", "1e-12", e < 1e-12)


def heston_synthetic() -> None:
    print("\n== Heston synthetic round trip (`synthetic` data, 0.3 vol-pt noise; D-035)")
    g = TC.heston_grid()
    for name, x, seed in (("S1", TC.S1, TC.SEED), ("S2", TC.S2, TC.SEED + 1)):
        cov = cal.sandwich_cov(lambda p: cal.heston_iv(p, g), x, g["w"], TC.SIG_EPS)
        pc = cal.precheck(x, cov, cal.BOUNDS)
        gate(
            f"{name} pre-check (18): +-3SE in bounds, 3SE <= 0.5|p|",
            f"3SE/|p| max {np.max(3 * pc['se'] / np.abs(x)):.3f}",
            "inside; <= 0.5",
            pc["ok"],
        )
        g1 = TC.heston_round_trip(x, None, tol=1e-14)
        e1 = float(np.max(np.abs(g1["x"] / x - 1)))
        gate(
            f"{name} G1 noise-free, 16 starts, tol 1e-14",
            f"max |p^/p* - 1| {e1:.1e}",
            "1e-6",
            e1 <= 1e-6,
        )
        out = TC.heston_round_trip(x, seed)
        gt = out["gates"]
        print(
            f"  {name} p^ = {fmt(out['x'], '.5f')}, SE {fmt(np.sqrt(np.diag(out['cov'])), '.2e')}"
        )
        gate(
            f"{name} G2 |p^ - p*| <= 3 SE (v0, kappa, theta, xi, rho)",
            f"z {fmt(gt['z'])}",
            "3",
            gt["G2a"],
        )
        gate(
            f"{name} G2 D^2 = dp' Sigma^-1 dp",
            f"{gt['D2']:.3f}",
            f"chi2_5(0.999) {TC.CHI2_5:.2f}",
            gt["G2b"],
        )
        tab = out["fit"]["table"]
        cs = 0.5 * float(
            np.sum(
                (cal.heston_residuals(x, cal.with_targets(g, cal.heston_iv(x, g) + out["eps"]))[0])
                ** 2
            )
        )
        gate(
            f"{name} G3 cost(p^) <= cost(p*)",
            f"{out['fit']['res'].cost:.4f} vs {cs:.4f}",
            "<=",
            gt["G3"],
        )
        gate(f"{name} G4 RMSE(p*) = RMS(eps)", f"{gt['plumb']:.1e}", "1e-12", gt["G4a"])
        gate(
            f"{name} G4 RMSE(p^) - sigma_eps sqrt((n-5)/n)",
            f"{1e4 * gt['rmse_dev']:+.3f}e-4",
            f"|.| <= {1e4 * gt['rmse_tol']:.3f}e-4",
            gt["G4b"],
        )
        print(
            f"  ({name}: converged {int(tab.status.isin(cal.CONVERGED).sum())}/16, basin "
            f"{int(tab.basin.sum())}/16)"
        )
        if name == "S1":
            pr = TC.heston_round_trip(x, seed, kind="price")
            dz = (pr["x"] - out["x"]) / np.sqrt(np.diag(out["cov"]))
            gate(
                "S1 objective (10) vs (8), same starts: |dp|/SE",
                f"{fmt(dz)}",
                "<= 1 (§5)",
                bool(np.all(np.abs(dz) <= 1)),
            )
    t0 = time.time()
    cov = cal.sandwich_cov(lambda p: cal.heston_iv(p, g), TC.S1, g["w"], TC.SIG_EPS)
    with ProcessPoolExecutor(TC.WORKERS) as ex:
        xs = np.array(list(ex.map(lin_job, range(64))))
    ratio = xs.std(axis=0, ddof=1) / np.sqrt(np.diag(cov))
    bias = (xs.mean(axis=0) - TC.S1) / (np.sqrt(np.diag(cov)) / 8)
    gate(
        "S1 linearisation, 64 seeds: sd(p^_j)/SE_j",
        fmt(ratio),
        "[0.73, 1.27]",
        bool(np.all((ratio >= 0.73) & (ratio <= 1.27))),
    )
    print(f"  (mean bias in units of SE/sqrt(64): {fmt(bias)}; {time.time() - t0:.0f} s)")


def lin_job(s: int) -> np.ndarray:
    """One linearisation seed (D-035); defined here so that worker processes can unpickle it."""
    g = TC.heston_grid()
    gs, _ = TC.synthetic(g, cal.heston_iv(TC.S1, g), TC.SEED + 100 + s)
    return cal.fit_start((s, TC.S1, gs, cal.COS_NL, "iv", 1e-10, 1e-6))["res"].x


def sabr_synthetic() -> None:
    print("\n== SABR synthetic round trip, eq. (20) (`synthetic`, 0.3 vol-pt noise)")
    for j, (e, qs) in enumerate(TC.SABR_SETS.items()):
        g, qs = TC.sabr_grid(e), np.array(qs)
        cov = cal.sandwich_cov(lambda q, g=g: cal.sabr_iv(q, g), qs, g["w"], TC.SIG_EPS)
        pc = cal.precheck(qs, cov, cal.SABR_BOUNDS)
        e1 = float(np.max(np.abs(TC.sabr_round_trip(e, j, False, 1e-14)["x"] / qs - 1)))
        out = TC.sabr_round_trip(e, j, True)
        gt = out["gates"]
        ok = pc["ok"] and e1 <= 1e-8 and gt["G2a"] and gt["G2b"] and gt["G3"]
        ok = ok and gt["G4a"] and gt["G4b"]
        gate(
            f"SABR {e} q* {fmt(qs)}: pre, G1-G4",
            f"G1 {e1:.0e} z {fmt(gt['z'], '.1f')} D2 {gt['D2']:.2f}",
            f"1e-8; 3; {TC.CHI2_3:.2f}",
            ok,
        )
        print(
            f"    3SE/|q| max {np.max(3 * pc['se'] / np.abs(qs)):.3f}; G3 {gt['G3']}; G4 plumbing "
            f"{gt['plumb']:.0e}, RMSE dev {1e4 * gt['rmse_dev']:+.2f}e-4 "
            f"(tol {1e4 * gt['rmse_tol']:.2f}e-4)"
            f"; starts dropped (F6) {out['dropped']}"
        )


def sabr_on_heston() -> None:
    """§5 diagnostic (not a ROADMAP gate): SABR fitted to the S1 Heston smiles on the 39 grids."""
    d = TC.sabr_on_heston()
    rho = float(spearmanr(d["T"], d.rmse).statistic)
    print(f"  SABR fitted to S1 Heston smiles (39 expiries): RMSE "
          f"{100 * d.rmse.iloc[0]:.3f} vol pt "
          f"(T = {d['T'].iloc[0]:.3f}) .. {100 * d.rmse.iloc[-1]:.3f} (T = {d['T'].iloc[-1]:.2f}), "
          f"max {100 * d.rmse.max():.3f}; Spearman(T, RMSE) {rho:.2f}")  # fmt: skip
    gate("diagnostic §5: SABR-on-Heston RMSE grows with T", f"Spearman {rho:.2f}",
         "> 0 (qualitative)", rho > 0 and d.rmse.iloc[-1] > d.rmse.iloc[0])  # fmt: skip


def real_data() -> None:
    print("\n== Real-data fit (ROADMAP: completes, RMSE table, no NaN)")
    g = TC.heston_grid()
    fit = cal.fit_heston(g, cal.sobol_starts(), workers=TC.WORKERS)
    tab, res = fit["table"], fit["res"]
    nconv = int(tab.status.isin(cal.CONVERGED).sum())
    gate(
        "Heston multi-start: converged starts",
        f"{nconv}/16 (basin {int(tab.basin.sum())})",
        ">= 8",
        nconv >= 8,
    )
    r, nfb = cal.heston_residuals(res.x, g)
    gate(
        "fallbacks (9) at the optimum; residuals finite",
        f"{nfb}; finite {np.isfinite(r).all()}",
        "0; True",
        nfb == 0 and bool(np.isfinite(r).all()),
    )
    gap = cal.cos_lewis_gap(res.x, g)
    gate(
        "post-fit COS vs Lewis IV agreement at p^ on I_H",
        f"max |dsigma| {gap['e_iv']:.1e}",
        "1e-5 (§5)",
        gap["e_iv"] <= 1e-5,
    )
    d = cal.ls_diagnostics(res)
    se = np.array(list(d["se"].values()))
    pr = cal.fit_heston(g, cal.sobol_starts(), kind="price", workers=TC.WORKERS)["res"].x
    dz = (pr - res.x) / se
    gate(
        "real objective (10) vs (8): |dp|/SE (curvature SE)",
        fmt(dz),
        "<= 1 (§5)",
        bool(np.all(np.abs(dz) <= 1)),
    )
    tag = Path(TC.targets()["S"]["meta"]["path"]).stem.removeprefix("spx_")
    csv, js = ROOT / "report" / f"calibration_expiry_{tag}.csv", ROOT / "data" / "processed"
    js = js / f"heston_fit_{tag}.json"
    if not (csv.exists() and js.exists()):
        gate("RMSE table produced by run_calibration.py", f"{csv.name} missing", "exists", False)
        return
    et, p = pd.read_csv(csv), json.loads(js.read_text())
    same = max(abs(p[n] - v) / abs(v) for n, v in zip(cal.NAMES, res.x, strict=True))
    num = et.select_dtypes("number").to_numpy(float)
    gate(
        "RMSE table: 39 expiries, no NaN; JSON p^ = this fit",
        f"rows {len(et)}, NaN {int(np.isnan(num).sum())}; rel {same:.1e}",
        "39, 0; 1e-8",
        len(et) == 39 and not np.isnan(num).any() and same <= 1e-8,
    )
    for smp in ("in", "out"):
        s = et[et["sample"] == smp]
        print(
            f"  Heston {'in-sample' if smp == 'in' else 'out-of-sample'}: {len(s)} expiries, "
            f"RMSE {s.heston_rmse_volpt.min():.3f}-"
            f"{s.heston_rmse_volpt.max():.3f} vol pt (median {s.heston_rmse_volpt.median():.3f})"
        )


def arbitrage() -> None:
    print("\n== No-arbitrage checks: size and power (§5), market surface (eqs. 21-23)")
    t = TC.targets()
    q, band = t["q"], t["band"]
    flat = TC.flat_surface(q[band])
    n0 = int(cal.calendar_check(flat).mid.sum() + cal.butterfly_check(flat).mid.sum())
    gate("flat-vol Black surface: mid violations (cal + bfly)", f"{n0}", "0", n0 == 0)
    b = cal.butterfly_check(q)
    st = int((b.strict & b.same_type).sum())
    gate(
        "real surface: same-type adjacent strict butterflies", f"{st}", "0 (validator A1)", st == 0
    )
    c = cal.calendar_check(q[band])
    print(
        f"  market: calendar {len(c)} points, mid {int(c.mid.sum())}, "
        f"strict {int(c.strict.sum())}; "
        f"butterfly {len(b)} triples, mid {int(b.mid.sum())}, strict {int(b.strict.sum())}, "
        f"max z {b.z.max():.3f}"
    )
    for fn, label in (
        (TC.test_arbitrage_checks_size_and_power, "bump +3h flagged; swapped smiles flagged"),
        (
            TC.test_arbitrage_real_surface_and_heston_surface,
            "S1 Heston surface: 0 strict (calendar, butterfly)",
        ),
    ):
        try:
            fn()
            ok = True
        except AssertionError:
            ok = False
        gate(label, "test assertions", "as in test", ok)


if __name__ == "__main__":
    logging.getLogger("qpricer").setLevel(logging.ERROR)
    t0 = time.time()
    print(f"Phase 5 gates (seeds and sets of tests/test_calibration.py; COS {cal.COS_NL})")
    cos_box()
    identities()
    heston_synthetic()
    sabr_synthetic()
    sabr_on_heston()
    real_data()
    arbitrage()
    n = sum(s == "PASS" for _, s in RESULTS)
    print(f"\n{n} PASS, {len(RESULTS) - n} FAIL ({len(RESULTS)} lines, {time.time() - t0:.0f} s)")
    for name, s in RESULTS:
        if s != "PASS":
            print(f"  FAIL: {name}")
