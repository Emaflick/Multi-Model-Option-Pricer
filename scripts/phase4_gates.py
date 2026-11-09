"""Print every Phase 4 gate (docs/ROADMAP.md) with measured value, tolerance and PASS/FAIL.

Settings, seeds and oracle tables are imported from tests/test_barrier.py, test_asian.py and
test_autocallable.py, so the numbers are those the tests assert (derivations barrier.md, asian.md,
autocallable.md §5; D-025 to D-031). The autocallable model gap uses scripts/autocall_compare.py
with the GS Heston set: ILLUSTRATIVE, PRE-CALIBRATION (D-029); Phase 5 re-runs it with fitted
parameters. Usage: .venv/bin/python scripts/phase4_gates.py [--seeds n]  (n: G2h fresh-seed scan)
"""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np

from qpricer.engines.analytic import (
    autocall_nocall,
    autocall_two_dates,
    barrier_price,
    bgk_barrier,
    geometric_asian_moments,
    geometric_asian_price,
    ratio_marginals,
)
from qpricer.engines.exotic_mc import (
    asian_averages,
    asian_mc,
    autocall_mc,
    autocall_pv,
    barrier_mc,
    control_variate,
    fixings,
    gbm_paths,
)
from qpricer.models import bsm
from qpricer.products import Autocall, autocall_flows

ROOT = Path(__file__).resolve().parents[1]
RESULTS: list[tuple[str, str]] = []  # (gate, PASS / FAIL / DEFERRED)


def load(name: str, folder: str = "tests"):
    spec = importlib.util.spec_from_file_location(name, ROOT / folder / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TB, TA, TC = load("test_barrier"), load("test_asian"), load("test_autocallable")
AC = load("autocall_compare", "scripts")


def gate(name: str, value: str, tol: str, ok: bool, status: str | None = None) -> None:
    status = status or ("PASS" if ok else "FAIL")
    RESULTS.append((name, status))
    print(f"  {name:<58} {value:<34} tol {tol:<22} {status}")


def z(v, se, ref):
    return (np.asarray(v) - ref) / np.asarray(se)


# ------------------------------------------------------------------ barrier
def barrier() -> None:
    print("\n== Barrier (barrier.md; tests/test_barrier.py)")
    worst = 0.0
    for m in TB.GATE_MODELS.values():
        for mon in (None, 10):
            kw = dict(n_steps=20, n_paths=20_000, seed=TB.SEED, b=0.04, monitor=mon)
            for cp, eta in TB.TYPES:
                H = 92.0 if eta == 1 else 108.0
                o, i, v = (
                    barrier_mc(101.0, [85.0, 100.0, 115.0], H, 1.0, 0.95, m, cp, eta, k, **kw)[0]
                    for k in ("out", "in", "vanilla")
                )
                worst = max(worst, np.abs(i + o - v).max())
    gate(
        "G1 in + out = vanilla, same paths (8 types, BSM+Heston, BB+disc)",
        f"max {worst:.2e}",
        "< 1e-12",
        worst < 1e-12,
    )
    for j, (b, M) in enumerate(TB.GATE_CFG):
        F, K, zs, zero_ok = 100 * np.exp(b * TB.HT), np.array([90.0, 110.0]), [], True
        for cp, eta in TB.TYPES:
            H = 95.0 if eta == 1 else 105.0
            for knock in ("out", "in"):
                ref = barrier_price(F, K, H, TB.HT, TB.HDF, 0.25, cp, eta, knock, b=b)
                v, se, _, _ = barrier_mc(
                    F,
                    K,
                    H,
                    TB.HT,
                    TB.HDF,
                    0.25,
                    cp,
                    eta,
                    knock,
                    n_steps=M,
                    n_paths=200_000,
                    seed=TB.SEED + 10 + j,
                    b=b,
                )
                nz = ref != 0
                zero_ok &= bool(np.all(v[~nz] == 0))
                zs += list(np.abs(z(v[nz], se[nz], ref[nz])))
        gate(
            f"G2 BB-MC vs RR, b={b}, M={M} ({len(zs)} cells, 2e5)",
            f"max |z| {max(zs):.2f}; zeros exact {zero_ok}",
            "|z| <= 3",
            max(zs) <= 3 and zero_ok,
        )
    p = TB.HestonParams(kappa=1.5, theta=0.04, xi=1e-4, rho=0.0, v0=0.04)
    zs = []
    for cp, eta, H in ((1, 1, 90.0), (-1, -1, 110.0)):
        for knock in ("out", "in"):
            ref = barrier_price(100.0, 100.0, H, 1.0, 0.97, 0.2, cp, eta, knock)
            v, se, _, _ = barrier_mc(
                100.0,
                100.0,
                H,
                1.0,
                0.97,
                p,
                cp,
                eta,
                knock,
                n_steps=50,
                n_paths=200_000,
                seed=TB.SEED + 30,
            )
            zs.append(abs(z(v, se, ref)))
    gate("G2h Heston BB-MC xi=1e-4, rho=0 vs RR at sqrt(theta)", f"max |z| {max(zs):.2f}",
         "|z| <= 3", max(zs) <= 3)  # fmt: skip
    for cp in (1, -1):  # diagnostic (D-028): the gate path set's own vanilla error
        v, se, _, _ = barrier_mc(100.0, 100.0, 90.0, 1.0, 0.97, p, cp, 1, "vanilla", n_steps=50,
                                 n_paths=200_000, seed=TB.SEED + 30)  # fmt: skip
        ref = bsm.price(100.0, 100.0, 1.0, 0.97, 0.2, cp)
        print(f"    same path set, vanilla cp={cp:+d}: z {(v - ref) / se:+.2f} (diagnostic)")
    print(
        "  eq. (7) model-free cdo(K = H = 90) = 10 DF, FO set, QE-M + v^+ bridge; ONE seed per dt"
    )
    print("  (noisy; 25-seed pooled bias +0.095/+0.055/+0.020/+0.012 at dt = 1/32..1/256, D-028):")
    for k, M in enumerate((16, 32, 64, 128)):
        seed = TB.SEED + 40 if M == 128 else TB.SEED + 41 + k
        v, se, _, _ = barrier_mc(
            100.0, 90.0, 90.0, 1.0, 0.98, TB.FO, 1, 1, "out", n_steps=M, n_paths=200_000, seed=seed
        )
        print(
            f"    dt = 1/{M:<4} MC - 9.8 = {v - 9.8:+.5f} (SE {se:.5f}, z {(v - 9.8) / se:+.2f})"
            f", seed {seed}"
        )
    gate(
        "G2m Heston model-free cdo at dt = 1/128 (bias < gate SE)",
        f"z {(v - 9.8) / se:+.2f}",
        "|z| <= 3",
        abs(v - 9.8) <= 3 * se,
    )
    errs = []
    for tab, m, F, eta in (
        (TB.BGK_21, 50, TB.BF, 1),
        (TB.BGK_22, 5, TB.BF, 1),
        (TB.BGK_26, 50, TB.BF_UP, -1),
    ):
        H = np.array(list(tab), float)
        cont, corr, _ = np.array(list(tab.values())).T
        errs.append(
            np.abs(
                barrier_price(F, 100.0, H, TB.BT, TB.BDF, TB.BS, 1, eta, "out", b=TB.BB) - cont
            ).max()
        )
        errs.append(
            np.abs(
                barrier_price(
                    F,
                    100.0,
                    bgk_barrier(H, TB.BS, TB.BT / m, eta),
                    TB.BT,
                    TB.BDF,
                    TB.BS,
                    1,
                    eta,
                    "out",
                    b=TB.BB,
                )
                - corr
            ).max()
        )
    gate(
        "G3i RR (cont.) and BGK-shifted RR vs BGK Tables 2.1/2.2/2.6",
        f"max {max(errs):.2e}",
        "<= 5e-4",
        max(errs) <= 5e-4,
    )
    ratio = []
    print("  discrete MC (exact steps at t_j) vs BGK True, and BGK-shifted RR vs discrete MC:")
    for k, (tab, m, F, eta) in enumerate(
        ((TB.BGK_21, 50, TB.BF, 1), (TB.BGK_22, 5, TB.BF, 1), (TB.BGK_26, 50, TB.BF_UP, -1))
    ):
        for H, (_, _, true) in tab.items():
            v, se, _, _ = barrier_mc(
                F,
                100.0,
                H,
                TB.BT,
                TB.BDF,
                TB.BS,
                1,
                eta,
                "out",
                n_steps=m,
                n_paths=200_000,
                seed=TB.SEED + 20 + k,
                b=TB.BB,
                monitor=m,
            )
            ratio.append(abs(v - true) / (3 * se + 0.001))
            bgk = barrier_price(
                F,
                100.0,
                bgk_barrier(H, TB.BS, TB.BT / m, eta),
                TB.BT,
                TB.BDF,
                TB.BS,
                1,
                eta,
                "out",
                b=TB.BB,
            )
            print(
                f"    m={m:<3} H={H:<4} MC {v:.4f} (SE {se:.4f}) True {true:.3f}; BGK {bgk:.4f}:"
                f" rel {(bgk - v) / v:+.2%}, z {(bgk - v) / se:+.2f}"
            )
    gate(
        "G3ii discrete MC vs BGK True (13 cells)",
        f"max |d|/(3SE+1e-3) {max(ratio):.2f}",
        "<= 1",
        max(ratio) <= 1,
    )
    import json  # Haug table per cell (D-027): golden file of the validator (RRS = PDE), read-only

    err = {"pub": {0.25: 0.0, 0.3: 0.0}, "e1": 0.0, "gold": 0.0}
    cells = json.loads(TB.GOLDEN_HAUG.read_text())["cells"]
    for c in cells:
        knock, cp, eta, H, sig, K = (c[k] for k in ("knock", "cp", "eta", "H", "sigma", "K"))
        v = float(barrier_price(TB.HF, K, H, TB.HT, TB.HDF, sig, cp, eta, knock, b=TB.HB, R=TB.HR))
        err["gold"] = max(err["gold"], abs(v - c["value"]))
        if c["published_correctly_rounded"]:
            err["pub"][sig] = max(err["pub"][sig], abs(v - c["published"]))
        else:
            err["e1"] = max(err["e1"], abs(v - c["value"]))
    n_ok = sum(c["published_correctly_rounded"] for c in cells)
    e25, e30 = err["pub"][0.25], err["pub"][0.3]
    worst = f"max s.25 {e25:.2e}, s.30 {e30:.2e}"
    gate(f"Oracle Haug, {n_ok} correctly rounded cells vs published", worst, "<= 5e-5 (note)",
         max(e25, e30) <= 5e-5)  # fmt: skip
    gate(f"Oracle Haug, {len(cells) - n_ok} E1 cells vs golden (RRS = PDE)", f"max {err['e1']:.2e}",
         "<= 1e-10", err["e1"] <= 1e-10)  # fmt: skip
    print(f"    all {len(cells)} cells vs golden value: max {err['gold']:.2e}")
    print("  erratum E1 evidence: Haug sigma = 0.30 cells off by > 5e-5 vs RR and vs quadrature")
    for knock, cp, eta, H, K, ref in (
        ("out", 1, -1, 105, 90.0, 2.6341),
        ("out", -1, -1, 105, 90.0, 4.2293),
        ("out", -1, -1, 105, 100.0, 5.8032),
        ("out", -1, -1, 105, 110.0, 7.5649),
        ("in", -1, 1, 95, 100.0, 7.7989),
        ("in", -1, 1, 95, 110.0, 13.3078),
    ):
        rr = float(
            barrier_price(TB.HF, K, H, TB.HT, TB.HDF, 0.30, cp, eta, knock, b=TB.HB, R=TB.HR)
        )
        q = TB.quad_barrier(TB.HF, K, H, TB.HT, TB.HDF, 0.30, cp, eta, knock, TB.HB, TB.HR)
        print(
            f"    {knock} cp={cp:+d} eta={eta:+d} H={H} K={K:g}: RR {rr:.8f} quad {q:.8f}"
            f" (|d| {abs(rr - q):.1e}) table {ref} RR-table {rr - ref:+.2e}"
        )


# ------------------------------------------------------------------ Asian
def asian() -> None:
    print("\n== Asian (asian.md; tests/test_asian.py)")
    K = 100.0 * np.array([0.8, 1.0, 1.2])
    cases = [(np.arange(1, n + 1) * T / n, np.full(n, 100.0), 1.0, s) for s, T, n in TA.GATE]
    cases.append((*TA.fm08(12), TA.FM_SIG))
    zg, zm = [], []
    for j, (t, F_fix, DF, s) in enumerate(cases):
        A, G = asian_averages(t, F_fix, s, 200_000, np.random.default_rng(TA.SEED + 10 + j))
        X = DF * np.maximum(G - K[:, None], 0)
        zg += list(
            np.abs(
                z(
                    X.mean(1),
                    X.std(1, ddof=1) / np.sqrt(2e5),
                    geometric_asian_price(F_fix, s**2 * t, K, DF, 1),
                )
            )
        )
        FG = geometric_asian_moments(F_fix, s**2 * t)[0]
        zm += [
            abs(z(G.mean(), G.std(ddof=1) / np.sqrt(2e5), FG)),
            abs(z(A.mean(), A.std(ddof=1) / np.sqrt(2e5), F_fix.mean())),
        ]
    gate(
        f"G4 geometric-Asian MC vs Kemna-Vorst ({len(zg)} cells, 2e5)",
        f"max |z| {max(zg):.2f}",
        "|z| <= 3",
        max(zg) <= 3,
    )
    gate(
        "G4m mean G vs F_G, mean A vs Abar_F (13 path sets)",
        f"max |z| {max(zm):.2f}",
        "|z| <= 3",
        max(zm) <= 3,
    )
    e = max(
        np.abs(
            geometric_asian_price(
                TA.fm08(n)[1], TA.FM_SIG**2 * TA.fm08(n)[0], TA.FM_K, TA.fm08(n)[2], 1
            )
            - ref
        ).max()
        for n, ref in TA.FM_GEO.items()
    )
    gate(
        "Oracle KV closed form vs [FM08] Table 4 (n = 12, 50, 250)",
        f"max {e:.2e}",
        "<= 3e-4",
        e <= 3e-4,
    )
    print("  CV variance reduction (eq. 15), same paths as the CV price:")
    rr = []
    for k, (n, ref) in enumerate(TA.FM_ARI.items()):
        t, F_fix, DF = TA.fm08(n)
        seed = TA.SEED + 40 + k
        A, G = asian_averages(t, F_fix, TA.FM_SIG, 200_000, np.random.default_rng(seed))
        Y, X = DF * np.maximum(A - TA.FM_K[:, None], 0), DF * np.maximum(G - TA.FM_K[:, None], 0)
        est, se, c, vr = control_variate(
            Y, X, geometric_asian_price(F_fix, TA.FM_SIG**2 * t, TA.FM_K, DF, 1)
        )
        rr.append(np.max((np.abs(est - ref) - 3 * se) / 3e-4))
        rho = [np.corrcoef(y, x)[0, 1] for y, x in zip(Y, X, strict=True)]
        for i, Ki in enumerate(TA.FM_K):
            print(
                f"    [FM08] n={n:<3} K={Ki:g}: CV {est[i]:.5f} (SE {se[i]:.1e}) vs Table 6"
                f" {ref[i]}; VR {vr[i]:.0f}, rho {rho[i]:.6f}, c {c[i]:.4f}, SE plain"
                f" {Y[i].std(ddof=1) / np.sqrt(2e5):.1e}, N 2e5, seed {seed}"
            )
    gate(
        "Oracle CV MC vs [FM08] Table 6 (n = 12, 50, 250)",
        f"max (|d| - 3SE)/3e-4 {max(rr):.2f}",
        "<= 1",
        max(rr) <= 1,
    )
    t, K3 = np.arange(1, 13) / 12, np.array([80.0, 100.0, 120.0])
    F_fix = 100.0 * np.exp(0.03 * t)
    cv, se_cv, _, _ = asian_mc(
        t, F_fix, K3, 0.97, 0.3, 1, n_paths=200_000, seed=TA.SEED + 30, kind="cv"
    )
    pl, se_pl, _, _ = asian_mc(
        t, F_fix, K3, 0.97, 0.3, 1, n_paths=200_000, seed=TA.SEED + 31, kind="arithmetic"
    )
    zz = np.abs(cv - pl) / np.sqrt(se_cv**2 + se_pl**2)
    gate(
        "CV vs plain arithmetic, independent seeds (sigma 0.3, n 12)",
        f"max |z| {zz.max():.2f}",
        "|z| <= 3",
        zz.max() <= 3,
    )
    print(
        f"    SE ratio plain/CV = {np.round(se_pl / se_cv, 1).tolist()}"
        f" -> VR ~ {np.round((se_pl / se_cv) ** 2).tolist()} (independent seeds)"
    )
    p = TA.HestonParams(kappa=1.5, theta=0.04, xi=1e-4, rho=0.0, v0=0.09)
    dt, E = 1 / 48, np.exp(-1.5 / 48)
    SQ = p.theta * t + (p.v0 - p.theta) * dt / 2 * (1 + E) * (1 - np.exp(-p.kappa * t)) / (1 - E)
    F_fix2 = 100.0 * np.exp(0.02 * t)
    K2 = np.array([90.0, 100.0, 110.0])
    v, se, _, _ = asian_mc(
        t, F_fix2, K2, 0.96, p, 1, n_paths=200_000, seed=TA.SEED + 50, kind="geometric", max_dt=dt
    )
    zh = np.abs(z(v, se, geometric_asian_price(F_fix2, SQ, K2, 0.96, 1))).max()
    gate(
        "Heston QE-M geometric xi=1e-4 vs (10) at Sigma_Q", f"max |z| {zh:.2f}", "|z| <= 3", zh <= 3
    )


# ------------------------------------------------------------------ autocallable
def autocallable() -> None:
    print("\n== Autocallable (autocallable.md; tests/test_autocallable.py)")
    sig, t = 0.25, np.asarray(TC.TQ)
    nc = replace(TC.SHEET, B_ac=np.inf, memory=False)
    zs = []
    for j, sh in enumerate((nc, replace(nc, B_c=0.0))):
        v, se, _, _ = autocall_mc(sh, TC.F_Q, TC.DF_Q, sig, n_paths=200_000, seed=TC.SEED + 10 + j)
        zs.append(abs(z(v, se, autocall_nocall(sh, TC.F_Q, TC.DF_Q, sig**2 * t))))
    one = Autocall((1.0,), 0.08, 1.0, 0.7, 0.6, True)
    v, se, _, _ = autocall_mc(
        one, TC.F_Q[3:4], TC.DF_Q[3:4], sig, n_paths=200_000, seed=TC.SEED + 12
    )
    zs.append(abs(z(v, se, autocall_nocall(one, TC.F_Q[3:4], TC.DF_Q[3:4], sig**2 * np.ones(1)))))
    two = Autocall((0.5, 1.0), 0.08, 1.0, 0.75, 0.6, False)
    for m in (False, True):
        sh = replace(two, memory=m)
        f2, D2 = TC.F_Q[[1, 3]], TC.DF_Q[[1, 3]]
        v, se, _, _ = autocall_mc(sh, f2, D2, sig, n_paths=200_000, seed=TC.SEED + 13 + m)
        zs.append(abs(z(v, se, autocall_two_dates(sh, f2, D2, sig**2 * np.array([0.5, 1.0])))))
    gate(
        "BS MC vs (11), (11) B_c=0, n=1, (12) m=0/1 (2e5)",
        f"max |z| {max(zs):.2f}",
        "|z| <= 3",
        max(zs) <= 3,
    )
    x, _ = fixings(TC.GS, t, 200_000, np.random.default_rng(TC.SEED + 20), 1 / 32)
    M = np.exp(x)
    zm = np.abs(z(M.mean(1), M.std(1, ddof=1) / np.sqrt(2e5), 1.0)).max()
    X, zd = TC.F_Q[:, None] * M, []
    for L in (0.6, 0.7, 1.0):
        ind = X >= L
        zd += list(
            np.abs(
                z(
                    ind.mean(1),
                    ind.std(1, ddof=1) / np.sqrt(2e5),
                    1 - ratio_marginals(TC.F_Q, t, L, TC.GS)[1],
                )
            )
        )
    pv = autocall_pv(X, nc, TC.DF_Q)
    znc = abs(
        z(pv.mean(), pv.std(ddof=1) / np.sqrt(2e5), autocall_nocall(nc, TC.F_Q, TC.DF_Q, TC.GS))
    )
    gate(
        "Heston: mean M_t = 1, D^+_i(L) vs COS, no-call vs (11)+COS",
        f"max |z| {max(zm, max(zd), znc):.2f}",
        "|z| <= 3",
        max(zm, max(zd), znc) <= 3,
    )
    p = TC.HestonParams(kappa=1.5, theta=0.04, xi=1e-4, rho=0.0, v0=0.09)
    dt, E = 1 / 32, np.exp(-1.5 / 32)
    SQ = p.theta * t + (p.v0 - p.theta) * dt / 2 * (1 + E) * (1 - np.exp(-p.kappa * t)) / (1 - E)
    xH, zz = fixings(p, t, 200_000, np.random.default_rng(TC.SEED + 30), dt)
    D = autocall_pv(TC.F_Q[:, None] * np.exp(xH), TC.SHEET, TC.DF_Q) - autocall_pv(
        TC.F_Q[:, None] * np.exp(gbm_paths(SQ, zz)), TC.SHEET, TC.DF_Q
    )
    zx = abs(D.mean() / (D.std(ddof=1) / np.sqrt(2e5)))
    gate("Heston xi=1e-4 vs BS(Sigma_Q), same draws (CRN)", f"|z| {zx:.2f}", "|z| <= 3", zx <= 3)
    print(f"\n  Model gap, eq. (19). Heston = {AC.GS}\n  LABEL: {AC.GS_LABEL}")
    for name, sheet in AC.SHEETS.items():
        f, DF, s_ref = AC.market_inputs(sheet)
        r = AC.attribution(AC.GS, sheet, f, DF)
        AC.report(name, sheet, r, s_ref, f, DF, np.nan, np.nan)
        add = abs(r["term1"] + r["term2"] + r["term3"] - r["gap"])
        znc = max(
            abs(z(r[f"nc_mc_{k}"], r[f"se_nc_mc_{k}"], r[f"nc_{k}"])) for k in ("H", "TS", "BS")
        )
        AC.sigma_grid(sheet, f, DF, r["_z"])
        gate(f"{name} (illustrative): eq. (19) adds up; V^nc paths vs (11)",
             f"|sum-gap| {add:.1e}; max|z| {znc:.2f}", "1e-12; |z| <= 3",
             add < 1e-12 and znc <= 3)  # fmt: skip
        gate(f"{name} (illustrative GS): expected sign V_H < V_BS (§3.5)",
             f"gap {r['gap']:+.5f} ({r['se_gap']:.5f})", "gap < -3 CRN SE",
             r["gap"] < -3 * r["se_gap"])  # fmt: skip
    gate("ROADMAP: autocall BSM (ATM) vs CALIBRATED Heston", "needs Phase 5 fit (D-029)",
         "report", False, "DEFERRED (Phase 5)")  # fmt: skip
    deng()


def deng() -> None:
    """Diagnostic, not a gate (autocallable.md §5): Deng-Mallett-McCann (2011) Case 2 with the
    accrual coupon schedule c_j = e^{0.092 t_j} - e^{0.092 t_{j-1}}, B_ac = B_c = 1.02, memory,
    B_p = 0.8, sigma = 0.2, F_0(t) = 100 e^{0.04 t}, DF(t) = e^{-0.06 t}; 1e6 paths."""
    t = np.arange(1, 13) / 12
    sh = Autocall(tuple(t), tuple(np.diff(np.exp(0.092 * t), prepend=1.0)), 1.02, 1.02, 0.8, True)
    f, DF, N = np.exp(0.04 * t), np.exp(-0.06 * t), 1_000_000
    v, se, _, _ = autocall_mc(sh, f, DF, 0.2, n_paths=N, seed=TC.SEED + 40)
    print(
        f"\n  Deng et al. Case 2 (diagnostic): qpricer MC {100 * v:.3f} (SE {100 * se:.3f}),",
        end="",
    )
    print(f" published 98.39 (z {(v - 0.9839) / se:+.1f}), validator exact 97.507", end="")
    print(f" (z {(v - 0.97507) / se:+.2f}) -> published value NOT reproduced (OPEN_ISSUES)")
    x, _ = fixings(0.2, t, N, np.random.default_rng(TC.SEED + 41))
    X = f[:, None] * np.exp(x)
    _, _, chi, alive = autocall_flows(X, sh)
    p = np.append(chi[:-1].mean(1), (alive & (X[-1] >= 1.02)).mean())
    tab = np.array([0.3767, 0.1435, 0.0781, 0.0506, 0.0361, 0.0275, 0.0218, 0.0178, 0.0149, 0.0127,
                    0.0110, 0.0096])  # fmt: skip
    zp = (p - tab) / np.sqrt(p * (1 - p) / N)
    print(
        f"  first-call p_i vs Table 1: max |z| {np.abs(zp).max():.2f};"
        + f" p_i {np.round(p, 4).tolist()}"
    )


def seed_scan(n: int) -> None:
    """Diagnostic (D-028): the G2h comparison on n fresh seeds SEED + 1000 + s (not a gate)."""
    p = TB.HestonParams(kappa=1.5, theta=0.04, xi=1e-4, rho=0.0, v0=0.04)
    zs = []
    for s in range(n):
        row = []
        for cp, eta, H in ((1, 1, 90.0), (-1, -1, 110.0)):
            for knock in ("out", "in"):
                ref = barrier_price(100.0, 100.0, H, 1.0, 0.97, 0.2, cp, eta, knock)
                v, se, _, _ = barrier_mc(100.0, 100.0, H, 1.0, 0.97, p, cp, eta, knock, n_steps=50,
                                         n_paths=200_000, seed=TB.SEED + 1000 + s)  # fmt: skip
                row.append((v - ref) / se)
        zs.append(row)
    zs = np.array(zs)
    print(f"G2h on {n} fresh seeds (cdo, cdi, puo, pui): mean z {np.round(zs.mean(0), 2).tolist()}")
    print(f"  sd {np.round(zs.std(0, ddof=1), 2).tolist()}, max |z| {np.abs(zs).max():.2f}")


if __name__ == "__main__":
    import sys

    if "--seeds" in sys.argv:
        seed_scan(int(sys.argv[sys.argv.index("--seeds") + 1]))
        raise SystemExit(0)
    print("Phase 4 gates (fixed seeds as in the tests; MC gates: |MC - ref| <= 3 SE)")
    barrier()
    asian()
    autocallable()
    count = {k: sum(st.startswith(k) for _, st in RESULTS) for k in ("PASS", "FAIL", "DEFERRED")}
    print(f"\n{count['PASS']} PASS, {count['FAIL']} FAIL, {count['DEFERRED']} DEFERRED", end="")
    print(f" ({len(RESULTS)} lines)")
    for name, st in RESULTS:
        if st != "PASS":
            print(f"  {st}: {name}")
