"""Phase 5 calibration: F5 closure, targets, Heston/SABR synthetic round trips, arbitrage checks.

docs/ROADMAP.md Phase 5, docs/derivations/calibration.md §5 (oracles) and §2.5 (gates G1-G4).
Synthetic data are labelled `synthetic`; seeds and tolerances fixed a priori (D-035).
COS at the pinned (N, L) = calibration.COS_NL (D-034). scripts/phase5_gates.py imports the helpers.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2, qmc, spearmanr

from qpricer import calibration as cal
from qpricer.models import bsm, sabr

SEED = 20261107  # D-035: S1 SEED, S2 SEED + 1, SABR SEED + 10 + j, linearisation SEED + 100 + s
SIG_EPS = 0.003  # 0.3 vol pt (ROADMAP)
S1 = np.array(
    [0.0354, 1.3253, 0.0354, 0.3877, -0.7165]
)  # GS, v0 := theta (x = v0, kappa, theta, xi, rho)
S2 = np.array([0.015, 3.0, 0.04, 0.4, -0.7])
SABR_SETS = {  # eq. (20): expiry -> q* = (alpha, rho, nu)
    "2026-11-13": (0.15, -0.7, 2.0),
    "2026-12-18": (0.15, -0.7, 1.2),
    "2027-05-21": (0.17, -0.7, 0.7),
    "2028-06-16": (0.19, -0.7, 0.45),
}
CHI2_5, CHI2_3 = chi2.ppf(0.999, 5), chi2.ppf(0.999, 3)  # 20.515, 16.266
WORKERS = max(1, min(8, (os.cpu_count() or 2) - 2))
BOX_BUDGET = 1e-5  # D-034: price error in market-vega units, max |dv|/vega(sigma^m), bounds box


@lru_cache(maxsize=1)
def targets() -> dict:
    return cal.build_targets()


def thinned() -> pd.DataFrame:
    t = targets()
    return t["q"][t["thin"]]


@lru_cache(maxsize=1)
def heston_grid() -> dict:
    q = thinned()
    return cal.grid_from_quotes(q[q["T"] >= cal.T_SPLIT])


def sabr_grid(expiry: str) -> dict:
    q = thinned()
    return cal.grid_from_quotes(q[q.expiry == expiry])


def synthetic(g: dict, sig_true: np.ndarray, seed: int | None) -> tuple[dict, np.ndarray]:
    """`synthetic` targets sigma* + eps, eps ~ N(0, SIG_EPS^2), default_rng(seed); None: eps = 0."""
    eps = np.zeros(sig_true.size)
    if seed is not None:
        eps = SIG_EPS * np.random.default_rng(seed).standard_normal(sig_true.size)
    return cal.with_targets(g, sig_true + eps), eps


def heston_round_trip(x_star, seed, tol=1e-10, kind="iv") -> dict:
    """Eq. (16)-(19) on the real Heston grid: fit, sandwich covariance (17) and gates G2-G4."""
    g = heston_grid()
    sig = cal.heston_iv(x_star, g)
    gs, eps = synthetic(g, sig, seed)
    fit = cal.fit_heston(gs, cal.sobol_starts(), tol=tol, kind=kind, workers=WORKERS)
    out = {"fit": fit, "x": fit["res"].x, "eps": eps}
    if seed is None:
        return out
    cov = cal.sandwich_cov(lambda x: cal.heston_iv(x, g), x_star, g["w"], SIG_EPS)
    r_star, _ = cal.heston_residuals(x_star, gs)
    rmse = lambda r: float(np.sqrt(np.mean((r / gs["w"]) ** 2)))  # noqa: E731
    if kind == "price":  # G2-G4 are defined for the IV objective; report the IV-space residual
        r_hat = cal.heston_residuals(fit["res"].x, gs)[0]
    else:
        r_hat = fit["res"].fun
    out["cov"], out["gates"] = cov, cal.round_trip_gates(
        fit["res"].x, x_star, cov, 0.5 * r_hat @ r_hat, 0.5 * r_star @ r_star, rmse(r_star),
        float(np.sqrt(np.mean(eps**2))), rmse(r_hat), sig.size, SIG_EPS, CHI2_5,
    )  # fmt: skip
    return out


def sabr_round_trip(expiry: str, j: int, noise: bool, tol=1e-10) -> dict:
    g, q_star = sabr_grid(expiry), np.array(SABR_SETS[expiry])
    sig = cal.sabr_iv(q_star, g)
    gs, eps = synthetic(g, sig, SEED + 10 + j if noise else None)
    starts, dropped = cal.sabr_starts(gs)
    fit = cal.fit_sabr(gs, starts, tol=tol)
    cov = cal.sandwich_cov(lambda q: cal.sabr_iv(q, g), q_star, g["w"], SIG_EPS)
    r_star = gs["w"] * (cal.sabr_iv(q_star, gs) - gs["sigma"])
    rmse = lambda r: float(np.sqrt(np.mean((r / gs["w"]) ** 2)))  # noqa: E731
    gates = cal.round_trip_gates(
        fit["res"].x, q_star, cov, fit["res"].cost, 0.5 * r_star @ r_star, rmse(r_star),
        float(np.sqrt(np.mean(eps**2))), rmse(fit["res"].fun), sig.size, SIG_EPS, CHI2_3,
    )  # fmt: skip
    return {"fit": fit, "x": fit["res"].x, "cov": cov, "gates": gates, "dropped": dropped}


def box_points(m: int = 7, seed: int = cal.SOBOL_SEED) -> np.ndarray:
    """32 corners of cal.BOUNDS plus 2^m scrambled Sobol points (log-uniform v0, kappa, theta, xi;
    uniform rho), D-034."""
    lo, hi = cal.BOUNDS
    corners = np.array([np.where(np.array(b), hi, lo) for b in np.ndindex(*(2,) * 5)])
    u = qmc.Sobol(d=5, scramble=True, rng=np.random.default_rng(seed)).random_base2(m)
    logs = np.exp(np.log(lo[:4]) + u[:, :4] * np.log(hi[:4] / lo[:4]))
    return np.vstack([corners, np.column_stack([logs, lo[4] + u[:, 4] * (hi[4] - lo[4])])])


def flat_surface(q: pd.DataFrame, sigma: float = 0.2) -> pd.DataFrame:
    """Black prices at one flat vol on the real strikes/forwards; bid = ask = mid."""
    v = bsm.price(q.F, q.strike, q["T"], q.DF, sigma, q.cp)
    return q.assign(mid=v, bid=v, ask=v, sigma_m=sigma, sigma_b=sigma, sigma_a=sigma)


# ------------------------------------------------------------------ 2.1 F5 closure, identities
def test_zero_curve_recovers_linear_rate_exactly():
    """§5: the WLS (2) recovers a linear r(T) exactly from noise-free DFs (any weights)."""
    T = np.linspace(0.1, 1.6, 21)
    r = -np.log(np.exp(-(0.045 + 0.003 * T) * T)) / T
    se = np.random.default_rng(SEED).uniform(1e-4, 5e-3, T.size)
    assert np.allclose(cal.zero_curve(T, r, se), [0.045, 0.003], rtol=0, atol=1e-14)


def test_short_end_smoothing_closes_f5():
    """Eq. (3) returns F, DF unchanged for T >= 0.1; below, DF~ = curve, monotone, in-band pairs."""
    t = targets()
    f5, tab = t["f5"], t["S"]["table"]
    long = f5[~f5.smoothed]
    assert len(long) == 21 and len(f5) == 39
    assert np.allclose(long.F_tilde, long.F, rtol=1e-12, atol=0)
    assert np.array_equal(long.DF_tilde, long.DF)
    sh = f5[f5.smoothed].sort_values("T")
    a, b = t["curve"]
    assert np.allclose(sh.r_tilde, a + b * sh["T"], rtol=1e-12)
    assert np.all(np.diff(sh.DF_tilde.to_numpy()) < 0)  # monotone DF below 0.1 y (F5 symptom)
    assert (f5.n_res_gt_h == 0).all() and f5.max_res_h_tilde.max() <= 1.0
    kept = tab[tab.kept.astype(bool)]
    assert np.array_equal(
        t["table"].set_index("expiry").loc[long.expiry, "F"],
        kept.set_index("expiry").loc[long.expiry, "F"],
    )


def test_targets_identities_and_selection():
    """Black(sigma^m) reproduces V^m to 1e-12 rel; selection (5)-(7) and thinning (6)."""
    t = targets()
    q, band, th = t["q"], t["band"], t["thin"]
    assert (q.status_m == 0).all() and np.all(np.isfinite(q[["sigma_b", "sigma_a"]]))
    back = bsm.price(q.F, q.strike, q["T"], q.DF, q.sigma_m, q.cp)
    assert np.max(np.abs(back / q.mid - 1)) < 1e-12
    assert len(q) == len(t["S"]["surface"]) == 7625  # re-cut surface keeps every quote (D-033)
    assert not (th & ~band).any()
    sel = q[th]
    assert (sel.mid >= cal.V_MIN).all() and (np.abs(sel.d1) <= cal.Z5).all()
    n = sel.groupby("expiry").size()
    assert n.between(cal.N_THIN, cal.N_THIN + 2).all() and len(n) == 39
    for _, g in sel.groupby("expiry"):  # ATM legs present; every d1 target within 0.14 of a pick
        allq = q[band & (q.expiry == g.expiry.iloc[0])]
        assert g.k[g.k < 0].max() == allq.k[allq.k < 0].max()
        assert g.k[g.k >= 0].min() == allq.k[allq.k >= 0].min()
    g = heston_grid()
    assert len(g["slices"]) == 21 and g["T"].size == 565
    gm = cal.grid_from_quotes(sel)  # make_grid's Black(sigma) = normalised market mid
    v = (sel.mid / (sel.DF * sel.F)).to_numpy()[gm["order"]]
    assert np.max(np.abs(gm["v"] / v - 1)) < 1e-12


# ------------------------------------------------------------------ Heston residuals and COS
def test_residual_plumbing_rmse_equals_rms_eps():
    """G4 plumbing: RMSE(p*) = RMS(eps) to 1e-12 on `synthetic` S1 data; fallback count 0."""
    g = heston_grid()
    gs, eps = synthetic(g, cal.heston_iv(S1, g), SEED)
    r, nfb = cal.heston_residuals(S1, gs)
    assert nfb == 0
    assert abs(np.sqrt(np.mean((r / gs["w"]) ** 2)) - np.sqrt(np.mean(eps**2))) < 1e-12


def test_fallback_residual_is_vega_scaled_price():
    """Eq. (9): where the model IV fails (status != 0) r = w (v^H - v) / vega, counted."""
    g = cal.make_grid([0.5, 0.5], [-12.0, 0.0], [-1.0, -1.0], [0.6, 0.2], [1.0, 1.0])
    r, nfb = cal.heston_residuals(S1, g)
    v, _ = cal.heston_prices(S1, g)
    s, st = cal.model_iv(v, g)
    assert nfb == 1 and st[0] != 0 and st[1] == 0
    assert r[0] == (v[0] - g["v"][0]) / g["vega"][0] and r[1] == s[1] - 0.2


def test_cos_pinned_vs_lewis_worst_corner_and_fit_point():
    """D-034 budget (price error in market-vega units) at the worst corner and a plausible point."""
    g = heston_grid()
    for x in ([0.01, 0.5, 0.01, 3.0, -0.99], [0.052, 4.6, 0.053, 1.65, -0.75]):
        assert cal.cos_lewis_gap(np.array(x), g)["e_vega"] <= BOX_BUDGET


@pytest.mark.slow
def test_cos_pinned_vs_lewis_over_bounds_box():
    """D-034: price error in market-vega units <= 1e-5 at 32 corners + 128 Sobol points."""
    g = heston_grid()
    worst = max(cal.cos_lewis_gap(x, g)["e_vega"] for x in box_points())
    assert worst <= BOX_BUDGET


# ------------------------------------------------------------------ Heston synthetic gates
def test_heston_precheck_s1_s2():
    """Eq. (18) on both synthetic sets (values in D-035)."""
    g = heston_grid()
    for x in (S1, S2):
        cov = cal.sandwich_cov(lambda p: cal.heston_iv(p, g), x, g["w"], SIG_EPS)
        assert cal.precheck(x, cov, cal.BOUNDS)["ok"]


@pytest.mark.slow
@pytest.mark.parametrize("x_star", [S1, S2], ids=["S1", "S2"])
def test_heston_g1_noise_free(x_star):
    """G1: noise-free, 16 starts, tol 1e-14: max |p^/p* - 1| <= 1e-6."""
    out = heston_round_trip(x_star, None, tol=1e-14)
    assert np.max(np.abs(out["x"] / x_star - 1)) <= 1e-6


@pytest.mark.slow
@pytest.mark.parametrize("x_star,seed", [(S1, SEED), (S2, SEED + 1)], ids=["S1", "S2"])
def test_heston_g2_g4_noisy(x_star, seed):
    """G2 (3 SE every j, D^2 <= chi2_5(0.999)), G3, G4 on `synthetic` data with 0.3 vol-pt noise."""
    gt = heston_round_trip(x_star, seed)["gates"]
    assert gt["G2a"] and gt["G2b"] and gt["G3"] and gt["G4a"] and gt["G4b"]


@pytest.mark.slow
def test_heston_objective10_matches_iv_objective_on_s1():
    """§5: objective (10) from the same starts gives p^(10) within 1 SE of p^ (S1, noisy)."""
    iv, pr = heston_round_trip(S1, SEED), heston_round_trip(S1, SEED, kind="price")
    assert np.all(np.abs(pr["x"] - iv["x"]) <= np.sqrt(np.diag(iv["cov"])))


def _linearisation_job(s: int):
    g = heston_grid()
    gs, _ = synthetic(g, cal.heston_iv(S1, g), SEED + 100 + s)
    return cal.fit_start((s, S1, gs, cal.COS_NL, "iv", 1e-10, 1e-6))["res"].x


@pytest.mark.slow
def test_heston_linearisation_64_seeds():
    """§5: 64 noise seeds on S1 (single start at p*): sd(p^_j)/SE_j in [0.73, 1.27]."""
    g = heston_grid()
    cov = cal.sandwich_cov(lambda p: cal.heston_iv(p, g), S1, g["w"], SIG_EPS)
    with ProcessPoolExecutor(WORKERS) as ex:
        xs = np.array(list(ex.map(_linearisation_job, range(64))))
    ratio = xs.std(axis=0, ddof=1) / np.sqrt(np.diag(cov))
    assert np.all((ratio >= 0.73) & (ratio <= 1.27)), ratio


# ------------------------------------------------------------------ SABR
def test_sabr_synthetic_gates():
    """Eq. (20): pre-check (18), G1 (1e-8), G2 (chi2_3), G3, G4 on the four expiries."""
    for j, (e, qs) in enumerate(SABR_SETS.items()):
        g = sabr_grid(e)
        cov = cal.sandwich_cov(lambda q, g=g: cal.sabr_iv(q, g), np.array(qs), g["w"], SIG_EPS)
        assert cal.precheck(np.array(qs), cov, cal.SABR_BOUNDS)["ok"]
        g1 = sabr_round_trip(e, j, noise=False, tol=1e-14)
        assert np.max(np.abs(g1["x"] / np.array(qs) - 1)) <= 1e-8
        gt = sabr_round_trip(e, j, noise=True)["gates"]
        assert gt["G2a"] and gt["G2b"] and gt["G3"] and gt["G4a"] and gt["G4b"], (e, gt)


def test_sabr_real_fits_finite_and_flagged():
    """Every expiry passes (7) and fits: finite residuals, 1 + c1 T > 0, nu^2 T flagged."""
    q = thinned()
    for _, g in q.groupby("expiry"):
        assert g.k.lt(0).sum() >= 2 and g.k.ge(0).sum() >= 2 and len(g) >= 8
        gr = cal.grid_from_quotes(g)
        starts, _ = cal.sabr_starts(gr)
        d = cal.fit_sabr(gr, starts)["diag"]
        assert d["bracket"] > 0 and np.isfinite(d["rmse"])
        assert d["nu2T_flag"] == (d["nu2T"] > cal.NU2T_MAX)


def sabr_on_heston(x=S1) -> pd.DataFrame:
    """§5 diagnostic: SABR (beta = 1) fitted to the noise-free Heston smiles sigma^H(x) on every
    expiry's real thinned grid (k, w); per expiry T and unweighted RMSE (vol)."""
    rows = []
    for _, q in thinned().groupby("expiry"):
        g = cal.grid_from_quotes(q)
        gs = cal.with_targets(g, cal.heston_iv(x, g))
        f = cal.fit_sabr(gs, cal.sabr_starts(gs)[0])
        rows.append({"T": g["T"][0], "rmse": f["diag"]["rmse"], "nu2T": f["diag"]["nu2T"]})
    return pd.DataFrame(rows).sort_values("T")


def test_sabr_fitted_to_heston_smiles_rmse_grows_with_T():
    """§5 (diagnostic, not a ROADMAP gate): SABR fitted to the S1 Heston smiles; its RMSE grows with
    T (positive rank correlation; Hagan's fixed-T smile cannot follow Heston's term structure)."""
    d = sabr_on_heston()
    assert len(d) == 39 and np.all(np.isfinite(d.rmse))
    assert spearmanr(d["T"], d.rmse).statistic > 0 and d.rmse.iloc[-1] > d.rmse.iloc[0]


def test_sabr_alpha_seed_roundtrip_and_f6_drop():
    """sabr.md (9) inside the seeds: alpha reproduces sigma_ATM; infeasible starts are dropped."""
    g = sabr_grid("2028-06-16")
    q0, s0 = cal.sabr_seed(g)
    atm = cal.make_grid([g["T"][0]], [0.0], [1.0], [s0], [1.0])
    assert abs(cal.sabr_iv(q0, atm)[0] - s0) < 1e-14
    with pytest.warns(RuntimeWarning):  # above the b^2/(4|a|) ceiling (F6): NaN
        assert np.isnan(sabr.alpha_from_atm(0.18, 2.0, -0.95, 4.0))
    bad = cal.make_grid([2.0] * 3, [-0.1, 0.0, 0.1], [-1.0, 1.0, 1.0], [0.18] * 3, [1.0] * 3)
    starts, dropped = cal.sabr_starts(bad)
    assert len(starts) + dropped == 5 and np.all(np.isfinite(starts))


# ------------------------------------------------------------------ no-arbitrage (power, size)
def test_arbitrage_checks_size_and_power():
    """§5: flat Black surface -> 0 mid violations; +3 half-width bump flagged by the butterfly;
    two expiries' total-variance smiles swapped flagged by the calendar check."""
    t = targets()
    q, band = t["q"], t["band"]
    flat = flat_surface(q[band])
    assert cal.calendar_check(flat).mid.sum() == 0
    assert cal.butterfly_check(flat).mid.sum() == 0
    e = "2027-03-31"
    i = flat.index[(flat.expiry == e) & (flat.k.abs() < 0.01)][0]
    bumped = flat.copy()
    bumped.loc[i, "mid"] += 3 * 0.5 * (q.loc[i, "ask"] - q.loc[i, "bid"])
    b = cal.butterfly_check(bumped)
    assert b[(b.expiry == e) & (b.K == q.loc[i, "strike"])].mid.all()
    a, c = q[band & (q.expiry == "2027-03-19")], q[band & (q.expiry == "2027-03-31")]
    sw = pd.concat([
        c.assign(expiry=a.expiry.iloc[0], T=a["T"].iloc[0]).pipe(_rescale, c["T"].iloc[0]),
        a.assign(expiry=c.expiry.iloc[0], T=c["T"].iloc[0]).pipe(_rescale, a["T"].iloc[0]),
    ])  # fmt: skip
    assert cal.calendar_check(sw).mid.sum() > 0


def _rescale(d: pd.DataFrame, T_old: float) -> pd.DataFrame:
    """Keep the total variance w = sigma^2 T_old when the row is relabelled to d['T']."""
    f = np.sqrt(T_old / d["T"])
    return d.assign(sigma_m=d.sigma_m * f, sigma_b=d.sigma_b * f, sigma_a=d.sigma_a * f)


def test_arbitrage_real_surface_and_heston_surface():
    """§5: real surface, same-type adjacent triples -> 0 strict butterflies (validator A1); the S1
    Heston surface (COS mid, bid/ask = Black(sigma^H -+ h), real IV half-widths) -> 0 strict
    calendar and butterfly violations."""
    t = targets()
    q, band = t["q"], t["band"]
    b = cal.butterfly_check(q)
    assert (b.strict & b.same_type).sum() == 0
    qb = q[band]
    g = cal.make_grid(qb["T"], qb.k, qb.cp, qb.sigma_m, qb.w)
    v, _ = cal.heston_prices(S1, g)
    hs = qb.iloc[g["order"]].copy()
    sig = cal.model_iv(v, g)[0]
    h = hs.h.to_numpy()
    assert np.all(np.isfinite(sig)) and np.all(sig - h > 0)
    px = {c: bsm.price(hs.F, hs.strike, hs["T"], hs.DF, sig + d, hs.cp) for c, d in
          (("mid", 0.0), ("bid", -h), ("ask", h))}  # fmt: skip
    hs = hs.assign(**px, sigma_m=sig, sigma_b=sig - h, sigma_a=sig + h)
    assert cal.calendar_check(hs).strict.sum() == 0 and cal.butterfly_check(hs).strict.sum() == 0


# ------------------------------------------------------------------ real data
@pytest.mark.slow
def test_real_heston_fit():
    """ROADMAP: the real fit completes (>= 8 converged starts), no NaN, 0 fallbacks at the optimum,
    COS vs Lewis <= 1e-5 at p^ (§5)."""
    g = heston_grid()
    fit = cal.fit_heston(g, cal.sobol_starts(), workers=WORKERS)
    tab, res = fit["table"], fit["res"]
    assert tab.status.isin(cal.CONVERGED).sum() >= 8
    r, nfb = cal.heston_residuals(res.x, g)
    assert nfb == 0 and np.all(np.isfinite(r))
    assert cal.cos_lewis_gap(res.x, g)["e_iv"] <= 1e-5
    d = cal.ls_diagnostics(res)
    assert np.all(np.isfinite(list(d["se"].values())))
