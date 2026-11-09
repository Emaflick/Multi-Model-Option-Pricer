"""Calibration to the SPX implied-vol surface: targets, Heston, SABR per expiry, arbitrage checks.

Equation numbers refer to docs/derivations/calibration.md; kernels: heston.md (COS 9-11, Lewis 14),
sabr.md (2, 9-11, 22), bsm.md (IV 11-15). Decisions D-032 to D-038 in docs/DECISIONS.md.
Model prices are normalised (F = DF = 1, strike e^k, k = ln(K/F_j); §1); residuals are in vol
space, r_i = w_i (sigma_model - sigma_mkt), w_i = 1 / max(h_i, h_min) (eqs. 8, 15).
Heston COS runs at one pinned (N, L) for the whole fit (D-034), verified against Lewis over BOUNDS.
"""

from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import ndtri
from scipy.stats import qmc

from qpricer.data import build_surface, remove_static_arbitrage, select_otm
from qpricer.engines.fourier import cos_price, lewis_price
from qpricer.iv import implied_vol
from qpricer.market import N_NEAR, ols_parity
from qpricer.models import bsm, sabr
from qpricer.models.heston import HestonParams

T_SPLIT = 0.1  # Heston window T >= 0.1; DF smoothing below (eqs. 2, 7)
S0_FLOOR = 1e-3  # 10 bp floor on r_se in the WLS weights, eq. (2)
V_MIN = 0.50  # 10 minimum SPX ticks, eq. (5)
Z5 = float(ndtri(0.95))  # 5-delta band |d1| <= 1.6449, eq. (5)
N_THIN = 25  # d1 targets per expiry, eq. (6)
H_MIN = 1e-3  # half-width floor of the weights (0.1 vol pt), eq. (8)
COS_NL = (16384, 64.0)  # pinned COS (N, L) for every fit (D-034)
NAMES = ("v0", "kappa", "theta", "xi", "rho")  # Heston parameter vector order
NOTE_BOUNDS = (
    np.array([1e-4, 1e-3, 1e-4, 0.01, -0.99]),
    np.array([1.0, 20.0, 1.0, 5.0, 0.0]),
)  # (11)
NOTE_START_BOX = {
    "sqrt_v": (0.08, 0.40),
    "kappa": (0.3, 8.0),
    "xi": (0.2, 3.0),
    "rho": (-0.95, -0.2),
}
BOUNDS = (np.array([0.01, 0.5, 0.01, 0.01, -0.99]), np.array([0.25, 20.0, 0.25, 3.0, 0.0]))
START_BOX = {"sqrt_v": (0.10, 0.40), "kappa": (0.5, 8.0), "xi": (0.2, 3.0), "rho": (-0.95, -0.2)}
SOBOL_SEED = 20261109  # Heston and SABR start points (D-037)
SABR_NAMES = ("alpha", "rho", "nu")
SABR_BOUNDS = (np.array([1e-4, -0.999, 1e-4]), np.array([5.0, 0.999, 5.0]))  # sabr.md (11)
NU2T_MAX = 0.16  # Hagan-vs-MC validated domain (Phase 3, D-023)
CONVERGED = (1, 2, 3, 4)  # least_squares status codes counted as converged (eq. 12)


# ------------------------------------------------------------------ 2.1 short-end DF (F5)
def zero_curve(T, r, r_se, s0: float = S0_FLOOR) -> np.ndarray:
    """WLS fit r(T) = a + b T with weights 1 / (r_se^2 + s0^2), eq. (2). Returns (a, b)."""
    T, r, r_se = (np.asarray(a, dtype=float) for a in (T, r, r_se))
    sw = 1.0 / np.sqrt(r_se**2 + s0**2)
    A = np.column_stack([np.ones_like(T), T]) * sw[:, None]
    return np.linalg.lstsq(A, r * sw, rcond=None)[0]


def parity_pairs(clean: pd.DataFrame, table: pd.DataFrame, n_near: int = N_NEAR) -> pd.DataFrame:
    """Put-call pairs used by the Phase 0 parity fit of each kept expiry (market.fit_expiry):
    the n_near strikes nearest F0 = K* + (C - P)(K*) minus the rejected strikes (D-004, D-005).

    Columns expiry, K, y = C - P (mids), h (C - P half-width). Raises unless OLS on these pairs
    reproduces the table's F and DF (rel. 1e-10), i.e. the selection is replicated exactly.
    """
    out = []
    for t in table[table.kept.astype(bool)].itertuples():
        g = clean[(clean.expiry == t.expiry) & (clean.root == t.root)]
        c = g[g.option_type == "C"].set_index("strike")
        p = g[g.option_type == "P"].set_index("strike")
        pr = pd.merge(c, p, left_index=True, right_index=True, suffixes=("_c", "_p")).sort_index()
        K, y = pr.index.to_numpy(float), (pr.mid_c - pr.mid_p).to_numpy()
        h = 0.5 * ((pr.ask_c - pr.bid_c) + (pr.ask_p - pr.bid_p)).to_numpy()
        i0 = np.argmin(np.abs(y))
        sel = np.sort(np.argsort(np.abs(K - (K[i0] + y[i0])), kind="stable")[:n_near])
        rej = [float(s) for s in str(t.rejected_K).split(";") if s not in ("", "nan", "<NA>")]
        sel = sel[~np.isin(K[sel], rej)]
        fit = ols_parity(K[sel], y[sel])
        if not (np.isclose(fit["F"], t.F, rtol=1e-10) and np.isclose(fit["DF"], t.DF, rtol=1e-10)):
            raise ValueError(f"{t.expiry}: parity pairs do not reproduce the table F, DF")
        out.append(pd.DataFrame({"expiry": t.expiry, "K": K[sel], "y": y[sel], "h": h[sel]}))
    return pd.concat(out, ignore_index=True)


def smooth_short_end(
    clean: pd.DataFrame, table: pd.DataFrame, t_split: float = T_SPLIT, s0: float = S0_FLOOR
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Eqs. (2)-(3): DF~ = exp(-r^(T) T) for T < t_split, parity DF kept for T >= t_split; then
    F~ = Kbar_u + ybar_u / DF~ on the used pairs (one-parameter OLS), se(F~) = s_u / (DF~ sqrt n_u).

    Returns (table~, report, (a, b)). table~ has F, DF, r replaced on T < t_split only (for
    T >= t_split eq. 3 reproduces F exactly, reported as F_tilde - F). report: per expiry F, F~,
    z_F = (F~ - F)/F_se, DF, DF~, r, r~, max|res|/h before (Phase 0) and after, n(|res|/h > 1).
    """
    kept = table[table.kept.astype(bool)].sort_values("T")
    long = kept["T"].to_numpy() >= t_split
    a, b = zero_curve(kept["T"][long], kept.r[long], kept.r_se[long], s0)
    pairs = parity_pairs(clean, table)
    rows = []
    for t, is_long in zip(kept.itertuples(), long, strict=True):
        dft = t.DF if is_long else float(np.exp(-(a + b * t.T) * t.T))
        pp = pairs[pairs.expiry == t.expiry]
        K, y, h = pp.K.to_numpy(), pp.y.to_numpy(), pp.h.to_numpy()
        ft = K.mean() + y.mean() / dft
        res = y - dft * (ft - K)
        s_u = np.sqrt(res @ res / (K.size - 1))
        rh = np.abs(res) / h
        rows.append(
            {
                "expiry": t.expiry,
                "T": t.T,
                "F": t.F,
                "F_se": t.F_se,
                "F_tilde": ft,
                "F_se_tilde": s_u / (dft * np.sqrt(K.size)),
                "z_F": (ft - t.F) / t.F_se,
                "DF": t.DF,
                "DF_tilde": dft,
                "r": t.r,
                "r_tilde": -np.log(dft) / t.T,
                "max_res_h": t.max_res_h,
                "max_res_h_tilde": rh.max(),
                "n_res_gt_h": int((rh > 1).sum()),
                "n_used": K.size,
                "smoothed": not is_long,
            }  # fmt: skip
        )
    rep = pd.DataFrame(rows)
    new = table.copy()
    m = rep.set_index("expiry")
    idx = new.expiry.isin(rep.expiry[rep.smoothed]) & new.kept.astype(bool)
    new.loc[idx, "F"] = new.loc[idx, "expiry"].map(m.F_tilde)
    new.loc[idx, "DF"] = new.loc[idx, "expiry"].map(m.DF_tilde)
    new.loc[idx, "r"] = -np.log(new.loc[idx, "DF"]) / new.loc[idx, "T"]
    return new, rep, np.array([a, b])


def recut_surface(clean: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """OTM cut and static-arbitrage filter (data.select_otm, data.remove_static_arbitrage) redone
    with the smoothed forwards, eq. (3): a strike between F and F~ changes type."""
    return remove_static_arbitrage(select_otm(clean, table))[0]


def build_targets(path=None, n_t: int = N_THIN) -> dict:
    """Pipeline of §2.1-2.2 on a raw snapshot: build_surface, eqs. (2)-(3), re-cut surface, eq. (1)
    IVs (``q``), in-band (5) and thinned (6) masks. Keys: S (build_surface dict), table, f5, curve,
    q, band, thin."""
    S = build_surface(path)
    table, f5, curve = smooth_short_end(S["clean"], S["table"])
    q = quote_ivs(recut_surface(S["clean"], table))
    return {"S": S, "table": table, "f5": f5, "curve": curve, "q": q, "band": in_band(q),
            "thin": thin(q, n_t)}  # fmt: skip


# ------------------------------------------------------------------ 2.2 targets and thinning
def quote_ivs(surface: pd.DataFrame, h_min: float = H_MIN) -> pd.DataFrame:
    """Eq. (1): sigma^{m,b,a} by iv.implied_vol (status kept, D-013), h = (sigma^a - sigma^b)/2,
    cp, d1 = -k/s + s/2 (s = sigma^m sqrt T), weight w = 1 / max(h, h_min) (eq. 8)."""
    cp = np.where(surface.option_type == "C", 1.0, -1.0)
    F, K, T, DF = (surface[c].to_numpy(float) for c in ("F", "strike", "T", "DF"))
    out = {"cp": cp}
    for c in ("mid", "bid", "ask"):
        out[f"sigma_{c[0]}"], info = implied_vol(surface[c], F, K, T, DF, cp, full_output=True)
        out[f"status_{c[0]}"] = info["status"]
    q = surface.assign(**out)
    s = q.sigma_m.to_numpy() * np.sqrt(T)
    q["d1"] = -q.k / s + 0.5 * s
    q["h"] = 0.5 * (q.sigma_a - q.sigma_b)
    q["w"] = 1.0 / np.maximum(q.h, h_min)
    return q


def in_band(q: pd.DataFrame, v_min: float = V_MIN, z: float = Z5) -> np.ndarray:
    """Eq. (5): finite sigma^m (status 0), mid >= v_min and |d1(sigma^m)| <= z (5-delta band)."""
    return ((q.status_m == 0) & (q.mid >= v_min) & (np.abs(q.d1) <= z)).to_numpy()


def thin(q: pd.DataFrame, n_t: int = N_THIN, z: float = Z5) -> np.ndarray:
    """Eq. (6), mask over q: per expiry, for each d1 target z_m = -z + 2 z (m-1)/(n_t-1) the unused
    in-band quote with the nearest d1 (targets in order), plus the put and the call nearest k = 0.
    Expiries with <= n_t in-band quotes keep them all. Up to n_t + 2 quotes per expiry (D-032)."""
    band = in_band(q)
    keep = np.zeros(len(q), bool)
    targets = -z + 2.0 * z * np.arange(n_t) / (n_t - 1)
    for _, g in q[band].groupby("expiry"):
        pos = q.index.get_indexer(g.index)
        if pos.size <= n_t:
            keep[pos] = True
            continue
        d1, k, used = g.d1.to_numpy(), g.k.to_numpy(), np.zeros(pos.size, bool)
        for zm in targets:
            used[np.argmin(np.where(used, np.inf, np.abs(d1 - zm)))] = True
        for side in (k < 0, k >= 0):
            if side.any():
                used[np.flatnonzero(side)[np.argmin(np.abs(k[side]))]] = True
        keep[pos[used]] = True
    return keep


def make_grid(T, k, cp, sigma, w) -> dict:
    """Fit grid sorted by (T, k): arrays T, k, K = e^k, cp, sigma (target), w, the normalised Black
    price v = Black(sigma) at F = DF = 1 and vega = phi(d1) sqrt T (§1), and per-expiry slices."""
    T, k, cp, sigma, w = (np.asarray(a, dtype=float) for a in (T, k, cp, sigma, w))
    o = np.lexsort((k, T))
    T, k, cp, sigma, w = T[o], k[o], cp[o], sigma[o], w[o]
    s = sigma * np.sqrt(T)
    d1 = -k / s + 0.5 * s
    g = {"T": T, "k": k, "K": np.exp(k), "cp": cp, "sigma": sigma, "w": w, "order": o}
    g["v"] = bsm.price(1.0, g["K"], T, 1.0, sigma, cp)
    g["vega"] = np.exp(-0.5 * d1 * d1) / np.sqrt(2.0 * np.pi) * np.sqrt(T)
    Tu, start = np.unique(T, return_index=True)
    stop = np.r_[start[1:], T.size]
    g["slices"] = [(float(t), slice(a, b)) for t, a, b in zip(Tu, start, stop, strict=True)]
    return g


def grid_from_quotes(q: pd.DataFrame) -> dict:
    """make_grid on the market targets of quote_ivs (sigma^m, w)."""
    return make_grid(q["T"], q.k, q.cp, q.sigma_m, q.w)


def with_targets(g: dict, sigma: np.ndarray) -> dict:
    """Same grid (k, T, cp, w) with new target vols (synthetic data, eq. 16), in grid order."""
    return make_grid(g["T"], g["k"], g["cp"], sigma, g["w"]) | {"order": g["order"]}


# ------------------------------------------------------------------ 2.3 Heston
def params(x) -> HestonParams:
    """HestonParams from the vector x = (v0, kappa, theta, xi, rho)."""
    v0, kappa, theta, xi, rho = (float(a) for a in x)
    return HestonParams(kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0)


def heston_prices(x, g: dict, nl=COS_NL, lewis: bool = False) -> tuple[np.ndarray, int]:
    """Normalised Heston prices on the grid: COS (heston.md 9-11, puts + parity) at the pinned
    (N, L), or Lewis (heston.md 14). Returns (v, number of COS eq.-13 series warnings)."""
    p, v = params(x), np.empty(g["T"].size)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for T, s in g["slices"]:
            if lewis:
                v[s] = lewis_price(1.0, g["K"][s], T, 1.0, p, g["cp"][s])
            else:
                v[s] = cos_price(1.0, g["K"][s], T, 1.0, p, g["cp"][s], N=nl[0], L=nl[1])
    return v, sum(str(r.message).startswith("COS") for r in rec)


def model_iv(v: np.ndarray, g: dict) -> tuple[np.ndarray, np.ndarray]:
    """IV of normalised prices on the grid (bsm.md 11-15); (sigma, status), NaN where status != 0.
    The RuntimeWarning of implied_vol is absorbed: the status is returned and counted (eq. 9)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        s, info = implied_vol(v, 1.0, g["K"], g["T"], 1.0, g["cp"], full_output=True)
    return s, info["status"]


def heston_iv(x, g: dict, nl=COS_NL, lewis: bool = False) -> np.ndarray:
    """Heston implied vols sigma^H(p) on the grid (NaN where the inversion fails)."""
    return model_iv(heston_prices(x, g, nl, lewis)[0], g)[0]


def heston_residuals(x, g: dict, nl=COS_NL, kind: str = "iv") -> tuple[np.ndarray, int]:
    """Weighted residuals and the fallback count.

    kind="iv": r_i = w_i (sigma^H_i - sigma_i), eq. (8); where the model IV has status != 0 the
    fallback (9) r_i = w_i (v^H_i - v_i) / vega_i (vega at the target vol).
    kind="price": r_i = w_i (v^H_i - v_i) / vega_i everywhere, eq. (10).
    """
    v, _ = heston_prices(x, g, nl)
    pr = g["w"] * (v - g["v"]) / g["vega"]
    if kind == "price":
        return pr, 0
    s, st = model_iv(v, g)
    ok = st == 0
    return np.where(ok, g["w"] * (s - g["sigma"]), pr), int((~ok).sum())


def sobol_starts(m: int = 4, seed: int = SOBOL_SEED, box: dict = START_BOX) -> np.ndarray:
    """2^m scrambled Sobol starts (eq. 11, D-034 box): sqrt(v0), sqrt(theta) uniform, ln kappa,
    ln xi uniform, rho uniform. Rows are x = (v0, kappa, theta, xi, rho)."""
    u = qmc.Sobol(d=5, scramble=True, rng=np.random.default_rng(seed)).random_base2(m)
    sv, kp, xi, rh = (box[n] for n in ("sqrt_v", "kappa", "xi", "rho"))
    lin = lambda a, b, t: a + (b - a) * t  # noqa: E731
    return np.column_stack([
        lin(*sv, u[:, 0]) ** 2,
        np.exp(lin(*np.log(kp), u[:, 1])),
        lin(*sv, u[:, 2]) ** 2,
        np.exp(lin(*np.log(xi), u[:, 3])),
        lin(*rh, u[:, 4]),
    ])  # fmt: skip


def fit_start(args) -> dict:
    """One TRF run (eq. 12); module-level so that starts can run in worker processes."""
    i, x0, g, nl, kind, tol, diff_step, *bnd = args
    bounds = bnd[0] if bnd else BOUNDS
    nfb: list[int] = []

    def fun(x):
        r, n = heston_residuals(x, g, nl, kind)
        nfb.append(n)
        return r

    t0 = perf_counter()
    row = {"start": i, **{f"{n}_0": v for n, v in zip(NAMES, x0, strict=True)}}
    try:
        res = least_squares(fun, x0, jac="2-point", diff_step=diff_step, x_scale="jac",
                            bounds=bounds, method="trf", ftol=tol, xtol=tol, gtol=tol,
                            max_nfev=2000)  # fmt: skip
    except (ValueError, np.linalg.LinAlgError) as e:  # non-finite start or Jacobian: logged
        return {"row": row | {"status": -99, "error": str(e), "seconds": perf_counter() - t0}}
    row |= {f"{n}": v for n, v in zip(NAMES, res.x, strict=True)}
    row |= {"cost": res.cost, "status": res.status, "nfev": res.nfev, "njev": res.njev,
            "optimality": res.optimality, "seconds": perf_counter() - t0,
            "max_fallbacks": max(nfb), "error": ""}  # fmt: skip
    return {"row": row, "res": res}


def fit_heston(
    g: dict,
    starts: np.ndarray,
    nl=COS_NL,
    kind: str = "iv",
    tol: float = 1e-10,
    diff_step: float = 1e-6,
    workers: int = 1,
    bounds=BOUNDS,
) -> dict:
    """Multi-start TRF fit of eq. (8) (or (10)), eqs. (11)-(12); bounds default BOUNDS (D-034),
    NOTE_BOUNDS = the note's box (11) for the audit run (D-034, `run_calibration.py --box note`).

    Returns {"table": per-start DataFrame (start, end, cost, status, nfev, njev, seconds, fallback
    counts), "res": OptimizeResult of the best converged start, "best": its index, "basin": mask
    of starts with cost <= cost_min (1 + 1e-4)}. Raises if no start converged.
    """
    jobs = [(i, np.asarray(x0, float), g, nl, kind, tol, diff_step, bounds)
            for i, x0 in enumerate(starts)]  # fmt: skip
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            out = list(ex.map(fit_start, jobs))
    else:
        out = [fit_start(j) for j in jobs]
    tab = pd.DataFrame([o["row"] for o in out])
    conv = tab.status.isin(CONVERGED).to_numpy()
    if not conv.any():
        raise RuntimeError(f"no Heston start converged: statuses {tab.status.tolist()}")
    cost = np.where(conv, tab.cost.to_numpy(float), np.inf)
    best = int(np.argmin(cost))
    tab["basin"] = conv & (cost <= cost[best] * (1.0 + 1e-4))
    return {"table": tab, "res": out[best]["res"], "best": best, "basin": tab.basin.to_numpy()}


def ls_diagnostics(res, names=NAMES) -> dict:
    """Gauss-Newton diagnostics at the optimum (eq. 12): s^2 = 2 cost / (n - m), C = s^2 (J^T J)^+
    by SVD (singular values below n eps s_max dropped and counted), SE, correlations R, condition
    number of J diag|x| and its weakest right singular vector, active bounds, optimality."""
    J, x = res.jac, res.x
    n, m = J.shape
    s2 = 2.0 * res.cost / (n - m)
    _, S, Vt = np.linalg.svd(J, full_matrices=False)
    keep = S > n * np.finfo(float).eps * S[0]
    C = s2 * (Vt[keep].T / S[keep] ** 2) @ Vt[keep]
    se = np.sqrt(np.diag(C))
    _, Ss, Vts = np.linalg.svd(J * np.abs(x), full_matrices=False)
    return {
        "x": dict(zip(names, x, strict=True)), "se": dict(zip(names, se, strict=True)),
        "corr": pd.DataFrame(C / np.outer(se, se), index=names, columns=names), "cov": C,
        "s2": s2, "n": n, "cond": Ss[0] / Ss[-1], "weakest": dict(zip(names, Vts[-1], strict=True)),
        "rank_dropped": int((~keep).sum()),
        "active": dict(zip(names, res.active_mask, strict=True)),
        "optimality": res.optimality, "cost": res.cost,
    }  # fmt: skip


def basin_spread(tab: pd.DataFrame, x_hat, se, names=NAMES) -> float:
    """max over basin starts and parameters of |x_j^(s) - x^_j| / SE_j (eq. 12)."""
    b = tab[tab.basin]
    d = (b[list(names)].to_numpy(float) - np.asarray(x_hat)) / np.asarray(se)
    return float(np.abs(d).max())


def sandwich_cov(f, x, w, sigma_eps: float, rel_step: float = 1e-5) -> np.ndarray:
    """Predicted covariance at the truth, eq. (17): J = d sigma / d x by central differences
    (step rel_step |x_j|), A = J^T W J, Sigma = sigma_eps^2 A^-1 (J^T W^2 J) A^-1, W = diag(w^2).
    f(x) returns the model vols on the grid."""
    x = np.asarray(x, float)
    cols = []
    for j in range(x.size):
        e = np.zeros_like(x)
        e[j] = rel_step * abs(x[j])
        cols.append((f(x + e) - f(x - e)) / (2.0 * e[j]))
    J, W = np.column_stack(cols), np.asarray(w, float) ** 2
    Ai = np.linalg.inv(J.T @ (W[:, None] * J))
    return sigma_eps**2 * Ai @ (J.T @ ((W * W)[:, None] * J)) @ Ai


def precheck(x, cov, bounds) -> dict:
    """Identifiability pre-check, eq. (18): x_j +- 3 SE_j inside the bounds, 3 SE_j <= 0.5 |x_j|."""
    x, se = np.asarray(x, float), np.sqrt(np.diag(cov))
    inside = (x - 3 * se >= bounds[0]) & (x + 3 * se <= bounds[1])
    rel = 3 * se <= 0.5 * np.abs(x)
    return {"se": se, "inside": inside, "rel": rel, "ok": bool(np.all(inside & rel))}


def round_trip_gates(x_hat, x_star, cov, cost_hat, cost_star, rmse_star, rms_eps, rmse_hat, n,
                     sigma_eps: float, chi2_q: float) -> dict:  # fmt: skip
    """Eq. (19) G2-G4 for a noisy synthetic fit with m parameters: |dx| <= 3 SE (all j),
    D^2 = dx^T Sigma^-1 dx <= chi2_q, cost(x^) <= cost(x*), RMSE(x*) = RMS(eps) to 1e-12 and
    |RMSE(x^) - sigma_eps sqrt((n-m)/n)| <= 3 sigma_eps / sqrt(2n)."""
    dx = np.asarray(x_hat, float) - np.asarray(x_star, float)
    se, m = np.sqrt(np.diag(cov)), dx.size
    D2 = float(dx @ np.linalg.solve(cov, dx))
    tgt, tol = sigma_eps * np.sqrt((n - m) / n), 3.0 * sigma_eps / np.sqrt(2.0 * n)
    return {
        "z": dx / se, "G2a": bool(np.all(np.abs(dx) <= 3 * se)), "D2": D2, "G2b": D2 <= chi2_q,
        "G3": bool(cost_hat <= cost_star), "plumb": abs(rmse_star - rms_eps),
        "G4a": abs(rmse_star - rms_eps) <= 1e-12, "rmse_dev": rmse_hat - tgt, "rmse_tol": tol,
        "G4b": abs(rmse_hat - tgt) <= tol,
    }  # fmt: skip


def cos_lewis_gap(x, g: dict, nl=COS_NL) -> dict:
    """Pinned COS vs Lewis on the grid: e_vega = max |v_COS - v_Lewis| / vega_i (first-order IV
    error at the target vol, the D-034 budget metric) and the direct max |sigma_COS - sigma_Lewis|
    where both inversions succeed; plus COS warning count and the number of status mismatches."""
    vc, nw = heston_prices(x, g, nl)
    vl, _ = heston_prices(x, g, lewis=True)
    (sc, stc), (sl, stl) = model_iv(vc, g), model_iv(vl, g)
    both = (stc == 0) & (stl == 0)
    return {"e_vega": float(np.max(np.abs(vc - vl) / g["vega"])),
            "e_iv": float(np.max(np.abs(sc - sl)[both], initial=0.0)),
            "n_iv": int(both.sum()), "status_mismatch": int((stc != stl).sum()),
            "cos_warn": nw}  # fmt: skip


# ------------------------------------------------------------------ 2.4 SABR per expiry
def sabr_iv(q, g: dict) -> np.ndarray:
    """Hagan beta = 1 vols (sabr.md 2) on the grid for q = (alpha, rho, nu); NaN where 1 + c1 T <= 0
    (sabr.md 22; its RuntimeWarning is absorbed, the NaN is returned)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return sabr.implied_vol(1.0, g["K"], g["T"], *(float(a) for a in q))


def sabr_seed(g: dict, k_max: float = 0.1, nu_min: float = 0.1) -> tuple[np.ndarray, float]:
    """Seed by sabr.md (10): weighted quadratic sigma ~ s0 + S k + C k^2/2 on |k| <= k_max (all
    quotes if fewer than 3 there); nu0 = sqrt(max(3 alpha C + 6 S^2, nu_min^2)) with alpha = s0,
    rho0 = clip(2 S / nu0, +-0.95), alpha0 from (9) at (rho0, nu0). Returns (q0, s0); q0 has a NaN
    alpha if (9) has no positive root (Phase 3 F6)."""
    k, s, w = g["k"], g["sigma"], g["w"]
    m = np.abs(k) <= k_max
    m = m if m.sum() >= 3 else np.ones_like(m)
    c2, c1, c0 = np.polyfit(k[m], s[m], 2, w=w[m])
    S, C = c1, 2.0 * c2
    nu0 = np.sqrt(max(3.0 * c0 * C + 6.0 * S * S, nu_min**2))
    rho0 = float(np.clip(2.0 * S / nu0, -0.95, 0.95))
    return np.array([_alpha9(c0, g["T"][0], rho0, nu0), rho0, nu0]), float(c0)


def _alpha9(s_atm, T, rho, nu) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # NaN = no positive root, handled by caller
        return float(sabr.alpha_from_atm(s_atm, T, rho, nu))


def sabr_starts(g: dict, n_log2: int = 2, seed: int = SOBOL_SEED) -> tuple[np.ndarray, int]:
    """Seed (10) plus 2^n_log2 Sobol points rho in [-0.95, 0.3], ln nu in [ln 0.1, ln 4], alpha from
    (9) at the seed's s0 (§2.4). Starts where (9) has no positive root (s0 above b^2/(4|a|), F6)
    are dropped (D-036). Returns (starts, number dropped)."""
    q0, s0 = sabr_seed(g)
    u = qmc.Sobol(d=2, scramble=True, rng=np.random.default_rng(seed)).random_base2(n_log2)
    rho, nu = -0.95 + 1.25 * u[:, 0], np.exp(np.log(0.1) + u[:, 1] * np.log(40.0))
    pts = [q0] + [
        np.array([_alpha9(s0, g["T"][0], r, n), r, n]) for r, n in zip(rho, nu, strict=True)
    ]
    ok = [p for p in pts if np.isfinite(p[0]) and np.all(np.isfinite(sabr_iv(p, g)))]
    return np.array(ok), len(pts) - len(ok)


def fit_sabr(g: dict, starts: np.ndarray, tol: float = 1e-10) -> dict:
    """Multi-start TRF fit of eq. (15) on one expiry, native (alpha, rho, nu), bounds sabr.md (11),
    default diff_step. Non-finite trial residuals (1 + c1 T <= 0) are rejected by TRF; a non-finite
    optimum raises. Returns {"res", "table", "diag"}: RMSE (vol), cond(J diag|q|), corr(rho, nu),
    active bounds, 1 + c1 T (sabr.md 22) and nu^2 T with its flag nu^2 T > 0.16 (D-036)."""

    def fun(q):
        return g["w"] * (sabr_iv(q, g) - g["sigma"])

    rows, best = [], None
    for i, q0 in enumerate(starts):
        try:
            r = least_squares(fun, q0, jac="2-point", x_scale="jac", bounds=SABR_BOUNDS,
                              method="trf", ftol=tol, xtol=tol, gtol=tol,
                              max_nfev=2000)  # fmt: skip
        except (ValueError, np.linalg.LinAlgError) as e:
            rows.append({"start": i, "status": -99, "error": str(e)})
            continue
        rows.append({"start": i, **dict(zip(SABR_NAMES, r.x, strict=True)), "cost": r.cost,
                     "status": r.status, "nfev": r.nfev, "error": ""})  # fmt: skip
        if r.status in CONVERGED and (best is None or r.cost < best.cost):
            best = r
    if best is None or not np.all(np.isfinite(best.fun)):
        raise RuntimeError(f"SABR fit failed at T = {g['T'][0]:.4f}: {rows}")
    d = ls_diagnostics(best, SABR_NAMES)
    a, rho, nu = best.x
    T = g["T"][0]
    d |= {"rmse": float(np.sqrt(np.mean((best.fun / g["w"]) ** 2))), "nu2T": nu * nu * T,
          "nu2T_flag": nu * nu * T > NU2T_MAX, "corr_rho_nu": float(d["corr"].loc["rho", "nu"]),
          "bracket": 1.0 + (rho * nu * a / 4 + (2 - 3 * rho**2) * nu**2 / 24) * T}  # fmt: skip
    return {"res": best, "table": pd.DataFrame(rows), "diag": d}


# ------------------------------------------------------------------ 2.6 no-arbitrage checks
def calendar_check(q: pd.DataFrame) -> pd.DataFrame:
    """Eq. (21), per test point: consecutive expiries (j, j+1), points = k of both expiries' quotes
    inside the common k-range; w^{m,b,a} = sigma^2 T interpolated linearly in k on each expiry's own
    quotes (no extrapolation). mid violation w^m_{j+1} < w^m_j; strict w^a_{j+1} < w^b_j;
    dsigma = sqrt(w^m_j / T_{j+1}) - sigma^m_{j+1} (vol).
    q: expiry, T, k, sigma_m, sigma_b, sigma_a."""
    ex = [g.sort_values("k") for _, g in q.sort_values("T").groupby("T", sort=True)]
    out = []
    for a, b in zip(ex[:-1], ex[1:], strict=True):
        lo, hi = max(a.k.min(), b.k.min()), min(a.k.max(), b.k.max())
        k = np.unique(np.r_[a.k, b.k])
        k = k[(k >= lo) & (k <= hi)]
        if k.size == 0:
            continue
        w = {(n, c): np.interp(k, d.k, d[f"sigma_{c}"] ** 2 * d["T"])
             for n, d in (("a", a), ("b", b))
             for c in "mba"}  # fmt: skip
        Tb = float(b["T"].iloc[0])
        out.append(pd.DataFrame({
            "expiry": a.expiry.iloc[0], "expiry_next": b.expiry.iloc[0], "k": k,
            "mid": w["b", "m"] < w["a", "m"], "strict": w["b", "a"] < w["a", "b"],
            "dsigma": np.sqrt(w["a", "m"] / Tb) - np.sqrt(w["b", "m"] / Tb),
        }))  # fmt: skip
    return pd.concat(out, ignore_index=True)


def butterfly_check(q: pd.DataFrame) -> pd.DataFrame:
    """Eq. (22), per strike triple of each expiry: undiscounted calls c = V/DF + (F - K)^+ (bid,
    ask, mid alike), lambda = (K_{i+1} - K_i)/(K_{i+1} - K_{i-1}),
    B = lambda c_{i-1} + (1-lambda) c_{i+1} - c_i. mid violation B^m < 0;
    strict lambda c^a_{i-1} + (1-lambda) c^a_{i+1} - c^b_i < 0;
    z = -B^m / h^B. same_type = the three quotes share an option type (no straddle of F)."""
    out = []
    for e, g in q.groupby("expiry"):
        g = g.sort_values("strike")
        K, F, DF = g.strike.to_numpy(), g.F.to_numpy(), g.DF.to_numpy()
        c = {x: g[x].to_numpy() / DF + np.maximum(F - K, 0.0) for x in ("mid", "bid", "ask")}
        if K.size < 3:
            continue
        lam = (K[2:] - K[1:-1]) / (K[2:] - K[:-2])
        B = lam * c["mid"][:-2] + (1 - lam) * c["mid"][2:] - c["mid"][1:-1]
        Bs = lam * c["ask"][:-2] + (1 - lam) * c["ask"][2:] - c["bid"][1:-1]
        sp = c["ask"] - c["bid"]
        hB = 0.5 * (lam * sp[:-2] + sp[1:-1] + (1 - lam) * sp[2:])
        t = g.option_type.to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(hB > 0, -B / hB, np.where(B < 0, np.inf, 0.0))
        out.append(pd.DataFrame({
            "expiry": e, "K": K[1:-1], "B": B, "mid": B < 0, "strict": Bs < 0, "z": z,
            "same_type": (t[:-2] == t[1:-1]) & (t[1:-1] == t[2:]),
        }))  # fmt: skip
    return pd.concat(out, ignore_index=True)


def arbitrage_summary(cal: pd.DataFrame, bfly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Eq. (23): per pair (calendar) and per expiry (butterfly): points tested, mid and strict
    violation counts, max dsigma (vol pts) or max z."""
    c = cal.groupby(["expiry", "expiry_next"], sort=False).agg(
        n=("k", "size"), mid=("mid", "sum"), strict=("strict", "sum"),
        max_dsigma_volpt=("dsigma", lambda s: 100 * s.max()))  # fmt: skip
    b = bfly.groupby("expiry").agg(n=("K", "size"), mid=("mid", "sum"), strict=("strict", "sum"),
                                   max_z=("z", "max"))  # fmt: skip
    return c.reset_index(), b.reset_index()
