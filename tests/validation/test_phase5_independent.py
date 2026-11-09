"""Independent validation of Phase 5 (calibration to SPX) and the calibrated Phase 4 autocall gate.

The qpricer API (``calibration.build_targets`` as the data loader, ``calibration.heston_iv``,
``calibration.sabr_iv``, ``engines.fourier.cos_price``) is only called as the system under test or
to load the snapshot rows. Every reference number is computed here with my own code:

* CF    -- Heston CF, little-trap form, checked against my own RK4 integration of the Riccati ODEs.
* LEWIS -- Lewis (2000) single integral by composite Gauss-Legendre (10 nodes, panels 0.25 below
           u = 20 and 1 above), cutoff where |phi(u - i/2)|/(u^2 + 1/4) < 1e-16 (implementation:
           COS pinned at (16384, 64); its own Lewis uses scipy quad_vec).
* IV    -- vectorised bisection on ln sigma (100 halvings) + 2 Newton polishing steps, my Black.
* LS    -- my residuals, my own central-difference Jacobian for the SEs, scipy least_squares only
           as the optimiser, on the NOTE'S ORIGINAL bounds box (11), not the D-034 box.
* HAGAN -- Hagan et al. (2002) beta = 1 transcribed from the paper.
* F5    -- parity pairs rebuilt from the Phase 0 clean chain, own OLS, own WLS zero curve and refit.
* ARB   -- own calendar (total variance, common k-range) and butterfly (undiscounted calls) checks.
* AC    -- exact BS / deterministic-variance autocall values (ACX backward induction), Heston
           marginals by Gil-Pelaez and Heston paths by the Phase 4 validator's own QE-M (OQEM),
           at two time steps; loaded from tests/validation/test_phase4_independent.py (validator
           code, not qpricer).

Seeds: validator base 5_050_000 (never re-seeded after a look). Run as a script to print the
numbers quoted in docs/validation/phase5.md:
``.venv/bin/python tests/validation/test_phase5_independent.py``.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.optimize import least_squares
from scipy.special import ndtr, ndtri
from scipy.stats import chi2, qmc

REPO = Path(__file__).resolve().parents[2]
TAG = "20261106_2213"
FIT_JSON = REPO / "data" / "processed" / f"heston_fit_{TAG}.json"
VSEED = 5_050_000
WORKERS = max(1, min(8, (os.cpu_count() or 2) - 2))
NAMES = ("v0", "kappa", "theta", "xi", "rho")
NOTE_BOX = (np.array([1e-4, 1e-3, 1e-4, 0.01, -0.99]), np.array([1.0, 20.0, 1.0, 5.0, 0.0]))
D034_BOX = (np.array([0.01, 0.5, 0.01, 0.01, -0.99]), np.array([0.25, 20.0, 0.25, 3.0, 0.0]))
S1 = np.array([0.0354, 1.3253, 0.0354, 0.3877, -0.7165])
S2 = np.array([0.015, 3.0, 0.04, 0.4, -0.7])
SIG_EPS, H_MIN, Z5 = 0.003, 1e-3, float(ndtri(0.95))
SQ2PI = np.sqrt(2.0 * np.pi)


# =========================================================================== Black / IV
def black_norm(k, T, sig, cp):
    """Undiscounted Black price at F = 1, strike e^k; cp = +1 call, -1 put."""
    k, T, sig, cp = np.broadcast_arrays(*(np.asarray(a, float) for a in (k, T, sig, cp)))
    s = sig * np.sqrt(T)
    d1 = -k / s + 0.5 * s
    return cp * (ndtr(cp * d1) - np.exp(k) * ndtr(cp * (d1 - s)))


def vega_norm(k, T, sig):
    s = sig * np.sqrt(T)
    d1 = -k / s + 0.5 * s
    return np.exp(-0.5 * d1 * d1) / SQ2PI * np.sqrt(T)


def iv_norm(v, k, T, cp, lo=1e-6, hi=10.0):
    """Implied vol of normalised prices by bisection on ln sigma; NaN outside the bracket."""
    v, k, T, cp = np.broadcast_arrays(*(np.asarray(a, float) for a in (v, k, T, cp)))
    a, b = np.full(v.shape, np.log(lo)), np.full(v.shape, np.log(hi))
    ok = (v > black_norm(k, T, lo, cp)) & (v < black_norm(k, T, hi, cp))
    for _ in range(100):
        m = 0.5 * (a + b)
        up = black_norm(k, T, np.exp(m), cp) > v
        b, a = np.where(up, m, b), np.where(up, a, m)
    s = np.exp(0.5 * (a + b))
    for _ in range(2):
        vg = vega_norm(k, T, s)
        step = np.where(vg > 0, (black_norm(k, T, s, cp) - v) / np.maximum(vg, 1e-300), 0.0)
        s = np.where(np.abs(step) < 1e-6 * s, s - step, s)
    return np.where(ok, s, np.nan)


# =========================================================================== Heston
def heston_cf(u, T, v0, kappa, theta, xi, rho):
    """E exp(i u ln(F_T/F_0)), little-trap form (Albrecher et al. 2007)."""
    u = np.asarray(u, dtype=complex)
    a = kappa - rho * xi * 1j * u
    d = np.sqrt(a * a + xi * xi * (1j * u + u * u))
    g = (a - d) / (a + d)
    e = np.exp(-d * T)
    C = kappa * theta / xi**2 * ((a - d) * T - 2.0 * np.log((1 - g * e) / (1 - g)))
    return np.exp(C + (a - d) / xi**2 * (1 - e) / (1 - g * e) * v0)


def heston_cf_rk4(u, T, v0, kappa, theta, xi, rho, n=20000):
    """Same CF from dD/dt = -(u^2 + iu)/2 - (kappa - rho xi i u) D + xi^2 D^2 / 2,
    dC/dt = kappa theta D, D(0) = C(0) = 0, by classical RK4 (a different method)."""
    u = np.asarray(u, dtype=complex)
    h = T / n

    def f(D):
        return -0.5 * (u * u + 1j * u) - (kappa - rho * xi * 1j * u) * D + 0.5 * xi * xi * D * D

    D, C = np.zeros_like(u), np.zeros_like(u)
    for _ in range(n):
        k1 = f(D)
        k2 = f(D + 0.5 * h * k1)
        k3 = f(D + 0.5 * h * k2)
        k4 = f(D + h * k3)
        C = C + kappa * theta * h / 6 * (6 * D + h * (k1 + k2 + k3))
        D = D + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return np.exp(C + D * v0)


_GX, _GW = leggauss(10)


def _nodes(U, h_fine=0.25, u_fine=20.0, h=1.0):
    e = np.concatenate((np.arange(0.0, u_fine, h_fine), np.arange(u_fine, U + h, h)))
    mid, half = 0.5 * (e[1:] + e[:-1]), 0.5 * np.diff(e)
    return (mid[:, None] + half[:, None] * _GX).ravel(), (half[:, None] * _GW).ravel()


def cutoff(T, p, tol=1e-16, umax=2e5):
    """u beyond which |phi(u - i/2)| / (u^2 + 1/4) < tol on a geometric grid (capped)."""
    u = 20.0 * 1.15 ** np.arange(0, 120)
    u = u[u <= umax]
    m = np.abs(heston_cf(u - 0.5j, T, *p)) / (u * u + 0.25)
    bad = np.flatnonzero(m >= tol)
    return float(u[min(bad[-1] + 1, u.size - 1)]) if bad.size else 20.0


def lewis_norm(k, T, p, cp, U=None, **kw):
    """Normalised Heston price (F = DF = 1), strike e^k (Lewis 2000):
    C = 1 - e^{k/2}/pi int_0^inf Re[e^{-iuk} phi(u - i/2)] du/(u^2 + 1/4); puts by parity."""
    k, cp = np.asarray(k, float), np.asarray(cp, float)
    u, w = _nodes(cutoff(T, p) if U is None else U, **kw)
    phi = heston_cf(u - 0.5j, T, *p) / (u * u + 0.25)
    integral = np.cos(np.outer(k, u)) @ (w * phi.real) + np.sin(np.outer(k, u)) @ (w * phi.imag)
    call = 1.0 - np.exp(0.5 * k) / np.pi * integral
    return np.where(cp > 0, call, call - 1.0 + np.exp(k))


def heston_px(p, T, k, cp):
    out = np.empty(k.size)
    for t in np.unique(T):
        m = T == t
        out[m] = lewis_norm(k[m], t, p, cp[m])
    return out


def heston_iv(p, T, k, cp):
    return iv_norm(heston_px(p, T, k, cp), k, T, cp)


# =========================================================================== SABR, LS helpers
def hagan_b1(k, T, alpha, rho, nu):
    """Hagan et al. (2002), beta = 1, k = ln(K/F): alpha z/x(z) [1 + (rho nu alpha/4 +
    (2 - 3 rho^2) nu^2/24) T], z = (nu/alpha) ln(F/K),
    x = ln((sqrt(1 - 2 rho z + z^2) + z - rho)/(1 - rho))."""
    z = -nu / alpha * np.asarray(k, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.log((np.sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho))
        zx = np.where(np.abs(z) < 1e-7, 1 - 0.5 * rho * z, z / x)
    return alpha * zx * (1 + (rho * nu * alpha / 4 + (2 - 3 * rho**2) * nu**2 / 24) * T)


def fd_jac(f, x, rel=1e-5):
    x = np.asarray(x, float)
    cols = []
    for j in range(x.size):
        e = np.zeros_like(x)
        e[j] = rel * max(abs(x[j]), 1e-3)
        cols.append((f(x + e) - f(x - e)) / (2 * e[j]))
    return np.column_stack(cols)


def sandwich(f, x, w):
    """Eq. (17) a-priori covariance, my own Jacobian: s_eps^2 A^-1 J'W^2 J A^-1, W = diag w^2."""
    J, W = fd_jac(f, x), w**2
    Ai = np.linalg.inv(J.T @ (W[:, None] * J))
    return SIG_EPS**2 * Ai @ (J.T @ ((W * W)[:, None] * J)) @ Ai


def fit_json():
    d = json.loads(FIT_JSON.read_text())
    return np.array([d[n] for n in NAMES]), np.array([d["se"][n] for n in NAMES]), d


# =========================================================================== market targets
@lru_cache(maxsize=1)
def market() -> pd.DataFrame:
    """Snapshot rows (prices, K, F~, DF~, T, type) from the implementation's loader; every IV,
    weight, band and thinning flag below is recomputed with my own code."""
    from qpricer import calibration as cal

    logging.getLogger("qpricer").setLevel(logging.ERROR)
    tg = cal.build_targets()
    q = tg["q"].assign(band_impl=tg["band"], thin_impl=tg["thin"])
    F, DF, K, T = (q[c].to_numpy(float) for c in ("F", "DF", "strike", "T"))
    k, cp = np.log(K / F), np.where(q.option_type == "C", 1.0, -1.0)
    iv = {c: iv_norm(q[c].to_numpy() / (DF * F), k, T, cp) for c in ("mid", "bid", "ask")}
    s = iv["mid"] * np.sqrt(T)
    q = q.assign(kk=k, cpp=cp, iv_m=iv["mid"], iv_b=iv["bid"], iv_a=iv["ask"])
    q["hh"] = 0.5 * (q.iv_a - q.iv_b)
    q["ww"] = 1.0 / np.maximum(q.hh, H_MIN)
    q["dd1"] = -k / s + 0.5 * s
    q["band"] = np.isfinite(q.iv_m) & (q.mid >= 0.5) & (np.abs(q.dd1) <= Z5)
    q["thin25"] = own_thin(q, 25)
    return q


def own_thin(q, nt):
    """Eq. (6): per expiry, nearest unused in-band d1 to each of nt targets, plus put and call
    nearest k = 0; all quotes if <= nt."""
    keep, tg = np.zeros(len(q), bool), -Z5 + 2 * Z5 * np.arange(nt) / (nt - 1)
    band, d1, k, ex = q.band.to_numpy(), q.dd1.to_numpy(), q.kk.to_numpy(), q.expiry.to_numpy()
    for e in np.unique(ex[band]):
        idx = np.flatnonzero(band & (ex == e))
        if idx.size <= nt:
            keep[idx] = True
            continue
        used = np.zeros(idx.size, bool)
        for zm in tg:
            used[np.argmin(np.where(used, np.inf, np.abs(d1[idx] - zm)))] = True
        for side in (k[idx] < 0, k[idx] >= 0):
            if side.any():
                used[np.flatnonzero(side)[np.argmin(np.abs(k[idx][side]))]] = True
        keep[idx[used]] = True
    return keep


def arrays(mask, order=False):
    q = market()[mask]
    a = {c: q[c].to_numpy(float) for c in ("T", "kk", "cpp", "iv_m", "ww", "hh")}
    if order:  # the implementation's grid order (T, k), needed to replay its noise seeds
        o = np.lexsort((a["kk"], a["T"]))
        a = {c: v[o] for c, v in a.items()}
    a["vm"] = black_norm(a["kk"], a["T"], a["iv_m"], a["cpp"])
    a["vg"] = vega_norm(a["kk"], a["T"], a["iv_m"])
    return a


def heston_mask(nt=25, tmin=0.1):
    q = market()
    th = q.thin25.to_numpy() if nt == 25 else own_thin(q, nt)
    return th & (q["T"].to_numpy() >= tmin)


def residual_fn(a, kind="iv", target=None):
    """Eq. (8) with fallback (9), or eq. (10) (kind='price'), with my pricer and my IV."""
    sm = a["iv_m"] if target is None else target

    def f(x):
        v = heston_px(x, a["T"], a["kk"], a["cpp"])
        pr = a["ww"] * (v - a["vm"]) / a["vg"]
        if kind == "price":
            return pr
        s = iv_norm(v, a["kk"], a["T"], a["cpp"])
        return np.where(np.isfinite(s), a["ww"] * (s - sm), pr)

    return f


def own_fit(f, x0, box=NOTE_BOX):
    return least_squares(f, x0, bounds=box, x_scale="jac", diff_step=1e-6, ftol=1e-10, xtol=1e-10,
                         gtol=1e-10, max_nfev=500)  # fmt: skip


def own_starts(n_log2=4):
    """My own 16 Sobol starts on the note's ORIGINAL start box (11)."""
    u = qmc.Sobol(d=5, scramble=True, rng=np.random.default_rng(VSEED)).random_base2(n_log2)
    return np.column_stack([(0.08 + 0.32 * u[:, 0]) ** 2,
                            np.exp(np.log(0.3) + u[:, 1] * np.log(8 / 0.3)),
                            (0.08 + 0.32 * u[:, 2]) ** 2,
                            np.exp(np.log(0.2) + u[:, 3] * np.log(15.0)),
                            -0.95 + 0.75 * u[:, 4]])  # fmt: skip


ADVERSARIAL = np.array([[1e-3, 0.01, 1e-3, 4.5, -0.95], [0.5, 15, 0.5, 0.05, -0.05],
                        [2e-3, 0.05, 0.6, 3.0, -0.3], [0.3, 0.01, 0.01, 4.0, -0.9]])  # fmt: skip


def _real_fit_job(x0):
    r = own_fit(residual_fn(arrays(heston_mask())), x0)
    return r.x, r.cost, r.status, r.nfev


def _synthetic_job(args):
    """Fresh-seed synthetic round trip with my pricer (single start at p*), noise on my grid."""
    name, s = args
    xs = S1 if name == "S1" else S2
    a = arrays(heston_mask())
    st = heston_iv(xs, a["T"], a["kk"], a["cpp"])
    rng = np.random.default_rng(VSEED + 10_000 + (0 if name == "S1" else 1000) + s)
    eps = SIG_EPS * rng.standard_normal(st.size)
    obs = st + eps
    r = own_fit(lambda x: a["ww"] * (heston_iv(x, a["T"], a["kk"], a["cpp"]) - obs), xs, D034_BOX)
    return r.x, r.cost, 0.5 * np.sum((a["ww"] * eps) ** 2), np.sqrt(np.mean((r.fun / a["ww"]) ** 2))


def _impl_seed_job(s):
    """The implementer's 64 linearisation seeds (default_rng(20261207 + s), noise in its (T, k)
    grid order) refitted with MY pricer from p* = S1."""
    a = arrays(heston_mask(), order=True)
    st = heston_iv(S1, a["T"], a["kk"], a["cpp"])
    obs = st + SIG_EPS * np.random.default_rng(20261207 + s).standard_normal(st.size)
    return own_fit(lambda x: a["ww"] * (heston_iv(x, a["T"], a["kk"], a["cpp"]) - obs), S1,
                   D034_BOX).x  # fmt: skip


def pmap(fn, items):
    with ProcessPoolExecutor(WORKERS) as ex:
        return list(ex.map(fn, items))


# =========================================================================== F5 (own refit)
@lru_cache(maxsize=1)
def f5_refit() -> pd.DataFrame:
    """Parity pairs rebuilt from the Phase 0 clean chain (20 strikes nearest F0 = K* + y*, minus
    the table's rejected strikes), own OLS reproducing (F, DF), own WLS curve (2), refit (3)."""
    from qpricer.data import build_surface

    logging.getLogger("qpricer").setLevel(logging.ERROR)
    clean = build_surface()["clean"]
    tab = pd.read_csv(REPO / "data" / "processed" / f"market_{TAG}.csv")
    kept = tab[tab.kept.astype(bool)].sort_values("T").reset_index(drop=True)
    L = (kept["T"] >= 0.1).to_numpy()
    wt = 1 / (kept.r_se[L].to_numpy() ** 2 + 1e-6)
    A = np.column_stack([np.ones(L.sum()), kept["T"][L]])
    coef = np.linalg.solve(A.T @ (wt[:, None] * A), A.T @ (wt * kept.r[L].to_numpy()))
    rows = []
    for t in kept.itertuples():
        g = clean[(clean.expiry == t.expiry) & (clean.root == t.root)]
        c, p = (g[g.option_type == x].set_index("strike") for x in ("C", "P"))
        Ks = np.array(sorted(set(c.index) & set(p.index)))
        y = c.loc[Ks, "mid"].to_numpy() - p.loc[Ks, "mid"].to_numpy()
        h = 0.5 * ((c.loc[Ks, "ask"] - c.loc[Ks, "bid"]) + (p.loc[Ks, "ask"] - p.loc[Ks, "bid"]))
        i0 = np.argmin(np.abs(y))
        sel = np.sort(np.argsort(np.abs(Ks - (Ks[i0] + y[i0])), kind="stable")[:20])
        rej = [float(s) for s in str(t.rejected_K).split(";") if s not in ("", "nan")]
        sel = sel[~np.isin(Ks[sel], rej)]
        K, yy, hh = Ks[sel], y[sel], h.to_numpy()[sel]
        b, a0 = np.polyfit(K, yy, 1)
        dft = t.DF if t.T >= 0.1 else np.exp(-(coef[0] + coef[1] * t.T) * t.T)
        Ft = np.mean(K + yy / dft)
        rows.append({"expiry": t.expiry, "T": t.T, "F": t.F, "DF": t.DF, "F_ols": -a0 / b,
                     "DF_ols": -b, "DFt": dft, "Ft": Ft, "zF": (Ft - t.F) / t.F_se,
                     "pred": (t.F - K.mean()) * (t.DF - dft) / dft, "Kbar": K.mean(),
                     "maxrh": np.max(np.abs(yy - dft * (Ft - K)) / hh)})  # fmt: skip
    return pd.DataFrame(rows).assign(a=coef[0], b=coef[1])


# =========================================================================== Phase 4 validator
@lru_cache(maxsize=1)
def p4():
    spec = importlib.util.spec_from_file_location(
        "p4_validator", REPO / "tests" / "validation" / "test_phase4_independent.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def autocall_refs(name, n_paths=1_000_000, dt=1 / 64, seed=VSEED + 70_001, s_scale=1.0):
    """Exact V_BS, V_TS, no-call values and OQEM V_H for the calibrated set (AC-Q / AC-A)."""
    m = p4()
    x, _, _ = fit_json()
    v0, ka, th, xi, rho = x
    prm = (ka, th, xi, rho, v0)
    t = np.arange(1, 7) / 4 if name == "AC-Q" else np.array([1.0, 2.0, 3.0])
    cp = 0.08 * np.diff(t, prepend=0.0)
    F0, D0, s0, _ = m.market_ref(t)
    f, T = F0 / s0 * s_scale, t[-1]
    sig = float(iv_norm(lewis_norm(np.zeros(1), T, x, np.ones(1)), 0.0, T, 1.0)[0])
    wts = th * t + (v0 - th) * (1 - np.exp(-ka * t)) / ka
    out = {"sigma_atm": sig, "V_BS": m.acx(t, cp, 1.0, 0.7, 0.6, True, f, D0, sig**2 * t),
           "V_TS": m.acx(t, cp, 1.0, 0.7, 0.6, True, f, D0, wts)}  # fmt: skip
    Dc = np.array([m.gp_below(0.7 / fi, ti, prm) for fi, ti in zip(f, t, strict=True)])
    dip = D0[-1] * (m.gp_put(f[-1], 0.6, T, prm) + 0.4 * m.gp_below(0.6 / f[-1], T, prm))
    out["nc_H"] = float(np.sum(D0 * cp * (1 - Dc)) + D0[-1] - dip)
    out["nc_TS"] = m.acx(t, cp, np.inf, 0.7, 0.6, False, f, D0, wts)
    out["term1"], out["term2"] = out["V_TS"] - out["V_BS"], out["nc_H"] - out["nc_TS"]
    xh = m.oqem(prm, t, dt, n_paths, np.random.default_rng(seed))
    pv = m.own_pv(f[:, None] * np.exp(xh), cp, 1.0, 0.7, 0.6, True, D0)
    out["V_H"], out["se_V_H"] = float(pv.mean()), float(pv.std(ddof=1) / np.sqrt(n_paths))
    out["gap"] = out["V_H"] - out["V_BS"]
    out["term3"] = out["V_H"] - out["V_TS"] - out["term2"]
    out["_pv"] = pv
    return out


def impl_autocall():
    return pd.read_csv(REPO / "report" / "figures" / f"autocall_attribution_heston_fit_{TAG}.csv")


# =========================================================================== tests: references
def test_cf_little_trap_vs_own_riccati_rk4():
    x, _, _ = fit_json()
    u = np.array([0.3, 2.0, 7.0, 25.0, 80.0]) - 0.5j
    for p in (x, S1, S2):
        for T in (0.11, 1.61):
            assert np.abs(heston_cf(u, T, *p) - heston_cf_rk4(u, T, *p)).max() < 1e-12


def test_own_lewis_quadrature_converged():
    """Doubling the cutoff and halving the panels changes prices by < 1e-14 at p^, S1, S2."""
    x, _, _ = fit_json()
    k = np.linspace(-0.6, 0.3, 27)
    cp = np.where(k < 0, -1.0, 1.0)
    for p in (x, S1, S2):
        for T in (0.114, 0.5, 1.61):
            U = cutoff(T, p)
            v = lewis_norm(k, T, p, cp)
            assert np.abs(v - lewis_norm(k, T, p, cp, U=2 * U)).max() < 1e-14
            assert np.abs(v - lewis_norm(k, T, p, cp, h_fine=0.125, h=0.5)).max() < 1e-14


# =========================================================================== tests: F5 (D-033)
def test_f5_closure_own_refit():
    """Own pairs reproduce Phase 0 (F, DF) to 1e-9; WLS curve 4.527% + 0.304% T; F~, DF~ equal
    the implementation's to 1e-9; max |z_F| = 3.56 on 2026-12-03 (erratum E3), 4 expiries above 2;
    E3's mechanism F~ - F = (F - Kbar)(DF - DF~)/DF~ holds exactly; max |res|/h 0.834;
    DF rises 10 -> 4; S_ref moves by +0.0745% (OPEN_ISSUES S_ref entry)."""
    d = f5_refit()
    assert np.allclose(d.F_ols, d.F, rtol=1e-9) and np.allclose(d.DF_ols, d.DF, rtol=1e-9)
    assert abs(d.a[0] - 0.04527) < 5e-6 and abs(d.b[0] - 0.00304) < 5e-6
    impl = pd.read_csv(REPO / "report" / f"calibration_f5_{TAG}.csv").set_index("expiry")
    di = d.set_index("expiry")
    assert np.allclose(di.Ft, impl.F_tilde.loc[di.index], rtol=1e-12)
    assert np.allclose(di.DFt, impl.DF_tilde.loc[di.index], rtol=1e-15)
    assert np.allclose(di.Ft - di.F, di.pred, atol=1e-8)
    assert di.zF.abs().idxmax() == "2026-12-03" and abs(di.zF.abs().max() - 3.562) < 1e-3
    assert set(di.index[di.zF.abs() > 2]) == {"2026-11-18", "2026-11-19", "2026-12-02",
                                               "2026-12-03"}  # fmt: skip
    assert abs(d.maxrh.max() - 0.8339) < 1e-4
    assert (np.diff(d.DF) > 0).sum() == 10 and (np.diff(d.DFt) > 0).sum() == 4
    rel = d.Ft[0] * d.DFt[0] / (d.F[0] * d.DF[0]) - 1
    assert abs(rel - 7.447e-4) < 1e-6


# =========================================================================== tests: targets, loss
def test_targets_own_ivs_band_and_thinning():
    q = market()
    assert np.abs(q.iv_m - q.sigma_m).max() < 1e-13
    assert np.abs(q.iv_b - q.sigma_b).max() < 1e-13 and np.abs(q.iv_a - q.sigma_a).max() < 1e-13
    assert ((q.cpp > 0) == (q.strike >= q.F)).all()  # OTM type vs F~
    assert (q.band == q.band_impl).all() and q.band.sum() == 5806
    assert (q.thin25 == q.thin_impl).all() and q.thin25.sum() == 1029
    assert heston_mask().sum() == 565
    assert abs((q.hh[q.thin25] < H_MIN).mean() - 0.609) < 1e-3


def test_loss_ses_and_diagnostics_at_reported_optimum():
    """Own cost at p^ = 2062.8978 (rel 1e-10), 0 fallbacks; own Gauss-Newton SEs = JSON SEs
    (rel 1e-4); cond(J diag p) 25.6; Feller 0.178; GN step from p^ < 1e-4 SE (stationary);
    interior: >= 38 SE from every D-034 bound."""
    x, se, d = fit_json()
    a = arrays(heston_mask())
    f = residual_fn(a)
    assert np.all(np.isfinite(heston_iv(x, a["T"], a["kk"], a["cpp"])))
    r = f(x)
    assert abs(0.5 * r @ r / d["cost"] - 1) < 1e-10
    J = fd_jac(f, x)
    C = (r @ r / (r.size - 5)) * np.linalg.inv(J.T @ J)
    assert np.allclose(np.sqrt(np.diag(C)), se, rtol=1e-4)
    assert abs(np.linalg.cond(J * np.abs(x)) - 25.57) < 0.05
    assert abs(2 * x[1] * x[2] / x[3] ** 2 - 0.1784) < 1e-4
    assert np.all(np.abs(np.linalg.solve(J.T @ J, -J.T @ r)) < 1e-4 * se)
    assert np.min(np.r_[(x - D034_BOX[0]) / se, (D034_BOX[1] - x) / se]) > 38


def test_rmse_table_independent_and_nan_free():
    """Per-expiry Heston RMSE (my pricer + my IV) = CSV to 1e-8 vol pt; 39 rows; in sample iff
    T >= 0.1 (21) and out of sample otherwise (18); no NaN anywhere; pooled 0.328 / 1.379."""
    x, _, _ = fit_json()
    q = market()[market().thin25]
    s = heston_iv(x, q["T"].to_numpy(), q.kk.to_numpy(), q.cpp.to_numpy())
    assert np.all(np.isfinite(s))
    d = q.assign(res=s - q.iv_m)
    rm = d.groupby("expiry").res.apply(lambda r: 100 * np.sqrt(np.mean(r**2)))
    et = pd.read_csv(REPO / "report" / f"calibration_expiry_{TAG}.csv").set_index("expiry")
    assert len(et) == 39 and not et.select_dtypes("number").isna().any().any()
    assert np.abs(rm - et.heston_rmse_volpt.loc[rm.index]).max() < 1e-8
    assert ((et["T"] >= 0.1) == (et["sample"] == "in")).all() and (et["sample"] == "in").sum() == 21
    ins = d["T"] >= 0.1
    assert abs(100 * np.sqrt(np.mean(d.res[ins] ** 2)) - 0.3278) < 1e-4
    assert abs(100 * np.sqrt(np.mean(d.res[~ins] ** 2)) - 1.3790) < 1e-4


def test_impl_cos_equals_own_lewis_at_optimum_all_expiries():
    """System under test calibration.heston_iv (pinned COS) vs my Lewis + my IV at p^ on all
    1029 thinned quotes (in and out of sample): < 1e-10."""
    from qpricer import calibration as cal

    x, _, _ = fit_json()
    q = market()[market().thin25]
    g = cal.make_grid(q["T"], q.kk, q.cpp, q.iv_m, q.ww)
    mine = heston_iv(x, g["T"], g["k"], g["cp"])
    assert np.abs(cal.heston_iv(x, g) - mine).max() < 1e-10


def test_d034_cos_claims_vs_own_lewis():
    """D-034 / erratum E1 with my Lewis: (N, L) = (1024, 12) at (0.03, 2, 0.04, 1, -0.7) has an IV
    error 9.9e-5 > 1e-5 (the note's setting fails its own post-fit check); the pinned (16384, 64)
    at the worst corner (0.01, 0.5, 0.01, 3, -0.99) has e_vega = 9.2e-6 <= 1e-5 but a direct IV
    error of 2.5e-3 (only meaningful where the model vega is tiny)."""
    from qpricer.engines.fourier import cos_price
    from qpricer.models.heston import HestonParams

    a = arrays(heston_mask())

    def gaps(x, N, L):
        p = HestonParams(kappa=x[1], theta=x[2], xi=x[3], rho=x[4], v0=x[0])
        vc = np.empty(a["T"].size)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for t in np.unique(a["T"]):
                m = a["T"] == t
                vc[m] = cos_price(1.0, np.exp(a["kk"][m]), t, 1.0, p, a["cpp"][m], N=N, L=L)
        vl = heston_px(x, a["T"], a["kk"], a["cpp"])
        sc, sl = (iv_norm(v, a["kk"], a["T"], a["cpp"]) for v in (vc, vl))
        both = np.isfinite(sc) & np.isfinite(sl)
        return np.max(np.abs(vc - vl) / a["vg"]), np.max(np.abs(sc - sl)[both])

    ev, ei = gaps([0.03, 2, 0.04, 1, -0.7], 1024, 12)
    assert 9.5e-5 < ei < 1.05e-4
    ev, ei = gaps([0.01, 0.5, 0.01, 3, -0.99], 16384, 64)
    assert 9.0e-6 < ev <= 1e-5 and ei > 1e-3


# =========================================================================== tests: synthetic
def test_synthetic_precheck_ses_independent():
    """Eq. (17) SEs with my pricer/Jacobian equal D-035 (rel 2e-3); pre-check (18) holds in the
    D-034 box and also in the note's original box."""
    a = arrays(heston_mask())
    ref = {"S1": [1.269e-4, 4.959e-2, 2.947e-4, 4.435e-3, 4.432e-3],
           "S2": [1.632e-4, 9.416e-2, 3.283e-4, 6.555e-3, 6.989e-3]}  # fmt: skip
    for name, xs in (("S1", S1), ("S2", S2)):
        se = np.sqrt(np.diag(sandwich(lambda x: heston_iv(x, a["T"], a["kk"], a["cpp"]), xs,
                                      a["ww"])))  # fmt: skip
        assert np.allclose(se, ref[name], rtol=2e-3)
        for lo, hi in (D034_BOX, NOTE_BOX):
            assert np.all((xs - 3 * se >= lo) & (xs + 3 * se <= hi) & (3 * se <= 0.5 * np.abs(xs)))


SABR_SETS = {"2026-11-13": (0.15, -0.7, 2.0), "2026-12-18": (0.15, -0.7, 1.2),
             "2027-05-21": (0.17, -0.7, 0.7), "2028-06-16": (0.19, -0.7, 0.45)}  # fmt: skip


def sabr_grid(expiry):
    q = market()
    return arrays((q.expiry == expiry).to_numpy() & q.thin25.to_numpy())


def test_sabr_hagan_own_vs_impl_and_precheck():
    """Own Hagan = calibration.sabr_iv to 1e-13 on the four eq. (20) grids; own sandwich SEs equal
    D-035's SABR SEs (rel 1e-2)."""
    from qpricer import calibration as cal

    ref = {
        "2026-11-13": (9.9e-4, 6.22e-2, 1.577e-1),
        "2026-12-18": (1.06e-3, 4.19e-2, 5.92e-2),
        "2027-05-21": (1.09e-3, 3.11e-2, 2.66e-2),
        "2028-06-16": (1.22e-3, 2.78e-2, 1.72e-2),
    }
    for e, qs in SABR_SETS.items():
        a = sabr_grid(e)
        g = cal.make_grid(a["T"], a["kk"], a["cpp"], a["iv_m"], a["ww"])
        assert np.abs(cal.sabr_iv(qs, g) - hagan_b1(g["k"], g["T"], *qs)).max() < 1e-13
        se = np.sqrt(np.diag(sandwich(lambda q, a=a: hagan_b1(a["kk"], a["T"], *q), qs, a["ww"])))
        assert np.allclose(se, ref[e], rtol=1e-2)


@pytest.mark.slow
def test_sabr_synthetic_fresh_seeds():
    """Eq. (20) G2 with my Hagan and fresh seeds (200 per set): false-fail rate of
    {any |z| > 3 or D^2 > chi2_3(0.999)} <= 3% per set (nominal 0.9%); D^2 mean in [2.5, 3.5]."""
    for j, (e, qs) in enumerate(SABR_SETS.items()):
        a = sabr_grid(e)
        f = lambda q, a=a: hagan_b1(a["kk"], a["T"], *q)  # noqa: E731
        C = sandwich(f, qs, a["ww"])
        se, fails, D2s = np.sqrt(np.diag(C)), 0, []
        for s in range(200):
            rng = np.random.default_rng(VSEED + 20_000 + 1000 * j + s)
            obs = f(qs) + SIG_EPS * rng.standard_normal(a["T"].size)
            r = least_squares(lambda q, a=a, f=f, obs=obs: a["ww"] * (f(q) - obs), qs,
                              bounds=([1e-4, -0.999, 1e-4], [5, 0.999, 5]), x_scale="jac",
                              ftol=1e-12, xtol=1e-12, gtol=1e-12)  # fmt: skip
            dq = r.x - np.array(qs)
            D2 = float(dq @ np.linalg.solve(C, dq))
            D2s.append(D2)
            fails += bool(np.any(np.abs(dq) > 3 * se) or D2 > chi2.ppf(0.999, 3))
        assert fails <= 6, (e, fails)
        assert 2.5 <= np.mean(D2s) <= 3.5, (e, np.mean(D2s))


@pytest.mark.slow
def test_heston_synthetic_fresh_seeds_z_and_d2():
    """100 fresh seeds per set, my pricer, D-034 box, single start at p*: G2 false-fail <= 4%
    per set (stated 1.44%), sd(z_j) in [0.73, 1.27], D^2 mean in [4, 6], G3 always, G4b >= 95%."""
    out = pmap(_synthetic_job, [(n, s) for n in ("S1", "S2") for s in range(100)])
    a = arrays(heston_mask())
    n = a["T"].size
    tgt, tol = SIG_EPS * np.sqrt((n - 5) / n), 3 * SIG_EPS / np.sqrt(2 * n)
    for i, xs in enumerate((S1, S2)):
        rows = out[100 * i : 100 * (i + 1)]
        C = sandwich(lambda x: heston_iv(x, a["T"], a["kk"], a["cpp"]), xs, a["ww"])
        se = np.sqrt(np.diag(C))
        dx = np.array([r[0] for r in rows]) - xs
        z, D2 = dx / se, np.einsum("ij,ij->i", dx, np.linalg.solve(C, dx.T).T)
        fail = (np.abs(z) > 3).any(1) | (D2 > chi2.ppf(0.999, 5))
        assert fail.mean() <= 0.04
        assert np.all((z.std(0, ddof=1) >= 0.73) & (z.std(0, ddof=1) <= 1.27))
        assert 4.0 <= D2.mean() <= 6.0
        assert all(r[1] <= r[2] for r in rows)
        assert np.mean([abs(r[3] - tgt) <= tol for r in rows]) >= 0.95


@pytest.mark.slow
def test_implementer_linearisation_seeds_reproduced_with_own_pricer():
    """The implementer's 64 seeds refitted with my pricer give its sd ratios [1.10, 1.20, 0.98,
    1.13, 1.00] to 0.01 (so the kappa ratio 1.20 is that seed set's sampling noise: 100 fresh
    seeds give 0.85, see test above / report)."""
    X = np.array(pmap(_impl_seed_job, range(64)))
    a = arrays(heston_mask(), order=True)
    se = np.sqrt(np.diag(sandwich(lambda x: heston_iv(x, a["T"], a["kk"], a["cpp"]), S1,
                                  a["ww"])))  # fmt: skip
    assert np.allclose(X.std(0, ddof=1) / se, [1.10, 1.199, 0.977, 1.127, 1.004], atol=0.01)


# =========================================================================== tests: real fit
@pytest.mark.slow
def test_real_fit_global_on_note_original_box():
    """D-034 check: my pricer, the NOTE'S ORIGINAL bounds box (11), 16 own Sobol starts on the
    original start box + 4 adversarial starts at its hard corners (tiny v0/theta, xi up to 4.5,
    kappa = 0.01): every start converges to the reported p^ (|dp| < 1e-3 SE) and cost
    (rel 1e-9). The narrowed box does not change the fit."""
    x, se, d = fit_json()
    out = pmap(_real_fit_job, list(own_starts()) + list(ADVERSARIAL))
    for xx, cost, status, _ in out:
        assert status in (1, 2, 3, 4)
        assert abs(cost / d["cost"] - 1) < 1e-9
        assert np.all(np.abs(xx - x) < 1e-3 * se)


@pytest.mark.slow
def test_sensitivity_runs_independent():
    """D-037 sensitivity numbers with my pricer/thinning: T >= 2 weeks moves kappa, xi by
    +12.97, +12.90 SE; n_t = 50 by <= 1.06 SE; objective (10) within 0.84 SE (all to 0.02 SE)."""
    x, se, _ = fit_json()
    cases = (("T>=2w", heston_mask(25, 14 / 365), "iv", [3.89, 12.97, -2.07, 12.90, -1.57]),
             ("n_t=50", heston_mask(50), "iv", [-0.95, -0.76, -0.12, -1.06, 0.09]),
             ("obj10", heston_mask(), "price", [-0.83, -0.46, 0.15, -0.79, -0.79]))  # fmt: skip
    for _, m, kind, ref in cases:
        f = residual_fn(arrays(m), kind)
        r = min((own_fit(f, x0) for x0 in (x, [0.03, 1.0, 0.08, 0.6, -0.5])), key=lambda r: r.cost)
        assert np.allclose((r.x - x) / se, ref, atol=0.02)


# =========================================================================== tests: SABR real
def test_sabr_real_fits_independent():
    """Own Hagan fits (9 own starts, no eq. 9 anywhere) on all 39 expiries reproduce the CSV:
    RMSE to 1e-6 vol pt, (alpha, rho, nu) to 1e-6; every expiry fitted; nu^2 T in
    [0.3515, 0.8901], all 39 above 0.16; no bound active."""
    q = market()[market().thin25]
    et = pd.read_csv(REPO / "report" / f"calibration_expiry_{TAG}.csv").set_index("expiry")
    nu2T = []
    for e, g in q.groupby("expiry"):
        k, T, sm, w = g.kk.to_numpy(), float(g["T"].iloc[0]), g.iv_m.to_numpy(), g.ww.to_numpy()
        best = None
        for rho0 in (-0.9, -0.6, -0.2):
            for nu0 in (0.3, 1.0, 3.0):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = least_squares(lambda z, w=w, k=k, T=T, sm=sm: w * (hagan_b1(k, T, *z) - sm),
                                      [np.interp(0, k, sm), rho0, nu0],
                                      bounds=([1e-4, -0.999, 1e-4], [5, 0.999, 5]), x_scale="jac",
                                      ftol=1e-12, xtol=1e-12, gtol=1e-12)  # fmt: skip
                if np.all(np.isfinite(r.fun)) and (best is None or r.cost < best.cost):
                    best = r
        E = et.loc[e]
        assert abs(100 * np.sqrt(np.mean((best.fun / w) ** 2)) - E.sabr_rmse_volpt) < 1e-6
        assert np.allclose(best.x, [E.sabr_alpha, E.sabr_rho, E.sabr_nu], atol=1e-6)
        assert np.all(best.active_mask == 0)
        nu2T.append(best.x[2] ** 2 * T)
    assert len(nu2T) == 39 and min(nu2T) > 0.16
    assert abs(min(nu2T) - 0.3516) < 1e-3 and abs(max(nu2T) - 0.8900) < 1e-3


# =========================================================================== tests: arbitrage
def test_arbitrage_checks_independent():
    """Own calendar (in-band quotes, consecutive expiries, common k-range, linear w) and
    butterfly (all surface quotes, undiscounted calls): 11054 points / 0 mid / 0 strict;
    7547 triples / 2126 mid / 0 strict / max z 0.427."""
    q = market()
    b = q[q.band]
    ex = b.groupby("expiry")["T"].first().sort_values().index
    npts = nmid = nstr = 0
    for e1, e2 in zip(ex[:-1], ex[1:], strict=True):
        A, B = (b[b.expiry == e].sort_values("kk") for e in (e1, e2))
        lo, hi = max(A.kk.min(), B.kk.min()), min(A.kk.max(), B.kk.max())
        kk = np.unique(np.r_[A.kk, B.kk])
        kk = kk[(kk >= lo) & (kk <= hi)]
        wA = {c: np.interp(kk, A.kk, A[f"iv_{c}"] ** 2 * A["T"]) for c in "mba"}
        wB = {c: np.interp(kk, B.kk, B[f"iv_{c}"] ** 2 * B["T"]) for c in "mba"}
        npts += kk.size
        nmid += int((wB["m"] < wA["m"]).sum())
        nstr += int((wB["a"] < wA["b"]).sum())
    assert (npts, nmid, nstr) == (11054, 0, 0)
    nt = nm = ns = 0
    zmax = 0.0
    for _, g in q.groupby("expiry"):
        g = g.sort_values("strike")
        K, F, D = g.strike.to_numpy(), g.F.iloc[0], g.DF.iloc[0]
        c = {x: g[x].to_numpy() / D + np.maximum(F - K, 0) for x in ("mid", "bid", "ask")}
        lam = (K[2:] - K[1:-1]) / (K[2:] - K[:-2])
        Bm = lam * c["mid"][:-2] + (1 - lam) * c["mid"][2:] - c["mid"][1:-1]
        Bs = lam * c["ask"][:-2] + (1 - lam) * c["ask"][2:] - c["bid"][1:-1]
        sp = c["ask"] - c["bid"]
        hB = 0.5 * (lam * sp[:-2] + sp[1:-1] + (1 - lam) * sp[2:])
        nt, nm, ns = nt + Bm.size, nm + int((Bm < 0).sum()), ns + int((Bs < 0).sum())
        zmax = max(zmax, float((-Bm / hB).max()))
    assert (nt, nm, ns) == (7547, 2126, 0) and abs(zmax - 0.4268) < 1e-3


# =========================================================================== tests: Phase 4 gate
@pytest.mark.slow
@pytest.mark.parametrize("name", ["AC-Q", "AC-A"])
def test_autocall_calibrated_gap_independent(name):
    """Calibrated autocall vs my references: sigma_ATM (Lewis + own IV) 1e-9; V_BS, V_TS (exact)
    within 4 SE of the CSV MC values; term2 (Gil-Pelaez vs exact) 1e-8; term1 within 4 CRN SE;
    V_H (OQEM, 1e6 paths, dt 1/64) within 4 combined SE; dt 1/256 within 3 combined SE of dt 1/64
    (no visible QE time-step bias); gap within 4 SE; the S_ref fix moves the gap by < 0.5 CRN SE."""
    r = impl_autocall().set_index("sheet").loc[name]
    col = {c.split(":")[0]: c for c in r.index}
    ref = autocall_refs(name)
    assert abs(ref["sigma_atm"] - r.sigma_atm) < 1e-9
    assert abs(r.V_BS - ref["V_BS"]) <= 4 * r.se_V_BS and abs(r.V_TS - ref["V_TS"]) <= 4 * r.se_V_TS
    assert abs(r[col["term2"]] - ref["term2"]) < 1e-8
    assert abs(r[col["term1"]] - ref["term1"]) <= 4 * r[col["se_term1"]]
    assert abs(r.V_H - ref["V_H"]) <= 4 * np.hypot(r.se_V_H, ref["se_V_H"])
    assert abs(r.gap - ref["gap"]) <= 4 * np.hypot(r.se_gap, np.hypot(ref["se_V_H"], r.se_V_BS))
    fine = autocall_refs(name, 400_000, 1 / 256, VSEED + 70_002)
    assert abs(fine["V_H"] - ref["V_H"]) <= 3 * np.hypot(fine["se_V_H"], ref["se_V_H"])
    sh = autocall_refs(name, s_scale=7374.4254632990705 / 7379.917502678466)
    dgap = (sh["_pv"] - ref["_pv"]).mean() - (sh["V_BS"] - ref["V_BS"])
    assert abs(dgap) < 0.5 * r.se_gap


# =========================================================================== report script
def _main() -> None:  # pragma: no cover
    t0 = time.time()
    x, se, d = fit_json()
    print("p^", x, "SE", se)
    a = arrays(heston_mask())
    f = residual_fn(a)
    r = f(x)
    J = fd_jac(f, x)
    C = (r @ r / (r.size - 5)) * np.linalg.inv(J.T @ J)
    print("own cost", 0.5 * r @ r, "own SE", np.sqrt(np.diag(C)), "cond", np.linalg.cond(J * x))
    print("dist to D-034 bounds (SE):", (x - D034_BOX[0]) / se, (D034_BOX[1] - x) / se)
    out = pmap(_real_fit_job, list(own_starts()) + list(ADVERSARIAL))
    for xx, c, st, nf in out:
        print("  original box start ->", np.round(xx, 6), round(c, 6), st, nf)
    syn = pmap(_synthetic_job, [(n, s) for n in ("S1", "S2") for s in range(100)])
    n = a["T"].size
    tgt, tol = SIG_EPS * np.sqrt((n - 5) / n), 3 * SIG_EPS / np.sqrt(2 * n)
    for i, (nm, xs) in enumerate((("S1", S1), ("S2", S2))):
        rows = syn[100 * i : 100 * (i + 1)]
        Cs = sandwich(lambda z: heston_iv(z, a["T"], a["kk"], a["cpp"]), xs, a["ww"])
        ses = np.sqrt(np.diag(Cs))
        dx = np.array([q[0] for q in rows]) - xs
        z, D2 = dx / ses, np.einsum("ij,ij->i", dx, np.linalg.solve(Cs, dx.T).T)
        print(nm, "SE", ses, "sd(z)", z.std(0, ddof=1).round(3), "mean z", z.mean(0).round(3))
        print("   any|z|>3", (np.abs(z) > 3).any(1).mean(), "D2>20.52", (D2 > 20.515).mean(),
              "D2 mean", D2.mean().round(3), "max", D2.max().round(2), "G4b",
              np.mean([abs(q[3] - tgt) <= tol for q in rows]))  # fmt: skip
    for name in ("AC-Q", "AC-A"):
        ref = autocall_refs(name)
        print(name, {k: round(v, 6) for k, v in ref.items() if not k.startswith("_")})
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "tests"))
    _main()
