"""Independent validation of Phase 4 (exotics: barrier, Asian, autocallable) -- validator.

The qpricer public API (``barrier_price``, ``bgk_barrier``, ``barrier_mc``, ``geometric_asian_*``,
``asian_averages``, ``asian_mc``, ``control_variate``, ``Autocall``, ``autocall_flows``,
``autocall_pv``, ``autocall_mc``, ``autocall_nocall``, ``autocall_two_dates``,
``ratio_marginals``, ``fixings``, ``fixing_grid``, ``gbm_paths``, ``forward_curve`` and
``scripts/autocall_compare.py``) is only called as the system under test. Every reference below
is my own code, built with a method that differs from the implementation:

* RRS  -- Reiner-Rubinstein in Haug's *spot* form (S, X, H, r, b; Haug 2007 §4.17.1), transcribed
          independently (the implementation is in forward form, with log_ndtr).
* PDE  -- Crank-Nicolson (4 Rannacher half-steps) on y = eta ln(S/H), absorbing barrier with the
          rebate as boundary value, Black far boundary, cell-averaged payoff, S0 on a node, one
          Richardson step (n, 2n). No reflection principle anywhere. In-options by in-out parity
          with a second PDE for Q(tau > T).
* DQ   -- discrete-monitoring barrier by backward quadrature: Gauss-Legendre on each smooth piece of
          the alive region + cubic splines of the continuation value (no MC, no BGK).
* GEO  -- discrete geometric Asian from the full covariance double sum sigma^2 min(t_i, t_j).
* KV   -- Kemna-Vorst continuous geometric call as printed in Haug (2007) §4.20.1.
* ARQ  -- arithmetic Asian by density recursion of Z_k = ln Y_k, Y_k = R_k (1 + Y_{k+1})
          (Carverhill-Clewlow / Fusai-Meucci type), banded trapezoid convolution, last step by
          Black (smooth in Z_2). Checked against 1-D quad for n = 2 (1e-12).
* ACX  -- autocallable under any deterministic-variance lognormal model by backward induction over
          dates with the memory state (index of the last paid coupon); GL per smooth piece between
          barrier levels, cubic splines of the continuation values.
* LOOP -- the term sheet of autocallable.md §1 transcribed path by path in plain Python.
* GP   -- own Heston CF (little-trap form) and Gil-Pelaez inversion (digital and share digital),
          composite Gauss-Legendre; Heston implied vol by brentq on my own Black.
* OQEM -- own Andersen (2008) QE-M with sub-steps ending on each date and a different RNG layout
          (uniform for the variance, then a normal for ln F).

Run as a script to print the numbers quoted in ``docs/validation/phase4.md``:
``.venv/bin/python tests/validation/test_phase4_independent.py`` (about 3 minutes); add
``--scan`` for the fresh-seed z distributions (G2 BSM 50 seeds, G2h 300 seeds, G2m 25 seeds per
step, about 20 minutes). ``--golden-out PATH`` writes the Haug-table golden JSON to PATH only if
PATH does not exist; ``tests/golden/barrier_haug.json`` was created from that output.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sps
from numpy.polynomial.legendre import leggauss
from scipy import integrate
from scipy.interpolate import CubicSpline
from scipy.linalg import solve_banded
from scipy.optimize import brentq, least_squares
from scipy.stats import norm

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
    fixing_grid,
    fixings,
    gbm_paths,
)
from qpricer.market import forward_curve
from qpricer.models.heston import HestonParams
from qpricer.products import Autocall, autocall_flows

REPO = Path(__file__).resolve().parents[2]
GOLDEN = REPO / "tests" / "golden" / "barrier_haug.json"
MARKET = REPO / "data" / "processed" / "market_20261106_2213.csv"
FIG = REPO / "report" / "figures"
VSEED = 4_040_000  # validator seed base (implementer: 2026xxxx); never re-seeded after a look
Nn = norm.cdf

GS_T = (1.3253, 0.0354, 0.3877, -0.7165, 0.0354)  # (kappa, theta, xi, rho, v0), D-029
FO_T = (1.5768, 0.0398, 0.5751, -0.5711, 0.0175)
GS, FO = HestonParams(*GS_T), HestonParams(*FO_T)


# =========================================================================== references: barrier
def rrs(S, X, H, T, r, b, v, phi, eta, knock, K=0.0):
    """RRS: Haug (2007) §4.17.1 standard barrier, spot form. X strike, K rebate."""
    S, X, H = (np.asarray(a, dtype=float) for a in (S, X, H))
    st = v * np.sqrt(T)
    mu = (b - v * v / 2) / (v * v)
    lam = np.sqrt(mu * mu + 2 * r / (v * v))
    x1 = np.log(S / X) / st + (1 + mu) * st
    x2 = np.log(S / H) / st + (1 + mu) * st
    y1 = np.log(H * H / (S * X)) / st + (1 + mu) * st
    y2 = np.log(H / S) / st + (1 + mu) * st
    z = np.log(H / S) / st + lam * st
    cb, dr, hs = np.exp((b - r) * T), np.exp(-r * T), H / S
    A = phi * S * cb * Nn(phi * x1) - phi * X * dr * Nn(phi * x1 - phi * st)
    B = phi * S * cb * Nn(phi * x2) - phi * X * dr * Nn(phi * x2 - phi * st)
    C = phi * S * cb * hs ** (2 * (mu + 1)) * Nn(eta * y1) - phi * X * dr * hs ** (2 * mu) * Nn(
        eta * y1 - eta * st
    )
    D = phi * S * cb * hs ** (2 * (mu + 1)) * Nn(eta * y2) - phi * X * dr * hs ** (2 * mu) * Nn(
        eta * y2 - eta * st
    )
    E = K * dr * (Nn(eta * x2 - eta * st) - hs ** (2 * mu) * Nn(eta * y2 - eta * st))
    F = K * (hs ** (mu + lam) * Nn(eta * z) + hs ** (mu - lam) * Nn(eta * z - 2 * eta * lam * st))
    hi = X > H
    if knock == "in":
        tab = {
            (1, 1): np.where(hi, C + E, A - B + D + E),
            (1, -1): np.where(hi, A + E, B - C + D + E),
            (-1, 1): np.where(hi, B - C + D + E, A + E),
            (-1, -1): np.where(hi, A - B + D + E, C + E),
        }
    else:
        tab = {
            (1, 1): np.where(hi, A - C + F, B - D + F),
            (1, -1): np.where(hi, F, A - B + C - D + F),
            (-1, 1): np.where(hi, A - B + C - D + F, F),
            (-1, -1): np.where(hi, B - D + F, A - C + F),
        }
    return tab[(phi, eta)]


def black_spot(S, X, T, r, b, v, phi):
    st = v * np.sqrt(T)
    d1 = (np.log(S / X) + (b + v * v / 2) * T) / st
    return phi * (S * np.exp((b - r) * T) * Nn(phi * d1) - X * np.exp(-r * T) * Nn(phi * (d1 - st)))


def _cn(V, bnd0, far_fn, a, c, r, dx, T, nt, rannacher=4):
    """Crank-Nicolson on a uniform grid, Dirichlet bnd0 at node 0 and far_fn(tau) at node n."""
    lo, di, up = a / dx**2 - c / (2 * dx), -2 * a / dx**2 - r, a / dx**2 + c / (2 * dx)
    m, dt, tau = V.size - 2, T / nt, 0.0
    steps = [(dt / 2, 1.0)] * rannacher + [(dt, 0.5)] * (nt - rannacher // 2)
    for h, th in steps:
        tau += h
        far = far_fn(tau)
        rhs = V[1:-1] + (1 - th) * h * (lo * V[:-2] + di * V[1:-1] + up * V[2:])
        rhs[0] += th * h * lo * bnd0
        rhs[-1] += th * h * up * far
        ab = np.zeros((3, m))
        ab[0, 1:], ab[1, :], ab[2, :-1] = -th * h * up, 1 - th * h * di, -th * h * lo
        V = np.concatenate(([bnd0], solve_banded((1, 1), ab, rhs), [far]))
    return V


def pde_out(S0, X, H, T, r, b, v, phi, eta, R, n0, nt, width=10.0):
    """PDE: knock-out with rebate R paid at the hit. S0 on node n0."""
    y0 = eta * np.log(S0 / H)
    dx = y0 / n0
    n = int(np.ceil((y0 + width * v * np.sqrt(T)) / dx))
    y = np.arange(n + 1) * dx
    lo_, hi_ = y - dx / 2, y + dx / 2  # exact cell average of (phi (H e^{eta y} - X))^+
    ys = eta * np.log(X / H)
    a_, b_ = (np.maximum(lo_, ys), hi_) if phi * eta > 0 else (lo_, np.minimum(hi_, ys))
    ok = b_ > a_
    a_, b_ = np.where(ok, a_, 0.0), np.where(ok, b_, 0.0)
    V = np.where(ok, phi * (H / eta * (np.exp(eta * b_) - np.exp(eta * a_)) - X * (b_ - a_)), 0)
    V = V / dx
    V[0] = R
    S_far = H * np.exp(eta * y[-1])
    V = _cn(
        V,
        R,
        lambda tau: black_spot(S_far, X, tau, r, b, v, phi),
        0.5 * v * v,
        eta * (b - 0.5 * v * v),
        r,
        dx,
        T,
        nt,
    )
    return V[n0]


def pde_survival(S0, H, T, b, v, eta, n0, nt, width=10.0):
    """PDE: Q(tau > T) under the spot drift b (r = 0, payoff 1, barrier value 0)."""
    y0 = eta * np.log(S0 / H)
    dx = y0 / n0
    n = int(np.ceil((y0 + width * v * np.sqrt(T)) / dx))
    V = np.ones(n + 1)
    V[0] = 0.0
    V = _cn(V, 0.0, lambda tau: 1.0, 0.5 * v * v, eta * (b - 0.5 * v * v), 0.0, dx, T, nt)
    return V[n0]


def pde_price(S0, X, H, T, r, b, v, phi, eta, knock, R, n0=100, nt=400):
    """PDE with one Richardson step; in = vanilla - out(R = 0) + R DF Q(tau > T)."""

    def rich(fn, *a):
        return (4 * fn(*a, 2 * n0, 2 * nt) - fn(*a, n0, nt)) / 3

    if knock == "out":
        return rich(pde_out, S0, X, H, T, r, b, v, phi, eta, R)
    out0 = rich(pde_out, S0, X, H, T, r, b, v, phi, eta, 0.0)
    q = rich(pde_survival, S0, H, T, b, v, eta) if R > 0 else 0.0
    return black_spot(S0, X, T, r, b, v, phi) - out0 + R * np.exp(-r * T) * q


GLX, GLW = leggauss(160)


def expect_pieces(x, mu, s, pieces, zmax=12.0):
    """E[f(x + mu + s Z)] for each x; f piecewise smooth, pieces = [(a, b, func)] in y."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    for a, b, func in pieces:
        za = np.clip((a - x - mu) / s, -zmax, zmax)
        zb = np.clip((b - x - mu) / s, -zmax, zmax)
        half, mid = 0.5 * (zb - za), 0.5 * (zb + za)
        z = mid[:, None] + half[:, None] * GLX[None, :]
        out += np.sum(half[:, None] * GLW[None, :] * norm.pdf(z) * func(x[:, None] + mu + s * z), 1)
    return out


def dq_out(S0, X, H, T, r, b, v, phi, eta, m, h_frac=0.125, width=10.0):
    """DQ: discretely monitored knock-out (t_j = jT/m, t = 0 not monitored), rebate 0."""
    dt = T / m
    s, mu = v * np.sqrt(dt), (b - 0.5 * v * v) * dt
    lH, lK, x0 = np.log(H), np.log(X), np.log(S0)
    A, B = (lH, x0 + width * v * np.sqrt(T) + 0.5)
    if eta == -1:
        A, B = x0 - width * v * np.sqrt(T) - 0.5, lH
    grid = np.linspace(A, B, int(np.ceil((B - A) / (s * h_frac))) + 1)

    def payoff(y):
        return np.maximum(phi * (np.exp(y) - X), 0.0)

    last = [(A, lK, payoff), (lK, B, payoff)] if A < lK < B else [(A, B, payoff)]
    disc = np.exp(-r * dt)
    if m == 1:
        return float(disc * expect_pieces(np.array([x0]), mu, s, last)[0])
    vals = disc * expect_pieces(grid, mu, s, last)
    for _ in range(m - 1):
        sp = CubicSpline(grid, vals)
        pts = np.array([x0]) if _ == m - 2 else grid
        vals = disc * expect_pieces(pts, mu, s, [(A, B, sp)])
    return float(vals[0])


# =========================================================================== references: Asian
def black(F, K, s, DF, cp):
    F, K, s = (np.asarray(a, dtype=float) for a in (F, K, s))
    d1 = np.log(F / K) / s + 0.5 * s
    return cp * DF * (F * Nn(cp * d1) - K * Nn(cp * (d1 - s)))


def geo_ref(t, F_fix, sig, K, DF, cp):
    """GEO: lognormal G with moments from the covariance double sum."""
    t, F_fix = np.asarray(t, float), np.asarray(F_fix, float)
    n = t.size
    mean = np.mean(np.log(F_fix) - 0.5 * sig**2 * t)
    var = sig**2 * np.minimum.outer(t, t).sum() / n**2
    return black(np.exp(mean + 0.5 * var), K, np.sqrt(var), DF, cp), np.exp(mean + 0.5 * var), var


def kemna_vorst(S, K, T, r, q, sig):
    ba, sa = 0.5 * (r - q - sig**2 / 6), sig / np.sqrt(3)
    d1 = (np.log(S / K) + (ba + 0.5 * sa**2) * T) / (sa * np.sqrt(T))
    return S * np.exp((ba - r) * T) * Nn(d1) - K * np.exp(-r * T) * Nn(d1 - sa * np.sqrt(T))


def arq(t, F_fix, sig, K, DF, cp, S0_fix=None, h_frac=0.1, nsd=12.0):
    """ARQ: arithmetic Asian, fixings at t > 0 (+ optional t = 0 fixing S0_fix), flat sigma."""
    t, F_fix, K = np.asarray(t, float), np.asarray(F_fix, float), np.atleast_1d(K).astype(float)
    n, c0 = t.size, (0.0 if S0_fix is None else float(S0_fix))
    ntot = n + (S0_fix is not None)
    dV = np.diff(sig**2 * t, prepend=0.0)
    s = np.sqrt(dV)
    m = np.concatenate(([-0.5 * dV[0]], np.log(F_fix[1:] / F_fix[:-1]) - 0.5 * dV[1:]))
    if n == 1:
        z, wts = np.array([-np.inf]), np.array([1.0])
    else:
        h = h_frac * s[1:].min()
        lo = min(m[1:].min(), 0.0) - nsd * s[1:].max()
        hi = np.log(n + 1.0) + abs(np.log(F_fix[-1] / F_fix[0])) + nsd * sig * np.sqrt(t[-1]) + 0.5
        z = np.arange(lo, hi + h, h)
        f = norm.pdf(z, loc=m[-1], scale=s[-1])
        W = np.logaddexp(0.0, z)
        for k in range(n - 2, 0, -1):
            band = int(np.ceil(nsd * s[k] / h)) + 2
            rows = (
                np.round((m[k] + W - z[0]) / h).astype(int)[None, :]
                + np.arange(-band, band + 1)[:, None]
            )
            cols = np.broadcast_to(np.arange(z.size)[None, :], rows.shape)
            ok = (rows >= 0) & (rows < z.size)
            rr, cc = rows[ok], cols[ok]
            vals = norm.pdf((z[rr] - m[k] - W[cc]) / s[k]) / s[k] * h
            f = sps.csr_matrix((vals, (rr, cc)), shape=(z.size, z.size)) @ f
        wts = f * h
    out = []
    for k in K:
        scale = F_fix[0] * (1.0 + np.exp(z)) / ntot
        Kp = (k - c0 / ntot) / scale
        pos = Kp > 0
        d1 = np.where(pos, -np.log(np.where(pos, Kp, 1.0)) / s[0] + 0.5 * s[0], np.inf)
        if cp == 1:
            val = np.where(pos, Nn(d1) - Kp * Nn(d1 - s[0]), 1.0 - Kp)
        else:
            val = np.where(pos, Kp * Nn(s[0] - d1) - Nn(-d1), 0.0)
        out.append(DF * np.sum(wts * scale * val))
    return np.array(out)


# =========================================================================== references: autocall
def loop_flows(X, coupons, B_ac, B_c, B_p, memory):
    """LOOP: term sheet path by path. Returns undiscounted flows (n, N) and call index (-1)."""
    n, N = X.shape
    flows, call = np.zeros((n, N)), np.full(N, -1)
    for j in range(N):
        unpaid = 0.0
        for i in range(n):
            cpn = coupons[i] + (unpaid if memory else 0.0)
            if X[i, j] >= B_c:
                flows[i, j] += cpn
                unpaid = 0.0
            else:
                unpaid = cpn if memory else 0.0
            if i < n - 1 and X[i, j] >= B_ac:
                flows[i, j] += 1.0
                call[j] = i
                break
            if i == n - 1:
                flows[i, j] += 1.0 if X[i, j] >= B_p else X[i, j]
    return flows, call


def own_pv(X, coupons, B_ac, B_c, B_p, memory, DF):
    """Own vectorised PV (for MC only; validated against LOOP)."""
    n, N = X.shape
    alive, U, P = np.ones(N, bool), np.zeros(N), np.zeros(N)
    for i in range(n):
        G = coupons[i] + (U if memory else 0.0)
        paid = alive & (X[i] >= B_c)
        P += DF[i] * np.where(paid, G, 0.0)
        U = np.where(paid, 0.0, G)
        if i < n - 1:
            c = alive & (X[i] >= B_ac)
            P += DF[i] * c
            alive &= ~c
    return P + DF[-1] * alive * np.where(X[-1] >= B_p, 1.0, X[-1])


def acx(t, coupons, B_ac, B_c, B_p, memory, f, DF, w, nx=1601, nsd=11.0):
    """ACX: exact autocallable value, X_i = f_i M_{t_i}, ln M Gaussian with Var = w_i."""
    t, coupons, f, DF, w = (np.asarray(a, float) for a in (t, coupons, f, DF, w))
    n, dw = t.size, np.diff(np.asarray(w, float), prepend=0.0)
    L = nsd * np.sqrt(w[-1]) + 1.0
    x = np.linspace(-L, L, nx)
    cum = np.concatenate(([0.0], np.cumsum(coupons)))

    def lx(B):
        return np.log(B / f_i) if B > 0 else -np.inf

    cont = None
    for i in range(n - 1, -1, -1):
        f_i, new = f[i], {}
        s, mu = np.sqrt(dw[i]), -0.5 * dw[i]
        for l_ in range(-1, i):
            G = coupons[i] + ((cum[i] - cum[l_ + 1]) if memory else 0.0)
            if i == n - 1:
                xc, xp = lx(B_c), lx(B_p)
                cuts = sorted({c for c in (xc, xp) if np.isfinite(c)})
                edges = [-np.inf, *cuts, np.inf]
                pcs = []
                for a, b in zip(edges[:-1], edges[1:], strict=True):
                    probe = b - 1.0 if np.isfinite(b) else a + 1.0
                    cpn = DF[i] * G if f_i * np.exp(probe) >= B_c else 0.0
                    pcs.append(
                        (
                            a,
                            b,
                            lambda y, cpn=cpn, fi=f_i, Di=DF[i]: (
                                cpn + Di * np.where(fi * np.exp(y) >= B_p, 1.0, fi * np.exp(y))
                            ),
                        )
                    )
            else:
                xc, xac = lx(B_c), lx(B_ac)
                pcs = [(-np.inf, min(xc, xac), cont[l_])]
                if xac > xc:
                    pcs.append((xc, xac, lambda y, s_=cont[i], G=G, Di=DF[i]: Di * G + s_(y)))
                pcs.append((xac, np.inf, lambda y, G=G, Di=DF[i]: Di * (1.0 + G) + 0.0 * y))
            pcs = [(max(a, -L - 5), min(b, L + 5), fn) for a, b, fn in pcs if b > a]
            if i == 0:
                return float(expect_pieces(np.array([0.0]), mu, s, pcs)[0])
            sp = CubicSpline(x, expect_pieces(x, mu, s, pcs))
            new[l_] = lambda y, sp=sp: sp(np.clip(y, -L, L))
        cont = new
    raise RuntimeError("unreachable")


# =========================================================================== references: Heston
def heston_cf(u, T, kappa, theta, xi, rho, v0):
    u = np.asarray(u, dtype=complex)
    a = kappa - rho * xi * 1j * u
    d = np.sqrt(a * a + xi * xi * (1j * u + u * u))
    g = (a - d) / (a + d)
    e = np.exp(-d * T)
    C = kappa * theta / xi**2 * ((a - d) * T - 2.0 * np.log((1 - g * e) / (1 - g)))
    return np.exp(C + (a - d) / xi**2 * (1 - e) / (1 - g * e) * v0)


_E = np.concatenate((np.linspace(0, 20, 81), np.linspace(20, 400, 201)[1:]))
_GX, _GW = leggauss(64)
U_N = (0.5 * np.diff(_E)[:, None] * _GX + 0.5 * (_E[:-1] + _E[1:])[:, None]).ravel()
U_W = (0.5 * np.diff(_E)[:, None] * _GW).ravel()


def gp_below(k, T, prm, share=False):
    """GP: Q(F_T/F < k), or the share-measure probability (share=True)."""
    phi = heston_cf(U_N - 1j, T, *prm) if share else heston_cf(U_N, T, *prm)
    integ = np.real(np.exp(-1j * U_N * np.log(k)) * phi / (1j * U_N))
    return 0.5 - np.sum(U_W * integ) / np.pi


def gp_put(f, k, T, prm):
    """E[(k - X)^+], X = f F_T/F (undiscounted)."""
    return k * gp_below(k / f, T, prm) - f * gp_below(k / f, T, prm, share=True)


def gp_iv(f, k, T, prm):
    call = gp_put(f, k, T, prm) + f - k
    return brentq(lambda s: float(black(f, k, s * np.sqrt(T), 1.0, 1)) - call, 1e-4, 3, xtol=1e-15)


def oqem(prm, t_obs, dt_max, n_paths, rng, psi_c=1.5):
    """OQEM: own QE-M log-forward paths at the dates (n_dates, n_paths)."""
    k, th, xi, rho, v0 = prm
    x, v, out, tp = np.zeros(n_paths), np.full(n_paths, float(v0)), [], 0.0
    for to in t_obs:
        m_i = int(np.ceil((to - tp) / dt_max - 1e-9))
        dt = (to - tp) / m_i
        ek = np.exp(-k * dt)
        s2a, s2b = xi * xi * ek * (1 - ek) / k, th * xi * xi * (1 - ek) ** 2 / (2 * k)
        k2 = 0.5 * dt * (k * rho / xi - 0.5) + rho / xi
        k3 = 0.5 * dt * (1 - rho * rho)
        A = k2 + 0.5 * k3
        for _ in range(m_i):
            uv, zx = rng.random(n_paths), rng.standard_normal(n_paths)
            mm = th + (v - th) * ek
            psi = (v * s2a + s2b) / (mm * mm)
            vn, lnM, q = np.empty(n_paths), np.empty(n_paths), psi <= psi_c
            b2 = 2 / psi[q] - 1 + np.sqrt(2 / psi[q]) * np.sqrt(2 / psi[q] - 1)
            a = mm[q] / (1 + b2)
            vn[q] = a * (np.sqrt(b2) + norm.ppf(uv[q])) ** 2
            lnM[q] = A * b2 * a / (1 - 2 * A * a) - 0.5 * np.log(1 - 2 * A * a)
            e = ~q
            pe = (psi[e] - 1) / (psi[e] + 1)
            be = (1 - pe) / mm[e]
            up = uv[e] > pe
            ve = np.zeros(pe.size)
            ve[up] = np.log((1 - pe[up]) / (1 - uv[e][up])) / be[up]
            vn[e], lnM[e] = ve, np.log(pe + be * (1 - pe) / (be - A))
            x = x - lnM - 0.5 * k3 * v + k2 * vn + np.sqrt(k3 * (v + vn)) * zx
            v = vn
        out.append(x.copy())
        tp = to
    return np.array(out)


# =========================================================================== shared data
HAUG = {  # Haug (1998) p. 72 via QuantLib test-suite/barrieroption.cpp, as quoted in barrier.md §5
    ("in", 1, 1): {
        95: ((7.7627, 4.0109, 2.0576), (9.0093, 5.1370, 2.8517)),
        100: ((13.8333, 7.8494, 3.9795), (14.8816, 9.2045, 5.3043)),
    },
    ("out", 1, 1): {
        95: ((9.0246, 6.7924, 4.8759), (8.8334, 7.0285, 5.4137)),
        100: ((3.0, 3.0, 3.0), (3.0, 3.0, 3.0)),
    },
    ("out", 1, -1): {105: ((2.6789, 2.3580, 2.3453), (2.6341, 2.4389, 2.4315))},
    ("in", 1, -1): {105: ((14.1112, 8.4482, 4.5910), (15.2098, 9.7278, 5.8350))},
    ("out", -1, 1): {
        95: ((2.2798, 2.2947, 2.6252), (2.4170, 2.4258, 2.6246)),
        100: ((3.0, 3.0, 3.0), (3.0, 3.0, 3.0)),
    },
    ("out", -1, -1): {105: ((3.7760, 5.4932, 7.5187), (4.2293, 5.8032, 7.5649))},
    ("in", -1, 1): {
        95: ((2.9586, 6.5677, 11.9752), (3.8769, 7.7989, 13.3078)),
        100: ((2.2845, 5.9085, 11.6465), (3.3328, 7.2636, 12.9713)),
    },
    ("in", -1, -1): {105: ((1.4653, 3.3721, 7.0846), (2.0658, 4.4226, 8.3686))},
}
H_S0, H_R, H_B, H_T, H_REB = 100.0, 0.08, 0.04, 0.5, 3.0
E1_CELLS = {  # (knock, cp, eta, H, K) of barrier.md erratum E1 / D-027, sigma = 0.30
    ("out", 1, -1, 105, 90.0),
    ("out", -1, -1, 105, 90.0),
    ("out", -1, -1, 105, 100.0),
    ("out", -1, -1, 105, 110.0),
    ("in", -1, 1, 95, 100.0),
    ("in", -1, 1, 95, 110.0),
}
BGK_TRUE = {  # BGK (1997) "True" column (barrier.md §5): (table, m, S0, eta) -> {H: True}
    ("2.1", 50, 100.0, 1): {
        85: 6.322,
        90: 6.098,
        95: 4.907,
        96: 4.427,
        97: 3.834,
        98: 3.126,
        99: 2.337,
    },
    ("2.2", 5, 100.0, 1): {95: 5.671, 99: 4.489},
    ("2.6", 50, 110.0, -1): {155: 12.894, 140: 10.551, 125: 4.616, 115: 0.807},
}
FM_R, FM_SIG, FM_K = 0.0367, 0.17801, np.array([90.0, 100.0, 110.0])
FM_GEO = {
    12: (11.66744, 4.703634, 1.251243),
    50: (11.7076, 4.769273, 1.299134),
    250: (11.71856, 4.787024, 1.312165),
}
FM_CTS = np.array([11.72135, 4.791541, 1.315486])
FM_ARI = {
    12: (11.90497, 4.88210, 1.36314),
    50: (11.93301, 4.93736, 1.40264),
    250: (11.94068, 4.95233, 1.41351),
}
TQ = np.arange(1, 7) / 4


def haug_cells():
    """(knock, cp, eta, H, sigma, K, published) for the 72 Haug cells."""
    for (knock, cp, eta), rows in HAUG.items():
        for H, (v25, v30) in rows.items():
            for sig, vals in ((0.25, v25), (0.30, v30)):
                for K, pub in zip((90.0, 100.0, 110.0), vals, strict=True):
                    yield knock, cp, eta, H, sig, K, pub


def haug_exact(knock, cp, eta, H, sig, K):
    """Closed form (RRS), or branch (4) when the barrier is breached at S0 = H = 100."""
    if H == H_S0:
        return H_REB if knock == "out" else float(black_spot(H_S0, K, H_T, H_R, H_B, sig, cp))
    return float(rrs(H_S0, K, H, H_T, H_R, H_B, sig, cp, eta, knock, H_REB))


def qp_haug(knock, cp, eta, H, sig, K):
    F, DF = H_S0 * np.exp(H_B * H_T), np.exp(-H_R * H_T)
    return float(barrier_price(F, K, H, H_T, DF, sig, cp, eta, knock, b=H_B, R=H_REB))


def market_ref(t):
    """Own log-linear forward / DF curve from the kept parity expiries (D-025 rule)."""
    tab = pd.read_csv(MARKET)
    kept = tab[tab.kept.astype(bool)].sort_values("T")
    T_, lF, lD = kept["T"].to_numpy(), np.log(kept.F.to_numpy()), np.log(kept.DF.to_numpy())
    fit = T_ > 0.3
    r_ex = -(T_[fit] @ lD[fit]) / (T_[fit] @ T_[fit])
    b_ex = np.polyfit(T_[fit], lF[fit], 1)[0]
    ext = np.maximum(np.asarray(t, float) - T_[-1], 0.0)
    F0 = np.exp(np.interp(t, T_, lF) + b_ex * ext)
    return F0, np.exp(np.interp(t, T_, lD) - r_ex * ext), float(np.exp(lF[0] + lD[0])), tab


def load_autocall_compare():
    spec = importlib.util.spec_from_file_location(
        "ac_cmp", REPO / "scripts" / "autocall_compare.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def zstat(est, se, ref):
    return (np.asarray(est) - np.asarray(ref)) / np.asarray(se)


# =========================================================================== tests: barrier
def test_rr_equals_own_spot_form_on_a_wide_grid():
    """RR (forward form, log_ndtr) vs RRS (spot form) for all 16 cells of table (3), with and
    without rebate, b in {-0.05, 0, 0.04, 0.1}, sigma in {0.05, 0.25, 0.8}, T in {0.01, 0.5, 3},
    DF in {0.9, 1.0}, K straddling H. abs/rel 1e-11."""
    worst = 0.0
    K = np.array([70.0, 94.0, 95.0, 96.0, 100.0, 104.0, 105.0, 106.0, 140.0])
    for b in (-0.05, 0.0, 0.04, 0.1):
        for sig in (0.05, 0.25, 0.8):
            for T in (0.01, 0.5, 3.0):
                for DF in (0.9, 1.0):
                    r, F = -np.log(DF) / T, 100.0 * np.exp(b * T)
                    for cp in (1, -1):
                        for eta, H in ((1, 95.0), (-1, 105.0)):
                            for knock in ("in", "out"):
                                for R in (0.0, 2.0):
                                    q = barrier_price(F, K, H, T, DF, sig, cp, eta, knock, b=b, R=R)
                                    ref = rrs(100.0, K, H, T, r, b, sig, cp, eta, knock, R)
                                    if not np.all(np.isfinite(ref)):
                                        continue  # spot form overflows; covered by PDE below
                                    err = np.abs(q - ref) / np.maximum(1.0, np.abs(ref))
                                    worst = max(worst, err.max())
    assert worst < 1e-11, worst


@pytest.mark.parametrize("cell", sorted(E1_CELLS))
def test_e1_cells_pde_confirms_rr_and_refutes_table(cell):
    """Erratum E1: an independent PDE reproduces RR on the six sigma = 0.30 cells to 1e-7, and the
    published 4-decimal entry is NOT the correctly rounded value (|table - exact| > 5e-5)."""
    knock, cp, eta, H, K = cell
    pde = pde_price(H_S0, K, H, H_T, H_R, H_B, 0.30, cp, eta, knock, H_REB)
    rr = qp_haug(knock, cp, eta, H, 0.30, K)
    pub = HAUG[(knock, cp, eta)][H][1][[90.0, 100.0, 110.0].index(K)]
    assert abs(pde - rr) < 1e-7, (pde, rr)
    assert abs(round(pde, 4) - pub) > 5e-5 and abs(pde - pub) > 5e-5, (pde, pub)


def test_haug_table_per_cell_without_loosening():
    """Proposed replacement for the 1e-4 sigma = 0.30 column test (E1): every cell whose published
    value is correctly rounded is held to 5e-5 (both sigma columns: 66 cells); the six E1 cells
    are held to the two-method golden value (RRS closed form = PDE, tests/golden/barrier_haug.json)
    at 1e-10; and the E1 set is exactly the set of cells that miss 5e-5."""
    gold = {(c["knock"], c["cp"], c["eta"], c["H"], c["sigma"], c["K"]): c for c in _golden()}
    misses = set()
    for knock, cp, eta, H, sig, K, pub in haug_cells():
        v = qp_haug(knock, cp, eta, H, sig, K)
        g = gold[(knock, cp, eta, H, sig, K)]
        assert abs(v - g["value"]) < 1e-10, (knock, cp, eta, H, sig, K, v, g["value"])
        if abs(v - pub) > 5e-5:
            misses.add((knock, cp, eta, H, K) if sig == 0.30 else ("sigma 0.25",))
    assert misses == E1_CELLS, misses


def _golden() -> list[dict]:
    return json.loads(GOLDEN.read_text())["cells"]


def test_golden_file_two_methods_agree():
    """tests/golden/barrier_haug.json: RRS vs PDE agree to 2e-8 on every H != S0 cell."""
    cells = _golden()
    assert len(cells) == 72
    assert max(c["max_abs_method_diff"] for c in cells) < 2e-8


def test_branch_continuity_at_k_equals_h_to_1e12():
    """barrier.md §5 'branch continuity at K = H' at the note's 1e-12, using K = H +- 1 ulp (the
    implementer tests 1e-11 at K = H(1 +- 1e-13), where the K-shift alone is ~2e-11)."""
    F, DF, T, sig = 100.0, 0.97, 0.75, 0.25
    for cp in (1, -1):
        for eta, H in ((1, 95.0), (-1, 105.0)):
            for knock in ("in", "out"):
                for b, R in ((0.0, 0.0), (0.04, 2.0)):
                    v = [
                        float(barrier_price(F, k, H, T, DF, sig, cp, eta, knock, b=b, R=R))
                        for k in (np.nextafter(H, 0), H, np.nextafter(H, 1e9))
                    ]
                    assert max(v) - min(v) < 1e-12, (cp, eta, knock, b, v)


def test_rho_to_one_converges_linearly_to_branch_4():
    """rho -> 1: |V(delta) - limit| = O(delta) for delta = 1e-4 .. 1e-12 (ratio ~100 per 2
    decades), out -> R and in -> vanilla, both barrier sides, with rebate."""
    F, DF, T, sig = 100.0, 0.97, 0.75, 0.25
    for cp in (1, -1):
        for eta in (1, -1):
            for knock in ("in", "out"):
                lim = 2.0 if knock == "out" else float(black_spot(F, 100.0, T, 0.0, 0.0, sig, cp))
                lim = lim if knock == "out" else lim * DF
                errs = [
                    abs(
                        float(
                            barrier_price(
                                F, 100.0, F * (1 - eta * d), T, DF, sig, cp, eta, knock, R=2.0
                            )
                        )
                        - lim
                    )
                    for d in (1e-4, 1e-6, 1e-8, 1e-10, 1e-12)
                ]
                assert errs[-1] < 1e-8 and all(
                    e1 < 0.05 * e0 for e0, e1 in zip(errs[:3], errs[1:4], strict=True)
                )


def test_extreme_inputs_vs_pde():
    """Small sigma with carry (pitfall 8), T -> 0, barrier 0.1% from spot, deep ITM/OTM, b < 0,
    vs PDE (with rebate 1.5): 2e-8."""
    cases = [
        (100, 100, 95, 2.0, 0.03, 0.05, 0.03, 1, 1),
        (100, 100, 105, 2.0, 0.03, 0.05, 0.03, -1, -1),
        (100, 100, 105, 2.0, 0.03, -0.05, 0.03, 1, -1),
        (100, 100, 99.9, 0.5, 0.02, 0.0, 0.2, 1, 1),
        (100, 95, 99.5, 0.005, 0.02, 0.01, 0.25, 1, 1),
        (100, 105, 100.5, 0.005, 0.02, 0.01, 0.25, -1, -1),
        (100, 60, 90, 1.0, 0.02, 0.0, 0.3, 1, 1),
        (100, 200, 90, 1.0, 0.02, 0.0, 0.3, 1, 1),
    ]
    for S0, X, H, T, r, b, v, cp, eta in cases:
        q = float(
            barrier_price(
                S0 * np.exp(b * T), X, H, T, np.exp(-r * T), v, cp, eta, "out", b=b, R=1.5
            )
        )
        assert abs(q - pde_price(S0, X, H, T, r, b, v, cp, eta, "out", 1.5)) < 2e-8, (H, T, v)


def test_bgk_true_column_vs_own_discrete_quadrature():
    """DQ reproduces BGK 'True' (trinomial, average accuracy ~0.001, BGK §2) to 1e-3 on all 13
    cells, and the BGK-shifted RR of the implementation equals RRS at the shifted barrier."""
    for (_, m, S0, eta), rows in BGK_TRUE.items():
        for H, true in rows.items():
            assert abs(dq_out(S0, 100.0, H, 0.2, 0.1, 0.1, 0.3, 1, eta, m) - true) < 1e-3, (m, H)
            Hs = float(bgk_barrier(H, 0.3, 0.2 / m, eta))
            assert abs(Hs - H * np.exp(-eta * 0.5825971579390107 * 0.3 * np.sqrt(0.2 / m))) < 1e-12
            q = float(
                barrier_price(
                    S0 * np.exp(0.02), 100.0, Hs, 0.2, np.exp(-0.02), 0.3, 1, eta, "out", b=0.1
                )
            )
            assert abs(q - float(rrs(S0, 100.0, Hs, 0.2, 0.1, 0.1, 0.3, 1, eta, "out"))) < 1e-11


def test_in_plus_out_equals_vanilla_extra_configurations():
    """Gate G1 on configurations the implementer did not use: Euler scheme, b < 0, DF > 1, odd
    M, monitor = 5, K = H included, FO (Feller-violating) and BSM; separate calls, < 1e-12."""
    K = np.array([80.0, 95.0, 100.0, 105.0, 125.0])
    worst = 0.0
    for model, scheme in ((FO, "euler"), (FO, "qe"), (0.4, "qe-m")):
        for b, DF, M, mon in ((-0.03, 1.01, 7, None), (0.05, 0.9, 15, 5)):
            for cp in (1, -1):
                for eta, H in ((1, 95.0), (-1, 105.0)):
                    kw = dict(
                        n_steps=M, n_paths=5000, seed=VSEED + 1, b=b, monitor=mon, scheme=scheme
                    )
                    o, i, v = (
                        barrier_mc(100.0, K, H, 1.3, DF, model, cp, eta, k, **kw)[0]
                        for k in ("out", "in", "vanilla")
                    )
                    worst = max(worst, np.abs(o + i - v).max())
    assert worst < 1e-12, worst


@pytest.mark.slow
def test_e2_slope_and_figure_csv_vs_exact_discrete_prices():
    """Erratum E2 and barrier_mc_vs_rr.csv: exact discrete prices (DQ) for m = 4..512 give a fitted
    log-log slope in [0.40, 0.46] with local slopes increasing towards 0.5; every CSV point
    (uncorrected MC vs DQ, bridge MC vs RRS) within 3 SE; BGK-shifted column = RRS."""
    df = pd.read_csv(FIG / "barrier_mc_vs_rr.csv")
    S0, K, H, T, r, b, v = 100.0, 100.0, 95.0, 0.2, 0.1, 0.1, 0.3
    cont = float(rrs(S0, K, H, T, r, b, v, 1, 1, "out"))
    ex = np.array([dq_out(S0, K, H, T, r, b, v, 1, 1, int(m)) for m in df.m])
    assert np.all(np.abs(zstat(df.mc_uncorrected, df.se_uncorrected, ex)) <= 3)
    assert np.all(np.abs(zstat(df.mc_bridge, df.se_bridge, cont)) <= 3)
    shifted = [
        float(
            rrs(S0, K, H * np.exp(-0.5825971579390107 * v * np.sqrt(dt)), T, r, b, v, 1, 1, "out")
        )
        for dt in df.dt
    ]
    assert np.abs(df.bgk_shifted_rr - shifted).max() < 1e-10
    slope = np.polyfit(np.log(df.dt), np.log(ex - cont), 1)[0]
    loc = np.diff(np.log(ex - cont)) / np.diff(np.log(df.dt.to_numpy()))
    assert 0.40 <= slope <= 0.46 and np.all(np.diff(loc) > 0) and loc[-1] < 0.5, (slope, loc)


@pytest.mark.slow
def test_discrete_mc_vs_exact_discrete_quadrature_fresh_seeds():
    """Discrete-monitoring MC (exact steps at t_j, no bridge) vs DQ, BGK cells, validator seeds,
    2e5 paths: max |z| <= 4 over 13 cells x 2 seeds."""
    zs = []
    for k, ((_, m, S0, eta), rows) in enumerate(BGK_TRUE.items()):
        for s in range(2):
            for H in rows:
                ref = dq_out(S0, 100.0, H, 0.2, 0.1, 0.1, 0.3, 1, eta, m)
                v, se, _, _ = barrier_mc(
                    S0 * np.exp(0.02),
                    100.0,
                    H,
                    0.2,
                    np.exp(-0.02),
                    0.3,
                    1,
                    eta,
                    "out",
                    n_steps=m,
                    n_paths=200_000,
                    seed=VSEED + 100 + 10 * k + s,
                    b=0.1,
                    monitor=m,
                )
                zs.append(float(zstat(v, se, ref)))
    assert np.max(np.abs(zs)) <= 4, zs


def g2_bsm_z(seed_index: int) -> list[float]:
    """Gate G2 configuration (Haug market, 8 types, K in {90, 110}, b in {0, 0.04}, M in {1, 50})
    on validator seeds, vs RRS; returns the z of non-zero cells (zero cells must be exactly 0)."""
    zs = []
    for j, (b, M) in enumerate([(b, M) for b in (0.0, 0.04) for M in (1, 50)]):
        F, K = 100 * np.exp(b * 0.5), np.array([90.0, 110.0])
        for cp in (1, -1):
            for eta, H in ((1, 95.0), (-1, 105.0)):
                for knock in ("out", "in"):
                    ref = rrs(100.0, K, H, 0.5, 0.08, b, 0.25, cp, eta, knock)  # r = -ln(DF)/T
                    v, se, _, _ = barrier_mc(
                        F,
                        K,
                        H,
                        0.5,
                        np.exp(-0.04),
                        0.25,
                        cp,
                        eta,
                        knock,
                        n_steps=M,
                        n_paths=200_000,
                        seed=VSEED + 1000 + 10 * seed_index + j,
                        b=b,
                    )
                    zero = ref == 0
                    assert np.all(v[zero] == 0)
                    zs += list(zstat(v[~zero], se[~zero], ref[~zero]))
    return zs


@pytest.mark.slow
def test_gate_g2_bsm_bridge_fresh_seeds():
    """G2 on 5 validator seeds (280 comparisons): mean z in +-0.3, sd in [0.8, 1.2],
    max |z| <= 4.5."""
    zs = np.array([z for s in range(5) for z in g2_bsm_z(s)])
    assert abs(zs.mean()) <= 0.3 and 0.8 <= zs.std() <= 1.2 and np.abs(zs).max() <= 4.5


def g2h_z(seed: int) -> list[float]:
    """Gate G2h (xi = 1e-4, rho = 0, v0 = theta = 0.04, K = 100, H = 90 / 110, T = 1, DF = 0.97,
    50 steps, 2e5 paths) vs RRS at sigma = 0.2: z of cdo, cdi, puo, pui, then vanilla call, put."""
    p = HestonParams(kappa=1.5, theta=0.04, xi=1e-4, rho=0.0, v0=0.04)
    r, out = -np.log(0.97), []
    for cp, eta, H in ((1, 1, 90.0), (-1, -1, 110.0)):
        for knock in ("out", "in"):
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
                seed=seed,
            )
            out.append(float(zstat(v, se, rrs(100.0, 100.0, H, 1.0, r, 0.0, 0.2, cp, eta, knock))))
    for cp in (1, -1):
        v, se, _, _ = barrier_mc(
            100.0,
            100.0,
            90.0,
            1.0,
            0.97,
            p,
            cp,
            1,
            "vanilla",
            n_steps=50,
            n_paths=200_000,
            seed=seed,
        )
        out.append(float(zstat(v, se, black_spot(100.0, 100.0, 1.0, r, 0.0, 0.2, cp))))
    return out


@pytest.mark.slow
def test_gate_g2h_heston_small_xi_declared_and_fresh_seeds():
    """G2h on the declared seed (20261138) reproduces max |z| 2.97 with vanilla z -3.15 / +3.13;
    on 40 validator seeds: per-cell mean z within +-0.5, pooled sd in [0.8, 1.2], max |z| <= 4.5."""
    zd = g2h_z(20261108 + 30)
    assert abs(max(abs(z) for z in zd[:4]) - 2.97) < 0.01 and abs(zd[4] + 3.15) < 0.01
    Z = np.array([g2h_z(VSEED + 2000 + s) for s in range(40)])
    assert np.all(np.abs(Z[:, :4].mean(0)) <= 0.5), Z.mean(0)
    assert 0.8 <= Z[:, :4].std() <= 1.2 and np.abs(Z[:, :4]).max() <= 4.5


@pytest.mark.slow
def test_gate_g2m_model_free_cdo_fresh_seeds():
    """G2m (cdo(K = H = 90) = 10 DF for any continuous martingale), FO set, QE-M + v+ bridge at
    the gate step dt = 1/128, 20 validator seeds x 2e5 paths. The v+ bridge has a real O(dt)
    bias (pooled +0.021 +- 0.005 at 1/128, +0.091 at 1/32; --scan), so zero bias is NOT
    asserted. Asserted: the pooled bias is below 1.5 single-run SE (the gate can still detect a
    1.5 SE error) and at most 3 of 20 seeds fail the single-run 3 SE gate."""
    vals = [
        barrier_mc(
            100.0,
            90.0,
            90.0,
            1.0,
            0.98,
            FO,
            1,
            1,
            "out",
            n_steps=128,
            n_paths=200_000,
            seed=VSEED + 3000 + s,
        )[:2]
        for s in range(20)
    ]
    v, se = np.array(vals, dtype=float).T
    assert abs(v.mean() - 9.8) <= 1.5 * se.mean(), (v.mean() - 9.8, se.mean())
    assert np.sum(np.abs(v - 9.8) > 3 * se) <= 3


# =========================================================================== tests: Asian
def test_geometric_closed_form_vs_own_covariance_sum():
    """Eqs. (7)-(10) vs GEO on random dates (with and without a t = 0 fixing), random forward
    curves, calls and puts: rel 1e-12; and KV continuous vs [FM08] 'cts' row to 3e-4 (note)."""
    rng = np.random.default_rng(VSEED + 5)
    for _ in range(20):
        n = int(rng.integers(1, 40))
        t = np.sort(rng.uniform(0.01, 3.0, n))
        if rng.random() < 0.5:
            t[0] = 0.0
        F_fix = 100 * np.exp(rng.normal(0, 0.05, n).cumsum())
        sig, DF, K = rng.uniform(0.05, 0.8), rng.uniform(0.8, 1.0), rng.uniform(60, 150, 5)
        for cp in (1, -1):
            ref, FG, var = geo_ref(t, F_fix, sig, K, DF, cp)
            v = geometric_asian_price(F_fix, sig**2 * t, K, DF, cp)
            assert np.allclose(v, ref, rtol=1e-12, atol=1e-12)
            fg, s2 = geometric_asian_moments(F_fix, sig**2 * t)
            assert abs(fg / FG - 1) < 1e-13 and abs(s2 / var - 1) < 1e-12
    assert np.abs(kemna_vorst(100.0, FM_K, 1.0, FM_R, 0.0, FM_SIG) - FM_CTS).max() < 3e-4


def test_arq_reference_is_exact_for_n2():
    """ARQ (my arithmetic reference) vs 1-D adaptive quadrature at n = 2: 1e-11."""
    sig, t, Ff, K, DF = 0.3, np.array([0.5, 1.0]), np.array([100.0, 102.0]), 101.0, 0.95

    def inner(z1):
        S1 = Ff[0] * np.exp(-0.5 * sig**2 * t[0] + sig * np.sqrt(t[0]) * z1)
        fwd2, s2, Kp = S1 * Ff[1] / Ff[0], sig * np.sqrt(t[1] - t[0]), 2 * K - S1
        if Kp <= 0:
            return norm.pdf(z1) * ((S1 + fwd2) / 2 - K)
        d1 = np.log(fwd2 / Kp) / s2 + 0.5 * s2
        return norm.pdf(z1) * 0.5 * (fwd2 * Nn(d1) - Kp * Nn(d1 - s2))

    kink = (np.log(2 * K / Ff[0]) + 0.5 * sig**2 * t[0]) / (sig * np.sqrt(t[0]))
    bf = DF * integrate.quad(inner, -12, 12, epsabs=1e-14, limit=400, points=[kink])[0]
    assert abs(arq(t, Ff, sig, K, DF, 1)[0] - bf) < 1e-11


def test_fm08_tables_explained_by_unrounded_sigma():
    """Both [FM08] tables sit 0.5-1.7e-4 above the stated-input exact values. One unrounded sigma
    (fitted on the 12 geometric values) reproduces the geometric table to 5e-6 and Table 6 at
    n = 12 to 2e-5 (ARQ). This justifies the note's 3e-4 tolerance (no loosening)."""

    def geo_vals(s):
        out = []
        for n, ref in FM_GEO.items():
            tk = np.arange(n + 1) / n
            out += list(geo_ref(tk, 100 * np.exp(FM_R * tk), s, FM_K, np.exp(-FM_R), 1)[0] - ref)
        return np.array(out + list(kemna_vorst(100.0, FM_K, 1.0, FM_R, 0.0, s) - FM_CTS))

    s_fit = least_squares(lambda q: geo_vals(q[0]), [FM_SIG], x_scale=1e-5).x[0]
    assert abs(s_fit - 0.1780163) < 2e-7 and np.abs(geo_vals(s_fit)).max() < 5e-6
    assert 1e-4 < np.abs(geo_vals(FM_SIG)).max() < 1.5e-4
    tk = np.arange(1, 13) / 12
    a_fit = arq(tk, 100 * np.exp(FM_R * tk), s_fit, FM_K, np.exp(-FM_R), 1, S0_fix=100.0)
    a_stated = arq(tk, 100 * np.exp(FM_R * tk), FM_SIG, FM_K, np.exp(-FM_R), 1, S0_fix=100.0)
    assert np.abs(a_fit - FM_ARI[12]).max() < 2e-5 and np.abs(a_stated - FM_ARI[12]).max() < 3e-4


def test_cv_report_csv_is_self_consistent():
    """asian_cv_vr.csv: VR = (se_plain/se_cv)^2 exactly (same paths, same ddof) and VR =
    1/(1 - rho^2) to 1e-6 relative; prices vs ARQ within 3 se_cv."""
    df = pd.read_csv(FIG / "asian_cv_vr.csv")
    assert np.allclose(df.VR, (df.se_plain / df.se_cv) ** 2, rtol=1e-9)
    assert np.allclose(df.VR, 1 / (1 - df.rho_hat**2), rtol=1e-6)
    t = np.arange(1, 13) / 12
    for sig, g in df.groupby("sigma"):
        ref = arq(t, np.full(12, 100.0), sig, 100 * g.K_over_F.to_numpy(), 1.0, 1)
        assert np.all(np.abs(zstat(g.price_cv, g.se_cv, ref)) <= 3), (
            sig,
            zstat(g.price_cv, g.se_cv, ref),
        )


ASIAN_CASES = {  # name: (t > 0 dates, forwards, S0 fixing or None, sigma, DF, K, cp)
    "fm08_n12": (
        np.arange(1, 13) / 12,
        100 * np.exp(FM_R * np.arange(1, 13) / 12),
        100.0,
        FM_SIG,
        np.exp(-FM_R),
        FM_K,
        1,
    ),
    "fm08_n50": (
        np.arange(1, 51) / 50,
        100 * np.exp(FM_R * np.arange(1, 51) / 50),
        100.0,
        FM_SIG,
        np.exp(-FM_R),
        FM_K,
        1,
    ),
    "carry_n12": (
        np.arange(1, 13) / 12,
        100 * np.exp(0.03 * np.arange(1, 13) / 12),
        None,
        0.3,
        0.97,
        np.array([80.0, 100.0, 120.0]),
        1,
    ),
    "put_s05_T3_n52": (
        np.arange(1, 53) * 3 / 52,
        np.full(52, 100.0),
        None,
        0.5,
        1.0,
        np.array([80.0, 100.0, 120.0]),
        -1,
    ),
    "irregular": (
        np.array([0.1, 0.35, 0.4, 0.9, 1.7]),
        np.array([99.0, 101.0, 100.5, 103.0, 104.0]),
        98.0,
        0.25,
        0.93,
        np.array([90.0, 100.0, 110.0]),
        1,
    ),
}


@pytest.mark.slow
def test_asian_mc_cv_plain_and_geometric_vs_exact_fresh_seeds():
    """CV, plain arithmetic and plain geometric MC (2e5, validator seeds) vs ARQ / GEO with no
    extra tolerance: max |z| <= 4 over 5 cases x 3 strikes x 3 estimators x 2 seeds."""
    zs = []
    for k, (name, (t, F, S0, sig, DF, K, cp)) in enumerate(ASIAN_CASES.items()):
        tt = t if S0 is None else np.concatenate(([0.0], t))
        FF = F if S0 is None else np.concatenate(([S0], F))
        ari = arq(t, F, sig, K, DF, cp, S0_fix=S0)
        geo = geo_ref(tt, FF, sig, K, DF, cp)[0]
        for s in range(2):
            for kind, ref in (("cv", ari), ("arithmetic", ari), ("geometric", geo)):
                v, se, _, _ = asian_mc(
                    tt,
                    FF,
                    K,
                    DF,
                    sig,
                    cp,
                    n_paths=200_000,
                    seed=VSEED + 4000 + 100 * k + 10 * s + len(kind),
                    kind=kind,
                )
                zs.append((name, kind, s, np.round(zstat(v, se, ref), 2).tolist()))
    zmax = max(np.abs(z[3]).max() for z in zs)
    assert zmax <= 4, zs


@pytest.mark.slow
def test_asian_bounds_19_and_own_variance_reduction():
    """Oracle (19) (not tested by the implementer beyond Y >= X): pathwise
    (A - K)^+ <= (G - K)^+ + (A - G) and C_A <= DF/n sum Black(F_i, K, sigma sqrt t_i) + 3 SE.
    Own VR (own paths, own regression) at the CSV settings within 15% of the CSV value."""
    t, K, DF = np.arange(1, 13) / 12, np.array([80.0, 100.0, 120.0]), 0.97
    F = 100 * np.exp(0.02 * t)
    A, G = asian_averages(t, F, 0.3, 100_000, np.random.default_rng(VSEED + 50))
    Y, X = np.maximum(A - K[:, None], 0), np.maximum(G - K[:, None], 0)
    assert np.all(Y <= X + (A - G) + 1e-12 * A)
    upper = DF * np.mean(
        [black(Fi, K, 0.3 * np.sqrt(ti), 1.0, 1) for Fi, ti in zip(F, t, strict=True)], 0
    )
    assert np.all(DF * Y.mean(1) <= upper + 3 * DF * Y.std(1) / np.sqrt(Y.shape[1]))
    df = pd.read_csv(FIG / "asian_cv_vr.csv")
    rng = np.random.default_rng(VSEED + 51)
    for sig in (0.1, 0.3, 0.5):
        Z = rng.standard_normal((200_000, 12))
        S = 100 * np.exp(np.cumsum(-0.5 * sig**2 / 12 + sig * np.sqrt(1 / 12) * Z, axis=1))
        a, g = S.mean(1), np.exp(np.log(S).mean(1))
        for kf in (0.8, 1.0, 1.2):
            y, x = np.maximum(a - 100 * kf, 0), np.maximum(g - 100 * kf, 0)
            c = np.cov(y, x)
            vr = c[0, 0] / (c[0, 0] - c[0, 1] ** 2 / c[1, 1])
            rep = df[(df.sigma == sig) & (np.abs(df.K_over_F - kf) < 1e-9)].VR.iloc[0]
            assert abs(vr / rep - 1) < 0.15, (sig, kf, vr, rep)


@pytest.mark.slow
def test_asian_heston_arithmetic_vs_own_qem():
    """Heston (GS) monthly arithmetic Asian, asian_mc with max_dt = 1/32 vs OQEM (dt 1/64),
    2e5 + 4e5 paths: |z| <= 4 (independent schemes and RNG layouts)."""
    t, F, K = np.arange(1, 13) / 12, 100 * np.exp(0.02 * np.arange(1, 13) / 12), FM_K
    v, se, _, _ = asian_mc(
        t, F, K, 0.96, GS, 1, n_paths=200_000, seed=VSEED + 60, kind="arithmetic", max_dt=1 / 32
    )
    x = oqem(GS_T, t, 1 / 64, 400_000, np.random.default_rng(VSEED + 61))
    Y = 0.96 * np.maximum((F[:, None] * np.exp(x)).mean(0) - K[:, None], 0)
    z = (v - Y.mean(1)) / np.sqrt(se**2 + Y.var(1) / Y.shape[1])
    assert np.all(np.abs(z) <= 4), z


def test_f2_heston_asian_default_grid_unbiased():
    """F2 (closed in fix round 1). Originally: asian_mc / asian_averages defaulted to max_dt = None
    under Heston (one QE-M step per fixing period, against asian.md eq. 4); quarterly fixings, GS,
    3e6 paths gave default minus max_dt = 1/32 of z = -2.6 / -3.6 / +5.9 at K = 90 / 100 / 110.
    Now (default MAX_DT = 1/32): the same comparison on the same seeds, and the default vs my OQEM
    at dt = 1/64 (2e6 paths), all |z| <= 3."""
    t, F, K = np.arange(1, 5) / 4, np.full(4, 100.0), np.array([90.0, 100.0, 110.0])
    v0, s0, _, _ = asian_mc(
        t, F, K, 1.0, GS, 1, n_paths=3_000_000, seed=VSEED + 70, kind="arithmetic"
    )
    v1, s1, _, _ = asian_mc(
        t, F, K, 1.0, GS, 1, n_paths=3_000_000, seed=VSEED + 71, kind="arithmetic", max_dt=1 / 32
    )
    z = (v0 - v1) / np.hypot(s0, s1)
    assert np.all(np.abs(z) <= 3), (v0, v1, z)
    x = oqem(GS_T, t, 1 / 64, 2_000_000, np.random.default_rng(VSEED + 72))
    Y = np.maximum((F[:, None] * np.exp(x)).mean(0) - K[:, None], 0)
    z2 = (v0 - Y.mean(1)) / np.hypot(s0, Y.std(1) / np.sqrt(Y.shape[1]))
    assert np.all(np.abs(z2) <= 3), z2


# =========================================================================== tests: autocallable
SHEET_VARIANTS = [
    dict(B_ac=1.0, B_c=0.7, B_p=0.6, memory=True),
    dict(B_ac=1.0, B_c=0.7, B_p=0.6, memory=False),
    dict(B_ac=1.02, B_c=1.02, B_p=0.8, memory=True),  # Deng-type: B_c = B_ac
    dict(B_ac=np.inf, B_c=0.7, B_p=0.6, memory=True),
    dict(B_ac=1.0, B_c=0.5, B_p=0.6, memory=True),  # B_p > B_c (erratum E1 of autocallable.md)
    dict(B_ac=0.0, B_c=0.0, B_p=0.0, memory=True),
    dict(B_ac=1.0, B_c=0.0, B_p=0.6, memory=False),
]


def test_autocall_flows_equal_loop_transcription():
    """autocall_flows (vectorised) vs LOOP on 20000 paths with exact ties at every barrier, unequal
    period lengths, 7 sheet variants: flows bit-equal to 1e-15, same call dates, sum p_i + A_n = 1;
    autocall_pv discounts each flow with its own DF_i."""
    t = np.array([0.2, 0.5, 0.75, 1.1, 1.5, 2.0])
    rng = np.random.default_rng(VSEED + 80)
    X = np.exp(np.cumsum(rng.normal(0, 0.15, (6, 20_000)), axis=0))
    for i, level in enumerate((0.6, 0.7, 1.0, 1.02, 0.5, 0.8)):
        X[i, 200 * i : 200 * i + 150] = level
    DF = np.exp(-0.04 * t)
    for var in SHEET_VARIANTS:
        sh = Autocall(tuple(t), 0.09, **var)
        C, R, chi, alive = autocall_flows(X, sh)
        fl, call = loop_flows(X, sh.coupons, var["B_ac"], var["B_c"], var["B_p"], var["memory"])
        assert np.abs(C + R - fl).max() < 1e-15, var
        assert np.array_equal(np.where(chi.any(0), chi.argmax(0), -1), call)
        assert np.array_equal(alive, call == -1)
        assert np.abs(autocall_pv(X, sh, DF) - DF @ fl).max() < 1e-15
        assert (
            np.abs(
                own_pv(X, sh.coupons, var["B_ac"], var["B_c"], var["B_p"], var["memory"], DF)
                - DF @ fl
            ).max()
            < 1e-14
        )


def test_autocall_closed_forms_vs_backward_induction():
    """Eq. (12) (n = 2, both memory flags, several B) and eq. (11) (no-call, Black and a
    deterministic term structure) vs ACX: 1e-9."""
    f2, D2 = np.exp(0.04 * np.array([0.5, 1.0])), np.exp(-0.05 * np.array([0.5, 1.0]))
    for sig in (0.1, 0.25, 0.45):
        w2 = sig**2 * np.array([0.5, 1.0])
        for B_ac, B_c, B_p in ((1.0, 0.75, 0.6), (1.05, 0.9, 0.7), (1e6, 0.75, 0.6)):
            for m in (False, True):
                sh = Autocall((0.5, 1.0), 0.08, B_ac, B_c, B_p, m)
                ref = acx(sh.t, sh.coupons, B_ac, B_c, B_p, m, f2, D2, w2)
                assert abs(autocall_two_dates(sh, f2, D2, w2) - ref) < 1e-9, (sig, B_ac, m)
    t, f, DF = TQ, np.exp(0.04 * TQ), np.exp(-0.05 * TQ)
    for w in (0.1**2 * t, 0.25**2 * t, 0.04 * t + 0.05 * (1 - np.exp(-1.5 * t)) / 1.5):
        for B_c, B_p in ((0.7, 0.6), (0.0, 0.6), (0.8, 0.0)):
            sh = Autocall(tuple(t), 0.08, 1.0, B_c, B_p, True)
            ref = acx(t, sh.coupons, np.inf, B_c, B_p, False, f, DF, w)
            assert abs(autocall_nocall(sh, f, DF, w) - ref) < 1e-9, (B_c, B_p)


def test_vector_coupon_schedule_vs_own_references():
    """Fix round 1 (Deng Case 2 path): Autocall with per-period coupons. Flows vs LOOP with random
    non-proportional schedules and ties (bit-equal); eq. (11) and eq. (12) with vector coupons vs
    ACX (1e-9); the Deng schedule is stored exactly; bad schedules raise."""
    rng = np.random.default_rng(VSEED + 8100)
    t = np.array([0.2, 0.5, 0.75, 1.1, 1.5, 2.0])
    X = np.exp(np.cumsum(rng.normal(0, 0.15, (6, 20_000)), axis=0))
    for i, level in enumerate((0.6, 0.7, 1.0, 1.02, 0.5, 0.8)):
        X[i, 200 * i : 200 * i + 150] = level
    for var in SHEET_VARIANTS:
        c = tuple(rng.uniform(0, 0.05, 6))
        C, R, chi, _ = autocall_flows(X, Autocall(tuple(t), c, **var))
        fl, call = loop_flows(X, np.array(c), var["B_ac"], var["B_c"], var["B_p"], var["memory"])
        assert np.abs(C + R - fl).max() < 1e-15
        assert np.array_equal(np.where(chi.any(0), chi.argmax(0), -1), call)
    f, DF, c6 = np.exp(0.04 * TQ), np.exp(-0.05 * TQ), rng.uniform(0.005, 0.04, 6)
    for w in (0.2**2 * TQ, 0.35**2 * TQ):
        sh = Autocall(tuple(TQ), tuple(c6), 1.0, 0.7, 0.6, True)
        ref = acx(TQ, c6, np.inf, 0.7, 0.6, False, f, DF, w)
        assert abs(autocall_nocall(sh, f, DF, w) - ref) < 1e-9
    f2, D2, w2 = f[[1, 3]], DF[[1, 3]], 0.25**2 * np.array([0.5, 1.0])
    for m in (False, True):
        sh = Autocall((0.5, 1.0), (0.013, 0.051), 1.0, 0.75, 0.6, m)
        ref = acx(sh.t, np.array([0.013, 0.051]), 1.0, 0.75, 0.6, m, f2, D2, w2)
        assert abs(autocall_two_dates(sh, f2, D2, w2) - ref) < 1e-9
    td = np.arange(1, 13) / 12
    cd = np.exp(0.092 * td) - np.exp(0.092 * np.r_[0.0, td[:-1]])
    assert np.array_equal(Autocall(tuple(td), tuple(cd), 1.02, 1.02, 0.8, True).coupons, cd)
    for bad in ((0.01,) * 5, (0.01, -0.01, 0.01, 0.01, 0.01, 0.01), (np.nan,) * 6):
        with pytest.raises(ValueError):
            Autocall(tuple(TQ), bad)


@pytest.mark.slow
def test_deng_case2_qpricer_mc_vs_exact():
    """Fix round 1: qpricer autocall_mc on the Deng Case 2 sheet (vector accrual coupons) vs ACX
    97.5068, 5 validator seeds x 4e5 paths: max |z| <= 4 and |mean z| <= 1.5."""
    t = np.arange(1, 13) / 12
    cpn = np.exp(0.092 * t) - np.exp(0.092 * np.r_[0.0, t[:-1]])
    f, DF = np.exp(0.04 * t), np.exp(-0.06 * t)
    ex = acx(t, cpn, 1.02, 1.02, 0.8, True, f, DF, 0.04 * t)
    sh = Autocall(tuple(t), tuple(cpn), 1.02, 1.02, 0.8, True)
    zs = [
        zstat(*autocall_mc(sh, f, DF, 0.2, n_paths=400_000, seed=VSEED + 8000 + s)[:2], ex)
        for s in range(5)
    ]
    assert np.max(np.abs(zs)) <= 4 and abs(np.mean(zs)) <= 1.5, zs


def test_heston_marginals_cos_vs_gil_pelaez():
    """ratio_marginals(Heston) (COS put, central-difference digital) vs GP at the AC-Q dates, the
    three barriers, sets GS and FO: put 1e-10, digital 1e-6; no-call Heston value vs GP 1e-7."""
    t, f, DF = TQ, np.exp(0.04 * TQ), np.exp(-0.05 * TQ)
    for prm, P in ((GS_T, GS), (FO_T, FO)):
        for L in (0.6, 0.7, 1.0):
            p, D = ratio_marginals(f, t, L, P)
            assert (
                np.abs(p - [gp_put(fi, L, ti, prm) for fi, ti in zip(f, t, strict=True)]).max()
                < 1e-10
            )
            assert (
                np.abs(D - [gp_below(L / fi, ti, prm) for fi, ti in zip(f, t, strict=True)]).max()
                < 1e-6
            )
        sh = Autocall(tuple(t), 0.08, 1.0, 0.7, 0.6, True)
        Dc = np.array([gp_below(0.7 / fi, ti, prm) for fi, ti in zip(f, t, strict=True)])
        dip = DF[-1] * (gp_put(f[-1], 0.6, t[-1], prm) + 0.4 * gp_below(0.6 / f[-1], t[-1], prm))
        ref = np.sum(DF * sh.coupons * (1 - Dc)) + DF[-1] - dip
        assert abs(autocall_nocall(sh, f, DF, P) - ref) < 1e-7


def test_crn_eq15_rebuilt_from_raw_draws():
    """Eq. (15): the z returned by fixings() under Heston equals sum_k sqrt(dt_k/tau_i)
    (rho Z_v,k + sqrt(1 - rho^2) Z_perp,k) rebuilt from the raw generator stream (exact), on an
    uneven date set; the grid ends exactly on every date with ceil(tau/dt) sub-steps."""
    t = np.array([0.25, 0.6, 1.0, 1.01])
    x, z = fixings(GS, t, 500, np.random.default_rng(77), 1 / 32)
    grid = fixing_grid(t, 1 / 32)
    assert set(t) <= set(grid)
    counts = [
        int(np.sum((grid > a) & (grid <= b))) for a, b in zip(np.r_[0, t[:-1]], t, strict=True)
    ]
    assert counts == [int(np.ceil(d * 32 - 1e-9)) for d in np.diff(np.r_[0, t])]
    rng, acc, tp, zz = np.random.default_rng(77), 0.0, 0.0, []
    for g, h in zip(grid, np.diff(grid, prepend=0.0), strict=True):
        d = rng.standard_normal((2, 500))
        acc = acc + np.sqrt(h) * (GS.rho * d[0] + np.sqrt(1 - GS.rho**2) * d[1])
        if g in t:
            zz.append(acc / np.sqrt(g - tp))
            acc, tp = 0.0, g
    assert np.abs(np.array(zz) - z).max() == 0.0


@pytest.mark.slow
def test_autocall_bs_full_product_mc_vs_exact_fresh_seeds():
    """Full product (autocall, memory, protection) under BS: autocall_mc vs ACX, AC-Q-like and
    AC-A-like synthetic sheets, sigma in {0.15, 0.25, 0.4}, 3 validator seeds, 2e5 paths:
    max |z| <= 4; ACX grid-converged (nx 1601 vs 3201) to 1e-8."""
    zs = []
    for k, t in enumerate((TQ, np.array([1.0, 2.0, 3.0]))):
        f, DF = np.exp(0.04 * t), np.exp(-0.05 * t)
        sh = Autocall(tuple(t), 0.08, 1.0, 0.7, 0.6, True)
        for sig in (0.15, 0.25, 0.4):
            ref = acx(t, sh.coupons, 1.0, 0.7, 0.6, True, f, DF, sig**2 * t)
            if sig == 0.25:
                fine = acx(t, sh.coupons, 1.0, 0.7, 0.6, True, f, DF, sig**2 * t, nx=3201)
                assert abs(fine - ref) < 1e-8
            for s in range(3):
                v, se, _, _ = autocall_mc(
                    sh,
                    f,
                    DF,
                    sig,
                    n_paths=200_000,
                    seed=VSEED + 5000 + 100 * k + 10 * s + int(sig * 100),
                )
                zs.append(float(zstat(v, se, ref)))
    assert np.max(np.abs(zs)) <= 4, zs


@pytest.mark.slow
def test_attribution_eq19_vs_independent_references():
    """scripts/autocall_compare.py attribution (GS, illustrative) vs my references on both sheets:
    market f_i, DF_i, S_ref from my own curve (1e-12); sigma_ATM (GP + brentq) 1e-7;
    V_BS, V_TS vs ACX within 4 SE; term1 vs exact within 4 CRN SE; term2 vs GP 1e-6; BS par coupon
    vs exact (affine in c) within 4 SE; V_H vs OQEM (1e6 paths, dt 1/64) within 4 combined SE."""
    ac = load_autocall_compare()
    for k, (name, sheet) in enumerate(ac.SHEETS.items()):
        t = np.asarray(sheet.t)
        f, DF, s_ref = ac.market_inputs(sheet)
        F0, D0, s0, _ = market_ref(t)
        assert abs(s_ref / s0 - 1) < 1e-12 and np.allclose(f, F0 / s0, rtol=1e-12)
        assert np.allclose(DF, D0, rtol=1e-12) and np.all(np.diff(DF) < 0)
        r = ac.attribution(GS, sheet, f, DF)
        sig = gp_iv(f[-1], f[-1], t[-1], GS_T)
        assert abs(r["sigma_atm"] - sig) < 1e-7
        cp = sheet.coupons
        wts = GS.theta * t + (GS.v0 - GS.theta) * (1 - np.exp(-GS.kappa * t)) / GS.kappa
        vbs = acx(t, cp, 1.0, 0.7, 0.6, True, f, DF, sig**2 * t)
        vts = acx(t, cp, 1.0, 0.7, 0.6, True, f, DF, wts)
        assert abs(r["V_BS"] - vbs) <= 4 * r["se_V_BS"] and abs(r["V_TS"] - vts) <= 4 * r["se_V_TS"]
        assert abs(r["term1"] - (vts - vbs)) <= 4 * r["se_term1"]
        Dc = np.array([gp_below(0.7 / fi, ti, GS_T) for fi, ti in zip(f, t, strict=True)])
        dip = DF[-1] * (gp_put(f[-1], 0.6, t[-1], GS_T) + 0.4 * gp_below(0.6 / f[-1], t[-1], GS_T))
        nch = np.sum(DF * cp * (1 - Dc)) + DF[-1] - dip
        ncts = acx(t, cp, np.inf, 0.7, 0.6, False, f, DF, wts)
        assert abs(r["term2"] - (nch - ncts)) < 1e-6
        a0 = acx(t, 0 * cp, 1.0, 0.7, 0.6, True, f, DF, sig**2 * t)
        b1 = acx(t, cp / 0.08, 1.0, 0.7, 0.6, True, f, DF, sig**2 * t) - a0
        assert abs(r["par_c_BS"] - (1 - a0) / b1) <= 4 * r["se_par_c_BS"]
        x = oqem(GS_T, t, 1 / 64, 1_000_000, np.random.default_rng(VSEED + 6000 + k))
        pv = own_pv(f[:, None] * np.exp(x), cp, 1.0, 0.7, 0.6, True, DF)
        assert abs(r["V_H"] - pv.mean()) <= 4 * np.hypot(r["se_V_H"], pv.std() / 1e3), name


def test_illustrative_labelling():
    """D-029: every GS-based output is labelled 'illustrative, pre-calibration'."""
    df = pd.read_csv(FIG / "autocall_attribution_illustrative.csv")
    assert df.label.str.contains("illustrative, pre-calibration").all()
    ac = load_autocall_compare()
    assert "illustrative, pre-calibration" in ac.GS_LABEL and ac.GS == GS
    text = (REPO / "docs" / "DECISIONS.md").read_text()
    assert "illustrative GS set" in text and "illustrative Heston" in text


@pytest.mark.slow
def test_deng_case1_reproduced_case2_published_value_not():
    """Deng-Mallett-McCann (2011) as mapped in autocallable.md §5 (B_ac = B_c = 1.02, memory,
    c_j = e^{0.092 t_j} - e^{0.092 t_(j-1)}, B_p = 0.8, F_0(t) = 100 e^{0.04t}, DF = e^{-0.06t},
    sigma 0.2). Case 1 (no call, coupon 9.2 at T): ACX 99.976 ~ par, as the note expects.
    Case 2: ACX 97.507 and an own MC agree (3 SE) and reproduce Table 1 p_i (4 SE), but the
    published V = 98.39 is 0.88 away (not reproduced; the note does not gate it). The
    implementer's Autocall cannot represent this coupon schedule (c_i = c tau_i only)."""
    t = np.arange(1, 13) / 12
    cpn = np.exp(0.092 * t) - np.exp(0.092 * np.r_[0.0, t[:-1]])
    f, DF, w = np.exp(0.04 * t), np.exp(-0.06 * t), 0.04 * t
    c1 = acx([1.0], [0.092], np.inf, 0.0, 0.8, False, [np.exp(0.04)], [np.exp(-0.06)], [0.04])
    assert abs(100 * c1 - 100) < 0.05
    v = acx(t, cpn, 1.02, 1.02, 0.8, True, f, DF, w)
    rng = np.random.default_rng(VSEED + 7000)
    X = f[:, None] * np.exp(
        np.cumsum(-0.5 * 0.04 / 12 + 0.2 / np.sqrt(12) * rng.standard_normal((12, 400_000)), axis=0)
    )
    pv = own_pv(X, cpn, 1.02, 1.02, 0.8, True, DF)
    assert abs(pv.mean() - v) <= 3 * pv.std() / np.sqrt(pv.size)
    assert abs(100 * v - 97.507) < 1e-3 and abs(100 * v - 98.39) > 0.8
    alive, p = np.ones(X.shape[1], bool), []
    for i in range(11):
        c = alive & (X[i] >= 1.02)
        p.append(c.mean())
        alive &= ~c
    p.append((alive & (X[11] >= 1.02)).mean())
    pub = [
        0.3767,
        0.1435,
        0.0781,
        0.0506,
        0.0361,
        0.0275,
        0.0218,
        0.0178,
        0.0149,
        0.0127,
        0.0110,
        0.0096,
    ]
    se = np.sqrt(np.array(pub) * (1 - np.array(pub)) / X.shape[1])
    assert np.all(np.abs(np.array(p) - pub) <= 4 * se + 5e-5), np.round(p, 4)


# =========================================================================== tests: market
def test_forward_curve_vs_own_interpolation():
    """market.forward_curve vs my own log-linear interpolation / extrapolation (D-025) on 200
    dates in [T_min, 4]: 1e-13 relative; continuity at T_max; raises before T_min."""
    F0, D0, _, tab = market_ref(np.linspace(0.02, 4.0, 200))
    F, D = forward_curve(tab, np.linspace(0.02, 4.0, 200))
    assert np.allclose(F, F0, rtol=1e-13) and np.allclose(D, D0, rtol=1e-13)
    kept = tab[tab.kept.astype(bool)].sort_values("T")
    Tm = float(kept["T"].iloc[-1])
    Fe, De = forward_curve(tab, [Tm, Tm + 1e-10])
    assert abs(Fe[1] / Fe[0] - 1) < 1e-10 and abs(De[1] / De[0] - 1) < 1e-10
    with pytest.raises(ValueError):
        forward_curve(tab, 0.001)


# =========================================================================== golden + report
def build_golden() -> dict:
    cells = []
    for knock, cp, eta, H, sig, K, pub in haug_cells():
        val = haug_exact(knock, cp, eta, H, sig, K)
        if H == H_S0:
            pde, src = None, "branch (4) of barrier.md: barrier at S0; out = rebate, in = vanilla"
            diff = 0.0
        else:
            pde = float(pde_price(H_S0, K, H, H_T, H_R, H_B, sig, cp, eta, knock, H_REB))
            src, diff = "RRS closed form; PDE cross-check", abs(pde - val)
        cells.append(
            dict(
                knock=knock,
                cp=cp,
                eta=eta,
                H=H,
                sigma=sig,
                K=K,
                value=val,
                pde=pde,
                published=pub,
                published_correctly_rounded=bool(abs(round(val, 4) - pub) < 1e-9),
                max_abs_method_diff=diff,
                method=src,
            )
        )
    return {
        "topic": "Haug (1998) standard barrier table, S0 = 100, r = 8%, b = 4%, T = 0.5, rebate 3",
        "created_by": "validator agent, Phase 4 independent validation, 2026-11-08",
        "source": "Agreement of two independent methods, neither using qpricer code: (A) RRS = "
        "Reiner-Rubinstein in Haug's spot form (Haug 2007 §4.17.1), transcribed independently; "
        "(B) PDE = Crank-Nicolson with 4 Rannacher half-steps on y = eta ln(S/H), absorbing "
        "barrier with the rebate as boundary value (paid at hit), Black far boundary, exact "
        "cell-averaged "
        "payoff, S0 on a node (n0 = 100 and 200 nodes between H and S0, 400 / 800 steps), one "
        "Richardson step; knock-ins by in-out parity with a second PDE for Q(tau > T). Stored "
        "value = (A). Rows with H = 100 are breached at S0: closed-form branch (4) (out = rebate "
        "3 paid now, in = vanilla). 'published' = Haug p. 72 via QuantLib "
        "test-suite/barrieroption.cpp as quoted in docs/derivations/barrier.md §5.",
        "conventions": "spot barrier; cp +1 call / -1 put; eta +1 down / -1 up; out-rebate paid at "
        "hit, in-rebate paid at T; forward form inputs: F = 100 e^{0.02}, DF = e^{-0.04}, b = 0.04",
        "accuracy": "max |A - B| stored per cell; overall < 2e-8 (PDE truncation). The six cells "
        "with published_correctly_rounded = false are the barrier.md erratum E1 cells.",
        "generator": "tests/validation/test_phase4_independent.py --golden-out <path>",
        "cells": cells,
    }


def _scan() -> None:
    t0 = time.time()
    zs = np.array([z for s in range(50) for z in g2_bsm_z(100 + s)])
    print(
        f"G2 BSM, 50 validator seeds, {zs.size} z: mean {zs.mean():+.3f} sd {zs.std():.3f} "
        f"max|z| {np.abs(zs).max():.2f} frac>3 {np.mean(np.abs(zs) > 3):.4f} "
        f"({time.time() - t0:.0f}s)"
    )
    Z = np.array([g2h_z(VSEED + 20000 + s) for s in range(300)])
    mx = np.abs(Z[:, :4]).max(1)
    print(
        f"G2h 300 validator seeds: per-cell mean {np.round(Z[:, :4].mean(0), 3).tolist()} "
        f"sd {np.round(Z[:, :4].std(0), 3).tolist()}; "
        f"vanilla mean {np.round(Z[:, 4:].mean(0), 3).tolist()}"
    )
    print(
        f"  per-comparison |z|>3: {np.mean(np.abs(Z[:, :4]) > 3):.4f}; seeds with max|z|>=2.97: "
        f"{np.mean(mx >= 2.97):.4f}; with max|z|>3: {np.mean(mx > 3):.4f}; "
        f"corr(z_cdo, z_vanilla call) {np.corrcoef(Z[:, 0], Z[:, 4])[0, 1]:+.3f} "
        f"({time.time() - t0:.0f}s)"
    )
    for M in (32, 64, 128, 256):
        a = np.array(
            [
                barrier_mc(
                    100.0,
                    90.0,
                    90.0,
                    1.0,
                    0.98,
                    FO,
                    1,
                    1,
                    "out",
                    n_steps=M,
                    n_paths=200_000,
                    seed=VSEED + 30000 + 1000 * M + s,
                )[:2]
                for s in range(25)
            ],
            dtype=float,
        )
        b, pse = a[:, 0].mean() - 9.8, np.sqrt((a[:, 1] ** 2).sum()) / len(a)
        print(
            f"G2m dt=1/{M}: pooled bias {b:+.5f} (SE {pse:.5f}, z {b / pse:+.2f}); seeds failing "
            f"3SE {np.mean(np.abs(a[:, 0] - 9.8) > 3 * a[:, 1]):.2f} ({time.time() - t0:.0f}s)"
        )


def _main() -> None:  # noqa: C901 -- a flat report script
    np.set_printoptions(linewidth=140)
    print("== Haug table: exact (RRS) vs published, cells missing 5e-5")
    for knock, cp, eta, H, sig, K, pub in haug_cells():
        v = haug_exact(knock, cp, eta, H, sig, K)
        if abs(v - pub) > 5e-5:
            pde = pde_price(H_S0, K, H, H_T, H_R, H_B, sig, cp, eta, knock, H_REB)
            print(
                f"  {knock} cp={cp:+d} eta={eta:+d} H={H} sig={sig} K={K:g}: RRS {v:.8f} PDE "
                f"{pde:.8f} published {pub} diff {v - pub:+.2e}"
            )
    res = np.array([haug_exact(*c[:6]) - c[6] for c in haug_cells() if c[3] != 100])
    sig_ = np.array([c[4] for c in haug_cells() if c[3] != 100])
    for s in (0.25, 0.30):
        r = res[sig_ == s]
        print(
            f"  sigma {s}: max |exact - table| {np.abs(r).max():.2e}, "
            f"rms {np.sqrt(np.mean(r**2)):.2e}"
            f" (uniform rounding rms 2.89e-05)"
        )
    print("== BGK True vs DQ")
    for (tab, m, S0, eta), rows in BGK_TRUE.items():
        for H, true in rows.items():
            print(
                f"  Tab {tab} m={m} H={H}: "
                f"DQ {dq_out(S0, 100.0, H, 0.2, 0.1, 0.1, 0.3, 1, eta, m):.5f}"
                f" True {true}"
            )
    df = pd.read_csv(FIG / "barrier_mc_vs_rr.csv")
    cont = float(rrs(100.0, 100.0, 95.0, 0.2, 0.1, 0.1, 0.3, 1, 1, "out"))
    ex = np.array([dq_out(100.0, 100.0, 95.0, 0.2, 0.1, 0.1, 0.3, 1, 1, int(m)) for m in df.m])
    print("== E2: exact discrete DOC (H = 95) vs m:", np.round(ex, 6).tolist())
    print(
        f"  fitted slope exact {np.polyfit(np.log(df.dt), np.log(ex - cont), 1)[0]:.4f}; local",
        np.round(np.diff(np.log(ex - cont)) / np.diff(np.log(df.dt.to_numpy())), 3).tolist(),
    )
    print(
        "  CSV uncorrected z vs exact:",
        np.round(zstat(df.mc_uncorrected, df.se_uncorrected, ex), 2).tolist(),
    )
    print("  CSV bridge z vs RRS:", np.round(zstat(df.mc_bridge, df.se_bridge, cont), 2).tolist())
    print("== Asian: FM08 tables vs exact at stated inputs")
    for n in (12, 50, 250):
        tk = np.arange(1, n + 1) / n
        a = arq(tk, 100 * np.exp(FM_R * tk), FM_SIG, FM_K, np.exp(-FM_R), 1, S0_fix=100.0)
        g = geo_ref(
            np.r_[0, tk], 100 * np.exp(FM_R * np.r_[0, tk]), FM_SIG, FM_K, np.exp(-FM_R), 1
        )[0]
        print(
            f"  n={n}: ARQ {np.round(a, 6).tolist()} "
            f"table-ARQ {np.round(np.array(FM_ARI[n]) - a, 6).tolist()};"
            f" GEO table-GEO {np.round(np.array(FM_GEO[n]) - g, 6).tolist()}"
        )
    print("== Autocallable attribution (illustrative GS) vs independent references")
    ac = load_autocall_compare()
    for name, sheet in ac.SHEETS.items():
        t = np.asarray(sheet.t)
        f, DF, _ = ac.market_inputs(sheet)
        r = ac.attribution(GS, sheet, f, DF)
        sig = gp_iv(f[-1], f[-1], t[-1], GS_T)
        wts = GS.theta * t + (GS.v0 - GS.theta) * (1 - np.exp(-GS.kappa * t)) / GS.kappa
        vbs = acx(t, sheet.coupons, 1.0, 0.7, 0.6, True, f, DF, sig**2 * t)
        vts = acx(t, sheet.coupons, 1.0, 0.7, 0.6, True, f, DF, wts)
        x = oqem(GS_T, t, 1 / 64, 1_000_000, np.random.default_rng(VSEED + 6000))
        pv = own_pv(f[:, None] * np.exp(x), sheet.coupons, 1.0, 0.7, 0.6, True, DF)
        print(
            f"  {name}: sigma_ATM {sig:.6f} (impl {r['sigma_atm']:.6f}); V_BS exact {vbs:.6f} vs "
            f"{r['V_BS']:.6f}+-{r['se_V_BS']:.6f}; V_TS exact {vts:.6f} vs {r['V_TS']:.6f}+-"
            f"{r['se_V_TS']:.6f}; term1 exact {vts - vbs:+.6f} vs {r['term1']:+.6f}+-"
            f"{r['se_term1']:.6f}; V_H OQEM {pv.mean():.6f}+-{pv.std() / 1e3:.6f} vs "
            f"{r['V_H']:.6f}+-{r['se_V_H']:.6f}; gap with OQEM and exact V_BS "
            f"{pv.mean() - vbs:+.6f}"
        )
    t = np.arange(1, 13) / 12
    cpn = np.exp(0.092 * t) - np.exp(0.092 * np.r_[0.0, t[:-1]])
    v = acx(t, cpn, 1.02, 1.02, 0.8, True, np.exp(0.04 * t), np.exp(-0.06 * t), 0.04 * t)
    print(f"== Deng case 2: ACX {100 * v:.4f} vs published 98.39")
    tq, F, K = np.arange(1, 5) / 4, np.full(4, 100.0), np.array([90.0, 100.0, 110.0])
    kw = dict(n_paths=2_000_000, kind="arithmetic")
    out = [
        asian_mc(tq, F, K, 1.0, GS, 1, seed=VSEED + 70, **kw)[:2],  # library default
        asian_mc(tq, F, K, 1.0, GS, 1, seed=VSEED + 71, max_dt=1 / 32, **kw)[:2],
        asian_mc(tq, F, K, 1.0, GS, 1, seed=VSEED + 72, max_dt=None, **kw)[:2],  # pre-fix grid
    ]
    for lab, (v_, s_) in zip(("default", "explicit None (pre-fix)"), (out[0], out[2]), strict=True):
        print(
            f"== Heston Asian {lab}: {np.round(v_, 5).tolist()} vs max_dt 1/32 "
            f"{np.round(out[1][0], 5).tolist()}: z "
            f"{np.round((v_ - out[1][0]) / np.hypot(s_, out[1][1]), 2).tolist()}"
        )
    vs = [control_variate, gbm_paths]  # imported for API coverage only
    del vs


if __name__ == "__main__":
    if "--golden-out" in sys.argv:
        path = Path(sys.argv[sys.argv.index("--golden-out") + 1])
        if path.exists():
            raise SystemExit(f"{path} exists; golden files are never overwritten")
        path.write_text(json.dumps(build_golden(), indent=1))
        print(f"wrote {path}")
    elif "--scan" in sys.argv:
        _scan()
    else:
        _main()
