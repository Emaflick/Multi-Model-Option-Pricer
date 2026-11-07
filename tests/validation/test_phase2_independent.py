"""Independent validation of Phase 2 (Heston: CF, COS, Lewis, Euler-FT / QE / QE-M MC) -- validator.

The qpricer public API (``HestonParams``, ``cf``, ``cumulants``, ``cos_price``, ``cos_range``,
``cos_series_error``, ``lewis_price``, ``mc.european_price``, ``mc.simulate``, ``mc.uniform_grid``)
is only called as the system under test. Every reference value is computed here, with methods
that differ from the implementation:

* RIC  -- characteristic function from my own fixed-step RK4 integration of the Heston Riccati
          ODEs D' = xi^2 D^2 / 2 - (kappa - i rho xi u) D - (u^2 + i u) / 2, C' = kappa theta D
          (derived from the pricing PDE, not copied from the note), one Richardson step
          (n, 2n), step count per u from the local stiffness |d| + |beta|. No closed form.
* GP   -- Gil-Pelaez / Heston (1993) inversion for the put,
          P = DF [ (K - F)/2 - 1/pi int_0^U Re( e^{-iuz} (K phi(u) - F phi(u - i)) / (iu) ) du ],
          composite Gauss-Legendre (fine panels near 0, wider beyond), cut-off U probed on RIC.
          Different contour (real axis + share measure) and quadrature from Lewis (quad_vec on
          Im u = -1/2) and from COS (cosine series).
* H93  -- second, closed-form route used only to certify golden values: Heston's original 1993
          P1/P2 formula with his e^{+dT} CF, complex log continued by phase unwrapping along u.
* MOM  -- c1, c2 from RK4 on the order-w and order-w^2 coefficient ODEs of the cumulant
          generating function (no closed form, stable as kappa -> 0).
* BLK  -- Black-76 from scipy.stats.norm cdf/sf; integrated variance by scipy quad of vbar(t).
* OQE / OEU -- my own Andersen (2008) QE / QE-M and Lord-Koekkoek-van Dijk Euler full
          truncation, written from the papers' recipes with a different random-number layout
          (U_v uniform drawn first, Z_v = Phi^{-1}(U_v), separate Z_perp draw).

Run as a script (``.venv/bin/python tests/validation/test_phase2_independent.py``) to print the
numbers quoted in ``docs/validation/phase2.md`` (add ``--mc`` for the multi-seed MC study,
about 4 minutes). ``--golden-out PATH`` writes the golden JSON to PATH only if PATH does not
exist (never overwrites); ``tests/golden/heston_prices.json`` was created from that output.
"""

from __future__ import annotations

import csv
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy import integrate
from scipy.stats import norm

from qpricer.engines import mc
from qpricer.engines.fourier import cos_price, cos_series_error, lewis_price
from qpricer.models.heston import HestonParams, cf, cumulants

REPO = Path(__file__).resolve().parents[2]
GOLDEN = REPO / "tests" / "golden" / "heston_prices.json"
BIAS_CSV = REPO / "report" / "figures" / "heston_euler_bias.csv"

# Parameter tuples (kappa, theta, xi, rho, v0). Sources: derivations/heston.md §5, heston-mc.md §5.
FO = (1.5768, 0.0398, 0.5751, -0.5711, 0.0175)  # Fang-Oosterlee 2008 (nu_F = 0.38)
GS = (1.3253, 0.0354, 0.3877, -0.7165, 0.0354)  # Gatheral 2006 SPX-like (nu_F = 0.62)
LW = (4.0, 0.25, 1.0, -0.5, 0.04)  # Lewis 2000 (nu_F = 2)
AT2 = (1.5, 0.04, 0.3, -0.9, 0.04)  # AMST Table 2
AI = (0.5, 0.04, 1.0, -0.9, 0.04)  # Andersen A-I (nu_F = 0.04)
BK2 = (2.0, 0.09, 1.0, -0.3, 0.09)  # Broadie-Kaya case 2 (nu_F = 0.36)
KJ = (1.0, 0.16, 2.0, -0.8, 0.16)  # Kahl-Jaeckel stress (nu_F = 0.08)
# Validator edge sets (not in the implementer's tests)
XI15 = (2.0, 0.04, 1.5, -0.7, 0.04)  # high vol-of-vol, nu_F = 0.071
RHO_M99 = (1.5, 0.04, 0.5, -0.99, 0.04)  # rho -> -1
RHO_P99 = (1.5, 0.04, 0.5, 0.99, 0.04)  # rho -> +1 (kappa - rho xi = 1.005 > 0)
RHO_P5 = (2.0, 0.04, 0.6, 0.5, 0.04)  # positive correlation
FELLER_X = (0.5, 0.01, 1.5, -0.7, 0.01)  # Feller strongly violated, nu_F = 0.0044
V0ZERO = (1.5, 0.04, 0.5, -0.7, 0.0)
# Found by the validator's random scan (seed 5, see __main__): L = 18 range too narrow, no warning
SILENT = (3.062, 0.01013, 0.6537, -0.6225, 0.02837)

GATE_SETS = {"FO": FO, "GS": GS, "LW": LW, "AT2": AT2}
T_GRID = (0.1, 0.25, 0.5, 1.0, 2.0, 3.0)
K_GRID = np.arange(60.0, 150.0 + 1e-9, 5.0)
F0 = 100.0


def hp(prm) -> HestonParams:
    return HestonParams(*prm)


# ============================================================================ RIC: Riccati RK4
def _rk4(u, T, k, th, xi, rho, n):
    beta = k - 1j * rho * xi * u
    A = u * u + 1j * u
    a2 = 0.5 * xi * xi

    def f(D):
        return a2 * D * D - beta * D - 0.5 * A

    D = np.zeros_like(u)
    C = np.zeros_like(u)
    h = T / n
    kt = k * th * h / 6.0
    for _ in range(n):
        k1 = f(D)
        d2 = D + 0.5 * h * k1
        k2 = f(d2)
        d3 = D + 0.5 * h * k2
        k3 = f(d3)
        d4 = D + h * k3
        k4 = f(d4)
        C = C + kt * (D + 2.0 * d2 + 2.0 * d3 + d4)  # C' = kappa theta D on the same stages
        D = D + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return C, D


def ref_logcf(u, T, prm, hd=0.1, nmin=32):
    """ln E[exp(iuX)], X = ln(F_T/F), by RK4 + one Richardson step; h (|d| + |beta|) <= hd."""
    k, th, xi, rho, v0 = prm
    u = np.atleast_1d(np.asarray(u, dtype=np.complex128))
    beta = k - 1j * rho * xi * u
    lam = np.abs(np.sqrt(beta * beta + xi * xi * (u * u + 1j * u))) + np.abs(beta)
    need = np.maximum(nmin, np.ceil(T * lam / hd))
    lev = 2 ** np.ceil(np.log2(need)).astype(int)
    out = np.empty_like(u)
    for n in np.unique(lev):
        m = lev == n
        C1, D1 = _rk4(u[m], T, k, th, xi, rho, int(n))
        C2, D2 = _rk4(u[m], T, k, th, xi, rho, 2 * int(n))
        out[m] = (16.0 * (C2 + v0 * D2) - (C1 + v0 * D1)) / 15.0
    return out


def ref_cutoff(T, prm, tol=1e-15, hd=0.25):
    """Smallest u = sqrt(2)^j with |phi(u)|, |phi(u - i)| < tol u (probed upward on RIC)."""
    u = 1.0
    while True:
        lc = ref_logcf(np.array([u, u - 1j]), T, prm, hd=hd)
        if np.all(np.real(lc) < np.log(tol * u)):
            return u
        u *= np.sqrt(2.0)


def _gl_nodes(U, fine=8.0, h_fine=0.25, h_coarse=2.0, nodes=16):
    x, w = leggauss(nodes)
    e1 = np.linspace(0.0, min(fine, U), int(np.ceil(min(fine, U) / h_fine)) + 1)
    e2 = np.linspace(e1[-1], max(U, e1[-1]), max(1, int(np.ceil((U - e1[-1]) / h_coarse))) + 1)
    edges = np.concatenate([e1, e2[1:]]) if U > e1[-1] else e1
    mid, half = 0.5 * (edges[1:] + edges[:-1]), 0.5 * (edges[1:] - edges[:-1])
    return (mid[:, None] + half[:, None] * x).ravel(), (half[:, None] * w).ravel()


def ref_put(F, K, T, DF, prm, hd=0.25, U=None, **quad):
    """GP put with the RIC characteristic function."""
    K = np.atleast_1d(np.asarray(K, dtype=np.float64))
    z = np.log(K / F)
    U = ref_cutoff(T, prm) if U is None else U
    u, wt = _gl_nodes(U, **quad)
    ph = np.exp(ref_logcf(np.concatenate([u, u - 1j]), T, prm, hd=hd))
    p0, p1 = ph[: u.size], ph[u.size :]
    integ = np.real(np.exp(-1j * np.outer(z, u)) * (K[:, None] * p0 - F * p1) / (1j * u)) @ wt
    return DF * (0.5 * (K - F) - integ / np.pi)


def h93_call(F, K, T, DF, prm, U, h=0.5, nodes=16):
    """Heston (1993) P1/P2 with his original e^{+dT} CF; log continued by np.unwrap along u."""
    k, th, s, rho, v0 = prm
    K = np.atleast_1d(np.asarray(K, dtype=np.float64))
    x, w = leggauss(nodes)
    edges = np.linspace(0.0, U, int(np.ceil(U / h)) + 1)
    mid, half = 0.5 * (edges[1:] + edges[:-1]), 0.5 * (edges[1:] - edges[:-1])
    ph = (mid[:, None] + half[:, None] * x).ravel()
    wt = (half[:, None] * w).ravel()
    P = []
    for uj, bj in ((0.5, k - rho * s), (-0.5, k)):
        q = bj - rho * s * ph * 1j
        d = np.sqrt(q * q - s * s * (2.0 * uj * ph * 1j - ph * ph))
        g = (q + d) / (q - d)
        with np.errstate(over="raise", invalid="raise"):
            edt = np.exp(d * T)
            G = (1.0 - g * edt) / (1.0 - g)
        logG = np.log(np.abs(G)) + 1j * np.unwrap(np.angle(G))
        C = k * th / s**2 * ((q + d) * T - 2.0 * logG)
        D = (q + d) / s**2 * (1.0 - edt) / (1.0 - g * edt)
        f = np.exp(C + D * v0 + 1j * ph * np.log(F))
        integ = np.real(np.exp(-1j * np.outer(np.log(K), ph)) * f / (1j * ph)) @ wt
        P.append(0.5 + integ / np.pi)
    return DF * (F * P[0] - K * P[1])


# ============================================================================ BLK, MOM
def ref_black(F, K, T, DF, sigma, cp):
    s = sigma * np.sqrt(T)
    d1 = np.log(F / np.asarray(K)) / s + 0.5 * s
    d2 = d1 - s
    if cp > 0:
        return DF * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return DF * (K * norm.sf(d2) - F * norm.sf(d1))


def ref_int_var(T, prm):
    k, th, _, _, v0 = prm
    val, _ = integrate.quad(lambda t: th + (v0 - th) * np.exp(-k * t), 0.0, T, epsabs=1e-15)
    return val


def ref_cumulants(T, prm):
    """c1, c2 from the CGF expansion D = D1 w + D2 w^2 + ... of the Riccati in w (u = -iw):
    D1' = -kappa D1 - 1/2, D2' = xi^2 D1^2 / 2 + rho xi D1 - kappa D2 + 1/2, C_j' = kappa theta D_j;
    c1 = C1 + v0 D1, c2 = 2 (C2 + v0 D2). Integrated numerically (DOP853, rtol 1e-13)."""
    k, th, xi, rho, v0 = prm

    def f(_, y):
        D1, D2 = y[0], y[1]
        return [
            -k * D1 - 0.5,
            0.5 * xi * xi * D1 * D1 + rho * xi * D1 - k * D2 + 0.5,
            k * th * D1,
            k * th * D2,
        ]

    sol = integrate.solve_ivp(f, (0.0, T), np.zeros(4), method="DOP853", rtol=1e-13, atol=1e-16)
    D1, D2, C1, C2 = sol.y[:, -1]
    return C1 + v0 * D1, 2.0 * (C2 + v0 * D2)


# ============================================================================ OQE, OEU
def own_qe(prm, T, n_steps, n_paths, rng, martingale=True, psi_c=1.5):
    """Andersen (2008) QE (psi_c = 1.5, gamma1 = gamma2 = 1/2) and QE-M; returns x_T = ln F_T/F."""
    k, th, xi, rho, v0 = prm
    dt = T / n_steps
    ek = np.exp(-k * dt)
    s2a, s2b = xi * xi * ek * (1 - ek) / k, th * xi * xi * (1 - ek) ** 2 / (2 * k)
    k0 = -rho * k * th * dt / xi
    k1 = 0.5 * dt * (k * rho / xi - 0.5) - rho / xi
    k2 = 0.5 * dt * (k * rho / xi - 0.5) + rho / xi
    k3 = k4 = 0.5 * dt * (1 - rho * rho)
    A = k2 + 0.5 * k4
    x, v = np.zeros(n_paths), np.full(n_paths, float(v0))
    for _ in range(n_steps):
        uv = rng.random(n_paths)
        zx = rng.standard_normal(n_paths)
        m = th + (v - th) * ek
        psi = (v * s2a + s2b) / (m * m)
        vn, lnM = np.empty(n_paths), np.empty(n_paths)
        q = psi <= psi_c
        pq, mq = psi[q], m[q]
        b2 = 2 / pq - 1 + np.sqrt(2 / pq) * np.sqrt(2 / pq - 1)
        a = mq / (1 + b2)
        vn[q] = a * (np.sqrt(b2) + norm.ppf(uv[q])) ** 2
        lnM[q] = A * b2 * a / (1 - 2 * A * a) - 0.5 * np.log(1 - 2 * A * a)
        e = ~q
        pe = (psi[e] - 1) / (psi[e] + 1)
        be = (1 - pe) / m[e]
        ue = uv[e]
        ve = np.zeros(ue.size)
        up = ue > pe
        ve[up] = np.log((1 - pe[up]) / (1 - ue[up])) / be[up]
        vn[e] = ve
        lnM[e] = np.log(pe + be * (1 - pe) / (be - A))
        K0 = (-lnM - (k1 + 0.5 * k3) * v) if martingale else k0
        x = x + K0 + k1 * v + k2 * vn + np.sqrt(k3 * v + k4 * vn) * zx
        v = vn
    return x


def own_euler(prm, T, n_steps, n_paths, rng):
    """Euler full truncation (LKD 2010): v+ in drift and diffusion, log-Euler for F."""
    k, th, xi, rho, v0 = prm
    dt = T / n_steps
    x, v = np.zeros(n_paths), np.full(n_paths, float(v0))
    sr = np.sqrt(1 - rho * rho)
    for _ in range(n_steps):
        w1 = rng.standard_normal(n_paths)
        w2 = rng.standard_normal(n_paths)
        vp = np.where(v > 0, v, 0.0)
        x = x - 0.5 * vp * dt + np.sqrt(vp * dt) * (rho * w1 + sr * w2)
        v = v + k * (th - vp) * dt + xi * np.sqrt(vp * dt) * w1
    return x


def own_estimate(g, DF=1.0):
    g = np.asarray(g, dtype=np.float64)
    return DF * g.mean(axis=-1), DF * g.std(axis=-1, ddof=1) / np.sqrt(g.shape[-1])


def cos_quiet(*a, **kw):
    """cos_price, returning (price, warned)."""
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        v = cos_price(*a, **kw)
    return v, any(issubclass(w.category, RuntimeWarning) for w in wl)


# ============================================================================ 1. CF
@pytest.mark.parametrize(
    "prm, T, umax",
    [
        (FO, 0.02, 300.0),
        (FO, 1.0, 60.0),
        (FO, 10.0, 12.0),
        (KJ, 1.0, 30.0),
        (KJ, 10.0, 6.0),
        (AI, 10.0, 12.0),
        (XI15, 1.0, 30.0),
        (RHO_P5, 2.0, 30.0),
        (RHO_M99, 1.0, 60.0),
    ],
)
def test_cf_vs_own_riccati_rk4(prm, T, umax):
    """cf vs RIC on the real axis (COS), Im u = -1/2 (Lewis), Im u = -1 (share measure)."""
    u = np.geomspace(0.05, umax, 9)
    uu = np.concatenate([u, -u, u - 0.5j, u - 1j])
    ref = np.exp(ref_logcf(uu, T, prm, hd=0.1))
    got = cf(uu, T, hp(prm))
    assert np.abs(got - ref).max() < 1e-10


# ============================================================================ 2. cumulants
SMALL_K = (0.05, 0.04, 0.5, -0.7, 0.04)
LARGE_K = (20.0, 0.04, 1.0, -0.9, 0.2)


@pytest.mark.parametrize("prm", [FO, GS, LW, AT2, AI, KJ, XI15, SMALL_K, LARGE_K])
@pytest.mark.parametrize("T", [0.02, 1.0, 10.0])
def test_cumulants_vs_own_moment_ode(prm, T):
    """Eqs. (6)-(7) vs MOM (main branch, kappa T >= 1e-3)."""
    c1, c2 = cumulants(T, hp(prm))
    r1, r2 = ref_cumulants(T, prm)
    assert abs(c1 / r1 - 1) < 1e-12
    assert abs(c2 / r2 - 1) < 1e-9


def test_cumulant_small_kappa_branch_error_is_range_only():
    """Finding F3 (cosmetic): the kappa T < 1e-3 branch drops the kappa theta terms of c2 (rel.
    error up to ~2% when theta >> v0). c1 stays exact, and c2 only sets the COS range, so the
    assertion is on the range: |c2 / c2_true - 1| < 5% keeps L = 18 between 17.5 and 18.5."""
    prm = (0.999e-3, 0.5, 0.5, -0.7, 0.01)
    c1, c2 = cumulants(1.0, hp(prm))
    r1, r2 = ref_cumulants(1.0, prm)
    assert abs(c1 / r1 - 1) < 1e-12
    assert abs(c2 / r2 - 1) < 5e-2  # measured 2.2e-2 (printed by __main__)


# ============================================================================ 3. gate: COS vs Lewis
def test_gate_cos_vs_lewis_off_lattice():
    """ROADMAP gate on random off-lattice points: K ~ U[60, 150] (+ ends), T ~ U[0.1, 3] (+ ends),
    DF ~ U[0.8, 1], calls and puts, the four designated sets. Tolerance 1e-6."""
    rng = np.random.default_rng(2026_11_07)
    K = np.concatenate([[60.0, 150.0], rng.uniform(60.0, 150.0, 23)])
    worst = 0.0
    for prm in GATE_SETS.values():
        p = hp(prm)
        for T in np.concatenate([[0.1, 3.0], rng.uniform(0.1, 3.0, 8)]):
            DF = rng.uniform(0.8, 1.0)
            for cp in (1, -1):
                c, warned = cos_quiet(F0, K, T, DF, p, cp)
                lw = lewis_price(F0, K, T, DF, p, cp)
                assert not warned
                worst = max(worst, np.abs(c - lw).max())
    assert worst < 1e-6, worst


@pytest.mark.parametrize("name, T", [("FO", 1.0), ("GS", 0.1), ("AT2", 3.0), ("LW", 1.0)])
def test_engines_vs_own_reference_fast(name, T):
    """COS and Lewis vs RIC+GP on the gate strike grid (DF = 0.95, calls via my own parity)."""
    prm, DF = GATE_SETS[name], 0.95
    put = ref_put(F0, K_GRID, T, DF, prm)
    call = put + DF * (F0 - K_GRID)
    p = hp(prm)
    assert np.abs(cos_price(F0, K_GRID, T, DF, p, -1) - put).max() < 1e-6
    assert np.abs(cos_price(F0, K_GRID, T, DF, p, 1) - call).max() < 1e-6
    assert np.abs(lewis_price(F0, K_GRID, T, DF, p, -1) - put).max() < 1e-8
    assert np.abs(lewis_price(F0, K_GRID, T, DF, p, 1) - call).max() < 1e-8


@pytest.mark.parametrize("prm", [FO, GS])
def test_short_maturity_and_wings(prm):
    """T = 0.02 (below the gate's 0.1) and the gate's wing corners (K = 60, 150 at T = 0.1)."""
    p = hp(prm)
    for T in (0.02, 0.1):
        put = ref_put(F0, K_GRID, T, 1.0, prm)
        assert np.abs(cos_price(F0, K_GRID, T, 1.0, p, -1) - put).max() < 1e-6
        assert np.abs(lewis_price(F0, K_GRID, T, 1.0, p, -1) - put).max() < 1e-8


def test_df_applied_once_and_homogeneity():
    """V(DF) = DF V(1) exactly in law; V(lam F, lam K) = lam V(F, K); cross-engine parity."""
    p = hp(GS)
    K = np.array([60.0, 95.0, 100.0, 150.0])
    for eng in (cos_price, lewis_price):
        for cp in (1, -1):
            v1, v9 = eng(F0, K, 1.0, 1.0, p, cp), eng(F0, K, 1.0, 0.9, p, cp)
            assert np.abs(v9 - 0.9 * v1).max() < 1e-12
            assert np.abs(eng(250.0, 2.5 * K, 1.0, 0.9, p, cp) - 2.5 * v9).max() < 1e-9
    c = lewis_price(F0, K, 2.0, 0.93, p, 1)
    put = cos_price(F0, K, 2.0, 0.93, p, -1)
    assert np.abs(c - put - 0.93 * (F0 - K)).max() < 1e-6


# ============================================================================ 4. gate: xi -> 0
@pytest.mark.parametrize(
    "k, th, v0", [(1.5, 0.04, 0.04), (1.5, 0.04, 0.09), (1.5, 0.09, 0.01), (5.0, 0.02, 0.2)]
)
def test_gate_xi_to_zero_rho0_vs_own_black(k, th, v0):
    """ROADMAP gate, rho = 0: xi = 1e-4 equals Black with sigma^2 = (1/T) int_0^T vbar dt (quad),
    < 1e-5, both engines, calls and puts, DF in {1, 0.95}. Includes v0 != theta (the implementer's
    gate grid uses v0 = theta only, where the time average is trivial)."""
    p = hp((k, th, 1e-4, 0.0, v0))
    worst = 0.0
    for T in T_GRID:
        sig = np.sqrt(ref_int_var(T, (k, th, 1e-4, 0.0, v0)) / T)
        for DF in (1.0, 0.95):
            for cp in (1, -1):
                b = ref_black(F0, K_GRID, T, DF, sig, cp)
                for eng in (cos_price, lewis_price):
                    worst = max(worst, np.abs(eng(F0, K_GRID, T, DF, p, cp) - b).max())
    assert worst < 1e-5, worst


@pytest.mark.parametrize("T", [0.1, 1.0, 3.0])
def test_xi_to_zero_first_order_term_eq16(T):
    """D-015 reference (Black + rho xi J d2V/dx dSigma) vs my own first-order coefficient, the
    central difference in xi of RIC+GP at xi = +-h (RIC allows xi < 0). Also: the rho = -0.7 gap
    to plain Black is model-driven (RIC+GP shows it too), i.e. the literal gate is unattainable."""
    k, th, v0, rho = 1.5, 0.04, 0.04, -0.7
    h = 1e-3
    pp = ref_put(F0, K_GRID, T, 1.0, (k, th, h, rho, v0), U=400.0)
    pm = ref_put(F0, K_GRID, T, 1.0, (k, th, -h, rho, v0), U=400.0)
    pp2 = ref_put(F0, K_GRID, T, 1.0, (k, th, 2 * h, rho, v0), U=400.0)
    pm2 = ref_put(F0, K_GRID, T, 1.0, (k, th, -2 * h, rho, v0), U=400.0)
    a1 = (8 * (pp - pm) - (pp2 - pm2)) / (12 * h)  # 4th-order central difference
    tau, E = (1 - np.exp(-k * T)) / k, np.exp(-k * T)
    S = th * T + (v0 - th) * tau
    J = th / k * (T - tau) + (v0 - th) / k * (tau - T * E)
    d1 = (np.log(F0 / K_GRID) + 0.5 * S) / np.sqrt(S)
    eq16 = rho * J * (-F0 * norm.pdf(d1) * (d1 - np.sqrt(S)) / (2 * S))
    assert np.abs(a1 - eq16).max() < 1e-6 * max(1.0, np.abs(a1).max())
    p = hp((k, th, 1e-4, rho, v0))
    black = ref_black(F0, K_GRID, T, 1.0, np.sqrt(ref_int_var(T, (k, th, 0, 0, v0)) / T), -1)
    for eng in (cos_price, lewis_price):
        v = eng(F0, K_GRID, T, 1.0, p, -1)
        assert np.abs(v - black - 1e-4 * a1).max() < 1e-5  # D-015 interpretation
    gap = np.abs(lewis_price(F0, K_GRID, T, 1.0, p, -1) - black).max()
    own_gap = np.abs(ref_put(F0, K_GRID, T, 1.0, (k, th, 1e-4, rho, v0), U=400.0) - black).max()
    assert abs(gap - own_gap) < 1e-9 and (T < 0.5 or gap > 1e-5)


# ============================================================================ 5. COS robustness
def test_cos_default_meets_gate_or_warns_on_plausible_set():
    """Finding F1 (reproduction, EXPECTED TO FAIL until fixed): on an SPX-plausible set inside the
    gate domain (K in [60, 150], T = 1) the default COS (N = 256, L = 18) misses Lewis and RIC+GP
    by 2.6e-6 > 1e-6 WITHOUT the RuntimeWarning: the L = 18 range is too narrow (error flat in
    N, falls with L), and the warning only models the series error (eq. 13)."""
    p = hp(SILENT)
    v, warned = cos_quiet(F0, K_GRID, 1.0, 1.0, p, -1)
    lw = lewis_price(F0, K_GRID, 1.0, 1.0, p, -1)
    own = ref_put(F0, K_GRID, 1.0, 1.0, SILENT)
    assert np.abs(lw - own).max() < 1e-8  # the reference is not the issue
    assert np.abs(v - lw).max() < 1e-6 or warned, np.abs(v - lw).max()


@pytest.mark.slow
@pytest.mark.parametrize(
    "prm, T", [(XI15, 0.5), (RHO_M99, 0.5), (RHO_P99, 0.5), (V0ZERO, 0.5), (XI15, 0.02)]
)
def test_edge_sets_lewis_exact_cos_warns(prm, T):
    """Edge sets: Lewis vs RIC+GP < 1e-8 always; COS either < 1e-6 or emits its RuntimeWarning."""
    p = hp(prm)
    own = ref_put(F0, K_GRID, T, 1.0, prm)
    assert np.abs(lewis_price(F0, K_GRID, T, 1.0, p, -1) - own).max() < 1e-8
    v, warned = cos_quiet(F0, K_GRID, T, 1.0, p, -1)
    assert np.abs(v - own).max() < 1e-6 or warned


# ============================================================================ 6. golden file
def _golden():
    if not GOLDEN.exists():
        pytest.fail(f"{GOLDEN} missing: run this file with --golden-out <path>")
    return json.loads(GOLDEN.read_text())


def test_golden_lewis_and_cos():
    """Frozen two-method oracle values (RIC+GP and H93 agree): Lewis everywhere < 1e-8; COS on
    the gate sets < 1e-6 (COS is not held to the golden values on the edge sets, finding F1)."""
    g = _golden()
    for c in g["cases"]:
        p = hp(tuple(c["params"]))
        K = np.array(c["K"])
        ref = np.array(c["put"])
        assert np.abs(lewis_price(c["F"], K, c["T"], c["DF"], p, -1) - ref).max() < 1e-8, c["set"]
        if c["set"] in GATE_SETS:
            assert np.abs(cos_price(c["F"], K, c["T"], c["DF"], p, -1) - ref).max() < 1e-6


# ============================================================================ 7. MC contract
def test_mc_api_contract_and_common_random_numbers():
    p = hp(FO)
    K = np.array([70.0, 100.0, 140.0])
    kw = dict(n_steps=8, n_paths=50_000, seed=9)
    a = mc.european_price(F0, K, 1.0, 1.0, p, -1, **kw)
    assert len(a) == 4 and isinstance(a[2], int) and isinstance(a[3], int)
    assert a[2] == 50_000 and a[3] == 9 and a[0].shape == a[1].shape == (3,)
    b = mc.european_price(F0, 100.0, 1.0, 1.0, p, -1, **kw)
    assert np.ndim(b[0]) == 0 and b[0] == a[0][1]  # strike subset does not change the paths
    assert np.array_equal(mc.european_price(F0, K, 1.0, 1.0, p, -1, **kw)[0], a[0])
    c = mc.european_price(F0, K, 1.0, 0.9, p, -1, **kw)
    assert np.allclose(c[0], 0.9 * a[0], rtol=1e-15, atol=0) and np.allclose(c[1], 0.9 * a[1])
    d = mc.european_price(F0, K, 1.0, 1.0, p, [1, -1, 1], **kw)
    e = mc.european_price(F0, K, 1.0, 1.0, p, 1, **kw)
    assert np.array_equal(d[0], [e[0][0], a[0][1], e[0][2]])
    f = mc.european_price(2 * F0, 2 * K, 1.0, 1.0, p, -1, **kw)
    assert np.allclose(f[0], 2 * a[0], rtol=1e-14)


@pytest.mark.parametrize(
    "prm, T, M", [(RHO_P5, 2.0, 8), (XI15, 1.0, 4), (FELLER_X, 1.0, 32), (RHO_M99, 1.0, 4)]
)
@pytest.mark.parametrize("scheme", ["qe-m", "euler"])
def test_mc_forward_martingale_extra_sets(prm, T, M, scheme):
    """E[F_T] = F within 3 SE for the martingale schemes on sets the implementer did not use."""
    x, _ = mc.simulate(hp(prm), mc.uniform_grid(T, M), 100_000, np.random.default_rng(31), scheme)
    m, se = own_estimate(F0 * np.exp(x[0]))
    assert abs(m - F0) <= 3 * se


@pytest.mark.parametrize("scheme", ["euler", "qe", "qe-m"])
def test_mc_scheme_equals_own_implementation_at_coarse_dt(scheme):
    """At dt = 1 on A-I (T = 10) the discretisation bias is large (Euler ~ +6.4, QE ~ +1.0,
    QE-M ~ +0.23) and scheme-specific, so agreement of the implementation with OQE/OEU (other
    random numbers) within 3 combined SE certifies the scheme, not only the limit."""
    n = 200_000
    p = hp(AI)
    v, se, _, _ = mc.european_price(
        F0, 100.0, 10.0, 1.0, p, 1, n_steps=10, n_paths=n, seed=101, scheme=scheme
    )
    rng = np.random.default_rng(202)
    if scheme == "euler":
        x = own_euler(AI, 10.0, 10, n, rng)
    else:
        x = own_qe(AI, 10.0, 10, n, rng, martingale=(scheme == "qe-m"))
    m, s = own_estimate(np.maximum(F0 * np.exp(x) - 100.0, 0.0))
    assert abs(v - m) <= 3 * np.hypot(se, s), (v, m)


# ============================================================================ 8. Euler bias figure
def _bias_rows():
    with open(BIAS_CSV) as f:
        return list(csv.DictReader(f))


def test_euler_bias_csv_consistent_and_reproducible():
    """The figure's CSV: bias = MC - COS and bias/SE are recomputable; Euler |bias| decreases
    monotonically until inside 3 SE; the COS reference matches RIC+GP; one row re-runs
    bit-for-bit from its stored seed; every set is Feller-violating."""
    rows = _bias_rows()
    for r in rows:
        assert abs(float(r["mc_price"]) - float(r["cos_ref"]) - float(r["bias"])) < 1e-12
        assert abs(float(r["bias"]) / float(r["stderr"]) - float(r["bias_over_se"])) < 1e-9
    for name in ("A-I", "FO"):
        e = sorted(
            (r for r in rows if r["set"] == name and r["scheme"] == "euler"),
            key=lambda r: -float(r["dt"]),
        )
        assert float(e[0]["bias_over_se"]) > 3
        for a, b in zip(e, e[1:], strict=False):
            inside_noise = abs(float(b["bias_over_se"])) <= 3
            assert abs(float(a["bias"])) > abs(float(b["bias"])) or inside_noise
    fo_ref = {float(r["cos_ref"]) for r in rows if r["set"] == "FO"}.pop()
    own = ref_put(F0, 100.0, 1.0, 1.0, FO)[0]  # ATM: call = put
    assert abs(fo_ref - own) < 1e-8
    r = next(r for r in rows if r["set"] == "FO" and r["scheme"] == "euler" and r["dt"] == "0.5")
    x, _ = mc.simulate(
        hp(FO),
        mc.uniform_grid(1.0, int(r["n_steps"])),
        int(r["n_paths"]),
        np.random.default_rng(int(r["seed"])),
        "euler",
    )
    price = np.maximum(F0 * np.exp(x[0]) - 100.0, 0.0).mean()
    assert price == float(r["mc_price"])
    assert hp(FO).feller < 1 and hp(AI).feller < 1


def test_gate_sets_include_feller_violation():
    """ROADMAP: one Feller-violating set. COS-gate sets FO, GS and MC-gate sets FO, GS, BK2, A-I."""
    assert hp(FO).feller < 1 and hp(GS).feller < 1
    assert all(hp(s).feller < 1 for s in (FO, GS, BK2, AI))


# ============================================================================ slow
@pytest.mark.slow
@pytest.mark.parametrize("name", list(GATE_SETS))
def test_gate_full_grid_vs_own_reference(name):
    """Full gate grid (K 60..150 step 5 x T in {0.1,..,3} x DF {1, 0.95} x C/P) vs RIC+GP:
    COS < 1e-6, Lewis < 1e-8."""
    prm = GATE_SETS[name]
    p = hp(prm)
    for T in T_GRID:
        put1 = ref_put(F0, K_GRID, T, 1.0, prm)
        for DF in (1.0, 0.95):
            put = DF * put1
            call = put + DF * (F0 - K_GRID)
            for cp, ref in ((-1, put), (1, call)):
                assert np.abs(cos_price(F0, K_GRID, T, DF, p, cp) - ref).max() < 1e-6
                assert np.abs(lewis_price(F0, K_GRID, T, DF, p, cp) - ref).max() < 1e-8


MC_GATE = {  # name: (params, T, F, DF) -- the implementer's MC gate sets (D-018)
    "FO": (FO, 1.0, 100.0, 1.0),
    "GS": (GS, 1.0, 100.0, 1.0),
    "BK2": (BK2, 5.0, 100.0 * np.exp(0.25), np.exp(-0.25)),
    "AI": (AI, 10.0, 100.0, 1.0),
}
N_SEEDS, SEED0 = 20, 777_000  # declared before the first run; never changed


@pytest.mark.slow
@pytest.mark.parametrize("scheme", ["qe", "qe-m"])
@pytest.mark.parametrize("name", list(MC_GATE))
def test_qe_gate_z_distribution_over_seeds(name, scheme):
    """Not cherry-picked? 20 fresh seeds at the gate settings (2e5 paths, dt = 1/32, puts
    70/100/140). Reference: Lewis (validated above). Each z must be within 3 (gate); the mean of
    the 20 z per strike within 3 standard errors of 0 (i.e. no bias visible at 20x the paths)."""
    prm, T, F, DF = MC_GATE[name]
    K = np.array([70.0, 100.0, 140.0])
    ref = lewis_price(F, K, T, DF, hp(prm), -1)
    z = np.array([_z(F, K, T, DF, prm, ref, s, scheme) for s in range(N_SEEDS)])
    assert np.abs(z).max() <= 4.0  # |z| > 3 is expected at the 0.27% rate; |z| > 4 (6e-5) is not
    assert np.all(np.abs(z.mean(0)) <= 3 * z.std(0, ddof=1) / np.sqrt(N_SEEDS)), z.mean(0)


def _z(F, K, T, DF, prm, ref, s, scheme):
    v, se, _, _ = mc.european_price(
        F,
        K,
        T,
        DF,
        hp(prm),
        -1,
        n_steps=int(32 * T),
        n_paths=200_000,
        seed=SEED0 + s,
        scheme=scheme,
    )
    return (v - ref) / se


@pytest.mark.slow
@pytest.mark.parametrize("name", list(MC_GATE))
def test_own_qe_m_passes_the_same_gate(name):
    """Independent MC (OQE, QE-M) at the gate settings vs Lewis within 3 SE."""
    prm, T, F, DF = MC_GATE[name]
    K = np.array([70.0, 100.0, 140.0])
    x = own_qe(prm, T, int(32 * T), 200_000, np.random.default_rng(31_337))
    v, se = own_estimate(np.maximum(K[:, None] - F * np.exp(x), 0.0), DF)
    ref = lewis_price(F, K, T, DF, hp(prm), -1)
    assert np.all(np.abs(v - ref) <= 3 * se), (v - ref) / se


@pytest.mark.slow
def test_euler_bias_figure_independent_euler():
    """Own Euler-FT (OEU) reproduces the figure's Euler biases (MC - COS) within 3 combined SE,
    1e6 paths, FO dt in {1/2, 1/4, 1/8, 1/16} and A-I dt in {1, 1/4}."""
    rows = _bias_rows()
    for name, prm, T, dts in (
        ("FO", FO, 1.0, (0.5, 0.25, 0.125, 0.0625)),
        ("A-I", AI, 10.0, (1.0, 0.25)),
    ):
        for j, dt in enumerate(dts):
            r = next(
                r
                for r in rows
                if r["set"] == name and r["scheme"] == "euler" and abs(float(r["dt"]) - dt) < 1e-12
            )
            x = own_euler(prm, T, int(round(T / dt)), 1_000_000, np.random.default_rng(55 + j))
            m, s = own_estimate(np.maximum(F0 * np.exp(x) - 100.0, 0.0))
            assert abs(m - float(r["mc_price"])) <= 3 * np.hypot(s, float(r["stderr"])), (name, dt)


@pytest.mark.slow
@pytest.mark.parametrize("prm, T", [(XI15, 1.0), (RHO_M99, 1.0), (FELLER_X, 1.0), (RHO_P5, 2.0)])
def test_qe_m_extreme_sets_vs_lewis(prm, T):
    """QE-M at dt = 1/32, 2e5 paths on edge sets (high xi, rho -> -1, nu_F = 0.004, rho > 0)."""
    K = np.array([70.0, 100.0, 140.0])
    ref = lewis_price(F0, K, T, 1.0, hp(prm), -1)
    v, se, _, _ = mc.european_price(
        F0, K, T, 1.0, hp(prm), -1, n_steps=int(32 * T), n_paths=200_000, seed=4242, scheme="qe-m"
    )
    assert np.all(np.abs(v - ref) <= 3 * se), (v - ref) / se


# ============================================================================ script mode
GOLDEN_CASES = [  # (set name, params, T); K = 60, 80, 100, 120, 150; F = 100, DF = 1; puts
    *((n, GATE_SETS[n], T) for n in GATE_SETS for T in (0.1, 1.0, 3.0)),
    ("FO", FO, 0.02),
    ("XI15", XI15, 0.1),
    ("XI15", XI15, 1.0),
    ("RHO_M99", RHO_M99, 0.1),
    ("RHO_M99", RHO_M99, 1.0),
    ("SILENT", SILENT, 1.0),
]
GOLDEN_K = [60.0, 80.0, 100.0, 120.0, 150.0]


def build_golden() -> dict:
    """Two independent methods per case: RIC+GP (hd = 0.1) and H93 (via parity); |A - B| < 2e-9."""
    cases = []
    for name, prm, T in GOLDEN_CASES:
        K = np.array(GOLDEN_K)
        a = ref_put(F0, K, T, 1.0, prm, hd=0.1, h_fine=0.125, h_coarse=1.0)
        U = ref_cutoff(T, prm)
        b = h93_call(F0, K, T, 1.0, prm, U=U, h=0.25) - (F0 - K)
        diff = float(np.abs(a - b).max())
        assert diff < 2e-9, (name, T, diff)
        cases.append(
            dict(
                set=name,
                params=list(prm),
                F=F0,
                DF=1.0,
                T=T,
                K=GOLDEN_K,
                put=[float(v) for v in a],
                h93_minus_ric=float(np.max(b - a)),
                max_abs_method_diff=diff,
            )
        )
        print(f"  golden {name:8s} T={T:<5} |RIC+GP - H93| = {diff:.1e}", flush=True)
    return {
        "topic": "Heston European put prices (forward-based)",
        "created_by": "validator agent, Phase 2 independent validation, 2026-11-07",
        "source": (
            "Agreement of two independent methods, neither using qpricer code: (A) RIC+GP = "
            "characteristic function from RK4 integration (one Richardson step, h(|d|+|beta|) "
            "<= 0.1) of the Heston Riccati ODEs, inverted by the Gil-Pelaez/Heston put formula "
            "with composite 16-point Gauss-Legendre (panels 0.125 on [0, 8], 1.0 beyond, cut-off "
            "where |phi| < 1e-15 u); (B) H93 = Heston (1993) closed-form P1/P2 with the original "
            "e^{+dT} CF, log continued by phase unwrapping, 16-point Gauss-Legendre on panels "
            "of 0.25, call converted to put by parity. Stored value = method A."
        ),
        "conventions": "put = DF E[(K - F_T)^+]; inputs F, DF, T (years); params "
        "[kappa, theta, xi, rho, v0]; dF = sqrt(v) F dW^F, "
        "dv = kappa(theta - v)dt + xi sqrt(v)dW^v",
        "accuracy": "max |A - B| over all cases is stored per case; all < 2e-9",
        "generator": "tests/validation/test_phase2_independent.py --golden-out <path>",
        "cases": cases,
    }


def _main() -> None:  # noqa: C901 -- a flat report script
    t00 = time.perf_counter()
    if "--golden-out" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--golden-out") + 1])
        if out.exists():
            raise SystemExit(f"{out} exists: never overwritten")
        out.write_text(json.dumps(build_golden(), indent=1) + "\n")
        print(f"wrote {out}")
        return

    print("== CF vs own Riccati RK4 (abs error, real axis + Im u = -1/2, -1)")
    for prm, T, umax, nm in (
        (FO, 0.02, 300, "FO"),
        (FO, 10, 12, "FO"),
        (KJ, 10, 6, "KJ"),
        (AI, 10, 12, "A-I"),
        (XI15, 1, 30, "XI15"),
        (RHO_M99, 1, 60, "RHO-.99"),
    ):
        u = np.geomspace(0.05, umax, 9)
        uu = np.concatenate([u, -u, u - 0.5j, u - 1j])
        e = np.abs(cf(uu, T, hp(prm)) - np.exp(ref_logcf(uu, T, prm))).max()
        print(f"  {nm:8s} T={T:<5} max|cf - RIC| = {e:.1e}")

    print("== cumulants vs MOM (relative)")
    w1 = w2 = 0.0
    for prm in (FO, GS, LW, AT2, AI, KJ, XI15, SMALL_K, LARGE_K):
        for T in (0.02, 1.0, 10.0):
            c, r = cumulants(T, hp(prm)), ref_cumulants(T, prm)
            w1, w2 = max(w1, abs(c[0] / r[0] - 1)), max(w2, abs(c[1] / r[1] - 1))
    print(f"  main branch: c1 {w1:.1e}, c2 {w2:.1e}")
    for th, v0 in ((0.5, 0.01), (0.04, 0.04)):
        prm = (0.999e-3, th, 0.5, -0.7, v0)
        print(
            f"  kappa T = 0.999e-3 branch (theta={th}, v0={v0}): c2 rel error "
            f"{cumulants(1.0, hp(prm))[1] / ref_cumulants(1.0, prm)[1] - 1:+.2e}"
        )

    print(
        "== Gate 1: COS vs Lewis and both vs RIC+GP, full grid "
        "(K 60..150/5, T 0.1..3, DF 1/.95, C/P)"
    )
    for name, prm in GATE_SETS.items():
        p, wcl, wco, wlo = hp(prm), 0.0, 0.0, 0.0
        for T in T_GRID:
            put1 = ref_put(F0, K_GRID, T, 1.0, prm)
            for DF in (1.0, 0.95):
                for cp in (1, -1):
                    ref = DF * put1 + (DF * (F0 - K_GRID) if cp > 0 else 0.0)
                    c, lw = (
                        cos_price(F0, K_GRID, T, DF, p, cp),
                        lewis_price(F0, K_GRID, T, DF, p, cp),
                    )
                    wcl = max(wcl, np.abs(c - lw).max())
                    wco = max(wco, np.abs(c - ref).max())
                    wlo = max(wlo, np.abs(lw - ref).max())
        print(
            f"  {name:4s} nu_F={p.feller:.2f}: |COS-Lewis| {wcl:.2e}  |COS-own| {wco:.2e}  "
            f"|Lewis-own| {wlo:.2e}"
        )

    print("== Gate 1 off-lattice (as test_gate_cos_vs_lewis_off_lattice): max |COS - Lewis|")
    rng = np.random.default_rng(2026_11_07)
    K = np.concatenate([[60.0, 150.0], rng.uniform(60.0, 150.0, 23)])
    for name, prm in GATE_SETS.items():
        w = 0.0
        for T in np.concatenate([[0.1, 3.0], rng.uniform(0.1, 3.0, 8)]):
            DF = rng.uniform(0.8, 1.0)
            for cp in (1, -1):
                c, _ = cos_quiet(F0, K, T, DF, hp(prm), cp)
                w = max(w, np.abs(c - lewis_price(F0, K, T, DF, hp(prm), cp)).max())
        print(f"  {name:4s}: {w:.2e}")

    print("== Gate 1 generalised: random SPX-plausible sets (seed 5, 300 sets x 6 T, puts, DF=1)")
    rng = np.random.default_rng(5)
    errs, warned, sets = [], [], []
    for _ in range(300):
        prm = (
            float(np.exp(rng.uniform(np.log(0.3), np.log(8.0)))),
            rng.uniform(0.01, 0.12),
            rng.uniform(0.1, 1.2),
            rng.uniform(-0.95, -0.2),
            rng.uniform(0.005, 0.12),
        )
        for T in T_GRID:
            c, w = cos_quiet(F0, K_GRID, T, 1.0, hp(prm), -1)
            errs.append(np.abs(c - lewis_price(F0, K_GRID, T, 1.0, hp(prm), -1)).max())
            warned.append(w)
            sets.append((prm, T))
    errs, warned = np.array(errs), np.array(warned)
    bad = errs > 1e-6
    print(
        f"  cells {errs.size}: |COS-Lewis| > 1e-6 in {bad.sum()} ({bad.mean():.1%}); "
        f"warned {warned.sum()}; silent failures {(bad & ~warned).sum()} "
        f"(max {errs[~warned].max():.2e}); worst warned {errs.max():.2e}"
    )
    j = int(np.argmax(np.where(warned, 0, errs)))
    print(f"  worst silent: params {np.round(sets[j][0], 4).tolist()} T={sets[j][1]}")
    for N, L in ((256, 18), (1024, 18), (4096, 18), (4096, 24), (4096, 30)):
        e = np.abs(
            cos_price(F0, K_GRID, 1.0, 1.0, hp(SILENT), -1, N=N, L=L)
            - ref_put(F0, K_GRID, 1.0, 1.0, SILENT)
        ).max()
        print(f"  SILENT set, T=1: N={N:<5d} L={L}: |COS - own| = {e:.2e}")
    print(
        f"  SILENT eq.13 estimate {cos_series_error(1.0, hp(SILENT)):.1e} (warning threshold 1e-5)"
    )

    print("== Edge sets: |COS-Lewis| (warned?) per T")
    for nm, prm in (
        ("XI15", XI15),
        ("RHO-.99", RHO_M99),
        ("RHO+.99", RHO_P99),
        ("FELLER_X", FELLER_X),
        ("V0=0", V0ZERO),
        ("KAPPA1e-4", (1e-4, 0.04, 0.5, -0.7, 0.04)),
    ):
        row = []
        for T in (0.02, 0.1, 0.5, 1.0, 3.0):
            c, w = cos_quiet(F0, K_GRID, T, 1.0, hp(prm), -1)
            row.append(
                f"T={T}: {np.abs(c - lewis_price(F0, K_GRID, T, 1.0, hp(prm), -1)).max():.1e}"
                f"{'(w)' if w else '   '}"
            )
        print(f"  {nm:9s} nu_F={hp(prm).feller:.3f} " + "  ".join(row))

    print("== Very short T, FO, K = 90..110: |COS - Lewis| and |COS - Black(sqrt v0)|")
    Ks = np.array([90.0, 95.0, 99.0, 100.0, 101.0, 105.0, 110.0])
    for T in (0.005, 0.001, 1e-4):
        c, w = cos_quiet(F0, Ks, T, 1.0, hp(FO), -1)
        lw = lewis_price(F0, Ks, T, 1.0, hp(FO), -1)
        b = ref_black(F0, Ks, T, 1.0, np.sqrt(FO[4]), -1)
        print(
            f"  T={T:g}: |COS-Lewis| {np.abs(c - lw).max():.1e} (warned {w}), "
            f"|COS-Black(sqrt v0)| {np.abs(c - b).max():.1e}"
        )

    print("== Published values vs own RIC+GP (hd = 0.1; calls by parity)")
    BK1 = (6.21, 0.019, 0.61, -0.70, 0.010201)
    for nm, prm, T, F, DF, pub in (
        ("FO1 Fang-Oosterlee T4", FO, 1.0, 100.0, 1.0, "5.785155450 (MPRA 5.785155435)"),
        ("FO2 Fang-Oosterlee T5", FO, 10.0, 100.0, 1.0, "22.318945791"),
        ("BK-1 Broadie-Kaya T4", BK1, 1.0, 100 * np.exp(0.0319), np.exp(-0.0319), "6.8061"),
        ("BK-2 Broadie-Kaya T5", BK2, 5.0, 100 * np.exp(0.25), np.exp(-0.25), "34.9998"),
        ("A-I Choi-Kwok", AI, 10.0, 100.0, 1.0, "13.08467014"),
    ):
        c = ref_put(F, 100.0, T, DF, prm, hd=0.1, h_fine=0.125)[0] + DF * (F - 100.0)
        lw = float(lewis_price(F, 100.0, T, DF, hp(prm), 1))
        print(f"  {nm:22s}: own {c:.10f}  Lewis {lw:.10f}  published {pub}")

    print("== Gate 2: xi = 1e-4")
    for k, th, v0 in ((1.5, 0.04, 0.04), (1.5, 0.04, 0.09), (1.5, 0.09, 0.01), (5.0, 0.02, 0.2)):
        p, w = hp((k, th, 1e-4, 0.0, v0)), 0.0
        for T in T_GRID:
            sig = np.sqrt(ref_int_var(T, (k, th, 1e-4, 0.0, v0)) / T)
            for DF in (1.0, 0.95):
                for cp in (1, -1):
                    b = ref_black(F0, K_GRID, T, DF, sig, cp)
                    for eng in (cos_price, lewis_price):
                        w = max(w, np.abs(eng(F0, K_GRID, T, DF, p, cp) - b).max())
        print(f"  rho=0 kappa={k} theta={th} v0={v0}: max |engine - Black(sigma_eff)| = {w:.2e}")
    k, th, v0, rho, h = 1.5, 0.04, 0.04, -0.7, 1e-3
    for T in (0.1, 1.0, 3.0):
        P = {
            s: ref_put(F0, K_GRID, T, 1.0, (k, th, s * h, rho, v0), U=400.0) for s in (1, -1, 2, -2)
        }
        a1 = (8 * (P[1] - P[-1]) - (P[2] - P[-2])) / (12 * h)
        tau, E = (1 - np.exp(-k * T)) / k, np.exp(-k * T)
        S = th * T + (v0 - th) * tau
        J = th / k * (T - tau) + (v0 - th) / k * (tau - T * E)
        d1 = (np.log(F0 / K_GRID) + 0.5 * S) / np.sqrt(S)
        eq16 = rho * J * (-F0 * norm.pdf(d1) * (d1 - np.sqrt(S)) / (2 * S))
        p = hp((k, th, 1e-4, rho, v0))
        black = ref_black(F0, K_GRID, T, 1.0, np.sqrt(S / T), -1)
        v = lewis_price(F0, K_GRID, T, 1.0, p, -1)
        vc = cos_price(F0, K_GRID, T, 1.0, p, -1)
        print(
            f"  rho=-0.7 T={T}: |dP/dxi(own FD) - eq.16| = {np.abs(a1 - eq16).max():.1e} "
            f"(max |dP/dxi| {np.abs(a1).max():.2f}); |Lewis - Black - xi a1| = "
            f"{np.abs(v - black - 1e-4 * a1).max():.1e}, "
            f"COS {np.abs(vc - black - 1e-4 * a1).max():.1e};"
            f" gap to plain Black {np.abs(v - black).max():.2e}"
        )

    print(
        "== MC: scheme identity at dt = 1, A-I, ATM call (theirs seed 101 vs own, 2e5 paths each)"
    )
    ref = float(lewis_price(F0, 100.0, 10.0, 1.0, hp(AI), 1))
    pub = {"euler": "+6.394 (0.029)", "qe": "+1.022 (0.013)", "qe-m": "+0.233 (n/a)"}
    for scheme in ("euler", "qe", "qe-m"):
        v, se, _, _ = mc.european_price(
            F0, 100.0, 10.0, 1.0, hp(AI), 1, n_steps=10, n_paths=200_000, seed=101, scheme=scheme
        )
        rng = np.random.default_rng(202)
        x = (
            own_euler(AI, 10.0, 10, 200_000, rng)
            if scheme == "euler"
            else own_qe(AI, 10.0, 10, 200_000, rng, martingale=(scheme == "qe-m"))
        )
        m, s = own_estimate(np.maximum(F0 * np.exp(x) - 100.0, 0.0))
        print(
            f"  {scheme:5s}: bias theirs {v - ref:+.4f} ({se:.4f}), own {m - ref:+.4f} ({s:.4f}), "
            f"z(diff) {(v - m) / np.hypot(se, s):+.2f}; Andersen 2006 Table 2 (1e6 paths, via "
            f"heston-mc.md): {pub[scheme]}"
        )

    print("== Euler bias CSV")
    rows = _bias_rows()
    for name in ("A-I", "FO"):
        e = [r for r in rows if r["set"] == name and r["scheme"] == "euler"]
        print(
            f"  {name}: "
            + ", ".join(
                f"dt={float(r['dt']):g}: {float(r['bias']):+.4f} "
                f"({float(r['bias_over_se']):+.1f} SE)"
                for r in e
            )
        )
        sig = [
            (float(r["dt"]), abs(float(r["bias"]))) for r in e if abs(float(r["bias_over_se"])) > 3
        ]
        lx, ly = np.log(np.array(sig)).T
        print(
            f"  {name}: fitted Euler order (|bias| > 3 SE points, n={len(sig)}): "
            f"{np.polyfit(lx, ly, 1)[0]:.3f}"
        )
    if "--mc" in sys.argv:
        print("== Scheme identity at dt = 1, A-I, ATM call, 5e6 paths each (10 batches)")
        ref = float(lewis_price(F0, 100.0, 10.0, 1.0, hp(AI), 1))
        for scheme in ("qe", "qe-m"):
            th, mi = [], []
            for b in range(10):
                x, _ = mc.simulate(
                    hp(AI),
                    mc.uniform_grid(10.0, 10),
                    500_000,
                    np.random.default_rng(9000 + b),
                    scheme,
                )
                th.append(np.maximum(F0 * np.exp(x[0]) - 100.0, 0.0))
                y = own_qe(
                    AI,
                    10.0,
                    10,
                    500_000,
                    np.random.default_rng(8000 + b),
                    martingale=(scheme == "qe-m"),
                )
                mi.append(np.maximum(F0 * np.exp(y) - 100.0, 0.0))
            a, sa = own_estimate(np.concatenate(th))
            m, sm = own_estimate(np.concatenate(mi))
            print(
                f"  {scheme:5s}: bias theirs {a - ref:+.4f} ({sa:.4f}), own {m - ref:+.4f} "
                f"({sm:.4f}), z(diff) {(a - m) / np.hypot(sa, sm):+.2f}"
            )
        print("== CSV points at ~3 SE re-run with 4e6 paths each (theirs vs own)")
        for nm, prm, T, M, scheme in (("A-I", AI, 10.0, 80, "qe"), ("FO", FO, 1.0, 2, "qe-m")):
            ref = float(lewis_price(F0, 100.0, T, 1.0, hp(prm), 1))
            th, mi = [], []
            for b in range(8):
                x, _ = mc.simulate(
                    hp(prm), mc.uniform_grid(T, M), 500_000, np.random.default_rng(5000 + b), scheme
                )
                th.append(np.maximum(F0 * np.exp(x[0]) - 100.0, 0.0))
                y = own_qe(
                    prm,
                    T,
                    M,
                    500_000,
                    np.random.default_rng(6000 + b),
                    martingale=(scheme == "qe-m"),
                )
                mi.append(np.maximum(F0 * np.exp(y) - 100.0, 0.0))
            a, sa = own_estimate(np.concatenate(th))
            m, sm = own_estimate(np.concatenate(mi))
            print(
                f"  {nm} {scheme} dt={T / M:g}: bias theirs {a - ref:+.4f} ({sa:.4f}), "
                f"own {m - ref:+.4f} ({sm:.4f})"
            )
        print("== Own Euler (1e6 paths) vs CSV Euler mc_price")
        for name, prm, T, dts in (
            ("FO", FO, 1.0, (0.5, 0.25, 0.125, 0.0625)),
            ("A-I", AI, 10.0, (1.0, 0.25)),
        ):
            for j, dt in enumerate(dts):
                r = next(
                    r
                    for r in rows
                    if r["set"] == name
                    and r["scheme"] == "euler"
                    and abs(float(r["dt"]) - dt) < 1e-12
                )
                x = own_euler(prm, T, int(round(T / dt)), 1_000_000, np.random.default_rng(55 + j))
                m, s = own_estimate(np.maximum(F0 * np.exp(x) - 100.0, 0.0))
                print(
                    f"  {name} dt={dt:g}: own {m:.4f} ({s:.4f}) vs CSV {float(r['mc_price']):.4f} "
                    f"({float(r['stderr']):.4f}): "
                    f"z {(m - float(r['mc_price'])) / np.hypot(s, float(r['stderr'])):+.2f}"
                )
        print(f"== QE gate over {N_SEEDS} fresh seeds (z = (MC - Lewis)/SE, puts 70/100/140)")
        K = np.array([70.0, 100.0, 140.0])
        allz = []
        for name, (prm, T, F, DF) in MC_GATE.items():
            ref = lewis_price(F, K, T, DF, hp(prm), -1)
            for scheme in ("qe", "qe-m"):
                z = np.array([_z(F, K, T, DF, prm, ref, s, scheme) for s in range(N_SEEDS)])
                allz.append(z)
                print(
                    f"  {name:4s} {scheme:5s}: mean z {np.round(z.mean(0), 2).tolist()}, sd z "
                    f"{np.round(z.std(0, ddof=1), 2).tolist()}, max|z| {np.abs(z).max():.2f}, "
                    f"#|z|>3 {(np.abs(z) > 3).sum()}"
                )
        allz = np.concatenate([a.ravel() for a in allz])
        print(
            f"  pooled: {allz.size} z-scores, mean {allz.mean():+.3f}, sd {allz.std(ddof=1):.3f}, "
            f"#|z|>3 {(np.abs(allz) > 3).sum()}"
        )
        print("== Own QE-M at the gate settings (seed 31337) and QE-M on edge sets (seed 4242)")
        for name, (prm, T, F, DF) in MC_GATE.items():
            x = own_qe(prm, T, int(32 * T), 200_000, np.random.default_rng(31_337))
            v, se = own_estimate(np.maximum(K[:, None] - F * np.exp(x), 0.0), DF)
            print(
                f"  own QE-M {name:4s}: z "
                f"{np.round((v - lewis_price(F, K, T, DF, hp(prm), -1)) / se, 2).tolist()}"
            )
        for nm, prm, T in (
            ("XI15", XI15, 1.0),
            ("RHO-.99", RHO_M99, 1.0),
            ("FELLER_X", FELLER_X, 1.0),
            ("RHO+.5", RHO_P5, 2.0),
        ):
            ref = lewis_price(F0, K, T, 1.0, hp(prm), -1)
            for scheme in ("qe-m", "euler"):
                v, se, _, _ = mc.european_price(
                    F0,
                    K,
                    T,
                    1.0,
                    hp(prm),
                    -1,
                    n_steps=int(32 * T),
                    n_paths=200_000,
                    seed=4242,
                    scheme=scheme,
                )
                print(f"  {nm:9s} {scheme:5s}: z {np.round((v - ref) / se, 2).tolist()}")
    print(f"(total {time.perf_counter() - t00:.0f} s)")


if __name__ == "__main__":
    _main()
