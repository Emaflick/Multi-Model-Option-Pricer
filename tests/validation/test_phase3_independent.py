"""Independent validation of Phase 3 (SABR, beta = 1: Hagan lognormal vol + MC) -- validator.

The qpricer public API (``sabr.z_over_x``, ``sabr.implied_vol``, ``sabr.alpha_from_atm``,
``sabr.simulate``, ``sabr.mc_price``) is only called as the system under test. Every reference
value is computed here, with methods that differ from the implementation:

* HAGd -- Hagan et al. (2002) eq. (2.17a-c) at beta = 1 (= (B.71b-c)), transcribed from the
          paper: sigma = alpha (z / x(z)) (1 + [rho nu alpha / 4 + (2 - 3 rho^2) nu^2 / 24] T),
          z = (nu / alpha) ln(F/K), x(z) = ln[(sqrt(1 - 2 rho z + z^2) + z - rho) / (1 - rho)],
          evaluated in stdlib ``decimal`` at 60 digits on the exact double inputs (no series, no
          branch split, no log1p rewrite). Correctly rounded reference for every z incl. z -> 0.
* HAGa -- the same formula in double precision but with x(z) in the closed-form arcsinh
          antiderivative of int_0^z du / sqrt((u - rho)^2 + 1 - rho^2):
          x = asinh((z - rho)/s) + asinh(rho/s), s = sqrt(1 - rho^2) (Obloj 2008 integral form).
          Used on wide vectorised grids where |z| is not small.
* BLK  -- own Black-76 in ln F (scipy.special.ndtr) and own implied vol by scipy brentq on the
          total vol (neither ``bsm`` nor ``iv`` is used for reference numbers).
* RQMC -- model price by the exact conditional (mixing) representation of beta = 1 SABR
          (Romano-Touzi 1997; Hull-White 1987 at rho = 0): given the alpha path,
          ln F_T ~ N(ln F + rho (alpha_T - alpha)/nu - I/2, (1 - rho^2) I), I = int alpha^2 dt.
          The alpha path is built *by Brownian bridge* from *scrambled Sobol* points (randomised
          QMC, R independent scrambles -> honest error bar), I by *Simpson's rule*; alpha is exact
          on the grid. Implementation: incremental pseudo-random draws, trapezoid I.
* SCH  -- exact expectation of the implementer's discrete scheme (14) (log-Euler, exact alpha)
          at the *same* dt: conditional on the alpha path the scheme's ln F_T is Gaussian with
          mean rho J_hat - I_hat/2 and variance (1 - rho^2) I_hat, I_hat = dt sum alpha_n^2
          (left point), J_hat = sum alpha_n (W_{n+1} - W_n). Computed by RQMC. MC-vs-SCH tests
          the implementation's discrete law free of both Hagan error and dt bias.
* PDE  -- 2-D backward PDE in (ln F, ln alpha), Modified Craig-Sneyd ADI (In 't Hout & Welfert
          2009, theta = 1/3), cell-averaged payoff, 2 damping Douglas theta = 1 half steps,
          Richardson over (n, 2n) grids. Independent of the mixing identity and of any MC.
* OMC  -- own pseudo-random payoff-level MC (the literal gate with an independent MC): exact
          conditional sampling of ln F_T given the alpha path (as RQMC but random draws), alpha
          path by cumulative sums of (N, M) row-major draws, Simpson I, dt = 1/500.

Run as a script (``.venv/bin/python tests/validation/test_phase3_independent.py``) to print the
numbers quoted in ``docs/validation/phase3.md``; ``--seeds`` adds the multi-seed z-score study
of the gate (100 fresh seeds, about 5 minutes); ``--golden-out PATH`` writes the golden JSON to
PATH only if PATH does not exist (never overwrites). ``tests/golden/sabr_prices.json`` was created
from that output (written to a scratch path, then copied verbatim; the guard hook blocks shell
writes into tests/golden/).
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sps
from scipy.optimize import brentq
from scipy.sparse.linalg import splu
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

from qpricer.models import sabr

REPO = Path(__file__).resolve().parents[2]
GOLDEN = REPO / "tests" / "golden" / "sabr_prices.json"
FIG_CSV = REPO / "report" / "figures" / "sabr_hagan_vs_mc.csv"

F0 = 100.0
G = (0.2, -0.75, 0.4)  # gate set (alpha, rho, nu) declared in sabr.md §5 / D-023
K_GATE = F0 * np.exp(np.array([-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2]))
T_GATE = (0.25, 0.5, 1.0)
N_GATE = 200_000
# Choi-Hu-Kwok (2025) Table 1, FDM (attributed to Cai-Song-Chen 2017): beta = 1, alpha = 0.2,
# T = 1, F = K = 1, call; 5 decimals. Row 5 is printed with nu = 0.6 in the table, 0.8 in the text.
CHK = (
    (-0.75, 0.2, 0.07910),
    (-0.50, 0.2, 0.07942),
    (-0.25, 0.2, 0.07969),
    (-0.75, 0.4, 0.07860),
)
CHK_ROW5 = (-0.75, 0.07811)  # nu ambiguous (0.6 / 0.8): resolved below


M_GATE = 250  # mirror of tests/test_sabr.py M_GATE / n_steps (fix round 1: dt = T/250 at every T)


def n_steps(T: float) -> int:
    """Mirror of the implementer's gate step count (tests/test_sabr.py::n_steps, read from the code
    after fix round 1): 250 steps at every maturity. Before the fix it was ceil(250 T)."""
    return M_GATE


def n_steps_old(T: float) -> int:
    """Pre-fix gate step count (dt = 1/250), kept only to reproduce finding F1's numbers."""
    return int(np.ceil(T * 250 - 1e-9))


# ============================================================================ BLK: Black-76
def black(lnF, K, s, cp, DF=1.0):
    """DF * Black-76 with forward e^{lnF}, total std s = sigma sqrt(T) > 0, cp = +1 / -1."""
    lnF, K, s, cp = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (lnF, K, s, cp)))
    d1 = (lnF - np.log(K)) / s + 0.5 * s
    return DF * cp * (np.exp(lnF) * ndtr(cp * d1) - K * ndtr(cp * (d1 - s)))


def black_vol(F, K, T, sigma, cp, DF=1.0):
    return black(np.log(F), K, np.asarray(sigma) * np.sqrt(T), cp, DF)


def iv_brent(price, F, K, T, cp, DF=1.0):
    """Own implied vol: brentq on sigma in [1e-4, 5] (scalar loop; references only)."""
    out = []
    for p, k, c in zip(*np.broadcast_arrays(np.asarray(price, float), K, cp), strict=True):
        f = lambda s, p=p, k=k, c=c: float(black_vol(F, k, T, s, c, DF)) - p  # noqa: E731
        out.append(brentq(f, 1e-4, 5.0, xtol=1e-15, rtol=1e-15, maxiter=500))
    return np.array(out)


def black_vega(F, K, T, sigma, DF=1.0):
    s = sigma * np.sqrt(T)
    d1 = np.log(F / K) / s + 0.5 * s
    return DF * F * np.sqrt(T) * np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi)


# ============================================================================ HAG: Hagan beta = 1
def zx_decimal(z: float, rho: float, prec: int = 60) -> float:
    """z / x(z) of [H02] (2.17b-c) in 60-digit decimal, on the exact binary inputs."""
    if z == 0.0:
        return 1.0
    with localcontext() as c:
        c.prec = prec
        Z, R = Decimal(z), Decimal(rho)
        D = (1 - 2 * R * Z + Z * Z).sqrt()
        x = ((D + Z - R) / (1 - R)).ln()
        return float(Z / x)


def hagan_decimal(F, K, T, alpha, rho, nu, prec: int = 60) -> float:
    """[H02] (2.17a) at beta = 1 in decimal: alpha (z / x(z)) (1 + c1 T), exact double inputs."""
    with localcontext() as c:
        c.prec = prec
        a, r, n, t = (Decimal(v) for v in (alpha, rho, nu, T))
        lfk = Decimal(F).ln() - Decimal(K).ln()
        z = n / a * lfk
        if z == 0:
            zx = Decimal(1)
        else:
            D = (1 - 2 * r * z + z * z).sqrt()
            zx = z / ((D + z - r) / (1 - r)).ln()
        br = 1 + (r * n * a / 4 + (2 - 3 * r * r) * n * n / 24) * t
        return float(a * zx * br)


def hagan_asinh(F, K, T, alpha, rho, nu):
    """[H02] beta = 1 with x(z) = asinh((z - rho)/s) + asinh(rho/s); valid for |z| not small."""
    F, K, T, alpha, rho, nu = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (F, K, T, alpha, rho, nu))
    )
    z = nu / alpha * np.log(F / K)
    s = np.sqrt((1 - rho) * (1 + rho))
    x = np.arcsinh((z - rho) / s) + np.arcsinh(rho / s)
    with np.errstate(invalid="ignore"):  # z = 0 -> 0/0; callers mask |z| small
        return alpha * (z / x) * (1 + (rho * nu * alpha / 4 + (2 - 3 * rho**2) * nu**2 / 24) * T)


def bracket(T, alpha, rho, nu):
    return 1 + (rho * nu * alpha / 4 + (2 - 3 * rho**2) * nu**2 / 24) * T


def hagan_put(K, T, alpha, rho, nu, F=F0):
    sig = np.array([hagan_decimal(F, k, T, alpha, rho, nu) for k in np.atleast_1d(K)])
    return black_vol(F, np.atleast_1d(K), T, sig, -1)


# ============================================================================ RQMC / SCH
def _bridge_plan(M: int):
    """Brownian-bridge order on grid indices 0..M: list of (mid, left, right)."""
    plan, queue = [], [(0, M)]
    while queue:
        nxt = []
        for lo, hi in queue:
            if hi - lo >= 2:
                m = (lo + hi) // 2
                plan.append((m, lo, hi))
                nxt += [(lo, m), (m, hi)]
        queue = nxt
    return plan


def _bridge_paths(z, T, M, plan):
    """W on t_j = j T / M from standard normals z (n, M), dim 0 -> W_T, then bisection."""
    dt = T / M
    W = np.zeros((z.shape[0], M + 1))
    W[:, M] = np.sqrt(T) * z[:, 0]
    for d, (m, lo, hi) in enumerate(plan, start=1):
        wl, wr = (hi - m) / (hi - lo), (m - lo) / (hi - lo)
        W[:, m] = (
            wl * W[:, lo] + wr * W[:, hi] + np.sqrt((m - lo) * (hi - m) / (hi - lo) * dt) * z[:, d]
        )
    return W


def _cond_moments(W, T, M, alpha, nu, rule):
    """(J, I) for the conditional representation: rule simpson / trap (model), scheme (eq. 14)."""
    t = np.linspace(0.0, T, M + 1)
    dt = T / M
    a = alpha * np.exp(nu * W - 0.5 * nu * nu * t)
    a2 = a * a
    if rule == "scheme":
        return (a[:, :-1] * np.diff(W, axis=1)).sum(1), dt * a2[:, :-1].sum(1)
    J = alpha * W[:, -1] if nu == 0 else (a[:, -1] - alpha) / nu
    if rule == "simpson":
        if M % 2:
            raise ValueError("Simpson needs even M")
        w = np.ones(M + 1)
        w[1:-1:2], w[2:-1:2] = 4.0, 2.0
        return J, dt / 3.0 * (a2 @ w)
    return J, dt * (a2[:, 1:-1].sum(1) + 0.5 * (a2[:, 0] + a2[:, -1]))


def rqmc_price(F, K, T, alpha, rho, nu, cp=-1, *, M, log2n=14, R=8, seed=7, rule="simpson"):
    """RQMC of E[Black(F xi, K, (1 - rho^2) I)], xi = exp(rho J - rho^2 I / 2). (mean, se)."""
    K = np.atleast_1d(np.asarray(K, float))
    cp = np.broadcast_to(np.asarray(cp, float), K.shape)
    plan = _bridge_plan(M)
    chunk = min(2**log2n, max(2**8, 2 ** int(np.log2(2**22 / (M + 1)))))
    est = []
    for rep in range(R):
        eng = qmc.Sobol(d=M, scramble=True, seed=np.random.default_rng([seed, rep]))
        acc, n = np.zeros(K.size), 0
        while n < 2**log2n:
            u = eng.random(chunk) + 2.0**-31  # Sobol points are k / 2^30: keep u in (0, 1)
            W = _bridge_paths(ndtri(u), T, M, plan)
            J, ivar = _cond_moments(W, T, M, alpha, nu, rule)
            lnf = np.log(F) + rho * J - 0.5 * rho * rho * ivar
            s = np.sqrt((1 - rho * rho) * ivar)
            acc += black(lnf[:, None], K, s[:, None], cp).sum(0)
            n += chunk
        est.append(acc / n)
    est = np.array(est)
    return est.mean(0), est.std(0, ddof=1) / np.sqrt(R)


def rqmc_halving(F, K, T, alpha, rho, nu, *, M, log2n=15, R=16, seed=7):
    """E[V_dt] - E[V_dt/2] of scheme (14) (SCH, puts) with dt = T/M, on *common* RQMC alpha paths:
    the coarse scheme uses every other node of the fine Brownian path, so the difference of the two
    exact conditional expectations is resolved far better than by two separate RQMC runs."""
    K = np.atleast_1d(np.asarray(K, float))
    plan = _bridge_plan(2 * M)
    chunk = 2 ** int(np.log2(2**22 / (2 * M + 1)))
    est = []
    for rep in range(R):
        eng = qmc.Sobol(d=2 * M, scramble=True, seed=np.random.default_rng([seed, rep]))
        acc, n = np.zeros(K.size), 0
        while n < 2**log2n:
            u = eng.random(chunk) + 2.0**-31
            W = _bridge_paths(ndtri(u), T, 2 * M, plan)
            v = []
            for Wg, MM in ((W[:, ::2], M), (W, 2 * M)):
                J, ivar = _cond_moments(Wg, T, MM, alpha, nu, "scheme")
                lnf = np.log(F) + rho * J - 0.5 * rho * rho * ivar
                v.append(
                    black(lnf[:, None], K, np.sqrt((1 - rho * rho) * ivar)[:, None], -1).sum(0)
                )
            acc += v[0] - v[1]
            n += chunk
        est.append(acc / n)
    est = np.array(est)
    return est.mean(0), est.std(0, ddof=1) / np.sqrt(R)


# ============================================================================ OMC: own payoff MC
def own_mc_put(F, K, T, alpha, rho, nu, *, M, N, seed, chunk=20_000):
    """Exact conditional sampling of ln F_T given the alpha path (Simpson I). (mean, se)."""
    K = np.atleast_1d(np.asarray(K, float))
    rng = np.random.default_rng(seed)
    dt, t = T / M, np.linspace(0.0, T, M + 1)
    w = np.ones(M + 1)
    w[1:-1:2], w[2:-1:2] = 4.0, 2.0
    s1, s2, n = np.zeros(K.size), np.zeros(K.size), 0
    while n < N:
        m = min(chunk, N - n)
        zperp = rng.standard_normal(m)  # layout: Z_perp first, then the (m, M) alpha block
        W = np.zeros((m, M + 1))
        W[:, 1:] = np.cumsum(np.sqrt(dt) * rng.standard_normal((m, M)), axis=1)
        a = alpha * np.exp(nu * W - 0.5 * nu * nu * t)
        ivar = dt / 3.0 * ((a * a) @ w)
        J = alpha * W[:, -1] if nu == 0 else (a[:, -1] - alpha) / nu
        lnFT = np.log(F) + rho * J - 0.5 * ivar + np.sqrt((1 - rho * rho) * ivar) * zperp
        g = np.maximum(K[:, None] - np.exp(lnFT)[None, :], 0.0)
        s1 += g.sum(1)
        s2 += (g * g).sum(1)
        n += m
    mean = s1 / N
    var = (s2 - N * mean * mean) / (N - 1)
    return mean, np.sqrt(var / N)


# ============================================================================ PDE: MCS ADI
def _d_ops(n, h):
    e = np.ones(n)
    D1 = sps.diags([-e[:-1], e[:-1]], [-1, 1], shape=(n, n)).tolil() / (2 * h)
    D2 = sps.diags([e[:-1], -2 * e, e[:-1]], [-1, 0, 1], shape=(n, n)).tolil() / (h * h)
    for D in (D1, D2):
        D[0, :] = 0
        D[-1, :] = 0
    return D1.tocsr(), D2.tocsr()


def pde_put(F, K, T, alpha, rho, nu, nx=401, ny=161, nt=200):
    """Put prices (DF = 1) of beta = 1 SABR on (x, y) = (ln F, ln alpha), backward in tau:
    u_tau = e^{2y}/2 (u_xx - u_x) + rho nu e^y u_xy + nu^2/2 (u_yy - u_y).
    x-edges frozen at the payoff (deep ITM put K - e^x, OTM 0); y-edges: alpha frozen (x-operator
    only). Spot node (ln F, ln alpha) is a grid node. Cell-averaged payoff; MCS theta = 1/3."""
    K = np.atleast_1d(np.asarray(K, float))
    X = max(8 * alpha * np.sqrt(T) * np.exp(2 * nu * np.sqrt(T)), 1.0)
    Y = max(6 * nu * np.sqrt(T), 0.5)
    x = np.log(F) + np.linspace(-X, X, nx)
    y = np.log(alpha) + np.linspace(-Y, Y, ny)
    hx = x[1] - x[0]
    Dx, Dxx = _d_ops(nx, hx)
    Dy, Dyy = _d_ops(ny, y[1] - y[0])
    ey = np.exp(y)
    A1 = sps.kron(sps.diags(0.5 * ey**2), Dxx - Dx, format="csr")
    A2 = sps.kron(0.5 * nu * nu * (Dyy - Dy), sps.identity(nx), format="csr")
    A0 = sps.kron(sps.diags(rho * nu * ey) @ Dy, Dx, format="csr")
    # x-edge rows of A1 inside the y-band are already zero (Dx, Dxx edge rows zeroed)
    A = (A0 + A1 + A2).tocsr()
    lo = x[:, None] - hx / 2
    hi = np.minimum(x[:, None] + hx / 2, np.log(K)[None, :])
    u0 = np.where(hi > lo, K * (hi - lo) - (np.exp(hi) - np.exp(lo)), 0.0) / hx
    u0[0], u0[-1] = K - np.exp(x[0]), 0.0
    U = np.tile(u0, (ny, 1))  # (nx*ny, nK), index = iy*nx + ix
    Id = sps.identity(nx * ny, format="csc")

    def run(U, dt, steps, th, mcs):
        L1 = splu((Id - th * dt * A1).tocsc())
        L2 = splu((Id - th * dt * A2).tocsc())
        for _ in range(steps):
            AU, A1U, A2U = A @ U, A1 @ U, A2 @ U
            Y0 = U + dt * AU
            Y2 = L2.solve(L1.solve(Y0 - th * dt * A1U) - th * dt * A2U)
            if not mcs:
                U = Y2
                continue
            Yt0 = Y0 + th * dt * (A0 @ (Y2 - U)) + (0.5 - th) * dt * (A @ Y2 - AU)
            U = L2.solve(L1.solve(Yt0 - th * dt * A1U) - th * dt * A2U)
        return U

    dt = T / nt
    U = run(U, dt / 2, 4, 1.0, False)
    U = run(U, dt, nt - 2, 1.0 / 3.0, True)
    return U.reshape(ny, nx, K.size)[ny // 2, nx // 2]


def pde_put_rich(F, K, T, alpha, rho, nu, n=(401, 161, 200)):
    """Richardson (2nd order) of grids n and 2n - 1 nodes; error estimate |fine - coarse| / 3."""
    c = pde_put(F, K, T, alpha, rho, nu, *n)
    f = pde_put(F, K, T, alpha, rho, nu, 2 * n[0] - 1, 2 * n[1] - 1, 2 * n[2])
    return f + (f - c) / 3, np.abs(f - c) / 3


# ============================================================================ shared fixtures
SEED_V = 73_102_026  # validator seeds: SEED_V + offset, fixed before any run (never re-seeded)


@pytest.fixture(scope="module")
def gate_refs():
    """Model truth (RQMC Simpson, M = 512) and exact scheme-(14) expectation at the gate dt."""
    a, r, nu = G
    out = {}
    for T in T_GATE:
        tru, tse = rqmc_price(F0, K_GATE, T, a, r, nu, -1, M=512, log2n=14, R=8, seed=101)
        sch, sse = rqmc_price(F0, K_GATE, T, a, r, nu, -1, M=n_steps(T), rule="scheme", log2n=14,
                              R=8, seed=303)  # fmt: skip
        out[T] = dict(tru=tru, tse=tse, sch=sch, sse=sse, hag=hagan_put(K_GATE, T, a, r, nu))
    return out


def _gate_se(T, seed):
    """SE of the implementer's gate estimator (N = 2e5, dt = 1/250, puts) for one seed."""
    a, r, nu = G
    return sabr.mc_price(F0, K_GATE, T, 1.0, a, r, nu, -1, n_steps=n_steps(T), n_paths=N_GATE,
                         seed=seed)  # fmt: skip


# ============================================================================ 1. gate: nu -> 0
def test_gate_nu_zero_is_flat_alpha():
    """ROADMAP gate: nu -> 0, beta = 1 -> flat vol alpha, tol 1e-10. nu = 0 on a domain wider than
    the gate (|k| <= 5, T up to 30, alpha 1e-4..5, rho = +-0.999); nu = 1e-12 on |k| <= 1, T <= 10,
    alpha <= 1 (by my first-order expansion the offset is ~1e-12 (|k|/2 + alpha^2 T/4))."""
    k = np.linspace(-5.0, 5.0, 401)
    worst0 = worst12 = 0.0
    for rho in (-0.999, -0.5, 0.0, 0.5, 0.999):
        for a in (1e-4, 0.05, 0.2, 1.0, 5.0):
            for T in (1e-4, 0.01, 1.0, 10.0, 30.0):
                s0 = sabr.implied_vol(1.0, np.exp(k), T, a, rho, 0.0)
                assert np.array_equal(s0, np.full_like(k, a))  # eq. (7): exact, series returns 1
                worst0 = max(worst0, np.abs(s0 - a).max())
                if T <= 10 and a <= 1:
                    m = np.abs(k) <= 1
                    s12 = sabr.implied_vol(1.0, np.exp(k[m]), T, a, rho, 1e-12)
                    worst12 = max(worst12, np.abs(s12 - a).max())
    assert worst0 < 1e-10 and worst12 < 1e-10, (worst0, worst12)


def test_nu_small_first_order_expansion():
    """My expansion of [H02] at beta = 1: sigma = alpha + rho nu (k/2 + alpha^2 T/4) + O(nu^2)
    (z/x = 1 - rho z/2 + O(z^2), z = -nu k/alpha). Approach to the limit, not only the limit."""
    k = np.array([-1.0, -0.2, -1e-9, 0.0, 0.3, 1.0])
    for a, rho, T in ((0.2, -0.7, 2.0), (0.05, 0.9, 0.1), (1.0, -0.999, 10.0)):
        s = sabr.implied_vol(1.0, np.exp(k), T, a, rho, 1e-7)
        slope = rho * (k / 2 + a * a * T / 4)
        assert np.all(np.abs((s - a) / 1e-7 - slope) < 1e-5), (s - a) / 1e-7 - slope


# ============================================================================ 2. Hagan formula
RHOS = (-0.999, -0.9, -0.5, 0.0, 0.3, 0.75, 0.999)


def test_z_over_x_vs_decimal_reference():
    """z/x(z) vs 60-digit decimal [H02] (2.17c) on |z| in [1e-12, 1e4], both signs, the series
    switch |z| = Z_C (+-1 ulp, +-1e-9 rel), z = rho, rho = +-0.999. Budget from sabr.md eq. (5):
    series truncation <= 0.05 z_c^4 = 5e-14, so 1e-13 for |z| < Z_C; the log1p branch is claimed
    O(eps) (D-024): 1e-14 = 45 eps for |z| >= Z_C."""
    zc = sabr.Z_C
    zs = np.concatenate(
        [
            np.geomspace(1e-12, 1e4, 200),
            zc * np.array([1 - 1e-9, 1 - 2**-52, 1, 1 + 2**-52, 1 + 1e-9]),
        ]
    )
    zs = np.concatenate([zs, -zs, [0.0]])
    for rho in RHOS:
        z = np.concatenate([zs, [rho, rho * (1 + 1e-12), rho * (1 - 1e-12)]])
        got = sabr.z_over_x(z, rho)
        ref = np.array([zx_decimal(v, rho) for v in z])
        err = np.abs(got / ref - 1)
        small = np.abs(z) < zc
        assert err[small].max() < 1e-13, (rho, err[small].max())
        assert err[~small].max() < 1e-14, (rho, err[~small].max())


def test_implied_vol_vs_decimal_random_domain():
    """sigma_H vs decimal [H02] on 1500 random points: F 1e-2..1e4, k in [-3, 3] (half of them
    |k| ~ 1e-3), T 1e-3..30, alpha 1e-3..5, nu 1e-4..5, rho in [-0.999, 0.999]. NaN exactly where my
    own bracket 1 + c1 T <= 0 (eq. 22), with a RuntimeWarning."""
    rng = np.random.default_rng(SEED_V + 1)
    n = 1500
    k = np.concatenate([rng.uniform(-3, 3, n // 2), rng.normal(0, 1e-3, n - n // 2)])
    T = np.exp(rng.uniform(np.log(1e-3), np.log(30), n))
    a = np.exp(rng.uniform(np.log(1e-3), np.log(5), n))
    r = rng.uniform(-0.999, 0.999, n)
    nu = np.exp(rng.uniform(np.log(1e-4), np.log(5), n))
    F = np.exp(rng.uniform(np.log(1e-2), np.log(1e4), n))
    K = F * np.exp(k)
    ok = bracket(T, a, r, nu) > 0
    assert (~ok).sum() > 5  # the NaN branch is exercised
    with pytest.warns(RuntimeWarning, match="1 \\+ c1 T <= 0"):
        got = sabr.implied_vol(F, K, T, a, r, nu)
    assert np.array_equal(np.isnan(got), ~ok)
    ref = np.array(
        [hagan_decimal(*v) for v in zip(F[ok], K[ok], T[ok], a[ok], r[ok], nu[ok], strict=True)]
    )
    err = np.abs(got[ok] / ref - 1)
    assert err.max() < 1e-13, err.max()


def test_implied_vol_vs_asinh_form_wide_grid():
    """Second closed form for x(z) (arcsinh antiderivative), vectorised grid with |z| >= 0.05."""
    k = np.linspace(-3, 3, 121)[:, None, None, None]
    T = np.array([0.01, 0.5, 2.0, 10.0])[None, :, None, None]
    rho = np.array(RHOS)[None, None, :, None]
    nu = np.array([0.1, 0.5, 1.5, 4.0])[None, None, None, :]
    a = 0.25
    with warnings.catch_warnings():  # T = 10, nu = 4: bracket <= 0 -> NaN (tested elsewhere)
        warnings.simplefilter("ignore", RuntimeWarning)
        got = sabr.implied_vol(100.0, 100.0 * np.exp(k), T, a, rho, nu)
    ref = hagan_asinh(100.0, 100.0 * np.exp(k), T, a, rho, nu)
    z = np.broadcast_to(np.abs(nu * k / a), got.shape)
    ok = (z >= 0.05) & np.isfinite(got)
    assert ok.sum() > 0.9 * got.size
    assert np.max(np.abs(got[ok] / ref[ok] - 1)) < 1e-12


def test_atm_is_alpha_times_bracket():
    """[H02] (2.18) at beta = 1: sigma_ATM = alpha (1 + c1 T), K = F exactly."""
    a = np.array([1e-3, 0.2, 2.0])[:, None, None]
    rho = np.array(RHOS)[None, :, None]
    T = np.array([1e-3, 1.0, 5.0])[None, None, :]
    ref = a * bracket(T, a, rho, 0.8)
    with pytest.warns(RuntimeWarning):  # rho = -0.999, alpha = 2, T = 5: bracket <= 0 -> NaN
        got = sabr.implied_vol(3.7, 3.7, T, a, rho, 0.8)
    ok = ref > 0
    assert np.array_equal(np.isnan(got), ~ok)
    assert np.max(np.abs(got[ok] / ref[ok] - 1)) < 4 * np.finfo(float).eps


def test_extreme_inputs_finite_no_warnings():
    """alpha 1e-4..5, nu 0..5, rho = +-0.999, k in [-10, 10], T 1e-4..30: where my bracket > 0 the
    result is finite, positive, warning-free; where it is <= 0 it is NaN (warned)."""
    k = np.linspace(-10, 10, 81)[:, None, None, None, None]
    a = np.array([1e-4, 0.2, 5.0])[None, :, None, None, None]
    rho = np.array([-0.999, 0.0, 0.999])[None, None, :, None, None]
    nu = np.array([0.0, 1e-3, 1.0, 5.0])[None, None, None, :, None]
    T = np.array([1e-4, 1.0, 30.0])[None, None, None, None, :]
    ok = np.broadcast_to(bracket(T, a, rho, nu) > 0, (81, 3, 3, 4, 3))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        s = sabr.implied_vol(1.0, np.exp(k), T, a, rho, nu)
    assert [str(w.message) for w in rec] == ["SABR: 1 + c1 T <= 0 (eq. 22), sigma_H set to NaN"]
    assert np.all(np.isfinite(s[ok]) & (s[ok] > 0)) and np.all(np.isnan(s[~ok]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s2 = sabr.implied_vol(1.0, np.exp(k), T[..., :2], a[:, :2], rho, nu[..., :3, :])
    assert np.all(np.isfinite(s2))  # all brackets > 0 on this sub-grid: no warning of any kind


def test_input_validation_raises():
    good = dict(F=100.0, K=100.0, T=1.0, alpha=0.2, rho=-0.5, nu=0.4)
    for key, bad in (("rho", 1.0), ("rho", -0.9991), ("rho", np.nan), ("alpha", 0.0),
                     ("alpha", -0.1), ("alpha", np.inf), ("nu", -1e-12), ("nu", np.nan),
                     ("F", 0.0), ("K", -1.0), ("K", np.nan), ("T", 0.0),
                     ("T", np.inf)):  # fmt: skip
        kw = {**good, key: bad}
        with pytest.raises(ValueError):
            sabr.implied_vol(kw["F"], kw["K"], kw["T"], kw["alpha"], kw["rho"], kw["nu"])
    assert sabr.implied_vol(100.0, 100.0, 1.0, 0.2, 0.999, 0.4) > 0  # the bound itself is legal
    assert sabr.implied_vol(100.0, 100.0, 1.0, 0.2, -0.999, 0.4) > 0


def test_shapes_and_elementwise_broadcast():
    """Broadcast over (3,1,1) x (1,4,1) x (1,1,5) with array parameters equals scalar calls
    bitwise; scalar in -> 0-d out."""
    K = np.array([80.0, 100.0, 125.0])[:, None, None]
    T = np.array([0.1, 0.5, 1.0, 3.0])[None, :, None]
    rho = np.array([-0.9, -0.3, 0.0, 0.3, 0.9])[None, None, :]
    s = sabr.implied_vol(100.0, K, T, 0.25, rho, 0.7)
    assert s.shape == (3, 4, 5)
    for i, j, m in np.ndindex(*s.shape):
        one = sabr.implied_vol(100.0, K[i, 0, 0], T[0, j, 0], 0.25, rho[0, 0, m], 0.7)
        assert np.ndim(one) == 0 and one == s[i, j, m]
    assert np.ndim(sabr.z_over_x(0.3, -0.5)) == 0


# ============================================================================ 3. alpha_from_atm
def _atm_coeffs(T, rho, nu):
    return rho * nu * T / 4, 1 + (2 - 3 * rho**2) * nu**2 * T / 24  # sigma = alpha (b + a alpha)


def test_alpha_from_atm_round_trip_and_root_choice():
    """Phase-5 domain (bounds eq. 11, T <= 2 as in the Phase-0 chain): sigma_ATM(alpha_hat) = s.
    D-024 claims 1e-14; that holds where b > 0. Where b < 0 < a (rho > 0.82, large nu^2 T) the form
    2s/(b + sqrt(b^2 + 4as)) cancels: my error model 8 eps (1 + b^2/(2as)) is asserted there.
    Root = smaller positive root (bracket >= b/2), equal to my own brentq on [0, vertex]."""
    rng = np.random.default_rng(SEED_V + 2)
    n = 6000
    s = np.exp(rng.uniform(np.log(0.03), np.log(1.5), n))
    T = np.exp(rng.uniform(np.log(1 / 365), np.log(2.0), n))
    r = rng.uniform(-0.999, 0.999, n)
    nu = np.exp(rng.uniform(np.log(1e-4), np.log(5.0), n))
    a, b = _atm_coeffs(T, r, nu)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        al = sabr.alpha_from_atm(s, T, r, nu)
    ok = np.isfinite(al)
    rt = np.abs(sabr.implied_vol(1.0, 1.0, T[ok], al[ok], r[ok], nu[ok]) / s[ok] - 1)
    eps = np.finfo(float).eps
    bpos = b[ok] > 0
    assert rt[bpos].max() < 1e-14, rt[bpos].max()
    bound = 8 * eps * (1 + b[ok] ** 2 / (2 * np.abs(a[ok]) * s[ok]))
    assert np.all(rt <= np.maximum(bound, 1e-14))
    # root choice: own brentq on the increasing branch [0, vertex] (a < 0) or [lo, inf) (a > 0)
    idx = np.flatnonzero(ok)[:400]
    for i in idx:
        f = lambda x, i=i: x * (b[i] + a[i] * x) - s[i]  # noqa: E731
        if a[i] < 0:
            lo, hi = 0.0, -b[i] / (2 * a[i])
        else:
            lo, hi = (0.0 if b[i] > 0 else -b[i] / a[i]), 1.0
            while f(hi) < 0:
                hi *= 2
        ref = brentq(f, lo, hi, xtol=1e-300, rtol=1e-15)
        assert abs(al[i] / ref - 1) < 1e-13
    assert np.all(b[ok] + a[ok] * al[ok] >= 0.5 * b[ok] - 1e-12)


def test_alpha_from_atm_nan_region_and_limits():
    """NaN (+ one RuntimeWarning) exactly where no positive root exists: a < 0 and (b <= 0 or
    s > b^2/(4|a|)), or a = 0 and b <= 0. nu = 0 gives alpha = s bitwise. Broadcast shape."""
    rng = np.random.default_rng(SEED_V + 3)
    n = 20000
    s = np.exp(rng.uniform(np.log(0.01), np.log(3), n))
    T = np.exp(rng.uniform(np.log(1e-3), np.log(30), n))
    r = rng.uniform(-0.999, 0.999, n)
    nu = np.exp(rng.uniform(np.log(1e-4), np.log(5), n))
    a, b = _atm_coeffs(T, r, nu)
    exists = np.where(a < 0, (b > 0) & (s < b * b / (4 * -a)), np.where(a > 0, True, b > 0))
    with pytest.warns(RuntimeWarning, match="no positive alpha"):
        al = sabr.alpha_from_atm(s, T, r, nu)
    assert (~exists).sum() > 50 and np.array_equal(np.isnan(al), ~exists)
    assert np.array_equal(sabr.alpha_from_atm(s[:500], T[:500], r[:500], 0.0), s[:500])
    shp = sabr.alpha_from_atm(0.2, np.array([[0.1], [1.0]]), np.array([-0.5, 0.0, 0.5]), 0.4)
    assert np.shape(shp) == (2, 3) and np.ndim(sabr.alpha_from_atm(0.2, 1.0, -0.5, 0.4)) == 0
    # SPX-like long-dated example: feasibility ceiling b^2/(4|a|) = 4.57% at rho -0.9, nu 2, T 5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        v = sabr.alpha_from_atm(np.array([0.045, 0.046, 0.20]), 5.0, -0.9, 2.0)
    assert np.isfinite(v[0]) and np.all(np.isnan(v[1:]))
    for bad in ((0.0, 1, 0, 0.4), (0.2, 0, 0, 0.4), (0.2, 1, 1.0, 0.4), (0.2, 1, 0, -0.1),
                (np.nan, 1, 0, 0.4)):  # fmt: skip
        with pytest.raises(ValueError):
            sabr.alpha_from_atm(*bad)


# ============================================================================ 4. MC: API contract
def test_simulate_matches_hand_recursion():
    """sabr.simulate on 2 given steps equals my hand evaluation of eqs. (13)-(14) (rel 1e-14)."""
    a0, rho, nu, dt = 0.3, -0.6, 0.9, 0.05
    z = [np.array([[0.7, -1.2], [0.1, 2.0]]), np.array([[-0.4, 0.5], [1.3, -0.8]])]
    x, aT, w, il = sabr.simulate(a0, rho, nu, dt, iter(z))
    rb, sq = np.sqrt(1 - rho * rho), np.sqrt(dt)
    xr, ar, wr, ir = np.zeros(2), np.full(2, a0), np.zeros(2), np.zeros(2)
    for n, zz in enumerate(z, start=1):
        xr = xr - 0.5 * ar * ar * dt + ar * sq * (rho * zz[0] + rb * zz[1])
        ir = ir + ar * ar * dt
        wr = wr + sq * zz[0]
        ar = a0 * np.exp(nu * wr - 0.5 * nu * nu * n * dt)
    for g, ref in ((x, xr), (aT, ar), (w, wr), (il, ir)):
        np.testing.assert_allclose(g, ref, rtol=1e-14, atol=1e-16)


def test_mc_contract_shapes_reproducibility_invariances():
    """(price, stderr, n_paths, seed): types, shapes (scalar K, (3,), (2,3) with cp broadcast),
    bitwise reproducibility, seed sensitivity, strikes on common paths, exact DF scaling,
    scale invariance of beta = 1 (lambda F, lambda K) -> lambda V."""
    kw = dict(n_steps=10, n_paths=2000, seed=SEED_V + 4)
    for method in ("euler", "mixing"):
        p, se, n, sd = sabr.mc_price(F0, 100.0, 1.0, 1.0, *G, -1, method=method, **kw)
        assert np.ndim(p) == 0 and np.ndim(se) == 0 and se > 0
        assert type(n) is int and n == 2000 and type(sd) is int and sd == SEED_V + 4
        K = np.array([80.0, 100.0, 120.0])
        v1 = sabr.mc_price(F0, K, 1.0, 1.0, *G, -1, method=method, **kw)
        v2 = sabr.mc_price(F0, K, 1.0, 1.0, *G, -1, method=method, **kw)
        assert np.array_equal(v1[0], v2[0]) and np.array_equal(v1[1], v2[1])
        v3 = sabr.mc_price(F0, K, 1.0, 1.0, *G, -1, method=method, **{**kw, "seed": SEED_V + 5})
        assert np.all(v3[0] != v1[0])
        assert sabr.mc_price(F0, 100.0, 1.0, 1.0, *G, -1, method=method, **kw)[0] == v1[0][1]
        Kb = np.array([[80.0, 100.0, 120.0], [90.0, 100.0, 110.0]])
        vb = sabr.mc_price(F0, Kb, 1.0, 1.0, *G, np.array([-1, -1, 1]), method=method, **kw)
        assert vb[0].shape == vb[1].shape == (2, 3)
        vd = sabr.mc_price(F0, K, 1.0, 0.9, *G, -1, method=method, **kw)
        np.testing.assert_allclose(vd[0], 0.9 * v1[0], rtol=1e-15)
        np.testing.assert_allclose(vd[1], 0.9 * v1[1], rtol=1e-15)
        vs = sabr.mc_price(2.5 * F0, 2.5 * K, 1.0, 1.0, *G, -1, method=method, **kw)
        np.testing.assert_allclose(vs[0], 2.5 * v1[0], rtol=1e-12)


def test_mc_input_validation():
    base = dict(F=F0, K=100.0, T=1.0, DF=1.0, alpha=0.2, rho=-0.5, nu=0.4, cp=-1)
    opts = dict(n_steps=4, n_paths=100, seed=1)
    cases = [
        ({"cp": 0}, {}),
        ({"rho": 1.0}, {}),
        ({"alpha": 0.0}, {}),
        ({"nu": -0.1}, {}),
        ({"T": 0.0}, {}),
        ({"K": -1.0}, {}),
        ({"DF": 0.0}, {}),
        ({"F": np.nan}, {}),
        ({}, {"method": "qe"}),
        ({}, {"n_paths": 1}),
        ({}, {"n_steps": 0}),
        ({}, {"n_steps": 2.7}),
        ({}, {"n_paths": 100.5}),
    ]  # fmt: skip  (F7: no truncation)
    for b_upd, o_upd in cases:
        b, o = {**base, **b_upd}, {**opts, **o_upd}
        with pytest.raises(ValueError):
            sabr.mc_price(b["F"], b["K"], b["T"], b["DF"], b["alpha"], b["rho"], b["nu"], b["cp"],
                          **o)  # fmt: skip
    with pytest.raises(TypeError):  # no implicit global RNG
        sabr.mc_price(F0, 100.0, 1.0, 1.0, *G, -1, n_steps=4, n_paths=100, seed=None)


def test_mc_se_scales_as_inverse_sqrt_n():
    v1 = sabr.mc_price(F0, K_GATE, 0.5, 1.0, *G, -1, n_steps=20, n_paths=20_000, seed=SEED_V + 6)
    v4 = sabr.mc_price(F0, K_GATE, 0.5, 1.0, *G, -1, n_steps=20, n_paths=80_000, seed=SEED_V + 7)
    ratio = v4[1] / v1[1]
    assert np.all(np.abs(ratio - 0.5) < 0.05), ratio


@pytest.mark.parametrize("method", ["euler", "mixing"])
def test_mc_nu_zero_is_black(method):
    """nu = 0: SABR is Black(alpha) in law at any dt; own Black, rho = -0.999, DF = 0.9."""
    K = F0 * np.exp(np.linspace(-0.4, 0.4, 9))
    cp = np.where(K < F0, -1.0, 1.0)
    v, se, *_ = sabr.mc_price(F0, K, 2.0, 0.9, 0.3, -0.999, 0.0, cp, n_steps=3, n_paths=100_000,
                              seed=SEED_V + 8, method=method)  # fmt: skip
    ref = black_vol(F0, K, 2.0, 0.3, cp, 0.9)
    assert np.all(np.abs(v - ref) <= 3 * se), (v - ref) / se


# ============================================================================ 5. MC: discrete law
COARSE = (  # (alpha, rho, nu, T, M): coarse dt on purpose, so scheme != model by many SE
    (0.2, -0.75, 1.0, 1.0, 10),
    (0.2, -0.999, 1.5, 1.0, 8),
    (0.2, 0.999, 1.5, 1.0, 8),
    (0.3, 0.0, 2.0, 0.5, 6),
)
K_COARSE = F0 * np.exp(np.array([-0.4, -0.2, 0.0, 0.2, 0.4]))


@pytest.mark.parametrize("i", range(len(COARSE)))
def test_euler_matches_exact_scheme_expectation(i):
    """Implementer's log-Euler (14) vs SCH (exact expectation of the same discrete scheme, RQMC):
    within 3 SE. Power check: SCH differs from the model (RQMC, M = 512) by > 3 SE somewhere, so a
    code that silently simulated the model (or another scheme) would fail."""
    a, r, nu, T, M = COARSE[i]
    sch, sse = rqmc_price(F0, K_COARSE, T, a, r, nu, -1, M=M, rule="scheme", log2n=13, R=8, seed=11)
    tru, _ = rqmc_price(F0, K_COARSE, T, a, r, nu, -1, M=256, log2n=12, R=8, seed=12)
    v, se, *_ = sabr.mc_price(F0, K_COARSE, T, 1.0, a, r, nu, -1, n_steps=M, n_paths=100_000,
                              seed=SEED_V + 10 + i)  # fmt: skip
    z = (v - sch) / np.hypot(se, sse)
    assert np.all(np.abs(z) <= 3), z
    assert np.max(np.abs(sch - tru) / se) > 3


@pytest.mark.parametrize("i", range(len(COARSE)))
def test_mixing_matches_trapezoid_expectation(i):
    """Implementer's conditional estimator (17) vs my RQMC of the same estimator's expectation
    (trapezoid I at the same dt, exact alpha_T): within 3 SE."""
    a, r, nu, T, M = COARSE[i]
    ref, rse = rqmc_price(F0, K_COARSE, T, a, r, nu, -1, M=M, rule="trap", log2n=13, R=8, seed=13)
    v, se, *_ = sabr.mc_price(F0, K_COARSE, T, 1.0, a, r, nu, -1, n_steps=M, n_paths=100_000,
                              seed=SEED_V + 20 + i, method="mixing")  # fmt: skip
    z = (v - ref) / np.hypot(se, rse)
    assert np.all(np.abs(z) <= 3), z


def test_rho_positive_model_is_strict_local_martingale():
    """[LM07]: beta = 1 SABR is a true martingale iff rho <= 0. Own RQMC of the model call and put
    (conditional representation): C - P - (F - K) = F (E xi - 1) is significantly < 0 at rho = +0.5
    and ~0 at rho = -0.5 (same nu, T). So "calls by parity with F" (sabr.md §3.4, pitfall 8) is the
    model call only for rho <= 0; the log-Euler scheme is an exact martingale for every rho
    (finding F4; the magnitude, -16.7% of F here, is printed by __main__)."""
    out = {}
    for rho in (0.5, -0.5):
        c, cs = rqmc_price(1.0, 1.0, 5.0, 0.3, rho, 1.0, 1, M=256, log2n=12, R=8, seed=SEED_V + 9)
        p, ps = rqmc_price(1.0, 1.0, 5.0, 0.3, rho, 1.0, -1, M=256, log2n=12, R=8, seed=SEED_V + 9)
        out[rho] = ((c - p)[0], np.hypot(cs, ps)[0])
    assert out[0.5][0] < -10 * out[0.5][1], out
    assert abs(out[-0.5][0]) < 3 * out[-0.5][1] + 1e-4, out


# ============================================================================ 6. gate 2 (slow)
@pytest.mark.slow
def test_gate_literal_with_own_mc():
    """ROADMAP gate with an independent MC (OMC: exact conditional sampling, Simpson, dt = 1/500,
    N = 2e5, own layout and seeds): |V_OMC - Black(sigma_H)| <= 3 SE on set G, 21 comparisons."""
    a, r, nu = G
    for i, T in enumerate(T_GATE):
        M = 2 * int(np.ceil(T * 250 - 1e-9))  # even, dt <= 1/500
        v, se = own_mc_put(F0, K_GATE, T, a, r, nu, M=M, N=N_GATE, seed=SEED_V + 30 + i)
        z = (v - hagan_put(K_GATE, T, a, r, nu)) / se
        assert np.all(np.abs(z) <= 3), (T, z)


@pytest.mark.slow
def test_impl_gate_estimator_vs_exact_scheme_fresh_seeds(gate_refs):
    """Implementer's gate estimator (scheme 14, N = 2e5, dt = 1/250) on fresh validator seeds vs
    SCH (its exact expectation): within 3 SE. Tests the code, free of Hagan and dt error."""
    for i, T in enumerate(T_GATE):
        v, se, *_ = _gate_se(T, SEED_V + 40 + i)
        ref = gate_refs[T]
        z = (v - ref["sch"]) / np.hypot(se, ref["sse"])
        assert np.all(np.abs(z) <= 3), (T, z)


@pytest.mark.slow
def test_gate_expected_offset_within_3se(gate_refs):
    """Seed-free version of the gate: the deterministic offset E[V_MC] - Black(sigma_H) = SCH -
    Hagan must be inside 3 SE_gate for all 21 comparisons (else the gate passes by luck)."""
    for i, T in enumerate(T_GATE):
        _, se, *_ = _gate_se(T, SEED_V + 45 + i)
        ref = gate_refs[T]
        off = (ref["sch"] - ref["hag"]) / se
        assert np.all(np.abs(off) <= 3), (T, off)


@pytest.mark.slow
def test_scheme14_bias_meets_d023_criterion_at_every_gate_maturity():
    """The implementer's replacement for note oracle (e) (OPEN_ISSUES, tests/test_sabr.py::
    test_halving_dt): |V_dt - V_dt/2| <= 0.1 SE_gate, i.e. |bias| <~ 0.2 SE. Here the *expectations*
    V_dt, V_dt/2 are computed exactly (SCH, RQMC) at every gate maturity with the gate's own step
    count n_steps(T) (mirror of the code). Pre-fix (dt = 1/250, M = 63, 125, 250) this failed at
    T = 0.25 (0.321) and 0.5 (0.160): finding F1. Post-fix (M = 250 at every T) the criterion (0.1)
    and the reference-precision guard (0.02 SE_gate) are unchanged; the difference is now computed
    on common RQMC paths (rqmc_halving), because two separate 250/500-dim RQMC runs no longer meet
    the guard (error 0.028 SE_gate at T = 0.25, k = -0.2)."""
    a, r, nu = G
    bad = {}
    for i, T in enumerate(T_GATE):
        d, dse = rqmc_halving(F0, K_GATE, T, a, r, nu, M=n_steps(T), seed=7)
        _, se_gate, *_ = _gate_se(T, SEED_V + 50 + i)
        ratio = np.abs(d) / se_gate
        assert np.all(dse < 0.02 * se_gate)  # reference resolves the criterion
        if np.any(ratio > 0.1):
            bad[T] = np.round(ratio, 3)
    assert not bad, f"|V_dt - V_dt/2| / SE_gate > 0.1 at {bad}"


@pytest.mark.slow
def test_multi_seed_z_distribution_vs_exact_scheme(gate_refs):
    """10 fresh seeds x 21 comparisons of the implementer's gate estimator vs SCH: pooled z has
    mean ~ 0 and sd ~ 1 (effective sample ~ 45 because strikes share paths: bounds 3 sigma)."""
    zs = []
    for j in range(10):
        for i, T in enumerate(T_GATE):
            v, se, *_ = _gate_se(T, SEED_V + 1000 + 10 * j + i)
            ref = gate_refs[T]
            zs.append((v - ref["sch"]) / np.hypot(se, ref["sse"]))
    z = np.concatenate(zs)
    assert abs(z.mean()) < 0.45 and 0.7 < z.std(ddof=1) < 1.3, (z.mean(), z.std(ddof=1))


# ============================================================================ 7. golden + published
def _golden():
    with open(GOLDEN) as f:
        return json.load(f)


@pytest.mark.slow
def test_golden_file_two_methods_agree():
    """Every golden case: RQMC and PDE (both stored) agree within the stated tol; a fresh RQMC run
    (new scrambles) reproduces the stored values within tol + 3 se. R = 16 scrambles x 2^14: with
    R = 8 x 2^13 the scramble-to-scramble spread of the deep-OTM put (k = -0.2, T = 0.25) is skewed
    and the 7-dof se was seen to understate the error ~4x (docs/validation/phase3.md, F8)."""
    g = _golden()
    for c in g["cases"]:
        assert abs(c["rqmc"] - c["pde"]) <= c["tol"], c
    cs = g["cases"][:7]  # set G, T = 0.25 (cheapest), fresh scrambles
    K = np.array([c["K"] for c in cs])
    v, se = rqmc_price(F0, K, 0.25, *G, -1, M=512, log2n=14, R=16, seed=SEED_V + 61)
    ref = np.array([c["price"] for c in cs])
    tol = np.array([c["tol"] for c in cs])
    assert all(c["set"] == "G" and c["T"] == 0.25 for c in cs)
    assert np.all(np.abs(v - ref) <= tol + 3 * se), (v - ref, se)


@pytest.mark.slow
def test_impl_mixing_vs_golden():
    """Implementer's conditional MC (17) (bias O(dt^2)) vs golden prices, set G: 3 SE + tol."""
    g = _golden()
    for i, T in enumerate(T_GATE):
        cs = [c for c in g["cases"] if c["set"] == "G" and c["T"] == T]
        K = np.array([c["K"] for c in cs])
        ref = np.array([c["price"] for c in cs])
        tol = np.array([c["tol"] for c in cs])
        v, se, *_ = sabr.mc_price(F0, K, T, 1.0, *G, -1, n_steps=n_steps(T), n_paths=N_GATE,
                                  seed=SEED_V + 70 + i, method="mixing")  # fmt: skip
        assert np.all(np.abs(v - ref) <= 3 * se + tol), (T, (v - ref) / se)


@pytest.mark.slow
def test_chk_published_fdm_values():
    """[CHK] Table 1 (FDM, 5 decimals): own RQMC within half a unit (5e-6) + 3 SE for rows 1-4.
    Row 5: nu = 0.6 (table) fits far better than nu = 0.8 (text), but misses 0.07811 by ~1.0e-5
    (outside the rounding; finding, not asserted as a pass)."""
    for r, nu, ref in CHK:
        v, se = rqmc_price(1.0, 1.0, 1.0, 0.2, r, nu, -1, M=512, log2n=14, R=8, seed=SEED_V + 80)
        assert abs(v[0] - ref) <= 5e-6 + 3 * se[0], (r, nu, v[0] - ref)
    r, ref = CHK_ROW5
    d = {}
    for nu in (0.6, 0.8):
        v, _ = rqmc_price(1.0, 1.0, 1.0, 0.2, r, nu, -1, M=512, log2n=14, R=8, seed=SEED_V + 81)
        d[nu] = abs(v[0] - ref)
    assert d[0.6] < 2e-5 < 1e-4 < d[0.8], d


@pytest.mark.slow
def test_figure_csv_spot_checks():
    """report/figures/sabr_hagan_vs_mc.csv (mixing rows): at the per-panel max-error strike for
    T >= 1 plus the ATM point, err_volpts agrees with my 100 (sigma_H - sigma_true) (RQMC, OTM
    option, own brentq IV) within 3 SE_sigma(csv) + 3 SE(mine)."""
    import csv

    with open(FIG_CSV) as f:
        rows = [r for r in csv.DictReader(f) if r["method"] == "mixing"]
    a, r = 0.2, -0.75
    for nu in (0.4, 1.0):
        for T in (1.0, 5.0, 10.0):
            panel = [q for q in rows if float(q["nu"]) == nu and float(q["T"]) == T]
            errs = np.array([abs(float(q["err_volpts"])) for q in panel])
            pick = [panel[int(np.argmax(errs))], min(panel, key=lambda q: abs(float(q["k"])))]
            for q in pick:
                K, cp = float(q["K"]), int(q["cp"])
                M = 512 if T <= 5 else 1024
                p, se = rqmc_price(F0, K, T, a, r, nu, cp, M=M, log2n=13, R=8, seed=SEED_V + 90)
                sig_t = iv_brent(p, F0, K, T, cp)[0]
                se_mine = se[0] / black_vega(F0, K, T, sig_t)
                err_mine = 100 * (hagan_decimal(F0, K, T, a, r, nu) - sig_t)
                tol = 300 * (float(q["se_sigma"]) + se_mine)
                assert abs(err_mine - float(q["err_volpts"])) <= tol, (nu, T, q["k"], err_mine)


# ============================================================================ __main__: numbers
def _p(*a):
    print(*a, flush=True)


def _fmt(v, f="+.3f"):
    return "[" + ", ".join(format(x, f) for x in np.atleast_1d(v)) + "]"


def report_formula():
    _p("== Formula level")
    zc, worst_s, worst_d = sabr.Z_C, 0.0, 0.0
    zs = np.concatenate([np.geomspace(1e-12, 1e4, 300), zc * np.array([1 - 2**-52, 1, 1 + 2**-52])])
    zs = np.concatenate([zs, -zs])
    for rho in RHOS:
        e = np.abs(sabr.z_over_x(zs, rho) / np.array([zx_decimal(v, rho) for v in zs]) - 1)
        worst_s = max(worst_s, e[np.abs(zs) < zc].max())
        worst_d = max(worst_d, e[np.abs(zs) >= zc].max())
    _p(f"  z/x vs decimal: series branch {worst_s:.1e}, log1p branch {worst_d:.1e}")
    k = np.linspace(-1, 1, 201)
    e12 = max(
        np.abs(sabr.implied_vol(1.0, np.exp(k), T, a, r, 1e-12) - a).max()
        for r in (-0.999, 0, 0.999)
        for a in (0.05, 0.2, 1.0)
        for T in (0.01, 1, 10)
    )
    _p(f"  gate 1: nu = 0 exact (bitwise alpha); nu = 1e-12: max |sigma - alpha| = {e12:.1e}")
    rng = np.random.default_rng(
        SEED_V + 1
    )  # same draws as test_implied_vol_vs_decimal_random_domain
    n = 1500
    k = np.concatenate([rng.uniform(-3, 3, n // 2), rng.normal(0, 1e-3, n - n // 2)])
    T = np.exp(rng.uniform(np.log(1e-3), np.log(30), n))
    a = np.exp(rng.uniform(np.log(1e-3), np.log(5), n))
    r = rng.uniform(-0.999, 0.999, n)
    nu = np.exp(rng.uniform(np.log(1e-4), np.log(5), n))
    F = np.exp(rng.uniform(np.log(1e-2), np.log(1e4), n))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = sabr.implied_vol(F, F * np.exp(k), T, a, r, nu)
    ok = bracket(T, a, r, nu) > 0
    ref = np.array([hagan_decimal(*v) for v in zip(F, F * np.exp(k), T, a, r, nu, strict=True)])
    err = np.abs(got[ok] / ref[ok] - 1).max()
    bad = np.sum(np.isnan(got) != ~ok)
    _p(f"  sigma_H vs decimal [H02], 1500 random points: max rel {err:.1e};"
       f" NaN where bracket <= 0: {(~ok).sum()} pts, mismatches {bad}")  # fmt: skip
    rng = np.random.default_rng(3)
    n = 20000
    s = np.exp(rng.uniform(np.log(0.03), np.log(1.5), n))
    T = np.exp(rng.uniform(np.log(1 / 365), np.log(30.0), n))  # to 30y: reaches b < 0 < a
    r = rng.uniform(-0.999, 0.999, n)
    nu = np.exp(rng.uniform(np.log(1e-4), np.log(5.0), n))
    a, b = _atm_coeffs(T, r, nu)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        al = sabr.alpha_from_atm(s, T, r, nu)
    ok = np.isfinite(al)
    rt = np.abs(sabr.implied_vol(1.0, 1.0, T[ok], al[ok], r[ok], nu[ok]) / s[ok] - 1)
    neg = (b[ok] < 0) & (a[ok] > 0)
    _p(f"  alpha_from_atm (T <= 30, Phase-5 bounds): {(~ok).sum()} NaN of {n}; round trip b > 0: "
       f"{rt[~neg].max():.1e}; rho <= 0: {rt[r[ok] <= 0].max():.1e}; b < 0 < a ({neg.sum()} pts): "
       f"{rt[neg].max():.1e}")  # fmt: skip


def gate_references(log2n=16, R=16):
    a, r, nu = G
    out = {}
    for T in T_GATE:
        M = n_steps(T)
        tru, tse = rqmc_price(F0, K_GATE, T, a, r, nu, -1, M=512, log2n=log2n, R=R, seed=101)
        pde, perr = pde_put_rich(F0, K_GATE, T, a, r, nu)
        sch, sse = rqmc_price(F0, K_GATE, T, a, r, nu, -1, M=M, rule="scheme", log2n=log2n, R=R,
                              seed=303)  # fmt: skip
        sch2, sse2 = rqmc_price(F0, K_GATE, T, a, r, nu, -1, M=2 * M, rule="scheme", log2n=log2n,
                                R=R, seed=303)  # fmt: skip
        trap, trse = rqmc_price(F0, K_GATE, T, a, r, nu, -1, M=M, rule="trap", log2n=log2n, R=R,
                                seed=404)  # fmt: skip
        out[T] = dict(tru=tru, tse=tse, pde=pde, perr=perr, sch=sch, sse=sse, sch2=sch2,
                      sse2=sse2, trap=trap, trse=trse,
                      hag=hagan_put(K_GATE, T, a, r, nu))  # fmt: skip
    return out


def report_gate(refs):
    a, r, nu = G
    _p("== Gate 2, set G: references (F = 100)")
    for i, T in enumerate(T_GATE):
        q = refs[T]
        v, se, *_ = _gate_se(T, 20261107 + i)  # the implementer's declared seed
        sig_t = iv_brent(q["tru"], F0, K_GATE, T, -1)
        vega = black_vega(F0, K_GATE, T, sig_t)
        _p(f" T={T}: truth {_fmt(q['tru'], '.5f')} (RQMC se <= {q['tse'].max():.1e});"
           f" PDE - RQMC {_fmt(q['pde'] - q['tru'], '+.1e')}")  # fmt: skip
        _p(f"   Hagan - truth (vol pt)          {_fmt(100 * (q['hag'] - q['tru']) / vega, '+.4f')}")
        _p(f"   Hagan - truth (SE_gate)         {_fmt((q['hag'] - q['tru']) / se, '+.3f')}")
        _p(f"   scheme(14) bias dt (vol pt)     {_fmt(100 * (q['sch'] - q['tru']) / vega, '+.4f')}")
        _p(f"   scheme(14) bias dt (SE_gate)    {_fmt((q['sch'] - q['tru']) / se, '+.3f')}")
        _p(f"   |V_dt - V_dt/2| / SE_gate       {_fmt(np.abs(q['sch'] - q['sch2']) / se, '.3f')}")
        _p(f"   mixing(17) bias dt (SE_gate)    {_fmt((q['trap'] - q['tru']) / se, '+.3f')}")
        _p(f"   offset (SCH - Hagan)/SE_gate    {_fmt((q['sch'] - q['hag']) / se, '+.3f')}")
        rich = 2 * (q["sch"] - q["sch2"])
        _p(f"   Richardson 2(V_dt - V_dt/2) vol pt {_fmt(100 * rich / vega, '+.4f')}")
        if n_steps_old(T) != n_steps(T):  # history (finding F1): pre-fix dt = 1/250
            Mo = n_steps_old(T)
            o1, _ = rqmc_price(F0, K_GATE, T, a, r, nu, -1, M=Mo, rule="scheme", log2n=14, R=8,
                               seed=303)  # fmt: skip
            o2, _ = rqmc_price(F0, K_GATE, T, a, r, nu, -1, M=2 * Mo, rule="scheme", log2n=14, R=8,
                               seed=303)  # fmt: skip
            _p(f"   pre-fix M={Mo}: bias (SE_gate)   {_fmt((o1 - q['tru']) / se, '+.3f')}")
            _p(f"   pre-fix M={Mo}: halving/SE_gate  {_fmt(np.abs(o1 - o2) / se, '.3f')}")
        _p(f"   impl. seed {20261107 + i}: z vs Hagan {_fmt((v - q['hag']) / se, '+.2f')}; "
           f"z vs SCH {_fmt((v - q['sch']) / np.hypot(se, q['sse']), '+.2f')}")  # fmt: skip
    for i, T in enumerate(T_GATE):
        M = 2 * int(np.ceil(T * 250 - 1e-9))
        v, se = own_mc_put(F0, K_GATE, T, a, r, nu, M=M, N=N_GATE, seed=SEED_V + 30 + i)
        _p(f"   own MC (OMC) T={T}: z vs Hagan {_fmt((v - refs[T]['hag']) / se, '+.2f')}; "
           f"z vs truth {_fmt((v - refs[T]['tru']) / se, '+.2f')}")  # fmt: skip


def study_seeds(refs, S=100):
    """Fresh-seed robustness of the 21-comparison gate (implementer's estimator and settings)."""
    from scipy.stats import norm

    _p(f"== Multi-seed study: {S} fresh seeds x 21 comparisons (euler gate, and mixing (17))")
    zh, zs, zmh, zmt = [], [], [], []
    for j in range(S):
        a1, a2, a3, a4 = [], [], [], []
        for i, T in enumerate(T_GATE):
            q = refs[T]
            v, se, *_ = _gate_se(T, SEED_V + 5000 + 10 * j + i)
            a1.append((v - q["hag"]) / se)
            a2.append((v - q["sch"]) / np.hypot(se, q["sse"]))
            vm, sm, *_ = sabr.mc_price(F0, K_GATE, T, 1.0, *G, -1, n_steps=n_steps(T),
                                       n_paths=N_GATE, seed=SEED_V + 9000 + 10 * j + i,
                                       method="mixing")  # fmt: skip
            a3.append((vm - q["hag"]) / sm)
            a4.append((vm - q["tru"]) / np.hypot(sm, q["tse"]))
        zh.append(np.concatenate(a1))
        zs.append(np.concatenate(a2))
        zmh.append(np.concatenate(a3))
        zmt.append(np.concatenate(a4))
    for name, z in (("euler vs Hagan (the gate)", zh), ("euler vs SCH (code only)", zs),
                    ("mixing vs Hagan", zmh), ("mixing vs truth", zmt)):  # fmt: skip
        z = np.array(z)
        fail = np.mean(np.any(np.abs(z) > 3, axis=1))
        _p(
            f"  {name:28s}: mean {z.mean():+.3f} sd {z.std(ddof=1):.3f} "
            f"max|z| {np.abs(z).max():.2f}"
            f"  P(|z|>3) per comparison {np.mean(np.abs(z) > 3):.4f}  gate fail rate {fail:.3f}"
        )
        _p("    per-comparison mean z (T rows): " + "; ".join(
            _fmt(z.mean(0)[7 * i:7 * i + 7], '+.2f') for i in range(3)))  # fmt: skip
    # predicted per-comparison failure rate from offsets alone (unit-variance normal)
    off = np.array(zh).mean(0)
    pf = norm.cdf(-3 - off) + norm.sf(3 - off)
    _p(f"  predicted from mean offsets: sum P(|z|>3) = {pf.sum():.3f} (vs 21 x 0.0027 = 0.057)")


def study_scheme_unbiased():
    """Implementer's log-Euler vs its exact expectation (SCH), many seeds: mean z ~ 0, sd ~ 1."""
    _p("== Scheme-(14) code vs SCH over many seeds")
    a, r, nu, T, M = COARSE[0]
    sch, sse = rqmc_price(
        F0, K_COARSE, T, a, r, nu, -1, M=M, rule="scheme", log2n=16, R=16, seed=99
    )
    z = []
    for s in range(300):
        v, se, *_ = sabr.mc_price(F0, K_COARSE, T, 1.0, a, r, nu, -1, n_steps=M, n_paths=100_000,
                                  seed=4_000_000 + s)  # fmt: skip
        z.append((v - sch) / np.hypot(se, sse))
    z = np.array(z)
    _p(f"  coarse set {COARSE[0]}, 300 seeds x N = 1e5: mean z {_fmt(z.mean(0))}, "
       f"sd {_fmt(z.std(0, ddof=1))} (se of mean {1 / np.sqrt(300):.3f})")  # fmt: skip
    sch, sse = rqmc_price(F0, K_GATE, 1.0, *G, -1, M=250, rule="scheme", log2n=15, R=16, seed=808)
    z = []
    for s in range(150):
        v, se, *_ = _gate_se(1.0, 6_000_000 + s)
        z.append((v - sch) / np.hypot(se, sse))
    z = np.array(z)
    _p(f"  gate settings T = 1, 150 seeds: mean z {_fmt(z.mean(0))}, sd {_fmt(z.std(0, ddof=1))} "
       f"(se of mean {1 / np.sqrt(150):.3f})")  # fmt: skip


def report_other_sets():
    _p("== Gate box on other (SPX-plausible) parameter sets: Hagan - truth, N = 2e5 euler z")
    kk = np.array([-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2])
    K = F0 * np.exp(kk)
    cp = np.where(kk > 0, 1.0, -1.0)
    sets = (("S1", 0.15, -0.7, 1.5, 0.1), ("S4 (fig.)", 0.2, -0.75, 1.0, 0.25),
            ("S5", 0.2, -0.9, 0.6, 1.0), ("S2", 0.18, -0.7, 1.0, 0.5),
            ("S3 (fig.)", 0.2, -0.75, 1.0, 1.0))  # fmt: skip
    for name, a, r, nu, T in sets:
        tru, _ = rqmc_price(F0, K, T, a, r, nu, cp, M=512, log2n=14, R=8, seed=5)
        sig_t = iv_brent(tru, F0, K, T, cp)
        sig_h = np.array([hagan_decimal(F0, k_, T, a, r, nu) for k_ in K])
        put_t = tru - np.where(cp > 0, F0 - K, 0.0)  # rho <= 0: martingale, parity exact
        v, se, *_ = sabr.mc_price(F0, K, T, 1.0, a, r, nu, -1, n_steps=n_steps(T),
                                  n_paths=N_GATE, seed=SEED_V + 300)  # fmt: skip
        hp = black_vol(F0, K, T, sig_h, -1)
        _p(f"  {name} alpha={a} rho={r} nu={nu} T={T} (nu^2T={nu * nu * T:.3f}):")
        _p(f"    Hagan - truth (vol pt) {_fmt(100 * (sig_h - sig_t), '+.3f')}")
        _p(f"    (Hagan - truth)/SE     {_fmt((hp - put_t) / se, '+.2f')}")
        _p(f"    euler z vs Hagan       {_fmt((v - hp) / se, '+.2f')}")


def report_chk():
    _p("== [CHK] Table 1 (FDM) vs own RQMC and PDE, F = K = 1, T = 1, alpha = 0.2")
    for r, nu, ref in CHK + ((-0.75, 0.6, 0.07811), (-0.75, 0.8, 0.07811)):
        v, se = rqmc_price(1.0, 1.0, 1.0, 0.2, r, nu, -1, M=512, log2n=16, R=16, seed=9)
        pd, _ = pde_put_rich(1.0, 1.0, 1.0, 0.2, r, nu)
        h = hagan_put(1.0, 1.0, 0.2, r, nu, F=1.0)[0]
        _p(f"  rho={r:+.2f} nu={nu}: RQMC {v[0]:.7f} +- {se[0]:.1e}  PDE {pd[0]:.7f}  FDM {ref}"
           f"  RQMC-FDM {v[0] - ref:+.1e}  Hagan-RQMC {h - v[0]:+.1e}")  # fmt: skip


def report_figure():
    import csv

    _p("== Figure CSV spot checks (mixing rows) vs own truth")
    with open(FIG_CSV) as f:
        rows = [r for r in csv.DictReader(f) if r["method"] == "mixing"]
    for nu in (0.4, 1.0):
        for T in (0.25, 1.0, 5.0, 10.0):
            panel = [q for q in rows if float(q["nu"]) == nu and float(q["T"]) == T]
            errs = np.array([abs(float(q["err_volpts"])) for q in panel])
            for q in (panel[int(np.argmax(errs))], min(panel, key=lambda q: abs(float(q["k"])))):
                K, cp = float(q["K"]), int(q["cp"])
                M = 512 if T <= 5 else 1024
                p, se = rqmc_price(F0, K, T, 0.2, -0.75, nu, cp, M=M, log2n=14, R=8, seed=77)
                sig_t = iv_brent(p, F0, K, T, cp)[0]
                se_m = 100 * se[0] / black_vega(F0, K, T, sig_t)
                e_m = 100 * (hagan_decimal(F0, K, T, 0.2, -0.75, nu) - sig_t)
                _p(f"  nu={nu} T={T:<5} k={float(q['k']):+.3f}: csv {float(q['err_volpts']):+.3f}"
                   f" +- {100 * float(q['se_sigma']):.3f}  mine {e_m:+.3f}"
                   f" +- {se_m:.3f}")  # fmt: skip


def report_martingale():
    _p("== rho > 0: loss of martingality, E F_T / F - 1 (alpha = 0.3, nu = 1, T = 5)")
    rng = np.random.default_rng(SEED_V + 500)
    for rho in (0.5, -0.5):
        c, cs = rqmc_price(1.0, 1.0, 5.0, 0.3, rho, 1.0, 1, M=512, log2n=13, R=8, seed=3)
        p, ps = rqmc_price(1.0, 1.0, 5.0, 0.3, rho, 1.0, -1, M=512, log2n=13, R=8, seed=3)
        # second method: P(explosion by T) of alpha under the share measure,
        # d ln a = (rho nu a - nu^2/2) dt + nu dZ (Lions-Musiela), Euler, absorbed at a = 1e4
        N, M = 50_000, 5000
        la, alive = np.full(N, np.log(0.3)), np.ones(N, bool)
        for _ in range(M):
            la = (
                la
                + (rho * np.exp(la) - 0.5) * (5.0 / M)
                + np.sqrt(5.0 / M) * rng.standard_normal(N)
            )
            alive &= la < np.log(1e4)
            la = np.minimum(la, np.log(1e4))
        pe = 1 - alive.mean()
        _p(f"  rho={rho:+.1f}: RQMC C-P-(F-K) = {(c - p)[0]:+.4f} +- {np.hypot(cs, ps)[0]:.4f};"
           f" -P(explosion) = {-pe:+.4f} +- {np.sqrt(pe * (1 - pe) / N):.4f}")  # fmt: skip


def make_golden(path: Path):
    if path.exists():
        _p(f"golden {path} exists: not overwritten")
        return
    cases = []

    def add(setname, F, K, T, a, r, nu, published=None, pde_n=(401, 161, 200)):
        K = np.atleast_1d(K)
        v, se = rqmc_price(F, K, T, a, r, nu, -1, M=512, log2n=16, R=16, seed=2026)
        pd, perr = pde_put_rich(F, K, T, a, r, nu, n=pde_n)
        for j in range(K.size):
            raw = max(3 * se[j] + abs(v[j] - pd[j]), 1e-7 * F)
            e10 = 10.0 ** (np.floor(np.log10(raw)) - 1)
            tol = float(f"{np.ceil(raw / e10) * e10:.2g}")  # rounded *up* to 2 significant digits
            c = dict(set=setname, F=F, K=float(K[j]), k=float(np.log(K[j] / F)), T=T, DF=1.0,
                     cp=-1, alpha=a, rho=r, nu=nu, price=float(v[j]), tol=tol,
                     rqmc=float(v[j]), rqmc_se=float(se[j]), pde=float(pd[j]),
                     pde_richardson_step=float(perr[j]))  # fmt: skip
            if published is not None:
                c["published_fdm_call_eq_put"] = published
            cases.append(c)
            _p(f"  {setname} T={T} K={K[j]:.4f}: {v[j]:.7f} tol {tol:.1e}")

    for T in T_GATE:
        add("G", F0, K_GATE, T, *G)
    add("NU1", F0, F0 * np.exp(np.array([-0.2, 0.0, 0.2])), 1.0, 0.2, -0.75, 1.0)
    add("SPX_SHORT", F0, F0 * np.exp(np.array([-0.2, -0.1, 0.0, 0.1, 0.2])), 0.1, 0.15, -0.7, 1.5)
    for r, nu, ref in CHK + ((-0.75, 0.6, 0.07811),):
        add("CHK", 1.0, 1.0, 1.0, 0.2, r, nu, published=ref)
    doc = {
        "description": "SABR beta = 1 European put prices (DF = 1). F = forward; puts; ATM call = "
        "put for the CHK cases. Phase 3 validation (validator agent), 2026-11-07.",
        "model": "dF = alpha_t F dW, d alpha = nu alpha dZ, d<W,Z> = rho dt (T-forward measure)",
        "source": "Agreement of two independent methods, neither using qpricer code: "
        "(A) randomised "
        "QMC (16 scrambled Sobol sequences x 2^16 points, Brownian-bridge alpha path, M = 512, "
        "Simpson integrated variance) of the exact conditional (mixing) representation E[Black(F "
        "xi, K, (1-rho^2) I)], xi = exp(rho (alpha_T - alpha)/nu - rho^2 I/2) (Romano-Touzi 1997); "
        "(B) 2-D PDE in (ln F, ln alpha), Modified Craig-Sneyd ADI (theta = 1/3), cell-averaged "
        "payoff, Richardson over (401x161x200, 801x321x400) grids. price = (A); tol = 3 se(A) + "
        "|A - B| (min 1e-7 F). CHK cases also carry the published FDM value of Choi-Hu-Kwok, "
        "arXiv:2408.01898v2 Table 1 (5 decimals); the last CHK row is printed with nu = 0.6 in "
        "the table (0.8 in the text): nu = 0.6 is used, and its published value misses (A) and "
        "(B) by ~1e-5.",
        "generator": "tests/validation/test_phase3_independent.py --golden-out PATH",
        "cases": cases,
    }
    with open(path, "x") as f:
        json.dump(doc, f, indent=1)
    _p(f"golden written: {path} ({len(cases)} cases)")


if __name__ == "__main__":
    t0 = time.time()
    if "--golden-out" in sys.argv:
        make_golden(Path(sys.argv[sys.argv.index("--golden-out") + 1]))
        sys.exit(0)
    warnings.simplefilter("ignore", RuntimeWarning)
    report_formula()
    refs = gate_references()
    report_gate(refs)
    report_other_sets()
    report_chk()
    report_figure()
    report_martingale()
    if "--seeds" in sys.argv:
        study_seeds(refs)
        study_scheme_unbiased()
    _p(f"done in {time.time() - t0:.0f}s")
