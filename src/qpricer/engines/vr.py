"""Phase 6 vanilla MC estimators: plain, antithetic, control variate, randomised Sobol' QMC.

Terminal sampling: BSM exact one step (asian.md eq. 3 with n = 1); Heston via the mc.py QE-M step
(heston-mc.md eqs. 7-12) driven by caller-supplied (2, N) normal blocks, one per step, in the
heston_steps order [Z_v; Z_perp]. Estimators: heston-mc.md eq. (1) (plain), the pair-mean form
below (antithetic), asian.md eqs. (14)-(15) (regression CV, exotic_mc.control_variate) and the
mean over independent Owen scramblings (RQMC). Every pricer returns (price, stderr, n_paths, seed).
References: Glasserman (2003) Monte Carlo Methods in Fin. Eng. §4.1-4.2, §5.4; Owen (1997) SIAM
J. Numer. Anal. 34(5) (scrambled nets); L'Ecuyer & Lemieux (2002) (RQMC error estimation).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtri
from scipy.stats import qmc

from qpricer.engines import mc
from qpricer.engines.exotic_mc import control_variate
from qpricer.models import bsm
from qpricer.models.heston import HestonParams
from qpricer.products import vanilla

METHODS = ("plain", "antithetic", "cv", "sobol")
SOBOL_BITS = 30  # scipy default scrambling depth


def antithetic_estimate(g: ArrayLike, DF: float = 1.0) -> tuple[NDArray, NDArray]:
    """Antithetic estimator over the last axis: first half of the N paths = Z, second half = -Z.

    The N/2 pair means h_j = (g_j + g_j') / 2 are iid, so V = DF mean h and SE = DF s_h / sqrt(N/2)
    (ddof = 1). The SE is never computed from the N correlated payoffs (Glasserman §4.2).
    """
    g = np.asarray(g, dtype=np.float64)
    n = g.shape[-1]
    if n % 2 or n < 4:
        raise ValueError("antithetic needs an even number of paths >= 4 (N/2 pairs)")
    return mc.mc_estimate(0.5 * (g[..., : n // 2] + g[..., n // 2 :]), DF)


def sobol_normals(dim: int, n: int, rng: np.random.Generator) -> NDArray:
    """(dim, n) N(0, 1) draws z = Phi^{-1}(u) from one Owen-scrambled Sobol' net (scipy.stats.qmc).

    scipy scrambles B = 30 digits, so u = k 2^-B lies on a grid and u = 0 occurs (prob 2^-B per
    coordinate). The scramble is completed to full resolution by a uniform jitter inside the
    point's own 2^-B cell, u <- u + U 2^-B (D-042): each point stays in its elementary interval
    (net property kept) and is exactly U(0, 1). n must be a power of 2. Raises if a coordinate
    is still 0 or 1 (z infinite).
    """
    m = int(np.log2(n)) if n >= 1 else -1
    if m < 0 or 2**m != n:
        raise ValueError("Sobol' sample size must be a power of 2")
    u = qmc.Sobol(int(dim), scramble=True, bits=SOBOL_BITS, rng=rng).random_base2(m)
    u = u + rng.random(u.shape) * 2.0**-SOBOL_BITS
    if np.any(u <= 0.0) or np.any(u >= 1.0):
        raise FloatingPointError("scrambled Sobol' point on the boundary of [0, 1)")
    return ndtri(u.T)


def heston_terminal(
    p: HestonParams, times: ArrayLike, blocks: Iterable[NDArray], scheme: str = "qe-m"
) -> NDArray:
    """x_T = ln(F_T/F) from given normal blocks, one (2, N) block [Z_v; Z_perp] per step of `times`.

    Steps by mc._qe_step (heston-mc.md eqs. 7-12; scheme "qe" or "qe-m"); bit-identical to
    mc.heston_steps when the blocks are rng.standard_normal((2, N)) draws in step order.
    """
    if scheme not in ("qe", "qe-m"):
        raise ValueError("scheme must be 'qe' or 'qe-m' (Euler: use mc.heston_steps)")
    t = mc._check_grid(times)
    x = v = None
    for dt, z in zip(np.diff(t, prepend=0.0), blocks, strict=True):
        if x is None:
            x, v = np.zeros(z.shape[1]), np.full(z.shape[1], p.v0)
        x, v = mc._qe_step(x, v, dt, p, z[0], z[1], scheme == "qe-m")
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(v))):
            raise FloatingPointError("non-finite Heston state")
    return x


def _terminal(model, T: float, n_steps: int, n: int, method: str, rng: np.random.Generator):
    """x_T on n paths; antithetic: the second half is driven by the negated normals."""
    heston = isinstance(model, HestonParams)
    rows = 2 if heston else 1  # per step: [Z_v; Z_perp] (Heston) or [Z] (BSM, one step)
    if method == "sobol":
        Z = sobol_normals(rows * n_steps, n, rng)  # dims (2k, 2k + 1) drive step k
        blocks = (Z[rows * k : rows * (k + 1)] for k in range(n_steps))
    elif method == "antithetic":
        draw = (rng.standard_normal((rows, n // 2)) for _ in range(n_steps))
        blocks = (np.concatenate([z, -z], axis=1) for z in draw)
    else:
        blocks = (rng.standard_normal((rows, n)) for _ in range(n_steps))
    if heston:
        return heston_terminal(model, mc.uniform_grid(T, n_steps), blocks, "qe-m")
    sigma = float(model)
    if not (np.isfinite(sigma) and sigma >= 0):
        raise ValueError("sigma must be finite and >= 0")
    return -0.5 * sigma**2 * T + sigma * np.sqrt(T) * next(blocks)[0]  # asian.md eq. (3)


def vanilla_mc(
    F: float,
    K: ArrayLike,
    T: float,
    DF: float,
    model,
    cp: ArrayLike,
    *,
    method: str,
    n_paths: int,
    seed: int,
    n_steps: int = 1,
    n_scramble: int = 16,
) -> tuple[NDArray, NDArray, int, int]:
    """European price by MC with variance reduction: (price, stderr, n_paths, seed).

    model = sigma (BSM, exact one step, n_steps must be 1) or HestonParams (QE-M on a uniform
    grid of n_steps). All strikes share one path set. n_paths is the total number of terminal
    samples (honest work count) for every method:
    plain      eq. (1), heston-mc.md (Heston: same stream as mc.european_price).
    antithetic N/2 pairs (Z, -Z); SE from the pair means (`antithetic_estimate`).
    cv         asian.md eqs. (14)-(15), OLS c_hat on the same paths, control X = F_T with
               E X = F exactly: BSM exact sampling, and QE-M is a discrete martingale,
               E[e^{x_{n+1} - x_n} | v_n] = 1 (heston-mc.md eq. 12), so no grid bias (D-041).
    sobol      R = n_scramble independent Owen scramblings of n_paths / R points (a power of 2),
               child seeds SeedSequence(seed).spawn(R); price = mean of the R estimates and
               SE = their std (ddof = 1) / sqrt(R) (randomised QMC).
    rng = np.random.default_rng(seed) for the other methods.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    K, cp = np.broadcast_arrays(np.asarray(K, dtype=np.float64), np.asarray(cp, dtype=np.float64))
    bsm.check_market(np.asarray(F), K, np.asarray(T), np.asarray(DF), cp)
    heston, N = isinstance(model, HestonParams), int(n_paths)
    if not heston and n_steps != 1:
        raise ValueError("BSM terminal sampling is exact: n_steps must be 1")
    if N < 2 or (method == "antithetic" and N % 2):
        raise ValueError("need n_paths >= 2 (antithetic: even, N/2 pairs)")
    Kc, cpc = K[..., None], cp[..., None]
    if method == "sobol":
        R = int(n_scramble)
        if R < 2 or N % R:
            raise ValueError("sobol needs n_scramble >= 2 dividing n_paths")
        est = []
        for ss in np.random.SeedSequence(seed).spawn(R):
            x = _terminal(model, T, n_steps, N // R, method, np.random.default_rng(ss))
            est.append(DF * vanilla(F * np.exp(x), Kc, cpc).mean(axis=-1))
        price, se = mc.mc_estimate(np.stack(est, axis=-1))
        return price[()], se[()], N, int(seed)
    x = _terminal(model, T, n_steps, N, method, np.random.default_rng(seed))
    g = DF * vanilla(F * np.exp(x), Kc, cpc)
    if method == "plain":
        price, se = mc.mc_estimate(g)
    elif method == "antithetic":
        price, se = antithetic_estimate(g)
    else:
        FT = np.broadcast_to(F * np.exp(x), g.shape)
        price, se, _, _ = control_variate(g, FT, np.full(K.shape, F))
    return price[()], se[()], N, int(seed)
