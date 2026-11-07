"""Heston Monte Carlo under the T-forward measure: Euler full truncation, Andersen QE and QE-M.

State x_t = ln(F_t / F) (x_0 = 0) and v_t, simulated on a time grid 0 < t_1 < ... < t_M that must
contain every observation date. One (2, N) block of iid N(0, 1) per step: row 0 = Z_v, row 1 =
Z_perp (same order in every scheme, pitfall 8). Equation numbers refer to
docs/derivations/heston-mc.md. MC prices return (price, stderr, n_paths, seed), eq. (1).
Path observations for Phase 4: S_t = F_0(t) e^{x_t} (eq. 17), F_0(t) from market.py.
References: Andersen (2008) J. Comput. Finance 11(3) (read as the 2006 WP); Lord, Koekkoek,
van Dijk (2010) Quant. Finance 10(2); Glasserman (2003) Monte Carlo Methods in Fin. Eng.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr

from qpricer.models import bsm
from qpricer.models.heston import HestonParams

SCHEMES = ("euler", "qe", "qe-m")
PSI_C = 1.5  # QE switching threshold psi_c in [1, 2], eqs. (8)-(9)


def _euler_step(x, v, dt, p: HestonParams, zv, zp):
    """Euler full truncation, eqs. (4)-(6): only v+ enters; E[e^{dx} | F_t] = 1 exactly."""
    vp = np.maximum(v, 0.0)
    sq = np.sqrt(vp * dt)
    zf = p.rho * zv + np.sqrt(1.0 - p.rho**2) * zp  # eq. (4)
    v_new = v + p.kappa * (p.theta - vp) * dt + p.xi * sq * zv  # eq. (5), v may go negative
    x_new = x - 0.5 * vp * dt + sq * zf  # eq. (6)
    return x_new, v_new


def _qe_step(x, v, dt, p: HestonParams, zv, zp, martingale: bool):
    """Andersen QE variance step (7)-(9), log-forward step (10), optional QE-M drift (11)-(12).

    Both branches are evaluated on min/max(psi, psi_c) so neither produces NaN (pitfall 3);
    1 - U_v = ndtr(-Z_v) (pitfall 4). Central weights gamma_1 = gamma_2 = 1/2.
    """
    k, th, xi, rho = p.kappa, p.theta, p.xi, p.rho
    E = np.exp(-k * dt)
    m = th + (v - th) * E  # eq. (7)
    s2 = v * xi**2 * E * (1.0 - E) / k + th * xi**2 * (1.0 - E) ** 2 / (2.0 * k)
    psi = s2 / (m * m)
    quad = psi <= PSI_C
    iq = 2.0 / np.minimum(psi, PSI_C)  # eq. (8)
    b2 = iq - 1.0 + np.sqrt(iq * (iq - 1.0))
    a = m / (1.0 + b2)
    pe = np.maximum(psi, PSI_C)  # eq. (9)
    pp = (pe - 1.0) / (pe + 1.0)
    beta = (1.0 - pp) / m
    one_minus_u = ndtr(-zv)
    v_exp = np.where(one_minus_u >= 1.0 - pp, 0.0, np.log((1.0 - pp) / one_minus_u) / beta)
    v_new = np.where(quad, a * (np.sqrt(b2) + zv) ** 2, v_exp)

    c = dt * (k * rho / xi - 0.5)  # eq. (10)
    K0, K1, K2 = -rho * k * th * dt / xi, 0.5 * c - rho / xi, 0.5 * c + rho / xi
    K3 = K4 = 0.5 * dt * (1.0 - rho**2)
    if martingale:  # eqs. (11)-(12): E[e^{x_{n+1} - x_n} | v_n] = 1
        A = K2 + 0.5 * K4
        if A > 0 and np.any(np.where(quad, 2.0 * A * a >= 1.0, A >= beta)):
            raise ValueError("QE-M: M(v_n) infinite (A >= 1/(2a) or A >= beta), eq. (13)")
        with np.errstate(invalid="ignore", divide="ignore"):  # unselected branch only (checked)
            ln_m_quad = A * b2 * a / (1.0 - 2.0 * A * a) - 0.5 * np.log1p(-2.0 * A * a)
            ln_m_exp = np.log(pp + beta * (1.0 - pp) / (beta - A))
        K0 = -np.where(quad, ln_m_quad, ln_m_exp) - (K1 + 0.5 * K3) * v
    x_new = x + K0 + K1 * v + K2 * v_new + np.sqrt(K3 * v + K4 * v_new) * zp
    return x_new, v_new


def _check_grid(times: ArrayLike) -> NDArray:
    t = np.atleast_1d(np.asarray(times, dtype=np.float64))
    if t.ndim != 1 or not (np.all(np.isfinite(t)) and t[0] > 0 and np.all(np.diff(t) > 0)):
        raise ValueError("times must be a finite, strictly increasing 1-D grid with t_1 > 0")
    return t


def uniform_grid(T: float, n_steps: int) -> NDArray:
    """t_n = n T / M, n = 1..M (heston-mc.md §1)."""
    if not (T > 0 and n_steps >= 1):
        raise ValueError("need T > 0 and n_steps >= 1")
    return np.linspace(0.0, T, n_steps + 1)[1:]


def heston_steps(
    p: HestonParams,
    times: ArrayLike,
    n_paths: int,
    rng: np.random.Generator,
    scheme: str = "qe-m",
) -> Iterator[tuple[float, NDArray, NDArray, NDArray]]:
    """Generator over the grid: yields (t_n, x_n, v_n, z_n) after each step n = 1..M.

    x_n = ln(F_{t_n}/F), v_n the scheme's variance state (Euler-FT: may be < 0, use v+ as the
    left-point variance), z_n the (2, N) draws [Z_v; Z_perp] of the step t_{n-1} -> t_n (for
    common random numbers, e.g. autocallable eq. 15). Yielded arrays are fresh at every step and
    never mutated later, so consumers may keep references (running min/max, fixings, bridges).
    scheme: "euler" (eqs. 4-6), "qe" (7-10), "qe-m" (7-12). Euler warns if kappa dt >= 1
    (pitfall 2). Raises if a state becomes non-finite.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}")
    if not (isinstance(rng, np.random.Generator) and int(n_paths) >= 2):
        raise ValueError("need an explicit np.random.Generator and n_paths >= 2")
    t = _check_grid(times)
    dts = np.diff(t, prepend=0.0)
    if scheme == "euler" and p.kappa * dts.max() >= 1.0:
        warnings.warn("Euler: kappa dt >= 1, the mean factor oscillates", RuntimeWarning, 2)

    def gen():
        x, v = np.zeros(int(n_paths)), np.full(int(n_paths), p.v0)
        for tn, dt in zip(t, dts, strict=True):
            z = rng.standard_normal((2, int(n_paths)))
            if scheme == "euler":
                x, v = _euler_step(x, v, dt, p, z[0], z[1])
            else:
                x, v = _qe_step(x, v, dt, p, z[0], z[1], scheme == "qe-m")
            if not (np.all(np.isfinite(x)) and np.all(np.isfinite(v))):
                raise FloatingPointError(f"non-finite Heston state at t = {tn}")
            yield float(tn), x, v, z

    return gen()


def simulate(
    p: HestonParams,
    times: ArrayLike,
    n_paths: int,
    rng: np.random.Generator,
    scheme: str = "qe-m",
    observe: ArrayLike | None = None,
) -> tuple[NDArray, NDArray]:
    """(x, v) at the observation dates, each of shape (n_obs, n_paths); default: final date only.

    Every observation date must be a grid point (relative tolerance 1e-12), else ValueError.
    """
    t = _check_grid(times)
    obs = t[-1:] if observe is None else np.atleast_1d(np.asarray(observe, dtype=np.float64))
    idx = np.searchsorted(t, obs * (1.0 - 1e-12))
    if np.any(idx >= t.size) or not np.allclose(t[np.minimum(idx, t.size - 1)], obs, rtol=1e-12):
        raise ValueError("every observation date must lie on the simulation grid")
    snaps = {}
    for n, (_, x, v, _) in enumerate(heston_steps(p, t, n_paths, rng, scheme)):
        if n in idx:
            snaps[n] = (x, v)
    return np.stack([snaps[n][0] for n in idx]), np.stack([snaps[n][1] for n in idx])


def mc_estimate(g: ArrayLike, DF: float = 1.0) -> tuple[NDArray, NDArray]:
    """MC estimator over the last axis (paths): (DF mean g, DF s_g / sqrt(N)), ddof = 1, eq. (1)."""
    g = np.asarray(g, dtype=np.float64)
    n = g.shape[-1]
    return DF * g.mean(axis=-1), DF * g.std(axis=-1, ddof=1) / np.sqrt(n)


def european_price(
    F: float,
    K: ArrayLike,
    T: float,
    DF: float,
    p: HestonParams,
    cp: ArrayLike,
    *,
    n_steps: int,
    n_paths: int,
    seed: int,
    scheme: str = "qe-m",
) -> tuple[NDArray, NDArray, int, int]:
    """Heston European price by MC on a uniform grid (dt = T / n_steps): (price, stderr, N, seed).

    V = DF E[(cp (F e^{x_T} - K))^+], estimator and SE per eq. (1); all strikes share one path
    set (common random numbers), so on-path parity C - P = DF (mean F_T - K) holds to round-off.
    cp (+1 call, -1 put) is required. rng = np.random.default_rng(seed). Gate use:
    |V - V_ref| <= 3 SE, eq. (2).
    """
    K, cp = np.broadcast_arrays(np.asarray(K, dtype=np.float64), np.asarray(cp, dtype=np.float64))
    bsm.check_market(np.asarray(F), K, np.asarray(T), np.asarray(DF), cp)
    rng = np.random.default_rng(seed)
    x, _ = simulate(p, uniform_grid(T, n_steps), n_paths, rng, scheme)
    FT = F * np.exp(x[0])
    g = np.maximum(cp[..., None] * (FT - K[..., None]), 0.0)
    price, se = mc_estimate(g, DF)
    return price[()], se[()], int(n_paths), int(seed)
