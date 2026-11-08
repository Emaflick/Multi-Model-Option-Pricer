"""SABR, beta = 1 (D-022): dF = alpha_t F dW, d alpha = nu alpha dZ, d<W, Z> = rho dt under Q^T.

Eq. numbers: docs/derivations/sabr.md. sigma_H feeds Black-76 (bsm.price); no r, q (D-001).
Inputs F, K, T > 0, alpha > 0, nu >= 0, |rho| <= 0.999 (pitfall 2), else ValueError. Refs: Hagan
et al. (2002) Wilmott (preprint 2.17-2.18, B.71); Lions-Musiela (2007); Choi-Hu-Kwok (2025) EJOR.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr

from qpricer.engines.mc import mc_estimate
from qpricer.models import bsm

RHO_MAX = 0.999  # code bound on |rho|, note §1 and pitfall 2
Z_C = 1e-3  # series switch for z / x(z), eq. (5)


def _check_params(alpha, rho, nu) -> None:
    a, r, n = (np.asarray(v, dtype=np.float64) for v in (alpha, rho, nu))
    if not np.all((0 < a) & (a < np.inf) & (0 <= n) & (n < np.inf) & (np.abs(r) <= RHO_MAX)):
        raise ValueError(f"need finite alpha > 0, nu >= 0 and |rho| <= {RHO_MAX}")


def z_over_x(z: ArrayLike, rho: ArrayLike) -> NDArray:
    """z / x(z), x(z) = ln[(D + z - rho) / (1 - rho)] = int_0^z du / D(u), eqs. (3), (5).

    Sign split of (5) (z >= rho / z < rho), each log as log1p of non-negative terms via
    D - 1 = z (z - 2 rho) / (D + 1): no cancellation (D-024). |z| < Z_C: cubic series of (5).
    z_over_x(-z, -rho) == z_over_x(z, rho) bitwise, so the §3.3 identities hold to round-off.
    """
    z, rho = np.broadcast_arrays(np.asarray(z, dtype=np.float64), np.asarray(rho, dtype=np.float64))
    small = np.abs(z) < Z_C
    zs = np.where(small, 1.0, z)  # both np.where branches are evaluated (pitfall 1)
    d1 = np.sqrt((zs - rho) ** 2 + (1.0 - rho) * (1.0 + rho)) + 1.0  # D + 1
    up = np.log1p(zs * (d1 + (zs - 2.0 * rho)) / (d1 * (1.0 - rho)))
    dn = -np.log1p(-zs * (d1 + (2.0 * rho - zs)) / (d1 * (1.0 + rho)))
    series = 1.0 - rho * z / 2 + (2 - 3 * rho**2) * z**2 / 12 + rho * (5 - 6 * rho**2) * z**3 / 24
    return np.where(small, series, zs / np.where(zs >= rho, up, dn))


def implied_vol(F: ArrayLike, K: ArrayLike, T: ArrayLike, alpha, rho, nu):
    """Hagan lognormal vol, beta = 1, eq. (2): sigma_H = alpha (z / x(z)) (1 + c1 T), broadcast.

    z = (nu / alpha) ln(F/K), c1 = rho nu alpha / 4 + (2 - 3 rho^2) nu^2 / 24; K = F gives (4),
    nu = 0 gives alpha exactly (7). 1 + c1 T <= 0 (eq. 22): NaN + RuntimeWarning (pitfall 3).
    An asymptotic in nu^2 T, alpha^2 T (§2.4): degrades at long T and in the wings (21).
    """
    F, K, T, alpha, rho, nu = np.broadcast_arrays(
        *(np.asarray(a, dtype=np.float64) for a in (F, K, T, alpha, rho, nu))
    )
    for name, a in (("F", F), ("K", K), ("T", T)):
        if not np.all(np.isfinite(a) & (a > 0)):
            raise ValueError(f"{name} must be finite and > 0")
    _check_params(alpha, rho, nu)
    bracket = 1.0 + (rho * nu * alpha / 4 + (2 - 3 * rho**2) * nu**2 / 24) * T
    sigma = alpha * z_over_x(nu / alpha * np.log(F / K), rho) * bracket
    if np.any(bracket <= 0):
        warnings.warn("SABR: 1 + c1 T <= 0 (eq. 22), sigma_H set to NaN", RuntimeWarning, 2)
        sigma = np.where(bracket > 0, sigma, np.nan)
    return sigma[()]


def alpha_from_atm(sigma_atm: ArrayLike, T: ArrayLike, rho, nu):
    """alpha with sigma_ATM = alpha (1 + c1(alpha) T), eq. (9): 2 s / (b + sqrt(b^2 + 4 a s)),
    a = rho nu T / 4, b = 1 + (2 - 3 rho^2) nu^2 T / 24. No positive root: NaN + RuntimeWarning.
    """
    s, T, rho, nu = np.broadcast_arrays(
        *(np.asarray(v, dtype=np.float64) for v in (sigma_atm, T, rho, nu))
    )
    if not np.all(np.isfinite(s) & (s > 0) & np.isfinite(T) & (T > 0)):
        raise ValueError("sigma_atm and T must be finite and > 0")
    _check_params(1.0, rho, nu)
    a, b = rho * nu * T / 4, 1.0 + (2 - 3 * rho**2) * nu**2 * T / 24
    disc = b * b + 4.0 * a * s
    den = b + np.sqrt(np.maximum(disc, 0.0))  # max only feeds the flagged branch
    ok = (disc > 0) & (den > 0)
    if not np.all(ok):
        warnings.warn("SABR ATM inversion (9): no positive alpha, NaN returned", RuntimeWarning, 2)
    return np.where(ok, 2.0 * s / np.where(ok, den, 1.0), np.nan)[()]


def simulate(alpha: float, rho: float, nu: float, dt: float, draws: Iterable[NDArray]):
    """Terminal SABR state on a uniform grid from one (2, N) block [Z_alpha; Z_perp] per step.

    Exact vol (13) as alpha_n = alpha exp(nu W_n - nu^2 t_n / 2), W_n = sqrt(dt) sum Z_alpha;
    log-Euler (14) with Z_F = rho Z_alpha + sqrt(1 - rho^2) Z_perp (positive, exact martingale).
    Returns (x_T = ln(F_T/F), alpha_T, W_T, I_left = dt sum_{n<M} alpha_n^2), eq. (16).
    """
    _check_params(alpha, rho, nu)
    sq, rb = np.sqrt(dt), np.sqrt(1.0 - rho * rho)
    a, w, x, i_left, n = float(alpha), 0.0, 0.0, 0.0, 0
    for n, z in enumerate(draws, start=1):
        x = x + a * (sq * (rho * z[0] + rb * z[1]) - 0.5 * a * dt)  # eq. (14), left point
        i_left = i_left + a * a
        w = w + sq * z[0]
        a = alpha * np.exp(nu * w - 0.5 * nu * nu * n * dt)  # eq. (13), exact in law
    if n == 0 or not (np.all(np.isfinite(x)) and np.all(np.isfinite(a))):
        raise FloatingPointError("SABR MC: no steps, or non-finite state")
    return x, a, w, i_left * dt


def mc_price(F, K, T, DF, alpha, rho, nu, cp, *, n_steps, n_paths, seed, method="euler"):
    """SABR (beta = 1) European price by MC, dt = T / n_steps: (price, stderr, N, seed).

    "euler": V = DF mean (cp (F e^{x_T} - K))^+, (13)-(14), weak order 1 (16). "mixing": eq.
    (17), V = DF mean Black_cp(F xi, K, sqrt((1 - rho^2) I / T)), ln xi = rho J - rho^2 I / 2,
    I trapezoid, J = (alpha_T - alpha)/nu = alpha expm1(nu W_T - nu^2 T/2)/nu (alpha W_T at
    nu = 0, pitfall 10); bias O(dt^2). SE per heston-mc.md eq. (1); strikes share the paths.
    Raw calls have infinite variance for rho > -1/sqrt 2: use puts for "euler" (pitfall 8); for
    "mixing" with rho <= 0, F xi <= F e^{|rho| alpha / nu} bounds the call too (D-023). rho > 0:
    F is a strict local martingale (E F_T < F, [LM07]) while (14) is an exact martingale, so
    "euler" calls (and parity with F) target P + DF (F - K), not the model call that "mixing"
    prices; puts agree. Calls and parity with F are the model's only for rho <= 0.
    """
    K, cp = np.broadcast_arrays(np.asarray(K, dtype=np.float64), np.asarray(cp, dtype=np.float64))
    bsm.check_market(np.asarray(F), K, np.asarray(T), np.asarray(DF), cp)
    if method not in ("euler", "mixing") or not all(
        float(n).is_integer() and n >= m for n, m in ((n_steps, 1), (n_paths, 2))
    ):
        raise ValueError("method 'euler'/'mixing', integer n_steps >= 1 and n_paths >= 2")
    rng, dt = np.random.default_rng(seed), T / int(n_steps)
    draws = (rng.standard_normal((2, int(n_paths))) for _ in range(int(n_steps)))
    x, aT, w, i_left = simulate(alpha, rho, nu, dt, draws)
    Kc, c = K[..., None], cp[..., None]
    if method == "euler":
        g = np.maximum(c * (F * np.exp(x) - Kc), 0.0)
    else:
        iv = i_left + 0.5 * dt * (aT * aT - alpha * alpha)  # trapezoid I, eq. (16)
        J = alpha * w if nu == 0 else alpha * np.expm1(nu * w - 0.5 * nu * nu * T) / nu
        lnf, s = np.log(F) + rho * J - 0.5 * rho * rho * iv, np.sqrt((1.0 - rho * rho) * iv)
        d1 = (lnf - np.log(Kc)) / s + 0.5 * s  # Black (bsm.md eq. 1) in ln F: F xi may underflow
        g = c * (np.exp(lnf) * ndtr(c * d1) - Kc * ndtr(c * (d1 - s)))
    if not np.all(np.isfinite(g)):
        raise FloatingPointError("SABR MC: non-finite payoff")
    price, se = mc_estimate(g, DF)
    return price[()], se[()], int(n_paths), int(seed)
