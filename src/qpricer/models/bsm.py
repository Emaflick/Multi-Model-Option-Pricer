"""Black-76 (BSM in forward form): price and Greeks, vectorised and forward-based.

Every kernel takes (F, K, T, DF, sigma, cp) and broadcasts them with NumPy rules; cp = +1 call,
-1 put (theta in the note). Equation numbers refer to docs/derivations/bsm.md. With
x = ln(F/K) and s = sigma sqrt(T):

    V = DF cp [F Phi(cp d1) - K Phi(cp d2)],   d_{1,2} = x/s +- s/2                        eq. (1)

Inputs: F, K, T, DF > 0 and finite (DF > 1, i.e. a negative rate, is accepted); sigma >= 0 and
finite; cp in {+1, -1}. Anything else raises ValueError. At s = 0 the kernels return the
continuous limits (pitfall 5, D-012), via d_{1,2} = +inf / -inf / 0 for x > 0 / x < 0 / x = 0.
Upper tails use ndtr(cp d), never 1 - ndtr (pitfall 1).
References: Black (1976), J. Financial Economics 3; Hull (2003), OFOD 5th ed.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr

_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _phi(d: NDArray) -> NDArray:
    """Standard normal pdf; phi(+-inf) = 0 (d*d -> inf for |d| > 1e154 is exact, not an error)."""
    with np.errstate(over="ignore"):
        return _INV_SQRT_2PI * np.exp(-0.5 * d * d)


def _d12(x: NDArray, s: NDArray) -> tuple[NDArray, NDArray]:
    """d_{1,2} = x/s +- s/2 (eq. 1); at s = 0 the limits sign(x)*inf, and 0 at x = 0 (pitfall 5)."""
    pos = s > 0
    s1 = np.where(pos, s, 1.0)
    lim = np.where(x > 0, np.inf, np.where(x < 0, -np.inf, 0.0))
    return np.where(pos, x / s1 + 0.5 * s, lim), np.where(pos, x / s1 - 0.5 * s, lim)


def check_market(F: NDArray, K: NDArray, T: NDArray, DF: NDArray, cp: NDArray) -> None:
    """Raise ValueError unless F, K, T, DF are finite and > 0 and cp is +-1 (note §1, pitfall 5)."""
    for name, a in (("F", F), ("K", K), ("T", T), ("DF", DF)):
        if not np.all(np.isfinite(a) & (a > 0)):
            raise ValueError(f"{name} must be finite and > 0")
    if not np.all(np.abs(cp) == 1):
        raise ValueError("cp must be +1 (call) or -1 (put)")


def _inputs(F, K, T, DF, sigma, cp) -> tuple[NDArray, ...]:
    """Broadcast and validate; return (F, K, T, DF, sigma, cp, x, s, d1, d2)."""
    F, K, T, DF, sigma, cp = np.broadcast_arrays(
        *(np.asarray(a, dtype=np.float64) for a in (F, K, T, DF, sigma, cp))
    )
    check_market(F, K, T, DF, cp)
    if not np.all(np.isfinite(sigma) & (sigma >= 0)):
        raise ValueError("sigma must be finite and >= 0")
    x, s = np.log(F / K), sigma * np.sqrt(T)
    return (F, K, T, DF, sigma, cp, x, s, *_d12(x, s))


def price(F: ArrayLike, K: ArrayLike, T: ArrayLike, DF: ArrayLike, sigma: ArrayLike, cp=1):
    """Black-76 price V = DF cp [F Phi(cp d1) - K Phi(cp d2)], eq. (1), derivations/bsm.md.

    Parity (2) C - P = DF (F - K) and bounds (4) hold. At sigma = 0, V = DF (cp (F - K))^+.
    Accuracy is absolute (~eps DF max(F, K)); relative accuracy is lost once Phi(cp d2)
    underflows (OTM prices ~1e-280 and below, D-013).
    """
    F, K, T, DF, sigma, cp, x, s, d1, d2 = _inputs(F, K, T, DF, sigma, cp)
    return (DF * cp * (F * ndtr(cp * d1) - K * ndtr(cp * d2)) + 0.0)[()]  # +0.0: no -0.0 puts


def normalised_price(x: ArrayLike, s: ArrayLike, cp=1):
    """b_cp(x, s) = cp [e^{x/2} Phi(cp d1) - e^{-x/2} Phi(cp d2)] = V / (DF sqrt(FK)), eq. (3).

    x = ln(F/K), s = sigma sqrt(T) >= 0; symmetry b_{-1}(x, s) = b_{+1}(-x, s).
    """
    x, s, cp = np.broadcast_arrays(*(np.asarray(a, dtype=np.float64) for a in (x, s, cp)))
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(s) & (s >= 0))):
        raise ValueError("x must be finite and s finite and >= 0")
    if not np.all(np.abs(cp) == 1):
        raise ValueError("cp must be +1 (call) or -1 (put)")
    d1, d2 = _d12(x, s)
    b = cp * (np.exp(0.5 * x) * ndtr(cp * d1) - np.exp(-0.5 * x) * ndtr(cp * d2))
    return (b + 0.0)[()]


def delta(F: ArrayLike, K: ArrayLike, T: ArrayLike, DF: ArrayLike, sigma: ArrayLike, cp=1):
    """Forward delta dV/dF = cp DF Phi(cp d1) at fixed (K, T, DF, sigma), eq. (5).

    Spot delta = delta * dF/dS, eq. (10). At sigma = 0: cp DF 1{cp x > 0}, cp DF / 2 at x = 0.
    """
    F, K, T, DF, sigma, cp, x, s, d1, d2 = _inputs(F, K, T, DF, sigma, cp)
    return (cp * DF * ndtr(cp * d1))[()]


def gamma(F: ArrayLike, K: ArrayLike, T: ArrayLike, DF: ArrayLike, sigma: ArrayLike, cp=1):
    """Forward gamma d2V/dF2 = DF phi(d1) / (F sigma sqrt T), eq. (6); same for call and put.

    At sigma = 0: 0 for x != 0 and +inf at x = 0 (pitfall 5).
    """
    F, K, T, DF, sigma, cp, x, s, d1, d2 = _inputs(F, K, T, DF, sigma, cp)
    s1 = np.where(s > 0, s, 1.0)
    return np.where(s > 0, DF * _phi(d1) / (F * s1), np.where(x == 0, np.inf, 0.0))[()]


def vega(F: ArrayLike, K: ArrayLike, T: ArrayLike, DF: ArrayLike, sigma: ArrayLike, cp=1):
    """Vega dV/dsigma = DF F phi(d1) sqrt(T) = DF K phi(d2) sqrt(T), eq. (7); per unit of vol.

    Divide by 100 for "per vol point". At sigma = 0: 0 for x != 0 and the right-derivative
    DF F sqrt(T) / sqrt(2 pi) at x = 0 (D-012).
    """
    F, K, T, DF, sigma, cp, x, s, d1, d2 = _inputs(F, K, T, DF, sigma, cp)
    return (DF * F * _phi(d1) * np.sqrt(T))[()]


def theta(F: ArrayLike, K: ArrayLike, T: ArrayLike, DF: ArrayLike, sigma: ArrayLike, cp=1):
    """Calendar theta dV/dt, per year, eq. (8): Theta = Theta_0 + r V.

    Convention (note §1): t is calendar time (tau = T - t shrinks) with the forward F to the same
    expiry date, K, sigma and the zero rate r = -ln(DF)/T held fixed, so DF(tau) = e^{-r tau}
    accretes towards 1. Theta_0 = -DF F phi(d1) sigma / (2 sqrt T) is the decay at fixed DF
    (= -dV/dT at fixed F, DF, sigma). This is the dt term of the Black PDE (9):
    Theta + sigma^2 F^2 Gamma / 2 - r V = 0. Per calendar day: Theta / 365.
    """
    F, K, T, DF, sigma, cp, x, s, d1, d2 = _inputs(F, K, T, DF, sigma, cp)
    V = DF * cp * (F * ndtr(cp * d1) - K * ndtr(cp * d2))
    theta0 = -DF * F * _phi(d1) * sigma / (2.0 * np.sqrt(T))
    return (theta0 - np.log(DF) * V / T)[()]  # r V as (-ln DF) V / T: no inf * 0 at tiny T
