"""Heston (1993) model under the T-forward measure: parameters, characteristic function, cumulants.

dF = sqrt(v) F dW^F,  dv = kappa (theta - v) dt + xi sqrt(v) dW^v,  d<W^F, W^v> = rho dt.
X = ln(F_T / F) and phi(u) = E[exp(i u X)]. Equation numbers refer to docs/derivations/heston.md.
References: Heston (1993) RFS 6(2); Albrecher, Mayer, Schoutens, Tistaert (2007) "The little
Heston trap", Wilmott; Fang & Oosterlee (2008) SIAM J. Sci. Comput. 31(2); Le Floc'h (2020),
arXiv:2005.13248.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class HestonParams:
    """Heston parameters (note §1): kappa, theta, xi > 0, v0 >= 0, |rho| < 1, kappa - rho xi > 0.

    v0 and theta are annualised variances, kappa in 1/yr, xi (vol-of-vol) in 1/sqrt(yr). The domain
    kappa - rho xi > 0 (pitfall 5) holds automatically for rho <= 0; other inputs raise ValueError.
    """

    kappa: float
    theta: float
    xi: float
    rho: float
    v0: float

    def __post_init__(self) -> None:
        vals = (self.kappa, self.theta, self.xi, self.rho, self.v0)
        if not all(np.isfinite(vals)):
            raise ValueError("Heston parameters must be finite")
        if min(self.kappa, self.theta, self.xi) <= 0 or self.v0 < 0:
            raise ValueError("need kappa, theta, xi > 0 and v0 >= 0")
        if not abs(self.rho) < 1:
            raise ValueError("need |rho| < 1")
        if self.kappa - self.rho * self.xi <= 0:
            raise ValueError("need kappa - rho xi > 0 (note pitfall 5)")

    @property
    def feller(self) -> float:
        """Feller ratio nu_F = 2 kappa theta / xi^2 (note §1); v never reaches 0 iff nu_F >= 1."""
        return 2.0 * self.kappa * self.theta / self.xi**2


def _log1p(w: NDArray) -> NDArray:
    """Principal Log(1 + w) = 1/2 log1p(2 Re w + |w|^2) + i atan2(Im w, 1 + Re w), pitfall 3."""
    wr, wi = w.real, w.imag
    return 0.5 * np.log1p(2.0 * wr + wr * wr + wi * wi) + 1j * np.arctan2(wi, 1.0 + wr)


def cf(u: ArrayLike, T: float, p: HestonParams) -> NDArray:
    """Characteristic function phi(u) = E[e^{iuX}], X = ln(F_T/F), little-trap form, eqs. (1)-(3).

    beta = kappa - i rho xi u, A = u^2 + iu, d = sqrt(beta^2 + xi^2 A) (Re d > 0), eq. (1).
    Eq. (3) without any subtraction beta - d (= -xi^2 A / (beta + d)): with e1 = 1 - e^{-dT} and
    q = xi^2 A e1 / (2 d (beta + d)), Upsilon / (2d) = (1 - g e^{-dT}) / (1 - g) = 1 - q, so
        D = -A e1 / (2 d (1 - q)),
        C = -kappa theta A T / (beta + d) - (2 kappa theta / xi^2) Log(1 - q),
    phi = exp(C + v0 D). Log(1 - q) is evaluated by `_log1p`, so the O(xi^2) Log term keeps full
    relative accuracy as xi -> 0 (pitfall 3). u may be complex, -w_+ < Im u < -w_- (eq. 17); COS
    uses real u, Lewis Im u = -1/2. phi(0) = phi(-i) = 1 exactly (eq. 5).
    """
    u = np.asarray(u, dtype=np.complex128)
    k, th, xi, rho, v0 = p.kappa, p.theta, p.xi, p.rho, p.v0
    beta = k - 1j * rho * xi * u
    A = u * u + 1j * u
    d = np.sqrt(beta * beta + xi * xi * A)
    s = beta + d
    e1 = -np.expm1(-d * T)
    q = xi * xi * A * e1 / (2.0 * d * s)
    D = -A * e1 / (2.0 * d * (1.0 - q))
    C = -k * th * A * T / s - (2.0 * k * th / (xi * xi)) * _log1p(-q)
    return np.exp(C + v0 * D)


def total_variance(T: ArrayLike, p: HestonParams) -> NDArray:
    """Integrated expected variance Sigma_bar = theta T + (v0 - theta) tau_k, eq. (6).

    vbar(t) = theta + (v0 - theta) e^{-kappa t}, tau_k = (1 - e^{-kappa T}) / kappa.
    Sigma_bar / T is sigma_eff^2 of the xi -> 0 limit, eq. (16).
    """
    T = np.asarray(T, dtype=np.float64)
    tau = -np.expm1(-p.kappa * T) / p.kappa
    return p.theta * T + (p.v0 - p.theta) * tau


def cumulants(T: float, p: HestonParams) -> tuple[float, float]:
    """First two cumulants (c1, c2) of X = ln(F_T/F), eqs. (6)-(7) (= Le Floc'h 2020 eq. B5).

    c1 = -Sigma_bar / 2. c2 per eq. (7) with alpha_0,1,2; NOT Fang-Oosterlee Table 11, whose
    u_bar (6 e^{-lambda T} - 7) is a misprint for u_bar (4 e^{-lambda T} - 5) (§2.2, pitfall 8).
    For kappa T < 1e-3 (pitfall 9) a first-order expansion in kappa of the §3.2 moment ODEs:
    c2 = v0 (T - rho xi T^2/2 + xi^2 T^3/12) + kappa [v0 (-T^2/2 + rho xi T^3/3 - xi^2 T^4/12)
    + theta (T^2/2 - rho xi T^3/6 + xi^2 T^4/48)] + O(kappa^2) (validator F3; rel. error <= 1e-5).
    """
    k, th, xi, rho, v0 = p.kappa, p.theta, p.xi, p.rho, p.v0
    c1 = -0.5 * float(total_variance(T, p))
    if k * T < 1e-3:
        c2 = v0 * (T - 0.5 * rho * xi * T**2 + xi**2 * T**3 / 12.0)
        c2 += k * v0 * (-(T**2) / 2 + rho * xi * T**3 / 3 - xi**2 * T**4 / 12)
        return c1, c2 + k * th * (T**2 / 2 - rho * xi * T**3 / 6 + xi**2 * T**4 / 48)
    E = np.exp(-k * T)
    tau = -np.expm1(-k * T) / k
    a0 = 0.5 - rho * xi / (2 * k) + xi**2 / (8 * k**2)
    a1 = rho * xi / (2 * k) - xi**2 / (4 * k**2)
    a2 = xi**2 / (8 * k**2)
    c2 = 2 * th * (a0 * (T - tau) + a1 * (tau - T * E) + 0.5 * a2 * k * tau**2)
    c2 += 2 * v0 * (a0 * tau + a1 * T * E + a2 * E * tau)
    return c1, float(c2)
