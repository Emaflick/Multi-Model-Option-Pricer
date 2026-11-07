"""Fourier pricing of European options from a characteristic function: COS and Lewis (2000).

Both engines are forward-based (F, K, T, DF; no r, q) and vectorised over the strikes of one expiry
(F, T, DF scalars, K any shape, cp = +1 call / -1 put broadcast with K). Equation numbers refer to
docs/derivations/heston.md. References: Fang & Oosterlee (2008) SIAM J. Sci. Comput. 31(2);
Le Floc'h (2020) arXiv:2005.13248 (forward-centred COS); Lewis (2000), Option Valuation under
Stochastic Volatility; Lipton, Gal, Lasis (2014) Quant. Finance 14(11).
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad_vec

from qpricer.models import bsm
from qpricer.models.heston import HestonParams, cf, cumulants


def _inputs(F, K, T, DF, cp) -> tuple[float, NDArray, float, float, NDArray]:
    F, T, DF = (float(a) for a in (F, T, DF))
    K, cp = np.broadcast_arrays(np.asarray(K, dtype=np.float64), np.asarray(cp, dtype=np.float64))
    bsm.check_market(np.asarray(F), K, np.asarray(T), np.asarray(DF), cp)
    return F, K, T, DF, cp


N_MAX = 2**14  # cap of the self-checking COS refinement (D-017)


def cos_range(T: float, p: HestonParams, L: float = 18.0) -> tuple[float, float]:
    """Truncation range [a, b] = [c1 - L sqrt(c2), c1 + L sqrt(c2)] for X = ln(F_T/F), eq. (8).

    c4 omitted; c2 from eq. (7), not the misprinted F&O Table 11 (no |c2|). L = 12 (the note) is
    too narrow for heavy left tails (§3.3 reflection term, pitfall 11); see D-017.
    """
    c1, c2 = cumulants(T, p)
    if not c2 > 0:
        raise ValueError(f"c2 = {c2} <= 0: cumulant formula outside its domain")
    h = L * np.sqrt(c2)
    return c1 - h, c1 + h


def cos_series_error(T: float, p: HestonParams, N: int = 256, L: float = 18.0) -> float:
    """Series-error estimate e^{-c u_N}, u_N = pi N / (b - a), in units of F/100, eq. (13).

    c = sqrt(1 - rho^2) (v0 + kappa theta T) / xi is the large-u decay rate of |phi| (§3.6). It is
    an (est.): measured series errors are ~20x smaller (D-017). It does not cover range error.
    """
    a, b = cos_range(T, p, L)
    c = np.sqrt(1.0 - p.rho**2) * (p.v0 + p.kappa * p.theta * T) / p.xi
    return float(np.exp(-c * np.pi * N / (b - a)))


def _cos_put(F: float, K: NDArray, T: float, DF: float, p: HestonParams, N: int, L: float):
    """COS put, eqs. (9)-(11), at fixed (N, L)."""
    a, b = cos_range(T, p, L)
    u = np.arange(N) * np.pi / (b - a)
    w = np.real(cf(u, T, p) * np.exp(-1j * u * a))
    w[0] = 0.5
    z = np.minimum(np.maximum(np.log(K / F), a), b)[..., None]
    arg = u * (z - a)
    ez = np.exp(z)
    chi = (ez * np.cos(arg) + u * ez * np.sin(arg) - np.exp(a)) / (1.0 + u * u)
    u1 = np.where(u > 0, u, 1.0)
    psi = np.where(u > 0, np.sin(arg) / u1, z - a)
    return DF * (2.0 / (b - a)) * ((K[..., None] * psi - F * chi) @ w)


def cos_price(
    F: float,
    K: ArrayLike,
    T: float,
    DF: float,
    p: HestonParams,
    cp: ArrayLike,
    N: int | None = None,
    L: float | None = None,
    tol: float = 1e-9,
) -> NDArray:
    """Heston European price by forward-centred COS: puts directly, calls by parity, eqs. (9)-(11).

    u_k = k pi / (b - a), w_k = Re[phi(u_k) e^{-i u_k a}], w_0 = 1/2                      eq. (9)
    V_k = 2/(b - a) [K psi_k(z_bar) - F chi_k(z_bar)], z_bar = min(max(ln(K/F), a), b)      eq. (10)
    P = DF sum_k w_k V_k (one (M x N) @ N product per expiry), C = P + DF (F - K)          eq. (11)
    The bound on z is the integration domain of eq. (10) (P = 0 for z <= a, P ~ DF (K - F) for
    z >= b), not a clip of the result; tiny negative prices (~1e-14) are returned as computed
    (pitfall 13). cp (+1 call, -1 put) is required (validator F4).
    Default (N = L = None), self-checking (D-017): start at (N, L) = (256, 18); while
    max_K |P(N, L) - P(2N, L + 6)| > tol F, set (N, L) <- (2N, L + 6); return the finer P. This
    bounds range (pitfall 11) and series (pitfall 10) error together; at N_MAX it warns. Fixed
    (N and/or L given; the other defaults to 256 / 18): one evaluation, no range check, and only
    the eq. (13) series warning (estimate > 1e-5, i.e. measured error ~5e-7 F/100).
    """
    F, K, T, DF, cp = _inputs(F, K, T, DF, cp)
    if N is None and L is None:
        N, L = 256, 18.0
        put = _cos_put(F, K, T, DF, p, N, L)
        while True:
            N, L, prev = 2 * N, L + 6.0, put
            put = _cos_put(F, K, T, DF, p, N, L)
            gap = float(np.max(np.abs(put - prev), initial=0.0))
            if gap <= tol * F:
                break
            if N >= N_MAX:
                msg = f"COS not converged: |P(N) - P(N/2)| = {gap:.1e} > tol F at N = {N}"
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
                break
    else:
        N, L = N or 256, 18.0 if L is None else L
        err = cos_series_error(T, p, N, L)
        if err > 1e-5:
            msg = f"COS N = {N}: series error estimate {err:.1e} F/100 (eq. 13); raise N"
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
        put = _cos_put(F, K, T, DF, p, N, L)
    return np.where(cp > 0, put + DF * (F - K), put)[()]


def lewis_price(
    F: float,
    K: ArrayLike,
    T: float,
    DF: float,
    p: HestonParams,
    cp: ArrayLike,
    tol: float = 1e-11,
) -> NDArray:
    """Heston European price by the Lewis (2000) single integral on Im u = -1/2, eqs. (14)-(15).

    C = DF [F - sqrt(FK)/pi int_0^inf Re[e^{iux} phi(u - i/2)] du / (u^2 + 1/4)], x = ln(F/K),
    P = DF [K - (same term)]. One `quad_vec` call (semi-infinite GK15) over the strike vector,
    norm='max', epsabs = epsrel = tol = 1e-11 (eq. 15: scipy's default 1.49e-8 gives ~2e-6 errors,
    pitfall 14). cp (+1 call, -1 put) is required. Raises RuntimeError unless quad_vec reports
    status 0.
    """
    F, K, T, DF, cp = _inputs(F, K, T, DF, cp)
    x = np.log(F / K)

    def integrand(u: float) -> NDArray:
        return np.real(np.exp(1j * u * x) * cf(u - 0.5j, T, p)) / (u * u + 0.25)

    integral, err, info = quad_vec(
        integrand, 0.0, np.inf, epsabs=tol, epsrel=tol, norm="max", full_output=True
    )
    if info.status != 0:
        raise RuntimeError(f"Lewis quad_vec failed: status {info.status}, error estimate {err:.2e}")
    core = np.sqrt(F * K) / np.pi * integral
    return np.where(cp > 0, DF * (F - core), DF * (K - core))[()]
