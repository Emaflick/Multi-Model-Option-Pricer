"""Black-76 implied volatility, eqs. (11)-(16) of docs/derivations/bsm.md.

Normalised OTM formulation (eq. 11): with v = V/DF, iota = (cp (F - K))^+, x = ln(F/K),
beta = (v - iota) / sqrt(FK), solve for the total vol s = sigma sqrt(T)

    b(s) := b_{+1}(-|x|, s) = beta,     b'(s) = exp(-x^2/(2 s^2) - s^2/8) / sqrt(2 pi) > 0,   (12)

i.e. always invert the OTM option (parity (2) maps an ITM quote to its time value). Start at the
exact ATM inverse s = 2 sqrt(2) erfinv(beta) for x = 0, else at the inflection s_c = sqrt(2|x|)
(Manaster-Koehler), eq. (13). Newton safeguarded by the bracket [0, sigma_max sqrt T] (rtsafe,
eq. 14); stop on |ds| <= 1e-13 sqrt(T) or |b - beta| <= 4 eps beta (eq. 15). Elements not
converged after max_iter go to scipy.optimize.brentq on their final bracket (D-011).

Prices outside the strict bounds (4) -- v <= iota, time value below 10 eps v (pitfall 7),
v >= F (call) or K (put) -- non-finite prices, vols above sigma_max, and roots where float64
cannot resolve the OTM tail (Phi(d2) < PHI_MIN, prices ~1e-280 and below) return NaN with a
RuntimeWarning and a status code (STATUS); nothing is clipped, never sigma = 0 (D-013).
References: Manaster & Koehler (1982), J. Finance 37; Press et al. (2007), NR 3rd ed. §9.4.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.special import erfinv, ndtr

from qpricer.models import bsm

EPS = np.finfo(np.float64).eps
SIGMA_MAX = 10.0  # bracket cap sigma_max, eq. (14)
N_MAX = 100  # Newton iterations before the Brent fallback, eq. (15)
STEP_TOL = 1e-13  # |d sigma| stopping tolerance, eq. (15)
STATUS = {
    0: "ok",
    1: "v <= intrinsic or time value < 10 eps v (eq. 4, pitfall 7)",
    2: "v >= F (call) or K (put) (eq. 4)",
    3: "implied vol > sigma_max (eq. 14)",
    4: "non-finite price",
    5: "OTM time value below float64 tail resolution: Phi(d2) < PHI_MIN at the root (D-013)",
}
PHI_MIN = 1e3 * np.finfo(np.float64).tiny  # ~2.2e-305, d2 ~ -37.3: Phi(d2) still a normal float


def _b(a: NDArray, s: NDArray) -> NDArray:
    """OTM normalised call b(s) = b_{+1}(-a, s), a = |ln(F/K)|; eqs. (3), (11)."""
    return bsm.normalised_price(-a, s, 1.0)


def _db(a: NDArray, s: NDArray) -> NDArray:
    """Normalised vega b'(s) = exp(-a^2/(2 s^2) - s^2/8) / sqrt(2 pi), eq. (12); s > 0."""
    with np.errstate(over="ignore"):  # (a/s)^2 -> inf gives the exact b' = 0
        return np.exp(-0.5 * (a / s) ** 2 - 0.125 * s * s) / np.sqrt(2.0 * np.pi)


def _brent(a: float, beta: float, lo: float, hi: float) -> float:
    """Fallback root of b(s) = beta on the Newton bracket, b(lo) < beta <= b(hi) (eq. 15, D-011)."""
    return brentq(lambda s: float(_b(a, s)) - beta, lo, hi, xtol=1e-15, rtol=4 * EPS)


def implied_vol(
    V: ArrayLike,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    DF: ArrayLike,
    cp=1,
    *,
    sigma_max: float = SIGMA_MAX,
    max_iter: int = N_MAX,
    full_output: bool = False,
):
    """Black-76 implied vol of the discounted price V (eqs. 11-15); inputs broadcast.

    cp = +1 call, -1 put; ITM and OTM quotes are both accepted (reduced to the OTM time value).
    Returns sigma (NaN where status != 0, with a RuntimeWarning). With full_output=True returns
    (sigma, info): info["status"] (codes in STATUS), info["n_iter"] (Newton evaluations, max_iter
    for Brent elements) and info["brent"] (True where the Brent fallback was used).
    Market inputs F, K, T, DF <= 0 or non-finite, or cp not +-1, raise ValueError.
    """
    V, F, K, T, DF, cp = np.broadcast_arrays(
        *(np.asarray(z, dtype=np.float64) for z in (V, F, K, T, DF, cp))
    )
    bsm.check_market(F, K, T, DF, cp)
    if not (np.isfinite(sigma_max) and sigma_max > 0 and max_iter >= 1):
        raise ValueError("need finite sigma_max > 0 and max_iter >= 1")
    shape = V.shape
    V, F, K, T, DF, cp = (z.ravel() for z in (V, F, K, T, DF, cp))

    v = V / DF  # eq. (11)
    iota = np.maximum(cp * (F - K), 0.0)
    tv = v - iota
    a = np.abs(np.log(F / K))
    beta = tv / np.sqrt(F * K)
    sqT = np.sqrt(T)
    hi = sigma_max * sqT

    status = np.zeros(V.size, dtype=np.int8)  # eq. (4) bounds, then the cap of eq. (14)
    status[~np.isfinite(V)] = 4
    status[(status == 0) & ~(tv > 10.0 * EPS * v)] = 1
    status[(status == 0) & ~(v < np.where(cp > 0, F, K))] = 2
    status[(status == 0) & (_b(a, hi) < beta)] = 3

    idx = np.flatnonzero(status == 0)
    a_, be, h, tol = a[idx], beta[idx], hi[idx], STEP_TOL * sqT[idx]
    s = np.where(a_ > 0, np.sqrt(2.0 * a_), 2.0 * np.sqrt(2.0) * erfinv(be))  # eq. (13)
    s = np.where(s < h, s, 0.5 * h)  # s_c above the cap: start inside the bracket
    lo = np.zeros_like(s)
    out = np.full(V.size, np.nan)
    n_iter = np.zeros(V.size, dtype=np.int64)
    brent = np.zeros(V.size, dtype=bool)

    for n in range(1, max_iter + 1):  # safeguarded Newton, eq. (14), on the active set only
        if idx.size == 0:
            break
        f = _b(a_, s) - be
        bp = _db(a_, s)
        lo = np.where(f < 0, s, lo)
        h = np.where(f > 0, s, h)
        ds = f / np.where(bp > 0, bp, np.nan)  # b' underflow -> NaN step -> bisection
        s_nt = s - ds
        s_new = np.where((s_nt > lo) & (s_nt < h), s_nt, 0.5 * (lo + h))
        res_ok = np.abs(f) <= 4.0 * EPS * be  # eq. (15), residual test
        nt_ok = np.abs(ds) <= tol  # Newton correction below tol (may land on a bracket end)
        done = res_ok | nt_ok | (np.abs(s_new - s) <= tol)  # eq. (15), step test
        fin = np.where(res_ok, s, np.where(nt_ok, s_nt, s_new))
        out[idx[done]] = fin[done] / sqT[idx[done]]
        n_iter[idx[done]] = n
        keep = ~done
        idx, a_, be, s, lo, h, tol = (z[keep] for z in (idx, a_, be, s_new, lo, h, tol))

    n_iter[idx] = max_iter  # Brent fallback over the (expected empty) remainder, D-011
    for j, aj, bj, lj, hj in zip(idx, a_, be, lo, h, strict=True):
        out[j] = _brent(aj, bj, lj, hj) / sqT[j]
        brent[j] = True

    # b(s) is only resolved where Phi(d2) at the root is a normal float (ndtr flushes to 0 near
    # d = -37.5, so b is too large by up to ~17x); reject such roots and s <= 0 (D-013)
    s_fin = out * sqT
    s1 = np.where(s_fin > 0, s_fin, 1.0)
    resolved = (s_fin > 0) & (ndtr(-a / s1 - 0.5 * s1) >= PHI_MIN)
    unres = (status == 0) & ~resolved
    status[unres], out[unres] = 5, np.nan

    bad = status != 0
    if bad.any():
        counts = {STATUS[c]: int((status == c).sum()) for c in np.unique(status[bad])}
        warnings.warn(
            f"implied_vol: {int(bad.sum())} of {bad.size} inputs -> NaN: {counts}",
            RuntimeWarning,
            stacklevel=2,
        )
    sigma = out.reshape(shape)[()]
    if not full_output:
        return sigma
    info = {"status": status.reshape(shape), "n_iter": n_iter.reshape(shape)}
    info["brent"] = brent.reshape(shape)
    return sigma, info
