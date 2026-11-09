"""Phase 4 closed forms, forward-based (F, DF, T; no r or q inside a kernel, D-001).

Barrier: Reiner-Rubinstein (1991) in forward form with a spot barrier S_t = F_t e^{-b(T-t)},
eqs. (1)-(4) of docs/derivations/barrier.md, and the Broadie-Glasserman-Kou (1997) shift (12).
Geometric Asian, discrete fixings (Kemna-Vorst 1990): eqs. (7)-(10) of derivations/asian.md.
Autocallable Black-76 / Heston-COS oracles: eqs. (7)-(12) of derivations/autocallable.md.
References: Haug (2007) Complete Guide, 2nd ed., §4.17 (via NAG S30FAF / QuantLib); BGK (1997)
Math. Finance 7(4); Kou (2003) Statistica Sinica 13; Fusai-Meucci (2008) JBF 32(10).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad
from scipy.special import log_ndtr, ndtr

from qpricer.engines.fourier import cos_price
from qpricer.models import bsm
from qpricer.models.heston import HestonParams

BGK_BETA = 0.5825971579390107  # -zeta(1/2) / sqrt(2 pi), barrier.md eq. (12)


def barrier_price(F, K, H, T, DF, sigma, cp: int, eta: int, knock: str, b=0.0, R=0.0):
    """Reiner-Rubinstein barrier, continuous monitoring, eqs. (1)-(4), derivations/barrier.md.

    cp = theta (+1 call, -1 put); eta = +1 down (S_0 > H), -1 up; knock "in" or "out". Barrier on
    spot S_t = F_t e^{-b(T-t)} (flat carry b, default 0 = barrier on the T-forward). Eq. (1):
    rho = (H/F) e^{bT}, mu = b/sigma^2 - 1/2, lambda = sqrt(mu^2 + 2r/sigma^2), r = -ln(DF)/T
    (rebate leg only), F* = rho^2 F. Eq. (2): G(X, L; k) = theta DF [X Phi(k d1) - K Phi(k d2)],
    d_{1,2} = d_{1,2}(X, L); A = G(F, K; theta), B = G(F, H; theta), C = rho^{2 mu} G(F*, K; eta),
    D = rho^{2 mu} G(F*, H; eta),
    E = R DF [Phi(eta d2(F, H)) - rho^{2 mu} Phi(eta d2(F*, H))], z = ln(rho)/s + lambda s,
    Fr = R [rho^{mu+lambda} Phi(eta z) + rho^{mu-lambda} Phi(eta z - 2 eta lambda s)].
    Table (3), with far := theta (K - H) >= 0 and the sign theta eta:
      in  = E  + {+1: C if far else A - B + D;      -1: A if far else B - C + D}
      out = Fr + {+1: A - C if far else B - D;      -1: 0 if far else A - B + C - D}
    Eq. (4): breached barrier (eta ln rho >= 0) gives out = R, in = A. Out-rebates pay at the hit,
    in-rebates at T. Powers rho^k Phi(.) are exp(k ln rho + log_ndtr(.)) (pitfall 8); R > 0 with
    lambda^2 < 0 (negative rate) raises (pitfall 9). Broadcasts over F, K, H, T, DF, sigma, b.
    """
    F, K, H, T, DF, sigma, b = np.broadcast_arrays(
        *(np.asarray(a, dtype=np.float64) for a in (F, K, H, T, DF, sigma, b))
    )
    bsm.check_market(F, K, T, DF, np.asarray(cp))
    if not (np.all(np.isfinite(H) & (H > 0)) and np.all(np.isfinite(sigma) & (sigma > 0))):
        raise ValueError("need finite H > 0 and sigma > 0")
    if abs(eta) != 1 or knock not in ("in", "out") or not (np.isfinite(R) and R >= 0):
        raise ValueError("eta must be +-1, knock 'in' or 'out', R finite and >= 0")
    s, lnr = sigma * np.sqrt(T), np.log(H / F) + b * T
    mu = b / sigma**2 - 0.5
    lnFs = np.log(F) + 2.0 * lnr  # ln F*

    def G(lnw, lnX, L, k):  # e^{lnw} G(X, L; k)
        d1 = (lnX - np.log(L)) / s + 0.5 * s
        up, dn = np.exp(lnw + lnX + log_ndtr(k * d1)), K * np.exp(lnw + log_ndtr(k * d1 - k * s))
        return cp * DF * (up - dn)

    with np.errstate(over="ignore", invalid="ignore"):  # breached cells are replaced by eq. (4)
        A, B = G(0.0, np.log(F), K, cp), G(0.0, np.log(F), H, cp)
        C, D = G(2 * mu * lnr, lnFs, K, eta), G(2 * mu * lnr, lnFs, H, eta)
        far = cp * (K - H) >= 0
        if cp * eta > 0:
            core = np.where(far, C, A - B + D) if knock == "in" else np.where(far, A - C, B - D)
        else:
            core = (
                np.where(far, A, B - C + D) if knock == "in" else np.where(far, 0.0, A - B + C - D)
            )
        if R > 0:
            lam2 = mu**2 - 2.0 * np.log(DF) / (T * sigma**2)
            if np.any(lam2 < 0):
                raise ValueError("rebate with lambda^2 < 0 (DF > 1): F term undefined (pitfall 9)")
            lam = np.sqrt(lam2)
            d2h, d2s = (np.log(F / H) / s - 0.5 * s), ((lnFs - np.log(H)) / s - 0.5 * s)
            if knock == "in":
                ex = np.exp(log_ndtr(eta * d2h)) - np.exp(2 * mu * lnr + log_ndtr(eta * d2s))
                core = core + R * DF * ex
            else:
                z = lnr / s + lam * s
                ex = np.exp((mu + lam) * lnr + log_ndtr(eta * z))
                core = core + R * (
                    ex + np.exp((mu - lam) * lnr + log_ndtr(eta * (z - 2 * lam * s)))
                )
    breached = eta * lnr >= 0
    if not np.all(np.isfinite(core) | breached):
        raise FloatingPointError("non-finite Reiner-Rubinstein value inside the domain")
    return (np.where(breached, A if knock == "in" else R, core) + 0.0)[()]


def bgk_barrier(H: ArrayLike, sigma: ArrayLike, dt: float, eta: int) -> NDArray:
    """BGK continuity correction, eq. (12), derivations/barrier.md: H e^{-eta beta sigma sqrt(dt)}.

    A discrete contract with date spacing dt (first date at dt) ~ the continuous one at the shifted
    barrier, moved away from S_0 (down barriers down, up barriers up), up to o(m^{-1/2}). BSM only.
    """
    return np.asarray(H, dtype=np.float64) * np.exp(
        -eta * BGK_BETA * np.asarray(sigma) * np.sqrt(dt)
    )


def geometric_asian_moments(F_fix: ArrayLike, V: ArrayLike) -> tuple[float, float]:
    """(F_G, sigma_G^2) of the discrete geometric average G, eqs. (7)-(9), derivations/asian.md.

    F_fix = fixing forwards F(0, t_i); V = total variances V(t_i) on sorted dates (BSM sigma^2 t_i,
    V = 0 for a t = 0 fixing). sigma_G^2 = sum_k w_k^2 dV_k, w_k = (n - k + 1)/n (eq. 8, Abel form);
    F_G = Gbar_F exp(-(Vbar - sigma_G^2)/2), Gbar_F = (prod F_i)^{1/n} (eq. 9).
    """
    F_fix, V = np.asarray(F_fix, dtype=np.float64), np.asarray(V, dtype=np.float64)
    if F_fix.ndim != 1 or F_fix.shape != V.shape or not np.all(F_fix > 0):
        raise ValueError("F_fix and V must be 1-D of equal length, F_fix > 0")
    dV = np.diff(V, prepend=0.0)
    if not (np.all(np.isfinite(V)) and np.all(dV >= 0)):
        raise ValueError("V must be finite, >= 0 and non-decreasing (sorted dates)")
    n = V.size
    s2 = float(np.sum(((n - np.arange(n)) / n) ** 2 * dV))
    return float(np.exp(np.mean(np.log(F_fix)) - 0.5 * (V.mean() - s2))), s2


def geometric_asian_price(F_fix: ArrayLike, V: ArrayLike, K: ArrayLike, DF: float, cp=1):
    """Discrete geometric Asian (Kemna-Vorst), eq. (10), derivations/asian.md.

    V_G = DF cp [F_G Phi(cp d1) - K Phi(cp d2)], d_{1,2} = ln(F_G/K)/sigma_G +- sigma_G/2: Black-76
    (bsm.md eq. 1) with F -> F_G, sigma sqrt(T) -> sigma_G (called with T = 1). Parity
    C_G - P_G = DF (F_G - K). sigma_G = 0 uses the D-012 limits.
    """
    FG, s2 = geometric_asian_moments(F_fix, V)
    return bsm.price(FG, K, 1.0, DF, np.sqrt(s2), cp)


def _dminus(f, L, s):
    """d_-(f, L, s) = [ln(f/L) - s^2/2] / s, autocallable.md §1 notation (s > 0)."""
    return (np.log(f / L) - 0.5 * s * s) / s


def ratio_marginals(f: ArrayLike, t: ArrayLike, k: float, model) -> tuple[NDArray, NDArray]:
    """(p_i(k), D^-_i(k)) at each date: undiscounted ratio put E(k - X_i)^+ and Q(X_i < k).

    X_i = f_i M_{t_i}. Black-76 (model = total variances w_i, array): Black put with forward f_i,
    total vol sqrt(w_i), and D^- = Phi(-d_-(f_i, k, sqrt(w_i))), autocallable.md eq. (8). Heston
    (model = HestonParams): self-checking COS put (forward f_i, expiry t_i, DF = 1), D^- = dp/dk by
    a central difference on one COS call at [k - h, k, k + h], h = 1e-4 k (one (N, L) for the three
    strikes, so the difference is the derivative of one series), §2.3.
    """
    f, t = np.asarray(f, dtype=np.float64), np.asarray(t, dtype=np.float64)
    if isinstance(model, HestonParams):
        h = 1e-4 * k
        P = np.array(
            [
                cos_price(fi, [k - h, k, k + h], ti, 1.0, model, -1)
                for fi, ti in zip(f, t, strict=True)
            ]
        )
        return P[:, 1], (P[:, 2] - P[:, 0]) / (2 * h)
    s = np.sqrt(np.asarray(model, dtype=np.float64))
    return bsm.price(f, k, 1.0, 1.0, s, -1), ndtr(-_dminus(f, k, s))


def autocall_nocall(sheet, f: ArrayLike, DF: ArrayLike, model) -> float:
    """No-call, no-memory value of the sheet's (t, c, B_c, B_p), eqs. (7), (8), (11),
    autocallable.md.

    V^nc = sum_i DF_i c_i D^+_i(B_c) + DF_n - DIP_E, DIP_E = DF_n [p_n(B_p) + (1 - B_p) D^-_n(B_p)]
    (eq. 7, static: a put plus (1 - B_p) digital puts at B_p). B_ac and memory are ignored by
    construction. Marginals from `ratio_marginals` (Black total variances w_i, or Heston COS).
    """
    t, DF = np.asarray(sheet.t), np.asarray(DF, dtype=np.float64)
    _, Dc = ratio_marginals(f, t, sheet.B_c, model) if sheet.B_c > 0 else (0, np.zeros(t.size))
    dip = 0.0
    if sheet.B_p > 0:
        p, Dp = ratio_marginals(f, t, sheet.B_p, model)
        dip = DF[-1] * (p[-1] + (1.0 - sheet.B_p) * Dp[-1])
    return float(np.sum(DF * sheet.coupons * (1.0 - Dc)) + DF[-1] - dip)


def autocall_two_dates(sheet, f: ArrayLike, DF: ArrayLike, w: ArrayLike) -> float:
    """n = 2 Black-76 value, any memory flag, B_c < B_ac, eq. (12), derivations/autocallable.md.

    V = DF_1 [c_1 Phi(d_-(f_1, B_c, s_1)) + Phi(d_-(f_1, B_ac, s_1))]
        + DF_2 int_{-inf}^{z_ac} phi(z) h(x(z)) dz,     z_L = -d_-(f_1, L, s_1),
    x(z) = f_1 e^{s_1 z - s_1^2/2}, g = x f_2/f_1, s_12^2 = w_2 - w_1, d_+ = d_- + s_12,
    h = (c_2 + m c_1 1{x < B_c}) Phi(d_-(g, B_c, s_12)) + 1 - Phi(-d_-(g, B_p, s_12))
        + g Phi(-d_+(g, B_p, s_12)).
    quad (epsabs 1e-13) on (-inf, z_c) and (z_c, z_ac) separately (memory jump at z_c).
    """
    f, DF, w = (np.asarray(a, dtype=np.float64) for a in (f, DF, w))
    if len(sheet.t) != 2 or not (0 < sheet.B_p and sheet.B_c < sheet.B_ac and w[1] > w[0] > 0):
        raise ValueError("eq. (12) needs n = 2, 0 < B_p, B_c < B_ac and 0 < w_1 < w_2")
    (c1, c2), m = sheet.coupons, float(sheet.memory)
    s1, s12 = np.sqrt(w[0]), np.sqrt(w[1] - w[0])

    def h(z: float) -> float:
        g = f[1] * np.exp(s1 * z - 0.5 * w[0])
        cpn = c2 + m * c1 * (g * f[0] / f[1] < sheet.B_c)
        dp = _dminus(g, sheet.B_p, s12)
        cap = 1.0 - ndtr(-dp) + g * ndtr(-dp - s12)
        return (
            np.exp(-0.5 * z * z)
            / np.sqrt(2 * np.pi)
            * (cpn * ndtr(_dminus(g, sheet.B_c, s12)) + cap)
        )

    zc, zac = -_dminus(f[0], sheet.B_c, s1), -_dminus(f[0], sheet.B_ac, s1)
    integral = sum(
        quad(h, lo, hi, epsabs=1e-13, limit=200)[0] for lo, hi in ((-np.inf, zc), (zc, zac))
    )
    first = DF[0] * (c1 * ndtr(-zc) + ndtr(-zac))
    return float(first + DF[1] * integral)
