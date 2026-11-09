"""Phase 4 Monte Carlo: barrier (Brownian bridge or discrete), Asian (geometric CV), autocallable.

model = sigma (float: BSM, exact steps between dates) or HestonParams (mc.heston_steps; QE-M by
default). State x_t = ln(F_t/F) as in mc.py; a fixing is S_t = F_0(t) e^{x_t} (heston-mc.md eq. 17).
Pricers build np.random.default_rng(seed) and return (price, stderr, n_paths, seed), estimator and
SE per heston-mc.md eq. (1). Equation numbers: docs/derivations/barrier.md, asian.md and
autocallable.md. References: Glasserman (2003) Monte Carlo Methods in Fin. Eng. (bridge, CV, CRN);
Shevchenko & Del Moral, arXiv:1405.5294 (bridge weight); Andersen (2008) (QE-M).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qpricer.engines import mc
from qpricer.engines.analytic import geometric_asian_price
from qpricer.models import bsm
from qpricer.models.heston import HestonParams
from qpricer.products import Autocall, autocall_flows, vanilla

MAX_DT = 1 / 32  # Heston sub-step cap between dates, asian eq. (4), autocallable eq. (14) (D-030)


def fixing_grid(t_obs: ArrayLike, max_dt: float | None = None) -> NDArray:
    """Grid containing every date: [t_{i-1}, t_i] split into m_i = ceil(tau_i / max_dt) equal steps
    (asian.md eq. 4, autocallable.md eq. 14). max_dt = None returns the dates themselves."""
    t = mc._check_grid(t_obs)
    if max_dt is None:
        return t
    if not max_dt > 0:
        raise ValueError("max_dt must be > 0")
    edges = np.concatenate(([0.0], t))
    m = np.ceil(np.diff(edges) / max_dt * (1.0 - 1e-12)).astype(int)
    ab = zip(edges[:-1], edges[1:], m, strict=True)
    return np.concatenate([np.linspace(a, b, k + 1)[1:] for a, b, k in ab])  # ends exactly on t_i


def obs_steps(
    model, t_obs: ArrayLike, n_paths: int, rng: np.random.Generator, max_dt=MAX_DT, scheme="qe-m"
) -> Iterator[tuple[float, NDArray, NDArray | float, NDArray]]:
    """Generator over the dates t_i: yields (t_i, x_i, v_i, z_i), fresh arrays at every date.

    BSM: x_i = x_{i-1} - sigma^2 tau_i/2 + sigma sqrt(tau_i) z_i, exact (barrier eq. 8, asian eq. 3,
    autocallable eq. 14); v_i = sigma^2; max_dt is ignored. Heston: mc.heston_steps on
    fixing_grid(t_obs, max_dt) (default MAX_DT = 1/32; None = one QE step per date interval, only
    for date grids that are already fine, as in barrier_mc), v_i the variance state, and the
    period's spot-Brownian increment
    z_i = sum_k sqrt(dt_k / tau_i) (rho Z_v,k + sqrt(1 - rho^2) Z_perp,k), exactly iid N(0, 1)
    across periods (common random numbers, autocallable eq. 15).
    """
    t = mc._check_grid(t_obs)
    if not (isinstance(rng, np.random.Generator) and int(n_paths) >= 2):
        raise ValueError("need an explicit np.random.Generator and n_paths >= 2")
    N = int(n_paths)
    if isinstance(model, HestonParams):
        grid = fixing_grid(t, max_dt)
        steps = mc.heston_steps(model, grid, N, rng, scheme)  # validates eagerly
        return _aggregate(steps, np.isin(grid, t), model.rho)
    sigma = float(model)
    if not (np.isfinite(sigma) and sigma >= 0):
        raise ValueError("sigma must be finite and >= 0")

    def gen():
        x, t_prev = np.zeros(N), 0.0
        for ti in t:
            z, tau, t_prev = rng.standard_normal(N), ti - t_prev, ti
            x = x - 0.5 * sigma**2 * tau + sigma * np.sqrt(tau) * z
            yield float(ti), x, sigma**2, z

    return gen()


def _aggregate(steps, is_obs: NDArray, rho: float):
    zacc, t_prev, t_last = 0.0, 0.0, 0.0
    for (tn, x, v, z), obs in zip(steps, is_obs, strict=True):
        zacc = zacc + np.sqrt(tn - t_prev) * (rho * z[0] + np.sqrt(1.0 - rho**2) * z[1])
        t_prev = tn
        if obs:
            yield tn, x, v, zacc / np.sqrt(tn - t_last)
            zacc, t_last = 0.0, tn


def fixings(model, t_obs, n_paths, rng, max_dt=MAX_DT, scheme="qe-m") -> tuple[NDArray, NDArray]:
    """(x, z) stacked at the dates, each (n_dates, n_paths); see `obs_steps`."""
    out = [(x, z) for _, x, _, z in obs_steps(model, t_obs, n_paths, rng, max_dt, scheme)]
    return np.stack([o[0] for o in out]), np.stack([o[1] for o in out])


def gbm_paths(w: ArrayLike, z: ArrayLike) -> NDArray:
    """BS log-paths on given normals z (n, N): x_i = sum_{k<=i} (-dw_k/2 + sqrt(dw_k) z_k).

    w = total variances at the dates (flat sigma: sigma^2 t_i; Heston's expected variance for the
    term-structure model). With z from `obs_steps` under Heston this is the CRN BS path driven by
    Heston's spot Brownian motion (autocallable eqs. 14-15; asian eq. 17).
    """
    dw = np.diff(np.asarray(w, dtype=np.float64), prepend=0.0)
    if not np.all(np.isfinite(dw) & (dw >= 0)):
        raise ValueError("w must be finite, >= 0 and non-decreasing")
    return np.cumsum(-0.5 * dw[:, None] + np.sqrt(dw)[:, None] * np.asarray(z), axis=0)


def bridge_survival(ey0, ey1, s2dt) -> NDArray:
    """1 - p_i, the bridge no-crossing probability of one step, eq. (9), derivations/barrier.md.

    p_i = exp(-2 (eta y_i)^+ (eta y_{i+1})^+ / (sigma_i^2 dt)); ey = eta y (signed distances). At
    s2dt = 0: p = 1 if an endpoint is at or beyond H, else 0 (pitfall 3). Underflow to 0 is exact.
    """
    a, c = np.maximum(ey0, 0.0), np.maximum(ey1, 0.0)
    pos = np.asarray(s2dt) > 0
    p = np.exp(-2.0 * a * c / np.where(pos, s2dt, 1.0))
    return 1.0 - np.where(pos, p, np.where(a * c > 0, 0.0, 1.0))


def barrier_mc(
    F, K, H, T, DF, model, cp, eta, knock, *, n_steps, n_paths, seed, b=0.0, monitor=None,
    scheme="qe-m",
):  # fmt: skip
    """Barrier option by MC on spot S_t = F_t e^{-b(T-t)}: (price, stderr, n_paths, seed).

    Grid t_i = i T/M (M = n_steps); y_i = ln(F/H) - b(T - t_i) + x_i, alive iff eta y_i > 0 (eq. 8).
    monitor = None (continuous): w = prod_i (1 - p_i), eqs. (9)-(10), with sigma_i^2 = sigma^2 (BSM,
    exact for every M) or v_i^+ (Heston, left point; approximate, D-028). monitor = m (M % m == 0):
    w = prod_j 1{eta y(t_j) > 0} at t_j = j T/m, no bridge, t = 0 not monitored (eq. 13).
    knock "out": g w, "in": g (1 - w), "vanilla": g, g = (cp (F e^{x_M} - K))^+; one path set per
    seed, so in + out = vanilla holds to round-off across separate calls (§5). Rebate R = 0.
    """
    K = np.asarray(K, dtype=np.float64)
    bsm.check_market(np.asarray(F), K, np.asarray(T), np.asarray(DF), np.asarray(cp))
    if not (H > 0 and abs(eta) == 1 and knock in ("out", "in", "vanilla")):
        raise ValueError("need H > 0, eta = +-1, knock in {'out', 'in', 'vanilla'}")
    if monitor is not None and not (monitor >= 1 and n_steps % monitor == 0):
        raise ValueError("monitor must divide n_steps")
    t, dt = mc.uniform_grid(T, n_steps), T / n_steps
    every = None if monitor is None else n_steps // monitor
    v = model.v0 if isinstance(model, HestonParams) else float(model) ** 2  # left-point variance
    ey, w = eta * (np.log(F / H) - b * T), 1.0
    steps = obs_steps(model, t, n_paths, np.random.default_rng(seed), None, scheme)
    for i, (ti, x, vi, _) in enumerate(steps):
        ey_new = eta * (np.log(F / H) - b * (T - ti) + x)
        if every is None:
            w = w * bridge_survival(ey, ey_new, np.maximum(v, 0.0) * dt)
        elif (i + 1) % every == 0:
            w = w * (ey_new > 0)
        ey, v = ey_new, vi
    g = vanilla(F * np.exp(x), K[..., None], cp)
    weight = {"out": w, "in": 1.0 - w, "vanilla": 1.0}[knock]
    price, se = mc.mc_estimate(g * weight, DF)
    return price[()], se[()], int(n_paths), int(seed)


def asian_averages(t_fix, F_fix, model, n_paths, rng, max_dt=MAX_DT, scheme="qe-m"):
    """(A, G) per path from running sums, eq. (5), derivations/asian.md (memory O(N)).

    Sigma_A += F_k e^{x_k}, Sigma_L += x_k; A = Sigma_A/n, G = Gbar_F exp(Sigma_L/n) (log-sum, no
    product overflow, pitfall 6). t_fix sorted, 0 <= t_1 < ... < t_n; a t = 0 fixing is today's
    F(0, 0) = S_0, supplied by the caller in F_fix (x = 0, dV = 0). Heston: at most max_dt = 1/32
    per QE-M step between fixings (eq. 4; one step per quarter is biased, D-030).
    """
    t, F_fix = np.asarray(t_fix, dtype=np.float64), np.asarray(F_fix, dtype=np.float64)
    if t.ndim != 1 or t.shape != F_fix.shape or not (t[0] >= 0 and np.all(F_fix > 0)):
        raise ValueError("t_fix, F_fix: equal 1-D shapes, t_1 >= 0, F_fix > 0")
    later = t > 0
    SA, SL = np.full(int(n_paths), F_fix[~later].sum()), 0.0
    steps = obs_steps(model, t[later], n_paths, rng, max_dt, scheme)
    for Fk, (_, x, _, _) in zip(F_fix[later], steps, strict=True):
        SA, SL = SA + Fk * np.exp(x), SL + x
    return SA / t.size, np.exp(np.mean(np.log(F_fix)) + SL / t.size)


def control_variate(y: ArrayLike, x: ArrayLike, mean_x: ArrayLike):
    """Regression control variate, eqs. (14)-(15), derivations/asian.md; paths on the last axis.

    e = y - c_hat (x - mean_x), c_hat = sum (y - ybar)(x - xbar) / sum (x - xbar)^2 (OLS on the
    same paths), SE = std(e, ddof=1)/sqrt(N); VR = s_y^2/s_e^2 = 1/(1 - rho_hat^2); bias O(1/N)
    (eq. 16). Returns (estimate, stderr, c_hat, VR). Raises if the control has zero sample variance.
    """
    y, x = np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64)
    xc, yc = x - x.mean(-1, keepdims=True), y - y.mean(-1, keepdims=True)
    sxx = (xc * xc).sum(-1)
    if np.any(sxx == 0):
        raise ValueError("control variate has zero sample variance")
    c = (xc * yc).sum(-1) / sxx
    e = y - c[..., None] * (x - np.asarray(mean_x)[..., None])
    est, se = mc.mc_estimate(e)
    return est, se, c, y.var(-1, ddof=1) / e.var(-1, ddof=1)


def asian_mc(
    t_fix, F_fix, K, DF, model, cp, *, n_paths, seed, kind="cv", max_dt=MAX_DT, scheme="qe-m"
):
    """Discrete Asian by MC: (price, stderr, n_paths, seed), derivations/asian.md §2.2-2.4.

    kind "arithmetic": Y = DF (cp (A - K))^+; "geometric": X = DF (cp (G - K))^+; "cv": arithmetic
    with the geometric control and its exact mean (10) (eq. 14). "cv" is BSM only: under Heston the
    Black V_G is not the mean of X (§3.5), so it raises. K may be an array (common paths).
    """
    K = np.asarray(K, dtype=np.float64)
    if kind not in ("arithmetic", "geometric", "cv"):
        raise ValueError("kind must be 'arithmetic', 'geometric' or 'cv'")
    if kind == "cv" and isinstance(model, HestonParams):
        raise ValueError("geometric CV mean is Black-only; use plain MC under Heston (asian §3.5)")
    rng = np.random.default_rng(seed)
    A, G = asian_averages(t_fix, F_fix, model, n_paths, rng, max_dt, scheme)
    Y, X = DF * vanilla(A, K[..., None], cp), DF * vanilla(G, K[..., None], cp)
    if kind == "cv":
        mean_x = geometric_asian_price(F_fix, float(model) ** 2 * np.asarray(t_fix), K, DF, cp)
        price, se, _, _ = control_variate(Y, X, mean_x)
    else:
        price, se = mc.mc_estimate(Y if kind == "arithmetic" else X)
    return price[()], se[()], int(n_paths), int(seed)


def autocall_pv(X: ArrayLike, sheet: Autocall, DF: ArrayLike) -> NDArray:
    """Path PV = sum_i DF_i (C_i + R_i), eq. (4), derivations/autocallable.md (no credit spread)."""
    C, R, _, _ = autocall_flows(X, sheet)
    return np.asarray(DF, dtype=np.float64) @ (C + R)


def autocall_mc(sheet: Autocall, f, DF, model, *, n_paths, seed, max_dt=MAX_DT, scheme="qe-m"):
    """Autocallable by MC: (price, stderr, n_paths, seed), eqs. (4), (14), (16), autocallable.md.

    X_i = f_i e^{x_{t_i}}, f_i = F_0(t_i)/S_ref, DF_i = DF(t_i). BSM: exact, one step per period;
    Heston: QE-M with ceil(tau_i / max_dt) equal sub-steps per period ending on t_i (eq. 14).
    """
    x, _ = fixings(model, sheet.t, n_paths, np.random.default_rng(seed), max_dt, scheme)
    price, se = mc.mc_estimate(autocall_pv(np.asarray(f)[:, None] * np.exp(x), sheet, DF))
    return float(price), float(se), int(n_paths), int(seed)
