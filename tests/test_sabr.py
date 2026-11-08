"""Phase 3 SABR (beta = 1): Hagan formula identities, MC exact properties, Hagan-vs-MC 3SE gate.

docs/ROADMAP.md Phase 3 and docs/derivations/sabr.md §5 (oracles (i)-(vi), MC (a)-(f), gate set
G). MC comparisons are in standard-error units with seeds fixed a priori, never re-seeded
(heston-mc.md eq. 2). Prices are puts (bounded payoff, pitfall 8); gate in price space (19).
"""

import warnings

import numpy as np
import pytest
from scipy.integrate import quad

from qpricer.engines.mc import mc_estimate
from qpricer.models import bsm, sabr

SEED = 20261107
F0, ALPHA, RHO, NU = 100.0, 0.2, -0.75, 0.4  # gate set G, sabr.md §5
K_GATE = F0 * np.exp(np.array([-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2]))
T_GATE = (0.25, 0.5, 1.0)
M_GATE, N_GATE = 250, 200_000  # dt = T/250 <= min(1/250, 0.5 / (nu^2 sqrt N)), eq. (18), D-023
RHOS = (-0.999, -0.75, -0.3, 0.0, 0.4, 0.999)


def n_steps(T: float) -> int:
    """250 steps at every gate maturity (validator F1): the (14) wing bias at dt = 1/250 is
    0.64 SE_gate at T = 0.25; at dt = T/250 it is <= 0.17 SE_gate (D-023)."""
    return M_GATE


def hagan_put(K, T, alpha=ALPHA, rho=RHO, nu=NU):
    return bsm.price(F0, K, T, 1.0, sabr.implied_vol(F0, K, T, alpha, rho, nu), -1)


def hagan_general_beta(F, K, T, alpha, beta, rho, nu):
    """Eq. (1), [H02] (2.17a-c), written out independently (direct log; use only |z| >~ 1e-2)."""
    fk, ell = (F * K) ** ((1 - beta) / 2), np.log(F / K)
    z = nu / alpha * fk * ell
    x = np.log((np.sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho))
    den = fk * (1 + (1 - beta) ** 2 / 24 * ell**2 + (1 - beta) ** 4 / 1920 * ell**4)
    corr = (1 - beta) ** 2 * alpha**2 / (24 * fk**2) + rho * beta * nu * alpha / (4 * fk)
    return alpha / den * z / x * (1 + (corr + (2 - 3 * rho**2) * nu**2 / 24) * T)


# ------------------------------------------------------------------ Hagan formula, exact oracles
def test_gate_nu_to_zero_flat_vol():
    """Gate: nu -> 0, beta = 1 -> sigma_H = alpha (< 1e-10); eq. (7), oracle (i)."""
    k = np.linspace(-1.0, 1.0, 201)
    T = np.array([0.01, 0.25, 1.0, 5.0, 10.0])[:, None]
    for rho in RHOS:
        for alpha in (0.05, 0.2, 1.0):
            assert np.array_equal(
                sabr.implied_vol(1.0, np.exp(k), T, alpha, rho, 0.0), alpha + 0 * T * k
            )
            s = sabr.implied_vol(1.0, np.exp(k), T, alpha, rho, 1e-12)
            assert np.max(np.abs(s - alpha)) < 1e-10


def test_nu_slope_matches_eq7():
    """(sigma_H - alpha) / nu at nu = 1e-7 equals rho (k/2 + alpha^2 T / 4) to rel 1e-4, eq. (7)."""
    k, T = np.array([-0.5, -0.1, 0.1, 0.5]), np.array([0.5, 2.0])[:, None]
    for rho in (-0.75, 0.4):
        slope = (sabr.implied_vol(1.0, np.exp(k), T, 0.2, rho, 1e-7) - 0.2) / 1e-7
        np.testing.assert_allclose(slope, rho * (k / 2 + 0.04 * T / 4), rtol=1e-4)


def test_atm_continuity_skew_and_curvature():
    """Oracle (ii): K = F continuity, ATM skew and curvature of eq. (8), jump at |z| = z_c."""
    T = 1.0
    for rho in (-0.75, 0.0, 0.5):
        b = 1 + (rho * NU * ALPHA / 4 + (2 - 3 * rho**2) * NU**2 / 24) * T
        atm = sabr.implied_vol(F0, F0, T, ALPHA, rho, NU)
        assert atm == ALPHA * b  # eq. (4)
        skew, curv = 0.5 * rho * NU * b, (2 - 3 * rho**2) * NU**2 / (6 * ALPHA) * b
        for h in (1e-10, 1e-8, 1e-6, 1e-4):
            s = sabr.implied_vol(F0, F0 * np.exp([h, -h]), T, ALPHA, rho, NU)
            # second-order Taylor bound (the note's (ii) omits the h^2 term, OPEN_ISSUES)
            assert np.all(np.abs(s - atm) <= abs(skew) * h + 0.5 * abs(curv) * h**2 + 1e-12 * atm)
        h = 1e-4
        sp, sm = sabr.implied_vol(F0, F0 * np.exp([h, -h]), T, ALPHA, rho, NU)
        if rho != 0:
            assert abs((sp - sm) / (2 * h) / skew - 1) < 1e-6
        assert abs((sp - 2 * atm + sm) / h**2 / curv - 1) < 1e-5
        kc = sabr.Z_C * ALPHA / NU  # |z| = z_c
        for sgn in (1, -1):
            k = sgn * kc * np.array([1 - 1e-9, 1 + 1e-9])
            s = sabr.implied_vol(F0, F0 * np.exp(k), T, ALPHA, rho, NU)
            assert abs(s[1] - s[0]) <= 2 * abs(skew) * 2e-9 * kc + 1e-12 * atm


def test_symmetry_and_rho_offset_identity():
    """Oracle (iii), §3.3: rho = 0 even in k; for rho != 0 the exact offset identity holds and the
    plain mirror symmetry sigma(k; rho) = sigma(-k; -rho) does not."""
    k = np.linspace(-1.5, 1.5, 61)
    T = np.array([0.1, 1.0, 5.0])[:, None]
    s0 = sabr.implied_vol(F0, F0 * np.exp(k), T, ALPHA, 0.0, NU)
    np.testing.assert_allclose(
        s0, sabr.implied_vol(F0, F0 * np.exp(-k), T, ALPHA, 0.0, NU), rtol=1e-14
    )
    for rho in (-0.999, -0.75, 0.3, 0.999):
        s = sabr.implied_vol(F0, F0 * np.exp(k), T, ALPHA, rho, NU)
        s_m = sabr.implied_vol(F0, F0 * np.exp(-k), T, ALPHA, -rho, NU)
        zx = sabr.z_over_x(-NU * k / ALPHA, rho)
        assert np.all(np.abs(s - s_m - ALPHA * zx * rho * NU * ALPHA * T / 2) <= 1e-14 * s)
        assert np.min(np.abs(s - s_m)) > 1e-5  # negative control: plain symmetry fails


def test_z_over_x_matches_integral_definition():
    """Eq. (3): x(z) = int_0^z du / D(u) by adaptive quadrature, incl. rho = +-0.999 (pitfall 2)."""
    for rho in RHOS:
        for z in (-8.0, -1.5, -0.999, -0.3, -2e-3, -5e-4, 5e-4, 2e-3, 0.3, 0.999, 1.5, 8.0):
            pts = [rho] if min(0, z) < rho < max(0, z) else None
            x, _ = quad(
                lambda u, r=rho: 1 / np.sqrt(1 - 2 * r * u + u * u),
                0,
                z,
                points=pts,
                epsabs=0,
                epsrel=1e-13,
                limit=200,
            )
            assert abs(sabr.z_over_x(z, rho) / (z / x) - 1) < 1e-11, (rho, z)


def test_beta_one_equals_general_formula():
    """Oracle (iv): eq. (1) at beta = 1 equals (2); at beta = 1 - 1e-6 differs by O(1e-6)."""
    k = np.concatenate([np.linspace(-1.0, -0.05, 20), np.linspace(0.05, 1.0, 20)])
    for rho in (-0.9, -0.75, 0.0, 0.5):
        for T in (0.25, 2.0):
            s2 = sabr.implied_vol(1.0, np.exp(k), T, ALPHA, rho, NU)
            np.testing.assert_allclose(
                hagan_general_beta(1.0, np.exp(k), T, ALPHA, 1.0, rho, NU), s2, rtol=1e-12
            )
            s1 = hagan_general_beta(1.0, np.exp(k), T, ALPHA, 1 - 1e-6, rho, NU)
            assert np.max(np.abs(s1 - s2) / s2) < 1e-5


def test_scale_free_and_vectorised():
    """Oracle (v): no F scale at beta = 1; broadcasting over T (column) x K (row) = scalar calls."""
    K, T = F0 * np.exp(np.linspace(-0.8, 0.6, 15)), np.array([0.1, 1.0, 3.0])[:, None]
    s = sabr.implied_vol(F0, K, T, ALPHA, RHO, NU)
    assert s.shape == (3, 15)
    for lam in (1e-3, 37.5, 1e4):
        np.testing.assert_allclose(
            sabr.implied_vol(lam * F0, lam * K, T, ALPHA, RHO, NU), s, rtol=1e-12
        )
    loop = [[sabr.implied_vol(F0, k, t, ALPHA, RHO, NU) for k in K] for t in T[:, 0]]
    np.testing.assert_array_equal(np.array(loop), s)


def test_alpha_from_atm_round_trip_and_flag():
    """Oracle (vi): eq. (9) inverts sigma_ATM(alpha) to 1e-14; no positive root -> NaN + warning."""
    alpha = np.array([0.05, 0.2, 0.6])[:, None, None]
    T, nu = np.array([0.02, 0.5, 2.0, 5.0])[None, :, None], np.array([0.0, 0.4, 1.5])[None, None, :]
    for rho in (-0.9, -0.5, 0.0, 0.7):
        with warnings.catch_warnings():  # NaN cells of eq. (22) are excluded; warning tested below
            warnings.simplefilter("ignore", RuntimeWarning)
            atm = sabr.implied_vol(1.0, 1.0, T, alpha, rho, nu)
        a_b, T_b, nu_b = np.broadcast_arrays(alpha, T, nu)
        ok = np.isfinite(atm)
        back = sabr.alpha_from_atm(atm[ok], T_b[ok], rho, nu_b[ok])
        np.testing.assert_allclose(back, a_b[ok], rtol=1e-14)
    with pytest.warns(RuntimeWarning):  # rho = -0.9, nu = 3, T = 10: b < 0, no positive root
        assert np.isnan(sabr.alpha_from_atm(0.2, 10.0, -0.9, 3.0))


def test_nonpositive_bracket_returns_nan_with_warning():
    """Eq. (22), pitfall 3: 1 + c1 T <= 0 (rho = -0.9, alpha = 0.5, nu = 1, T >= 7.7) -> NaN."""
    c1 = -0.9 * 0.5 / 4 + (2 - 3 * 0.81) / 24
    T = np.array([1.0, 5.0, 10.0])
    with pytest.warns(RuntimeWarning):
        s = sabr.implied_vol(1.0, 1.1, T, 0.5, -0.9, 1.0)
    assert np.all(np.isfinite(s[:2])) and np.isnan(s[2]) and 1 + c1 * 10 <= 0 < 1 + c1 * 5


def test_input_validation():
    for args in (
        (100, -1, 1, 0.2, 0, 0.4),
        (100, 100, 0, 0.2, 0, 0.4),
        (100, 100, 1, 0, 0, 0.4),
        (100, 100, 1, 0.2, 0.9995, 0.4),
        (100, 100, 1, 0.2, 0, -0.1),
        (100, 100, 1, 0.2, np.nan, 0.4),
        (np.inf, 100, 1, 0.2, 0, 0.4),
    ):
        with pytest.raises(ValueError):
            sabr.implied_vol(*args)
    with pytest.raises(ValueError):
        sabr.mc_price(100, 100, 1, 1, 0.2, 0, 0.4, -1, n_steps=4, n_paths=100, seed=1, method="x")
    with pytest.raises(ValueError):
        sabr.mc_price(100, 100, 1, 1, 0.2, -1.0, 0.4, -1, n_steps=4, n_paths=100, seed=1)
    with pytest.raises(ValueError):  # validator F7: no silent truncation of n_steps
        sabr.mc_price(100, 100, 1, 1, 0.2, 0, 0.4, -1, n_steps=2.7, n_paths=100, seed=1)
    with pytest.raises(TypeError):  # cp is required, as in mc.european_price (D-021)
        sabr.mc_price(100, 100, 1, 1, 0.2, 0, 0.4, n_steps=4, n_paths=100, seed=1)


# ------------------------------------------------------------------ MC: API and exact properties
def test_mc_api_contract_reproducibility_and_parity():
    K = np.array([90.0, 100.0, 110.0])
    out = sabr.mc_price(F0, K, 1.0, 0.95, ALPHA, RHO, NU, -1, n_steps=8, n_paths=4000, seed=SEED)
    price, se, n, seed = out
    assert price.shape == se.shape == K.shape and n == 4000 and seed == SEED and np.all(se > 0)
    again = sabr.mc_price(F0, K, 1.0, 0.95, ALPHA, RHO, NU, -1, n_steps=8, n_paths=4000, seed=SEED)
    assert np.array_equal(price, again[0]) and np.array_equal(se, again[1])
    call = sabr.mc_price(F0, K, 1.0, 0.95, ALPHA, RHO, NU, 1, n_steps=8, n_paths=4000, seed=SEED)[0]
    rng = np.random.default_rng(SEED)
    x, *_ = sabr.simulate(ALPHA, RHO, NU, 1 / 8, (rng.standard_normal((2, 4000)) for _ in range(8)))
    np.testing.assert_allclose(call - price, 0.95 * (F0 * np.exp(x).mean() - K), atol=1e-11)


@pytest.mark.parametrize("method", ["euler", "mixing"])
def test_mc_nu_zero_is_black(method):
    """MC oracle (a): nu = 0 is Black(alpha) in law at any dt (4 steps), within 3 SE."""
    K = F0 * np.exp(np.linspace(-0.3, 0.3, 7))
    v, se, *_ = sabr.mc_price(
        F0,
        K,
        1.0,
        0.9,
        ALPHA,
        RHO,
        0.0,
        -1,
        n_steps=4,
        n_paths=100_000,
        seed=SEED + 1,
        method=method,
    )
    assert np.all(np.abs(v - bsm.price(F0, K, 1.0, 0.9, ALPHA, -1)) <= 3 * se)


def test_mc_vol_moments_and_left_point_variance():
    """MC oracle (b): mean alpha_T = alpha, mean alpha_T^2 = alpha^2 e^{nu^2 T}, mean I_left =
    alpha^2 dt (e^{nu^2 T} - 1) / (e^{nu^2 dt} - 1), eqs. (13), (16)."""
    T, M, N, nu = 1.0, 50, 100_000, 1.0
    rng, dt = np.random.default_rng(SEED + 2), T / M
    draws = (rng.standard_normal((2, N)) for _ in range(M))
    _, aT, _, il = sabr.simulate(ALPHA, RHO, nu, dt, draws)
    ref_i = ALPHA**2 * dt * np.expm1(nu**2 * T) / np.expm1(nu**2 * dt)
    for g, ref in ((aT, ALPHA), (aT**2, ALPHA**2 * np.exp(nu**2 * T)), (il, ref_i)):
        m, se = mc_estimate(g)
        assert abs(m - ref) <= 3 * se


def test_mc_martingale_at_rho_minus_09():
    """MC oracle (c): mean F_T = F (euler) and mean xi = 1 (mixing) at rho = -0.9 (E F^4 < inf)."""
    T, M, N = 1.0, 100, 100_000
    rng, dt = np.random.default_rng(SEED + 3), T / M
    draws = (rng.standard_normal((2, N)) for _ in range(M))
    x, aT, w, il = sabr.simulate(ALPHA, -0.9, NU, dt, draws)
    m, se = mc_estimate(F0 * np.exp(x))
    assert abs(m - F0) <= 3 * se
    iv = il + 0.5 * dt * (aT**2 - ALPHA**2)
    ln_xi = -0.9 * (aT - ALPHA) / NU - 0.5 * 0.81 * iv
    m, se = mc_estimate(np.exp(ln_xi))
    assert abs(m - 1.0) <= 3 * se
    assert np.all(ln_xi <= 0.9 * ALPHA / NU)  # D-023 (iv): conditional forward bounded, rho <= 0


# ------------------------------------------------------------------ Phase 3 gate: Hagan vs MC
def test_gate_hagan_vs_mc_within_3se():
    """Gate: |P_MC - Black(sigma_H)| <= 3 SE on set G (nu^2 T <= 0.16), T in {0.25, 0.5, 1},
    |k| <= 0.2, puts, N = 2e5, dt = T/250, scheme (14). 21 correlated comparisons with small
    deterministic offsets (mainly Hagan's wing error): 4.0% of 100 fresh seeds fail at dt = T/250
    (validator: 9.0% at dt = 1/250), scripts/phase3_gates.py --seeds 100 (D-023)."""
    for i, T in enumerate(T_GATE):
        v, se, *_ = sabr.mc_price(
            F0,
            K_GATE,
            T,
            1.0,
            ALPHA,
            RHO,
            NU,
            -1,
            n_steps=n_steps(T),
            n_paths=N_GATE,
            seed=SEED + i,
        )
        z = (v - hagan_put(K_GATE, T)) / se
        assert np.all(np.abs(z) <= 3), (T, z)


def test_euler_vs_mixing_independent_mc():
    """MC oracle (f): (14) and (17), independent seeds, within 3 sqrt(SE1^2 + SE2^2) at T = 1."""
    T = 1.0
    kw = dict(n_steps=n_steps(T), n_paths=N_GATE)
    v1, s1, *_ = sabr.mc_price(F0, K_GATE, T, 1.0, ALPHA, RHO, NU, -1, seed=SEED + 2, **kw)
    kw["method"] = "mixing"
    v2, s2, *_ = sabr.mc_price(F0, K_GATE, T, 1.0, ALPHA, RHO, NU, -1, seed=SEED + 12, **kw)
    assert np.all(np.abs(v1 - v2) <= 3 * np.hypot(s1, s2))


def halving_payoffs(T, M, N, seed):
    """Per-path put payoffs of (14) and (17) at dt = T/M and dt/2, common random numbers: the
    coarse increments are (Z_1 + Z_2) / sqrt 2 of the fine pair (MC oracle (e))."""

    def fine():
        rng = np.random.default_rng(seed)
        for _ in range(2 * M):
            yield rng.standard_normal((2, N))

    def coarse():
        rng = np.random.default_rng(seed)
        for _ in range(M):
            yield (rng.standard_normal((2, N)) + rng.standard_normal((2, N))) / np.sqrt(2)

    out = []
    for dt, draws in ((T / M, coarse()), (T / (2 * M), fine())):
        x, aT, w, il = sabr.simulate(ALPHA, RHO, NU, dt, draws)
        eul = np.maximum(K_GATE[:, None] - F0 * np.exp(x), 0.0)
        iv = il + 0.5 * dt * (aT**2 - ALPHA**2)
        lnf = np.log(F0) + RHO * (aT - ALPHA) / NU - 0.5 * RHO**2 * iv
        mix = bsm.price(np.exp(lnf), K_GATE[:, None], T, 1.0, np.sqrt((1 - RHO**2) * iv / T), -1)
        out.append((eul, mix))
    return out


@pytest.mark.slow
@pytest.mark.parametrize("T, seed", [(0.25, SEED + 21), (0.5, SEED + 22), (1.0, SEED + 20)])
def test_halving_dt(T, seed):
    """MC oracle (e) at the gate dt = T/250, every gate maturity. (17): |V_dt - V_dt/2| <= 3 SE of
    the CRN difference (bias O(dt^2)). (14): weak order 1, so CRN resolves its true bias
    (OPEN_ISSUES); the gate needs it small against the gate SE: |V_dt - V_dt/2| ~ bias/2 <=
    0.1 SE_gate, i.e. |bias| <~ 0.2 SE, so the per-comparison false-fail rate stays <= 0.33%
    (vs 0.27%), heston-mc.md eq. (3)."""
    (e_c, m_c), (e_f, m_f) = halving_payoffs(T, n_steps(T), N_GATE, seed)
    d, se_d = mc_estimate(m_c - m_f)
    assert np.all(np.abs(d) <= 3 * se_d)
    d, _ = mc_estimate(e_c - e_f)
    _, se_gate = mc_estimate(e_f)
    assert np.all(np.abs(d) <= 0.1 * se_gate)  # se_gate: put SE at N_GATE, as in the gate
