"""Tests for qpricer.models.bsm (Black-76), Phase 1 gates and the oracles of derivations/bsm.md §5.

Gates (docs/ROADMAP.md): put-call parity < 1e-12 at F = 100 on K/F in [0.25, 4] (pitfall 4);
Greeks vs central finite differences, relative error < 1e-5, steps per §2.4 / eq. (17)-(18)
(gamma step: D-014). The gate grid is the `bsm_gate_grid` fixture (conftest.py, §2.5).
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import erf
from scipy.stats import norm

from qpricer.models import bsm

EPS = np.finfo(np.float64).eps
SIG = np.linspace(0.05, 1.0, 40)[:, None, None, None]
TT = np.geomspace(0.02, 5.0, 30)[None, :, None, None]
DFS = np.array([0.8, 0.9, 1.0])[None, None, None, :]


def unpack(g):
    return g["F"], g["K"], g["T"], g["DF"], g["sigma"]


def rel_err(approx, exact):
    return np.abs(approx / exact - 1.0)


def assert_identity(a, b):
    """§5 tolerance for identities, per element: abs <= 1e-12 (F = 100) or rel <= 1e-12."""
    err = np.abs(a - b)
    assert np.all((err <= 1e-12) | (err <= 1e-12 * np.abs(b))), err.max()


def d12(F, K, T, sigma):
    s = sigma * np.sqrt(T)
    return np.log(F / K) / s + s / 2, np.log(F / K) / s - s / 2


# ---------------------------------------------------------------- gate: put-call parity (2)
def test_parity_gate():
    """|C - P - DF (F - K)| < 1e-12, C and P evaluated independently (pitfall 4)."""
    K = 100.0 * np.geomspace(0.25, 4.0, 61)[None, None, :, None]
    F, K, T, DF, sig = np.broadcast_arrays(100.0, K, TT, DFS, SIG)
    err = np.abs(bsm.price(F, K, T, DF, sig, 1) - bsm.price(F, K, T, DF, sig, -1) - DF * (F - K))
    assert err.max() < 1e-12


# ---------------------------------------------------------------- gate: Greeks vs central FD
@pytest.mark.parametrize("cp", [1, -1])
def test_delta_fd_gate(bsm_gate_grid, cp):
    """Delta (5) vs D_1 in F, h = 1e-5 F sigma sqrt T, h_eff = (F+h) - (F-h) (§2.4)."""
    F, K, T, DF, sig = unpack(bsm_gate_grid)
    h = 1e-5 * F * sig * np.sqrt(T)
    up, dn = F + h, F - h
    fd = (bsm.price(up, K, T, DF, sig, cp) - bsm.price(dn, K, T, DF, sig, cp)) / (up - dn)
    assert rel_err(fd, bsm.delta(F, K, T, DF, sig, cp)).max() < 1e-5


def test_gamma_fd_gate(bsm_gate_grid):
    """Gamma (6) vs D_2 in F on the OTM option (pitfall 3; gamma is the same for C and P).

    Step h = 4.5e-4 F sigma sqrt T: the minimax step of eq. (17) given the kernel's rounding
    noise eps F Phi(cp d1) (pitfall 6), not eps V; the §2.4 table value 1e-4 gives 1.8e-5 (D-014).
    """
    F, K, T, DF, sig = unpack(bsm_gate_grid)
    cp = np.where(np.log(F / K) > 0, -1.0, 1.0)  # OTM: put if F > K, else call
    h = (F + 4.5e-4 * F * sig * np.sqrt(T)) - F
    f = [bsm.price(F + k * h, K, T, DF, sig, cp) for k in (-1, 0, 1)]
    fd = (f[2] - 2.0 * f[1] + f[0]) / h**2
    assert rel_err(fd, bsm.gamma(F, K, T, DF, sig, cp)).max() < 1e-5


@pytest.mark.parametrize("cp", [1, -1])
def test_vega_fd_gate(bsm_gate_grid, cp):
    """Vega (7) vs D_1 in sigma, h = 1e-5 sigma (§2.4), ITM and OTM."""
    F, K, T, DF, sig = unpack(bsm_gate_grid)
    up, dn = sig * (1 + 1e-5), sig * (1 - 1e-5)
    fd = (bsm.price(F, K, T, DF, up, cp) - bsm.price(F, K, T, DF, dn, cp)) / (up - dn)
    assert rel_err(fd, bsm.vega(F, K, T, DF, sig, cp)).max() < 1e-5


def test_theta_fd_gate(bsm_gate_grid):
    """Theta (8) vs the joint (T, DF) bump of eq. (18), h = 1e-5 T, on OTM options (pitfall 11)."""
    F, K, T, DF, sig = unpack(bsm_gate_grid)
    cp = np.where(np.log(F / K) > 0, -1.0, 1.0)
    r = -np.log(DF) / T
    h = (T + 1e-5 * T) - T
    vm = bsm.price(F, K, T - h, DF * np.exp(r * h), sig, cp)
    vp = bsm.price(F, K, T + h, DF * np.exp(-r * h), sig, cp)
    th = bsm.theta(F, K, T, DF, sig, cp)
    assert np.all(th < 0)  # OTM: r V / |Theta_0| < 1 (pitfall 11), so no zero crossing
    assert rel_err((vm - vp) / (2 * h), th).max() < 1e-5


def test_theta_t_only_bump_measures_theta0(bsm_gate_grid):
    """Bumping T alone at fixed DF gives Theta_0 = Theta - r V, not Theta (pitfall 10)."""
    F, K, T, DF, sig = unpack(bsm_gate_grid)
    cp = np.where(np.log(F / K) > 0, -1.0, 1.0)
    h = (T + 1e-5 * T) - T
    fd = (bsm.price(F, K, T - h, DF, sig, cp) - bsm.price(F, K, T + h, DF, sig, cp)) / (2 * h)
    theta0 = bsm.theta(F, K, T, DF, sig, cp) + np.log(DF) / T * bsm.price(F, K, T, DF, sig, cp)
    assert rel_err(fd, theta0).max() < 1e-5


# ---------------------------------------------------------------- identities (§5, eq. 9)
def test_greek_identities(bsm_gate_grid):
    """Delta_C - Delta_P = DF; Black PDE (9); vega = sigma T F^2 Gamma = DF K phi(d2) sqrt T;
    Theta_C - Theta_P = r DF (F - K) (ITM theta check, pitfall 11)."""
    F, K, T, DF, sig = unpack(bsm_gate_grid)
    r = -np.log(DF) / T
    dc, dp = bsm.delta(F, K, T, DF, sig, 1), bsm.delta(F, K, T, DF, sig, -1)
    assert np.abs(dc - dp - DF).max() < 1e-12
    g, v = bsm.gamma(F, K, T, DF, sig), bsm.vega(F, K, T, DF, sig)
    assert rel_err(v, sig * T * F**2 * g).max() < 1e-12
    d1, d2 = d12(F, K, T, sig)
    assert rel_err(v, DF * K * norm.pdf(d2) * np.sqrt(T)).max() < 1e-12
    for cp in (1, -1):
        th, V = bsm.theta(F, K, T, DF, sig, cp), bsm.price(F, K, T, DF, sig, cp)
        terms = np.abs(th) + 0.5 * sig**2 * F**2 * g + np.abs(r * V)
        assert (np.abs(th + 0.5 * sig**2 * F**2 * g - r * V) / terms).max() < 1e-12
    thc, thp = bsm.theta(F, K, T, DF, sig, 1), bsm.theta(F, K, T, DF, sig, -1)
    scale = np.abs(thc) + np.abs(thp) + np.abs(r * DF * (F - K))
    assert (np.abs(thc - thp - r * DF * (F - K)) / scale).max() < 1e-12


@pytest.mark.parametrize("cp", [1, -1])
def test_homogeneity_and_euler(bsm_gate_grid, cp):
    """V(lam F, lam K) = lam V; Euler V = F Delta_F + K dV/dK, dV/dK = -cp DF Phi(cp d2) (§3.2)."""
    F, K, T, DF, sig = unpack(bsm_gate_grid)
    V = bsm.price(F, K, T, DF, sig, cp)
    assert_identity(bsm.price(3.7 * F, 3.7 * K, T, DF, sig, cp) / 3.7, V)
    _, d2 = d12(F, K, T, sig)
    fd_, kd = F * bsm.delta(F, K, T, DF, sig, cp), -K * cp * DF * norm.cdf(cp * d2)
    assert (np.abs(V - fd_ - kd) / (np.abs(fd_) + np.abs(kd))).max() < 1e-12


def test_swap_and_put_call_symmetry(bsm_gate_grid):
    """C(F, K) = P(K, F) and C(F, K) = (K/F) P(F, F^2/K) (Carr-Ellis-Gupta 1998), §5."""
    F, K, T, DF, sig = unpack(bsm_gate_grid)
    C = bsm.price(F, K, T, DF, sig, 1)
    assert_identity(bsm.price(K, F, T, DF, sig, -1), C)
    assert_identity(K / F * bsm.price(F, F**2 / K, T, DF, sig, -1), C)


@pytest.mark.parametrize("cp", [1, -1])
def test_normalised_form(bsm_gate_grid, cp):
    """V = DF sqrt(FK) b_cp(x, s), eq. (3), and b_{-1}(x, s) = b_{+1}(-x, s)."""
    F, K, T, DF, sig = unpack(bsm_gate_grid)
    x, s = np.log(F / K), sig * np.sqrt(T)
    b = bsm.normalised_price(x, s, cp)
    assert rel_err(DF * np.sqrt(F * K) * b, bsm.price(F, K, T, DF, sig, cp)).max() < 1e-12
    assert rel_err(bsm.normalised_price(-x, s, -cp), b).max() < 1e-12


# ---------------------------------------------------------------- limits, bounds, shape (§5)
@pytest.mark.parametrize("cp", [1, -1])
def test_bounds_strict(bsm_gate_grid, cp):
    """(4): (cp (F - K))^+ < v < F (call) or K (put), strictly, on the gate grid."""
    F, K, T, DF, sig = unpack(bsm_gate_grid)
    v = bsm.price(F, K, T, DF, sig, cp) / DF
    assert np.all(v > np.maximum(cp * (F - K), 0.0))
    assert np.all(v < (F if cp == 1 else K))


def test_sigma_to_zero_intrinsic():
    """s = 1e-8, |x| >= 0.01: V = DF (cp (F - K))^+ to machine precision (§5)."""
    x = np.concatenate([-np.geomspace(0.01, 1.0, 20), np.geomspace(0.01, 1.0, 20)])
    F, K, T = 100.0, 100.0 * np.exp(-x), 0.5
    sig = 1e-8 / np.sqrt(T)
    for cp in (1, -1):
        V = bsm.price(F, K, T, 0.9, sig, cp)
        assert np.abs(V - 0.9 * np.maximum(cp * (F - K), 0.0)).max() <= 1e-13


def test_sigma_zero_limits():
    """sigma = 0 exactly: limits of pitfall 5 (vega at x = 0: right-derivative, D-012)."""
    F, K, T, DF = 100.0, np.array([90.0, 100.0, 110.0]), 0.5, 0.95
    r = -np.log(DF) / T
    for cp in (1, -1):
        V = bsm.price(F, K, T, DF, 0.0, cp)
        np.testing.assert_array_equal(V, DF * np.maximum(cp * (F - K), 0.0))
        np.testing.assert_array_equal(
            bsm.delta(F, K, T, DF, 0.0, cp), cp * DF * np.array([cp > 0, 0.5, cp < 0], float)
        )
        np.testing.assert_array_equal(bsm.theta(F, K, T, DF, 0.0, cp), r * V)
    np.testing.assert_array_equal(bsm.gamma(F, K, T, DF, 0.0), [0.0, np.inf, 0.0])
    np.testing.assert_allclose(
        bsm.vega(F, K, T, DF, 0.0), [0.0, DF * F * np.sqrt(T / (2 * np.pi)), 0.0], rtol=1e-15
    )


def test_sigma_to_infinity():
    """s = 40: C = DF F and P = DF K (residual ~ Phi(-20)), §5."""
    K, T = np.array([25.0, 100.0, 400.0]), 2.0
    sig = 40.0 / np.sqrt(T)
    np.testing.assert_allclose(bsm.price(100.0, K, T, 0.9, sig, 1), 90.0, rtol=1e-14)
    np.testing.assert_allclose(bsm.price(100.0, K, T, 0.9, sig, -1), 0.9 * K, rtol=1e-14)


def test_atm_exact_and_brenner_subrahmanyam():
    """ATM: C = DF F erf(s / (2 sqrt 2)) exactly; B-S C ~ DF F s / sqrt(2 pi), rel. err -s^2/24."""
    sig, T, DF = np.broadcast_arrays(SIG, TT, DFS)
    s = sig * np.sqrt(T)
    C = bsm.price(100.0, 100.0, T, DF, sig, 1)
    assert rel_err(C, DF * 100.0 * erf(s / (2 * np.sqrt(2)))).max() < 1e-13
    s_small = np.array([1e-3, 3e-3, 1e-2])
    bs = s_small / np.sqrt(2 * np.pi)
    ratio = (bsm.price(1.0, 1.0, 1.0, 1.0, s_small) / bs - 1) / (-(s_small**2) / 24)
    np.testing.assert_allclose(ratio, 1.0, atol=1e-3)


def test_shape_monotone_convex():
    """C increases in sigma, decreases in K and is convex in K (§5); 41 strikes within 3 sd."""
    z = np.linspace(-3.0, 3.0, 41)[None, None, :, None]
    sig, T, DF = np.broadcast_arrays(SIG, TT, DFS)
    K = 100.0 * np.exp(-z * sig * np.sqrt(T))  # K decreasing along axis 2
    C = bsm.price(100.0, K, T, DF, sig, 1)
    slope = np.diff(C, axis=2) / np.diff(K, axis=2)  # dC/dK between adjacent strikes
    assert np.all(slope < 0) and np.all(slope > -DF)
    assert np.all(np.diff(slope, axis=2) < 0)  # slope rises with K (K falls along axis 2)
    Kf = np.array([90.0, 100.0, 110.0])[:, None, None]
    sig_f = np.linspace(0.05, 1.0, 40)[None, :, None]
    Cf = bsm.price(100.0, Kf, np.array([0.25, 1.0, 5.0])[None, None, :], 0.95, sig_f, 1)
    assert np.all(np.diff(Cf, axis=1) > 0)


@pytest.mark.parametrize(("sig", "T", "DF"), [(0.05, 0.02, 0.8), (0.2, 1.0, 0.95), (1.0, 5.0, 1.0)])
def test_strike_density_integrates_to_df(sig, T, DF):
    """d2C/dK2 = F^2 Gamma / K^2 (homogeneity, §3.2) integrates to DF over K in (0, inf)."""
    F, s = 100.0, sig * np.sqrt(T)

    def dens(u):  # in u = ln(K/F): d2C/dK2 dK = F^2 Gamma / K du
        K = F * np.exp(u)
        return float(F**2 * bsm.gamma(F, K, T, DF, sig) / K)

    val, _ = quad(dens, -12 * s - s * s, 12 * s, epsabs=1e-13, epsrel=1e-12, limit=200)
    assert abs(val - DF) < 1e-10


# ---------------------------------------------------------------- precision and API
def test_upper_tail_no_cancellation():
    """Deep-OTM put delta = -DF Phi(-d1) keeps relative precision at d1 in [5, 30] (pitfall 1)."""
    d1 = np.linspace(5.0, 30.0, 26)
    s = 0.2
    K = 100.0 * np.exp(-(d1 - s / 2) * s)
    np.testing.assert_allclose(bsm.delta(100.0, K, 1.0, 0.9, s, -1), -0.9 * norm.sf(d1), rtol=1e-12)


def test_hull_published_value():
    """Hull (2003) 5th ed. pp. 287-288 (via MathWorks blkprice): P = C = 1.1166 (half-unit tol.)."""
    args = (20.0, 20.0, 4 / 12, np.exp(-0.09 * 4 / 12), 0.25)
    for cp in (1, -1):
        assert abs(bsm.price(*args, cp) - 1.1166) <= 5e-5


def test_broadcast_and_scalar():
    K = np.array([80.0, 100.0, 120.0])[:, None]
    sig = np.array([0.1, 0.2, 0.3, 0.4])[None, :]
    for fn in (bsm.price, bsm.delta, bsm.gamma, bsm.vega, bsm.theta):
        assert fn(100.0, K, 1.0, 0.97, sig).shape == (3, 4)
        out = fn(100.0, 100.0, 1.0, 0.97, 0.2, -1)
        assert np.ndim(out) == 0 and np.isfinite(out)


def test_extreme_inputs_clean():
    """No -0.0 prices, no spurious overflow warning for |d| > 1e154, finite theta at tiny T."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for args in ((100.0, 100.0, 1e-300, 1.0, 0.2), (100.0, 50.0, 1.0, 1.0, 0.01)):
            assert not np.signbit(bsm.price(*args, -1))
            assert not np.signbit(bsm.normalised_price(np.log(100.0 / args[1]), 1e-150, -1))
        assert bsm.vega(100.0, 110.0, 1.0, 1.0, 1e-300) == 0.0
        assert bsm.gamma(100.0, 110.0, 1.0, 1.0, 1e-300) == 0.0
        assert bsm.theta(100.0, 110.0, 1e-310, 0.9, 0.2, 1) == 0.0


@pytest.mark.parametrize(
    "bad",
    [
        {"F": 0.0},
        {"K": -1.0},
        {"T": 0.0},
        {"DF": 0.0},
        {"F": np.nan},
        {"sigma": -0.1},
        {"sigma": np.inf},
        {"cp": 0},
    ],
)
def test_invalid_inputs_raise(bad):
    args = {"F": 100.0, "K": 100.0, "T": 1.0, "DF": 0.97, "sigma": 0.2, "cp": 1} | bad
    with pytest.raises(ValueError):
        bsm.price(**args)
