"""Phase 2 (Heston) gates and oracles: CF, cumulants, COS vs Lewis, xi -> 0 limit, published values.

docs/ROADMAP.md Phase 2; oracles from docs/derivations/heston.md §5; xi -> 0 per D-015; COS range
L = 18 per D-017. Sets (F = 100 unless stated): FO (Feller-violating, nu_F = 0.38), GS (SPX-like,
nu_F = 0.62), LW, AT2, KJ (stress, CF check only).
"""

import dataclasses
import warnings

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.stats import norm

from qpricer.engines.fourier import cos_price, cos_range, lewis_price
from qpricer.models import bsm
from qpricer.models.heston import HestonParams, cf, cumulants, total_variance

FO = HestonParams(kappa=1.5768, theta=0.0398, xi=0.5751, rho=-0.5711, v0=0.0175)
GS = HestonParams(kappa=1.3253, theta=0.0354, xi=0.3877, rho=-0.7165, v0=0.0354)
LW = HestonParams(kappa=4.0, theta=0.25, xi=1.0, rho=-0.5, v0=0.04)
AT2 = HestonParams(kappa=1.5, theta=0.04, xi=0.3, rho=-0.9, v0=0.04)
KJ = HestonParams(kappa=1.0, theta=0.16, xi=2.0, rho=-0.8, v0=0.16)
AI = HestonParams(kappa=0.5, theta=0.04, xi=1.0, rho=-0.9, v0=0.04)
SILENT = HestonParams(kappa=3.062, theta=0.01013, xi=0.6537, rho=-0.6225, v0=0.02837)  # F1
GATE_SETS = {"FO": FO, "GS": GS, "LW": LW, "AT2": AT2}
K_GRID = np.arange(60.0, 150.0 + 1e-9, 5.0)
T_GRID = (0.1, 0.25, 0.5, 1.0, 2.0, 3.0)


def test_params_domain():
    """Note §1 / pitfall 5: kappa, theta, xi > 0, v0 >= 0, |rho| < 1, kappa - rho xi > 0."""
    for bad in (
        dict(kappa=0.0),
        dict(theta=-0.01),
        dict(xi=0.0),
        dict(v0=-1e-9),
        dict(rho=1.0),
        dict(rho=-1.0),
        dict(rho=0.9, xi=2.0),  # kappa - rho xi = -0.2 < 0
        dict(kappa=np.nan),
    ):
        with pytest.raises(ValueError):
            dataclasses.replace(FO, **bad)
    assert FO.feller == pytest.approx(0.3795, abs=5e-5)
    assert GS.feller < 1 and LW.feller == pytest.approx(2.0)


# ------------------------------------------------------------------ characteristic function
@pytest.mark.parametrize("p", [FO, GS, LW, AT2, KJ])
@pytest.mark.parametrize("T", [0.1, 1.0, 10.0])
def test_cf_identities(p, T):
    """Eq. (5): phi(0) = phi(-i) = 1 (1e-15), |phi| <= 1 and phi(-u) = conj(phi(u)) on real u,
    phi(-i/2) = E[(F_T/F)^(1/2)] in (0, 1]."""
    assert np.abs(cf(np.array([0.0, -1j]), T, p) - 1.0).max() <= 1e-15
    u = np.linspace(0.0, 200.0, 2001)
    ph = cf(u, T, p)
    assert np.all(np.abs(ph) <= 1.0 + 1e-15)
    assert np.abs(cf(-u, T, p) - np.conj(ph)).max() <= 1e-15
    h = cf(-0.5j, T, p)
    assert abs(h.imag) <= 1e-15 and 0.0 < h.real <= 1.0


def _cf_ode(u: complex, T: float, p: HestonParams) -> complex:
    """Independent CF from the §3.1 Riccati: D' = xi^2 D^2/2 - beta D - A/2, C' = kappa theta D."""
    beta, A = p.kappa - 1j * p.rho * p.xi * u, u * u + 1j * u

    def rhs(_, y):
        D = y[0]
        return [0.5 * p.xi**2 * D * D - beta * D - 0.5 * A, p.kappa * p.theta * D]

    sol = solve_ivp(rhs, (0.0, T), [0j, 0j], method="DOP853", rtol=1e-12, atol=1e-14)
    assert sol.success
    C, D = sol.y[1, -1], sol.y[0, -1]
    return np.exp(C + p.v0 * D)


def _cf_heston1993(u, T, p: HestonParams):
    """Original Heston (1993) form, eq. (4), principal Log: negative control only."""
    beta, A = p.kappa - 1j * p.rho * p.xi * u, u * u + 1j * u
    d = np.sqrt(beta * beta + p.xi**2 * A)
    g1 = (beta + d) / (beta - d)
    D1 = (beta + d) / p.xi**2 * (1 - np.exp(d * T)) / (1 - g1 * np.exp(d * T))
    lg = np.log((1 - g1 * np.exp(d * T)) / (1 - g1))
    C1 = p.kappa * p.theta / p.xi**2 * ((beta + d) * T - 2 * lg)
    return np.exp(C1 + p.v0 * D1)


@pytest.mark.parametrize("p", [FO, KJ])
@pytest.mark.parametrize("T", [0.1, 3.0, 10.0])
def test_cf_vs_riccati_ode(p, T):
    """§5: eq. (3) vs DOP853 (rtol 1e-12) at u in {0.5, 5, 50}, real and on Im u = -1/2: < 1e-9."""
    for u in (0.5, 5.0, 50.0, 0.5 - 0.5j, 5.0 - 0.5j, 50.0 - 0.5j):
        ref = _cf_ode(u, T, p)
        assert abs(cf(u, T, p) / ref - 1.0) < 1e-9, (u, T)


def test_cf_heston1993_form_fails_at_long_T():
    """Negative control (pitfall 1): form (4) jumps across the Log branch cut at FO, T = 10,
    while form (3) matches the ODE there."""
    u = np.linspace(0.05, 30.0, 600)
    good, bad = cf(u, 10.0, FO), _cf_heston1993(u, 10.0, FO)
    assert np.abs(bad / good - 1.0).max() > 1e-2
    for uu in (3.0, 12.0, 25.0):
        assert abs(cf(uu, 10.0, FO) / _cf_ode(uu, 10.0, FO) - 1.0) < 1e-9


def test_cf_small_xi_no_cancellation():
    """Pitfall 3: at xi = 1e-6, rho = 0 the CF equals Black's e^{-A Sigma/2} to O(xi^2)."""
    p = HestonParams(kappa=1.5, theta=0.04, xi=1e-6, rho=0.0, v0=0.04)
    u = np.linspace(0.0, 20.0, 201)
    black = np.exp(-0.5 * (u * u + 1j * u) * float(total_variance(1.0, p)))
    assert np.abs(cf(u, 1.0, p) - black).max() < 1e-10


# ------------------------------------------------------------------ cumulants and range
@pytest.mark.parametrize("p", [FO, GS, LW, AT2, KJ])
@pytest.mark.parametrize("T", [0.1, 1.0, 3.0, 10.0])
def test_cumulants_vs_cgf_differences(p, T):
    """§5: c1, c2 (eqs. 6-7) vs central differences of K(w) = ln phi(-i w), rel < 1e-5.

    Base step h = 1e-3 with one Richardson step (h, h/2): the plain h = 1e-3 stencil carries an
    O(h^2 c4 / c2) truncation error of 1.3e-5 for the stress set KJ at T = 10 (measured)."""

    def diffs(h):
        Kp, Km = (np.log(cf(-1j * w, T, p)).real for w in (h, -h))
        return np.array([(Kp - Km) / (2 * h), (Kp + Km) / h**2])

    d1, d2 = (4 * diffs(5e-4) - diffs(1e-3)) / 3
    c1, c2 = cumulants(T, p)
    assert abs(d1 / c1 - 1.0) < 1e-5 and abs(d2 / c2 - 1.0) < 1e-5


@pytest.mark.parametrize(
    "p",
    [
        HestonParams(kappa=0.999e-3, theta=0.5, xi=0.5, rho=-0.7, v0=0.01),
        HestonParams(kappa=5e-4, theta=0.5, xi=1.0, rho=-0.9, v0=0.005),
        HestonParams(kappa=0.999e-3, theta=0.04, xi=0.5, rho=-0.7, v0=0.04),
    ],
)
def test_cumulants_small_kappa_branch(p):
    """Validator F3, pitfall 9: the kappa T < 1e-3 branch keeps the first-order kappa (incl.
    kappa theta) terms. Vs the Richardson CGF differences (T = 1): rel < 1e-5 (the old v0-only
    branch: -2.25e-2 at theta / v0 = 50); continuous across the switch at kappa T = 1e-3."""
    h, T = 1e-3, 1.0
    d = [sum(np.log(cf(1j * w, T, p)).real for w in (s, -s)) / s**2 for s in (h, h / 2)]
    assert abs(cumulants(T, p)[1] / ((4 * d[1] - d[0]) / 3) - 1.0) < 1e-5
    lo, hi = (dataclasses.replace(p, kappa=k) for k in (0.9999e-3, 1.0001e-3))
    assert abs(cumulants(T, lo)[1] / cumulants(T, hi)[1] - 1.0) < 1e-5


def _fo_table11_c2(T: float, p: HestonParams, misprint: bool) -> float:
    """Fang-Oosterlee (2008) Table 11 c2, with lambda, u_bar, u0, eta = kappa, theta, v0, xi."""
    lam, ub, u0, eta, rho = p.kappa, p.theta, p.v0, p.xi, p.rho
    E = np.exp(-lam * T)
    t = ub * ((6 * E - 7) if misprint else (4 * E - 5))
    return (
        eta * T * lam * E * (u0 - ub) * (8 * lam * rho - 4 * eta)
        + lam * rho * eta * (1 - E) * (16 * ub - 8 * u0)
        + 2 * ub * lam * T * (-4 * lam * rho * eta + eta**2 + 4 * lam**2)
        + eta**2 * ((ub - 2 * u0) * E**2 + t + 2 * u0)
        + 8 * lam**2 * (u0 - ub) * (1 - E)
    ) / (8 * lam**3)


@pytest.mark.parametrize("p", [FO, GS, LW, AT2])
@pytest.mark.parametrize("T", [0.1, 1.0, 10.0])
def test_c2_fang_oosterlee_misprint(p, T):
    """§2.2: eq. (7) = F&O Table 11 with u_bar (4E - 5); the printed (6E - 7) is smaller by
    xi^2 theta (1 - E) / (4 kappa^3)."""
    c2 = cumulants(T, p)[1]
    assert abs(_fo_table11_c2(T, p, False) / c2 - 1.0) < 1e-13
    gap = p.xi**2 * p.theta * (1 - np.exp(-p.kappa * T)) / (4 * p.kappa**3)
    assert abs(c2 - _fo_table11_c2(T, p, True) - gap) < 1e-13


def test_cumulant_hand_check_fo():
    """§2.2 hand check (4 significant digits), FO at T = 1: c2 = 0.03157, Sigma_bar = 0.02858."""
    assert abs(cumulants(1.0, FO)[1] - 0.03157) < 5e-6
    assert abs(float(total_variance(1.0, FO)) - 0.02858) < 5e-6
    assert cos_range(1.0, FO)[0] < 0 < cos_range(1.0, FO)[1]


# ------------------------------------------------------------------ gate: COS vs Lewis
@pytest.mark.parametrize("name", list(GATE_SETS))
def test_gate_cos_vs_lewis(name):
    """ROADMAP gate: |COS - Lewis| < 1e-6 on K in {60..150} x T in {0.1..3} x DF in {1, 0.95},
    puts and calls (COS: self-checking default, D-017; Lewis: quad_vec 1e-11). FO, GS violate
    Feller."""
    p, worst = GATE_SETS[name], 0.0
    for T in T_GRID:
        for DF in (1.0, 0.95):
            for cp in (1, -1):
                c = cos_price(100.0, K_GRID, T, DF, p, cp)
                lw = lewis_price(100.0, K_GRID, T, DF, p, cp)
                worst = max(worst, np.abs(c - lw).max())
    assert worst < 1e-6, worst


def test_cross_engine_parity():
    """§5: C_Lewis - P_COS = DF (F - K) to < 1e-6 (parity inside one engine is tautological)."""
    for p in GATE_SETS.values():
        for T in (0.25, 2.0):
            c = lewis_price(100.0, K_GRID, T, 0.95, p, 1)
            pu = cos_price(100.0, K_GRID, T, 0.95, p, -1)
            assert np.abs(c - pu - 0.95 * (100.0 - K_GRID)).max() < 1e-6


def test_share_measure_symmetry():
    """Eq. (18): C(F, K; kappa, theta, xi, rho, v0) = K/F P(F, F^2/K; kappa - rho xi,
    kappa theta / (kappa - rho xi), xi, -rho, v0), < 1e-6."""
    for p in (FO, GS):
        ks = p.kappa - p.rho * p.xi
        q = HestonParams(ks, p.kappa * p.theta / ks, p.xi, -p.rho, p.v0)
        for T in (0.5, 2.0):
            lhs = cos_price(100.0, K_GRID, T, 0.97, p, 1)
            rhs = K_GRID / 100.0 * cos_price(100.0, 100.0**2 / K_GRID, T, 0.97, q, -1)
            assert np.abs(lhs - rhs).max() < 1e-6


def test_no_arbitrage_shape():
    """§5: P increasing and convex in K, 0 <= dP/dK <= DF, calls non-decreasing in T (fixed F),
    on the ROADMAP strike domain K in [60, 150] (step 0.5). Allowance 1e-9 for COS numerical
    noise (1000x below the 1e-6 gate; measured worst violation 2.1e-10, slope vs DF)."""
    K, tol = np.linspace(60.0, 150.0, 181), 1e-9
    for p in GATE_SETS.values():
        prev = None
        for T in T_GRID:
            P = cos_price(100.0, K, T, 0.9, p, -1)
            slope = np.diff(P) / np.diff(K)
            assert np.all(slope >= -tol) and np.all(slope <= 0.9 + tol)
            assert np.all(np.diff(P, 2) >= -tol)
            C = cos_price(100.0, K, T, 1.0, p, 1)
            if prev is not None:
                assert np.all(C - prev >= -tol)
            prev = C


def test_cos_strikes_outside_range_and_warning():
    """Fixed (N, L) = (256, 18): pitfall 13, z <= a gives P = 0 exactly; pitfall 10, N far too
    small for A-I at T = 10 triggers the eq. (13) series warning."""
    a, b = cos_range(1.0, FO, 18.0)
    K = 100.0 * np.exp(np.array([a - 1.0, b + 1.0]))
    P = cos_price(100.0, K, 1.0, 1.0, FO, -1, N=256, L=18.0)
    assert P[0] == 0.0 and abs(P[1] - (K[1] - 100.0)) < 1e-6 * K[1]
    with pytest.warns(RuntimeWarning, match="series error"):
        cos_price(100.0, 100.0, 10.0, 1.0, AI, -1, N=256)


@pytest.mark.parametrize("p, T", [(SILENT, 1.0), (AI, 10.0), (FO, 1.0)])
def test_cos_default_self_check_hard_sets(p, T):
    """Validator F1 / D-017: the default (self-checking) COS meets the gate without any warning
    where fixed (256, 18) fails: SILENT (validator scan, 2.6e-6 unwarned at T = 1), A-I at T = 10
    (1.7e-4), FO at T = 1 (1.0e-7 -> now near Lewis)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for cp in (1, -1):
            c = cos_price(100.0, K_GRID, T, 1.0, p, cp)
            assert np.abs(c - lewis_price(100.0, K_GRID, T, 1.0, p, cp)).max() < 1e-6


def test_cos_default_warns_when_not_converged():
    """D-017: if the self-check cannot reach tol F by N = N_MAX it must warn, never stay silent."""
    with pytest.warns(RuntimeWarning, match="not converged"):
        cos_price(100.0, 100.0, 1.0, 1.0, FO, -1, tol=1e-30)


def test_cp_is_required():
    """Validator F4: no silent default option type in the Heston engines (bsm.price defaults to
    a call, the Heston engines used to default to a put)."""
    for engine in (cos_price, lewis_price):
        with pytest.raises(TypeError):
            engine(100.0, 100.0, 1.0, 1.0, FO)


def test_lewis_raises_on_quadrature_failure(monkeypatch):
    """Eq. (15): quad_vec status != 0 must raise, never return an unconverged price."""
    import qpricer.engines.fourier as fourier

    def fake(*args, **kwargs):
        return np.zeros(1), 1.0, type("Info", (), {"status": 1})()

    monkeypatch.setattr(fourier, "quad_vec", fake)
    with pytest.raises(RuntimeError, match="status 1"):
        fourier.lewis_price(100.0, [100.0], 1.0, 1.0, FO, -1)


# ------------------------------------------------------------------ gate: xi -> 0 (D-015)
def _black_limit(F, K, T, DF, p, cp):
    """Eq. (16): V_B(sigma_eff) + rho xi J dx dSigma V_B, dx dSigma V_B = -DF F n(d1) d2 / (2S)."""
    S = float(total_variance(T, p))
    k, E = p.kappa, np.exp(-p.kappa * T)
    tau = (1 - E) / k
    J = p.theta / k * (T - tau) + (p.v0 - p.theta) / k * (tau - T * E)
    d1 = (np.log(F / K) + 0.5 * S) / np.sqrt(S)
    d2 = d1 - np.sqrt(S)
    black = bsm.price(F, K, T, DF, np.sqrt(S / T), cp)
    return black, black + p.rho * p.xi * J * (-DF * F * norm.pdf(d1) * d2 / (2 * S))


@pytest.mark.parametrize("theta, v0", [(0.04, 0.04), (0.04, 0.09), (0.09, 0.01)])
@pytest.mark.parametrize("rho", [0.0, -0.7])
@pytest.mark.parametrize("engine", [cos_price, lewis_price])
def test_gate_xi_to_zero(rho, engine, theta, v0):
    """ROADMAP gate per D-015, xi = 1e-4, kappa = 1.5, K x T grid, DF in {1, 0.95}, calls and
    puts: rho = 0 vs plain Black at sigma_eff^2 = Sigma_bar/T, < 1e-5; rho = -0.7 vs Black + the
    eq. (16) rho xi J term, < 1e-5. v0 != theta exercises the time-average (validator F2)."""
    p = HestonParams(kappa=1.5, theta=theta, xi=1e-4, rho=rho, v0=v0)
    worst = 0.0
    for T in T_GRID:
        for DF in (1.0, 0.95):
            for cp in (1, -1):
                black, corrected = _black_limit(100.0, K_GRID, T, DF, p, cp)
                ref = black if rho == 0.0 else corrected
                worst = max(worst, np.abs(engine(100.0, K_GRID, T, DF, p, cp) - ref).max())
    assert worst < 1e-5, worst


def test_xi_to_zero_power_check():
    """§5: at rho = -0.7 the gap to plain Black at K = 120, T = 1 is about -3.3e-4 (2 digits),
    i.e. the rho = 0 gate would not hold with correlation (D-015)."""
    p = HestonParams(kappa=1.5, theta=0.04, xi=1e-4, rho=-0.7, v0=0.04)
    gap = cos_price(100.0, 120.0, 1.0, 1.0, p, 1) - bsm.price(100.0, 120.0, 1.0, 1.0, 0.2, 1)
    assert abs(gap + 3.3e-4) < 5e-6


# ------------------------------------------------------------------ published values (§5)
def test_published_fang_oosterlee():
    """FO1 = 5.785155450 (T = 1), FO2 = 22.318945791 (T = 10), F = K = 100, DF = 1, tol 5e-8,
    default (self-checking) COS and Lewis. Fixed (256, 18) is 1.1e-7 off at T = 1 (D-017)."""
    assert abs(lewis_price(100.0, 100.0, 1.0, 1.0, FO, 1) - 5.785155450) < 5e-8
    assert abs(cos_price(100.0, 100.0, 1.0, 1.0, FO, 1) - 5.785155450) < 5e-8
    assert abs(cos_price(100.0, 100.0, 1.0, 1.0, FO, 1, N=256, L=18.0) - 5.785155450) > 5e-8
    assert abs(lewis_price(100.0, 100.0, 10.0, 1.0, FO, 1) - 22.318945791) < 5e-8
    assert abs(cos_price(100.0, 100.0, 10.0, 1.0, FO, 1) - 22.318945791) < 5e-8


def test_published_lewis_quantlib():
    """L1-5: A. Lewis reference prices via QuantLib testAlanLewisReferencePrices, set LW, T = 1,
    F = 100 e^{-0.01}, DF = e^{-0.01}, tol 1e-9, both engines."""
    F, DF, K = 100.0 * np.exp(-0.01), np.exp(-0.01), np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    C = [26.774758743998854, 20.933349000596710, 16.070154917028834, 12.132211516709845,
         9.024913483457836]  # fmt: skip
    P = [7.958878113256768, 12.017966707346305, 17.055270961270109, 23.017825898442801,
         29.811026202682472]  # fmt: skip
    for engine in (cos_price, lewis_price):
        assert np.abs(engine(F, K, 1.0, DF, LW, 1) - C).max() < 1e-9
        assert np.abs(engine(F, K, 1.0, DF, LW, -1) - P).max() < 1e-9


def test_published_broadie_kaya():
    """BK-1 = 6.8061 (T = 1) and BK-2 = 34.9998 (T = 5), ATM calls, F = 100 e^{rT}, DF = e^{-rT}
    (heston-mc.md §5, two sources each), tol 5e-5 (4 decimals)."""
    bk1 = HestonParams(kappa=6.21, theta=0.019, xi=0.61, rho=-0.7, v0=0.010201)
    bk2 = HestonParams(kappa=2.0, theta=0.09, xi=1.0, rho=-0.3, v0=0.09)
    for p, T, r, ref in ((bk1, 1.0, 0.0319, 6.8061), (bk2, 5.0, 0.05, 34.9998)):
        F, DF = 100.0 * np.exp(r * T), np.exp(-r * T)
        for engine in (cos_price, lewis_price):
            assert abs(engine(F, 100.0, T, DF, p, 1) - ref) < 5e-5
