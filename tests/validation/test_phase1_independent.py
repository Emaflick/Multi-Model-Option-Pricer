"""Independent validation of Phase 1 (Black-76 price, Greeks, implied vol) -- validator agent.

The qpricer public API (``bsm.price/delta/gamma/vega/theta/normalised_price``,
``iv.implied_vol``) is only called as the system under test. Every reference value is computed
here, with methods that differ from the implementation:

* HP   -- stdlib ``decimal`` at 60 significant digits with an own normal-tail series
          (erf = 2/sqrt(pi) e^{-z^2} sum 2^n z^{2n+1}/(2n+1)!!, all terms positive, working
          precision raised by z^2/ln 10 so 1 - erf keeps >= 50 digits). Gives the exact Black-76
          value of the exact double inputs; Greeks by central differences *in decimal* (no
          double-precision cancellation) and, separately, by the closed forms in decimal.
* CS   -- complex-step derivatives Im f(u + i h)/h, h = 1e-30 u, of an own complex Black-76
          (``scipy.special.ndtr`` on complex input = Faddeeva algorithm, not the real ndtr path).
* RFD  -- own double-precision central differences with one Richardson step (the literal gate:
          "greeks vs central finite differences").
* QUAD -- ``scipy.integrate.quad`` of the payoff, written with expm1, against the N(0,1) density.
* BIS  -- own vectorised bisection root-finder for IV on an own double pricer (scipy.stats.norm).

Run as a script (``.venv/bin/python tests/validation/test_phase1_independent.py``) to print the
numbers quoted in ``docs/validation/phase1.md``; ``--write-golden`` creates
``tests/golden/bsm_black76.json`` only if it does not exist (never overwrites).
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate, special, stats

from qpricer.iv import implied_vol
from qpricer.models import bsm

REPO = Path(__file__).resolve().parents[2]
GOLDEN = REPO / "tests" / "golden" / "bsm_black76.json"
EPS = np.finfo(np.float64).eps
F0 = 100.0
PREC = 60  # decimal working precision (significant digits)


# ============================================================================ HP reference
_PI: dict[int, Decimal] = {}


def _pi(prec: int) -> Decimal:
    """pi by Machin's formula at `prec` digits (cached)."""
    if prec not in _PI:
        with localcontext() as ctx:
            ctx.prec = prec + 10

            def acot(n: int) -> Decimal:
                x2, term = Decimal(n * n), 1 / Decimal(n)
                tot, k, tiny = term, 0, Decimal(10) ** (-(prec + 10))
                while abs(term) > tiny:
                    k += 1
                    term = -term / x2
                    tot += term / (2 * k + 1)
                return tot

            p = 4 * (4 * acot(5) - acot(239))
        with localcontext() as ctx:
            ctx.prec = prec
            _PI[prec] = +p
    return _PI[prec]


def hp_tail(y: Decimal, digits: int = 50) -> Decimal:
    """Q(y) = P(Z > y), Z ~ N(0,1), to ~`digits` significant digits, any sign of y."""
    if y < 0:
        with localcontext() as ctx:
            ctx.prec = digits + 5
            return Decimal(1) - hp_tail(-y, digits)
    if y == 0:
        return Decimal("0.5")
    z2f = float(y) ** 2 / 2.0
    with localcontext() as ctx:
        ctx.prec = digits + int(z2f / math.log(10)) + 15
        z = y / Decimal(2).sqrt()
        z2 = z * z
        term = tot = z
        n, tiny = 0, Decimal(10) ** (-(ctx.prec + 5))
        while True:
            n += 1
            term = term * 2 * z2 / (2 * n + 1)
            tot += term
            if n > z2f and term <= tot * tiny:
                break
        erf = 2 / _pi(ctx.prec).sqrt() * (-z2).exp() * tot
        q = (1 - erf) / 2
    with localcontext() as ctx:
        ctx.prec = digits
        return +q


def _dec(*a):
    return tuple(x if isinstance(x, Decimal) else Decimal(float(x)) for x in a)


def hp_price(F, K, T, DF, sig, cp) -> Decimal:
    """Black-76 V = DF cp [F Phi(cp d1) - K Phi(cp d2)] in decimal (exact double inputs)."""
    with localcontext() as ctx:
        ctx.prec = PREC
        F, K, T, DF, sig = _dec(F, K, T, DF, sig)
        s = sig * T.sqrt()
        d1 = (F / K).ln() / s + s / 2
        d2 = d1 - s
        if cp > 0:
            return DF * (F * hp_tail(-d1) - K * hp_tail(-d2))
        return DF * (K * hp_tail(d2) - F * hp_tail(d1))


def hp_greeks_fd(F, K, T, DF, sig, cp) -> tuple[Decimal, ...]:
    """(V, delta, gamma, vega, theta, theta0) by central differences in 60-digit decimal.

    theta: calendar dV/dt at fixed F, K, sigma and r = -ln(DF)/T (DF = e^{-r tau}), note §1;
    theta0: -dV/dT at fixed F, K, sigma and DF.
    """
    with localcontext() as ctx:
        ctx.prec = PREC
        F, K, T, DF, sig = _dec(F, K, T, DF, sig)
        s = sig * T.sqrt()
        p0 = hp_price(F, K, T, DF, sig, cp)
        h = F * s * Decimal("1e-12")
        delta = (hp_price(F + h, K, T, DF, sig, cp) - hp_price(F - h, K, T, DF, sig, cp)) / (2 * h)
        h = F * s * Decimal("1e-9")
        gamma = (
            hp_price(F + h, K, T, DF, sig, cp) - 2 * p0 + hp_price(F - h, K, T, DF, sig, cp)
        ) / (h * h)
        h = sig * Decimal("1e-12")
        vega = (hp_price(F, K, T, DF, sig + h, cp) - hp_price(F, K, T, DF, sig - h, cp)) / (2 * h)
        r, h = -DF.ln() / T, T * Decimal("1e-12")
        theta = (
            hp_price(F, K, T - h, (-r * (T - h)).exp(), sig, cp)
            - hp_price(F, K, T + h, (-r * (T + h)).exp(), sig, cp)
        ) / (2 * h)
        theta0 = (hp_price(F, K, T - h, DF, sig, cp) - hp_price(F, K, T + h, DF, sig, cp)) / (2 * h)
        return p0, delta, gamma, vega, theta, theta0


def hp_greeks_closed(F, K, T, DF, sig, cp) -> tuple[Decimal, ...]:
    """(V, delta, gamma, vega, theta) from the textbook closed forms, evaluated in decimal."""
    with localcontext() as ctx:
        ctx.prec = PREC
        F, K, T, DF, sig = _dec(F, K, T, DF, sig)
        s = sig * T.sqrt()
        d1 = (F / K).ln() / s + s / 2
        pdf = (-d1 * d1 / 2).exp() / (2 * _pi(PREC)).sqrt()
        V = hp_price(F, K, T, DF, sig, cp)
        delta = DF * hp_tail(-d1) if cp > 0 else -DF * hp_tail(d1)
        gamma = DF * pdf / (F * s)
        vega = DF * F * pdf * T.sqrt()
        theta = -DF * F * pdf * sig / (2 * T.sqrt()) - DF.ln() / T * V
        return V, delta, gamma, vega, theta


# ============================================================================ double references
def ref_price(F, K, T, DF, sig, cp):
    """Own double Black-76 (scipy.stats.norm; put written with survival functions)."""
    s = sig * np.sqrt(T)
    d1 = np.log(F / K) / s + s / 2
    d2 = d1 - s
    call = DF * (F * stats.norm.cdf(d1) - K * stats.norm.cdf(d2))
    put = DF * (K * stats.norm.sf(d2) - F * stats.norm.sf(d1))
    return np.where(np.asarray(cp) > 0, call, put)


def cs_price(F, K, T, DF, sig, cp):
    """Black-76 valid for complex arguments (complex-step reference)."""
    s = sig * np.sqrt(T)
    d1 = np.log(F / K) / s + s / 2
    d2 = d1 - s
    return DF * cp * (F * special.ndtr(cp * d1) - K * special.ndtr(cp * d2))


def cs_delta(F, K, T, DF, sig, cp):
    """Textbook delta cp DF Phi(cp d1), complex-capable (used only to complex-step gamma)."""
    s = sig * np.sqrt(T)
    return cp * DF * special.ndtr(cp * (np.log(F / K) / s + s / 2))


def cs_greeks(F, K, T, DF, sig, cp):
    h = 1e-30
    r = -np.log(DF) / T
    Tc = T + 1j * h * T
    return {
        "delta": cs_price(F + 1j * h * F, K, T, DF, sig, cp).imag / (h * F),
        "gamma": cs_delta(F + 1j * h * F, K, T, DF, sig, cp).imag / (h * F),
        "vega": cs_price(F, K, T, DF, sig + 1j * h * sig, cp).imag / (h * sig),
        "theta": -cs_price(F, K, Tc, np.exp(-r * Tc), sig, cp).imag / (h * T),
    }


def quad_price(F, K, T, DF, sig, cp) -> float:
    """DF E[(cp(F_T - K))^+] by quadrature: F_T = F e^{s u - s^2/2}, written K expm1(s(u - u*))."""
    s = sig * math.sqrt(T)
    us = math.log(K / F) / s + s / 2  # F_T = K at u = u*
    f = lambda u: cp * K * math.expm1(s * (u - us)) * math.exp(-u * u / 2) / math.sqrt(2 * math.pi)  # noqa: E731
    a, b = (us, max(us, 0.0) + 40.0) if cp > 0 else (min(us, 0.0) - 40.0, us)
    val, _ = integrate.quad(f, a, b, epsabs=0.0, epsrel=2e-14, limit=400)
    return DF * val


def rfd1(f, u, h):
    """Central difference + one Richardson step, O(h^4); h_eff = (u + h) - u."""

    def d(hh):
        hh = (u + hh) - u
        return (f(u + hh) - f(u - hh)) / (2 * hh)

    return (4 * d(h) - d(2 * h)) / 3


def rfd2(f, u, h):
    def d(hh):
        hh = (u + hh) - u
        return (f(u + hh) - 2 * f(u) + f(u - hh)) / hh**2

    return (4 * d(h) - d(2 * h)) / 3


def bisect_iv(V, F, K, T, DF, cp, n_iter=64, hi=12.0):
    """Own IV: bisection in sigma on the OTM-side price (ITM quote mapped by parity)."""
    V, F, K, T, DF, cp = np.broadcast_arrays(*(np.asarray(a, float) for a in (V, F, K, T, DF, cp)))
    otm = np.where(np.log(F / K) > 0, -1.0, 1.0)
    Vo = np.where(cp == otm, V, V - cp * DF * (F - K))
    lo, up = np.zeros_like(V), np.full_like(V, hi)
    for _ in range(n_iter):
        mid = 0.5 * (lo + up)
        below = ref_price(F, K, T, DF, np.maximum(mid, 1e-300), otm) < Vo
        lo, up = np.where(below, mid, lo), np.where(below, up, mid)
    return 0.5 * (lo + up)


def iv_full(*a, **k):
    """implied_vol with full_output, capturing RuntimeWarnings."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sig, info = implied_vol(*a, full_output=True, **k)
    return sig, info, [x for x in w if issubclass(x.category, RuntimeWarning)]


# ============================================================================ samples
def gate_grid():
    """Same lattice as the implementer's fixture is NOT reused; this is an independent copy."""
    sig = np.linspace(0.05, 1.0, 40)[:, None, None, None]
    T = np.geomspace(0.02, 5.0, 30)[None, :, None, None]
    z = np.linspace(-3.0, 3.0, 41)[None, None, :, None]
    DF = np.array([0.8, 0.9, 1.0])[None, None, None, :]
    K = F0 * np.exp(-z * sig * np.sqrt(T))
    return np.broadcast_arrays(sig, T, K, DF)


def random_domain(n, seed, z_max=3.0, edges=True):
    """Off-lattice sample: sigma U[0.05,1], T log-U[0.02,5], z U[-z_max,z_max], DF U[0.8,1.02],
    cp random. With edges=True the 2^4 * 2 domain corners (sigma, T, z = +-3, DF) are appended."""
    rng = np.random.default_rng(seed)
    sig = rng.uniform(0.05, 1.0, n)
    T = np.exp(rng.uniform(np.log(0.02), np.log(5.0), n))
    z = rng.uniform(-z_max, z_max, n)
    DF = rng.uniform(0.8, 1.02, n)
    cp = rng.choice([-1.0, 1.0], n)
    if edges:
        c = np.array(
            np.meshgrid([0.05, 1.0], [0.02, 5.0], [-3.0, 0.0, 3.0], [0.8, 1.0], [-1.0, 1.0])
        )
        c = c.reshape(5, -1)
        sig, T, z, DF, cp = (
            np.concatenate([a, b]) for a, b in zip((sig, T, z, DF, cp), c, strict=True)
        )
    K = F0 * np.exp(-z * sig * np.sqrt(T))
    return sig, T, K, DF, cp


@pytest.fixture(scope="module")
def hp_sample():
    """~170 points (random + corners), both cp, with HP price and HP-FD Greeks."""
    sig, T, K, DF, _ = random_domain(140, seed=20261107)
    rows = []
    for cp in (1.0, -1.0):
        for i in range(sig.size):
            g = hp_greeks_fd(F0, K[i], T[i], DF[i], sig[i], cp)
            rows.append((sig[i], T[i], K[i], DF[i], cp, *[float(v) for v in g]))
    a = np.array(rows)
    keys = ("sig", "T", "K", "DF", "cp", "V", "delta", "gamma", "vega", "theta", "theta0")
    return {k: a[:, j] for j, k in enumerate(keys)}


def _impl_greeks(d):
    args = (F0, d["K"], d["T"], d["DF"], d["sig"], d["cp"])
    return {
        "V": bsm.price(*args),
        "delta": bsm.delta(*args),
        "gamma": bsm.gamma(*args),
        "vega": bsm.vega(*args),
        "theta": bsm.theta(*args),
    }


# ============================================================================ 0. reference sanity
def test_reference_self_consistency():
    """HP tail vs scipy at moderate args; HP closed-form Greeks vs HP FD; HP vs QUAD prices."""
    for y in (-5.0, -1.3, 0.2, 2.0, 7.5, 20.0):
        assert abs(float(hp_tail(Decimal(y))) / special.ndtr(-y) - 1) < 1e-13  # ndtr: 3e-14 at 20
    for case in [(100.0, 80.0, 0.5, 0.95, 0.3, 1), (100.0, 130.0, 2.0, 0.9, 0.25, -1)]:
        fd, cf = hp_greeks_fd(*case), hp_greeks_closed(*case)
        for a, b in zip(fd[:5], cf, strict=True):
            assert abs(a / b - 1) < Decimal("1e-15")
        assert abs(float(fd[0]) / quad_price(*case) - 1) < 1e-12


# ============================================================================ 1. GATE parity
def test_gate_parity_abs_1e12():
    """Gate: |C - P - DF(F - K)| < 1e-12 at F = 100, independent call and put evaluations.

    Domain: K/F in [0.25, 4] (note §2.5), sigma in [0.05, 1], T in [0.02, 5], DF in [0.8, 1.02].
    """
    rng = np.random.default_rng(1)
    n = 200_000
    K = F0 * np.exp(rng.uniform(np.log(0.25), np.log(4.0), n))
    sig, T = rng.uniform(0.05, 1.0, n), np.exp(rng.uniform(np.log(0.02), np.log(5.0), n))
    DF = rng.uniform(0.8, 1.02, n)
    err = np.abs(bsm.price(F0, K, T, DF, sig, 1) - bsm.price(F0, K, T, DF, sig, -1) - DF * (F0 - K))
    assert err.max() < 1e-12


def test_parity_full_iv_domain_at_ulp_floor():
    """On the whole IV-gate domain |ln K/F| <= 3 sigma sqrt T (K up to 8.2e4) the absolute 1e-12
    is not representable (ulp(8e4) = 1.5e-11); parity must still hold to a few ulps of DF max(F,K).
    """
    sig, T, K, DF = gate_grid()
    err = np.abs(bsm.price(F0, K, T, DF, sig, 1) - bsm.price(F0, K, T, DF, sig, -1) - DF * (F0 - K))
    assert (err / np.spacing(DF * np.maximum(F0, K))).max() <= 4.0


# ============================================================================ 2. GATE IV round trip
def test_gate_iv_round_trip_hp_prices(hp_sample):
    """Gate: |sigma_hat - sigma| < 1e-10, input = HP Black price rounded to double (not qpricer)."""
    d = hp_sample
    sh, info, w = iv_full(d["V"], F0, d["K"], d["T"], d["DF"], d["cp"])
    assert not w and np.all(info["status"] == 0)
    assert np.abs(sh - d["sig"]).max() < 1e-10


def test_gate_iv_round_trip_dense_independent_pricer():
    """Gate on 300k off-lattice points + domain corners, both cp, ITM and OTM inputs, own pricer."""
    sig, T, K, DF, cp = random_domain(300_000, seed=2)
    V = ref_price(F0, K, T, DF, sig, cp)
    sh, info, w = iv_full(V, F0, K, T, DF, cp)
    assert not w and np.all(info["status"] == 0)
    assert np.abs(sh - sig).max() < 1e-10
    sig, T, K, DF = gate_grid()  # independent lattice, both cp
    for c in (1.0, -1.0):
        sh = implied_vol(ref_price(F0, K, T, DF, sig, c), F0, K, T, DF, c)
        assert np.abs(sh - sig).max() < 1e-10


def test_iv_agrees_with_own_bisection():
    """Second root-finder: vectorised bisection on an own pricer agrees to 1e-11."""
    sig, T, K, DF, cp = random_domain(40_000, seed=3)
    V = ref_price(F0, K, T, DF, sig, cp)
    assert np.abs(implied_vol(V, F0, K, T, DF, cp) - bisect_iv(V, F0, K, T, DF, cp)).max() < 1e-11


def test_iv_atm_exact_inverse_and_scale_invariance():
    """ATM: C = DF F erf(s / 2 sqrt 2) inverted in closed form; IV(lam V, lam F, lam K) = IV."""
    T = np.geomspace(0.02, 5.0, 25)
    sig = np.linspace(0.05, 1.0, 25)[:, None]
    DF = 0.93
    C = DF * F0 * special.erf(sig * np.sqrt(T) / (2 * np.sqrt(2)))
    exact = 2 * np.sqrt(2) * special.erfinv(C / (DF * F0)) / np.sqrt(T)
    assert np.abs(implied_vol(C, F0, F0, T, DF, 1) - exact).max() < 1e-13
    sig, T, K, DF, cp = random_domain(5000, seed=4)
    V = ref_price(F0, K, T, DF, sig, cp)
    base = implied_vol(V, F0, K, T, DF, cp)
    for lam in (60.0, 1e-4):  # F = 6000 (SPX level) and F = 0.01
        assert np.abs(implied_vol(lam * V, lam * F0, lam * K, T, DF, cp) - base).max() < 1e-11


# ============================================================================ 3. GATE Greeks
def test_gate_greeks_vs_own_richardson_fd():
    """Gate (literal): analytic Greeks vs own central FD (+ Richardson), rel < 1e-5.

    100k off-lattice points, both cp, ITM *and* OTM for every Greek (the implementer restricted
    gamma and theta to OTM). Steps: 1e-3 x (F s, sigma, T) for D1, 3e-3 F s for gamma.
    Theta FD holds r fixed: DF(T) = e^{-rT}. Near a theta zero (deep-ITM, r > 0) the relative
    error is measured against max(|Theta|, 1e-3 |Theta_0|).
    """
    sig, T, K, DF, cp = random_domain(100_000, seed=5)
    s, r = sig * np.sqrt(T), -np.log(DF) / T
    P = ref_price
    fd = {
        "delta": rfd1(lambda u: P(u, K, T, DF, sig, cp), F0, 1e-3 * F0 * s),
        "gamma": rfd2(lambda u: P(u, K, T, DF, sig, cp), F0, 3e-3 * F0 * s),
        "vega": rfd1(lambda u: P(F0, K, T, DF, u, cp), sig, 1e-3 * sig),
        "theta": -rfd1(lambda u: P(F0, K, u, np.exp(-r * u), sig, cp), T, 1e-3 * T),
    }
    an = {g: getattr(bsm, g)(F0, K, T, DF, sig, cp) for g in fd}
    theta0 = an["theta"] + np.log(DF) / T * bsm.price(F0, K, T, DF, sig, cp)
    for g in fd:
        den = np.abs(an[g])
        if g == "theta":
            den = np.maximum(den, 1e-3 * np.abs(theta0))
        assert (np.abs(fd[g] - an[g]) / den).max() < 1e-5, g


def test_greeks_vs_complex_step_full_lattice():
    """Complex-step (no cancellation) on the full 147,600-point lattice, both cp: rel < 1e-10."""
    sig, T, K, DF = gate_grid()
    F = np.full_like(K, F0)
    for cp in (1.0, -1.0):
        ref = cs_greeks(F, K, T, DF, sig, cp)
        for g, v in ref.items():
            assert np.abs(getattr(bsm, g)(F, K, T, DF, sig, cp) / v - 1).max() < 1e-10, (g, cp)


def test_price_and_greeks_vs_high_precision(hp_sample):
    """Implementation vs exact (60-digit) values: price rel < 1e-11, Greeks rel < 1e-11.

    This is the accuracy behind the 1e-5 gate: the gate is passed by 6 orders of magnitude.
    """
    d = hp_sample
    im = _impl_greeks(d)
    assert np.abs(im["V"] / d["V"] - 1).max() < 1e-11
    for g in ("delta", "gamma", "vega"):
        assert np.abs(im[g] / d[g] - 1).max() < 1e-11, g
    scale = np.abs(d["theta0"]) + np.abs(d["theta"] - d["theta0"])  # |Theta_0| + |r V|
    assert (np.abs(im["theta"] - d["theta"]) / scale).max() < 1e-11


# ============================================================================ 4. conventions
def test_theta_convention_and_sign(hp_sample):
    """Theta = dV/dt calendar, per year, F/K/sigma/r fixed (DF accretes); Theta - rV = Theta_0.

    Checks against HP finite differences, the Black PDE with HP values, and the sign pattern.
    """
    d = hp_sample
    r = -np.log(d["DF"]) / d["T"]
    th = bsm.theta(F0, d["K"], d["T"], d["DF"], d["sig"], d["cp"])
    V = bsm.price(F0, d["K"], d["T"], d["DF"], d["sig"], d["cp"])
    assert np.abs((th - r * V) / d["theta0"] - 1).max() < 1e-11  # Theta_0 = -dV/dT | DF fixed
    pde = d["theta"] + 0.5 * d["sig"] ** 2 * F0**2 * d["gamma"] - r * d["V"]  # on HP values
    assert np.abs(pde).max() < 1e-12 * np.abs(d["theta0"]).max()
    sig, T, K, DF = gate_grid()
    x = np.log(F0 / K)
    otm = np.where(x > 0, -1.0, 1.0)
    assert np.all(bsm.theta(F0, K, T, DF, sig, otm) < 0)  # note pitfall 11: OTM theta < 0
    assert np.all(bsm.theta(F0, F0, T, 1.0, sig, 1) < 0)  # ATM, r = 0: pure decay
    # deep-ITM European put with r > 0 has positive calendar theta (carry on the premium)
    assert bsm.theta(F0, 300.0, 1.0, 0.9, 0.2, -1) > 0


def test_discounting_applied_once():
    """V, Delta, Gamma, Vega are linear in DF (DF enters once); IV is invariant under V -> V DF."""
    sig, T, K, _, cp = random_domain(2000, seed=6)
    for f in (bsm.price, bsm.delta, bsm.gamma, bsm.vega):
        a, b = f(F0, K, T, 0.85, sig, cp), f(F0, K, T, 1.0, sig, cp)
        assert np.abs(a - 0.85 * b).max() <= 4 * EPS * np.abs(b).max(), f.__name__
    V1 = ref_price(F0, K, T, 1.0, sig, cp)
    assert np.abs(implied_vol(0.85 * V1, F0, K, T, 0.85, cp) - sig).max() < 1e-10


def test_call_put_swaps():
    """cp = -1 is a put: vs own put; Gamma, Vega cp-independent; Delta_C - Delta_P = DF;
    Theta_C - Theta_P = r DF (F - K); IV(C) = IV(P)."""
    sig, T, K, DF = (a.ravel() for a in gate_grid())
    P = bsm.price(F0, K, T, DF, sig, -1)
    ref = ref_price(F0, K, T, DF, sig, -1)
    assert np.all(np.abs(P - ref) <= 1e-13 + 1e-11 * ref)
    assert np.array_equal(bsm.gamma(F0, K, T, DF, sig, 1), bsm.gamma(F0, K, T, DF, sig, -1))
    assert np.array_equal(bsm.vega(F0, K, T, DF, sig, 1), bsm.vega(F0, K, T, DF, sig, -1))
    dd = bsm.delta(F0, K, T, DF, sig, 1) - bsm.delta(F0, K, T, DF, sig, -1)
    assert np.abs(dd - DF).max() < 4 * EPS
    r = -np.log(DF) / T
    dth = bsm.theta(F0, K, T, DF, sig, 1) - bsm.theta(F0, K, T, DF, sig, -1)
    assert np.abs(dth - r * DF * (F0 - K)).max() < 1e-12 * np.maximum(1, np.abs(r * DF * K)).max()
    C = ref_price(F0, K, T, DF, sig, 1)
    ivc, ivp = implied_vol(C, F0, K, T, DF, 1), implied_vol(ref, F0, K, T, DF, -1)
    assert np.abs(ivc - ivp).max() < 1e-10


def test_normalised_price_public_form(hp_sample):
    """V = DF sqrt(FK) b_cp(x, s), b_{-1}(x, s) = b_{+1}(-x, s), against HP prices."""
    d = hp_sample
    x, s = np.log(F0 / d["K"]), d["sig"] * np.sqrt(d["T"])
    b = bsm.normalised_price(x, s, d["cp"])
    assert np.abs(d["DF"] * np.sqrt(F0 * d["K"]) * b / d["V"] - 1).max() < 1e-11
    assert np.array_equal(bsm.normalised_price(x, s, -1), bsm.normalised_price(-x, s, 1))


# ============================================================================ 5. published values
def test_hull_futures_option_published():
    """Hull OFOD 5th ed. pp. 287-288 (via MathWorks blkprice docs, as cited in the note):
    F = K = 20, T = 4/12, r = 9%, sigma = 25% -> C = P = 1.1166 (4 d.p.)."""
    for cp in (1, -1):
        v = bsm.price(20.0, 20.0, 4 / 12, math.exp(-0.09 * 4 / 12), 0.25, cp)
        assert abs(v - 1.1166) <= 0.5e-4
        assert abs(float(hp_price(20.0, 20.0, 4 / 12, math.exp(-0.03), 0.25, cp)) - v) < 1e-14


def test_hull_greek_letters_example_spot_mapping():
    """Hull OFOD, 'The Greek Letters' running example (ch. 17 in 7th ed. / ch. 18 in 8th ed.):
    S0 = 49, K = 50, r = 5%, q = 0, sigma = 20%, T = 20/52. Published: price 2.40,
    Delta 0.522, Gamma 0.066, Vega 12.1 (per unit vol), Theta -4.31 per year (-0.0118 per day).
    Hull's Greeks are spot Greeks; map the forward kernels by eq. (10): F = S e^{rT},
    Delta_S = Delta_F F/S, Gamma_S = Gamma_F (F/S)^2, Theta_S = Theta - r F Delta_F.
    The kernel Theta (F fixed) is NOT Hull's theta; the mapping is what makes them agree.
    """
    S, K, r, T, sig = 49.0, 50.0, 0.05, 20 / 52, 0.20
    F, DF = S * math.exp(r * T), math.exp(-r * T)
    a = (F, K, T, DF, sig, 1)
    m = F / S
    assert abs(bsm.price(*a) - 2.40) <= 0.005
    assert abs(bsm.delta(*a) * m - 0.522) <= 0.0005
    assert abs(bsm.gamma(*a) * m**2 - 0.066) <= 0.0005
    assert abs(bsm.vega(*a) - 12.1) <= 0.05
    theta_s = bsm.theta(*a) - r * F * bsm.delta(*a)
    assert abs(theta_s - (-4.31)) <= 0.005
    assert abs(theta_s / 365 - (-0.0118)) <= 0.00005
    assert abs(bsm.theta(*a) - (-4.31)) > 0.5  # kernel theta differs by r F Delta_F ~ 1.3


def _golden_cases():
    def k(z, sig, T):
        return F0 * math.exp(-z * sig * math.sqrt(T))

    c = [
        ("hull5_futures_call", 20.0, 20.0, 4 / 12, math.exp(-0.03), 0.25, 1),
        ("hull5_futures_put", 20.0, 20.0, 4 / 12, math.exp(-0.03), 0.25, -1),
        ("atm_df1", 100.0, 100.0, 1.0, 1.0, 0.2, 1),
        ("corner_lowvol_shortT_otm_call", 100.0, k(-3, 0.05, 0.02), 0.02, 0.8, 0.05, 1),
        ("corner_lowvol_shortT_itm_call", 100.0, k(3, 0.05, 0.02), 0.02, 0.8, 0.05, 1),
        ("corner_lowvol_shortT_otm_put", 100.0, k(3, 0.05, 0.02), 0.02, 0.8, 0.05, -1),
        ("corner_highvol_longT_otm_call", 100.0, k(-3, 1.0, 5.0), 5.0, 0.8, 1.0, 1),
        ("corner_highvol_longT_otm_put", 100.0, k(3, 1.0, 5.0), 5.0, 0.8, 1.0, -1),
        ("itm_put_positive_theta", 100.0, 130.0, 1.0, 0.9, 0.15, -1),
        ("negative_rate", 100.0, 95.0, 2.0, 1.02, 0.3, -1),
        ("deep_otm_call_z-8", 100.0, k(-8, 0.3, 1.0), 1.0, 0.9, 0.3, 1),
        ("deep_otm_put_z8", 100.0, k(8, 0.3, 1.0), 1.0, 0.9, 0.3, -1),
    ]
    return c


def build_golden() -> dict:
    """Method A: HP closed forms (decimal). Method B: QUAD (price), HP FD (Greeks)."""
    out = []
    for name, F, K, T, DF, sig, cp in _golden_cases():
        A = hp_greeks_closed(F, K, T, DF, sig, cp)
        B = hp_greeks_fd(F, K, T, DF, sig, cp)
        q = quad_price(F, K, T, DF, sig, cp)
        agree = {"price_quad_rel": abs(q / float(A[0]) - 1)}
        for j, g in enumerate(("delta", "gamma", "vega", "theta"), start=1):
            agree[f"{g}_fd_rel"] = float(abs(B[j] / A[j] - 1))
        row = {"id": name, "F": F, "K": K, "T": T, "DF": DF, "sigma": sig, "cp": cp}
        for j, g in enumerate(("price", "delta", "gamma", "vega", "theta")):
            row[g] = format(A[j], ".25e")
        row["agreement"] = agree
        out.append(row)
    return {
        "description": "Black-76 oracle values (price, forward Greeks, calendar theta at fixed r).",
        "conventions": "V = DF E^T[(cp(F_T-K))^+]; delta/gamma wrt F at fixed DF; vega per unit "
        "vol; theta = dV/dt per year with F, K, sigma and r = -ln(DF)/T fixed "
        "(docs/derivations/bsm.md §1, eq. 8). Inputs are exact IEEE doubles (repr).",
        "source": "Method A: closed forms eqs. (1),(5)-(8) evaluated in Python stdlib decimal at "
        "60 significant digits with the validator's own normal-tail series (independent of "
        "qpricer and of scipy). Method B (independent): price by scipy.integrate.quad of the "
        "payoff against the normal density (epsrel 1e-14); Greeks by central finite differences "
        "of the 60-digit price (steps 1e-12, gamma 1e-9, relative). Stored values = Method A; "
        "'agreement' gives the A-vs-B relative differences. Hull 5th ed. pp. 287-288 publishes "
        "1.1166 for the two hull5 cases.",
        "generator": "tests/validation/test_phase1_independent.py --write-golden (validator)",
        "cases": out,
    }


def test_golden_file():
    """Implementation vs the frozen oracle file (created by the validator, two methods)."""
    if not GOLDEN.exists():
        pytest.fail(f"{GOLDEN} missing: run this module with --write-golden")
    g = json.loads(GOLDEN.read_text())
    for c in g["cases"]:
        a = (c["F"], c["K"], c["T"], c["DF"], c["sigma"], c["cp"])
        assert max(c["agreement"].values()) < 1e-12, c["id"]
        for name, f in (
            ("price", bsm.price),
            ("delta", bsm.delta),
            ("gamma", bsm.gamma),
            ("vega", bsm.vega),
            ("theta", bsm.theta),
        ):
            ref = float(c[name])
            assert abs(f(*a) / ref - 1) < 1e-11, (c["id"], name)
        assert abs(implied_vol(float(c["price"]), *a[:4], a[5]) - c["sigma"]) < 1e-10, c["id"]


# ============================================================================ 6. edge cases
def test_sigma_zero_and_small_total_vol_limits():
    """sigma = 0 -> DF (cp(F-K))^+ exactly; tiny s -> intrinsic; T -> 0 at fixed sigma likewise."""
    K = np.array([60.0, 90.0, 99.0, 101.0, 110.0, 150.0])
    for cp in (1, -1):
        intr = 0.9 * np.maximum(cp * (F0 - K), 0)
        assert np.array_equal(bsm.price(F0, K, 1.0, 0.9, 0.0, cp), intr)
        assert np.abs(bsm.price(F0, K, 1.0, 0.9, 1e-8, cp) - intr).max() <= 1e-13
        assert np.abs(bsm.price(F0, K, 1e-14, 0.9, 0.2, cp) - intr).max() <= 1e-13
        assert np.all(bsm.vega(F0, K, 1.0, 0.9, 0.0, cp) == 0)
        assert np.all(bsm.gamma(F0, K, 1.0, 0.9, 0.0, cp) == 0)
    # ATM s -> 0: exact C = DF F erf(s / 2 sqrt 2). eq. (1) computes F [Phi(s/2) - Phi(-s/2)],
    # a difference of two ~0.5 numbers: absolute error ~eps F (relative ~eps/s), never negative.
    for s in (1e-4, 1e-6, 1e-10, 1e-14):
        for cp in (1, -1):
            c = bsm.price(F0, F0, 1.0, 1.0, s, cp)
            assert c >= 0 and abs(c - F0 * special.erf(s / (2 * math.sqrt(2)))) <= 4 * EPS * F0
    # small T inside the IV domain shape: round trip still fine at T = 1e-4 (outside gate)
    z = np.linspace(-3, 3, 13)
    K = F0 * np.exp(-z * 0.2 * 1e-2)
    for cp in (1, -1):
        V = np.array([float(hp_price(F0, k, 1e-4, 0.99, 0.2, cp)) for k in K])
        assert np.abs(implied_vol(V, F0, K, 1e-4, 0.99, cp) - 0.2).max() < 1e-10


def test_iv_no_arbitrage_bounds_are_explicit():
    """Prices at or outside (4) give NaN + one RuntimeWarning + the documented status code."""
    F, K, T, DF = 100.0, 110.0, 1.0, 0.9
    cases = [  # (V, cp, expected status)
        (0.0, 1, 1),
        (-1e-3, 1, 1),
        (DF * (K - F), -1, 1),
        (np.nextafter(DF * (K - F), 1e9), -1, 1),
        (DF * F, 1, 2),
        (DF * F * (1 + 1e-15), 1, 2),
        (DF * K, -1, 2),
        (np.inf, 1, 4),
        (np.nan, -1, 4),
        (np.nextafter(DF * F, 0.0), 1, 3),
        (np.nextafter(DF * K, 0.0), -1, 3),
    ]
    for V, cp, st in cases:
        s, info, w = iv_full(V, F, K, T, DF, cp)
        assert np.isnan(s) and int(info["status"]) == st and len(w) == 1, (V, cp)
    # vol cap: sigma_true = 12 > sigma_max = 10 -> status 3; 9.5 is recovered
    s, info, w = iv_full(bsm.price(F, K, 0.05, DF, 12.0, 1), F, K, 0.05, DF, 1)
    assert np.isnan(s) and int(info["status"]) == 3 and len(w) == 1
    assert abs(implied_vol(bsm.price(F, K, 0.05, DF, 9.5, 1), F, K, 0.05, DF, 1) - 9.5) < 1e-10
    for bad in (dict(F=0.0), dict(K=-1.0), dict(T=0.0), dict(DF=np.nan), dict(cp=0)):
        kw = dict(V=1.0, F=F, K=K, T=T, DF=DF, cp=1) | bad
        with pytest.raises(ValueError):
            implied_vol(kw["V"], kw["F"], kw["K"], kw["T"], kw["DF"], kw["cp"])
    with pytest.raises(ValueError):
        bsm.price(F, K, -1.0, DF, 0.2)
    with pytest.raises(ValueError):
        bsm.price(F, K, T, DF, -0.2)


def test_iv_deep_itm_is_conditioning_limited():
    """Beyond |z| = 3 an ITM quote carries the time value only to eps max(V, DF F, DF K).
    Output must be NaN + status, or within 10x that conditioning bound of the true sigma."""
    for sig, T in [(0.05, 0.02), (0.3, 1.0), (1.0, 5.0), (0.2, 1e-6), (0.05, 5.0)]:
        for z in np.arange(3.25, 10.01, 0.25):
            for cp in (1, -1):
                zz = z * cp  # ITM side
                K = F0 * math.exp(-zz * sig * math.sqrt(T))
                V = float(hp_price(F0, K, T, 0.9, sig, cp))
                s, info, w = iv_full(V, F0, K, T, 0.9, cp)
                if np.isnan(s):
                    assert int(info["status"]) != 0 and len(w) == 1
                    continue
                cond = EPS * max(V, 0.9 * F0, 0.9 * K) / bsm.vega(F0, K, T, 0.9, sig)
                assert abs(s - sig) <= 10 * cond + 1e-14, (sig, T, zz, cp)


DEEP_OTM = [
    (sig, T, z)
    for sig, T in [(0.05, 0.02), (0.3, 1.0), (1.0, 5.0)]
    for z in (12, 20, 30, 34, 36, 37, 37.5, 38, 38.5, 39, 40)
]


@pytest.mark.parametrize("sig,T,z", DEEP_OTM)
def test_iv_deep_otm_never_silent_wrong(sig, T, z):
    """Beyond the gate domain (|z| up to 40) the OTM inversion must be accurate (1e-8) or
    NaN + warning + status != 0. Never a silent wrong number, never sigma = 0 (pitfall 7)."""
    for cp in (1, -1):
        K = F0 * math.exp(cp * z * sig * math.sqrt(T))  # OTM side for this cp
        V = float(hp_price(F0, K, T, 1.0, sig, cp))
        s, info, w = iv_full(V, F0, K, T, 1.0, cp)
        if np.isnan(s):
            assert int(info["status"]) != 0 and len(w) == 1
        else:
            assert int(info["status"]) == 0 and abs(s - sig) < 1e-8, (
                f"cp={cp} V={V:.3e}: sigma_hat={s!r} (true {sig}), status 0, no warning"
            )


def test_vectorisation_and_shapes():
    """Broadcasting, scalar returns, list inputs, mixed cp; vectorised == element-wise."""
    Fs = np.array([[95.0], [105.0]])
    Ks = [90.0, 100.0, 110.0]
    cps = np.array([1, -1, 1])
    Ts = np.array([[[0.5]], [[2.0]]])
    for f in (bsm.price, bsm.delta, bsm.gamma, bsm.vega, bsm.theta):
        out = f(Fs, Ks, Ts, 0.95, 0.25, cps)
        assert out.shape == (2, 2, 3) and out.dtype == np.float64
        el = np.array(
            [
                [
                    [f(Fs[i, 0], Ks[j], Ts[t, 0, 0], 0.95, 0.25, cps[j]) for j in range(3)]
                    for i in range(2)
                ]
                for t in range(2)
            ]
        )
        assert np.array_equal(out, el)
        assert isinstance(f(100, 100, 1, 1, 0.2), float)
    V = bsm.price(Fs, Ks, Ts, 0.95, 0.25, cps)
    s, info = implied_vol(V, Fs, Ks, Ts, 0.95, cps, full_output=True)
    assert s.shape == (2, 2, 3) and info["status"].shape == (2, 2, 3)
    assert np.abs(s - 0.25).max() < 1e-12
    assert isinstance(implied_vol(8.0, 100, 100, 1, 1), float)
    assert np.isnan(implied_vol([8.0, -1.0], 100, 100, 1, 1)).tolist() == [False, True]


# ============================================================================ script mode
def _report() -> None:  # pragma: no cover - diagnostics for docs/validation/phase1.md
    np.set_printoptions(precision=3)
    sig, T, K, DF = gate_grid()
    err = np.abs(bsm.price(F0, K, T, DF, sig, 1) - bsm.price(F0, K, T, DF, sig, -1) - DF * (F0 - K))
    m = (K / F0 >= 0.25) & (K / F0 <= 4)
    ulp = (err / np.spacing(DF * np.maximum(F0, K))).max()
    print(
        f"parity lattice K/F in [0.25,4]: max {err[m].max():.3e}; full IV domain: max "
        f"{err.max():.3e} at K={K.flat[err.argmax()]:.0f}, "
        f"{int((err > 1e-12).sum())} pts > 1e-12, max {ulp:.2f} ulp"
    )
    rng = np.random.default_rng(1)
    n = 200_000
    Kp = F0 * np.exp(rng.uniform(np.log(0.25), np.log(4.0), n))
    sp, Tp = rng.uniform(0.05, 1.0, n), np.exp(rng.uniform(np.log(0.02), np.log(5.0), n))
    DFp = rng.uniform(0.8, 1.02, n)
    e = np.abs(
        bsm.price(F0, Kp, Tp, DFp, sp, 1) - bsm.price(F0, Kp, Tp, DFp, sp, -1) - DFp * (F0 - Kp)
    )
    print(f"parity random 200k, K/F in [0.25,4]: max {e.max():.3e}")

    sig_r, T_r, K_r, DF_r, cp_r = random_domain(140, seed=20261107)
    rows = []
    for cp in (1.0, -1.0):
        for i in range(sig_r.size):
            g = hp_greeks_fd(F0, K_r[i], T_r[i], DF_r[i], sig_r[i], cp)
            rows.append((sig_r[i], T_r[i], K_r[i], DF_r[i], cp, *[float(v) for v in g]))
    a = np.array(rows)
    d = {
        k: a[:, j]
        for j, k in enumerate(
            ("sig", "T", "K", "DF", "cp", "V", "delta", "gamma", "vega", "theta", "theta0")
        )
    }
    im = _impl_greeks(d)
    print(f"HP sample n={a.shape[0]}: price rel {np.abs(im['V'] / d['V'] - 1).max():.3e}")
    for g in ("delta", "gamma", "vega"):
        print(f"  {g} rel vs HP-FD {np.abs(im[g] / d[g] - 1).max():.3e}")
    sc = np.abs(d["theta0"]) + np.abs(d["theta"] - d["theta0"])
    print(
        f"  theta err/(|Th0|+|rV|) {(np.abs(im['theta'] - d['theta']) / sc).max():.3e}, "
        f"plain rel {np.abs(im['theta'] / d['theta'] - 1).max():.3e}"
    )
    sh = implied_vol(d["V"], F0, d["K"], d["T"], d["DF"], d["cp"])
    print(f"  IV round trip on HP prices: {np.abs(sh - d['sig']).max():.3e}")

    sr, Tr, Kr, DFr, cpr = random_domain(300_000, seed=2)
    V = ref_price(F0, Kr, Tr, DFr, sr, cpr)
    sh, info = implied_vol(V, F0, Kr, Tr, DFr, cpr, full_output=True)
    otm = cpr * np.log(F0 / Kr) <= 0
    e = np.abs(sh - sr)
    print(
        f"IV dense 300k: max {e.max():.3e} (OTM {e[otm].max():.3e}, ITM {e[~otm].max():.3e}),"
        f" Newton evals max {info['n_iter'].max()}, brent {int(info['brent'].sum())}"
    )
    for c in (1.0, -1.0):
        shg = implied_vol(ref_price(F0, K, T, DF, sig, c), F0, K, T, DF, c)
        print(f"IV lattice cp={c:+.0f}: max {np.abs(shg - sig).max():.3e}")
    sb, Tb, Kb, DFb, cpb = random_domain(40_000, seed=3)
    Vb = ref_price(F0, Kb, Tb, DFb, sb, cpb)
    ib, bb = implied_vol(Vb, F0, Kb, Tb, DFb, cpb), bisect_iv(Vb, F0, Kb, Tb, DFb, cpb)
    print(
        f"IV vs own bisection: max |diff| {np.abs(ib - bb).max():.3e}, bisection vs true "
        f"{np.abs(bb - sb).max():.3e}"
    )

    s5, T5, K5, DF5, cp5 = random_domain(100_000, seed=5)
    s, r = s5 * np.sqrt(T5), -np.log(DF5) / T5
    P = ref_price
    fd = {
        "delta": rfd1(lambda u: P(u, K5, T5, DF5, s5, cp5), F0, 1e-3 * F0 * s),
        "gamma": rfd2(lambda u: P(u, K5, T5, DF5, s5, cp5), F0, 3e-3 * F0 * s),
        "vega": rfd1(lambda u: P(F0, K5, T5, DF5, u, cp5), s5, 1e-3 * s5),
        "theta": -rfd1(lambda u: P(F0, K5, u, np.exp(-r * u), s5, cp5), T5, 1e-3 * T5),
    }
    otm5 = cp5 * np.log(F0 / K5) <= 0
    th0 = bsm.theta(F0, K5, T5, DF5, s5, cp5) + np.log(DF5) / T5 * bsm.price(
        F0, K5, T5, DF5, s5, cp5
    )
    for g, v in fd.items():
        an = getattr(bsm, g)(F0, K5, T5, DF5, s5, cp5)
        den = np.maximum(np.abs(an), 1e-3 * np.abs(th0)) if g == "theta" else np.abs(an)
        rr = np.abs(v - an) / den
        print(
            f"RFD {g}: max rel {rr.max():.3e} (OTM {rr[otm5].max():.3e}, ITM {rr[~otm5].max():.3e})"
        )
    for cp in (1.0, -1.0):
        Fg = np.full_like(K, F0)
        ref = cs_greeks(Fg, K, T, DF, sig, cp)
        msg = ", ".join(
            f"{g} {np.abs(getattr(bsm, g)(Fg, K, T, DF, sig, cp) / v - 1).max():.2e}"
            for g, v in ref.items()
        )
        print(f"complex-step lattice cp={cp:+.0f}: {msg}")
    print("deep OTM probes (V from HP, DF = 1):")
    for sg, Tt, z in DEEP_OTM:
        for cp in (1, -1):
            Kd = F0 * math.exp(cp * z * sg * math.sqrt(Tt))
            Vd = float(hp_price(F0, Kd, Tt, 1.0, sg, cp))
            s_, info, w = iv_full(Vd, F0, Kd, Tt, 1.0, cp)
            pe = bsm.price(F0, Kd, Tt, 1.0, sg, cp)
            if not np.isnan(s_) and abs(s_ - sg) > 1e-8 or (z >= 37 and cp == 1):
                print(
                    f"  sig={sg} T={Tt} z={z} cp={cp:+d} V={Vd:.3e} bsm.price={pe:.3e} "
                    f"sigma_hat={s_!r} status={int(info['status'])} warn={len(w)}"
                )


if __name__ == "__main__":  # pragma: no cover
    if "--write-golden" in sys.argv:
        if GOLDEN.exists():
            sys.exit(f"{GOLDEN} exists; golden files are never overwritten")
        GOLDEN.write_text(json.dumps(build_golden(), indent=1) + "\n")
        print(f"wrote {GOLDEN}")
    _report()
