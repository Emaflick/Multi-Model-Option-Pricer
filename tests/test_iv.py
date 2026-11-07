"""Tests for qpricer.iv (eqs. 11-16, derivations/bsm.md), Phase 1 IV gate and §5 IV oracles.

Gate (docs/ROADMAP.md): round-trip |sigma_hat - sigma| < 1e-10 for sigma in [0.05, 1],
T in [0.02, 5], |ln(K/F)| <= 3 sigma sqrt T (fixture `bsm_gate_grid`), both cp, ITM and OTM inputs.
"""

import warnings

import numpy as np
import pytest
from scipy.special import erfinv

from qpricer.iv import N_MAX, STATUS, implied_vol
from qpricer.models import bsm

EPS = np.finfo(np.float64).eps


@pytest.mark.parametrize("cp", [1, -1])
def test_round_trip_gate(bsm_gate_grid, cp):
    """IV(price(sigma)) = sigma within 1e-10 on the gate grid; Newton only (no Brent, D-011)."""
    g = bsm_gate_grid
    V = bsm.price(g["F"], g["K"], g["T"], g["DF"], g["sigma"], cp)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no NaN / out-of-bounds warning on the gate domain
        sig, info = implied_vol(V, g["F"], g["K"], g["T"], g["DF"], cp, full_output=True)
    assert np.abs(sig - g["sigma"]).max() < 1e-10
    assert np.all(info["status"] == 0)
    assert not info["brent"].any()
    assert info["n_iter"].max() <= 20  # pitfall 8: ~10 expected at the slow convex corner


def test_call_put_same_iv(bsm_gate_grid):
    """IV(call) = IV(put) at the same K, the put taken from parity (2) (§5)."""
    g = bsm_gate_grid
    F, K, T, DF = g["F"], g["K"], g["T"], g["DF"]
    C = bsm.price(F, K, T, DF, g["sigma"], 1)
    sc = implied_vol(C, F, K, T, DF, 1)
    sp = implied_vol(C - DF * (F - K), F, K, T, DF, -1)
    assert np.abs(sc - sp).max() < 1e-10


def test_atm_exact_inverse():
    """x = 0: solver = exact inverse 2 sqrt 2 erfinv(beta), eq. (13); near-ATM via Newton."""
    sig = np.linspace(0.05, 1.0, 40)[:, None]
    T = np.geomspace(0.02, 5.0, 30)[None, :]
    C = bsm.price(100.0, 100.0, T, 0.9, sig, 1)
    exact = 2 * np.sqrt(2) * erfinv(C / (0.9 * 100.0)) / np.sqrt(T)
    assert np.abs(exact - sig).max() < 1e-10
    assert np.abs(implied_vol(C, 100.0, 100.0, T, 0.9, 1) - exact).max() < 1e-12
    for K in (100.0 * (1 + 1e-9), 100.0 * (1 - 1e-9)):  # x != 0: starts at s_c ~ 4.5e-5
        V = bsm.price(100.0, K, T, 0.9, sig, 1)
        assert np.abs(implied_vol(V, 100.0, K, T, 0.9, 1) - sig).max() < 1e-10


def test_out_of_bounds_nan_with_warning():
    """Outside the strict bounds (4), non-finite, below the precision of intrinsic (pitfall 7) or
    above sigma_max -> NaN + RuntimeWarning + status code; valid neighbours are unaffected."""
    F, T, DF = 100.0, 0.5, 0.9
    K = np.array([90.0, 90.0, 110.0, 90.0, 100.0, 100.0, 100.0, 50.0, 100.0])
    cp = np.array([1, 1, -1, 1, 1, 1, 1, 1, 1])
    V = np.array(
        [
            DF * 10.0,  # = intrinsic
            DF * 10.0 - 1e-3,  # below intrinsic
            DF * 110.0,  # put >= DF K
            DF * 100.0,  # call >= DF F
            np.nan,
            np.inf,
            -1.0,
            DF * 50.0 * (1 + 4 * EPS),  # time value < 10 eps v (pitfall 7)
            float(bsm.price(F, 100.0, T, DF, 12.0, 1)),  # sigma = 12 > sigma_max = 10
        ]
    )
    with pytest.warns(RuntimeWarning, match="-> NaN"):
        sig, info = implied_vol(V, F, K, T, DF, cp, full_output=True)
    assert np.all(np.isnan(sig))
    np.testing.assert_array_equal(info["status"], [1, 1, 2, 2, 4, 4, 1, 1, 3])
    assert set(STATUS) == {0, 1, 2, 3, 4, 5}
    assert abs(implied_vol(V[-1], F, 100.0, T, DF, 1, sigma_max=20.0) - 12.0) < 1e-10
    mixed = np.array([float(bsm.price(F, 105.0, T, DF, 0.3, 1)), -1.0])
    with pytest.warns(RuntimeWarning):
        out = implied_vol(mixed, F, 105.0, T, DF, 1)
    assert abs(out[0] - 0.3) < 1e-12 and np.isnan(out[1])


def test_deep_otm_unresolved_tail_is_nan():
    """Phi(d2) below PHI_MIN at the root (ndtr flushes near d = -37.5): NaN + status 5, never a
    silent wrong sigma or sigma = 0 (D-013); just inside (z = -36) the inversion is accurate."""
    F, T, sig = 100.0, 5.0, 1.0
    for z, ok in ((36.0, True), (37.0, False), (38.5, False), (39.0, False)):
        for cp in (1, -1):
            K = F * np.exp(cp * z * sig * np.sqrt(T))  # OTM side for this cp
            V = bsm.price(F, K, T, 1.0, sig, cp)
            if ok:
                assert abs(implied_vol(V, F, K, T, 1.0, cp) - sig) < 1e-8
                continue
            with pytest.warns(RuntimeWarning, match="-> NaN"):
                s, info = implied_vol(V, F, K, T, 1.0, cp, full_output=True)
            assert np.isnan(s) and int(info["status"]) == (5 if V > 0 else 1)  # V == 0: v <= iota
    with pytest.warns(RuntimeWarning):  # beta underflows to a subnormal: no sigma ~ 0 output
        s, info = implied_vol(1e-320, F, 200.0, 1.0, 1.0, 1, full_output=True)
    assert np.isnan(s) and int(info["status"]) == 5


def test_brent_fallback_path(bsm_gate_grid):
    """max_iter = 1 forces the non-vectorised brentq fallback (eq. 15, D-011); same accuracy."""
    g = {k: (v[::3, ::3, ::4, :] if np.ndim(v) else v) for k, v in bsm_gate_grid.items()}
    V = bsm.price(g["F"], g["K"], g["T"], g["DF"], g["sigma"], -1)
    sig, info = implied_vol(V, g["F"], g["K"], g["T"], g["DF"], -1, max_iter=1, full_output=True)
    assert info["brent"].mean() > 0.9
    assert np.all(info["n_iter"][info["brent"]] == 1)
    assert np.abs(sig - g["sigma"]).max() < 1e-10


def test_shapes_and_scalars():
    assert N_MAX == 100
    K = np.array([90.0, 100.0, 110.0])[:, None]
    T = np.array([0.1, 1.0])[None, :]
    V = bsm.price(100.0, K, T, 0.95, 0.2, 1)
    sig = implied_vol(V, 100.0, K, T, 0.95, 1)
    assert sig.shape == (3, 2) and np.abs(sig - 0.2).max() < 1e-12
    out = implied_vol(float(V[1, 1]), 100.0, 100.0, 1.0, 0.95)
    assert np.ndim(out) == 0 and abs(out - 0.2) < 1e-12


@pytest.mark.parametrize(
    "bad",
    [{"F": 0.0}, {"K": np.nan}, {"T": -1.0}, {"DF": 0.0}, {"cp": 2}, {"sigma_max": 0.0}],
)
def test_invalid_market_inputs_raise(bad):
    args = {"V": 5.0, "F": 100.0, "K": 100.0, "T": 1.0, "DF": 0.97, "cp": 1} | bad
    with pytest.raises(ValueError):
        implied_vol(**args)
