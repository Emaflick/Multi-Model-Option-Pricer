"""Phase 2 Heston Monte Carlo (Euler-FT, QE, QE-M): API contract, exact identities, 3SE gates.

docs/ROADMAP.md Phase 2 and docs/derivations/heston-mc.md §5. All MC comparisons are in standard-
error units, |MC - ref| <= 3 SE (eq. 2), with seeds fixed a priori (never re-seeded to pass).
Gate: QE / QE-M vs COS at 2e5 paths and dt = 1/32 on four Feller-violating sets (D-018).
"""

import numpy as np
import pytest

from qpricer.engines import mc
from qpricer.engines.fourier import cos_price, lewis_price
from qpricer.models import bsm
from qpricer.models.heston import HestonParams

SEED = 20261107
FO = HestonParams(kappa=1.5768, theta=0.0398, xi=0.5751, rho=-0.5711, v0=0.0175)
GS = HestonParams(kappa=1.3253, theta=0.0354, xi=0.3877, rho=-0.7165, v0=0.0354)
AI = HestonParams(kappa=0.5, theta=0.04, xi=1.0, rho=-0.9, v0=0.04)
BK2 = HestonParams(kappa=2.0, theta=0.09, xi=1.0, rho=-0.3, v0=0.09)


def within_3se(est, se, ref):
    return np.abs(np.asarray(est) - ref) <= 3.0 * np.asarray(se)


# ------------------------------------------------------------------ API and exact identities
def test_api_contract_and_reproducibility():
    """Project convention: MC returns (price, stderr, n_paths, seed); same seed -> bit-identical."""
    K = np.array([80.0, 100.0, 120.0])
    out = mc.european_price(100.0, K, 1.0, 0.9, FO, -1, n_steps=8, n_paths=4000, seed=SEED)
    price, se, n, seed = out
    assert price.shape == se.shape == K.shape and n == 4000 and seed == SEED
    assert np.all(se > 0)
    again = mc.european_price(100.0, K, 1.0, 0.9, FO, -1, n_steps=8, n_paths=4000, seed=SEED)
    assert np.array_equal(price, again[0]) and np.array_equal(se, again[1])
    other = mc.european_price(100.0, K, 1.0, 0.9, FO, -1, n_steps=8, n_paths=4000, seed=SEED + 1)
    assert not np.array_equal(price, other[0])


def test_input_validation():
    rng = np.random.default_rng(SEED)
    with pytest.raises(ValueError):
        mc.heston_steps(FO, [0.5, 1.0], 100, rng, scheme="milstein")
    with pytest.raises(ValueError):
        mc.heston_steps(FO, [0.5, 0.5, 1.0], 100, rng)
    with pytest.raises(ValueError):
        mc.heston_steps(FO, [0.0, 1.0], 100, rng)
    with pytest.raises(ValueError):
        mc.heston_steps(FO, [1.0], 100, SEED)  # an int is not a Generator
    with pytest.raises(ValueError):
        mc.simulate(FO, mc.uniform_grid(1.0, 4), 100, rng, observe=[0.3])
    with pytest.raises(ValueError):
        mc.european_price(100.0, -1.0, 1.0, 1.0, FO, -1, n_steps=4, n_paths=100, seed=SEED)
    with pytest.raises(TypeError):  # validator F4: cp is required, no silent put default
        mc.european_price(100.0, 100.0, 1.0, 1.0, FO, n_steps=4, n_paths=100, seed=SEED)


def test_qe_m_domain_raises_for_positive_rho():
    """Eq. (13): for rho > 0, M(v_n) can be infinite at large dt; QE-M must raise, not fall back."""
    p = HestonParams(kappa=10.0, theta=0.04, xi=5.0, rho=0.9, v0=0.5)
    with pytest.raises(ValueError, match="QE-M"):
        mc.simulate(p, [2.0], 1000, np.random.default_rng(SEED), "qe-m")


def test_euler_warns_when_kappa_dt_large():
    """Pitfall 2: kappa dt >= 1 makes the Euler mean factor (1 - kappa dt)^n oscillate."""
    with pytest.warns(RuntimeWarning, match="kappa dt"):
        mc.heston_steps(FO, [1.0], 100, np.random.default_rng(SEED), "euler")


def test_step_generator_contract():
    """Path API for Phase 4: per-step (t, x, v, z), fresh arrays, draws [Z_v; Z_perp] of shape
    (2, N); `simulate` returns exactly the generator's states at the observation dates."""
    t = mc.uniform_grid(1.0, 8)
    steps = list(mc.heston_steps(FO, t, 500, np.random.default_rng(SEED), "qe-m"))
    assert [s[0] for s in steps] == list(t)
    assert all(s[3].shape == (2, 500) for s in steps)
    assert all(steps[i][1] is not steps[i + 1][1] for i in range(len(steps) - 1))
    assert not np.array_equal(steps[2][1], steps[3][1])  # earlier snapshot not overwritten
    x, v = mc.simulate(FO, t, 500, np.random.default_rng(SEED), "qe-m", observe=[t[2], t[7]])
    assert x.shape == (2, 500)
    assert np.array_equal(x[0], steps[2][1]) and np.array_equal(v[1], steps[7][2])


@pytest.mark.parametrize("scheme", mc.SCHEMES)
def test_on_path_parity_and_shape(scheme):
    """§5 exact on the same paths (1e-12): C - P = DF (mean F_T - K); MC calls non-increasing
    and convex in K (common random numbers)."""
    K = np.linspace(60.0, 150.0, 19)
    kw = dict(n_steps=16, n_paths=20_000, seed=SEED, scheme=scheme)
    C = mc.european_price(100.0, K, 1.0, 0.95, GS, 1, **kw)[0]
    P = mc.european_price(100.0, K, 1.0, 0.95, GS, -1, **kw)[0]
    x, _ = mc.simulate(GS, mc.uniform_grid(1.0, 16), 20_000, np.random.default_rng(SEED), scheme)
    FT_bar = (100.0 * np.exp(x[0])).mean()
    assert np.abs(C - P - 0.95 * (FT_bar - K)).max() < 1e-12
    assert np.all(np.diff(C) <= 1e-12) and np.all(np.diff(C, 2) >= -1e-12)


# ------------------------------------------------------------------ limits and moments
def test_xi_to_zero_discrete_limits():
    """Eq. (15), xi = 1e-4, rho = 0, v0 = 0.09 != theta = 0.04, kappa = 1.5, T = 1, dt = 1/4:
    QE vs Black at Sigma_Q, Euler vs Black at Sigma_E, within 3SE. Power: Euler is > 3SE away
    from Black at the exact Sigma_bar (its discrete limit differs)."""
    p = HestonParams(kappa=1.5, theta=0.04, xi=1e-4, rho=0.0, v0=0.09)
    T, M, K = 1.0, 4, np.array([80.0, 100.0, 125.0])
    dt, E, eT = T / M, np.exp(-p.kappa * T / M), np.exp(-p.kappa * T)
    sig_q = p.theta * T + (p.v0 - p.theta) * dt / 2 * (1 + E) * (1 - eT) / (1 - E)
    sig_e = p.theta * T + (p.v0 - p.theta) * (1 - (1 - p.kappa * dt) ** M) / p.kappa
    sig_x = p.theta * T + (p.v0 - p.theta) * (1 - eT) / p.kappa
    kw = dict(n_steps=M, n_paths=100_000, seed=SEED)
    for scheme, S in (("qe", sig_q), ("qe-m", sig_q), ("euler", sig_e)):
        v, se, _, _ = mc.european_price(100.0, K, T, 1.0, p, -1, scheme=scheme, **kw)
        assert np.all(within_3se(v, se, bsm.price(100.0, K, T, 1.0, np.sqrt(S / T), -1)))
    v, se, _, _ = mc.european_price(100.0, 100.0, T, 1.0, p, -1, scheme="euler", **kw)
    assert abs(v - bsm.price(100.0, 100.0, T, 1.0, np.sqrt(sig_x / T), -1)) > 3 * se


@pytest.mark.parametrize("p, T, M", [(FO, 1.0, 4), (AI, 10.0, 10), (BK2, 5.0, 5)])
def test_variance_moments(p, T, M):
    """Eq. (14): QE reproduces E v_T and E v_T^2 exactly for any dt (within 3SE, coarse grids);
    Euler-FT mean lies below the Euler-ODE mean theta + (v0 - theta)(1 - kappa dt)^M
    ([LKD06] Lemma 1; one-sided, 3SE)."""
    E = np.exp(-p.kappa * T)
    mean = p.theta + (p.v0 - p.theta) * E
    var = p.v0 * p.xi**2 * E * (1 - E) / p.kappa + p.theta * p.xi**2 * (1 - E) ** 2 / (2 * p.kappa)
    t = mc.uniform_grid(T, M)
    _, v = mc.simulate(p, t, 200_000, np.random.default_rng(SEED), "qe")
    m1, s1 = mc.mc_estimate(v[0])
    m2, s2 = mc.mc_estimate(v[0] ** 2)
    assert within_3se(m1, s1, mean) and within_3se(m2, s2, var + mean**2)
    tf = mc.uniform_grid(T, 32 * int(T))
    _, ve = mc.simulate(p, tf, 200_000, np.random.default_rng(SEED), "euler")
    me, se = mc.mc_estimate(ve[0])
    assert me <= p.theta + (p.v0 - p.theta) * (1 - p.kappa * T / tf.size) ** tf.size + 3 * se


@pytest.mark.parametrize("scheme", ["euler", "qe-m"])
@pytest.mark.parametrize("p, T, M", [(FO, 1.0, 4), (AI, 10.0, 10), (BK2, 5.0, 20)])
def test_forward_is_martingale(scheme, p, T, M):
    """§5: mean F_T = F within 3SE for Euler-FT (eq. 6) and QE-M (eq. 12) at coarse dt."""
    x, _ = mc.simulate(p, mc.uniform_grid(T, M), 200_000, np.random.default_rng(SEED), scheme)
    m, se = mc.mc_estimate(100.0 * np.exp(x[0]))
    assert within_3se(m, se, 100.0)


def test_plain_qe_forward_defect_power():
    """§5 diagnostic / power check: without the martingale correction, plain QE at dt = 1 on FO
    misses F by > 3SE (measured +12.8 SE), so the martingale test above can detect a defect."""
    x, _ = mc.simulate(FO, [1.0], 200_000, np.random.default_rng(SEED), "qe")
    m, se = mc.mc_estimate(100.0 * np.exp(x[0]))
    assert not within_3se(m, se, 100.0)


# ------------------------------------------------------------------ gate: QE vs COS (slow)
GATE_CASES = {  # name: (params, T, F, DF, COS N, COS L); all Feller-violating (nu_F < 1)
    "FO": (FO, 1.0, 100.0, 1.0, 256, 18.0),
    "GS": (GS, 1.0, 100.0, 1.0, 256, 18.0),
    "BK2": (BK2, 5.0, 100.0 * np.exp(0.25), np.exp(-0.25), 256, 18.0),
    "AI": (AI, 10.0, 100.0, 1.0, 4096, 40.0),  # T = 10 stress: wider range (D-017)
}


@pytest.mark.slow
@pytest.mark.parametrize("scheme", ["qe", "qe-m"])
@pytest.mark.parametrize("name", list(GATE_CASES))
def test_gate_qe_vs_cos(name, scheme):
    """ROADMAP gate: QE MC vs COS within 3SE at 2e5 paths, dt = 1/32, puts K in {70, 100, 140}.
    Reference error checked against Lewis (< 1e-6, far below SE ~ 1e-2)."""
    p, T, F, DF, N, L = GATE_CASES[name]
    assert p.feller < 1.0
    K = np.array([70.0, 100.0, 140.0])
    ref = cos_price(F, K, T, DF, p, -1, N=N, L=L)
    assert np.abs(ref - lewis_price(F, K, T, DF, p, -1)).max() < 1e-6
    i = list(GATE_CASES).index(name)
    price, se, n, _ = mc.european_price(
        F, K, T, DF, p, -1, n_steps=int(32 * T), n_paths=200_000, seed=SEED + i, scheme=scheme
    )
    assert n == 200_000
    assert np.all(within_3se(price, se, ref)), (price - ref) / se


@pytest.mark.slow
def test_gate_euler_bias_shrinks_with_dt():
    """ROADMAP gate (test form of report/figures/heston_euler_bias.png): Euler-FT ATM call bias
    on A-I (T = 10) at dt in {1, 1/4, 1/16}, 2e5 paths: > 3SE at dt = 1 and strictly decreasing."""
    ref = cos_price(100.0, 100.0, 10.0, 1.0, AI, 1, N=4096, L=40.0)
    bias, ses = [], []
    for j, M in enumerate((10, 40, 160)):
        kw = dict(n_steps=M, n_paths=200_000, seed=SEED + j, scheme="euler")
        v, se, _, _ = mc.european_price(100.0, 100.0, 10.0, 1.0, AI, 1, **kw)
        bias.append(v - ref)
        ses.append(se)
    assert bias[0] > 3 * ses[0]
    assert bias[0] > bias[1] > bias[2]
