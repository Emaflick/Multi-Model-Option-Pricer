"""Phase 4 Asians: discrete geometric closed form (Kemna-Vorst), arithmetic MC, geometric CV.

docs/ROADMAP.md Phase 4 and docs/derivations/asian.md §5. MC comparisons |MC - ref| <= 3 SE
(heston-mc.md eq. 2), seeds SEED + offset fixed a priori. Gate: geometric-Asian MC vs Kemna-Vorst
within 3 SE; the CV variance-reduction factor is reported by scripts/phase4_gates.py.
"""

import numpy as np
import pandas as pd
import pytest

from qpricer.engines.analytic import geometric_asian_moments, geometric_asian_price
from qpricer.engines.exotic_mc import (
    asian_averages,
    asian_mc,
    barrier_mc,
    control_variate,
    fixings,
)
from qpricer.engines.fourier import cos_price
from qpricer.market import forward_curve
from qpricer.models import bsm
from qpricer.models.heston import HestonParams

SEED = 20261208
GS = HestonParams(kappa=1.3253, theta=0.0354, xi=0.3877, rho=-0.7165, v0=0.0354)
MARKET = "data/processed/market_20261106_2213.csv"
# [FM08] Fusai-Meucci (2008) Tables 4 / 6, Gaussian model, t_k = k/n, k = 0..n (asian.md §5)
FM_R, FM_SIG, FM_K = 0.0367, 0.17801, np.array([90.0, 100.0, 110.0])
FM_GEO = {12: (11.66744, 4.703634, 1.251243), 50: (11.7076, 4.769273, 1.299134),
          250: (11.71856, 4.787024, 1.312165)}  # fmt: skip
FM_GEO_CTS = (11.72135, 4.791541, 1.315486)
FM_ARI = {12: (11.90497, 4.88210, 1.36314), 50: (11.93301, 4.93736, 1.40264),
          250: (11.94068, 4.95233, 1.41351)}  # fmt: skip


def within_3se(est, se, ref, extra=0.0):
    return np.abs(np.asarray(est) - ref) <= 3.0 * np.asarray(se) + extra


def fm08(n):
    t = np.arange(n + 1) / n
    return t, 100.0 * np.exp(FM_R * t), np.exp(-FM_R)


# ------------------------------------------------------------------ closed form (exact)
def test_geometric_reduces_to_black():
    """§5: n = 1, t_1 = T equals bsm eq. (1); n = 1, t_1 < T equals Black on (F_1, sqrt V(t_1));
    all fixings at one date give sigma_G^2 = V(t), F_G = F(0, t). rel 1e-12."""
    K = np.array([70.0, 100.0, 140.0])
    for cp in (1, -1):
        v = geometric_asian_price([103.0], [0.3**2 * 2.0], K, 0.9, cp)
        assert np.allclose(v, bsm.price(103.0, K, 2.0, 0.9, 0.3, cp), rtol=1e-12, atol=0)
        v = geometric_asian_price([101.0], [0.3**2 * 0.7], K, 0.9, cp)  # paid at T > t_1
        assert np.allclose(v, bsm.price(101.0, K, 0.7, 0.9, 0.3, cp), rtol=1e-12, atol=0)
    FG, s2 = geometric_asian_moments([98.0] * 5, [0.06] * 5)
    assert abs(FG / 98.0 - 1) < 1e-14 and abs(s2 / 0.06 - 1) < 1e-14


def test_three_forms_of_eq8_and_closed_forms_11_12():
    """Eq. (8): double sum, (2n - 2i + 1) weights and Abel form agree; (11) (t_0 excluded) and
    (12) (t_0 included) match (7)-(9) on equally spaced dates; eq. (13) finite-n gaps. rel 1e-12."""
    rng = np.random.default_rng(SEED)
    t = np.sort(rng.uniform(0.0, 3.0, 17))
    V = np.cumsum(rng.uniform(0.0, 0.05, 17))  # arbitrary deterministic variance curve
    n = t.size
    double = np.minimum.outer(V, V).sum() / n**2
    weights = np.sum((2 * n - 2 * np.arange(1, n + 1) + 1) * V) / n**2
    s2 = geometric_asian_moments(np.full(n, 100.0), V)[1]
    assert abs(double / s2 - 1) < 1e-12 and abs(weights / s2 - 1) < 1e-12
    sig, T, S0, b = 0.3, 2.0, 100.0, 0.03
    for n in (1, 12, 50, 250):
        ti = np.arange(1, n + 1) * T / n  # (11)
        FG, s2 = geometric_asian_moments(S0 * np.exp(b * ti), sig**2 * ti)
        s2_11 = sig**2 * T * (n + 1) * (2 * n + 1) / (6 * n**2)
        FG_11 = S0 * np.exp(b * T * (n + 1) / (2 * n) - sig**2 * T * (n**2 - 1) / (12 * n**2))
        assert abs(s2 / s2_11 - 1) < 1e-12 and abs(FG / FG_11 - 1) < 1e-12
        assert abs((s2 - sig**2 * T / 3) / (sig**2 * T * (3 * n + 1) / (6 * n**2)) - 1) < 1e-10
        tk = np.arange(n + 1) * T / n  # (12)
        FG, s2 = geometric_asian_moments(S0 * np.exp(b * tk), sig**2 * tk)
        s2_12 = sig**2 * T * (2 * n + 1) / (6 * (n + 1))
        FG_12 = S0 * np.exp(0.5 * b * T - sig**2 * T * (n + 2) / (12 * (n + 1)))
        assert abs(s2 / s2_12 - 1) < 1e-12 and abs(FG / FG_12 - 1) < 1e-12
        assert abs((s2 - sig**2 * T / 3) / (-(sig**2) * T / (6 * (n + 1))) - 1) < 1e-10


def test_geometric_parity_and_zero_vol():
    """Eq. (10) parity C_G - P_G = DF (F_G - K); sigma -> 0: V_G = DF (theta (Gbar_F - K))^+."""
    F_fix, V = np.array([100.0, 101.0, 103.0, 104.0]), 0.04 * np.array([0.25, 0.5, 0.75, 1.0])
    K = np.linspace(60.0, 150.0, 19)
    FG, _ = geometric_asian_moments(F_fix, V)
    C, P = (geometric_asian_price(F_fix, V, K, 0.95, cp) for cp in (1, -1))
    assert np.abs(C - P - 0.95 * (FG - K)).max() < 1e-12
    G0 = np.exp(np.log(F_fix).mean())
    for cp in (1, -1):
        v0 = geometric_asian_price(F_fix, np.zeros(4), K, 0.95, cp)
        assert np.abs(v0 - 0.95 * np.maximum(cp * (G0 - K), 0)).max() < 1e-12


def test_fm08_geometric_table4():
    """§5 published: [FM08] Table 4 geometric (Carr-Madan FFT) for n = 12, 50, 250 and the
    continuous limit (13), abs 3e-4 (tolerance from the note's hand check, unrounded sigma)."""
    for n, ref in FM_GEO.items():
        t, F_fix, DF = fm08(n)
        v = geometric_asian_price(F_fix, FM_SIG**2 * t, FM_K, DF, 1)
        assert np.abs(v - ref).max() <= 3e-4, (n, v - ref)
    FGc = 100.0 * np.exp(0.5 * (FM_R - FM_SIG**2 / 6))  # eq. (13), T = 1
    vc = bsm.price(FGc, FM_K, 1.0, np.exp(-FM_R), FM_SIG / np.sqrt(3.0), 1)
    assert np.abs(vc - FM_GEO_CTS).max() <= 3e-4


def test_moments_input_validation():
    with pytest.raises(ValueError):
        geometric_asian_moments([100.0, 100.0], [0.04, 0.02])  # V decreasing
    with pytest.raises(ValueError):
        geometric_asian_moments([100.0, -1.0], [0.01, 0.02])


# ------------------------------------------------------------------ fixing-forward builder
def test_forward_curve_builder():
    """asian.md pitfall 3 / autocallable.md §1 (D-025): exact at kept expiries, ln F and ln DF
    linear between them, continuous flat extrapolation beyond T_max, raises before T_min; n = 1
    geometric Asian at t_1 < T from the builder equals Black-76 on F(0, t_1)."""
    tab = pd.read_csv(MARKET)
    kept = tab[tab.kept.astype(bool)].sort_values("T")
    F, DF = forward_curve(tab, kept["T"])
    assert np.allclose(F, kept.F, rtol=1e-12, atol=0) and np.allclose(DF, kept.DF, rtol=1e-12)
    T = kept["T"].to_numpy()
    mid = 0.5 * (T[25] + T[26])
    Fm, Dm = forward_curve(tab, mid)
    assert abs(np.log(Fm[0]) - 0.5 * np.log(kept.F.iloc[25] * kept.F.iloc[26])) < 1e-12
    assert abs(np.log(Dm[0]) - 0.5 * np.log(kept.DF.iloc[25] * kept.DF.iloc[26])) < 1e-12
    Fe, De = forward_curve(tab, [T[-1], T[-1] + 1e-9, 2.5])
    assert abs(Fe[1] / Fe[0] - 1) < 1e-9 and Fe[2] > Fe[0] and De[2] < De[0]
    with pytest.raises(ValueError):
        forward_curve(tab, 0.5 * T[0])
    t1, K = 0.5, np.array([7000.0, 7500.0, 8000.0])
    (F1,), (D1,) = forward_curve(tab, t1)
    v = geometric_asian_price([F1], [0.18**2 * t1], K, D1, 1)
    assert np.allclose(v, bsm.price(F1, K, t1, D1, 0.18, 1), rtol=1e-12)


# ------------------------------------------------------------------ MC: common paths (exact)
def test_common_path_identities():
    """§5 exact on common MC paths: Y >= X for calls (rel 1e-12); parity (18) C - P = DF (mean A
    - K); calls non-increasing and convex in K; VR = 1/(1 - rho_hat^2) to 1e-10."""
    t = np.arange(1, 13) / 12
    F_fix, DF = 100.0 * np.exp(0.02 * t), 0.97
    A, G = asian_averages(t, F_fix, 0.3, 50_000, np.random.default_rng(SEED + 1))
    assert np.all(A >= G * (1 - 1e-12))
    K = np.linspace(70.0, 140.0, 15)
    Y, Yp = (DF * np.maximum(cp * (A - K[:, None]), 0) for cp in (1, -1))
    assert np.abs(Y.mean(1) - Yp.mean(1) - DF * (A.mean() - K)).max() < 1e-12
    C = Y.mean(1)
    assert np.all(np.diff(C) <= 1e-12) and np.all(np.diff(C, 2) >= -1e-12)
    X = DF * np.maximum(G - K[:, None], 0)
    _, _, _, vr = control_variate(Y, X, np.zeros(K.size))
    rho = np.array([np.corrcoef(y, x)[0, 1] for y, x in zip(Y, X, strict=True)])
    assert np.abs(vr * (1 - rho**2) - 1).max() < 1e-10


def test_single_fixing_equals_vanilla_mc_same_draws():
    """§5: n = 1, t_1 = T: the arithmetic Asian MC is the vanilla MC on the same draws."""
    K = np.array([90.0, 100.0, 120.0])
    a = asian_mc([1.5], [100.0], K, 0.93, 0.25, 1, n_paths=10_000, seed=SEED, kind="arithmetic")
    v = barrier_mc(
        100.0, K, 1e-3, 1.5, 0.93, 0.25, 1, 1, "vanilla", n_steps=1, n_paths=10_000, seed=SEED
    )
    assert np.array_equal(a[0], v[0]) and a[2:] == (10_000, SEED)
    assert np.allclose(a[1], v[1], rtol=1e-13, atol=0)  # DF inside vs outside the std


def test_api_and_validation():
    t, F_fix = [0.0, 0.5, 1.0], [99.0, 100.0, 101.0]  # t_1 = 0: today's fixing (S_0)
    price, se, n, seed = asian_mc(t, F_fix, [95.0, 105.0], 0.95, 0.2, -1, n_paths=1000, seed=7)
    assert price.shape == se.shape == (2,) and n == 1000 and seed == 7
    with pytest.raises(ValueError, match="Heston"):
        asian_mc(t, F_fix, 100.0, 0.95, GS, 1, n_paths=1000, seed=7, kind="cv")
    with pytest.raises(ValueError):
        asian_mc(t, F_fix, 100.0, 0.95, 0.2, 1, n_paths=1000, seed=7, kind="median")
    with pytest.raises(ValueError):
        control_variate(np.ones(10), np.zeros(10), 0.0)


def test_heston_default_grid_is_substepped():
    """asian.md eq. (4), D-030 (validator F2): under Heston the default sub-step cap is 1/32, so a
    default call equals max_dt = 1/32 bitwise and differs from one QE-M step per quarter; under
    BSM max_dt is ignored (exact steps at the dates)."""
    t, F_fix, K = np.arange(1, 5) / 4, np.full(4, 100.0), np.array([90.0, 100.0, 110.0])
    kw = dict(n_paths=2000, seed=SEED + 2, kind="arithmetic")
    d = asian_mc(t, F_fix, K, 1.0, GS, 1, **kw)
    assert np.array_equal(d[0], asian_mc(t, F_fix, K, 1.0, GS, 1, max_dt=1 / 32, **kw)[0])
    assert not np.array_equal(d[0], asian_mc(t, F_fix, K, 1.0, GS, 1, max_dt=None, **kw)[0])
    b = asian_mc(t, F_fix, K, 1.0, 0.2, 1, **kw)[0]
    assert np.array_equal(b, asian_mc(t, F_fix, K, 1.0, 0.2, 1, max_dt=None, **kw)[0])


def test_eq19_bounds_pathwise_and_in_expectation():
    """asian.md eq. (19) (validator F7a). Pathwise forms on common BSM paths (abs 1e-12 at F ~ 100):
    C_G <= C_A <= C_G + DF mean(A - G), C_A >= DF (mean A - K)^+, C_A <= DF/n sum_i mean(S_i - K)^+,
    P_G - DF mean(A - G) <= P_A <= P_G. In expectation (closed forms vs the CV price, 3 SE):
    max{DF (Abar_F - K)^+, C_G} <= C_A <= min{C_G + DF (Abar_F - F_G), DF/n sum_i c_B(F_i, K)},
    P_G - DF (Abar_F - F_G) <= P_A <= P_G. sigma = 0.3, n = 12 monthly, 4% carry, K = 70..140."""
    t, sig, DF, N = np.arange(1, 13) / 12, 0.3, 0.97, 50_000
    F_fix, K = 100.0 * np.exp(0.04 * t), np.linspace(70.0, 140.0, 15)
    x, _ = fixings(sig, t, N, np.random.default_rng(SEED + 3))
    S = F_fix[:, None] * np.exp(x)
    A, G = asian_averages(t, F_fix, sig, N, np.random.default_rng(SEED + 3))  # same draws
    assert np.allclose(A, S.mean(0), rtol=1e-13, atol=0)
    m = lambda a: DF * a.mean(-1)  # noqa: E731
    CA, CG = m(np.maximum(A - K[:, None], 0)), m(np.maximum(G - K[:, None], 0))
    PA, PG = m(np.maximum(K[:, None] - A, 0)), m(np.maximum(K[:, None] - G, 0))
    strip = m(np.maximum(S[None] - K[:, None, None], 0).mean(1))
    gap, tol = m(A - G), 1e-12
    assert np.all(CG <= CA + tol) and np.all(CA <= CG + gap + tol) and np.all(CA <= strip + tol)
    assert np.all(CA >= DF * np.maximum(A.mean() - K, 0) - tol)
    assert np.all(PG - gap <= PA + tol) and np.all(PA <= PG + tol)
    FG, s2 = geometric_asian_moments(F_fix, sig**2 * t)
    cG, pG = (geometric_asian_price(F_fix, sig**2 * t, K, DF, cp) for cp in (1, -1))
    c_strip = DF * np.mean(
        [bsm.price(Fi, K, ti, 1.0, sig, 1) for Fi, ti in zip(F_fix, t, strict=True)], 0
    )
    cA, se_c, _, _ = asian_mc(t, F_fix, K, DF, sig, 1, n_paths=N, seed=SEED + 4)
    pA, se_p, _, _ = asian_mc(t, F_fix, K, DF, sig, -1, n_paths=N, seed=SEED + 4)
    lo, hi = (
        np.maximum(DF * np.maximum(F_fix.mean() - K, 0), cG),
        np.minimum(cG + DF * (F_fix.mean() - FG), c_strip),
    )
    assert np.all(cA >= lo - 3 * se_c) and np.all(cA <= hi + 3 * se_c)
    assert np.all(pA >= pG - DF * (F_fix.mean() - FG) - 3 * se_p) and np.all(pA <= pG + 3 * se_p)


# ------------------------------------------------------------------ MC: statistical (3 SE)
GATE = [(s, T, n) for s in (0.2, 0.5) for T in (1.0, 3.0) for n in (1, 12, 52)]


@pytest.mark.slow
def test_gate_geometric_mc_vs_kemna_vorst():
    """ROADMAP gate: plain MC of X = DF (G - K)^+ under BSM (exact at the dates) vs eq. (10), 2e5
    paths, sigma in {0.2, 0.5}, T in {1, 3}, n in {1, 12, 52} (t_i = iT/n), K/F in {0.8, 1, 1.2},
    F_i = 100, DF = 1, seed SEED + 10 + j per path set; plus the [FM08] curve at n = 12 (t_0 incl.).
    Also mean G vs F_G and mean A vs Abar_F (3 SE)."""
    K = 100.0 * np.array([0.8, 1.0, 1.2])
    cases = [(np.arange(1, n + 1) * T / n, np.full(n, 100.0), 1.0, s) for s, T, n in GATE]
    t, F_fix, DF = fm08(12)
    cases.append((t, F_fix, DF, FM_SIG))
    for j, (t, F_fix, DF, s) in enumerate(cases):
        V = s**2 * t
        A, G = asian_averages(t, F_fix, s, 200_000, np.random.default_rng(SEED + 10 + j))
        v, se = (
            DF * np.maximum(G - K[:, None], 0).mean(1),
            DF * np.maximum(G - K[:, None], 0).std(1, ddof=1) / np.sqrt(2e5),
        )
        assert np.all(within_3se(v, se, geometric_asian_price(F_fix, V, K, DF, 1))), (j, v)
        FG = geometric_asian_moments(F_fix, V)[0]
        assert within_3se(G.mean(), G.std(ddof=1) / np.sqrt(2e5), FG)
        assert within_3se(A.mean(), A.std(ddof=1) / np.sqrt(2e5), F_fix.mean())


@pytest.mark.slow
def test_cv_vs_plain_independent_seeds():
    """§5: CV arithmetic vs plain arithmetic on independent seeds, |diff| <= 3 sqrt(SE1^2 + SE2^2),
    sigma = 0.3, T = 1, n = 12 monthly, K in {80, 100, 120}, 2e5 paths each."""
    t, K = np.arange(1, 13) / 12, np.array([80.0, 100.0, 120.0])
    F_fix = 100.0 * np.exp(0.03 * t)
    cv, se_cv, _, _ = asian_mc(
        t, F_fix, K, 0.97, 0.3, 1, n_paths=200_000, seed=SEED + 30, kind="cv"
    )
    pl, se_pl, _, _ = asian_mc(
        t, F_fix, K, 0.97, 0.3, 1, n_paths=200_000, seed=SEED + 31, kind="arithmetic"
    )
    assert np.all(np.abs(cv - pl) <= 3 * np.sqrt(se_cv**2 + se_pl**2))
    assert np.all(se_cv < se_pl / 5)


@pytest.mark.slow
def test_fm08_arithmetic_table6_cv():
    """§5 published: [FM08] Table 6 arithmetic (quadrature) vs CV MC, 2e5 paths, n = 12, 50, 250,
    |MC - ref| <= 3 SE + 3e-4 (T = 1 assumed for Table 6, as the note states)."""
    for k, (n, ref) in enumerate(FM_ARI.items()):
        t, F_fix, DF = fm08(n)
        v, se, _, _ = asian_mc(t, F_fix, FM_K, DF, FM_SIG, 1, n_paths=200_000, seed=SEED + 40 + k)
        assert np.all(within_3se(v, se, ref, 3e-4)), (n, v - ref, se)


@pytest.mark.slow
def test_heston_geometric_small_xi_and_bounds():
    """§5 Heston: (i) xi = 1e-4, rho = 0: QE-M geometric MC vs (10) with V(t_i) = Sigma_Q(t_i)
    (heston-mc eq. 15, uniform grid dt = 1/48 with the monthly dates on nodes), 3 SE.
    (ii) GS set: mean A = Abar_F (QE-M martingale) and C_A <= DF/n sum_i COS(F_i, K, t_i) + 3 SE."""
    t, K, DF = np.arange(1, 13) / 12, np.array([90.0, 100.0, 110.0]), 0.96
    p = HestonParams(kappa=1.5, theta=0.04, xi=1e-4, rho=0.0, v0=0.09)
    dt, E = 1 / 48, np.exp(-1.5 / 48)
    SQ = p.theta * t + (p.v0 - p.theta) * dt / 2 * (1 + E) * (1 - np.exp(-p.kappa * t)) / (1 - E)
    F_fix = 100.0 * np.exp(0.02 * t)
    v, se, _, _ = asian_mc(
        t, F_fix, K, DF, p, 1, n_paths=200_000, seed=SEED + 50, kind="geometric", max_dt=dt
    )
    assert np.all(within_3se(v, se, geometric_asian_price(F_fix, SQ, K, DF, 1))), (v, se)
    A, _ = asian_averages(t, F_fix, GS, 200_000, np.random.default_rng(SEED + 51), max_dt=1 / 32)
    assert within_3se(A.mean(), A.std(ddof=1) / np.sqrt(2e5), F_fix.mean())
    Y = DF * np.maximum(A - K[:, None], 0)
    upper = DF * np.mean(
        [cos_price(Fi, K, ti, 1.0, GS, 1) for Fi, ti in zip(F_fix, t, strict=True)], 0
    )
    assert np.all(Y.mean(1) <= upper + 3 * Y.std(1, ddof=1) / np.sqrt(2e5))
