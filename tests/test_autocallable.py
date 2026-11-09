"""Phase 4 mono-asset autocallable: cash flows, closed-form oracles, MC, BSM-vs-Heston attribution.

docs/ROADMAP.md Phase 4 and docs/derivations/autocallable.md §5. MC comparisons |MC - ref| <= 3 SE
(heston-mc.md eq. 2), seeds SEED + offset fixed a priori. The model-gap numbers (eq. 19) come from
scripts/autocall_compare.py, which this module loads, with the illustrative GS Heston set (D-029).
"""

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ndtr

from qpricer.engines.analytic import autocall_nocall, autocall_two_dates, ratio_marginals
from qpricer.engines.exotic_mc import autocall_mc, autocall_pv, fixings, gbm_paths, obs_steps
from qpricer.engines.fourier import cos_price
from qpricer.models import bsm
from qpricer.models.heston import HestonParams, total_variance
from qpricer.products import Autocall, autocall_flows

SEED = 20261408
GS = HestonParams(kappa=1.3253, theta=0.0354, xi=0.3877, rho=-0.7165, v0=0.0354)
TQ = tuple(np.arange(1, 7) / 4)  # AC-Q dates
SHEET = Autocall(TQ, 0.08, 1.0, 0.7, 0.6, True)
F_Q = np.exp(0.04 * np.asarray(TQ))  # f_i with 4% carry (synthetic market)
DF_Q = np.exp(-0.05 * np.asarray(TQ))


def within_3se(est, se, ref):
    return np.abs(np.asarray(est) - ref) <= 3.0 * np.asarray(se)


def bs_paths(sigma, t, n, seed):
    x, _ = fixings(sigma, t, n, np.random.default_rng(seed))
    return x


# ------------------------------------------------------------------ hand paths (§5)
def test_hand_paths():
    """§5 (a)-(e), n = 4 quarterly, c = 8% (c_i = 0.02), B_ac = 1, B_c = 0.7, B_p = 0.6, memory."""
    sh = Autocall((0.25, 0.5, 0.75, 1.0), 0.08, 1.0, 0.7, 0.6, True)
    X = np.array([
        [1.05, 0.65, 0.90, 0.90, 0.90],   # (a) called at t_1
        [0.50, 0.80, 0.95, 0.50, 0.95],   # (b) X_1 < B_c missed, paid at t_2 with memory
        [0.40, 1.01, 0.85, 0.55, 0.99],   # (b) called at t_3
        [0.30, 0.50, 0.65, 0.40, 1.20],   # (c) B_p <= X_n < B_c, (d) X_n < B_p, (e) X_n >= B_ac
    ])  # fmt: skip
    C, R, chi, alive = autocall_flows(X, sh)
    exp_C = np.array([
        [0.02, 0.00, 0.02, 0.02, 0.02],
        [0.00, 0.04, 0.02, 0.00, 0.02],
        [0.00, 0.02, 0.02, 0.00, 0.02],
        [0.00, 0.00, 0.00, 0.00, 0.02],
    ])  # fmt: skip
    exp_R = np.zeros((4, 5))
    exp_R[0, 0], exp_R[2, 1], exp_R[3, 2:] = 1.0, 1.0, [1.0, 0.4, 1.0]
    assert np.allclose(C, exp_C, rtol=0, atol=1e-15) and np.allclose(R, exp_R, rtol=0, atol=1e-15)
    assert chi[0, 0] and chi[2, 1] and chi.sum() == 2 and list(alive) == [0, 0, 1, 1, 1]
    nomem = autocall_flows(X, replace(sh, memory=False))[0]
    assert abs(nomem[1, 1] - 0.02) < 1e-15  # without memory only c_2 at t_2


# ------------------------------------------------------------------ exact on common paths (§5)
def test_exact_identities_common_paths():
    """§5 (1e-12): sum p_i + mean A_n = 1; PV(c') - PV(c) = (c' - c) b; 0 <= PV <= max DF +
    sum DF c; pathwise monotone in B_c, B_p (non-increasing) and c (non-decreasing); memory >= no
    memory, equal at B_c = 0; B_ac = 0: PV = DF_1 (1 + c_1); B_ac = inf, B_c = B_p = 0: bond."""
    X = F_Q[:, None] * np.exp(bs_paths(0.25, TQ, 20_000, SEED))
    _, _, chi, alive = autocall_flows(X, SHEET)
    assert abs(chi[:-1].mean(1).sum() + alive.mean() - 1.0) < 1e-12 and not chi[-1].any()
    pv = autocall_pv(X, SHEET, DF_Q)
    b = autocall_pv(X, replace(SHEET, c=1.0), DF_Q) - autocall_pv(X, replace(SHEET, c=0.0), DF_Q)
    assert np.abs(autocall_pv(X, replace(SHEET, c=0.11), DF_Q) - pv - 0.03 * b).max() < 1e-12
    assert pv.min() >= 0 and pv.max() <= DF_Q.max() + DF_Q @ SHEET.coupons + 1e-12
    base = replace(SHEET, B_p=0.5)
    pv_bc = [autocall_pv(X, replace(base, B_c=bc), DF_Q) for bc in (0.5, 0.7, 0.9)]
    pv_bp = [autocall_pv(X, replace(base, B_p=bp), DF_Q) for bp in (0.0, 0.3, 0.5)]
    pv_c = [autocall_pv(X, replace(SHEET, c=c), DF_Q) for c in (0.0, 0.05, 0.1)]
    assert all(np.all(np.diff(np.stack(v), axis=0) <= 1e-15) for v in (pv_bc, pv_bp))
    assert np.all(np.diff(np.stack(pv_c), axis=0) >= -1e-15)
    assert np.all(pv >= autocall_pv(X, replace(SHEET, memory=False), DF_Q) - 1e-15)
    z0 = replace(SHEET, B_c=0.0, B_p=0.0)
    assert np.array_equal(autocall_pv(X, z0, DF_Q), autocall_pv(X, replace(z0, memory=False), DF_Q))
    call0 = Autocall(TQ, 0.08, 0.0, 0.0, 0.0, True)
    assert np.abs(autocall_pv(X, call0, DF_Q) - DF_Q[0] * (1 + call0.coupons[0])).max() < 1e-12
    bond = Autocall(TQ, 0.08, np.inf, 0.0, 0.0, False)
    assert np.abs(autocall_pv(X, bond, DF_Q) - DF_Q @ bond.coupons - DF_Q[-1]).max() < 1e-12


def test_zero_vol_term_sheet_by_hand():
    """Eq. (13): sigma = 0, X_i = f_i; f = (0.98, 0.99, 1.01, 1.02) is called at t_3: V = DF_1 c_1 +
    DF_2 c_2 + DF_3 (c_3 + 1)."""
    sh = Autocall((0.25, 0.5, 0.75, 1.0), 0.08, 1.0, 0.7, 0.6, True)
    f, DF = np.array([0.98, 0.99, 1.01, 1.02]), np.array([0.99, 0.98, 0.97, 0.96])
    v, se, n, seed = autocall_mc(sh, f, DF, 0.0, n_paths=100, seed=SEED)
    assert abs(v - (0.99 * 0.02 + 0.98 * 0.02 + 0.97 * 1.02)) < 1e-15 and se < 1e-15
    assert (n, seed) == (100, SEED)


def test_input_validation():
    with pytest.raises(ValueError):
        Autocall((0.5, 0.25), 0.08)
    with pytest.raises(ValueError):
        Autocall((0.5, 1.0), 0.08, B_ac=1.0, B_c=1.2, B_p=0.6)
    with pytest.raises(ValueError):
        autocall_flows(np.ones((3, 10)), Autocall((0.5, 1.0), 0.08))
    with pytest.raises(ValueError):
        autocall_two_dates(SHEET, F_Q, DF_Q, 0.04 * np.asarray(TQ))


# ------------------------------------------------------------------ closed forms (exact)
def test_closed_form_subcases_and_consistency():
    """Eq. (11) sub-cases: B_c = 0 -> sum DF c + DF_n - DIP_E; B_p = 0 too -> sum DF c + DF_n;
    DIP_E (8) = DF_n [p_n(B_p) + (1 - B_p) D^-_n(B_p)] by Black; eq. (12) at B_ac -> inf, m = 0
    equals (11); V^nc(sigma) strictly decreasing (§3.4). abs 1e-12 (quad: 1e-10)."""
    t, w = np.asarray(TQ), 0.3**2 * np.asarray(TQ)
    sh = replace(SHEET, B_c=0.0)
    s = np.sqrt(w[-1])
    dm = (np.log(F_Q[-1] / 0.6) - 0.5 * s * s) / s
    dip = DF_Q[-1] * (ndtr(-dm) - F_Q[-1] * ndtr(-dm - s))  # eq. (8)
    bond = DF_Q @ sh.coupons + DF_Q[-1]
    assert abs(autocall_nocall(sh, F_Q, DF_Q, w) - (bond - dip)) < 1e-12
    assert abs(autocall_nocall(replace(sh, B_p=0.0), F_Q, DF_Q, w) - bond) < 1e-15
    p, D = ratio_marginals(F_Q, t, 0.6, w)
    assert abs(dip - DF_Q[-1] * (p[-1] + 0.4 * D[-1])) < 1e-12
    two = Autocall((0.5, 1.0), 0.08, 1e6, 0.75, 0.6, False)
    f2, D2, w2 = F_Q[[1, 3]], DF_Q[[1, 3]], 0.25**2 * np.array([0.5, 1.0])
    assert abs(autocall_two_dates(two, f2, D2, w2) - autocall_nocall(two, f2, D2, w2)) < 1e-10
    m1 = autocall_two_dates(replace(two, B_ac=1.0, memory=True), f2, D2, w2)
    assert m1 > autocall_two_dates(replace(two, B_ac=1.0), f2, D2, w2)
    vs = [autocall_nocall(SHEET, F_Q, DF_Q, s**2 * t) for s in (0.05, 0.1, 0.2, 0.3, 0.5)]
    assert np.all(np.diff(vs) < 0)


def test_deng_first_call_probability_p1():
    """§5 Deng-Mallett-McCann (2011) Table 1: p_1 = Q(X_1 >= 1.02) = 1 - D^-_1(1.02) via
    ratio_marginals (Black, f_1 = e^{0.04/12}, w_1 = 0.2^2/12) = 0.3767 to 5e-5 (4-decimal table).
    The published Case 2 value is not reproduced (OPEN_ISSUES; scripts/phase4_gates.py)."""
    t1 = np.array([1 / 12])
    _, Dm = ratio_marginals(np.exp(0.04 * t1), t1, 1.02, 0.2**2 * t1)
    assert abs(1.0 - Dm[0] - 0.3767) < 5e-5


def test_coupon_schedule_vector():
    """Per-period coupons c_i (accrual schedule): flows by hand on two paths; a scalar c equals the
    vector c tau_i; a vector of the wrong length raises."""
    t = (0.5, 1.0, 1.5)
    cpn = tuple(np.diff(np.exp(0.09 * np.array(t)), prepend=1.0))
    sh = Autocall(t, cpn, 1.02, 1.02, 0.8, True)
    X = np.array([[0.9, 1.0], [1.05, 0.95], [1.1, 0.7]])  # path 0 called at t_2; path 1 to T
    C, R, _, alive = autocall_flows(X, sh)
    assert abs(C[1, 0] - (np.exp(0.09) - 1)) < 1e-15 and R[1, 0] == 1.0 and C[:, 1].sum() == 0
    assert list(alive) == [False, True] and abs(R[2, 1] - 0.7) < 1e-15
    lin = Autocall(t, 0.08, 1.0, 0.7, 0.6, True)
    vec = Autocall(t, tuple(0.08 * np.full(3, 0.5)), 1.0, 0.7, 0.6, True)
    assert np.allclose(lin.coupons, vec.coupons, rtol=1e-15)
    with pytest.raises(ValueError):
        Autocall(t, (0.01, 0.02))


def test_heston_marginals_reduce_to_black():
    """Heston COS put and digital (strike-derivative) at xi = 1e-4, rho = 0 equal Black with the
    integrated variance (heston.md eq. 16): |put| < 1e-7, |digital| < 1e-6."""
    p = HestonParams(kappa=1.5, theta=0.04, xi=1e-4, rho=0.0, v0=0.09)
    t = np.array([0.25, 1.0, 1.5])
    for k in (0.6, 0.8, 1.0):
        ph, dh = ratio_marginals(F_Q[:3], t, k, p)
        pb, db = ratio_marginals(F_Q[:3], t, k, total_variance(t, p))
        assert np.abs(ph - pb).max() < 1e-7 and np.abs(dh - db).max() < 1e-6


def test_crn_normals_are_standard():
    """Eq. (15): the period-aggregated Heston spot normals z_i are N(0, 1), uncorrelated across
    periods (sample mean, variance and lag-1 correlation within 3 SE, 50000 paths, dt = 1/32)."""
    _, z = fixings(GS, TQ, 50_000, np.random.default_rng(SEED + 1), 1 / 32)
    n = z.shape[1]
    assert np.all(np.abs(z.mean(1)) <= 3 / np.sqrt(n))
    assert np.all(np.abs(z.var(1) - 1) <= 3 * np.sqrt(2 / n))
    assert np.all(np.abs((z[1:] * z[:-1]).mean(1)) <= 3 / np.sqrt(n))
    steps = list(obs_steps(GS, TQ, 10, np.random.default_rng(1), 1 / 32))
    assert [s[0] for s in steps] == list(TQ)


# ------------------------------------------------------------------ MC vs closed form (slow)
@pytest.mark.slow
def test_bs_mc_vs_closed_forms():
    """§5 BS within 3 SE, 2e5 paths: AC-Q no-call sheet (11) and its B_c = 0 sub-case; n = 1 (11)
    for any B_ac; n = 2 (12) for m in {0, 1}."""
    sig, t = 0.25, np.asarray(TQ)
    w = sig**2 * t
    nc = replace(SHEET, B_ac=np.inf, memory=False)
    for j, sh in enumerate((nc, replace(nc, B_c=0.0))):
        v, se, _, _ = autocall_mc(sh, F_Q, DF_Q, sig, n_paths=200_000, seed=SEED + 10 + j)
        assert within_3se(v, se, autocall_nocall(sh, F_Q, DF_Q, w)), (j, v, se)
    one = Autocall((1.0,), 0.08, 1.0, 0.7, 0.6, True)
    v, se, _, _ = autocall_mc(one, F_Q[3:4], DF_Q[3:4], sig, n_paths=200_000, seed=SEED + 12)
    assert within_3se(v, se, autocall_nocall(one, F_Q[3:4], DF_Q[3:4], sig**2 * np.ones(1)))
    two = Autocall((0.5, 1.0), 0.08, 1.0, 0.75, 0.6, False)
    f2, D2 = F_Q[[1, 3]], DF_Q[[1, 3]]
    for m in (False, True):
        sh = replace(two, memory=m)
        v, se, _, _ = autocall_mc(sh, f2, D2, sig, n_paths=200_000, seed=SEED + 13 + m)
        assert within_3se(v, se, autocall_two_dates(sh, f2, D2, sig**2 * np.array([0.5, 1.0])))


@pytest.mark.slow
def test_heston_mc_vs_cos():
    """§5 Heston (GS, QE-M, dt = 1/32, 2e5 paths): at every t_i, MC D^+_i(L), L in {B_p, B_c, B_ac},
    vs COS and mean M_{t_i} = 1, within 3 SE; the no-call sheet MC vs (11) with COS terms."""
    t = np.asarray(TQ)
    x, _ = fixings(GS, t, 200_000, np.random.default_rng(SEED + 20), 1 / 32)
    M = np.exp(x)
    assert np.all(within_3se(M.mean(1), M.std(1, ddof=1) / np.sqrt(2e5), 1.0))
    X = F_Q[:, None] * M
    for L in (0.6, 0.7, 1.0):
        ind = X >= L
        _, Dm = ratio_marginals(F_Q, t, L, GS)
        assert np.all(within_3se(ind.mean(1), ind.std(1, ddof=1) / np.sqrt(2e5), 1 - Dm)), L
    nc = replace(SHEET, B_ac=np.inf, memory=False)
    pv = autocall_pv(X, nc, DF_Q)
    assert within_3se(pv.mean(), pv.std(ddof=1) / np.sqrt(2e5), autocall_nocall(nc, F_Q, DF_Q, GS))


@pytest.mark.slow
def test_heston_small_xi_vs_bs_same_draws():
    """§5: xi = 1e-4, rho = 0, v0 != theta: full product under Heston QE-M vs BS with QE's discrete
    variance Sigma_Q(t_i) (heston-mc eq. 15) on the same draws (eq. 15), |mean D| <= 3 SE_D."""
    p = HestonParams(kappa=1.5, theta=0.04, xi=1e-4, rho=0.0, v0=0.09)
    t, dt = np.asarray(TQ), 1 / 32
    E = np.exp(-p.kappa * dt)
    SQ = p.theta * t + (p.v0 - p.theta) * dt / 2 * (1 + E) * (1 - np.exp(-p.kappa * t)) / (1 - E)
    xH, z = fixings(p, t, 200_000, np.random.default_rng(SEED + 30), dt)
    D = autocall_pv(F_Q[:, None] * np.exp(xH), SHEET, DF_Q)
    D = D - autocall_pv(F_Q[:, None] * np.exp(gbm_paths(SQ, z)), SHEET, DF_Q)
    assert within_3se(D.mean(), D.std(ddof=1) / np.sqrt(2e5), 0.0), (D.mean(), D.std())


@pytest.mark.slow
def test_model_gap_attribution_illustrative():
    """ROADMAP gate (report): eq. (19) on the AC-Q sheet with market F, DF (D-025) and the
    illustrative GS set (D-029), as printed by scripts/autocall_compare.py. Checks: the three terms
    add up to V_H - V_BS (1e-12); V^nc on the paths vs exact within 3 SE (H, TS, BS); the §3.5 (i)
    surface check D^-_{n,H}(B_p) > flat-ATM D^-_n(B_p); the expected sign V_H < V_BS beyond 3 SE."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "autocall_compare.py"
    spec = importlib.util.spec_from_file_location("autocall_compare", path)
    ac = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ac)
    sheet = ac.SHEETS["AC-Q"]
    f, DF, _ = ac.market_inputs(sheet)
    r = ac.attribution(GS, sheet, f, DF)
    assert abs(r["term1"] + r["term2"] + r["term3"] - r["gap"]) < 1e-12
    for k in ("H", "TS", "BS"):
        assert within_3se(r[f"nc_mc_{k}"], r[f"se_nc_mc_{k}"], r[f"nc_{k}"]), k
    t = np.asarray(sheet.t)
    _, d_h = ratio_marginals(f[-1:], t[-1:], sheet.B_p, GS)
    _, d_b = ratio_marginals(f[-1:], t[-1:], sheet.B_p, r["sigma_atm"] ** 2 * t[-1:])
    assert d_h[0] > d_b[0]
    assert r["gap"] < -3 * r["se_gap"]
    assert bsm.price(f[-1], f[-1], t[-1], 1.0, r["sigma_atm"], 1) == pytest.approx(
        cos_price(f[-1], f[-1], t[-1], 1.0, GS, 1), rel=1e-10
    )
