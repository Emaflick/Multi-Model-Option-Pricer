"""Phase 4 barriers: Reiner-Rubinstein closed form, Brownian-bridge MC, BGK discrete monitoring.

docs/ROADMAP.md Phase 4 and docs/derivations/barrier.md §5. MC comparisons are |MC - ref| <= 3 SE
(heston-mc.md eq. 2) with seeds fixed a priori (SEED + offset, never re-seeded). Gates: in + out =
vanilla on the same paths (< 1e-12); BB-MC vs RR within 3 SE; discrete MC vs BGK "True".
"""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr, zeta

from qpricer.engines.analytic import BGK_BETA, barrier_price, bgk_barrier
from qpricer.engines.exotic_mc import barrier_mc, bridge_survival
from qpricer.models import bsm
from qpricer.models.heston import HestonParams

SEED = 20261108
TYPES = [(cp, eta) for cp in (1, -1) for eta in (1, -1)]  # (theta, eta)
FO = HestonParams(kappa=1.5768, theta=0.0398, xi=0.5751, rho=-0.5711, v0=0.0175)
GOLDEN_HAUG = Path(__file__).resolve().parent / "golden" / "barrier_haug.json"  # validator, frozen

# Haug (1998) p. 72 via QuantLib test-suite/barrieroption.cpp (barrier.md §5): S0 = 100, r = 8%,
# b = 4%, T = 0.5, rebate R = 3, K = 90 / 100 / 110; per type: H -> (sigma = 0.25, sigma = 0.30).
HAUG = {
    ("in", 1, 1): {95: ((7.7627, 4.0109, 2.0576), (9.0093, 5.1370, 2.8517)),
                   100: ((13.8333, 7.8494, 3.9795), (14.8816, 9.2045, 5.3043))},
    ("out", 1, 1): {95: ((9.0246, 6.7924, 4.8759), (8.8334, 7.0285, 5.4137)),
                    100: ((3.0, 3.0, 3.0), (3.0, 3.0, 3.0))},
    ("out", 1, -1): {105: ((2.6789, 2.3580, 2.3453), (2.6341, 2.4389, 2.4315))},
    ("in", 1, -1): {105: ((14.1112, 8.4482, 4.5910), (15.2098, 9.7278, 5.8350))},
    ("out", -1, 1): {95: ((2.2798, 2.2947, 2.6252), (2.4170, 2.4258, 2.6246)),
                     100: ((3.0, 3.0, 3.0), (3.0, 3.0, 3.0))},
    ("out", -1, -1): {105: ((3.7760, 5.4932, 7.5187), (4.2293, 5.8032, 7.5649))},
    ("in", -1, 1): {95: ((2.9586, 6.5677, 11.9752), (3.8769, 7.7989, 13.3078)),
                    100: ((2.2845, 5.9085, 11.6465), (3.3328, 7.2636, 12.9713))},
    ("in", -1, -1): {105: ((1.4653, 3.3721, 7.0846), (2.0658, 4.4226, 8.3686))},
}  # fmt: skip
HF, HDF, HT, HB, HR = 100 * np.exp(0.02), np.exp(-0.04), 0.5, 0.04, 3.0
KH = np.array([90.0, 100.0, 110.0])

# BGK (1997) Tables 2.1, 2.2, 2.6: b = r = 0.1, T = 0.2, sigma = 0.3, K = 100, R = 0.
BF, BDF, BT, BB, BS = 100 * np.exp(0.02), np.exp(-0.02), 0.2, 0.1, 0.3
BGK_21 = {  # DOC, S0 = 100, m = 50: H -> (continuous, corrected, True)
    85: (6.308, 6.322, 6.322), 90: (5.977, 6.098, 6.098), 95: (4.398, 4.907, 4.907),
    96: (3.792, 4.428, 4.427), 97: (3.060, 3.836, 3.834), 98: (2.189, 3.121, 3.126),
    99: (1.171, 2.271, 2.337),
}  # fmt: skip
BGK_22 = {95: (4.398, 5.646, 5.671), 99: (1.171, 4.050, 4.489)}  # DOC, m = 5
BGK_26 = {155: (12.775, 12.905, 12.894), 140: (10.144, 10.581, 10.551),  # UOC, S0 = 110, m = 50
          125: (4.012, 4.649, 4.616), 115: (0.545, 0.819, 0.807)}  # fmt: skip
BF_UP = 110 * np.exp(0.02)


def within_3se(est, se, ref, extra=0.0):
    return np.abs(np.asarray(est) - ref) <= 3.0 * np.asarray(se) + extra


def quad_barrier(F, K, H, T, DF, sig, cp, eta, knock, b, R):
    """Independent oracle: integrate the payoff against the killed density of Y = ln(S/S0) and the
    rebate against the first-passage density (barrier.md §3.2-3.3, not the A..F algebra)."""
    S0, r, nu, v = F * np.exp(-b * T), -np.log(DF) / T, b - 0.5 * sig**2, sig**2 * T
    lv = np.log(H / S0)
    n = lambda y: np.exp(-((y - nu * T) ** 2) / (2 * v)) / np.sqrt(2 * np.pi * v)  # noqa: E731
    p = lambda y: n(y) - np.exp(2 * nu * lv / sig**2) * n(y - 2 * lv)  # noqa: E731
    lo, hi = (lv, lv + 14 * np.sqrt(v)) if eta == 1 else (lv - 14 * np.sqrt(v), lv)
    kink = np.log(K / S0)
    pts = [kink] if lo < kink < hi else None
    kw = dict(epsabs=1e-14, epsrel=1e-13, limit=500)
    out0 = (
        DF * quad(lambda y: max(cp * (S0 * np.exp(y) - K), 0.0) * p(y), lo, hi, points=pts, **kw)[0]
    )
    if knock == "in":
        return bsm.price(F, K, T, DF, sig, cp) - out0 + R * DF * quad(p, lo, hi, **kw)[0]
    c = abs(lv) / (
        sig * np.sqrt(2 * np.pi)
    )  # first-passage density c t^{-3/2} e^{-(l - nu t)^2/..}
    ft = lambda t: c * t**-1.5 * np.exp(-((lv - nu * t) ** 2) / (2 * sig**2 * t))  # noqa: E731
    return out0 + R * quad(lambda t: np.exp(-r * t) * ft(t), 0, T, **kw)[0]


# ------------------------------------------------------------------ closed form
def test_bgk_beta_constant():
    """Eq. (12): beta = -zeta(1/2)/sqrt(2 pi) = 0.5825971579..."""
    assert abs(BGK_BETA - (-zeta(0.5) / np.sqrt(2 * np.pi))) < 1e-15
    assert abs(BGK_BETA - 0.5825971579) < 1e-10


def test_rr_haug_table():
    """§5 published values, per cell and without loosening (D-027, erratum E1 of barrier.md §8):
    the 66 correctly rounded cells at the note's 5e-5; the six E1 cells (flagged
    published_correctly_rounded = false in the validator's golden file barrier_haug.json; two
    independent methods, RRS closed form = Crank-Nicolson PDE) against the golden value at 1e-10.
    Every cell also equals the golden value to 1e-10, and the golden 'published' column equals the
    note's table (HAUG)."""
    cells = json.loads(GOLDEN_HAUG.read_text())["cells"]
    assert len(cells) == 72 and sum(not c["published_correctly_rounded"] for c in cells) == 6
    err_pub, err_e1, err_gold = 0.0, 0.0, 0.0
    for c in cells:
        knock, cp, eta, H, sig, K = (c[k] for k in ("knock", "cp", "eta", "H", "sigma", "K"))
        assert c["published"] == HAUG[(knock, cp, eta)][H][int(sig == 0.3)][list(KH).index(K)]
        v = float(barrier_price(HF, K, H, HT, HDF, sig, cp, eta, knock, b=HB, R=HR))
        err_gold = max(err_gold, abs(v - c["value"]))
        if c["published_correctly_rounded"]:
            err_pub = max(err_pub, abs(v - c["published"]))
        else:
            err_e1 = max(err_e1, abs(v - c["value"]))
    assert err_pub <= 5e-5 and err_e1 <= 1e-10 and err_gold <= 1e-10, (err_pub, err_e1, err_gold)


def test_rr_vs_independent_quadrature():
    """RR (A..F table) vs direct quadrature of the killed / first-passage densities, every Haug
    cell with H != S0 plus b = 0 and an off-table sigma: abs 1e-12 at F ~ 100 (§5 identities)."""
    for (knock, cp, eta), rows in HAUG.items():
        for H in rows:
            if H == 100:
                continue  # breached at S0 = 100: branch (4), tested below
            for sig, b, R in ((0.25, HB, HR), (0.30, HB, HR), (0.45, 0.0, 0.0)):
                F = 100 * np.exp(b * HT)
                v = barrier_price(F, KH, H, HT, HDF, sig, cp, eta, knock, b=b, R=R)
                q = [quad_barrier(F, K, H, HT, HDF, sig, cp, eta, knock, b, R) for K in KH]
                assert np.abs(v - q).max() < 1e-12, (knock, cp, eta, H, sig)


def test_rr_bgk_tables():
    """§5 (i): continuous columns (RR at b != 0) and the BGK-corrected column (RR at the shifted
    barrier, eq. 12) of BGK Tables 2.1, 2.2, 2.6, to 5e-4 (3-decimal tables)."""
    for tab, m, F, eta in ((BGK_21, 50, BF, 1), (BGK_22, 5, BF, 1), (BGK_26, 50, BF_UP, -1)):
        H = np.array(list(tab), float)
        cont, corr, _ = np.array(list(tab.values())).T
        v = barrier_price(F, 100.0, H, BT, BDF, BS, 1, eta, "out", b=BB)
        vc = barrier_price(
            F, 100.0, bgk_barrier(H, BS, BT / m, eta), BT, BDF, BS, 1, eta, "out", b=BB
        )
        assert np.abs(v - cont).max() <= 5e-4 and np.abs(vc - corr).max() <= 5e-4


@pytest.mark.parametrize("b", [0.0, 0.04])
@pytest.mark.parametrize("R", [0.0, 3.0])
def test_rr_in_out_parity_all_cells(b, R):
    """Eq. (5) on all 16 cells of (3) (8 types x K >= H / K < H): in + out = A + E + F, with E, F
    taken from in/out at R vs R = 0; at R = 0 in + out = vanilla. abs 1e-12 (F ~ 100)."""
    F, DF, T, sig = 100 * np.exp(b * 0.5), 0.96, 0.5, 0.3
    K = np.array([80.0, 95.0, 100.0, 104.0, 125.0])
    for cp, eta in TYPES:
        H = 97.0 if eta == 1 else 103.0
        out, inn = (barrier_price(F, K, H, T, DF, sig, cp, eta, k, b=b, R=R) for k in ("out", "in"))
        out0, in0 = (barrier_price(F, K, H, T, DF, sig, cp, eta, k, b=b) for k in ("out", "in"))
        assert np.abs(out0 + in0 - bsm.price(F, K, T, DF, sig, cp)).max() < 1e-12
        rebate = (out - out0) + (inn - in0)  # E + F, independent of K and theta
        assert np.abs(out + inn - bsm.price(F, K, T, DF, sig, cp) - rebate).max() < 1e-12
        assert np.ptp(rebate) < 1e-12 and (R > 0 or np.all(rebate == 0))


def test_rr_rebate_identity_at_zero_rate():
    """§5: E + F = R at DF = 1 (r = 0: rebate paid with probability one, undiscounted)."""
    for cp, eta in TYPES:
        H, ins, outs = (90.0 if eta == 1 else 112.0), [], []
        for knock, acc in (("in", ins), ("out", outs)):
            for R in (0.0, 2.5):
                acc.append(
                    barrier_price(100.0, 100.0, H, 1.3, 1.0, 0.35, cp, eta, knock, b=0.03, R=R)
                )
        assert abs((ins[1] - ins[0]) + (outs[1] - outs[0]) - 2.5) < 1e-12


def test_rr_branch_and_limit_continuity():
    """§5: branches agree at K = H -+ 1 ulp; rho -> 1 tends to (4) (out -> R, in -> A); cdo -> A
    as H -> 0,
    cuo(K < H) -> A as H -> inf (abs 1e-12, except rho -> 1 at distance 1e-10: 1e-7)."""
    F, DF, T, sig = 100.0, 0.97, 0.75, 0.25
    for cp, eta in TYPES:
        H = 95.0 if eta == 1 else 105.0
        for knock in ("in", "out"):
            lo, hi = (
                barrier_price(F, np.nextafter(H, d), H, T, DF, sig, cp, eta, knock)
                for d in (-np.inf, np.inf)
            )
            assert abs(lo - hi) < 1e-12  # K = H -+ 1 ulp (validator F8)
            near = barrier_price(F, 100.0, F * (1 - eta * 1e-10), T, DF, sig, cp, eta, knock, R=2.0)
            ref = 2.0 if knock == "out" else bsm.price(F, 100.0, T, DF, sig, cp)
            assert abs(near - ref) < 1e-7
    A = bsm.price(F, KH, T, DF, sig, 1)
    assert np.abs(barrier_price(F, KH, 1e-6, T, DF, sig, 1, 1, "out") - A).max() < 1e-12
    assert np.abs(barrier_price(F, KH, 1e6, T, DF, sig, 1, -1, "out") - A).max() < 1e-12


@pytest.mark.parametrize("sig", [0.1, 0.25, 0.6])
def test_rr_image_static_hedge_and_model_free(sig):
    """Eqs. (6), (7) at b = 0, K in {H, 0.9 H} or the image range, rel/abs 1e-12."""
    F, DF, T = 100.0, 0.95, 1.2
    H, K = 90.0, np.array([90.0, 100.0, 120.0])  # K >= H: cdi image / static hedge
    cdi = barrier_price(F, K, H, T, DF, sig, 1, 1, "in")
    assert np.allclose(cdi, F / H * bsm.price(H**2 / F, K, T, DF, sig, 1), rtol=1e-12, atol=1e-12)
    assert np.allclose(cdi, K / H * bsm.price(F, H**2 / K, T, DF, sig, -1), rtol=1e-12, atol=1e-12)
    Hu, Ku = 110.0, np.array([80.0, 100.0, 110.0])  # K <= H: pui
    pui = barrier_price(F, Ku, Hu, T, DF, sig, -1, -1, "in")
    assert np.allclose(
        pui, Ku / Hu * bsm.price(F, Hu**2 / Ku, T, DF, sig, 1), rtol=1e-12, atol=1e-12
    )
    s = sig * np.sqrt(T)
    d1 = np.log(F / H) / s + 0.5 * s
    q = ndtr(d1 - s) - F / H * ndtr(-d1)
    for Kc in (H, 0.9 * H):  # eq. (7), K <= H
        ref = DF * ((F - H) + (H - Kc) * q)
        assert abs(barrier_price(F, Kc, H, T, DF, sig, 1, 1, "out") - ref) < 1e-12
    assert abs(barrier_price(F, Hu, Hu, T, DF, sig, -1, -1, "out") - DF * (Hu - F)) < 1e-12


def test_rr_input_validation():
    with pytest.raises(ValueError):
        barrier_price(100.0, 100.0, -1.0, 1.0, 1.0, 0.2, 1, 1, "out")
    with pytest.raises(ValueError):
        barrier_price(100.0, 100.0, 90.0, 1.0, 1.0, 0.2, 1, 0, "out")
    with pytest.raises(ValueError):
        barrier_price(100.0, 100.0, 90.0, 1.0, 1.0, 0.0, 1, 1, "out")
    with pytest.raises(ValueError):
        barrier_price(100.0, 100.0, 90.0, 1.0, 1.0, 0.2, 1, 1, "knock")
    with pytest.raises(ValueError, match="pitfall 9"):  # DF > 1 with a rebate: lambda^2 < 0
        barrier_price(100.0, 100.0, 90.0, 1.0, 1.2, 0.2, 1, 1, "out", R=1.0)


# ------------------------------------------------------------------ MC: exact identities
def test_bridge_survival_pitfalls():
    """Eq. (9), positive parts: an endpoint on/beyond H -> p = 1; s2dt = 0 -> 0 or 1 (pitfall 3)."""
    s = bridge_survival(
        np.array([0.1, -0.1, 0.0, 0.2, 0.2]), np.array([-0.1, 0.1, 0.3, 0.3, 0.0]), 0.04
    )
    assert np.array_equal(s[:3], [0.0, 0.0, 0.0]) and s[4] == 0.0 and 0 < s[3] < 1
    assert abs(s[3] - (1 - np.exp(-2 * 0.2 * 0.3 / 0.04))) < 1e-15
    z = bridge_survival(np.array([0.2, 0.0]), np.array([0.3, 0.3]), np.array([0.0, 0.0]))
    assert np.array_equal(z, [1.0, 0.0])


GATE_MODELS = {"bsm": 0.25, "heston": FO}


@pytest.mark.parametrize("monitor", [None, 10])
@pytest.mark.parametrize("model", list(GATE_MODELS))
def test_gate_in_plus_out_equals_vanilla(model, monitor):
    """ROADMAP gate: in + out = vanilla on the same paths, < 1e-12, all 8 types, separate pricer
    calls (§5), bridge (monitor=None) and discrete weights, BSM and Heston (QE-M), b = 0.04."""
    m = GATE_MODELS[model]
    K = np.array([85.0, 100.0, 115.0])
    kw = dict(n_steps=20, n_paths=20_000, seed=SEED, b=0.04, monitor=monitor)
    worst = 0.0
    for cp, eta in TYPES:
        H = 92.0 if eta == 1 else 108.0
        o, i, v = (
            barrier_mc(101.0, K, H, 1.0, 0.95, m, cp, eta, k, **kw)[0]
            for k in ("out", "in", "vanilla")
        )
        worst = max(worst, np.abs(i + o - v).max())
    assert worst < 1e-12, worst


def test_mc_ordering_and_limits_common_paths():
    """§5: out_cont <= out_disc, in_cont >= in_disc on common grid paths (14); H -> 0: out ==
    vanilla; barrier at/beyond S_0: out == 0, in == vanilla; calls non-increasing in K, cdo
    non-increasing in H (common random numbers)."""
    K = np.linspace(80.0, 120.0, 9)
    kw = dict(n_steps=40, n_paths=20_000, seed=SEED + 1)
    van = barrier_mc(100.0, K, 90.0, 1.0, 1.0, 0.3, 1, 1, "vanilla", **kw)[0]
    oc, od = (
        barrier_mc(100.0, K, 90.0, 1.0, 1.0, 0.3, 1, 1, "out", monitor=mo, **kw)[0]
        for mo in (None, 40)
    )
    ic, idd = (
        barrier_mc(100.0, K, 90.0, 1.0, 1.0, 0.3, 1, 1, "in", monitor=mo, **kw)[0]
        for mo in (None, 40)
    )
    assert np.all(oc <= od + 1e-14) and np.all(ic >= idd - 1e-14)
    assert np.array_equal(barrier_mc(100.0, K, 1e-3, 1.0, 1.0, 0.3, 1, 1, "out", **kw)[0], van)
    for H in (100.0, 104.0):
        assert np.all(barrier_mc(100.0, K, H, 1.0, 1.0, 0.3, 1, 1, "out", **kw)[0] == 0.0)
        assert np.array_equal(barrier_mc(100.0, K, H, 1.0, 1.0, 0.3, 1, 1, "in", **kw)[0], van)
    assert np.all(np.diff(oc) <= 1e-14)
    cdo_H = [
        barrier_mc(100.0, 100.0, H, 1.0, 1.0, 0.3, 1, 1, "out", **kw)[0] for H in (70, 80, 90, 95)
    ]
    assert np.all(np.diff(cdo_H) <= 1e-14)


def test_mc_api_contract():
    out = barrier_mc(
        100.0, [90.0, 110.0], 95.0, 1.0, 0.9, 0.2, 1, 1, "out", n_steps=5, n_paths=1000, seed=SEED
    )
    price, se, n, seed = out
    assert price.shape == se.shape == (2,) and n == 1000 and seed == SEED and np.all(se > 0)
    with pytest.raises(ValueError):
        barrier_mc(
            100.0, 100.0, 95.0, 1.0, 1.0, 0.2, 1, 1, "out", n_steps=5, n_paths=10, seed=1, monitor=3
        )
    with pytest.raises(ValueError):
        barrier_mc(100.0, 100.0, 95.0, 1.0, 1.0, 0.2, 1, 1, "up", n_steps=5, n_paths=10, seed=1)


# ------------------------------------------------------------------ MC gates (slow)
GATE_CFG = [(b, M) for b in (0.0, 0.04) for M in (1, 50)]  # §5: BB-MC unbiased for every M (BSM)


@pytest.mark.slow
@pytest.mark.parametrize("j", range(len(GATE_CFG)))
def test_gate_bb_mc_vs_rr(j):
    """ROADMAP gate: BSM BB-MC (8)-(10) vs RR (3), all 8 types, K in {90, 110} (both branches of
    (3) for H = 95 down / 105 up), Haug market (F = 100 e^{bT}, DF = e^{-0.04}, T = 0.5,
    sigma = 0.25), 2e5 paths, seed SEED + 10 + j shared by the 16 calls of a configuration.
    Cells whose value is identically 0 (cuo K >= H, pdo K <= H) must be exactly 0 in MC too."""
    b, M = GATE_CFG[j]
    F, K = 100 * np.exp(b * HT), np.array([90.0, 110.0])
    for cp, eta in TYPES:
        H = 95.0 if eta == 1 else 105.0
        for knock in ("out", "in"):
            ref = barrier_price(F, K, H, HT, HDF, 0.25, cp, eta, knock, b=b)
            v, se, _, _ = barrier_mc(
                F,
                K,
                H,
                HT,
                HDF,
                0.25,
                cp,
                eta,
                knock,
                n_steps=M,
                n_paths=200_000,
                seed=SEED + 10 + j,
                b=b,
            )
            zero = ref == 0
            assert np.all(v[zero] == 0) and np.all(within_3se(v[~zero], se[~zero], ref[~zero])), (
                cp,
                eta,
                knock,
                (v - ref) / se,
            )


@pytest.mark.slow
def test_gate_discrete_mc_vs_bgk_true():
    """§5 (ii): discrete-monitoring BSM MC (exact steps at t_j, no bridge) vs BGK "True",
    |MC - True| <= 3 SE + 0.001, Tables 2.1 / 2.2 (DOC) and 2.6 (UOC), 2e5 paths."""
    for k, (tab, m, F, eta) in enumerate(
        ((BGK_21, 50, BF, 1), (BGK_22, 5, BF, 1), (BGK_26, 50, BF_UP, -1))
    ):
        for H, (_, _, true) in tab.items():
            v, se, _, _ = barrier_mc(
                F,
                100.0,
                H,
                BT,
                BDF,
                BS,
                1,
                eta,
                "out",
                n_steps=m,
                n_paths=200_000,
                seed=SEED + 20 + k,
                b=BB,
                monitor=m,
            )
            assert within_3se(v, se, true, 0.001), (H, m, v, true, se)


@pytest.mark.slow
def test_gate_heston_small_xi_vs_rr():
    """§5: Heston BB-MC (QE-M, v_i^+ bridge) at xi = 1e-4, rho = 0, v0 = theta vs RR at
    sigma = sqrt(theta), within 3 SE: cdo/cdi (H = 90) and puo/pui (H = 110), K = 100, T = 1."""
    p = HestonParams(kappa=1.5, theta=0.04, xi=1e-4, rho=0.0, v0=0.04)
    for cp, eta, H in ((1, 1, 90.0), (-1, -1, 110.0)):
        for knock in ("out", "in"):
            ref = barrier_price(100.0, 100.0, H, 1.0, 0.97, 0.2, cp, eta, knock)
            v, se, _, _ = barrier_mc(
                100.0,
                100.0,
                H,
                1.0,
                0.97,
                p,
                cp,
                eta,
                knock,
                n_steps=50,
                n_paths=200_000,
                seed=SEED + 30,
            )
            assert within_3se(v, se, ref), (cp, eta, knock, (v - ref) / se)


@pytest.mark.slow
def test_gate_heston_model_free_cdo():
    """§5 eq. (7): cdo(K = H = 90, b = 0) = DF (F - H) = 10 DF for any continuous martingale; FO
    set (Feller-violating), QE-M with the v_i^+ bridge at dt = 1/128, 2e5 paths, within 3 SE
    (the dt-dependence of the bridge bias is reported by scripts/phase4_gates.py)."""
    v, se, _, _ = barrier_mc(
        100.0, 90.0, 90.0, 1.0, 0.98, FO, 1, 1, "out", n_steps=128, n_paths=200_000, seed=SEED + 40
    )
    assert within_3se(v, se, 0.98 * 10.0), (v, se)
