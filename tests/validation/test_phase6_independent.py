"""Independent validation of Phase 6: benchmark MC vs closed form (vr.py, run_benchmarks.py).

No qpricer helper is used to produce a reference value. qpricer is called only as the system under
test (``engines.vr.vanilla_mc``, ``engines.vr.sobol_normals``) or to read its CSV outputs. Every
reference here is my own code:

* B76   -- Black-76 closed form, written here.
* GP    -- Heston by Gil-Pelaez (P1/P2 form), little-trap CF (Albrecher et al. 2007), scipy quad on
           [0, inf). This is independent of the implementer's COS and of Lewis.
* VRCF  -- population VR factors under BSM: the CV with X = F_T in closed form (lognormal partial
           moments), the antithetic with Cov(g(Z), g(-Z)) by Gauss-Hermite quadrature.
* CVB   -- the leading O(1/N) bias of the regression CV, -E[x^2 e] / (N Var X), by quadrature.
* OQE   -- my own Andersen (2008) QE scheme with the martingale correction (QE-M), central weights
           gamma1 = gamma2 = 1/2, psi_c = 1.5. The exponential branch uses an independent uniform
           (the implementer uses Phi(Z_v)), so the code is independent and the law is the same.
* EST   -- my own plain, antithetic (pair means), regression-CV and RQMC estimators. For RQMC I use
           scipy's scrambled Sobol' with bits = 53 rather than 30, so no grid jitter is needed. This
           is a different fix from D-042's.

Seeds: validator base 6_060_000. They were fixed before the first look and never re-seeded.
Run as a script to print the numbers quoted in docs/validation/phase6.md. That takes a few minutes
and uses a process pool:
``.venv/bin/python tests/validation/test_phase6_independent.py``.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.special import ndtr, ndtri, roots_hermitenorm
from scipy.stats import qmc

from qpricer.engines import vr  # system under test only
from qpricer.models.heston import HestonParams  # parameter container only

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "report" / "figures"
VSEED = 6_060_000

# case inputs, copied from benchmark_references.csv (they are inputs, not references)
F_C, DF_C, T_C = 7677.961694908149, 0.95134635345534, 1.0347338597158802
SIG_C = 0.1840369273335226
CAL = (
    4.575131438111167,
    0.05292535169697239,
    1.6476980696550318,
    -0.7517989242308154,
    0.05237208868271241,
)  # fmt: skip  kappa, theta, xi, rho, v0
FO = (1.5768, 0.0398, 0.5751, -0.5711, 0.0175)
CASES = {
    "bsm": dict(F=F_C, DF=DF_C, T=T_C, model=SIG_C, n_steps=1),
    "heston_cal": dict(F=F_C, DF=DF_C, T=T_C, model=CAL, n_steps=34),
    "heston_fo": dict(F=100.0, DF=1.0, T=1.0, model=FO, n_steps=32),
}
MONEY, CPS = np.array([1.0, 0.9]), np.array([1.0, -1.0])
LABELS = ("ATM call", "90% put")


def hp(m) -> HestonParams:
    return HestonParams(*m)


# ===================================================================== references
def black76(F, K, T, DF, sig, cp):
    F, K, cp = np.asarray(F, float), np.asarray(K, float), np.asarray(cp, float)
    s = sig * np.sqrt(T)
    d1 = np.log(F / K) / s + 0.5 * s
    d2 = d1 - s
    return DF * cp * (F * ndtr(cp * d1) - K * ndtr(cp * d2))


def heston_cf(u, T, kappa, theta, xi, rho, v0):
    """E exp(i u ln(F_T/F)), little-trap form."""
    iu = 1j * u
    b = kappa - rho * xi * iu
    d = np.sqrt(b * b + xi * xi * (iu + u * u))
    g = (b - d) / (b + d)
    e = np.exp(-d * T)
    C = kappa * theta / xi**2 * ((b - d) * T - 2.0 * np.log((1 - g * e) / (1 - g)))
    D = (b - d) / xi**2 * (1 - e) / (1 - g * e)
    return np.exp(C + D * v0)


def heston_gp(F, K, T, DF, m, cp):
    """Gil-Pelaez: C = DF (F P1 - K P2); put by parity."""
    k = np.log(K / F)
    o = dict(limit=4000, epsabs=1e-14, epsrel=1e-13)
    f1 = lambda u: (np.exp(-1j * u * k) * heston_cf(u - 1j, T, *m) / (1j * u)).real  # noqa: E731
    f2 = lambda u: (np.exp(-1j * u * k) * heston_cf(u, T, *m) / (1j * u)).real  # noqa: E731
    P1 = 0.5 + quad(f1, 0, np.inf, **o)[0] / np.pi
    P2 = 0.5 + quad(f2, 0, np.inf, **o)[0] / np.pi
    c = DF * (F * P1 - K * P2)
    return c if cp > 0 else c - DF * (F - K)


def reference(name: str) -> np.ndarray:
    c = CASES[name]
    K = c["F"] * MONEY
    if name == "bsm":
        return black76(c["F"], K, c["T"], c["DF"], c["model"], CPS)
    return np.array(
        [heston_gp(c["F"], k, c["T"], c["DF"], c["model"], s) for k, s in zip(K, CPS, strict=True)]
    )


def vr_closed_form_bsm(F, K, T, sig, cp):
    """Population VR at equal N under exact BSM: (CV with X = F_T, antithetic). DF cancels."""
    s = sig * np.sqrt(T)
    d1 = np.log(F / K) / s + 0.5 * s
    d2, d0 = d1 - s, d1 + s
    e2 = np.exp(s * s)
    # call partial moments: E[F_T^j 1{F_T > K}]
    m0, m1, m2 = ndtr(d2), F * ndtr(d1), F * F * e2 * ndtr(d0)
    if cp > 0:
        Eg, Eg2, EgX = m1 - K * m0, m2 - 2 * K * m1 + K * K * m0, m2 - K * m1
    else:  # put: complements
        n0, n1, n2 = 1 - m0, F - m1, F * F * e2 - m2
        Eg, Eg2, EgX = K * n0 - n1, K * K * n0 - 2 * K * n1 + n2, K * n1 - n2
    varg, varX, cov = Eg2 - Eg**2, F * F * (e2 - 1), EgX - Eg * F
    vr_cv = 1.0 / (1.0 - cov**2 / (varg * varX))
    z, w = roots_hermitenorm(400)
    w = w / w.sum()
    g = lambda zz: np.maximum(cp * (F * np.exp(-0.5 * s * s + s * zz) - K), 0.0)  # noqa: E731
    c_anti = np.sum(w * g(z) * g(-z)) - Eg**2
    vr_anti = varg / (varg + c_anti)
    return vr_cv, vr_anti


def cv_bias_leading(F, K, T, DF, sig, cp, N):
    """Leading bias of the OLS-beta CV: E[Ybar - c_hat (Xbar - mu)] - E Y = -E[x^2 e]/(N Var X)."""
    s = sig * np.sqrt(T)
    z, w = roots_hermitenorm(400)
    w = w / w.sum()
    X = F * np.exp(-0.5 * s * s + s * z)
    Y = DF * np.maximum(cp * (X - K), 0.0)
    x, y = X - F, Y - np.sum(w * Y)
    c = np.sum(w * x * y) / np.sum(w * x * x)
    e = y - c * x
    return -np.sum(w * x * x * e) / (N * np.sum(w * x * x))


# ===================================================================== my own MC
def oqe_terminal(m, T, n_steps, n, rng, anti=False):
    """My QE-M (Andersen 2008, sec. 3.2.4 + martingale correction 4.2.2); returns ln(F_T/F)."""
    kappa, theta, xi, rho, v0 = m
    dt = T / n_steps
    E = np.exp(-kappa * dt)
    k0 = -rho * kappa * theta * dt / xi
    k1 = 0.5 * dt * (kappa * rho / xi - 0.5) - rho / xi
    k2 = 0.5 * dt * (kappa * rho / xi - 0.5) + rho / xi
    k3 = k4 = 0.5 * dt * (1 - rho * rho)
    A = k2 + 0.5 * k4
    h = n // 2 if anti else n
    x, v = np.zeros(n), np.full(n, v0)
    for _ in range(n_steps):
        z1, uu, z2 = rng.standard_normal(h), rng.random(h), rng.standard_normal(h)
        if anti:
            z1, uu, z2 = np.r_[z1, -z1], np.r_[uu, 1 - uu], np.r_[z2, -z2]
        mm = theta + (v - theta) * E
        s2 = v * xi * xi * E * (1 - E) / kappa + theta * xi * xi * (1 - E) ** 2 / (2 * kappa)
        psi = s2 / mm**2
        vn = np.empty(n)
        lnM = np.empty(n)
        q = psi <= 1.5
        iq = 2.0 / psi[q]
        b2 = iq - 1 + np.sqrt(iq) * np.sqrt(iq - 1)
        a = mm[q] / (1 + b2)
        vn[q] = a * (np.sqrt(b2) + z1[q]) ** 2
        lnM[q] = A * b2 * a / (1 - 2 * A * a) - 0.5 * np.log(1 - 2 * A * a)
        r = ~q
        p = (psi[r] - 1) / (psi[r] + 1)
        beta = (1 - p) / mm[r]
        ur = uu[r]
        vn[r] = np.where(ur <= p, 0.0, np.log((1 - p) / np.maximum(1 - ur, 1e-300)) / beta)
        lnM[r] = np.log(p + beta * (1 - p) / (beta - A))
        x = x - lnM - (k1 + 0.5 * k3) * v + k1 * v + k2 * vn + np.sqrt(k3 * v + k4 * vn) * z2
        v = vn
    del k0  # the drift constant is replaced by the martingale correction
    return x


def my_estimate(kind, g, X=None, EX=None):
    """(mean, se) along the last axis; antithetic: halves are the pairs."""
    n = g.shape[-1]
    if kind == "antithetic":
        h = 0.5 * (g[..., : n // 2] + g[..., n // 2 :])
        return h.mean(-1), h.std(-1, ddof=1) / np.sqrt(n // 2)
    if kind == "cv":
        xc = X - X.mean(-1, keepdims=True)
        c = (xc * (g - g.mean(-1, keepdims=True))).sum(-1) / (xc * xc).sum(-1)
        e = g - c[..., None] * (X - EX)
        return e.mean(-1), e.std(-1, ddof=1) / np.sqrt(n)
    return g.mean(-1), g.std(-1, ddof=1) / np.sqrt(n)


def my_bsm_rqmc(F, K, T, DF, sig, cp, n_total, seed, R=16):
    """RQMC with independent scramblings: bits = 53 scrambled Sobol', no jitter."""
    est = []
    for ss in np.random.SeedSequence(seed).spawn(R):
        u = qmc.Sobol(1, scramble=True, bits=53, rng=np.random.default_rng(ss)).random(n_total // R)
        z = ndtri(np.clip(u[:, 0], 1e-300, 1 - 2**-53))
        FT = F * np.exp(-0.5 * sig * sig * T + sig * np.sqrt(T) * z)
        est.append(DF * np.maximum(cp[:, None] * (FT - K[:, None]), 0).mean(-1))
    est = np.array(est)
    return est.mean(0), est.std(0, ddof=1) / np.sqrt(R)


def my_mc(name, kind, N, seed, n_steps=None):
    """My own estimator on my own paths: plain / antithetic / cv. Returns (price[2], se[2])."""
    c = CASES[name]
    F, DF, T = c["F"], c["DF"], c["T"]
    K = F * MONEY
    rng = np.random.default_rng(seed)
    anti = kind == "antithetic"
    if name == "bsm":
        z = rng.standard_normal(N // 2 if anti else N)
        z = np.r_[z, -z] if anti else z
        x = -0.5 * c["model"] ** 2 * T + c["model"] * np.sqrt(T) * z
    else:
        x = oqe_terminal(c["model"], T, n_steps or c["n_steps"], N, rng, anti)
    FT = F * np.exp(x)
    g = DF * np.maximum(CPS[:, None] * (FT[None] - K[:, None]), 0.0)
    if kind == "cv":
        return my_estimate("cv", g, np.broadcast_to(FT, g.shape), F)
    return my_estimate(kind, g)


def ols_slope(N, y):
    x, z = np.log(np.asarray(N, float)), np.log(np.asarray(y, float))
    xc = x - x.mean()
    return float(xc @ (z - z.mean()) / (xc @ xc))


# ===================================================================== fast tests
def test_heston_references_gp_vs_implementation_cos():
    """My Gil-Pelaez agrees with the COS values stored by run_benchmarks.py to 1e-7 absolute."""
    refs = pd.read_csv(FIG / "benchmark_references.csv")
    for name in ("bsm", "heston_cal", "heston_fo"):
        mine = reference(name)
        theirs = refs[refs.set == name].set_index("strike").loc[list(LABELS), "ref"].to_numpy()
        assert np.allclose(mine, theirs, rtol=0, atol=1e-7), (name, mine - theirs)
    # Fang-Oosterlee (2008) Table 4 published 5.785155450 (10 digits)
    assert abs(heston_gp(100, 100, 1, 1, FO, 1) - 5.785155450) < 5e-8


def test_golden_phase6_refs():
    """Validator-created golden phase6_benchmark_refs.json vs my references and the CSV."""
    import json

    gold = json.loads((ROOT / "tests" / "golden" / "phase6_benchmark_refs.json").read_text())
    refs = pd.read_csv(FIG / "benchmark_references.csv")
    for c in gold["cases"]:
        j = LABELS.index(c["strike"])
        assert reference(c["set"])[j] == pytest.approx(c["price"], rel=0, abs=1e-9)
        r = refs[(refs.set == c["set"]) & (refs.strike == c["strike"])].iloc[0]
        assert abs(r.ref - c["price"]) <= 1e-6 and r.K == pytest.approx(c["K"], rel=1e-15)


def test_bsm_vol_is_heston_atm_iv():
    """D-039: sigma = calibrated-Heston ATM implied vol, so Black(sigma) = Heston ATM (my GP)."""
    gap = black76(F_C, F_C, T_C, DF_C, SIG_C, 1) - heston_gp(F_C, F_C, T_C, DF_C, CAL, 1)
    assert abs(gap) < 1e-6


def test_vr_reported_bsm_matches_closed_form():
    """Reported BSM VR at N = 2^20 (rms SE^2 ratio over 16 runs) = the population VR (0.5 %)."""
    s = pd.read_csv(FIG / "benchmark_summary.csv")
    s = s[s.set == "bsm"].set_index(["strike", "method"])
    for j, lab in enumerate(LABELS):
        cv, anti = vr_closed_form_bsm(F_C, F_C * MONEY[j], T_C, SIG_C, CPS[j])
        assert s.loc[(lab, "cv"), "VR_se_N1048576"] == pytest.approx(cv, rel=5e-3)
        assert s.loc[(lab, "antithetic"), "VR_se_N1048576"] == pytest.approx(anti, rel=5e-3)


def test_efficiency_definition_consistent():
    """eff = VR_se x t_plain / t_m at N = 2^20, recomputed from the timing table."""
    s = pd.read_csv(FIG / "benchmark_summary.csv")
    t = pd.read_csv(FIG / "benchmark_timing.csv")
    t = t[t.N == 2**20].set_index(["set", "method"]).median_s
    for _, r in s.iterrows():
        eff = r["VR_se_N1048576"] * t[(r.set, "plain")] / t[(r.set, r.method)]
        assert r["eff_N1048576"] == pytest.approx(eff, rel=1e-12)
        assert r["time_s_N1048576"] == pytest.approx(t[(r.set, r.method)], rel=1e-12)


def test_slope_metric_excludes_heston_bias():
    """Recompute the gate metric from benchmark_runs.csv with my own code: Heston err = replication
    SD about the replication mean (a constant bias cancels exactly); BSM err = RMSE vs my Black."""
    runs = pd.read_csv(FIG / "benchmark_runs.csv")
    conv = pd.read_csv(FIG / "benchmark_convergence.csv")
    for name in CASES:
        ref = dict(zip(LABELS, reference(name), strict=True))
        for (lab, meth, N), g in runs[runs.set == name].groupby(["strike", "method", "N"]):
            if name == "bsm":
                err = np.sqrt(np.mean((g.price - ref[lab]) ** 2))
            else:
                err = g.price.std(ddof=1)
                shifted = (g.price + 123.0).std(ddof=1)  # any constant bias drops out
                assert shifted == pytest.approx(err, rel=1e-9)
            row = conv[(conv.set == name) & (conv.strike == lab) & (conv.method == meth)
                       & (conv.N == N)]  # fmt: skip
            assert len(row) == 1 and row.err.iloc[0] == pytest.approx(err, rel=1e-9, abs=1e-12)
            assert len(g) == 16 and sorted(g.rep) == list(range(16))
    # the fitted plain slopes equal my own OLS over all 11 points (nothing dropped)
    summ = pd.read_csv(FIG / "benchmark_summary.csv")
    for (name, lab), g in conv[conv.method == "plain"].groupby(["set", "strike"]):
        g = g.sort_values("N")
        assert len(g) == 11
        r = summ[(summ.set == name) & (summ.strike == lab) & (summ.method == "plain")].iloc[0]
        assert r.slope_err == pytest.approx(ols_slope(g.N, g.err), abs=1e-12)
        assert r.slope_se == pytest.approx(ols_slope(g.N, g.se_r0), abs=1e-12)


def test_antithetic_counts_pairs_and_se_from_pair_means():
    """vr antithetic on BSM = my pair-mean estimator on my own (Z, -Z) with Z of size N/2."""
    N, seed = 2**12, VSEED + 1
    K = F_C * MONEY
    p, se, n, s = vr.vanilla_mc(F_C, K, T_C, DF_C, SIG_C, CPS, method="antithetic", n_paths=N,
                                seed=seed)  # fmt: skip
    z = np.random.default_rng(seed).standard_normal(N // 2)
    FT = F_C * np.exp(-0.5 * SIG_C**2 * T_C + SIG_C * np.sqrt(T_C) * np.r_[z, -z])
    g = DF_C * np.maximum(CPS[:, None] * (FT - K[:, None]), 0)
    mp, mse = my_estimate("antithetic", g)
    assert n == N and s == seed
    assert np.allclose(p, mp, rtol=1e-13) and np.allclose(se, mse, rtol=1e-12)
    # the naive SE (N correlated payoffs treated as iid) is different -> the code does not use it
    naive = g.std(-1, ddof=1) / np.sqrt(N)
    assert np.all(np.abs(se / naive - 1) > 0.05)


def test_cv_matches_my_regression_cv_bsm():
    N, seed = 2**12, VSEED + 2
    K = F_C * MONEY
    p, se, _, _ = vr.vanilla_mc(F_C, K, T_C, DF_C, SIG_C, CPS, method="cv", n_paths=N, seed=seed)
    z = np.random.default_rng(seed).standard_normal((1, N))[0]
    FT = F_C * np.exp(-0.5 * SIG_C**2 * T_C + SIG_C * np.sqrt(T_C) * z)
    g = DF_C * np.maximum(CPS[:, None] * (FT - K[:, None]), 0)
    mp, mse = my_estimate("cv", g, np.broadcast_to(FT, g.shape), F_C)
    assert np.allclose(p, mp, rtol=1e-12) and np.allclose(se, mse, rtol=1e-10)


def test_cv_bias_negligible_closed_form():
    """Leading O(1/N) bias of the estimated-beta CV (quadrature): < 0.1 SE at N = 2^10 (measured
    0.043 ATM), < 0.003 SE at N = 2^20; bias/SE falls as N^-1/2."""
    z, w = roots_hermitenorm(400)
    w = w / w.sum()
    FT = F_C * np.exp(-0.5 * SIG_C**2 * T_C + SIG_C * np.sqrt(T_C) * z)
    for j in range(2):
        K = F_C * MONEY[j]
        b = cv_bias_leading(F_C, K, T_C, DF_C, SIG_C, CPS[j], 2**10)
        vr_cv, _ = vr_closed_form_bsm(F_C, K, T_C, SIG_C, CPS[j])
        g = DF_C * np.maximum(CPS[j] * (FT - K), 0.0)
        se_cv = np.sqrt((np.sum(w * g * g) - np.sum(w * g) ** 2) / vr_cv / 2**10)
        assert abs(b) / se_cv < 0.1
        assert abs(b / 1024) / (se_cv / 32) < 0.003


def test_sobol_scramblings_independent_and_uniform():
    """The 16 RQMC child streams give 16 different nets; each coordinate's marginal is U(0, 1)
    (jitter fills the 2^-30 cell): the fractional cell position is not stuck at 0."""
    kids = np.random.SeedSequence(VSEED + 3).spawn(16)
    zs = [vr.sobol_normals(2, 256, np.random.default_rng(k)) for k in kids]
    firsts = {float(z[0, 0]) for z in zs}
    assert len(firsts) == 16
    u = ndtr(np.concatenate([z.ravel() for z in zs]))
    frac = (u * 2**30) % 1.0  # position inside the 2^-30 cell (ndtr round trip is ~1e-16)
    mid = (u > 1e-3) & (u < 1 - 1e-3)
    assert 0.4 < frac[mid].mean() < 0.6  # without the jitter this would be ~0 (or ~1)
    # pairwise coordinate stratification at 1/n (one point per cell per dimension)
    for z in zs:
        cells = np.sort(np.floor(ndtr(z) * 256).astype(int), axis=1)
        assert np.array_equal(cells, np.broadcast_to(np.arange(256), (2, 256)))


def test_sobol_jitter_negligible_vs_bits53_reference():
    """Jittered bits = 30 RQMC vs my bits = 53 RQMC on BSM, both pooled within 3 SE of Black."""
    K = F_C * MONEY
    a, b = [], []
    for r in range(40):
        p, se, _, _ = vr.vanilla_mc(F_C, K, T_C, DF_C, SIG_C, CPS, method="sobol", n_paths=2**14,
                                    seed=VSEED + 100 + r)  # fmt: skip
        a.append((p, se))
        b.append(my_bsm_rqmc(F_C, K, T_C, DF_C, SIG_C, CPS, 2**14, VSEED + 200 + r))
    ref = reference("bsm")
    for est in (a, b):
        P = np.mean([e[0] for e in est], 0)
        S = np.sqrt(np.sum([e[1] ** 2 for e in est], 0)) / len(est)
        assert np.all(np.abs(P - ref) <= 3 * S), (P - ref) / S


# ===================================================================== slow tests
@pytest.mark.slow
def test_se_honesty_fresh_seeds_bsm_all_methods():
    """Replication SD of vr estimates (200 fresh seeds, N = 2^14) vs their quoted SE: ratio within
    the chi^2(199) band (+-15 %) for all four methods, and the mean within 3 SE of Black."""
    K, ref = F_C * MONEY, reference("bsm")
    for mi, meth in enumerate(vr.METHODS):
        P, S = [], []
        for r in range(200):
            seed = VSEED + 10_000 * (mi + 1) + r
            p, se, _, _ = vr.vanilla_mc(F_C, K, T_C, DF_C, SIG_C, CPS, method=meth,
                                        n_paths=2**14, seed=seed)  # fmt: skip
            P.append(p)
            S.append(se)
        P, S = np.array(P), np.array(S)
        ratio = P.std(0, ddof=1) / np.sqrt(np.mean(S**2, 0))
        assert np.all(np.abs(ratio - 1) < 0.15), (meth, ratio)
        assert np.all(np.abs(P.mean(0) - ref) <= 3 * np.sqrt(np.mean(S**2, 0) / 200)), meth


@pytest.mark.slow
def test_se_honesty_fresh_seeds_heston_antithetic_sobol():
    """Same honesty check under Heston FO (32 steps, N = 2^12, 120 seeds), the two SEs that are not
    from iid payoffs: antithetic pair means and RQMC scramblings. Band +-20 % (chi^2(119))."""
    K = 100.0 * MONEY
    for mi, meth in enumerate(("antithetic", "sobol")):
        P, S = [], []
        for r in range(120):
            p, se, _, _ = vr.vanilla_mc(100.0, K, 1.0, 1.0, hp(FO), CPS, method=meth,
                                        n_paths=2**12, seed=VSEED + 50_000 + 1000 * mi + r,
                                        n_steps=32)  # fmt: skip
            P.append(p)
            S.append(se)
        P, S = np.array(P), np.array(S)
        ratio = P.std(0, ddof=1) / np.sqrt(np.mean(S**2, 0))
        assert np.all(np.abs(ratio - 1) < 0.20), (meth, ratio)


@pytest.mark.slow
def test_cv_bias_empirical_small_n():
    """20 000 CV runs at N = 2^10 (BSM): mean - Black equals the predicted leading O(1/N) bias
    within 3 SE (so the CV has no bias beyond the known O(1/N) term)."""
    K, ref = F_C * MONEY, reference("bsm")
    P = np.array([vr.vanilla_mc(F_C, K, T_C, DF_C, SIG_C, CPS, method="cv", n_paths=2**10,
                                seed=VSEED + 100_000 + r)[0] for r in range(20_000)])  # fmt: skip
    pred = np.array([cv_bias_leading(F_C, k, T_C, DF_C, SIG_C, c, 2**10) for k, c in zip(K, CPS,
                     strict=True)])  # fmt: skip
    se_mean = P.std(0, ddof=1) / np.sqrt(len(P))
    assert np.all(np.abs(P.mean(0) - ref - pred) <= 3 * se_mean), (P.mean(0) - ref, pred, se_mean)


@pytest.mark.slow
def test_plain_slope_fresh_seeds_bsm():
    """D-040 procedure on 10 fresh seed sets (R = 16, N = 2^10..2^20, RMSE vs my Black, OLS over
    all 11 points): every fitted slope in [-0.6, -0.4] and the mean within 0.03 of -0.5."""
    K, ref = F_C * MONEY, reference("bsm")
    Ns = 2 ** np.arange(10, 21)
    slopes = []
    for t in range(10):
        err = []
        for i, N in enumerate(Ns):
            P = np.array([vr.vanilla_mc(F_C, K, T_C, DF_C, SIG_C, CPS, method="plain",
                                        n_paths=int(N), seed=VSEED + 200_000 + 10_000 * t
                                        + 100 * i + r)[0] for r in range(16)])  # fmt: skip
            err.append(np.sqrt(np.mean((P - ref) ** 2, 0)))
        err = np.array(err)
        slopes += [ols_slope(Ns, err[:, j]) for j in range(2)]
    slopes = np.array(slopes)
    assert np.all((slopes >= -0.6) & (slopes <= -0.4)), slopes
    assert abs(slopes.mean() + 0.5) < 0.03, slopes.mean()


@pytest.mark.slow
def test_heston_vr_vs_my_qem():
    """VR at N = 2^18 of vr's antithetic and CV on heston_cal vs my own QE-M estimators (4 seeds
    each, SE-squared ratios): agreement within 15 %."""
    c = CASES["heston_cal"]
    K = c["F"] * MONEY
    se = {}
    for meth in ("plain", "antithetic", "cv"):
        theirs = [vr.vanilla_mc(c["F"], K, c["T"], c["DF"], hp(CAL), CPS, method=meth,
                                n_paths=2**18, seed=VSEED + 300_000 + r, n_steps=34)[1]
                  for r in range(4)]  # fmt: skip
        mine = [my_mc("heston_cal", meth, 2**18, VSEED + 310_000 + r)[1] for r in range(4)]
        se[meth] = (np.sqrt(np.mean(np.square(theirs), 0)), np.sqrt(np.mean(np.square(mine), 0)))
    for meth in ("antithetic", "cv"):
        vt = (se["plain"][0] / se[meth][0]) ** 2
        vm = (se["plain"][1] / se[meth][1]) ** 2
        assert np.allclose(vt, vm, rtol=0.15), (meth, vt, vm)


@pytest.mark.slow
def test_heston_qem_bias_claims_coarse_and_base():
    """D-043 bias claims (CV, 8 x 2^20 paths) vs my own QE-M + CV (8 x 2^20 paths) and my GP:
    the implementation's V_dt and mine agree within 3 combined SE at dt = T/9 and T/34, and the
    T/9 ATM bias is significantly positive in my run too."""
    b = pd.read_csv(FIG / "benchmark_bias.csv")
    ref = reference("heston_cal")
    for m in (9, 34):
        P, S = [], []
        for ch in range(8):
            p, s = my_mc("heston_cal", "cv", 2**20, VSEED + 400_000 + 1000 * m + ch, n_steps=m)
            P.append(p)
            S.append(s)
        mine, mse = np.mean(P, 0), np.sqrt(np.sum(np.square(S), 0)) / 8
        for j, lab in enumerate(LABELS):
            row = b[(b.set == "heston_cal") & (b.strike == lab) & (b.n_steps == m)].iloc[0]
            assert abs(row.price - mine[j]) <= 3 * np.hypot(row.se, mse[j]), (m, lab)
            if m == 9 and j == 0:
                assert (mine[j] - ref[j]) / mse[j] > 5


@pytest.mark.slow
def test_rqmc_scramble_errors_skewed_t_tails_heavy():
    """Finding 3 (documentation, not a code defect): the per-scrambling RQMC error of the BSM ATM
    call is right-skewed, for vr's jittered bits = 30 nets and for my bits = 53 nets alike. So the
    16-scramble t-statistic has P(|t| > 3) well above t_15's 0.9 %. The SE is right in mean square
    (variance level) but its tails are not Gaussian/t, so no single-run 3SE claim on Sobol'."""
    K, cp = np.array([F_C]), np.array([1.0])
    ref = black76(F_C, K, T_C, DF_C, SIG_C, cp)
    E = []
    for k in np.random.SeedSequence(VSEED + 600_000).spawn(8000):
        z = vr.sobol_normals(1, 512, np.random.default_rng(k))[0]
        FT = F_C * np.exp(-0.5 * SIG_C**2 * T_C + SIG_C * np.sqrt(T_C) * z)
        E.append(DF_C * np.maximum(FT - K[0], 0).mean() - ref[0])
    E = np.array(E)
    assert abs(E.mean()) < 3 * E.std() / np.sqrt(E.size)  # unbiased
    sk = np.mean((E - E.mean()) ** 3) / E.std() ** 3
    G = E.reshape(-1, 16)
    t = G.mean(1) / (G.std(1, ddof=1) / 4.0)
    assert sk > 1.0 and np.mean(np.abs(t) > 3) > 0.009


# ===================================================================== report numbers (script)
def _slope_trial(args):
    name, t, kind = args
    c = CASES[name]
    K = c["F"] * MONEY
    model = c["model"] if name == "bsm" else hp(c["model"])
    Ns = 2 ** np.arange(10, 21)
    out = []
    ref = reference(name) if kind == "rmse" else None
    for i, N in enumerate(Ns):
        P = np.array([vr.vanilla_mc(c["F"], K, c["T"], c["DF"], model, CPS, method="plain",
                                    n_paths=int(N), seed=VSEED + 500_000 + 50_000 * t + 100 * i
                                    + r, n_steps=c["n_steps"])[0] for r in range(16)])  # fmt: skip
        out.append(np.sqrt(np.mean((P - ref) ** 2, 0)) if ref is not None else P.std(0, ddof=1))
    out = np.array(out)
    return name, t, [ols_slope(Ns, out[:, j]) for j in range(2)]


def _bias_chunk(args):
    m, ch = args
    return m, my_mc("heston_cal", "cv", 2**20, VSEED + 400_000 + 1000 * m + ch, n_steps=m)


def main() -> None:
    np.set_printoptions(precision=5, suppress=False, linewidth=140)
    print("== references (mine) vs implementation COS")
    refs = pd.read_csv(FIG / "benchmark_references.csv")
    for name in CASES:
        mine = reference(name)
        th = refs[refs.set == name].set_index("strike").loc[list(LABELS), "ref"].to_numpy()
        print(f"  {name}: mine {mine}  diff {mine - th}")
    print("== BSM closed-form VR (cv, antithetic) vs reported VR_se at 2^20")
    s = pd.read_csv(FIG / "benchmark_summary.csv").set_index(["set", "strike", "method"])
    for j, lab in enumerate(LABELS):
        cv, an = vr_closed_form_bsm(F_C, F_C * MONEY[j], T_C, SIG_C, CPS[j])
        print(
            f"  {lab}: cv {cv:.4f} (rep {s.loc[('bsm', lab, 'cv'), 'VR_se_N1048576']:.4f}), "
            f"anti {an:.4f} (rep {s.loc[('bsm', lab, 'antithetic'), 'VR_se_N1048576']:.4f})"
        )
        for N in (2**10, 2**20):
            print(
                f"    CV leading bias at N = {N}: "
                f"{cv_bias_leading(F_C, F_C * MONEY[j], T_C, DF_C, SIG_C, CPS[j], N):.3e}"
            )
    workers = min(8, os.cpu_count() or 1)
    with ProcessPoolExecutor(workers) as ex:
        print("== fresh-seed plain slope distribution (D-040 procedure)", flush=True)
        jobs = [("bsm", t, "rmse") for t in range(40)] + [
            (n, t, "std") for n in ("heston_cal", "heston_fo") for t in range(6)
        ]
        res = list(ex.map(_slope_trial, jobs))
        for name in CASES:
            sl = np.array([r[2] for r in res if r[0] == name])
            for j, lab in enumerate(LABELS):
                x = sl[:, j]
                out = int(np.sum(np.abs(x + 0.5) > 0.1))
                print(
                    f"  {name} {lab}: n = {x.size}, mean {x.mean():+.4f}, "
                    f"sd {x.std(ddof=1):.4f}, min {x.min():+.4f}, max {x.max():+.4f}, "
                    f"outside band {out}"
                )
        print("== my QE-M + CV bias (8 x 2^20 per level) vs GP and vs implementation", flush=True)
        b = pd.read_csv(FIG / "benchmark_bias.csv")
        ref = reference("heston_cal")
        jobs = [(m, ch) for m in (9, 17, 34, 68) for ch in range(8)]
        res = list(ex.map(_bias_chunk, jobs))
    for m in (9, 17, 34, 68):
        P = np.array([r[1][0] for r in res if r[0] == m])
        S = np.array([r[1][1] for r in res if r[0] == m])
        mine, mse = P.mean(0), np.sqrt((S**2).sum(0)) / len(P)
        for j, lab in enumerate(LABELS):
            row = b[(b.set == "heston_cal") & (b.strike == lab) & (b.n_steps == m)].iloc[0]
            print(
                f"  m = {m:>2} {lab}: mine bias {mine[j] - ref[j]:+.4f} +- {mse[j]:.4f} "
                f"({(mine[j] - ref[j]) / mse[j]:+.2f} SE) | impl {row.bias:+.4f} +- {row.se:.4f}"
                f" | impl - mine {(row.price - mine[j]) / np.hypot(row.se, mse[j]):+.2f} SE"
            )


if __name__ == "__main__":
    sys.exit(main())
