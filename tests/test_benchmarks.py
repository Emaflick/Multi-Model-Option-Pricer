"""Phase 6 estimators (engines/vr.py) and the analysis of scripts/run_benchmarks.py.

docs/ROADMAP.md Phase 6; design D-039..D-042. Exact identities are tested to round-off; MC
comparisons are 3SE (heston-mc.md eq. 2) with seeds fixed a priori. The full benchmark and its
gates run in scripts/run_benchmarks.py and scripts/phase6_gates.py.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtr

from qpricer.engines import mc, vr
from qpricer.engines.fourier import cos_price
from qpricer.models import bsm
from qpricer.models.heston import HestonParams

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261111
FO = HestonParams(kappa=1.5768, theta=0.0398, xi=0.5751, rho=-0.5711, v0=0.0175)
K, CP = np.array([100.0, 90.0]), np.array([1.0, -1.0])


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RB = load_script("run_benchmarks")


# ------------------------------------------------------------------ exact identities and contracts
def test_heston_terminal_bit_identical_to_mc_simulate():
    """Given rng.standard_normal((2, N)) blocks in step order, the QE-M state equals mc.simulate."""
    t = mc.uniform_grid(1.0, 8)
    x_ref, _ = mc.simulate(FO, t, 3000, np.random.default_rng(SEED))
    rng = np.random.default_rng(SEED)
    x = vr.heston_terminal(FO, t, (rng.standard_normal((2, 3000)) for _ in t))
    assert np.array_equal(x, x_ref[0])


def test_plain_heston_equals_european_price_and_contract():
    """method='plain' is mc.european_price on the same paths; returns (price, se, n_paths, seed)."""
    a = vr.vanilla_mc(100.0, K, 1.0, 0.95, FO, CP, method="plain", n_paths=4096, seed=SEED,
                      n_steps=16)  # fmt: skip
    b = mc.european_price(100.0, K, 1.0, 0.95, FO, CP, n_steps=16, n_paths=4096, seed=SEED)
    # same paths; DF is applied before (vr) vs after (mc) the mean, so equal to round-off
    assert np.allclose(a[0], b[0], rtol=1e-14, atol=0) and np.allclose(a[1], b[1], rtol=1e-13)
    assert a[2:] == (4096, SEED)


@pytest.mark.parametrize("method", vr.METHODS)
def test_reproducible_and_seed_sensitive(method):
    kw = dict(method=method, n_paths=2048, n_steps=4)
    a = vr.vanilla_mc(100.0, K, 1.0, 1.0, FO, CP, seed=SEED, **kw)
    b = vr.vanilla_mc(100.0, K, 1.0, 1.0, FO, CP, seed=SEED, **kw)
    c = vr.vanilla_mc(100.0, K, 1.0, 1.0, FO, CP, seed=SEED + 1, **kw)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(a[0], c[0]) and a[2] == 2048 and np.all(a[1] > 0)


def test_antithetic_pair_counting():
    """N paths = N/2 pairs: SE is the SD of the N/2 pair means over sqrt(N/2), never over N."""
    rng = np.random.default_rng(SEED)
    z = rng.standard_normal(500)
    g = np.concatenate([np.exp(z), np.exp(-z)])
    est, se = vr.antithetic_estimate(g, DF=0.9)
    h = 0.5 * (np.exp(z) + np.exp(-z))
    assert est == pytest.approx(0.9 * h.mean(), rel=1e-15)
    assert se == pytest.approx(0.9 * h.std(ddof=1) / np.sqrt(500), rel=1e-13)
    lin = np.concatenate([2.0 + 3.0 * z, 2.0 - 3.0 * z])  # linear payoff: pair means are constant
    assert vr.antithetic_estimate(lin)[1] < 1e-14
    with pytest.raises(ValueError):
        vr.antithetic_estimate(g[:-1])
    with pytest.raises(ValueError):
        vr.vanilla_mc(100.0, 100.0, 1.0, 1.0, 0.2, 1, method="antithetic", n_paths=1001, seed=1)


def test_antithetic_bsm_uses_negated_draws():
    """BSM antithetic = Black payoffs on (Z, -Z), Z = rng.standard_normal((1, N/2))."""
    F, T, DF, s, N = 100.0, 0.5, 0.97, 0.25, 4000
    p, se, n, _ = vr.vanilla_mc(F, K, T, DF, s, CP, method="antithetic", n_paths=N, seed=SEED)
    z = np.random.default_rng(SEED).standard_normal((1, N // 2))[0]
    z = np.concatenate([z, -z])
    g = DF * np.maximum(
        CP[:, None] * (F * np.exp(-0.5 * s * s * T + s * np.sqrt(T) * z) - K[:, None]), 0
    )
    assert np.allclose(p, 0.5 * (g[:, : N // 2] + g[:, N // 2 :]).mean(1), rtol=1e-14) and n == N


def test_sobol_normals_stratified_and_validated():
    """Scrambled Sobol' net: each 1-D projection has exactly one point per cell [j/n, (j+1)/n)."""
    z = vr.sobol_normals(5, 1024, np.random.default_rng(SEED))
    assert z.shape == (5, 1024) and np.all(np.isfinite(z))
    cells = np.sort(np.floor(ndtr(z) * 1024).astype(int), axis=1)
    assert np.array_equal(cells, np.broadcast_to(np.arange(1024), (5, 1024)))
    z2 = vr.sobol_normals(5, 1024, np.random.default_rng(SEED + 1))
    assert not np.array_equal(z, z2)
    with pytest.raises(ValueError):
        vr.sobol_normals(2, 1000, np.random.default_rng(1))
    with pytest.raises(ValueError):  # 16 scramblings must divide N
        vr.vanilla_mc(100.0, 100.0, 1.0, 1.0, 0.2, 1, method="sobol", n_paths=1000, seed=1)


def test_sobol_se_is_from_the_scramblings():
    """RQMC: price = mean of R scrambled estimates, SE = their SD / sqrt(R) (D-042)."""
    F, T, DF, s, R, n = 100.0, 1.0, 0.9, 0.2, 8, 256
    p, se, N, _ = vr.vanilla_mc(F, 100.0, T, DF, s, 1, method="sobol", n_paths=R * n, seed=SEED,
                                n_scramble=R)  # fmt: skip
    est = []
    for ss in np.random.SeedSequence(SEED).spawn(R):
        z = vr.sobol_normals(1, n, np.random.default_rng(ss))[0]
        est.append(
            DF * np.maximum(F * np.exp(-0.5 * s * s * T + s * np.sqrt(T) * z) - 100.0, 0).mean()
        )
    assert N == R * n and p == pytest.approx(np.mean(est), rel=1e-14)
    assert se == pytest.approx(np.std(est, ddof=1) / np.sqrt(R), rel=1e-12)


def test_input_validation():
    with pytest.raises(ValueError):
        vr.vanilla_mc(100.0, 100.0, 1.0, 1.0, 0.2, 1, method="qmc", n_paths=64, seed=1)
    with pytest.raises(ValueError):  # BSM is exact in one step
        vr.vanilla_mc(100.0, 100.0, 1.0, 1.0, 0.2, 1, method="plain", n_paths=64, seed=1, n_steps=4)
    with pytest.raises(ValueError):
        vr.vanilla_mc(100.0, 100.0, 1.0, 1.0, -0.2, 1, method="plain", n_paths=64, seed=1)


# ------------------------------------------------------------------ 3SE unbiasedness (fast)
@pytest.mark.parametrize("method", vr.METHODS)
def test_bsm_methods_unbiased_3se(method):
    F, T, DF, s = 100.0, 1.0, 0.95, 0.2
    p, se, _, _ = vr.vanilla_mc(F, K, T, DF, s, CP, method=method, n_paths=2**16, seed=SEED)
    assert np.all(np.abs(p - bsm.price(F, K, T, DF, s, CP)) <= 3 * se)


@pytest.mark.parametrize("method", vr.METHODS)
def test_heston_methods_vs_cos_3se(method):
    """FO, dt = 1/32, 2^14 paths: QE-M bias (Phase 2, ~1e-2) is far below 3 SE (~0.05) here."""
    p, se, _, _ = vr.vanilla_mc(100.0, K, 1.0, 1.0, FO, CP, method=method, n_paths=2**14,
                                seed=SEED, n_steps=32)  # fmt: skip
    assert np.all(np.abs(p - cos_price(100.0, K, 1.0, 1.0, FO, CP)) <= 3 * se)


def test_variance_reduction_orders_bsm():
    """At equal N on BSM: antithetic and CV (F_T control) reduce the SE, Sobol' RQMC most."""
    se = {m: vr.vanilla_mc(100.0, 100.0, 1.0, 1.0, 0.2, 1, method=m, n_paths=2**14, seed=SEED)[1]
          for m in vr.METHODS}  # fmt: skip
    assert se["sobol"] < se["cv"] < se["plain"] and se["antithetic"] < se["plain"]


# ------------------------------------------------------------------ driver analysis functions
def test_loglog_slope_exact_and_noisy():
    N = 2.0 ** np.arange(10, 21)
    s, sd = RB.loglog_slope(N, 3.0 * N**-0.5)
    assert s == pytest.approx(-0.5, abs=1e-12) and sd < 1e-12
    y = N**-0.5 * np.exp(np.random.default_rng(SEED).normal(0, 0.1, N.size))
    s, sd = RB.loglog_slope(N, y)
    assert abs(s + 0.5) < 4 * sd and 0 < sd < 0.05
    with pytest.raises(ValueError):
        RB.loglog_slope(N, np.r_[0.0, y[1:]])


def test_pooled_is_work_weighted():
    g = pd.DataFrame({"N": [100, 300], "price": [1.0, 2.0], "se": [0.3, 0.1]})
    v, se = RB.pooled(g)
    assert v == pytest.approx(1.75) and se == pytest.approx(np.hypot(30, 30) / 400)


def test_seeds_distinct_over_the_full_design():
    s = {
        RB.seed_of(a, b, c, r)
        for a in range(3)
        for b in range(4)
        for c in range(11)
        for r in range(16)
    }
    assert len(s) == 3 * 4 * 11 * 16


@pytest.mark.slow
def test_driver_quick_run_writes_manifest_and_is_deterministic(tmp_path, monkeypatch):
    """--quick end to end, twice: every figure/table listed with a matching sha256; the
    replication table is identical across runs and worker counts (seeded tasks)."""
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))  # spawned workers import run_benchmarks
    monkeypatch.setitem(sys.modules, "run_benchmarks", RB)  # pickle resolves run_task by name
    a = RB.main(["--quick", "--workers", "2", "--out", str(tmp_path / "a")])
    b = RB.main(["--quick", "--workers", "1", "--out", str(tmp_path / "b")])
    man = pd.read_csv(a / "benchmark_manifest.csv")
    assert set(man.file) == set(RB.FIGURES + RB.TABLES)
    assert all(RB.sha256(a / f) == h for f, h in zip(man.file, man.sha256, strict=True))
    cols = ["set", "method", "N", "rep", "seed", "strike", "price", "se"]
    ra, rb = pd.read_csv(a / "benchmark_runs.csv"), pd.read_csv(b / "benchmark_runs.csv")
    pd.testing.assert_frame_equal(ra[cols], rb[cols])
