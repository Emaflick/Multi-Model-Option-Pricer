import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(20261105)


@pytest.fixture(scope="session")
def bsm_gate_grid():
    """Phase 1 gate domain (docs/ROADMAP.md; derivations/bsm.md §2.5), dense and vectorised.

    sigma in [0.05, 1] (40), T in [0.02, 5] (30, geometric), z = ln(F/K)/(sigma sqrt T) in [-3, 3]
    (41, includes 0), DF in {0.8, 0.9, 1}, F = 100. Arrays broadcast to shape (40, 30, 41, 3).
    """
    sig = np.linspace(0.05, 1.0, 40)[:, None, None, None]
    T = np.geomspace(0.02, 5.0, 30)[None, :, None, None]
    z = np.linspace(-3.0, 3.0, 41)[None, None, :, None]
    DF = np.array([0.8, 0.9, 1.0])[None, None, None, :]
    F = 100.0
    K = F * np.exp(-z * sig * np.sqrt(T))
    sig, T, K, DF = np.broadcast_arrays(sig, T, K, DF)
    return {"F": F, "K": K, "T": T, "DF": DF, "sigma": sig}
