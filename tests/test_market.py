"""Tests for qpricer.market: parity regression (eqs. P1-P3), outlier rejection, Phase 0 gates.

Constructed inputs below are unit-test data (not market data). The last test runs the gates on the
real snapshot behind the committed data/processed/market_*.csv and FAILS (never skips) if the
gitignored raw parquet is missing (D-006).
"""

import logging
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from qpricer.data import PROCESSED_DIR, RAW_DIR, arbitrage_pairs, build_surface
from qpricer.market import (
    DF_BOUNDS,
    INBAND_MAX,
    N_MIN,
    N_NEAR,
    R2_MIN,
    REJ_FRAC_MAX,
    MarketData,
    fit_expiry,
    implied_forwards,
    market_data,
    ols_parity,
    theil_sen,
)

F0, DF0 = 7432.123456, 0.9617  # constructed parameters


def legs(K, F=F0, DF=DF0):
    """Exact parity-consistent (C, P): P = DF*max(K - F, 0) + 20, C = P + DF*(F - K)."""
    P = DF * np.maximum(K - F, 0.0) + 20.0
    return P + DF * (F - K), P


def test_ols_exact_parity_recovers_F_DF():
    K = np.arange(6000.0, 9000.0, 25.0)
    C, P = legs(K)
    fit = ols_parity(K, C - P)
    assert abs(fit["F"] / F0 - 1.0) < 1e-12
    assert abs(fit["DF"] - DF0) < 1e-12
    assert fit["R2"] == pytest.approx(1.0, abs=1e-14)


def test_ols_noisy_standard_errors_calibrated(rng):
    """eq. (P3): empirical std of F, DF over 1000 noisy replications matches the reported se."""
    K = np.arange(7300.0, 7570.0, 10.0)
    C, P = legs(K)
    fits = [ols_parity(K, C - P + rng.normal(0.0, 0.5, K.size)) for _ in range(1000)]
    est = pd.DataFrame(fits)
    for p, true in (("F", F0), ("DF", DF0)):
        assert np.std(est[p]) / est[f"{p}_se"].mean() == pytest.approx(1.0, abs=0.1)
        assert abs(est[p].mean() - true) < 4 * est[f"{p}_se"].mean() / np.sqrt(len(est))
    assert (est.R2 > 0.999).all() and (est.R2 < 1.0).all()


def test_ols_rejects_degenerate_input():
    with pytest.raises(ValueError):
        ols_parity(np.array([1.0, 2.0]), np.array([1.0, 0.0]))
    with pytest.raises(ValueError):
        ols_parity(np.full(5, 7400.0), np.arange(5.0))


def test_theil_sen_robust_to_outliers():
    K = np.arange(7300.0, 7500.0, 10.0)
    y = DF0 * (F0 - K)
    y[[3, 11]] += [250.0, -120.0]
    a, b = theil_sen(K, y)
    assert b == pytest.approx(-DF0, abs=1e-12)
    assert a == pytest.approx(DF0 * F0, rel=1e-12)


def test_fit_expiry_rejects_stale_quotes_and_recovers_exactly():
    K = np.arange(7300.0, 7600.0, 5.0)
    C, P = legs(K)
    near = np.argsort(np.abs(K - F0))[:N_NEAR]
    bad = np.sort(near[[0, 5, 12]])
    C[bad[0]] += 200.0  # stale call
    P[bad[1]] -= 100.0  # stale put
    C[bad[2]] += 30.0
    fit = fit_expiry(K, C, P, np.ones_like(K))
    assert fit["n_sel"] == N_NEAR and fit["n_rej"] == 3 and fit["n_used"] == N_NEAR - 3
    assert fit["rejected_K"] == ";".join(f"{k:g}" for k in K[bad])
    assert abs(fit["F"] / F0 - 1) < 1e-12 and abs(fit["DF"] - DF0) < 1e-12
    assert fit["R2_all"] < R2_MIN  # plain OLS on the same selection would fail the gate


def test_fit_expiry_no_rejection_on_clean_noisy_data(rng):
    K = np.arange(7300.0, 7600.0, 5.0)
    C, P = legs(K)
    C = C + rng.normal(0.0, 0.2, K.size)
    P = P + rng.normal(0.0, 0.2, K.size)
    fit = fit_expiry(K, C, P, np.full(K.size, 0.5))
    assert fit["n_rej"] == 0 and fit["n_used"] == N_NEAR
    assert abs(fit["F"] - F0) < 4 * fit["F_se"] and abs(fit["DF"] - DF0) < 4 * fit["DF_se"]


def test_fit_expiry_uses_nearest_strikes_to_forward():
    K = np.arange(5000.0, 9000.0, 10.0)
    C, P = legs(K)
    fit = fit_expiry(K, C, P, np.ones_like(K))
    nearest = K[np.argsort(np.abs(K - F0))[:N_NEAR]]
    assert fit["n_sel"] == N_NEAR
    assert fit["logm_max"] == pytest.approx(np.max(np.abs(np.log(nearest / F0))), rel=1e-12)


def chain(expiry, K, C, P, T=0.5, root="SPX", half=0.5):
    """Clean-quote frame for implied_forwards: bid/ask = mid -/+ half (pair half-width 2*half)."""
    rows = [
        pd.DataFrame({"option_type": t, "strike": K, "mid": m, "bid": m - half, "ask": m + half})
        for t, m in (("C", C), ("P", P))
    ]
    return pd.concat(rows).assign(expiry=expiry, root=root, T=T)


def test_inband_gate_drops_noisy_expiry_that_passes_R2(rng):
    """D-005: R^2 > 0.999 is not enough; every used pair must sit inside its own bid-ask band."""
    K = np.arange(7300.0, 7600.0, 5.0)
    C, P = legs(K)
    tab = implied_forwards(chain("2027-01-15", K, C + rng.normal(0.0, 0.6, K.size), P, half=0.25))
    assert tab.R2.iloc[0] > R2_MIN and tab.max_res_h.iloc[0] > INBAND_MAX
    assert tab.reason.iloc[0] == f"max|res|/h>{INBAND_MAX:g}" and not tab.kept.iloc[0]


@pytest.mark.parametrize("n_bad", [5, 6])
def test_rejected_fraction_capped_at_theil_sen_breakdown(n_bad):
    """D-005: 5/20 rejected (< 0.293) is kept and exact; 6/20 (>= 0.293) drops the expiry."""
    K = np.arange(7300.0, 7600.0, 5.0)
    C, P = legs(K)
    near = np.sort(np.argsort(np.abs(K - F0))[:N_NEAR])
    C[near[[1, 4, 7, 10, 13, 16][:n_bad]]] += np.array([300, -250, 280, -260, 310, -240])[:n_bad]
    row = implied_forwards(chain("2027-01-15", K, C, P)).iloc[0]
    assert row.n_rej == n_bad
    if n_bad < REJ_FRAC_MAX * N_NEAR:
        assert row.kept and abs(row.F / F0 - 1) < 1e-12
    else:
        assert not row.kept and row.reason == f"n_rej/n_sel>={REJ_FRAC_MAX:.3f}"


def test_unbracketed_forward_is_flagged_and_logged(caplog):
    """F6: F outside [min K, max K] of the used pairs -> bracketed False + WARNING (not a gate)."""
    K = np.arange(7460.0, 7800.0, 10.0)
    with caplog.at_level(logging.WARNING, logger="qpricer.market"):
        tab = implied_forwards(chain("2027-01-15", K, *legs(K)))
    assert tab.kept.iloc[0] and not tab.bracketed.iloc[0]
    assert any("outside the strikes used" in r.message for r in caplog.records)
    K = np.arange(7300.0, 7600.0, 10.0)
    assert implied_forwards(chain("2027-01-15", K, *legs(K))).bracketed.iloc[0]


def test_implied_forwards_gates_drop_and_log(rng, caplog):
    K = np.arange(7200.0, 7700.0, 10.0)
    noise = rng.normal(0.0, 30.0, K.size)
    frames = [
        chain("2027-01-15", K, *legs(K)),  # good
        chain("2027-02-19", K, *legs(K, DF=1.002)),  # DF > 1
        chain("2027-03-19", K[: N_MIN - 1], *legs(K[: N_MIN - 1])),  # too few pairs
        chain("2027-04-16", K, 100.0 + noise, 100.0 - noise),  # no parity line: R2 ~ 0
        chain("2027-05-21", K, *legs(K, DF=0.75)),  # DF < 0.8
    ]
    with caplog.at_level(logging.WARNING, logger="qpricer.market"):
        tab = implied_forwards(pd.concat(frames, ignore_index=True)).set_index("expiry")
    assert tab.kept.tolist() == [True, False, False, False, False]
    assert tab.loc["2027-02-19", "reason"].startswith("DF not in")
    assert tab.loc["2027-03-19", "reason"] == f"n_pairs<{N_MIN}"
    assert tab.loc["2027-04-16", "reason"] == f"R2<={R2_MIN}"
    assert tab.loc["2027-05-21", "reason"].startswith("DF not in")
    assert sum("dropping expiry" in r.message for r in caplog.records) == 4
    md = market_data(tab.reset_index())
    assert list(md) == ["2027-01-15"]
    assert md["2027-01-15"].F == pytest.approx(F0, rel=1e-12)
    assert md["2027-01-15"].DF == pytest.approx(DF0, abs=1e-12)
    assert tab.loc["2027-01-15", "r"] == pytest.approx(-np.log(DF0) / 0.5, rel=1e-10)


def test_market_data_frozen_validated():
    md = MarketData(F=7400.0, DF=0.98, T=0.5)
    assert md.r == pytest.approx(-np.log(0.98) / 0.5, rel=1e-15)
    with pytest.raises(FrozenInstanceError):
        md.F = 1.0
    for bad in ((-1.0, 0.98, 0.5), (7400.0, np.nan, 0.5), (7400.0, 0.98, 0.0)):
        with pytest.raises(ValueError):
            MarketData(*bad)


def test_real_snapshot_parity_gates():
    """Gates on the real SPX snapshot (ROADMAP Phase 0) + reproduction of the committed table.

    Every kept expiry: R^2 > 0.999 and DF in (0.8, 1]. Non-vacuity: >= 75% of the snapshot's
    expiries kept and the kept set reaches T >= 1 (usable surface for Phase 5).
    """
    csvs = sorted(PROCESSED_DIR.glob("market_*.csv"))
    assert csvs, "no data/processed/market_*.csv: run scripts/build_market.py"
    raw = RAW_DIR / csvs[-1].name.replace("market_", "spx_").replace(".csv", ".parquet")
    if not raw.exists():
        pytest.fail(f"raw snapshot {raw} missing (data/raw is gitignored): run fetch_spx.py")
    s = build_surface(raw)
    tab = s["table"]
    kept = tab[tab.kept]
    assert (kept.R2 > R2_MIN).all()
    assert ((kept.DF > DF_BOUNDS[0]) & (kept.DF <= DF_BOUNDS[1])).all()
    assert (kept.n_used >= N_MIN).all()
    assert (kept.max_res_h <= INBAND_MAX).all() and (kept.n_rej < REJ_FRAC_MAX * kept.n_sel).all()
    assert (tab.reason[~tab.kept] != "").all() and (kept.reason == "").all()
    assert tab.kept.mean() >= 0.75 and kept["T"].max() >= 1.0
    otm = s["otm"]
    assert ((otm.option_type == "P") == (otm.strike < otm.F)).all()
    surf = s["surface"]  # calibration input: no strict static arbitrage left (D-010)
    for (_, t), g in surf.groupby(["expiry", "option_type"]):
        g = g.sort_values("strike")
        assert not arbitrage_pairs(
            g.strike.to_numpy(), g.bid.to_numpy(), g.ask.to_numpy(), t == "P"
        ).any()
    assert len(surf) + len(s["removed"]) == len(otm)
    stamp = csvs[-1].stem.removeprefix("market_")
    ref_s = pd.read_csv(PROCESSED_DIR / f"surface_{stamp}.csv")
    assert ref_s[["rows_in", "rows_out"]].to_numpy().tolist() == (
        s["surface_stats"][["rows_in", "rows_out"]].to_numpy().tolist()
    )
    ref_rm = pd.read_csv(PROCESSED_DIR / f"arb_removed_{stamp}.csv")
    assert ref_rm.contractSymbol.tolist() == s["removed"].contractSymbol.tolist()

    ref = pd.read_csv(csvs[-1], keep_default_na=False, na_values=[""])
    assert ref.expiry.tolist() == tab.expiry.tolist()
    assert ref.kept.tolist() == tab.kept.tolist()
    num = ["T", "n_pairs", "n_used", "F", "F_se", "DF", "DF_se", "r", "R2", "R2_all", "max_res_h"]
    got = tab[num].to_numpy(float, na_value=np.nan)
    np.testing.assert_allclose(ref[num].to_numpy(float), got, rtol=1e-12)
