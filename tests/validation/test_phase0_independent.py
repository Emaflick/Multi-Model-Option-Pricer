"""Independent validation of Phase 0 (setup & SPX data) -- validator agent.

Every reference value here is computed from scratch from the raw snapshot with pandas / numpy /
scipy only. No qpricer helper is used to produce a reference value; the qpricer public API
(``clean_chain``, ``implied_forwards``, ``build_surface``, ``market_data``) is only called as the
system under test. Methods deliberately differ from the implementation where possible:

* cleaning: re-implemented from the ROADMAP rules (own root parsing, own filters);
* T: ``datetime`` + ``zoneinfo`` per expiry, cross-checked against a hand-coded US DST rule;
* parity fit (replication of the documented rule): ``scipy.stats.theilslopes(method='joint')``,
  ``scipy.stats.median_abs_deviation(scale='normal')``, ``np.polyfit``; standard errors from the
  full (uncentred, QR) covariance + delta method, not the centred closed form of ``market.py``;
* second method: least-absolute-deviations (LAD, L1) regression via ``scipy.optimize.linprog``
  on the 20 nearest pairs with NO outlier rejection;
* outlier justification: strict static-arbitrage bounds on bid/ask (monotonicity, vertical spread).

Synthetic chains in the gate-power tests are labelled ``synthetic`` and are unit-test inputs only.
Run as a script (``.venv/bin/python tests/validation/test_phase0_independent.py``) to print the
diagnostics quoted in ``docs/validation/phase0.md``.
"""

from __future__ import annotations

import ast
import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from scipy import optimize, stats

REPO = Path(__file__).resolve().parents[2]
STAMP = "20261106_2213"
RAW = REPO / "data" / "raw" / f"spx_{STAMP}.parquet"
MARKET_CSV = REPO / "data" / "processed" / f"market_{STAMP}.csv"
CLEANING_CSV = REPO / "data" / "processed" / f"cleaning_{STAMP}.csv"
ET = ZoneInfo("America/New_York")
YEAR_S = 365 * 86400  # ACT/365F
N_NEAR, N_MIN, Z = 20, 10, 4.0  # documented in D-004 / D-005 (rule under test)
# NYSE early-close days (13:00 ET); PM-settled SPXW settle on the 13:00 closing value.
# Derived from the NYSE rule (13:00 close on the day after Thanksgiving, and on July 3 / Dec 24 when
# they are regular weekday sessions): Thanksgiving 2026-11-26, 2027-11-25, 2028-11-23; 2026-12-24
# Thu; 2028-07-03 Mon (July 4 Tue). Not early closes: 2027-12-24 (Christmas observed, closed),
# 2027-07-02 (July 4 observed Mon 07-05), 2028-12-24 (Sunday).
EARLY_CLOSE = {"2026-11-27", "2026-12-24", "2027-11-26", "2028-07-03", "2028-11-24"}
REJ_BREAKDOWN = 1 - 1 / np.sqrt(2)  # Theil-Sen breakdown point


# ----------------------------------------------------------------------------- fixtures
def _need(path: Path) -> Path:
    if not path.exists():
        pytest.fail(f"{path} missing: run scripts/fetch_spx.py and scripts/build_market.py")
    return path


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return pd.read_parquet(_need(RAW))


@pytest.fixture(scope="module")
def meta() -> dict:
    return json.loads(pq.read_schema(_need(RAW)).metadata[b"qpricer"])


@pytest.fixture(scope="module")
def csv() -> pd.DataFrame:
    return pd.read_csv(_need(MARKET_CSV))


@pytest.fixture(scope="module")
def cleaning_csv() -> pd.DataFrame:
    return pd.read_csv(_need(CLEANING_CSV))


@pytest.fixture(scope="module")
def mine(raw) -> dict:
    return independent_pipeline(raw)


# ------------------------------------------------------------------ independent pipeline
def my_clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """ROADMAP cleaning re-implemented; returns (clean rows, row counts after each step)."""
    d = raw.copy()
    d["root"] = d.contractSymbol.str[:-15]  # OCC: ROOT + YYMMDD + C/P + 8-digit strike
    spx_dates = set(d.expiry[d.root == "SPX"])
    counts = [len(d)]
    d = d[(d.root == "SPX") | ~d.expiry.isin(spx_dates)]
    counts.append(len(d))
    d = d[d.bid > 0]
    counts.append(len(d))
    d = d[np.isfinite(d.ask) & (d.ask >= d.bid)]
    counts.append(len(d))
    d = d[(d.ask - d.bid) / ((d.ask + d.bid) / 2) <= 0.5]
    counts.append(len(d))
    d = d[d.openInterest > 0].copy()
    counts.append(len(d))
    d["mid"] = (d.bid + d.ask) / 2
    return d, counts


def us_dst(day: date) -> bool:
    """US DST rule (2007+): from 2nd Sunday of March to 1st Sunday of November."""

    def nth_sunday(y: int, m: int, n: int) -> date:
        first = date(y, m, 1)
        return first + timedelta(days=(6 - first.weekday()) % 7 + 7 * (n - 1))

    return nth_sunday(day.year, 3, 2) <= day < nth_sunday(day.year, 11, 1)


def settle_et(expiry: str, root: str) -> datetime:
    """Settlement instant in ET: SPX AM 09:30 (SOQ), SPXW PM 16:00 (13:00 on early-close days)."""
    y, m, d = map(int, expiry.split("-"))
    if root == "SPX":
        hh, mm = 9, 30
    else:
        hh, mm = (13, 0) if expiry in EARLY_CLOSE else (16, 0)
    return datetime(y, m, d, hh, mm, tzinfo=ET)


def my_T(expiry: str, root: str, snap: datetime) -> float:
    return (settle_et(expiry, root) - snap).total_seconds() / YEAR_S


def pairs_of(g: pd.DataFrame) -> pd.DataFrame:
    """Strikes with both a clean call and a clean put: K, y = C - P (mids), h = half-width."""
    c = g[g.option_type == "C"].set_index("strike")
    p = g[g.option_type == "P"].set_index("strike")
    assert c.index.is_unique and p.index.is_unique
    K = np.intersect1d(c.index.to_numpy(float), p.index.to_numpy(float))
    c, p = c.loc[K], p.loc[K]
    return pd.DataFrame(
        {
            "K": K,
            "y": c.mid.to_numpy() - p.mid.to_numpy(),
            "h": 0.5 * ((c.ask - c.bid).to_numpy() + (p.ask - p.bid).to_numpy()),
        }
    )


def near_window(pp: pd.DataFrame, n_near: int) -> tuple[pd.DataFrame, float]:
    i0 = int(np.argmin(np.abs(pp.y.to_numpy())))
    F0 = float(pp.K.iloc[i0] + pp.y.iloc[i0])
    sel = np.sort(np.argsort(np.abs(pp.K.to_numpy() - F0), kind="stable")[:n_near])
    return pp.iloc[sel].reset_index(drop=True), F0


def ols_qr(K: np.ndarray, y: np.ndarray) -> dict:
    """OLS y = A + B K with full covariance s^2 (X'X)^-1 via QR; F = -A/B by the delta method."""
    X = np.column_stack([np.ones_like(K), K])
    Q, R = np.linalg.qr(X)
    A, B = np.linalg.solve(R, Q.T @ y)
    res = y - A - B * K
    s2 = res @ res / (K.size - 2)
    Rinv = np.linalg.inv(R)
    cov = s2 * Rinv @ Rinv.T
    g = np.array([-1.0 / B, A / B**2])
    return {"F_se": float(np.sqrt(g @ cov @ g)), "DF_se": float(np.sqrt(cov[1, 1]))}


def my_fit(pp: pd.DataFrame, n_near: int = N_NEAR) -> dict:
    """Documented rule (D-004/D-005) coded independently with scipy primitives."""
    w, F0 = near_window(pp, n_near)
    K, y, h = w.K.to_numpy(), w.y.to_numpy(), w.h.to_numpy()
    ts = stats.theilslopes(y, K, method="joint")
    e = y - ts.intercept - ts.slope * K
    keep = np.abs(e) <= np.maximum(Z * stats.median_abs_deviation(e, scale="normal"), h)
    out = {"F0": F0, "n_sel": len(K), "n_used": int(keep.sum())}
    out["rejected_K"] = ";".join(f"{k:g}" for k in K[~keep])
    if keep.sum() < 3:
        return out
    b, a = np.polyfit(K[keep] - F0, y[keep], 1)
    out |= {"DF": -b, "F": F0 - a / b, "R2": np.corrcoef(K[keep], y[keep])[0, 1] ** 2}
    out |= ols_qr(K[keep], y[keep])
    res = y[keep] - out["DF"] * (out["F"] - K[keep])
    out["max_res_over_h"] = float(np.max(np.abs(res) / h[keep]))
    out["Kmin"], out["Kmax"] = float(K[keep].min()), float(K[keep].max())
    return out


def lad(K: np.ndarray, y: np.ndarray, K0: float) -> tuple[float, float]:
    """L1 regression y = a + b (K - K0) as an LP; returns (F, DF)."""
    n = K.size
    cost = np.r_[0.0, 0.0, np.ones(2 * n)]
    A_eq = np.column_stack([np.ones(n), K - K0, np.eye(n), -np.eye(n)])
    bounds = [(None, None)] * 2 + [(0, None)] * (2 * n)
    sol = optimize.linprog(cost, A_eq=A_eq, b_eq=y, bounds=bounds, method="highs")
    assert sol.status == 0
    a, b = sol.x[:2]
    return K0 - a / b, -b


def accept(f: dict, n_pairs: int) -> bool:
    """ROADMAP gates + D-005 acceptance (in band, rejected fraction < breakdown), coded here."""
    return bool(
        n_pairs >= N_MIN
        and f.get("n_used", 0) >= N_MIN
        and f["n_sel"] - f["n_used"] < REJ_BREAKDOWN * f["n_sel"]
        and f["R2"] > 0.999
        and f["max_res_over_h"] <= 1.0
        and 0.8 < f["DF"] <= 1.0
    )


def independent_pipeline(raw: pd.DataFrame) -> dict:
    clean, counts = my_clean(raw)
    snap = raw.snapshot_utc.iloc[0].to_pydatetime()
    rows, pairs = [], {}
    for (e, r), g in clean.groupby(["expiry", "root"]):
        pp = pairs_of(g)
        pairs[e] = pp
        row = {"expiry": e, "root": r, "T": my_T(e, r, snap), "n_pairs": len(pp)}
        if len(pp) >= N_MIN:
            row |= my_fit(pp)
            w, F0 = near_window(pp, N_NEAR)
            row["F_lad"], row["DF_lad"] = lad(w.K.to_numpy(), w.y.to_numpy(), F0)
        rows.append(row | {"kept": accept(row, len(pp))})
    return {"clean": clean, "counts": counts, "table": pd.DataFrame(rows), "pairs": pairs}


def arbitrage_reasons(clean: pd.DataFrame, expiry: str, K: float) -> list[str]:
    """Strict static-arbitrage violations of the call or put at K vs adjacent clean strikes.

    Using bid/ask (tradeable prices) and DF <= 1: calls non-increasing in K, puts non-decreasing,
    and a vertical spread worth at most (K2 - K1).
    """
    out = []
    for t in ("C", "P"):
        s = clean[(clean.expiry == expiry) & (clean.option_type == t)].sort_values("strike")
        s = s.reset_index(drop=True)
        idx = s.index[s.strike == K]
        if len(idx) == 0:
            continue
        i = int(idx[0])
        for j in (i - 1, i + 1):
            if not 0 <= j < len(s):
                continue
            lo, hi = (s.iloc[j], s.iloc[i]) if j < i else (s.iloc[i], s.iloc[j])
            dK = hi.strike - lo.strike
            if t == "C" and hi.bid > lo.ask:
                out.append(f"C({hi.strike:g}).bid > C({lo.strike:g}).ask")
            if t == "C" and lo.bid - hi.ask > dK:
                out.append(f"C({lo.strike:g}) - C({hi.strike:g}) > dK")
            if t == "P" and lo.bid > hi.ask:
                out.append(f"P({lo.strike:g}).bid > P({hi.strike:g}).ask")
            if t == "P" and hi.bid - lo.ask > dK:
                out.append(f"P({hi.strike:g}) - P({lo.strike:g}) > dK")
    return out


def otm_monotonicity_violations(otm: pd.DataFrame) -> pd.DataFrame:
    """Strict adjacent-strike monotonicity violations (bid/ask) inside an OTM surface."""
    v = []
    for (e, t), g in otm.groupby(["expiry", "option_type"]):
        g = g.sort_values("strike")
        K, b, a = g.strike.to_numpy(), g.bid.to_numpy(), g.ask.to_numpy()
        bad = np.where(b[1:] > a[:-1])[0] if t == "C" else np.where(b[:-1] > a[1:])[0]
        v += [(e, t, K[i], K[i + 1]) for i in bad]
    return pd.DataFrame(v, columns=["expiry", "type", "K1", "K2"])


def synthetic_chain(K, y, h_leg=0.5, DF=0.99, F=7400.0, T=0.25, expiry="2099-01-01"):
    """synthetic: legs with C - P = y exactly, P = DF*max(K-F,0) + 20, bid/ask = mid -/+ h_leg/2."""
    P = DF * np.maximum(K - F, 0.0) + 20.0
    C = P + y
    frames = [
        pd.DataFrame(
            {"option_type": t, "strike": K, "mid": m, "bid": m - h_leg / 2, "ask": m + h_leg / 2}
        )
        for t, m in (("C", C), ("P", P))
    ]
    return pd.concat(frames, ignore_index=True).assign(expiry=expiry, root="SPXW", T=T)


# ------------------------------------------------------------------------ 1. raw snapshot
def test_raw_snapshot_integrity(raw, meta):
    snap = datetime.fromisoformat(meta["snapshot_utc"])
    assert raw.snapshot_utc.nunique() == 1
    assert raw.snapshot_utc.iloc[0].to_pydatetime() == snap
    assert snap.strftime("%Y%m%d_%H%M") == STAMP  # file stamp = snapshot UTC
    assert raw.spot.nunique() == 1 and raw.spot.iloc[0] == meta["spot"]
    assert not raw.contractSymbol.duplicated().any()
    assert meta["failures"] == {}
    # 1w-2y window from the listed expiries, ET calendar date of the snapshot
    today = snap.astimezone(ET).date()
    want = [e for e in meta["expiries_listed"] if 7 <= (date.fromisoformat(e) - today).days <= 730]
    assert sorted(raw.expiry.unique()) == sorted(want) == meta["expiries_saved"]
    assert raw.lastTradeDate.max() < raw.snapshot_utc.iloc[0]  # no quote from the future
    assert set(raw.option_type) == {"C", "P"}


# --------------------------------------------------------------------------- 2. cleaning
def test_cleaning_counts_match_log(mine, cleaning_csv):
    c = mine["counts"]
    assert cleaning_csv.rows_in.tolist() == c[:-1]
    assert cleaning_csv.rows_out.tolist() == c[1:]
    names = " | ".join(cleaning_csv["filter"])
    for rule in ("bid > 0", "(ask-bid)/mid <= 0.5", "open interest > 0"):
        assert rule in names


def test_clean_set_and_mid_match_module(raw, mine):
    from qpricer.data import clean_chain

    clean, _ = clean_chain(raw)
    assert set(clean.contractSymbol) == set(mine["clean"].contractSymbol)
    assert (clean.bid > 0).all() and (clean.openInterest > 0).all()
    assert ((clean.ask - clean.bid) / ((clean.ask + clean.bid) / 2) <= 0.5).all()
    np.testing.assert_array_equal(clean.mid.to_numpy(), ((clean.bid + clean.ask) / 2).to_numpy())
    assert (clean.groupby("expiry").root.nunique() == 1).all()  # one root per date (D-002)


# -------------------------------------------------------------------------------- 3. T
def test_T_act365f_matches_independent_clock(mine, csv):
    t = mine["table"].merge(csv, on=["expiry", "root"], suffixes=("", "_csv"))
    assert len(t) == len(csv) == 40
    np.testing.assert_allclose(t["T_csv"], t["T"], rtol=0, atol=1e-12)  # all 40, no exclusion
    # zoneinfo ET offset equals a hand-coded US DST rule on every settlement date
    for e, r in zip(t.expiry, t.root, strict=True):
        off = settle_et(e, r).utcoffset()
        assert off == timedelta(hours=-4 if us_dst(date.fromisoformat(e)) else -5)


def test_T_early_close_settlement(csv):
    """SPXW expiring on an NYSE early-close day settles on the 13:00 ET close, not 16:00 (F1)."""
    from qpricer.data import settlement_utc

    snap = datetime(2026, 11, 6, 22, 13, 53, tzinfo=timezone.utc)
    hit = csv[csv.expiry.isin(EARLY_CLOSE) & (csv.root == "SPXW")]
    assert hit.expiry.tolist() == ["2026-11-27"]
    for r in hit.itertuples():
        want = my_T(r.expiry, r.root, snap)
        assert abs(r.T - want) < 1e-12, (
            f"{r.expiry} SPXW: T={r.T:.9f}, 13:00 ET early close gives {want:.9f}"
            f" (dT = {r.T - want:.3e} y = {(r.T - want) * 8760:.2f} h)"
        )
    # public settlement_utc on dates future snapshots will contain; AM settlement unaffected
    exp = sorted(EARLY_CLOSE) + ["2027-11-19", "2028-11-24"]
    root = ["SPXW"] * len(EARLY_CLOSE) + ["SPXW", "SPX"]
    got = settlement_utc(pd.Series(exp), pd.Series(root))
    want = [pd.Timestamp(settle_et(e, r)).tz_convert("UTC") for e, r in zip(exp, root, strict=True)]
    assert got.tolist() == want


# ---------------------------------------------------------------------- 4. parity fit
def test_parity_fit_reproduced_by_independent_code(mine, csv):
    t = mine["table"].merge(csv, on=["expiry", "root"], suffixes=("", "_csv"))
    assert t.kept.tolist() == t.kept_csv.tolist()
    assert t.n_pairs.tolist() == t.n_pairs_csv.tolist()
    k = t[t.kept]
    assert k.n_used.tolist() == k.n_used_csv.astype(int).tolist()
    assert k.rejected_K.tolist() == k.rejected_K_csv.fillna("").tolist()
    np.testing.assert_allclose(k.F, k.F_csv, rtol=1e-10, atol=0)
    np.testing.assert_allclose(k.DF, k.DF_csv, rtol=0, atol=1e-10)
    np.testing.assert_allclose(k.R2, k.R2_csv, rtol=0, atol=1e-10)
    np.testing.assert_allclose(k.F_se, k.F_se_csv, rtol=1e-8)  # QR covariance vs closed form
    np.testing.assert_allclose(k.DF_se, k.DF_se_csv, rtol=1e-8)
    np.testing.assert_allclose(k.r, -np.log(k.DF) / k["T"], rtol=1e-9)
    np.testing.assert_allclose(k.max_res_h, k.max_res_over_h, rtol=1e-9)  # D-005 in-band stat
    mine_br = (k.Kmin <= k.F) & (k.F <= k.Kmax)  # D-004 'bracketed' flag (F6)
    assert k.bracketed.astype(bool).tolist() == mine_br.tolist()
    assert k.loc[~mine_br, "expiry"].tolist() == ["2026-12-07"]


def test_gate_R2_each_kept_expiry(mine, csv):
    k = mine["table"][mine["table"].kept]
    assert (k.R2 > 0.999).all(), k.loc[k.R2 <= 0.999, ["expiry", "R2"]]
    kc = csv[csv.kept]
    assert (kc.R2 > 0.999).all() and (kc.n_used >= N_MIN).all()


def test_gate_DF_in_range(mine, csv):
    k = mine["table"][mine["table"].kept]
    assert ((k.DF > 0.8) & (k.DF <= 1.0)).all()
    kc = csv[csv.kept]
    assert ((kc.DF > 0.8) & (kc.DF <= 1.0)).all()


def test_dropped_expiries_have_reason_and_are_logged(raw, csv, mine, caplog):
    from qpricer.data import clean_chain
    from qpricer.market import implied_forwards

    dropped = csv[~csv.kept]
    assert dropped.reason.notna().all() and (dropped.reason.str.len() > 0).all()
    assert csv[csv.kept].reason.isna().all()
    with caplog.at_level(logging.WARNING, logger="qpricer.market"):
        implied_forwards(clean_chain(raw)[0])
    logged = {e for e in dropped.expiry if any(e in m.getMessage() for m in caplog.records)}
    assert logged == set(dropped.expiry)
    # independent count confirms the drop reason (2026-12-08: 7 clean pairs < N_MIN)
    mt = mine["table"].set_index("expiry")
    for e in dropped.expiry:
        assert mt.loc[e, "n_pairs"] < N_MIN or not mt.loc[e, "kept"]


def test_second_method_LAD_agrees_within_3se(mine, csv):
    """L1 regression on the 20 nearest pairs, no outlier rejection at all (independent method)."""
    t = mine["table"].merge(csv, on=["expiry", "root"], suffixes=("", "_csv"))
    k = t[t.kept]
    zF = (k.F_lad - k.F_csv) / k.F_se_csv
    zD = (k.DF_lad - k.DF_csv) / k.DF_se_csv
    assert zF.abs().max() < 3.0, k.loc[zF.abs().idxmax(), "expiry"]
    assert zD.abs().max() < 3.0, k.loc[zD.abs().idxmax(), "expiry"]


# ------------------------------------------------- 5. can the selection game the R^2 gate?
def test_fitted_parity_line_inside_every_used_quote_band(mine):
    """Economic non-vacuity: the kept line lies within each used pair's C-P bid-ask band."""
    k = mine["table"][mine["table"].kept]
    assert (k.max_res_over_h <= 1.0).all(), k.loc[k.max_res_over_h > 1, ["expiry"]]


def test_window_is_nearest_pairs_to_final_forward(mine):
    """The window placed around the first-order F0 equals the N_NEAR pairs nearest the fitted F."""
    for r in mine["table"][mine["table"].kept].itertuples():
        pp = mine["pairs"][r.expiry]
        K = pp.K.to_numpy()
        by_F = set(K[np.argsort(np.abs(K - r.F), kind="stable")[:N_NEAR]])
        assert set(near_window(pp, N_NEAR)[0].K) == by_F, r.expiry


def test_rejections_are_minority_and_static_arbitrage(mine, csv):
    """Every rejected strike has a leg in strict static arbitrage vs its clean neighbours, and the
    rejected fraction stays below the Theil-Sen breakdown point 1 - 1/sqrt(2) ~ 0.293."""
    kc = csv[csv.kept]
    assert (kc.n_rej / kc.n_sel < 1 - 1 / np.sqrt(2)).all()
    for r in kc[kc.rejected_K.notna()].itertuples():
        for K in map(float, r.rejected_K.split(";")):
            assert arbitrage_reasons(mine["clean"], r.expiry, K), f"{r.expiry} K={K:g}"


def test_rejection_cannot_rescue_noisy_parity():
    """synthetic: exact parity line + pure Gaussian noise (no outliers by construction).

    sigma = 2 pt on a 20-pair, 5-pt window gives population R^2 ~ 0.995 (5x the unexplained
    variance the gate allows) and quotes 4x wider than their half-spread from parity: the
    expiry must be dropped. Plain OLS on the same window drops all of them, so any pass can only
    come from the outlier-rejection step removing ordinary noise.
    """
    from qpricer.market import implied_forwards

    rng = np.random.default_rng(20261106)
    K = np.arange(7300.0, 7500.0, 5.0)  # 40 strikes, 5-pt grid as on SPX weeklies
    y0 = 0.99 * (7400.0 - K)
    bad = [
        synthetic_chain(K, y0 + rng.normal(0, 2.0, K.size), h_leg=0.5, expiry=f"b{i:03d}")
        for i in range(500)
    ]
    good = [
        synthetic_chain(K, y0 + rng.normal(0, 0.2, K.size), h_leg=1.0, expiry=f"g{i:03d}")
        for i in range(200)
    ]
    tb = implied_forwards(pd.concat(bad, ignore_index=True))
    tg = implied_forwards(pd.concat(good, ignore_index=True))
    ols_pass = (tb.R2_all > 0.999) & (tb.DF > 0.8) & (tb.DF <= 1.0)
    assert not ols_pass.any()
    rescued = tb[tb.kept]
    assert rescued.empty, (
        f"{len(rescued)}/{len(tb)} pure-noise chains kept after rejecting "
        f"{rescued.n_rej.tolist()} ordinary-noise pairs (R2_all {rescued.R2_all.round(5).tolist()})"
    )
    assert tg.kept.all()
    assert ((tg.F - 7400.0).abs() < 4 * tg.F_se).mean() > 0.99
    # no parity relation at all / smooth non-linear C - P: dropped
    smooth = synthetic_chain(K, y0 + 3.0 * np.sin(K / 30.0), expiry="s001")
    flat = synthetic_chain(K, rng.normal(0, 30.0, K.size), expiry="s002")
    assert not implied_forwards(pd.concat([smooth, flat], ignore_index=True)).kept.any()


def test_window_choice_not_cherry_picked(raw, mine):
    """Window sensitivity, independent code vs the module at n_near = 10 / 20 / 40.

    Independent count with all acceptance rules: 34 / 39 / 38 kept (at 40 pairs the in-band rule
    drops 2026-11-17, max |res|/h = 1.057; D-004's '40 keeps 39/40' predates that rule).
    F moves < 5e-4 relative (<= 0.1 vol pt at |dsigma/dk| <= 2).
    """
    from qpricer.data import clean_chain
    from qpricer.market import implied_forwards

    clean = clean_chain(raw)[0]
    k = mine["table"][mine["table"].kept].set_index("expiry")
    for n, n_kept in ((10, 34), (20, 39), (40, 38)):
        alt = {}
        for e, pp in mine["pairs"].items():
            if len(pp) < N_MIN:
                continue
            f = my_fit(pp, n)
            if accept(f, len(pp)):
                alt[e] = f["F"]
        assert len(alt) == n_kept
        mod = implied_forwards(clean, n_near=n)
        assert sorted(mod.expiry[mod.kept]) == sorted(alt)
        common = k.index.intersection(list(alt))
        rel = (pd.Series(alt)[common] - k.F[common]).abs() / k.F[common]
        assert rel.max() < 5e-4


def test_new_acceptance_rules_drop_nothing_on_snapshot(csv):
    """D-005 (i) in band and (ii) breakdown add no drop at the production setting."""
    reasons = set(csv.reason.dropna())
    assert reasons == {"n_pairs<10"}
    kc = csv[csv.kept]
    assert kc.max_res_h.max() <= 1.0 and (kc.n_rej / kc.n_sel).max() < REJ_BREAKDOWN


def test_inband_gate_size_on_valid_synthetic_chains():
    """synthetic: exact parity + Gaussian noise sigma = 0.2 h (real snapshot: rms/h median 0.11,
    max 0.37). A valid expiry must survive the in-band gate in >= 99% of 1000 chains."""
    from qpricer.market import implied_forwards

    rng = np.random.default_rng(11)
    K = np.arange(7300.0, 7500.0, 5.0)
    y0 = 0.99 * (7400.0 - K)
    ch = [
        synthetic_chain(K, y0 + rng.normal(0, 0.2, K.size), h_leg=1.0, expiry=f"v{i:04d}")
        for i in range(1000)
    ]
    tb = implied_forwards(pd.concat(ch, ignore_index=True))
    assert tb.kept.mean() >= 0.99


# ------------------------------------------- 5b. static-arbitrage filter (D-010, F4)
def strict_arb_flags(K: np.ndarray, bid: np.ndarray, ask: np.ndarray, put: bool) -> np.ndarray:
    """O(n) running-extremum test (a different method from the module's pairwise matrix).

    flag[i] = quote i is the lower-strike end of a strict arbitrage with some K_j > K_i:
    calls max_{j>i} bid_j > ask_i or bid_i + K_i > min_{j>i}(ask_j + K_j);
    puts  bid_i > min_{j>i} ask_j or max_{j>i}(bid_j - K_j) > ask_i - K_i.
    """
    o = np.argsort(K, kind="stable")
    K, b, a = K[o], bid[o], ask[o]
    if K.size < 2:
        return np.zeros(K.size, bool)

    def after(x: np.ndarray, op) -> np.ndarray:  # op over j > i, for i = 0 .. n-2
        return op.accumulate(x[::-1])[::-1][1:]

    if put:
        f = (b[:-1] > after(a, np.minimum)) | (after(b - K, np.maximum) > (a - K)[:-1])
    else:
        f = (after(b, np.maximum) > a[:-1]) | ((b + K)[:-1] > after(a + K, np.minimum))
    return np.r_[f, False]


def _n_arb(df: pd.DataFrame) -> int:
    return int(
        sum(
            strict_arb_flags(
                g.strike.to_numpy(float), g.bid.to_numpy(float), g.ask.to_numpy(float), t == "P"
            ).sum()
            for (_, t), g in df.groupby(["expiry", "option_type"])
        )
    )


def _n_butterfly(df: pd.DataFrame) -> int:
    """Strict adjacent-triple butterfly arbitrage: wings bought at ask cost less than body bid."""
    n = 0
    for _, g in df.groupby(["expiry", "option_type"]):
        g = g.sort_values("strike")
        K, b, a = g.strike.to_numpy(), g.bid.to_numpy(), g.ask.to_numpy()
        lam = (K[2:] - K[1:-1]) / (K[2:] - K[:-2])
        n += int((lam * a[:-2] + (1 - lam) * a[2:] < b[1:-1]).sum())
    return n


@pytest.fixture(scope="module")
def surface():
    from qpricer.data import build_surface

    return build_surface(_need(RAW))


def test_surface_free_of_static_arbitrage(surface):
    otm, sur, rem = surface["otm"], surface["surface"], surface["removed"]
    assert _n_arb(otm) > 0  # the plain ROADMAP cut does contain stale quotes (F4)
    assert _n_arb(sur) == 0
    assert _n_butterfly(sur) == 0
    assert set(sur.contractSymbol) <= set(otm.contractSymbol)
    assert set(otm.contractSymbol) - set(sur.contractSymbol) == set(rem.contractSymbol)
    st = pd.read_csv(_need(REPO / "data" / "processed" / f"surface_{STAMP}.csv"))
    assert st.rows_out.tolist() == [len(otm), len(sur)]
    ar = pd.read_csv(_need(REPO / "data" / "processed" / f"arb_removed_{STAMP}.csv"))
    assert set(ar.contractSymbol) == set(rem.contractSymbol)


def test_removed_quotes_are_necessary_and_stale(surface):
    """Locally minimal: restoring any single removed quote re-creates a strict arbitrage; and each
    removed mid is further than its own full spread from its surviving neighbours' interpolation."""
    otm, sur, rem = surface["otm"], surface["surface"], surface["removed"]
    for r in rem.itertuples():
        g = sur[(sur.expiry == r.expiry) & (sur.option_type == r.option_type)]
        g2 = pd.concat([g, otm[otm.contractSymbol == r.contractSymbol]])
        flags = strict_arb_flags(
            g2.strike.to_numpy(float),
            g2.bid.to_numpy(float),
            g2.ask.to_numpy(float),
            r.option_type == "P",
        )
        assert flags.any(), r.contractSymbol
        gs = g.sort_values("strike")
        interp = np.interp(r.strike, gs.strike, gs.mid)
        assert abs((r.bid + r.ask) / 2 - interp) > r.ask - r.bid, r.contractSymbol


# ------------------------------------------------------- 6. no hard-coded r / q / spot
def test_spot_and_yahoo_derived_columns_unused(raw):
    from qpricer.data import clean_chain
    from qpricer.market import implied_forwards

    base = implied_forwards(clean_chain(raw)[0])
    pert = raw.assign(
        spot=raw.spot * 1.1,
        impliedVolatility=raw.impliedVolatility * 2.0,
        lastPrice=raw.lastPrice + 100.0,
        inTheMoney=~raw.inTheMoney,
    )
    alt = implied_forwards(clean_chain(pert)[0])
    cols = ["T", "F", "DF", "R2", "F_se", "DF_se"]
    pd.testing.assert_frame_equal(base[cols], alt[cols])
    assert base.kept.tolist() == alt.kept.tolist()


SUSPECT_NAMES = {"dividend", "dividends", "div", "yield", "rate", "rates", "rf", "sofr", "ois"}


def test_no_rate_or_yield_literals_in_phase0_code():
    """No numeric literal that looks like a rate / dividend yield (0, 0.2] in Phase 0 code."""
    files = [
        REPO / "src/qpricer/data.py",
        REPO / "src/qpricer/market.py",
        REPO / "scripts/fetch_spx.py",
        REPO / "scripts/build_market.py",
    ]
    hits = []
    for f in files:
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, float):
                if 0.0 < node.value <= 0.2:
                    hits.append((f.name, node.lineno, node.value))
            if isinstance(node, ast.Name | ast.arg):
                name = (node.id if isinstance(node, ast.Name) else node.arg).lower()
                if set(name.split("_")) & SUSPECT_NAMES:
                    hits.append((f.name, node.lineno, name))
    assert hits == []


def test_extreme_carry_recovered_without_clipping():
    """synthetic: DF = 0.83 at T = 0.1 (r ~ 186%) is recovered exactly: nothing assumes a rate."""
    from qpricer.market import implied_forwards, market_data

    K = np.arange(5000.0, 5400.0, 10.0)
    tab = implied_forwards(synthetic_chain(K, 0.83 * (5123.4 - K), DF=0.83, F=5123.4, T=0.1))
    assert tab.kept.all()
    md = market_data(tab)["2099-01-01"]
    assert abs(md.F - 5123.4) < 1e-9 and abs(md.DF - 0.83) < 1e-12
    assert abs(md.r - (-np.log(0.83) / 0.1)) < 1e-10


# ----------------------------------------------------------------- 7. OTM selection
def test_otm_surface_matches_independent_cut(mine):
    from qpricer.data import build_surface

    s = build_surface(RAW)
    t = mine["table"][mine["table"].kept].set_index("expiry")
    c = mine["clean"][mine["clean"].expiry.isin(t.index)].copy()
    c["F"] = c.expiry.map(t.F)
    rej = {(e, float(k)) for e, ks in t.rejected_K.items() if ks for k in ks.split(";")}
    flagged = np.array([(e, k) in rej for e, k in zip(c.expiry, c.strike, strict=True)])
    otm = np.where(c.option_type == "P", c.strike < c.F, c.strike >= c.F)
    want = c[otm & ~flagged]
    got = s["otm"]
    assert set(got.contractSymbol) == set(want.contractSymbol)
    assert len(got) == len(want)
    g = got.set_index("contractSymbol").loc[want.contractSymbol]
    np.testing.assert_allclose(g.F.to_numpy(), want.F.to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(g.k.to_numpy(), np.log(want.strike / want.F).to_numpy(), atol=1e-14)
    assert ((g.mid - (g.bid + g.ask) / 2).abs() == 0).all()


def test_market_data_matches_csv(csv):
    from qpricer.data import build_surface
    from qpricer.market import market_data

    md = market_data(build_surface(RAW)["table"])
    kc = csv[csv.kept].set_index("expiry")
    assert sorted(md) == sorted(kc.index)
    for e, m in md.items():
        assert abs(m.F / kc.loc[e, "F"] - 1) < 1e-12 and abs(m.DF - kc.loc[e, "DF"]) < 1e-12
        assert abs(m.T - kc.loc[e, "T"]) < 1e-12


# ----------------------------------------------------- 8. AM/PM root choice (D-002)
def test_am_pm_forwards_agree_on_overlap_dates(raw):
    """D-002 claims SPX vs SPXW forwards agree within 0.4 pt; recomputed by LAD, both roots."""
    d = raw.assign(root=raw.contractSymbol.str[:-15])
    d = d[(d.bid > 0) & (d.ask >= d.bid) & (d.openInterest > 0)]
    d = d[(d.ask - d.bid) / ((d.ask + d.bid) / 2) <= 0.5].assign(mid=lambda x: (x.bid + x.ask) / 2)
    both = d.groupby("expiry").root.nunique()
    dates = both[both > 1].index
    assert len(dates) == 5
    for e in dates:
        F = {}
        for r in ("SPX", "SPXW"):
            w, F0 = near_window(pairs_of(d[(d.expiry == e) & (d.root == r)]), N_NEAR)
            F[r] = lad(w.K.to_numpy(), w.y.to_numpy(), F0)[0]
        assert abs(F["SPX"] - F["SPXW"]) < 0.4, (e, F)


# -------------------------------------------------- 9. long-end DF identified (D-007)
def test_long_end_DF_monotone_within_noise(csv):
    k = csv[csv.kept & (csv["T"] > 0.3)].sort_values("T")
    dDF = np.diff(k.DF.to_numpy())
    se = np.hypot(k.DF_se.to_numpy()[1:], k.DF_se.to_numpy()[:-1])
    assert (dDF / se < 3.0).all()


# ----------------------------------------------------------------- diagnostics (script)
def _diagnostics() -> None:  # pragma: no cover - prints the numbers quoted in the report
    from qpricer.data import build_surface

    logging.disable(logging.WARNING)
    raw = pd.read_parquet(RAW)
    m = independent_pipeline(raw)
    csv = pd.read_csv(MARKET_CSV)
    t = m["table"].merge(csv, on=["expiry", "root"], suffixes=("", "_csv"))
    k = t[t.kept]
    pd.set_option("display.width", 200)
    print("cleaning counts:", m["counts"])
    print(f"kept {int(k.shape[0])}/{len(t)}; dropped: {t.loc[~t.kept, 'expiry'].tolist()}")
    print(f"max |F_mine - F_csv| = {np.max(np.abs(k.F - k.F_csv)):.3e}")
    print(f"max |DF_mine - DF_csv| = {np.max(np.abs(k.DF - k.DF_csv)):.3e}")
    print(
        f"min R2 = {k.R2.min():.6f} ({k.expiry[k.R2.idxmin()]}); "
        f"DF range [{k.DF.min():.6f}, {k.DF.max():.6f}]"
    )
    zF = (k.F_lad - k.F_csv) / k.F_se_csv
    zD = (k.DF_lad - k.DF_csv) / k.DF_se_csv
    print(
        f"LAD: max |dF|/se = {zF.abs().max():.3f} ({k.expiry[zF.abs().idxmax()]}), max |dF| = "
        f"{np.max(np.abs(k.F_lad - k.F_csv)):.3f} pt; max |dDF|/se = {zD.abs().max():.3f}"
    )
    print(f"max |res|/h over used pairs = {k.max_res_over_h.max():.3f}")
    print(f"max |T_mine - T_csv| over 40 expiries = {np.max(np.abs(t['T'] - t['T_csv'])):.3e}")
    for e in sorted(EARLY_CLOSE & set(csv.expiry)):
        print(f"early close {e}: T_csv={csv.loc[csv.expiry == e, 'T'].iloc[0]:.7f}")
    kc = csv[csv.kept].sort_values("T")
    dDF = np.diff(kc.DF.to_numpy())
    z = dDF / np.hypot(kc.DF_se.to_numpy()[1:], kc.DF_se.to_numpy()[:-1])
    print(
        f"DF increases with T: {(dDF > 0).sum()}/{dDF.size}; max z = {z.max():.2f} "
        f"({kc.expiry.iloc[int(np.argmax(z))]} -> {kc.expiry.iloc[int(np.argmax(z)) + 1]})"
    )
    sl = kc[kc["T"] < 0.1]
    print(f"short-end r (T<0.1): [{sl.r.min():.4f}, {sl.r.max():.4f}]")
    bs = build_surface(RAW)
    v = otm_monotonicity_violations(bs["otm"])
    print(
        f"plain OTM cut: strict adjacent monotonicity violations {len(v)} on "
        f"{v.expiry.nunique()} expiries\n{v.groupby('expiry').size().to_string()}"
    )
    print(
        f"surface: {len(bs['otm'])} -> {len(bs['surface'])} quotes; "
        f"adjacent monotonicity {len(otm_monotonicity_violations(bs['surface']))}, "
        f"all-pairs strict arb flags {_n_arb(bs['surface'])} (OTM cut {_n_arb(bs['otm'])}), "
        f"butterfly {_n_butterfly(bs['surface'])}"
    )
    for r in csv[csv.rejected_K.notna()].itertuples():
        for K in map(float, r.rejected_K.split(";")):
            print(f"rejected {r.expiry} K={K:g}: {arbitrage_reasons(m['clean'], r.expiry, K)}")
    h = []
    for e, pp in m["pairs"].items():
        if len(pp) >= N_MIN:
            w, _ = near_window(pp, N_NEAR)
            yc = w.y - w.y.mean()
            h.append((e, np.sqrt(1e-3 * (yc @ yc) / len(w)), w.h.median()))
    h = pd.DataFrame(h, columns=["expiry", "rms_allowed_by_R2", "median_h"])
    print(
        "residual RMS the R2 gate tolerates (index pts): "
        f"[{h.rms_allowed_by_R2.min():.2f}, {h.rms_allowed_by_R2.max():.2f}]"
    )
    out = k[(k.Kmin > k.F) | (k.Kmax < k.F)]
    for r in out.itertuples():
        print(
            f"F outside used strike range: {r.expiry} F={r.F:.2f} K in [{r.Kmin:g}, {r.Kmax:g}]"
            f" (n_used={r.n_used}, F_se={r.F_se:.3f})"
        )
    from qpricer.market import implied_forwards

    K = np.arange(7300.0, 7500.0, 5.0)
    y0 = 0.99 * (7400.0 - K)
    for sig in (0.9, 1.0, 1.5, 2.0, 3.0):
        rng = np.random.default_rng(7)
        ch = [
            synthetic_chain(K, y0 + rng.normal(0, sig, K.size), expiry=f"e{i:04d}")
            for i in range(1000)
        ]
        tb = implied_forwards(pd.concat(ch, ignore_index=True))
        ols = ((tb.R2_all > 0.999) & (tb.DF > 0.8) & (tb.DF <= 1.0)).mean()
        print(
            f"synthetic gate power sigma={sig} (h=0.5): kept {tb.kept.mean():.3f}, "
            f"plain-OLS R2 pass {ols:.3f}, max n_rej {tb.n_rej.max()}"
        )
    ratios = []
    for r in k.itertuples():
        w, _ = near_window(m["pairs"][r.expiry], N_NEAR)
        w = w[~w.K.isin([float(x) for x in r.rejected_K.split(";") if x])]
        res = w.y - r.DF * (r.F - w.K)
        ratios.append(np.sqrt(np.mean(res**2)) / np.median(w.h))
    print(
        f"real rms(res)/median(h): min {min(ratios):.3f}, median {np.median(ratios):.3f}, "
        f"max {max(ratios):.3f}"
    )
    for ratio in (0.2, 0.3, 0.35, 0.5):
        rng = np.random.default_rng(11)
        ch = [
            synthetic_chain(K, y0 + rng.normal(0, ratio, K.size), h_leg=1.0, expiry=f"e{i:04d}")
            for i in range(1000)
        ]
        tb = implied_forwards(pd.concat(ch, ignore_index=True))
        print(
            f"in-band size, valid chains sigma/h={ratio}: kept {tb.kept.mean():.3f}, "
            f"reasons {tb.reason[~tb.kept].value_counts().to_dict()}"
        )
    clean = bs["clean"]
    for n in (10, 20, 40):
        mod = implied_forwards(clean, n_near=n)
        print(
            f"module n_near={n}: kept {int(mod.kept.sum())}, dropped "
            f"{mod.loc[~mod.kept, ['expiry', 'reason']].values.tolist()}"
        )


if __name__ == "__main__":
    _diagnostics()
