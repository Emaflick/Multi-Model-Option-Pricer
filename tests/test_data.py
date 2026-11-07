"""Tests for qpricer.data: symbol parsing, settlement times / ACT-365F, cleaning filters, OTM cut.

All frames here are constructed unit-test inputs, not market data.
"""

import numpy as np
import pandas as pd
import pytest

from qpricer.data import (
    arbitrage_free_mask,
    arbitrage_pairs,
    clean_chain,
    latest_raw,
    parse_symbols,
    remove_static_arbitrage,
    select_otm,
    settlement_utc,
    year_fraction,
)

SNAP = pd.Timestamp("2026-11-06 22:13:53", tz="UTC")


def symbol(root, expiry, cp, K):
    return f"{root}{pd.Timestamp(expiry):%y%m%d}{cp}{round(K * 1000):08d}"


def test_parse_symbols():
    p = parse_symbols(pd.Series(["SPXW261113C04000000", "SPX261120P07375500"]))
    assert p.root.tolist() == ["SPXW", "SPX"]
    assert p.expiry.tolist() == ["2026-11-13", "2026-11-20"]
    assert p.option_type.tolist() == ["C", "P"]
    assert p.strike.tolist() == [4000.0, 7375.5]
    with pytest.raises(ValueError):
        parse_symbols(pd.Series(["SPX26112P0737"]))


def test_settlement_times_dst_aware():
    exp = pd.Series(["2026-11-13", "2027-06-30", "2026-11-20", "2027-06-17"])
    root = pd.Series(["SPXW", "SPXW", "SPX", "SPX"])
    got = settlement_utc(exp, root)
    want = ["2026-11-13 21:00", "2027-06-30 20:00", "2026-11-20 14:30", "2027-06-17 13:30"]
    assert (got == pd.to_datetime(want).tz_localize("UTC")).all()  # EST = UTC-5, EDT = UTC-4
    with pytest.raises(ValueError):
        settlement_utc(pd.Series(["2026-11-13"]), pd.Series(["XSP"]))


def test_settlement_early_close_days():
    """D-003: SPXW on NYSE 13:00 early closes settle at 13:00 ET; SPX (AM) unaffected."""
    exp = pd.Series(["2026-11-27", "2026-12-24", "2027-11-26", "2028-07-03", "2026-11-27"])
    root = pd.Series(["SPXW", "SPXW", "SPXW", "SPXW", "SPX"])
    got = settlement_utc(exp, root)
    want = ["2026-11-27 18:00", "2026-12-24 18:00", "2027-11-26 18:00", "2028-07-03 17:00"]
    want = pd.to_datetime(want + ["2026-11-27 14:30"]).tz_localize("UTC")
    assert (got == want).all()
    # 2027-12-24 is a full NYSE holiday, not an early close: no special case
    assert settlement_utc(pd.Series(["2027-12-24"]), pd.Series(["SPXW"])).iloc[0] == pd.Timestamp(
        "2027-12-24 21:00", tz="UTC"
    )
    with pytest.warns(UserWarning, match="early-close calendar ends"):
        late = settlement_utc(pd.Series(["2029-01-19"]), pd.Series(["SPXW"]))
    assert late.iloc[0] == pd.Timestamp("2029-01-19 21:00", tz="UTC")


def test_arbitrage_pairs_monotonicity_and_vertical_spread():
    K = np.array([100.0, 105.0, 110.0])
    # calls: C(110).bid 6 > C(105).ask 5.5 (monotonicity); C(100).bid - C(105).ask = 12 - 5.5 > 5
    bid, ask = np.array([12.0, 5.0, 6.0]), np.array([12.5, 5.5, 6.5])
    v = arbitrage_pairs(K, bid, ask, put=False)
    assert v.tolist() == [[False, True, False], [False, False, True], [False, False, False]]
    # puts mirror: P(100).bid 6 > P(105).ask 5.5; P(110).bid - P(105).ask = 12 - 5.5 > 5
    v = arbitrage_pairs(K, bid[::-1], ask[::-1], put=True)
    assert v.tolist() == [[False, True, False], [False, False, True], [False, False, False]]
    # a clean, decreasing call strip with slope above -1 has no violation
    K = np.arange(100.0, 150.0, 5.0)
    mid = 8.0 * np.exp(-(K - 100.0) / 20.0)  # slope <= 0.4 < 1
    assert not arbitrage_pairs(K, mid - 0.1, mid + 0.1, put=False).any()


def test_arbitrage_free_mask_removes_stale_quote_and_ties_by_age():
    K = np.arange(100.0, 150.0, 5.0)
    mid = 8.0 * np.exp(-(K - 100.0) / 20.0)  # slope <= 0.4 < 1
    bid, ask, age = mid - 0.1, mid + 0.1, np.arange(K.size, dtype=float)
    bid[6], ask[6] = 6.0, 6.2  # stale call far above its neighbours (~1.8)
    keep, n = arbitrage_free_mask(K, bid, ask, age, put=False)
    assert keep.tolist() == [i != 6 for i in range(K.size)] and n[6] > 1 and n[keep].sum() == 0
    # two quotes, one violating pair: the one with the older last trade goes (input unsorted)
    K2, b2, a2 = np.array([105.0, 100.0]), np.array([9.0, 8.0]), np.array([9.5, 8.5])
    keep, _ = arbitrage_free_mask(K2, b2, a2, np.array([5.0, 1.0]), put=False)
    assert keep.tolist() == [True, False]
    keep, _ = arbitrage_free_mask(K2, b2, a2, np.array([1.0, 5.0]), put=False)
    assert keep.tolist() == [False, True]


def test_remove_static_arbitrage_on_frame():
    K = np.arange(7300.0, 7550.0, 50.0)
    put = 5.0 + (K - 7300.0) / 10.0  # increasing in K, slope 0.1 < 1
    call = put[::-1].copy()
    call[2] = 80.0  # stale call (neighbours 20 and 15)
    t0 = pd.Timestamp("2026-11-06", tz="UTC")
    rows = [
        pd.DataFrame({"option_type": t, "strike": K, "bid": m - 0.5, "ask": m + 0.5})
        for t, m in (("C", call), ("P", put))
    ]
    otm = pd.concat(rows, ignore_index=True).assign(
        expiry="2027-01-15",
        contractSymbol=lambda d: d.option_type + d.strike.astype(str),
        lastTradeDate=t0,
    )
    surf, removed = remove_static_arbitrage(otm)
    assert removed.contractSymbol.tolist() == ["C7400.0"] and len(surf) == len(otm) - 1


def test_year_fraction_act365f():
    t1 = pd.Series([pd.Timestamp("2026-11-13 21:00", tz="UTC")])
    seconds = 6 * 86400 + 22 * 3600 + 46 * 60 + 7
    assert year_fraction(SNAP, t1)[0] == seconds / (365 * 86400)


def raw_frame():
    """12 rows; comment = the filter expected to drop the row (None = survives)."""
    r = [  # root, expiry, type, K, bid, ask, OI
        ("SPXW", "2026-11-20", "C", 7400, 10.0, 10.5, 5),  # root dedup (SPX listed on date)
        ("SPX", "2026-11-20", "C", 7400, 10.0, 10.5, 5),  # None
        ("SPX", "2026-11-20", "P", 7400, 0.0, 0.5, 5),  # bid > 0
        ("SPX", "2026-11-20", "P", 7300, np.nan, 0.5, 5),  # bid > 0 (NaN)
        ("SPX", "2026-11-20", "C", 7500, 2.0, 1.5, 5),  # ask >= bid (crossed)
        ("SPX", "2026-11-20", "P", 7200, 2.0, np.nan, 5),  # ask >= bid (NaN)
        ("SPX", "2026-11-20", "C", 7600, 1.0, 2.0, 5),  # spread: 1/1.5 > 0.5
        ("SPX", "2026-11-20", "C", 7700, 1.0, 1.4, 5),  # None (spread 1/3)
        ("SPX", "2026-11-20", "P", 7100, 1.0, 1.2, 0),  # OI > 0
        ("SPX", "2026-11-20", "P", 7000, 1.0, 1.2, np.nan),  # OI > 0 (NaN)
        ("SPXW", "2026-11-13", "C", 7400, 20.0, 21.0, 5),  # None
        ("SPXW", "2026-11-13", "P", 7400, 19.0, 20.0, 5),  # None
    ]
    df = pd.DataFrame(r, columns=["root", "expiry", "option_type", "strike", "bid", "ask", "oi"])
    sym = [symbol(*x) for x in df[["root", "expiry", "option_type", "strike"]].itertuples(False)]
    return pd.DataFrame(
        {
            "contractSymbol": sym,
            "strike": df.strike.astype(float),
            "bid": df.bid,
            "ask": df.ask,
            "openInterest": df.oi.astype(float),
            "option_type": df.option_type,
            "expiry": df.expiry,
            "spot": 7376.86,
            "snapshot_utc": SNAP,
        }
    )


def test_clean_chain_filters_and_stats():
    clean, stats = clean_chain(raw_frame())
    assert stats.rows_in.tolist() == [12, 11, 9, 7, 6]
    assert stats.rows_out.tolist() == [11, 9, 7, 6, 4]
    assert (stats.dropped == stats.rows_in - stats.rows_out).all()
    got = list(zip(clean.root, clean.expiry, clean.option_type, clean.strike, strict=True))
    assert got == [
        ("SPX", "2026-11-20", "C", 7400.0),
        ("SPX", "2026-11-20", "C", 7700.0),
        ("SPXW", "2026-11-13", "C", 7400.0),
        ("SPXW", "2026-11-13", "P", 7400.0),
    ]
    np.testing.assert_allclose(clean.mid, [10.25, 1.2, 20.5, 19.5], rtol=0, atol=1e-15)
    t_am = (pd.Timestamp("2026-11-20 14:30", tz="UTC") - SNAP).total_seconds() / (365 * 86400)
    t_pm = (pd.Timestamp("2026-11-13 21:00", tz="UTC") - SNAP).total_seconds() / (365 * 86400)
    np.testing.assert_allclose(clean["T"], [t_am, t_am, t_pm, t_pm], rtol=1e-15)


def test_clean_chain_without_dedupe_keeps_both_roots():
    clean, stats = clean_chain(raw_frame(), dedupe=False)
    assert len(stats) == 4 and stats.rows_in.iloc[0] == 12
    assert set(clean.root[clean.expiry == "2026-11-20"]) == {"SPX", "SPXW"}


def test_clean_chain_raises_on_symbol_column_mismatch():
    raw = raw_frame()
    raw.loc[3, "strike"] = 7305.0
    with pytest.raises(ValueError, match="strike"):
        clean_chain(raw)


def test_select_otm_cut_and_parity_rejects():
    K = np.arange(7300.0, 7550.0, 50.0)
    clean = pd.concat(
        [
            pd.DataFrame({"expiry": e, "root": "SPX", "option_type": t, "strike": K, "mid": 1.0})
            for e in ("2027-01-15", "2027-02-19")
            for t in ("C", "P")
        ]
    )
    table = pd.DataFrame(
        {
            "expiry": ["2027-01-15", "2027-02-19"],
            "root": "SPX",
            "F": [7400.0, 7450.0],
            "DF": [0.99, 0.98],
            "kept": [True, False],
            "rejected_K": ["7350", np.nan],
        }
    )
    otm = select_otm(clean, table)
    assert set(otm.expiry) == {"2027-01-15"}
    assert otm[otm.option_type == "P"].strike.tolist() == [7300.0]  # 7350 rejected, 7400 = F -> C
    assert otm[otm.option_type == "C"].strike.tolist() == [7400.0, 7450.0, 7500.0]
    np.testing.assert_allclose(otm.k, np.log(otm.strike / 7400.0), rtol=1e-15)


def test_latest_raw_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        latest_raw(tmp_path)
