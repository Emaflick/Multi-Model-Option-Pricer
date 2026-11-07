"""SPX option-chain loading and cleaning: raw Yahoo snapshot -> clean quotes -> OTM surface.

Pipeline (``build_surface``): ``load_raw`` -> ``clean_chain`` (one contract root per expiry,
quote filters, mid, T) -> ``market.implied_forwards`` (parity F, DF per expiry) -> ``select_otm``
(puts K < F, calls K >= F) -> ``remove_static_arbitrage`` (stale quotes that break monotonicity /
vertical-spread bounds in K, D-010). Calibration (Phase 5) uses ``build_surface(...)["surface"]``.

Time: T = (t_settle - t_snapshot) / 365 days (ACT/365F, in seconds), t_settle = expiry date at
16:00 ET for PM-settled SPXW (13:00 ET on NYSE early-close days), 09:30 ET for AM-settled SPX
(D-003, docs/DECISIONS.md).
Mid = (bid + ask) / 2. Filters (rows in/out reported, D-006): bid > 0; finite ask >= bid;
(ask - bid)/mid <= 0.5; open interest > 0 (NaN fails every filter and is counted, not masked).
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from qpricer.market import implied_forwards

REPO = Path(__file__).resolve().parents[2]
RAW_DIR = REPO / "data" / "raw"
PROCESSED_DIR = REPO / "data" / "processed"
ET = ZoneInfo("America/New_York")
SETTLE_TIME = {"SPXW": "16:00", "SPX": "09:30"}  # PM / AM settlement, US/Eastern (D-003)
# NYSE 13:00 ET early closes; PM-settled SPXW settle on the 13:00 close that day (D-003).
# Source: nyse.com/markets/hours-calendars (2026-2028 table, retrieved 2026-11-06). 2027-12-24 is
# a full holiday (Christmas observed), not an early close. No dates published after 2028.
EARLY_CLOSE = frozenset({"2026-11-27", "2026-12-24", "2027-11-26", "2028-07-03", "2028-11-24"})
EARLY_CLOSE_TIME = "13:00"
EARLY_CLOSE_KNOWN_UNTIL = "2028-12-31"
ROOT_PREFERENCE = ("SPX", "SPXW")  # on a date listing both roots keep the first (D-002)
MAX_REL_SPREAD = 0.5
YEAR_SECONDS = 365.0 * 86400.0
_SYMBOL = r"^(?P<root>[A-Z]+)(?P<yymmdd>\d{6})(?P<cp>[CP])(?P<k>\d{8})$"


def latest_raw(raw_dir: Path = RAW_DIR) -> Path:
    """Most recent ``spx_<YYYYMMDD_HHMM>.parquet`` (UTC stamp sorts lexicographically)."""
    files = sorted(Path(raw_dir).glob("spx_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no spx_*.parquet in {raw_dir}: run scripts/fetch_spx.py first")
    return files[-1]


def read_raw_metadata(path: Path) -> dict:
    """Snapshot metadata (spot, timestamps, expiries, failures) written by scripts/fetch_spx.py."""
    md = pq.read_schema(path).metadata or {}
    if b"qpricer" not in md:
        raise KeyError(f"{path} has no 'qpricer' metadata: not written by scripts/fetch_spx.py")
    return json.loads(md[b"qpricer"])


def load_raw(path: Path | str | None = None) -> tuple[pd.DataFrame, dict]:
    """Raw snapshot (latest if ``path`` is None) and its metadata."""
    path = Path(path) if path is not None else latest_raw()
    return pd.read_parquet(path), read_raw_metadata(path) | {"path": str(path)}


def parse_symbols(sym: pd.Series) -> pd.DataFrame:
    """OCC symbol ROOT|YYMMDD|C/P|strike*1000 (8 digits) -> root, expiry, option_type, strike."""
    p = sym.str.extract(_SYMBOL)
    if p.isna().any(axis=None):
        bad = sym[p.isna().any(axis=1)].head().tolist()
        raise ValueError(f"unparsable contract symbols, e.g. {bad}")
    expiry = pd.to_datetime(p.yymmdd, format="%y%m%d").dt.strftime("%Y-%m-%d")
    strike = p.k.astype(np.int64) / 1000.0
    return pd.DataFrame({"root": p.root, "expiry": expiry, "option_type": p.cp, "strike": strike})


def settlement_utc(expiry: pd.Series, root: pd.Series) -> pd.Series:
    """Settlement instant in UTC: expiry date at SETTLE_TIME[root] US/Eastern (DST-aware);
    PM-settled SPXW on an NYSE early-close day (EARLY_CLOSE) settle at 13:00 ET.

    Warns if a PM expiry lies beyond the published early-close calendar (16:00 ET is then assumed).
    """
    hhmm = root.map(SETTLE_TIME)
    if hhmm.isna().any():
        raise ValueError(f"no settlement time for roots {sorted(set(root[hhmm.isna()]))}")
    pm = (root == "SPXW").to_numpy()
    hhmm = hhmm.where(~(pm & expiry.isin(EARLY_CLOSE).to_numpy()), EARLY_CLOSE_TIME)
    beyond = sorted(set(expiry[pm & (expiry > EARLY_CLOSE_KNOWN_UNTIL).to_numpy()]))
    if beyond:
        warnings.warn(
            f"early-close calendar ends {EARLY_CLOSE_KNOWN_UNTIL}; 16:00 ET assumed for {beyond}",
            stacklevel=2,
        )
    return pd.to_datetime(expiry + " " + hhmm).dt.tz_localize(ET).dt.tz_convert("UTC")


def year_fraction(t0: pd.Series | pd.Timestamp, t1: pd.Series) -> np.ndarray:
    """ACT/365F year fraction (t1 - t0) / 365 days, computed in seconds."""
    return ((t1 - t0).dt.total_seconds() / YEAR_SECONDS).to_numpy()


def preferred_root_mask(df: pd.DataFrame, preference: tuple[str, ...] = ROOT_PREFERENCE):
    """True for rows of the highest-preference root present on their expiry date (D-002)."""
    rank = df.root.map({r: i for i, r in enumerate(preference)})
    if rank.isna().any():
        raise ValueError(f"roots not in preference list: {sorted(set(df.root[rank.isna()]))}")
    return (rank == rank.groupby(df.expiry).transform("min")).to_numpy()


def _filters(dedupe: bool) -> list[tuple[str, Callable[[pd.DataFrame], np.ndarray]]]:
    f = [
        ("bid > 0", lambda d: (d.bid > 0).to_numpy()),
        ("finite ask >= bid", lambda d: (np.isfinite(d.ask) & (d.ask >= d.bid)).to_numpy()),
        (
            f"(ask-bid)/mid <= {MAX_REL_SPREAD}",
            lambda d: ((d.ask - d.bid) / (0.5 * (d.ask + d.bid)) <= MAX_REL_SPREAD).to_numpy(),
        ),
        ("open interest > 0", lambda d: (d.openInterest > 0).to_numpy()),
    ]
    return ([("one root per expiry", preferred_root_mask)] if dedupe else []) + f


def clean_chain(raw: pd.DataFrame, dedupe: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the quote filters in order; add root, mid, settle_utc, T.

    Returns ``(clean, stats)``; ``stats`` has rows_in / rows_out / dropped per filter.
    Raises if the contract symbols disagree with the yfinance strike/type/expiry columns.
    """
    parsed = parse_symbols(raw.contractSymbol)
    for col in ("expiry", "option_type", "strike"):
        bad = parsed[col].to_numpy() != raw[col].to_numpy()
        if bad.any():
            raise ValueError(f"{bad.sum()} rows: contractSymbol disagrees with column '{col}'")
    df = raw.assign(root=parsed.root)
    stats = []
    for name, keep in _filters(dedupe):
        m = keep(df)
        stats.append({"filter": name, "rows_in": len(df), "rows_out": int(m.sum())})
        df = df[m]
    stats = pd.DataFrame(stats).assign(dropped=lambda s: s.rows_in - s.rows_out)
    df = df.assign(mid=0.5 * (df.bid + df.ask), settle_utc=settlement_utc(df.expiry, df.root))
    df["T"] = year_fraction(df.snapshot_utc, df.settle_utc)
    if not (df["T"] > 0).all():
        raise ValueError("non-positive T: snapshot after settlement for some contracts")
    return df.reset_index(drop=True), stats


def select_otm(clean: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """OTM quotes of the kept expiries: puts K < F, calls K >= F, F from the parity fit.

    Both legs of strikes rejected as parity outliers (``table.rejected_K``) are removed: parity
    cannot tell which leg is stale (D-005). Adds F, DF and log-moneyness k = ln(K/F).
    """
    kept = table.loc[table.kept.astype(bool), ["expiry", "root", "F", "DF", "rejected_K"]]
    df = clean.merge(kept.drop(columns="rejected_K"), on=["expiry", "root"], validate="m:1")
    lists = kept.rejected_K.fillna("").astype(str).str.split(";")
    rej = kept.assign(strike=lists).explode("strike")
    rej = rej[rej.strike != ""].astype({"strike": float})
    key = ["expiry", "root", "strike"]
    flagged = df.set_index(key).index.isin(rej.set_index(key).index)
    otm = np.where(df.option_type == "P", df.strike < df.F, df.strike >= df.F)
    out = df[otm & ~flagged].assign(k=lambda d: np.log(d.strike / d.F))
    return out.sort_values(["expiry", "strike"]).reset_index(drop=True)


def arbitrage_pairs(K: np.ndarray, bid: np.ndarray, ask: np.ndarray, put: bool) -> np.ndarray:
    """V[i, j] = True (i < j, K ascending) if the quotes at K_i < K_j are a strict static arbitrage.

    Tradeable prices, model-free, DF <= 1 (D-010):
      calls: bid_j > ask_i (C must be non-increasing in K) or bid_i - ask_j > K_j - K_i (spread);
      puts:  bid_i > ask_j (P must be non-decreasing in K) or bid_j - ask_i > K_j - K_i.
    """
    lo_b, hi_b = bid[:, None], bid[None, :]
    lo_a, hi_a = ask[:, None], ask[None, :]
    dK = K[None, :] - K[:, None]
    if put:
        v = (lo_b > hi_a) | (hi_b - lo_a > dK)
    else:
        v = (hi_b > lo_a) | (lo_b - hi_a > dK)
    return np.triu(v, k=1)


def arbitrage_free_mask(
    K: np.ndarray, bid: np.ndarray, ask: np.ndarray, last_trade: np.ndarray, put: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy removal until no pair violates ``arbitrage_pairs`` (one expiry, one option type).

    Each step removes the quote in the most violating pairs; ties -> oldest ``last_trade``
    (seconds, NaN = unknown = oldest), then lowest strike. Returns (keep, violation count of each
    quote at its removal; 0 for kept quotes).
    """
    order = np.argsort(K, kind="stable")
    K, bid, ask, lt = (
        K[order],
        bid[order],
        ask[order],
        np.nan_to_num(last_trade[order], nan=-np.inf),
    )
    keep, count = np.ones(K.size, bool), np.zeros(K.size, int)
    while True:
        idx = np.flatnonzero(keep)
        v = arbitrage_pairs(K[idx], bid[idx], ask[idx], put)
        n = v.sum(axis=0) + v.sum(axis=1)
        if n.max(initial=0) == 0:
            break
        worst = idx[np.lexsort((K[idx], lt[idx], -n))[0]]
        keep[worst], count[worst] = False, n[np.searchsorted(idx, worst)]
    out_keep, out_count = np.empty_like(keep), np.empty_like(count)
    out_keep[order], out_count[order] = keep, count
    return out_keep, out_count


def remove_static_arbitrage(otm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop stale quotes that create strict static arbitrage in K within each (expiry, type) slice.

    Returns ``(surface, removed)``; ``removed`` lists each dropped quote with its violation count.
    """
    t = otm.lastTradeDate
    secs = ((t - pd.Timestamp(0, tz="UTC")).dt.total_seconds()).to_numpy(float)
    keep = np.ones(len(otm), bool)
    nviol = np.zeros(len(otm), int)
    for (_, typ), g in otm.groupby(["expiry", "option_type"], sort=False):
        i = otm.index.get_indexer(g.index)
        k, c = arbitrage_free_mask(
            g.strike.to_numpy(float),
            g.bid.to_numpy(float),
            g.ask.to_numpy(float),
            secs[i],
            put=typ == "P",
        )
        keep[i], nviol[i] = k, c
    cols = ["contractSymbol", "expiry", "option_type", "strike", "bid", "ask", "lastTradeDate"]
    removed = otm.loc[~keep, cols].assign(n_violations=nviol[~keep])
    return otm[keep].reset_index(drop=True), removed.reset_index(drop=True)


def build_surface(path: Path | str | None = None) -> dict:
    """Full pipeline on a raw snapshot.

    Keys: 'meta', 'clean', 'stats' (chain filters), 'table' (parity, per expiry), 'otm' (OTM cut),
    'surface' (OTM minus static-arbitrage quotes: the calibration input), 'removed' (quotes dropped
    by ``remove_static_arbitrage``), 'surface_stats' (rows in/out of the two surface stages).
    """
    raw, meta = load_raw(path)
    clean, stats = clean_chain(raw)
    table = implied_forwards(clean)
    otm = select_otm(clean, table)
    surface, removed = remove_static_arbitrage(otm)
    surface_stats = pd.DataFrame(
        {
            "stage": [
                "OTM cut (kept expiries, minus parity-rejected strikes)",
                "static arbitrage in K",
            ],
            "rows_in": [len(clean), len(otm)],
            "rows_out": [len(otm), len(surface)],
        }
    ).assign(dropped=lambda s: s.rows_in - s.rows_out)
    return {
        "meta": meta,
        "clean": clean,
        "stats": stats,
        "table": table,
        "otm": otm,
        "surface": surface,
        "removed": removed,
        "surface_stats": surface_stats,
    }
