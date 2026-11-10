"""Download the current ^SPX option chain from Yahoo Finance (yfinance) and save it raw.

Output: ``data/raw/spx_<YYYYMMDD_HHMM>.parquet`` (stamp = snapshot time in UTC) with every yfinance
column for calls and puts of all expiries between 7 days and 2 years (730 calendar days) from the
snapshot date (US/Eastern), plus ``expiry``, ``option_type`` ('C'/'P'), ``spot`` and
``snapshot_utc``. Spot, timestamps, the expiry list and any failures are also stored as JSON in the
parquet schema metadata under the key ``qpricer`` (read with ``qpricer.data.read_raw_metadata``).

No cleaning happens here: the file is the untouched market snapshot. If the download fails the
script exits non-zero and writes nothing (no synthetic fallback: project hard rule).

Usage: ``.venv/bin/python scripts/fetch_spx.py [--min-days 7] [--max-days 730]``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
ET = ZoneInfo("America/New_York")
TICKER = "^SPX"


def select_expiries(expiries: list[str], today: date, min_days: int, max_days: int) -> list[str]:
    """Keep expiry strings 'YYYY-MM-DD' with min_days <= (expiry - today).days <= max_days."""
    out = []
    for e in expiries:
        d = (date.fromisoformat(e) - today).days
        if min_days <= d <= max_days:
            out.append(e)
    return out


def fetch_one(tk: yf.Ticker, expiry: str, retries: int = 3, pause: float = 1.0):
    """Return (calls+puts DataFrame, underlying dict) for one expiry; retry with linear backoff."""
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            ch = tk.option_chain(expiry)
            calls, puts = ch.calls.assign(option_type="C"), ch.puts.assign(option_type="P")
            df = pd.concat([calls, puts], ignore_index=True)
            if df.empty:
                raise RuntimeError("empty chain")
            return df.assign(expiry=expiry), dict(ch.underlying or {})
        except Exception as err:  # network / parsing errors from yfinance
            last_err = err
            time.sleep(pause * attempt)
    raise RuntimeError(f"expiry {expiry}: {last_err!r}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--min-days", type=int, default=7)
    ap.add_argument("--max-days", type=int, default=730)
    ap.add_argument("--pause", type=float, default=0.3, help="seconds between requests")
    args = ap.parse_args(argv)

    snap = datetime.now(timezone.utc).replace(microsecond=0)
    tk = yf.Ticker(TICKER)
    try:
        all_exp = list(tk.options)
    except Exception as err:
        print(f"ERROR: could not list {TICKER} expiries: {err!r}", file=sys.stderr)
        return 1
    if not all_exp:
        print(f"ERROR: yfinance returned no expiries for {TICKER}", file=sys.stderr)
        return 1
    exps = select_expiries(all_exp, snap.astimezone(ET).date(), args.min_days, args.max_days)
    window = f"[{args.min_days}, {args.max_days}] d"
    print(f"{len(all_exp)} expiries listed, {len(exps)} within {window}")

    frames, failures, underlying = [], {}, {}
    for e in exps:
        try:
            df, und = fetch_one(tk, e)
            frames.append(df)
            underlying = underlying or und
        except RuntimeError as err:
            failures[e] = str(err)
            print(f"WARNING: {err}", file=sys.stderr)
        time.sleep(args.pause)
    if not frames:
        print("ERROR: no expiry could be downloaded; nothing written.", file=sys.stderr)
        return 1

    spot = underlying.get("regularMarketPrice")
    spot_src = "option_chain.underlying.regularMarketPrice"
    if spot is None:
        spot = float(tk.history(period="5d")["Close"].iloc[-1])
        spot_src = "history(period='5d').Close[-1]"
    spot_time = underlying.get("regularMarketTime")
    spot_time_utc = (
        datetime.fromtimestamp(int(spot_time), timezone.utc).isoformat() if spot_time else None
    )

    raw = pd.concat(frames, ignore_index=True)
    raw["spot"] = float(spot)
    raw["snapshot_utc"] = pd.Timestamp(snap)
    meta = {
        "ticker": TICKER,
        "snapshot_utc": snap.isoformat(),
        "fetch_end_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "spot": float(spot),
        "spot_source": spot_src,
        "spot_time_utc": spot_time_utc,
        "expiries_listed": all_exp,
        "expiries_saved": sorted(raw["expiry"].unique().tolist()),
        "failures": failures,
        "yfinance_version": yf.__version__,
    }
    table = pa.Table.from_pandas(raw, preserve_index=False)
    table = table.replace_schema_metadata(
        {**(table.schema.metadata or {}), b"qpricer": json.dumps(meta).encode()}
    )
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"spx_{snap:%Y%m%d_%H%M}.parquet"
    pq.write_table(table, path)
    print(
        f"saved {path.relative_to(ROOT)}: {len(raw)} rows, {raw['expiry'].nunique()} expiries, "
        f"spot={spot} ({spot_src}, {spot_time_utc}), snapshot={snap.isoformat()}, "
        f"failures={len(failures)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
