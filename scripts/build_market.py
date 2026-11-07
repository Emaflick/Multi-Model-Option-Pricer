"""Per-expiry put-call-parity table (F, DF, implied r, R^2, kept/dropped) for a raw SPX snapshot.

Prints the cleaning stats (rows in/out per filter) and the per-expiry table, and writes
    data/processed/market_<YYYYMMDD_HHMM>.csv        per-expiry parity table (committed)
    data/processed/cleaning_<YYYYMMDD_HHMM>.csv      rows in/out per chain filter (ROADMAP list)
    data/processed/surface_<YYYYMMDD_HHMM>.csv       rows in/out: OTM cut, static-arbitrage filter
    data/processed/arb_removed_<YYYYMMDD_HHMM>.csv   quotes dropped by the static-arbitrage filter
``--compare-roots`` also fits SPX (AM) and SPXW (PM) separately on dates listing both (D-002).
``--n-near N`` (sensitivity, D-004) refits with N pairs per expiry and prints only (no files).

Usage: .venv/bin/python scripts/build_market.py [--raw data/raw/spx_<stamp>.parquet]
                                                [--compare-roots] [--n-near N]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from qpricer.data import PROCESSED_DIR, build_surface, clean_chain, latest_raw, load_raw
from qpricer.market import N_NEAR, implied_forwards

SHOW = ["expiry", "root", "T", "n_used", "n_rej", "F", "F_se", "DF", "r", "r_se", "R2"]
SHOW += ["max_res_h", "bracketed", "kept"]


def compare_roots(path: Path) -> pd.DataFrame:
    """Parity fits of both roots on dates where SPX and SPXW are both listed."""
    raw, _ = load_raw(path)
    clean, _ = clean_chain(raw, dedupe=False)
    both = clean.groupby("expiry").root.nunique()
    tab = implied_forwards(clean[clean.expiry.isin(both[both > 1].index)])
    return tab[["expiry", "root", "T", "n_pairs", "n_used", "F", "F_se", "DF", "r", "R2", "reason"]]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, default=None, help="raw parquet (default: latest)")
    ap.add_argument("--compare-roots", action="store_true")
    ap.add_argument("--n-near", type=int, default=N_NEAR, help="sensitivity run, prints only")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    path = args.raw or latest_raw()
    if args.n_near != N_NEAR:
        clean, _ = clean_chain(load_raw(path)[0])
        tab = implied_forwards(clean, n_near=args.n_near)
        with pd.option_context("display.width", 200, "display.max_rows", 500):
            print(tab[SHOW + ["reason"]].to_string(index=False, float_format="{:.6g}".format))
        print(f"n_near={args.n_near}: kept {int(tab.kept.sum())} / {len(tab)} (nothing written)")
        return 0
    s = build_surface(path)
    stamp = path.stem.removeprefix("spx_")
    meta, stats, table = s["meta"], s["stats"], s["table"]
    print(f"snapshot {meta['snapshot_utc']}  spot {meta['spot']}  source {path.name}\n")
    print(stats.to_string(index=False), "\n")
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(table[SHOW + ["reason"]].to_string(index=False, float_format="{:.6g}".format))
    kept = int(table.kept.sum())
    print(f"\nkept {kept} / {len(table)} expiries")
    for r in table[~table.kept].itertuples():
        print(f"  dropped {r.expiry} ({r.root}): {r.reason}")
    for r in table[table.kept & ~table.bracketed.fillna(True)].itertuples():
        print(f"  flag {r.expiry} ({r.root}): F outside the strikes used (bracketed=False)")
    print("\n" + s["surface_stats"].to_string(index=False))
    rem = s["removed"]
    print(f"static-arbitrage quotes removed: {len(rem)} on {rem.expiry.nunique()} expiries")
    print(rem.groupby(["expiry", "option_type"]).size().to_string())

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = table.assign(snapshot_utc=meta["snapshot_utc"], source=path.name)
    out.to_csv(PROCESSED_DIR / f"market_{stamp}.csv", index=False)
    stats.to_csv(PROCESSED_DIR / f"cleaning_{stamp}.csv", index=False)
    s["surface_stats"].to_csv(PROCESSED_DIR / f"surface_{stamp}.csv", index=False)
    rem.to_csv(PROCESSED_DIR / f"arb_removed_{stamp}.csv", index=False)
    print(f"wrote market/cleaning/surface/arb_removed _{stamp}.csv in {PROCESSED_DIR}")

    if args.compare_roots:
        print("\nSPX (AM) vs SPXW (PM) on dates listing both:")
        print(compare_roots(path).to_string(index=False, float_format="{:.6g}".format))
    return 0


if __name__ == "__main__":
    sys.exit(main())
