"""Phase 5 real-data calibration (calibration.md eq. 24): Heston T >= 0.1, SABR per expiry, checks.

Writes (tag = snapshot stamp, e.g. 20261106_2213):
  data/processed/heston_fit_<tag>.json      p^, SE, cost, provenance (input to autocall_compare.py)
  report/calibration_expiry_<tag>.csv       per expiry: T, n_band, n_used, Heston RMSE (in/out of
                                            sample), weighted RMS, % in bid-ask, SABR fit, flags
  report/calibration_heston_<tag>.csv       p^, SE, correlations, cond, weakest direction, Feller
  report/calibration_starts_<tag>.csv       16-start table (start, end, cost, status, nfev, seconds)
  report/calibration_sensitivity_<tag>.csv  n_t = 50, T >= 2 weeks, objective (10): dp / SE
  report/calibration_f5_<tag>.csv           DF, r, F before/after eqs. (2)-(3), put/call IV split
  report/calibration_arbitrage_<tag>.csv    calendar / butterfly summaries, market and SABR (eq. 23)
  report/figures/calibration_smiles_{long,short}_<tag>.png, calibration_df_r_<tag>.png
Every number printed or written comes from this run (D-032 to D-037). Usage:
  .venv/bin/python scripts/run_calibration.py [--workers 8] [--skip-sensitivity] [--box note]
--box note: audit only (D-034). Refit on the note's original bounds box (11) from its 16 Sobol
starts plus 4 adversarial corner starts; report the endpoints against the production p^ (in SE)
and p^'s distance to the narrowed bounds (in SE). Writes report/calibration_notebox_<tag>.csv.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from qpricer import calibration as cal
from qpricer.iv import implied_vol
from qpricer.models import bsm

ROOT = Path(__file__).resolve().parents[1]
REPORT, FIG, PROC = ROOT / "report", ROOT / "report" / "figures", ROOT / "data" / "processed"


def per_expiry_heston(q: pd.DataFrame, x) -> pd.DataFrame:
    """Heston IV on the thinned quotes of every expiry (COS pinned), RMSE in vol pts (unweighted),
    weighted RMS, % of quotes with |sigma^H - sigma^m| <= h (inside the IV bid-ask)."""
    g = cal.grid_from_quotes(q)
    s = cal.heston_iv(x, g)
    if not np.all(np.isfinite(s)):
        raise FloatingPointError("Heston IV not finite on the thinned grid")
    d = pd.DataFrame({"T": g["T"], "res": s - g["sigma"], "w": g["w"]})
    d["h"] = q.h.to_numpy()[g["order"]]
    d["expiry"] = q.expiry.to_numpy()[g["order"]]
    d["sigma_H"], d["k"], d["sigma_m"] = s, g["k"], g["sigma"]
    return d


def smile_figure(q, qt, hd, sabr_fits, x, expiries, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(expiries)
    nc = 6
    fig, axes = plt.subplots(int(np.ceil(n / nc)), nc, figsize=(3.2 * nc, 2.5 * np.ceil(n / nc)))
    for ax, e in zip(axes.ravel(), expiries, strict=False):
        b = q[q.expiry == e]
        ax.errorbar(
            b.k,
            100 * b.sigma_m,
            yerr=100 * b.h,
            fmt=".",
            ms=2,
            lw=0.5,
            color="0.6",
            label="mid ± h (in band)",
        )
        t = qt[qt.expiry == e]
        ax.plot(t.k, 100 * t.sigma_m, "k.", ms=3, label="fit targets")
        kk = np.linspace(b.k.min(), b.k.max(), 120)
        T = float(b["T"].iloc[0])
        one = np.ones(kk.size)
        g = cal.make_grid(T * one, kk, np.where(kk < 0, -1.0, 1.0), 0.2 * one, one)
        ax.plot(g["k"], 100 * cal.heston_iv(x, g), "C0-", lw=1, label="Heston")
        ax.plot(g["k"], 100 * cal.sabr_iv(sabr_fits[e]["res"].x, g), "C3--", lw=1, label="SABR")
        rm = 100 * np.sqrt(np.mean(hd[hd.expiry == e].res ** 2))
        ax.set_title(f"{e}  T={T:.3f}  H rmse {rm:.2f}vp", fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=5)
    fig.suptitle(title, fontsize=9)
    fig.supxlabel("k = ln(K/F)", fontsize=8)
    fig.supylabel("implied vol (%)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def df_figure(f5, curve, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(f5["T"], f5.DF, "o", ms=3, label="parity DF (Phase 0)")
    ax[0].plot(f5["T"], f5.DF_tilde, "x", ms=4, label="after eqs. (2)-(3)")
    ax[1].plot(f5["T"], 100 * f5.r, "o", ms=3, label="parity r")
    ax[1].plot(f5["T"], 100 * f5.r_tilde, "x", ms=4, label="after (2)")
    tt = np.linspace(0, 1.65, 100)
    ax[1].plot(tt, 100 * (curve[0] + curve[1] * tt), "k-", lw=0.7, label="WLS r(T), T >= 0.1")
    for a, yl in zip(ax, ("DF(T)", "r(T) (%)"), strict=True):
        a.axvline(cal.T_SPLIT, color="0.7", lw=0.7)
        a.set_xscale("log")
        a.set_xlabel("T (y)")
        a.set_ylabel(yl)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def put_call_split(clean, pairs, table) -> pd.DataFrame:
    """Per expiry: median |sigma_C - sigma_P| at the used parity strikes, and the count with
    |sigma_C - sigma_P| > h_C + h_P (IV half-widths), under the given (F, DF) table (pitfall 9)."""
    m = table[table.kept.astype(bool)].set_index("expiry")
    rows = []
    for e, pp in pairs.groupby("expiry"):
        g = clean[(clean.expiry == e) & (clean.root == m.loc[e, "root"]) & clean.strike.isin(pp.K)]
        F, DF, T = m.loc[e, "F"], m.loc[e, "DF"], float(g["T"].iloc[0])
        iv = {}
        for typ, cp in (("C", 1), ("P", -1)):
            x = g[g.option_type == typ].set_index("strike").loc[pp.K]
            iv[typ] = [
                implied_vol(x[c].to_numpy(), F, x.index.to_numpy(), T, DF, cp)
                for c in ("mid", "bid", "ask")
            ]
        d = np.abs(iv["C"][0] - iv["P"][0])
        h = 0.5 * (iv["C"][2] - iv["C"][1] + iv["P"][2] - iv["P"][1])
        rows.append(
            {"expiry": e, "pc_med_volpt": 100 * np.median(d), "pc_n_gt_h": int((d > h).sum())}
        )
    return pd.DataFrame(rows)


def bound_distance_se(x, se, bounds=cal.BOUNDS) -> np.ndarray:
    """Per parameter, min distance of x to the lower / upper bound in units of its SE."""
    return np.minimum(x - bounds[0], bounds[1] - x) / se


ADVERSARIAL = np.array(
    [  # note-box audit starts at extreme corners (D-034)
        [1e-3, 0.01, 1e-3, 4.5, -0.9],
        [0.5, 15.0, 0.5, 0.05, -0.1],
        [1e-3, 15.0, 0.5, 4.5, -0.98],
        [0.5, 0.01, 1e-3, 0.05, -0.5],
    ]
)


def note_box_audit(g, workers: int, tag: str) -> None:
    """D-034 audit: production fit (narrowed box) vs a fit on the note's box (11) from the note's
    16 Sobol starts + 4 adversarial starts, same pinned COS; endpoints checked against Lewis."""
    prod = cal.fit_heston(g, cal.sobol_starts(), workers=workers)
    x, d = prod["res"].x, cal.ls_diagnostics(prod["res"])
    se = np.array(list(d["se"].values()))
    starts = np.vstack([cal.sobol_starts(box=cal.NOTE_START_BOX), ADVERSARIAL])
    nb = cal.fit_heston(g, starts, workers=workers, bounds=cal.NOTE_BOUNDS)
    tab = nb["table"]
    conv = tab.status.isin(cal.CONVERGED).to_numpy()
    end = tab[list(cal.NAMES)].to_numpy(float)
    tab["max_abs_dp_over_SE_vs_production"] = np.abs((end - x) / se).max(axis=1)
    tab["cost_rel_vs_production"] = tab.cost / prod["res"].cost - 1
    tab["cos_lewis_e_iv"] = [cal.cos_lewis_gap(e, g)["e_iv"] if c else np.nan
                             for e, c in zip(end, conv, strict=True)]  # fmt: skip
    tab["start_set"] = ["note Sobol"] * 16 + ["adversarial"] * len(ADVERSARIAL)
    same = conv & (tab.max_abs_dp_over_SE_vs_production < 1e-2)
    dist = bound_distance_se(x, se)
    print(
        f"Note-box audit (D-034), COS {cal.COS_NL}, bounds {[b.tolist() for b in cal.NOTE_BOUNDS]}"
    )
    print(
        f"  production p^ (narrowed box) = {np.round(x, 6).tolist()}, cost {prod['res'].cost:.6f}"
    )
    print(tab[["start", "start_set", *cal.NAMES, "cost", "status", "nfev",
               "max_abs_dp_over_SE_vs_production", "cos_lewis_e_iv"]]
          .to_string(index=False))  # fmt: skip
    print(f"  starts reaching the production p^ (|dp| < 0.01 SE): {int(same.sum())}/{len(tab)}; "
          f"converged {int(conv.sum())}; best note-box cost rel. to production "
          f"{tab.cost_rel_vs_production[conv].min():+.1e}")  # fmt: skip
    print(f"  production p^ distance to the narrowed bounds, in SE: {np.round(dist, 1).tolist()} "
          f"(min {dist.min():.1f})")  # fmt: skip
    tab.to_csv(REPORT / f"calibration_notebox_{tag}.csv", index=False)
    print(f"wrote report/calibration_notebox_{tag}.csv")


def heston_block(fit, d, g) -> dict:
    x = fit["res"].x
    p = cal.params(x)
    se = np.array(list(d["se"].values()))
    out = {f"{n}": v for n, v in d["x"].items()} | {f"se_{n}": v for n, v in d["se"].items()}
    out |= {
        "cost": d["cost"],
        "n": d["n"],
        "s2": d["s2"],
        "cond_Jdiagp": d["cond"],
        "feller_nu_F": p.feller,
        "optimality": d["optimality"],
        "n_converged": int(fit["table"].status.isin(cal.CONVERGED).sum()),
        "n_basin": int(fit["basin"].sum()),
        "basin_spread_SE": cal.basin_spread(fit["table"], x, se),
        "rank_dropped": d["rank_dropped"],
    }
    out |= {f"active_{n}": int(v) for n, v in d["active"].items()}
    out |= {f"weakest_{n}": v for n, v in d["weakest"].items()}
    for i, a in enumerate(cal.NAMES):
        for b in cal.NAMES[i + 1 :]:
            out[f"corr_{a}_{b}"] = float(d["corr"].loc[a, b])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--skip-sensitivity", action="store_true")
    ap.add_argument("--box", choices=("narrowed", "note"), default="narrowed")
    args = ap.parse_args(argv)
    logging.getLogger("qpricer").setLevel(logging.ERROR)  # Phase 0 drop log, already in D-004
    t0 = time.time()
    tg = cal.build_targets()
    S, q, band, th = tg["S"], tg["q"], tg["band"], tg["thin"]
    tag = Path(S["meta"]["path"]).stem.removeprefix("spx_")
    qb, qt = q[band], q[th]
    long = qt["T"] >= cal.T_SPLIT
    print(
        f"snapshot {tag}: surface {len(q)} quotes, in band {band.sum()}, thinned {th.sum()}, "
        f"Heston n = {long.sum()} on {qt[long].expiry.nunique()} expiries; "
        f"h_min binds on {100 * (qt.h < cal.H_MIN).mean():.1f}% of thinned quotes"
    )

    if args.box == "note":
        note_box_audit(cal.grid_from_quotes(qt[long]), args.workers, tag)
        return 0

    # ---- F5 closure (eqs. 2-3)
    f5, curve = tg["f5"], tg["curve"]
    pairs = cal.parity_pairs(S["clean"], S["table"])
    before = put_call_split(S["clean"], pairs, S["table"]).rename(
        columns=lambda c: c if c == "expiry" else c + "_before"
    )
    after = put_call_split(S["clean"], pairs, tg["table"]).rename(
        columns=lambda c: c if c == "expiry" else c + "_after"
    )
    f5o = f5.merge(before, on="expiry").merge(after, on="expiry")
    sh = f5o[f5o.smoothed]
    wz = sh.loc[sh.z_F.abs().idxmax()]
    print(
        f"\nF5: r(T) = {100 * curve[0]:.3f}% + {100 * curve[1]:.3f}% T (WLS, T >= 0.1, s0 = 10 bp)"
    )
    print(
        f"  T < 0.1: parity r {100 * sh.r.min():.2f}%..{100 * sh.r.max():.2f}% -> "
        f"{100 * sh.r_tilde.min():.3f}%..{100 * sh.r_tilde.max():.3f}%; DF rises "
        f"{int((np.diff(f5.DF) > 0).sum())} -> {int((np.diff(f5.DF_tilde) > 0).sum())} times "
        f"between consecutive expiries"
    )
    print(
        f"  max |F~ - F|/F_se = {sh.z_F.abs().max():.2f} ({wz.expiry}, "
        f"{wz.F_tilde - wz.F:+.3f} pt); "
        f"max |res|/h under (F~, DF~) = {sh.max_res_h_tilde.max():.3f}; pairs |res|/h > 1: "
        f"{int(f5o.n_res_gt_h.sum())}"
    )
    print(
        f"  put/call IV split at the used strikes, T < 0.1: median |sC - sP| "
        f"{sh.pc_med_volpt_before.median():.4f} -> {sh.pc_med_volpt_after.median():.4f} vol pt; "
        f"pairs with |sC - sP| > hC + hP: {int(sh.pc_n_gt_h_before.sum())} -> "
        f"{int(sh.pc_n_gt_h_after.sum())} (of {int(sh.n_used.sum())})"
    )
    key = ["expiry", "strike", "option_type"]
    same = S["surface"].set_index(key).index.equals(q.set_index(key).index)
    print(f"  re-cut surface identical to build_surface()['surface']: {same}")

    # ---- Heston fit
    g = cal.grid_from_quotes(qt[long])
    starts = cal.sobol_starts()
    t1 = time.time()
    fit = cal.fit_heston(g, starts, workers=args.workers)
    d = cal.ls_diagnostics(fit["res"])
    hb = heston_block(fit, d, g) | {"fit_seconds_wall": time.time() - t1}
    x = fit["res"].x
    r, nfb = cal.heston_residuals(x, g)
    hb["fallbacks_at_optimum"] = nfb
    dist = bound_distance_se(x, np.array(list(d["se"].values())))
    hb |= {f"bound_dist_SE_{n}": v for n, v in zip(cal.NAMES, dist, strict=True)}
    gap = cal.cos_lewis_gap(x, g)
    hb |= {f"cos_lewis_{k}": v for k, v in gap.items()}
    ends = [
        cal.cos_lewis_gap(row[list(cal.NAMES)].to_numpy(float), g)["e_iv"]
        for _, row in fit["table"].iterrows()
        if row.status in cal.CONVERGED
    ]
    hb["cos_lewis_e_iv_all_endpoints_max"] = max(ends)
    tab = fit["table"]
    print(
        f"\nHeston (T >= 0.1, n = {d['n']}, COS {cal.COS_NL}, {len(starts)} Sobol starts, "
        f"{hb['fit_seconds_wall']:.0f} s wall on {args.workers} workers)"
    )
    for n in cal.NAMES:
        print(f"  {n:<6} {d['x'][n]: .6f}  SE {d['se'][n]:.6f}  active {int(d['active'][n])}")
    print(
        f"  cost {d['cost']:.4f}, s^2 {d['s2']:.3f}, Feller 2 kappa theta/xi^2 "
        f"= {hb['feller_nu_F']:.4f}, "
        f"cond(J diag|p|) {d['cond']:.2f}, optimality {d['optimality']:.2e}"
    )
    print(f"  distance to the narrowed bounds (D-034), in SE: "
          f"{np.round(dist, 1).tolist()} (min {dist.min():.1f})")  # fmt: skip
    print(f"  weakest direction {np.round(list(d['weakest'].values()), 3).tolist()}")
    print(f"  correlations:\n{d['corr'].round(3).to_string()}")
    print(
        f"  converged {hb['n_converged']}/16, basin {hb['n_basin']}/16, basin spread "
        f"{hb['basin_spread_SE']:.2e} SE, distinct minima "
        f"{tab[tab.status.isin(cal.CONVERGED)].cost.round(4).nunique()}; "
        f"nfev {tab.nfev.min()}-{tab.nfev.max()}, "
        f"{tab.seconds.min():.1f}-{tab.seconds.max():.1f} s "
        f"per start"
    )
    print(
        f"  fallbacks at optimum {nfb}; post-fit COS vs Lewis IV agreement at p^ on I_H: max "
        f"|dsigma| {gap['e_iv']:.2e} "
        f"(e_vega {gap['e_vega']:.2e}); all endpoints max "
        f"{hb['cos_lewis_e_iv_all_endpoints_max']:.2e}"
    )

    # ---- per-expiry RMSE table (in sample T >= 0.1, out of sample below), SABR per expiry
    hd = per_expiry_heston(qt, x)
    g_out = cal.grid_from_quotes(qt[~long])
    hb["cos_lewis_e_iv_out_of_sample"] = cal.cos_lewis_gap(x, g_out)["e_iv"]
    rows, sabr_fits = [], {}
    for e, ge in qt.groupby("expiry"):
        gr = cal.grid_from_quotes(ge)
        st, dropped = cal.sabr_starts(gr)
        sf = cal.fit_sabr(gr, st)
        sabr_fits[e] = sf
        sd = sf["diag"]
        h = hd[hd.expiry == e]
        T = float(ge["T"].iloc[0])
        rows.append(
            {
                "expiry": e,
                "T": T,
                "n_band": int((qb.expiry == e).sum()),
                "n_used": len(ge),
                "sample": "in" if T >= cal.T_SPLIT else "out",
                "heston_rmse_volpt": 100 * np.sqrt(np.mean(h.res**2)),
                "heston_max_abs_volpt": 100 * h.res.abs().max(),
                "heston_weighted_rms": np.sqrt(np.mean((h.w * h.res) ** 2)),
                "heston_pct_inside_bidask": 100 * np.mean(h.res.abs() <= h.h),
                "sabr_alpha": sd["x"]["alpha"],
                "sabr_rho": sd["x"]["rho"],
                "sabr_nu": sd["x"]["nu"],
                "sabr_rmse_volpt": 100 * sd["rmse"],
                "sabr_nu2T": sd["nu2T"],
                "sabr_nu2T_gt_0.16": sd["nu2T_flag"],
                "sabr_bracket_1_c1T": sd["bracket"],
                "sabr_cond": sd["cond"],
                "sabr_corr_rho_nu": sd["corr_rho_nu"],
                "sabr_active": "".join(str(int(a)) for a in sd["active"].values()),
                "sabr_starts_dropped_F6": dropped,
            }
        )
    et = pd.DataFrame(rows)
    num = et.select_dtypes("number")
    if not np.all(np.isfinite(num.to_numpy(float))):
        raise FloatingPointError("NaN in the per-expiry table")
    print("\nPer-expiry table (vol pts):")
    print(
        et[
            [
                "expiry",
                "T",
                "n_band",
                "n_used",
                "sample",
                "heston_rmse_volpt",
                "heston_weighted_rms",
                "heston_pct_inside_bidask",
                "sabr_alpha",
                "sabr_rho",
                "sabr_nu",
                "sabr_rmse_volpt",
                "sabr_nu2T",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    for smp in ("in", "out"):
        s = et[et["sample"] == smp]
        hs = hd[hd["T"] >= cal.T_SPLIT] if smp == "in" else hd[hd["T"] < cal.T_SPLIT]
        print(
            f"  Heston {'in-sample' if smp == 'in' else 'out-of-sample'} ({len(s)} expiries, "
            f"{len(hs)} quotes): pooled RMSE "
            f"{100 * np.sqrt(np.mean(hs.res**2)):.3f} vol pt; per-expiry RMSE "
            f"{s.heston_rmse_volpt.min():.3f}-{s.heston_rmse_volpt.max():.3f}; inside bid-ask "
            f"{100 * np.mean(hs.res.abs() <= hs.h):.1f}%"
        )
    print(f"  COS vs Lewis at p^ out of sample: {hb['cos_lewis_e_iv_out_of_sample']:.2e}")
    print(
        f"  SABR: {len(et)} expiries fitted, RMSE {et.sabr_rmse_volpt.min():.3f}-"
        f"{et.sabr_rmse_volpt.max():.3f} vol pt; nu^2 T "
        f"{et.sabr_nu2T.min():.3f}-{et.sabr_nu2T.max():.3f},"
        f" flagged > 0.16: {int(et['sabr_nu2T_gt_0.16'].sum())}/{len(et)}; active bounds "
        f"{int((et.sabr_active != '000').sum())}; F6 starts dropped "
        f"{int(et.sabr_starts_dropped_F6.sum())}"
    )

    # ---- arbitrage checks (eqs. 21-23): market and fitted SABR smiles
    cal_m, bf_m = cal.calendar_check(qb), cal.butterfly_check(q)
    sab = []
    for e, ge in q.groupby("expiry"):
        gr = cal.make_grid(ge["T"], ge.k, ge.cp, ge.sigma_m, ge.w)
        s = cal.sabr_iv(sabr_fits[e]["res"].x, gr)
        o = ge.iloc[gr["order"]]
        v = bsm.price(o.F, o.strike, o["T"], o.DF, s, o.cp)
        sab.append(o.assign(mid=v, bid=v, ask=v, sigma_m=s, sigma_b=s, sigma_a=s))
    sab = pd.concat(sab)
    cal_s, bf_s = cal.calendar_check(sab[cal.in_band(q.loc[sab.index])]), cal.butterfly_check(sab)
    arb = []
    for name, c, b in (("market", cal_m, bf_m), ("SABR", cal_s, bf_s)):
        cs, bs = cal.arbitrage_summary(c, b)
        print(
            f"\nArbitrage ({name}): calendar {len(cs)} pairs, {len(c)} points, "
            f"mid {int(c.mid.sum())}, strict {int(c.strict.sum())}, "
            f"max dsigma {100 * c.dsigma.max():+.3f} vol pt; butterfly "
            f"{len(b)} triples on {b.expiry.nunique()} expiries, mid {int(b.mid.sum())}, strict "
            f"{int(b.strict.sum())}, same-type strict {int((b.strict & b.same_type).sum())}, "
            f"max z {b.z.replace(np.inf, np.nan).max():.3f}"
        )
        if name == "market" and b.strict.any():
            print(b[b.strict].to_string(index=False))
        arb.append(cs.assign(check="calendar", surface=name))
        arb.append(bs.assign(check="butterfly", surface=name))

    # ---- sensitivity runs and objective (10)
    sens = []
    se = np.array(list(d["se"].values()))
    variants = [] if args.skip_sensitivity else ["n_t=50", "T>=2w", "objective (10)"]
    for v in variants:
        if v == "n_t=50":
            gv = cal.grid_from_quotes(q[cal.thin(q, 50) & (q["T"] >= cal.T_SPLIT).to_numpy()])
            fv = cal.fit_heston(gv, starts, workers=args.workers)
        elif v == "T>=2w":
            gv = cal.grid_from_quotes(qt[qt["T"] >= 14 / 365])
            fv = cal.fit_heston(gv, starts, workers=args.workers)
        else:
            gv = g
            fv = cal.fit_heston(g, starts, kind="price", workers=args.workers)
        dx = (fv["res"].x - x) / se
        sens.append(
            {
                "variant": v,
                "n": gv["T"].size,
                "expiries": len(gv["slices"]),
                **dict(zip(cal.NAMES, fv["res"].x, strict=True)),
                **{f"dp_over_SE_{n}": z for n, z in zip(cal.NAMES, dx, strict=True)},
                "max_abs_dp_over_SE": np.abs(dx).max(),
                "cost": fv["res"].cost,
                "active": "".join(str(int(a)) for a in fv["res"].active_mask),
            }
        )
        print(
            f"\nSensitivity {v}: n = {gv['T'].size}, p = {np.round(fv['res'].x, 5).tolist()}, "
            f"dp/SE = {np.round(dx, 2).tolist()}"
        )

    if sens:  # M6: curvature SE vs the sensitivity range (window / sample / objective)
        sx = np.array([[r[n] for n in cal.NAMES] for r in sens])
        print("\nParameter uncertainty: curvature SE vs sensitivity range (D-037)")
        for j, n in enumerate(cal.NAMES):
            lo, hi = min(sx[:, j].min(), x[j]), max(sx[:, j].max(), x[j])
            print(f"  {n:<6} {x[j]: .5f}  SE {se[j]:.5f}  range [{lo:.5f}, {hi:.5f}]  "
                  f"max |dp|/SE {np.abs((sx[:, j] - x[j]) / se[j]).max():.2f}")  # fmt: skip
        print("  (curvature SEs are conditional on the model and window; quote the range)")

    # ---- write outputs
    REPORT.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    et.to_csv(REPORT / f"calibration_expiry_{tag}.csv", index=False)
    pd.DataFrame([hb]).to_csv(REPORT / f"calibration_heston_{tag}.csv", index=False)
    tab.to_csv(REPORT / f"calibration_starts_{tag}.csv", index=False)
    f5o.to_csv(REPORT / f"calibration_f5_{tag}.csv", index=False)
    pd.concat(arb).to_csv(REPORT / f"calibration_arbitrage_{tag}.csv", index=False)
    if sens:
        pd.DataFrame(sens).to_csv(REPORT / f"calibration_sensitivity_{tag}.csv", index=False)
    js = {n: float(v) for n, v in d["x"].items()} | {
        "se": {n: float(v) for n, v in d["se"].items()}
    }
    js |= {
        "cost": float(d["cost"]),
        "n_quotes": int(d["n"]),
        "feller_nu_F": hb["feller_nu_F"],
        "label": f"calibrated Heston, SPX snapshot {tag} (T >= 0.1, {len(g['slices'])} expiries; "
        "scripts/run_calibration.py)",
        "provenance": {
            "snapshot": S["meta"]["path"].split("/")[-1],
            "script": "scripts/run_calibration.py",
            "cos_N_L": list(cal.COS_NL),
            "bounds": [b.tolist() for b in cal.BOUNDS],
            "objective": "IV residuals eq. (8), w = 1/max(h, 0.001)",
            "starts": "16 scrambled Sobol, seed 20261109",
            "decisions": "D-032..D-038",
        },
    }
    (PROC / f"heston_fit_{tag}.json").write_text(json.dumps(js, indent=2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        exl = sorted(et.expiry[et["T"] >= cal.T_SPLIT])
        exs = sorted(et.expiry[et["T"] < cal.T_SPLIT])
        smile_figure(
            qb,
            qt,
            hd,
            sabr_fits,
            x,
            exl,
            FIG / f"calibration_smiles_long_{tag}.png",
            f"SPX {tag}: Heston in sample (T >= 0.1) and SABR per expiry",
        )
        smile_figure(
            qb,
            qt,
            hd,
            sabr_fits,
            x,
            exs,
            FIG / f"calibration_smiles_short_{tag}.png",
            f"SPX {tag}: Heston OUT of sample (T < 0.1) and SABR per expiry",
        )
        df_figure(f5, curve, FIG / f"calibration_df_r_{tag}.png")
    print(
        f"\nwrote report/calibration_*_{tag}.csv, data/processed/heston_fit_{tag}.json, "
        f"report/figures/calibration_*_{tag}.png ({time.time() - t0:.0f} s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
