"""Phase 6 benchmark: vanilla MC vs closed form / COS, plain vs antithetic vs CV vs Sobol' RQMC.

Cases (D-039): an SPX-like expiry near 1 y from the Phase 0 parity table (F, DF, T), K/F = 1.0 call
and 0.9 put, under (i) BSM at the calibrated-Heston ATM implied vol (exact one-step lognormal, no
bias), (ii) the calibrated Heston set (QE-M, n_steps = ceil(32 T)) and (iii) the textbook FO set
(Fang-Oosterlee 2008, F = 100, DF = 1, T = 1, 32 steps). References: Black-76 (bsm.md eq. 1) and
forward-centred COS (heston.md eqs. 9-11), checked against Lewis (eqs. 14-15).
Pre-declared procedure (D-039): N = 2^10..2^20, R = 16 independent replications per (case,
method, N), seeds BASE_SEED + 10^6 set + 10^5 method + 10^3 iN + r. Error metric for the slope:
BSM RMSE vs Black; Heston the replication std (ddof = 1), i.e. noise only, bias excluded (D-040).
Heston bias: separate CV runs (8 x 2^20 paths) at n_steps = base x {1/4, 1/2, 1, 2} vs COS.
Timing: serial pass, median of 3 repeats of the r = 0 run (time.perf_counter); machine-dependent.

Writes report/figures/benchmark_*.{png,csv} and benchmark_manifest.csv (sha256 of every output).
Usage: .venv/bin/python scripts/run_benchmarks.py [--workers 8] [--quick --out DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from qpricer.engines import mc, vr  # noqa: E402
from qpricer.engines.exotic_mc import control_variate  # noqa: E402
from qpricer.engines.fourier import cos_price, lewis_price  # noqa: E402
from qpricer.iv import implied_vol  # noqa: E402
from qpricer.models import bsm  # noqa: E402
from qpricer.models.heston import HestonParams, total_variance  # noqa: E402
from qpricer.products import vanilla  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
FIT = ROOT / "data" / "processed" / "heston_fit_20261106_2213.json"
MARKET = ROOT / "data" / "processed" / "market_20261106_2213.csv"
EXPIRY = "2027-11-19"  # kept expiry closest to T = 1 y (T = 1.0347)
FO = HestonParams(kappa=1.5768, theta=0.0398, xi=0.5751, rho=-0.5711, v0=0.0175)
FO_PUBLISHED = 5.785155450  # Fang-Oosterlee (2008) Table 4, K = 100 call (heston-mc.md §5)
SETS = ("bsm", "heston_cal", "heston_fo")
METHODS = vr.METHODS
MONEYNESS, CPS = (1.0, 0.9), (1.0, -1.0)
STRIKE_LABELS = ("ATM call", "90% put")
BASE_SEED = 20261110
FULL = {"N": 2 ** np.arange(10, 21), "R": 16, "bias_chunks": 8, "bias_n": 2**20, "repeats": 3}
QUICK = {"N": 2 ** np.arange(10, 13), "R": 4, "bias_chunks": 2, "bias_n": 2**12, "repeats": 1}
N_SCRAMBLE = 16
BIAS_FACTORS = (0.25, 0.5, 1.0, 2.0)
FIGURES = (
    "benchmark_convergence_bsm.png",
    "benchmark_convergence_heston_cal.png",
    "benchmark_convergence_heston_fo.png",
    "benchmark_time_vs_error.png",
    "benchmark_heston_bias.png",
)
TABLES = (
    "benchmark_references.csv",
    "benchmark_runs.csv",
    "benchmark_convergence.csv",
    "benchmark_timing.csv",
    "benchmark_bias.csv",
    "benchmark_cv_choice.csv",
    "benchmark_summary.csv",
)
COLORS = {"plain": "C0", "antithetic": "C1", "cv": "C2", "sobol": "C3"}


# ---------------------------------------------------------------- cases and references
def calibrated() -> HestonParams:
    d = json.loads(FIT.read_text())
    return HestonParams(d["kappa"], d["theta"], d["xi"], d["rho"], d["v0"])


def cases() -> dict[str, dict]:
    """Case definitions: market inputs, model, grid, strikes, reference prices (D-039)."""
    row = pd.read_csv(MARKET).set_index("expiry").loc[EXPIRY]
    F, DF, T = float(row.F), float(row.DF), float(row["T"])
    p = calibrated()
    sig = float(implied_vol(cos_price(F, F, T, DF, p, 1.0), F, F, T, DF, 1))  # Heston ATM IV
    out = {}
    for name, model, f, df, t, m in [
        ("bsm", sig, F, DF, T, 1),
        ("heston_cal", p, F, DF, T, int(np.ceil(32 * T))),
        ("heston_fo", FO, 100.0, 1.0, 1.0, 32),
    ]:
        K, cp = f * np.array(MONEYNESS), np.array(CPS)
        if name == "bsm":
            ref, lew = bsm.price(f, K, t, df, model, cp), np.full(2, np.nan)
        else:
            ref, lew = cos_price(f, K, t, df, model, cp), lewis_price(f, K, t, df, model, cp)
        out[name] = {"F": f, "DF": df, "T": t, "model": model, "n_steps": m, "K": K, "cp": cp,
                     "ref": np.asarray(ref), "lewis": np.asarray(lew)}  # fmt: skip
    return out


def seed_of(set_i: int, method_i: int, n_i: int, r: int) -> int:
    return BASE_SEED + 1_000_000 * set_i + 100_000 * method_i + 1000 * n_i + r


# ---------------------------------------------------------------- tasks (top level: picklable)
def run_task(task: tuple) -> dict:
    """One vanilla_mc call; returns the 4-tuple fields per strike and the wall time."""
    name, method, N, r, seed, n_steps = task
    c = _cases()[name]
    t0 = time.perf_counter()
    price, se, n, s = vr.vanilla_mc(
        c["F"], c["K"], c["T"], c["DF"], c["model"], c["cp"], method=method, n_paths=N,
        seed=seed, n_steps=n_steps, n_scramble=N_SCRAMBLE,
    )  # fmt: skip
    dt = time.perf_counter() - t0
    return {"set": name, "method": method, "N": n, "rep": r, "seed": s, "n_steps": n_steps,
            "price": price.tolist(), "se": se.tolist(), "wall_s": dt}  # fmt: skip


_CASES: dict = {}


def _cases() -> dict:
    """Per-process cache of `cases()` (inputs are read once per worker)."""
    if not _CASES:
        _CASES.update(cases())
    return _CASES


def run_pool(tasks: list[tuple], workers: int) -> list[dict]:
    if workers <= 1:
        return [run_task(t) for t in tasks]
    order = sorted(range(len(tasks)), key=lambda i: -tasks[i][2] * tasks[i][5])  # big first
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(run_task, [tasks[i] for i in order], chunksize=1))
    back = [None] * len(tasks)
    for i, r in zip(order, res, strict=True):
        back[i] = r
    return back


def explode(rows: list[dict]) -> pd.DataFrame:
    """One row per (run, strike)."""
    out = []
    for r in rows:
        for j, lab in enumerate(STRIKE_LABELS):
            out.append({k: r[k] for k in ("set", "method", "N", "rep", "seed", "n_steps", "wall_s")}
                       | {"strike": lab, "price": r["price"][j], "se": r["se"][j]})  # fmt: skip
    return pd.DataFrame(out)


# ---------------------------------------------------------------- analysis (shared with gates)
def loglog_slope(N, y) -> tuple[float, float]:
    """OLS of ln y on ln N: (slope, its standard error) (D-040)."""
    N, y = np.asarray(N, float), np.asarray(y, float)
    if not (np.all(np.isfinite(y) & (y > 0)) and np.all(N > 0)):
        raise ValueError("loglog_slope needs finite y > 0 and N > 0")
    x, z = np.log(N), np.log(y)
    A = np.column_stack([np.ones_like(x), x])
    coef, res, *_ = np.linalg.lstsq(A, z, rcond=None)
    dof = x.size - 2
    s2 = float(res[0]) / dof if dof > 0 and res.size else 0.0
    return float(coef[1]), float(np.sqrt(s2 / np.sum((x - x.mean()) ** 2)))


def convergence(runs: pd.DataFrame, refs: pd.DataFrame) -> pd.DataFrame:
    """Per (set, strike, method, N): r = 0 price/SE/|error|, RMSE vs ref, replication std,
    rms SE, and the slope metric `err` (BSM: RMSE; Heston: replication std, D-040)."""
    m = runs.merge(refs[["set", "strike", "ref"]], on=["set", "strike"])
    m["e"] = m.price - m.ref
    rows = []
    for key, g in m.groupby(["set", "strike", "method", "N"], sort=False):
        g0 = g[g.rep == 0].iloc[0]
        rmse, noise = float(np.sqrt(np.mean(g.e**2))), float(g.price.std(ddof=1))
        rows.append(dict(zip(["set", "strike", "method", "N"], key, strict=True))
                    | {"price_r0": g0.price, "se_r0": g0.se, "abs_err_r0": abs(g0.e),
                       "z_r0": g0.e / g0.se, "mean": g.price.mean(), "rmse": rmse,
                       "noise_std": noise, "rms_se": float(np.sqrt(np.mean(g.se**2))),
                       "err": rmse if key[0] == "bsm" else noise, "R": len(g)})  # fmt: skip
    return pd.DataFrame(rows)


def pooled(g: pd.DataFrame) -> tuple[float, float]:
    """Work-weighted pool over runs: V = sum N V / sum N, SE = sqrt(sum N^2 SE^2) / sum N."""
    w = g.N.to_numpy(float)
    return float(w @ g.price / w.sum()), float(np.sqrt(np.sum((w * g.se) ** 2)) / w.sum())


def summary(runs, refs, conv, timing) -> pd.DataFrame:
    """Per (set, strike, method): slopes, VR at N_max, work-normalised efficiency, unbiasedness."""
    Nmax = int(conv.N.max())
    rows = []
    for (s, k), cg in conv.groupby(["set", "strike"], sort=False):
        ref = float(refs[(refs.set == s) & (refs.strike == k)].ref.iloc[0])
        rg = runs[(runs.set == s) & (runs.strike == k)]
        vp, sp = pooled(rg[rg.method == "plain"])
        top = cg[cg.N == Nmax].set_index("method")
        tm = timing[(timing.set == s) & (timing.N == Nmax)].set_index("method").median_s
        for meth in METHODS:
            g = cg[cg.method == meth].sort_values("N")
            se_sl, se_sl_se = loglog_slope(g.N, g.se_r0)
            er_sl, er_sl_se = loglog_slope(g.N, g.err)
            v, sv = pooled(rg[rg.method == meth])
            if s == "bsm" or meth == "plain":  # vs closed form / COS (Heston plain: incl. bias)
                target, z = ref, (v - ref) / sv
            else:  # vs plain MC at the same dt: isolates the estimator from the scheme bias
                target, z = vp, (v - vp) / np.hypot(sv, sp)
            vr_se = (top.loc["plain", "rms_se"] / top.loc[meth, "rms_se"]) ** 2
            vr_err = (top.loc["plain", "err"] / top.loc[meth, "err"]) ** 2
            rows.append({"set": s, "strike": k, "method": meth, "slope_se": se_sl,
                         "slope_se_sd": se_sl_se, "slope_err": er_sl, "slope_err_sd": er_sl_se,
                         f"VR_se_N{Nmax}": vr_se, f"VR_err_N{Nmax}": vr_err,
                         f"time_s_N{Nmax}": tm[meth],
                         f"eff_N{Nmax}": vr_se * tm["plain"] / tm[meth],
                         "pooled": v, "pooled_se": sv, "unbiased_target": target, "z_pooled": z,
                         "se_ratio": float(np.sqrt(np.mean(g.err**2 / g.rms_se**2)))})  # fmt: skip
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- side studies
def timing_pass(cs, grid, repeats) -> pd.DataFrame:
    """Serial: median wall time of `repeats` identical r = 0 calls (machine-dependent)."""
    rows = []
    for si, s in enumerate(SETS):
        for mi, meth in enumerate(METHODS):
            for ni, N in enumerate(grid):
                ts = [run_task((s, meth, int(N), 0, seed_of(si, mi, ni, 0), cs[s]["n_steps"]))
                      ["wall_s"] for _ in range(repeats)]  # fmt: skip
                rows.append({"set": s, "method": meth, "N": int(N), "repeats": repeats,
                             "median_s": float(np.median(ts))})  # fmt: skip
    return pd.DataFrame(rows)


def bias_tasks(cs, cfg) -> list[tuple]:
    tasks = []
    for si, s in enumerate(SETS[1:], start=1):
        for li, f in enumerate(BIAS_FACTORS):
            m = int(np.ceil(cs[s]["n_steps"] * f))
            for ch in range(cfg["bias_chunks"]):
                seed = BASE_SEED + 1_000_000 * si + 900_000 + 1000 * li + ch
                tasks.append((s, "cv", cfg["bias_n"], ch, seed, m))
    return tasks


def bias_table(rows, refs) -> pd.DataFrame:
    """Pool the CV chunks per (set, n_steps); bias = V_dt - COS (heston-mc.md §1 sign)."""
    d = explode(rows).merge(refs[["set", "strike", "ref"]], on=["set", "strike"])
    out = []
    for (s, k, m), g in d.groupby(["set", "strike", "n_steps"], sort=False):
        v, se = pooled(g)
        T = float(refs[refs.set == s]["T"].iloc[0])
        out.append({"set": s, "strike": k, "n_steps": m, "dt": T / m, "n_paths": int(g.N.sum()),
                    "seeds": f"{g.seed.min()}..{g.seed.max()}", "price": v, "se": se,
                    "cos": g.ref.iloc[0], "bias": v - g.ref.iloc[0],
                    "bias_over_se": (v - g.ref.iloc[0]) / se})  # fmt: skip
    b = pd.DataFrame(out).sort_values(["set", "strike", "n_steps"]).reset_index(drop=True)
    b["halving_diff"], b["halving_se"] = np.nan, np.nan  # V(dt) - V(dt/2), where m doubles
    for i in range(1, len(b)):
        prev, cur = b.iloc[i - 1], b.iloc[i]
        if (prev.set, prev.strike) == (cur.set, cur.strike) and cur.n_steps == 2 * prev.n_steps:
            b.loc[i - 1, "halving_diff"] = prev.price - cur.price
            b.loc[i - 1, "halving_se"] = np.hypot(prev.se, cur.se)
    return b


def cv_choice(cs, n: int) -> pd.DataFrame:
    """VR of the two candidate Heston controls on the same paths: X = F_T (chosen) vs the shadow
    BSM payoff of asian.md eq. (17) (D-041). BSM: F_T only."""
    rows = []
    for si, s in enumerate(SETS):
        c = cs[s]
        rng = np.random.default_rng(BASE_SEED + 1_000_000 * si + 950_000)
        F, K, cp, DF, T = c["F"], c["K"][:, None], c["cp"][:, None], c["DF"], c["T"]
        if s == "bsm":
            x, xb = vr._terminal(c["model"], T, 1, n, "plain", rng), None
        else:
            p, xb = c["model"], np.zeros(n)
            t = mc.uniform_grid(T, c["n_steps"])
            dvb = np.diff(total_variance(t, p), prepend=0.0)
            blocks = [rng.standard_normal((2, n)) for _ in t]
            x = vr.heston_terminal(p, t, blocks, "qe-m")
            for dv, z in zip(dvb, blocks, strict=True):
                xb = xb - 0.5 * dv + np.sqrt(dv) * (p.rho * z[0] + np.sqrt(1 - p.rho**2) * z[1])
        g = DF * vanilla(F * np.exp(x), K, cp)
        vr_f = control_variate(g, np.broadcast_to(F * np.exp(x), g.shape), np.full(2, F))[3]
        if xb is None:
            vr_b = np.full(2, np.nan)
        else:
            sb = np.sqrt(float(total_variance(T, p)) / T)
            X = DF * vanilla(F * np.exp(xb), K, cp)
            vr_b = control_variate(g, X, bsm.price(F, K[:, 0], T, DF, sb, cp[:, 0]))[3]
        for j, lab in enumerate(STRIKE_LABELS):
            rows.append({"set": s, "strike": lab, "n_paths": n, "seed": BASE_SEED + 1_000_000
                         * si + 950_000, "VR_F_T": vr_f[j], "VR_shadow_bsm": vr_b[j]})  # fmt: skip
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures
def fig_convergence(conv, bias, s, base_steps, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, k in zip(axes, STRIKE_LABELS, strict=True):
        g = conv[(conv.set == s) & (conv.strike == k)]
        for meth in METHODS:
            h = g[g.method == meth].sort_values("N")
            ax.loglog(h.N, h.err, "o-", color=COLORS[meth], label=f"{meth}: {err_name(s)}")
            ax.loglog(h.N, h.se_r0, "--", color=COLORS[meth], lw=1, alpha=0.8)
        h = g[g.method == "plain"].sort_values("N")
        ax.loglog(h.N, h.abs_err_r0, "kx", ms=6, label="plain: single-seed |error| (r = 0)")
        ax.loglog(h.N, h.err.iloc[0] * (h.N / h.N.iloc[0]) ** -0.5, "k:", lw=1, label="N^-1/2")
        if bias is not None:
            b = bias[(bias.set == s) & (bias.strike == k) & (bias.n_steps == base_steps)]
            if len(b):
                ax.axhline(abs(b.bias.iloc[0]), color="grey", lw=1,
                           label=f"|bias| at base dt = {abs(b.bias.iloc[0]):.2g}")  # fmt: skip
        ax.set(title=f"{s}: {k}", xlabel="N (total terminal samples)", ylabel="price error")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("solid: R-replication error metric, dashed: single-run SE (r = 0)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def err_name(s: str) -> str:
    return "RMSE vs Black" if s == "bsm" else "replication std (noise)"


def fig_time(conv, timing, path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for col, s in enumerate(SETS):
        for row, k in enumerate(STRIKE_LABELS):
            ax = axes[row, col]
            for meth in METHODS:
                g = conv[(conv.set == s) & (conv.strike == k) & (conv.method == meth)]
                g = g.merge(timing[timing.set == s], on=["method", "N"]).sort_values("N")
                ax.loglog(g.median_s, g.err, "o-", color=COLORS[meth], label=meth)
            ax.set(title=f"{s}: {k}", xlabel="median wall time [s] (this machine)",
                   ylabel=err_name(s))  # fmt: skip
            ax.grid(True, which="both", alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_bias(bias, path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for row, s in enumerate(SETS[1:]):
        for col, k in enumerate(STRIKE_LABELS):
            ax, b = axes[row, col], bias[(bias.set == s) & (bias.strike == k)]
            ax.errorbar(
                b.dt, b.bias, yerr=3 * b.se, fmt="o-", capsize=3, label="CV MC - COS +- 3SE"
            )
            ax.axhline(0.0, color="k", lw=0.8)
            ax.set_xscale("log", base=2)
            ax.set(title=f"{s}: {k} (QE-M)", xlabel="dt", ylabel="bias")
            ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- main
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--quick", action="store_true", help="tiny grid (tests only)")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    cfg, out = (QUICK if a.quick else FULL), a.out
    out.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t_start = time.perf_counter()
    cs = _cases()
    refs = pd.DataFrame([{"set": s, "strike": lab, "F": c["F"], "DF": c["DF"], "T": c["T"],
                          "K": c["K"][j], "cp": c["cp"][j], "n_steps": c["n_steps"],
                          "model": repr(c["model"]), "ref": c["ref"][j], "lewis": c["lewis"][j],
                          "cos_minus_lewis": c["ref"][j] - c["lewis"][j]}
                         for s, c in cs.items()
                         for j, lab in enumerate(STRIKE_LABELS)])  # fmt: skip
    if "heston_fo" in cs:
        print(f"FO COS {cs['heston_fo']['ref'][0]:.9f} vs published {FO_PUBLISHED:.9f}")
    tasks = [(s, meth, int(N), r, seed_of(si, mi, ni, r), cs[s]["n_steps"])
             for si, s in enumerate(SETS) for mi, meth in enumerate(METHODS)
             for ni, N in enumerate(cfg["N"]) for r in range(cfg["R"])]  # fmt: skip
    print(f"{len(tasks)} replication runs on {a.workers} workers ...", flush=True)
    runs = explode(run_pool(tasks, a.workers))
    print(f"  done in {time.perf_counter() - t_start:.0f} s; bias runs ...", flush=True)
    bias = bias_table(run_pool(bias_tasks(cs, cfg), a.workers), refs)
    print(f"  done in {time.perf_counter() - t_start:.0f} s; serial timing pass ...", flush=True)
    timing = timing_pass(cs, cfg["N"], cfg["repeats"])
    conv = convergence(runs, refs)
    summ = summary(runs, refs, conv, timing)
    cvc = cv_choice(cs, min(2**18, int(cfg["N"][-1])))
    for name, df in zip(TABLES, (refs, runs, conv, timing, bias, cvc, summ), strict=True):
        df.to_csv(out / name, index=False)
    for s in SETS:
        b = bias if s != "bsm" else None
        fig_convergence(conv, b, s, cs[s]["n_steps"], out / f"benchmark_convergence_{s}.png")
    fig_time(conv, timing, out / "benchmark_time_vs_error.png")
    fig_bias(bias, out / "benchmark_heston_bias.png")
    man = [{"file": f, "sha256": sha256(out / f), "bytes": (out / f).stat().st_size,
            "written_by": "scripts/run_benchmarks.py", "run_id": run_id,
            "config": "quick" if a.quick else "full"} for f in FIGURES + TABLES]  # fmt: skip
    pd.DataFrame(man).to_csv(out / "benchmark_manifest.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summ.round(4).to_string(index=False))
        print(bias.round(5).to_string(index=False))
        print(cvc.round(3).to_string(index=False))
    print(f"total {time.perf_counter() - t_start:.0f} s; outputs in {out}")
    return out


if __name__ == "__main__":
    main()
