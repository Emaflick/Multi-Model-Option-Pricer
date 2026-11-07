"""Per-expiry forward F and discount factor DF implied from put-call parity (model-free).

For European calls and puts with the same expiry T and strike K (no model, no r or q needed):

    C(K) - P(K) = DF * (F - K) = DF*F - DF*K                                          (P1)

so the OLS line y = a + b*K fitted to y = C - P gives

    DF = -b,   F = a / DF = Kbar - ybar / b,   r_implied = -ln(DF) / T.                  (P2)

Standard errors (homoscedastic OLS, s^2 = SS_res / (n - 2), Sxx = sum (K - Kbar)^2):
    se(DF) = s / sqrt(Sxx);  se(F)^2 = s^2 / (n b^2) + ybar^2 s^2 / (Sxx b^4)  (delta method). (P3)

Only near-ATM strikes enter (the N_NEAR quoted pairs nearest the C-P sign change): deep ITM legs
carry wide absolute spreads and stale quotes. Gross outliers (stale Yahoo quotes, D-005) are
rejected against a Theil-Sen line before the OLS fit. Gates (docs/ROADMAP.md, Phase 0):
R^2 > 0.999 and DF in (0.8, 1]; plus (D-005) every used pair inside its own bid-ask band,
max_i |res_i| / h_i <= 1, and rejected fraction below the Theil-Sen breakdown point.
Failing expiries are dropped and logged.
Reference: Hull, *Options, Futures and Other Derivatives*, put-call parity; D-001, D-004, D-005,
D-007 in docs/DECISIONS.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

N_NEAR = 20  # pairs nearest the C-P sign change used in the regression (D-004)
N_MIN = 10  # minimum pairs left after outlier rejection; fewer -> expiry dropped (D-004)
REJECT_Z = 4.0  # outlier threshold in robust (MAD) sigmas, floored by the pair's half-spread
R2_MIN = 0.999
DF_BOUNDS = (0.8, 1.0)  # open below, closed above
INBAND_MAX = 1.0  # max |OLS residual| / half-width over used pairs (D-005)
REJ_FRAC_MAX = 1.0 - 1.0 / np.sqrt(2.0)  # Theil-Sen breakdown point ~0.293 (D-005)


@dataclass(frozen=True)
class MarketData:
    """Inputs of one expiry for the forward-based kernels: forward F, discount DF, T in years."""

    F: float
    DF: float
    T: float

    def __post_init__(self) -> None:
        vals = np.array([self.F, self.DF, self.T], dtype=float)
        if not np.all(np.isfinite(vals)) or np.any(vals <= 0.0):
            raise ValueError(f"MarketData needs finite F, DF, T > 0, got {self}")

    @property
    def r(self) -> float:
        """Continuously compounded zero rate r = -ln(DF)/T, eq. (P2)."""
        return float(-np.log(self.DF) / self.T)


def ols_parity(K: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """OLS of y = C - P on K, eqs. (P1)-(P3): returns F, DF, R2 and their standard errors.

    Strikes are centred before fitting (K ~ 7000 makes the raw normal equations ill-conditioned).
    """
    K, y = np.asarray(K, float), np.asarray(y, float)
    n = K.size
    if n < 3 or y.size != n:
        raise ValueError(f"need >= 3 matching (K, C-P) points, got {n} and {y.size}")
    Kbar, ybar = K.mean(), y.mean()
    Kc = K - Kbar
    Sxx = Kc @ Kc
    if Sxx <= 0.0:
        raise ValueError("all strikes identical: slope not identified")
    b = (Kc @ (y - ybar)) / Sxx
    res = y - ybar - b * Kc
    ss_res, ss_tot = res @ res, (y - ybar) @ (y - ybar)
    s2 = ss_res / (n - 2)
    DF = -b
    F = Kbar - ybar / b
    F_se = np.sqrt(s2 / (n * b**2) + ybar**2 * s2 / (Sxx * b**4))
    return {
        "F": float(F),
        "DF": float(DF),
        "R2": float(1.0 - ss_res / ss_tot),
        "F_se": float(F_se),
        "DF_se": float(np.sqrt(s2 / Sxx)),
        "n": n,
    }


def theil_sen(K: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Robust line y = a + b K: b = median pairwise slope, a = median(y - b K) (Sen 1968)."""
    i, j = np.triu_indices(K.size, k=1)
    dK = K[j] - K[i]
    b = np.median((y[j] - y[i])[dK != 0] / dK[dK != 0])
    return float(np.median(y - b * K)), float(b)


def fit_expiry(
    K: np.ndarray,
    C: np.ndarray,
    P: np.ndarray,
    half_width: np.ndarray,
    n_near: int = N_NEAR,
    z: float = REJECT_Z,
) -> dict:
    """Near-ATM selection + outlier rejection + OLS parity fit for one expiry.

    1. F0 = K* + (C - P)(K*) at K* = argmin |C - P| (first-order sign-change locator, DF ~ 1).
    2. Keep the ``n_near`` strikes closest to F0.
    3. Theil-Sen residuals e_i; reject |e_i| > max(z * 1.4826 * MAD(e), h_i), where h_i is the
       half-width of the synthetic C - P quote, ((C_ask - C_bid) + (P_ask - P_bid)) / 2: a pair is
       never rejected while the robust line lies inside its own quoted band.
    4. OLS (eqs. P1-P3) on the remaining pairs; ``max_res_h`` = max |res_i| / h_i over them
       (gated <= 1 in ``implied_forwards``); ``bracketed`` = F inside [min K, max K] used.
    ``R2_all`` is the OLS R^2 on the selection before rejection (reported, not gated).
    """
    K, C, P, h = (np.asarray(a, float) for a in (K, C, P, half_width))
    y = C - P
    i0 = np.argmin(np.abs(y))
    F0 = K[i0] + y[i0]
    sel = np.sort(np.argsort(np.abs(K - F0), kind="stable")[:n_near])
    Ks, ys, hs = K[sel], y[sel], h[sel]
    a, b = theil_sen(Ks, ys)
    e = ys - a - b * Ks
    mad = 1.4826 * np.median(np.abs(e - np.median(e)))
    keep = np.abs(e) <= np.maximum(z * mad, hs)
    out = {
        "n_sel": int(sel.size),
        "n_rej": int((~keep).sum()),
        "R2_all": ols_parity(Ks, ys)["R2"] if sel.size >= 3 else np.nan,
        "rejected_K": ";".join(f"{k:g}" for k in Ks[~keep]),
    }
    if keep.sum() < 3:
        return {**out, "n_used": int(keep.sum())}
    Ku, yu, hu = Ks[keep], ys[keep], hs[keep]
    fit = ols_parity(Ku, yu)
    out["n_used"] = fit.pop("n")
    out["logm_max"] = float(np.max(np.abs(np.log(Ku / fit["F"]))))
    res = np.abs(yu - fit["DF"] * (fit["F"] - Ku))
    with np.errstate(divide="ignore", invalid="ignore"):  # h = 0 -> inf (fails the gate)
        out["max_res_h"] = float(np.max(np.where(res == 0.0, 0.0, res / hu)))
    out["bracketed"] = bool(Ku.min() <= fit["F"] <= Ku.max())
    return {**out, **fit}


def implied_forwards(
    clean: pd.DataFrame,
    n_near: int = N_NEAR,
    n_min: int = N_MIN,
    r2_min: float = R2_MIN,
    df_bounds: tuple[float, float] = DF_BOUNDS,
) -> pd.DataFrame:
    """Parity fit per (expiry, root) on cleaned quotes; one table row per expiry with gate outcome.

    ``clean`` needs columns expiry, root, T, option_type ('C'/'P'), strike, bid, ask, mid.
    Pairs = strikes with both a clean call and a clean put. Dropped expiries (reason in the
    ``reason`` column) are also logged at WARNING level.
    """
    rows = []
    for (expiry, root), g in clean.groupby(["expiry", "root"], sort=True):
        c = g[g.option_type == "C"].set_index("strike")
        p = g[g.option_type == "P"].set_index("strike")
        pair = pd.merge(c, p, left_index=True, right_index=True, suffixes=("_c", "_p")).sort_index()
        if g["T"].nunique() != 1:
            raise ValueError(f"expiry {expiry} ({root}): several T values in one slice")
        row = {"expiry": expiry, "root": root, "T": float(g["T"].iloc[0]), "n_pairs": len(pair)}
        if len(pair) >= n_min:
            hw = 0.5 * ((pair.ask_c - pair.bid_c) + (pair.ask_p - pair.bid_p))
            K = pair.index.to_numpy(float)
            C, P = pair.mid_c.to_numpy(), pair.mid_p.to_numpy()
            row |= fit_expiry(K, C, P, hw.to_numpy(), n_near)
        reason = ""
        if len(pair) < n_min:
            reason = f"n_pairs<{n_min}"
        elif row["n_used"] < n_min:
            reason = f"n_used<{n_min}"
        elif not row["n_rej"] < REJ_FRAC_MAX * row["n_sel"]:
            reason = f"n_rej/n_sel>={REJ_FRAC_MAX:.3f}"
        elif not row["R2"] > r2_min:
            reason = f"R2<={r2_min}"
        elif not row["max_res_h"] <= INBAND_MAX:
            reason = f"max|res|/h>{INBAND_MAX:g}"
        elif not df_bounds[0] < row["DF"] <= df_bounds[1]:
            reason = f"DF not in ({df_bounds[0]}, {df_bounds[1]}]"
        row["kept"], row["reason"] = reason == "", reason
        if reason:
            log.warning("dropping expiry %s (%s): %s", expiry, root, reason)
        elif not row["bracketed"]:
            log.warning("expiry %s (%s): F=%.2f outside the strikes used", expiry, root, row["F"])
        rows.append(row)
    cols = ["expiry", "root", "T", "n_pairs", "n_sel", "n_rej", "n_used", "logm_max", "F", "F_se"]
    cols += ["DF", "DF_se", "R2", "R2_all", "max_res_h", "bracketed", "kept", "reason"]
    tab = pd.DataFrame(rows).reindex(columns=cols + ["rejected_K"])
    tab = tab.astype({c: "Int64" for c in ("n_sel", "n_rej", "n_used")})  # NA on n_pairs drops
    tab = tab.astype({"bracketed": "boolean"})
    # r undefined for DF <= 0 (only possible on dropped rows, whose reason column says why)
    tab.insert(tab.columns.get_loc("DF_se") + 1, "r", -np.log(tab.DF.where(tab.DF > 0)) / tab["T"])
    tab.insert(tab.columns.get_loc("r") + 1, "r_se", tab.DF_se / (tab.DF * tab["T"]))
    return tab


def market_data(table: pd.DataFrame) -> dict[str, MarketData]:
    """MarketData per kept expiry (key: expiry 'YYYY-MM-DD') from an ``implied_forwards`` table."""
    kept = table[table.kept.astype(bool)]
    if kept.expiry.duplicated().any():
        raise ValueError("more than one kept root per expiry: deduplicate roots first")
    cols = kept[["expiry", "F", "DF", "T"]].itertuples(index=False)
    return {e: MarketData(F, DF, T) for e, F, DF, T in cols}
