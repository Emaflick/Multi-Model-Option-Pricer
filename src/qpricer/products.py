"""Payoff definitions (Phase 4): vanilla payoff, autocallable term sheet and its path cash flows.

Barrier and Asian payoffs are the vanilla payoff of S_T or of the average, times a survival weight
or indicator built by the engine (derivations/barrier.md eq. 10, asian.md §1). Autocallable cash
flows follow eqs. (1)-(3) of derivations/autocallable.md (notional 1, levels as ratios to S_ref).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def vanilla(S: ArrayLike, K: ArrayLike, cp) -> NDArray:
    """(cp (S - K))^+, cp = +1 call, -1 put (barrier.md §1, asian.md §1)."""
    return np.maximum(np.asarray(cp) * (np.asarray(S) - np.asarray(K)), 0.0)


@dataclass(frozen=True)
class Autocall:
    """Mono-asset autocallable term sheet, autocallable.md §1 (D-026).

    t: dates 0 < t_1 < ... < t_n = T (ACT/365F); c: annual simple coupon, c_i = c tau_i, or a tuple
    of the n per-period coupons c_i (e.g. an accrual schedule, Deng et al. 2011); barriers as
    ratios X = S/S_ref with 0 <= B_c <= B_ac (a call pays its coupon; B_ac = inf: no autocall) and
    B_p >= 0 (sheets have B_p <= B_c; eq. 6 needs B_p <= 1). B_ac is tested at t_i < T, B_c at
    every date, B_p (European protection) at T only; every test is ">=".
    memory: an unpaid coupon is paid at the next date whose coupon test passes (eq. 2).
    """

    t: tuple[float, ...]
    c: float | tuple[float, ...]
    B_ac: float = 1.0
    B_c: float = 0.7
    B_p: float = 0.6
    memory: bool = False

    def __post_init__(self) -> None:
        t = np.asarray(self.t, dtype=np.float64)
        if t.ndim != 1 or t.size < 1 or not (t[0] > 0 and np.all(np.diff(t) > 0)):
            raise ValueError("t must be strictly increasing with t_1 > 0")
        c = np.asarray(self.c, dtype=np.float64)
        if c.shape not in ((), t.shape) or not np.all(np.isfinite(c) & (c >= 0)):
            raise ValueError("c must be a finite scalar >= 0 or one coupon >= 0 per date")
        if not (0 <= self.B_c <= self.B_ac and self.B_p >= 0):
            raise ValueError("need 0 <= B_c <= B_ac and B_p >= 0")

    @property
    def coupons(self) -> NDArray:
        """c_i = c tau_i, tau_i = t_i - t_{i-1}, t_0 = 0 (§1), or the given per-period coupons."""
        if np.ndim(self.c):
            return np.asarray(self.c, dtype=np.float64)
        return self.c * np.diff(np.asarray(self.t, dtype=np.float64), prepend=0.0)


def autocall_flows(X: ArrayLike, sheet: Autocall) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Path cash flows (C, R, chi, A_n) of fixings X (n, N), eqs. (1)-(3), autocallable.md.

    Per date, in this order (pitfall 3): G = c_i + m U; pi = A 1{X_i >= B_c}; C_i = pi G;
    U = (1 - pi) G; chi_i = A 1{X_i >= B_ac} (chi_n = 0); R_i = chi_i; A <- A - chi_i. At T:
    R_n = A_n [1 - (1 - X_n) 1{X_n < B_p}]. Returns coupons C (n, N), redemptions R (n, N),
    first-call indicators chi (n, N, bool) and A_n (N, bool), the not-called-before-T indicator.
    """
    X = np.asarray(X, dtype=np.float64)
    n = len(sheet.t)
    if X.ndim != 2 or X.shape[0] != n or not np.all(np.isfinite(X) & (X >= 0)):
        raise ValueError("X must be finite, >= 0 and of shape (n_dates, n_paths)")
    C, R, chi = np.zeros(X.shape), np.zeros(X.shape), np.zeros(X.shape, dtype=bool)
    alive, U = np.ones(X.shape[1], dtype=bool), np.zeros(X.shape[1])
    for i, ci in enumerate(sheet.coupons):  # loop over dates; vectorised over paths
        G = ci + sheet.memory * U
        pi = alive & (X[i] >= sheet.B_c)
        C[i], U = pi * G, np.where(pi, 0.0, G)
        if i < n - 1:
            chi[i] = alive & (X[i] >= sheet.B_ac)
            R[i], alive = chi[i], alive & ~chi[i]
    R[-1] = alive * (1.0 - (1.0 - X[-1]) * (X[-1] < sheet.B_p))
    return C, R, chi, alive
