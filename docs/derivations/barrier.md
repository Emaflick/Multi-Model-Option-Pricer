# Barrier options: Reiner–Rubinstein closed form, Brownian-bridge MC, BGK discrete-monitoring correction

Phase 4 · closed form `engines/analytic.py`, path weights `engines/mc.py`, payoffs `products.py`. Docstrings cite the eq. numbers
below. Black-76 notation follows `bsm.md`; MC estimator, SE, seeds and Heston schemes follow `heston-mc.md`. Asian and autocallable
products have their own notes. ⚑ = fewer than two external sources checked; (der.) = derived here; (est.) = hand estimate.

## 1. Setup

- **Measure.** $\mathbb Q^T$ with deterministic rates and carry (heston-mc.md §1). $F_t=F(t,T)$ is a $\mathbb Q^T$-martingale:
  BSM $dF_t=\sigma F_tdW_t$, or Heston as in heston-mc.md (4)–(12). State $x_t=\ln(F_t/F)$, $x_0=0$. Price $=DF\cdot\mathbb E^T[\cdot]$.
- **The barrier is on spot (convention, to be recorded as a D-entry).** Contracts, Haug and BGK all define $H$ on the index
  level $S_t$. Spot comes from the forward through a **flat carry** $b$ (Haug's cost of carry $b=r-q$), a kernel input with default 0:
  $\;S_t=F_te^{-b(T-t)}$, $S_0=Fe^{-bT}$, $S_T=F_T$. This is heston-mc.md eq. (17) with $F_0(t)=Fe^{-b(T-t)}$. With $b=0$ the barrier sits
  on the $T$-forward itself (driftless, pure Black-76); that is the gate configuration. $r:=-\ln(DF)/T$ is derived from $DF$ and
  enters only the rebate-at-hit term $\mathcal F$. Neither $r$ nor $q$ is a kernel input.
  *Rejected.* (i) A constant barrier on $F_t$ at any carry. It is forward-pure, but it equals the spot barrier $He^{-b(T-t)}$: for $b>0$
  the down barrier sits too low early on (relative error $\approx bT$ at $t=0$, est.), knock-outs are over-priced, and the Haug/BGK tables
  ($b=4\%$, $10\%$) cannot be reproduced. (ii) $S_0$ from the data feed, which contradicts D-008. Here $S_0$ is implied by $(F,b)$, and
  $b=\ln(F(T_2)/F(T_1))/(T_2-T_1)$ comes from two parity forwards, or $b=0$. *Consequence:* in $x$-coordinates the barrier
  $\ln(H/F)+b(T-t)$ is linear in $t$. Closed form, MC and bridge share the single log-distance (8), so they are consistent by construction.
- **Products.** Strike $K$; $\theta=+1$ call, $-1$ put (Haug's $\phi$); $\eta=+1$ down ($S_0>H$), $-1$ up ($S_0<H$).
  $\tau=\inf\{t\ge0:\eta(S_t-H)\le0\}$. A knock-out pays $(\theta(S_T-K))^+\mathbf 1_{\tau>T}$ at $T$; a knock-in pays the same on $\{\tau\le T\}$.
  Optional rebate $R\ge0$: out-options pay $R$ at $\tau$, in-options pay $R$ at $T$ if $\tau>T$ (Haug, QuantLib, MathWorks). Gates use $R=0$.
- **Monitoring.** Continuous, or discrete at $t_j=j\Delta t$, $j=1..m$, $\Delta t=T/m$ (BGK §1.1: $T$ is monitored, $t=0$ is not).
- **Notation.** $s=\sigma\sqrt T$; $d_{1,2}(X,L)=\ln(X/L)/s\pm s/2$; $\mathrm{Bl}_\theta(X,K)$ is bsm.md eq. (1) with forward $X$ and the same $DF,T,\sigma$.
  Haug's symbols map as $S\to S_0$, $X\to K$ (strike), $K\to R$ (rebate), $\phi\to\theta$.

## 2. Result

### 2.1 Closed form, continuous monitoring, BSM (Reiner–Rubinstein in forward form)
**(1)** $\;\rho=\dfrac H{S_0}=\dfrac HFe^{bT},\quad \mu=\dfrac b{\sigma^2}-\dfrac12,\quad \lambda=\sqrt{\mu^2+\dfrac{2r}{\sigma^2}},\quad F^*=\rho^2F\ \text{(image forward)},\quad z=\dfrac{\ln\rho}s+\lambda s.$

**(2)** With $G(X,L;\kappa):=\theta\,DF\,[X\,\Phi(\kappa d_1(X,L))-K\,\Phi(\kappa d_2(X,L))]$:
$A=G(F,K;\theta)=\mathrm{Bl}_\theta(F,K)$, $\;B=G(F,H;\theta)$, $\;C=\rho^{2\mu}G(F^*,K;\eta)$, $\;D=\rho^{2\mu}G(F^*,H;\eta)$,
$E=R\,DF\,[\Phi(\eta d_2(F,H))-\rho^{2\mu}\Phi(\eta d_2(F^*,H))]$, $\;\mathcal F=R\,[\rho^{\mu+\lambda}\Phi(\eta z)+\rho^{\mu-\lambda}\Phi(\eta z-2\eta\lambda s)]$.
Meaning: $B=DF\,\mathbb E[\theta(S_T-K)\mathbf 1\{\theta(S_T-H)>0\}]$; $C,D$ are the mirror images of $A,B$; $E=R\,DF\,\mathbb Q(\tau>T)$;
$\mathcal F=R\,\mathbb E[e^{-r\tau}\mathbf 1\{\tau\le T\}]$. $\mathcal F$ is Haug's F, not the forward.

**(3)** The eight cases (Haug; identical in NAG S30FAF and QuantLib):

| type | $\eta$ | $\theta$ | $K\ge H$ | $K<H$ | | type | $\eta$ | $\theta$ | $K\ge H$ | $K<H$ |
|---|---|---|---|---|---|---|---|---|---|---|
| cdi | +1 | +1 | $C+E$ | $A-B+D+E$ | | cdo | +1 | +1 | $A-C+\mathcal F$ | $B-D+\mathcal F$ |
| cui | −1 | +1 | $A+E$ | $B-C+D+E$ | | cuo | −1 | +1 | $\mathcal F$ | $A-B+C-D+\mathcal F$ |
| pdi | +1 | −1 | $B-C+D+E$ | $A+E$ | | pdo | +1 | −1 | $A-B+C-D+\mathcal F$ | $\mathcal F$ |
| pui | −1 | −1 | $A-B+D+E$ | $C+E$ | | puo | −1 | −1 | $B-D+\mathcal F$ | $A-C+\mathcal F$ |

**(4)** *Domain.* (3) requires $\eta\ln\rho<0$. If the barrier is already breached ($\eta(S_0-H)\le0$): out $=R$ (paid now), in $=A$.
Both $K$-branches agree at $K=H$ (since $A=B$, $C=D$), and (3) tends continuously to (4) as $\rho\to1$.

**(5)** *In–out parity:* in + out $=A+E+\mathcal F$ for each $(\eta,\theta,K)$. For $R=0$: **in + out = vanilla**. With $DF=1$ ($r=0$): $E+\mathcal F=R$.

**(6)** *Image form* ($K\ge H$): $\mathrm{cdi}=\rho^{2\mu}\mathrm{Bl}_+(F^*,K)$; likewise $\mathrm{pui}=\rho^{2\mu}\mathrm{Bl}_-(F^*,K)$ for $K\le H$. *Zero carry* ($b=0$:
$\rho^{2\mu}=F/H$, $F^*=H^2/F$) gives the static hedge $\mathrm{cdi}=\tfrac FH\mathrm{Bl}_+(H^2/F,K)=\tfrac KH\,\mathrm{Bl}_-(F,H^2/K)$ and $\mathrm{pui}=\tfrac KH\,\mathrm{Bl}_+(F,H^2/K)$.

**(7)** *Zero carry, $K\le H$* (der.): $\mathrm{cdo}=DF\,[(F-H)+(H-K)\,q]$ with $q=\mathbb Q(\tau>T)=\Phi(d_2(F,H))-\tfrac FH\Phi(-d_1(F,H))$.
*Model-free, $K=H$:* $\mathrm{cdo}=DF\,(F-H)$ ($F>H$) and $\mathrm{puo}=DF\,(H-F)$ ($F<H$). This holds for **any continuous martingale**
$F$ (Heston included) and does not depend on $\sigma$. *Greeks* are not gated: if needed, use central FD on (3) with the bsm.md (17) steps (pitfall 7).

### 2.2 Monte Carlo, continuous monitoring with Brownian bridge (BB)
**(8)** Grid $t_i=i\Delta$, $\Delta=T/M$, containing every monitoring date. Signed log-distance
$y_i=\ln(S_{t_i}/H)=\ln(F/H)-b(T-t_i)+x_i$; alive at $t_i$ iff $\eta y_i>0$. BSM exact step: $x_{i+1}=x_i-\tfrac12\sigma^2\Delta+\sigma\sqrt\Delta Z_i$.
Heston: $x$ from heston-mc.md (5)–(6) (Euler-FT) or (8)–(12) (QE, QE-M).

**(9)** Step crossing probability: $\;p_i=\exp\!\Big(-\dfrac{2\,(\eta y_i)^+(\eta y_{i+1})^+}{\sigma_i^2\,\Delta}\Big)$. Under BSM $\sigma_i^2=\sigma^2$, which is **exact** for any $b$
and any $\Delta$. Under Heston $\sigma_i^2=v_i^+$ (left-point variance, as in Euler-FT), which is an **approximation**. The positive parts make
$p_i=1$ automatically when an endpoint is at or beyond $H$.

**(10)** Survival weight $w=\prod_{i=0}^{M-1}(1-p_i)$ (conditional expectation). With $g=(\theta(Fe^{x_M}-K))^+$: out sample $g\,w$,
in sample $g\,(1-w)$, vanilla sample $g$. Price $=DF\times$mean and SE per heston-mc.md (1), **all from the same paths and the same $w$**.
The in-rebate sample is $R\,w$. The out-rebate needs $\tau$ and is outside the gates; pay it at the end of the step ($O(\Delta)$ bias, est.).

**(11)** *Uniform variant:* alive iff $U_i>p_i$ for all $i$ (QuantLib samples the bridge extreme instead, which is equivalent). It has the
same mean, but its variance is larger by $\mathbb E[g^2w(1-w)]$ per sample (Rao–Blackwell). Use (10): lower variance and smooth in the
parameters (CRN Greeks). Under BSM, (9)–(10) are unbiased for **every** $M$, $M=1$ included.

### 2.3 Discrete monitoring
**(12)** *BGK correction:* $\;V_m(H)=V\big(H\,e^{-\eta\beta\sigma\sqrt{\Delta t}}\big)+o(m^{-1/2})$, $\;\beta=-\zeta(\tfrac12)/\sqrt{2\pi}=0.5825971579\ldots$, $\Delta t=T/m$.
The barrier is shifted **away from $S_0$**: a down barrier moves down, an up barrier moves up. $V$ is (3) at the shifted barrier.
Proved for all 8 types under BSM (Kou 2003); there is no justification under Heston (pitfall 6).

**(13)** *Monitoring bias* (der., Taylor expansion of (12)): $V_m(H)-V(H)\approx-\eta\beta\sigma\sqrt{\Delta t}\;H\,\partial_HV(H)=O(m^{-1/2})$.
Without the correction the error is $O(m^{-1/2})$ (BGK §1.1), and discrete knock-outs are worth more than continuous ones. MC for a
discrete contract: weight $w^{(m)}=\prod_j\mathbf 1\{\eta y(t_j)>0\}$ with **no** bridge, and exact steps between dates under BSM.

**(14)** *Pathwise ordering* when the dates $\{t_j\}$ lie on the grid: $w\le w^{(m)}$, hence on the same paths out$_{\rm cont}\le$ out$_{\rm disc}$, in$_{\rm cont}\ge$ in$_{\rm disc}$.

### 2.4 Cross-check status (sources actually consulted)
| eq. | external sources | status |
|---|---|---|
| (1)–(3) | NAG S30FAF docs (A–F, 16-case table); QuantLib `analyticbarrierengine.cpp` (same table). Forward form: der. A hand evaluation reproduces Haug cdi/cdo($K$=100, $H$=95, σ=.25) = 4.0111/6.7925 and BGK Tab. 2.6 continuous ($H$=155) 12.776 (est.) | 2 |
| (5) | follows from (3); Kou (2003) §3; Howison–Steinberg §1 | 2 |
| (6) | image: BGK Prop. 3.1 ($b=r$), Howison–Steinberg eq. (7) ($\alpha_1=(r-q)/\sigma^2$). Static-hedge form: der. + bsm.md put-call symmetry | 2 · hedge ⚑ der. |
| (7) | optional stopping; matches $B-D$ by algebra | ⚑ der. |
| (9) | Shevchenko–Del Moral eq. (6) (per-step σ, step-wise barrier); QuantLib `mcbarrierengine.cpp` (inverse cdf of the bridge minimum, vol at step start) | 2 · Heston ⚑ approx. |
| (10)–(11) | Shevchenko–Del Moral eq. (27) (weight form); Rao–Blackwell (der.) | ⚑ 1 + der. |
| (12) | BGK Thm 1.1 (4 cases); Kou (2003) Thm 2.1 (all 8); Howison–Steinberg (re-derivation, $\beta$ from the Spitzer function) | 2 |
| (13) | BGK abstract and §1.1 (order); Taylor expansion: der. | ⚑ 1 + der. |

## 3. Derivation

**3.1 Spot from forward.** Deterministic carry gives $F(t,T)=S_te^{b(T-t)}$. With $\ln F_t=\ln F-\tfrac12\sigma^2t+\sigma W_t$ this gives
$Y_t:=\ln(S_t/S_0)=\nu t+\sigma W_t$ with $\nu=b-\tfrac12\sigma^2=\mu\sigma^2$, so the barrier is a constant level $\ell=\ln\rho$ for a drifted BM.

**3.2 Image identity** (down barrier, $\ell<0$; the up barrier is the same with the inequalities reversed). For any $f$:
$$\mathbb E\big[f(Y_T);\ \min_{[0,T]}Y\le\ell,\ Y_T>\ell\big]=e^{2\nu\ell/\sigma^2}\,\mathbb E\big[f(2\ell+Y_T);\ 2\ell+Y_T>\ell\big].\tag{†}$$
*Proof.* Change to the driftless measure: $d\mathbb Q/d\tilde{\mathbb Q}=e^{(\nu/\sigma^2)Y_T-\nu^2T/2\sigma^2}$. Reflect the path after $\tau$; this maps {hit, $Y_T>\ell$} onto
$\{Y_T<\ell\}$ with $Y_T\to2\ell-Y_T$, and the density at the reflected endpoint carries the factor $e^{2\nu\ell/\sigma^2}$. Then use $Y\to-Y$
symmetry under $\tilde{\mathbb Q}$ and change back. In spot terms $e^{2\nu\ell/\sigma^2}=\rho^{2\mu}$, $S_0e^{2\ell+Y_T}=\rho^2S_T$ and $\mathbb E^T[\rho^2S_T]=F^*$, so the
mirror world is Black-76 with forward $F^*$.

**3.3 Assembling (3).** Split each payoff on {$S_T$ beyond $H$} (which forces a hit) and its complement. *cdi, $K\ge H$:* the payoff
vanishes on $S_T\le H$, and (†) gives $\rho^{2\mu}DF\,\mathbb E(\rho^2S_T-K)^+=C$. *cdi, $K<H$:* $DF\,\mathbb E[(S_T-K)^+;S_T\le H]=A-B$, plus
$\rho^{2\mu}DF\,\mathbb E[(\rho^2S_T-K);\rho^2S_T>H]=D$. Puts follow by $\theta\to-1$ and up barriers by $\eta\to-1$. Each out = vanilla − in + rebate legs,
which gives the out column and (5). *E:* (†) with $f=1$ gives $\mathbb Q(\tau>T)=\Phi(\eta d_2(F,H))-\rho^{2\mu}\Phi(\eta d_2(F^*,H))$. *$\mathcal F$:* $\int_0^Te^{-rt}f_\tau(t)dt$
with the first-passage density $f_\tau(t)=\lvert\ell\rvert(2\pi\sigma^2t^3)^{-1/2}e^{-(\ell-\nu t)^2/2\sigma^2t}$. Completing the square with $\sigma^2\lambda=\sqrt{\nu^2+2r\sigma^2}$
turns the integrand into $\rho^{\mu\pm\lambda}\times$ first-passage densities of BMs with drift $\mp\sigma^2\lambda$, which integrate to $\mathcal F$. Checks: as $T\to\infty$,
$\mathcal F/R\to\rho^{\mu+\lambda}$ (down), the decaying root of $\tfrac12\sigma^2f''+\nu f'-rf=0$; at $r=0$, $\mathcal F/R=\mathbb Q(\tau\le T)$, which gives $E+\mathcal F=R$.

**3.4 Static hedge and model-free cases ($b=0$, so $S\equiv F$).** (6) follows from $\rho^{2\mu}=F/H$, homogeneity and put-call
symmetry (bsm.md §5). For (7): $F$ is a continuous martingale and $\tau$ a stopping time, so $\mathbb E\,F_{T\wedge\tau}=F$. On $\{\tau\le T\}$ we have
$F_\tau=H$, hence $\mathbb E[(F_T-H)\mathbf 1_{\tau>T}]=F-H$. For $K<H$ add $(H-K)\mathbb Q(\tau>T)$. Only continuity and the martingale property were used.
Under discrete monitoring $F_{\tau_d}<H$ (undershoot), so the $K=H$ value exceeds $DF(F-H)$, consistent with (13).

**3.5 Bridge (9).** Divide (†) at $T=\Delta$, $Y_0=a$, by the free density of $Y_\Delta$ at $c$. The exponent difference is
$\tfrac{2\nu(\ell-a)}{\sigma^2}-\tfrac{(c+a-2\ell-\nu\Delta)^2-(c-a-\nu\Delta)^2}{2\sigma^2\Delta}=-\tfrac{2(a-\ell)(c-\ell)}{\sigma^2\Delta}$: the **drift cancels**. A barrier that is linear in $t$
equals a constant barrier for $Y$ minus a linear function, i.e. a different drift, so (9) is exact for a spot barrier at any $b$. Under
Heston, given the variance path, $\ln S$ is Gaussian with time-varying variance and a $v$-dependent drift (heston-mc.md §3.6).
Freezing $v$ at $v_i^+$ over a step is the local-vol rule (QuantLib uses the diffusion at the step start); it is exact only as $\Delta\to0$.

**3.6 Variance (11).** $I=\prod_i\mathbf 1\{U_i>p_i\}$ satisfies $\mathbb E[I\mid\text{path}]=w$ and $\mathrm{Var}[I\mid\text{path}]=w(1-w)$. Hence $\mathrm{Var}(gI)=\mathrm{Var}(gw)+\mathbb E[g^2w(1-w)]$.

**3.7 BGK (12)** (BGK §1.2). The discrete walk first crosses at a monitored point that overshoots $H$ by $R_m$. Reflect there instead
of at $H$ and use $\mathbb E R_m\to\beta$, Siegmund's constant for Gaussian steps (in units of $\sigma\sqrt{\Delta t}$). The discrete price then equals the
continuous price at $H$ moved away from $S_0$ by $\beta\sigma\sqrt{\Delta t}$, up to $o(m^{-1/2})$. Kou (2003) proves all 8 cases by a discrete Girsanov
argument. Howison–Steinberg obtain $\beta$ as the asymptotic offset of the Spitzer function, and show the correction is weaker when
the payoff does not vanish at the barrier.

## 4. Numerical pitfalls (symptom → fix)

1. **Monitoring bias.** Checking only at grid points prices the $M$-date discrete contract. Knock-outs come out too high, and the error
   shrinks only as $\sqrt\Delta$ (13): halving $\Delta$ removes about 29% of it. BGK Tab. 2.1, $H=97$: continuous 3.060 vs daily 3.834 (+25%). Fix:
   use the bridge weight (10) for continuous contracts, and exact simulation at the dates (no bridge) for discrete ones. Never mix the two.
2. **Barrier at or near spot.** (3) with $\eta\ln\rho\ge0$ returns non-zero knock-outs. Fix: the branch (4), which applies to continuous monitoring
   only; a discrete contract is *not* knocked at $t=0$. BGK fails close to spot: −2.8% at $H=99$, $m=50$ (distance $0.53\sigma\sqrt{\Delta t}$), and −9.8% at $m=5$
   (Tab. 2.1–2.2). There, use discrete MC. Also, $\Delta t$ must be the date spacing, with the first date at $\Delta t$ (Howison–Steinberg §2).
3. **Overflow, underflow and 0/0 in $p_i$.** Without positive parts, endpoints on opposite sides give $p>1$ or inf, and negative weights.
   If $\sigma_i^2\Delta=0$ (Heston $v_i^+=0$) with an endpoint exactly on $H$, the result is NaN. Fix: (9) with $(\cdot)^+$, and
   `p = where(s2dt>0, exp(-2*a*c/s2dt), where(a*c>0, 0, 1))`. Underflow of `exp` to 0 for far-away paths is correct. $w\in[0,1]$ needs no logs.
4. **In and out on different randomness.** Separate seeds or grids, a re-drawn uniform for the in-leg, an in-weight built as a product of
   $p_i$, or inconsistent ties at $y=0$ all break in + out = vanilla by $O(\mathrm{SE})$ instead of $10^{-16}$. Fix: one path set, one $w$, in $:=g(1-w)$, and
   "alive iff $\eta y>0$" everywhere.
5. **Spot/forward mismatch.** $b$ in the closed form but $b=0$ in the MC (or the reverse) gives a gap $\propto bT$ that no $\Delta$-refinement removes.
   Fix: a single function for $y_i$ (8); test at $b\ne0$ (Haug, BGK).
6. **Heston bridge.** Under BSM the BB-MC is exact for any $\Delta$; under Heston it moves with $\Delta$. Using $v_{i+1}$ or an average instead of $v_i^+$
   changes finite-$\Delta$ prices, and at $v_i^+=0$ the step cannot cross. Fix: $v_i^+$ (D-entry); refine $\Delta$ until the change is below SE; validate with
   the $\xi\to0$ limit and with (7). No BGK under Heston: discrete contracts get discrete MC.
7. **Greeks blow up near the barrier.** Knock-outs whose payoff is non-zero at $H$ (cdo, cuo with $K<H$; pdo, puo with $K>H$; "reverse":
   cuo, pdo) jump by $\lvert H-K\rvert$ there. Near expiry delta scales like $1/(\sigma\sqrt\tau)$ and gamma like $1/(\sigma^2\tau)$ (est.); vega is negative near $H$.
   A FD bump that crosses $H$ jumps to branch (4). Fix: bump by less than half the distance to $H$; in MC use CRN with the smooth $w$, never $w^{(m)}$.
8. **Small σ or extreme $\rho$.** $\mu=b/\sigma^2-\tfrac12$ is large, so $\rho^{2\mu}\cdot\Phi(\cdot)$ can be inf·0 = NaN. Fix: compute $\exp(k\ln\rho+\log\Phi(\cdot))$ with
   `log_ndtr`, and write upper tails as $\Phi(-x)$ (bsm.md pitfalls 1–2).
9. **Negative rates with a rebate.** If $DF>1$ (accepted per D-012), $\lambda^2=\mu^2+2r/\sigma^2$ can be negative and $\mathcal F$ is NaN. Fix: skip $\mathcal F$
   if $R=0$, otherwise raise.
10. **Multiple 3SE tests.** Across 8 types × strikes × barriers, about 0.27% of comparisons fail for correct code (heston-mc.md (2)); in
    and out on the same paths are not independent tests. Fix: fixed seed; diagnose every failure; never re-seed.

## 5. Test oracles

**Exact on the same paths** (any model, BB or discrete weight; abs. $<10^{-12}$ at $F=100$; gate):
- in + out = vanilla for all 8 types at $R=0$. On each path $gw+g(1-w)=g$ holds to about 2 ulps, so the three means differ by roughly
  $\varepsilon\log_2N\max g\approx10^{-13}$ (est.). The identity is algebraic, not statistical: it fails at $O(\mathrm{SE})$ exactly when in and out see
  different paths or weights. Compute in and out in **separate** pricer calls; in := vanilla − out would make the test tautological.
- Ordering (14); monotonicity in $K$ and $H$ on common paths. $H\to0$ (down): $w\equiv1$ ($p$ underflows), so out ≡ vanilla.
  $H$ at or beyond $S_0$: $w\equiv0$, so out $=0$ and in = vanilla.

**Closed-form identities** (abs. $10^{-12}$ at $F=100$ or rel. $10^{-12}$): (5) for all 16 cells of (3); $E+\mathcal F=R$ at $DF=1$; branch continuity
at $K=H$; $\rho\to1$ continuity to (4); cdo → $A$ as $H\to0$ and cuo($K<H$) → $A$ as $H\to\infty$; (6) and (7) at $b=0$, $K\in\{H,0.9H\}$, several σ.

**MC vs closed form** (gate, 3SE): BSM BB-MC (8)–(10) vs (3) for all 8 types, $b\in\{0,0.04\}$, $M=1$ and $M\ge50$ (both unbiased).
Heston: $v_0=\theta$, $\xi=10^{-4}$, $\rho=0$ vs (3) at $\sigma=\sqrt\theta$ (3SE). Model-free (7): cdo($K=H=90$, $b=0$) $=10\,DF$ on the FO set
(heston-mc.md §5). Report the bias against $\Delta$ and require 3SE at the finest $\Delta$.

**Discrete monitoring** (approximate; tolerances fixed in advance). (i) Deterministic: (3) at the shifted barrier (12) reproduces the
BGK "corrected (1)" column to $5\cdot10^{-4}$. (ii) Discrete BSM MC with exact steps at $t_j$ vs BGK "True" (trinomial, average accuracy ≈0.001,
BGK §2): $\lvert\Delta\rvert\le3\,\mathrm{SE}+0.001$. (iii) BGK-corrected (3) vs discrete MC is a diagnostic, not a gate. Expected error ≤0.1% for a standard
DOC when $\ln(S_0/H)\ge1.6\sigma\sqrt{\Delta t}$ ($m=50$, est. from Tab. 2.1), and 0.1–1.5% for the reverse UOC (Tab. 2.6). (iv) Figure: uncorrected
discrete minus continuous vs $\Delta t$ on log-log axes, slope ≈ 0.5 (13).

**Published values. Haug's table:** $S_0=100$, $r=8\%$, $b=4\%$, $T=0.5$, $R=3$, continuous monitoring ⇒ $F=100e^{0.02}=102.0201340$,
$DF=e^{-0.04}=0.9607894392$, carry input $b=0.04$. Tolerance $5\cdot10^{-5}$ (QuantLib tests at $10^{-4}$). Source: QuantLib
`test-suite/barrieroption.cpp` (cites Haug 1998, p. 72). ² = the σ=0.25 entries also appear identically in OptionStratLib issue #646
(cites Haug 2007, Table 4-13). All other entries (σ=0.30, and rows without ²) are ⚑ QuantLib only. **Haug's book itself was not accessed.**

| type | $H$ | σ=0.25: $K$=90 / 100 / 110 | σ=0.30: $K$=90 / 100 / 110 |
|---|---|---|---|
| cdo² | 95 | 9.0246 / 6.7924 / 4.8759 | 8.8334 / 7.0285 / 5.4137 |
| cdo | 100 | 3.0000 / 3.0000 / 3.0000 | 3.0000 / 3.0000 / 3.0000 |
| cuo² | 105 | 2.6789 / 2.3580 / 2.3453 | 2.6341 / 2.4389 / 2.4315 |
| cdi² | 95 | 7.7627 / 4.0109 / 2.0576 | 9.0093 / 5.1370 / 2.8517 |
| cdi | 100 | 13.8333 / 7.8494 / 3.9795 | 14.8816 / 9.2045 / 5.3043 |
| cui² | 105 | 14.1112 / 8.4482 / 4.5910 | 15.2098 / 9.7278 / 5.8350 |
| pdo² | 95 | 2.2798 / 2.2947 / 2.6252 | 2.4170 / 2.4258 / 2.6246 |
| pdo | 100 | 3.0000 / 3.0000 / 3.0000 | 3.0000 / 3.0000 / 3.0000 |
| puo² | 105 | 3.7760 / 5.4932 / 7.5187 | 4.2293 / 5.8032 / 7.5649 |
| pdi² | 95 | 2.9586 / 6.5677 / 11.9752 | 3.8769 / 7.7989 / 13.3078 |
| pdi | 100 | 2.2845 / 5.9085 / 11.6465 | 3.3328 / 7.2636 / 12.9713 |
| pui² | 105 | 1.4653 / 3.3721 / 7.0846 | 2.0658 / 4.4226 / 8.3686 |

Isolated-block checks: cuo($K$=110) and pdo($K$=90) equal $\mathcal F$ alone. The $H=100$ rows test (4) and satisfy parity (bsm.md eq. 2).
Internal consistency, computed here from the table: in + out − vanilla $=E+\mathcal F$ is independent of $K$ and θ. It equals 2.9539–2.9540 ($H$=95)
and 2.9568 ($H$=105) at σ=.25, and 2.9610–2.9611 ($H$=95) at σ=.30.

**BGK (1997)**, read in the author copy: $R=0$, $q=0$ ⇒ $b=r=0.10$, $T=0.2$, $DF=e^{-0.02}=0.9801986733$, $m=50$ ($\sigma\sqrt{\Delta t}=0.018974$).
Values are continuous / corrected (12) / True. Tab. 2.1 (DOC, $S_0=100$ ⇒ $F=102.0201340$, $K=100$, σ=0.30), ⚑ primary only:

| $H$ | 85 | 90 | 95 | 96 | 97 | 98 | 99 |
|---|---|---|---|---|---|---|---|
| cont. / corr. / True | 6.308 / 6.322 / 6.322 | 5.977 / 6.098 / 6.098 | 4.398 / 4.907 / 4.907 | 3.792 / 4.428 / 4.427 | 3.060 / 3.836 / 3.834 | 2.189 / 3.121 / 3.126 | 1.171 / 2.271 / 2.337 |

Tab. 2.2 ($m=5$): $H$=95: 4.398 / 5.646 / 5.671; $H$=99: 1.171 / 4.050 / 4.489. Tab. 2.6 (UOC, $S_0=110$ ⇒ $F=112.2221474$, $K=100$, σ=0.30),
reproduced identically in Kou (2003) Tab. 2.1 (2 sources): $H$=155: 12.775 / 12.905 / 12.894; 140: 10.144 / 10.581 / 10.551; 125: 4.012 / 4.649 /
4.616; 115: 0.545 / 0.819 / 0.807. The continuous columns are zero-rebate RR oracles at $b\ne0$ (tolerance $5\cdot10^{-4}$).

## 6. Remarks

1. **In + out = vanilla, and the knock-in is a mirror image.** On every path exactly one of the pair pays, so together they are the
   vanilla, in any model; the MC must reproduce this to round-off. Reflect a path at the barrier: with zero carry, a down-and-in call
   becomes $K/H$ puts struck at $H^2/K$ (put-call symmetry). That is a static hedge: buy the puts, and swap them for the call at no cost if
   the barrier trades. Carry tilts the mirror (the factor $\rho^{2\mu}$), which is why you must say whether the barrier is on spot or the forward.
2. **A discretely monitored barrier is a continuous barrier moved away by 0.58σ√Δt.** Between fixings a path can dip through the barrier
   unseen, so discrete knock-outs are worth more. The first observed breach overshoots by on average 0.58 step standard deviations,
   hence the BGK shift. A naive MC error decays only like √Δt. The Brownian bridge removes it for continuous contracts by weighting each
   path with the probability that it did not touch the barrier between grid points, given the endpoints. That is exact for GBM and
   approximate under stochastic vol.
3. **Barrier risk sits at the barrier.** Near the barrier a knock-out loses value when vol rises (negative vega). A reverse knock-out is short a
   digital at the barrier, so near expiry delta and gamma explode and cannot be hedged smoothly; desks shift the barrier. The price depends on
   the vol at the barrier and on forward skew, not on ATM vol. That is why BSM and Heston disagree on barrier products (autocallable gate).

## 7. References

- Reiner, E., Rubinstein, M. (1991). Breaking down the barriers. *Risk* 4(8), 28–35. Not read; formulas via Haug/NAG/QuantLib.
- Merton, R.C. (1973). Theory of rational option pricing. *Bell J. Econ. Manag. Sci.* 4(1), 141–183. Not read; via BGK.
- Haug, E.G. (2007). *The Complete Guide to Option Pricing Formulas*, 2nd ed., McGraw-Hill (1st ed. 1998). Not accessed; table via QuantLib.
- Broadie, M., Glasserman, P., Kou, S. (1997). A continuity correction for discrete barrier options. *Math. Finance* 7(4), 325–348.
  Read: author copy, Thm 1.1, Prop. 3.1, Tables 2.1–2.6.
- Kou, S.G. (2003). On pricing of discrete barrier options. *Statistica Sinica* 13, 955–964. Read (Thm 2.1, Table 2.1).
- Howison, S., Steinberg, M. (2005). A matched asymptotic expansions approach to continuity corrections for discretely sampled
  options. Part 1: barrier options. Oxford preprint, 5 April 2005 (read; journal version not checked).
- Shevchenko, P.V., Del Moral, P. Valuation of barrier options using sequential Monte Carlo. arXiv:1405.5294 (eqs. 6, 27 read).
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer, pp. 368–370 (via Shevchenko–Del Moral; not read).
- Beaglehole, D.R., Dybvig, P.H., Zhou, G. (1997). Going to extremes: correcting simulation bias in exotic option valuation. *Financial
  Analysts J.* 53(1). Not read. · Gobet, E. (2000). Weak approximation of killed diffusion using Euler schemes. *Stoch. Proc. Appl.* 87(2),
  167–197. Not read (BB-Euler weak order 1).
- Siegmund, D. (1985). *Sequential Analysis*. Springer (via BGK/Kou) · Karatzas, I., Shreve, S. (1991). *Brownian Motion and Stochastic
  Calculus*, 2nd ed., Springer (via BGK) · Carr, P., Ellis, K., Gupta, V. (1998). Static hedging of exotic options. *J. Finance* 53(3), 1165–1190.
- NAG Library Mark 29, S30FAF docs; QuantLib `analyticbarrierengine.cpp`, `mcbarrierengine.cpp`, `test-suite/barrieroption.cpp`;
  OptionStratLib issue #646 (all accessed 2026-11-07).

## 8. Errata (Phase 4 implementation, 2026-11-08)

- **E1 (§5, Haug table, σ = 0.30 column; D-027; revised in fix round 1).** The 5e-5 tolerance holds for 66 of the
  72 cells: all 36 σ = 0.25 cells (two sources; max 4.96e-5) and 30 of the σ = 0.30 cells (max 4.59e-5). It does not
  hold for six σ = 0.30 cells, which are QuantLib only (⚑): cuo(105, K = 90), puo(105, K = 90/100/110), pdi(95,
  K = 100/110), off by 5.2e-5 to 6.25e-5. Three independent methods agree on these cells:
  - direct quadrature of the killed density (§3.2) and of the first-passage density (§3.3), within ≤ 3.2e-14 of (3);
  - the validator's Haug spot-form formula;
  - a Crank–Nicolson PDE, within ≤ 5.5e-10 of the spot-form formula.
  So the printed entries are not correctly rounded, e.g. puo K = 90 = 4.22923747, printed 4.2293.
  **Correction:** keep 5e-5 on the 66 correctly rounded cells. Test the six cells against the two-method values in
  `tests/golden/barrier_haug.json` (validator; flag `published_correctly_rounded = false`) at 1e-10, and every cell vs
  that file at 1e-10. Add RR vs independent quadrature at 1e-12 on all H ≠ S_0 cells. No tolerance is loosened. (The
  first version of this erratum moved the whole σ = 0.30 column to 1e-4; that version is withdrawn.) Numbers:
  `scripts/phase4_gates.py`, `docs/validation/phase4.md`.
- **E2 (§5 (iv), figure slope; D-028).** "Slope ≈ 0.5" is the asymptotic order of (13). On the Table 2.1 DOC (H = 95)
  over m = 4…512 the uncorrected-MC bias has fitted slope 0.422. The noise-free BGK prediction V(H e^{−βσ√Δt}) − V(H) on
  the same m gives 0.426; its local slope rises from 0.32 (m = 4 → 8) to 0.50 (m ≈ 10⁶). That is pre-asymptotic curvature of
  V in H, not a defect. The figure reports both slopes (`scripts/exotics_figures.py`).
