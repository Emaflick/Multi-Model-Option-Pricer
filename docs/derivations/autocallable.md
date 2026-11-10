# Mono-asset autocallable: cash flows, decomposition, closed-form oracles, MC, BSM-vs-Heston gap

Phase 4 · kernels `src/qpricer/products.py` (payoff), `src/qpricer/engines/mc.py` (paths) · docstrings cite the eq. numbers
below. Barrier and Asian options have their own notes. Heston paths: `heston-mc.md` (QE-M, eqs. 7–12); Black-76: `bsm.md`.
⚑ = fewer than two external sources checked; (der.) = derived in this note. No model price appears here; the only published
numbers are quoted from Deng–Mallett–McCann (2011) in §5.

## 1. Setup

- **Measure.** Risk-neutral $\mathbb Q$ (bank-account numéraire) with **deterministic rates and carry** (D-entry). Then $P(0,t)=DF(t)$,
  every $t$-forward measure equals $\mathbb Q$, and $F_0(t):=F(0,t)=\mathbb E^{\mathbb Q}S_t$. Simulated state: $M_t:=S_t/F_0(t)$, a martingale,
  $M_0=1$. BS: $dM_t=\sigma M_tdW_t$. Heston: $dM_t=\sqrt{v_t}M_tdW^S_t$, $v$ as in heston-mc.md §1 ($x_t=\ln M_t$ is the state there, eq. 17).
- **Term sheet** (notional 1; reference level $S_{\rm ref}$ fixed at inception; all levels are ratios to $S_{\rm ref}$). Dates
  $0=t_0<t_1<\dots<t_n=T$, $\tau_i=t_i-t_{i-1}$ (ACT/365F); coupon $c_i=c\,\tau_i$ ($c$ = annual rate, simple); barriers
  $0\le B_p\le B_c\le B_{ac}$, default $B_{ac}=1$, $B_c\in[0.7,0.8]$, $B_p=0.6$; memory flag $m\in\{0,1\}$. Fixings
  $X_i:=S_{t_i}/S_{\rm ref}=f_iM_{t_i}$, $f_i:=F_0(t_i)/S_{\rm ref}$. Every test is "$\ge$" (at the barrier counts as above).
  - $t_i<T$: if $X_i\ge B_{ac}$, pay $1+$coupon and terminate; otherwise pay the coupon if $X_i\ge B_c$.
  - $t_n=T$ (no autocall test): coupon if $X_n\ge B_c$; capital $1$ if $X_n\ge B_p$, else $X_n$ (European protection, observed at $T$ only).
  - Coupon $=c_i$ (no memory), or $c_i$ plus all unpaid earlier coupons (memory).
  - Variants (D-entry): **AC-Q** quarterly, $t_i=i/4$, $n=6$, $T=1.5$ y (inside the last kept expiry, 1.61 y); **AC-A** annual,
    $t_i=i$, $n=3$ (carry and Heston are extrapolated beyond 1.61 y; label this in the report).
- **Payment and discounting.** A cash flow fixed at $t_i$ is paid at $t_i$ (no lag) and discounted with $DF(t_i)$. Discounting is
  risk-free (parity DF), with no issuer credit or funding spread. Real notes are issuer debt; Deng et al. add the issuer CDS
  spread to the discount rate (D-entry).
- **Market inputs.** $DF(t_i)$ and $F_0(t_i)$ come from linear interpolation of $\ln DF$ and $\ln F$ in $T$ between the kept expiries of
  `market.py` (piecewise-flat forward rate and carry). Outside $[T_{\min},T_{\max}]$, use a constant zero rate and carry fitted on
  $T>0.3$ (D-007). $S_{\rm ref}$ is a contract constant. For the study set $S_{\rm ref}:=F_0(T_{(1)})DF(T_{(1)})$ (spot minus the PV of
  dividends to the first expiry, about 1 week) rather than the non-synchronous Yahoo spot (D-008).
- **Notation.** $d_\pm(f,L,s)=[\ln(f/L)\pm s^2/2]/s$ (bsm.md's $d_{1,2}$ with $F\to f$, $K\to L$). Total std. dev. to $t_i$ is $s_i$:
  $\sigma\sqrt{t_i}$ (flat) or $\sqrt{w(t_i)}$ (deterministic variance $w$). $s_{i-1,i}^2=s_i^2-s_{i-1}^2$. $p_i(k)=\mathbb E(k-X_i)^+$ (undiscounted
  ratio put); $D_i^-(k)=\mathbb Q(X_i<k)$, $D_i^+=1-D_i^-$; $\sigma_i(k)$ = implied vol at strike $kS_{\rm ref}$, expiry $t_i$, $\sigma_i'=\partial_k\sigma_i$; $DF_i=DF(t_i)$.

## 2. Result

### 2.1 Cash flows on one path
**(1)** Alive and call indicators: $A_1=1$; $\chi_i=A_i\mathbf 1\{X_i\ge B_{ac}\}$ for $i<n$, $\chi_n=0$; $A_{i+1}=A_i-\chi_i$.\
**(2)** Coupon: $\pi_i=A_i\mathbf 1\{X_i\ge B_c\}$, entitlement $G_i=c_i+m\,U_i$, paid $C_i=\pi_iG_i$; memory state $U_1=0$, $U_{i+1}=(1-\pi_i)G_i$.
For $m=1$ this is $C_i=\pi_i\sum_{j=\ell_i+1}^{i}c_j$, where $\ell_i$ is the last payment date before $i$ (0 if none).\
**(3)** Redemption: $R_i=\chi_i$ for $i<n$; $\;R_n=A_n\,[1-(1-X_n)\mathbf 1\{X_n<B_p\}]$.\
**(4)** $\mathrm{PV}=\sum_{i=1}^nDF_i\,(C_i+R_i)$, $\qquad V=\mathbb E^{\mathbb Q}[\mathrm{PV}]$.\
**(5)** Affine in $c$: $\mathrm{PV}=a+c\,b$ with $a=\sum_iDF_iR_i$ and $b=\sum_iDF_iC_i/c$. No indicator depends on $c$.

### 2.2 Decomposition and who holds what (exact; needs $B_p\le1$)
**(6)** With $\tau^\star$ = call date, or $T$ if never called:
$V=\underbrace{\mathbb E[DF(\tau^\star)]}_{\text{bond, random maturity}}+\underbrace{\textstyle\sum_iDF_i\,\mathbb E[C_i]}_{\text{coupon digitals}}-\underbrace{DF_n\,\mathbb E[A_n(1-X_n)\mathbf 1\{X_n<B_p\}]}_{\text{DIP knocked in at }T\text{, if alive}}$.
The **investor** is long the first two terms and short the put. The **issuer** holds the mirror position: long the down-and-in
put (DIP), short the coupon digitals, short the bond. Since $A_n\le1$, the embedded put is worth at most the European DIP\
**(7)** $\mathrm{DIP}_E=DF_n\,\mathbb E[(1-X_n)\mathbf 1\{X_n<B_p\}]=DF_n\,[\,p_n(B_p)+(1-B_p)\,D^-_n(B_p)\,]$: a put at $B_p$ plus $(1-B_p)$ digital puts
at $B_p$. This is static and model-free given the $T$-smile.\
**(8)** Black-76: $\mathrm{DIP}_E=DF_n[\Phi(-d_-)-f_n\Phi(-d_+)]$ with $d_\pm=d_\pm(f_n,B_p,s_n)$; $\;D_i^+(k)=\Phi(d_-(f_i,k,s_i))$.\
**(9)** Smile, evaluated at $\sigma=\sigma_i(k)$: $D_i^-(k)=\Phi(-d_-)+k\,\varphi(d_-)\sqrt{t_i}\,\sigma_i'(k)$ and $D_i^+(k)=\Phi(d_-)-k\varphi(d_-)\sqrt{t_i}\,\sigma_i'(k)$, so
$\mathrm{DIP}_E=DF_n[\Phi(-d_-)-f_n\Phi(-d_+)+(1-B_p)B_p\,\varphi(d_-)\sqrt T\,\sigma_n'(B_p)]$ at $\sigma_n(B_p)$. No single flat vol prices both legs:
the barrier vol prices the put, but the digital also needs the slope.\
**(10)** Vega at inception (der., §3.4): $\partial_s\mathrm{DIP}_E>0$ for $B_p<f_n$; $\partial_sD_i^+(k)<0$ for $k<f_ie^{s_i^2/2}$. This covers the coupon digitals
at $B_c$ and the call digitals at $B_{ac}=1\le f_i$ (positive carry). So the no-call sheet has $\partial V/\partial\sigma<0$:
**the investor is short vega, the issuer long vega.** With the autocall there is no general sign.

### 2.3 Closed-form oracles (Black-76, flat or deterministic variance)
**(11)** No call, no memory ($B_{ac}=\infty$, $m=0$; marginals only): $V^{nc}=\sum_iDF_ic_i\,\Phi(d_-(f_i,B_c,s_i))+DF_n-\mathrm{DIP}_E$, with (8).
Sub-cases: $B_c=0\Rightarrow V=\sum_iDF_ic_i+DF_n-\mathrm{DIP}_E$; adding $B_p=0$ gives $V=\sum_iDF_ic_i+DF_n$ (zero variance). For $n=1$, (11) holds for any $B_{ac}$.\
**(12)** $n=2$, any $m$, $B_c<B_{ac}$:
$V=DF_1[c_1\Phi(d_-(f_1,B_c,s_1))+\Phi(d_-(f_1,B_{ac},s_1))]+DF_2\int_{-\infty}^{z_{ac}}\varphi(z)\,h(x(z))\,dz$, where $x(z)=f_1e^{s_1z-s_1^2/2}$,
$g=x f_2/f_1$, $s_{12}=s_{1,2}$, $z_L=-d_-(f_1,L,s_1)$ and
$h(x)=(c_2+m\,c_1\mathbf 1\{x<B_c\})\,\Phi(d_-(g,B_c,s_{12}))+1-\Phi(-d_-(g,B_p,s_{12}))+g\,\Phi(-d_+(g,B_p,s_{12}))$.
Integrate (`scipy.integrate.quad`, `epsabs=1e-13`) separately on $(-\infty,z_c)$ and $(z_c,z_{ac})$, because of the memory jump at $z_c$.\
**(13)** $\sigma=0$: $X_i=f_i$ exactly, and $V$ is the term sheet evaluated by hand (e.g. $f_1\ge B_{ac}\Rightarrow V=DF_1(1+c_1)$).\
Heston analogue of (11): $D_i^\pm$ and $\mathrm{DIP}_E$ from COS (heston.md) at each $t_i$. Get the digital as the strike derivative of the COS put
(central difference) or from the $\psi_k$ coefficients.

### 2.4 Monte Carlo
**(14)** BS (exact, one step per period): $\ln M_{t_i}=\ln M_{t_{i-1}}-\tfrac12s_{i-1,i}^2+s_{i-1,i}Z_i$. Heston: QE-M (heston-mc eqs. 7–12) with
$m_i=\lceil\tau_i/\Delta_{\max}\rceil$ equal sub-steps per period, $\Delta_{\max}=1/32$, ending exactly at $t_i$; store $x$ only at the $t_i$.\
**(15)** Common random numbers (CRN): for period $i$, set $Z_i:=\sum_k\sqrt{\Delta_k/\tau_i}\,(\rho Z_{v,k}+\sqrt{1-\rho^2}\,Z_{\perp,k})$, summing over the Heston sub-step
draws of that period. These $Z_i$ are exactly iid $N(0,1)$, so BS is driven by Heston's spot Brownian motion.\
**(16)** $\hat V=\overline{\mathrm{PV}}$, SE $=s_{\rm PV}/\sqrt N$ (heston-mc eq. 1). Model gap: $D_j=\mathrm{PV}^H_j-\mathrm{PV}^{BS}_j$, $\widehat{\Delta V}=\bar D$, SE $=s_D/\sqrt N$.
On CRN paths never use $\sqrt{\mathrm{SE}_H^2+\mathrm{SE}_{BS}^2}$.\
**(17)** Par coupon (target $P^\star=1$, or $1-$fee): $\hat c=(P^\star-\bar a)/\bar b$, with $\mathrm{SE}(\hat c)=s(a_j+\hat c\,b_j)/(\sqrt N\,\bar b)$ (delta method).
For the gap: $\mathrm{SE}(\hat c_H-\hat c_{BS})=s(\psi_j)/\sqrt N$, $\;\psi_j=(a^H_j+\hat c_Hb^H_j)/\bar b_H-(a^{BS}_j+\hat c_{BS}b^{BS}_j)/\bar b_{BS}$.\
**(18)** First-call probabilities $\hat p_i=\overline{\chi_i}$ satisfy $\sum_{i<n}\hat p_i+\overline{A_n}=1$ exactly; expected life is $\sum_{i<n}t_i\hat p_i+T\,\overline{A_n}$.

### 2.5 BSM-vs-Heston attribution (one set of CRN paths)
**(19)** $V_H-V_{BS}=\underbrace{(V_{TS}-V_{BS})}_{\text{ATM term structure}}+\underbrace{(V^{nc}_H-V^{nc}_{TS})}_{\text{marginal smile at }B_p,\,B_c}+\underbrace{[(V_H-V_{TS})-(V^{nc}_H-V^{nc}_{TS})]}_{\text{path: forward skew, vol-of-vol}}$.
- $V_{BS}$: flat $\sigma_{ATM}$ = Heston implied vol at $(K=F_0(T),T)$, so both models price the ATM-forward $T$-vanilla alike (D-entry). Also report
  BS at the market ATM vol, at $K=S_{\rm ref}$ and at $K=B_pS_{\rm ref}$.
- $V_{TS}$: BS with Heston's expected variance $w(t)=\theta t+(v_0-\theta)(1-e^{-\kappa t})/\kappa$ (the $\xi\to0$ limit, heston-mc eq. 15).
- "nc": same sheet with $B_{ac}=\infty$, $m=0$. Exact via (11) and COS; also priced on the MC paths as a 3SE check.

**Expected sign (§3.5): $V_H<V_{BS}(\sigma_{ATM})$, i.e. $\hat c_H>\hat c_{BS}$.** The marginal-smile term drives it. The path term is the model-risk part,
and its sign is not asserted.

### 2.6 Cross-check status (sources actually read)
| eq. | external sources | status |
|---|---|---|
| (1)–(4) | term sheet (ROADMAP); cash-flow/probability form of Deng et al. (2011) eqs. (6)–(7) | ⚑ 1 + der. |
| (6)–(7) | Langnau (2010) eq. (6) (digital put $=\partial\,$Put$/\partial K$, model-free); Breeden–Litzenberger (1978); algebra §3.2 | 2 |
| (8) | bsm.md §3.1; Langnau (2010) eq. (8) (BS digital put $=Df\,N(-d_2)$) | 2 |
| (9) | chain rule on (7) (der.); qualitative only in Langnau §5 | ⚑ der. |
| (10) | der. §3.4. Issuer long vega, peak-vega flip: Bartholomew–Davis (Risk.net 2020), Conde (SRP 2020); both trade press | 2 (press) |
| (11)–(13), (17)–(19) | derivation only; (17) is the standard ratio-estimator delta method | ⚑ der. |
| (14)–(16) | heston-mc.md (2 sources per eq.); CRN: Glasserman (2003) | 2 |
| §3.5 (iv) | Farkas–Ferrari–Ulrych (2022) abstract; Mazzon–Pascucci intro (local-vol forward skews flat, citing Schoutens et al. 2004) | ⚑ abstract/intro |

## 3. Derivation

**3.1 Measure.** With deterministic $r$, $V=\sum_i\mathbb E[e^{-\int_0^{t_i}r}CF_i]=\sum_iDF_i\,\mathbb E[CF_i]$, and $F_0(t)=\mathbb ES_t$ (forward = futures). So $M$ is a
martingale and (4) holds. Under a single $T$-forward measure the value is $DF(T)\sum_i\mathbb E^T[CF_i/P(t_i,T)]$. A flow paid at $t_i<T$
must be rolled to $T$ at the random bond price $P(t_i,T)$, and $S_{t_i}$ is no longer a function of $F(t_i,T)$ alone, because
$S_t=F(t,T)\,P(t,T)\times$carry. Only with deterministic rates is $P(t_i,T)=DF_n/DF_i$; then (4) is recovered with $X_i=f_i\,F(t_i,T)/F_0(T)$
(heston-mc eq. 17). Simulating $M$ under $\mathbb Q$ and discounting each flow at its own date avoids the issue.

**3.2 Cash-flow algebra.** (1)–(3) transcribe the term sheet, with $A_n=\prod_{i<n}\mathbf 1\{X_i<B_{ac}\}$. On a call date $X_i\ge B_{ac}\ge B_c$, so a call always
pays the coupon. Capital: $\mathbf 1\{X\ge B_p\}+X\mathbf 1\{X<B_p\}=1-(1-X)\mathbf 1\{X<B_p\}$, and for $B_p\le1$,
$(1-X)\mathbf 1\{X<B_p\}=(B_p-X)^++(1-B_p)\mathbf 1\{X<B_p\}$. This gives (6)–(7). Grouping the "1" terms of $\sum DF_i(C_i+R_i)$ by redemption
date gives the random-maturity bond. Linearity (5) holds because $c$ enters only through $c_j=c\tau_j$ inside $G_i$.

**3.3 Black-76 pieces.** (8) follows from $\mathbb Q(X<L)=\Phi(-d_-)$ and $\mathbb E[X\mathbf 1\{X<L\}]=f\Phi(-d_+)$ (share-measure change, bsm.md §3.1).
(9): if $p(k)=p_{BS}(k,\sigma(k))$, then $D^-(k)=\partial_kp=\partial_kp_{BS}+\partial_\sigma p_{BS}\,\sigma'(k)$, with $\partial_kp_{BS}=\Phi(-d_-)$ and
$\partial_\sigma p_{BS}=k\varphi(d_-)\sqrt t$ (bsm.md eqs. 7, 9; Langnau eq. 6). (11): with $B_{ac}=\infty$ and $m=0$, each term of (4) depends on a single $X_i$.
(12): condition on $X_1$. Since $\ln M$ has independent Gaussian increments, given $X_1=x$ the fixing $X_2$ is lognormal with mean $g$ and
total std. dev. $s_{12}$, so (8) applies to the $t_2$ flows. Finally $X_1<L\iff z<z_L$.

**3.4 Vega signs.** $\partial_sd_-(f,L,s)=-\ln(f/L)/s^2-\tfrac12$. So $\partial_s\Phi(d_-)<0$ and $\partial_s\Phi(-d_-)>0$ exactly when $L<fe^{s^2/2}$. Together with
$\partial_sp>0$ this gives (10), and through (11) $\partial_\sigma V^{nc}<0$. With the autocall, $\sigma\to0$ gives (13) and $\sigma\to\infty$ sends $X_i\to0$ in
probability, so $V\to0$ (bounded payoff). In between, monotonicity is not guaranteed, because vol also moves the call timing.
As spot falls towards $B_p$ near $T$, the digital of size $1-B_p=0.4$ dominates. An OTM digital put has positive vega and an ITM one
negative vega: the sign of $\partial_s\Phi(-d_-)$ flips at $L=fe^{s^2/2}$. So the issuer's vega grows as spot falls, peaks above the barrier,
then falls and flips. Dealers report this "peak vega" (Risk.net 2020); SRP (2020) puts the peak about 20–25% below the issue level,
depending on the structure, and says dealers sell volatility, often OTM puts, to hedge.

**3.5 Why a flat ATM vol overvalues the note (expected sign).** (i) *Protection barrier.* The issuer-held put sits at $B_p=0.6$, deep in
the SPX downside skew. In (9), $\sigma_n(B_p)\gg\sigma_{ATM}$ raises both the put and $\Phi(-d_-)$, while the slope term ($\sigma'<0$) lowers the
digital. The net $D^-_n(B_p)$ exceeds its flat-ATM value whenever the smile's left tail is fatter than lognormal. That is a property of
the surface: check it with (9) or COS before running any MC. (ii) *Coupon barrier.* The same fat tail lowers $D^+_i(B_c)$ at every $t_i$, so
the coupons are worth less. (iii) *Autocall barrier ≈ ATM.* By (9), $D^+_i(1)$ exceeds its flat value when $\sigma'<0$. That means more early
calls, a shorter life, fewer coupons and less put exposure; the price effect is second order and its sign depends on the product.
(iv) *Path: forward skew and vol-of-vol.* Flows at $t_i$ are conditional on survival ($X_j<B_{ac}$ for $j<i$). With $\rho<0$, Heston variance
rises after down moves, so the conditional law at later dates, given a low path, has a fatter left tail than BS. Heston's forward smile
stays skewed because its parameters are time-homogeneous; BS has no forward skew at all. In pure Heston, $\xi$ (curvature) and $\rho\xi$
(skew) are pinned by the vanilla fit, so this effect cannot be moved on its own. With local-stochastic vol, Farkas et al. (2022) report
autocallable prices typically above local vol at the same vanilla fit and attribute the gap to forward skew; local-vol forward skews
are known to be flat (Mazzon–Pascucci). Terms (i) and (ii) are first order in $\sigma(B)-\sigma_{ATM}$, so the expected net is $V_H<V_{BS}$. Eq. (19)
measures each piece.

## 4. Numerical pitfalls (symptom → fix)

1. **Digitals make Greeks unstable.** Pathwise delta/vega drops the Dirac term of every indicator (biased, far too small). Bump-and-reprice
   variance grows like $1/h$, even with CRN. Fix: the price needs no fix. For Greeks (outside the gates): BS likelihood-ratio delta
   $\mathbb E[\mathrm{PV}\,Z_1]/(S\,s_{0,1})$ (Glasserman ch. 7); call-spread smoothing of width $\varepsilon$ (bias $O(\varepsilon^2)$, report $\varepsilon$); or Fries–Joshi conditional MC.
2. **Variance from jumps.** The capital jumps by 0.4 at $B_p$ and the redemption date is random, so the SE is that of a few Bernoullis.
   Fix: $N\ge2\cdot10^5$. Optional under BS: replace the last-period flows by their conditional expectation given $(X_{n-1},U_n)$, as in $h$
   of (12); this is exact and lowers variance.
3. **Memory bookkeeping.** Symptom: (12) with $m=1$ misses by about $DF_2c_1\times$prob., or memory is paid on dead paths or never reset.
   Fix: on each date run, in this order, $G=c_i+mU$; $\pi=A\mathbf 1\{X\ge B_c\}$; $C=\pi G$; $U=(1-\pi)G$; $\chi$; $A\mathrel{-}=\chi$. Check with hand path (b) of §5.
4. **Termination masks.** Symptoms: the price breaks the §5 bound, $\sum\hat p_i+\overline{A_n}\ne1$, or the "1" is paid twice at $T$. Fix: $\chi_n=0$;
   multiply the flows by $A$ before updating $A$; store $A$ as bool.
5. **Discounting.** Symptom: the $\sigma=0$ test misses by $DF_1$ vs $DF_n$. Fix: a vector $DF_i$, never $e^{-rt_i}$ with $r$ taken from $T$. Assert that the
   interpolated $DF_i$ are non-increasing (short-end noise, D-007).
6. **Heston observation grid.** Symptom: the MC digitals at $t_i$ miss COS by more than 3SE, or $\overline{M_{t_i}}\ne1$. Causes: dates rounded to the grid,
   one QE step per quarter, or plain (non-martingale) QE. Fix: sub-steps (14) ending exactly at $t_i$, with the QE constants $E,K_0..K_4$
   recomputed for each step size (as in asian.md), using QE-M.
7. **Barrier ties.** Symptom: hand-path or $\sigma=0$ tests fail at $X=B$, because rounding turns $1.0$ into $0.9999999999999999$. Fix: "$\ge$" in ratio space;
   in deterministic tests keep $f_i$ away from barriers or build $X$ exactly.
8. **Independent seeds for the model gap.** Symptom: the gap SE is about $\sqrt2\times$ a single SE and the sign is not resolved. Fix: CRN (15)–(16).
   The digitals cap the correlation (paths near a barrier flip), so always report $s_D$.
9. **Fair coupon by root search.** Symptom: $V(c)$ is noisy and non-monotone when each iterate uses fresh paths. Fix: the ratio (17); its bias is $O(1/N)$.
10. **"ATM vol" and extrapolation.** Symptom: the BS price moves by more than the reported gap between $\sigma_{ATM}$ at $F_0(T)$ and at $S_{\rm ref}$.
    Fix: fix the choice in a D-entry and report the sensitivity rows of (19). AC-A beyond 1.61 y extrapolates both carry and Heston.
11. **Feller-violating fits.** QE handles $v\to0$, and QE-M is always defined for $\rho\le0$ (heston-mc eq. 13). The payoff is bounded, so the
    variance is finite (unlike calls, heston-mc §3.1).

## 5. Test oracles

**Exact on the same paths ($10^{-12}$, any model):** $\sum_{i<n}\hat p_i+\overline{A_n}=1$. $\mathrm{PV}(c')-\mathrm{PV}(c)=(c'-c)\,b$. $0\le\mathrm{PV}\le\max_iDF_i+\sum_iDF_ic_i$ (needs
non-increasing DF). With CRN, pathwise: PV is non-increasing in $B_c$ and $B_p$, non-decreasing in $c$, and memory $\ge$ no memory, with
equality at $B_c=0$. $B_{ac}=0\Rightarrow\mathrm{PV}\equiv DF_1(1+c_1)$ ($n\ge2$). $B_{ac}=\infty,B_c=0,B_p=0\Rightarrow\mathrm{PV}\equiv\sum DF_ic_i+DF_n$. BS with $\sigma=0$ gives (13).\
**Hand paths** (an $X$ matrix fed to the payoff function, flows computed by hand): (a) called at $t_1$; (b) $X_1<B_c$ missed, paid at $t_2$
with memory ($C_2=c_1+c_2$), called at $t_3$; (c) never called with $B_p\le X_n<B_c$ (capital 1, no coupon); (d) $X_n<B_p$ (capital $X_n$);
(e) $X_n\ge B_{ac}$ at $T$ (coupon $+1$, nothing doubled).\
**BS within 3SE** ($N=2\cdot10^5$, fixed seed): (11) and its sub-cases; $n=1$; $n=2$ via (12) for $m\in\{0,1\}$; the AC-Q no-call sheet.
$V^{nc}(\sigma)$ is strictly decreasing (closed form, exact). For the full product, report a CRN $\sigma$-grid as a diagnostic, not a gate (§3.4).\
**Heston:** at every $t_i$, MC $D_i^\pm(L)$ for $L\in\{B_p,B_c,B_{ac}\}$ vs COS, and $\overline{M_{t_i}}=1$, within 3SE. The no-call sheet: MC vs (11)
with COS terms. Small vol-of-vol, $\xi=10^{-4}$, $\rho=0$: the full product under Heston MC vs BS with QE's per-period discrete variance
(heston-mc eq. 15) on the same draws, difference within $3\,\mathrm{SE}_D$.\
**Model-gap gate (ROADMAP):** table of $V_{BS}$, $V_{TS}$, $V_H$, the three terms of (19) with CRN SEs, $\hat c_{BS}$ and $\hat c_H$ with (17) SEs, $\hat p_i$ and the
expected life. If the total gap comes out positive, first check (i) of §3.5 on the surface ($D^-_{n,H}(B_p)$ vs flat ATM) before suspecting code.

**Published values** (Deng–Mallett–McCann 2011, SLCG WP of 16 Aug 2011, §3 "Case 2"; journal version not read):

| inputs (mapped to this note) | value | status |
|---|---|---|
| $S_{\rm ref}=100$, $t_i=i/12$, $i\le12$; call at $S\ge102$ paying $100e^{0.092t_i}$ → $B_{ac}=B_c=1.02$, $m=1$, $c_j=e^{0.092t_j}-e^{0.092t_{j-1}}$; $L=80$ → $B_p=0.8$; $r=5\%$, $q=1\%$, $\sigma=20\%$, CDS $1\%$ → $F_0(t)=100e^{0.04t}$, $DF(t)=e^{-0.06t}$ | $V=98.39$ per 100 (explicit FD, 1000×500 grid) | read in the source; ⚑ 1 source; FD accuracy not stated → report in SE units, not a gate |
| same; first-call probabilities $p_1..p_{12}$ (their Table 1, defined on p. 7 as "not called before and called at $t_i$", although the caption says "conditional") | 0.3767, 0.1435, 0.0781, 0.0506, 0.0361, 0.0275, 0.0218, 0.0178, 0.0149, 0.0127, 0.0110, 0.0096 | $p_1=\Phi(d_-(e^{0.04/12},1.02,0.2/\sqrt{12}))=\Phi(-0.3141)=0.3767$ reproduced by hand; others ⚑ 1 source. Compare $p_{12}$ with $\overline{A_n\mathbf 1\{X_n\ge1.02\}}$, since $\chi_n=0$ here |

Their "Case 1" (no autocall, fixed 9.2 coupon at $T$) is a direct instance of (11). They set the coupon to make it par, rounded to 9.2%, so
expect about 100, not 100.00 exactly. No published Heston autocallable price was found; none is used.

## 6. Remarks

1. **What the investor sold.** An autocallable is a bond plus a strip of coupon digitals, financed by a 60% down-and-in put the investor
   sells to the bank, with automatic early redemption when the market is up. The investor is short vol and short the downside. The
   bank is long vega at inception. That vega grows as the market falls, peaks above the barrier, then flips as the put turns into an
   in-the-money digital near expiry. Dealers hedge by selling vol, so issuance is a steady supply of vol.
2. **Why a flat ATM vol is wrong.** The risk is not at the money. The put sits at 60% and the coupon barrier at 70–80%, where SPX implied
   vol is far above ATM. A flat vol makes crashes too unlikely, so it overvalues the note to the investor, undervalues the put the bank
   buys, and gives too low a fair coupon. The terminal piece is static (a put plus digitals at the barrier), so it is model-free from
   the smile. But a digital depends on the smile's slope, not only its level, so no single vol prices it.
3. **What the vanilla smile does not pin down.** Every coupon and the final put pay only on paths that were not called, i.e. that stayed
   below 100%. So the price depends on the smile the model generates at future dates given such a path: forward skew, driven by
   vol-of-vol and spot-vol correlation. Two models fitted to the same vanillas (local vol vs stochastic vol) disagree here, and that
   residual is model risk. The price is a sum of digitals: easy to compute, but its Greeks are unstable near the barriers.

## 7. References

- Deng, G., Mallett, J., McCann, C. (2011). Modeling autocallable structured products. *J. Derivatives & Hedge Funds* 17, 326–340.
  Read as the SLCG working paper of 16 Aug 2011 (eqs. 6–7, §3, Table 1).
- Langnau, A. (2010). A model-insensitive determination of first-hitting-time densities with application to equity default-swaps.
  arXiv:1002.2573 (eqs. 6, 8; §5 on spot vs forward skew of American digital puts).
- Breeden, D.T., Litzenberger, R.H. (1978). Prices of state-contingent claims implicit in option prices. *J. Business* 51(4), 621–651 (not re-read).
- Farkas, W., Ferrari, F., Ulrych, U. (2022). Pricing autocallables under local-stochastic volatility. Swiss Finance Institute Research
  Paper 22-71 (abstract only).
- Mazzon, A., Pascucci, A. The forward smile in stochastic local volatility models. Preprint (introduction read; listed as *J. Computational
  Finance*, not accessed).
- Schoutens, W., Simons, E., Tistaert, J. (2004). A perfect calibration! Now what? *Wilmott Magazine*, March (via Mazzon–Pascucci; not read).
- Fries, C.P., Joshi, M.S. (2011). Perturbation stable conditional analytic Monte-Carlo pricing scheme for auto-callable products.
  *Int. J. Theor. Appl. Finance* 14(2), 197–219 (abstract only).
- Bartholomew, H., Davis, C. (2020). Autocalls hit peak vega, where hedging costs mount. *Risk.net*, 16 March 2020 (lede only, paywalled).
- Conde, P. (2020). Peak vega: vol swings at 2008 levels make hedging a challenge. *StructuredRetailProducts.com*, 25 March 2020
  (interview with S. Carroll, UBS).
- Bouzoubaa, M., Osseiran, A. (2010). *Exotic Options and Hybrids*. Wiley, ch. 12 "Autocallable Structures" (table of contents only, not read).
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer (CRN, ch. 7 sensitivities, ratio estimators).
- Andersen, L. (2008). Simple and efficient simulation of the Heston stochastic volatility model. *J. Comput. Finance* 11(3) (via heston-mc.md).

## 8. Errata (Phase 4 implementation, 2026-11-08)

- **E1 (§1 barrier ordering vs §5 sub-case; D-026).** §1 states 0 ≤ B_p ≤ B_c ≤ B_ac, but the §5 / eq. (11) sub-case
  "B_c = 0 ⇒ V = Σ DF_i c_i + DF_n − DIP_E" has B_p = 0.6 > B_c. Eqs. (1)–(11) need only 0 ≤ B_c ≤ B_ac (on a call date
  X ≥ B_ac ≥ B_c, so a call pays its coupon) and B_p ≥ 0; the decomposition (6) also needs B_p ≤ 1. **Correction:** read
  B_p ≤ B_c as a term-sheet convention; `products.Autocall` validates 0 ≤ B_c ≤ B_ac and B_p ≥ 0, so the sub-case is
  admissible.
- **E2 (§1 extrapolation wording; D-025).** §1 says that outside [T_min, T_max] a constant zero rate and carry fitted
  on T > 0.3 are used. `market.forward_curve` implements this as flat extrapolation beyond T_max with the fitted r and
  b, anchored at the last kept expiry: ln DF(t) = ln DF(T_max) − r(t − T_max), ln F(t) = ln F(T_max) + b(t − T_max).
  This is continuous at T_max. Dates before T_min raise, because they would need S_0 (D-008).
- **E3 (eq. 19 label; D-031).** The first term V_TS − V_BS is labelled "ATM term structure". It compares the flat
  σ_ATM with Heston's expected-variance curve, so it holds a level part, √(w(T)/T) vs σ_ATM, and a shape part. When
  v0 = θ, w(t) = θt is flat and the term is a pure level gap. The code labels it "flat σ_ATM vs Heston expected-variance
  curve (level + term structure)".
