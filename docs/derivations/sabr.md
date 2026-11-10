# SABR (β = 1): Hagan lognormal implied vol, Monte Carlo check, calibration seeds

Phase 3 · kernels `src/qpricer/models/sabr.py` (implied vol, fed to Black-76, bsm.md eq. 1) and `src/qpricer/engines/mc.py`
(estimator, SE, 3SE gate as in heston-mc.md eqs. 1–3) · docstrings cite the eq. numbers below. ⚑ = fewer than two external
sources checked; (der.) = derived in this note; (est.) = hand estimate, neither published nor measured. **[H02]** = Hagan et al.
(2002), read as the preprint (its eq./page numbers). **[O08]** = Obłój, arXiv:0708.0998v1. **[LM07]** = Lions–Musiela (2007).
**[CHK]** = Choi–Hu–Kwok, arXiv:2408.01898v2.

## 1. Setup

- **Measure and dynamics.** $T$-forward measure $\mathbb Q^T$ (numéraire $P(t,T)$), $F_t=F(t,T)$, [H02] (2.15a–c):
  $dF_t=\alpha_tF_t^\beta dW_t$, $d\alpha_t=\nu\alpha_t\,dZ_t$, $d\langle W,Z\rangle_t=\rho\,dt$, $F_0=F$, $\alpha_0=\alpha$. The code fixes $\beta=1$:
  $d\ln F_t=-\tfrac12\alpha_t^2dt+\alpha_tdW_t$ and $\alpha_t=\alpha\,e^{\nu Z_t-\nu^2t/2}$ (exact).
- **Price.** $V=DF\cdot\mathbb E^T[(\theta(F_T-K))^+]$, quoted as $V=\mathrm{Black}_\theta(F,K,T,DF,\sigma_H(K))$ (bsm.md eq. 1). The SABR kernel returns
  $\sigma_H$ only and never sees $r,q$ (D-001).
- **Inputs.** $F,K>0$; $T>0$ (ACT/365F); $\alpha>0$ (for $\beta=1$ it is the instantaneous lognormal vol, annualised decimal); $\nu\ge0$
  (vol-of-vol, per $\sqrt{\text{yr}}$); $|\rho|<1$ (code bound $|\rho|\le0.999$).
- **Notation.** $k=\ln(K/F)$ (bsm.md's $x=-k$); $z=\tfrac\nu\alpha\ln(F/K)=-\nu k/\alpha$; $D(z)=\sqrt{1-2\rho z+z^2}$; $c_1=\tfrac{\rho\nu\alpha}4+\tfrac{(2-3\rho^2)\nu^2}{24}$ (1/yr);
  $\mathcal V$ = Black vega (bsm.md eq. 7); $\varepsilon=2^{-52}$. MC: grid $t_n=n\Delta$, $M=\lceil T/\Delta\rceil$, $N$ paths, explicit `np.random.Generator`.
- **Model facts, β = 1** ([LM07] §1, Prop. 2.2, §2.3, Thm. 2.3 + Remark (i)). $F$ is a true martingale iff $\rho\le0$; for $\rho>0$,
  $\mathbb EF_t<F$. For $m>1$: $\mathbb E F_T^m<\infty$ iff $\rho\le-\sqrt{(m-1)/m}$, and if $\rho>-\sqrt{(m-1)/m}$ it is $+\infty$ for all $t>0$. At $\rho=0$ every
  moment $m>1$ is infinite. Hence $\mathrm{Var}\,F_T<\infty$ iff $\rho\le-1/\sqrt2$ (consequence: pitfall 8).
- **β = 1 for SPX (text for the DECISIONS.md entry).** *Context:* β is not identifiable from one smile. β and ρ both make the
  skew: [H02] (3.1a) gives the ATM slope as $-\tfrac12(1-\beta)\sigma_{ATM}+\tfrac12\rho\nu$. [H02] §3.1, Fig. 3.3 says "β cannot be determined by
  fitting a market smile since this would clearly amount to 'fitting the noise'". The level $\sigma_{ATM}\approx\alpha F^{\beta-1}$ makes α trade off
  against β, so a free β gives an ill-conditioned (α, β, ρ) fit. *Decision:* β = 1. (i) Equity convention: positive index, Black (lognormal)
  quotes. With β = 1, $\sigma_H$ depends on $K/F$ only, so the model is scale-free (no length scale in $F$, [H02] p. 7). α is then directly a vol,
  the backbone is flat ([H02] p. 10) and all spot–vol co-movement sits in ρ. [H02] p. 12 calls β = 1 the usual a-priori choice of desks that
  prefer lognormal models or a horizontal backbone. (ii) Numerics: no boundary at $F=0$; log-Euler MC that is positive and an exact
  martingale; the ATM α-inversion is a quadratic (9); Hagan and Obłój coincide (6). *Alternatives:* β = 0.5; β from a log-log regression of
  historical $(F,\sigma_{ATM})$ ([H02] (3.3), noisy). *Consequences:* the model delta is sticky-moneyness (12); heavy right tail (facts above).

## 2. Result

### 2.1 Implied volatility
**(1)** General β [H02 (2.17a–c)], with $\ell=\ln(F/K)$ and $z=\tfrac\nu\alpha(FK)^{(1-\beta)/2}\ell$:
$\sigma_H=\dfrac{\alpha}{(FK)^{\frac{1-\beta}2}\big[1+\frac{(1-\beta)^2}{24}\ell^2+\frac{(1-\beta)^4}{1920}\ell^4\big]}\cdot\dfrac z{x(z)}\cdot\Big\{1+\Big[\dfrac{(1-\beta)^2\alpha^2}{24(FK)^{1-\beta}}+\dfrac{\rho\beta\nu\alpha}{4(FK)^{\frac{1-\beta}2}}+\dfrac{2-3\rho^2}{24}\nu^2\Big]T\Big\}$

**(2)** β = 1 (implemented) [H02 (B.71b–c)]: $\;\sigma_H(K)=\alpha\,\dfrac{z}{x(z)}\,(1+c_1T),\quad z=\dfrac\nu\alpha\ln\dfrac FK,\quad c_1=\dfrac{\rho\nu\alpha}4+\dfrac{2-3\rho^2}{24}\nu^2$.

**(3)** $x(z)=\ln\dfrac{D(z)+z-\rho}{1-\rho}=\displaystyle\int_0^z\frac{du}{D(u)}$, $\;x'(z)=1/D(z)$ ([H02] (B.36a)).

**(4)** ATM [H02 (2.18)]: $\sigma_{ATM}=\frac{\alpha}{F^{1-\beta}}\{1+[\frac{(1-\beta)^2\alpha^2}{24F^{2-2\beta}}+\frac{\rho\beta\alpha\nu}{4F^{1-\beta}}+\frac{2-3\rho^2}{24}\nu^2]T\}$; for β = 1, $\sigma_{ATM}=\alpha(1+c_1T)$ (the $K\to F$ limit of (2), since $z/x\to1$).

**(5)** Stable evaluation (der.). Take $D=\sqrt{(z-\rho)^2+(1-\rho)(1+\rho)}$. For $z\ge\rho$: $x=\ln\frac{D+z-\rho}{1-\rho}$. For $z<\rho$: $x=\ln\frac{1+\rho}{D+\rho-z}$, from
$(D+z-\rho)(D-z+\rho)=1-\rho^2$. In both branches every sum has non-negative terms. For $|z|<z_c=10^{-3}$ use
$\dfrac z{x(z)}=1-\dfrac{\rho z}2+\dfrac{(2-3\rho^2)z^2}{12}+\dfrac{\rho(5-6\rho^2)z^3}{24}+O(z^4)$, with $|z^4$-coefficient$|\le0.05$ for $|\rho|\le1$ (der., §3.5).
Error (est.): the direct log is $\lesssim4\varepsilon/|z|\approx10^{-12}$ at $z_c$, and the series is $\le0.05z_c^4=5\times10^{-14}$. So the jump at $z_c$ is $\lesssim10^{-12}$ relative.

**(6)** Obłój variant (not implemented; matters only for β < 1) [O08 Table, eq. (3)]: replace the leading factor by $\nu\ell/x(\zeta)$ with
$\zeta=\frac\nu\alpha\frac{F^{1-\beta}-K^{1-\beta}}{1-\beta}$, and keep Hagan's bracket ($I_B=I^0_B(1+I^1_H\tau)$). At β = 1, $\zeta=z$ and (6) ≡ (2). **Adopted: (2).** At β = 1
"Hagan vs Obłój" is not a choice. [O08] (3) prints $I^1_H$ without the $(FK)^{(\beta-1)/2}$ factors, which is immaterial at β = 1.

**(7)** $\nu\to0$: at $\nu=0$, $z=0$ and $c_1=0$, so $\sigma_H=\alpha$ exactly. For small ν, $\sigma_H-\alpha=\rho\nu\big(\tfrac k2+\tfrac{\alpha^2T}4\big)+O(\nu^2)$ (der.).

### 2.2 Shape near the money and Phase-5 calibration seeds (β = 1)
**(8)** $\sigma_H=(1+c_1T)\big[\alpha+\tfrac12\rho\nu k+\tfrac{(2-3\rho^2)\nu^2}{12\alpha}k^2\big]+O(k^3)$ ([H02] (3.1a) at β = 1). So the ATM skew is $\partial_k\sigma=\tfrac12\rho\nu(1+c_1T)$ and
the curvature is $\partial_k^2\sigma=\tfrac{(2-3\rho^2)\nu^2}{6\alpha}(1+c_1T)$. α sets the level, ρν the skew, ν²/α the smile.

**(9)** α from the market ATM vol (invert (4); [H02] p. 13 recommends parameterising by $\sigma_{ATM}$). Solve $a\alpha^2+b\alpha-\sigma_{ATM}=0$ with
$a=\rho\nu T/4$ and $b=1+(2-3\rho^2)\nu^2T/24$, giving $\alpha=2\sigma_{ATM}/(b+\sqrt{b^2+4a\sigma_{ATM}})$. This root form is stable as $a\to0$; if $b^2+4a\sigma_{ATM}\le0$, flag.

**(10)** Seeds (der.): fit $\sigma_{\rm mkt}(k)\approx\sigma_0+Sk+\tfrac12Ck^2$ on $|k|\le0.1$. Then $\alpha_0$ from (9), $\nu_0=\sqrt{\max(3\alpha_0C+6S^2,\nu_{\min}^2)}$, and
$\rho_0=\mathrm{clip}(2S/\nu_0,\pm0.95)$. This inverts (8) with $1+c_1T\approx1$.

**(11)** Bounds (TRF): $\alpha\in[10^{-4},5]$, $\rho\in[-0.999,0.999]$, $\nu\in[10^{-4},5]$. An active bound is logged as a diagnostic, never passed silently.
Expect $\rho<0$ for SPX (the martingale region). Expect ν large at short expiries and falling with T ([H02] p. 14 reports this for rates).

**(12)** Model delta at fixed α (sticky moneyness, der.): $\partial_FV=\Delta_F^{\rm Black}-\mathcal V\,\partial_k\sigma_H/F$, by the chain rule on $k=\ln(K/F)$.

### 2.3 Monte Carlo (β = 1)
**(13)** Exact vol step: $\alpha_{n+1}=\alpha_n\exp(\nu\sqrt\Delta Z_{\alpha,n}-\tfrac12\nu^2\Delta)$.\
**(14)** Log-Euler: $\ln F_{n+1}=\ln F_n-\tfrac12\alpha_n^2\Delta+\alpha_n\sqrt\Delta Z_{F,n}$, with $Z_F=\rho Z_\alpha+\sqrt{1-\rho^2}Z_\perp$. Draw `rng.standard_normal((2, N))` per step,
row 0 = $Z_\alpha$ and row 1 = $Z_\perp$ (as in heston-mc.md eq. 4). Then $\hat F>0$ always, $\mathbb E[\hat F_{n+1}|\mathcal F_{t_n}]=\hat F_n$ exactly for every Δ and ρ, $\hat\alpha$ is exact in law
on the grid, and the weak order is 1.\
**(15)** Mixing identity (exact). Since $\int_0^T\alpha\,dZ=(\alpha_T-\alpha)/\nu$:
$\ln F_T=\ln F+\tfrac\rho\nu(\alpha_T-\alpha)-\tfrac12I+\sqrt{(1-\rho^2)I}\,Z$, with $I=\int_0^T\alpha_t^2dt$ and $Z\perp\alpha$-path ([CHK] §2.3; for ρ = 0, Hull–White 1987).
Scheme (14) is the same expression with $(\hat I,\hat J)=(\Delta\sum\alpha_n^2,\sqrt\Delta\sum\alpha_nZ_{\alpha,n})$ in place of $(I,(\alpha_T-\alpha)/\nu)$.\
**(16)** Integrated-variance bias (der.). Left point: $\mathbb E\hat I/\mathbb EI=\frac{\nu^2\Delta}{e^{\nu^2\Delta}-1}=1-\tfrac12\nu^2\Delta+O(\Delta^2)$. Trapezoid $\Delta\sum\tfrac12(\alpha_n^2+\alpha_{n+1}^2)$:
$\tfrac{\nu^2\Delta}2\coth\tfrac{\nu^2\Delta}2=1+\tfrac1{12}\nu^4\Delta^2+O(\Delta^4)$. The leading ATM-vol bias of (14) at ρ = 0 is $\delta\sigma\approx-\alpha\nu^2\Delta/4$ (est.). For ρ ≠ 0,
$\hat J-J$ has mean zero, is $O(\sqrt\Delta)$ strong and adds an $O(\Delta)$ weak term, plausibly $O(|\rho|\nu\alpha^2\Delta)$ (est.; measure it, §5). **Do not** put the
trapezoid into the drift of (14) alone: $\mathbb E\,e^{-\frac14(\alpha_n^2+\alpha_{n+1}^2)\Delta+\alpha_n\sqrt\Delta Z_F}\ne1$ when $\rho\ne0$, so the martingale property breaks. Go to higher order through (17).\
**(17)** Conditional (mixing) estimator, an independent second MC (der. from (15); requires ν > 0):
$\hat V=DF\cdot\tfrac1N\sum_i\mathrm{Black}_\theta\big(F\xi_i,K,\sqrt{(1-\rho^2)\hat I_i/T}\big)$ with $\xi_i=\exp\{\tfrac\rho\nu(\alpha_{T,i}-\alpha)-\tfrac12\rho^2\hat I_i\}$ and $\hat I$ from the trapezoid rule.
The bias is $O(\Delta^2)$ ($\mathbb E\xi=1+O(\Delta^2)$), and the $Z_\perp$ noise is integrated out.\
**(18)** Step size (est.). ATM, small $s$: put-payoff sd $\approx0.584\,Fs$ and $\mathcal V\approx0.399F\sqrt T$, so $\mathrm{SE}_\sigma\approx1.46\,\sigma/\sqrt N$. The condition $|\delta\sigma|\le0.1\,\mathrm{SE}_\sigma$
gives $\Delta\le0.58/(\nu^2\sqrt N)$. Use $\Delta=\min(1/250,\,0.5/(\nu^2\sqrt N))$, then check by halving (§5). Gate set: α = 0.2, ν = 0.4, N = 2e5, Δ = 1/250 gives
$|\delta\sigma|\approx3\times10^{-5}$ (0.003 vol pt) against $\mathrm{SE}_\sigma\approx6.5\times10^{-4}$ (0.065 vol pt). *Erratum (validator Phase 3, F1):* this is an ATM
estimate. The wing bias from the ρ-term ($\hat J-J$) is larger: at Δ = 1/250 it is 0.0415 vol pt = 0.64 SE_gate at T = 0.25, k = −0.2. The gate
therefore uses **M = 250 steps at every maturity (Δ = T/250)**, where the measured Richardson bias is ≤ 0.17 SE_gate (D-023).\
**(19)** Comparison space: **price**. Gate: $|\hat P_{MC}-P_{\rm Black}(\sigma_H)|\le3\,\mathrm{SE}_P$, using **puts at every strike** (bounded payoff, pitfall 8). The Black
side is deterministic, so all the noise is in SE. Vol view (figure only): $\hat\sigma=$ IV($\hat P$) by bsm.md (11)–(15), with $\mathrm{SE}_\sigma\approx\mathrm{SE}_P/\mathcal V(\hat\sigma)$ (delta method).
The inversion bias is $\approx-\tfrac12(\text{volga}/\mathcal V)\mathrm{SE}_\sigma^2$, negligible at these SEs.

### 2.4 Validity of (2)
**(20)** Small parameter. [H02] App. B (B.1) rescales $\alpha\to\varepsilon\alpha$, $\nu\to\varepsilon\nu$ ("small volatility expansion", ε ≪ 1). It keeps $z/x(z)$ exact in $z$
and truncates at $O(\varepsilon^2T)$; (B.71b) is stated "through $O(\varepsilon^4)$". For β = 1, prices depend only on $(k,\alpha\sqrt T,\nu\sqrt T,\rho)$ (time change
$\tau=\nu^2t$, cf. Paulot §4: $\alpha\to\alpha/\nu$, $T\to\nu^2T$). So the expansion is in $\nu^2T$, $\alpha^2T$ and $\rho\nu\alpha T$. The first correction is $c_1T$; the neglected terms are
$O((\nu^2T)^2,\dots)$ relative. [H02] p. 32: the bracket is "usually … around 1%"; p. 9: dropping it can cost >3% "in extreme cases".\
**(21)** Wings (der.). As $|k|\to\infty$, $x\approx\pm\ln\frac{2|z|}{1\mp\rho}$, so $\sigma_H\approx(1+c_1T)\,\nu|k|/\ln\frac{2\nu|k|}{\alpha(1\mp\rho)}$ and $\sigma_H^2T/|k|\to\infty$. This violates
Lee's (2004) bound $\limsup\sigma^2T/|k|\le2$: (2) is butterfly-arbitrageable far enough out in both wings for every T, and the onset moves in as $\nu^2T$
grows.\
**(22)** $c_1<0$ iff $\rho<0$ and $\alpha>(2-3\rho^2)\nu/(6|\rho|)$, or $|\rho|>\sqrt{2/3}$ (where the ν² term is itself negative). Then $1+c_1T\le0$ for $T\ge1/|c_1|$, and (2)
returns a non-positive vol (der.).\
**Published:** [O08] p. 2 says the formula gives negative prices or densities "in region of small strikes for large maturities"
(Figs. 1–2: β = 0.4/0.6, T = 15/20). Grzelak–Oosterlee (2017) §1: the density is "not always free of arbitrage, especially not for very low
strikes". Paulot (2015) §4.6, Fig. 1 (β = 0.7, α = 0.3, ν = 0.4, ρ = −0.5, F₀ = 4): the HKLW relative vol error vs FDM, at the lowest strikes, reads
off the plots as ≈3%, 9%, 15% and 21% at T = 2.5, 5, 7.5 and 10 y ($\nu^2T$ = 1.6 at 10 y). No β = 1 accuracy study was found.

### 2.5 Cross-check status (sources actually consulted)
| eq. | external sources | status |
|---|---|---|
| (1) | [H02] (2.17a–c); QuantLib `unsafeSabrLogNormalVolatility` (same $(FK)^{1-\beta}$, /24, /1920 structure) | 2 |
| (2), (4) β=1 | [H02] (B.71b–c), (2.18); [O08] Table ($I^0|_{\beta=1}$) and eq. (3) ($I^1$ at β=1); QuantLib | 2+ |
| (3) | [H02] (2.17c), (B.36a); integral form: derivation | 2 |
| (5) to $z^2$ | QuantLib small-$z$ series $1-\rho z/2-(3\rho^2-2)z^2/12$; [H02] (3.1a) | 2 · $z^3$ term and branch split ⚑ der. |
| (6) | [O08] only | ⚑ 1 |
| (7)–(10), (12) | (8) = [H02] (3.1a); (9) = [H02] p. 13 + algebra; (7), (10), (12) derivation | ⚑ 1 + der. |
| (13)–(15) | exact lognormal α: [LM07] eq. (3), [CHK] §2; martingale step as heston-mc.md (6); mixing: [CHK] §2.3 + Itô | 2 |
| (16)–(18), (21), (22) | derivation; Lee (2004) for the bound only | ⚑ der. |
| moments (§1) | [LM07] Prop. 2.2, §2.3, Thm. 2.3; martingale iff ρ ≤ 0 also Jourdain (2004) via a secondary citation | martingale 2 · moments ⚑ 1 |

## 3. Derivation

**3.1 Leading term (heuristic version; matches [H02] (B.21a), (B.36a) and the exact short-time limit of Berestycki–Busca–Florent, per
[O08]).** Measure the distance to the strike in units of the current vol, $u_t=\ln(F_t/K)/\alpha_t$. By Itô, $du=dW-\nu u\,dZ+O(dt)$, so
$d\langle u\rangle/dt=1-2\rho\nu u+\nu^2u^2=D(\nu u)^2$. This is the only place ρ and ν enter at leading order; [H02] (B.25) notes that α itself only enters
at $O(\varepsilon)$. Over short times, the probability of travelling from $u_0$ to the strike ($u=0$) is $\sim e^{-d^2/2T}$, with distance
$d=\int_0^{u_0}du/D(\nu u)=x(z)/\nu$, where $z=\nu u_0$. In Black's model with vol σ, $u^B=\ln(F/K)/\sigma$ has unit diffusion, so $d_B=\ln(F/K)/\sigma$. Equating
$d=d_B$ gives $\sigma=\nu\ln(F/K)/x(z)=\alpha z/x(z)$, the leading factor of (2). As $K\to F$, $z/x\to1$.

**3.2 Correction $c_1$ (checks, not a full derivation; the full one is [H02] App. B).** *ρ = 0 part:* by (15), $V=\mathbb E[\mathrm{Black}(I)]$. ATM Black is
linear in $\sqrt I$ at leading order, so $\sigma_{ATM}\approx\mathbb E\sqrt{I/T}$. With $A=I/(\alpha^2T)=1+2\nu\bar Y+\nu^2(2\bar M-\tfrac T2)+O(\nu^3)$, $\bar Y=\tfrac1T\int Z$, $\bar M=\tfrac1T\int Z^2$,
we have $\mathbb E\bar Y^2=T/3$ and $\mathbb E\bar M=T/2$. Then $\mathbb E\sqrt A\approx1+\tfrac12\mathbb E(A-1)-\tfrac18\mathbb E(A-1)^2=1+\tfrac{\nu^2T}4-\tfrac{\nu^2T}6=1+\tfrac{2\nu^2T}{24}$, which is
$c_1$ at ρ = 0 (der.). *ρνα/4 (heuristic):* conditioning on $F_t=F$ forces $W_t\approx\alpha t/2$ (to offset the drift $-\alpha^2t/2$). Correlation then gives
$\mathbb E[\alpha_t^2|F_t=F]\approx\alpha^2(1+\rho\nu\alpha t)$. Time-averaging the ATM local variance gives $\alpha^2(1+\rho\nu\alpha T/2)$, and the square root gives
$\rho\nu\alpha T/4$. The $-\rho^2\nu^2/8$ piece needs the full expansion.

**3.3 Symmetry (der.).** Under the share measure $d\tilde{\mathbb Q}/d\mathbb Q^T=F_T/F$, $Y=1/F$ has $dY/Y=-\alpha\,d\tilde W$ and $d\alpha=\nu\alpha(d\tilde Z+\rho\alpha\,dt)$ ([LM07] (12)).
Since $C_F(K)=FK\,P_Y(1/K)$ and Black is homogeneous, $\sigma_F(k)=\sigma_Y(-k)$. $Y$ is SABR with correlation $-\rho$, **plus** an α-drift $\rho\nu\alpha^2$.
(a) ρ = 0: $\sigma(k)=\sigma(-k)$ exactly in the model, and (2) obeys it exactly ($z/x$ is even, $c_1$ has no odd part). (b) ρ ≠ 0: from
$x(-z;-\rho)=-x(z;\rho)$, the identity $\sigma_H(k;\rho)-\sigma_H(-k;-\rho)=\alpha\frac z{x(z)}\cdot\frac{\rho\nu\alpha}2T$ holds exactly. This matches the drift heuristic (α grows by
≈ $\rho\nu\alpha^2t$, so it averages $\rho\nu\alpha^2T/2$). So the exact ρ → −ρ, K ↔ F²/K symmetry does **not** hold for ρ ≠ 0, in the model or in (2).

**3.4 MC.** (13): $\ln\alpha$ is a Brownian motion with drift, so the step is exact. (14): $\alpha_n$ is $\mathcal F_{t_n}$-measurable and $Z_{F,n}$ is independent of it, so
$\mathbb Ee^{-\frac12\alpha_n^2\Delta+\alpha_n\sqrt\Delta Z_F}=1$. (15): $W=\rho Z+\sqrt{1-\rho^2}Z^\perp$, and given the α-path $\int\alpha\,dZ^\perp\sim N(0,I)$. (16): $\mathbb E\alpha_t^2=\alpha^2e^{\nu^2t}$, so the
left and trapezoid sums are geometric series. At ρ = 0, (14) equals (15) with $\hat I$, and the ATM price $\propto\sqrt{\mathbb E I}$ to leading order, giving
$\delta\sigma/\sigma\approx\tfrac12(-\tfrac12\nu^2\Delta)$. *SE (pitfall 8):* the CLT needs $\mathrm{Var}\,g<\infty$. Puts satisfy $0\le g\le K$. Calls need $\mathbb EF_T^2<\infty$, i.e. ρ ≤ −1/√2
([LM07] Thm. 2.3), and a reliable SE estimate needs $\mathbb EF_T^4<\infty$, i.e. ρ ≤ −√3/2. The scheme inherits the tail: at ρ = 0,
$\mathbb E[\hat F_T^2|\alpha]=F^2e^{\hat I}$ and $\mathbb Ee^{\hat I}=\infty$. Parity $\hat C=\hat P+DF(F-K)$ is unbiased **for ρ ≤ 0** because $\mathbb E\hat F_T=F$ exactly and
$\mathbb EF_T=F$ in the model. *Erratum (validator Phase 3, F4):* for ρ > 0, β = 1 SABR is a strict local martingale ([LM07]), $\mathbb EF_T<F$
(e.g. α = 0.3, ρ = 0.5, ν = 1, T = 5: $\mathbb EF_T/F-1=-0.167$, validator RQMC). Scheme (14) is still an exact martingale, so its calls and "parity with F"
target $P_{\rm model}+DF(F-K)$, not $C_{\rm model}$, while (17) prices the model call: the two estimators differ for ρ > 0 calls (puts agree), and
call- and put-implied vols differ in the model.

**3.5 Series (5).** The Legendre generating function $1/D(u)=\sum_nP_n(\rho)u^n$ integrates to $x=z+\tfrac\rho2z^2+\tfrac{3\rho^2-1}6z^3+\tfrac{5\rho^3-3\rho}8z^4+\dots$;
inverting the series gives (5). Checks: at ρ = 0, $z/\operatorname{asinh}z=1+z^2/6-17z^4/360$; at ρ = 1, $z/(-\ln(1-z))=1-\tfrac z2-\tfrac{z^2}{12}-\tfrac{z^3}{24}-\tfrac{19z^4}{720}$ (Gregory
coefficients). The $z^4$ coefficient is $-0.047$ at ρ = 0 and $-0.026$ at ρ = 1 (der.).

## 4. Numerical pitfalls (symptom → fix)

1. **$z/x(z)$ at $z\to0$** ($K=F$ or $\nu=0$). Symptom: 0/0 NaN; also $\ln(1+O(z))$ has relative error $\sim\varepsilon/|z|$. QuantLib switches at $z^2=10\varepsilon$,
   which leaves up to ~1e-8 relative noise just above the switch (est.). Fix: (5) with $z_c=10^{-3}$ and the cubic series. Evaluate the direct
   branch on `z_safe = where(|z|<z_c, 1, z)`, because `np.where` evaluates both branches.
2. **ρ → ±1 in $x(z)$.** Symptom: $(D+z-\rho)/(1-\rho)$ cancels to 0/0 when $z<\rho\approx1$, and $D+z-\rho\to0$ when $z<\rho\approx-1$; the result is inf/NaN or
   garbage wing vols. Fix: the sign-split (5), $D^2=(z-\rho)^2+(1-\rho)(1+\rho)$, and the bound $|\rho|\le0.999$.
3. **Non-positive vol** from $1+c_1T\le0$ (22). Fix: return NaN with a flag; never clip.
4. **Arbitrageable smile** in the wings and at long T (21). Symptom: negative second differences of $\mathrm{Black}(\sigma_H)$ in K
   (Breeden–Litzenberger density). Fix: never use (2) as a density source outside its validity region; the figure shades $p(K)<0$.
5. **Hagan treated as truth.** A gate at long T or high ν fails because of the expansion, not the code. Fix: gate only where $\nu^2T\lesssim0.2$ and
   $|k|\le0.2$; elsewhere it is a figure.
6. **Δ bias.** The left-point $\hat I$ is low by $\approx\tfrac12\nu^2\Delta$ (16), so MC smiles are too low by $\alpha\nu^2\Delta/4$. Fix: Δ from (18), plus the halving test.
7. **Trapezoid in the log-Euler drift only.** This breaks the martingale (16): parity and the $\hat F_T$ mean drift. Fix: keep (14), or use (17).
8. **Heavy right tail.** The call payoff has infinite variance for ρ > −1/√2 ([LM07]). Call SEs jump between seeds, the CLT fails, and the sample
   mean of $\hat F_T$ is noisy. Fix: price **puts** at every strike, get calls by parity with the exact $F$ (**valid only for ρ ≤ 0**; for ρ > 0 the
   model has $\mathbb EF_T<F$, §3.4 erratum), and run the martingale check at ρ = −0.9.
9. **Positivity.** Euler on $F$ itself ($F(1+\alpha\sqrt\Delta Z)$) or on α ($\alpha(1+\nu\sqrt\Delta Z)$) can go negative. Fix: log-Euler (14) and exact (13).
10. **(17) at ν → 0.** $(\rho/\nu)(\alpha_T-\alpha)$ is 0/0. Fix: for ν < 1e-8 use $\hat J=\sqrt\Delta\sum\alpha_nZ_{\alpha,n}$, or route ν = 0 to Black(α).
11. **Vol-space gate.** IV of a noisy MC price can fall outside the bounds (NaN in the wings), and the inversion is non-linear. Fix: gate in price
    space (19), and use vols with $\mathrm{SE}_P/\mathcal V$ bars only in the figure.
12. **Correlated tests.** All strikes on the same paths give strike-to-strike correlated errors, and m gates fail with probability $\approx1-0.9973^m$
    (heston-mc.md (2)). Fix: one fixed recorded seed; never re-seed to pass.
13. **Calibration conditioning.** At long T the bracket couples α with ρν; short SPX expiries push ν up. Fix: the σ_ATM parameterisation (9) inside the
    loss, seeds (10), bounds (11), and log active bounds.

## 5. Test oracles

**Exact / algebraic (double precision):** (i) **ν → 0 gate:** ν = 0 gives $\sigma_H=\alpha$ exactly (the series returns 1). ν = 1e-12 gives $|\sigma_H-\alpha|<10^{-10}$ for $|k|\le1$, $T\le10$,
since by (7) the error is $\approx10^{-12}(|k|/2+\alpha^2T/4)$. Slope check: $(\sigma_H-\alpha)/\nu$ at ν = 1e-7 equals $\rho(k/2+\alpha^2T/4)$ to rel. 1e-4.
(ii) **K = F continuity:** for $h\in\{10^{-10},10^{-8},10^{-6},10^{-4}\}$, $|\sigma_H(\pm h)-\sigma_{ATM}|\le|\partial_k\sigma|h+10^{-12}\sigma$. $(\sigma_H(h)-\sigma_H(-h))/2h$ at $h=10^{-4}$ equals
$\tfrac12\rho\nu(1+c_1T)$ to rel. 1e-6, and the second difference gives the curvature in (8). The jump across $|z|=z_c$ is $\le10^{-12}\sigma$. *Erratum (Phase 3, OPEN_ISSUES):* the bound above is first order only; at
$h=10^{-4}$ the −h side exceeds it by exactly $\tfrac12\partial_k^2\sigma\,h^2$ (2.06e-10 on set G). Use $|\partial_k\sigma|h+\tfrac12|\partial_k^2\sigma|h^2+10^{-12}\sigma$ (D-024).
(iii) **Symmetry:** ρ = 0 gives $\sigma_H(k)=\sigma_H(-k)$ to 1e-14 rel. For ρ ≠ 0 the identity of §3.3 holds to 1e-14 rel. A test asserting plain
$\sigma_H(k;\rho)=\sigma_H(-k;-\rho)$ must fail for ρ ≠ 0. (iv) (1) at β = 1 equals (2), and $|\sigma_{(1)}(\beta=1-10^{-6})-\sigma_{(2)}|=O(10^{-6})$.
(v) (2) has no F scale: $\sigma_H(\lambda F,\lambda K)=\sigma_H(F,K)$. (vi) Seeds: (9) reproduces α from $\sigma_{ATM}(\alpha)$ to 1e-14.

**MC (3SE, fixed seed, puts):** (a) ν = 0: MC = Black(α) at any Δ (exact in law). (b) $\overline{\alpha_T}=\alpha$, $\overline{\alpha_T^2}=\alpha^2e^{\nu^2T}$, and $\overline{\hat I}$ = the left-point
mean in (16). (c) $\overline{\hat F_T}=F$ at ρ = −0.9 (4th moment finite). (d) ρ = 0: $\hat\sigma(k)$ vs $\hat\sigma(-k)$ on independent seeds, within 3 combined SE
(a model identity, §3.3). (e) **Halving:** $|\hat V_\Delta-\hat V_{\Delta/2}|\le3$ SE of the difference, with common random numbers: the coarse increments are
$(Z_1+Z_2)/\sqrt2$ for both $Z_\alpha$ and $Z_\perp$, and $\alpha$ is rebuilt exactly. *Erratum (Phase 3, OPEN_ISSUES):* as a 3SE identity this holds only for (17). For the order-1
scheme (14), CRN makes the SE of the difference far smaller than the true O(Δ) bias, so correct code fails it. For (14) use a size check,
$|\hat V_\Delta-\hat V_{\Delta/2}|\le0.1\,\mathrm{SE_{gate}}$ at every gate maturity, plus an order check that $M(\hat V_{1/M}-\hat V_{1/2M})$ is flat in M (D-023). (f) (14) vs (17): two independent MCs agree within $3\sqrt{\mathrm{SE}_1^2+\mathrm{SE}_2^2}$.

**Gate "Hagan vs MC within 3SE":** set G = (F = 100, DF = 1, α = 0.2, ρ = −0.75, ν = 0.4), T ∈ {0.25, 0.5, 1},
k ∈ {0, ±0.05, ±0.1, ±0.2}, puts, N = 2e5, Δ = T/250 (M = 250 at every T; F1 erratum to (18)), scheme (14), with (17) reported
alongside. **Domain: set G, ν²T ≤ 0.16.** *Erratum (validator Phase 3, F2/F3):* the 21 comparisons share paths and carry small
deterministic offsets, mainly Hagan's own wing error (mean z ≈ −1 at T = 1, k = −0.2). So the iid figure 1 − 0.9973²¹ ≈ 5.5% does not apply.
Measured over 100 fresh seeds: 4.0% of seeds fail at Δ = T/250 (0.29% per comparison; `scripts/phase3_gates.py --seeds 100`). The validator
measured 9.0% at Δ = 1/250. Outside the domain the box fails because of the expansion, not the code: at ν = 1, T = 0.5, Hagan is +0.40 vol pt
off at k = −0.2 (5.4 SE). Pitfall 5 applies. Expected (est.): Hagan error ≈ 0.008 vol pt ATM at T = 1 (see the table: Hagan − FDM ≈ +3e-5 in price), against $\mathrm{SE}_\sigma\approx0.065$ vol pt.
Wing Hagan errors: measured by the validator at +0.058 vol pt at T = 1, k = −0.2 on set G (`docs/validation/phase3.md`); if only wings fail, rerun (17) at N = 1e6 to separate expansion error from code error before touching anything.

**Published values** ([CHK] Table 1, FDM benchmark, which [CHK] §5 attributes to Cai–Song–Chen (2017); β = 1, σ₀ = α = 0.2, T = 1,
F₀ = K = 1, European call; r is not stated, and r = 0 is consistent with Black at σ ≈ 0.199). Test: MC vs value within 3SE + 5e-6. Hagan's
value is a reported diagnostic, not a gate.

| ρ, ν | FDM value | Hagan (2)+(4)→Black, hand evaluation (est.) | status |
|---|---|---|---|
| −0.75, 0.2 | 0.07910 | 0.079101 | ⚑ 1 source; consistent |
| −0.50, 0.2 | 0.07942 | 0.079424 | ⚑ 1 source; consistent |
| −0.25, 0.2 | 0.07969 | 0.079698 | ⚑ 1 source; Hagan +1e-5 |
| −0.75, 0.4 | 0.07860 | 0.078630 | ⚑ 1 source; Hagan +3e-5 ≈ 0.008 vol pt |
| −0.75, **0.6** | 0.07811 | — | *Erratum (validator Phase 3, F5):* ν = 0.6 (the table, not the text). Two independent methods (RQMC, PDE) agree on 0.0781203, so the published value is off by ≈ +1.0e-5, about 2x its 5-decimal rounding ⚑ |

No second source for β = 1 reference values was found ([H02], [O08] and Paulot give figures only).

**Figure "where Hagan degrades"** (`scripts/make_figures.py`): α = 0.2, ρ = −0.75, ν ∈ {0.4, 1.0}, T ∈ {0.25, 1, 5, 10} ($\nu^2T$ from 0.04 to 10), 41
strikes on $k/(\alpha\sqrt T)\in[-3,2]$. Use the conditional MC (17) with N = 2e5 and Δ = 1/100 (trapezoid), and overlay (14) for T ≤ 1. Top row: $\sigma_H-\hat\sigma_{MC}$ in
vol pts, with ±3SE_σ bands and the gate box (T ≤ 1, |k| ≤ 0.2) marked. Bottom row: the Hagan density from second differences of $\mathrm{Black}(\sigma_H)$
(step $10^{-3}F$), with $p<0$ shaded. Write max |error| per panel to CSV. Expected qualitatively (from §2.4): error grows with $\nu^2T$ and $|k|$, and is
largest at low strikes.

## 6. Remarks

1. **What SABR's smile is.** It is Black with a random vol correlated with the forward. The implied vol at a strike is roughly the vol seen
   along the most likely route to that strike. With negative correlation, going down means vol goes up, which makes the skew, set by
   correlation times vol-of-vol. Vol-of-vol makes both wings dear, because big moves happen in high-vol states. That gives three almost
   orthogonal knobs per expiry: α sets the level, ρ the skew, ν the smile.
2. **β is a dynamics choice, not a fit parameter.** One smile cannot separate β from ρ. β decides how the ATM vol moves when the forward
   moves (the backbone), so it changes deltas, not today's prices. For SPX, β = 1 means lognormal and scale-free: the smile slides with
   moneyness, and all spot–vol correlation lives in ρ.
3. **Hagan is a short-time, near-the-money asymptotic.** It is excellent while vol and vol-of-vol are small over the option's life. It
   degrades with maturity and in the wings: negative densities at low strikes, and wings that grow faster than no-arbitrage allows. So it
   is checked against Monte Carlo, with errors in standard-error units. For β = 1 the MC is clean: the vol is simulated exactly, the forward
   stays positive and driftless, and the only bias is first order in the step, from the integrated variance. Compare on puts, because the
   β = 1 right tail can have infinite variance.

## 7. References

- Hagan, P.S., Kumar, D., Lesniewski, A.S., Woodward, D.E. (2002). Managing smile risk. *Wilmott Magazine*, July 2002, 84–108.
  Read as the preprint (eqs. 2.15–2.18, 3.1–3.3, App. B incl. B.1, B.21–B.36, B.59, B.69–B.71).
- Obłój, J. (2008). Fine-tune your smile: correction to Hagan et al. *Wilmott Magazine*, May 2008; arXiv:0708.0998 (v1 read).
- Berestycki, H., Busca, J., Florent, I. (2004). Computing the implied volatility in stochastic volatility models. *CPAM* 57(10), 1352–1373 (not read).
- Hagan, P.S., Kumar, D., Lesniewski, A.S., Woodward, D.E. (2014). Arbitrage-free SABR. *Wilmott* 2014(69), 60–75 (catalogue abstract only).
- Paulot, L. (2015). Asymptotic implied volatility at the second order with application to the SABR model. In *Large Deviations and
  Asymptotic Methods in Finance*, Springer; arXiv:0906.0658v3 (§4.6 and Figs. 1–2 read).
- Grzelak, L.A., Oosterlee, C.W. (2017). From arbitrage to arbitrage-free implied volatilities. *J. Computational Finance* 20(3), 31–49.
- Lions, P.-L., Musiela, M. (2007). Correlations and bounds for stochastic volatility models. *Ann. IHP (C)* 24(1), 1–16 (§1–2.4 read).
- Jourdain, B. (2004). Loss of martingality in asset price models with lognormal stochastic volatility. CERMICS report 2004-267 (not read).
- Lee, R.W. (2004). The moment formula for implied volatility at extreme strikes. *Mathematical Finance* 14(3), 469–480.
- Choi, J., Hu, L., Kwok, Y.K. (2025). Efficient and accurate simulation of the SABR model. *European J. Operational Research*,
  doi:10.1016/j.ejor.2025.09.027; arXiv:2408.01898v2 (§2, §5.1, Table 1 read).
- Cai, N., Song, Y., Chen, N. (2017). Exact simulation of the SABR model. *Operations Research* 65(4), 931–951 (not read; FDM source per [CHK]).
- Hull, J., White, A. (1987). The pricing of options on assets with stochastic volatilities. *J. Finance* 42(2), 281–300 (mixing, ρ = 0).
- Breeden, D., Litzenberger, R. (1978). Prices of state-contingent claims implicit in option prices. *J. Business* 51(4), 621–651.
- QuantLib `ql/termstructures/volatility/sabr.cpp` (master, accessed 2026-11-07): `unsafeSabrLogNormalVolatility`, `validateSabrParameters`.
