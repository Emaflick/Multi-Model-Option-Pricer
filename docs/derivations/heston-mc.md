# Heston Monte Carlo: Euler full truncation, Andersen QE / QE-M, MC estimator and 3SE gates

Phase 2 · kernels `src/qpricer/engines/mc.py` (parameters as in `models/heston.py`) · docstrings cite the eq. numbers
below. The characteristic function, COS and Lewis are in `heston.md` and not repeated here. ⚑ = fewer than two external
sources checked; (der.) = derived in this note; (est.) = hand estimate, neither published nor measured.
**[A06]** = Andersen's working paper, version of 12 Dec 2006. This is the version read; the 2008 journal version was not
accessed and its numbering may differ. QE-M is [A06] §4.3, Props. 7 and 9. **[LKD06]** = Lord–Koekkoek–van Dijk,
TI DP 2006-046/4, version of 17 May 2006 (2010 journal version not accessed).

## 1. Setup

- **Measure.** $T$-forward measure $\mathbb Q^T$ (numéraire $P(t,T)$) with **deterministic rates and carry**. Then $\mathbb Q^T$ and the
  risk-neutral measure give the same Heston dynamics for $F_t:=F(t,T)$ (to be recorded as a D-entry):
  $dF_t=\sqrt{v_t}F_t\,dW^F_t$, $\;dv_t=\kappa(\theta-v_t)dt+\xi\sqrt{v_t}\,dW^v_t$, $\;d\langle W^F,W^v\rangle_t=\rho\,dt$, $\;F_0=F$. Simulated
  state: $x_t=\ln(F_t/F)$ with $dx_t=-\tfrac12v_tdt+\sqrt{v_t}dW^F_t$, $x_0=0$. $F$ is a martingale: $\mathbb E^Te^{x_T}=1$.
- **Price.** $V_0=DF\cdot\mathbb E^T[g(Fe^{x_T})]$ for a payoff paid at $T$, with $DF=P(0,T)$. No $r,q$ inside kernels (D-001).
- **Inputs.** $F>0$, $DF\in(0,1]$, $T>0$ (ACT/365F), $v_0\ge0$, $\kappa,\theta,\xi>0$, $|\rho|<1$. Uniform grid $t_n=n\Delta$, $\Delta=T/M$,
  which must contain every observation date. $N$ paths. An explicit `np.random.Generator` is passed in. Every MC
  kernel returns `(price, stderr, n_paths, seed)`.
- **Notation.** $v_n,x_n$ = state at $t_n$; $y^+=\max(y,0)$; $E:=e^{-\kappa\Delta}$. Each step uses one draw $(Z_v,Z_\perp)$ iid $N(0,1)$
  per path (rows 0 and 1 of `rng.standard_normal((2, N))`). Feller ratio $\nu_F:=2\kappa\theta/\xi^2$; when $\nu_F\ge1$, $v$ never hits 0
  ([A06] Prop. 2). Symbol map: $\xi$ = Andersen's $\varepsilon$ = LKD's $\eta$ = Broadie–Kaya's $\sigma_v$ = Fang–Oosterlee's $\eta$, and
  FO's $(\lambda,\bar u,u_0)$ = our $(\kappa,\theta,v_0)$.
- **Bias sign.** bias $:=\mathbb E[\hat V_\Delta]-V_{\rm exact}$ (LKD convention; [A06] tabulates $e=-$bias).

## 2. Result

### 2.1 MC estimator and the 3SE gate
**(1)** $\hat V=DF\,\bar g$, $\bar g=\frac1N\sum_ig_i$; $\;\mathrm{SE}=DF\,s_g/\sqrt N$, $s_g^2=\frac1{N-1}\sum_i(g_i-\bar g)^2$ (`ddof=1`); 95% CI $\hat V\pm1.96\,\mathrm{SE}$.\
**(2)** Gate: $|\hat V-V_{\rm ref}|\le3\,\mathrm{SE}$, with reference error $\ll$ SE. For an unbiased scheme, P(false fail) $=2\Phi(-3)=0.27\%$
per comparison, and $1-0.9973^m$ over $m$ independent comparisons (5.3% at $m=20$, 24% at $m=100$). Two independent
MC estimates are compared with $\sqrt{\mathrm{SE}_1^2+\mathrm{SE}_2^2}$.\
**(3)** Power: a true bias $b$ is flagged with probability $\approx\Phi(|b|/\mathrm{SE}-3)$, i.e. 50% at $3$SE and 97.7% at $5$SE. The gate sees
the total error (discretisation bias + noise), so it bounds the bias only to a few SE.

### 2.2 Euler with full truncation (FT) ([LKD06] Table 1; [A06] eqs. 6–7)
**(4)** $Z_F=\rho Z_v+\sqrt{1-\rho^2}\,Z_\perp$.\
**(5)** $v_{n+1}=v_n+\kappa(\theta-v_n^+)\Delta+\xi\sqrt{v_n^+\Delta}\,Z_v$. The stored state $v_n$ may be negative; only $v_n^+$ enters.\
**(6)** $x_{n+1}=x_n-\tfrac12v_n^+\Delta+\sqrt{v_n^+\Delta}\,Z_F$, so that $\mathbb E[e^{x_{n+1}-x_n}\mid\mathcal F_{t_n}]=1$ **exactly, for every $\Delta$**.\
Empirically the bias is $\approx c\Delta$ (fitted order 0.95 in [LKD06] Table 3; order 1 in [A06] Fig. 5). No theorem covers $\nu_F<1$
(§3.3). Keep $\kappa\Delta<1$, the hypothesis of [LKD06] Lemma 1.

### 2.3 QE variance step ([A06] §3.2, $\psi_c=1.5$)
**(7)** $m=\theta+(v_n-\theta)E$, $\;s^2=\dfrac{v_n\xi^2E(1-E)}{\kappa}+\dfrac{\theta\xi^2(1-E)^2}{2\kappa}$, $\;\psi=s^2/m^2\in(0,1/\nu_F]$ (max at $v_n=0$; [A06] eq. 20).\
**(8)** $\psi\le\psi_c$: $\;b^2=\tfrac2\psi-1+\sqrt{\tfrac2\psi}\sqrt{\tfrac2\psi-1}\ge0$, $\;a=m/(1+b^2)$, $\;v_{n+1}=a\,(b+Z_v)^2$.\
**(9)** $\psi>\psi_c$: $\;p=\dfrac{\psi-1}{\psi+1}\in[0,1)$, $\;\beta=\dfrac{1-p}m=\dfrac2{m(\psi+1)}$, $U_v=\Phi(Z_v)$: $\;v_{n+1}=0$ if $U_v\le p$, else $\beta^{-1}\ln\dfrac{1-p}{1-U_v}$.\
Both branches match $m$ and $s^2$ exactly. The threshold must satisfy $\psi_c\in[1,2]$, because (8) needs $\psi\le2$ and (9) needs
$\psi\ge1$. If $\nu_F\ge1/\psi_c=2/3$, the exponential branch never fires.

### 2.4 QE log-forward step ([A06] eq. 33, central weights $\gamma_1=\gamma_2=\tfrac12$)
**(10)** $x_{n+1}=x_n+K_0+K_1v_n+K_2v_{n+1}+\sqrt{K_3v_n+K_4v_{n+1}}\;Z_\perp$, with $K_0=-\rho\kappa\theta\Delta/\xi$,
$K_1=\gamma_1\Delta(\tfrac{\kappa\rho}{\xi}-\tfrac12)-\tfrac\rho\xi$, $K_2=\gamma_2\Delta(\tfrac{\kappa\rho}{\xi}-\tfrac12)+\tfrac\rho\xi$, $K_3=\gamma_1\Delta(1-\rho^2)$, $K_4=\gamma_2\Delta(1-\rho^2)$.
$Z_\perp$ is independent of $Z_v$; correlation enters only through $K_2v_{n+1}$, and QE has no $Z_F$.

### 2.5 Martingale correction (QE-M) ([A06] Props. 7 and 9)
**(11)** $A=K_2+\tfrac12K_4=\tfrac\rho\xi(1+\kappa\gamma_2\Delta)-\tfrac12\gamma_2\Delta\rho^2$, $\;M(v_n)=\mathbb E[e^{Av_{n+1}}\mid v_n]$:
quadratic branch $\ln M=\dfrac{Ab^2a}{1-2Aa}-\tfrac12\ln(1-2Aa)$, which exists iff $A<\dfrac1{2a}$; exponential branch $M=p+\dfrac{\beta(1-p)}{\beta-A}$,
which exists iff $A<\beta$.\
**(12)** QE-M replaces $K_0$ in (10), per path and per step, by $K_0^*(v_n)=-\ln M(v_n)-(K_1+\tfrac12K_3)\,v_n$. This gives $\mathbb E[e^{x_{n+1}-x_n}\mid v_n]=1$.\
**(13)** $\rho\le0\Rightarrow A\le0$, so both conditions always hold (typical SPX fits). For $\rho>0$ they limit $\Delta$, roughly to $\rho\xi\Delta<2$
([A06] §4.3.2); if they fail, raise an error, never fall back silently.
**[A06] typos, do not copy:** the WP's eq. (42) prints $M=\beta(1-p)/(\beta-A)$, dropping the atom $p$ even though its own proof
line has "$p+$". Its $K_0^*$ display after Prop. 9 prints $-(K_1+\tfrac12\Delta\gamma_1)$, with no $v_n$. (11)–(12) match [A06] eq. (35),
QuantLib and Choi–Kwok (2024) §2.5. Sanity check: $A=0\Rightarrow M=1$.

### 2.6 Oracle formulas
**(14)** $\mathbb Ev_T=\theta+(v_0-\theta)e^{-\kappa T}$ and $\mathrm{Var}\,v_T=\dfrac{v_0\xi^2e^{-\kappa T}(1-e^{-\kappa T})}{\kappa}+\dfrac{\theta\xi^2(1-e^{-\kappa T})^2}{2\kappa}$. QE reproduces both exactly for any
$\Delta$ (der., §3.4). Euler-FT does not: its mean lies below the Euler-ODE mean $\theta+(v_0-\theta)(1-\kappa\Delta)^n$ ([LKD06] Lemma 1).\
**(15)** Limit $\xi\to0$: the price tends to Black-76 (bsm.md eq. 1) with $\sigma^2T=\bar\Sigma:=\theta T+(v_0-\theta)(1-e^{-\kappa T})/\kappa$. The discrete
limits the schemes reach as $\xi\to0$ (der.) are: Euler (any $\rho$) $\bar\Sigma_E=\theta T+(v_0-\theta)\big(1-(1-\kappa\Delta)^M\big)/\kappa$; QE with $\rho=0$
(exact ODE plus trapezoid) $\bar\Sigma_Q=\theta T+(v_0-\theta)\tfrac\Delta2(1+E)(1-e^{-\kappa T})/(1-E)$.\
**(16)** Plain-QE drift defect (der.): each step adds a deterministic log-drift error $\approx\tfrac\rho\xi(v_n-\theta)\tfrac{(\kappa\Delta)^3}{12}$. At fixed $\Delta$ it diverges
as $\xi\to0$ when $\rho\ne0$. For the FO set at $\xi=10^{-4}$ and $\Delta=1/32$ it adds up to $\approx2\%$ on $F_T$ (est.). QE-M removes it.\
**(17)** Path observations (forward pointer to Phase 4; deterministic carry): $S_t=F_0(t)\,e^{x_t}$, because $S_t/F_0(t)=F(t,T)/F_0(T)$.
$F_0(t)$ comes from `market.py`, and cash flows at $t$ are discounted with $DF(t)$.

### 2.7 Cross-check status (sources actually consulted)
| eq. | external sources | status |
|---|---|---|
| (1)–(3) | CLT; the 3-SD significance convention of [A06] Tables 2–4 | n/a |
| (4)–(6) | [A06] eqs. (5)–(7), fn. 7 ("Euler satisfies $\mathbb E[\hat X(t{+}\Delta)\mid\hat X(t)]=\hat X(t)$"); [LKD06] eqs. (9), (12), Table 1 + remark after (12); QuantLib `FullTruncation` | 2+ |
| (7) | [A06] Cor. 1, eqs. (17)–(20); QuantLib; $\chi'^2$ law [LKD06] eq. (4) + §3.4 | 2 |
| (8)–(9) | [A06] Props. 5–6, eqs. (25)–(30); QuantLib `QuadraticExponential`; re-derived §3.5 | 2 |
| (10) | [A06] eq. (33); QuantLib; Choi–Kwok (2024) §2.5 (equivalent $A_1,A_2$ form) | 2 |
| (11)–(13) | [A06] eqs. (35), (38), (40)–(43); QuantLib `QuadraticExponentialMartingale`; Choi–Kwok §2.5 | 2 (WP typos above) |
| (14) | mean: [A06] Cor. 1, [LKD06] §2.4; variance: [A06] Cor. 1; QE exactness: der. | mean 2 · var ⚑ 1 + der. |
| (15)–(17) | derivation only | ⚑ der. |

## 3. Derivation

**3.1 Estimator.** By the CLT, $\sqrt N(\bar g-\mathbb Eg)\Rightarrow N(0,\mathrm{Var}\,g)$, which needs $\mathrm{Var}\,g<\infty$. Puts are bounded. Calls need
$\mathbb EF_T^2<\infty$, which can fail in Heston (moment explosion; [A06] §4.3 citing Andersen–Piterbarg). $P(|Z|>3)=0.0027$, and
independence across $m$ tests gives (2). Under a bias $b$, $(\hat V-V_{\rm ref})/\mathrm{SE}\approx N(b/\mathrm{SE},1)$, which gives (3).

**3.2 Euler-FT.** Itô gives $d\ln F=-\tfrac12v\,dt+\sqrt v\,dW^F$. In (6), $v_n^+$ is $\mathcal F_{t_n}$-measurable and $Z_F\sim N(0,1)$ is independent of $\mathcal F_{t_n}$,
so $\mathbb E[e^{-\frac12v^+\Delta+\sqrt{v^+\Delta}Z_F}]=1$. This is exact; it would break if the coefficient used $v_{n+1}$ or an average.
*Why FT.* Suppose the scheme gives $v=-\delta<0$ while the true process sits at $0^+$. The true process is strongly reflecting:
zero vol at 0, and it leaves with drift $\kappa\theta$. FT copies this (zero effective variance, drift $\kappa\theta$). Absorption restarts at
0 and reflection jumps to $+\delta$; both add variance ([LKD06] Table 2). Lemma 1 orders the variance means as reflection >
absorption > Higham–Mao = partial truncation > FT. In LKD's test every fix over-prices the ATM call, and FT the least
([LKD06] Table 3, 20 steps/yr: bias 0.050 for FT, 2.102 for absorption, 4.360 for reflection).

**3.3 Weak order.** Weak order 1 needs smooth coefficients (Kloeden–Platen Thm 14.5.2, via [LKD06]), but $\sqrt v$ is not Lipschitz
at 0. [LKD06] prove strong convergence of FT and convergence of vanilla and barrier prices (Thms. 1–4), with no weak rate.
Empirically the order is ≈1 in both published Feller-violating tests. When $\nu_F\ll1$ the error concentrates near $v=0$, so
pre-asymptotic behaviour can be irregular. The gate therefore asks for a monotone decrease beyond noise; the fitted slope
is only a diagnostic.

**3.4 CIR law and moments.** $v_{t+\Delta}=c\,\chi'^2_d(\lambda)$ with $c=\xi^2(1-E)/(4\kappa)$, $d=4\kappa\theta/\xi^2$, $\lambda=v_tE/c$ ([A06] Prop. 1, [LKD06] eq. 4).
$\mathbb E\chi'^2=d+\lambda$ and $\mathrm{Var}\,\chi'^2=2(d+2\lambda)$ give (7); (14) is (7) with $\Delta=T$. At $v_t=0$, $\psi=\xi^2/(2\kappa\theta)$. *QE exactness:* $m(v)$ and
$m^2+s^2$ are polynomials of degree 1 and 2 in $v$ with exactly the CIR coefficients. The tower property and induction over
steps make $\mathbb E\hat v_T$ and $\mathbb E\hat v_T^2$ exact.

**3.5 QE moment matching.** (8): $a(b+Z)^2=a\,\chi'^2_1(b^2)$ has mean $a(1+b^2)$ and variance $2a^2(1+2b^2)$. With $x=b^2$,
$\psi(1+x)^2=2(1+2x)\iff x^2+2x(1-\tfrac2\psi)+1-\tfrac2\psi=0$. The discriminant $\tfrac2\psi(\tfrac2\psi-1)\ge0$ iff $\psi\le2$; take the non-negative root.
(9): the mixture $p\,\delta_0+(1-p)\mathrm{Exp}(\beta)$ has mean $(1-p)/\beta$, $\mathbb Ev^2=2(1-p)/\beta^2$ and variance $(1-p^2)/\beta^2$. So $\psi=(1+p)/(1-p)$,
which gives $p$ and needs $\psi\ge1$. Invert the cdf $\Psi(y)=p+(1-p)(1-e^{-\beta y})$.

**3.6 Log-forward.** Integrating $dv$ gives $\int\!\sqrt v\,dW^v=\xi^{-1}(v_{t+\Delta}-v_t-\kappa\theta\Delta+\kappa I)$ with $I=\int_t^{t+\Delta}v\,du$. With $W^F=\rho W^v+\sqrt{1-\rho^2}W^\perp$:
$x_{t+\Delta}-x_t=\tfrac\rho\xi(v_{t+\Delta}-v_t-\kappa\theta\Delta)+(\tfrac{\kappa\rho}\xi-\tfrac12)I+\sqrt{1-\rho^2}\int\!\sqrt v\,dW^\perp$ ([A06] eq. 11; Broadie–Kaya). Given the
$v$-path, the last term is $N(0,(1-\rho^2)I)$. Replacing $I\approx\Delta(\gamma_1v_t+\gamma_2v_{t+\Delta})$ gives (10). The naive alternative, Gaussian
log-Euler on QE variance with $\mathrm{corr}(Z_F,Z_v)=\rho$, leaks correlation because $v_{n+1}$ is non-linear in $Z_v$ ([A06] §4.1).

**3.7 Martingale correction.** Conditioning on $v_{n+1}$ and using the Gaussian MGF of $Z_\perp$:
$\mathbb E[e^{x_{n+1}-x_n}\mid v_n]=e^{K_0+(K_1+\frac12K_3)v_n}\,\mathbb E[e^{Av_{n+1}}\mid v_n]$. Setting this to 1 gives (12). $M$ follows from the $\chi'^2_1$ MGF
$\mathbb Ee^{u(b+Z)^2}=e^{ub^2/(1-2u)}/\sqrt{1-2u}$ with $u=Aa<\tfrac12$, and from $p+(1-p)\beta\int_0^\infty e^{(A-\beta)y}dy$ for the exponential branch.
For $\rho\le0$ every term of $A$ is $\le0$. *Why $\mathbb EF_T\ne F$ without it:* the continuous model satisfies
$\mathbb E\exp\{\tfrac\rho\xi(v'-v-\kappa\theta\Delta)+(\tfrac{\kappa\rho}\xi-\tfrac12)I+\tfrac12(1-\rho^2)I\}=1$ under the exact joint law of $(v',I)$. QE replaces $I$ by a trapezoid
and the law of $v'$ by a proxy that matches only two moments. An exponential moment at $A\approx\rho/\xi$ depends on the whole
distribution, so the identity fails at finite $\Delta$. The defect vanishes as $\Delta\to0$ (weak consistency, [A06] Prop. 10).

**3.8 Limits (der.).** When $\xi\to0$, $v$ becomes deterministic and $F_T$ is lognormal with total variance $\int v$, which gives Black-76.
Euler-ODE: $v_n-\theta=(v_0-\theta)(1-\kappa\Delta)^n$, and a left-point sum gives $\bar\Sigma_E$. QE returns $v_{n+1}=m$, so $v_n-\theta=(v_0-\theta)E^n$, and the
trapezoid gives $\bar\Sigma_Q$. For $\rho\ne0$, the deterministic part of (10) carries $\tfrac{\kappa\rho}\xi\times$(trapezoid error of $\int m$) $=$
$\tfrac\rho\xi(v_n-\theta)\big[E-1+\tfrac12\kappa\Delta(1+E)\big]=\tfrac\rho\xi(v_n-\theta)\tfrac{(\kappa\Delta)^3}{12}+O(\Delta^4)$, which is (16). (17): $F(t,T)=S_te^{\int_t^T(r-q)}$ and $F_0(t)=S_0e^{\int_0^t(r-q)}$.

## 4. Numerical pitfalls (symptom → fix)

1. **Negative variance (Euler).** `sqrt` of a negative gives NaN, or an ad-hoc fix gives a silent bias. Fix: FT (5)–(6). Keep the
   untruncated $v_n$ as the state; never absorb or reflect (both bias upward, §3.2).
2. **$\kappa\Delta\ge1$ (Euler).** The mean factor $(1-\kappa\Delta)^n$ oscillates. Fix: $\kappa\Delta<1$, preferably $\le\tfrac12$ (BK-1 has $\kappa=6.21$, so use $\Delta\le1/16$).
   QE uses the exact $E$ and has no such limit on $v$.
3. **Vectorised branching.** `np.where` evaluates both branches, so $\sqrt{2/\psi-1}$ is NaN for $\psi>2$ and $\ln(1-U)$ can be infinite. Fix:
   evaluate (8) on $\min(\psi,\psi_c)$ and (9) on $\max(\psi,\psi_c)$, or use boolean subsets; assert no NaN.
4. **$U_v\to1$ in (9).** `1 - ndtr(Z)` is exactly 0 for $Z\gtrsim8.3$, giving $v=\infty$. Fix: $1-U_v=$`ndtr(-Z_v)` (bsm.md pitfall 1). As $p\to0$
   ($\psi\to1^+$) the branch is benign, since $\beta\to1/m$ and $m\ge\theta(1-E)>0$.
5. **$\psi$ near $\psi_c$.** Both branches match two moments, so there is no bias jump. But a path can switch branch under a
   parameter bump, which makes bump-and-reprice Greeks noisy. Fix: common random numbers (same seed, same draw order). Always
   draw one $Z_v$ per variance step and use $U_v=\Phi(Z_v)$ (as QuantLib does), so streams stay aligned across branches.
6. **QE-M domain and overflow.** For $\rho>0$, $A\ge1/(2a)$ or $A\ge\beta$ makes $M=\infty$. Fix: check per path and raise. For small $\xi$,
   $|A|\sim|\rho|/\xi$ and $e^{Ab^2a/(1-2Aa)}$ underflows. Fix: compute $\ln M$ in log form (11), never `log(exp(.))`.
7. **Plain QE with small $\xi$ and $\rho\ne0$.** The forward drifts by (16). Fix: run the $\xi\to0$ test with $\rho=0$, or use QE-M.
8. **Correlation errors.** (i) Leakage: Euler-type $x$ on QE variance (§3.6) gives wings that are too thin. (ii) Double counting:
   a $Z$ in (10) correlated with $Z_v$. (iii) Wrong Cholesky row. Fix: (4) and (10) as written, with row 0 = $Z_v$ and row 1 = $Z_\perp$
   in both schemes. Check the sample $\mathrm{corr}(Z_F,Z_v)\approx\rho$, and check that $\rho<0$ gives a negative MC skew.
9. **Non-martingale scheme in a CV or parity.** Under plain QE, $\mathbb E\hat F_T\ne F$, so an $F_T$ control variate, or a call computed as
   put + $DF(F-K)$, imports or hides the defect. Fix: use either only with Euler-FT or QE-M.
10. **Heavy right tail.** At large $\xi$, long $T$ or $\rho>0$, the call SE is unstable and may be infinite (§3.1). Fix: price the OTM side and
    get ITM via parity (martingale scheme only). Report the SE and do not trust a single run.
11. **Reproducibility.** Global `np.random` state, or results that change with batch size. Fix: pass a `Generator`, use a fixed draw
    order (one `(2, N)` block per step), split batches with `SeedSequence.spawn`, and store the seed with the result.
12. **Memory.** Full paths take $N\times M\times8$ B per array, e.g. $2\cdot10^5\times320\approx0.5$ GB, and so do pre-drawn normals. Fix: keep only
    the current $(x,v)$ and draw per step. Accumulate running path functionals (min/max, sums, snapshots) only if the payoff needs them.
13. **Bias below the noise floor.** In the $\Delta$-figure the fine-$\Delta$ points are pure noise. Fix: plot ±3SE and a noise line, and do not
    claim a slope below it. For Euler, building coarse increments as $\sum Z^{\rm fine}/\sqrt k$ correlates the estimates across $\Delta$.

## 5. Test oracles

**Exact on the same paths (any scheme, $10^{-12}$):** $\hat C-\hat P=DF(\overline{F_T}-K)$; MC call prices on common paths are non-increasing and
convex in $K$. **Within 3SE:** $\overline{F_T}=F$ for Euler-FT and QE-M (exact martingales, (6), (12)). Report plain QE's $\overline{F_T}-F$ as a diagnostic
that should shrink with $\Delta$. For QE at any $\Delta$, check $\overline{v_T}$ and $\overline{v_T^2}$ against (14); for Euler-FT these are convergence diagnostics only.
Branch diagnostics: the share of exponential-branch draws and of negative Euler states.

**Limits:** $\xi=10^{-4}$, $\rho=0$: QE MC vs Black-76 at $\bar\Sigma_Q$ and Euler MC vs Black-76 at $\bar\Sigma_E$ (15), within 3SE. As $\Delta\to0$ both tend
to $\bar\Sigma$, the same target as the COS gate in heston.md.

**QE vs COS gate (2e5 paths, fixed seed, $\Delta=1/32$):** sets FO ($T=1$), BK-2 and A-I ($K\in\{70,100,140\}$), all Feller-violating. The
SPX-like FO set is the designated "Feller-violating set". Euler-FT is expected to **fail** a 3SE comparison on A-I: the published
bias at $\Delta=1/32$ is 0.243, against $3\,\mathrm{SE}\approx0.09$ at 2e5 paths ([A06]'s 0.014 at $10^6$ scaled by $\sqrt5$, est.). That is the scheme,
not a bug, so Euler enters through the bias figure only.

**Bias figure ("Euler bias shrinks with $\Delta$"):** plot MC − COS with ±3SE bars on log-log $|\cdot|$ axes, for $\Delta\in\{1,\tfrac12,\dots,\tfrac1{64}\}$,
Euler-FT/QE/QE-M, on A-I ($K=100$) and FO ($K=100$). Pass: the Euler bias exceeds 3SE at the coarsest $\Delta$ and decreases
monotonically until it is inside the noise band. Report the fitted slope (expected ≈1).

**Seeds and multiple testing:** fix and record the seed, so each run reproduces bit-for-bit. A failure is then a reproducible
event to diagnose; re-seeding until a test passes is loosening the tolerance and is forbidden. With $m$ 3SE gates, about
$m\times0.27\%$ fail with a correct implementation. A pass is statistical evidence, not proof.

**Published values** (all Feller-violating; spot-based sources mapped to $F=S_0e^{rT}$, $DF=e^{-rT}$, $q=0$):

| set | $\kappa,\theta,\xi,\rho,v_0$ | $T$ | $F$, $DF$ | $K$ | value | $\nu_F$ | source · status |
|---|---|---|---|---|---|---|---|
| FO | 1.5768, 0.0398, 0.5751, −0.5711, 0.0175 | 1 / 10 | 100, 1 | 100 | 5.785155450 / 22.318945791 | 0.38 | Fang–Oosterlee (2008) eq. (53), Tables 4–5 (their ref.: Carr–Madan, $N=2^{17}$) · ⚑ 1 + our COS |
| A-I | 0.5, 0.04, 1.0, −0.9, 0.04 | 10 | 100, 1 | 70/100/140 | $K{=}100$: 13.08467014 | 0.04 | params: [A06] Table 1 = Choi–Kwok (2024) Table 1 (2) · value: Choi–Kwok only ⚑; $K{=}70,140$: none found |
| A-II | 0.3, 0.04, 0.9, −0.5, 0.04 | 15 | 100, 1 | 70/100/140 | $K{=}100$: 16.64922292 | 0.030 | as A-I |
| A-III | 1.0, 0.09, 1.0, −0.3, 0.09 | 5 | 100, 1 | 70/100/140 | none found → COS | 0.18 | params: [A06] Table 1 only ⚑ |
| BK-1 | 6.21, 0.019, 0.61, −0.70, 0.010201 | 1 | $100e^{0.0319}$, $e^{-0.0319}$ | 100 | 6.8061 (6.80611331) | 0.63 | Broadie–Kaya (2006) Table 4; Choi–Kwok Table 1 (2) |
| BK-2 | 2.0, 0.09, 1.0, −0.3, 0.09 | 5 | $100e^{0.25}$, $e^{-0.25}$ | 100 | 34.9998 | 0.36 | Broadie–Kaya (2006) Table 5; [LKD06] Table 3 (2) |

**Published bias findings to reproduce qualitatively** (bias = MC − exact, SE in brackets):
- [A06] Table 2, A-I, $K=100$, $10^6$ paths: Euler-FT +6.394 (0.029) at $\Delta=1$, +1.051 (0.015) at 1/8, +0.243 (0.014) at 1/32. QE: +1.022
  (0.013) at $\Delta=1$, +0.049 (0.013) at 1/4, insignificant from 1/8. QE-M: +0.233 at $\Delta=1$, insignificant from 1/4. Text: QE at $\Delta=1$ or
  1/2 is about as accurate as Euler at 1/32 for ATM/OTM strikes; Euler converges with order 1; QE-M helps mostly ITM ($K=70$).
  Cost relative to Euler: QE 1.21×, QE-M 1.38× (Table 5).
- [LKD06] Table 3, BK-2 ATM, $10^8$ paths (SE ≈ 0.006): FT 0.050 / 0.025 / 0.013 / 0.007 at 20/40/80/160 steps per year (fitted
  order 0.95). Absorption gives 2.102 → 0.923 and reflection 4.360 → 1.771. The bias is upward, i.e. MC over-prices.

**Variance reduction (forward pointers, Phases 4 and 6):** *antithetic* $(Z_v,Z_\perp)\to(-Z_v,-Z_\perp)$, which also maps $U_v\to1-U_v$ in (9); compute
the SE over the $N/2$ pair means. *Control variates:* $F_T$ (known mean $F$, martingale schemes only), or a Black-76 price on the same
draws. *Conditional MC* (Willard 1997, used by Broadie–Kaya): given the $v$-path, $F_T$ is lognormal.

## 6. Interview angle

1. **The variance process is the hard part.** A square-root diffusion with realistic (Feller-violating) SPX parameters piles
   probability up near zero. Plain Euler then produces negative variances, and the choice of repair moves prices by whole
   vol points. Full truncation is least biased because it copies the true boundary behaviour: no vol while "below zero",
   and a pull back up. QE samples the variance from a law with the exact mean and variance: a squared Gaussian away from
   zero, and an atom at zero plus an exponential tail near zero.
2. **Leverage lives in the variance move, and the forward must stay driftless.** In QE the forward's shock is mostly the
   realised change in variance, scaled by correlation over vol-of-vol, plus independent noise; that keeps the skew even with
   large steps. Under the forward measure the only drift is the convexity term. Log-Euler keeps the forward an exact
   martingale; QE needs a small per-step correction, without which ITM prices, parity and a forward control variate drift.
3. **Two errors, one statistical gate.** Bias falls with the time step (about linearly for Euler) and noise falls with the
   square root of the path count. For an order-1 scheme, halving the step should come with four times the paths
   (Duffie–Glynn). A "within 3 SE" test only sees bias larger than the noise, and it fails about 1 time in 370 even for
   correct code. Fixed seeds make tests reproducible, many tests make some false alarms likely, and a pass is evidence,
   not proof.

## 7. References

- Andersen, L. (2008). Simple and efficient simulation of the Heston stochastic volatility model. *J. Computational Finance*
  11(3), 1–42. Read as the working paper "Efficient simulation of the Heston stochastic volatility model" (12 Dec 2006).
- Lord, R., Koekkoek, R., van Dijk, D. (2010). A comparison of biased simulation schemes for stochastic volatility models.
  *Quantitative Finance* 10(2), 177–194. Read as Tinbergen Institute DP 2006-046/4 (17 May 2006).
- Broadie, M., Kaya, Ö. (2006). Exact simulation of stochastic volatility and other affine jump diffusion processes.
  *Operations Research* 54(2), 217–231.
- Fang, F., Oosterlee, C.W. (2008). A novel pricing method for European options based on Fourier-cosine series expansions.
  *SIAM J. Sci. Comput.* 31(2), 826–848.
- Choi, J., Kwok, Y.K. (2024). Simulation schemes for the Heston model with Poisson conditioning. *European J. Operational
  Research* 314(1), 363–376 (arXiv:2301.02800).
- Heston, S.L. (1993). A closed-form solution for options with stochastic volatility with applications to bond and currency
  options. *Rev. Financial Studies* 6(2), 327–343.
- Andersen, L., Piterbarg, V. (2007). Moment explosions in stochastic volatility models. *Finance Stoch.* 11, 29–50 (not read).
- Kloeden, P.E., Platen, E. (1999). *Numerical Solution of Stochastic Differential Equations*. Springer (via [LKD06]).
- Duffie, D., Glynn, P. (1995). Efficient Monte Carlo simulation of security prices. *Ann. Appl. Probab.* 5(4), 897–905.
- Willard, G.A. (1997). Calculating prices and sensitivities for path-independent derivative securities in multifactor
  models. *J. Derivatives* 5(1), 45–61 (via Broadie–Kaya).
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer (antithetics, control variates).
- QuantLib `ql/processes/hestonprocess.cpp` (master, accessed 2026-11-06): FullTruncation, QuadraticExponential(Martingale).
