# Arithmetic Asian options, discrete fixings: fixing forwards, MC, geometric closed form, control variate

Phase 4 · payoff in `src/qpricer/products.py`, paths in `engines/mc.py`, geometric price in `engines/analytic.py`. Docstrings
cite the eq. numbers below. Barrier and autocallable have their own notes. The Heston path machinery (QE-M, grid, MC estimator,
3SE gate) is in `heston-mc.md` and is not repeated here. ⚑ = fewer than two external sources checked; (der.) = derived in this
note; (est.) = hand estimate, neither published nor measured. **[FM08]** = Fusai–Meucci (2008), read as the City Research
Online draft (file `…Fusai_Meucci_JBF.pdf`; its cover sheet wrongly carries a Fusai–Marazzina–Marena citation). The published
version was not accessed, so its numbering may differ.

## 1. Setup

- **Measure and rates.** Rates and carry are deterministic, as in heston-mc.md §1. The $T$-forward measure $\mathbb Q^T$, the
  $T_{\rm pay}$-forward measure and the risk-neutral measure therefore coincide for payoffs that depend only on $S$.
  $F(t,u)$ is the forward at time $t$ for delivery at $u$. The **fixing forwards** are $F_i:=F(0,t_i)$.
- **Product.** Fixing dates $0\le t_1<\dots<t_n\le T$ (ACT/365F from the valuation instant, D-003). $t_1=0$ is allowed: it is
  today's fixing. Weights are equal. $A=\frac1n\sum_{i=1}^nS_{t_i}$ and the payoff $(\theta(A-K))^+$ is paid at $T_{\rm pay}\ge t_n$
  (default $T_{\rm pay}=t_n=T$), with $\theta=+1$ call, $-1$ put. The geometric twin is $G=(\prod_iS_{t_i})^{1/n}$. Strike is fixed
  and no fixing lies in the past (seasoned deals: pitfall 12).
- **Kernel inputs.** The date vector $t$, the vector $F_{\rm fix}=(F_1,\dots,F_n)$, $DF=P(0,T_{\rm pay})$, $K$, $\theta$, and the model
  parameters ($\sigma$, or Heston $(v_0,\kappa,\theta,\xi,\rho)$). No $r$, $q$ or $S_0$ enters the kernel (D-001). A $t=0$ fixing enters
  as $F(0,0)=S_0$, supplied by the caller.
- **Fixing-forward builder** (outside the kernel; D-entry needed). (a) SPX: interpolate the parity forwards of `market.py`
  log-linearly in $t$. Fixings before the first listed expiry need $S_0$, which is informational only (D-008), so flag them.
  (b) Constant carry from one $(F,DF,T)$ plus $S_0$: eq. (2). (c) Synthetic tests: zero carry $F_i\equiv F$, labelled `synthetic`.
- **Notation.** $V(t)=\int_0^t\sigma^2(s)ds$ is the total variance to $t$ (BSM: $\sigma^2t$). $t_0:=0$, $\Delta V_k=V(t_k)-V(t_{k-1})$, $w_k=(n-k+1)/n$.
  $\bar A_F=\frac1n\sum F_i$, $\bar G_F=(\prod F_i)^{1/n}$, $\bar V=\frac1n\sum V(t_i)$. Per path: $Y=DF(\theta(A-K))^+$ and $X=DF(\theta(G-K))^+$.
  $c_B(F,K,s)$ is the undiscounted Black-76 call with total vol $s$ (bsm.md eq. 1 divided by $DF$).

## 2. Result

### 2.1 Fixing forwards and price
**(1)** $\;\mathbb E^T[S_{t_i}]=F(0,t_i)=F_i$ (deterministic rates and carry). Hence $\mathbb E^TA=\bar A_F$ and $V_A=DF\cdot\mathbb E^T[(\theta(A-K))^+]$.\
**(2)** Constant carry $b$: $F_i=F\,e^{-b(T-t_i)}$. With $b=\ln(F/S_0)/T$ this is $F_i=S_0(F/S_0)^{t_i/T}$; zero carry gives $F_i\equiv F$.\
*Consequence.* The price depends on the whole curve $F_1..F_n$, not only on $F(0,T)$; e.g. $\partial F_i/\partial b=-(T-t_i)F_i$. The MC and
the closed form share $F_{\rm fix}$, so no MC-vs-formula gate can detect a wrong curve. The builder needs its own test (§5).

### 2.2 Paths at the fixing dates
**(3)** BSM is exact on the dates for any spacing: $x_k=x_{k-1}-\tfrac12\Delta V_k+\sqrt{\Delta V_k}\,Z_k$, $x_0=0$, $S_{t_k}=F_ke^{x_k}$, with $Z_k$ iid
$N(0,1)$ and one $(N,)$ draw per date.\
**(4)** Heston: $x_t=\ln(F(t,T)/F(0,T))$ by QE-M (heston-mc eqs. 7–12) on a grid that contains every $t_i$. Split each $[t_{i-1},t_i]$ into
$m_i=\lceil(t_i-t_{i-1})/\Delta_{\max}\rceil$ equal steps and recompute the QE constants $K_0..K_4,E$ for each step size. Then $S_{t_i}=F_ie^{x_{t_i}}$ (heston-mc eq. 17).\
**(5)** Keep only running sums: $\Sigma_A\mathrel{+}=F_ke^{x_k}$ and $\Sigma_L\mathrel{+}=x_k$; then $A=\Sigma_A/n$ and $G=\bar G_F\exp(\Sigma_L/n)$. Memory is $O(N)$.\
**(6)** Estimator, SE and 3SE gate: heston-mc eqs. (1)–(3). Kernels return `(price, stderr, n_paths, seed)`. Greeks are out of Phase 4 scope.

### 2.3 Geometric Asian, discrete fixings, arbitrary dates
**(7)** $\;\mu_G:=\mathbb E\ln G=\ln\bar G_F-\tfrac12\bar V$\
**(8)** $\;\sigma_G^2:=\mathrm{Var}\ln G=\dfrac1{n^2}\sum_{i,j}V(\min(t_i,t_j))=\dfrac1{n^2}\sum_{i=1}^n(2n-2i+1)\,V(t_i)=\sum_{k=1}^nw_k^2\,\Delta V_k$\
**(9)** $\;F_G:=\mathbb EG=e^{\mu_G+\sigma_G^2/2}=\bar G_F\,\exp\!\big(-\tfrac12(\bar V-\sigma_G^2)\big)$, with $0\le\sigma_G^2\le\bar V$, so $F_G\le\bar G_F\le\bar A_F$. The factor
$e^{-\frac12(\bar V-\sigma_G^2)}$ is the convexity term: the forward of $G$ is not the geometric mean of the forwards.\
**(10)** $\;V_G=DF\,\theta\,[F_G\Phi(\theta d_1)-K\Phi(\theta d_2)]$, $\;d_{1,2}=\ln(F_G/K)/\sigma_G\pm\sigma_G/2$. This is bsm.md eq. (1) with $F\to F_G$ and
$\sigma\sqrt T\to\sigma_G$: call the Black kernel with $\sigma=\sigma_G$, $T=1$ (its theta is then meaningless). Parity: $C_G-P_G=DF(F_G-K)$.\
**(11)** Equally spaced, $t_0$ **excluded** ($t_i=iT/n$, $i=1..n$), flat $\sigma$: $\bar V=\sigma^2T\frac{n+1}{2n}$, $\;\sigma_G^2=\sigma^2T\dfrac{(n+1)(2n+1)}{6n^2}$.
Constant carry: $F_G=S_0\exp\!\big(bT\frac{n+1}{2n}-\sigma^2T\frac{n^2-1}{12n^2}\big)$; zero carry: $F_G=F\exp\!\big(-\sigma^2T\frac{n^2-1}{12n^2}\big)$.\
**(12)** Equally spaced, $t_0=0$ **included** ($n{+}1$ fixings $t_k=kT/n$, $k=0..n$, divisor $n{+}1$; the [FM08] convention):
$\sigma_G^2=\sigma^2T\dfrac{2n+1}{6(n+1)}$, $\;F_G=S_0\exp\!\big(\tfrac12bT-\sigma^2T\frac{n+2}{12(n+1)}\big)$. Versus (11), $\sigma_G$ is smaller by exactly the factor $n/(n+1)$.\
**(13)** Continuous limit $n\to\infty$ of both (Kemna–Vorst): $\sigma_G^2\to\sigma^2T/3$, $F_G\to S_0\exp(\tfrac12(b-\sigma^2/6)T)$. Rates:
$\sigma_G^2|_{(11)}-\tfrac{\sigma^2T}3=\sigma^2T\frac{3n+1}{6n^2}>0$ and $\sigma_G^2|_{(12)}-\tfrac{\sigma^2T}3=-\frac{\sigma^2T}{6(n+1)}<0$.

### 2.4 Control variate (BSM)
**(14)** $\;\hat V_{CV}=\frac1N\sum_ie_i$ with $e_i=Y_i-\hat c\,(X_i-V_G)$ and $\hat c=\dfrac{\sum(Y_i-\bar Y)(X_i-\bar X)}{\sum(X_i-\bar X)^2}$; $\;\mathrm{SE}=\mathrm{std}(e,\text{ddof}=1)/\sqrt N$.
With $c=1$: $e_i=Y_i-X_i+V_G$, which is exactly unbiased.\
**(15)** $c^*=\mathrm{Cov}(X,Y)/\mathrm{Var}X$ gives $\mathrm{Var}(Y-c^*X)=\mathrm{Var}Y(1-\rho^2)$, so $\;\mathrm{VR}=\mathrm{Var}Y/\mathrm{Var}(Y-c^*X)=1/(1-\rho^2)$. For example
$\rho=0.9,\,0.99,\,0.999$ give VR $=5.3,\,50,\,500$ (arithmetic). **Report** $\widehat{\rm VR}=s_Y^2/s_e^2$ from the same paths, with $\hat\rho$, $\hat c$, $N$, seed.
With OLS $\hat c$ and the same ddof, $\widehat{\rm VR}=1/(1-\hat\rho^2)$ exactly (test to $10^{-10}$). Efficiency is $\widehat{\rm VR}\times t_{\rm plain}/t_{CV}\approx\widehat{\rm VR}$,
because $G$ comes from the same paths.\
**(16)** Bias of fitting $\hat c$ on the same paths: $\mathbb E\hat V_{CV}-V_A=-\mathrm{Cov}(\hat c,\bar X)=O(1/N)$, against $\mathrm{SE}=O(N^{-1/2})$. If exact
unbiasedness is required, fit $\hat c$ on a pilot run ([FM08] §5.2: $10^5$ pilot paths, then $10^6$ fresh ones) or set $c=1$.\
**(17)** Heston: the BSM $V_G$ is **not** a valid mean; see §3.5. Recommended shadow control on the same grid and the same normals:
$x^B_{k+1}=x^B_k-\tfrac12\Delta V^B_k+\sqrt{\Delta V^B_k}\,(\rho Z_{v,k}+\sqrt{1-\rho^2}Z_{\perp,k})$, with $V^B(t)=\theta t+(v_0-\theta)(1-e^{-\kappa t})/\kappa$ and $S^B_{t_i}=F_ie^{x^B_{t_i}}$.
Then $X^B=DF(\theta(G^B-K))^+$ and $\mathbb EX^B=$ (10) evaluated with $V=V^B$ in (7)–(9).

### 2.5 Parity and bounds (model-free given $F_i$, pathwise)
**(18)** $\;C_A-P_A=DF(\bar A_F-K)$. On common paths, $\hat C_A-\hat P_A=DF(\overline{A}_{MC}-K)$ holds exactly.\
**(19)** Calls: $\max\{DF(\bar A_F-K)^+,\,C_G\}\le C_A\le\min\{C_G+DF(\bar A_F-F_G),\ \tfrac{DF}n\sum_ic_i\}$, with $c_i=\mathbb E^T(S_{t_i}-K)^+$
(BSM: $c_B(F_i,K,\sqrt{V(t_i)})$; Heston: COS with forward $F_i$ and expiry $t_i$). Puts: $P_G-DF(\bar A_F-F_G)\le P_A\le P_G$.

### 2.6 Cross-check status (sources actually consulted)
| eq. | external sources | status |
|---|---|---|
| (1),(2) | derivation §3.1; consistent with heston-mc eq. 17 (internal) | ⚑ der. |
| (3)–(6) | exact GBM solution; heston-mc eqs. 1–17 (2 sources each there) | as heston-mc |
| (7)–(10), arbitrary dates | derivation §3.3; [FM08] eq. (8), Gaussian case (equally spaced, $t_0$ incl.) | ⚑ 1 + der. |
| (11) | derived from (8); checked against $n=1$ and against (13) | ⚑ der. |
| (12) | [FM08] eqs. (6),(8) Gaussian; [FM08] Table 4 reproduced by hand to $\le1.3\cdot10^{-4}$ (§5) | ⚑ 1 + numeric |
| (13) | Wikipedia "Asian option" ($\sigma_G=\sigma/\sqrt3$, $b_G=\frac12(r-\frac12\sigma_G^2)$); [FM08] §3.4 (mean $\ln S_0+m\tau/2$, var. $\sigma^2\tau/3$) | 2 |
| (14)–(16) | algebra; [FM08] §5.2 (pilot $\hat\beta$); Glasserman (2004) §4.1 (cited, not re-read) | ⚑ 1 + algebra |
| (17) | derivation §3.5; existence of Heston geometric formulas: Kim et al. (2016), Custers et al. (2026) (abstracts only) | ⚑ der. |
| (18),(19) | pathwise derivation; "geometric call is a lower bound for the arithmetic call" (secondary, via KV citations) | ⚑ der. |

## 3. Derivation

**3.1 Fixing forwards.** With deterministic carry, $F(t,T)=S_te^{\int_t^T(r-q)}$ and $F(0,t)=S_0e^{\int_0^t(r-q)}$. Hence $S_t=F(t,T)\,F(0,t)/F(0,T)$.
$F(\cdot,T)$ is a $\mathbb Q^T$-martingale, so $\mathbb E^TS_t=F(0,t)$, which is (1). For payment at $T_{\rm pay}$, $V=P(0,T_{\rm pay})\mathbb E^{T_{\rm pay}}[\cdot]$. The
density $d\mathbb Q^{T_{\rm pay}}/d\mathbb Q=1/(P(0,T_{\rm pay})B_{T_{\rm pay}})$ is deterministic and therefore equal to 1. With stochastic rates correlated with
$S$, $\mathbb E^{T_{\rm pay}}S_{t_i}\ne F(0,t_i)$ (a timing/convexity adjustment, out of scope). Constant carry makes $\ln F(0,t)$ linear in $t$, which gives (2).

**3.2 Fixing law.** Take deterministic $\sigma(t)$ under $\mathbb Q^T$: $F(t,T)=F(0,T)\exp(M_t-\frac12V(t))$, with $M_t=\int_0^t\sigma\,dW$ and
$\mathrm{Cov}(M_s,M_t)=V(s\wedge t)$. By §3.1, $\ln S_{t_i}=\ln F_i-\frac12V(t_i)+M_{t_i}$. The increments of $M$ are independent Gaussians, so (3) is exact.

**3.3 Geometric average.** $\ln G=\frac1n\sum_i\ln S_{t_i}$ is linear in a Gaussian vector, so $G$ is lognormal. Its mean is (7). Its variance is
$n^{-2}\sum_{i,j}V(t_{i\wedge j})$. For sorted dates, $\#\{(i,j):i\wedge j=k\}=2(n-k)+1$, which gives the second form of (8). For the third form,
Abel summation gives $\sum_iM_{t_i}=\sum_k(n-k+1)\Delta M_k$ with independent $\Delta M_k$. $F_G$ is the lognormal mean. The bound $\sigma_G^2\le\bar V$ holds
because the variance of an average is at most the average of the variances (Cauchy–Schwarz), and $\bar G_F\le\bar A_F$ is AM–GM applied to the
forwards. The derivation in bsm.md §3.1 uses only $\ln F_T\sim N(\ln F-\frac{s^2}2,s^2)$; here $\ln G\sim N(\ln F_G-\frac{\sigma_G^2}2,\sigma_G^2)$, which gives (10).
*(11):* $\sum i=\frac{n(n+1)}2$ and $\sum(2n-2i+1)\,i=\frac{n(n+1)(2n+1)}6$, so $-\frac12(\bar V-\sigma_G^2)=-\sigma^2T\frac{(n+1)(n-1)}{12n^2}$. Check: $n=1$ gives $\sigma^2T$ and $F$.
*(12):* the $t_0$ fixing adds $\ln S_0$ with zero variance and the divisor becomes $n+1$. So $\sigma_G^2=\sigma^2\frac Tn\frac{n(n+1)(2n+1)}{6(n+1)^2}$, and the mean of the $t_k$ is $T/2$.
*Discrete vs Kemna–Vorst.* KV (1990) give the closed form for continuous geometric averaging and use the geometric option as a
control (per [FM08] §1 and secondary sources; KV was not read). The argument is the same in discrete time ("the log of a geometric
average of GBM is Gaussian"). [FM08] eq. (8) in the Gaussian case gives (12) and states "we obtain the result of Kemna and Vorst".

**3.4 Control variate.** $\mathrm{Var}(Y-cX)=\mathrm{Var}Y-2c\,\mathrm{Cov}+c^2\mathrm{Var}X$ is minimised at $c^*$, which gives (15). For the bias:
$\hat V_{CV}-V_A=(\bar Y-\mathbb EY)-\hat c(\bar X-\mu_X)$ and $\mathbb E[\hat c(\bar X-\mu_X)]=\mathrm{Cov}(\hat c,\bar X)$. Both $\hat c-c^*$ and $\bar X-\mu_X$ are $O_p(N^{-1/2})$, which gives (16).
*Why $\rho$ is close to 1.* With $y_i=\ln S_{t_i}$, $A/G=\frac1n\sum e^{y_i-\bar y}=1+\frac1{2n}\sum(y_i-\bar y)^2+\dots$. So $A-G$ is **second order** in the
dispersion of the fixings, $O(\sigma^2T)\,F$, while the payoffs fluctuate at first order, $O(\sigma\sqrt T)\,F$. Near the money and at moderate $\sigma\sqrt T$,
the two payoffs move almost one-for-one: $c^*\approx1$ and $\rho\to1$. Deep OTM, both payoffs are mostly zero and the exercise sets
$\{A>K\}\supseteq\{G>K\}$ differ in relative terms, so $\rho$ falls (est.; measure and report it).

**3.5 Control variates under Heston.** A CV needs $\mathbb EX$ **under the law that generated** $X$. On Heston paths $G^H$ is not lognormal,
and $\mathbb E^H(G^H-K)^+\ne V_G(\sigma)$ in general. Plugging $V_G$ in shifts $\hat V_{CV}$ by $c(\mathbb E^HX-V_G)$. That bias does not shrink with $N$ and comes
with a small SE: a confident wrong number. The valid options are:
- **(H1) Plain MC.** Always valid.
- **(H2) Shadow BSM on the same normals, eq. (17).** $Z^B=\rho Z_v+\sqrt{1-\rho^2}Z_\perp$ is $N(0,1)$ and independent across steps, because
  $(Z_v,Z_\perp)$ are iid $N(0,1)$ and $\rho^2+(1-\rho^2)=1$. So $x^B$ is an exact BSM path with variance $V^B$, and $\mathbb EX^B$ is (10) **exactly**, on
  any grid and whatever the bias of the Heston scheme. It correlates with the Heston payoff through the shared draws. Fit $\hat c$ by
  regression ($c^*\ne1$ in general). $V^B$ is Heston's expected integrated variance, chosen only to match the vol level; any $V^B$ is valid.
- **(H3) Controls with a known Heston mean.** Examples: $A$ itself ($\mathbb EA=\bar A_F$, exact under QE-M), $S_{t_n}$, or a vanilla on $S_{t_n}$ priced
  by COS. Caveat: $X$ is computed on the discretised scheme, so $\mathbb E_\Delta X\ne\mathbb EX$ except for martingale quantities. A COS-mean control
  adds $-c(\mathbb E_\Delta X-\mathbb EX)$ to the bias (heston-mc pitfall 9).
- **(H4) Heston geometric Asian by Fourier.** A recursive affine CF exists (Kim–Kim–Kim–Wee 2016; Volterra–Heston: Custers–Karbach–Friesen
  2026). It gives the best correlation but needs the most code, and it has the same discretisation caveat as H3.

**Recommendation.** Run the Phase 4 CV gate and the VR report under **BSM only** (exact simulation, exact mean). Under Heston use **H1**
by default. If an SE target forces variance reduction, use **H2**: about 10 lines, it reuses (10), and it is unbiased with $c=1$ or a pilot
$\hat c$. A CV reduces noise; it never removes the bias of the scheme.

## 4. Numerical pitfalls (symptom → fix)

1. **Continuous formula for discrete fixings.** Symptom: geometric MC fails 3SE. At $n=12$ ($t_0$ incl.) the continuous price exceeds the
   discrete one by 0.087907 at $K=100$ and 0.064243 at $K=110$ ([FM08] Table 4: 4.791541 vs 4.703634; 1.315486 vs 1.251243). Fix:
   use (7)–(10) on the actual dates, and use (13) only as a limit test.
2. **$t_0$ included or not; divisor $n$ or $n+1$.** The literature differs: [FM08] eq. (22) averages $k=0..T$ with divisor $T+1$, while
   Pirjol–Zhu (2017) eqs. (1)–(2) average $i=1..n$ with $t_i=i\tau$. Symptom: $\sigma_G$ is off by the factor $(n+1)/n$ (8.3% at $n=12$) and
   published values are not reproduced. Fix: the product stores the explicit date vector, and a $t=0$ fixing is an ordinary entry
   with $F(0,0)=S_0$ and $\Delta V=0$. Test (11) and (12) separately.
3. **Wrong fixing forwards.** Symptom: nothing fails, because MC and (10) share $F_{\rm fix}$, and the price is silently off. Typical
   errors: $F(0,T)$ or spot used for every fixing; interpolating $F$ instead of $\ln F$. Fix: test the builder on its own ($n=1$ at $t_1<T$
   vs Black-76 on $F(0,t_1)$), and log the SPX interpolation rule as a D-entry.
4. **Grid misses the fixing dates (Heston).** Symptom: a bias that does not shrink with $N$ and moves with $\Delta$. Fix: (4); the fixing
   dates are grid nodes, with per-step QE constants when step sizes differ.
5. **Non-martingale scheme.** Under plain QE, $\mathbb EA\ne\bar A_F$, so the closed-form parity (18) fails and an $A$-control is biased. Fix: QE-M.
6. **Overflow of $\prod S$.** With $n=250$ and $S\approx100$ the product is $10^{500}>$ DBL_MAX ($\approx1.8\cdot10^{308}$): `inf` payoffs, then `nan` in the
   CV regression. Fix: use the log-sum (5) and never form the product.
7. **Memory.** Storing paths costs $n\times N\times8$ B: 2 GB per array at $n=250$, $N=10^6$, and as much again for pre-drawn normals. Fix:
   (5). Loop over dates (vectorised over paths), draw per date or step, keep $\Sigma_A,\Sigma_L$, and evaluate all strikes at the end on $(N,)$ arrays.
8. **$\hat c$ bookkeeping.** Symptom: SE quoted from $s_Y$ (no gain shows), or VR computed with $\hat c$ from one seed and variances from
   another. Fix: compute (14)–(16) on the same paths and report $\hat c,\hat\rho,\widehat{\rm VR}$. The $O(1/N)$ bias is negligible; use a pilot run or $c=1$ if not.
9. **Wrong CV mean (Heston simulation with BSM $V_G$).** Symptom: a tight SE, but many SE away from plain MC at large $N$. Fix: §3.5.
10. **AM–GM in floating point.** $G=\exp(\text{mean log})$ can exceed $A$ by a few ulps when the fixings are nearly equal ($n=1$, $\sigma^2\Delta t\to0$).
    Symptom: a strict pathwise test $Y_i\ge X_i$ fails at the $10^{-16}$ level. Fix: compare at relative $10^{-12}$; never clip.
11. **$\sigma_G=0$** ($\sigma=0$, or all fixings at $t=0$). (10) is $0/0$ at $F_G=K$. Fix: use the $s=0$ limits of D-012.
12. **Seasoned Asians** (past fixings). With $m$ of $n$ fixings still to come, $(A-K)^+=\frac mn(A_{\rm fut}-K')^+$ with $K'=(nK-\Sigma_{\rm past})/m$. If
    $K'\le0$ the call is a forward, and Black with $K'\le0$ fails on the log. Out of Phase 4 scope: reject at the API.
13. **[FM08] draft eq. (17)** gives $c=\frac{(2+T)(3+2T)}{6(1+T)}$, which is one term more than their eq. (8) (der.). Table 4 is consistent with (8)/(12) (§5), so do not use (17).

## 5. Test oracles

**Exact** (rel. $10^{-12}$ unless stated):
- (10) at $n=1$, $t_1=T$ equals bsm.md eq. (1); at $n=1$, $t_1<T$ it equals Black-76 with $(F_1,\sqrt{V(t_1)},DF)$. All $t_i=t$ give $\sigma_G^2=V(t)$, $F_G=F(0,t)$.
- The three forms of (8) agree, and (11), (12) agree with (7)–(9) on equally spaced dates. $n(\sigma_G^2-\frac{\sigma^2T}3)\to\frac{\sigma^2T}2$ (11) and $\to-\frac{\sigma^2T}6$ (12).
- $\sigma\to0$ gives $V_G\to DF(\theta(\bar G_F-K))^+$. Geometric parity (10) holds.
- **Common MC paths:** $Y_i\ge X_i$ for calls (rel. $10^{-12}$); parity (18) and the pathwise forms of (19); calls non-increasing and convex in $K$;
  $\widehat{\rm VR}=1/(1-\hat\rho^2)$ to $10^{-10}$; at $n=1$, $t_1=T$, the arithmetic MC equals the vanilla MC on the same draws.

**Statistical** (3SE, heston-mc eq. 2; fixed, recorded seeds):
- **Gate.** Plain MC of $X$ under BSM vs (10), with $2\cdot10^5$ paths. Suggested grid: $\sigma\in\{0.2,0.5\}$, $T\in\{1,3\}$, $n\in\{1,12,52\}$,
  $K/F\in\{0.8,1,1.2\}$, $F_i\equiv100$, $DF=1$, plus the [FM08] forward curve. Simulation is exact at the dates, so persistent failures are bugs.
- $\overline{A}_{MC}$ vs $\bar A_F$ and $\overline{G}_{MC}$ vs $F_G$. CV vs plain arithmetic on **independent** seeds, within $3\sqrt{\mathrm{SE}_1^2+\mathrm{SE}_2^2}$.
- **Heston.** (i) $\xi=10^{-4}$, $\rho=0$, uniform grid with the dates on nodes: QE-M geometric MC vs (10) with $V(t_i)=\bar\Sigma_Q(t_i)$
  (heston-mc eq. 15 with $T\to t_i$; at $\rho=0$, QE-M reduces to QE and the first correction is $O(\xi^2)$). (ii) Full parameters: $\overline A_{MC}=\bar A_F$
  (QE-M) and $\hat C_A\le\frac{DF}n\sum_i$COS$(F_i,K,t_i)+3$SE. (iii) H2 vs H1 on independent seeds.

**Published values.** Source: [FM08], Gaussian model.
- **Inputs:** $S_0=100$, $r=0.0367$, $\sigma=0.17801$, $T=1$. $T=1$ is stated for Table 4 (and in §5.3); the Table 6 caption omits it, and
  $T=1$ is assumed there. $q=0$ is not stated anywhere; it is assumed, and the hand check below supports it.
- **Fixings:** $t_k=k/n$, $k=0..n$ ($t_0$ included, divisor $n+1$; [FM08] eqs. (6), (22)). **Forward form:** $F_k=100e^{0.0367t_k}$, $DF=e^{-0.0367}$.
- **Methods:** Table 4 is geometric by Carr–Madan FFT. Table 6 is arithmetic by recursive quadrature with 10000 points.

| $n$ | type | $K=90$ | $K=100$ | $K=110$ | status |
|---|---|---|---|---|---|
| 12 | geometric | 11.66744 | 4.703634 | 1.251243 | verified in [FM08] Table 4 (draft) · ⚑ 1 |
| 50 | geometric | 11.7076 | 4.769273 | 1.299134 | as above |
| 250 | geometric | 11.71856 | 4.787024 | 1.312165 | as above |
| cts | geometric | 11.72135 | 4.791541 | 1.315486 | as above; equals (13) |
| 12 | arithmetic | 11.90497 | 4.88210 | 1.36314 | verified in [FM08] Table 6 (draft) · ⚑ 1; $T$ assumed |
| 50 | arithmetic | 11.93301 | 4.93736 | 1.40264 | as above |
| 250 | arithmetic | 11.94068 | 4.95233 | 1.41351 | as above |

- **Hand check of the geometric rows** (this note), using (12)/(13) with the printed inputs: $n=12$, $K=100$: 4.70350; continuous
  $K=90/100/110$: 11.72131 / 4.79141 / 1.31539. Each value is $0.4$–$1.3\cdot10^{-4}$ below the table. The gaps are proportional to the
  geometric vega ($\approx8/20/17$ per unit $\sigma$), which is consistent with an unrounded $\sigma\approx0.178016$ (est.).
- **Tolerances:** $3\cdot10^{-4}$ abs for the closed form, and $3\,$SE$\,+3\cdot10^{-4}$ for MC. They come from this hand check, not from the paper.
- **Not used:** [FM08]'s MC+CV column, and its "se" column (0.15–0.21, far too large for a $10^6$-path SE; meaning unclear).
- **Consistency with (19)** at $n=12$, $K=100$: $C_A-C_G=0.1785\le DF(\bar A_F-F_G)\approx0.285$ (est., hand).
- **UNVERIFIED, not used:** the Levy (1992), Curran (1994) and Turnbull–Wakeman (1991) tables were not accessed. Linetsky (2004) and
  Fu–Madan–Wang (1999) are **continuous**-averaging benchmarks and are not valid for discrete fixings, except as $n\to\infty$ limits (not accessed).
  No published Heston discrete arithmetic Asian value was found.

## 6. Interview angle

1. **An Asian is an option on an average of fixings, priced off the forward curve.** Each fixing has its own forward, and the
   expected average is the average of those forwards, so carry and dividend timing matter, not just the terminal forward.
   Averaging across dates lowers the effective vol: for continuous averaging from inception it is about $1/\sqrt3$ of the
   terminal vol, a bit more for coarse fixings. Asians are therefore cheaper than vanillas and robust to one bad settlement print.
   Delta and vega roll off as fixings set.
2. **The geometric average is the solvable twin and the ideal control.** The log of a geometric average is a sum of Gaussians, so
   $G$ is lognormal and the Black formula applies, with a reduced vol and a forward that carries a convexity discount. AM ≥ GM path
   by path, and the gap is second order in vol, so the two payoffs move almost one-for-one. Regressing on the geometric leaves only
   $(1-\rho^2)$ of the variance: a correlation of 0.99 means 50 times fewer paths for the same error.
3. **A control variate needs a known mean under the model you simulate.** Under Heston, the Black geometric price is not the mean
   of the Heston geometric payoff, and plugging it in gives a confident, biased number. Build a shadow Black path on the same random
   numbers (known exact mean), or use a quantity whose Heston mean you know. A CV cuts noise only; the scheme's discretisation bias stays.

## 7. References

- Kemna, A.G.Z., Vorst, A.C.F. (1990). A pricing method for options based on average asset values. *J. Banking & Finance* 14(1),
  113–129 (not read; content via [FM08] and secondary sources).
- Fusai, G., Meucci, A. (2008). Pricing discretely monitored Asian options under Lévy processes. *J. Banking & Finance* 32(10),
  2076–2088. Read as the draft at openaccess.city.ac.uk/id/eprint/6061: eqs. (6), (8), (17), (22), §3.4, §5.2, Tables 4 and 6.
- Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*. Springer, §4.1 control variates (cited, not re-read).
- Boyle, P., Broadie, M., Glasserman, P. (1997). Monte Carlo methods for security pricing. *J. Econ. Dynamics & Control* 21, 1267–1321 (via secondary citation).
- Pirjol, D., Zhu, L. (2017). Asymptotics for the discrete-time average of the geometric Brownian motion and Asian options.
  arXiv:1706.09659 (read pp. 1–4: averaging convention $t_i=i\tau$, $i=1..n$).
- Kim, B., Kim, J., Kim, J., Wee, I.-S. (2016). A recursive method for discretely monitored geometric Asian option prices. *Bull. Korean Math. Soc.* (abstract only).
- Custers, G., Karbach, S., Friesen, M. (2026). Pricing and hedging of discretely monitored Asian options in the Volterra–Heston
  model. arXiv:2609.07169 (abstract only: semi-closed geometric Asian prices used as control variates).
- Andersen, L. (2008). Simple and efficient simulation of the Heston stochastic volatility model. *J. Comput. Finance* 11(3) (see heston-mc.md).
- Black, F. (1976). The pricing of commodity contracts. *J. Financial Economics* 3, 167–179. Geman, H., El Karoui, N., Rochet, J.-C.
  (1995). Changes of numéraire, changes of probability measure and option pricing. *J. Applied Probability* 32(2), 443–458.
- Levy, E. (1992). Pricing European average rate currency options. *J. Int. Money Finance* 11(5), 474–491 (not accessed).
- Curran, M. (1994). Valuing Asian and portfolio options by conditioning on the geometric mean price. *Management Science* 40(12), 1705–1711 (not accessed).
- Wikipedia, "Asian option" (accessed 2026-11-07): continuous geometric formula (secondary).
