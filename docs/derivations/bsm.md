# Black-76 (BSM in forward form): price, Greeks, implied volatility

Phase 1 · kernels `src/qpricer/models/bsm.py`, `src/qpricer/iv.py` · docstrings cite the eq. numbers below.
Cross-check status of every formula: §2.6 (⚑ = fewer than two external sources checked). "(est.)" marks a
hand estimate made in this note from the error model, not a measured or published number.

## 1. Setup

- **Measure.** $T$-forward measure $\mathbb Q^T$, numéraire the zero-coupon bond $P(t,T)$. $F_t=F(t,T)$ is the
  forward price for delivery at $T$. For SPX it already contains dividends and repo, and is implied per expiry from
  put-call parity (D-001). $F_t$ is a $\mathbb Q^T$-martingale; Black's model is
  $dF_t=\sigma F_t\,dW^T_t$, $F_T=S_T$, and $V_0=DF\cdot\mathbb E^T[g(F_T)]$ with $DF=P(0,T)$.
- **Payoff.** $g=(\theta(F_T-K))^+$, $\theta=+1$ call, $\theta=-1$ put; European, paid at $T$.
- **Inputs.** $F,K>0$; $T>0$ in years ACT/365F; $DF\in(0,1]$; $\sigma>0$ annualised decimal. No $r$, $q$ in kernels.
  A rate appears only in theta: implied zero rate $r:=-\ln(DF)/T$ (continuous compounding).
- **Notation.** $x=\ln(F/K)$ (call ITM iff $x>0$); $s=\sigma\sqrt T$ (total vol); $d_{1,2}=x/s\pm s/2$;
  $\Phi,\varphi$ standard normal cdf/pdf; $v=V/DF$ (undiscounted); intrinsic $\iota=(\theta(F-K))^+$;
  $\varepsilon=2^{-52}\approx2.2\times10^{-16}$.
- **Greeks.** $\Delta_F,\Gamma_F$ are derivatives with respect to $F$ at fixed $(K,T,DF,\sigma)$. Vega $\mathcal V=\partial V/\partial\sigma$ is
  per unit of vol (divide by 100 to get "per vol point").
- **Theta convention (kernel).** $\Theta:=\partial V/\partial t$ in calendar time ($\tau=T-t$ shrinks). It holds
  **$F$ (forward to the same expiry date), $K$, $\sigma$ and the zero rate $r$** fixed, so $DF(\tau)=e^{-r\tau}$
  accretes towards 1. Units are per year; per calendar day use $\Theta/365$. Reason for this choice: it is the
  $dt$ term of the Black PDE, i.e. the expected-P&L theta under $\mathbb Q^T$ (where $F$ is driftless). The pure
  decay at fixed $DF$ is reported separately as $\Theta_0:=-\partial V/\partial T|_{F,DF,\sigma}$.

## 2. Result

### 2.1 Price, parity, normalised form, bounds
**(1)** $\;V=DF\,\theta\,[F\,\Phi(\theta d_1)-K\,\Phi(\theta d_2)],\qquad d_{1,2}=\dfrac{\ln(F/K)}{\sigma\sqrt T}\pm\dfrac{\sigma\sqrt T}{2}$

**(2)** $\;C-P=DF\,(F-K)$ (model-free).\
**(3)** $\;V=DF\sqrt{FK}\;b_\theta(x,s)$, with $b_\theta(x,s)=\theta\,[e^{x/2}\Phi(\theta d_1)-e^{-x/2}\Phi(\theta d_2)]$ and $b_{-1}(x,s)=b_{+1}(-x,s)$.\
**(4)** Bounds, strict for $0<\sigma<\infty$: $(F-K)^+<c<F$ and $(K-F)^+<p<K$ ($c,p$ undiscounted). For the OTM
option ($\theta x\le0$): $0<b_\theta(x,s)<e^{-|x|/2}$.

### 2.2 Greeks (vectorised, closed form)
**(5)** $\;\Delta_F=\theta\,DF\,\Phi(\theta d_1)$ $\qquad$ **(6)** $\;\Gamma_F=\dfrac{DF\,\varphi(d_1)}{F\,\sigma\sqrt T}$ (same for call and put)

**(7)** $\;\mathcal V=DF\,F\,\varphi(d_1)\sqrt T=DF\,K\,\varphi(d_2)\sqrt T$ (same for call and put)

**(8)** $\;\Theta=\Theta_0+r\,V,\qquad\Theta_0=-\dfrac{DF\,F\,\varphi(d_1)\,\sigma}{2\sqrt T},\qquad r=-\ln(DF)/T$

**(9)** Identities: $\Theta+\tfrac12\sigma^2F^2\Gamma_F-rV=0$ (Black PDE); $\mathcal V=\sigma T F^2\Gamma_F$;
$\Theta_C-\Theta_P=r\,DF(F-K)$; $\partial C/\partial K=-DF\,\Phi(d_2)$; $\partial^2C/\partial K^2=DF\,\varphi(d_2)/(Ks)$;
$F\varphi(d_1)=K\varphi(d_2)$.

**(10)** Spot mapping ($F=F(S)$ linear in $S$): $\Delta_S=\Delta_F\,\partial F/\partial S$,
$\Gamma_S=\Gamma_F(\partial F/\partial S)^2$, $\mathcal V_S=\mathcal V$. With a yield model $F=Se^{(r-q)T}$ we get
$\partial F/\partial S=F/S$ and $\Theta_S=\Theta-(r-q)F\Delta_F$ (at fixed $S,r,q$). For the call this gives
$\Delta_S=e^{-qT}\Phi(d_1)$, $\Gamma_S=e^{-qT}\varphi(d_1)/(S\sigma\sqrt T)$ and
$\Theta_S=-\tfrac{e^{-qT}S\varphi(d_1)\sigma}{2\sqrt T}+qSe^{-qT}\Phi(d_1)-rKe^{-rT}\Phi(d_2)$.
With cash dividends of fixed PV $D$: $F=(S-D)/DF$, so $\partial F/\partial S=1/DF$.

### 2.3 Implied volatility (normalised, always on the OTM option)
**(11)** Set $v=V/DF$ and return NaN+flag unless $\iota<v<F$ (call) or $\iota<v<K$ (put), per (4). Reduce via
(2): $\beta=(v-\iota)/\sqrt{FK}$, then solve $b(s):=b_{+1}(-|x|,s)=\beta$ (an OTM call in normalised form).

**(12)** $\;b'(s)=\dfrac{1}{\sqrt{2\pi}}\exp\!\Big(-\dfrac{x^2}{2s^2}-\dfrac{s^2}{8}\Big)>0,\qquad
b''(s)=b'(s)\Big(\dfrac{x^2}{s^3}-\dfrac s4\Big)$. The unique inflection is at $s_c=\sqrt{2|x|}$, where $b'$ is
maximal. The link to vega is $\mathcal V=DF\sqrt{FK}\,\sqrt T\,b'(s)$.

**(13)** Initial guess. For $x=0$ the inverse is exact: $s=2\sqrt2\,\mathrm{erfinv}(\beta)$, since $b(0,s)=\mathrm{erf}(s/2\sqrt2)$;
Brenner–Subrahmanyam $s\approx\sqrt{2\pi}\,\beta$ is its first-order term. For $x\ne0$ use
$s_0=s_c=\sqrt{2|x|}$ (Manaster–Koehler), which converges monotonically (§3.3).

**(14)** Safeguarded Newton: $s_{n+1}=s_n-(b(s_n)-\beta)/b'(s_n)$. Start the bracket at $[\ell,h]=[0,\sigma_{\max}\sqrt T]$
with $\sigma_{\max}=10$; after each evaluation set $\ell=s_n$ if $b(s_n)<\beta$, else $h=s_n$. If $s_{n+1}\notin(\ell,h)$,
take $s_{n+1}=(\ell+h)/2$ instead (rtsafe). If $b(h)<\beta$ at the start, return NaN+flag ("vol > cap").

**(15)** Stopping rule: stop when $|s_{n+1}-s_n|\le10^{-13}\sqrt T$ (i.e. $|\Delta\sigma|\le10^{-13}$) **or**
$|b(s_n)-\beta|\le4\varepsilon\beta$, with $n_{\max}=100$. Non-converged elements go to `scipy.optimize.brentq` on
$[\ell,h]$ with `xtol=1e-15` and `rtol=4*eps` (scipy's minimum). Output $\sigma=s/\sqrt T$. The Brent loop runs only
over failures, expected to be none on the gate domain; record this as a D-entry, since it is the one non-vectorised path.

**(16)** Attainable accuracy: $|\delta\sigma|\approx|\delta v_{\rm tv}|/(\sqrt{FK}\sqrt T\,b'(s))$. Here $\delta v_{\rm tv}$ (abs.
error of the time value) is $\approx\varepsilon\,v_{\rm OTM}$ for an OTM input but $\approx\varepsilon\,\max(F,K)$ for an
ITM input. On the gate domain this gives $\lesssim10^{-14}$ (OTM input) vs $\sim10^{-12}$ (ITM input) (est.);
the gate is $10^{-10}$.

### 2.4 Finite-difference checks of the Greeks (central differences, natural units)
**(17)** $D_1f=\dfrac{f(u+h)-f(u-h)}{2h}$, error $\approx\dfrac{h^2|f'''|}{6}+\dfrac{\varepsilon|f|}{h}$, optimal $h\propto\varepsilon^{1/3}$;
$\;D_2f=\dfrac{f(u+h)-2f(u)+f(u-h)}{h^2}$, error $\approx\dfrac{h^2|f''''|}{12}+\dfrac{4\varepsilon|f|}{h^2}$, optimal $h\propto\varepsilon^{1/4}$.

| Greek | bump | step $h$ | scheme | predicted rel. err. (est.) |
|---|---|---|---|---|
| $\Delta_F$ | $F$ | $10^{-5}\,F\,\sigma\sqrt T$ | $D_1$ | $\sim10^{-10}$–$10^{-9}$ |
| $\Gamma_F$ | $F$, **OTM option** | $4.5\times10^{-4}\,F\,\sigma\sqrt T$ (erratum, D-014) | $D_2$ | $\lesssim10^{-6}$ measured (the original $10^{-4}$ step gives $1.8\times10^{-5}$ at $\sigma\sqrt T\approx0.007$: price noise there is $\sim600\,\varepsilon V$, pitfall 6) |
| $\mathcal V$ | $\sigma$ | $10^{-5}\,\sigma$ | $D_1$ | $\sim10^{-9}$ |
| $\Theta$ | $T$ **and** $DF$ jointly | $10^{-5}\,T$ | (18) | $\sim10^{-9}$ |

**(18)** $\;\Theta_{\rm FD}=\dfrac{V(F,K,T-h,\,DF\,e^{+rh},\sigma)-V(F,K,T+h,\,DF\,e^{-rh},\sigma)}{2h}$ with $r=-\ln(DF)/T$ fixed. Bumping $T$
alone measures $\Theta_0$. Bumps are in the variable's natural scale: $F\sigma\sqrt T$ is a one-sd move of $F$, and
in these units the derivative ratios in (17) are $O(1$–$50)$ on the gate domain. Use $h_{\rm eff}=(u+h)-u$
(Press et al., §5.7).

### 2.5 Domain of validity of the gates
$\sigma\in[0.05,1]$, $T\in[0.02,5]$, $|x|\le3s$, $F=100$, $DF\in[0.8,1]$. Parity at $10^{-12}$ needs $K/F\in[0.25,4]$ (§4.4).

### 2.6 Cross-check status (sources actually consulted for this note)
| eq. | external sources | status |
|---|---|---|
| (1),(2),(4) | Hull 5th ed. example, reproduced numerically by MathWorks `blkprice`; Hallerbach (2004) eqs (1),(3),(8) in spot form ($S\to DF\,F$, $X\to DF\,K$) | 2 |
| (3) | Jäckel (2015) (via a port's docs) + algebra | ⚑ 1 |
| (5),(6) | Hallerbach (2004) eqs (10),(13) (strike derivatives), mapped by homogeneity §3.2 | ⚑ 1 + derivation |
| (7) | Kozdron (2014) spot vega $S\sqrt T\varphi(d_1)$, equal to (7) with $DF\,F=S$ | ⚑ 1 + derivation |
| (8),(10) | Kozdron (2014) spot $\Theta=-rKe^{-rT}\Phi(d_2)-\sigma S\varphi(d_1)/2\sqrt T$, reproduced by (10) at $q=0$; PDE (9) | ⚑ 1 + PDE identity |
| (12),(13) $s_c$ | Manaster–Koehler (1982) abstract; Kozdron (2014) (19.3)–(19.5); Hallerbach (2004) fn. 3 (ATM exception) | 2 |
| (13) ATM / B–S | Hallerbach (2004) eqs (3),(5) | ⚑ 1 + derivation |
| (14)–(18) | numerical-analysis design (Press et al. §5.7, §9.4; Brent 1973), not a published formula | n/a |

## 3. Derivation

**3.1 Price.** Under $\mathbb Q^T$, $F_T=F\exp(sZ-s^2/2)$ with $Z\sim N(0,1)$; this is a martingale because $\mathbb E\,e^{sZ}=e^{s^2/2}$.
For the call, $\mathbb E[(F_T-K)^+]=\mathbb E[F_T\mathbf 1_{F_T>K}]-K\,\mathbb Q^T(F_T>K)$. The second term: $F_T>K\iff Z>-d_2$,
so the probability is $\Phi(d_2)$. The first term: change to the share measure $d\tilde{\mathbb Q}/d\mathbb Q^T=F_T/F$; by
Girsanov $Z=\tilde Z+s$, so $\ln F_T=\ln F+s^2/2+s\tilde Z$ and the term is $F\,\Phi(d_1)$. Multiply by $DF$ to get (1)
for $\theta=+1$; the put is the same computation on $\{F_T<K\}$.
Parity (2) follows from $(F_T-K)^+-(K-F_T)^+=F_T-K$ and $\mathbb E^T F_T=F$, with no model needed. For (3), divide (1)
by $DF\sqrt{FK}$ and use $\sqrt{F/K}=e^{x/2}$. Under $x\to-x$, $(d_1,d_2)\to(-d_2,-d_1)$, which gives the symmetry in (3).

**3.2 Greeks.** Key identity: $\varphi(d_2)/\varphi(d_1)=e^{(d_1^2-d_2^2)/2}=e^{x}$, so $F\varphi(d_1)=K\varphi(d_2)$ (*).
Differentiate (1) with respect to any input $y$. The terms coming from $\partial_y d_1$ and $\partial_y d_2$ combine to
$DF\,F\varphi(d_1)\,\partial_y(d_1-d_2)=DF\,F\varphi(d_1)\,\partial_y s$, because $d_1-d_2=s$ (an envelope property). Hence
$y=F$ ($\partial_F s=0$) gives (5), and then $\Gamma_F=DF\,\varphi(d_1)\,\partial_F d_1$ with $\partial_Fd_1=1/(Fs)$ gives (6);
$y=\sigma$ ($\partial_\sigma s=\sqrt T$) gives (7); $y=T$ at fixed $DF$ ($\partial_T s=\sigma/(2\sqrt T)$) gives $\Theta_0$.

Calendar theta: write $V=DF(\tau)\,v(\tau)$ with $DF=e^{-r\tau}$ and $\partial_t=-\partial_\tau$; then
$\Theta=rV+DF(-\partial_\tau v)=rV+\Theta_0$, which is (8). Black PDE: $v(t,F_t)$ is a $\mathbb Q^T$-martingale, so
$\partial_tv+\tfrac12\sigma^2F^2v_{FF}=0$. Multiplying by $DF$ gives $\Theta-rV+\tfrac12\sigma^2F^2\Gamma_F=0$, which
independently confirms (6)+(8).

Strike Greeks: homogeneity $V(\lambda F,\lambda K)=\lambda V$ gives Euler's relation $V=F\Delta_F+K\,\partial_KV$, hence
$\partial_KC=-DF\Phi(d_2)$ and $F^2\Gamma_F=K^2\partial_K^2V$. Combining (6) and (7) gives $\mathcal V=\sigma TF^2\Gamma_F$.
Spot mapping (10): chain rule with $F=Se^{(r-q)(T-t)}$, so $\partial_tF|_S=-(r-q)F$.

**3.3 Implied volatility.**
- *Uniqueness.* From (12), $b'>0$, with $b(0^+)=0$ (OTM) and $b(\infty)=e^{-|x|/2}$. So each $\beta$ in (4) has exactly one root.
- *Normalised vega (12).* Differentiate (3) and apply (*) in the form
  $e^{x/2}\varphi(d_1)=e^{-x/2}\varphi(d_2)=(2\pi)^{-1/2}e^{-x^2/2s^2-s^2/8}$. Then $b''$ follows from the exponent;
  it equals $b'\,d_1d_2/s$, i.e. volga $=\mathcal V\,d_1d_2/\sigma$ (Kozdron 19.3).
- *Monotone Newton from $s_c$ (Manaster–Koehler).* $b$ is convex on $(0,s_c)$ and concave on $(s_c,\infty)$. If
  $\beta>b(s_c)$, the root lies in the concave part, where tangents lie above the graph, so every iterate stays below
  the root and increases. If $\beta<b(s_c)$, the root lies in the convex part, where tangents lie below the graph, so
  the iterates decrease monotonically. Starting at maximal slope never overshoots, so the bracket in (14) only guards
  against rounding. At $x=0$, $s_c=0$ and in $\sigma$-space $d_1=0/0$ (the known ATM exception): use the exact inverse (13).
- *ATM inverse.* $b(0,s)=2\Phi(s/2)-1=\mathrm{erf}(s/2\sqrt2)$. Using $\mathrm{erf}(z)\approx2z/\sqrt\pi$ gives
  Brenner–Subrahmanyam, which slightly underestimates the vol because $b$ is concave. Using `erfinv` avoids the
  cancellation in $\Phi^{-1}((1+\beta)/2)$ for small $\beta$.
- *Why invert the OTM option.* By (2), the OTM undiscounted price equals the ITM time value $v-\iota$, and vega is
  the same for both. An ITM quote carries absolute rounding $\varepsilon v\approx\varepsilon\max(F,K)$, all of which
  lands on the time value. Through (16) this amplifies the $\sigma$ error by $\approx v_{\rm ITM}/v_{\rm OTM}$.

## 4. Numerical pitfalls (symptom → fix)

1. **Computing $1-\Phi(d)$ for an upper tail.** Symptom: $1-\Phi(d)$ has absolute error $\sim10^{-16}$. So a deep-OTM
   put, or $\Phi(-d_1)$ in $\Delta$, is off by ~10% at $d=8$ (where $\Phi(-8)\approx6\times10^{-16}$, only ~5 ulps of 1)
   and is exactly 0 beyond $d\approx8.3$. Fix: use the θ-form (1)/(5) with `ndtr(θ*d)`; never write `1 - ndtr`.
2. **`norm.cdf` vs `ndtr`.** `scipy.stats.norm.cdf` calls `scipy.special.ndtr` (same algorithm, same accuracy); the
   only cost is per-call overhead, which matters inside calibration loops. Fix: use `ndtr`, `erfinv`/`ndtri`, and
   `log_ndtr` whenever $\ln\Phi$ is needed (Φ underflows to 0 below $d\approx-38$).
3. **ITM cancellation.** Symptom: the ITM time value, FD gamma of an ITM option and IV from an ITM quote all carry
   absolute error $\varepsilon(F+K)$. Fix: invert on the OTM option (11), run FD gamma on the OTM option, and never
   build the OTM put as $C-DF(F-K)$.
4. **Parity floor.** If calls and puts are evaluated independently, $|C-P-DF(F-K)|\approx c\,\varepsilon\,DF(F+K)$ with
   $c\approx2$–$4$. This exceeds $10^{-12}$ once $K\gtrsim10^3$ at $F=100$. Fix: run the parity grid on
   $K/F\in[0.25,4]$. Do not derive P from C: that makes the test tautological and destroys OTM-put precision.
5. **$s\to0$ ($\sigma\to0$ or $T\to0$).** $x/s=\pm\infty$ is harmless (`ndtr` returns 0/1), but $F=K$ gives $0/0$, and so
   does $\Gamma=\varphi/(Fs)$. Fix: compute with `s_safe = where(s>0, s, 1)` and then overwrite with the limits:
   $V=DF\,\iota$, $\Delta_F=\theta DF\,\mathbf 1_{\theta x>0}$ ($\theta DF/2$ at $x=0$), $\mathcal V=0$, $\Theta=rV$,
   $\Gamma=0$ ($x\ne0$) or $+\infty$ ($x=0$). Reject $T\le0$ and $\sigma<0$ at the API.
6. **Deep wings / $d_1$ overflow.** For $|x|/s\gg1$, both $\Phi(d)$ and $b'$ underflow ($x^2/s^2\gtrsim1400$). Then
   $V=0$ and the Newton step is $0/0$. Even before underflow, $e^{x/2}\Phi(d_1)-e^{-x/2}\Phi(d_2)$ loses
   $\approx\log_{10}(|d_2|/s)$ digits; that is ≤3 on the gate domain (est.). Fix: outside the gate domain return
   NaN+flag, or use Jäckel (2015) (asymptotic expansions, log-space branches).
7. **Price below the precision of the intrinsic.** An ITM quote with $v-\iota\lesssim10\,\varepsilon v$ carries no vol
   information. Fix: return NaN+flag, never $\sigma=0$.
8. **Newton crawl in the convex branch.** When $\beta\ll b(s_c)$, convergence is monotone but slow: about 10
   iterations at $\sigma=5\%$, $T=0.02$, $x=-3s$ (est.). Fix: set $n_{\max}=100$ with Brent behind it; optionally run
   Newton on $\ln b$ (with $g'=b'/b$) in this branch.
9. **Premature step-based stop.** A small $|\Delta s|$ certifies convergence only in the quadratic regime. Fix: keep
   the residual test in (15) as a second criterion and log the maximum iteration count.
10. **Theta FD false failure.** Bumping only $T$ measures $\Theta_0$, which is off by $rV$. Fix: use the joint bump (18).
11. **Relative error at a zero.** For an ITM option with $r>0$, $\Theta$ can cross 0 (when $rV>|\Theta_0|$). For OTM
    options, $rV/|\Theta_0|=2rT\,b/(s\,b')\lesssim0.7$ on the gate domain (est.), so $\Theta<0$. Fix: FD-test $\Theta$ on
    OTM options and check ITM theta through $\Theta_C-\Theta_P=r\,DF(F-K)$.
12. **`brentq` tolerances.** `rtol < 4*eps` raises ValueError, and `xtol` is absolute in the variable being solved
    ($s$, not $\sigma$).

## 5. Test oracles

**Identities** (abs. $10^{-12}$ at $F=100$, $K/F\in[0.25,4]$, or rel. $10^{-12}$): parity (2); swap $C(F,K)=P(K,F)$;
put-call symmetry $C(F,K)=\tfrac KF\,P(F,F^2/K)$ (Carr–Ellis–Gupta 1998); homogeneity $V(\lambda F,\lambda K)=\lambda V$ and
Euler $V=F\Delta_F+K\partial_KV$; $\Delta_C-\Delta_P=DF$; Black PDE (9); $\mathcal V=\sigma TF^2\Gamma_F$;
$F\varphi(d_1)=K\varphi(d_2)$; $b_{-1}(x,s)=b_{+1}(-x,s)$.

**Limits:** $\sigma\to0$: $V\to DF\,\iota$, exact to machine precision at $\sigma\sqrt T=10^{-8}$ for $|x|\ge0.01$.
$\sigma\to\infty$: $C\to DF\,F$ and $P\to DF\,K$; at $\sigma\sqrt T=40$ the residual is $\sim\Phi(-20)$, i.e. machine-exact.
ATM exact $C=DF\,F\,\mathrm{erf}(s/2\sqrt2)$; Brenner–Subrahmanyam $C\approx DF\,F\,s/\sqrt{2\pi}$ (rel. err. $O(s^2)$).
Bounds (4) hold.

**Shape:** C increases in σ, decreases in K and is convex in K; $\partial_K^2C>0$ and $\int\partial_K^2C\,dK=DF$ (numerical quadrature).

**IV:** round-trip on the full gate grid, both θ, ITM and OTM inputs. Expected (est.): OTM $\lesssim10^{-14}$, ITM
$\sim10^{-12}$; the gate is $10^{-10}$. Also: solver vs exact ATM inverse (13); IV(call) = IV(put) at the same K;
NaN outside (4).

**Greeks:** FD per §2.4. Gate: rel. err. $<10^{-5}$.

**Cross-model (later phases):** Heston COS/Lewis with $\xi\to0$ reproduces (1) at the integrated variance (Phase 2);
GBM MC agrees within 3 SE (Phase 6).

**Published values** (test tolerance = half a unit in the last published digit):

| source | inputs | value | status |
|---|---|---|---|
| Hull (2003), *OFOD* 5th ed., futures-options section, pp. 287–288 (futures-option example as reproduced by MathWorks) | $F=K=20$, $T=4/12$, $r=9\%\Rightarrow DF=e^{-0.03}$, $\sigma=25\%$ | $P=C=1.1166$ | value verified in MathWorks `blkprice` docs, which cite Hull 5th ed. pp. 287–288; Hull's page not seen directly (page ref is secondary) |
| Haug (2007), *Complete Guide…* 2nd ed., §1.1 (Black-76 example) | $F=K=19$, $T=0.75$, $r=10\%$, $\sigma=28\%$ | $c=p=1.7011$ | **UNVERIFIED** (recalled; not found in any accessible source) |

An independent hand evaluation (ATM series of Φ) gives 1.11664 and 1.70105; consistent, but not a source.

## 6. Remarks

1. **Black-76 is Black–Scholes on the forward.** Under the T-forward measure the forward has no drift, so all carry
   (rates, dividends, repo) lives in $F$ and $DF$. Option value then depends only on moneyness relative to the forward
   and on total vol $\sigma\sqrt T$. That is why the kernels never see $r$ or $q$, and why parity-implied $F$/$DF$
   keep carry assumptions out of the vol surface.
2. **Gamma pays for theta.** A delta-hedged long option earns ½·gamma·(realised − implied variance) in cash terms per
   unit time. Theta (net of carry on the premium) is exactly the rent for gamma at implied vol. For short-dated ATM
   options, gamma and theta blow up like $1/\sqrt T$ while vega shrinks like $\sqrt T$. In this model long gamma and
   long vega go together.
3. **Implied vol is a quoting convention, and its information is in the time value.** Price rises strictly with vol
   between intrinsic and the forward, so every arbitrage-free price maps to exactly one vol. You invert the OTM side
   because the ITM price is intrinsic plus that same time value. In the deep wings vega vanishes, so tiny price
   errors become huge vol errors, and below machine precision there is no vol at all. Newton from the inflection
   point (maximum vega) converges without overshooting.

## 7. References

- Black, F. (1976). The pricing of commodity contracts. *J. Financial Economics* 3(1–2), 167–179.
- Geman, H., El Karoui, N., Rochet, J.-C. (1995). Changes of numéraire, changes of probability measure and option
  pricing. *J. Applied Probability* 32(2), 443–458.
- Hull, J.C. (2003). *Options, Futures, and Other Derivatives*, 5th ed., Prentice Hall (pp. 287–288 per MathWorks).
- Haug, E.G. (2007). *The Complete Guide to Option Pricing Formulas*, 2nd ed., McGraw-Hill, §1.1.
- Manaster, S., Koehler, G. (1982). The calculation of implied variances from the Black–Scholes model: a note.
  *J. Finance* 37(1), 227–230.
- Brenner, M., Subrahmanyam, M.G. (1988). A simple formula to compute the implied standard deviation.
  *Financial Analysts Journal* 44(5), 80–83.
- Corrado, C.J., Miller, T.W. (1996). A note on a simple, accurate formula to compute implied standard deviations.
  *J. Banking & Finance* 20(3), 595–603 (not used: its square root can go negative off-ATM).
- Hallerbach, W.G. (2004). An improved estimator for Black–Scholes–Merton implied volatility. ERIM Report ERS-2004-054-F&A.
- Jäckel, P. (2015). Let's be rational. *Wilmott Magazine*, January 2015, doi:10.1002/wilm.10395.
- Kozdron, M. (2014). Lecture #30: Implied volatility. Stat 441 notes, University of Regina (secondary).
- Carr, P., Ellis, K., Gupta, V. (1998). Static hedging of exotic options. *J. Finance* 53(3), 1165–1190.
- Press, W.H., Teukolsky, S.A., Vetterling, W.T., Flannery, B.P. (2007). *Numerical Recipes*, 3rd ed., CUP, §5.7, §9.4.
- Brent, R.P. (1973). *Algorithms for Minimization without Derivatives*. Prentice-Hall.
- MathWorks, Financial Toolbox docs, `blkprice` · SciPy source `scipy/stats/_continuous_distns.py` (both accessed 2026-11-06).
