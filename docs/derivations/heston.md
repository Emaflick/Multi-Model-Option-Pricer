# Heston: characteristic function and Fourier pricing (COS, Lewis)

Phase 2 · `src/qpricer/models/heston.py` (CF, cumulants), `src/qpricer/engines/fourier.py` (COS, Lewis) · Monte Carlo:
`heston-mc.md` (same symbols: $W^F,W^v,\nu_F,\bar\Sigma$, set names FO, A-I…). Notation otherwise as in `bsm.md`. ⚑ = fewer than two external
sources checked (§2.6). "(est.)" = hand estimate made in this note, not a measured or published number.

## 1. Setup

- **Measure, dynamics.** $T$-forward measure $\mathbb Q^T$, numéraire $P(t,T)$. $F_t=F(t,T)$ is a $\mathbb Q^T$-martingale, $F_T=S_T$,
  and $V_0=DF\cdot\mathbb E^T[g(F_T)]$. Rates and carry are deterministic in the kernels ($DF$ is an input), so Heston's risk-neutral
  dynamics carry over: $dF_t=\sqrt{v_t}F_t\,dW^F_t$, $\;dv_t=\kappa(\theta-v_t)dt+\xi\sqrt{v_t}\,dW^v_t$, $\;d\langle W^F,W^v\rangle_t=\rho\,dt$.
- **Parameters.** $v_0,\theta>0$ are annualised variances, $\kappa>0$ (1/yr), $\xi>0$ is vol-of-vol (1/√yr), $|\rho|<1$. The domain is
  $\kappa-\rho\xi>0$, which holds automatically for $\rho\le0$ (SPX); reject other inputs at the API. Feller ratio $\nu_F=2\kappa\theta/\xi^2$; when
  $\nu_F\ge1$, $v$ never reaches 0.
- **CF.** $X:=\ln(F_T/F)$ and $\varphi(u):=\mathbb E^T[e^{iuX}]$, with $u$ complex, $-w_+<\operatorname{Im}u<-w_-$. Here $(w_-,w_+)\supset[0,1]$ is the
  set of finite exponential moments (eq. 17). COS uses real $u$; Lewis uses $\operatorname{Im}u=-\tfrac12$.
- **Payoff, strikes.** The put $(K-F_T)^+$ is priced directly; the call follows from parity $C=P+DF(F-K)$. $x=\ln(F/K)$ as in bsm.md,
  and $z:=\ln(K/F)=-x$. $n(\cdot)$ is the standard normal pdf.
- **Conventions.** $T$ ACT/365F; no $r,q$ in kernels. Complex $\sqrt\cdot$ and $\operatorname{Log}$ are principal ($\operatorname{Re}\sqrt\cdot\ge0$, $\arg\in(-\pi,\pi]$, the
  numpy defaults). Spot-based sources convert as $F=S_0e^{(r-q)T}$, $DF=e^{-rT}$; their CF of $\ln(S_T/S_0)-(r-q)T$ is $\varphi$. FO's
  $(\lambda,\bar u,u_0,\eta)$ are our $(\kappa,\theta,v_0,\xi)$.

## 2. Result

### 2.1 Characteristic function
**(1)** $\beta=\kappa-i\rho\xi u,\quad A=u^2+iu,\quad d=\sqrt{\beta^2+\xi^2A}\ \ (\operatorname{Re}d>0),\quad g=(\beta-d)/(\beta+d).$\
**(2)** Albrecher–Mayer–Schoutens–Tistaert (AMST) "little trap" form: $\varphi(u)=\exp(C+v_0D)$ with
$D=\dfrac{\beta-d}{\xi^2}\dfrac{1-e^{-dT}}{1-ge^{-dT}},\qquad C=\dfrac{\kappa\theta}{\xi^2}\Big[(\beta-d)T-2\operatorname{Log}\dfrac{1-ge^{-dT}}{1-g}\Big].$\
**(3)** Implementation form. It is identical to (2), but $\beta-d$ is never formed by subtraction. Let $\Upsilon=\beta+d+(d-\beta)e^{-dT}$. Then
$D=-\dfrac{A(1-e^{-dT})}{\Upsilon}$ and $C=-\dfrac{\kappa\theta AT}{\beta+d}-\dfrac{2\kappa\theta}{\xi^2}\operatorname{Log}\dfrac{\Upsilon}{2d}$. Take Log of the ratio
$\Upsilon/2d=(1-ge^{-dT})/(1-g)$ (AMST's $G_2$), never a difference of two logs.\
**(4)** Original Heston (1993) form, for use as a negative control only: $g_1=1/g$,
$D_1=\frac{\beta+d}{\xi^2}\frac{1-e^{dT}}{1-g_1e^{dT}}$, $C_1=\frac{\kappa\theta}{\xi^2}\big[(\beta+d)T-2\operatorname{Log}\frac{1-g_1e^{dT}}{1-g_1}\big]$. It is algebraically equal to (2).
With principal branches, however, it jumps wherever $G_1=\frac{1-g_1e^{dT}}{1-g_1}$ crosses $(-\infty,0]$. For any $T$ large enough this happens
unless $\nu_F\in\mathbb N$ (AMST Prop. 1).\
**(5)** Exact values: $\varphi(0)=\varphi(-i)=1$ (because $A=0$); $\varphi(-\tfrac i2)=\mathbb E[(F_T/F)^{1/2}]\in(0,1]$; for real $u$, $|\varphi|\le1$ and $\varphi(-u)=\overline{\varphi(u)}$.

### 2.2 Cumulants and truncation range
**(6)** $E=e^{-\kappa T}$ (not heston-mc's per-step $E$), $\tau_\kappa=(1-E)/\kappa$, $\bar\Sigma=\int_0^T\bar v\,dt=\theta T+(v_0-\theta)\tau_\kappa$, $c_1=-\tfrac12\bar\Sigma$.\
**(7)** $c_2=2\theta\big[\alpha_0(T-\tau_\kappa)+\alpha_1(\tau_\kappa-TE)+\tfrac12\alpha_2\kappa\tau_\kappa^2\big]+2v_0\big[\alpha_0\tau_\kappa+\alpha_1TE+\alpha_2E\tau_\kappa\big]$, where
$\alpha_0=\tfrac12-\tfrac{\rho\xi}{2\kappa}+\tfrac{\xi^2}{8\kappa^2}$, $\alpha_1=\tfrac{\rho\xi}{2\kappa}-\tfrac{\xi^2}{4\kappa^2}$, $\alpha_2=\tfrac{\xi^2}{8\kappa^2}$. If $\kappa T<10^{-3}$, use the $\kappa\to0$ limit
$c_2=v_0(T-\tfrac12\rho\xi T^2+\tfrac1{12}\xi^2T^3)$. Eq. (7) equals Le Floc'h (2020) eq. (B5) term by term.
**Fang–Oosterlee (2008) Table 11 is misprinted:** its $\bar u(6e^{-\lambda T}-7)$ must read $\bar u(4e^{-\lambda T}-5)$. The printed $c_2$ is too
small by $\xi^2\theta(1-E)/(4\kappa^3)$, tends to $v_0T-\xi^2\theta T/(4\kappa^2)$ as $T\to0$, and can go negative. That is why F&O take
$|c_2|$. Hand check (est.) for set FO (§5) at $T=1$: $c_2=0.03157$ (misprinted formula: $0.03091$), $\bar\Sigma=0.02858$.\
**(8)** $[a,b]=[c_1-L\sqrt{c_2},\ c_1+L\sqrt{c_2}]$ with $L=12$ and $c_4$ omitted, which is F&O §5.3's choice for Heston. **Erratum (D-017, validation F1):** $L=12$ is truncation-limited (error $\approx4\times10^{-5}$ on set FO, independent of $N$), and a fixed $L=18$ still misses $10^{-6}$ on ~23% of random SPX-plausible sets; the implementation uses a self-checking default ($N=256, L=18$, refined to $(2N, L+6)$ until agreement within $10^{-9}F$). The range is for $X$,
so it does not depend on the strike. Since $c_2>0$, no $|\cdot|$ is needed. Le Floc'h (2020, App. B) gives a CAS-derived closed-form $c_4$;
it is not used here and was not cross-checked.

### 2.3 COS put, forward-centred (Le Floc'h 2020 variant of F&O 2008)
**(9)** $u_k=k\pi/(b-a)$ for $k=0,\dots,N-1$, with $N=256$; $\;w_k=\operatorname{Re}[\varphi(u_k)e^{-iu_ka}]$, then overwrite $w_0=\tfrac12$.\
**(10)** Per strike: $\bar z=\min(\max(z,a),b)$ and $V_k=\frac{2}{b-a}\big[K\psi_k(\bar z)-F\chi_k(\bar z)\big]$, where
$\chi_k(z)=\dfrac{e^{z}\cos(u_k(z-a))+u_ke^{z}\sin(u_k(z-a))-e^{a}}{1+u_k^2}$, $\psi_k(z)=\dfrac{\sin(u_k(z-a))}{u_k}$ for $k\ge1$, and $\psi_0(z)=z-a$.\
**(11)** $P=DF\sum_kw_kV_k$, computed as an $(M\times N)$ matrix times an $N$-vector for the $M$ strikes of one expiry ($\varphi$ is
evaluated once per expiry); $C=P+DF(F-K)$. The clipping gives $P=0$ when $z\le a$ and $P\approx DF(K-F)$ when $z\ge b$.\
**(12)** Greeks (not gated) ⚑: $\Delta^P_F=-DF\sum_kw_k\frac{2}{b-a}\chi_k(\bar z)$ and $\Gamma_F=DF\frac{K}{F^2}\hat f(\bar z)$, where
$\hat f(y)=\frac{2}{b-a}\sum_kw_k\cos(u_k(y-a))$ is the COS density of $X$. Also $\partial P/\partial v_0=DF\sum_k\operatorname{Re}[D(u_k)\varphi(u_k)e^{-iu_ka}]V_k$ (halve $k=0$).\
**(13)** Error model (est.) ⚑: $|\varepsilon|\approx\varepsilon_{\rm tail}(L)+\varepsilon_{\rm ser}(N)$, with $\varepsilon_{\rm ser}\approx\frac{F}{100}e^{-c\,u_N}$, $u_N=\frac{\pi N}{b-a}$, and
$c=\sqrt{1-\rho^2}\,(v_0+\kappa\theta T)/\xi$, the large-$u$ decay rate of $|\varphi|$ (§3.6). The resulting requirement is $N_{\rm req}\approx20(b-a)/(\pi c)$ for
$\sim10^{-9}$. Check against F&O Table 4 ($T=1$, their $c_2$): the model predicts $2.8\times10^{-4}$ and $1.8\times10^{-5}$ at $N=96,128$; the paper
reports $2.99\times10^{-4}$ and $1.94\times10^{-5}$. For the gate sets of §5 on $T\in[0.1,3]$, $N_{\rm req}\approx120$–$200$, so $N=256$ suffices.

### 2.4 Lewis (2000) single integral
**(14)** $C=DF\Big[F-\frac{\sqrt{FK}}{\pi}\int_0^\infty\operatorname{Re}\big[e^{iux}\varphi(u-\tfrac i2)\big]\frac{du}{u^2+1/4}\Big]$ with $x=\ln(F/K)$, and $P=DF\big[K-(\text{same term})\big]$.
On this contour $A=u^2+\tfrac14$ is real and $\beta=\kappa-\tfrac12\rho\xi-i\rho\xi u$; evaluate $\varphi$ with (3).\
**(15)** Quadrature: `scipy.integrate.quad_vec` on $[0,\infty)$ (semi-infinite GK15) over the strike vector of one expiry, with
`norm='max'`, `epsabs=epsrel=1e-11`, `full_output=True`; assert `info.status == 0`. The price error is at most $DF\sqrt{FK}\cdot\mathrm{err}/\pi$.
Holding it below $10^{-7}$ (a tenth of the gate) at $K\le150$ needs err $\le2.6\times10^{-9}$. quad's default `1.49e-8` allows about
$2\times10^{-6}$ and so fails the gate. A per-strike scalar `quad(f, 0, np.inf, limit=500)` is an acceptable fallback (it is a loop:
record a D-entry). A finite cut at $U=40/c$ (eq. 13) is safe (est.).

### 2.5 Limits, explosion, symmetry
**(16)** Limit $\xi\to0$. With $\bar v(t)=\theta+(v_0-\theta)e^{-\kappa t}$, $\sigma_{\rm eff}^2=\bar\Sigma/T=\theta+(v_0-\theta)\frac{1-e^{-\kappa T}}{\kappa T}$ and
$V_{\rm H}=V_{\rm B}(\sigma_{\rm eff})+\rho\xi J\,\partial_x\partial_\Sigma V_{\rm B}+O(\xi^2)$. Here
$J=\int_0^T\bar v(t)\frac{1-e^{-\kappa(T-t)}}{\kappa}dt=\frac{\theta}{\kappa}(T-\tau_\kappa)+\frac{v_0-\theta}{\kappa}(\tau_\kappa-TE)$ and $\partial_x\partial_\Sigma V_{\rm B}=-\frac{DF\,F\,n(d_1)\,d_2}{2\bar\Sigma}$, where $x=\ln F$
at fixed $K$, $\Sigma=\sigma^2T$, and $d_{1,2}$ are bsm.md eq. (1) at $\sigma_{\rm eff}$. **Consequence:** for $\rho\ne0$ the gap to plain Black is
$O(\xi)$, up to about $0.12|\rho|\xi F\tau_\kappa$. Example (est.): $\kappa=1.5$, $v_0=\theta=0.04$, $\rho=-0.7$, $\xi=10^{-4}$, $T=1$, $K=120$, $F=100$, $DF=1$
gives a correction of $-3.3\times10^{-4}$, 30 times the gate tolerance $10^{-5}$. **The roadmap gate (plain Black, $<10^{-5}$) therefore holds
only for $\rho=0$**, where the residual is $O(\xi^2)\lesssim10^{-7}$ (est.). For $\rho\ne0$ the reference must include the $\rho\xi J$ term.\
**(17)** Moment explosion for $\rho\le0$: $\mathbb E[(F_T/F)^\omega]<\infty$ iff $T<T^*(\omega)$. With $\chi=\rho\xi\omega-\kappa$ and $\Delta=\chi^2-\xi^2(\omega^2-\omega)$:
$T^*=\infty$ if $\Delta\ge0$, else $T^*=\frac{2}{\sqrt{-\Delta}}\big[\arctan\frac{\sqrt{-\Delta}}{\chi}+\pi\mathbf 1_{\chi<0}\big]$. The bounds $w_\pm(T)$ of §1 solve $T^*(w)=T$; no $\omega\in[0,1]$
ever explodes. Example (est.): set FO has $T^*(-2)=4.28$ y.\
**(18)** Share-measure symmetry ⚑: $C(F,K;\kappa,\theta,\xi,\rho,v_0)=\frac KF\,P\big(F,\frac{F^2}{K};\ \kappa-\rho\xi,\ \frac{\kappa\theta}{\kappa-\rho\xi},\ \xi,\ -\rho,\ v_0\big)$, at the same $DF$ and $T$; $\nu_F$ is invariant.

### 2.6 Cross-check status (sources actually read for this note)
| eq. | sources | status |
|---|---|---|
| (1),(2) | AMST (2007) §3 $\phi_2$; F&O (2008) §3.3 | 2 |
| (3) | derivation §3.1; Lipton–Gal–Lasis (2014) eqs (191)–(194), same form on the Lewis contour | 2 |
| (4) | AMST (2007) eq. (1); Heston (1993) itself not seen | ⚑ 1 |
| (6),(7) | two derivations (§3.2); Le Floc'h (2020) (B4)–(B5), identical; F&O Table 11 differs by the misprinted term | 2 |
| (8)–(11) | F&O eqs (22)–(25), (30), §5.3; Le Floc'h (2020) eqs (2)–(10) | 2 |
| (12),(13),(18) | derivation only; (13) is also fitted to F&O Tables 4–5 | ⚑ |
| (14) | derivation §3.4; Gobet (2010) slide 4 (Lewis '00, contour form); Lipton–Gal–Lasis (2014) eqs (180)–(197) | 2 |
| (16) | derivation §3.5; Benhamou–Gobet–Miri (2010) $a_{1,T}$, read from Gobet's slides | 2 |
| (17) | Keller-Ressel (2011) eq. (6.7) without jumps; derivation §3.7 (Andersen–Piterbarg 2007 not seen) | 2 |

## 3. Derivation

**3.1 CF.** By Feynman–Kac, $f=\mathbb E[e^{iuX}\mid x_t,v_t]$ solves $f_t+\tfrac12v(f_{xx}-f_x)+\kappa(\theta-v)f_v+\tfrac12\xi^2vf_{vv}+\rho\xi vf_{xv}=0$. The
affine ansatz $f=\exp(iux+C(\tau)+vD(\tau))$, with $\tau=T-t$, gives $D'=\tfrac12\xi^2D^2-\beta D-\tfrac12A$, $C'=\kappa\theta D$, and $C(0)=D(0)=0$.
The quadratic has roots $r_\pm=(\beta\pm d)/\xi^2$. Separating variables gives $\frac1d\ln\frac{D-r_+}{D-r_-}=\tau+\frac1d\ln\frac{r_+}{r_-}$, hence
$D=r_-\frac{1-e^{-d\tau}}{1-ge^{-d\tau}}$ with $g=r_-/r_+$. Then $C=\kappa\theta\int D$ gives (2). Replacing $d\to-d$ gives (4). This is the same function,
because the Riccati solution is unique and even in $d$ (AMST §3).
*Branches.* $\operatorname{Re}d^2=\kappa^2+(1-\rho^2)\xi^2u^2>0$ on the real axis, and $\operatorname{Re}d^2=(\kappa-\tfrac12\rho\xi)^2+(1-\rho^2)\xi^2u^2+\tfrac14\xi^2>0$ on
$\operatorname{Im}u=-\frac12$, so the principal $\sqrt\cdot$ is continuous on both contours. In (2), $|e^{-dT}|<1$ and $G_2\to1/(1-g)$ as $T\to\infty$: it
does not wind around 0. AMST Thm 3 proves this on Carr–Madan contours; F&O cite Lord–Kahl (2010) for the whole strip. In (4),
$G_1\approx-g_1e^{dT}/(1-g_1)$, with argument about $T\operatorname{Im}d+$const, which grows with $u$. $G_1$ spirals around 0, so the principal Log jumps by
$2\pi i$ and $\varphi_1$ is multiplied by $e^{\mp4\pi i\kappa\theta/\xi^2}\ne1$ unless $\nu_F\in\mathbb N$. Eq. (3) follows from $\beta-d=-\xi^2A/(\beta+d)$, $1-g=2d/(\beta+d)$
and $1-ge^{-dT}=\Upsilon/(\beta+d)$.

**3.2 Cumulants.** The CGF $\mathcal K(w)=\ln\mathbb Ee^{wX}$ is (2) at $u=-iw$. The Riccati becomes $D'=\frac12(w^2-w)+(\rho\xi w-\kappa)D+\frac12\xi^2D^2$. Expanding
$D=D_1w+D_2w^2+\dots$ gives $D_1'=-\frac12-\kappa D_1$ and $D_2'=\frac12+\rho\xi D_1+\frac12\xi^2D_1^2-\kappa D_2$. Solving with the kernel $e^{-\kappa(\tau-s)}$:
$D_1=-\frac12\tau_\kappa$ and $D_2=\alpha_0\tau_\kappa+\alpha_1\tau e^{-\kappa\tau}+\alpha_2e^{-\kappa\tau}\tau_\kappa$; then $C_n=\kappa\theta\int_0^TD_n$, $c_1=C_1+v_0D_1$ and $c_2=2(C_2+v_0D_2)$,
which is (6)–(7). Second route, for $v_0=\theta$: $\operatorname{Var}(-\frac12I+M)=\frac14\operatorname{Var}I+\mathbb EI-\operatorname{Cov}(I,M)$. Here
$\operatorname{Var}I=\frac{\xi^2\theta}{\kappa^2}[T-2\tau_\kappa+\frac{1-E^2}{2\kappa}]$, $\operatorname{Cov}(I,M)=\frac{\rho\xi\theta}{\kappa}(T-\tau_\kappa)$ and $\mathbb EI=\theta T$, which reproduces (7). Small-$T$ check: $c_2\to v_0T$.

**3.3 COS.** On $[a,b]$, $f(y)\approx\frac{2}{b-a}\sum'_k\operatorname{Re}[\int_a^bf e^{iu_k(y-a)}]\cos(u_k(y-a))$. Replacing $\int_a^b$ by $\int_{\mathbb R}$ gives the coefficient $w_k$.
Integrating $(K-Fe^y)^+$ against each cosine on $[a,\bar z]$ ($\int e^y\cos=\chi$, $\int\cos=\psi$) gives (10). The error (Le Floc'h 2020, App. A) is
$\varepsilon=\int_{\mathbb R\setminus[a,b]}(v-\hat v)f+\sum_{k\ge N}w_kV_k$, where $\hat v$ is the even $2(b-a)$-periodic extension of the payoff. For the put,
$|\hat v-v|=2Fe^a\sinh(a-y)$ below $a$ (zero at $a$, at most $K$), and both are zero beyond $b$, so $\varepsilon_{\rm tail}\le K\,\mathbb Q(X\notin[a,b])$. For the call,
$\hat v\approx Ke^b$ beyond $b$ and $V^{\rm call}_k\propto e^b$, so every error is amplified by $e^b$ (F&O Remark 5.2). The density is smooth with an
analytic CF, so $w_k$ decays geometrically at the rate of $|\varphi(u_k)|$ (F&O Lemma 4.2). This gives (13).

**3.4 Lewis.** $(F_T-K)^+=F_T-\min(F_T,K)$, and $\min(F_T,K)=K\,g(x+X)$ with $g(y)=\min(e^y,1)$. For $0<\operatorname{Im}\zeta<1$,
$\hat g(\zeta)=\int e^{i\zeta y}g\,dy=\frac{1}{1+i\zeta}-\frac{1}{i\zeta}=\frac{1}{\zeta^2-i\zeta}$. Fourier inversion on $\zeta=u+\frac i2$ gives
$\mathbb E\min(F_T,K)=\frac{K}{2\pi}\int\hat g(\zeta)e^{-i\zeta x}\varphi(-\zeta)\,du$, where $\zeta^2-i\zeta=u^2+\frac14$ and $Ke^{x/2}=\sqrt{FK}$. Substituting $u\to-u$ and using
$\varphi(-u-\frac i2)=\overline{\varphi(u-\frac i2)}$ folds this onto $[0,\infty)$ as $\operatorname{Re}$, which is (14). Black check: $\varphi(u-\frac i2)=e^{-(u^2+1/4)s^2/2}$.
The contour needs only $\mathbb E[F_T^{1/2}]\le F^{1/2}$, which is always finite.

**3.5 $\xi\to0$.** Write $D=D_0+\xi D_1+O(\xi^2)$ in the §3.1 Riccati: $D_0=-\frac12A\tau_\kappa(\tau)$ and $D_1'=-\kappa D_1+i\rho uD_0$. Integrating, together with $C=\kappa\theta\int D$,
gives $\ln\varphi=-\frac12A\bar\Sigma+\xi\rho(iu)(-\frac12A)J+O(\xi^2)$; the $v_0$ and $\theta$ coefficients of $J$ come from direct integration. Black's CF at total
variance $\Sigma$ is $e^{-A\Sigma/2}$, so a factor $-\frac12A$ acts as $\partial_\Sigma$. A factor $iu$ acts as $\partial_x$ on $V(x)=DF\,\mathbb E[G(x+X)]$ (integration
by parts). Hence (16). In detail, $\partial_\Sigma V_{\rm B}=DF\,Fn(d_1)/(2\sqrt\Sigma)$ and $\partial_x[Fn(d_1)/2\sqrt\Sigma]=-Fn(d_1)d_2/2\Sigma$. Two checks: BGM's
$a_{1,T}=\rho\xi\int_0^Te^{\kappa t}\bar v(t)\int_t^Te^{-\kappa s}ds\,dt=\rho\xi J$ multiplies $\partial^2_{xy}P_{BS}$; and as $T\to0$, $J\to v_0T^2/2$, so (16) gives an
implied-vol shift $\delta\sigma\approx\rho\xi\ln(K/F)/(4\sqrt{v_0})$, the textbook short-dated skew ⚑.

**3.6 CF decay.** For real $u\to\infty$ and $|\rho|<1$, $\beta+d\approx\xi u(\sqrt{1-\rho^2}-i\rho)$. So $D\approx-A/(\beta+d)\approx-u(\sqrt{1-\rho^2}+i\rho)/\xi$, the
first term of $C$ is $\kappa\theta T$ times the same, and the Log term tends to a constant. Hence $\operatorname{Re}\ln\varphi\approx-cu$. This needs
$u\gg\max(\kappa,1/T)/(\xi\sqrt{1-\rho^2})$; below that, $|\varphi|\approx e^{-c_2u^2/2}$. At $\rho=\pm1$, $d^2=\kappa^2+i\xi u(\xi-2\kappa\rho)$, so $d\sim\sqrt u$ and
$|\varphi|$ decays only like $e^{-c'\sqrt u}$.

**3.7 Explosion, symmetry.** *Explosion:* for real $\omega$, $\psi=D(-i\omega)$ solves $\psi'=R(\psi)=\frac12(\omega^2-\omega)+\chi\psi+\frac12\xi^2\psi^2$, and $T^*=\int_0^\infty d\psi/R$. If
$\Delta<0$, $R$ has no real root and the integral is (17). If $\Delta\ge0$ and $\chi<0$, $\psi$ converges to the lower root and $T^*=\infty$. For $\rho\le0$, $\chi>0$
forces $\omega<0$ and then $\Delta<0$. *Symmetry:* take $d\tilde{\mathbb Q}/d\mathbb Q^T=F_T/F$. Girsanov gives $dW^v=d\tilde W^v+\rho\sqrt v\,dt$, so
$dv=[\kappa\theta-(\kappa-\rho\xi)v]dt+\xi\sqrt v\,d\tilde W^v$, and $dX=\frac12v\,dt+\sqrt v\,d\tilde W^F$. Then $Y=-X$ is a Heston log-forward with correlation $-\rho$,
and $C=DF\,F\,\tilde{\mathbb E}[(1-\frac KFe^{Y})^+]=\frac KF\tilde P(F,F^2/K)$.

## 4. Numerical pitfalls (symptom → fix)

1. **Branch cut (form 4).** Prices jump in $u$; at long $T$ they are silently wrong. Example: FO ATM at $T=10$ gives 29.83 instead of 33.84
   (AMST Table 1). Fix: use (2)/(3) with principal branches; keep (4) only as a negative control in tests.
2. **Overflow.** Form (4) returns NaN at large $uT$ ($e^{dT}\to\infty$, giving inf/inf). Fix: (3) uses $e^{-dT}$ with $|e^{-dT}|\le1$; underflow to 0 is benign.
3. **Cancellation in $\beta-d$** (small $\xi$, small $|u|$). Naive (2) loses about $\log_{10}(\kappa^2/\xi^2|u|)$ digits, and the $\xi\to0$ gate drifts once
   $\xi\lesssim10^{-5}$. Fix: (3). The remaining Log term is $O(\xi^2)$ with absolute error $\varepsilon$, amplified by $2\kappa\theta/\xi^2$ ($\approx3\times10^{-9}$ at $\xi=10^{-4}$,
   est.). For smaller $\xi$, compute $\operatorname{Log}(1+p+iq)=\frac12\mathrm{log1p}(2p+p^2+q^2)+i\,\mathrm{atan2}(q,1+p)$; do not rely on complex `np.log1p`,
   whose accuracy near 0 is UNVERIFIED.
4. **$u=0$ is not singular.** Unlike the Gil-Pelaez $P_1,P_2$ integrands ($1/u$), COS has $\varphi(0)=1$ exactly and the Lewis integrand at 0 is
   $4\varphi(-\frac i2)\le4$. Fix: add no special cases; test (5).
5. **Domain $\kappa-\rho\xi\le0$.** Then $\beta+d=0$ at $u=-i$ (0/0 in (3)) and $\operatorname{Re}\beta$ can be $\le0$ on the Lewis contour. Fix: reject at the API (D-entry).
6. **Uncentred COS range** (classic F&O: fixed range for $\ln(F_T/K)$). Short-dated ITM puts are off by up to $10^{-2}$ (Le Floc'h 2020,
   Figs. 1–2). Fix: (10), with the range on $X$ and the kink at $z$.
7. **Direct COS calls.** Errors grow with $L$ and $T$ because $V^{\rm call}_k\propto e^b$: $e^b\approx70$ for FO at $T=3$ and $\approx3\times10^4$ for LW at $T=3$
   (est.). The slow right tail adds to this. Fix: price puts and use parity (F&O Remark 5.2).
8. **Misprinted F&O $c_2$.** $c_2<0$ or too small, and the $|c_2|$ workaround. Fix: (7), tested against finite differences of $\mathcal K$.
9. **Cumulants as $\kappa T\to0$.** Noise appears in $c_2$ for $\kappa T\lesssim10^{-4}$ (a $1/\kappa^3$ prefactor on an $O(\kappa^3)$ numerator). Fix: the $\kappa\to0$ branch
   of (7); bound $\kappa\ge10^{-3}$ in calibration.
10. **$N=256$ too small.** The COS–Lewis gap shrinks with $N$ but stays above $10^{-6}$, and (13) gives $N_{\rm req}>256$. This happens for $|\rho|\to1$ at
    long $T$ ($c\propto\sqrt{1-\rho^2}$); for small $v_0$ with large $\xi$ (e.g. $v_0=0.005$, $\xi=1$, $\rho=-0.7$, $T=0.1$: $e^{-cu_N}\approx4\times10^{-6}$, est.); and
    for long $T$ with large $\xi$ (case KJ, $N_{\rm req}\approx660$, est.). Fix: use $N=\max(256,N_{\rm req})$ (D-entry).
11. **Range too narrow for fat tails.** The error stops falling with $N$ but changes with $L$. The left tail is heavy when $T$ approaches $T^*(\omega)$ for
    $\omega\approx-2\ldots-3$ (eq. 17), and a $c_2$-only range ignores kurtosis. Fix: diagnostic comparing $L=12$ with $L=16$ at $N\propto L$; if needed,
    add $\sqrt{c_4}$ (F&O eq. 49, $L=10$).
12. **$\rho=\pm1$.** COS convergence becomes algebraic, because at $\rho=-1$, $X\le(v_0+\kappa\theta T)/\xi$ a.s. (the density has an edge). Fix: bound $|\rho|\le0.99$.
13. **Strikes outside $[a,b]$.** Unclipped $\chi_k,\psi_k$ give wrong, even negative, prices. Fix: clip as in (10). The resulting zero deep-OTM prices
    must give NaN plus a flag in IV inversion (bsm.md pitfall 7). Do not clip tiny negative COS prices or densities ($\sim10^{-14}$) in kernels or tests.
14. **Lewis tolerances.** quad's defaults give errors of about $2\times10^{-6}$. Fix: (15).
15. **Feller.** Violation (FO, GS and typical SPX fits) does not affect (2)/(3): the CIR transform is the same whether or not 0 is attainable
    (it is instantaneously reflecting). It matters only for Monte Carlo discretisation (heston-mc.md). Never impose Feller in
    calibration for the pricer's sake; report $\nu_F$.
16. **Moment explosion and contours.** Carr–Madan contours $\operatorname{Im}u=-(1+\alpha)$ and call-payoff transforms diverge once $T\ge T^*(1+\alpha)$, and the call's right
    tail weighted by $e^y$ decays slowly as $w_+\downarrow1$. Fix: keep Lewis at $-\frac12$ and COS on puts; check (17) before moving any contour.

## 5. Test oracles

**Parameter sets** ($F$, $DF$ form). **FO** (as in heston-mc.md): $\kappa=1.5768$, $\theta=0.0398$, $\xi=0.5751$, $\rho=-0.5711$, $v_0=0.0175$; a Eurostoxx-50
calibration (AMST §4), Feller-violating ($\nu_F=0.38$). **GS**, SPX-like: $\kappa=1.3253$, $\theta=0.0354$, $\xi=0.3877$, $\rho=-0.7165$ (Gatheral 2006 Table 3.2,
as quoted in Keller-Ressel 2011 Fig. 1), with $v_0:=\theta$ our choice; Feller-violating ($\nu_F=0.62$). **LW**, the Lewis set: $\kappa=4$, $\theta=0.25$, $\xi=1$,
$\rho=-0.5$, $v_0=0.04$ ($\nu_F=2$). **AT2**, the AMST Table 2 set: $\kappa=1.5$, $\theta=v_0=0.04$, $\xi=0.3$, $\rho=-0.9$.

**Gates and identities.**
- **COS ($L=12$, $N=256$) vs Lewis (15): $<10^{-6}$**, puts and calls, on $K\in\{60,65,\dots,150\}\times T\in\{0.1,0.25,0.5,1,2,3\}$, $F=100$, $DF\in\{1,0.95\}$,
  sets FO, GS, LW, AT2. Diagnostic: $L=16$, $N=512$ against $L=12$, $N=256$ should agree to $<10^{-8}$; a larger gap means $\varepsilon_{\rm tail}$ (pitfall 11).
- **CF vs an independent ODE:** integrate the §3.1 Riccati (`solve_ivp`, DOP853, rtol $=10^{-12}$) at $u\in\{0.5,5,50\}$ and on $\operatorname{Im}u=-\frac12$, for
  $T\in\{0.1,3,10\}$, sets FO and KJ: relative error $<10^{-9}$. Form (4) at FO, $T=10$ must fail somewhere (negative control).
- **Identities.** (5) to $10^{-15}$. $c_1,c_2$ against central differences of $\mathcal K(w)$ with $h=10^{-3}$: relative error $<10^{-5}$. (7) minus the
  misprinted F&O $c_2$ equals $\xi^2\theta(1-E)/(4\kappa^3)$. Cross-engine parity $C_{\rm Lewis}-P_{\rm COS}=DF(F-K)$, $<10^{-6}$ (parity within one engine is
  tautological). Symmetry (18), $<10^{-6}$. $P$ increasing and convex in $K$, $0\le\partial_KP\le DF$, $\mathbb E(F_T-K)^+$ non-decreasing in $T$. Error vs $N$
  decays exponentially down to a floor of about $10^{-13}$ (figure).
- **$\xi\to0$** ($\xi=10^{-4}$, $\kappa=1.5$, $v_0=\theta=0.04$, same $K,T$ grid). $\rho=0$ vs Black at $\sigma_{\rm eff}$ (bsm.md eq. 1): $<10^{-5}$, expected
  $\lesssim10^{-7}$ (est.); this is the same $\bar\Sigma$ target as the MC limit in heston-mc.md. $\rho=-0.7$ vs (16) including the $\rho\xi J$ term: $<10^{-5}$.
  Power check: at $\rho=-0.7$, the gap to plain Black at $K=120$, $T=1$ must be about $-3.3\times10^{-4}$ (est.).

**Published values** (converted to $F$, $DF$):

| id | source | inputs | value | tol | status |
|---|---|---|---|---|---|
| L1–5 | A. Lewis, Wilmott forum thread 90957, via QuantLib `testAlanLewisReferencePrices` | LW; $T=1$, $F=100e^{-0.01}$, $DF=e^{-0.01}$; $K=80,90,100,110,120$ | C: 26.774758743998854, 20.933349000596710, 16.070154917028834, 12.132211516709845, 9.024913483457836; P: 7.958878113256768, 12.017966707346305, 17.055270961270109, 23.017825898442801, 29.811026202682472 | $10^{-9}$ | verified in the QuantLib source (it checks rel. $10^{-12}$); Lewis's post not seen; table parity checked by hand at $K=80,100$ ($\sim10^{-14}$) |
| FO1 | F&O (2008) §5.3, Table 4 | FO; $F=K=100$, $DF=1$, $T=1$ | 5.785155450 (MPRA preprint: 5.785155435) | $5\cdot10^{-8}$ | verified in both versions |
| FO2 | F&O (2008) Table 5 | FO; $F=K=100$, $DF=1$, $T=10$ | 22.318945791 | $5\cdot10^{-8}$ | verified |
| A1 | AMST (2007) Table 1, row H2, % of spot | FO; $K=S_0=100$, $r=2.5\%$, $q=0$: $F=100e^{0.025T}$, $DF=e^{-0.025T}$ | T=1: 7.27; 2: 11.73; 3: 15.48; 5: 21.75 (form 4 gives 21.70); 10: 33.84 (form 4: 29.83) | 0.005 | values verified; "ATM = $K=S_0$" is inferred ($K=F$ would give 5.785 at $T=1$ by FO1) |
| A2 | AMST (2007) Table 2, row H2 | AT2; same $K$, $r$, $q$ as A1 | T=1: 8.8948; 2: 13.20; 3: 16.79 | $5\cdot10^{-5}$ / 0.005 | verified; same $K$ caveat |
| KJ | Kahl–Jäckel (2005), via QuantLib `testKahlJaeckelCase` | $F=100$, $DF=1$, $K=200$, $T=10$, $v_0=\theta=0.16$, $\kappa=1$, $\xi=2$, $\rho=-0.8$ | C = 4.95212 | $2\cdot10^{-5}$ | QuantLib source only, primary not seen. Stress case ($\nu_F=0.08$), needs $N\approx1024$ |

The MC note's Andersen sets (A-I, A-II, A-III) and Broadie–Kaya sets (BK-1, BK-2) are also valid COS/Lewis checks; their values are in heston-mc.md §5.

## 6. Remarks

1. **What the parameters do to the smile.** Heston is Black–Scholes with a random variance budget. Leverage ($\rho<0$: vol rises as spot falls)
   tilts the smile into a skew at first order in $\rho\xi$. Vol-of-vol fattens both tails (curvature, second order). Mean reversion $\kappa$ sets how fast
   the skew flattens with maturity, and $v_0$ versus $\theta$ sets the ATM term structure. With $\xi\to0$ you get Black at the average variance.
2. **Why Fourier, and what the "trap" is.** The model is affine, so the characteristic function is known in closed form and the density is not.
   COS reads the density's cosine coefficients straight off the CF and converges exponentially because the density is smooth. Price puts,
   because their payoff is bounded; calls follow from parity. Lewis's single integral is an independent check. The trap: two algebraically
   equal formulas differ in whether the complex logarithm winds around its branch cut at long maturities, and only one is continuous.
3. **Feller and fat tails.** Feller is about whether variance can touch zero, a path property: it matters for simulation, not for the transform.
   SPX fits usually violate it, and forcing it would wreck the short-dated skew. High vol-of-vol makes some moments infinite in finite time
   (moment explosion). That fixes the wing slopes (Lee's moment formula) and decides which Fourier contours and payoffs are safe.

## 7. References

- Heston, S.L. (1993). A closed-form solution for options with stochastic volatility with applications to bond and currency options. *RFS* 6(2).
- Albrecher, H., Mayer, P., Schoutens, W., Tistaert, J. (2007). The little Heston trap. *Wilmott Magazine*, Jan., 83–92 (author version of 11 Sep 2006 read).
- Fang, F., Oosterlee, C.W. (2008). A novel pricing method for European options based on Fourier-cosine series expansions. *SIAM J. Sci.
  Comput.* 31(2), 826–848 (published version and MPRA preprint 7700 read).
- Le Floc'h, F. (2020). More robust pricing of European options based on Fourier cosine series expansions. arXiv:2005.13248v2.
- Lewis, A.L. (2000). *Option Valuation under Stochastic Volatility*. Finance Press; Lewis (2001), SSRN 282110 (neither read directly).
- Lipton, A., Gal, A., Lasis, A. (2014). Pricing of vanilla and first-generation exotic options in the local stochastic volatility framework. *Quant. Finance* 14(11); arXiv:1312.5693.
- Benhamou, E., Gobet, E., Miri, M. (2010). Time dependent Heston model. *SIAM J. Financial Math.* 1, 289–325 (via Gobet's Bachelier 2010 slides).
- Andersen, L., Piterbarg, V. (2007). Moment explosions in stochastic volatility models. *Finance Stoch.* 11, 29–50 (not read).
- Keller-Ressel, M. (2011). Moment explosions and long-term behavior of affine stochastic volatility models. *Math. Finance* 21(1); arXiv:0802.1823v2.
- Lord, R., Kahl, C. (2010). Complex logarithms in Heston-like models. *Math. Finance* 20(4) (cited via F&O; not read).
- Kahl, C., Jäckel, P. (2005). Not-so-complex logarithms in the Heston model. *Wilmott Magazine*, Sept. (value via QuantLib).
- Gatheral, J. (2006). *The Volatility Surface*. Wiley (Table 3.2 parameters via Keller-Ressel).
- QuantLib `test-suite/hestonmodel.cpp`; SciPy `integrate.quad_vec` documentation (both accessed 2026-11-06).
