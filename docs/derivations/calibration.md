# Calibration to the SPX implied-vol surface: targets, Heston (surface), SABR (per expiry), arbitrage checks

Phase 5 · `src/qpricer/calibration.py`, `scripts/run_calibration.py` · kernels: `bsm.md` (IV eqs. 1, 7, 11–15), `heston.md` (COS eqs. 3, 6–11;
Lewis eq. 14), `sabr.md` (eqs. 2, 5, 9–11, 22). ⚑ = fewer than two external sources checked (§2.8). (der.) = derived here. (est.) = hand
estimate, not measured. Data: `data/processed/market_20261106_2213.csv` (39 kept expiries, $T=0.0190\ldots1.6100$ y; 18 with $T<0.1$, 21 with
$T\ge0.1$) and `build_surface()["surface"]` (7625 OTM quotes, ≈196 per expiry; `surface_*.csv` stores only the stage counts).

## 1. Setup

- **Measure, models.** One $T_j$-forward measure per expiry (numéraire $P(t,T_j)$). Heston as heston.md §1, with $p=(v_0,\kappa,\theta,\xi,\rho)$ shared by all fitted
  expiries; SABR with β = 1 as sabr.md §1, with $q_j=(\alpha_j,\rho_j,\nu_j)$ per expiry. Both are scale-free in $F$ (Heston $P(F,K)=F\,P(1,K/F)$; SABR β = 1 depends on
  $K/F$ only), so model prices are computed **undiscounted and normalised**: $F=1$, $DF=1$, strike $e^{k}$, $k=\ln(K/F_j)$.
- **Market inputs.** $(F_j,DF_j,T_j)$ from parity (D-001, D-004, D-005); eqs. (2)–(3) replace the short-end DF. Quote $i$: expiry $j(i)$, strike $K_i$, OTM type $\vartheta_i$
  (put, $\vartheta=-1$, if $K<F$; call, $\vartheta=+1$, if $K\ge F$), discounted bid/ask/mid $V^b,V^a,V^m$, normalised undiscounted mid $v^m_i=V^m_i/(DF_jF_j)$.
- **Conventions.** σ decimal, 1 vol pt = 0.01; $T$ ACT/365F (D-003); $r=-\ln DF/T$; $s=\sigma\sqrt T$; $d_1=-k/s+s/2$ (bsm.md eq. 1, $x=-k$); normalised vega
  $\mathcal V=\varphi(d_1)\sqrt T$ (bsm.md eq. 7 at $F=DF=1$); total variance $w=\sigma^2T$; forward call delta $\Phi(d_1)$; randomness via `np.random.default_rng(seed)`.
- **Noise model.** $\sigma^m_i=\sigma^{\rm true}_i+\varepsilon_i$, independent, with sd proportional to the quote's IV half-width (real data) or $\sigma_\varepsilon=0.003$ (synthetic gate,
  ROADMAP). Residuals are therefore defined in vol space (§2.3).

## 2. Result

### 2.1 Targets and the short-end discount factor (closes OPEN_ISSUES F5)
**(1)** $\sigma^{m,b,a}_i=\mathrm{IV}(V^{m,b,a}_i;F_j,K_i,T_j,DF_j,\vartheta_i)$ by bsm.md (11)–(15), with half-width $h_i=\tfrac12(\sigma^a_i-\sigma^b_i)$. OTM quotes have zero intrinsic,
so bid and ask IVs exist whenever bid > 0 (D-006).\
**(2)** Zero curve fitted to the identified parity DFs: a WLS fit $\hat r(T)=a+bT$ over $\mathcal J_{\rm id}=\{j:T_j\ge0.1\}$ (21 expiries), with weights $1/(r_{{\rm se},j}^2+s_0^2)$
and $s_0=10$ bp. The floor is needed because the residual-based $r_{\rm se}$ understates the window sensitivity (F5). For $T_j<0.1$ (18 expiries) set
$\widetilde{DF}_j=\exp(-\hat r(T_j)T_j)$; for $T_j\ge0.1$ keep the parity DF.\
**(3)** Forward at fixed $\widetilde{DF}$: a one-parameter OLS of $y=C-P=\widetilde{DF}(F-K)$ on the same used pairs gives $\tilde F_j=\bar K_u+\bar y_u/\widetilde{DF}_j$ and
${\rm se}(\tilde F_j)=s_u/(\widetilde{DF}_j\sqrt{n_u})$, with $s_u$ the residual sd. Record $(\tilde F_j-F_j)/F_{{\rm se},j}$ and $\max|{\rm res}|/h$ under $(\tilde F,\widetilde{DF})$. Redo the
OTM cut with $\tilde F$, because a strike between $F$ and $\tilde F$ changes type.\
**(4)** IV sensitivity of an OTM quote to its inputs (der., §3.1). Discount factor: $\delta\sigma=-\sigma E\,\delta DF/DF$ with $E=b/(s\,b')$ (bsm.md 3, 12). ATM,
$E=1+s^2/12+O(s^4)$; at $|k|=2s$, $s=0.15$, $E\approx0.16$ (est.), so ATM is the worst case. Forward: $\delta\sigma=-\vartheta\,\Phi(\vartheta d_1)/(\varphi(d_1)\sqrt T)\cdot\delta F/F$,
which is $\mp1.2533\,\delta F/(F\sqrt T)$ ATM: a put/call IV split at the money.\
*Size on this snapshot (est.).* The worst parity DF is on 2026-11-19 ($r=16.89\%$, $T=0.0355$). Against $\hat r\approx4.7\%$ it gives $\delta DF/DF\approx0.43\%$,
so $\delta\sigma\approx0.43\%$ of σ (0.065 vol pt at σ = 15%), systematic for the whole smile. The largest F shift is on 2026-12-07, whose window is
unbracketed ($K\in[7425,7800]$, D-004): $\tilde F-F=(F-\bar K_u)(DF-\widetilde{DF})/\widetilde{DF}\approx+0.6$ pt for $\bar K_u\approx7600$, i.e. ≈1.5 $F_{\rm se}$ and a ±0.035
vol-pt put/call split. Any $\hat r$ within ±0.5% of the truth moves IVs at $T<0.1$ by at most 0.01 vol pt, so the curve's form does not matter.

### 2.2 Quote selection and thinning
**(5)** Keep quote $i$ if and only if: it is in `surface` (validator F4: never `otm`); $\sigma^m_i$ is finite (IV status 0, D-013); $V^m_i\ge0.50$, i.e. 10
minimum ticks (SPX ticks are 0.05 below 3.00 and 0.10 otherwise, Cboe); and $|d_1(\sigma^m_i)|\le z_5=\Phi^{-1}(0.95)=1.6449$, i.e. 5Δ put to 5Δ call. The
delta uses the quote's own IV, so the band widens with the skew.\
**(6)** Thinning to $n_t=25$ per expiry. Targets are $z_m=-z_5+2z_5(m-1)/(n_t-1)$; for each target take the unused quote with the nearest $d_1$, and always
add the put and the call nearest $k=0$. If an expiry has fewer quotes, keep them all. *Why 25:* a 5Δ smile carries about 4–5 shape numbers (level,
skew, curvature, two wing slopes), so 25 points give at least 5 per shape number. Adjacent 5-pt strikes ($\Delta k\approx7\cdot10^{-4}\ll s$) quote the same
market-maker surface, so their errors are not independent and add little information; spacing uniform in $d_1$ (0.14 sd apart) makes the iid
assumption behind (12) more credible. Thinning also equalises expiry weights (weeklies list hundreds of strikes), and cost is linear in $n$.
Sensitivity run: $n_t=50$, report $\Delta p$ in SE units.\
**(7)** Sets. Heston: $\mathcal I_H$ = thinned quotes with $T_j\ge0.1$ (21 expiries, ≤ 525 residuals). SABR: $\mathcal I_j$ for every expiry with $n_j\ge8$ and at least 2
quotes on each side of $F$; any other expiry gets no fit and a logged reason. Heston errors on the 18 expiries with $T<0.1$ are reported **out of sample**.

### 2.3 Heston objective (recommended: IV residuals)
**(8)** $r_i(p)=w_i\,[\sigma^H_i(p)-\sigma^m_i]$ for $i\in\mathcal I_H$, with $\sigma^H_i(p)=\mathrm{IV}(v^H_i(p);1,e^{k_i},T_j,1,\vartheta_i)$. $v^H$ comes from COS (heston.md 9–11; puts, with
calls by parity), with $N=1024$ and $L=12$ frozen for the whole run. Weights are $w_i=1/\max(h_i,h_{\min})$ with $h_{\min}=0.001$ (0.1 vol pt); the floor caps the
weight of ultra-tight quotes, and how often it binds is reported. So $|r_i|\le1$ means the model is inside the quote.\
**(9)** Fallback, element by element, when the model IV has status ≠ 0 (D-013): $r_i=w_i(v^H_i-v^m_i)/\mathcal V^m_i$, with $\mathcal V^m$ at $\sigma^m$. Count the fallbacks per
evaluation; the count must be 0 at the reported optimum.\
**(10)** Price-residual equivalence (der., §3.2). With $\Delta=\sigma^H-\sigma^m$: $(v^H-v^m)/\mathcal V^m=\Delta\,[1+d_1d_2\Delta/(2\sigma^m)]+O(\Delta^3)$, where $d_1d_2=k^2/s^2-s^2/4$.
"Price errors weighted by 1/vega²" therefore equal (8) to first order (relative gap ≈2% at the 5Δ edge for Δ = 0.3 vol pt, σ = 0.2, est.). (10) is
cheaper (no inversion), but far from the optimum Δ is not small and it over-weights the wings. **Recommended:** (8) with the fallback (9); run (10)
once as an independent cross-check (§5).\
**(11)** Bounds, in native parameters with no reparametrisation and `x_scale='jac'`: $v_0,\theta\in[10^{-4},1]$; $\kappa\in[10^{-3},20]$ (heston.md pitfall 9); $\xi\in[0.01,5]$;
$\rho\in[-0.99,0]$ (pitfall 12). Keeping ρ ≤ 0 makes the domain condition $\kappa-\rho\xi>0$ (pitfall 5) hold everywhere in the box. Starts: 16 = 2⁴ scrambled
Sobol points (`qmc.Sobol(d=5, scramble=True, rng=…).random_base2(4)`; a power of 2 keeps the balance properties), mapped to $\sqrt{v_0},\sqrt\theta\in[0.08,0.40]$,
$\ln\kappa\in[\ln0.3,\ln8]$, $\ln\xi\in[\ln0.2,\ln3]$, $\rho\in[-0.95,-0.2]$. The box is data-independent and is reused by the synthetic gate.\
**(12)** Solver: `least_squares(method='trf', jac='2-point', diff_step=1e-6, x_scale='jac', bounds, ftol=xtol=gtol=1e-10, max_nfev=2000)`. A start has
converged if and only if status ∈ {1,2,3,4}; $\hat p$ is the argmin of cost over converged starts; the global basin is the set of starts with
cost ≤ cost$_{\min}(1+10^{-4})$. Diagnostics at $\hat p$: $J$ = `res.jac` (weighted); $s^2=2\,{\rm cost}/(n-5)$; $C=s^2(J^\top J)^{-1}$ via SVD; ${\rm SE}_j=\sqrt{C_{jj}}$; correlations
$R_{jk}=C_{jk}/({\rm SE}_j{\rm SE}_k)$ (on real data these are curvature SEs, conditional on the model). Also: condition number of $J\,{\rm diag}|\hat p|$ and its weakest
right singular vector, `active_mask`, `optimality`, $\nu_F=2\kappa\theta/\xi^2$ (reported, never imposed), basin spread in SE units, runtime and nfev per start.\
**(13)** Identifiability (der. from heston.md 16, $\xi\to0$). ATM variance: $\bar v(T)=\theta+(v_0-\theta)\frac{1-e^{-\kappa T}}{\kappa T}=v_0+\tfrac12\kappa(\theta-v_0)T-\tfrac16\kappa^2(\theta-v_0)T^2+\dots$
If $\kappa T_{\max}\ll1$, only $v_0$ and $\kappa(\theta-v_0)$ are identified (a κ–θ ridge along a hyperbola); if $\kappa T_{\min}\gg1$, only θ and $(v_0-\theta)/\kappa$. Separation
needs $\kappa T\approx1$ inside $[0.11,1.61]$ y. ρ–ξ: the skew is first order in ρξ and the curvature second order in ξ, so with a narrow band they come
out anti-correlated.\
**(14)** Heston ATM skew, first order in ξ (der. from heston.md 16): $\psi_H(T):=\partial_k\sigma|_{k=0}=\rho\xi J(T)/(2\bar v^{3/2}T^2)$. For $v_0=\theta$,
$\psi_H(T)/\psi_H(0)=g(\kappa T)$ with $g(x)=2(x-1+e^{-x})/x^2=1-x/3+x^2/12-\dots$ Limits: $\psi_H(0)=\rho\xi/(4\sqrt{v_0})$ ⚑, and $\psi_H\approx\rho\xi/(2\kappa\sqrt\theta\,T)$ for $\kappa T\gg1$.
SPX ATM skew instead follows a power law $\approx T^{-0.4}$ ⚑ (Bayer–Friz–Gatheral fit exponent 0.407 on the 2013-08-14 surface). So from $T=0.019$ to $0.1$
the market skew roughly doubles ($(0.1/0.019)^{0.41}\approx2.0$, est.), while Heston's falls by only $1-g(0.3)/g(0.057)\approx8\%$ at κ = 3 (est.).

### 2.4 SABR per expiry (β = 1)
**(15)** $r_i(q_j)=w_i\,[\sigma_H(k_i;\alpha,\rho,\nu,T_j)-\sigma^m_i]$ for $i\in\mathcal I_j$. $\sigma_H$ is sabr.md (2) with (5), an IV formula, so no inversion is needed. Seeds:
sabr.md (9)–(10), from a weighted quadratic fit on $|k|\le\min(0.1,\text{band edge})$. Bounds: sabr.md (11). Starts: the seed plus 4 Sobol points
($\rho\in[-0.95,0.3]$, $\ln\nu\in[\ln0.1,\ln4]$, α from (9)). Solver as in (12), with the default `diff_step` (closed form, no solver noise). Per-expiry
diagnostics: RMSE, condition number, corr(ρ, ν), active bounds (expect ν large at short T, sabr.md 11), and $1+c_1T$ (sabr.md 22).

### 2.5 Synthetic round-trip gate (tolerances fixed before the noisy fit)
**(16)** Data, labelled `synthetic`, on the real grid $\mathcal I_H$ (its $k_i$, $T_j$, $w_i$): $\sigma^{\rm obs}_i=\sigma^H_i(p^*)+\varepsilon_i$, with $\varepsilon\sim N(0,\sigma_\varepsilon^2I)$,
$\sigma_\varepsilon=0.003$, `default_rng(20261107)`, and the same COS $N$ as the fit. **S1** = GS (heston.md §5; Gatheral 2006 Table 3.2): κ = 1.3253, θ = 0.0354,
ξ = 0.3877, ρ = −0.7165, $v_0$ = 0.0354, $\nu_F=0.62$ (Feller-violating). **S2** (our choice): κ = 3, θ = 0.04, ξ = 0.4, ρ = −0.7, $v_0$ = 0.015, $\nu_F=1.5$
(Feller holds; upward ATM term structure).\
**(17)** Predicted covariance at $p^*$ (no fit needed). $J_\sigma=\partial\sigma^H/\partial p$ by central differences with step $10^{-5}|p_j|$; $W={\rm diag}(w_i^2)$; $A=J_\sigma^\top WJ_\sigma$.
Then $\Sigma_p=\sigma_\varepsilon^2A^{-1}(J_\sigma^\top W^2J_\sigma)A^{-1}$ (sandwich form; $=\sigma_\varepsilon^2(J_\sigma^\top J_\sigma)^{-1}$ if $w\equiv1$), and ${\rm SE}_j=(\Sigma_p)_{jj}^{1/2}$.\
**(18)** Identifiability pre-check: for every $j$, $p^*_j\pm3{\rm SE}_j$ lies inside the bounds (11) and $3{\rm SE}_j\le0.5|p^*_j|$. If it fails, change $p^*$ and record
why, before any noisy fit is run.\
**(19)** Gates. **G1, noise-free** ($\varepsilon=0$, full 16-start pipeline, ftol = xtol = gtol = $10^{-14}$): $\max_j|\hat p_j/p^*_j-1|\le10^{-6}$. **G2, noisy:** $|\hat p_j-p^*_j|\le3\,{\rm SE}_j$
for all $j$, **and** $D^2=\delta p^\top\Sigma_p^{-1}\delta p\le\chi^2_5(0.999)$ (= 20.52; compute with `scipy.stats.chi2.ppf`). **G3:** cost$(\hat p)\le$ cost$(p^*)$, a necessary
condition for a global minimum. **G4:** RMSE$(p^*)$ = RMS(ε) to $10^{-12}$ (plumbing), and $|{\rm RMSE}(\hat p)-\sigma_\varepsilon\sqrt{(n-5)/n}|\le3\sigma_\varepsilon/\sqrt{2n}$. The seed is fixed
and never changed; P(false fail of G2) ≤ $1-0.9973^{10}+2\cdot0.001\approx2.9\%$ (union bound over 2 sets).\
**(20)** SABR analogue on 4 expiries: 2026-11-13, 2026-12-18, 2027-05-21 and 2028-06-16, with $q^*$ = (0.15, −0.7, 2.0), (0.15, −0.7, 1.2),
(0.17, −0.7, 0.7) and (0.19, −0.7, 0.45) (our choice). It uses the pre-check (18), G1 at $10^{-8}$ (closed form), G2 with $\chi^2_3(0.999)$ (= 16.27), G3, and G4
with 3 parameters.

### 2.6 No-arbitrage checks on the market surface (diagnostics only; D-010 stays the only filter) and deliverables
**(21)** Calendar (Gatheral–Jacquier 2014, Lemma 2.1, proportional dividends): $\partial_Tw(k,T)\ge0$ at fixed $k=\ln(K/F_T)$. For each of the 38 consecutive
pairs $(j,j+1)$, the test points are the $k$ of the in-band, unthinned quotes of both expiries inside their common $k$-range. On each expiry,
$w^{m,b,a}=(\sigma^{m,b,a})^2T$ is interpolated linearly in $k$ between its own quotes, with no extrapolation. Mid violation: $w^m_{j+1}(k)<w^m_j(k)$. Strict:
$w^a_{j+1}(k)<w^b_j(k)$. Magnitude: $\Delta\sigma=\sqrt{w^m_j/T_{j+1}}-\sigma^m_{j+1}(k)$, in vol pts.\
**(22)** Butterfly (density $=\partial^2C/\partial K^2\ge0$; Breeden–Litzenberger, Gatheral–Jacquier §2.2). Per expiry, on all `surface` quotes, form undiscounted calls
$c_i=V_i/DF+(F-K_i)^+$ by parity (the same map for bid, ask and mid). For consecutive strikes, $\lambda_i=\frac{K_{i+1}-K_i}{K_{i+1}-K_{i-1}}$ and
$B_i=\lambda_ic_{i-1}+(1-\lambda_i)c_{i+1}-c_i\ge0$. Mid violation: $B^m_i<0$. Strict (tradeable): $\lambda_ic^a_{i-1}+(1-\lambda_i)c^a_{i+1}-c^b_i<0$. Size:
$z_i=-B^m_i/h^B_i$ with $h^B_i=\tfrac12[\lambda_i(a-b)_{i-1}+(a-b)_i+(1-\lambda_i)(a-b)_{i+1}]$ (undiscounted). The sign of $B$ does not depend on DF; $F$ enters only for
triples that straddle $F$.\
**(23)** Report per pair or expiry: points tested, mid and strict violation counts, max Δσ or max $z$, and the list of strict violations; run the same
checks on the fitted SABR smiles (sabr.md 21).\
**(24)** `run_calibration.py` writes: a per-expiry table ($T$, $n_{\rm band}$, $n_{\rm used}$; Heston RMSE in vol pts, unweighted, in sample for $T\ge0.1$ and out of sample
below; weighted RMS; % of quotes inside bid–ask; SABR (α, ρ, ν) with RMSE and flags); the Heston block ($\hat p$, SE, $R$, condition number, $\nu_F$, cost,
active bounds, basin spread, and the 16-start table: start, end, cost, status, nfev, seconds); per-expiry smile figures (mid with ±h bars, Heston,
SABR); and DF(T), r(T) before and after (2). Every number is re-runnable and contains no NaN.

### 2.7 Decision texts for DECISIONS.md
- **D-016 — Calibration targets and Heston maturity window.** *Context:* 7625 OTM quotes; 18 of 39 expiries below 0.1 y, with 5-pt strikes. Heston's
  ATM skew is flat for $T\ll1/\kappa$ (14) while SPX skew keeps steepening; weekly T counts weekends as variance time (D-003). *Decision:* (5)–(7), weights
  (8); Heston on $T\ge0.1$ (21 expiries), out of sample below; SABR on every expiry passing (7). *Alternatives:* all 39 expiries (weeklies dominate and push
  ξ up); $T\ge2$ weeks (33 expiries; run as a sensitivity, report $\Delta p$/SE); a 10Δ band; no thinning. *Consequences:* Heston targets the 0.1–1.6 y range
  that drives the Phase-4 autocallable; its short-end failure is measured, not hidden; the Heston fit never uses a smoothed DF.
- **D-017 — Short-end DF from a parity-fitted zero curve (closes F5).** *Decision:* (2)–(3). *Alternatives:* external OIS curve (breaks D-001); curve DF on
  every expiry (monotone, but it discards well-identified parity DFs, whose $r_{\rm se}$ is ≤ 0.09% on the SPX monthlies with T > 0.3); flat r.
  *Consequences:* IVs at $T<0.1$ move by at most 0.43% of σ (est.) and F by ≲ 1.5 $F_{\rm se}$; a DF step at T = 0.1 can remain and is reported.
- **D-018 — Heston objective and synthetic-gate tolerance.** *Decision:* IV residuals (8)–(9); bounds and 16 Sobol starts (11); TRF settings (12), with COS
  at N = 1024 instead of the Phase-2 N = 256; Feller reported, not imposed; synthetic tolerance (17)–(19), computed before fitting. *Alternatives:*
  vega-weighted prices (10); Feller as a constraint (forces small ξ or large κθ, spoils the skew); LM with an analytic gradient (Cui et al. 2017).

### 2.8 Cross-check status
| eq. | sources | status |
|---|---|---|
| (1), (5) | bsm.md (2 sources); Cboe SPX specification for the tick sizes (read) | 2 / exchange spec |
| (2), (3), (6)–(8), (11), (12), (15) | design choices (D-016 to D-018); `least_squares`, `qmc.Sobol` and trf.py semantics from the SciPy docs/source (read) | n/a |
| (4) | derivation §3.1; consistent with the D-007 estimate | ⚑ der. |
| (10) | derivation §3.2; first-order claim in secondary sources only (arXiv:2512.19611, citing Le Floc'h; lecture notes) | ⚑ |
| (13), (14) | from heston.md (16), itself 2-source. "Flat at short T, ∝1/T at long T" and the 0.407 exponent: Bayer–Friz–Gatheral via search summary and Gatheral's slides. $\rho\xi/(4\sqrt{v_0})$ also attributed to Pirjol–Wang–Zhu (2024) by a search snippet (not read) | ⚑ |
| (17)–(19) | standard Gauss–Newton / sandwich algebra, derivation §3.3; no source read | ⚑ der. |
| (21) | Gatheral–Jacquier (2014) Lemma 2.1 (read); Carr–Madan (2005) (not read) | ⚑ 1 read |
| (22) | Gatheral–Jacquier §2.2, $p=\partial^2C/\partial K^2$ (read); Breeden–Litzenberger (1978) (classical, not re-read) | ⚑ 1 read |

## 3. Derivation

**3.1 Which input moves the IV (4).** IV inversion sees only the undiscounted OTM price $v=V/DF$ and $F$. A DF error rescales $v$: $\delta v/v=-\delta DF/DF$, so
$\delta\sigma=\delta v/\mathcal V_u=-(v/\mathcal V_u)\delta DF/DF$, with $v/\mathcal V_u=b/(\sqrt T\,b')=\sigma E$ (bsm.md 3, 12). ATM, $b(0,s)=(2\pi)^{-1/2}\int_0^se^{-u^2/8}du$ gives
$E=(1-s^2/24)e^{s^2/8}+O(s^4)=1+s^2/12$. In the wing, $b\approx b'\,s/(|d_1||d_2|)$ (Mills ratio), so $E\approx1/(|d_1||d_2|)\approx s^2/k^2$: the price's vol elasticity grows, and a
relative price error matters least there. An F error at fixed $v$ has $\partial v/\partial F=\vartheta\Phi(\vartheta d_1)$ and $\mathcal V_u=F\varphi(d_1)\sqrt T$; setting $\delta v=0$ gives (4). For the
refit, the OLS line passes through $(\bar K,\bar y)$ with $\bar y=DF(F-\bar K)$, so $F=\bar K+\bar y/DF$ and $\tilde F-F=(F-\bar K)(DF-\widetilde{DF})/\widetilde{DF}$. This is small because
the window is centred ($\bar K\approx F$), which is why D-007 found F well identified when DF is not.

**3.2 Price versus IV residuals (10).** Expand Black in σ around $\sigma^m$: $v(\sigma^H)-v(\sigma^m)=\mathcal V\Delta+\tfrac12\mathcal V'\Delta^2+O(\Delta^3)$, with volga/vega $=d_1d_2/\sigma$
(bsm.md §3.3). Dividing by $\mathcal V^m$ gives (10). The Jacobians agree to $O(\Delta)$, so Gauss–Newton SEs coincide at the optimum. Under IV noise the
vega-scaled residual has mean $\tfrac12(d_1d_2/\sigma)\sigma_\varepsilon^2\approx0.006$ vol pt at the 5Δ edge (est.), which is negligible. Far from the optimum the two differ:
in the wing $v$ is convex and steep in σ, so $(v^H-v^m)/\mathcal V^m\gg\Delta$ when $\sigma^H>\sigma^m$, and the wings dominate the early iterations. Vol residuals stay
bounded by the vol range. Using the evaluation loss as the estimation loss (Christoffersen–Jacobs 2004) also favours (8), because the deliverable is
an IV RMSE.

**3.3 Linearised estimator (17), (19).** Write $\sigma^{\rm obs}=\sigma(p^*)+\varepsilon$ and minimise $\|W^{1/2}(\sigma(p)-\sigma^{\rm obs})\|^2$. The linearised normal equations
$J_\sigma^\top W(J_\sigma\delta p-\varepsilon)=0$ give $\delta p=A^{-1}J_\sigma^\top W\varepsilon$ and ${\rm Cov}=\sigma_\varepsilon^2A^{-1}J_\sigma^\top W^2J_\sigma A^{-1}$; since $\delta p$ is Gaussian to first order,
$D^2\sim\chi^2_5$. With $w\equiv1$ the fitted residual is $(I-P)\varepsilon$, with $P$ the rank-5 hat matrix, so $\|r\|^2\sim\sigma_\varepsilon^2\chi^2_{n-5}$: RMSE² has mean $\sigma_\varepsilon^2(n-5)/n$ and RMSE
has sd ≈ $\sigma_\varepsilon/\sqrt{2n}$ (weights change this only at $O(5/n)$). G3: ${\rm cost}(p^*)-{\rm cost}(\hat p)=\tfrac12\|PW^{1/2}\varepsilon\|^2\ge0$. *Noise-free accuracy (G1):* the
residual floor is η ≈ $10^{-13}$ (the IV stop, D-011; COS errors are identical in data and model and cancel), so $|\delta p_j/p^*_j|\approx\eta/s_{\min}(J_\sigma{\rm diag}\,p^*)$. The
pre-check (18) implies $s_{\min}\ge6\sigma_\varepsilon/\sqrt5$, so the floor is ~$10^{-11}$ (est.). G1's $10^{-6}$ leaves room for the optimiser stopping early in weak
directions, where $\|g\|\propto s_{\min}^2|\delta p|$. *Why a priori:* $\Sigma_p$ depends only on $(p^*,\text{grid},w,\sigma_\varepsilon)$, never on the noisy fit.

**3.4 Skew and identifiability (13), (14).** heston.md (16) gives $V_H=V_B(\bar v)+\rho\xi J\,\partial_x\partial_\Sigma V_B$. Dividing by vega gives $\delta\sigma=-\rho\xi J\,d_2/(2\Sigma\sqrt T)$.
With $d_2=-k/s-s/2$, $\partial_k$ at $k=0$ gives $\psi_H$. For $v_0=\theta$, $J=\theta(T-\tau_\kappa)/\kappa$ with $\tau_\kappa=(1-e^{-\kappa T})/\kappa$, which gives $g$; as $T\to0$, $J\to v_0T^2/2$.
Expanding $\bar v(T)$ for small or large κT gives (13). The market skew is a power law with no single time scale, while Heston has one ($1/\kappa$). That is
the reason for the $T\ge0.1$ window: a κ large enough to steepen the weeklies would flatten the long end.

**3.5 Arbitrage checks (21), (22).** *Calendar (Gatheral–Jacquier):* with proportional dividends $X_t=S_t/F_t$ is a martingale, so
$\mathbb E(X_{T_2}-e^k)^+\ge\mathbb E(X_{T_1}-e^k)^+$ (conditional Jensen). The normalised undiscounted call is Black$(k,w)$, increasing in $w$, hence $w$ is non-decreasing.
Linear interpolation errs by at most $\max|\partial_k^2w|\,\Delta k^2/8$; for strike steps up to 100 pt on F ≈ 7400–7900 ($\Delta k\le0.013$) this is below $10^{-5}$ in $w$
(est.). *Butterfly:* $B_i$ is the price of a tent payoff (long λ at $K_{i-1}$, long $1-\lambda$ at $K_{i+1}$, short 1 at $K_i$). The payoff is ≥ 0, so $B_i\ge0$ is static.
Undiscounted parity adds a function linear in $K$ on each side of $F$, which drops out of $B_i$ unless the triple straddles $F$.

## 4. Numerical pitfalls (symptom → fix)

1. **Model IV inversion fails** (price ≤ 10ε·v; COS clipping outside $[a,b]$, heston.md pitfall 13; extreme trial p). NaN would stop TRF → fallback (9),
   counted, 0 at the optimum. TRF rejects non-finite *trial* residuals by shrinking its radius (scipy `trf.py`); the FD Jacobian is not protected.
2. **COS accuracy in the loop.** heston.md (13) gives $N_{\rm req}\approx370$–400 at ξ = 1 for T = 0.11 and 1.61, ≈1000 at ξ = 2 (est.); the IV error (price error/$\mathcal V$)
   grows in the wings → freeze N = 1024; post-fit check max|Δσ| ≤ $10^{-5}$ on $\mathcal I_H$ at N = 4096, L = 16, else refit there. Never adapt N inside the loop.
3. **FD Jacobian noise.** SciPy's default relative step √ε ≈ 1.5·10⁻⁸ meets the IV floor ($10^{-13}$): derivative errors near $10^{-4}$ (relative) in the wings,
   erratic steps, premature gtol stops → `diff_step=1e-6` (optimum ≈ $\sqrt{\eta/|r''|}$, est.); `x_scale='jac'` set explicitly (SciPy 1.16 changed the default).
4. **Local minima, flat valleys.** Different starts end at different p with near-equal cost → 16 starts; report basin spread in SE units and the number
   of distinct minima; never report a single start.
5. **Parameters at bounds** (`active_mask` ≠ 0: ξ → 5, κ → 20 or $10^{-3}$, ρ → 0). SEs invalid; usual cause is misspecification (chasing short-end skew) →
   report, compare with the $T\ge2$-week sensitivity. Widening a bound needs a D-entry *before* the refit, never after a failed gate.
6. **Ill-conditioned $J^\top J$** (cond$(J{\rm diag}\,\hat p)\gtrsim10^6$ (est.), $|R_{jk}|>0.95$): huge or negative variances from a naive inverse → SVD pseudo-inverse;
   report the weakest direction (typically κ–θ or ρ–ξ, (13)); read parameters through identified combinations $\bar v(T)$, $\psi_H(T)$.
7. **Feller violated.** No problem for COS (heston.md pitfall 15); $2\kappa\theta\ge\xi^2$ is not a box constraint and imposing it wrecks the skew → report $\nu_F$.
8. **Short-end DF bias** (F5): systematic IV offset up to 0.43% of σ on a whole weekly smile → (2)–(3).
9. **Put/call IV split at the money.** δF gives a jump of ±1.25 δF/(F√T) between the last put and the first call (4) → compare $\sigma_C$, $\sigma_P$ at strikes with
   both legs (`clean`); they should agree within $h_C+h_P$.
10. **Weight blow-up.** Quotes with tiny $h_i$ dominate → floor $h_{\min}$; report how often it binds.
11. **Strike-density bias.** 5-pt weeklies outnumber long expiries roughly 10:1 → thinning (6), window (7).
12. **Stale quotes re-entering** (reading `otm`, or forgetting both legs of D-005 rejected strikes) → assert $\mathcal I\subset$ `surface`.
13. **Hagan domain.** $1+c_1T\le0$ gives NaN (sabr.md 22); TRF rejects such trial steps, so flag it only at the optimum. ν at its bound of 5 on weeklies →
    report; changing the bound needs a D-entry first.
14. **Calendar artefacts.** Extrapolating w past the last quote invents violations → common k-range only. Consecutive daily weeklies differ by
    ≈ σ²/365 in w, comparable to the half-width in w, so mid violations there are expected noise; judge by the strict count.
15. **Runtime.** No Python loop over quotes → per expiry one φ, one (M×N) product, a vectorised Newton; loop only over expiries and starts. Budget
    (est.): ~5 ms per evaluation, ~6 per TRF iteration, ≲ 1 min for 16 starts; report the measured time.

## 5. Test oracles

- **Synthetic (the gates).** Heston S1 and S2: G1–G4 of (19). SABR: (20). Slow test: 64 noise seeds on S1, with sd($\hat p_j$)/SE$_j$ ∈ [0.73, 1.27] (3σ for a sample
  sd from 64 draws, whose sd is ≈ 1/√126). This validates the linearisation behind the tolerance.
- **Independent methods.** Objective (10) from the same starts: $\hat p^{(10)}$ within 1 SE of $\hat p$, on S1 and on real data (§3.2). Lewis (heston.md 14) at $\hat p$ on
  $\mathcal I_H$: $|\sigma_{\rm COS}-\sigma_{\rm Lewis}|\le10^{-5}$. SABR fitted to the S1 Heston smiles: RMSE should grow with T (diagnostic, not gated).
- **Identities.** Black$(\sigma^m)$ reproduces $V^m$ to $10^{-12}$ (relative). (3) returns $(F,DF)$ unchanged for $T\ge0.1$. The WLS in (2) recovers a synthetic linear r(T)
  exactly from noise-free DFs. RMSE$(p^*)$ = RMS(ε).
- **Arbitrage checks: power and size.** A flat-vol Black surface gives 0 mid violations; the S1 Heston surface (COS) gives 0 strict violations; one mid
  bumped by +3 half-widths is flagged by the butterfly check; two expiries' smiles swapped are flagged by the calendar check; on the real surface,
  same-type adjacent triples give 0 strict butterflies (reproduces validator A1).
- **F5 closure.** Report the used pairs with $|{\rm res}|/h>1$ under $(\tilde F,\widetilde{DF})$, and the put/call IV agreement near the money before and after (2)–(3).
- **Published reference values.** None apply: calibrated parameters are snapshot-specific, and no published study uses this chain. The GS set is
  synthetic truth only. This note quotes no published calibration number.

## 6. Interview angle

1. **Fit in vol space, weight by confidence, and know which input moves the vol.** Vol is the market's normalised price, so vol errors compare like
   with like across strikes and maturities. Price errors divided by vega are the same to first order but blow up in the wings when the model is far
   off. Bid–ask weighting trusts tight quotes more. The forward sets moneyness, so an error in F tilts calls against puts at the money; a discount
   factor only rescales prices, so a 0.4% DF error moves each vol by 0.4% of itself.
2. **Heston has one clock; the market skew has many.** κ is Heston's only time scale: its skew is flat below 1/κ and decays like 1/T beyond, while SPX
   skew keeps steepening as maturity shrinks, roughly as a power law. One parameter set cannot fit weeklies and two-year options, so fit the
   maturities the product depends on and report the rest out of sample. If those maturities do not straddle 1/κ, speed and level of mean reversion
   trade off along a ridge; skew pins ρξ, and only curvature separates ρ from ξ. SABR per expiry fits each smile with three nearly orthogonal knobs,
   but says nothing about how smiles connect across maturities.
3. **A calibration is an estimate with an error bar you can compute before looking.** Multi-start shows whether the minimum is unique; the Jacobian
   gives standard errors and correlations; the synthetic round-trip checks recovery against noise-propagated errors fixed in advance ("recovered"
   means within 3 SE, not "looks close"). Feller is reported, not imposed: it is a property of the variance paths, which matters for Monte Carlo
   but not for the transform, and forcing it ruins the skew fit.

## 7. References

- Gatheral, J., Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces. *Quantitative Finance* 14(1), 59–71; arXiv:1204.0646v4 (§1–2 read).
- Carr, P., Madan, D. (2005). A note on sufficient conditions for no arbitrage. *Finance Research Letters* 2(3), 125–130 (not read).
- Breeden, D., Litzenberger, R. (1978). Prices of state-contingent claims implicit in option prices. *J. Business* 51(4), 621–651.
- Bayer, C., Friz, P., Gatheral, J. (2016). Pricing under rough volatility. *Quantitative Finance* 16(6) (not read; exponent via search summary, Gatheral's slides).
- Gatheral, J. (2006). *The Volatility Surface*. Wiley (Table 3.2 parameters via Keller-Ressel 2011, as in heston.md).
- Christoffersen, P., Jacobs, K. (2004). The importance of the loss function in option valuation. *J. Financial Economics* 72(2), 291–318 (abstract only).
- Cui, Y., del Baño Rollin, S., Germano, G. (2017). Full and fast calibration of the Heston stochastic volatility model. *EJOR* 263(2), 625–638 (abstract only).
- Guillaume, F., Schoutens, W. (2012). Calibration risk: illustrating the impact of calibration risk under the Heston model. *Rev. Deriv. Res.* 15(1), 57–79 (abstract only).
- Pirjol, D., Wang, X., Zhu, L. (2024). Short-maturity asymptotics for VIX and European options in local-stochastic volatility models. arXiv:2407.16813 (abstract only).
- Heston (1993); Fang–Oosterlee (2008); Le Floc'h (2020); Hagan et al. (2002): as cited in heston.md and sabr.md.
- Cboe, SPX options product specifications (tick sizes, AM/PM settlement), cboe.com, accessed 2026-11-07.
- SciPy documentation: `optimize.least_squares`, `stats.qmc.Sobol`; source `scipy/optimize/_lsq/trf.py` (main). All accessed 2026-11-07.

## 8. Errata (implementation, Phase 5)

- **E1, eqs. (8) and (11), pitfall 2: COS settings and bounds box (D-034).** N = 1024, L = 12 frozen is not accurate
  enough. At a plausible SPX point, (0.03, 2, 0.04, 1, −0.7), L = 12 gives an IV error of 9.9e-5 on the real Heston grid,
  so the note's own post-fit check (≤ 1e-5) would fail. Over the bounds box (11), no fixed (N, L) is accurate, and
  Lewis's `quad_vec` fails at 4 corners (v0 = θ = 1e-4, ξ = 5). The hard region is small v0 and θ with small κ and
  large ξ (ν_F ≈ 0). There X is strongly leptokurtic and the cumulant range needs L ≈ 80. The implementation uses
  bounds v0, θ ∈ [0.01, 0.25], κ ∈ [0.5, 20], ξ ∈ [0.01, 3], ρ ∈ [−0.99, 0], with COS pinned at (16384, 64). It is
  verified against Lewis at the 32 corners and 128 Sobol points of that box. The a-priori budget is a price error in
  market-vega units, max |Δv|/𝒱(σ^m) ≤ 1e-5 (measured 9.21e-6). It is not IV accuracy: the direct model-IV error is
  2.5e-3 at the worst corner, where the model vega is tiny. At p̂, COS and Lewis agree in IV to 4.9e-13. The note's
  original box was re-run as an audit (`run_calibration.py --box note`): 20/20 starts reach the same p̂, which lies
  ≥ 38 SE inside the narrowed bounds.
  The start box (11) becomes √v0, √θ ∈ [0.10, 0.40] and κ ∈ [0.5, 8], so that it lies inside the bounds.
- **E2, eq. (7), "≤ 525 residuals".** Thinning (6) also adds the ATM put and call, so an expiry has up to 27 quotes.
  On this snapshot n_H = 565.
- **E3, §2.1 size estimate (est.).** "F by ≲ 1.5 F_se" is exceeded. The measured maximum is |F̃ − F|/F_se = 3.56
  (2026-12-03, +0.65 pt). Four short expiries exceed 2 F_se, because the parity DFs there are low by up to 0.4%. Every
  used pair still lies inside its own bid–ask band under (F̃, D̃F): max |res|/h = 0.834. The IV effect stays below the
  half-widths (D-033).
- **E4, eq. (19), false-fail rate.** "≈ 2.9% (union bound over 2 sets)" covers only the two Heston G2 gates. The whole
  synthetic suite adds 4 SABR G2 gates (≈ 0.9% each) and 6 G4b gates (≈ 0.27% each). Its union bound is ≈ 8%, or
  ≈ 9.5% with the 64-seed linearisation test (D-035).
