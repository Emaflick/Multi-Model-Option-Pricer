# Hands-on: il mini-pricer che scrivi tu (~150 righe, 1.5–2 giorni)

File: `handson/mini_pricer.py` (+ `handson/test_mini.py`). Solo numpy/scipy. Claude Code può spiegarti la sintassi
Python (`/explain`), ma **non scriverti le funzioni**. Alla fine, confronta i tuoi numeri con `src/qpricer`.

| # | Funzione | Cosa impari (domande tipiche) | Check |
|---|---|---|---|
| 1 | `black76(F, K, T, DF, sigma, cp)` | d1/d2, perché si lavora col forward, misura forward | parità put-call < 1e-12 |
| 2 | `greeks(...)` delta, gamma, vega | delta hedging, gamma/theta P&L: ½Γ(ΔS)² − θΔt | differenze finite centrali |
| 3 | `implied_vol(price, ...)` Newton + Brent | perché Newton può divergere (vega→0 nelle ali) | round-trip < 1e-10 |
| 4 | `simulate_gbm(F, sigma, T, n_steps, n_paths, rng)` | log-Euler esatto, vettorizzazione, seed | media di F_T ≈ F (martingala) |
| 5 | `mc_price(payoff_values, DF)` → (prezzo, SE) | SE = std/√N, IC 95%, convergenza N^-½ | vanilla MC vs black76 entro 3SE |
| 6 | antithetic + control variate | perché riducono la varianza, quando l'antitetico non aiuta | fattore di riduzione della varianza |
| 7 | barriera down-and-out MC, con e senza Brownian bridge | bias del monitoraggio discreto, P(crossing)=exp(−2 ln(S_i/B) ln(S_{i+1}/B)/(σ²Δt)) | in+out = vanilla; confronto col closed form della pipeline |
| 8 | Asian aritmetica con CV geometrica | perché la geometrica ha forma chiusa (lognormale) | VR factor |
| 9 | autocallable semplice (osservazioni annuali) | early redemption, perché è short vol / short skew sulla barriera di protezione | prezzo vs pipeline (stesso seed) |

Ordine: 1→5 il primo giorno (è il 60% delle domande), 6→9 il secondo.
