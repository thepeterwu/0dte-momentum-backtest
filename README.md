git clone <https://github.com/thepeterwu/0dte-momentum-backtest>
cd <your-repo-folder>
python -m venv .venv

Activate the environment:
Windows: .venv\Scripts\activate
Mac/Linux: source .venv/bin/activate

Install Dependencies
pip install -r requirements.txt

# Momentum 



# Vol surface

### Project Structure

```text
0dte-momentum-backtest/
│
├── data/
│   └── spx_options_surface.csv       # Raw 1-minute delayed snapshots from IBKR
│
├── vol_surface/                      # Core modeling package
│   ├── __init__.py
│   ├── black76.py                    # Forward extraction, Black-76 pricer, IV root solver
│   ├── smile.py                      # 1D smile spline, wing extrapolation, delta-inversion solver
│   ├── surface.py                    # 2D (Delta, Tau) surface container & interpolation
│   ├── arbitrage.py                  # Butterfly density g(x) & calendar spread auditors
│   └── visualization.py              # 2D smile slices & 3D Plotly surface visualizers
│
├── vol_capture_option_price.py       # existing ib_async data collection loop
└── vol_run_surface_pipeline.py       # Top-level orchestrator & batch processing runner
```
### Summary
End-to-end Python pipeline that captures delayed SPX option chain 1 min snapshots via ib_async and constructs an 
arbitrage-audited 2D Implied Volatility Surface $\Sigma(\Delta, \tau)$. Rather than relying on external interest rate or 
dividend curves, the pipeline extracts synthetic forwards and discount factors directly from liquid near-the-money 
options via Put-Call Parity OLS regression. It then inverts out-of-the-money quotes to Black-76 implied volatilities, 
fits $C^2$ natural cubic splines in Total Implied Variance space $w(x) = \sigma^2(x)\tau$ with linear wing clamping 
bounded by Lee’s Moment Formula ($\vert{}w'\vert{} \le 2$), and resolves the strike-to-delta circularity via fixed-point 
iteration. Every stage is fail-fast and audit-logged, explicitly testing for both cross-sectional butterfly arbitrage 
($g(x) \ge 0$) and cross-maturity calendar spread arbitrage ($\partial_\tau w \ge 0$).

### Data Hygiene & Parity Calibration
- Clean raw quotes: duplicate strike quotes, missing or non-positive bid/ask (IBKR convention for no bids), crossed-spreads, wide spreads for deep ITM strikes
- extract implied forward $F(\tau)$ and discount factor $D(\tau)$ per maturity slice; partition into liquid OTM contracts.
  - Put-Call Parity OLS: \
    $C_{\text{mid}}(K) - P_{\text{mid}}(K) = \beta_0 + \beta_1 K$ \
    $D = -\beta_1 = e^{-r\tau}, \quad F = \frac{\beta_0}{D}$
- Eliminates model risk from guessing SOFR rates or discrete S&P 500 dividend schedules. Restricting to OTM options ($K < F$ Puts, $K \ge F$ Calls) avoids wide ITM bid-ask spreads and early-exercise/liquidity distortions.

### Black-76 IV Inversion
- Invert OTM mid-prices to implied volatility $\sigma_i$; transform strikes to forward log-moneyness $x_i$.
  - $x_i = \ln(K_i / F)$
  - Hybrid Newton-Raphson ($\sigma_{k+1} = \sigma_k - \frac{P(\sigma_k) - P_{\text{mkt}}}{\mathcal{V}(\sigma_k)}$) with Brent’s Method (brentq) fallback.
- Forward log-moneyness $x = \ln(K/F)$ centers ATM at $x=0$ across all maturities, decoupling smile geometry from spot drift. Newton-Raphson provides quadratic convergence near ATM, while bracketed brentq guarantees convergence in low-Vega deep wings. 

### Total Variance Smile & Wing Clamping
- Fit 1D continuous curve per slice; extrapolate wings linearly.
  - Fit Natural Cubic Spline ($w''(x_1) = w''(x_n) = 0$) on Total Implied Variance:$w(x) = \sigma^2(x)\tau$
  - Clamp wings to $w'(x) \in [-2, 0]$ (left) and $[0, 2]$ (right).
- Standard cubic splines extrapolate cubically ($w \propto x^3$), violating Roger Lee’s (2004) Moment Formula ($\limsup_{\vert{}x\vert{}\to\infty} \frac{w(x)}{\vert{}x\vert{}} \le 2$) and causing negative risk-neutral densities ($g(x) \to -\infty$). 
- Fitting $w(x)$ with natural boundaries + linear wings ensures $C^2$ continuity at boundary knots.

### Strike-to-Delta Inversion
- Convert strike-based smiles onto standardized delta pillars $\Delta^* \in \{\pm 10\Delta, \pm 25\Delta, 50\Delta\}$.
- Solved via Fixed-Point Iteration on $w(x)$:
  - $d_1^* = N^{-1}(\Delta_C^*)$
  - $x^{(k+1)} = \frac{1}{2}w(x^{(k)}) - d_1^*\sqrt{w(x^{(k)})}$
- Resolves the circularity where finding $\Delta$ requires $x$, which depends on $\sigma(x)$, which depends on $x$. Standardizing on $\Delta$ holds probability moneyness ($\frac{\ln(F/K)}{\sigma\sqrt{\tau}}$) constant across maturities.
### 2D Surface Assembly & Arbitrage Audit
- Assemble $(\Delta, \tau)$ surface; interpolate across maturities; audit static no-arbitrage conditions.
- Term Structure - Linear in $w(\Delta, \tau)$:
  - $w(\tau) = w(\tau_k) + \frac{\tau - \tau_k}{\tau_{k+1}-\tau_k}(w(\tau_{k+1}) - w(\tau_k))$
  - Butterfly no-arb condition (Gatheral): $g(x) \ge 0$
  - Calendar no-arb condition: $\frac{\partial w}{\partial \tau} \ge 0$
- Interpolating linearly in total variance $w$ (rather than $\sigma$) guarantees piecewise-constant positive forward variance $\sigma_{\text{fwd}}^2 = \frac{w_2 - w_1}{\tau_2 - \tau_1} \ge 0$ and never introduces artificial calendar arbitrage between valid slices.

## Assumptions
1. **Zero Silent Fallbacks:**
   - *Assumption*: Corrupted quotes must never be masked with heuristic defaults (such as assuming $F = S$ or $D = 1.0$).
   - *Reasoning*: On SPX (~7,700), misestimating the forward by even 5 index points shifts every call and put moneyness coordinate $x = \ln(K/F)$, inducing an artificial kink at $x=0$ where OTM puts transition to OTM calls. Every rejected quote or slice is logged with reasons.
2. **ACT/365 Intraday Minute-Precision $\tau$:**
   - *Assumption*: Time-to-maturity is measured continuously in minutes until 16:00 ET settlement ($\tau = \frac{\text{minutes}}{365 \times 1440}$).
   - *Reasoning*: For 0-DTE to 3-DTE options, integer day-count conventions introduce massive step-function distortions in annualized implied volatility $\sigma = \sqrt{w/\tau}$.
3. **Pre-Spline Quote Monotonicity & Minimum-Tick Truncation:**
   - *Assumption*: Deep wing options at the minimum exchange tick (bid <= $0.05) and non-monotonic adjacent mid-quotes are filtered prior to spline fitting.
   - *Reasoning*: Consecutive strikes quoted at the same $\$0.05$ bid require exponentially higher implied variance at further OTM strikes to justify the same dollar price, creating artificial convexity spikes ($w''(x) \gg 0$).

## Limitations

