# AI/ML-Augmented Black-Scholes Framework

**Three-month student project** - Department of Mathematics, BITS Pilani, Hyderabad Campus
**Student:** Sanskar
**Primary reference:** Carol Alexander, *Market Risk Analysis, Volume I: Quantitative Methods in Finance* (Wiley, 2008)

---

## Central research question

> Can machine-learning methods augment the classical Black-Scholes framework to improve
> option-pricing accuracy while preserving the mathematical interpretability and market-risk
> sensitivities of the traditional model?

The governing principle of the project is stated in the plan as:

> Do not replace the mathematical model with AI; use AI to identify and learn its residual
> limitations.

Progression: **Mathematics -> Numerics -> Data -> Machine Learning -> Hybrid Pricing -> Market Risk**

---

## Status

| Stage | Content | Status |
|---|---|---|
| August Week 1 | Financial returns, probability, data handling | **Complete** |
| September Week 1 | Brownian motion and geometric Brownian motion | **Complete** |
| September Week 2 | Ito's lemma and the Black-Scholes equation | **Complete** |
| Post-midsem Week 3 | Greeks and implied volatility | **Complete** |
| Post-midsem Week 4 | Binomial pricing and convergence | **Complete** |
| Post-midsem Week 5 | Monte Carlo and three-method cross-validation | **Complete** |
| Post-midsem Week 6 | ML experiment plan, synthetic dataset, regression baselines | **Complete** |
| Post-midsem Week 7 | Tree ensemble and small neural network; Stage Gate 2 | In progress |

Work follows the plan in `docs/Sanskar_Plan_to_15Dec.xlsx`, which carries the status and
evidence link for every task.

**Stage Gate 1** (31 Oct, Greeks and implied volatility) passed on 2 November.
**The computational gate** (3 Nov) passed: Black-Scholes, CRR and Monte Carlo agree across
thirty cases to within known convergence and sampling error, so the machine-learning stage
is cleared to begin.

One item is outstanding: a sample month of NIFTY option data. NSE returns HTTP 403 to
programmatic requests, so it has to be downloaded by hand. See `data_sources.md`.

---

## Repository layout

```
.
├── black_scholes.py        European call/put pricing module (importable)
├── greeks.py               analytic Delta, Gamma, Vega, Theta, Rho
├── implied_volatility.py   Newton + bisection fallback + bracket checks
├── binomial.py             CRR lattice, European and American, two routes
├── monte_carlo.py          risk-neutral terminal sampling, s.e. and 95% CI
├── ml_dataset.py           synthetic dataset, split, and the traceability audit
├── linear_models.py        OLS and Ridge in closed form, scaler, metrics
├── ML_experiment_plan.md   the specification required by master plan 5.31
├── week1.py                August Week 1    - returns, distributions, normality
├── week2.py                September Wk 1   - Brownian motion and GBM
├── week3.py                September Wk 2   - Ito's lemma and Black-Scholes
├── postmidsem_week3.py     Post-midsem Wk 3 - Greeks and implied volatility
├── postmidsem_week4.py     Post-midsem Wk 4 - binomial pricing and convergence
├── postmidsem_week5.py     Post-midsem Wk 5 - Monte Carlo and the three-method grid
├── postmidsem_week6.py     Post-midsem Wk 6 - synthetic dataset and regression
├── week3_evidence.py       regenerates the Stage Gate 1 evidence summary
├── data_sources.md         NSE / RBI sources, field mapping, access notes
├── data/                   cached NIFTY 50 price history
├── figures/                figures from weeks 1-3 (w01_*, w02_*, w03_*)
├── greeks_plots/           sensitivity and validation figures (post-midsem Wk 3)
├── binomial_plots/         convergence and oscillation figures (post-midsem Wk 4)
├── mc_plots/               Monte Carlo convergence and coverage (post-midsem Wk 5)
├── ml_plots/               dataset overview and regression residuals (post-midsem Wk 6)
├── results/                all numerical output as CSV, plus console logs
├── reports/                submitted Word reports and the mid-semester deck
├── notebooks/              Jupyter notebooks and superseded component scripts
└── docs/                   project plans and progress reviews from the supervisor
```

All five pricers take a continuous dividend yield `q`, defaulting to zero, because NIFTY is
a price index whose constituents pay dividends.

Note on naming: `week3.py` is September Week 2 (Ito and Black-Scholes), whereas
`postmidsem_week3.py` is Week 3 of the separate post-midsem plan. The two
numbering schemes come from different plan documents, both in `docs/`.

Scripts live at the repository root deliberately: they write to `data/`, `figures/` and
`results/` using relative paths, so they must be run from the root, and the run commands in
the three submitted reports refer to that location.

---

## Reproducing the results

### 1. Environment

```
pip install -r requirements.txt
```

Developed and run on **Python 3.12** (Windows). The only external data dependency is
`yfinance`, and even that is optional after the first run because prices are cached.

### 2. Run, in order

```
python week1.py             # ~1 min   August Week 1
python week2.py             # ~20 s    September Week 1
python week3.py             # ~3 s     September Week 2
python postmidsem_week3.py  # ~2 min   Post-midsem Week 3
python postmidsem_week4.py  # ~1 min   Post-midsem Week 4
python postmidsem_week5.py  # ~3 s     Post-midsem Week 5
python postmidsem_week6.py  # ~6 s     Post-midsem Week 6
python week3_evidence.py    # ~20 s    Stage Gate 1 evidence summary
```

`week1.py` downloads NIFTY 50 (`^NSEI`, 2015-01-02 to 2024-12-31) on first run and caches it
to `data/nifty50_prices.csv`. Every later run, and both other scripts, read the cache. Results
are therefore reproducible without network access and are unaffected by later data revisions.

Each script opens figure windows and pauses until each is closed. To run unattended, set
`SHOW_PLOTS = False` near the top of the file; figures are still written to `figures/`.

`week2.py` and `week3.py` also mirror their entire console output to
`results/w0*_console_output.txt`, so a complete run can be reviewed as a single file.

### 3. Verify the module independently

```
python black_scholes.py        # parity, bounds, T -> 0 and sigma -> 0 limits
python greeks.py               # Greek identities and the PDE with exact derivatives
python implied_volatility.py   # recovery of a known sigma, accuracy by vega
python binomial.py             # two CRR routes agree; convergence order; American put
python monte_carlo.py          # unbiasedness, s.e. scaling, interval coverage
python ml_dataset.py           # traceability chain, split, sampling distribution
python linear_models.py        # exact recovery, ridge limits, scaling invariance
```

Each runs a self-test over tens of thousands of random parameter sets and reports every
check individually.

---

## Determinism

Every script uses **`SEED = 20260825`**, unchanged since the first week. Given the cached price
file, every script reproduces its published numbers exactly, on any machine.

Reported figures were regenerated from the committed code and match the values quoted in the
reports.

---

## Headline results

### August Week 1 - empirical behaviour of NIFTY 50 returns

| Quantity | Value |
|---|---|
| Observations | 2,458 daily log returns (2015-2024) |
| Annualised volatility | 0.167389 |
| Skewness | -1.407643 |
| Excess kurtosis | 20.413236 |
| Jarque-Bera p-value | 0.000 |
| Days beyond 5 sigma | 10 (against 0.0014 expected under normality) |
| `r ~= R` scaling exponent | 0.500245 +/- 0.000127 (theory 0.5) |

### September Week 1 - Brownian motion and GBM against the data

| Quantity | Value |
|---|---|
| `SD(dW)` scaling exponent | 0.500245 (theory 0.5), R-squared 1.000000 |
| Quadratic variation at n = 100,000 | 0.994982 (theory T = 1) |
| First variation at n = 100,000 | 251.30 and diverging |
| NIFTY vs GBM skewness | -1.408 vs -0.022 |
| NIFTY vs GBM excess kurtosis | 20.413 vs 0.165 |
| Days beyond 4 sigma, NIFTY vs GBM | **15 vs 0** |
| Days beyond 5 sigma, NIFTY vs GBM | **10 vs 0** |

GBM calibrated to the same mean and volatility reproduces neither the tails, the skewness, nor
the volatility clustering.

### September Week 2 - Ito's lemma and Black-Scholes

| Quantity | Value |
|---|---|
| Ito identity for `d(W^2)`, max error | 1.23e-14 |
| Error of ordinary calculus | 1.0012 (equals T, and does not shrink with refinement) |
| `ln S` drift correction, observed vs theory | -0.061385 vs -0.061250 |
| Hedged mean P&L spread across mu | **0.018164** |
| Unhedged mean P&L spread across mu | **26.555024** |
| Black-Scholes PDE residual, max | 3.343e-05 (prices of order 16) |
| Put-call parity, max absolute error | 4.263e-14 over 13,500 grid points |

### Post-midsem Week 3 - Greeks and implied volatility

| Quantity | Value |
|---|---|
| Greek structural identities (20,000 sets) | all at machine precision |
| **PDE residual with ANALYTIC Greeks** | **~1e-14**, against 3.3e-05 by finite differences in Week 2 |
| Finite-difference truncation slope | 2.000 for delta, gamma and vega (theory 2) |
| Optimal step size, delta / gamma | 1e-4 / 1e-2 |
| Gamma error at h = 1e-12 | **7.1e+09** - smaller h is not safer |
| Implied vol recovered | 2,912 of 3,000; 88 rejected as priced at intrinsic |
| Newton / bisection split | 98.8% / 1.2% |
| Max IV error where vega > 1e-2 | 1.96e-12 |
| Max IV error where vega < 1e-8 | 4.44e-04 |
| Recovered smile on synthetic data | flat to 7.4e-15 |

Two findings worth carrying forward. Finite-difference accuracy has an **optimum** in step
size - below it, cancellation destroys the answer, catastrophically so for second
derivatives. And implied volatility is only as well determined as vega allows, since
`delta_sigma ~ residual / vega`; deep-wing implied vols carry far less information than
their decimal places suggest, which matters when IV becomes an ML feature in October.

### Post-midsem Week 4 - binomial pricing and convergence

| Quantity | Value |
|---|---|
| Backward induction vs the binomial-CDF route | agree to 4.1e-12 |
| Measured convergence order in `N` | **-0.9999** (theory -1) |
| Error sign fixed by the parity of `N` | negative for even, positive for odd, 391 of 391 cases |
| American call = European call at `q = 0` | holds exactly |
| American put > European put | early-exercise premium recovered |

Averaging the `N` and `N+1` lattices cancels most of the oscillation, which is what makes
the Week 5 comparison tight.

### Post-midsem Week 5 - Monte Carlo and three-method validation

| Quantity | Value |
|---|---|
| MC at `M = 1e6` vs Black-Scholes | call `z = -0.94`, put `z = -1.59`, both inside the 95% interval |
| Sample parity, `C - P = e^-rT (mean S_T - K)` | **2.8e-14**, exact by construction |
| Standard-error scaling in `M` | **-0.4925** (theory -0.5) |
| Interval coverage, 200 replications | **93.5%** call and put (nominal 95%, s.e. 1.5%) |
| max abs(CRR - BS), 30 cases at `N = 1000` | **3.46e-03** (target < 5e-03) |
| ... averaged over `N`, `N+1` | **1.28e-03** |
| MC within 2 s.e. of Black-Scholes | **29 of 30** (96.7%) |
| Antithetic variates | 1.80x variance reduction |
| Parity and PDE residual at `q = 0.015` | 0.0 and 2.2e-16 |

The two residual differences are of different kinds, and that is the substance of the gate.
CRR error is deterministic: it shrinks predictably with `N` and its sign is known in advance.
Monte Carlo error is random: it does not shrink predictably in any single run, but it is
quantified ahead of time by an interval that never needs the true price. Neither is
unexplained.

### Post-midsem Week 6 - synthetic dataset and regression baselines

20,000 synthetic calls, features `[m, T, r, sigma]`, target `C_BS / K`, 70/15/15 split
stored in the data file. Errors in currency units at `K = 100`.

| Quantity | Value |
|---|---|
| Traceability: recompute the whole chain from the saved CSV | worst **7.8e-14** |
| Put-call parity on the sampled parameters, independent route | 2.8e-14 |
| OLS `[m,T,r,sigma]`, test RMSE / R² | **10.4565** / 0.8230 |
| Ridge, test RMSE | 10.4565 - alpha at the grid floor, **no effect** |
| OLS extended with Greeks, test RMSE | 7.9591 |
| **OLS quadratic in the same four features** | **2.2614** - a 4.6x cut, no new information |
| Black-Scholes benchmark, test RMSE | 1.5e-14 (machine precision) |
| Mean residual: at the money / wings | **+12.85 / -12.49** |
| corr(abs residual, `K*Gamma`) | **+0.235** |

Three things worth carrying forward. The benchmark here is exact, so **no model can beat
it** and a model that appeared to would be a bug; the point of the synthetic stage is a
calibrated failure, not a contest. Ridge does nothing because the error is **bias, not
variance** - a coefficient penalty cannot touch bias. And the failure is the **functional
form, not the features**: a degree-two expansion of the same four variables cuts the error
by a factor of 4.6 without adding any information.

The dataset is calls only. A call and a put at the same parameters share a feature vector
but not a price, so admitting puts makes the target two-valued and drops `R²` from 0.823
to 0.060; puts follow exactly by parity instead. This was caught by measurement rather
than inspection, and the mixed-set number is kept in the results as the evidence.

---

## A note carried forward

The Black-Scholes derivation is mathematically exact **given its assumptions**. Those
assumptions have been measured against market data and each is contradicted:

| Classical assumption | Empirical observation |
|---|---|
| Continuous diffusion, no jumps | 15 days beyond 4 sigma, 10 beyond 5 sigma in 2,458 observations |
| Constant volatility | Volatility clustering: absolute-return autocorrelation 0.31 at lag 1, 0.09 at lag 30 |
| Gaussian log returns | Skewness -1.41, excess kurtosis 20.41 |
| Independent increments | Return magnitudes show persistent dependence |

The purpose is not to reject the model but to record precisely what is being assumed before it
is used as the benchmark. These measured gaps define what the later machine-learning stage is
asked to close.

---

## Files not tracked

The Alexander textbook PDF is present locally as the primary reference but is excluded from
version control for copyright and file-size reasons. See `.gitignore`.
