**AI/ML-Augmented Black-Scholes Framework for Option Pricing and Market-Risk Analysis**

**TECHNICAL REPORT**

**Sanskar Joshi** · Supervisor: Prof. Ashok Kumar Vaikuntam\
BITS Pilani Hyderabad · 29 November 2026

---

# 1. Introduction

## 1.1 The question

Black-Scholes prices an option from five numbers and a set of assumptions that market data
contradicts. Machine learning can fit an option surface to arbitrary accuracy and has no
notion of what an option price is allowed to be. The question this project asks is whether
the two can be combined so that each supplies what the other lacks:

> **Does a machine-learning correction to Black-Scholes price options better than either
> Black-Scholes or machine learning alone, and does it remain a price while doing so?**

The second half of that sentence is not decoration. A model that fits well and returns a
negative price has not solved the problem; it has moved it somewhere harder to see.

## 1.2 What the project does

The work runs in four stages, each gated on the previous one.

**Establish the benchmark and prove it correct.** Black-Scholes, the Greeks, implied
volatility, a binomial lattice and Monte Carlo, each validated independently and then against
each other. No machine learning is attempted until the three pricing methods agree to within
their own known error.

**Build a dataset whose every row can be audited.** Synthetic first, where the answer is
known exactly, then real NIFTY 50 index options, cleaned through a filter chain that reports
where every discarded row went.

**Fit machine-learning models and ask more of them than accuracy.** OLS, Ridge, a random
forest and a small neural network, compared on error *and* on whether their outputs satisfy
the arbitrage bounds an option price must satisfy.

**Combine them.** The hybrid learns only the Black-Scholes residual
$\varepsilon = C_{market} - C_{BS}$, so the formula supplies the structure and the model
supplies the correction.

## 1.3 The principal results

| | |
|---|---|
| Black-Scholes, binomial and Monte Carlo agree | to within known convergence and sampling error, 30 cases |
| Linear models fail **where curvature predicts** | residual $+12.85$ at the money, $-12.49$ in the wings |
| A model linear in $\ln(S/K)$ is **necessarily concave in $K$** | derived, then measured: predicted $-1.1857\times10^{-2}$, swept $-1.177\times10^{-2}$ |
| The best-fitting model is the worst-behaved | lowest RMSE, 476 negative prices |
| Constraining the output costs nothing | zero negative prices across 8 seeds, accuracy indistinguishable |
| **Residual learning beats Black-Scholes out of sample on real data** | **RMSE 35.00 to 27.33, a 22% reduction** |
| Pure machine learning fails on the same data | best 116.85 against a benchmark of 35.00 |

The last two lines together are the project's answer. The hybrid works; the machine learning
does not work without the formula underneath it.

## 1.4 How this report is organised

Eight chapters. This is the organisation recommended in section 12.19 of the project plan,
with the market-risk and stress-testing chapters removed, those having been descoped on
24 November in favour of completing the documentation.

| | |
|---|---|
| 2 | **Mathematical Foundation** - returns, Brownian motion, Itô, and where geometric Brownian motion leaves the data |
| 3 | **The Black-Scholes Model** - the hedging argument, the closed form, the Greeks, implied volatility and what limits it |
| 4 | **Numerical Methods and Validation** - finite differences, the binomial lattice, Monte Carlo, and the three-method gate |
| 5 | **Dataset and Feature Engineering** - features, the synthetic dataset, leakage, and the real-market data |
| 6 | **Machine-Learning Models** - regression baselines, tree ensemble, neural network, and the financial checks |
| 7 | **The Hybrid Black-Scholes-ML Framework** - residual learning, on synthetic and on market data |
| 8 | **Limitations, Conclusions and Future Work** |

## 1.5 On reproducibility, and one habit worth stating early

Every number in this report is reproducible from the repository under a single fixed seed,
**20260825**, unchanged since the first week. Each chapter names the scripts that generate
its numbers. Every module carries a self-test that runs in seconds and checks its own claims.

One methodological habit shaped the project more than any modelling choice, and it is worth
stating at the front because several results below exist only because of it:

> A number that agrees with theory is not evidence unless something that disagreed with
> theory would have produced a different number.

That rule was adopted after an early claim failed it. The Week 1 report presented the gap
between the arithmetic mean return and $\mu - \sigma^2/2$ as empirical evidence of the Itô
correction; it is in fact a Taylor identity that a deterministic sequence reproduces exactly,
and so could never have come out otherwise. Section 2.4 records it. The same rule later
forced a seed study that withdrew a 21% accuracy claim (section 6.5), and a materiality
threshold that reversed a ranking (section 6.4).

*Chapters 2 to 5 were submitted on 27 November and are reproduced here unchanged.*
---

# 2. Mathematical Foundation

## 2.1 Returns, and why the logarithm

For a price series $S_t$ the simple return is $R_t = S_t/S_{t-1} - 1$ and the log return is
$r_t = \ln(S_t/S_{t-1})$. Log returns are used throughout, for three reasons that matter
later rather than for convention.

They are additive over time, so a $T$-period return is a sum rather than a product, which is
what allows the central limit theorem to be invoked at all. They are the natural variable
for a process whose increments are proportional to its level. And they are unbounded below
in a way that respects the limited liability of the asset: $S_t > 0$ for all $t$ precisely
because $r_t$ is finite.

For small moves the two agree, since $r = \ln(1+R) = R - R^2/2 + O(R^3)$. Measured on 2,458
NIFTY 50 daily returns from January 2015 to December 2024, the maximum absolute discrepancy
is **9.23e-03**, and a log-log fit of the discrepancy against return magnitude gives a slope
of **2.000** against the theoretical 2. The approximation is therefore second order, as the
expansion says, and the agreement of the measured slope with the predicted one is the first
of many such checks in this project: a derivation is not accepted until a number confirms it.

## 2.2 What the NIFTY 50 returns actually look like

| Quantity | Value |
|---|---|
| Observations | 2,459 prices, 2,458 returns |
| Mean log return (daily) | 4.213e-04 |
| Standard deviation (daily) | 1.054e-02 |
| **Annualised volatility** | **16.74%** |
| Annualised drift | 12.02% |
| **Skewness** | **-1.408** |
| **Excess kurtosis** | **20.413** |
| Jarque-Bera $p$ | $< 10^{-16}$ |

Two of these numbers do the work. Skewness of $-1.41$ says large moves are more often down
than up. Excess kurtosis of $20.4$ against zero for a normal distribution says that extreme
moves are vastly more common than a Gaussian allows: there were **15 days beyond four
standard deviations and 10 beyond five** in 2,458 observations, where a normal distribution
predicts roughly 0.2 and 0.001 respectively.

This is recorded at the outset because the entire project is built on a model that assumes
the opposite. The point is not that Black-Scholes is wrong - every model is - but that the
*direction and size* of its wrongness should be measured before it is used as a benchmark,
not discovered afterwards.

## 2.3 Brownian motion

A standard Brownian motion $W_t$ is defined by $W_0 = 0$, independent increments,
$W_t - W_s \sim N(0, t-s)$, and continuous paths. Three properties are used repeatedly.

**Scaling.** $\text{sd}(W_{t+h} - W_t) \propto h^{1/2}$. Simulated and fitted, the exponent
is **0.50024** against a theoretical 0.5.

**Independent increments.** The maximum absolute autocorrelation across lags is
**4.26e-03**, inside the 95% band of **4.38e-03** - that is, indistinguishable from zero at
the sample size used.

**Quadratic variation.** This is the property that makes stochastic calculus different from
ordinary calculus, and it deserves to be stated carefully. For a partition of $[0,T]$ into
$n$ intervals,

$$\sum_{i=1}^{n}\left(W_{t_i} - W_{t_{i-1}}\right)^2 \longrightarrow T \quad \text{as } n \to \infty,$$

**not** to zero. Measured at $n = 10^5$ over $[0,1]$: **0.99498** against a theoretical 1.
The first variation $\sum|W_{t_i} - W_{t_{i-1}}|$ diverges over the same partition, reaching
**251.3** and still growing.

So a Brownian path has finite quadratic variation and infinite first variation. A smooth
function has the reverse. That single difference is why $(dW)^2 = dt$ rather than $0$, and
everything in chapter 3 follows from it.

## 2.4 Geometric Brownian motion and Itô's lemma

Geometric Brownian motion is the solution of

$$dS_t = \mu S_t\,dt + \sigma S_t\,dW_t,$$

and it is chosen because it keeps $S_t$ positive and makes returns rather than price levels
the stationary quantity.

Applying Itô's lemma to $f(S) = \ln S$,

$$d(\ln S_t) = \left(\mu - \tfrac{1}{2}\sigma^2\right)dt + \sigma\,dW_t.$$

The $-\sigma^2/2$ term is the entire content of Itô calculus in one place: ordinary calculus
would give $d(\ln S) = dS/S = \mu\,dt + \sigma\,dW$ and would be wrong. Verified numerically:

| Check | Observed | Theory |
|---|---|---|
| $(dW)^2$ accumulating to $T$ | residual **1.23e-14** | exact |
| $(dS)^2 / (\sigma^2S^2dt)$ at $n = 10^6$ | **0.99751** | 1 |
| Ordinary-calculus error over $[0,T]$ | **1.0012** | $T = 1$ |
| $\ln S$ drift correction | **-0.061385** | $-0.061250$ |

**A correction recorded here because it was made during the project.** The Week 1 report
initially presented the gap between the arithmetic mean return and $\mu - \sigma^2/2$ as
empirical evidence of the Itô correction in market data. That was wrong, and the supervisor
was right to challenge it. The relationship between the arithmetic and geometric mean of any
sequence is a Taylor identity, not a stochastic one: a deterministic alternating sequence
$\pm\sigma$ with no randomness whatever reproduces the same gap to a ratio of **0.999**. The
identity converges to **5.4e-13** by eighth order. It therefore has *zero* power to
discriminate between a stochastic model and any other, and the claim was withdrawn.

The episode is kept in the report because it is the clearest example of a rule adopted
thereafter: a number that agrees with theory is not evidence unless something that disagrees
with theory would have produced a different number.

## 2.5 Where GBM departs from the data

Simulating GBM with the NIFTY's own $\mu$ and $\sigma$ and comparing:

| Property | NIFTY 50 | Simulated GBM |
|---|---|---|
| Skewness | **-1.408** | -0.022 |
| Excess kurtosis | **20.413** | 0.165 |
| Jarque-Bera $p$ | $<10^{-16}$ | 0.236 |
| Days beyond $4\sigma$ | **15** | **0** |
| Days beyond $5\sigma$ | **10** | **0** |

The simulated series is Gaussian because it was built to be. The real series is not, and the
tail columns are the sharpest statement of it: GBM calibrated to the right volatility
produced *no* four-sigma days at all where the market had fifteen.

Volatility clustering is the other visible departure. Absolute-return autocorrelation in the
NIFTY is **0.31 at lag 1** and still **0.09 at lag 30**; under GBM it is zero by
construction.

*Reproduced by `week1.py`, `week2.py`, `week3.py`.*

---

# 3. The Black-Scholes Model

## 3.1 The hedging argument

Consider a portfolio long one option $V(S,t)$ and short $\Delta$ units of the underlying,
$\Pi = V - \Delta S$. Over $dt$,

$$d\Pi = dV - \Delta\,dS = \left(\frac{\partial V}{\partial t} + \tfrac12\sigma^2S^2\frac{\partial^2V}{\partial S^2}\right)dt + \left(\frac{\partial V}{\partial S} - \Delta\right)dS,$$

where the $\tfrac12\sigma^2S^2\partial^2_SV$ term arises only because $(dS)^2 = \sigma^2S^2dt$
rather than zero - section 2.3 again. Choosing $\Delta = \partial V/\partial S$ removes the
$dS$ term entirely. The portfolio is then riskless over the interval and must earn $r$, which
gives the Black-Scholes equation:

$$\frac{\partial V}{\partial t} + \tfrac12\sigma^2S^2\frac{\partial^2V}{\partial S^2} + (r-q)S\frac{\partial V}{\partial S} - rV = 0.$$

**The drift $\mu$ has vanished.** This is the result that makes option pricing possible: two
people who disagree entirely about the expected return of the underlying must still agree on
the option price, because the hedge removes the exposure that the disagreement concerns.

Verified directly. Delta-hedged portfolio outcomes were simulated under a range of $\mu$;
the spread of hedged outcomes across different drifts is **0.075** of a unit of $\mu$, while
the unhedged spread is **26.6** - a factor of 350. Invariance to $\mu$ is therefore a measured
property of the implementation, not only an algebraic one.

## 3.2 The closed-form solution

With a continuous dividend yield $q$,

$$C = Se^{-qT}N(d_1) - Ke^{-rT}N(d_2), \qquad
d_{1,2} = \frac{\ln(S/K) + \left(r - q \pm \tfrac12\sigma^2\right)T}{\sigma\sqrt{T}}.$$

`black_scholes.py` implements this with the degenerate limits handled explicitly rather than
left to produce `nan`: as $T \to 0$ the price tends to intrinsic value, and as $\sigma \to 0$
it tends to the discounted forward intrinsic.

Three structural checks over 20,000 random parameter sets:

| Check | Max absolute error |
|---|---|
| Put-call parity, $C - P = Se^{-qT} - Ke^{-rT}$ | **4.26e-14** |
| Price within the no-arbitrage bounds | exact |
| Monotonicity in $S$, $K$, $T$, $\sigma$ | all hold |

Put-call parity is worth isolating. It is an **arbitrage identity**, derivable without any
model of how $S$ moves, so it tests the implementation independently of whether the
Black-Scholes assumptions hold. That property is used again in chapter 5 on market data,
where the model assumptions certainly do not hold but parity still must.

## 3.3 The Greeks

Analytic expressions for $\Delta$, $\Gamma$, $\nu$, $\Theta$ and $\rho$ are implemented in
`greeks.py`. Four identities hold at machine precision over 20,000 sets:
$\Delta_{call} - \Delta_{put} = e^{-qT}$; $\Gamma$ and $\nu$ identical for calls and puts;
and $\nu = S^2\sigma T\Gamma$.

The strongest check is the PDE itself. Substituting the *analytic* Greeks into the
Black-Scholes equation gives a residual of **3.6e-14**, against **3.34e-05** when the same
substitution is made with finite-difference derivatives. The four orders of magnitude between
those two numbers is the subject of chapter 4.

## 3.4 Implied volatility, and what limits it

Implied volatility inverts the pricing formula for $\sigma$ given a price.
`implied_volatility.py` uses a safeguarded Newton iteration from the Manaster-Koehler
starting point with a bisection fallback and explicit no-arbitrage bracket checks.

The design decision that matters is the convergence criterion. Converging on the *price*
residual is wrong, because near-zero vega makes a small price residual consistent with a
large volatility error. The solver therefore converges on the width of the $\sigma$ interval
and reports

$$\delta\sigma \approx \frac{\text{price residual}}{\nu}$$

as an uncertainty alongside each answer. Over 3,000 random options:

| Quantity | Value |
|---|---|
| Recovered | 2,909 of 3,000 |
| Rejected as priced at intrinsic | 91 |
| Max error where $\nu > 10^{-2}$ | **1.96e-12** |
| Max error where $\nu < 10^{-8}$ | **4.44e-04** |
| Bisection fallback used | 1.07% overall, 28.6% in the lowest vega band |

The eight orders of magnitude between the two error rows is the result: **implied volatility
is only as well determined as vega allows**. The 91 rejections are not failures. At those
parameters the price equals intrinsic value to within double precision, so no finite $\sigma$
exists, and the solver returns the reason rather than a fabricated number. A silently wrong
volatility would propagate into every model fitted on the surface, which is exactly the
failure this project is trying to avoid.

**A correction made during the project.** The bracket tolerances were originally absolute.
At NIFTY index scale, a price three units in the last place below the no-arbitrage bound -
ordinary floating-point rounding - was being rejected as an arbitrage violation. The failure
was reproduced first and then fixed by making the tolerances relative to the price level;
after the fix, 1,990 of 2,000 synthetic index-scale options recover to **2.7e-12** where
vega exceeds 1, and genuine violations are still caught.

*Reproduced by `black_scholes.py`, `greeks.py`, `implied_volatility.py`,
`postmidsem_week3.py`.*

---

# 4. Numerical Methods and Validation

Three independent pricing methods were implemented. The purpose is not redundancy: it is
that **agreement between methods with different error structures is the only practical
evidence that any of them is right**, since all three are approximations to a quantity that
is only known in closed form in the simplest case.

## 4.1 Finite differences, and why a smaller step is not a safer step

A central difference carries truncation error $O(h^2)$ and round-off error $O(\epsilon/h)$,
so total error is V-shaped in $h$ and has an interior optimum. Scanning twelve orders of
magnitude:

| Quantity | Variable bumped | Scaled optimum | Observed | Ratio | Error at $h = 10^{-12}$ |
|---|---|---|---|---|---|
| Delta | $S$ | $S\epsilon^{1/3}$ = 6.06e-04 | 1.00e-04 | 0.17 | **1.28e-02** |
| Gamma | $S$ | $S\epsilon^{1/4}$ = 1.22e-02 | 1.00e-02 | **0.82** | **7.11e+09** |
| Vega | $\sigma$ | $\sigma\epsilon^{1/3}$ = 1.51e-06 | 5.62e-06 | 3.71 | 5.58e-03 |

Measured truncation-branch slopes: **2.000, 2.000, 2.010** against a theoretical 2.

**On the scaling, which was corrected after review.** The reference optima were first drawn
unscaled and appeared to miss by two orders of magnitude. $\epsilon^{1/3}$ and
$\epsilon^{1/4}$ are dimensionless whereas $h$ carries the units of the variable being
perturbed, so the reference must be scaled by that variable. Scaled, it predicts the
observed optimum - the Gamma reference $S\epsilon^{1/4} = 0.0122$ lands essentially on the
observed 0.01, a ratio of 0.82. Gamma is the sharpest test of the three because its optimum
lies furthest from machine epsilon.

The last column is the practical result. At $h = 10^{-12}$, which looks conservative, the
Gamma error reaches $7\times10^{9}$ and the computed value bears no relation to the true
Gamma of 0.015. This matters directly in chapter 7, where hybrid models have no closed-form
derivative and finite differences become unavoidable.

## 4.2 The binomial lattice

A Cox-Ross-Rubinstein lattice was implemented by two independent routes - backward induction
over the tree, and a closed form through the binomial CDF - which agree to **4.1e-12**.

| Quantity | Value |
|---|---|
| Error at $N = 10$ | 2.44e-01 |
| Error at $N = 1000$ | 2.47e-03 |
| **Measured convergence order** | **-0.9999** (theory $-1$) |
| Error sign, even $N$ | negative in **100%** of cases |
| Error sign, odd $N$ | positive in **100%** of cases |

The parity result is the useful one and it is not folklore: across 391 cases the sign of the
error was determined entirely by whether $N$ was even or odd. Averaging the $N$ and $N+1$
lattices therefore cancels most of it, cutting the maximum error from **2.44e-01 to
1.83e-02** for one extra lattice.

American options were implemented through the same lattice. An American call equals its
European counterpart when $q = 0$, and is strictly more valuable when $q > 0$ because early
exercise before a dividend can be optimal. That check is therefore *conditioned* on $q = 0$
rather than removed, since it is a property of the contract and not a defect.

## 4.3 Monte Carlo

For a European payoff only $S_T$ matters and its exact transition density is known, so the
terminal price is sampled directly with no time-stepping:

$$S_T^{(j)} = S_0\exp\left[\left(r - q - \tfrac12\sigma^2\right)T + \sigma\sqrt{T}Z_j\right],
\qquad \hat{C} = e^{-rT}\frac{1}{M}\sum_j\max(S_T^{(j)} - K, 0).$$

Terminal sampling is **unbiased**: the only error is statistical. An Euler path would add
discretisation bias on top and buy nothing.

| Quantity | Value |
|---|---|
| Standard-error scaling in $M$ | **-0.4925** (theory $-0.5$) |
| Interval coverage, 200 replications | **93.5%** calls and puts (nominal 95%) |
| $z$-statistics | mean $-0.018$, sd $1.028$ |
| Internal parity identity | **2.75e-14**, exact by construction |
| Antithetic variance reduction | 1.80x |

Coverage is the check that matters, because the standard error is a **prediction of accuracy
made without knowing the answer**. That it is calibrated is what makes the interval
meaningful on market data, where no analytical benchmark exists.

## 4.4 The three-method gate

Thirty cases, $S \in \{80,\dots,120\} \times T \in \{0.25, 1, 2\}$, calls and puts.

| Criterion | Target | Result | |
|---|---|---|---|
| max abs(CRR $-$ BS), $N = 1000$ | < 5e-03 | **3.46e-03** | pass |
| max abs(CRR averaged over $N$, $N{+}1$) | < 5e-03 | **1.28e-03** | pass |
| Share with abs(MC $-$ BS) < 2 s.e. | about 95% | **96.7%** (29/30) | pass |

The single exceedance, at $z = -2.12$, is the smallest price in the grid and the most deeply
out of the money at the shortest maturity. On thirty cases at a 95% level roughly 1.5
exceedances are expected, so one is not anomalous.

**The two residual errors are of different kinds, and that distinction is the substance of
the result.** CRR error is deterministic: it shrinks predictably with $N$ and its sign is
known in advance from the parity of $N$. Monte Carlo error is random: it does not shrink
predictably in any single run, but it is quantified ahead of time by an interval that never
requires the true price. Neither is unexplained, and both are bounded by quantities derived
before the comparison was run.

*Reproduced by `binomial.py`, `monte_carlo.py`, `postmidsem_week4.py`,
`postmidsem_week5.py`.*

---

# 5. Dataset and Feature Engineering

## 5.1 Features, and why these

The raw pricing variables are $S, K, T, r, \sigma$. The feature vector used is

$$x = [m, T, r, \sigma], \qquad m = \ln(S/K),$$

with target $y = C_{BS}/K$.

Both choices follow from one fact: **Black-Scholes is homogeneous of degree one in
$(S, K)$**. Scaling both by a positive constant scales the price by that constant, so the
price over strike depends on $S$ and $K$ only through their ratio. Feeding $S$ and $K$
separately would ask a model to rediscover a structural identity that is already known, and
would let it fit level effects that cannot exist. Dividing the target by $K$ removes a scale
that the scale-free features cannot explain.

The logarithm rather than the ratio, because the lognormal transition density is symmetric
in $\log S$.

An extended set $x^+ = [m, T, r, \sigma, \Delta, \Gamma, \nu]$ is reported alongside but is
deliberately **not** the headline. The Greeks are derivatives of the target; including them
hands a model most of the local shape of the function it is supposed to be learning. The
measured effect is reported in chapter 6.

## 5.2 The synthetic dataset

20,000 synthetic call options, $S = 100$, $q = 1.5\%$:

| Parameter | Distribution | Range | Reason |
|---|---|---|---|
| $K$ | log-uniform | $100e^{\mp0.6}$ | makes $m$ **uniform** |
| $T$ | log-uniform | 1 week to 2 years | keeps short maturities, where curvature lives |
| $r$ | uniform | 3% to 8% | plausible Indian policy range |
| $\sigma$ | uniform | 10% to 60% | NIFTY realised volatility including stress |

Log-uniform $T$ is the one non-obvious choice. Under uniform $T$ on $[0.02, 2]$ only about
1% of rows would fall below $T = 0.04$; here **15.8%** do, and that region is precisely where
the pricing surface is most curved.

### Why the dataset is calls only

The first version sampled calls and puts equally. That is wrong, and the way it is wrong was
found by measurement rather than inspection.

A call and a put sharing $(S, K, T, r, \sigma, q)$ have the **same feature vector** but
different prices. The target is therefore two-valued at every point in feature space and no
function of $x$ can separate them. Measured: the median $|y_{call} - y_{put}|$ at one and
the same $x$ is **0.2777** against a target spread of **0.2093** - the ambiguity exceeds the
signal, and test $R^2$ for a linear model falls from **0.823** to **0.060**.

An indicator variable would resolve it but would be redundant, because put-call parity makes
the put an exact affine function of the call at the same parameters. Puts are therefore
recovered by parity and add no information. The parity audit does not need them either, since
it is computed from the parameters.

### Traceability

The Week 6 milestone requires a demonstrated chain from input parameters through
Black-Scholes quantities and features to the target. The audit reloads the saved CSV from
disk - which also tests that the file round-trips at full precision - and recomputes every
stage from the stored input columns alone.

| Stage | Max discrepancy |
|---|---|
| $m = \ln(S/K)$ | 5.07e-16 |
| $C_{BS}$ | 6.04e-14 |
| Greeks | 7.82e-14 |
| $y = C_{BS}/K$ | 9.99e-16 |
| **Put-call parity (independent route)** | **2.84e-14** |

The last line carries the weight. Every other check recomputes with the same pricer that
generated the data, so agreement is necessary but not sufficient; a bug inside the pricer
would be invisible to it. Parity is an arbitrage identity rather than a property of the
formula, so it tests the parameters independently of the implementation.

## 5.3 Splitting, scaling and leakage

The split is 70 / 15 / 15, drawn **once** and written into the data file as a column, so
every downstream script sees the same partition without redrawing it. That is what guarantees
the chapter 6 and chapter 7 models are scored on identical test rows.

Features are standardised with $\mu_j$ and $s_j$ estimated on the **training split only**.
A free consistency check comes with it: OLS is invariant to any invertible linear rescaling,
so its predictions must be identical with and without the scaler - which catches a scaler
applied in the wrong direction, a mistake that would otherwise show only as slightly worse
numbers.

**Leakage rules adopted for the project:**

1. Fit every preprocessing step on training data only.
2. Derive nothing from the target.
3. **On market data, never use the option's own implied volatility as a feature.**
4. Draw the split before the model exists and store it with the data.
5. Touch the test set once.
6. On market data, respect time order in the split and in any rolling statistic.

Rule 3 was added after review and is the one that would actually have caused damage. On
synthetic data $\sigma$ is an *input*, drawn before the price exists. On market data there is
no such thing as "the" volatility - there is a traded price, and implied volatility is
obtained by **inverting** it. Using it as a feature would feed a model an invertible
transformation of the answer; the model would score beautifully and would have learned
nothing, and the failure would be invisible in the metrics. Nothing in the symbol $\sigma$
signals that its meaning changes between the two stages, which is why it is a named rule.

The admissible alternatives are **$\sigma_{hist}$**, realised volatility over the previous 21
trading days, which uses no option data at all, and **$\sigma_{atm,prev}$**, the previous
session's at-the-money implied volatility for the same expiry.

For real data the split also becomes **chronological**, $t_{train} < t_{val} < t_{test}$. A
random split on time-ordered option data leaks, both because the same underlying on the same
day would appear on both sides and because an option's price on day $t+1$ is nearly
determined by its price on day $t$.

## 5.4 Real market data: sources and access

| Quantity | Source |
|---|---|
| NIFTY option prices | NSE F&O UDiFF bhavcopy |
| Risk-free rate | RBI 91-day Treasury bill |
| Dividend yield | **per expiry, from the implied forward** - see 5.6 |
| Underlying spot | `UndrlygPric` in the same file |

NSE rejects programmatic requests without a browser session, returning HTTP 403, so files are
downloaded manually. The format changed to UDiFF in July 2024, so the sample window must sit
entirely after that date.

`nse_data.py` resolves every column through an alias table and **fails loudly** if one
cannot be matched, printing the columns actually found. It never guesses by position: a
half-parsed options file that looks plausible is worse than an exception. On the first real
file every expected column name resolved.

## 5.5 Two findings from the raw files

### The expiry-day settlement trap

**On expiry day, NSE writes the underlying's final settlement value into `SttlmPric` for
every expiring contract** - the same number for calls and puts at all strikes. It is not an
option price.

$C - P$ is then identically zero across that series, so the search for the reference strike
(the minimiser of $|C - P|$) selects an arbitrary one, and the implied forward returns that
strike:

| | Before | After |
|---|---|---|
| Implied forward, expiring series | **27,900** (spot 25,350) | refused |
| Parity residual, median | **5,675** index points | **5.84** |

What makes this worth reporting is that **nothing looked wrong**. The file passed every other
quality check, the forward was a plausible-looking number, and only the parity residual gave
it away. Three fixes were applied: refuse any expiry with $T \le 0$; refuse any expiry whose
settlement price is constant across strikes, which is the signature itself; and choose the
reference strike only among pairs where both legs actually traded.

### The moneyness band alone is insufficient

A deep out-of-the-money put worth **Rs 0.60** converts by parity to a deep in-the-money call
worth **Rs 6,453** - exact arithmetic, still carrying Rs 0.60 of information. Implied
volatility dispersion by quoted premium:

| Premium (Rs) | $n$ | IV median | IV IQR |
|---|---|---|---|
| 1 - 2 | 37 | **44.77%** | 7.36 pp |
| 2 - 5 | 138 | **28.18%** | 10.55 pp |
| 5 - 20 | 244 | 20.02% | 5.85 pp |
| 20 - 100 | 404 | 16.63% | 3.09 pp |
| > 100 | 548 | **14.43%** | **1.91 pp** |

The 1 to 5 rupee band is not a smile, it is noise - and it is the chapter 3 vega result in
different clothes. A **minimum quoted premium of Rs 5.00** is applied to the raw quote, set
from the principle that one tick should be at most 1% of the quote. It keeps 85% of rows and
cuts the overall IV interquartile range from 5.15 to 3.61 percentage points.

## 5.6 The cleaning pipeline

`clean_options.py` applies the filters in a specified order, each for a reason:

| # | Filter | Why here |
|---|---|---|
| 1 | per-expiry implied forward | everything downstream is measured against $F$, not spot |
| 2 | maturity $\ge$ 7 days | before anything that uses $T$, so expiry-week distortion cannot spread |
| 3 | traded today, open interest > 0 | before moneyness: a stale wing quote is not evidence about the wings |
| 4 | out of the money, against $F$ | liquidity sits on the OTM side |
| 5 | $\lvert\ln(F/K)\rvert \le 0.30$ | measured against $F$, not spot |
| 6 | premium $\ge$ Rs 5.00 | on the **raw** quote, before conversion |
| 7 | parity conversion of puts | $C = P + e^{-rT}(F - K)$ |
| 8 | no-arbitrage screening | last, on the prices the model will see |

A **filter waterfall** is reported rather than an output row count, because a cleaning script
that reports only how many rows survived is asking to be trusted rather than checked. On the
files processed: 4,408 raw rows to **1,198** clean, zero no-arbitrage failures.

**The validation that matters.** Where the parity-converted put side meets the native call
side, the implied volatilities agree to **0.1 to 0.4 percentage points** on every liquid
expiry. On synthetic input the conversion reproduces the true call price to **0.006 index
points**.

### Why $q$ is taken per expiry

The synthetic stage used a constant $q = 1.5\%$. The market disagrees, systematically:

| Expiry (days out) | 4 | 10 | 18 | 25 | 32 | 46 | 73 |
|---|---|---|---|---|---|---|---|
| Implied $q$ | **+4.10%** | +3.17% | +2.59% | +1.97% | +0.75% | +1.02% | **-0.12%** |

The monotone decline is the signature of **discrete, seasonal dividends**: Indian companies
cluster their ex-dates, so a near-dated forward prices in an imminent cluster and a
longer-dated one averages it away. A continuous yield is a convenient fiction and a single
constant is wrong by more than two percentage points at both ends. $q$ is therefore inverted
from the implied forward per expiry, which also means no borrowed dividend series has to be
trusted.

## 5.7 A finding that constrains the sampling design

The Black-Scholes residual on the real quotes is **99% one-sided**, which a smile correction
would not be - it would be signed by moneyness. Repricing with the *current* at-the-money
volatility separates the components:

| Component | Mean $\lvert\varepsilon\rvert$ | Share |
|---|---|---|
| **Level** - the benchmark is stale | **23.97** | **63%** |
| **Shape** - the smile itself | 13.94 | 37% |

At-the-money implied volatility moved two full points over the ten-day gap between the two
files.

This is a finding about the **sampling cadence**, not about NIFTY. $\sigma_{atm,prev}$ goes
stale at exactly the rate the sample is thinned, and a level move over a week is the
*unsystematic* part of the model error - roughly a random walk, unpredictable from
$[m, T, r]$. Asking a hybrid to learn it would be asking it to forecast volatility, which is
a different and much harder problem.

It is the direct argument for the sampling design finally adopted: **consecutive daily
files**, for which the benchmark is one day old and the stale-level component largely
disappears. $\sigma_{hist}$ is immune to the issue entirely, being computed from the daily
close series regardless of how the option files are sampled.

*Reproduced by `ml_dataset.py`, `nse_data.py`, `clean_options.py`,
`build_real_dataset.py`, `postmidsem_week6.py`.*
---

# 6. Machine-Learning Models

## 6.1 The specification, fixed before any model was fitted

Section 5.31 of the project plan requires an answer, for every model, to seven questions, and
requires them **before** fitting rather than written up afterwards to match the outcome. The
specification was recorded in `ML_experiment_plan.md` on 11 November.

| # | Question | Answer |
|---|---|---|
| 1 | Inputs | $x = [m, T, r, \sigma]$; extended set adds $\Delta, \Gamma, \nu$ |
| 2 | Target | $y = C_{BS}/K$ |
| 3 | Split | 70/15/15, drawn once, stored as a column in the data file |
| 4 | Loss | mean squared error; Ridge adds $\alpha\lVert\beta\rVert^2$, intercept unpenalised |
| 5 | Hyperparameters | Ridge $\alpha$ on 25 log-spaced values, chosen on **validation** only |
| 6 | Benchmark | Black-Scholes itself |
| 7 | Success | **not** a small error - see below |

Point 7 is the one that mattered. On synthetic data the target is generated by the
Black-Scholes formula with no noise and the benchmark is that same formula, so the benchmark
error is machine precision (RMSE $1.5\times10^{-14}$) and **no model can beat it**. A
regression that appeared to would be evidence of a bug. The criterion for the synthetic stage
was therefore that models fail *where theory says they must*, and that the failure be visible
in the residual structure rather than in a summary number.

## 6.2 Regression baselines

OLS and Ridge are solved in closed form from the normal equations by QR least squares, so
there is no convergence tolerance to argue about and the condition number is not squared.
Both are cross-checked against `scikit-learn` where it is installed, agreeing to
$2.8\times10^{-13}$.

**Ridge does nothing.** The penalty $\alpha$ fell to the grid floor of $10^{-6}$ and the test
RMSE matches OLS to eleven significant figures. This is a diagnostic rather than a
disappointment: with 14,000 training rows and four features there is no variance problem to
regularise. The error is **bias**, and a penalty on the coefficients cannot touch bias.

### Where the linear model fails, predicted first

$y = C/K$ is convex in $m$. A least-squares line through a convex function over-predicts in
the interior and under-predicts at both ends - for $y = x^2$ on $[-1,1]$ the fitted line is
$y = 1/3$, so the residual $1/3 - x^2$ is positive in the middle and negative at the ends.
The prediction, written down before fitting, was therefore: mean residual **positive** near
the money, **negative** in both wings, largest at short maturity, and correlated with Gamma.

| $m$ bucket | $n$ | Mean residual |
|---|---|---|
| $(-0.61, -0.4]$ | 3309 | **-11.79** |
| $(-0.4, -0.2]$ | 3318 | +0.28 |
| $(-0.2, -0.05]$ | 2457 | +9.48 |
| $(-0.05, 0.05]$ | 1693 | **+12.85** |
| $(0.05, 0.2]$ | 2556 | +11.26 |
| $(0.2, 0.4]$ | 3332 | +2.94 |
| $(0.4, 0.61]$ | 3335 | **-13.19** |

Negative in both wings, strongly positive through the middle, crossing zero twice. Mean
$\lvert$residual$\rvert$ falls monotonically from 10.10 at the shortest maturities to 7.35 at
the longest, and $\lvert$residual$\rvert$ correlates with $K\Gamma$ at **+0.235** - curvature
the model never received.

### Is it the features or the functional form?

A large error admits two explanations, and they are distinguishable. Expanding the **same
four features** to degree two - products of what is already there, no new information - cuts
test RMSE from **10.4565 to 2.2614**, a factor of 4.6, and raises $R^2$ from 0.823 to 0.992.

If the features were inadequate this could not have happened. The failure is the functional
form. That is the project plan's section 11.7 milestone - *why a linear model cannot
reproduce the full nonlinear Black-Scholes structure* - stated as a measurement.

## 6.3 Tree ensemble and neural network

Both are written out rather than imported, because section 5.31 requires an answer to what is
being minimised with what initialisation, and "the library does it" is not an answer.

**The random forest** uses histogram splits on quantile bins, so a node's split search is a
`bincount` plus a `cumsum` rather than a sort. The gain criterion reduces to two moments,
$\text{sum}_L^2/n_L + \text{sum}_R^2/n_R - \text{sum}^2/n$, because the $\sum y^2$ term is
common to both sides and cancels. It agrees with `scikit-learn`'s `RandomForestRegressor` to
1.8% on a common problem.

**The network** is $p \to 32 \to 16 \to 1$ with ReLU and Adam, the architecture fixed by
section 5.20's restriction on complexity. Its analytic gradients agree with central finite
differences to $2.1\times10^{-10}$ - the only reliable way to catch a backpropagation sign
error, which otherwise still trains, just worse, and is invisible.

Test set, errors in currency units at $K = 100$:

| Model | MAE | RMSE | $R^2$ |
|---|---|---|---|
| OLS / Ridge | 8.9559 | 10.4565 | 0.8230 |
| OLS quadratic | 1.8292 | 2.2614 | 0.9917 |
| Random forest | 0.5001 | 0.8195 | 0.9989 |
| **Neural net** | 0.1829 | **0.2478** | 0.9999 |
| Black-Scholes | $10^{-14}$ | $1.5\times10^{-14}$ | 1.000000 |

Against the **quadratic** baseline - the honest bar, since polynomial regression is nearly
free - the forest is 2.8 times better and the network 9.1 times.

## 6.4 Financial consistency, and why counts rather than averages

Section 5.28 requires predicted prices to be checked against the bounds an option price must
satisfy. These are reported as **counts of violations**, not averages, because an average
hides the failure mode completely: a model can have excellent mean error and still return a
negative price on one point in fifty.

Checks 1 to 3 are pointwise on the test rows. Checks 4 to 6 are statements about the *fitted
function*, so they are swept on a grid holding other inputs fixed, which also probes regions
the test split does not cover.

| Model | Total violations | **Material (> Rs 0.05)** | Negative prices |
|---|---|---|---|
| OLS / Ridge | 11,491 | 1,943 | 679 |
| OLS quadratic | 10,361 | 3,375 | 586 |
| Random forest | **6,655** | **3,612** | **0** |
| Neural net | **20,923** | 1,653 | 476 |
| Neural net (softplus) | 7,927 | **1,059** | **0** |
| **Black-Scholes** | **0** | **0** | **0** |

Black-Scholes is clean on all six checks across roughly 49,000 grid points. That is what
validates the framework: if the true pricer were flagged, the checker would be wrong.

**Three findings.**

**The accuracy ranking and the consistency ranking are opposites.** The neural network has
the lowest error of any unconstrained model and the highest violation count, including 476
negative prices. A single global error statistic would rank it first and say nothing about
them.

**A model linear in log-moneyness is necessarily concave in strike.** With $y = C/K$ linear
in $m$,

$$\frac{d^2C}{dK^2} = \frac{y'' - y'}{K} = -\frac{1}{K}\frac{dy}{dm} < 0,$$

so OLS and Ridge violate the butterfly no-arbitrage condition at *every* grid point. The
fitted $dy/dm = 0.6507$ predicts a worst violation of $1.1857\times10^{-2}$; the sweep found
$1.177\times10^{-2}$. This is a theorem that the checker reproduced, not an empirical
accident.

**Materiality reverses the ordering, which is why the threshold exists.** At a numerical
tolerance the forest looks best behaved and the network worst. Counting only violations
larger than one tick turns that round: the forest becomes the worst (3,612 material, worst
butterfly **Rs 6.04**) because a step function's second difference swings hard at every step,
while the OLS concavity - universal though it is - is entirely sub-tick, the worst butterfly
being worth Rs 0.0074. The theorem holds and nothing tradable follows from it.

## 6.5 A constraint that is free, and a claim that was withdrawn

The negative prices were predicted before any model was fitted: the true price in the deep
wings is near zero and squared-error loss has no reason to prefer a small positive error to a
small negative one. The remedy was committed to in advance as well - fix it in the model
**specification** rather than by clipping outputs. A softplus output head makes $C \ge 0$
structurally impossible to violate.

A single run suggested the constraint also improved accuracy by 21%. **It does not.** Over
eight seeds:

| | Test RMSE | Negative prices |
|---|---|---|
| Unconstrained | $0.2105 \pm 0.0337$ | 136 to 476 |
| **Softplus** | $0.2132 \pm 0.0558$ | **0 in every seed** |

The paired difference is $+0.0027 \pm 0.0740$, softplus better on 5 of 8, paired $t = 0.10$.
The seed-to-seed spread is an order of magnitude larger than the gap claimed, and the
original seed happened to give the unconstrained network its *worst* result of the eight -
the maximally flattering draw. The accuracy claim was withdrawn.

What survives is not statistical at all. Zero negative prices in every seed against 136 to
476 is a property of the functional form: a softplus output cannot emit a negative number.
**Structural guarantees do not have standard errors**, which is exactly why they are worth
more than a mean.

*Reproduced by `linear_models.py`, `tree_models.py`, `neural_net.py`,
`financial_checks.py`, `postmidsem_week6.py`, `postmidsem_week7.py`.*
---

# 7. The Hybrid Black-Scholes-ML Framework

## 7.1 The formulation

Hybrid A, in the project plan's section 5.4, decomposes the market price rather than
replacing the formula:

$$\varepsilon_i = C^{(i)}_{market} - C^{(i)}_{BS}, \qquad x_i \longrightarrow \varepsilon_i,
\qquad \hat{C}_{hybrid} = C_{BS} + \hat{\varepsilon}.$$

Section 5.5 states the purpose exactly: the task is reduced from *learn the complete pricing
function* to *learn only the systematic part of the model error*. Black-Scholes supplies the
structure - discounting, the payoff shape, the bulk of the moneyness dependence - and the
model supplies what the formula misses.

## 7.2 The benchmark volatility, and the trap underneath it

The project plan's section 11.9 milestone is a single sentence and it is the most important
constraint in this chapter:

> The same option's implied volatility shall not be used to define the residual-learning
> benchmark.

The reason is worth being explicit about. If $C_{BS}$ were priced with the option's own
implied volatility then, by the definition of implied volatility, $C_{BS} = C_{market}$, the
residual would be identically zero, and the model would learn nothing. **The metrics would
look perfect**, which is what makes this dangerous rather than merely wrong.

So the benchmark volatility must be computable without the contract's own price.
`hybrid_models.residual_dataset()` refuses a benchmark that reproduces the market price too
closely and names the milestone in the error, so the mistake raises rather than passing
silently.

Two admissible choices were specified: $\sigma_{hist}$, 21-day realised volatility, which
uses no option data at all; and $\sigma_{atm,prev}$, the previous session's at-the-money
implied volatility for the same expiry - observable before the date it is used on, one number
per expiry rather than per contract, and never the contract's own.

## 7.3 A market where the answer is known

On real data the residual is whatever it is, and "the hybrid improved RMSE" could mean it
learned the model error or that it memorised noise. So the first experiment manufactures a
market where the truth is known in closed form: prices generated from a volatility smile,

$$\sigma(m) = \sigma_{atm}\left(1 - 0.35\,m + 0.90\,m^2\right),$$

with the benchmark priced at the flat at-the-money volatility - what a desk carrying one
number per expiry would quote. The residual is then exactly zero at the money, where the
smile touches the benchmark, and grows into both wings.

Test set, 3,000 rows, errors in index points:

| Learner | Pure ML | **Hybrid A** |
|---|---|---|
| OLS | 12.1384 | **1.0243** |
| Random forest | 0.6518 | **0.1116** |
| Neural net | 0.1635 | **0.0572** |
| *Black-Scholes benchmark* | | *1.4218* |

**The hybrid beats the pure model with every learner**, on the same split and the same
hyperparameters. The OLS row makes the point most sharply: a linear map in
$[m, T, r, \sigma]$ cannot represent a price surface at all - 8.5 times worse than simply
quoting the benchmark - but it can represent a smile *correction* well enough to beat it.
Same model, same features; only the target changed.

## 7.4 Market data

Twenty consecutive daily NIFTY bhavcopies, 28 October to 26 November 2026. Cleaning per
chapter 5: 41,176 raw rows to **11,147** clean, zero no-arbitrage failures, of which
**10,264** carry a benchmark volatility.

### Why the sampling cadence was changed, and what it bought

An earlier two-file sample showed the residual was **99% one-sided**, which a smile
correction would not be - it would be signed by moneyness. Repricing with the current
at-the-money volatility separated the components and found that the benchmark being ten days
stale accounted for most of it. Sampling was therefore changed from weekly to **consecutive
daily** files, so the benchmark is one day old instead of ten:

| Component | 10-day gap | **1-day gap** |
|---|---|---|
| Level, stale volatility | 23.97 (63%) | **3.34 (14%)** |
| Shape, the smile | 13.94 (37%) | **19.88 (86%)** |
| One-sided | 99% | 75% |

Smile shape is now 86% of what the hybrid has to learn. This matters because section 5.5 asks
the model to learn *only the systematic part* of the model error, and a volatility level move
over a week is the **unsystematic** part - roughly a random walk, unpredictable from
$[m, T, r]$. Asking a hybrid to learn it would be asking it to forecast volatility, a
different and much harder problem.

### A recommendation that inverted

The two-file analysis recommended preferring $\sigma_{hist}$ over $\sigma_{atm,prev}$,
because realised volatility is immune to the sampling cadence. With daily files that
recommendation is wrong. On the same rows, $\sigma_{atm,prev}$ gives mean
$\lvert\varepsilon\rvert$ of **23.88** against $\sigma_{hist}$'s **33.88**. One-day-stale
implied volatility beats 21-day realised, because realised volatility is backward-looking and
sits systematically below implied. The earlier recommendation was conditional on weekly
sampling and did not survive the change.

## 7.5 Out-of-sample result

The split is **by trading date**, not by row: 12 dates to train, 3 to validate, the last 4
held out. A random row split would put the same contract on the same day on both sides, and
an option's price is nearly determined by its own price a day earlier, so a random split
would measure memory rather than generalisation.

**2,295 quotes on four dates no model saw:**

| Model | Kind | MAE | RMSE | $R^2$ |
|---|---|---|---|---|
| Black-Scholes | benchmark | 22.354 | 35.002 | 0.9993 |
| OLS | pure ML | 333.843 | 473.472 | 0.8663 |
| **OLS** | **hybrid A** | 19.111 | **28.655** | 0.9995 |
| Random forest | pure ML | 46.079 | 116.853 | 0.9919 |
| **Random forest** | **hybrid A** | **17.432** | **27.332** | **0.9996** |
| Neural net | pure ML | 132.690 | 239.542 | 0.9658 |
| Neural net | hybrid A | 88.292 | 150.927 | 0.9864 |

**Residual learning reduces out-of-sample Black-Scholes error by 22%**, from RMSE 35.00 to
27.33.

Mean $\lvert$error$\rvert$ by bucket, Black-Scholes against the best hybrid:

| $m$ bucket | $n$ | BS | Hybrid | Ratio |
|---|---|---|---|---|
| $(-0.31, -0.15]$ | 13 | 22.01 | 21.35 | 0.97 |
| $(-0.15, -0.05]$ | 316 | 10.98 | 12.04 | 1.10 |
| $(-0.05, 0.05]$ | 1186 | 22.05 | 21.19 | 0.96 |
| $(0.05, 0.15]$ | 634 | 25.17 | **11.69** | **0.46** |
| $(0.15, 0.31]$ | 146 | 37.30 | **23.19** | **0.62** |

| $T$ bucket | $n$ | BS | Hybrid | Ratio |
|---|---|---|---|---|
| $(0, 0.05]$ | 520 | 16.15 | 12.90 | 0.80 |
| $(0.05, 0.1]$ | 1029 | 18.16 | **12.27** | **0.68** |
| $(0.1, 0.2]$ | 589 | 24.39 | 20.22 | 0.83 |
| $(0.2, 0.5]$ | 84 | 21.94 | 48.32 | **2.20** |
| $(0.5, 3.0]$ | 60 | 116.22 | **51.32** | **0.44** |

The hybrid wins in 4 of 5 moneyness buckets and 4 of 5 maturity buckets, so the gain is not
one region of the surface. The one clear loss, $T \in (0.2, 0.5]$ with 84 quotes, is the
sparsest maturity bucket in the sample.

### Two results that matter more than the headline

**Pure machine learning fails on the same data.** The best pure model reaches RMSE 116.85
against a benchmark of 35.00 - three times worse than simply using the formula. Without
Black-Scholes supplying the structure, 6,443 training quotes are nowhere near enough to learn
a pricing surface. The hybrid's advantage is not that machine learning is powerful here; it
is that the formula does most of the work and the model is asked for very little.

**The ranking inverts from the synthetic case.** The neural network was the *best* hybrid on
noiseless data (0.0572) and is the *worst* on market data (150.93), while the forest and even
OLS generalise. On a noiseless target capacity wins; on twelve dates of noisy quotes it
overfits. This is the clearest argument in the project for validating on the data you
actually have rather than on the data you can generate.

## 7.6 Does the hybrid remain a price?

| Model | $C \ge 0$ | $C \ge$ intrinsic | $C \le Se^{-qT}$ | Worst |
|---|---|---|---|---|
| Black-Scholes | 0 | 4 | 0 | 17.64 |
| Hybrid A (forest) | **2** | **2** | 0 | 38.35 |

Out of 2,295 test quotes, the hybrid produced two negative prices and Black-Scholes four
breaches of the intrinsic bound - the latter a property of the market quotes rather than of
the formula, since a traded price below intrinsic is a data-quality artefact.

Two violations in 2,295 is small but it is not zero, and it is the same unconstrained-output
failure mode identified on synthetic data in chapter 6. The remedy is known and costs
nothing: a non-negative output head. It was not applied here because the forest, not the
network, was the best hybrid, and a tree cannot emit a negative price *unless* the benchmark
it is added to is already near zero - which is what happened in these two cases.

*Reproduced by `hybrid_models.py`, `build_real_dataset.py`, `postmidsem_week8.py`.*
---

# 8. Limitations, Conclusions and Future Work

## 8.1 Limitations

These are listed before the conclusions deliberately. Several of them bound what the
conclusions are allowed to say.

### The sample is one month and one volatility regime

The real-data result rests on **20 consecutive trading days**, 28 October to 26 November
2026, with four held out for testing. At-the-money implied volatility moved over a range of
only **13.05% to 15.76%** across the whole window, a standard deviation of 0.81 percentage
points.

So the models have seen one state of the world. Nothing here says how the hybrid behaves
when volatility doubles, when the smile steepens in a selloff, or across a regime change -
and chapter 2 established that NIFTY returns have fat tails and clustered volatility, so
those regimes exist and matter. A 22% error reduction measured in a calm month is evidence
that the method works, not evidence of how much it is worth over a year.

This is the single largest limitation of the project and it is a consequence of timing
rather than design: the data window was compressed from twelve months to one month on
24 November so that documentation could be completed before the examination period.

### The test set is four dates

Four held-out dates is enough for a genuine chronological test and not enough for a
confidence interval on the improvement. The 22% figure should be read as a point estimate
with no error bar attached. Chapter 6 shows what happens when a single draw is mistaken for
a robust result; the same caution applies here, and a seed study cannot fix it because the
uncertainty is in the data, not the initialisation.

### Only one hybrid formulation was tested

Hybrid B, the implied-volatility hybrid $x \to \sigma_{IV}$ then $C_{BS}(\hat{\sigma}_{IV})$,
was descoped on 17 November. It is arguably the more interesting of the two, because
predicting a volatility and pricing through the formula guarantees an arbitrage-free output
by construction - which chapter 7 shows Hybrid A does not.

### The benchmark volatility is a choice, and the choice moved

Section 7.4 records that the recommended benchmark volatility inverted when the sampling
cadence changed. That is a reminder that the hybrid's measured advantage is partly an
artefact of how good the benchmark is: a better-specified Black-Scholes benchmark would leave
less residual for the model to capture and would shrink the 22%. The result is therefore
*"residual learning improves on this benchmark"*, not *"on Black-Scholes in general"*.

### Market-risk analysis was not done

Portfolio Greeks, the Taylor-versus-full-repricing comparison and the stress matrix were
descoped on 24 November. The project therefore demonstrates better *pricing* and says nothing
measured about better *risk management*, which was part of the original title.

### Smaller ones

- $r$ is a single constant (6.5%) rather than the RBI 91-day Treasury bill series, which was
  not downloaded. Over a one-month window the error this introduces is small, but it is not
  zero.
- Transaction costs and bid-ask spreads are absent. Settlement prices are used throughout,
  and the parity residuals in chapter 5 are a direct measure of how wide that gap is.
- The Rs 5.00 minimum premium filter was set from a principle and corroborated on two days,
  not validated across the full window.
- American-style early exercise is implemented and tested but unused: NIFTY index options are
  European.

## 8.2 Conclusions

**The three numerical methods agree, and that is what licensed everything after it.**
Black-Scholes, a CRR lattice and Monte Carlo agree across thirty cases to within their own
known error - deterministic $O(1/N)$ for the lattice, statistical $O(1/\sqrt{M})$ for Monte
Carlo. No machine learning was attempted before that gate passed.

**Linear models fail exactly where theory says they must.** The residual is positive at the
money and negative in both wings, correlates with Gamma, and is largest at short maturity -
all predicted before fitting. And a model linear in log-moneyness is *necessarily* concave in
strike, which was derived and then confirmed by a sweep to within 1%.

**The failure is the functional form, not the features.** A degree-two expansion of the same
four variables cuts the error by a factor of 4.6 with no new information.

**Accuracy and financial consistency are different things, and can rank models oppositely.**
The best-fitting unconstrained model on synthetic data was also the worst-behaved, with 476
negative prices out of 3,000. Only counting violations makes that visible, and only applying
a materiality threshold makes the counts interpretable - it reversed which model was worst.

**A structural constraint is free.** A non-negative output head eliminated every negative
price across eight seeds at statistically indistinguishable accuracy. The accuracy gain first
claimed for it did not survive repetition and was withdrawn; the structural guarantee needs
no repetition, because it is a property of the functional form rather than of a fit.

**Residual learning works on real market data.** Out of sample, on four dates no model saw,
the hybrid reduced Black-Scholes RMSE from 35.00 to **27.33**, a 22% improvement, winning in
4 of 5 moneyness buckets and 4 of 5 maturity buckets.

**Pure machine learning does not.** On identical data the best pure model reached 116.85
against the benchmark's 35.00 - three times worse than using the formula alone. This is the
project's central finding. The hybrid's advantage does not come from machine learning being
powerful on this problem; it comes from Black-Scholes doing most of the work and the model
being asked for very little. **The structure is doing more than the capacity.**

**And the ranking inverts between synthetic and real data.** The neural network was the best
hybrid on a noiseless target and the worst on market quotes. Any conclusion drawn only from
synthetic data would have been exactly wrong about which model to deploy.

## 8.3 Future work

In the order I would actually do it.

**Extend the window to twelve months, and report the improvement by volatility regime.**
This is the one change that would most strengthen the conclusions, because it directly
attacks the largest limitation. The pipeline already runs unmodified on any number of files;
the cost is download time.

**Implement Hybrid B.** Predicting implied volatility and pricing through Black-Scholes
yields an arbitrage-free price by construction, which would close the two negative prices in
section 7.6 structurally rather than by a corrective head.

**Add the market-risk stage.** Portfolio Greeks under the hybrid against Black-Scholes, and
whether the pricing difference survives into a stress P&L - the question the original title
poses and this report does not answer.

**Replace the constant $r$ with the RBI 91-day series**, and re-estimate. Cheap, and removes
a known small bias.

**Report the improvement with an error bar**, by repeated chronological splits across a
longer window. The 22% figure deserves the same scepticism that the softplus accuracy claim
received, and over one month it cannot be given it.

## 8.4 A closing note on method

The result I trust most in this report is not the 22%. It is the set of claims that were
written down *before* the evidence arrived and then tested: that linear residuals would be
signed by convexity, that an unconstrained network would price the deep wings below zero,
that daily sampling would collapse the stale-volatility component of the residual. Each was
recorded in advance and each held.

The claims I trust least are the ones that came out of a single run and looked good - a 21%
accuracy gain from one seed, a universal convexity violation that turned out to be entirely
sub-tick, a benchmark recommendation that inverted when the sampling changed. Each was
withdrawn or corrected, and each is recorded in this report rather than removed from it,
because a report that shows only the claims that survived gives no way to judge how hard they
were tested.
