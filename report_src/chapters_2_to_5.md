**AI/ML-Augmented Black-Scholes Framework for Option Pricing and Market-Risk Analysis**

**TECHNICAL REPORT - CHAPTERS 2 TO 5**

**Sanskar Joshi** · Supervisor: Prof. Ashok Kumar Vaikuntam\
BITS Pilani Hyderabad · 27 November 2026

---

### Note on structure

Eight chapters, which is the organisation recommended in section 12.19 of the project plan
with the market-risk and stress-testing chapters removed, those having been descoped on
24 November:

| | |
|---|---|
| 1 | Introduction |
| **2** | **Mathematical Foundation** |
| **3** | **The Black-Scholes Model** |
| **4** | **Numerical Methods and Validation** |
| **5** | **Dataset and Feature Engineering** |
| 6 | Machine-Learning Models |
| 7 | The Hybrid Black-Scholes-ML Framework |
| 8 | Limitations, Conclusions and Future Work |

Chapters 2 to 5 follow. Every number quoted is reproducible from the repository under seed
**20260825**; the script that generates each is named at the end of the chapter that uses it.

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
