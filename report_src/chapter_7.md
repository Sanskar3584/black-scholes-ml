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
