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
