# ML Experiment Plan

**Project:** AI/ML-Augmented Black-Scholes Framework for Option Pricing and Market-Risk Analysis
**Author:** Sanskar Joshi · **Supervisor:** Prof. Ashok Kumar Vaikuntam
**Written:** 11 November 2026 · **Plan ref:** master plan sections 5.1 to 5.31, 11.6 and 11.7
**Status:** design document, fixed before any model is fitted

---

## Why this document exists first

Section 5.31 of the master plan is explicit: *"No machine-learning result shall be accepted
without this specification."* Seven questions must have an answer for every model, and the
answers must be fixed before the models are fitted, not written up afterwards to match
whatever came out. This document fixes them.

Everything below applies to the **synthetic** stage, which is Weeks 6 and 7. The real-market
stage changes two things only: the split becomes chronological (section 3.3 below) and the
target becomes a traded price rather than an exact one. Those changes are flagged where they
arise.

---

## 1. The seven-point specification (section 5.31)

| # | Question | Answer for the synthetic stage |
|---|---|---|
| 1 | Input variables | `x = [m, T, r, sigma]`, where `m = ln(S/K)`. Extended set `x+ = [m, T, r, sigma, Delta, Gamma, Vega]` is reported alongside but is **not** the headline model (see section 2.2). |
| 2 | Prediction target | `y = C_BS / K`, the Black-Scholes price scaled by strike. The scaling is argued in section 2.3. |
| 3 | Train / validation / test | A single random 70 / 15 / 15 partition of the synthetic rows, drawn once under seed 20260825 and stored as a `split` column in the dataset. Membership is a property of the data file, not of the script that reads it. |
| 4 | Loss minimised | Mean squared error. OLS minimises it unpenalised; Ridge minimises it plus `alpha * ||beta||^2` with the intercept excluded from the penalty. |
| 5 | Hyperparameters | OLS has none. Ridge has `alpha`, chosen on the **validation** split alone from a logarithmic grid of 25 values from 1e-6 to 1e3. |
| 6 | Financial benchmark | Black-Scholes itself. On synthetic data this is exact, so the benchmark error is zero and the comparison is one-sided by construction. This is a feature of the week, not a flaw: see section 5. |
| 7 | Success criterion | **Not** a small error. The criterion for Week 6 is that the linear model fails *in the places theory says it must* - at short maturity and away from the money, where the pricing surface has the most curvature - and that the failure is visible in the residual structure rather than only in a summary number. |

Points 6 and 7 are the ones most likely to be challenged at the review, so they are argued
in full in section 5.

---

## 2. Features and target

### 2.1 Why log-moneyness rather than S and K separately

Black-Scholes is homogeneous of degree one in `(S, K)`: scaling both by the same positive
constant scales the price by that constant. So the price divided by strike depends on `S`
and `K` only through their ratio. Feeding `S` and `K` as separate features would ask the
model to rediscover a structural identity that is already known, and would let it fit
spurious level effects. Using `m = ln(S/K)` imposes the identity instead of learning it.

The logarithm rather than the raw ratio, because the lognormal transition density is
symmetric in `log S`, so `m` is the variable in which the pricing problem is naturally
symmetric.

### 2.2 Why the Greeks are reported but are not the headline model

The extended set `x+` includes Delta, Gamma and Vega. These are **derivatives of the target**.
Including them is not leakage in the strict sense - they are computable from `x` alone without
knowing `y` - but it hands the model most of the local shape of the function it is supposed to
be learning, and `Delta = dC/dS` in particular carries nearly all the first-order information.
A model given Delta and scoring well has demonstrated very little.

Both feature sets are fitted and reported. The headline model is the basic set. The extended
set is included to quantify exactly how much the Greeks give away, which is itself a result
worth stating.

### 2.3 Why the target is scaled by strike

The same homogeneity argument. With `K` varying over the sample, an unscaled `C_BS` has a
strike-driven scale that the features cannot explain, because `m` is scale-free. Dividing by
`K` removes it. The scaling is exact and invertible, so every reported error can be converted
back to currency units by multiplying by `K`, and the report quotes both.

---

## 3. Splitting

### 3.1 The split for synthetic data

70% training, 15% validation, 15% test, drawn once by a random permutation under seed
20260825. The three sets are disjoint by construction and the partition is written into the
dataset as a column, so every downstream script sees the same split without re-drawing it.

Random splitting is legitimate **here and only here**, because synthetic rows are
independent draws with no time index. There is no later observation that could leak into an
earlier one, because there is no ordering at all.

### 3.2 What the test set is for

The test split is evaluated **once**, after both models are frozen. Section 5.10.3 of the
master plan is explicit that repeated inspection converts the test set into a second
validation set and produces an optimistic estimate. In practice this means:

- Ridge `alpha` is selected on validation, never on test.
- No model choice, feature change, or scaling change is made after a test number is seen.
- The driver script prints test metrics last, in a clearly marked block.

### 3.3 What changes when real data arrives

For market data the split becomes **chronological**, per section 5.11:

```
earlier dates -> train    middle dates -> validation    later dates -> test
t_train  <  t_validation  <  t_test
```

A random split on time-ordered option data leaks, because the same underlying on the same
day appears in both train and test, and because an option's price on day `t+1` is nearly
determined by its price on day `t`. The chronological split is also the honest
representation of the forecasting problem: future information is not available at fit time.

This is recorded now so that the switch is a planned design change rather than a correction
made later.

---

## 4. Scaling and leakage

Features are standardised, `z_j = (x_j - mu_j) / s_j`, with `mu_j` and `s_j` estimated on the
**training split only** and then applied unchanged to validation and test (sections 5.12 and
5.13).

Why it matters here specifically: `T` ranges over roughly `[0.02, 2]` while `r` ranges over
roughly `[0.03, 0.08]`. Ridge penalises `||beta||^2`, so a feature on a small numerical scale
needs a large coefficient to have any effect and is therefore penalised harder. Without
standardisation, `alpha` would be silently selecting *which features to shrink* on the basis
of their units. OLS is invariant to this, which gives a useful check: if standardisation has
been implemented correctly, OLS predictions must be identical with and without it, to
floating-point precision. The driver asserts this.

**Leakage rules adopted for the whole project:**

1. Fit every preprocessing step on training data only. This covers scaling, feature
   selection, PCA and any imputation.
2. Derive nothing from the target. No feature may be a function of `y`.
3. Draw the split before the model exists, and store it with the data.
4. Touch the test set once.
5. For market data, respect time order in the split and in any rolling statistic.

Rule 2 deserves a note. `C_BS` is computed from the same parameters that produce the
features, so in a trivial sense everything here is a function of everything else. The
operative distinction is whether a feature could be computed at prediction time without
already knowing the answer. `m`, `T`, `r` and `sigma` can be. The Greeks can be too, which
is why the extended set is permitted at all - but see section 2.2.

---

## 5. What counts as success this week

This is the part worth being careful about, because the obvious reading is wrong.

On synthetic data the target is generated by the Black-Scholes formula with no noise. The
benchmark is that same formula. So the benchmark has zero error and **no model can beat it**.
Reporting that a regression has a lower error than Black-Scholes would be evidence of a bug,
not of success.

The purpose of the synthetic stage is therefore not to win a comparison. It is to establish
three things:

1. **Traceability.** Every row reproduces its own target from its own stored inputs, to
   machine precision, with no hidden state. The chain `(S, K, T, r, sigma) -> (C_BS, Delta,
   Gamma, Vega) -> x -> y` is auditable row by row. This is the Week 6 project milestone.

2. **A calibrated failure.** The linear model must fail, and the failure must be structured.
   The Black-Scholes surface is convex in `m` and strongly curved near `T -> 0`, so a model
   linear in `[m, T, r, sigma]` cannot reproduce it. The prediction is that residuals are
   large and systematically signed near the money and at short maturity, and small in the
   deep wings where the surface is close to linear. If residuals came out unstructured, the
   data generation would be suspect.

3. **A floor for later weeks.** The linear error is the number the tree ensemble and the
   neural network in Week 7 must beat by a margin that justifies their complexity. Without
   it, "the neural network works well" is unfalsifiable.

Accordingly, the Week 6 criterion is:

> The dataset is fully traceable, and the linear baseline's residuals are concentrated
> where the pricing surface has the most curvature, in the direction convexity predicts.

A quantitative error target is deliberately **not** set for Week 6. Setting one would invite
tuning the synthetic distribution until the target is met, which would make the number
meaningless.

---

## 6. Sampling distribution for the synthetic dataset

| Parameter | Distribution | Range | Reason |
|---|---|---|---|
| `S` | fixed at 100 | - | Homogeneity means only `S/K` matters; varying both is redundant and would disguise it. |
| `K` | log-uniform | `S*exp(-0.6)` to `S*exp(0.6)` | Gives `m` uniform on `[-0.6, 0.6]`, so moneyness is sampled evenly rather than bunching at the money. |
| `T` | log-uniform | 1 week to 2 years | Short maturities matter disproportionately: that is where curvature lives. Uniform sampling would under-represent them. |
| `r` | uniform | 3% to 8% | Spans the plausible Indian policy-rate range over the data window. |
| `sigma` | uniform | 10% to 60% | Spans NIFTY realised volatility including stressed periods. |
| `q` | fixed at 1.5% | - | The NIFTY 50 index dividend yield, carried through from Week 5. |
| type | call and put | 50 / 50 | Both, so parity can be checked inside the dataset. |

Size: 20,000 rows. Large enough that the 15% test split has 3,000 rows and the metric
standard errors are small; small enough to regenerate in seconds.

Log-uniform sampling for `K` and `T` is the one choice here that is not obvious, and it is
deliberate. Under uniform `T` on `[0.02, 2]`, only about 1% of rows would have `T < 0.04`,
which is exactly the region the week is about.

---

## 7. Models and metrics

**Models.** OLS and Ridge, both fitted in closed form from the normal equations rather than
by an iterative solver, so the fit is exact and reproducible to machine precision and there
is no convergence tolerance to argue about. Where scikit-learn is installed, the driver
cross-checks both fits against it and reports the agreement; where it is not, the results are
unchanged. This mirrors the two-route checks used for the binomial lattice in Week 4.

**Metrics.** MAE, RMSE and `R^2`, reported on train, validation and test, in scaled units
(`C/K`) and in currency units at `K = 100`.

`R^2` is reported because the plan asks for it, but it is the least informative of the three
here: with a wide spread of option prices in the sample, a high `R^2` is easy and says little
about whether the model is useful at any particular point on the surface. The residual plots
against `m` and `T` carry the actual finding.

**Diagnostics.** Residuals against `m`, residuals against `T`, residuals against fitted value,
and a binned mean-residual surface over `(m, T)`. The binned surface is the one that shows
the convexity signature directly.

---

## 8. Register of decisions that could be challenged

Recorded now rather than defended later.

| Decision | Alternative | Why this one |
|---|---|---|
| Target `C/K` | raw `C` | Homogeneity; removes a scale the features cannot see. Both reported. |
| `S` fixed at 100 | sample `S` too | Redundant under homogeneity; fixing it makes the redundancy visible rather than hiding it. |
| Random split | chronological | No time index exists in synthetic data. Chronological is adopted the moment real data arrives, per section 5.11. |
| Log-uniform `T` | uniform | Uniform under-samples short maturity, which is where the result is. |
| Greeks reported but not headline | include them | They are derivatives of the target and hand over most of the local shape. |
| No error target for Week 6 | set one | A target would invite tuning the data distribution to meet it. |

---

## 9. What this plan commits to for Week 7

Week 7 adds a tree ensemble and a small neural network (`p -> 32 -> 16 -> 1`, per section
5.20's complexity restriction). The specification above applies unchanged: same features,
same target, same split, same scaler, same once-only test evaluation. Hyperparameters for
both are selected on validation alone.

Stage Gate 2 on 17 November asks whether added model complexity gives a meaningful gain over
regression. The linear numbers produced this week are the baseline that question is measured
against, which is why they are produced before the complex models exist rather than
alongside them.
