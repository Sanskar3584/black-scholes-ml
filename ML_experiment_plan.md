# ML Experiment Plan

**Project:** AI/ML-Augmented Black-Scholes Framework for Option Pricing and Market-Risk Analysis
**Author:** Sanskar Joshi · **Supervisor:** Prof. Ashok Kumar Vaikuntam
**Written:** 11 November 2026 · **Plan ref:** master plan sections 5.1 to 5.31, 11.6 and 11.7
**Status:** design document, fixed before any model is fitted
**Revision 2, 11 November 2026**, after the Week 6 review: section 4.1 (which volatility is
the feature on market data) and section 6.2 (how puts enter on market data) are new; the
sampling table in section 6 was stale and is corrected in section 6.1.

---

## Why this document exists first

Section 5.31 of the master plan is explicit: *"No machine-learning result shall be accepted
without this specification."* Seven questions must have an answer for every model, and the
answers must be fixed before the models are fitted, not written up afterwards to match
whatever came out. This document fixes them.

Everything below applies to the **synthetic** stage, which is Weeks 6 and 7. The real-market
stage changes four things, each flagged where it arises:

| Change | Section |
|---|---|
| the split becomes chronological | 3.3 |
| the target becomes a traded price rather than an exact one | 1, row 2 |
| `sigma` can no longer be the contract's own implied volatility | **4.1** |
| calls only becomes out-of-the-money only, converted by parity | **6.2** |

The last two were added after the Week 6 review. Both are cases where a specification that
is correct on synthetic data becomes silently wrong on market data without changing a single
symbol, which is exactly the kind of thing this document exists to catch.

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
3. **On market data, never use the option's own implied volatility as a feature.**
   See section 4.1 - this is the one that would actually have bitten.
4. Draw the split before the model exists, and store it with the data.
5. Touch the test set once.
6. For market data, respect time order in the split and in any rolling statistic.

Rule 2 deserves a note. `C_BS` is computed from the same parameters that produce the
features, so in a trivial sense everything here is a function of everything else. The
operative distinction is whether a feature could be computed at prediction time without
already knowing the answer. `m`, `T`, `r` and `sigma` can be. The Greeks can be too, which
is why the extended set is permitted at all - but see section 2.2.

### 4.1 Which volatility is the feature

*Added 11 November 2026, after the Week 6 review. The plan said `sigma` was a feature and
left it at that, which is fine on synthetic data and silently wrong on market data.*

On synthetic data `sigma` is an **input**: it is drawn first and the price is computed from
it. There is no circularity, and nothing to decide.

On market data there is no such thing as "the" volatility. What exists is a traded price.
Implied volatility is obtained by **inverting that price** through Black-Scholes, so for a
given contract on a given day,

```
sigma_implied  =  f(market price)  =  f(target)
```

Using it as a feature would be feeding the model an invertible transformation of the answer.
The model would score beautifully and would have learned nothing beyond the Black-Scholes
formula it was handed, and the failure would be invisible in the metrics - which is the
dangerous kind.

This is a strict violation of rule 2 and it is now rule 3 in its own right, because it is
not obvious from the symbol `sigma` alone that it has changed meaning between the two
stages.

**What is used instead.** Three admissible candidates, all computable at prediction time
without the contract's own price:

| Candidate | Definition | Comment |
|---|---|---|
| **`sigma_hist`** | realised volatility of NIFTY log returns over the previous 21 trading days, annualised | the primary choice: no option data at all, so it cannot leak |
| `sigma_iv_prev` | the same contract's implied volatility from the **previous** trading day's close | legitimate under rule 6, but carries stale-quote risk and is undefined on a contract's first day |
| `sigma_atm_prev` | previous day's at-the-money implied volatility for the same expiry | a single level per expiry rather than per contract; smoother, and the usual practitioner choice |

**Decision:** `sigma_hist` is the baseline feature. `sigma_atm_prev` is reported as a second
specification, because the gap between them measures how much of the surface is *forward*
information rather than backward-looking realised volatility, which is itself a result worth
having. `sigma_iv_prev` is not used: per-contract lagged IV in the illiquid wings is mostly
stale-quote noise, which Week 3 already quantified.

Both admissible features need a 21-day warm-up before the first usable date, and
`sigma_atm_prev` needs the previous session. The chronological split in section 3.3 must
therefore start at least one month after the raw data window begins. Recorded here so the
window is chosen with the warm-up in mind rather than being trimmed afterwards.

**A note on the residual-learning target (Week 8).** Hybrid A predicts
`epsilon = C_market - C_BS`. Computing `C_BS` needs a volatility, and if that volatility
were the contract's own implied volatility then `C_BS` would equal `C_market` by
construction and the residual would be identically zero. The same rule applies: the
benchmark `C_BS` is computed with `sigma_hist` or `sigma_atm_prev`, never with the
contract's own IV.

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
| `sigma` | uniform | 10% to 60% | Spans NIFTY realised volatility including stressed periods. On market data this feature changes meaning entirely - see section 4.1. |
| `q` | fixed at 1.5% | - | The NIFTY 50 index dividend yield, carried through from Week 5. |
| type | **calls only** | - | Puts follow exactly by parity; including them makes the target two-valued. See section 6.1. |

Size: 20,000 rows. Large enough that the 15% test split has 3,000 rows and the metric
standard errors are small; small enough to regenerate in seconds.

Log-uniform sampling for `K` and `T` is the one choice here that is not obvious, and it is
deliberate. Under uniform `T` on `[0.02, 2]`, only about 1% of rows would have `T < 0.04`,
which is exactly the region the week is about.

### 6.1 Calls only, and why - corrected 11 November 2026

*This table previously said "call and put, 50/50". That was the original design; it was
changed during Week 6 after measurement, and this section was not updated at the time.
The code and the Week 6 report were correct; this document was stale.*

A call and a put sharing `(S, K, T, r, sigma, q)` have the **same feature vector**
`x = [m, T, r, sigma]` but different prices. The target is therefore two-valued at every
point in feature space and no function of `x` can separate the two. Measured on the
dataset: median `|y_call - y_put|` at one and the same `x` is **0.2777**, against a target
spread of **0.2093** - the ambiguity exceeds the signal. Test `R^2` for OLS falls from
**0.823** on calls alone to **0.060** on the mixed set.

An `is_call` indicator would resolve the ambiguity but would be redundant: put-call parity
makes the put an exact affine function of the call at the same parameters, so a model
trained on calls prices puts exactly with no further fitting. Puts add no information and
double the data. The parity audit does not need put rows either, since
`put_call_parity_error` computes both sides from the parameters.

`build_dataset(include_puts=True)` still produces the mixed set; the Week 6 driver uses it
once, as a labelled diagnostic, so the cost is measured in the results rather than asserted
in prose.

### 6.2 How puts enter on market data

*Added 11 November 2026 after the Week 6 review. Calls only is right for synthetic data and
would be wrong for NIFTY, for a reason that has nothing to do with the argument above.*

The synthetic argument assumed the call price is **observable** at every strike. On NIFTY it
is not. Liquidity migrates to whichever side is out of the money: below the spot the traded
contracts are puts, above it they are calls. In-the-money quotes on either side are wide,
stale, or absent. So "use calls" at low strikes would mean using the illiquid side of the
market, and parity would then be importing noise rather than information.

**Convention adopted: out-of-the-money only, converted to a call-equivalent price.**

1. For each (expiry, strike), compute the **implied forward** `F` from the most liquid
   put-call pair for that expiry, by parity: `F = K* + e^{rT}(C* - P*)` at the strike `K*`
   where `|C - P|` is smallest, which is the strike nearest the forward and the one where
   both sides are liquid. The forward is taken from the market rather than from
   `S e^{(r-q)T}`, which would require trusting a borrowed dividend-yield series; the
   difference between the two is worth recording as a data-quality measure.
2. Keep **OTM contracts only**: puts for `K < F`, calls for `K > F`. At `K ~ F` keep the
   side with greater volume.
3. Convert every retained put to its **call-equivalent** price by parity,
   `C = P + e^{-rT}(F - K)`, and model `y = C/K` exactly as on synthetic data.
4. Record the conversion in a `source_side` column, so any result can be re-examined by
   side. If model error turns out to differ systematically between converted puts and
   native calls, that is a data-quality finding, not a modelling one, and it must stay
   visible.

Two consequences follow and are accepted deliberately. The conversion makes each retained
contract's target depend on `F`, which is estimated from a different pair of contracts, so a
mis-estimated forward biases a whole expiry rather than one row; step 1 is therefore checked
per expiry rather than assumed. And the OTM filter interacts with the moneyness filter
`|ln(S/K)| <= 0.3` already proposed in `data_sources.md` section 4 - the two must be applied
in the order OTM first, moneyness second, so the moneyness band is measured against the
implied forward rather than against spot.

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
| Calls only (synthetic) | mix in puts, or add an `is_call` flag | Mixing makes the target two-valued; the flag is redundant under parity. Measured: `R^2` 0.823 to 0.060 (6.1). |
| OTM only, converted by parity (market) | calls only, as synthetic | NIFTY liquidity is on the OTM side; "calls only" at low strikes means using the illiquid side (6.2). |
| `sigma_hist` as the volatility feature | the contract's own IV | The contract's own IV is inverted from the target. Strict leakage (4.1). |
| Implied forward from the market | `S e^{(r-q)T}` | Avoids trusting a borrowed dividend series; the gap between the two measures data quality (6.2). |

---

## 9. What this plan commits to for Week 7

Week 7 adds a tree ensemble and a small neural network (`p -> 32 -> 16 -> 1`, per section
5.20's complexity restriction). The specification above applies unchanged: same features,
same target, same split, same scaler, same once-only test evaluation. Hyperparameters for
both are selected on validation alone. The network is implemented in NumPy with the forward
and backward passes written out and the gradients verified against finite differences, so
every question in section 5.31 has an answer that does not begin "the library does it".

### 9.1 Why RMSE alone will not settle Stage Gate 2

On noiseless synthetic data a tree ensemble or a network will beat the quadratic baseline of
2.2614 without difficulty. That tells us the model has capacity, which was never in doubt;
it does not tell us the model has learned anything a pricing desk could use.

The gate evidence is therefore, for **every** model including the Week 6 baselines:

1. **The same residual tables** by moneyness and by maturity as Week 6, so the structure of
   the error can be compared across models on identical buckets rather than collapsed into
   one number.

2. **Financial consistency checks on the predictions**, per master plan 5.28, reported as
   **counts of violations** rather than averages. An average hides the failure mode
   completely: a model can have excellent mean error and still return a negative price on
   one point in fifty.

   | Check | Condition |
   |---|---|
   | non-negativity | `C_hat >= 0` |
   | intrinsic lower bound | `C_hat >= max(0, S e^{-qT} - K e^{-rT})` |
   | upper bound | `C_hat <= S e^{-qT}` |
   | monotone in moneyness | `C_hat` non-decreasing in `m` at fixed `(T, r, sigma)` |
   | convex in strike | second difference of `C_hat` in `K` is non-negative |

   Each is reported as a count, a percentage, and the worst single violation, broken down by
   moneyness bucket so it is visible *where* a model misbehaves.

3. **Monotonicity and convexity are tested on a grid**, not on the test sample. They are
   statements about the fitted function, so they are evaluated by sweeping `m` and `K` on a
   fine grid at fixed other parameters - which also probes regions the test split happens
   not to cover.

An unconstrained network is expected to fail non-negativity in the deep wings, where the
true price is near zero and nothing in the squared-error loss penalises going slightly
below it. That is a prediction, recorded here before the model is fitted. If it happens, the
remedy belongs in the model specification (a non-negative output transform) rather than in
post-hoc clipping, and the choice will be stated rather than quietly applied.
