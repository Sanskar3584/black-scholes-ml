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
