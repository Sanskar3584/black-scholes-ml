"""
hybrid_models.py - Black-Scholes residual learning (Hybrid A)
==============================================================

Project : AI/ML-Augmented Black-Scholes Framework
Ref     : master plan 5.4 to 5.6, 11.9; ML_experiment_plan.md 4.1
Text    : Carol Alexander, Market Risk Analysis Vol. I

The formulation (master plan 5.4)
----------------------------------
    epsilon_i = C_market_i - C_BS_i          the Black-Scholes residual
    x_i -> epsilon_i                          what the model learns
    C_hybrid = C_BS + epsilon_hat             the hybrid price

Section 5.5 states the point exactly: the task is reduced from "learn the
complete pricing function" to "learn only the systematic part of the model
error". Black-Scholes supplies the structure - moneyness, discounting,
the shape of the payoff - and the model supplies what the formula misses.

Which volatility prices the benchmark, and why it matters
----------------------------------------------------------
The 11.9 milestone is explicit:

    "The same option's implied volatility shall not be used to define the
     residual-learning benchmark."

If C_BS were priced with the option's own implied volatility then, by the
definition of implied volatility, C_BS would equal C_market and the residual
would be identically zero. The model would learn nothing and the error would
be invisible, because the metrics would look perfect.

So the benchmark volatility must be something computable without the
contract's own price. This module takes it as an argument and never derives
it internally - the caller is forced to say what it used, and
`residual_dataset()` refuses a benchmark volatility that reproduces the market
price too closely, which is the signature of that mistake.

Why the synthetic smile experiment
-----------------------------------
On market data the residual is whatever it is, and there is no ground truth to
check a hybrid against. So the experiment below manufactures one. Prices are
generated from a volatility SMILE,

    sigma_local(m) = sigma_atm * (1 + a*m + b*m^2),

and the benchmark is priced with the flat at-the-money volatility, which is
what a desk carrying a single number per expiry would use. The residual is
then real, structured, and known in closed form - so "the hybrid learned the
smile" is a checkable claim rather than a hopeful one.

This is also the right rehearsal for the real data, where the residual has
exactly this character: a single volatility cannot fit a surface that has one.
"""

import numpy as np
import pandas as pd

from black_scholes import bs_call

__all__ = ["smile_sigma", "make_smile_dataset", "residual_dataset",
           "HybridA", "SMILE_A", "SMILE_B"]

# Skew and curvature of the manufactured smile. Negative skew is the equity
# convention: downside strikes trade at higher implied volatility.
SMILE_A = -0.35
SMILE_B = 0.90


def smile_sigma(m, sigma_atm, a=SMILE_A, b=SMILE_B):
    """Local volatility as a function of log-moneyness.

    Floored at 1% so the manufactured smile cannot produce a negative
    volatility in the wings, which would be a property of the fixture rather
    than of anything financial.
    """
    m = np.asarray(m, float)
    return np.maximum(0.01, sigma_atm * (1.0 + a * m + b * m * m))


def make_smile_dataset(n=20000, seed=20260825, q=0.015, a=SMILE_A, b=SMILE_B):
    """Synthetic market where Black-Scholes is WRONG in a known way.

    Returns a frame with the market price, the benchmark price computed from
    the at-the-money volatility only, and the residual between them.
    """
    rng = np.random.default_rng(seed)
    S = 100.0
    m = rng.uniform(-0.6, 0.6, n)
    K = S * np.exp(-m)
    T = np.exp(rng.uniform(np.log(1 / 52), np.log(2.0), n))
    r = rng.uniform(0.03, 0.08, n)
    sigma_atm = rng.uniform(0.10, 0.60, n)

    sigma_true = smile_sigma(m, sigma_atm, a, b)
    C_market = bs_call(S, K, T, r, sigma_true, q)
    # the benchmark a desk with one volatility per expiry would quote
    C_bench = bs_call(S, K, T, r, sigma_atm, q)

    return pd.DataFrame({
        "S": S, "K": K, "T": T, "r": r, "q": q, "m": m,
        "sigma_atm": sigma_atm, "sigma_true": sigma_true,
        "C_market": C_market, "C_BS": C_bench,
        "epsilon": C_market - C_bench,
    })


def residual_dataset(df, sigma_bench_col="sigma_atm", price_col="C_market",
                     strict=True):
    """Attach C_BS and epsilon, refusing a leaking benchmark volatility.

    `strict` rejects a benchmark that reproduces the market price so closely
    that it must be the option's own implied volatility - the failure mode
    master plan 11.9 warns about, which otherwise shows up as a suspiciously
    perfect result rather than as an error.
    """
    out = df.copy()
    out["C_BS"] = bs_call(out["S"], out["K"], out["T"], out["r"],
                          out[sigma_bench_col], out.get("q", 0.0))
    out["epsilon"] = out[price_col] - out["C_BS"]

    rel = float(np.mean(np.abs(out["epsilon"]) /
                        np.maximum(np.abs(out[price_col]), 1e-9)))
    out.attrs["mean_relative_residual"] = rel
    if strict and rel < 1e-6:
        raise ValueError(
            "The benchmark reproduces the market price to %.2e on average.\n"
            "That is the signature of pricing C_BS with the option's OWN\n"
            "implied volatility, which makes the residual identically zero\n"
            "and the whole exercise vacuous. Master plan 11.9: 'The same\n"
            "option's implied volatility shall not be used to define the\n"
            "residual-learning benchmark.' Use lagged or historical\n"
            "volatility instead - see ML_experiment_plan.md 4.1." % rel)
    return out


class HybridA:
    """C_hybrid = C_BS + epsilon_hat, with epsilon learned from x.

    The learner is injected rather than fixed, so the same hybrid can be run
    over any of the Week 7 models and the comparison stays like for like.
    `learner` must expose fit(X, y) and predict(X).

    Scaling note. The residual is learned in units of C/K for the same
    homogeneity reason the price was: with K varying, an unscaled residual
    carries a strike-driven scale the scale-free features cannot explain.
    """

    def __init__(self, learner, name="hybrid"):
        self.learner = learner
        self.name = name

    def fit(self, X, K, epsilon):
        self.learner.fit(np.asarray(X, float),
                         np.asarray(epsilon, float) / np.asarray(K, float))
        return self

    def predict_residual(self, X, K):
        return self.learner.predict(np.asarray(X, float)) * np.asarray(K, float)

    def predict(self, X, K, C_BS):
        return np.asarray(C_BS, float) + self.predict_residual(X, K)


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def _self_test(verbose=True):
    from linear_models import Standardiser, fit_ols
    from neural_net import MLP

    if verbose:
        print("=" * 78)
        print("hybrid_models.py  SELF-TEST")
        print("=" * 78)
    checks = []

    def chk(name, cond, extra=""):
        checks.append(bool(cond))
        if verbose:
            print("  %-56s %s %s" % (name, "ok" if cond else "FAIL", extra))

    # ---- the smile is a smile ----
    m = np.linspace(-0.6, 0.6, 101)
    s = smile_sigma(m, 0.25)
    chk("smile is higher in the downside wing than at the money",
        s[0] > s[50], "(%.3f vs %.3f)" % (s[0], s[50]))
    chk("smile is convex in m",
        bool((np.diff(s, 2) > -1e-12).all()))
    chk("smile never goes negative", float(s.min()) > 0)

    # ---- the manufactured market is wrong in the intended direction ----
    d = make_smile_dataset(n=4000)
    chk("benchmark differs from market (the residual is real)",
        float(np.abs(d.epsilon).mean()) > 0.05,
        "(mean |eps| %.4f)" % np.abs(d.epsilon).mean())
    # The residual is EXACTLY zero at m = 0 by construction, since
    # smile_sigma(0) = sigma_atm. Near zero it is dominated by the linear skew
    # term, so mean |epsilon| should scale with the band width. Checking the
    # scaling is a far stronger statement than any single threshold - an
    # earlier version of this test used one, picked it badly, and failed on
    # correct behaviour.
    chk("sigma_local(0) equals sigma_atm exactly",
        bool(np.allclose(smile_sigma(0.0, 0.25), 0.25)))
    bands = [0.005, 0.01, 0.02, 0.04]
    means = [float(np.abs(d[np.abs(d.m) < w].epsilon).mean()) for w in bands]
    ratios = [means[i + 1] / means[i] for i in range(len(means) - 1)]
    if verbose:
        print("    mean |eps| by band: "
              + ", ".join("%.4f@%.3f" % (v, w) for v, w in zip(means, bands)))
    chk("residual scales linearly with |m| near the money",
        all(1.4 < r < 2.8 for r in ratios),
        "(ratios %s, expect ~2 per doubling)"
        % " ".join("%.2f" % r for r in ratios))
    chk("residual grows in the wings",
        float(np.abs(d[np.abs(d.m) > 0.4].epsilon).mean()) > 10 * means[2])
    chk("market prices stay arbitrage-free",
        bool((d.C_market >= -1e-9).all()
             and (d.C_market <= d.S * np.exp(-d.q * d["T"]) + 1e-9).all()))

    # ---- the leakage guard fires ----
    leak = d.copy()
    leak["sigma_own"] = d["sigma_true"]          # the option's own IV
    try:
        residual_dataset(leak, sigma_bench_col="sigma_own")
        chk("refuses a benchmark priced with the option's own IV", False)
    except ValueError as e:
        chk("refuses a benchmark priced with the option's own IV",
            "11.9" in str(e))
    ok = residual_dataset(d, sigma_bench_col="sigma_atm")
    chk("accepts a legitimate benchmark volatility",
        float(np.abs(ok.epsilon - d.epsilon).max()) < 1e-9)

    # ---- the hybrid beats both its parts ----
    # Split by index; this is a self-test, not the project's split.
    n = len(d)
    tr, te = np.arange(0, int(0.7 * n)), np.arange(int(0.7 * n), n)
    feats = ["m", "T", "r", "sigma_atm"]
    X = d[feats].to_numpy()
    K = d["K"].to_numpy()

    sc = Standardiser().fit(X[tr])

    class _OLS:
        def fit(s_, A, y):
            s_.f = fit_ols(sc.transform(A), y)
            return s_

        def predict(s_, A):
            return s_.f.predict(sc.transform(A))

    hyb = HybridA(_OLS()).fit(X[tr], K[tr], d.epsilon.to_numpy()[tr])
    pred_h = hyb.predict(X[te], K[te], d.C_BS.to_numpy()[te])

    pure = _OLS().fit(X[tr], d.C_market.to_numpy()[tr] / K[tr])
    pred_p = pure.predict(X[te]) * K[te]

    truth = d.C_market.to_numpy()[te]
    rmse = lambda p: float(np.sqrt(np.mean((p - truth) ** 2)))
    r_bs, r_h, r_p = rmse(d.C_BS.to_numpy()[te]), rmse(pred_h), rmse(pred_p)
    if verbose:
        print("    Black-Scholes %.4f | hybrid %.4f | pure ML %.4f"
              % (r_bs, r_h, r_p))
    chk("hybrid beats the Black-Scholes benchmark", r_h < r_bs)
    chk("hybrid beats pure ML on the same learner", r_h < r_p)

    # ---- a zero residual learner returns the benchmark exactly ----
    class _Zero:
        def fit(s_, A, y):
            return s_

        def predict(s_, A):
            return np.zeros(len(A))

    z = HybridA(_Zero()).predict(X[te], K[te], d.C_BS.to_numpy()[te])
    chk("a zero-residual learner reproduces Black-Scholes exactly",
        float(np.abs(z - d.C_BS.to_numpy()[te]).max()) < 1e-12)

    # ---- the hybrid is a strict generalisation, with a real learner ----
    # 600 epochs, not the 120 an earlier version used. The residual target is
    # an order of magnitude smaller than the price target the network's
    # defaults were set for, so it needs longer to converge: measured 3.61 at
    # 120 epochs, 2.04 at 300, 0.92 at 600 against the linear hybrid's 1.00.
    # Under-training the network and concluding it loses would have been the
    # easy mistake here.
    net = MLP(4, hidden=(32, 16), lr=3e-3, epochs=600, seed=20260825)

    class _Net:
        def fit(s_, A, y):
            net.fit(sc.transform(A), y)
            return s_

        def predict(s_, A):
            return net.predict(sc.transform(A))

    hn = HybridA(_Net()).fit(X[tr], K[tr], d.epsilon.to_numpy()[tr])
    r_hn = rmse(hn.predict(X[te], K[te], d.C_BS.to_numpy()[te]))
    if verbose:
        print("    network hybrid %.4f" % r_hn)
    chk("a network hybrid improves on the linear hybrid", r_hn < r_h)

    n_fail = sum(1 for c in checks if not c)
    if verbose:
        print("-" * 78)
        print("  %d checks, %d failed" % (len(checks), n_fail))
        print("  ALL CHECKS PASSED" if n_fail == 0
              else "  *** FAILURES ABOVE ***")
        print("=" * 78)
    return n_fail == 0


if __name__ == "__main__":
    raise SystemExit(0 if _self_test() else 1)
