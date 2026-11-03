"""
monte_carlo.py - Monte Carlo pricing of European options
========================================================

Project : AI/ML-Augmented Black-Scholes Framework
Ref     : Week 5 task sheet, Task 1; post-midsem plan Week 5
Text    : Carol Alexander, Market Risk Analysis Vol. I, sections I.5.7 and I.5.8

Method
------
Under the risk-neutral measure the terminal price is known in closed form, so
the whole path is unnecessary. Sample the endpoint directly:

    S_T^(j) = S_0 exp[ (r - q - sigma^2/2) T + sigma sqrt(T) Z_j ],   Z_j ~ N(0,1)

    C_hat = e^{-rT} (1/M) sum_j max(S_T^(j) - K, 0)

    s.e.  = s_Y / sqrt(M),        CI_95 = C_hat +/- 1.96 s.e.

where s_Y is the sample standard deviation of the DISCOUNTED payoffs.

Why terminal sampling rather than an Euler path
-----------------------------------------------
For a European payoff only S_T matters. Stepping a path through time would add
discretisation bias on top of the sampling error and buy nothing, since the exact
transition density is available. Terminal sampling is therefore unbiased: the
only error is statistical, and it is fully described by the standard error.

This is the key difference from the binomial method of Week 4. There the error
was deterministic discretisation error, O(1/N), which shrinks predictably. Here
the error is random, O(1/sqrt(M)), and is quantified by a confidence interval
rather than a convergence order.

Antithetic variates
-------------------
Optional. Each draw Z is paired with -Z and the two payoffs averaged. Because the
payoff is monotone in Z, the pair is negatively correlated and the variance of
the average falls. The standard error must then be computed on the M/2 PAIR
AVERAGES, not on the M individual payoffs, which are no longer independent.
Getting that wrong understates the error, so it is handled explicitly below.
"""

from dataclasses import dataclass, asdict

import numpy as np

from black_scholes import bs_price

__all__ = ["MCResult", "mc_price", "mc_call", "mc_put", "mc_terminal_prices"]

Z95 = 1.959963984540054          # exact two-sided 95% normal quantile


@dataclass
class MCResult:
    """Monte Carlo price with its sampling uncertainty."""
    price: float
    stderr: float
    ci_low: float
    ci_high: float
    M: int
    antithetic: bool
    mean_ST: float                # diagnostic: should estimate S e^{(r-q)T}
    payoff_sd: float

    @property
    def ci(self):
        return (self.ci_low, self.ci_high)

    def contains(self, x, tol=0.0):
        """Does the 95% interval contain x?"""
        return (self.ci_low - tol) <= x <= (self.ci_high + tol)

    def as_dict(self):
        return asdict(self)

    def __repr__(self):
        return (f"MCResult(price={self.price:.6f}, se={self.stderr:.2e}, "
                f"CI=[{self.ci_low:.6f}, {self.ci_high:.6f}], M={self.M:,}"
                f"{', antithetic' if self.antithetic else ''})")


def mc_terminal_prices(S, T, r, sigma, M, q=0.0, rng=None, antithetic=False):
    """Draw M terminal prices under the risk-neutral measure.

    With antithetic=True, M must be even: M/2 draws are each used twice, as Z
    and -Z, and the returned array is ordered so that element i and element
    i + M/2 form a pair.
    """
    rng = np.random.default_rng() if rng is None else rng
    drift = (r - q - 0.5 * sigma ** 2) * T
    vol = sigma * np.sqrt(T)
    if antithetic:
        if M % 2:
            raise ValueError("M must be even when antithetic=True")
        z = rng.standard_normal(M // 2)
        z = np.concatenate([z, -z])
    else:
        z = rng.standard_normal(M)
    return S * np.exp(drift + vol * z)


def mc_price(S, K, T, r, sigma, M=1_000_000, q=0.0, option="call",
             seed=None, antithetic=False, rng=None):
    """Monte Carlo price, standard error and 95% confidence interval.

    Pass `seed` for a reproducible run, or `rng` to share a generator across
    calls (used by the coverage study, where each replication needs a fresh
    draw from one stream).
    """
    opt = str(option).lower()
    if opt not in ("c", "call", "p", "put"):
        raise ValueError(f"option must be 'call' or 'put', got {option!r}")
    is_call = opt in ("c", "call")
    M = int(M)
    if M < 2:
        raise ValueError("M must be at least 2")

    if rng is None:
        rng = np.random.default_rng(seed)

    ST = mc_terminal_prices(S, T, r, sigma, M, q, rng=rng, antithetic=antithetic)
    payoff = np.maximum(ST - K, 0.0) if is_call else np.maximum(K - ST, 0.0)
    disc = np.exp(-r * T) * payoff

    if antithetic:
        # The M samples are not independent. Average each (Z, -Z) pair and treat
        # the M/2 pair averages as the independent sample.
        half = M // 2
        pairs = 0.5 * (disc[:half] + disc[half:])
        price = float(pairs.mean())
        sd = float(pairs.std(ddof=1))
        n_eff = half
    else:
        price = float(disc.mean())
        sd = float(disc.std(ddof=1))
        n_eff = M

    se = sd / np.sqrt(n_eff)
    return MCResult(price=price, stderr=float(se),
                    ci_low=float(price - Z95 * se), ci_high=float(price + Z95 * se),
                    M=M, antithetic=bool(antithetic),
                    mean_ST=float(ST.mean()), payoff_sd=sd)


def mc_call(S, K, T, r, sigma, M=1_000_000, q=0.0, **kw):
    return mc_price(S, K, T, r, sigma, M, q, "call", **kw)


def mc_put(S, K, T, r, sigma, M=1_000_000, q=0.0, **kw):
    return mc_price(S, K, T, r, sigma, M, q, "put", **kw)


# ---------------------------------------------------------------- self-test

def _self_test(verbose=True):
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.25
    checks, notes = {}, []

    # ---- 1. acceptance criterion: BS inside the 95% CI at M = 1e6 ----------
    if verbose:
        print("  Acceptance criterion: at M = 1,000,000 the Black-Scholes price"
              " lies inside the 95% interval\n")
        print(f"  {'option':>7}{'MC price':>13}{'s.e.':>11}{'95% CI':>26}"
              f"{'BS price':>13}{'inside':>9}{'z':>8}")
    inside_all = True
    for option in ("call", "put"):
        res = mc_price(S, K, T, r, sigma, 1_000_000, 0.0, option, seed=20260825)
        bs = float(bs_price(S, K, T, r, sigma, 0.0, option))
        inside = res.contains(bs)
        inside_all &= inside
        z = (res.price - bs) / res.stderr
        if verbose:
            print(f"  {option:>7}{res.price:>13.6f}{res.stderr:>11.2e}"
                  f"  [{res.ci_low:>10.6f}, {res.ci_high:>10.6f}]"
                  f"{bs:>13.6f}{('yes' if inside else 'NO'):>9}{z:>8.2f}")
    checks["BS inside 95% CI at M=1e6 (call and put)"] = 0.0 if inside_all else 1.0

    # ---- 2. standard error scales as 1/sqrt(M) ----------------------------
    Ms = [10 ** k for k in (3, 4, 5, 6)]
    ses = [mc_price(S, K, T, r, sigma, m, seed=20260825 + i).stderr
           for i, m in enumerate(Ms)]
    slope = np.polyfit(np.log(Ms), np.log(ses), 1)[0]
    checks["s.e. scaling exponent vs -0.5"] = abs(slope + 0.5)
    notes.append(f"s.e. slope against M: {slope:.4f} (theory -0.5)")

    # ---- 3. internal parity identity --------------------------------------
    # With the SAME terminal sample, mean(max(S-K,0)) - mean(max(K-S,0))
    # = mean(S_T) - K exactly, so the MC call and put must satisfy
    #   C - P = e^{-rT} (mean(S_T) - K)
    # to machine precision. This checks the estimator, not the model.
    rng = np.random.default_rng(20260825)
    ST = mc_terminal_prices(S, T, r, sigma, 200_000, 0.0, rng=rng)
    disc = np.exp(-r * T)
    c = disc * np.maximum(ST - K, 0).mean()
    p = disc * np.maximum(K - ST, 0).mean()
    checks["sample parity C - P = e^-rT (mean S_T - K)"] = abs(
        (c - p) - disc * (ST.mean() - K))

    # ---- 4. dividend yield ------------------------------------------------
    q = 0.015
    for option in ("call", "put"):
        res = mc_price(S, K, T, r, sigma, 1_000_000, q, option, seed=20260825)
        bs = float(bs_price(S, K, T, r, sigma, q, option))
        checks[f"q=0.015 {option}: |MC - BS| / s.e."] = abs(res.price - bs) / res.stderr
    # E[S_T] must estimate S e^{(r-q)T}
    res = mc_price(S, K, T, r, sigma, 1_000_000, q, "call", seed=20260825)
    checks["mean S_T vs S e^{(r-q)T}, relative"] = abs(
        res.mean_ST - S * np.exp((r - q) * T)) / (S * np.exp((r - q) * T))

    # ---- 5. antithetic variance reduction ---------------------------------
    plain = mc_price(S, K, T, r, sigma, 200_000, 0.0, "call", seed=7)
    anti = mc_price(S, K, T, r, sigma, 200_000, 0.0, "call", seed=7, antithetic=True)
    vr = (plain.stderr / anti.stderr) ** 2
    checks["antithetic reduces variance (factor > 1)"] = 0.0 if vr > 1 else 1.0
    notes.append(f"antithetic variance-reduction factor: {vr:.2f}x "
                 f"(s.e. {plain.stderr:.2e} -> {anti.stderr:.2e})")
    bs_c = float(bs_price(S, K, T, r, sigma, 0.0, "call"))
    checks["antithetic CI still covers BS"] = 0.0 if anti.contains(bs_c) else 1.0

    tol = {
        "BS inside 95% CI at M=1e6 (call and put)": 0.5,
        "s.e. scaling exponent vs -0.5": 0.05,
        "sample parity C - P = e^-rT (mean S_T - K)": 1e-10,
        "q=0.015 call: |MC - BS| / s.e.": 3.0,
        "q=0.015 put: |MC - BS| / s.e.": 3.0,
        "mean S_T vs S e^{(r-q)T}, relative": 2e-3,
        "antithetic reduces variance (factor > 1)": 0.5,
        "antithetic CI still covers BS": 0.5,
    }
    ok = True
    if verbose:
        print(f"\n  {'check':<46}{'value':>13}{'tol':>11}")
    for k, v in checks.items():
        bad = v > tol[k]
        ok &= not bad
        if verbose:
            print(f"  {k:<46}{v:>13.3e}{tol[k]:>11.1e}{'  FAIL' if bad else ''}")
    if verbose:
        for n in notes:
            print(f"\n  {n}")
        print(f"\n  {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    return ok


if __name__ == "__main__":
    print("monte_carlo.py self-test")
    print("=" * 78)
    _self_test()
