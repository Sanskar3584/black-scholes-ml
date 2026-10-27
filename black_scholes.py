"""
black_scholes.py - European option pricing under the Black-Scholes model
=======================================================================

Project : AI/ML-Augmented Black-Scholes Framework
Ref     : September Week 2 plan, sections 16.15 - 16.17
Text    : Carol Alexander, Market Risk Analysis Vol. I, section I.5

This module is the classical pricing benchmark for the whole project. Every later
numerical method (binomial, Monte Carlo) and every machine-learning model is
compared against the functions defined here, so correctness matters more than
speed.

Contents
--------
    d1_d2              the two standard normal arguments
    bs_call            European call price
    bs_put             European put price
    bs_price           dispatcher on option type
    put_call_parity_error
    intrinsic_value, moneyness, moneyness_region

Convention
----------
    S      spot price of the underlying
    K      strike
    T      time to maturity in years
    r      continuously compounded risk-free rate
    sigma  volatility (annualised)
    q      continuous dividend yield (default 0)

All functions are vectorised over NumPy arrays.

Edge cases
----------
T -> 0 or sigma -> 0 make d1, d2 singular. Both are handled explicitly by
returning the discounted intrinsic value, which is the correct limit.
"""

import numpy as np
from scipy.stats import norm

__all__ = [
    "d1_d2", "bs_call", "bs_put", "bs_price",
    "put_call_parity_error", "intrinsic_value",
    "moneyness", "moneyness_region", "forward_price",
]

_TINY = 1e-12


# --------------------------------------------------------------------- core

def d1_d2(S, K, T, r, sigma, q=0.0):
    """Return (d1, d2).

        d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt(T))
        d2 = d1 - sigma sqrt(T)

    Where sigma*sqrt(T) is (numerically) zero the pair is undefined; NaN is
    returned there and the pricing functions substitute the correct limit.
    """
    S, K, T, r, sigma, q = map(np.asarray, np.broadcast_arrays(S, K, T, r, sigma, q))
    S = S.astype(float); K = K.astype(float); T = T.astype(float)
    r = r.astype(float); sigma = sigma.astype(float); q = q.astype(float)

    vol = sigma * np.sqrt(np.maximum(T, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / vol
        d2 = d1 - vol
    degenerate = vol < _TINY
    d1 = np.where(degenerate, np.nan, d1)
    d2 = np.where(degenerate, np.nan, d2)
    return d1, d2


def _degenerate_mask(T, sigma):
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    return (sigma * np.sqrt(np.maximum(T, 0.0))) < _TINY


def bs_call(S, K, T, r, sigma, q=0.0):
    """European call price.

        C = S e^{-qT} N(d1) - K e^{-rT} N(d2)

    At T = 0 or sigma = 0 this returns the discounted intrinsic value
    max(S e^{-qT} - K e^{-rT}, 0), which is the correct limit.
    """
    S, K, T, r, sigma, q = np.broadcast_arrays(S, K, T, r, sigma, q)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    with np.errstate(invalid="ignore"):
        price = (np.asarray(S) * np.exp(-np.asarray(q) * np.asarray(T)) * norm.cdf(d1)
                 - np.asarray(K) * np.exp(-np.asarray(r) * np.asarray(T)) * norm.cdf(d2))
    limit = np.maximum(np.asarray(S) * np.exp(-np.asarray(q) * np.asarray(T))
                       - np.asarray(K) * np.exp(-np.asarray(r) * np.asarray(T)), 0.0)
    out = np.where(_degenerate_mask(T, sigma), limit, price)
    return out if out.ndim else float(out)


def bs_put(S, K, T, r, sigma, q=0.0):
    """European put price.

        P = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)
    """
    S, K, T, r, sigma, q = np.broadcast_arrays(S, K, T, r, sigma, q)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    with np.errstate(invalid="ignore"):
        price = (np.asarray(K) * np.exp(-np.asarray(r) * np.asarray(T)) * norm.cdf(-d2)
                 - np.asarray(S) * np.exp(-np.asarray(q) * np.asarray(T)) * norm.cdf(-d1))
    limit = np.maximum(np.asarray(K) * np.exp(-np.asarray(r) * np.asarray(T))
                       - np.asarray(S) * np.exp(-np.asarray(q) * np.asarray(T)), 0.0)
    out = np.where(_degenerate_mask(T, sigma), limit, price)
    return out if out.ndim else float(out)


def bs_price(S, K, T, r, sigma, q=0.0, option="call"):
    """Dispatcher. option is 'call' or 'put'."""
    opt = str(option).lower()
    if opt in ("c", "call"):
        return bs_call(S, K, T, r, sigma, q)
    if opt in ("p", "put"):
        return bs_put(S, K, T, r, sigma, q)
    raise ValueError(f"option must be 'call' or 'put', got {option!r}")


# ------------------------------------------------------------- consistency

def put_call_parity_error(S, K, T, r, sigma, q=0.0):
    """Section 16.17.

        eps = C - P - S e^{-qT} + K e^{-rT}

    Put-call parity is model-free: it follows from a static arbitrage argument,
    not from the Black-Scholes dynamics. A non-zero eps therefore indicates an
    implementation error rather than a modelling limitation.
    """
    C = bs_call(S, K, T, r, sigma, q)
    P = bs_put(S, K, T, r, sigma, q)
    return (C - P
            - np.asarray(S) * np.exp(-np.asarray(q) * np.asarray(T))
            + np.asarray(K) * np.exp(-np.asarray(r) * np.asarray(T)))


def intrinsic_value(S, K, option="call"):
    """max(S - K, 0) for a call, max(K - S, 0) for a put."""
    opt = str(option).lower()
    if opt in ("c", "call"):
        return np.maximum(np.asarray(S, dtype=float) - np.asarray(K, dtype=float), 0.0)
    if opt in ("p", "put"):
        return np.maximum(np.asarray(K, dtype=float) - np.asarray(S, dtype=float), 0.0)
    raise ValueError(f"option must be 'call' or 'put', got {option!r}")


def forward_price(S, T, r, q=0.0):
    """F = S e^{(r-q)T}."""
    return np.asarray(S, dtype=float) * np.exp((np.asarray(r) - np.asarray(q))
                                               * np.asarray(T))


# ---------------------------------------------------------------- moneyness

def moneyness(S, K):
    """Log moneyness  m = ln(S/K)   (section 16.18.5)."""
    return np.log(np.asarray(S, dtype=float) / np.asarray(K, dtype=float))


def moneyness_region(S, K, atm_band=0.01):
    """Classify by m = ln(S/K): m < 0 OTM, m ~ 0 ATM, m > 0 ITM (for a call)."""
    m = np.atleast_1d(moneyness(S, K))
    out = np.where(m < -atm_band, "OTM", np.where(m > atm_band, "ITM", "ATM"))
    return out if out.size > 1 else str(out[0])


# --------------------------------------------------------------- self-test

def _self_test(verbose=True):
    """Checks that must hold for any correct implementation."""
    rng = np.random.default_rng(20260825)
    n = 20_000
    S = rng.uniform(20, 200, n)
    K = rng.uniform(20, 200, n)
    T = rng.uniform(0.01, 3.0, n)
    r = rng.uniform(-0.01, 0.12, n)
    sig = rng.uniform(0.02, 1.20, n)
    q = rng.uniform(0.0, 0.06, n)

    C = bs_call(S, K, T, r, sig, q)
    P = bs_put(S, K, T, r, sig, q)
    disc_K = K * np.exp(-r * T)
    fwd_S = S * np.exp(-q * T)

    checks = {
        "put-call parity |eps| max":  np.abs(put_call_parity_error(S, K, T, r, sig, q)).max(),
        "call >= 0 (min)":            C.min(),
        "put  >= 0 (min)":            P.min(),
        "call <= S e^-qT (max viol)": np.maximum(C - fwd_S, 0).max(),
        "put  <= K e^-rT (max viol)": np.maximum(P - disc_K, 0).max(),
        "call >= lower bound":        np.maximum(np.maximum(fwd_S - disc_K, 0) - C, 0).max(),
        "put  >= lower bound":        np.maximum(np.maximum(disc_K - fwd_S, 0) - P, 0).max(),
    }

    # T -> 0 limit must equal intrinsic
    S0 = np.array([80.0, 100.0, 120.0])
    K0 = np.full(3, 100.0)
    checks["T=0 call vs intrinsic"] = np.abs(
        bs_call(S0, K0, 0.0, 0.05, 0.2) - intrinsic_value(S0, K0, "call")).max()
    checks["sigma=0 call vs disc. intrinsic"] = np.abs(
        bs_call(S0, K0, 1.0, 0.05, 0.0)
        - np.maximum(S0 - K0 * np.exp(-0.05), 0)).max()

    ok = True
    if verbose:
        print(f"  {'check':<32}{'value':>14}")
    for k, v in checks.items():
        bad = v > 1e-9 if ("viol" in k or "eps" in k or "vs" in k or "bound" in k) else v < -1e-12
        ok &= not bad
        if verbose:
            print(f"  {k:<32}{v:>14.3e}{'   FAIL' if bad else ''}")
    if verbose:
        print(f"\n  {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}  (n = {n:,} random parameter sets)")
    return ok


if __name__ == "__main__":
    print("black_scholes.py self-test")
    print("=" * 50)
    _self_test()
