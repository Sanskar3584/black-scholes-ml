"""
greeks.py - analytic option sensitivities under Black-Scholes
============================================================

Project : AI/ML-Augmented Black-Scholes Framework
Ref     : Post-midsem plan, Week 3 ("Implement analytic Delta, Gamma, Vega for
          European calls and puts"); September plan section 15.6.1
Text    : Carol Alexander, Market Risk Analysis Vol. I, section I.5

The plan requires Delta, Gamma and Vega. Theta and Rho are included as well:
they cost nothing extra, they are needed for the PDE identity check below, and
Week 6 asks for portfolio exposures where theta carry matters.

Definitions (European options, continuous dividend yield q)
-----------------------------------------------------------
    Delta = dV/dS        call   e^{-qT} N(d1)
                         put   -e^{-qT} N(-d1)

    Gamma = d2V/dS2      e^{-qT} phi(d1) / (S sigma sqrt(T))      same for call and put

    Vega  = dV/dsigma    S e^{-qT} sqrt(T) phi(d1)                same for call and put

    Theta = dV/dt        call  -S e^{-qT} phi(d1) sigma / (2 sqrt(T))
                               - r K e^{-rT} N(d2) + q S e^{-qT} N(d1)
                         put   -S e^{-qT} phi(d1) sigma / (2 sqrt(T))
                               + r K e^{-rT} N(-d2) - q S e^{-qT} N(-d1)

    Rho   = dV/dr        call   K T e^{-rT} N(d2)
                         put   -K T e^{-rT} N(-d2)

Conventions
-----------
Everything is returned in RAW units: per 1.00 of the underlying, per 1.00 of
volatility (not per 1%), per 1.00 of rate, and per YEAR for theta. Helper
functions convert to the quoting conventions used on a trading desk.

Two identities worth knowing, both checked in the self-test
-----------------------------------------------------------
    Delta_call - Delta_put = e^{-qT}            (from put-call parity)
    Vega = S^2 sigma T Gamma                    (both carry the same phi(d1))

and the strongest check available, the PDE itself evaluated with ANALYTIC
derivatives rather than finite differences:

    Theta + (1/2) sigma^2 S^2 Gamma + (r-q) S Delta - r V = 0
"""

import numpy as np
from scipy.stats import norm

from black_scholes import d1_d2, bs_call, bs_put, bs_price

__all__ = [
    "delta", "gamma", "vega", "theta", "rho", "all_greeks",
    "vega_per_pct", "theta_per_day", "rho_per_pct", "pde_residual",
]

_TINY = 1e-12


def _prep(S, K, T, r, sigma, q):
    S, K, T, r, sigma, q = np.broadcast_arrays(
        *[np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q)])
    degenerate = (sigma * np.sqrt(np.maximum(T, 0.0))) < _TINY
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    return S, K, T, r, sigma, q, d1, d2, degenerate


def _scalarise(x):
    return x if np.ndim(x) else float(x)


# ----------------------------------------------------------------- first order

def delta(S, K, T, r, sigma, q=0.0, option="call"):
    """dV/dS. In (0,1) for a call, (-1,0) for a put.

    At T = 0 or sigma = 0 the limit is a step: 1 (or -1) if the option is in the
    money, 0 if out, and the convention here is 0.5 (or -0.5) exactly at the money.
    """
    S, K, T, r, sigma, q, d1, _, deg = _prep(S, K, T, r, sigma, q)
    opt = str(option).lower()
    with np.errstate(invalid="ignore"):
        if opt in ("c", "call"):
            out = np.exp(-q * T) * norm.cdf(d1)
            lim = np.where(S > K, 1.0, np.where(S < K, 0.0, 0.5)) * np.exp(-q * T)
        elif opt in ("p", "put"):
            out = -np.exp(-q * T) * norm.cdf(-d1)
            lim = np.where(S < K, -1.0, np.where(S > K, 0.0, -0.5)) * np.exp(-q * T)
        else:
            raise ValueError(f"option must be 'call' or 'put', got {option!r}")
    return _scalarise(np.where(deg, lim, out))


# ---------------------------------------------------------------- second order

def gamma(S, K, T, r, sigma, q=0.0, option="call"):
    """d2V/dS2. Identical for calls and puts, because they differ by a linear
    function of S (put-call parity), which has zero second derivative.

    The degenerate limit is a Dirac spike at S = K; 0 is returned away from the
    strike and np.inf at it, which is the honest answer rather than a large number.
    """
    S, K, T, r, sigma, q, d1, _, deg = _prep(S, K, T, r, sigma, q)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
        lim = np.where(np.isclose(S, K), np.inf, 0.0)
    return _scalarise(np.where(deg, lim, out))


def vega(S, K, T, r, sigma, q=0.0, option="call"):
    """dV/dsigma, per 1.00 of volatility (not per 1%). Identical for calls and puts.

    Always strictly positive for T > 0, sigma > 0. Note that phi(d1) underflows to
    exactly zero once |d1| exceeds about 38, so deep out-of-the-money vega is
    returned as 0 even though it is mathematically positive - it is below the
    smallest representable double. This matters for implied-volatility inversion
    and is why implied_volatility.py falls back to bisection.
    """
    S, K, T, r, sigma, q, d1, _, deg = _prep(S, K, T, r, sigma, q)
    with np.errstate(invalid="ignore"):
        out = S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1)
    return _scalarise(np.where(deg, 0.0, out))


# ------------------------------------------------------------- time and rates

def theta(S, K, T, r, sigma, q=0.0, option="call"):
    """dV/dt, per YEAR. Usually negative: options decay as time passes.

    A deep in-the-money European put can have POSITIVE theta, because the
    dominant term is the discounted strike it will receive, which draws nearer.
    """
    S, K, T, r, sigma, q, d1, d2, deg = _prep(S, K, T, r, sigma, q)
    opt = str(option).lower()
    with np.errstate(invalid="ignore", divide="ignore"):
        common = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        if opt in ("c", "call"):
            out = common - r * K * np.exp(-r * T) * norm.cdf(d2) \
                  + q * S * np.exp(-q * T) * norm.cdf(d1)
        elif opt in ("p", "put"):
            out = common + r * K * np.exp(-r * T) * norm.cdf(-d2) \
                  - q * S * np.exp(-q * T) * norm.cdf(-d1)
        else:
            raise ValueError(f"option must be 'call' or 'put', got {option!r}")
    return _scalarise(np.where(deg, 0.0, out))


def rho(S, K, T, r, sigma, q=0.0, option="call"):
    """dV/dr, per 1.00 of rate. Positive for calls, negative for puts."""
    S, K, T, r, sigma, q, _, d2, deg = _prep(S, K, T, r, sigma, q)
    opt = str(option).lower()
    with np.errstate(invalid="ignore"):
        if opt in ("c", "call"):
            out = K * T * np.exp(-r * T) * norm.cdf(d2)
        elif opt in ("p", "put"):
            out = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        else:
            raise ValueError(f"option must be 'call' or 'put', got {option!r}")
    return _scalarise(np.where(deg, 0.0, out))


# ------------------------------------------------------------------ convenience

def all_greeks(S, K, T, r, sigma, q=0.0, option="call"):
    """Every Greek plus the price, as a dict, in raw units."""
    return {
        "price": bs_price(S, K, T, r, sigma, q, option),
        "delta": delta(S, K, T, r, sigma, q, option),
        "gamma": gamma(S, K, T, r, sigma, q, option),
        "vega":  vega(S, K, T, r, sigma, q, option),
        "theta": theta(S, K, T, r, sigma, q, option),
        "rho":   rho(S, K, T, r, sigma, q, option),
    }


def vega_per_pct(*a, **kw):
    """Vega for a 1 percentage-point move in volatility - the desk convention."""
    return vega(*a, **kw) / 100.0


def theta_per_day(*a, calendar_days=365.0, **kw):
    """Theta per calendar day. Use calendar_days=252 for trading days."""
    return theta(*a, **kw) / calendar_days


def rho_per_pct(*a, **kw):
    """Rho for a 1 percentage-point move in the rate."""
    return rho(*a, **kw) / 100.0


def pde_residual(S, K, T, r, sigma, q=0.0, option="call"):
    """Black-Scholes PDE evaluated with ANALYTIC Greeks:

        Theta + (1/2) sigma^2 S^2 Gamma + (r-q) S Delta - r V

    This is a stronger check than the finite-difference version used in Week 2,
    because every derivative here is exact. It should be zero to machine
    precision, not to truncation error.
    """
    g = all_greeks(S, K, T, r, sigma, q, option)
    S = np.asarray(S, dtype=float)
    return (g["theta"] + 0.5 * np.asarray(sigma) ** 2 * S ** 2 * g["gamma"]
            + (np.asarray(r) - np.asarray(q)) * S * g["delta"]
            - np.asarray(r) * g["price"])


# ---------------------------------------------------------------- self-test

def _self_test(verbose=True):
    rng = np.random.default_rng(20260825)
    n = 20_000
    S   = rng.uniform(20, 200, n)
    K   = rng.uniform(20, 200, n)
    T   = rng.uniform(0.05, 3.0, n)
    r   = rng.uniform(0.0, 0.10, n)
    sig = rng.uniform(0.05, 1.00, n)
    q   = rng.uniform(0.0, 0.05, n)

    dC = delta(S, K, T, r, sig, q, "call")
    dP = delta(S, K, T, r, sig, q, "put")
    gC = gamma(S, K, T, r, sig, q, "call")
    gP = gamma(S, K, T, r, sig, q, "put")
    vC = vega(S, K, T, r, sig, q, "call")
    vP = vega(S, K, T, r, sig, q, "put")

    checks = {
        # structural identities
        "Delta_call - Delta_put = e^-qT":  np.abs(dC - dP - np.exp(-q * T)).max(),
        "Gamma identical call vs put":     np.abs(gC - gP).max(),
        "Vega identical call vs put":      np.abs(vC - vP).max(),
        "Vega = S^2 sigma T Gamma":        np.abs(vC - S**2 * sig * T * gC).max(),
        # sign and range constraints
        "Delta_call in (0, e^-qT)":        max(0.0, -dC.min(), (dC - np.exp(-q*T)).max()),
        "Delta_put in (-e^-qT, 0)":        max(0.0, dP.max(), (-dP - np.exp(-q*T)).max()),
        "Gamma > 0":                       max(0.0, -gC.min()),
        "Vega > 0":                        max(0.0, -vC.min()),
        "Rho_call > 0":                    max(0.0, -rho(S, K, T, r, sig, q, "call").min()),
        "Rho_put < 0":                     max(0.0, rho(S, K, T, r, sig, q, "put").max()),
        # the PDE, with exact derivatives
        "PDE residual (call)": np.abs(pde_residual(S, K, T, r, sig, q, "call")).max(),
        "PDE residual (put)":  np.abs(pde_residual(S, K, T, r, sig, q, "put")).max(),
    }

    # finite-difference agreement at a representative point
    h = 1e-4
    S0, K0, T0, r0, s0 = 100.0, 100.0, 1.0, 0.05, 0.25
    fd_delta = (bs_call(S0+h, K0, T0, r0, s0) - bs_call(S0-h, K0, T0, r0, s0)) / (2*h)
    fd_vega  = (bs_call(S0, K0, T0, r0, s0+h) - bs_call(S0, K0, T0, r0, s0-h)) / (2*h)
    checks["Delta vs central difference"] = abs(fd_delta - delta(S0, K0, T0, r0, s0))
    checks["Vega vs central difference"]  = abs(fd_vega - vega(S0, K0, T0, r0, s0))

    ok = True
    if verbose:
        print(f"  {'check':<34}{'value':>14}")
    for k, v in checks.items():
        bad = v > 1e-6
        ok &= not bad
        if verbose:
            print(f"  {k:<34}{v:>14.3e}{'   FAIL' if bad else ''}")
    if verbose:
        print(f"\n  {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}"
              f"  (n = {n:,} random parameter sets)")
    return ok


if __name__ == "__main__":
    print("greeks.py self-test")
    print("=" * 52)
    _self_test()
