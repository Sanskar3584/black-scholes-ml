"""
binomial.py - Cox-Ross-Rubinstein binomial option pricing
=========================================================

Project : AI/ML-Augmented Black-Scholes Framework
Ref     : Post-midsem plan, Week 4 ("Implement CRR European call/put and
          risk-neutral probabilities"); September plan section 15.7.1
Text    : Carol Alexander, Market Risk Analysis Vol. I, section I.5.6
Source  : Cox, Ross and Rubinstein (1979), Journal of Financial Economics 7(3)

The lattice
-----------
Over a step dt = T/N the underlying moves up by a factor u or down by d:

    u = exp(sigma sqrt(dt)),    d = 1/u = exp(-sigma sqrt(dt))

so the lattice recombines and its centre stays at S_0. The risk-neutral
probability follows from requiring the asset to grow at the risk-free rate:

    E[S_{t+dt}] = S_t e^{(r-q)dt}
    p u + (1-p) d = e^{(r-q)dt}
    =>  p = ( e^{(r-q)dt} - d ) / ( u - d )

and the option price is the discounted risk-neutral expectation, computed by
backward induction from the terminal payoff.

Why this is a worthwhile independent check
------------------------------------------
The binomial model is a DISCRETE-time construction. It shares with Black-Scholes
the no-arbitrage principle and the lognormal limit, but not the continuous-time
machinery: no Ito, no PDE, no normal distribution. Agreement between the two is
therefore evidence that the Week 2 derivation is right, not merely that the same
formula was coded twice.

Validity of p
-------------
p lies in (0,1) only if  d < e^{(r-q)dt} < u, i.e.

    sigma sqrt(dt)  >  |r - q| dt      =>    N  >  T (r-q)^2 / sigma^2

For ordinary parameters this is satisfied at any N worth using, but the check is
enforced rather than assumed: outside it the "probabilities" are negative and the
lattice is an arbitrage, not a model.
"""

import numpy as np
from scipy.stats import binom

from black_scholes import bs_price

__all__ = [
    "crr_params", "binomial_price", "binomial_price_closed_form",
    "binomial_tree_valid", "convergence_table",
]


# ------------------------------------------------------------------ lattice

def crr_params(T, N, r, sigma, q=0.0):
    """Return (dt, u, d, p, disc) for the CRR lattice."""
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    return dt, u, d, p, disc


def binomial_tree_valid(T, N, r, sigma, q=0.0):
    """True if the risk-neutral probability lies strictly in (0,1).

    Fails only when the drift over one step outruns the volatility over that
    step, which needs an absurdly small N or an enormous rate.
    """
    _, u, d, p, _ = crr_params(T, N, r, sigma, q)
    return bool(0.0 < p < 1.0)


# ------------------------------------------------------------------ pricing

def binomial_price(S, K, T, r, sigma, N, q=0.0, option="call", american=False):
    """CRR price by backward induction.

    Terminal stock prices are formed in logs,

        S_T(j) = S exp[ (2j - N) sigma sqrt(dt) ],   j = 0..N

    rather than as S u^j d^(N-j), which overflows for large N and small dt.

    Set american=True for early exercise. The plan requires European only; the
    American branch costs three lines and is needed later in the project, where
    the early-exercise premium becomes the quantity of interest.
    """
    if N < 1:
        raise ValueError("N must be at least 1")
    dt, u, d, p, disc = crr_params(T, N, r, sigma, q)
    if not (0.0 < p < 1.0):
        raise ValueError(
            f"risk-neutral probability p = {p:.4f} outside (0,1) at N = {N}; "
            f"the lattice is not arbitrage-free. Need N > T(r-q)^2/sigma^2.")

    opt = str(option).lower()
    if opt not in ("c", "call", "p", "put"):
        raise ValueError(f"option must be 'call' or 'put', got {option!r}")
    is_call = opt in ("c", "call")

    j = np.arange(N + 1)
    log_move = (2 * j - N) * sigma * np.sqrt(dt)
    ST = S * np.exp(log_move)
    V = np.maximum(ST - K, 0.0) if is_call else np.maximum(K - ST, 0.0)

    for i in range(N - 1, -1, -1):
        V = disc * (p * V[1:] + (1.0 - p) * V[:-1])
        if american:
            ji = np.arange(i + 1)
            Si = S * np.exp((2 * ji - i) * sigma * np.sqrt(dt))
            exercise = np.maximum(Si - K, 0.0) if is_call else np.maximum(K - Si, 0.0)
            V = np.maximum(V, exercise)
    return float(V[0])


def binomial_price_closed_form(S, K, T, r, sigma, N, q=0.0, option="call"):
    """European CRR price in closed form, via the binomial distribution.

        C = S e^{-qT} P[Bin(N, p') >= a]  -  K e^{-rT} P[Bin(N, p) >= a]

    where p' = p u e^{-(r-q)dt} and a is the smallest number of up-moves that
    finishes in the money. Mathematically identical to backward induction, but
    computed by a completely different route, so it catches indexing mistakes
    that a single implementation would hide.
    """
    dt, u, d, p, _ = crr_params(T, N, r, sigma, q)
    if not (0.0 < p < 1.0):
        raise ValueError(f"p = {p:.4f} outside (0,1)")

    # smallest j with S u^j d^(N-j) >= K
    with np.errstate(divide="ignore"):
        a = np.ceil((np.log(K / S) - N * np.log(d)) / (np.log(u) - np.log(d)))
    a = int(max(0, min(N + 1, a)))

    p_star = p * u * np.exp(-(r - q) * dt)
    # P[X >= a] = sf(a-1)
    prob_star = binom.sf(a - 1, N, p_star) if a <= N else 0.0
    prob = binom.sf(a - 1, N, p) if a <= N else 0.0

    call = S * np.exp(-q * T) * prob_star - K * np.exp(-r * T) * prob
    opt = str(option).lower()
    if opt in ("c", "call"):
        return float(call)
    # put by parity
    return float(call - S * np.exp(-q * T) + K * np.exp(-r * T))


# ------------------------------------------------------------- convergence

def convergence_table(S, K, T, r, sigma, q=0.0, option="call",
                      Ns=(10, 25, 50, 100, 250, 500, 1000)):
    """Error against the analytical Black-Scholes price for each N.

    E_N = | C_binomial(N) - C_BS |
    """
    bs = float(bs_price(S, K, T, r, sigma, q, option))
    rows = []
    for N in Ns:
        c = binomial_price(S, K, T, r, sigma, N, q, option)
        _, u, d, p, _ = crr_params(T, N, r, sigma, q)
        rows.append({"N": N, "binomial": c, "black_scholes": bs,
                     "error": c - bs, "abs_error": abs(c - bs),
                     "u": u, "d": d, "p": p})
    return rows


# ---------------------------------------------------------------- self-test

def _self_test(verbose=True):
    rng = np.random.default_rng(20260825)
    checks = {}

    # 1. backward induction vs closed form - two independent routes
    n = 300
    S = rng.uniform(50, 150, n); K = rng.uniform(50, 150, n)
    T = rng.uniform(0.1, 2.0, n); r = rng.uniform(0.0, 0.08, n)
    sig = rng.uniform(0.1, 0.6, n); q = rng.uniform(0.0, 0.03, n)
    diffs = []
    for i in range(n):
        for opt in ("call", "put"):
            a = binomial_price(S[i], K[i], T[i], r[i], sig[i], 200, q[i], opt)
            b = binomial_price_closed_form(S[i], K[i], T[i], r[i], sig[i], 200, q[i], opt)
            diffs.append(abs(a - b))
    checks["backward induction vs closed form"] = max(diffs)

    # 2. put-call parity holds ON THE LATTICE, exactly, at any N
    par = []
    for i in range(100):
        for N in (7, 50, 201):
            c = binomial_price(S[i], K[i], T[i], r[i], sig[i], N, q[i], "call")
            pu = binomial_price(S[i], K[i], T[i], r[i], sig[i], N, q[i], "put")
            par.append(abs(c - pu - S[i] * np.exp(-q[i] * T[i])
                           + K[i] * np.exp(-r[i] * T[i])))
    checks["put-call parity on the lattice"] = max(par)

    # 3. risk-neutral probability valid across a wide grid
    bad = 0
    for N in (5, 10, 50, 500):
        for i in range(n):
            if not binomial_tree_valid(T[i], N, r[i], sig[i], q[i]):
                bad += 1
    checks["p outside (0,1) count"] = bad

    # 4. convergence to Black-Scholes at large N
    conv = []
    for i in range(60):
        b = binomial_price(S[i], K[i], T[i], r[i], sig[i], 2000, q[i], "call")
        a = float(bs_price(S[i], K[i], T[i], r[i], sig[i], q[i], "call"))
        conv.append(abs(b - a))
    checks["max |binomial(N=2000) - BS|"] = max(conv)

    # 5. American >= European, and equal for a non-dividend call
    am_ge = []
    for i in range(60):
        e = binomial_price(S[i], K[i], T[i], r[i], sig[i], 300, 0.0, "put")
        a = binomial_price(S[i], K[i], T[i], r[i], sig[i], 300, 0.0, "put", american=True)
        am_ge.append(e - a)                       # must be <= 0
    checks["American put >= European (max violation)"] = max(0.0, max(am_ge))
    call_gap = []
    for i in range(40):
        e = binomial_price(S[i], K[i], T[i], r[i], sig[i], 300, 0.0, "call")
        a = binomial_price(S[i], K[i], T[i], r[i], sig[i], 300, 0.0, "call", american=True)
        call_gap.append(abs(a - e))
    checks["American = European call, no dividend"] = max(call_gap)

    tolerances = {
        "backward induction vs closed form": 1e-9,
        "put-call parity on the lattice": 1e-9,
        "p outside (0,1) count": 0,
        "max |binomial(N=2000) - BS|": 0.05,
        "American put >= European (max violation)": 1e-12,
        "American = European call, no dividend": 1e-9,
    }
    ok = True
    if verbose:
        print(f"  {'check':<44}{'value':>14}{'tol':>12}")
    for k, v in checks.items():
        bad_k = v > tolerances[k]
        ok &= not bad_k
        if verbose:
            print(f"  {k:<44}{v:>14.3e}{tolerances[k]:>12.0e}"
                  f"{'   FAIL' if bad_k else ''}")
    if verbose:
        print(f"\n  {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    return ok


if __name__ == "__main__":
    print("binomial.py self-test")
    print("=" * 72)
    _self_test()
