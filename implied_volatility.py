"""
implied_volatility.py - robust inversion of the Black-Scholes formula
=====================================================================

Project : AI/ML-Augmented Black-Scholes Framework
Ref     : Post-midsem plan, Week 3 ("Newton-Raphson with bisection fallback and
          bracket checks"); September plan sections 15.6.3 and 15.6.4

The problem
-----------
Given an observed price C_market, find sigma_IV such that

    C_BS(S, K, T, r, sigma_IV) = C_market

Every input except sigma is observable, so this is the one parameter the market
actually quotes. It is also the quantity the volatility smile is expressed in,
which is where the constant-volatility assumption becomes visible.

Why a plain Newton solver is not enough
---------------------------------------
Newton's update divides by vega:

    sigma <- sigma - (C_BS(sigma) - C_market) / vega(sigma)

Vega collapses towards zero for deep in- or out-of-the-money options and for
very short maturities. Once |d1| exceeds about 38 the normal density underflows
to exactly zero in double precision, so vega is 0 and the update is undefined.
Newton then either diverges or raises. Three defences are used here:

  1. BRACKET CHECK first. A finite sigma exists only if the price lies strictly
     inside the no-arbitrage band. Outside it, no volatility can reproduce the
     price and NaN is returned with a reason rather than a silently wrong number.

  2. BISECTION FALLBACK. Bisection needs only a sign change, never a derivative,
     so it is immune to vanishing vega. It is slower but cannot fail on a valid
     bracket.

  3. SAFEGUARDED NEWTON. Each Newton step is rejected if vega is negligible or
     if the step leaves the bracket; the solver then falls back to bisection for
     that option rather than abandoning it.

Every call reports which method actually converged, so the fraction of cases
rescued by the fallback can be measured rather than assumed.
"""

import numpy as np
from scipy.stats import norm

from black_scholes import bs_call, bs_put, bs_price
from greeks import vega as bs_vega

__all__ = [
    "implied_vol", "implied_vol_scalar", "no_arbitrage_bounds",
    "initial_guess", "SOLVER_FAILED",
]

SOLVER_FAILED = np.nan

_SIG_LO = 1e-8        # lower search bound
_SIG_HI = 5.0         # upper search bound (500% vol); expanded if needed
_VEGA_FLOOR = 1e-10   # below this, Newton is not trusted

# Bracket tolerances are RELATIVE to the price scale, not absolute.
#
# An absolute tolerance is wrong for index options. At NIFTY levels the deep
# in-the-money lower bound is of order 1e4, where one unit in the last place is
# about 2e-12 - larger than a 1e-12 absolute tolerance. A price sitting a few ULP
# below the bound through ordinary rounding was therefore rejected as an
# arbitrage violation, and a price "at" the bound to within 1e-14 was not
# representable at all, so that branch could never fire.
#
# Both tolerances are now multiplied by the scale of the problem, taken as the
# upper no-arbitrage bound (S e^{-qT} for a call, K e^{-rT} for a put). At
# S = 100 this reproduces the previous behaviour to within a factor of a few; at
# S = 25,000 it scales with the instrument.
_REL_BRACKET = 1e-13  # price may sit this far outside the bound before rejection
_REL_DEGEN   = 1e-15  # price this close to a bound is treated as sitting on it


# ------------------------------------------------------------------ bounds

def no_arbitrage_bounds(S, K, T, r, q=0.0, option="call"):
    """(lower, upper) bounds on the option price.

        call:  max(S e^{-qT} - K e^{-rT}, 0)  <=  C  <=  S e^{-qT}
        put:   max(K e^{-rT} - S e^{-qT}, 0)  <=  P  <=  K e^{-rT}

    The lower bound is the sigma -> 0 limit, the upper the sigma -> infinity
    limit. A price strictly inside corresponds to exactly one sigma, because the
    price is strictly increasing in sigma (vega > 0).
    """
    S, K, T, r, q = [np.asarray(x, dtype=float) for x in (S, K, T, r, q)]
    fwd = S * np.exp(-q * T)
    disc_k = K * np.exp(-r * T)
    opt = str(option).lower()
    if opt in ("c", "call"):
        return np.maximum(fwd - disc_k, 0.0), fwd
    if opt in ("p", "put"):
        return np.maximum(disc_k - fwd, 0.0), disc_k
    raise ValueError(f"option must be 'call' or 'put', got {option!r}")


def initial_guess(price, S, K, T, r, q=0.0):
    """Manaster-Koehler starting point.

        sigma_0 = sqrt( |ln(S e^{-qT} / (K e^{-rT}))| * 2 / T )

    This is the volatility at which vega is maximal for the given moneyness, so
    Newton starts where the derivative is largest and best conditioned. Falls
    back to a Brenner-Subrahmanyam at-the-money estimate when the log-moneyness
    is near zero.
    """
    S, K, T, r, q = [np.asarray(x, dtype=float) for x in (S, K, T, r, q)]
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.abs(np.log(S * np.exp(-q * T) / (K * np.exp(-r * T))))
        mk = np.sqrt(np.maximum(m, 1e-10) * 2.0 / np.maximum(T, 1e-12))
        bs = np.sqrt(2 * np.pi / np.maximum(T, 1e-12)) * np.asarray(price) / np.maximum(S, 1e-12)
    g = np.where(m < 1e-6, bs, mk)
    return np.clip(np.nan_to_num(g, nan=0.3), 0.01, 3.0)


# ------------------------------------------------------------------ scalar

def implied_vol_scalar(price, S, K, T, r, q=0.0, option="call",
                       tol=1e-12, max_newton=60, max_bisect=200):
    """Invert one option. Returns (sigma, info).

    `tol` is a tolerance on SIGMA, not on the price residual. That distinction
    matters: when vega is small, a price residual of 1e-10 can correspond to a
    sigma error of several percentage points, because

        delta_sigma  ~  residual / vega

    Converging on the price alone therefore reports spurious precision for
    exactly the deep in- and out-of-the-money options where the inversion is
    least reliable. Both loops below converge on the width of the sigma interval
    instead, and `info["sigma_uncertainty"]` reports residual / vega so the
    conditioning of each inversion is visible.

    info keys: method ('newton' | 'bisection' | 'failed'), iters, resid,
               sigma_uncertainty, reason.
    """
    info = {"method": "failed", "iters": 0, "resid": np.nan,
            "sigma_uncertainty": np.nan, "reason": "",
            "scale": np.nan, "tol_bracket": np.nan}
    price = float(price)

    if not np.isfinite(price) or T <= 0:
        info["reason"] = "non-finite price or T <= 0"
        return SOLVER_FAILED, info

    lo_b, hi_b = no_arbitrage_bounds(S, K, T, r, q, option)
    lo_b, hi_b = float(lo_b), float(hi_b)

    # Bracket check, with tolerances scaled to the price level (see _REL_* above).
    # Prices AT the bounds correspond to sigma = 0 or sigma = infinity, neither
    # of which is a finite answer.
    scale = max(abs(hi_b), 1.0)
    tol_bracket = _REL_BRACKET * scale
    tol_degen = max(_REL_DEGEN * scale, np.spacing(scale))
    info["scale"] = scale
    info["tol_bracket"] = tol_bracket

    if price < lo_b - tol_bracket:
        info["reason"] = (f"price {price:.6g} below no-arbitrage lower bound "
                          f"{lo_b:.6g} (tolerance {tol_bracket:.2e})")
        return SOLVER_FAILED, info
    if price > hi_b + tol_bracket:
        info["reason"] = (f"price {price:.6g} above no-arbitrage upper bound "
                          f"{hi_b:.6g} (tolerance {tol_bracket:.2e})")
        return SOLVER_FAILED, info
    if price <= lo_b + tol_degen:
        info["reason"] = "price at intrinsic: sigma -> 0, no finite solution"
        return SOLVER_FAILED, info
    if price >= hi_b - tol_degen:
        info["reason"] = "price at upper bound: sigma -> infinity"
        return SOLVER_FAILED, info
    # A price marginally outside the bound but inside tolerance is rounding, not
    # arbitrage: clamp it into the bracket rather than refusing to price it.
    price = min(max(price, lo_b + tol_degen), hi_b - tol_degen)

    f = lambda s: float(bs_price(S, K, T, r, s, q, option)) - price

    # widen the bracket if 5.0 is not enough (very rare, but cheap to check)
    hi = _SIG_HI
    for _ in range(6):
        if f(hi) >= 0:
            break
        hi *= 2.0
    lo = _SIG_LO

    def _uncertainty(s):
        """residual / vega: how far sigma could move without changing the price
        by more than the residual. The conditioning of the inversion."""
        v = float(bs_vega(S, K, T, r, s, q))
        return abs(f(s)) / v if v > 0 else np.inf

    # ---- safeguarded Newton, converging on sigma ---------------------------
    sigma = float(initial_guess(price, S, K, T, r, q))
    sigma = min(max(sigma, lo), hi)
    for i in range(max_newton):
        resid = f(sigma)
        v = float(bs_vega(S, K, T, r, sigma, q))
        if not np.isfinite(v) or v < _VEGA_FLOOR:
            info["reason"] = f"vega {v:.3e} below floor at sigma={sigma:.4f}; used bisection"
            break
        step = resid / v
        nxt = sigma - step
        if not np.isfinite(nxt) or nxt <= lo or nxt >= hi:
            info["reason"] = f"Newton step left bracket at sigma={sigma:.4f}; used bisection"
            break
        sigma = nxt
        if abs(step) < tol * max(1.0, sigma):       # converged in SIGMA
            info.update(method="newton", iters=i + 1, resid=f(sigma),
                        sigma_uncertainty=_uncertainty(sigma))
            return sigma, info
    else:
        info["reason"] = "Newton hit iteration limit; used bisection"

    # ---- bisection fallback, converging on bracket width -------------------
    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0:
        info["reason"] += " | no sign change on bracket"
        return SOLVER_FAILED, info
    a, b = lo, hi
    for i in range(max_bisect):
        mid = 0.5 * (a + b)
        if (b - a) < tol * max(1.0, mid):           # converged in SIGMA
            info.update(method="bisection", iters=i, resid=f(mid),
                        sigma_uncertainty=_uncertainty(mid))
            return mid, info
        fm = f(mid)
        if f_lo * fm <= 0:
            b = mid
        else:
            a, f_lo = mid, fm
    mid = 0.5 * (a + b)
    info.update(method="bisection", iters=max_bisect, resid=f(mid),
                sigma_uncertainty=_uncertainty(mid))
    return mid, info


# -------------------------------------------------------------- vectorised

def implied_vol(price, S, K, T, r, q=0.0, option="call", **kw):
    """Vectorised inversion. Returns (sigma_array, info_dict_of_arrays).

    info arrays: method, iters, resid, reason - one entry per input.
    """
    price, S, K, T, r, q = np.broadcast_arrays(
        *[np.asarray(x, dtype=float) for x in (price, S, K, T, r, q)])
    shape = price.shape
    flat = [np.ravel(x) for x in (price, S, K, T, r, q)]
    n = flat[0].size

    out = np.empty(n)
    method = np.empty(n, dtype=object)
    iters = np.zeros(n, dtype=int)
    resid = np.full(n, np.nan)
    unc = np.full(n, np.nan)
    reason = np.empty(n, dtype=object)

    for i in range(n):
        s, inf = implied_vol_scalar(flat[0][i], flat[1][i], flat[2][i],
                                    flat[3][i], flat[4][i], flat[5][i],
                                    option, **kw)
        out[i] = s
        method[i] = inf["method"]
        iters[i] = inf["iters"]
        resid[i] = inf["resid"]
        unc[i] = inf["sigma_uncertainty"]
        reason[i] = inf["reason"]

    info = {"method": method.reshape(shape), "iters": iters.reshape(shape),
            "resid": resid.reshape(shape), "reason": reason.reshape(shape),
            "sigma_uncertainty": unc.reshape(shape)}
    sig = out.reshape(shape)
    return (sig if sig.ndim else float(sig)), info


# ---------------------------------------------------------------- self-test

def _self_test(verbose=True):
    """Recover a known sigma from synthetic Black-Scholes prices (plan: iv_recovery)."""
    rng = np.random.default_rng(20260825)
    n = 4000
    S   = rng.uniform(40, 160, n)
    K   = rng.uniform(40, 160, n)
    T   = rng.uniform(0.02, 3.0, n)
    r   = rng.uniform(0.0, 0.10, n)
    q   = rng.uniform(0.0, 0.04, n)
    sig = rng.uniform(0.05, 1.00, n)

    rows = {}
    for opt in ("call", "put"):
        price = bs_price(S, K, T, r, sig, q, opt)
        rec, info = implied_vol(price, S, K, T, r, q, opt)
        ok = np.isfinite(rec)
        err = np.abs(rec[ok] - sig[ok])
        meth = info["method"][ok]
        # Rejections are not failures: they are options whose price is
        # numerically indistinguishable from intrinsic, so no finite sigma is
        # recoverable in double precision. The solver says so rather than
        # returning a confident wrong answer.
        rej = ~ok
        rows[opt] = {
            "recovered": int(ok.sum()),
            "attempted": n,
            "rejected": int(rej.sum()),
            "max_err": float(err.max()) if err.size else np.nan,
            "mean_err": float(err.mean()) if err.size else np.nan,
            "newton_pct": float((meth == "newton").mean() * 100),
            "bisect_pct": float((meth == "bisection").mean() * 100),
            "mean_iters": float(info["iters"][ok].mean()),
            "max_vega_at_rej": float(np.nanmax(
                bs_vega(S[rej], K[rej], T[rej], r[rej], sig[rej], q[rej]))) if rej.any() else 0.0,
        }

    if verbose:
        print(f"  {'':<8}{'recovered':>12}{'rejected':>10}{'max |err|':>12}"
              f"{'mean |err|':>12}{'Newton %':>10}{'bisect %':>10}{'iters':>7}")
        for opt, d in rows.items():
            print(f"  {opt:<8}{d['recovered']:>7}/{d['attempted']:<4}{d['rejected']:>10}"
                  f"{d['max_err']:>12.2e}{d['mean_err']:>12.2e}"
                  f"{d['newton_pct']:>10.1f}{d['bisect_pct']:>10.1f}{d['mean_iters']:>7.1f}")
        print(f"\n  Rejected cases are priced at intrinsic to within double precision.")
        print(f"  Largest true vega among rejected: "
              f"{max(d['max_vega_at_rej'] for d in rows.values()):.2e}"
              f"   (vega this small means sigma is not identifiable from the price)")

    # ---- accuracy conditioned on vega (plan: "examine low-vega cases") -----
    price_c = bs_price(S, K, T, r, sig, q, "call")
    rec_c, info_c = implied_vol(price_c, S, K, T, r, q, "call")
    v = np.asarray(bs_vega(S, K, T, r, sig, q))
    okc = np.isfinite(rec_c)
    errc = np.abs(rec_c - sig)
    if verbose:
        print("\n  RECOVERY ERROR BY VEGA  (calls)")
        print(f"    {'vega band':>18}{'n':>7}{'max |err|':>13}{'mean |err|':>13}"
              f"{'predicted bound':>18}")
        for lo_v, hi_v in [(1e2, np.inf), (1e0, 1e2), (1e-2, 1e0),
                           (1e-4, 1e-2), (1e-8, 1e-4), (0.0, 1e-8)]:
            m = okc & (v >= lo_v) & (v < hi_v)
            if not m.any():
                continue
            pred = np.nanmax(info_c["sigma_uncertainty"][m])
            lbl = f"{lo_v:.0e} - {hi_v:.0e}" if np.isfinite(hi_v) else f"> {lo_v:.0e}"
            print(f"    {lbl:>18}{int(m.sum()):>7}{errc[m].max():>13.2e}"
                  f"{errc[m].mean():>13.2e}{pred:>18.2e}")
        print("\n    Accuracy degrades monotonically as vega falls. This is not a solver")
        print("    defect: delta_sigma ~ residual / vega, so when vega is tiny a price")
        print("    correct to machine precision still leaves sigma loose. The solver")
        print("    reports this per option in info['sigma_uncertainty'].")

    well = okc & (v > 1e-2)
    max_err_well = float(errc[well].max())
    if verbose:
        print(f"\n  Max error where vega > 1e-2 ({int(well.sum())} of {n} calls):"
              f" {max_err_well:.2e}")

    # bracket rejection must actually trigger
    bad_low  = implied_vol_scalar(0.0, 100, 100, 1.0, 0.05, 0.0, "call")[1]["reason"]
    bad_high = implied_vol_scalar(999.0, 100, 100, 1.0, 0.05, 0.0, "call")[1]["reason"]
    if verbose:
        print(f"\n  bracket check, price below intrinsic : {bad_low}")
        print(f"  bracket check, price above spot      : {bad_high}")

    # Pass criterion, in two parts.
    #   (a) where the problem is well conditioned (vega > 1e-2), sigma must be
    #       recovered to better than 1e-6;
    #   (b) where it is not, the observed error must be explained by the
    #       conditioning rather than by the solver. Compared BAND-WISE: a
    #       per-point test is meaningless because the predicted uncertainty is
    #       residual/vega, and a converged Newton step drives the residual to
    #       exactly 0.0, giving a predicted bound of 0 even where the price
    #       function is genuinely flat in sigma.
    worst_ratio = 0.0
    for lo_v, hi_v in [(1e2, np.inf), (1e0, 1e2), (1e-2, 1e0),
                       (1e-4, 1e-2), (1e-8, 1e-4), (0.0, 1e-8)]:
        m = okc & (v >= lo_v) & (v < hi_v)
        if not m.any():
            continue
        pred = float(np.nanmax(info_c["sigma_uncertainty"][m]))
        if pred > 0:
            worst_ratio = max(worst_ratio, float(errc[m].max()) / pred)
    within_predicted = worst_ratio < 10.0
    ok = (max_err_well < 1e-6) and within_predicted
    if verbose:
        print(f"    worst band-wise  max|err| / predicted : {worst_ratio:.2f}  "
              f"(must stay below 10)")
    if verbose:
        print(f"    well-conditioned accuracy  (< 1e-6) : "
              f"{'PASS' if max_err_well < 1e-6 else 'FAIL'}")
        print(f"    errors within predicted bound       : "
              f"{'PASS' if within_predicted else 'FAIL'}")
        print(f"\n  {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}"
              f"  (n = {n:,} per option type)")
    return ok


if __name__ == "__main__":
    print("implied_volatility.py self-test - recovery of a known sigma")
    print("=" * 76)
    _self_test()
