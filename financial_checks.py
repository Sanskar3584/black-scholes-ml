"""
financial_checks.py - do the predictions behave like option prices?
====================================================================

Project : AI/ML-Augmented Black-Scholes Framework
Ref     : master plan 5.28; ML_experiment_plan.md 9.1; supervisor note 11 Nov
Text    : Carol Alexander, Market Risk Analysis Vol. I, section I.5 (bounds)

Why counts, not averages
------------------------
A model with an excellent mean error can still return a negative price on one
point in fifty. An average hides that completely; a count does not. Every
check below therefore reports the NUMBER of violations, the percentage, and
the single worst one, broken down by moneyness so it is visible WHERE a model
misbehaves rather than only that it does.

This was the supervisor's specific request on 11 November, and the reason
given was concrete: an unconstrained network can produce negative prices on a
noticeable share of points.

The checks
----------
For a European call with continuous dividend yield q, writing the scaled
target y = C/K and m = ln(S/K):

  1  non-negativity        C >= 0
  2  intrinsic lower bound C >= max(0, S e^{-qT} - K e^{-rT})
  3  upper bound           C <= S e^{-qT}
  4  monotone in S         dC/dS >= 0   (equivalently, non-decreasing in m)
  5  monotone in sigma     dC/dsigma >= 0
  6  convex in K           d2C/dK2 >= 0

1 to 3 are pointwise and are evaluated on the test sample.

4 to 6 are statements about the FITTED FUNCTION, not about any particular
sample, so evaluating them on scattered test points would be meaningless: two
test rows that differ in m also differ in T, r and sigma, so a decrease
between them says nothing. They are therefore swept on a grid, holding the
other inputs fixed and varying one at a time. That also probes regions the
test split happens not to cover, which is where an unconstrained model is
most likely to misbehave.

Convexity in K is the subtle one. In terms of the scaled target the
relationship is not simply convexity of y in m, because both the target and
the feature move with K. The sweep therefore reconstructs the unscaled price
C = K * y at fixed S and varies K directly, which is the form the
no-arbitrage result is stated in: a butterfly spread cannot have negative
value, so the second difference of C in K must be non-negative.
"""

import numpy as np
import pandas as pd

__all__ = ["pointwise_checks", "sweep_checks", "check_table", "summarise",
           "TICK"]

_EPS = 1e-9

# One tick on an NSE index option is Rs 0.05. Supervisor note, 17 November:
# a count taken at a numerical tolerance of ~1e-7 includes violations far
# below anything tradable, so every check is reported at BOTH thresholds.
#
# The distinction is not cosmetic. A price 1e-9 below zero is a rounding
# artefact and nobody could act on it; a price Rs 0.40 below zero is a quote
# a desk would have to refuse. On real data, where the bid-ask spread is
# itself several ticks, only the material count will mean anything - so the
# habit of reporting both starts here, where the answer is already known.
TICK = 0.05


# --------------------------------------------------------------------------
# pointwise checks, on the test sample
# --------------------------------------------------------------------------

def pointwise_checks(pred_scaled, S, K, T, r, q, m=None, tol=_EPS,
                     tick=TICK):
    """Checks 1 to 3 on actual sample points.

    pred_scaled is the model's prediction of y = C/K; it is converted back to
    a price before the bounds are applied, because the bounds are statements
    about prices.
    """
    pred_scaled = np.asarray(pred_scaled, float)
    S, K = np.asarray(S, float), np.asarray(K, float)
    T, r, q = np.asarray(T, float), np.asarray(r, float), np.asarray(q, float)
    C = pred_scaled * K
    if m is None:
        m = np.log(S / K)

    lower = np.maximum(0.0, S * np.exp(-q * T) - K * np.exp(-r * T))
    upper = S * np.exp(-q * T)

    return {
        "m": m,
        "non_negative": C >= -tol,
        "above_intrinsic": C >= lower - tol,
        "below_upper": C <= upper + tol,
        "viol_non_negative": np.maximum(0.0, -C),
        "viol_above_intrinsic": np.maximum(0.0, lower - C),
        "viol_below_upper": np.maximum(0.0, C - upper),
        # material versions: violations larger than one tick
        "tick_non_negative": C >= -tick,
        "tick_above_intrinsic": C >= lower - tick,
        "tick_below_upper": C <= upper + tick,
    }


# --------------------------------------------------------------------------
# grid sweeps, on the fitted function
# --------------------------------------------------------------------------

def sweep_checks(predict_scaled, S=100.0, q=0.015, n_grid=201,
                 T_list=(0.02, 0.08, 0.25, 1.0, 2.0),
                 sigma_list=(0.10, 0.25, 0.45, 0.60),
                 r_list=(0.03, 0.055, 0.08),
                 m_range=(-0.6, 0.6), tol=1e-7, tick=TICK):
    """Checks 4 to 6 by sweeping one input at a time.

    predict_scaled(features) takes an (n, 4) array of [m, T, r, sigma] and
    returns the predicted y = C/K.

    Returns a dict with, for each check, the number of grid points violating
    it, the total swept, and the worst violation.
    """
    out = {}

    # ---- 4: monotone in S, i.e. non-decreasing in m at fixed T, r, sigma --
    # C = K y and K = S e^{-m} with S fixed, so varying m at fixed S varies K.
    # dC/dS at fixed K is what the theory constrains; sweeping m at fixed S
    # moves K, so the quantity that must be non-decreasing is the price of a
    # FIXED strike as S rises. Hold K fixed and vary S instead.
    K_fix = 100.0
    worst_mono_S, n_bad_S, n_tot_S, n_tick_S = 0.0, 0, 0, 0
    for T in T_list:
        for sg in sigma_list:
            for r in r_list:
                S_grid = K_fix * np.exp(np.linspace(m_range[0], m_range[1],
                                                    n_grid))
                m_grid = np.log(S_grid / K_fix)
                X = np.column_stack([m_grid, np.full(n_grid, T),
                                     np.full(n_grid, r), np.full(n_grid, sg)])
                C = predict_scaled(X) * K_fix
                d = np.diff(C)
                n_tot_S += len(d)
                bad = d < -tol
                n_bad_S += int(bad.sum())
                n_tick_S += int((d < -tick).sum())
                if bad.any():
                    worst_mono_S = max(worst_mono_S, float(-d[bad].min()))
    out["monotone_in_S"] = {"n_bad": n_bad_S, "n_total": n_tot_S,
                            "worst": worst_mono_S, "n_tick": n_tick_S}

    # ---- 5: monotone in sigma at fixed m, T, r --------------------------
    sg_grid = np.linspace(0.10, 0.60, n_grid)
    worst_v, n_bad_v, n_tot_v, n_tick_v = 0.0, 0, 0, 0
    for T in T_list:
        for r in r_list:
            for m in np.linspace(m_range[0], m_range[1], 25):
                X = np.column_stack([np.full(n_grid, m), np.full(n_grid, T),
                                     np.full(n_grid, r), sg_grid])
                C = predict_scaled(X) * K_fix
                d = np.diff(C)
                n_tot_v += len(d)
                bad = d < -tol
                n_bad_v += int(bad.sum())
                n_tick_v += int((d < -tick).sum())
                if bad.any():
                    worst_v = max(worst_v, float(-d[bad].min()))
    out["monotone_in_sigma"] = {"n_bad": n_bad_v, "n_total": n_tot_v,
                                "worst": worst_v, "n_tick": n_tick_v}

    # ---- 6: convex in K at fixed S, T, r, sigma -------------------------
    # Reported as the BUTTERFLY VALUE rather than as the second derivative.
    # For equally spaced K1 < K2 < K3 the spread (long K1, short two K2, long
    # K3) costs C(K1) - 2C(K2) + C(K3) and can never pay out less than zero,
    # so a negative value is a free lunch. That quantity is in rupees, which
    # the second derivative - units of 1/rupee - is not, and only the rupee
    # version can be compared against a tick.
    #
    # The strike grid is therefore UNIFORM here, unlike the m-uniform grid
    # used for the other sweeps, because a butterfly needs equal spacing.
    worst_c, n_bad_c, n_tot_c, n_tick_c = 0.0, 0, 0, 0
    K_grid = np.linspace(S * np.exp(-m_range[1]), S * np.exp(-m_range[0]),
                         n_grid)
    m_grid = np.log(S / K_grid)
    for T in T_list:
        for sg in sigma_list:
            for r in r_list:
                X = np.column_stack([m_grid, np.full(n_grid, T),
                                     np.full(n_grid, r), np.full(n_grid, sg)])
                C = predict_scaled(X) * K_grid
                fly = C[:-2] - 2.0 * C[1:-1] + C[2:]      # rupees
                n_tot_c += len(fly)
                bad = fly < -tol
                n_bad_c += int(bad.sum())
                n_tick_c += int((fly < -tick).sum())
                if bad.any():
                    worst_c = max(worst_c, float(-fly[bad].min()))
    out["convex_in_K"] = {"n_bad": n_bad_c, "n_total": n_tot_c,
                          "worst": worst_c, "n_tick": n_tick_c,
                          "spacing": float(K_grid[1] - K_grid[0])}
    return out


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

_M_BINS = [-0.61, -0.4, -0.2, -0.05, 0.05, 0.2, 0.4, 0.61]


def check_table(name, pw, sw, K_ref=100.0):
    """One row per check for one model."""
    rows = []
    for key, label in (("non_negative", "C >= 0"),
                       ("above_intrinsic", "C >= intrinsic"),
                       ("below_upper", "C <= S e^-qT")):
        ok = pw[key]
        v = pw["viol_" + key]
        okt = pw["tick_" + key]
        rows.append({
            "model": name, "check": label, "scope": "test sample",
            "n_total": int(len(ok)), "n_violations": int((~ok).sum()),
            "pct": 100.0 * float((~ok).mean()),
            "n_material": int((~okt).sum()),
            "pct_material": 100.0 * float((~okt).mean()),
            "worst": float(v.max()) if len(v) else 0.0,
        })
    for key, label in (("monotone_in_S", "dC/dS >= 0"),
                       ("monotone_in_sigma", "dC/dsigma >= 0"),
                       ("convex_in_K", "d2C/dK2 >= 0")):
        d = sw[key]
        rows.append({
            "model": name, "check": label, "scope": "grid sweep",
            "n_total": int(d["n_total"]), "n_violations": int(d["n_bad"]),
            "pct": 100.0 * d["n_bad"] / max(d["n_total"], 1),
            "n_material": int(d.get("n_tick", 0)),
            "pct_material": 100.0 * d.get("n_tick", 0) / max(d["n_total"], 1),
            "worst": float(d["worst"]),
        })
    return pd.DataFrame(rows)


def by_moneyness(name, pw):
    """Where in moneyness do the pointwise violations sit?"""
    d = pd.DataFrame({
        "m": pw["m"],
        "bad": ~(pw["non_negative"] & pw["above_intrinsic"]
                 & pw["below_upper"]),
    })
    d["bin"] = pd.cut(d["m"], _M_BINS)
    g = d.groupby("bin", observed=True)["bad"].agg(["size", "sum"])
    g.columns = ["n", "violations"]
    g["pct"] = (100.0 * g["violations"] / g["n"]).round(2)
    g.insert(0, "model", name)
    return g.reset_index()


def summarise(tab, verbose=True):
    """Print the check table grouped by model."""
    if verbose:
        print("  violations at a numerical tolerance (1e-9 / 1e-7), and")
        print("  MATERIAL violations larger than one tick (Rs %.2f).\n" % TICK)
        print("  %-24s %-16s %-11s %8s %7s %9s %11s"
              % ("model", "check", "scope", "n viol", "%", "material",
                 "worst"))
        for mdl, g in tab.groupby("model", sort=False):
            for _, rrow in g.iterrows():
                flag = "" if rrow.n_material == 0 else "  <--"
                print("  %-24s %-16s %-11s %8d %6.2f%% %9d %11.3e%s"
                      % (mdl[:24], rrow.check, rrow.scope,
                         rrow.n_violations, rrow.pct, rrow.n_material,
                         rrow.worst, flag))
    return tab


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def _self_test(verbose=True):
    if verbose:
        print("=" * 78)
        print("financial_checks.py  SELF-TEST")
        print("=" * 78)

    from black_scholes import bs_call

    checks = []

    def chk(name, cond):
        checks.append(bool(cond))
        if verbose:
            print("  %-62s %s" % (name, "ok" if cond else "FAIL"))

    # ---- Black-Scholes itself must pass every check ----
    # If the framework flags the true pricer, the framework is wrong.
    def bs_pred(X):
        m, T, r, sg = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        K = 100.0
        S = K * np.exp(m)
        return bs_call(S, K, T, r, sg, 0.015) / K

    sw = sweep_checks(bs_pred, n_grid=101)
    for k, v in sw.items():
        chk("Black-Scholes satisfies %s (%d/%d bad)"
            % (k, v["n_bad"], v["n_total"]), v["n_bad"] == 0)

    rng = np.random.default_rng(20260825)
    n = 2000
    m = rng.uniform(-0.6, 0.6, n)
    K = np.full(n, 100.0)
    S = K * np.exp(m)
    T = rng.uniform(0.02, 2.0, n)
    r = rng.uniform(0.03, 0.08, n)
    sg = rng.uniform(0.1, 0.6, n)
    q = np.full(n, 0.015)
    y = bs_call(S, K, T, r, sg, q) / K
    pw = pointwise_checks(y, S, K, T, r, q, m=m)
    chk("Black-Scholes passes non-negativity", pw["non_negative"].all())
    chk("Black-Scholes passes the intrinsic bound", pw["above_intrinsic"].all())
    chk("Black-Scholes passes the upper bound", pw["below_upper"].all())

    # ---- deliberately broken predictors must be caught ----
    pw_bad = pointwise_checks(y - 0.02, S, K, T, r, q, m=m)
    chk("a downward shift breaks non-negativity",
        (~pw_bad["non_negative"]).sum() > 0)
    chk("the worst violation is reported in price units",
        pw_bad["viol_non_negative"].max() > 0.5)

    pw_hi = pointwise_checks(y * 5.0 + 1.0, S, K, T, r, q, m=m)
    chk("an inflated prediction breaks the upper bound",
        (~pw_hi["below_upper"]).sum() > 0)

    # a decreasing function of m must fail monotonicity in S
    sw_dec = sweep_checks(lambda X: -0.3 * X[:, 0] + 0.5, n_grid=51)
    chk("a decreasing predictor fails dC/dS >= 0",
        sw_dec["monotone_in_S"]["n_bad"] > 0)

    # a function decreasing in sigma must fail vega positivity
    sw_v = sweep_checks(lambda X: 0.2 * np.exp(X[:, 0]) - 0.1 * X[:, 3],
                        n_grid=51)
    chk("a sigma-decreasing predictor fails dC/dsigma >= 0",
        sw_v["monotone_in_sigma"]["n_bad"] > 0)

    # a concave-in-K price must fail the butterfly test.
    # C = K*y with y = a - b*m, m = ln(S/K): C(K) = K(a - b ln S + b ln K),
    # whose second derivative in K is b/K > 0 for b > 0, so take b < 0.
    sw_cc = sweep_checks(lambda X: 0.3 + 0.25 * X[:, 0], n_grid=51)
    chk("a concave-in-K predictor fails d2C/dK2 >= 0",
        sw_cc["convex_in_K"]["n_bad"] > 0)

    # ---- the table reports counts, not averages ----
    tab = check_table("test-model", pw_bad, sw_dec)
    chk("table has one row per check", len(tab) == 6)
    chk("table carries violation counts", "n_violations" in tab.columns)
    chk("counts are non-zero where they should be",
        int(tab.loc[tab.check == "C >= 0", "n_violations"].iloc[0]) > 0)

    # ---- the materiality threshold separates noise from tradable ----
    # A shift of half a tick must register at the numerical tolerance and
    # NOT at the tick threshold; a shift of ten ticks must register at both.
    pw_small = pointwise_checks(y - 0.5 * TICK / 100.0, S, K, T, r, q, m=m)
    t_small = check_table("tiny", pw_small, sw)
    row = t_small[t_small.check == "C >= 0"].iloc[0]
    chk("a sub-tick breach is counted at the tolerance",
        int(row.n_violations) > 0)
    chk("a sub-tick breach is NOT counted as material",
        int(row.n_material) == 0)

    pw_big = pointwise_checks(y - 10 * TICK / 100.0, S, K, T, r, q, m=m)
    t_big = check_table("big", pw_big, sw)
    rowb = t_big[t_big.check == "C >= 0"].iloc[0]
    chk("a ten-tick breach is counted as material",
        int(rowb.n_material) > 0)
    chk("material count never exceeds the tolerance count",
        bool((pd.concat([tab, t_small, t_big]).eval(
            "n_material <= n_violations")).all()))

    # ---- convexity is reported in rupees, so a tick means something ----
    chk("convexity is reported as a butterfly value in price units",
        "spacing" in sw["convex_in_K"])
    bm = by_moneyness("test-model", pw_bad)
    chk("moneyness breakdown covers the sample",
        int(bm["n"].sum()) == len(m))

    n_fail = sum(1 for ok in checks if not ok)
    if verbose:
        print("-" * 78)
        print("  %d checks, %d failed" % (len(checks), n_fail))
        print("  ALL CHECKS PASSED" if n_fail == 0 else "  *** FAILURES ABOVE ***")
        print("=" * 78)
    return n_fail == 0


if __name__ == "__main__":
    raise SystemExit(0 if _self_test() else 1)
