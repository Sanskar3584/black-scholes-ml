"""
Post-midsem Week 3 - Greeks and Implied Volatility
==================================================

Project : AI/ML-Augmented Black-Scholes Framework
Ref     : Post-midsem plan, Week 3 (five work packages)
Text     : Carol Alexander, Market Risk Analysis Vol. I, section I.5

Work packages
-------------
    1  Analytic Delta, Gamma, Vega for European calls and puts   -> greeks.py
    2  Check Greeks with central finite differences;
       quantify error vs step size                               -> section 3 below
    3  Plot sensitivity vs spot, moneyness, volatility, maturity -> greeks_plots/
    4  Newton-Raphson with bisection fallback, bracket checks    -> implied_volatility.py
    5  Recover known sigma from synthetic prices;
       examine low-vega cases                                    -> sections 5-6 below

Naming note: the project's earlier drivers are week1.py (August W1), week2.py
(September W1) and week3.py (September W2). This file is the POST-MIDSEM Week 3
package and is named accordingly to avoid a collision with week3.py.

Usage
-----
    python postmidsem_week3.py

Outputs
-------
    greeks_plots/*.png
    results/w04_*.csv
    results/w04_console_output.txt
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

from black_scholes import bs_call, bs_put, bs_price, moneyness
import greeks as gk
import implied_volatility as iv

# =========================================================================== config

SEED       = 20260825
SHOW_PLOTS = True
PLOTDIR    = "greeks_plots"

rng = np.random.default_rng(SEED)
if not SHOW_PLOTS:
    matplotlib.use("Agg")

plt.rcParams.update({
    "figure.figsize": (11, 4.2), "axes.grid": True, "grid.alpha": 0.3,
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
})


class _Tee:
    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8"); self.stdout = sys.stdout
    def write(self, s): self.stdout.write(s); self.file.write(s)
    def flush(self): self.stdout.flush(); self.file.flush()
    def close(self): self.file.close()


def rule(t=""):
    print("\n" + "=" * 78)
    if t: print(t); print("=" * 78)


def finish(fig, name):
    fig.tight_layout()
    fig.savefig(f"{PLOTDIR}/{name}.png", dpi=150, bbox_inches="tight")
    plt.show() if SHOW_PLOTS else plt.close(fig)


# ============================================== 2. FINITE-DIFFERENCE VALIDATION

def fd_greeks(S, K, T, r, sigma, q=0.0, option="call", h_S=1e-3, h_sig=1e-4, h_T=1e-5):
    """Central differences on the closed-form price."""
    p = lambda s=S, k=K, t=T, rr=r, sg=sigma: float(bs_price(s, k, t, rr, sg, q, option))
    return {
        "delta": (p(s=S + h_S) - p(s=S - h_S)) / (2 * h_S),
        "gamma": (p(s=S + h_S) - 2 * p() + p(s=S - h_S)) / h_S ** 2,
        "vega":  (p(sg=sigma + h_sig) - p(sg=sigma - h_sig)) / (2 * h_sig),
        "theta": -(p(t=T + h_T) - p(t=T - h_T)) / (2 * h_T),   # t = -tau
    }


def section2_fd_check(K=100.0, r=0.05, sigma=0.25, T=1.0):
    """Analytic against central differences at representative points."""
    rows = []
    for option in ("call", "put"):
        for S in (70., 85., 100., 115., 130.):
            a = gk.all_greeks(S, K, T, r, sigma, 0.0, option)
            f = fd_greeks(S, K, T, r, sigma, 0.0, option)
            for g in ("delta", "gamma", "vega", "theta"):
                rows.append({"option": option, "S": S, "greek": g,
                             "analytic": a[g], "finite diff": f[g],
                             "abs error": abs(a[g] - f[g]),
                             "rel error": abs(a[g] - f[g]) / max(abs(a[g]), 1e-12)})
    tab = pd.DataFrame(rows)
    show = tab[(tab["option"] == "call")]
    print("  European call, K = 100, T = 1, r = 5%, sigma = 25%\n")
    print(f"  {'S':>6}{'greek':>8}{'analytic':>14}{'finite diff':>14}"
          f"{'abs error':>13}{'rel error':>12}")
    for _, x in show.iterrows():
        print(f"  {x['S']:>6.0f}{x['greek']:>8}{x['analytic']:>14.8f}"
              f"{x['finite diff']:>14.8f}{x['abs error']:>13.2e}{x['rel error']:>12.2e}")
    print(f"\n  Across calls AND puts ({len(tab)} comparisons):")
    print(f"    max absolute error  {tab['abs error'].max():.3e}")
    print(f"    max relative error  {tab['rel error'].max():.3e}")
    print("\n  The agreement is limited by the finite-difference step, not by the")
    print("  analytic formulas. Section 3 quantifies that trade-off directly.")
    return tab


# ==================================================== 3. STEP-SIZE STUDY

def section3_step_size(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.25):
    """Error against step size h - the central question the plan asks.

    A central difference carries two competing errors:

        truncation  O(h^2)        from the Taylor remainder
        round-off   O(eps / h)    from cancelling nearly equal prices

    Total error is therefore V-shaped in h on log-log axes, with a minimum at
    h* ~ eps^(1/3) ~ 6e-6 for a FIRST derivative and h* ~ eps^(1/4) ~ 1.2e-4 for
    a SECOND derivative, where the cancellation is worse (O(eps/h^2)).

    Taking h too small is as damaging as taking it too large. That is the point
    of the experiment.
    """
    eps = np.finfo(float).eps
    hs = np.logspace(-12, 0, 49)

    a_delta = gk.delta(S, K, T, r, sigma)
    a_gamma = gk.gamma(S, K, T, r, sigma)
    a_vega  = gk.vega(S, K, T, r, sigma)

    # The sigma step must keep sigma positive: a central difference with
    # h >= sigma would evaluate the formula at a negative volatility, which is
    # meaningless rather than merely inaccurate. Cap the vega scan at sigma/2.
    h_max_vega = sigma / 2
    e_d, e_g, e_v = [], [], []
    for h in hs:
        p = lambda s=S, sg=sigma: float(bs_call(s, K, T, r, sg))
        e_d.append(abs((p(s=S + h) - p(s=S - h)) / (2 * h) - a_delta))
        e_g.append(abs((p(s=S + h) - 2 * p() + p(s=S - h)) / h ** 2 - a_gamma))
        e_v.append(abs((p(sg=sigma + h) - p(sg=sigma - h)) / (2 * h) - a_vega)
                   if h < h_max_vega else np.nan)
    e_d, e_g, e_v = map(np.array, (e_d, e_g, e_v))

    tab = pd.DataFrame({"h": hs, "delta error": e_d,
                        "gamma error": e_g, "vega error": e_v})
    best = {"delta": hs[np.nanargmin(e_d)], "gamma": hs[np.nanargmin(e_g)],
            "vega": hs[np.nanargmin(e_v)]}

    print("  Optimal step size found by scan\n")
    print(f"  {'quantity':>10}{'h* observed':>15}{'scale-free':>13}"
          f"{'min error':>13}{'error at h=1e-12':>19}")
    scale_free = {"delta": eps ** (1 / 3), "gamma": eps ** (1 / 4), "vega": eps ** (1 / 3)}
    for name, err in (("delta", e_d), ("gamma", e_g), ("vega", e_v)):
        print(f"  {name:>10}{best[name]:>15.2e}{scale_free[name]:>13.2e}"
              f"{np.nanmin(err):>13.2e}{err[0]:>19.2e}")
    print("""
  The 'scale-free' column is eps^(1/3) for a first derivative and eps^(1/4) for a
  second. It is an order-of-magnitude guide, not a prediction: the true optimum is

      h* = ( 3 eps |f| / |f'''| )^(1/3)      and      ( 48 eps |f| / |f''''| )^(1/4)

  so it shifts with the ratio |f| / |higher derivative|, which for an option price
  is large. The observed optima sit one to two orders above the scale-free value
  for exactly that reason. What IS universal is the SHAPE - and the slopes below.""")

    # truncation branch: slope should be 2. Fit well above each optimum, since
    # the window must not straddle the minimum.
    def slope(err, lo, hi):
        m = (hs > lo) & (hs < hi) & np.isfinite(err)
        return stats.linregress(np.log(hs[m]), np.log(np.maximum(err[m], 1e-300))).slope
    sl_d = slope(e_d, 1e-2, 1e0)
    sl_g = slope(e_g, 1e-1, 1e0)
    sl_v = slope(e_v, 1e-3, 1e-1)
    print(f"\n  Truncation-branch slope (theory +2):  delta {sl_d:.3f}, "
          f"gamma {sl_g:.3f}, vega {sl_v:.3f}")
    # round-off branch: -1 for a first derivative, -2 for a second
    sl_rd = slope(e_d, 1e-12, 1e-8)
    sl_rg = slope(e_g, 1e-12, 1e-8)
    print(f"  Round-off-branch slope  (theory -1 / -2): delta {sl_rd:.3f}, "
          f"gamma {sl_rg:.3f}")
    print("  (the round-off branch is noisy by nature - it is rounding, not a smooth law)")

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
    ax[0].loglog(hs, np.maximum(e_d, 1e-20), "o-", ms=3, color="#1f4e79", label="delta")
    mv = np.isfinite(e_v)
    ax[0].loglog(hs[mv], np.maximum(e_v[mv], 1e-20), "s-", ms=3, color="#e08214", label="vega")
    ax[0].axvline(eps ** (1 / 3), color="r", ls="--", lw=1.2,
                  label=r"theory $h^*=\epsilon^{1/3}$")
    ax[0].set(xlabel="step size h", ylabel="absolute error",
              title="First derivatives: error is V-shaped in h")
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].loglog(hs, np.maximum(e_g, 1e-20), "o-", ms=3, color="#c0504d", label="gamma")
    ax[1].axvline(eps ** (1 / 4), color="r", ls="--", lw=1.2,
                  label=r"theory $h^*=\epsilon^{1/4}$")
    ax[1].set(xlabel="step size h", ylabel="absolute error",
              title="Second derivative: cancellation is worse, optimum larger")
    ax[1].legend(frameon=False, fontsize=8)
    finish(fig, "fd_step_size_study")

    print("""
  WHAT THIS MEANS
    Shrinking h does NOT keep improving a finite difference. Below the optimum
    the two prices being subtracted agree to so many digits that the difference
    is mostly rounding noise, and the error grows again. Gamma is worse than
    delta because the second difference divides by h^2.

    Practical consequence: analytic Greeks are preferable not only for speed but
    for accuracy. Where finite differences are unavoidable - the hybrid ML models
    later in the project - h must be chosen near eps^(1/3), not as small as
    possible.""")
    return tab, best


# ================================================= 4. SENSITIVITY PLOTS

def section4_plots(K=100.0, r=0.05, sigma=0.25, T=1.0):
    """greeks_plots/: sensitivity vs spot, moneyness, volatility, maturity."""

    # ---- vs spot ----
    S = np.linspace(20, 180, 500)
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 7.2))
    for opt, col in (("call", "#1f4e79"), ("put", "#c0504d")):
        ax[0, 0].plot(S, gk.delta(S, K, T, r, sigma, 0, opt), lw=1.8, color=col, label=opt)
        ax[1, 1].plot(S, gk.theta(S, K, T, r, sigma, 0, opt), lw=1.8, color=col, label=opt)
    ax[0, 1].plot(S, gk.gamma(S, K, T, r, sigma), lw=1.8, color="#1f4e79")
    ax[1, 0].plot(S, gk.vega(S, K, T, r, sigma), lw=1.8, color="#1f4e79")
    for a, t, yl in ((ax[0, 0], "Delta", r"$\Delta$"), (ax[0, 1], "Gamma", r"$\Gamma$"),
                     (ax[1, 0], "Vega", r"$\nu$"), (ax[1, 1], "Theta (per year)", r"$\Theta$")):
        a.axvline(K, color="grey", lw=0.8); a.set(xlabel="S", ylabel=yl, title=t)
    ax[0, 0].legend(frameon=False, fontsize=8); ax[1, 1].legend(frameon=False, fontsize=8)
    ax[0, 0].axhline(0, color="k", lw=0.6); ax[1, 1].axhline(0, color="k", lw=0.6)
    fig.suptitle("Greeks against spot  (K = 100, T = 1, r = 5%, $\\sigma$ = 25%)", y=1.0)
    finish(fig, "greeks_vs_spot")

    # ---- vs moneyness, several maturities ----
    m = np.linspace(-0.8, 0.8, 500)
    Sm = K * np.exp(m)
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for TT, lab in ((0.08, "1 month"), (0.5, "6 months"), (2.0, "2 years")):
        ax[0].plot(m, gk.delta(Sm, K, TT, r, sigma), lw=1.6, label=lab)
        ax[1].plot(m, gk.gamma(Sm, K, TT, r, sigma), lw=1.6, label=lab)
        ax[2].plot(m, gk.vega(Sm, K, TT, r, sigma), lw=1.6, label=lab)
    for a, t, yl in ((ax[0], "Delta", r"$\Delta$"), (ax[1], "Gamma", r"$\Gamma$"),
                     (ax[2], "Vega", r"$\nu$")):
        a.axvline(0, color="grey", lw=0.9)
        a.set(xlabel=r"$m=\ln(S/K)$", ylabel=yl, title=f"{t} vs moneyness")
        a.legend(frameon=False, fontsize=8)
    fig.suptitle("Gamma and vega peak at the money and collapse as maturity shortens",
                 y=1.0, fontsize=10)
    finish(fig, "greeks_vs_moneyness")

    # ---- vs volatility ----
    sg = np.linspace(0.02, 1.2, 500)
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for S0, lab in ((80, "OTM S=80"), (100, "ATM S=100"), (120, "ITM S=120")):
        ax[0].plot(sg, gk.delta(S0, K, T, r, sg), lw=1.6, label=lab)
        ax[1].plot(sg, gk.gamma(S0, K, T, r, sg), lw=1.6, label=lab)
        ax[2].plot(sg, gk.vega(S0, K, T, r, sg), lw=1.6, label=lab)
    for a, t, yl in ((ax[0], "Delta", r"$\Delta$"), (ax[1], "Gamma", r"$\Gamma$"),
                     (ax[2], "Vega", r"$\nu$")):
        a.set(xlabel=r"$\sigma$", ylabel=yl, title=f"{t} vs volatility")
        a.legend(frameon=False, fontsize=8)
    finish(fig, "greeks_vs_volatility")

    # ---- vs maturity ----
    Tg = np.linspace(0.01, 3.0, 500)
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for S0, lab in ((80, "OTM S=80"), (100, "ATM S=100"), (120, "ITM S=120")):
        ax[0].plot(Tg, gk.delta(S0, K, Tg, r, sigma), lw=1.6, label=lab)
        ax[1].plot(Tg, gk.gamma(S0, K, Tg, r, sigma), lw=1.6, label=lab)
        ax[2].plot(Tg, gk.vega(S0, K, Tg, r, sigma), lw=1.6, label=lab)
    for a, t, yl in ((ax[0], "Delta", r"$\Delta$"), (ax[1], "Gamma", r"$\Gamma$"),
                     (ax[2], "Vega", r"$\nu$")):
        a.set(xlabel="T (years)", ylabel=yl, title=f"{t} vs maturity")
        a.legend(frameon=False, fontsize=8)
    finish(fig, "greeks_vs_maturity")

    print("""  Saved four figures to greeks_plots/

  READING THEM
    Delta  is a smoothed step. It sharpens towards a true step as T -> 0, which
           is why hedging a near-expiry at-the-money option is so unstable.
    Gamma  peaks at the money and spikes as T -> 0. Gamma is the cost of discrete
           rebalancing: the Week 2 hedging experiment left residual dispersion
           precisely because gamma was not zero between rebalances.
    Vega   also peaks at the money but FALLS as T -> 0 - a one-week option barely
           cares about volatility. Long-dated options carry the vega risk.
    Theta  is negative almost everywhere (time decay) and most negative at the
           money near expiry, mirroring gamma. Gamma and theta trade off: the
           convexity you are paid for in gamma you pay for in decay.""")


# ============================================ 5-6. IMPLIED VOLATILITY RECOVERY

def section5_iv_recovery(n=3000):
    """Recover a known sigma from synthetic Black-Scholes prices, and examine
    how accuracy depends on vega."""
    S   = rng.uniform(40, 160, n)
    K   = rng.uniform(40, 160, n)
    T   = rng.uniform(0.02, 3.0, n)
    r   = rng.uniform(0.0, 0.10, n)
    q   = np.zeros(n)
    sig = rng.uniform(0.05, 1.00, n)

    price = bs_call(S, K, T, r, sig)
    t0 = time.time()
    rec, info = iv.implied_vol(price, S, K, T, r, q, "call")
    dt = time.time() - t0

    ok = np.isfinite(rec)
    err = np.abs(rec - sig)
    v = np.asarray(gk.vega(S, K, T, r, sig))
    meth = info["method"]

    print(f"  {n:,} synthetic European calls, sigma drawn from U(0.05, 1.00)\n")
    print(f"    recovered                {int(ok.sum()):>8,} / {n:,}")
    print(f"    rejected (no finite IV)  {int((~ok).sum()):>8,}")
    print(f"    solved by Newton         {int((meth[ok]=='newton').sum()):>8,}"
          f"  ({(meth[ok]=='newton').mean()*100:.1f}%)")
    print(f"    solved by bisection      {int((meth[ok]=='bisection').sum()):>8,}"
          f"  ({(meth[ok]=='bisection').mean()*100:.1f}%)")
    print(f"    mean iterations          {info['iters'][ok].mean():>8.1f}")
    print(f"    wall time                {dt:>8.1f} s  ({1000*dt/n:.2f} ms per option)")
    print(f"    max  |sigma_IV - sigma|  {err[ok].max():>8.2e}")
    print(f"    mean |sigma_IV - sigma|  {err[ok].mean():>8.2e}")

    rows = []
    print("\n  ACCURACY BY VEGA  (work package 5: 'examine low-vega cases')\n")
    print(f"    {'vega band':>16}{'n':>7}{'max |err|':>12}{'mean |err|':>12}"
          f"{'bisection %':>14}")
    for lo, hi in [(1e2, np.inf), (1e0, 1e2), (1e-2, 1e0),
                   (1e-4, 1e-2), (1e-8, 1e-4), (0.0, 1e-8)]:
        mm = ok & (v >= lo) & (v < hi)
        if not mm.any():
            continue
        lbl = f"> {lo:.0e}" if not np.isfinite(hi) else f"{lo:.0e} - {hi:.0e}"
        bis = (meth[mm] == "bisection").mean() * 100
        print(f"    {lbl:>16}{int(mm.sum()):>7}{err[mm].max():>12.2e}"
              f"{err[mm].mean():>12.2e}{bis:>14.1f}")
        rows.append({"vega band": lbl, "n": int(mm.sum()), "max err": err[mm].max(),
                     "mean err": err[mm].mean(), "bisection pct": bis})

    well = ok & (v > 1e-2)
    print(f"\n    Where vega > 1e-2 ({int(well.sum()):,} options): "
          f"max error {err[well].max():.2e}")
    print("""
  INTERPRETATION
    Implied volatility is only as well determined as vega allows. Since

        delta_sigma  ~  residual / vega

    an option whose price barely responds to sigma cannot pin sigma down, no
    matter how good the solver is. The error above degrades smoothly with vega
    and is NOT a convergence failure - the solver reports a per-option
    sigma_uncertainty so this is visible rather than hidden.

    This is why market implied-volatility surfaces are built from near-the-money
    options and why deep-wing quotes are treated with suspicion. It will matter
    directly in October, when the ML stage uses implied volatility as a feature:
    deep-wing IVs carry far less information than their decimal places suggest.""")

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))
    ax[0].plot(sig[ok], rec[ok], ".", ms=2, alpha=0.4, color="#1f4e79")
    ax[0].plot([0.05, 1.0], [0.05, 1.0], "r--", lw=1.4, label="45 degrees")
    ax[0].set(xlabel=r"true $\sigma$", ylabel=r"recovered $\sigma_{IV}$",
              title="Recovery of a known volatility")
    ax[0].legend(frameon=False, fontsize=8)

    pos = ok & (err > 0) & (v > 0)
    ax[1].loglog(v[pos], err[pos], ".", ms=2, alpha=0.4, color="#1f4e79")
    ax[1].set(xlabel="vega", ylabel=r"$|\sigma_{IV}-\sigma|$",
              title="Error is governed by vega")

    bis = ok & (meth == "bisection")
    ax[2].semilogy(np.abs(moneyness(S[ok], K[ok])), np.maximum(v[ok], 1e-20),
                   ".", ms=2, alpha=0.3, color="#9aa7bd", label="Newton")
    if bis.any():
        ax[2].semilogy(np.abs(moneyness(S[bis], K[bis])), np.maximum(v[bis], 1e-20),
                       "o", ms=4, color="#c0504d", label="bisection fallback")
    ax[2].set(xlabel=r"$|\ln(S/K)|$", ylabel="vega",
              title="Where the fallback is needed")
    ax[2].legend(frameon=False, fontsize=8)
    finish(fig, "iv_recovery")

    return pd.DataFrame(rows), pd.DataFrame({
        "true_sigma": sig, "recovered": rec, "vega": v, "abs_err": err,
        "method": meth.astype(str), "S": S, "K": K, "T": T, "r": r})


# ============================================================================ main

def run():
    t0 = time.time()
    print("=" * 78)
    print("POST-MIDSEM WEEK 3 - GREEKS AND IMPLIED VOLATILITY")
    print("=" * 78)
    if SHOW_PLOTS:
        print("\nNOTE: 6 figure windows will open; the script pauses until each is closed.")
        print("Set SHOW_PLOTS = False to skip this.")

    rule("MODULE SELF-TESTS")
    print("greeks.py")
    g_ok = gk._self_test()
    print("\nimplied_volatility.py")
    iv_ok = iv._self_test()

    rule("2. GREEKS VS CENTRAL FINITE DIFFERENCES")
    fd_tab = section2_fd_check()

    rule("3. ERROR AGAINST STEP SIZE")
    step_tab, best_h = section3_step_size()

    rule("4. SENSITIVITY PLOTS  ->  greeks_plots/")
    section4_plots()

    rule("5-6. IMPLIED-VOLATILITY RECOVERY AND LOW-VEGA CASES")
    vega_tab, iv_detail = section5_iv_recovery()

    fd_tab.to_csv("results/w04_fd_validation.csv", index=False)
    step_tab.to_csv("results/w04_step_size_study.csv", index=False)
    vega_tab.to_csv("results/w04_iv_accuracy_by_vega.csv", index=False)
    iv_detail.to_csv("results/w04_iv_recovery_detail.csv", index=False)

    rule("SUMMARY")
    ok_mask = np.isfinite(iv_detail["recovered"])
    well = ok_mask & (iv_detail["vega"] > 1e-2)
    summary = {
        "seed": SEED,
        "greeks_self_test": g_ok,
        "iv_self_test": iv_ok,
        "fd_max_abs_error": float(fd_tab["abs error"].max()),
        "fd_max_rel_error": float(fd_tab["rel error"].max()),
        "optimal_h_delta": float(best_h["delta"]),
        "optimal_h_gamma": float(best_h["gamma"]),
        "optimal_h_vega": float(best_h["vega"]),
        "pde_residual_analytic": float(np.abs(
            gk.pde_residual(100., 100., 1.0, 0.05, 0.25))),
        "iv_recovered": int(ok_mask.sum()),
        "iv_rejected": int((~ok_mask).sum()),
        "iv_max_err_all": float(iv_detail.loc[ok_mask, "abs_err"].max()),
        "iv_max_err_vega_gt_1e-2": float(iv_detail.loc[well, "abs_err"].max()),
        "iv_bisection_pct": float((iv_detail.loc[ok_mask, "method"] == "bisection").mean() * 100),
    }
    pd.Series(summary).to_csv("results/w04_summary.csv", header=False)
    for k, v in summary.items():
        print(f"  {k:<28} {v}")
    print(f"\n  runtime {time.time()-t0:.1f} s")
    print("\nSaved: results/w04_*.csv   Figures: greeks_plots/*.png")


def main():
    for d in (PLOTDIR, "results"):
        os.makedirs(d, exist_ok=True)
    log = "results/w04_console_output.txt"
    tee = _Tee(log); sys.stdout = tee
    try:
        run(); print(f"\nFull console output written to: {log}")
    finally:
        sys.stdout = tee.stdout; tee.close()
    print(f"\n>>> Share this one file for review: {log}")


if __name__ == "__main__":
    main()
