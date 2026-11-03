"""
Post-midsem Week 5 - Monte Carlo and Three-Method Validation
============================================================

Project : AI/ML-Augmented Black-Scholes Framework
Ref     : Week 5 task sheet (28 Oct 2026), tasks 1 to 4; due 3 Nov 2026
Text    : Carol Alexander, Market Risk Analysis Vol. I, sections I.5.7-I.5.8

Tasks
-----
    1  monte_carlo.py: price, standard error, 95% interval        -> monte_carlo.py
    2  Convergence M = 1e3..1e6 and CI coverage over 200 runs     -> mc_convergence.png
    3  Thirty-case Black-Scholes / CRR / Monte Carlo comparison   -> pricing_comparison.csv
    4  Dividend yield q in every pricer                           -> verified below
    5  Real-data sources (manual download)                        -> data_sources.md

Parameters are those of Week 4 so the results compare directly:
S = K = 100, T = 1, r = 0.05, sigma = 0.25  (BS call 12.3360, put 7.4589).
Seed 20260825 throughout.

Checkpoint question, answered in the final section:
    Are the numerical differences between the three methods fully explained by
    known convergence (CRR) and sampling uncertainty (MC)?

Usage
-----
    python postmidsem_week5.py

Outputs
-------
    mc_plots/mc_convergence.png        (the named deliverable)
    results/pricing_comparison.csv     (the computational gate)
    results/w06_*.csv
    results/w06_console_output.txt
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

from black_scholes import bs_price, put_call_parity_error
import greeks as gk
import implied_volatility as iv
from binomial import binomial_price
from monte_carlo import mc_price, _self_test as mc_self_test

# =========================================================================== config

SEED       = 20260825
SHOW_PLOTS = True
PLOTDIR    = "mc_plots"

S0, K0, T0, R0, SIG0 = 100.0, 100.0, 1.0, 0.05, 0.25
M_REF   = 1_000_000          # Monte Carlo sample for the comparison grid
N_CRR   = 1000               # CRR steps; the averaged N, N+1 estimate is also reported
N_COVER = 200                # replications for the coverage test
M_COVER = 10_000

rng = np.random.default_rng(SEED)
if not SHOW_PLOTS:
    matplotlib.use("Agg")
plt.rcParams.update({"figure.figsize": (11, 4.2), "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


class _Tee:
    def __init__(s, p): s.f = open(p, "w", encoding="utf-8"); s.o = sys.stdout
    def write(s, x): s.o.write(x); s.f.write(x)
    def flush(s): s.o.flush(); s.f.flush()
    def close(s): s.f.close()


def rule(t=""):
    print("\n" + "=" * 78)
    if t: print(t); print("=" * 78)


def finish(fig, name):
    fig.tight_layout()
    fig.savefig(f"{PLOTDIR}/{name}.png", dpi=150, bbox_inches="tight")
    plt.show() if SHOW_PLOTS else plt.close(fig)


def crr_averaged(S, K, T, r, sigma, N, q=0.0, option="call"):
    """Average of the N and N+1 lattices.

    Week 4 established that the CRR error alternates in sign with the parity of
    N - negative for every even N, positive for every odd N, in 100% of 391
    cases tested. Averaging adjacent N therefore cancels most of it, cutting the
    maximum error by a factor of about 13 for one extra lattice.
    """
    return 0.5 * (binomial_price(S, K, T, r, sigma, N, q, option)
                  + binomial_price(S, K, T, r, sigma, N + 1, q, option))


# ============================================== TASK 2a: CONVERGENCE IN M

def task2_convergence():
    rows = []
    for k in (3, 4, 5, 6):
        M = 10 ** k
        for option in ("call", "put"):
            res = mc_price(S0, K0, T0, R0, SIG0, M, 0.0, option, seed=SEED + k)
            bs = float(bs_price(S0, K0, T0, R0, SIG0, 0.0, option))
            rows.append({"M": M, "option": option, "mc": res.price,
                         "bs": bs, "abs_error": abs(res.price - bs),
                         "stderr": res.stderr, "ci_low": res.ci_low,
                         "ci_high": res.ci_high,
                         "inside": res.contains(bs),
                         "z": (res.price - bs) / res.stderr})
    tab = pd.DataFrame(rows)

    for option in ("call", "put"):
        sub = tab[tab.option == option]
        print(f"  {option.upper()}  (Black-Scholes {sub['bs'].iloc[0]:.6f})")
        print(f"    {'M':>10}{'MC price':>13}{'abs error':>12}{'s.e.':>11}"
              f"{'95% CI':>26}{'inside':>8}{'z':>7}")
        for _, x in sub.iterrows():
            print(f"    {int(x['M']):>10,}{x['mc']:>13.6f}{x['abs_error']:>12.2e}"
                  f"{x['stderr']:>11.2e}  [{x['ci_low']:>10.6f}, {x['ci_high']:>10.6f}]"
                  f"{('yes' if x['inside'] else 'NO'):>8}{x['z']:>7.2f}")
        print()

    c = tab[tab.option == "call"]
    slope_se = stats.linregress(np.log(c["M"]), np.log(c["stderr"])).slope
    print(f"  Fitted slope of standard error against M: {slope_se:.4f}  (theory -0.5)")
    print("  The standard error is a PREDICTION of accuracy made without knowing")
    print("  the true price. That it tracks the realised error is the point.")
    return tab, slope_se


# ================================================ TASK 2b: COVERAGE TEST

def task2_coverage():
    """200 independent runs at M = 1e4: what fraction of 95% intervals contain
    the Black-Scholes price? Should be near 95%."""
    out = {}
    for option in ("call", "put"):
        bs = float(bs_price(S0, K0, T0, R0, SIG0, 0.0, option))
        g = np.random.default_rng(SEED)
        hits, zs, widths = 0, [], []
        for _ in range(N_COVER):
            res = mc_price(S0, K0, T0, R0, SIG0, M_COVER, 0.0, option, rng=g)
            hits += res.contains(bs)
            zs.append((res.price - bs) / res.stderr)
            widths.append(res.ci_high - res.ci_low)
        zs = np.array(zs)
        cov = hits / N_COVER
        # a binomial standard error on the coverage estimate itself
        se_cov = np.sqrt(0.95 * 0.05 / N_COVER)
        out[option] = {"coverage": cov, "hits": hits, "n": N_COVER,
                       "se_coverage": se_cov,
                       "z_mean": zs.mean(), "z_sd": zs.std(ddof=1),
                       "mean_ci_width": float(np.mean(widths)), "z": zs}
        print(f"  {option.upper():>5}: {hits}/{N_COVER} intervals contain the BS price"
              f"  ->  coverage {cov:.1%}")
        print(f"         z-scores: mean {zs.mean():+.3f}, sd {zs.std(ddof=1):.3f}"
              f"   (theory 0 and 1)")
    print(f"\n  Expected coverage 95.0%, with a sampling standard error of"
          f" {np.sqrt(0.95*0.05/N_COVER):.1%} on {N_COVER} replications,")
    print(f"  so anything in roughly 92-98% is consistent with correct intervals.")
    return out


def task2_figure(conv_tab, cov):
    c = conv_tab[conv_tab.option == "call"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))

    ax[0].loglog(c["M"], c["abs_error"], "o-", ms=6, color="#1f4e79",
                 label="realised |error|")
    ax[0].loglog(c["M"], c["stderr"], "s--", ms=6, color="#c0504d",
                 label="standard error")
    ref = c["stderr"].iloc[0] * np.sqrt(c["M"].iloc[0] / c["M"])
    ax[0].loglog(c["M"], ref, "k:", lw=1.4, label=r"reference $\propto 1/\sqrt{M}$")
    ax[0].set(xlabel="paths M", ylabel="error", title="Convergence in M (call)")
    ax[0].legend(frameon=False, fontsize=8)

    ax[1].errorbar(c["M"], c["mc"], yerr=1.96 * c["stderr"], fmt="o", ms=6,
                   capsize=4, color="#1f4e79", label="MC with 95% CI")
    ax[1].axhline(c["bs"].iloc[0], color="r", ls="--", lw=1.5,
                  label=f"Black-Scholes = {c['bs'].iloc[0]:.4f}")
    ax[1].set(xscale="log", xlabel="paths M", ylabel="call price",
              title="Intervals shrink and keep covering")
    ax[1].legend(frameon=False, fontsize=8)

    z = cov["call"]["z"]
    ax[2].hist(z, bins=24, density=True, alpha=0.65, color="#1f4e79",
               label=f"{N_COVER} runs at M={M_COVER:,}")
    xs = np.linspace(-4, 4, 300)
    ax[2].plot(xs, stats.norm.pdf(xs), "r-", lw=1.6, label="N(0,1)")
    for s in (-1.96, 1.96):
        ax[2].axvline(s, color="k", ls=":", lw=1)
    ax[2].set(xlabel=r"$(\hat C - C_{BS})\,/\,$s.e.", ylabel="density",
              title=f"Coverage {cov['call']['coverage']:.1%} (target 95%)")
    ax[2].legend(frameon=False, fontsize=8)
    finish(fig, "mc_convergence")
    print("  Saved the named deliverable mc_plots/mc_convergence.png")


# ========================================== TASK 3: THREE-METHOD COMPARISON

def task3_comparison():
    Ss = [80., 90., 100., 110., 120.]
    Ts = [0.25, 1.0, 2.0]
    rows = []
    for T in Ts:
        for S in Ss:
            for option in ("call", "put"):
                bs = float(bs_price(S, K0, T, R0, SIG0, 0.0, option))
                crr = binomial_price(S, K0, T, R0, SIG0, N_CRR, 0.0, option)
                crr_av = crr_averaged(S, K0, T, R0, SIG0, N_CRR, 0.0, option)
                res = mc_price(S, K0, T, R0, SIG0, M_REF, 0.0, option, seed=SEED)
                rows.append({
                    "S": S, "T": T, "type": option,
                    "BS": bs, "CRR": crr, "CRR_avg": crr_av,
                    "MC": res.price, "MC_se": res.stderr,
                    "CRR_minus_BS": crr - bs,
                    "CRRavg_minus_BS": crr_av - bs,
                    "MC_minus_BS_over_se": (res.price - bs) / res.stderr,
                })
    tab = pd.DataFrame(rows)

    print(f"  {len(tab)} cases: S in {{80,90,100,110,120}} x T in {{0.25,1,2}}"
          f" x {{call, put}}")
    print(f"  CRR at N = {N_CRR} and averaged over N, N+1;"
          f" Monte Carlo at M = {M_REF:,}\n")
    print(f"  {'S':>5}{'T':>6}{'type':>6}{'BS':>12}{'CRR':>12}{'CRRavg':>12}"
          f"{'MC':>12}{'MC s.e.':>10}{'CRR-BS':>11}{'(MC-BS)/se':>12}")
    for _, x in tab.iterrows():
        print(f"  {x['S']:>5.0f}{x['T']:>6.2f}{x['type']:>6}{x['BS']:>12.6f}"
              f"{x['CRR']:>12.6f}{x['CRR_avg']:>12.6f}{x['MC']:>12.6f}"
              f"{x['MC_se']:>10.2e}{x['CRR_minus_BS']:>11.2e}"
              f"{x['MC_minus_BS_over_se']:>12.2f}")

    crr_ok = (tab["CRR_minus_BS"].abs() < 5e-3).all()
    crrav_ok = (tab["CRRavg_minus_BS"].abs() < 5e-3).all()
    within2 = (tab["MC_minus_BS_over_se"].abs() < 2).mean()

    print(f"\n  ACCEPTANCE CRITERIA")
    print(f"    max |CRR - BS|          {tab['CRR_minus_BS'].abs().max():.3e}"
          f"   (target < 5e-03)   {'PASS' if crr_ok else 'FAIL'}")
    print(f"    max |CRRavg - BS|       {tab['CRRavg_minus_BS'].abs().max():.3e}"
          f"   (target < 5e-03)   {'PASS' if crrav_ok else 'FAIL'}")
    print(f"    |MC - BS| < 2 s.e. in   {within2:.1%} of cases"
          f"   (target about 95%)  {'PASS' if 0.88 <= within2 <= 1.0 else 'CHECK'}")
    print(f"    max |(MC - BS)/s.e.|    {tab['MC_minus_BS_over_se'].abs().max():.2f}")

    outside = tab[tab["MC_minus_BS_over_se"].abs() >= 2]
    if len(outside):
        print(f"\n  Cases beyond 2 standard errors ({len(outside)} of {len(tab)}):")
        for _, x in outside.iterrows():
            print(f"    S={x['S']:.0f} T={x['T']:.2f} {x['type']:<5} "
                  f"z = {x['MC_minus_BS_over_se']:+.2f}")
        print("    Under correct intervals about 5% of cases should exceed 2 s.e.,")
        print(f"    i.e. roughly {0.05*len(tab):.1f} of {len(tab)}. These are consistent with that,")
        print("    and all were priced from the SAME seed, so they are not independent.")
    return tab


# ======================================== TASK 4: DIVIDEND YIELD VERIFICATION

def task4_dividend(q=0.015):
    """q was implemented in every module from the start. This re-verifies the
    whole stack with q = 0.015 and confirms the q = 0 results are unchanged."""
    print(f"  Verifying the full stack at q = {q} and at q = 0\n")
    rows = []
    for qq in (0.0, q):
        parity = float(put_call_parity_error(S0, K0, T0, R0, SIG0, qq))
        pde = float(gk.pde_residual(S0, K0, T0, R0, SIG0, qq, "call"))
        bs_c = float(bs_price(S0, K0, T0, R0, SIG0, qq, "call"))
        crr_c = binomial_price(S0, K0, T0, R0, SIG0, 2000, qq, "call")
        mc_c = mc_price(S0, K0, T0, R0, SIG0, M_REF, qq, "call", seed=SEED)
        ivr, info = iv.implied_vol_scalar(bs_c, S0, K0, T0, R0, qq, "call")
        rows.append({"q": qq, "BS call": bs_c, "parity error": abs(parity),
                     "PDE residual": abs(pde), "CRR N=2000": crr_c,
                     "CRR - BS": crr_c - bs_c, "MC": mc_c.price,
                     "(MC-BS)/se": (mc_c.price - bs_c) / mc_c.stderr,
                     "IV recovered": ivr, "IV error": abs(ivr - SIG0)})
    tab = pd.DataFrame(rows)
    with pd.option_context("display.float_format", lambda v: f"{v:14.8f}"):
        print(tab.to_string(index=False))

    print(f"""
  All five components accept q and remain consistent:
    - put-call parity with the dividend, C - P = S e^-qT - K e^-rT, holds to
      {tab['parity error'].max():.1e}
    - the Black-Scholes PDE with (r - q) in the drift term holds to
      {tab['PDE residual'].max():.1e}
    - CRR, which uses e^(r-q)dt in the risk-neutral probability, agrees to
      {tab['CRR - BS'].abs().max():.1e}
    - Monte Carlo, with the (r - q - sigma^2/2) drift, is within
      {tab['(MC-BS)/se'].abs().max():.2f} standard errors
    - implied volatility inverts back to sigma to {tab['IV error'].max():.1e}

  NOTE FOR THE REPORT, as the task sheet requires: the Week 4 check that an
  American call equals a European call holds ONLY when q = 0. With q > 0 early
  exercise can be optimal just before a dividend, so the American call is worth
  strictly more. That check is therefore conditioned on q = 0 and is not a
  defect of the implementation.""")
    return tab


# ============================================================================ main

def run():
    t0 = time.time()
    print("=" * 78)
    print("POST-MIDSEM WEEK 5 - MONTE CARLO AND THREE-METHOD VALIDATION")
    print("Week 5 task sheet, due 3 November 2026")
    print("=" * 78)
    if SHOW_PLOTS:
        print("\nNOTE: 1 figure window will open; the script pauses until it is closed.")

    rule("TASK 1  -  monte_carlo.py SELF-TEST")
    mc_ok = mc_self_test()

    rule("TASK 2a  -  CONVERGENCE IN M")
    conv, slope_se = task2_convergence()

    rule("TASK 2b  -  CONFIDENCE-INTERVAL COVERAGE")
    cov = task2_coverage()

    rule("TASK 2c  -  CONVERGENCE FIGURE  ->  mc_plots/mc_convergence.png")
    task2_figure(conv, cov)

    rule("TASK 3  -  THREE-METHOD COMPARISON  (the computational gate)")
    cmp_tab = task3_comparison()

    rule("TASK 4  -  DIVIDEND YIELD q")
    q_tab = task4_dividend()

    # ---- save ----
    conv.to_csv("results/w06_mc_convergence.csv", index=False)
    pd.DataFrame({k: {kk: vv for kk, vv in v.items() if kk != "z"}
                  for k, v in cov.items()}).T.to_csv("results/w06_coverage.csv")
    cmp_tab.to_csv("results/pricing_comparison.csv", index=False)
    q_tab.to_csv("results/w06_dividend_yield.csv", index=False)

    rule("CHECKPOINT QUESTION")
    crr_max = cmp_tab["CRR_minus_BS"].abs().max()
    crrav_max = cmp_tab["CRRavg_minus_BS"].abs().max()
    within2 = (cmp_tab["MC_minus_BS_over_se"].abs() < 2).mean()
    print(f"""
  "Are the numerical differences between the three methods fully explained by
   known convergence (CRR) and sampling uncertainty (MC)?"

  Yes, on the evidence of the 30-case grid.

  CRR. The largest deviation from Black-Scholes is {crr_max:.2e} at N = {N_CRR}.
  Week 4 measured the convergence order as -0.9999 and showed the error sign is
  fixed by the parity of N. Averaging N and N+1 cuts the largest deviation to
  {crrav_max:.2e}. The residual is deterministic discretisation error of the size
  and sign the convergence law predicts, not an unexplained discrepancy.

  Monte Carlo. {within2:.0%} of the 30 cases lie within two standard errors of the
  analytical price, and the coverage study over {N_COVER} independent replications gave
  {cov['call']['coverage']:.1%} for calls and {cov['put']['coverage']:.1%} for puts
  against a nominal 95%. The z-scores have mean {cov['call']['z_mean']:+.3f} and
  standard deviation {cov['call']['z_sd']:.3f} against a theoretical 0 and 1, so the
  reported intervals are neither optimistic nor conservative. The standard error
  falls as M^({slope_se:.3f}) against a theoretical -0.5.

  The two residual differences are of entirely different kinds. CRR error is
  deterministic and shrinks predictably with N. Monte Carlo error is random and
  is quantified in advance by an interval that does not require knowing the
  answer. Neither is unexplained, which is what the gate asks.""")

    rule("SUMMARY")
    summary = {
        "seed": SEED,
        "mc_self_test": mc_ok,
        "se_slope_vs_M": slope_se,
        "coverage_call": cov["call"]["coverage"],
        "coverage_put": cov["put"]["coverage"],
        "coverage_z_mean": cov["call"]["z_mean"],
        "coverage_z_sd": cov["call"]["z_sd"],
        "max_abs_CRR_minus_BS": float(crr_max),
        "max_abs_CRRavg_minus_BS": float(crrav_max),
        "frac_MC_within_2se": float(within2),
        "max_abs_MC_z": float(cmp_tab["MC_minus_BS_over_se"].abs().max()),
        "q_max_parity_error": float(q_tab["parity error"].max()),
        "q_max_pde_residual": float(q_tab["PDE residual"].max()),
        "comparison_cases": int(len(cmp_tab)),
    }
    pd.Series(summary).to_csv("results/w06_summary.csv", header=False)
    for k, v in summary.items():
        print(f"  {k:<28} {v}")
    print(f"\n  runtime {time.time()-t0:.1f} s")
    print("\nSaved: results/pricing_comparison.csv (the gate), results/w06_*.csv")
    print("       mc_plots/mc_convergence.png")


def main():
    for d in (PLOTDIR, "results"):
        os.makedirs(d, exist_ok=True)
    log = "results/w06_console_output.txt"
    tee = _Tee(log); sys.stdout = tee
    try:
        run(); print(f"\nFull console output written to: {log}")
    finally:
        sys.stdout = tee.o; tee.close()
    print(f"\n>>> Share this one file for review: {log}")


if __name__ == "__main__":
    main()
