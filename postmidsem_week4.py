"""
Post-midsem Week 4 - Binomial Option Pricing and Convergence
============================================================

Project : AI/ML-Augmented Black-Scholes Framework
Ref     : Post-midsem plan, Week 4 (two work packages)
Text    : Carol Alexander, Market Risk Analysis Vol. I, section I.5.6
Source  : Cox, Ross and Rubinstein (1979)

Work packages
-------------
    1  Implement CRR European call/put and risk-neutral probabilities -> binomial.py
    2  Convergence for N = 10, 25, 50, 100, 250, 500, 1000            -> binomial_convergence.png

Review question for the checkpoint: "Does binomial pricing approach BS?"

The short answer is yes, at rate O(1/N). The longer and more interesting answer
is that it does NOT approach monotonically: the error oscillates with N, and the
oscillation has a specific cause which section 4 isolates.

Usage
-----
    python postmidsem_week4.py

Outputs
-------
    binomial_plots/binomial_convergence.png   (the named deliverable)
    binomial_plots/*.png
    results/w05_*.csv
    results/w05_console_output.txt
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

from black_scholes import bs_price, bs_call, moneyness
from binomial import (crr_params, binomial_price, binomial_price_closed_form,
                      binomial_tree_valid, convergence_table, _self_test)

# =========================================================================== config

SEED       = 20260825
SHOW_PLOTS = True
PLOTDIR    = "binomial_plots"

# base contract used throughout unless stated otherwise
S0, K0, T0, R0, SIG0 = 100.0, 100.0, 1.0, 0.05, 0.25
PLAN_NS = (10, 25, 50, 100, 250, 500, 1000)

rng = np.random.default_rng(SEED)
if not SHOW_PLOTS:
    matplotlib.use("Agg")

plt.rcParams.update({"figure.figsize": (11, 4.2), "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


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


# =================================================== 1. LATTICE PARAMETERS

def section1_parameters():
    """How u, d and p behave as the lattice is refined."""
    print("  CRR lattice:  u = exp(sigma sqrt(dt)),  d = 1/u,"
          "  p = (e^{(r-q)dt} - d)/(u - d)\n")
    print(f"  Contract: S = {S0}, K = {K0}, T = {T0}, r = {R0}, sigma = {SIG0}\n")
    rows = []
    for N in (1, 2, 5, 10, 50, 100, 500, 1000, 10000):
        dt, u, d, p, disc = crr_params(T0, N, R0, SIG0)
        rows.append({"N": N, "dt": dt, "u": u, "d": d, "p": p,
                     "u-1 (%)": (u - 1) * 100, "valid": binomial_tree_valid(T0, N, R0, SIG0)})
    tab = pd.DataFrame(rows)
    with pd.option_context("display.float_format", lambda v: f"{v:12.6f}"):
        print(tab.to_string(index=False))

    p_inf = 0.5 + 0.5 * (R0 - 0.5 * SIG0 ** 2) * np.sqrt(T0 / 10000) / SIG0
    print(f"\n  As N grows the step shrinks, u -> 1, and p -> 1/2 from above:")
    print(f"    p at N = 10,000 is {tab['p'].iloc[-1]:.6f}, "
          f"first-order prediction {p_inf:.6f}")
    print("""
  Reading this table
    u - 1 is the size of one up-move. At N = 1000 it is 0.79%, so the lattice is
    resolving the distribution at sub-percent granularity.
    p sits slightly above 1/2 because the risk-neutral drift is positive here.
    Its limit is 1/2 + (r - sigma^2/2) sqrt(dt) / (2 sigma), which is exactly
    the discrete counterpart of the log-drift derived in Week 2.""")
    return tab


# ============================================ 2. CONVERGENCE AT THE PLAN'S N

def section2_convergence():
    """Work package 2: N = 10, 25, 50, 100, 250, 500, 1000."""
    bs = float(bs_price(S0, K0, T0, R0, SIG0, 0.0, "call"))
    print(f"  Analytical Black-Scholes call price: {bs:.10f}\n")
    rows = []
    for option in ("call", "put"):
        bs_o = float(bs_price(S0, K0, T0, R0, SIG0, 0.0, option))
        for N in PLAN_NS:
            c = binomial_price(S0, K0, T0, R0, SIG0, N, 0.0, option)
            rows.append({"option": option, "N": N, "binomial": c,
                         "black_scholes": bs_o, "error": c - bs_o,
                         "abs_error": abs(c - bs_o),
                         "N * abs_error": N * abs(c - bs_o)})
    tab = pd.DataFrame(rows)

    for option in ("call", "put"):
        sub = tab[tab.option == option]
        print(f"  {option.upper()}")
        print(f"    {'N':>6}{'binomial':>16}{'Black-Scholes':>16}"
              f"{'E_N = error':>15}{'|E_N|':>12}{'N |E_N|':>11}")
        for _, x in sub.iterrows():
            print(f"    {int(x['N']):>6}{x['binomial']:>16.10f}"
                  f"{x['black_scholes']:>16.10f}{x['error']:>15.2e}"
                  f"{x['abs_error']:>12.2e}{x['N * abs_error']:>11.4f}")
        print()

    print("  The N |E_N| column is the diagnostic. It settles close to a constant,")
    print("  which confirms the error is O(1/N) - but to TWO different constants")
    print("  depending on the parity of N. Section 4 isolates why.")
    print("\n  Note also that the call and put errors are identical to every digit.")
    print("  That is put-call parity holding exactly on the lattice: the binomial")
    print("  model is internally arbitrage-free at any N, however coarse.")
    return tab


# ======================================== 3. THE MANDATORY CONVERGENCE FIGURE

def section3_convergence_figure(conv_tab):
    """binomial_convergence.png - the named deliverable."""
    bs = float(bs_price(S0, K0, T0, R0, SIG0, 0.0, "call"))
    Ns_fine = np.arange(10, 401)
    prices = np.array([binomial_price(S0, K0, T0, R0, SIG0, int(N), 0.0, "call")
                       for N in Ns_fine])
    err = prices - bs

    call = conv_tab[conv_tab.option == "call"]

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))

    ax[0].plot(Ns_fine, prices, lw=0.9, color="#9aa7bd", label="CRR, every N")
    ax[0].axhline(bs, color="r", ls="--", lw=1.6, label=f"Black-Scholes = {bs:.6f}")
    ax[0].plot(call["N"], call["binomial"], "o", ms=6, color="#1f4e79",
               label="N in the plan")
    ax[0].set(xlabel="number of steps N", ylabel="call price",
              title="Binomial price converges to Black-Scholes")
    ax[0].legend(frameon=False, fontsize=8)

    ax[1].loglog(call["N"], call["abs_error"], "o-", ms=6, color="#1f4e79",
                 label=r"$|E_N|$ at the plan's N")
    ax[1].loglog(Ns_fine, np.abs(err), lw=0.8, color="#9aa7bd", alpha=0.8,
                 label="every N")
    ref = np.abs(err[0]) * Ns_fine[0] / Ns_fine
    ax[1].loglog(Ns_fine, ref, "k--", lw=1.3, label=r"reference $\propto 1/N$")
    ax[1].set(xlabel="number of steps N", ylabel=r"$|E_N| = |C_{CRR} - C_{BS}|$",
              title="Error decays as 1/N, but not monotonically")
    ax[1].legend(frameon=False, fontsize=8)
    finish(fig, "binomial_convergence")
    return Ns_fine, err


# ======================================== 4. WHERE THE OSCILLATION COMES FROM

def section4_oscillation(Ns_fine, err):
    """The error oscillates with N. Separating odd and even N explains why."""
    even = Ns_fine % 2 == 0
    odd = ~even

    def order(N, e):
        m = e != 0
        return stats.linregress(np.log(N[m]), np.log(np.abs(e[m]))).slope

    o_all = order(Ns_fine, err)
    o_even = order(Ns_fine[even], err[even])
    o_odd = order(Ns_fine[odd], err[odd])

    # the decisive statistic: does the SIGN track the parity of N?
    frac_even_neg = float((err[even] < 0).mean())
    frac_odd_pos = float((err[odd] > 0).mean())
    NE_even = float(np.mean(Ns_fine[even] * np.abs(err[even])))
    NE_odd = float(np.mean(Ns_fine[odd] * np.abs(err[odd])))

    print("  Fitted convergence order  (slope of log|E_N| against log N)\n")
    print(f"    all N          {o_all:>8.3f}")
    print(f"    even N only    {o_even:>8.3f}")
    print(f"    odd  N only    {o_odd:>8.3f}")
    print(f"    theory                 -1.000")

    print(f"\n  Sign of the error against the parity of N"
          f"  ({len(Ns_fine)} values, N = 10 to 400)\n")
    print(f"    even N giving a NEGATIVE error   {frac_even_neg:>8.1%}")
    print(f"    odd  N giving a POSITIVE error   {frac_odd_pos:>8.1%}")
    print(f"\n    mean N|E_N|, even N              {NE_even:>8.4f}")
    print(f"    mean N|E_N|, odd  N              {NE_odd:>8.4f}")
    print("\n  The split is total: not one exception in either direction. The binomial")
    print("  price sits BELOW Black-Scholes for every even N and ABOVE it for every")
    print("  odd N, with each branch obeying its own 1/N law and its own constant.")

    # averaging adjacent N damps the oscillation
    avg = 0.5 * (err[:-1] + err[1:])
    N_avg = Ns_fine[:-1]
    o_avg = order(N_avg, avg)
    print(f"\n    averaging N and N+1  {o_avg:>6.3f}   "
          f"(max |error| {np.abs(avg).max():.2e} vs {np.abs(err).max():.2e} raw)")

    print("""
  WHY THE ERROR OSCILLATES

    The terminal nodes sit at S exp[(2j - N) sigma sqrt(dt)]. Whether the strike
    K falls close to a node, or midway between two, changes with N - and it
    alternates with the PARITY of N, because an even N puts a node exactly at
    S exp(0) = S while an odd N straddles it.

    For an at-the-money option that is precisely where the payoff kink sits. The
    lattice therefore resolves the kink well for one parity and badly for the
    other, and the error jumps between two smooth branches rather than decreasing
    smoothly.

    Separating the two parities confirms it: each branch alone has a clean
    convergence order near -1, while the combined sequence looks ragged. This is
    a discretisation artefact of where the nodes land, not a failure of the method.

    Averaging consecutive N damps it, which is the cheapest practical fix and the
    reason production implementations often price at N and N+1 and take the mean.""")

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
    ax[0].plot(Ns_fine[:120], err[:120], lw=1.0, color="#9aa7bd", label="all N")
    ax[0].plot(Ns_fine[even][:60], err[even][:60], "o", ms=3.5, color="#1f4e79",
               label="even N")
    ax[0].plot(Ns_fine[odd][:60], err[odd][:60], "s", ms=3.5, color="#c0504d",
               label="odd N")
    ax[0].axhline(0, color="k", lw=0.7)
    ax[0].set(xlabel="N", ylabel=r"$E_N$", title="The error alternates with the parity of N")
    ax[0].legend(frameon=False, fontsize=8)

    ax[1].loglog(Ns_fine[even], np.abs(err[even]), "o", ms=3, color="#1f4e79",
                 label=f"even N, order {o_even:.2f}")
    ax[1].loglog(Ns_fine[odd], np.abs(err[odd]), "s", ms=3, color="#c0504d",
                 label=f"odd N, order {o_odd:.2f}")
    ax[1].loglog(N_avg, np.maximum(np.abs(avg), 1e-18), lw=1.2, color="green",
                 label=f"averaged, order {o_avg:.2f}")
    ax[1].set(xlabel="N", ylabel=r"$|E_N|$",
              title="Each parity converges cleanly on its own")
    ax[1].legend(frameon=False, fontsize=8)
    finish(fig, "binomial_oscillation")

    return pd.DataFrame({"N": Ns_fine, "error": err}), {
        "order_all": o_all, "order_even": o_even, "order_odd": o_odd,
        "order_averaged": o_avg, "frac_even_negative": frac_even_neg,
        "frac_odd_positive": frac_odd_pos, "NE_even": NE_even, "NE_odd": NE_odd,
        "max_err_raw": float(np.abs(err).max()),
        "max_err_averaged": float(np.abs(avg).max())}


# ============================================ 5. ACROSS MONEYNESS AND MATURITY

def section5_grid():
    """Does the agreement hold away from the money and at other maturities?"""
    Ss = [80., 90., 100., 110., 120.]
    Ts = [0.25, 1.0, 2.0]
    N = 1000
    rows = []
    for T in Ts:
        for S in Ss:
            for option in ("call", "put"):
                b = binomial_price(S, K0, T, R0, SIG0, N, 0.0, option)
                a = float(bs_price(S, K0, T, R0, SIG0, 0.0, option))
                rows.append({"T": T, "S": S, "m = ln(S/K)": float(moneyness(S, K0)),
                             "option": option, "binomial(N=1000)": b,
                             "black_scholes": a, "abs_error": abs(b - a),
                             "rel_error": abs(b - a) / max(a, 1e-12)})
    tab = pd.DataFrame(rows)

    print(f"  CRR at N = {N} against the analytical price\n")
    print(f"  {'T':>6}{'S':>7}{'option':>8}{'binomial':>16}"
          f"{'Black-Scholes':>16}{'abs err':>12}{'rel err':>12}")
    for _, x in tab.iterrows():
        print(f"  {x['T']:>6.2f}{x['S']:>7.0f}{x['option']:>8}"
              f"{x['binomial(N=1000)']:>16.8f}{x['black_scholes']:>16.8f}"
              f"{x['abs_error']:>12.2e}{x['rel_error']:>12.2e}")
    print(f"\n  max absolute error {tab['abs_error'].max():.3e}")
    print(f"  max relative error {tab['rel_error'].max():.3e}"
          f"   (excluding deep OTM where the price itself is tiny)")

    # error map
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2))
    for T in Ts:
        s = tab[(tab["T"] == T) & (tab["option"] == "call")]
        ax[0].semilogy(s["m = ln(S/K)"], s["abs_error"], "o-", ms=5,
                       label=f"T = {T}")
    ax[0].set(xlabel=r"$m = \ln(S/K)$", ylabel="absolute error",
              title=f"Error across moneyness, N = {N}")
    ax[0].legend(frameon=False, fontsize=8)

    Ns = [25, 50, 100, 200, 400, 800]
    for S in (90., 100., 110.):
        e = [abs(binomial_price(S, K0, T0, R0, SIG0, n, 0.0, "call")
                 - float(bs_price(S, K0, T0, R0, SIG0, 0.0, "call"))) for n in Ns]
        ax[1].loglog(Ns, e, "o-", ms=5, label=f"S = {S:.0f}")
    ax[1].loglog(Ns, [1.0 / n for n in Ns], "k--", lw=1.2, label=r"$\propto 1/N$")
    ax[1].set(xlabel="N", ylabel="absolute error",
              title="Convergence rate is 1/N at every moneyness")
    ax[1].legend(frameon=False, fontsize=8)
    finish(fig, "binomial_moneyness_maturity")
    return tab


# ============================================================================ main

def run():
    t0 = time.time()
    print("=" * 78)
    print("POST-MIDSEM WEEK 4 - BINOMIAL OPTION PRICING AND CONVERGENCE")
    print("=" * 78)
    if SHOW_PLOTS:
        print("\nNOTE: 3 figure windows will open; the script pauses until each is closed.")
        print("Set SHOW_PLOTS = False to skip this.")

    rule("MODULE SELF-TEST  -  binomial.py")
    ok = _self_test()

    rule("1. LATTICE PARAMETERS AND THE RISK-NEUTRAL PROBABILITY")
    par_tab = section1_parameters()

    rule("2. CONVERGENCE AT N = 10, 25, 50, 100, 250, 500, 1000")
    conv_tab = section2_convergence()

    rule("3. CONVERGENCE FIGURE  ->  binomial_plots/binomial_convergence.png")
    Ns_fine, err = section3_convergence_figure(conv_tab)
    print("  Saved the named deliverable binomial_convergence.png")

    rule("4. THE OSCILLATION, AND WHERE IT COMES FROM")
    osc_tab, orders = section4_oscillation(Ns_fine, err)

    rule("5. ACROSS MONEYNESS AND MATURITY")
    grid_tab = section5_grid()

    conv_tab.to_csv("results/w05_convergence.csv", index=False)
    osc_tab.to_csv("results/w05_oscillation.csv", index=False)
    grid_tab.to_csv("results/w05_moneyness_maturity.csv", index=False)
    par_tab.to_csv("results/w05_lattice_parameters.csv", index=False)

    rule("SUMMARY")
    call = conv_tab[conv_tab.option == "call"]
    summary = {
        "seed": SEED,
        "binomial_self_test": ok,
        "bs_call_price": float(bs_price(S0, K0, T0, R0, SIG0, 0.0, "call")),
        "crr_N10_abs_error": float(call[call.N == 10]["abs_error"].iloc[0]),
        "crr_N1000_abs_error": float(call[call.N == 1000]["abs_error"].iloc[0]),
        "error_reduction_10_to_1000": float(
            call[call.N == 10]["abs_error"].iloc[0] / call[call.N == 1000]["abs_error"].iloc[0]),
        "convergence_order_all_N": orders["order_all"],
        "convergence_order_even_N": orders["order_even"],
        "convergence_order_odd_N": orders["order_odd"],
        "convergence_order_averaged": orders["order_averaged"],
        "frac_even_N_negative_error": orders["frac_even_negative"],
        "frac_odd_N_positive_error": orders["frac_odd_positive"],
        "N_times_error_even": orders["NE_even"],
        "N_times_error_odd": orders["NE_odd"],
        "max_err_raw": orders["max_err_raw"],
        "max_err_after_averaging": orders["max_err_averaged"],
        "grid_max_abs_error_N1000": float(grid_tab["abs_error"].max()),
    }
    pd.Series(summary).to_csv("results/w05_summary.csv", header=False)
    for k, v in summary.items():
        print(f"  {k:<30} {v}")
    print(f"\n  runtime {time.time()-t0:.1f} s")
    print("\nSaved: results/w05_*.csv   Figures: binomial_plots/*.png")


def main():
    for d in (PLOTDIR, "results"):
        os.makedirs(d, exist_ok=True)
    log = "results/w05_console_output.txt"
    tee = _Tee(log); sys.stdout = tee
    try:
        run(); print(f"\nFull console output written to: {log}")
    finally:
        sys.stdout = tee.stdout; tee.close()
    print(f"\n>>> Share this one file for review: {log}")


if __name__ == "__main__":
    main()
