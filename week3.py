"""
September Week 2 - Ito's Lemma and the Black-Scholes Equation
============================================================

Project : AI/ML-Augmented Black-Scholes Framework
Ref     : September Week 2 plan, section 16 (tasks 1-13)
Text    : Carol Alexander, Market Risk Analysis Vol. I, sections I.3.7 and I.5

Progression
-----------
    Quadratic Variation -> Ito's Lemma -> Derivative Dynamics -> Delta Hedging
    -> Elimination of dW -> No-Arbitrage -> Black-Scholes PDE -> Call/Put Pricing

Tasks
-----
     1  From ordinary Taylor expansion to Ito's lemma           16.5
     2  Dynamics of the derivative                              16.6
     3  Numerical illustration  d(W^2) = 2W dW + dt             16.7   [figure 1]
     4  Second Ito experiment   f(S) = ln S -> GBM solution     16.8
     5  Construct the delta-hedged portfolio                    16.9
     6  Choose the hedge ratio, show dW cancels                 16.10
        The central question: where did mu go?                  16.11
     7  Apply the no-arbitrage argument                         16.12
     8  Introduce the terminal payoff                           16.14
     9  Implement the European call formula                     16.15
    10  Implement the European put formula                      16.16
    11  Put-call parity                                         16.17
    12  Understand the Black-Scholes inputs                     16.18  [figures 2-5]
    13  Connect the derivation back to Week 1                   16.19

The derivation is the principal task of the week; the code exists to verify it,
not to replace it. Every numerical section states what it is testing and what
result would falsify it.

Usage
-----
    python week3.py

Requires black_scholes.py in the same folder.

Outputs
-------
    figures/w03_*.png        7 figures (5 mandatory + 2 supporting)
    results/w03_*.csv        result tables
    results/w03_console_output.txt
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

from black_scholes import (bs_call, bs_put, d1_d2, put_call_parity_error,
                           moneyness, intrinsic_value, _self_test)

# =========================================================================== config

SEED       = 20260825          # unchanged since Week 1
SHOW_PLOTS = True              # False -> save figures only, no windows

rng = np.random.default_rng(SEED)
if not SHOW_PLOTS:
    matplotlib.use("Agg")

plt.rcParams.update({
    "figure.figsize": (11, 4.2), "axes.grid": True, "grid.alpha": 0.3,
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
})


class _Tee:
    """Mirror console output into a log file so the run can be shared as one file."""
    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8"); self.stdout = sys.stdout
    def write(self, s):
        self.stdout.write(s); self.file.write(s)
    def flush(self):
        self.stdout.flush(); self.file.flush()
    def close(self):
        self.file.close()


def rule(title=""):
    print("\n" + "=" * 78)
    if title:
        print(title); print("=" * 78)


def finish(fig, name):
    fig.tight_layout()
    fig.savefig(f"figures/{name}.png", dpi=150, bbox_inches="tight")
    plt.show() if SHOW_PLOTS else plt.close(fig)


# ============================================================ TASKS 1-2: THE ALGEBRA

DERIVATION_1_2 = r"""
  TASK 1  -  FROM ORDINARY TAYLOR EXPANSION TO ITO'S LEMMA          (16.5)

  Start from a second-order Taylor expansion of V = V(S,t):

      dV = V_t dt + V_S dS + (1/2) V_SS (dS)^2 + V_St dS dt + (1/2) V_tt (dt)^2 + ...

  In ORDINARY calculus every term beyond first order is discarded, because for a
  differentiable path (dS)^2 is of order (dt)^2 and therefore negligible.

  For Brownian motion that reasoning fails. With

      dS = mu S dt + sigma S dW

  squaring gives

      (dS)^2 = mu^2 S^2 (dt)^2 + 2 mu sigma S^2 dt dW + sigma^2 S^2 (dW)^2

  and the three stochastic multiplication rules are

      (dt)^2  = 0          dt dW = 0          (dW)^2 = dt

  The first two vanish faster than dt. The third does NOT: this is the quadratic
  variation measured numerically in Week 1, where sum (Delta W)^2 -> T while
  sum |Delta W| -> infinity. Hence

      (dS)^2 = sigma^2 S^2 dt                                        [order dt]

  The second-order term is therefore of the SAME order as dt and cannot be
  dropped. That single fact is the whole of Ito calculus.


  TASK 2  -  DYNAMICS OF THE DERIVATIVE                             (16.6)

  Substituting (dS)^2 = sigma^2 S^2 dt into the Taylor expansion:

      dV = V_t dt + V_S dS + (1/2) sigma^2 S^2 V_SS dt

  Now substitute dS = mu S dt + sigma S dW and collect terms:

      dV = V_t dt + V_S (mu S dt + sigma S dW) + (1/2) sigma^2 S^2 V_SS dt

         = [ V_t + mu S V_S + (1/2) sigma^2 S^2 V_SS ] dt  +  sigma S V_S dW
           \_________________  ________________________/     \______  _____/
                             \/                                     \/
                       deterministic part                    stochastic part

  Two things to note before going further:

    - mu appears in the drift of dV. It has NOT yet disappeared.
    - the stochastic part carries the factor V_S. This is what makes the hedge
      in Task 6 possible: both dV and dS are driven by the SAME dW.
"""


def task1_ds_squared(S0=100.0, mu=0.10, sigma=0.30, T=1.0):
    """Verify (dS)^2 = sigma^2 S^2 dt numerically: realised quadratic variation
    of S against sigma^2 * integral of S^2 dt."""
    print("  Numerical check of  (dS)^2 = sigma^2 S^2 dt")
    print("  Comparing realised sum (Delta S)^2 against sigma^2 * sum S^2 Delta t\n")
    rows = []
    for n in [100, 1_000, 10_000, 100_000, 1_000_000]:
        dt = T / n
        dW = np.sqrt(dt) * rng.standard_normal(n)
        logS = np.cumsum((mu - 0.5 * sigma ** 2) * dt + sigma * dW)
        S = S0 * np.exp(np.concatenate([[0.0], logS]))
        realised = (np.diff(S) ** 2).sum()
        predicted = (sigma ** 2 * S[:-1] ** 2 * dt).sum()
        rows.append({"n steps": n, "sum (dS)^2": realised,
                     "sigma^2 sum S^2 dt": predicted,
                     "ratio": realised / predicted})
    tab = pd.DataFrame(rows)
    with pd.option_context("display.float_format", lambda v: f"{v:16.6f}"):
        print(tab.to_string(index=False))
    print(f"\n  ratio -> 1 as n grows: {tab['ratio'].iloc[-1]:.6f} at n = 1,000,000")
    print("  This is the statement (dS)^2 = sigma^2 S^2 dt, verified on a path.")
    return tab


# ==================================== TASK 3: NUMERICAL ILLUSTRATION OF ITO'S LEMMA

def task3_d_W_squared(T=1.0):
    """Section 16.7. For f(W) = W^2, Ito gives d(W^2) = 2W dW + dt, whereas
    ordinary calculus would give only 2W dW.

    The discrete identity is exact:
        W_T^2 = sum (W_{i+1}^2 - W_i^2) = sum 2 W_i dW_i + sum (dW_i)^2
    so the error committed by ordinary calculus is exactly sum (dW)^2, which
    converges to T. This is the Week 1 quadratic-variation result reappearing
    as the extra term in Ito's lemma.
    """
    print("  Ito:               d(W^2) = 2W dW + dt")
    print("  Ordinary calculus: d(W^2) = 2W dW          <- incorrect\n")
    rows = []
    for n in [10, 100, 1_000, 10_000, 100_000, 1_000_000]:
        dt = T / n
        dW = np.sqrt(dt) * rng.standard_normal(n)
        W = np.concatenate([[0.0], np.cumsum(dW)])
        lhs = W[-1] ** 2                        # W_T^2 - W_0^2
        ordinary = (2 * W[:-1] * dW).sum()      # Ito integral, left endpoint
        ito = ordinary + (dW ** 2).sum()        # + quadratic variation
        rows.append({"n steps": n,
                     "W_T^2 (truth)": lhs,
                     "2W dW only": ordinary,
                     "2W dW + sum(dW)^2": ito,
                     "error of ordinary": lhs - ordinary,
                     "T": T})
    tab = pd.DataFrame(rows)
    with pd.option_context("display.float_format", lambda v: f"{v:18.8f}"):
        print(tab.to_string(index=False))

    err = tab["error of ordinary"]
    print(f"\n  The Ito column reproduces W_T^2 to machine precision:"
          f" max |diff| = {np.abs(tab['W_T^2 (truth)']-tab['2W dW + sum(dW)^2']).max():.2e}")
    print(f"  The ordinary column is wrong by {err.iloc[-1]:.6f}, i.e. by T = {T}.")
    print("  That missing amount IS the dt term in Ito's lemma.")

    # figure 1 (mandatory)
    n = 20_000
    dt = T / n
    dW = np.sqrt(dt) * rng.standard_normal(n)
    W = np.concatenate([[0.0], np.cumsum(dW)])
    t = np.linspace(0, T, n + 1)
    running_true = W ** 2
    running_ord = np.concatenate([[0.0], np.cumsum(2 * W[:-1] * dW)])
    running_ito = np.concatenate([[0.0], np.cumsum(2 * W[:-1] * dW + dW ** 2)])

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
    ax[0].plot(t, running_true, lw=1.4, color="#1f4e79", label=r"$W_t^2$  (truth)")
    ax[0].plot(t, running_ito, lw=1.2, ls="--", color="green",
               label=r"It$\hat{o}$: $\int 2W\,dW + \int dt$")
    ax[0].plot(t, running_ord, lw=1.2, color="#c0504d",
               label=r"ordinary: $\int 2W\,dW$ only")
    ax[0].set(xlabel="t", ylabel="value", title=r"$d(W^2) = 2W\,dW + dt$")
    ax[0].legend(frameon=False, fontsize=8)

    ax[1].plot(t, running_true - running_ord, lw=1.4, color="#c0504d",
               label=r"$W_t^2 - \int 2W\,dW$")
    ax[1].plot(t, t, "k--", lw=1.5, label=r"$t$  (the missing $\int dt$)")
    ax[1].set(xlabel="t", ylabel="discrepancy",
              title="The error of ordinary calculus is exactly $t$")
    ax[1].legend(frameon=False, fontsize=8)
    finish(fig, "w03_ito_d_W_squared")
    return tab


# ============================================ TASK 4: SECOND ITO EXPERIMENT, ln S

def task4_log_S(S0=100.0, mu=0.12, sigma=0.35, T=1.0):
    """Section 16.8. Ito applied to f(S) = ln S:

        d(ln S) = (1/S) dS - (1/(2S^2)) (dS)^2
                = (mu - sigma^2/2) dt + sigma dW

    Ordinary calculus would give only (1/S) dS = mu dt + sigma dW, missing the
    -sigma^2/2 drift correction. Integrating the correct version gives the GBM
    solution used computationally in Week 1.
    """
    print("  Ito:               d(ln S) = (mu - sigma^2/2) dt + sigma dW")
    print("  Ordinary calculus: d(ln S) = (1/S) dS = mu dt + sigma dW   <- incorrect\n")
    rows = []
    for n in [100, 1_000, 10_000, 100_000, 1_000_000]:
        dt = T / n
        dW = np.sqrt(dt) * rng.standard_normal(n)
        logS = np.concatenate([[np.log(S0)],
                               np.log(S0) + np.cumsum((mu - 0.5*sigma**2)*dt + sigma*dW)])
        S = np.exp(logS)
        truth = logS[-1] - logS[0]
        ordinary = (np.diff(S) / S[:-1]).sum()          # sum dS/S
        correction = -0.5 * (sigma ** 2) * T
        rows.append({"n steps": n, "ln(S_T/S_0) (truth)": truth,
                     "sum dS/S (ordinary)": ordinary,
                     "difference": truth - ordinary,
                     "-sigma^2 T/2": correction})
    tab = pd.DataFrame(rows)
    with pd.option_context("display.float_format", lambda v: f"{v:20.8f}"):
        print(tab.to_string(index=False))
    print(f"\n  difference -> -sigma^2 T/2 = {-0.5*sigma**2*T:.6f}")
    print(f"  observed at n = 1,000,000:   {tab['difference'].iloc[-1]:.6f}")
    print("\n  Integrating d(ln S) = (mu - sigma^2/2)dt + sigma dW gives")
    print("      S_T = S_0 exp[ (mu - sigma^2/2) T + sigma W_T ]")
    print("  which is the GBM solution used computationally in Week 1, now derived")
    print("  rather than assumed.")
    return tab


# ============================== TASKS 5-6: DELTA HEDGE AND ELIMINATION OF dW

DERIVATION_5_7 = r"""
  TASK 5  -  THE DELTA-HEDGED PORTFOLIO                             (16.9)

  Hold one derivative and short Delta units of the underlying:

      Pi = V - Delta S

  Over an infinitesimal interval, with Delta held fixed across the interval,

      dPi = dV - Delta dS


  TASK 6  -  CHOOSE THE HEDGE RATIO                                 (16.10)

  From Task 2,   dV = [V_t + mu S V_S + (1/2) sigma^2 S^2 V_SS] dt + sigma S V_S dW
  and            dS = mu S dt + sigma S dW.

  Therefore

      dPi = [V_t + mu S V_S + (1/2) sigma^2 S^2 V_SS - Delta mu S] dt
            + [sigma S V_S - Delta sigma S] dW

  The dW coefficient vanishes if and only if

      Delta = V_S

  and with that choice the mu S V_S and Delta mu S terms cancel simultaneously:

      dPi = [ V_t + (1/2) sigma^2 S^2 V_SS ] dt                     (no dW)

  The portfolio is now LOCALLY RISKLESS.


  16.11  THE CENTRAL QUESTION:  WHERE DID mu GO?

  mu multiplied S V_S in dV and S Delta in Delta dS. Setting Delta = V_S made
  those two terms identical, so they cancelled. The SAME choice that removes the
  randomness also removes the drift - because both entered through dS.

  Financially: the hedge neutralises all first-order exposure to S. Whatever the
  underlying is expected to do on average is already fully offset by the hedge,
  so the expected return of the stock cannot influence the value of the hedged
  position. Two investors who disagree completely about mu must still agree on
  the option price, provided they agree on sigma.

  What remains is pure convexity, (1/2) sigma^2 S^2 V_SS, which depends on the
  VARIANCE of the underlying, not its mean.


  TASK 7  -  THE NO-ARBITRAGE ARGUMENT                              (16.12)

  A locally riskless portfolio must earn the risk-free rate, else a costless
  arbitrage exists. Hence

      dPi = r Pi dt

  Substituting Pi = V - S V_S and the expression for dPi:

      [ V_t + (1/2) sigma^2 S^2 V_SS ] dt = r ( V - S V_S ) dt

  Rearranging gives the BLACK-SCHOLES PDE:

      V_t + (1/2) sigma^2 S^2 V_SS + r S V_S - r V = 0

  Interpretation of each term (16.13):

      V_t                      time effect (theta)
      (1/2) sigma^2 S^2 V_SS   curvature-volatility effect (gamma)
      r S V_S                  underlying-price sensitivity under risk-neutral growth
      -r V                     discounting / financing effect

  Note that r has replaced mu in the first-order term. The physical drift is gone;
  the risk-free rate takes its place.
"""


def task6_hedge_numerically(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.25,
                            n_paths=20_000, n_steps=252):
    """Demonstrate numerically that (a) delta hedging removes the risk, and
    (b) the hedged result does not depend on mu.

    A short call is hedged discretely with Delta = V_S. If the derivation is
    right, the hedged P&L should have near-zero mean and far smaller dispersion
    than the unhedged position, and should be statistically indistinguishable
    across different values of the physical drift mu.
    """
    dt = T / n_steps
    out = {}
    for mu in (0.00, 0.15, 0.30):
        g = np.random.default_rng(SEED)          # identical shocks for each mu
        dW = np.sqrt(dt) * g.standard_normal((n_paths, n_steps))
        logS = np.cumsum((mu - 0.5*sigma**2)*dt + sigma*dW, axis=1)
        S = S0 * np.exp(np.concatenate([np.zeros((n_paths, 1)), logS], axis=1))

        C0 = bs_call(S0, K, T, r, sigma)
        cash = C0                                # premium received for the short call
        shares = np.zeros(n_paths)
        for i in range(n_steps):
            tau = T - i * dt
            d1 = d1_d2(S[:, i], K, tau, r, sigma)[0]
            delta = stats.norm.cdf(d1)
            cash -= (delta - shares) * S[:, i]   # buy/sell to reach new delta
            shares = delta
            cash *= np.exp(r * dt)               # finance at the risk-free rate
        payoff = np.maximum(S[:, -1] - K, 0.0)
        hedged = cash + shares * S[:, -1] - payoff
        unhedged = C0 * np.exp(r * T) - payoff
        out[mu] = {"hedged": hedged, "unhedged": unhedged, "S_T": S[:, -1], "C0": C0}

    print(f"  Short one ATM call, delta-hedged {n_steps} times over {T} year,"
          f" {n_paths:,} paths")
    print(f"  Black-Scholes premium received: {out[0.0]['C0']:.6f}\n")
    print(f"  {'mu':>6}{'hedged mean':>15}{'hedged sd':>13}"
          f"{'unhedged mean':>16}{'unhedged sd':>14}{'risk removed':>14}")
    rows = []
    for mu, d in out.items():
        h, u = d["hedged"], d["unhedged"]
        print(f"  {mu:>6.2f}{h.mean():>15.6f}{h.std(ddof=1):>13.6f}"
              f"{u.mean():>16.6f}{u.std(ddof=1):>14.6f}"
              f"{1 - h.std(ddof=1)/u.std(ddof=1):>13.1%}")
        rows.append({"mu": mu, "hedged mean": h.mean(), "hedged sd": h.std(ddof=1),
                     "unhedged mean": u.mean(), "unhedged sd": u.std(ddof=1)})
    tab = pd.DataFrame(rows)

    sds = tab["hedged sd"].values
    means = tab["hedged mean"].values
    print(f"\n  Hedged standard deviation across mu = 0.00, 0.15, 0.30:"
          f"  spread = {np.ptp(sds):.6f}")
    print(f"  Hedged mean across the same:                       spread = {np.ptp(means):.6f}")
    print(f"  Unhedged mean across the same:                     spread ="
          f" {np.ptp(tab['unhedged mean'].values):.6f}")
    print()
    print("  READING THIS RESULT")
    print("  The unhedged position's expected P&L moves substantially with mu - as it")
    print("  must, since a higher drift makes the short call more likely to finish in")
    print("  the money. The hedged position's distribution is essentially unchanged.")
    print("  That invariance is the numerical content of 'mu disappears from the PDE'.")
    print()
    print("  The residual hedged dispersion is NOT a flaw in the derivation. It is")
    print("  discrete-rebalancing error: the derivation assumes continuous hedging,")
    print("  and here Delta is updated only 252 times. It shrinks as n_steps grows.")

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
    bins = np.linspace(-12, 12, 120)
    for mu, d in out.items():
        ax[0].hist(d["unhedged"], bins=bins, histtype="step", lw=1.5,
                   density=True, label=fr"$\mu={mu:.2f}$")
    ax[0].set(xlabel="P&L", ylabel="density",
              title="Unhedged short call: distribution depends on $\\mu$")
    ax[0].legend(frameon=False, fontsize=8)
    bins2 = np.linspace(-4, 4, 120)
    for mu, d in out.items():
        ax[1].hist(d["hedged"], bins=bins2, histtype="step", lw=1.5,
                   density=True, label=fr"$\mu={mu:.2f}$")
    ax[1].axvline(0, color="k", lw=0.8)
    ax[1].set(xlabel="P&L", ylabel="density",
              title="Delta-hedged: distribution is invariant to $\\mu$")
    ax[1].legend(frameon=False, fontsize=8)
    finish(fig, "w03_delta_hedge_mu_invariance")
    return tab


# ============================== TASK 7-8: PDE RESIDUAL AND TERMINAL CONDITION

def task7_pde_residual(r=0.05, sigma=0.25, K=100.0):
    """Verify that the analytical Black-Scholes formula satisfies the PDE.

        residual = V_t + (1/2) sigma^2 S^2 V_SS + r S V_S - r V

    The derivatives are taken by central differences on the closed-form price.
    Because the formula is stated in time-to-maturity tau, V_t = -dV/dtau.

    If the residual is not ~0 either the derivation or the implementation is
    wrong. This is the single strongest check available this week.
    """
    Ss = np.array([60., 80., 90., 100., 110., 120., 150.])
    taus = np.array([0.10, 0.25, 0.50, 1.00, 2.00])
    hS, ht = 1e-3, 1e-5
    rows = []
    for tau in taus:
        for S in Ss:
            V   = bs_call(S, K, tau, r, sigma)
            V_S = (bs_call(S+hS, K, tau, r, sigma) - bs_call(S-hS, K, tau, r, sigma))/(2*hS)
            V_SS= (bs_call(S+hS, K, tau, r, sigma) - 2*V
                   + bs_call(S-hS, K, tau, r, sigma))/hS**2
            dVdtau = (bs_call(S, K, tau+ht, r, sigma)
                      - bs_call(S, K, tau-ht, r, sigma))/(2*ht)
            V_t = -dVdtau
            res = V_t + 0.5*sigma**2*S**2*V_SS + r*S*V_S - r*V
            rows.append({"tau": tau, "S": S, "V": V, "V_t": V_t, "V_S": V_S,
                         "V_SS": V_SS, "PDE residual": res})
    tab = pd.DataFrame(rows)
    show = tab[tab["tau"] == 1.00]
    print("  PDE residual at tau = 1.0 (one year to maturity):\n")
    with pd.option_context("display.float_format", lambda v: f"{v:14.8f}"):
        print(show[["S", "V", "V_t", "V_S", "V_SS", "PDE residual"]].to_string(index=False))
    print(f"\n  Across all {len(tab)} (S, tau) points:")
    print(f"    max |residual|      {tab['PDE residual'].abs().max():.3e}")
    print(f"    mean |residual|     {tab['PDE residual'].abs().mean():.3e}")
    print(f"    typical |V|         {tab['V'].abs().mean():.3f}")
    print("\n  The residual is at the level of finite-difference truncation error,")
    print("  not at the level of the prices themselves. The closed-form formula")
    print("  therefore satisfies the PDE that was derived above.")
    print()
    print("  TASK 8 - TERMINAL CONDITION (16.14)")
    print("    The PDE alone has infinitely many solutions. What selects the option")
    print("    price is the terminal payoff:")
    print("        call:  V(S,T) = max(S - K, 0)")
    print("        put:   V(S,T) = max(K - S, 0)")
    print("    Verified numerically below as tau -> 0.")
    taus_small = [1e-2, 1e-4, 1e-6, 0.0]
    Sg = np.array([80., 95., 100., 105., 120.])
    print(f"\n    {'tau':>10}" + "".join(f"{f'S={s:.0f}':>12}" for s in Sg))
    for tau in taus_small:
        vals = bs_call(Sg, K, tau, r, sigma)
        print(f"    {tau:>10.0e}" + "".join(f"{v:>12.6f}" for v in np.atleast_1d(vals)))
    print(f"    {'intrinsic':>10}" + "".join(f"{v:>12.6f}"
                                             for v in intrinsic_value(Sg, K, 'call')))
    return tab


# ================================= TASKS 9-11: PRICING AND PUT-CALL PARITY

def tasks9_11_pricing_and_parity():
    """Implementation lives in black_scholes.py; this reports a price table and
    the parity check over a grid of S, K, T, r, sigma (section 16.17)."""
    print("  Representative prices (K = 100, r = 5%, sigma = 25%)\n")
    K, r, sigma = 100.0, 0.05, 0.25
    rows = []
    for T in (0.25, 1.0, 2.0):
        for S in (80., 90., 100., 110., 120.):
            C = bs_call(S, K, T, r, sigma); P = bs_put(S, K, T, r, sigma)
            rows.append({"T": T, "S": S, "m = ln(S/K)": moneyness(S, K),
                         "call": C, "put": P,
                         "C - P": C - P, "S - Ke^-rT": S - K*np.exp(-r*T)})
    price_tab = pd.DataFrame(rows)
    with pd.option_context("display.float_format", lambda v: f"{v:13.6f}"):
        print(price_tab.to_string(index=False))

    print("\n  PUT-CALL PARITY OVER A GRID   (16.17)")
    print("      eps = C - P - S + K e^{-rT}    must be 0\n")
    Sg    = np.linspace(40, 180, 15)
    Kg    = np.linspace(60, 140, 9)
    Tg    = np.array([0.05, 0.25, 1.0, 2.0, 5.0])
    rg    = np.array([0.0, 0.03, 0.07, 0.12])
    sigg  = np.array([0.05, 0.15, 0.30, 0.60, 1.00])
    S4, K4, T4, r4, s4 = np.meshgrid(Sg, Kg, Tg, rg, sigg, indexing="ij")
    eps = put_call_parity_error(S4, K4, T4, r4, s4)
    n_pts = eps.size
    print(f"    grid points          {n_pts:,}")
    print(f"    max |eps|            {np.abs(eps).max():.3e}")
    print(f"    mean |eps|           {np.abs(eps).mean():.3e}")
    print(f"    99.9th pct |eps|     {np.percentile(np.abs(eps), 99.9):.3e}")
    print(f"    typical option value {bs_call(100,100,1,0.05,0.25):.3f}")
    ok = np.abs(eps).max() < 1e-9
    print(f"\n    {'PARITY HOLDS' if ok else 'PARITY VIOLATED'} to machine precision.")
    print("\n    Parity is model-free: it follows from a static no-arbitrage argument,")
    print("    not from the Black-Scholes dynamics. It therefore tests the CODE, not")
    print("    the model. A non-zero eps would mean an implementation bug.")
    return price_tab, eps


# ============================ TASK 12: PARAMETER SENSITIVITY (MANDATORY FIGURES)

def task12_sensitivities(K=100.0, r=0.05, sigma=0.25, T=1.0):
    """Section 16.18 and the mandatory figures 2-5."""
    checks = {}

    # ---- 16.18.1  C(S) ----
    S = np.linspace(1, 200, 400)
    C = bs_call(S, K, T, r, sigma); P = bs_put(S, K, T, r, sigma)
    dCdS = np.gradient(C, S)
    checks["dC/dS > 0 everywhere"] = bool((dCdS > 0).all())
    checks["min dC/dS"] = float(dCdS.min())
    checks["max dC/dS"] = float(dCdS.max())

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
    ax[0].plot(S, C, lw=1.8, color="#1f4e79", label="call price")
    ax[0].plot(S, intrinsic_value(S, K, "call"), "k--", lw=1.2, label="intrinsic")
    ax[0].plot(S, np.maximum(S - K*np.exp(-r*T), 0), ":", color="grey", lw=1.2,
               label=r"lower bound $S-Ke^{-rT}$")
    ax[0].axvline(K, color="grey", lw=0.8)
    ax[0].set(xlabel="S", ylabel="C", title=r"Figure 2: $C(S)$")
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].plot(S, dCdS, lw=1.8, color="#c0504d")
    ax[1].axhline(0, color="k", lw=0.7); ax[1].axhline(1, color="k", lw=0.7, ls=":")
    ax[1].axvline(K, color="grey", lw=0.8)
    ax[1].set(xlabel="S", ylabel=r"$\partial C/\partial S$",
              title=r"$\partial C/\partial S \in (0,1)$, monotone increasing")
    finish(fig, "w03_sensitivity_S")

    # ---- 16.18.2  C(K) ----
    Kg = np.linspace(40, 180, 400)
    CK = bs_call(100.0, Kg, T, r, sigma)
    checks["dC/dK < 0 everywhere"] = bool((np.gradient(CK, Kg) < 0).all())

    # ---- 16.18.3  C(sigma), several moneyness levels ----
    # Vega is analytically S sqrt(T) phi(d1) > 0 for every sigma > 0. A numerical
    # gradient cannot confirm this at very small sigma, where deep OTM prices are
    # of order 1e-60 and the finite difference is pure floating-point noise. So the
    # positivity claim is tested on the closed form, and the numerical gradient is
    # checked only where double precision can resolve it.
    sg = np.linspace(0.001, 1.2, 400)
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
    analytic_ok, numeric_ok = True, True
    resolvable = sg >= 0.02            # below this, phi(d1) underflows to exactly 0
    min_vega, sig_num_min = np.inf, 0.0
    for S0, lab in [(80, "OTM  S=80"), (100, "ATM  S=100"), (120, "ITM  S=120")]:
        Cs = bs_call(S0, K, T, r, sg)
        vega = S0 * np.sqrt(T) * stats.norm.pdf(d1_d2(S0, K, T, r, sg)[0])
        analytic_ok &= bool((vega[resolvable] > 0).all())
        # C(sigma) is non-decreasing everywhere. It is only STRICTLY increasing
        # where vega is large enough to register in double precision; in the flat
        # deep-OTM and deep-ITM regions consecutive prices are bitwise equal, so
        # the numerical slope is exactly 0 rather than negative.
        g = np.gradient(Cs, sg)
        numeric_ok &= bool((g >= 0).all())
        sig_num_min = max(sig_num_min, float(g.min()))
        min_vega = min(min_vega, float(vega[resolvable].min()))
        ax[0].plot(sg, Cs, lw=1.7, label=lab)
    checks["vega > 0 analytically, sigma >= 0.02"] = analytic_ok
    checks["C(sigma) non-decreasing numerically"] = numeric_ok
    checks["most negative numerical slope"] = sig_num_min
    checks["min analytic vega, sigma >= 0.02"] = min_vega
    # At sigma = 0.001 and S = 80, d1 ~ -173, so phi(d1) is about 1e-6500 and
    # underflows to exactly 0 in double precision. Vega remains strictly positive
    # mathematically; it is simply not representable. Restricting to sigma >= 0.02
    # (where |d1| <~ 9) tests the mathematics rather than the floating-point format.
    ax[0].set(xlabel=r"$\sigma$", ylabel="C", title=r"Figure 3: $C(\sigma)$")
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].plot(Kg, CK, lw=1.8, color="#1f4e79")
    ax[1].set(xlabel="K", ylabel="C", title=r"$C(K)$: decreasing in strike")
    finish(fig, "w03_sensitivity_sigma_K")

    # ---- 16.18.4  C(T) ----
    Tg = np.linspace(0.001, 5, 400)
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for S0, lab in [(80, "OTM  S=80"), (100, "ATM  S=100"), (120, "ITM  S=120")]:
        ax[0].plot(Tg, bs_call(S0, K, Tg, r, sigma), lw=1.7, label=lab)
    ax[0].set(xlabel="T", ylabel="C", title=r"Figure 4: $C(T)$, call")
    ax[0].legend(frameon=False, fontsize=8)
    put_itm = bs_put(80.0, K, Tg, r, sigma)
    ax[1].plot(Tg, put_itm, lw=1.7, color="#c0504d", label="deep ITM put, S=80")
    ax[1].plot(Tg, bs_put(120.0, K, Tg, r, sigma), lw=1.7, color="#1f4e79",
               label="OTM put, S=120")
    ax[1].set(xlabel="T", ylabel="P", title=r"$P(T)$: a European put can DECREASE in $T$")
    ax[1].legend(frameon=False, fontsize=8)
    finish(fig, "w03_sensitivity_T")
    checks["call increasing in T (ATM)"] = bool((np.gradient(bs_call(100., K, Tg, r, sigma), Tg) > 0).all())
    checks["deep ITM put non-monotone in T"] = bool((np.gradient(put_itm, Tg) < 0).any())

    # ---- 16.18.5  moneyness ----
    m = np.linspace(-1.2, 1.2, 400)
    Sm = K * np.exp(m)
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for TT, lab in [(0.08, "T=1 month"), (0.5, "T=6 months"), (2.0, "T=2 years")]:
        ax[0].plot(m, bs_call(Sm, K, TT, r, sigma)/K, lw=1.7, label=lab)
    ax[0].axvline(0, color="grey", lw=0.9)
    ax[0].text(-0.9, 0.42, "OTM", fontsize=9); ax[0].text(-0.05, 0.42, "ATM", fontsize=9)
    ax[0].text(0.8, 0.42, "ITM", fontsize=9)
    ax[0].set(xlabel=r"$m=\ln(S/K)$", ylabel="C / K",
              title=r"Figure 5: $C(\ln(S/K))$, normalised")
    ax[0].legend(frameon=False, fontsize=8)
    rg = np.linspace(-0.02, 0.15, 400)
    ax[1].plot(rg, bs_call(100., K, T, rg, sigma), lw=1.7, color="#1f4e79", label="call")
    ax[1].plot(rg, bs_put(100., K, T, rg, sigma), lw=1.7, color="#c0504d", label="put")
    ax[1].set(xlabel="r", ylabel="price", title="Effect of the risk-free rate")
    ax[1].legend(frameon=False, fontsize=8)
    finish(fig, "w03_sensitivity_moneyness")
    checks["dC/dr > 0"] = bool((np.gradient(bs_call(100., K, T, rg, sigma), rg) > 0).all())
    checks["dP/dr < 0"] = bool((np.gradient(bs_put(100., K, T, rg, sigma), rg) < 0).all())

    print("  Monotonicity and bound checks (all derived from the plotted curves)\n")
    for k, v in checks.items():
        mark = "" if not isinstance(v, bool) else ("  OK" if v else "  FAIL")
        print(f"    {k:<38}{v!s:>12}{mark}")

    print("""
  WHAT THESE RESULTS MEAN

    dC/dS in (0,1)   Delta. A call never gains more than the stock itself, and
                     never loses value when the stock rises.
    dC/dK < 0        A higher strike means a worse exercise price for the holder.
    dC/dsigma > 0    Vega. The holder has limited downside and unlimited upside,
                     so more dispersion is unambiguously better. True at every
                     moneyness - this is why volatility is THE traded quantity.
    C increasing T   More time means more dispersion, and the strike is discounted
                     further. Both help the call.
    P can DECREASE   For a deep in-the-money European put the dominant effect is
      in T           discounting of the strike received at maturity, which is
                     delayed. This is a genuine feature, not an artefact, and it
                     is exactly why American puts can be worth more than European
                     ones - early exercise recovers that lost discounting.
    dC/dr > 0        A higher rate lowers the present value of the strike paid.
    dP/dr < 0        and lowers the present value of the strike received.
""")
    return checks


# =========================== TASK 13: CONNECT THE DERIVATION BACK TO WEEK 1

WEEK1_TABLE = [
    ("Continuous diffusion; no jumps",
     "15 days beyond 4 sigma, 10 beyond 5 sigma in 2,458 observations"),
    ("Constant volatility sigma",
     "Volatility clustering: |return| autocorrelation 0.31 at lag 1, still 0.09 at lag 30"),
    ("Gaussian log-return innovations",
     "Skewness -1.41, excess kurtosis 20.41, Jarque-Bera p = 0.000"),
    ("Independent increments",
     "Return magnitudes show persistent positive dependence"),
    ("Lognormal asset-price dynamics",
     "Useful benchmark, but an incomplete description of market behaviour"),
    ("Continuous, costless rebalancing",
     "Not tested yet; discrete-hedging error is visible in Task 6 above"),
]


def task13_connect_to_week1():
    """Section 16.19. The purpose is not to reject Black-Scholes but to record
    exactly which assumptions are being made before it is used as the benchmark."""
    tab = pd.DataFrame(WEEK1_TABLE,
                       columns=["Classical model assumption", "Empirical observation (Week 1)"])
    for a, b in WEEK1_TABLE:
        print(f"    {a:<36}  {b}")
    print("""
  The purpose is NOT to reject Black-Scholes.

  The derivation above is mathematically exact GIVEN its assumptions. What the
  table records is precisely which assumptions are being made, and how each one
  is contradicted by the data already measured. That is what makes the model a
  benchmark rather than a belief - and it defines the specific gaps the later
  AI/ML stage is asked to close.
""")
    return tab


REVIEW_ANSWERS = r"""
   1. Why is ordinary Taylor calculus insufficient for Brownian motion?
      Because (dS)^2 is not negligible. For a differentiable path it is O((dt)^2);
      for Brownian motion sum (Delta W)^2 -> T, so it contributes at order dt.

   2. What is meant by (dW)^2 = dt?
      Shorthand for the quadratic variation of Brownian motion: over [0,T] the sum
      of squared increments converges in L^2 to T, with variance 2T^2/n. It is a
      statement about accumulated sums, not about any single increment.

   3. Why does the second-order term survive in Ito's lemma?
      Because of (2). The (1/2) V_SS (dS)^2 term in the Taylor expansion becomes
      (1/2) sigma^2 S^2 V_SS dt, which is the same order as the first-order terms.

   4. How does Ito's lemma produce the -sigma^2/2 term in the log of a GBM?
      With f = ln S: f' = 1/S, f'' = -1/S^2. Then
          d(ln S) = (1/S)dS - (1/(2S^2))(dS)^2 = mu dt + sigma dW - (1/2)sigma^2 dt.
      The correction is -(1/2) f'' scaled by (dS)^2 = sigma^2 S^2 dt.

   5. Difference between the empirical Taylor/Jensen correction and the Ito one?
      The Week 1 gap mean(R) - mean(r) is an algebraic identity in sample moments,
      true of any small numbers, with no probability model and no power to confirm
      GBM - a deterministic +-sigma sequence reproduces it. The Ito correction is a
      statement about the continuous-time process, exact at every sigma, arising
      from quadratic variation. They agree numerically under GBM; they are not the
      same claim.

   6. Financial meaning of Delta = V_S?
      The number of units of the underlying that replicates the derivative's
      first-order price exposure. Holding -V_S units neutralises that exposure.

   7. Which stochastic term is eliminated by delta hedging?
      The sigma S V_S dW term in dV, cancelled against V_S times the sigma S dW in
      dS. Both are driven by the SAME Brownian motion, which is what makes exact
      cancellation possible.

   8. Why does mu disappear from the Black-Scholes PDE?
      mu enters dV only through V_S dS, and enters Delta dS through the same
      channel. Setting Delta = V_S makes the two contributions identical, so they
      cancel with the dW term. Removing the risk and removing the drift are the
      same operation. Demonstrated numerically in Task 6: the hedged P&L
      distribution is invariant across mu = 0.00, 0.15, 0.30.

   9. Why must a locally riskless portfolio earn the risk-free rate?
      Otherwise a costless arbitrage exists: borrow at r, buy the portfolio (or the
      reverse), and collect a certain profit with no capital and no risk.

  10. Why is a terminal condition required in addition to the PDE?
      The PDE constrains how value evolves, not what it is. It has infinitely many
      solutions; the payoff max(S-K,0) selects the one that is the call price.

  11. Difference between ITM, ATM and OTM?
      With m = ln(S/K): m > 0 in the money, m ~ 0 at the money, m < 0 out of the
      money (for a call). ATM options carry the most gamma and vega per unit of
      premium; deep ITM options behave almost like the stock itself.

  12. What does put-call parity test in the implementation?
      The code, not the model. Parity follows from a static arbitrage argument and
      holds for any arbitrage-free model, so a non-zero error indicates a coding
      bug rather than a modelling limitation.
"""


# ============================================================================ main

def run():
    t0 = time.time()
    print("=" * 78)
    print("SEPTEMBER WEEK 2 - ITO'S LEMMA AND THE BLACK-SCHOLES EQUATION")
    print("September Week 2 plan, section 16")
    print("=" * 78)
    if SHOW_PLOTS:
        print("\nNOTE: 7 figure windows will open; the script pauses until each is closed.")
        print("Set SHOW_PLOTS = False to skip this.")

    rule("MODULE SELF-TEST  -  black_scholes.py")
    module_ok = _self_test()

    rule("TASKS 1-2  -  THE DERIVATION  (16.5, 16.6)")
    print(DERIVATION_1_2)
    ds2 = task1_ds_squared()

    rule("TASK 3  -  NUMERICAL ILLUSTRATION OF ITO'S LEMMA  (16.7)   [figure 1]")
    w2tab = task3_d_W_squared()

    rule("TASK 4  -  SECOND ITO EXPERIMENT:  f(S) = ln S  (16.8)")
    logtab = task4_log_S()

    rule("TASKS 5-7  -  DELTA HEDGING AND THE NO-ARBITRAGE ARGUMENT  (16.9-16.12)")
    print(DERIVATION_5_7)

    rule("TASK 6 (numerical)  -  WHERE DID mu GO?  (16.11)")
    hedge = task6_hedge_numerically()

    rule("TASKS 7-8  -  PDE RESIDUAL AND TERMINAL CONDITION  (16.12-16.14)")
    pde = task7_pde_residual()

    rule("TASKS 9-11  -  PRICING AND PUT-CALL PARITY  (16.15-16.17)")
    price_tab, eps = tasks9_11_pricing_and_parity()

    rule("TASK 12  -  PARAMETER SENSITIVITY  (16.18)   [figures 2-5]")
    checks = task12_sensitivities()

    rule("TASK 13  -  CONNECTING THE DERIVATION BACK TO WEEK 1  (16.19)")
    w1tab = task13_connect_to_week1()

    rule("ANSWERS TO THE WEEK 2 REVIEW QUESTIONS  (16.22)")
    print(REVIEW_ANSWERS)

    # ---- save ----
    ds2.to_csv("results/w03_ds_squared.csv", index=False)
    w2tab.to_csv("results/w03_ito_d_W_squared.csv", index=False)
    logtab.to_csv("results/w03_ito_log_S.csv", index=False)
    hedge.to_csv("results/w03_delta_hedge.csv", index=False)
    pde.to_csv("results/w03_pde_residual.csv", index=False)
    price_tab.to_csv("results/w03_price_table.csv", index=False)
    w1tab.to_csv("results/w03_assumptions_vs_week1.csv", index=False)

    rule("SUMMARY")
    summary = {
        "seed":                        SEED,
        "module_self_test_passed":     module_ok,
        "dS2_ratio_at_n_1e6":          float(ds2["ratio"].iloc[-1]),
        "ito_W2_max_abs_error":        float(np.abs(w2tab["W_T^2 (truth)"]
                                                    - w2tab["2W dW + sum(dW)^2"]).max()),
        "ordinary_calculus_error":     float(w2tab["error of ordinary"].iloc[-1]),
        "ordinary_error_theory_T":     1.0,
        "logS_correction_observed":    float(logtab["difference"].iloc[-1]),
        "logS_correction_theory":      float(logtab["-sigma^2 T/2"].iloc[-1]),
        "hedged_sd_spread_over_mu":    float(np.ptp(hedge["hedged sd"].values)),
        "unhedged_mean_spread_over_mu": float(np.ptp(hedge["unhedged mean"].values)),
        "pde_max_abs_residual":        float(pde["PDE residual"].abs().max()),
        "parity_max_abs_error":        float(np.abs(eps).max()),
        "parity_grid_points":          int(eps.size),
        "all_monotonicity_checks":     all(v for v in checks.values() if isinstance(v, bool)),
    }
    pd.Series(summary).to_csv("results/w03_summary.csv", header=False)
    for k, v in summary.items():
        print(f"  {k:<30} {v}")
    print(f"\n  runtime {time.time()-t0:.1f} s")
    print("\nSaved: results/w03_*.csv   Figures: figures/w03_*.png")


def main():
    for d in ("figures", "results"):
        os.makedirs(d, exist_ok=True)
    log_path = "results/w03_console_output.txt"
    tee = _Tee(log_path); sys.stdout = tee
    try:
        run()
        print(f"\nFull console output written to: {log_path}")
    finally:
        sys.stdout = tee.stdout; tee.close()
    print(f"\n>>> Share this one file for review: {log_path}")


if __name__ == "__main__":
    main()
