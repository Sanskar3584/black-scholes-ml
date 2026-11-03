"""
September Week 1 (Part A) - Brownian Motion
===========================================

Project : AI/ML-Augmented Black-Scholes Framework
Ref     : September plan sections 15.4.1 and 15.4.2
Text    : Carol Alexander, Market Risk Analysis Vol. I, section I.3.7

Learning objective (15.4.1)
---------------------------
The Brownian-motion increment satisfies

        dW = W_{t+dt} - W_t,        dW ~ N(0, dt)

or equivalently

        dW = sqrt(dt) * Z,          Z ~ N(0,1)

Computational work (15.4.2)
---------------------------
    1. Generate standard normal random variables
    2. Generate Brownian increments
    3. Construct complete Brownian-motion paths
    4. Verify numerically that  E[dW] ~= 0
    5. Verify                   Var(dW) ~= dt
    6. Repeat the experiment for different values of dt
    7. Demonstrate numerically that  SD(dW)  proportional to  sqrt(dt)

Additional experiment (preparation for September Week 2)
-------------------------------------------------------
    8. Independence and stationarity of increments
    9. Quadratic variation:  sum (dW)^2 -> T  while  sum |dW| -> infinity

Item 9 is not required by section 15.4.2. It is included because the term
(1/2) sigma^2 S^2 V_SS in the Black-Scholes PDE originates from exactly this
property, and section 15.5.3 question 2 asks where that term comes from.

Usage
-----
    python week02a_brownian_motion.py

Outputs
-------
    figures/w02_bm_*.png
    results/w02_bm_summary.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

# --------------------------------------------------------------------------- config

SEED       = 20260825          # same seed convention as Week 1
SHOW_PLOTS = True              # set False to save figures without opening windows

rng = np.random.default_rng(SEED)

if not SHOW_PLOTS:
    matplotlib.use("Agg")

plt.rcParams.update({
    "figure.figsize": (11, 4.2),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def rule(title=""):
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def finish(fig, name):
    fig.tight_layout()
    fig.savefig(f"figures/{name}.png", dpi=150, bbox_inches="tight")
    plt.show() if SHOW_PLOTS else plt.close(fig)


# ------------------------------------------------------------- 1. standard normals

def task1_standard_normals(n=1_000_000):
    """Generate Z ~ N(0,1) and confirm the generator behaves as claimed."""
    Z = rng.standard_normal(n)
    print(f"  n            {n:,}")
    print(f"  mean         {Z.mean():+.6f}   (theory 0)")
    print(f"  variance     {Z.var(ddof=1):.6f}   (theory 1)")
    print(f"  skewness     {stats.skew(Z):+.6f}   (theory 0)")
    print(f"  ex. kurtosis {stats.kurtosis(Z):+.6f}   (theory 0)")

    # standard error of the mean is 1/sqrt(n): the observed mean must sit inside it
    se = 1 / np.sqrt(n)
    print(f"\n  |mean| / (1/sqrt(n)) = {abs(Z.mean())/se:.3f} standard errors")
    assert abs(Z.mean()) < 5 * se, "generator mean is implausible"

    jb, p = stats.jarque_bera(Z[:5000])
    print(f"  Jarque-Bera on first 5000: p = {p:.3f}  "
          f"({'consistent with normal' if p > 0.01 else 'REJECTED'})")
    return Z


# ---------------------------------------------------------- 2-3. increments, paths

def task2_increments(dt=1/252, n=10_000):
    """dW = sqrt(dt) * Z."""
    Z  = rng.standard_normal(n)
    dW = np.sqrt(dt) * Z
    print(f"  dt = {dt:.6f},  n = {n:,}")
    print(f"  mean(dW)  {dW.mean():+.8f}   (theory 0)")
    print(f"  var(dW)   {dW.var(ddof=1):.8f}   (theory dt = {dt:.8f})")
    print(f"  sd(dW)    {dW.std(ddof=1):.8f}   (theory sqrt(dt) = {np.sqrt(dt):.8f})")
    return dW


def task3_paths(n_paths=8, n_steps=1000, T=1.0):
    """W_t = cumulative sum of increments, with W_0 = 0."""
    dt = T / n_steps
    dW = np.sqrt(dt) * rng.standard_normal((n_paths, n_steps))
    W  = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(dW, axis=1)], axis=1)
    t  = np.linspace(0, T, n_steps + 1)

    assert np.all(W[:, 0] == 0), "W_0 must be 0"
    print(f"  {n_paths} paths, {n_steps} steps, T = {T},  W_0 = 0 for all paths")

    # the spread of W_t should grow as sqrt(t) - needs many paths to check
    n_big = 2000
    big = np.concatenate([np.zeros((n_big, 1)),
                          np.cumsum(np.sqrt(dt) * rng.standard_normal((n_big, n_steps)), axis=1)],
                         axis=1)
    emp_sd = big.std(axis=0, ddof=1)

    # W_T ~ N(0, T): test on the large sample, not on the 8 plotted paths
    WT = big[:, -1]
    se_mean = np.sqrt(T / n_big)
    print(f"\n  W_T moments from {n_big:,} paths (8 paths is far too few to test):")
    print(f"    mean  {WT.mean():+.4f}   theory 0        "
          f"({abs(WT.mean())/se_mean:.2f} standard errors)")
    print(f"    sd    {WT.std(ddof=1):.4f}   theory sqrt(T) = {np.sqrt(T):.4f}")
    jb, p = stats.jarque_bera(WT)
    print(f"    Jarque-Bera p = {p:.3f}  "
          f"({'consistent with normal' if p > 0.01 else 'REJECTED'})")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for i in range(n_paths):
        ax[0].plot(t, W[i], lw=0.9)
    ax[0].axhline(0, color="k", lw=0.7)
    ax[0].set(xlabel="t", ylabel="$W_t$", title=f"{n_paths} Brownian-motion paths")

    ax[1].plot(t, emp_sd, lw=1.4, color="#1f4e79", label="empirical SD (2000 paths)")
    ax[1].plot(t, np.sqrt(t), "r--", lw=1.6, label=r"theory: $\sqrt{t}$")
    ax[1].fill_between(t, -2*np.sqrt(t), 2*np.sqrt(t), alpha=0.10, color="grey",
                       label=r"$\pm 2\sqrt{t}$ envelope")
    for i in range(n_paths):
        ax[1].plot(t, W[i], lw=0.5, alpha=0.45)
    ax[1].set(xlabel="t", ylabel="$W_t$", title=r"Spread grows as $\sqrt{t}$, not $t$")
    ax[1].legend(frameon=False, fontsize=8)
    finish(fig, "w02_bm_paths")
    return t, W


# --------------------------------------------------- 4-7. moments and dt-dependence

def tasks4to7_scaling(dts=None, n=200_000):
    """E[dW] ~= 0, Var(dW) ~= dt, and SD(dW) proportional to sqrt(dt)."""
    if dts is None:
        dts = [1.0, 1/2, 1/4, 1/12, 1/52, 1/252, 1/1000, 1/5000, 1/20000]

    rows = []
    for dt in dts:
        dW = np.sqrt(dt) * rng.standard_normal(n)
        mean, var, sd = dW.mean(), dW.var(ddof=1), dW.std(ddof=1)
        se_mean = np.sqrt(dt / n)                 # SD of the sample mean
        rows.append({
            "dt":            dt,
            "mean(dW)":      mean,
            "mean / s.e.":   mean / se_mean,      # should be O(1), NOT growing
            "var(dW)":       var,
            "var / dt":      var / dt,            # should be ~1
            "sd(dW)":        sd,
            "sd / sqrt(dt)": sd / np.sqrt(dt),    # should be ~1
        })
    tab = pd.DataFrame(rows)

    with pd.option_context("display.float_format", lambda v: f"{v:12.6f}"):
        print(tab.to_string(index=False))

    print("\n  Task 4:  mean(dW) is within a few standard errors of 0 at every dt")
    print("           (the raw mean shrinks with dt only because the scale does;")
    print("            the column 'mean / s.e.' is the scale-free test)")
    print("  Task 5:  var(dW)/dt ~= 1 at every dt")
    print(f"           max deviation {np.abs(tab['var / dt']-1).max():.4f}")

    # --- task 7: fit log SD = a + b log dt, expect b = 1/2 -------------------
    x, y = np.log(tab["dt"].values), np.log(tab["sd(dW)"].values)
    slope, intercept, rval, _, stderr = stats.linregress(x, y)
    print(f"\n  Task 7:  log SD(dW) = a + b log(dt)")
    print(f"           slope b     {slope:.6f}  +/- {stderr:.6f}   (theory 0.5)")
    print(f"           intercept a {intercept:+.6f}                 (theory 0)")
    print(f"           R^2         {rval**2:.6f}")

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))
    ax[0].plot(tab["dt"], tab["var(dW)"], "o-", color="#1f4e79", label="empirical")
    ax[0].plot(tab["dt"], tab["dt"], "r--", lw=1.5, label=r"theory: Var = $\Delta t$")
    ax[0].set(xscale="log", yscale="log", xlabel=r"$\Delta t$", ylabel=r"Var$(\Delta W)$",
              title=r"Task 5:  Var$(\Delta W) = \Delta t$")
    ax[0].legend(frameon=False, fontsize=8)

    ax[1].plot(np.sqrt(tab["dt"]), tab["sd(dW)"], "o-", color="#1f4e79", label="empirical")
    lim = np.sqrt(tab["dt"]).max()
    ax[1].plot([0, lim], [0, lim], "r--", lw=1.5, label="slope 1 through origin")
    ax[1].set(xlabel=r"$\sqrt{\Delta t}$", ylabel=r"SD$(\Delta W)$",
              title=r"Task 7:  SD $\propto \sqrt{\Delta t}$")
    ax[1].legend(frameon=False, fontsize=8)

    ax[2].plot(x, y, "o", color="#1f4e79")
    ax[2].plot(x, intercept + slope * x, "r--", lw=1.5, label=f"fit, slope = {slope:.4f}")
    ax[2].set(xlabel=r"$\ln \Delta t$", ylabel=r"$\ln$ SD$(\Delta W)$",
              title="Log-log fit confirms exponent 1/2")
    ax[2].legend(frameon=False, fontsize=8)
    finish(fig, "w02_bm_scaling")

    # --- increment distribution at one dt -----------------------------------
    dt = 1/252
    dW = np.sqrt(dt) * rng.standard_normal(100_000)
    grid = np.linspace(dW.min(), dW.max(), 500)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].hist(dW, bins=120, density=True, alpha=0.65, color="#1f4e79", label="simulated")
    ax[0].plot(grid, stats.norm.pdf(grid, 0, np.sqrt(dt)), "r-", lw=1.8,
               label=r"$N(0,\Delta t)$")
    ax[0].set(xlabel=r"$\Delta W$", ylabel="density",
              title=fr"Increment distribution, $\Delta t = 1/252$")
    ax[0].legend(frameon=False)
    stats.probplot(dW[:20000], dist="norm", plot=ax[1])
    ax[1].get_lines()[0].set(markersize=2, alpha=0.4, color="#1f4e79")
    ax[1].get_lines()[1].set(color="red", lw=1.5)
    ax[1].set_title("QQ plot: increments are exactly normal by construction")
    finish(fig, "w02_bm_increment_distribution")

    return tab, slope


# --------------------------------------------- 8. independence and stationarity

def task8_independence(n=200_000, dt=1/252):
    """Increments must be independent and identically distributed."""
    dW = np.sqrt(dt) * rng.standard_normal(n)

    print("  Autocorrelation of increments (theory: 0 at all non-zero lags)")
    acf = [np.corrcoef(dW[:-k], dW[k:])[0, 1] for k in range(1, 11)]
    band = 1.96 / np.sqrt(n)
    for k, a in enumerate(acf, 1):
        flag = "" if abs(a) < band else "   <- outside 95% band"
        print(f"    lag {k:>2}   {a:+.5f}{flag}")
    print(f"    95% band  +/-{band:.5f}")

    # stationarity: split into blocks, moments must not drift
    blocks = np.array_split(dW, 8)
    print("\n  Stationarity: moments across 8 equal blocks")
    print(f"    {'block':>6}{'mean':>14}{'sd':>14}")
    for i, b in enumerate(blocks, 1):
        print(f"    {i:>6}{b.mean():>14.6f}{b.std(ddof=1):>14.6f}")
    sds = [b.std(ddof=1) for b in blocks]
    print(f"    spread of block SDs: {np.ptp(sds):.6f}  "
          f"(compare sqrt(dt) = {np.sqrt(dt):.6f})")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.0))
    ax[0].stem(range(1, 11), acf, basefmt=" ")
    ax[0].axhline(band, color="r", ls="--", lw=1)
    ax[0].axhline(-band, color="r", ls="--", lw=1, label="95% band")
    ax[0].axhline(0, color="k", lw=0.7)
    ax[0].set(xlabel="lag", ylabel="autocorrelation",
              title="Task 8: increments are uncorrelated")
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].plot(range(1, 9), sds, "o-", color="#1f4e79", label="block SD")
    ax[1].axhline(np.sqrt(dt), color="r", ls="--", lw=1.5, label=r"$\sqrt{\Delta t}$")
    ax[1].set(xlabel="block", ylabel="SD", title="Increments are stationary")
    ax[1].legend(frameon=False, fontsize=8)
    finish(fig, "w02_bm_independence")
    return acf


# ------------------------------------------ 9. quadratic variation (for Week 2)

def task9_quadratic_variation(T=1.0):
    """sum (dW)^2 -> T, while sum |dW| diverges. This is why Ito != chain rule."""
    ns = [10, 50, 100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]
    rows = []
    for n in ns:
        dt = T / n
        dW = np.sqrt(dt) * rng.standard_normal(n)
        qv = (dW ** 2).sum()
        # Var(sum (dW)^2) = n * Var((dW)^2) = n * 2*dt^2 = 2T^2/n
        se_qv = T * np.sqrt(2 / n)
        rows.append({
            "n steps":        n,
            "dt":             dt,
            "sum (dW)^2":     qv,                   # -> T
            "s.e. of QV":     se_qv,                # = T sqrt(2/n)
            "(QV - T)/s.e.":  (qv - T) / se_qv,     # should stay O(1)
            "sum |dW|":       np.abs(dW).sum(),     # -> infinity
            "sum (dW)^3":     (dW ** 3).sum(),      # -> 0
        })
    tab = pd.DataFrame(rows)
    with pd.option_context("display.float_format", lambda v: f"{v:14.6f}"):
        print(tab.to_string(index=False))

    print(f"\n  Convergence is not eyeballed: Var(sum (dW)^2) = 2T^2/n exactly, so the")
    print(f"  standard error is T*sqrt(2/n). The column '(QV - T)/s.e.' stays of order 1")
    print(f"  at every n, which is what convergence in L^2 means.")
    print(f"  max |(QV - T)/s.e.| = {np.abs(tab['(QV - T)/s.e.']).max():.2f}")

    print(f"\n  sum (dW)^2  ->  T = {T}          [quadratic variation is FINITE and non-zero]")
    print( "  sum |dW|    ->  infinity          [first variation is INFINITE]")
    print( "  sum (dW)^3  ->  0                 [higher orders vanish]")
    print()
    print( "  Consequence.  In an ordinary Taylor expansion the (dx)^2 term is")
    print( "  negligible.  For Brownian motion it is NOT: (dW)^2 accumulates at")
    print( "  rate dt.  That single fact is why Ito's lemma carries the extra term")
    print( "    (1/2) sigma^2 S^2 V_SS")
    print( "  and therefore why the Black-Scholes PDE has a second-derivative term")
    print( "  at all.  See September plan section 15.5.3, question 2.")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].semilogx(tab["n steps"], tab["sum (dW)^2"], "o-", color="#1f4e79",
                   label=r"$\sum (\Delta W)^2$")
    ax[0].axhline(T, color="r", ls="--", lw=1.5, label=f"T = {T}")
    ax[0].set(xlabel="number of steps n", ylabel=r"$\sum (\Delta W)^2$",
              title="Quadratic variation converges to T")
    ax[0].legend(frameon=False)
    ax[1].loglog(tab["n steps"], tab["sum |dW|"], "o-", color="#c0504d",
                 label=r"$\sum |\Delta W|$")
    ax[1].loglog(tab["n steps"], np.sqrt(2/np.pi) * np.sqrt(tab["n steps"]), "k--",
                 lw=1.4, label=r"theory: $\sqrt{2n/\pi}$")
    ax[1].set(xlabel="number of steps n", ylabel=r"$\sum |\Delta W|$",
              title="First variation diverges")
    ax[1].legend(frameon=False)
    finish(fig, "w02_bm_quadratic_variation")
    return tab


# --------------------------------------------------------------------------- main

def main():
    for d in ("figures", "results"):
        os.makedirs(d, exist_ok=True)

    rule("TASK 1  -  STANDARD NORMAL RANDOM VARIABLES")
    task1_standard_normals()

    rule("TASK 2  -  BROWNIAN INCREMENTS   dW = sqrt(dt) Z")
    task2_increments()

    rule("TASK 3  -  COMPLETE BROWNIAN-MOTION PATHS")
    task3_paths()

    rule("TASKS 4-7  -  MOMENTS AND DEPENDENCE ON dt")
    tab, slope = tasks4to7_scaling()

    rule("TASK 8  -  INDEPENDENCE AND STATIONARITY OF INCREMENTS")
    acf = task8_independence()

    rule("TASK 9  -  QUADRATIC VARIATION      [preparation for September Week 2]")
    qv = task9_quadratic_variation()

    rule("SUMMARY")
    summary = {
        "seed":                    SEED,
        "sd_scaling_exponent":     float(slope),
        "sd_scaling_theory":       0.5,
        "max_|var/dt - 1|":        float(np.abs(tab["var / dt"] - 1).max()),
        "max_|sd/sqrt(dt) - 1|":   float(np.abs(tab["sd / sqrt(dt)"] - 1).max()),
        "max_|autocorrelation|":   float(np.max(np.abs(acf))),
        "QV_at_n=100000":          float(qv["sum (dW)^2"].iloc[-1]),
        "QV_theory":               1.0,
        "first_variation_n=100000": float(qv["sum |dW|"].iloc[-1]),
    }
    pd.Series(summary).to_csv("results/w02_bm_summary.csv", header=False)
    tab.to_csv("results/w02_bm_scaling.csv", index=False)
    qv.to_csv("results/w02_bm_quadratic_variation.csv", index=False)
    for k, v in summary.items():
        print(f"  {k:<28} {v}")
    print("\nSaved: results/w02_bm_*.csv   Figures: figures/w02_bm_*.png")


if __name__ == "__main__":
    main()
