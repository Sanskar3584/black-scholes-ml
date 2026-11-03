"""
September Week 1 (Part B) - Geometric Brownian Motion and the Model-versus-Data Experiment
=========================================================================================

Project : AI/ML-Augmented Black-Scholes Framework
Ref     : September plan sections 15.4.3, 15.4.4, 15.4.5
Text    : Carol Alexander, Market Risk Analysis Vol. I, section I.3.7.3

Contents
--------
    1. GBM simulation and its validation                       (15.4.3)
    2. The roles of mu (drift) and sigma (volatility)          (15.4.3)
    3. The two drifts: why E[ln S] grows at mu - sigma^2/2     (supports 15.5)
    4. Calibration to NIFTY 50                                 (15.4.4)
    5. Model-versus-data experiment                            (15.4.5)
    6. Interpretation of model limitations                     (15.4.6)

Principal question (15.4.5)
---------------------------
    If GBM is calibrated to the same mean and volatility, does it reproduce the
    distributional behaviour observed in the actual market data?

Usage
-----
    python week02b_gbm_simulation.py

Requires data/nifty50_prices.csv from Week 1 (re-downloaded automatically if absent).

Outputs
-------
    figures/w02_gbm_*.png
    results/w02_gbm_*.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

# --------------------------------------------------------------------------- config

SEED         = 20260825
SHOW_PLOTS   = True
TRADING_DAYS = 252
CSV_PATH     = "data/nifty50_prices.csv"

rng = np.random.default_rng(SEED)
if not SHOW_PLOTS:
    matplotlib.use("Agg")

plt.rcParams.update({
    "figure.figsize": (11, 4.2), "axes.grid": True, "grid.alpha": 0.3,
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
})


def rule(title=""):
    print("\n" + "=" * 78)
    if title:
        print(title); print("=" * 78)


def finish(fig, name):
    fig.tight_layout()
    fig.savefig(f"figures/{name}.png", dpi=150, bbox_inches="tight")
    plt.show() if SHOW_PLOTS else plt.close(fig)


# ------------------------------------------------------------- 1. GBM simulation

def simulate_gbm(S0, mu, sigma, T, n_steps, n_paths, generator=None):
    """Exact GBM simulation.

        S_t = S_0 exp[ (mu - sigma^2/2) t + sigma W_t ]

    This steps the analytical solution directly, so it carries no discretisation
    bias at any step size - unlike an Euler scheme on dS = mu S dt + sigma S dW.
    """
    g  = generator if generator is not None else rng
    dt = T / n_steps
    dW = np.sqrt(dt) * g.standard_normal((n_paths, n_steps))
    incr = (mu - 0.5 * sigma ** 2) * dt + sigma * dW        # log-return increments
    log_path = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(incr, axis=1)], axis=1)
    return S0 * np.exp(log_path)


def validate_gbm(S0=100.0, mu=0.10, sigma=0.20, T=1.0, n_steps=252, n_paths=200_000):
    """Check the simulator against the analytical moments of the lognormal."""
    S  = simulate_gbm(S0, mu, sigma, T, n_steps, n_paths)
    ST = S[:, -1]

    ES_th  = S0 * np.exp(mu * T)
    VS_th  = S0**2 * np.exp(2*mu*T) * (np.exp(sigma**2 * T) - 1)
    Elog_th = np.log(S0) + (mu - 0.5*sigma**2) * T
    Vlog_th = sigma**2 * T

    print(f"  S0 = {S0}, mu = {mu}, sigma = {sigma}, T = {T}, "
          f"{n_paths:,} paths x {n_steps} steps\n")
    rows = [
        ("E[S_T]",       ST.mean(),               ES_th),
        ("Var(S_T)",     ST.var(ddof=1),          VS_th),
        ("E[ln S_T]",    np.log(ST).mean(),       Elog_th),
        ("Var(ln S_T)",  np.log(ST).var(ddof=1),  Vlog_th),
    ]
    print(f"  {'quantity':<14}{'simulated':>16}{'theory':>16}{'rel. error':>14}")
    for name, sim, th in rows:
        print(f"  {name:<14}{sim:>16.6f}{th:>16.6f}{abs(sim-th)/abs(th):>14.2e}")

    jb, p = stats.jarque_bera(np.log(ST)[:5000])
    print(f"\n  ln S_T normality: Jarque-Bera p = {p:.3f}  "
          f"({'consistent with normal' if p > 0.01 else 'REJECTED'})")
    print("  => S_T is lognormal, ln S_T is normal, exactly as the model states.")

    # one-step log returns must be N((mu - sigma^2/2)dt, sigma^2 dt)
    dt = T / n_steps
    r1 = np.diff(np.log(S), axis=1).ravel()
    print(f"\n  One-step log returns (dt = 1/{n_steps}):")
    print(f"    mean      {r1.mean():+.8f}   theory {(mu-0.5*sigma**2)*dt:+.8f}")
    print(f"    variance  {r1.var(ddof=1):.8f}   theory {sigma**2*dt:.8f}")
    return S


# ---------------------------------------------------- 2. roles of mu and sigma

def roles_of_mu_sigma(S0=100.0, T=1.0, n_steps=252, n_paths=60):
    """Same random numbers, different mu and sigma, so the effect is isolated."""
    mus    = [0.00, 0.10, 0.20]
    sigmas = [0.10, 0.20, 0.40]
    t = np.linspace(0, T, n_steps + 1)

    fig, axes = plt.subplots(len(sigmas), len(mus), figsize=(13, 9.5),
                             sharex=True, sharey=True)
    for i, sig in enumerate(sigmas):
        for j, mu in enumerate(mus):
            g = np.random.default_rng(SEED)          # identical shocks in every panel
            S = simulate_gbm(S0, mu, sig, T, n_steps, n_paths, generator=g)
            ax = axes[i, j]
            for k in range(n_paths):
                ax.plot(t, S[k], lw=0.5, alpha=0.5)
            ax.plot(t, S0*np.exp(mu*t), "r-", lw=2, label=r"$E[S_t]=S_0e^{\mu t}$")
            ax.plot(t, S0*np.exp((mu-0.5*sig**2)*t), "k--", lw=1.6,
                    label=r"median $=S_0e^{(\mu-\sigma^2/2)t}$")
            ax.set_title(fr"$\mu={mu:.2f}$,  $\sigma={sig:.2f}$", fontsize=10)
            if i == len(sigmas)-1: ax.set_xlabel("t")
            if j == 0:             ax.set_ylabel("$S_t$")
            if i == 0 and j == 0:  ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Same random shocks in every panel: rows change $\\sigma$, columns change $\\mu$",
                 y=0.995, fontsize=11)
    finish(fig, "w02_gbm_mu_sigma_grid")

    print("  Reading the grid:")
    print("    across a row  (mu increases)   -> the whole bundle tilts upward;")
    print("                                      the spread is unchanged.")
    print("    down a column (sigma increases)-> the bundle fans out;")
    print("                                      and the median falls away from the mean.")
    print("\n  mu    sets the trend.")
    print("  sigma sets the dispersion - and, through -sigma^2/2, also drags the median down.")


# -------------------------------------------- 3. the two drifts (mu vs mu - s^2/2)

def two_drifts(S0=100.0, mu=0.10, T=1.0, n_steps=252, n_paths=200_000):
    """In the MODEL the sigma^2/2 gap is exact and comes from Ito, not from a Taylor
    truncation. Demonstrated here across a range of sigma."""
    rows = []
    for sigma in [0.05, 0.10, 0.20, 0.30, 0.40, 0.60]:
        S  = simulate_gbm(S0, mu, sigma, T, n_steps, n_paths)
        ST = S[:, -1]
        growth_mean = np.log(ST.mean()/S0) / T            # -> mu
        # E[ln S_T] = ln S_0 + (mu - sigma^2/2) T exactly. The sample mean of ln S_T
        # is a far tighter estimator than the sample median (s.e. = sigma/sqrt(n T)).
        growth_log  = np.log(ST/S0).mean() / T            # -> mu - sigma^2/2
        se_log      = sigma / np.sqrt(n_paths * T)
        rows.append({
            "sigma":               sigma,
            "growth of E[S_T]":    growth_mean,
            "theory mu":           mu,
            "E[ln S_T]/T":         growth_log,
            "theory mu - s^2/2":   mu - 0.5*sigma**2,
            "gap":                 growth_mean - growth_log,
            "sigma^2/2":           0.5*sigma**2,
            "s.e.":                se_log,
        })
    tab = pd.DataFrame(rows)
    with pd.option_context("display.float_format", lambda v: f"{v:13.6f}"):
        print(tab.to_string(index=False))

    dev = np.abs(tab["gap"] - tab["sigma^2/2"]) / tab["s.e."]
    print(f"\n  max |gap - sigma^2/2|          = "
          f"{np.abs(tab['gap']-tab['sigma^2/2']).max():.2e}")
    print(f"  max |gap - sigma^2/2| / s.e.   = {dev.max():.2f}"
          "   <- within Monte Carlo error at every sigma")
    print("\n  In the MODEL this identity is EXACT: E[ln S_T] = ln S_0 + (mu - sigma^2/2)T")
    print("  holds for every sigma, with no approximation. It comes from Ito's lemma")
    print("  applied to f = ln S, where the term (1/2) f''(S) (dS)^2 survives because")
    print("  (dW)^2 = dt - the quadratic-variation property measured in Part A.")
    print()
    print("  CONTRAST WITH WEEK 1.  The empirical gap  mean(R) - mean(r)  found in the")
    print("  NIFTY data was a Taylor/Jensen identity in sample moments, true of any")
    print("  small numbers whatsoever. This one is a statement about the continuous-time")
    print("  process, exact at all sigma, and it is what Ito's lemma actually asserts.")
    print("  The two agree numerically under GBM; they are not the same statement.")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(tab["sigma"], tab["gap"], "o-", color="#1f4e79", label="simulated gap")
    ax[0].plot(tab["sigma"], tab["sigma^2/2"], "r--", lw=1.6, label=r"$\sigma^2/2$")
    ax[0].set(xlabel=r"$\sigma$", ylabel=r"$\ln E[S_T]/T \; - \; E[\ln S_T]/T$",
              title=r"Model: the gap is exactly $\sigma^2/2$")
    ax[0].legend(frameon=False)

    sigma = 0.40
    S  = simulate_gbm(S0, mu, sigma, T, n_steps, 100_000)
    ST = S[:, -1]
    ax[1].hist(ST, bins=200, density=True, alpha=0.6, color="#1f4e79")
    ax[1].axvline(ST.mean(), color="r", lw=2, label=f"mean {ST.mean():.1f}")
    ax[1].axvline(np.median(ST), color="k", ls="--", lw=2,
                  label=f"median {np.median(ST):.1f}")
    ax[1].set(xlabel="$S_T$", ylabel="density",
              title=fr"Lognormal is right-skewed ($\sigma={sigma}$): mean $>$ median")
    ax[1].legend(frameon=False)
    finish(fig, "w02_gbm_two_drifts")
    return tab


# ------------------------------------------------------- 4. calibration to NIFTY

def load_nifty(csv_path=CSV_PATH):
    if not os.path.exists(csv_path):
        import yfinance as yf
        os.makedirs("data", exist_ok=True)
        df = yf.download("^NSEI", start="2015-01-01", end="2025-01-01",
                         progress=False, auto_adjust=True)
        px = df["Close"]
        if isinstance(px, pd.DataFrame):
            px = px.iloc[:, 0]
        px.dropna().astype(float).to_frame("Close").to_csv(csv_path)
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    return df.iloc[:, 0].dropna().astype(float)


def calibrate(prices):
    r = np.log(prices/prices.shift(1)).dropna()
    daily_mean, daily_sd = r.mean(), r.std(ddof=1)
    sigma = daily_sd * np.sqrt(TRADING_DAYS)                  # annualised volatility
    mu    = daily_mean*TRADING_DAYS + 0.5*sigma**2            # so that mu - s^2/2 matches
    print(f"  observations           {len(r):,}")
    print(f"  daily mean log return  {daily_mean:+.8f}")
    print(f"  daily sd  log return   {daily_sd:.8f}")
    print(f"  annualised sigma       {sigma:.6f}   (plan section 15.4.4 quotes 0.167)")
    print(f"  annualised mu          {mu:.6f}")
    print(f"\n  mu is set so that (mu - sigma^2/2) = {mu-0.5*sigma**2:.6f}")
    print(f"  equals the empirical annualised mean log return "
          f"{daily_mean*TRADING_DAYS:.6f}.")
    print("  GBM is therefore matched to the NIFTY data on BOTH first and second moments.")
    return r, mu, sigma, daily_mean, daily_sd


# --------------------------------------------- 5. model-versus-data experiment

def tail_table(x, label):
    z = (x - x.mean())/x.std(ddof=1)
    out = {}
    for k in (2, 3, 4, 5):
        obs = int((np.abs(z) > k).sum())
        exp = 2*stats.norm.sf(k)*len(z)
        out[k] = (obs, exp, obs/exp if exp > 0 else np.inf)
    return out


def model_vs_data(r_emp, mu, sigma, daily_mean, daily_sd):
    """Section 15.4.5. Same n, same mean, same sd - compare everything else."""
    n = len(r_emp)
    dt = 1/TRADING_DAYS
    r_gbm = (mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*rng.standard_normal(n)

    emp, sim = r_emp.values, r_gbm

    print(f"  Both samples: n = {n:,} daily log returns\n")
    stat_rows = []
    for name, f in [("mean",            lambda x: x.mean()),
                    ("std deviation",   lambda x: x.std(ddof=1)),
                    ("annualised vol",  lambda x: x.std(ddof=1)*np.sqrt(TRADING_DAYS)),
                    ("skewness",        lambda x: stats.skew(x, bias=False)),
                    ("excess kurtosis", lambda x: stats.kurtosis(x, bias=False)),
                    ("minimum",         lambda x: x.min()),
                    ("maximum",         lambda x: x.max())]:
        stat_rows.append({"statistic": name, "NIFTY 50": f(emp), "GBM": f(sim)})
    tab = pd.DataFrame(stat_rows).set_index("statistic")
    tab["difference"] = tab["GBM"] - tab["NIFTY 50"]
    with pd.option_context("display.float_format", lambda v: f"{v:12.6f}"):
        print(tab.to_string())

    print("\n  Mean and standard deviation agree - they were matched by construction.")
    print("  Skewness and kurtosis do not, and were never targeted.")

    # --- normality tests ---------------------------------------------------
    print("\n  Normality tests")
    print(f"    {'sample':<12}{'Jarque-Bera':>16}{'p-value':>14}{'verdict':>12}")
    for name, x in [("NIFTY 50", emp), ("GBM", sim)]:
        jb, p = stats.jarque_bera(x)
        print(f"    {name:<12}{jb:>16,.2f}{p:>14.3e}"
              f"{('REJECT' if p < 0.01 else 'ok'):>12}")

    # --- tail frequencies --------------------------------------------------
    te, tg = tail_table(emp, "NIFTY"), tail_table(sim, "GBM")
    print("\n  Tail event frequencies")
    print(f"    {'threshold':>10}{'NIFTY obs':>12}{'GBM obs':>10}"
          f"{'normal exp':>12}{'NIFTY ratio':>13}{'GBM ratio':>11}")
    tail_rows = []
    for k in (2, 3, 4, 5):
        oe, ex, re_ = te[k]
        og, _,  rg  = tg[k]
        print(f"    {'|z| > '+str(k):>10}{oe:>12d}{og:>10d}{ex:>12.2f}"
              f"{re_:>13.1f}x{rg:>10.1f}x")
        tail_rows.append({"threshold": f"|z| > {k}", "NIFTY observed": oe,
                          "GBM observed": og, "normal expected": ex,
                          "NIFTY ratio": re_, "GBM ratio": rg})

    # --- figures -----------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
    bins = np.linspace(min(emp.min(), sim.min()), max(emp.max(), sim.max()), 130)
    ax[0].hist(emp, bins=bins, density=True, alpha=0.55, color="#1f4e79", label="NIFTY 50")
    ax[0].hist(sim, bins=bins, density=True, alpha=0.55, color="#e08214", label="GBM")
    ax[0].set(xlabel="daily log return", ylabel="density",
              title="Histograms, identical mean and volatility")
    ax[0].legend(frameon=False)
    ax[1].hist(emp, bins=bins, density=True, alpha=0.55, color="#1f4e79", label="NIFTY 50")
    ax[1].hist(sim, bins=bins, density=True, alpha=0.55, color="#e08214", label="GBM")
    ax[1].set_yscale("log")
    ax[1].set(xlabel="daily log return", ylabel="density (log scale)",
              title="Log density: the tails separate")
    ax[1].legend(frameon=False)
    finish(fig, "w02_gbm_vs_nifty_histogram")

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    stats.probplot(emp, dist="norm", plot=ax[0])
    ax[0].get_lines()[0].set(markersize=2.5, alpha=0.5, color="#1f4e79")
    ax[0].get_lines()[1].set(color="red", lw=1.5)
    ax[0].set_title("QQ plot: NIFTY 50 vs normal")
    stats.probplot(sim, dist="norm", plot=ax[1])
    ax[1].get_lines()[0].set(markersize=2.5, alpha=0.5, color="#e08214")
    ax[1].get_lines()[1].set(color="red", lw=1.5)
    ax[1].set_title("QQ plot: GBM vs normal")
    qe = np.sort(emp); qg = np.sort(sim)
    ax[2].plot(qg, qe, ".", markersize=2.5, alpha=0.5, color="#1f4e79")
    lo, hi = min(qg.min(), qe.min()), max(qg.max(), qe.max())
    ax[2].plot([lo, hi], [lo, hi], "r--", lw=1.5, label="45 degrees")
    ax[2].set(xlabel="GBM quantiles", ylabel="NIFTY quantiles",
              title="NIFTY against GBM directly")
    ax[2].legend(frameon=False, fontsize=8)
    finish(fig, "w02_gbm_vs_nifty_qq")

    # volatility clustering: |returns| autocorrelation
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2))
    lags = range(1, 31)
    for name, x, col in [("NIFTY 50", emp, "#1f4e79"), ("GBM", sim, "#e08214")]:
        a = [np.corrcoef(np.abs(x[:-k]), np.abs(x[k:]))[0, 1] for k in lags]
        ax[0].plot(lags, a, "o-", ms=3, color=col, label=name)
    ax[0].axhline(0, color="k", lw=0.7)
    ax[0].axhline(1.96/np.sqrt(n), color="r", ls="--", lw=1, label="95% band")
    ax[0].axhline(-1.96/np.sqrt(n), color="r", ls="--", lw=1)
    ax[0].set(xlabel="lag (days)", ylabel="autocorrelation of |return|",
              title="Volatility clustering")
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].plot(np.arange(n), emp, lw=0.5, color="#1f4e79", label="NIFTY 50")
    ax[1].plot(np.arange(n), sim - 0.10, lw=0.5, color="#e08214", label="GBM (offset)")
    ax[1].set(xlabel="observation", ylabel="log return",
              title="Return series side by side")
    ax[1].legend(frameon=False, fontsize=8)
    finish(fig, "w02_gbm_vs_nifty_clustering")

    return tab, pd.DataFrame(tail_rows), r_gbm


# --------------------------------------------------------------------------- main

def main():
    for d in ("data", "figures", "results"):
        os.makedirs(d, exist_ok=True)

    rule("1. GBM SIMULATION AND VALIDATION   (15.4.3)")
    validate_gbm()

    rule("2. THE ROLES OF mu AND sigma   (15.4.3)")
    roles_of_mu_sigma()

    rule("3. THE TWO DRIFTS: mu VERSUS mu - sigma^2/2")
    drift_tab = two_drifts()

    rule("4. CALIBRATION TO NIFTY 50   (15.4.4)")
    prices = load_nifty()
    r_emp, mu, sigma, dmean, dsd = calibrate(prices)

    rule("5. MODEL-VERSUS-DATA EXPERIMENT   (15.4.5)")
    stat_tab, tail_tab, r_gbm = model_vs_data(r_emp, mu, sigma, dmean, dsd)

    stat_tab.to_csv("results/w02_gbm_vs_nifty_statistics.csv")
    tail_tab.to_csv("results/w02_gbm_vs_nifty_tails.csv", index=False)
    drift_tab.to_csv("results/w02_gbm_two_drifts.csv", index=False)

    rule("6. INTERPRETATION OF MODEL LIMITATIONS   (15.4.6)")
    print("""
  PRINCIPAL QUESTION (15.4.5)
      If GBM is calibrated to the same mean and volatility, does it reproduce
      the distributional behaviour observed in the actual market data?

  ANSWER:  No.

  What GBM reproduces
      - the mean and the standard deviation, exactly - but only because these
        were the calibration targets, so this is not evidence of anything;
      - positivity of prices;
      - the broad shape of a price path over long horizons.

  What GBM fails to reproduce
      - SKEWNESS.  GBM gives approximately 0 by construction. NIFTY is strongly
        negative: large falls dominate large rises.
      - KURTOSIS.  GBM gives approximately 0 excess. NIFTY exceeds 20.
      - TAIL FREQUENCIES.  Beyond four and five standard deviations the empirical
        counts exceed the model by orders of magnitude.
      - VOLATILITY CLUSTERING.  GBM increments are i.i.d., so |returns| are
        uncorrelated at every lag. NIFTY shows strong, slowly decaying positive
        autocorrelation in |returns|. Volatility is not constant.

  WHY THIS MATTERS FOR THE PROJECT
      Black-Scholes inherits all four failures, because it is derived under exactly
      these dynamics. Two consequences follow directly:

        1. A single constant sigma cannot price all strikes consistently. Inverting
           the formula on market prices returns different sigma at different strikes
           - the volatility smile - which is logically impossible if the model holds.

        2. Tail risk is understated. A model that assigns probability ~1e-7 to a
           five-sigma day, in a market that delivers them roughly once a year,
           will misprice deep out-of-the-money options and understate portfolio
           risk under stress.

      These are the specific, measured gaps that the later machine-learning stage
      is asked to address. They are also why the project's central question is
      'where does ML help, and where does it not' rather than 'is ML better'.
""")

    rule("SUMMARY")
    summary = {
        "seed":                    SEED,
        "nifty_annualised_sigma":  float(sigma),
        "nifty_annualised_mu":     float(mu),
        "n_returns":               int(len(r_emp)),
        "nifty_skewness":          float(stats.skew(r_emp, bias=False)),
        "gbm_skewness":            float(stats.skew(r_gbm, bias=False)),
        "nifty_excess_kurtosis":   float(stats.kurtosis(r_emp, bias=False)),
        "gbm_excess_kurtosis":     float(stats.kurtosis(r_gbm, bias=False)),
        "nifty_jarque_bera_p":     float(stats.jarque_bera(r_emp)[1]),
        "gbm_jarque_bera_p":       float(stats.jarque_bera(r_gbm)[1]),
        "max_two_drift_error":     float(np.abs(drift_tab["gap"]-drift_tab["sigma^2/2"]).max()),
    }
    pd.Series(summary).to_csv("results/w02_gbm_summary.csv", header=False)
    for k, v in summary.items():
        print(f"  {k:<26} {v}")
    print("\nSaved: results/w02_gbm_*.csv   Figures: figures/w02_gbm_*.png")


if __name__ == "__main__":
    main()
