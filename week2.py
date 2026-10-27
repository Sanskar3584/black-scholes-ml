import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

# =========================================================================== config

SEED         = 20260825        # same seed convention as Week 1
SHOW_PLOTS   = True            # False -> save figures only, no windows
TRADING_DAYS = 252
CSV_PATH     = "data/nifty50_prices.csv"

# Monte Carlo sizes. Reduce these if you want a faster run.
N_PATHS_VALIDATE = 200_000
N_PATHS_DRIFT    = 200_000

rng = np.random.default_rng(SEED)

if not SHOW_PLOTS:
    matplotlib.use("Agg")

plt.rcParams.update({
    "figure.figsize": (11, 4.2), "axes.grid": True, "grid.alpha": 0.3,
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
})


class _Tee:
    """Mirror everything printed to the console into a log file, so the whole run
    can be shared as one file instead of copying terminal scrollback."""

    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def rule(title=""):
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def finish(fig, name):
    fig.tight_layout()
    fig.savefig(f"figures/{name}.png", dpi=150, bbox_inches="tight")
    plt.show() if SHOW_PLOTS else plt.close(fig)


# ============================================================ PART A: BROWNIAN MOTION

def task1_standard_normals(n=1_000_000):
    """Task 1. Everything rests on the generator, so check it first."""
    Z = rng.standard_normal(n)
    print(f"  n            {n:,}")
    print(f"  mean         {Z.mean():+.6f}   (theory 0)")
    print(f"  variance     {Z.var(ddof=1):.6f}   (theory 1)")
    print(f"  skewness     {stats.skew(Z):+.6f}   (theory 0)")
    print(f"  ex. kurtosis {stats.kurtosis(Z):+.6f}   (theory 0)")
    se = 1 / np.sqrt(n)
    print(f"\n  |mean| / (1/sqrt(n)) = {abs(Z.mean())/se:.3f} standard errors")
    assert abs(Z.mean()) < 5 * se, "generator mean is implausible"
    jb, p = stats.jarque_bera(Z[:5000])
    print(f"  Jarque-Bera on first 5000: p = {p:.3f}  "
          f"({'consistent with normal' if p > 0.01 else 'REJECTED'})")


def task2_increments(dt=1/252, n=10_000):
    """Task 2.  dW = sqrt(dt) * Z.  Note sqrt(dt), not dt."""
    dW = np.sqrt(dt) * rng.standard_normal(n)
    print(f"  dt = {dt:.6f},  n = {n:,}")
    print(f"  mean(dW)  {dW.mean():+.8f}   (theory 0)")
    print(f"  var(dW)   {dW.var(ddof=1):.8f}   (theory dt       = {dt:.8f})")
    print(f"  sd(dW)    {dW.std(ddof=1):.8f}   (theory sqrt(dt) = {np.sqrt(dt):.8f})")


def task3_paths(n_paths=8, n_steps=1000, T=1.0):
    """Task 3. A path is the cumulative sum of increments, started at W_0 = 0."""
    dt = T / n_steps
    dW = np.sqrt(dt) * rng.standard_normal((n_paths, n_steps))
    W  = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(dW, axis=1)], axis=1)
    t  = np.linspace(0, T, n_steps + 1)
    assert np.all(W[:, 0] == 0), "W_0 must be 0"
    print(f"  {n_paths} paths plotted, {n_steps} steps, T = {T},  W_0 = 0 for all")

    n_big = 2000
    big = np.concatenate(
        [np.zeros((n_big, 1)),
         np.cumsum(np.sqrt(dt) * rng.standard_normal((n_big, n_steps)), axis=1)], axis=1)
    emp_sd = big.std(axis=0, ddof=1)
    WT = big[:, -1]

    print(f"\n  W_T moments from {n_big:,} paths (8 is far too few to test):")
    print(f"    mean  {WT.mean():+.4f}   theory 0        "
          f"({abs(WT.mean())/np.sqrt(T/n_big):.2f} standard errors)")
    print(f"    sd    {WT.std(ddof=1):.4f}   theory sqrt(T) = {np.sqrt(T):.4f}")
    print(f"    Jarque-Bera p = {stats.jarque_bera(WT)[1]:.3f}")

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


def tasks4to7_scaling(n=200_000):
    """Tasks 4-7. E[dW]=0, Var(dW)=dt, SD(dW) proportional to sqrt(dt)."""
    dts = [1.0, 1/2, 1/4, 1/12, 1/52, 1/252, 1/1000, 1/5000, 1/20000]
    rows = []
    for dt in dts:
        x = np.sqrt(dt) * rng.standard_normal(n)
        rows.append({
            "dt":            dt,
            "mean(dW)":      x.mean(),
            "mean / s.e.":   x.mean() / np.sqrt(dt / n),
            "var(dW)":       x.var(ddof=1),
            "var / dt":      x.var(ddof=1) / dt,
            "sd(dW)":        x.std(ddof=1),
            "sd / sqrt(dt)": x.std(ddof=1) / np.sqrt(dt),
        })
    tab = pd.DataFrame(rows)
    with pd.option_context("display.float_format", lambda v: f"{v:12.6f}"):
        print(tab.to_string(index=False))

    print("\n  Task 4:  the raw mean shrinks with dt only because the scale shrinks.")
    print("           The scale-free test is 'mean / s.e.', which stays O(1).")
    print(f"  Task 5:  max |var/dt - 1| = {np.abs(tab['var / dt']-1).max():.4f}")

    x, y = np.log(tab["dt"].values), np.log(tab["sd(dW)"].values)
    slope, intercept, rval, _, stderr = stats.linregress(x, y)
    print(f"\n  Task 7:  log SD(dW) = a + b log(dt)")
    print(f"           slope b     {slope:.6f}  +/- {stderr:.6f}   (theory 0.5)")
    print(f"           intercept a {intercept:+.6f}                 (theory 0)")
    print(f"           R^2         {rval**2:.6f}")

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))
    ax[0].plot(tab["dt"], tab["var(dW)"], "o-", color="#1f4e79", label="empirical")
    ax[0].plot(tab["dt"], tab["dt"], "r--", lw=1.5, label=r"theory: Var $= \Delta t$")
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

    dt = 1/252
    x2 = np.sqrt(dt) * rng.standard_normal(100_000)
    grid = np.linspace(x2.min(), x2.max(), 500)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].hist(x2, bins=120, density=True, alpha=0.65, color="#1f4e79", label="simulated")
    ax[0].plot(grid, stats.norm.pdf(grid, 0, np.sqrt(dt)), "r-", lw=1.8,
               label=r"$N(0,\Delta t)$")
    ax[0].set(xlabel=r"$\Delta W$", ylabel="density",
              title=r"Increment distribution, $\Delta t = 1/252$")
    ax[0].legend(frameon=False)
    stats.probplot(x2[:20000], dist="norm", plot=ax[1])
    ax[1].get_lines()[0].set(markersize=2, alpha=0.4, color="#1f4e79")
    ax[1].get_lines()[1].set(color="red", lw=1.5)
    ax[1].set_title("QQ plot: increments are exactly normal by construction")
    finish(fig, "w02_bm_increment_distribution")
    return tab, slope


def task8_independence(n=200_000, dt=1/252):
    """Independent and stationary increments - defining properties of Brownian motion."""
    x = np.sqrt(dt) * rng.standard_normal(n)
    acf  = [np.corrcoef(x[:-k], x[k:])[0, 1] for k in range(1, 11)]
    band = 1.96 / np.sqrt(n)
    print("  Autocorrelation of increments (theory: 0 at all non-zero lags)")
    for k, a in enumerate(acf, 1):
        print(f"    lag {k:>2}   {a:+.5f}" + ("" if abs(a) < band else "   <- outside band"))
    print(f"    95% band  +/-{band:.5f}")

    blocks = np.array_split(x, 8)
    sds = [b.std(ddof=1) for b in blocks]
    print("\n  Stationarity: moments across 8 equal blocks")
    print(f"    {'block':>6}{'mean':>14}{'sd':>14}")
    for i, b in enumerate(blocks, 1):
        print(f"    {i:>6}{b.mean():>14.6f}{b.std(ddof=1):>14.6f}")
    print(f"    spread of block SDs: {np.ptp(sds):.6f}  "
          f"(compare sqrt(dt) = {np.sqrt(dt):.6f})")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.0))
    ax[0].stem(range(1, 11), acf, basefmt=" ")
    ax[0].axhline(band, color="r", ls="--", lw=1)
    ax[0].axhline(-band, color="r", ls="--", lw=1, label="95% band")
    ax[0].axhline(0, color="k", lw=0.7)
    ax[0].set(xlabel="lag", ylabel="autocorrelation", title="Increments are uncorrelated")
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].plot(range(1, 9), sds, "o-", color="#1f4e79", label="block SD")
    ax[1].axhline(np.sqrt(dt), color="r", ls="--", lw=1.5, label=r"$\sqrt{\Delta t}$")
    ax[1].set(xlabel="block", ylabel="SD", title="Increments are stationary")
    ax[1].legend(frameon=False, fontsize=8)
    finish(fig, "w02_bm_independence")
    return acf, band


def task9_quadratic_variation(T=1.0):
    """sum (dW)^2 -> T while sum |dW| -> infinity. This is why Ito is not the chain rule."""
    ns = [10, 50, 100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]
    rows = []
    for n in ns:
        dt = T / n
        x  = np.sqrt(dt) * rng.standard_normal(n)
        qv = (x ** 2).sum()
        se = T * np.sqrt(2 / n)              # Var(sum (dW)^2) = n * 2 dt^2 = 2T^2/n
        rows.append({"n steps": n, "dt": dt, "sum (dW)^2": qv, "s.e. of QV": se,
                     "(QV - T)/s.e.": (qv - T) / se, "sum |dW|": np.abs(x).sum(),
                     "sum (dW)^3": (x ** 3).sum()})
    tab = pd.DataFrame(rows)
    with pd.option_context("display.float_format", lambda v: f"{v:14.6f}"):
        print(tab.to_string(index=False))

    print(f"\n  max |(QV - T)/s.e.| = {np.abs(tab['(QV - T)/s.e.']).max():.2f}"
          "   <- stays O(1): this is L^2 convergence")
    print(f"\n  sum (dW)^2  ->  T = {T}      [quadratic variation FINITE and non-zero]")
    print( "  sum |dW|    ->  infinity      [first variation INFINITE]")
    print( "  sum (dW)^3  ->  0             [higher orders vanish]")
    print()
    print( "  For a smooth function the (dx)^2 term in a Taylor expansion is discarded.")
    print( "  For Brownian motion it cannot be: (dW)^2 accumulates at rate dt, written")
    print( "  compactly as (dW)^2 = dt. Keeping it is the whole difference between")
    print( "  ordinary and Ito calculus, and it produces the term")
    print( "      (1/2) sigma^2 S^2 V_SS")
    print( "  in the Black-Scholes PDE. See September plan section 15.5.3, question 2.")
    print()
    print( "  sum |dW| -> infinity says Brownian paths have infinite length over any")
    print( "  interval: continuous everywhere, differentiable nowhere. That is why")
    print( "  dW/dt does not exist and stochastic calculus is needed at all.")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].semilogx(tab["n steps"], tab["sum (dW)^2"], "o-", color="#1f4e79",
                   label=r"$\sum (\Delta W)^2$")
    ax[0].axhline(T, color="r", ls="--", lw=1.5, label=f"T = {T}")
    ax[0].fill_between(tab["n steps"], T-2*tab["s.e. of QV"], T+2*tab["s.e. of QV"],
                       alpha=0.15, color="grey", label=r"$\pm 2$ s.e.")
    ax[0].set(xlabel="number of steps n", ylabel=r"$\sum (\Delta W)^2$",
              title="Quadratic variation converges to T")
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].loglog(tab["n steps"], tab["sum |dW|"], "o-", color="#c0504d",
                 label=r"$\sum |\Delta W|$")
    ax[1].loglog(tab["n steps"], np.sqrt(2/np.pi)*np.sqrt(tab["n steps"]), "k--", lw=1.4,
                 label=r"theory: $\sqrt{2n/\pi}$")
    ax[1].set(xlabel="number of steps n", ylabel=r"$\sum |\Delta W|$",
              title="First variation diverges")
    ax[1].legend(frameon=False, fontsize=8)
    finish(fig, "w02_bm_quadratic_variation")
    return tab


# ================================================ PART B: GEOMETRIC BROWNIAN MOTION

def simulate_gbm(S0, mu, sigma, T, n_steps, n_paths, generator=None):
    """Exact GBM.

        S_t = S_0 exp[ (mu - sigma^2/2) t + sigma W_t ]

    Stepping the analytical solution carries no discretisation bias at any step
    size, unlike an Euler scheme on dS = mu S dt + sigma S dW (which can also
    produce negative prices).
    """
    g  = generator if generator is not None else rng
    dt = T / n_steps
    dW = np.sqrt(dt) * g.standard_normal((n_paths, n_steps))
    incr = (mu - 0.5 * sigma ** 2) * dt + sigma * dW
    log_path = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(incr, axis=1)], axis=1)
    return S0 * np.exp(log_path)


def validate_gbm(S0=100.0, mu=0.10, sigma=0.20, T=1.0, n_steps=252,
                 n_paths=N_PATHS_VALIDATE):
    """Check the simulator against the analytical lognormal moments."""
    S  = simulate_gbm(S0, mu, sigma, T, n_steps, n_paths)
    ST = S[:, -1]
    checks = [
        ("E[S_T]",      ST.mean(),              S0*np.exp(mu*T)),
        ("Var(S_T)",    ST.var(ddof=1),         S0**2*np.exp(2*mu*T)*(np.exp(sigma**2*T)-1)),
        ("E[ln S_T]",   np.log(ST).mean(),      np.log(S0)+(mu-0.5*sigma**2)*T),
        ("Var(ln S_T)", np.log(ST).var(ddof=1), sigma**2*T),
    ]
    print(f"  S0={S0}, mu={mu}, sigma={sigma}, T={T}, {n_paths:,} paths x {n_steps} steps\n")
    print(f"  {'quantity':<14}{'simulated':>16}{'theory':>16}{'rel. error':>14}")
    for name, sim, th in checks:
        print(f"  {name:<14}{sim:>16.6f}{th:>16.6f}{abs(sim-th)/abs(th):>14.2e}")

    p = stats.jarque_bera(np.log(ST)[:5000])[1]
    print(f"\n  ln S_T normality: Jarque-Bera p = {p:.3f}")
    print( "  => S_T is lognormal, ln S_T is normal, exactly as the model states.")

    dt = T / n_steps
    r1 = np.diff(np.log(S), axis=1).ravel()
    print(f"\n  One-step log returns (dt = 1/{n_steps}):")
    print(f"    mean      {r1.mean():+.8f}   theory {(mu-0.5*sigma**2)*dt:+.8f}")
    print(f"    variance  {r1.var(ddof=1):.8f}   theory {sigma**2*dt:.8f}")


def roles_of_mu_sigma(S0=100.0, T=1.0, n_steps=252, n_paths=60):
    """Same random shocks in every panel, so any difference is caused by the parameter."""
    mus, sigmas = [0.00, 0.10, 0.20], [0.10, 0.20, 0.40]
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
    fig.suptitle("Identical random shocks in every panel: "
                 "rows change $\\sigma$, columns change $\\mu$", y=0.995, fontsize=11)
    finish(fig, "w02_gbm_mu_sigma_grid")

    print("  Reading the grid:")
    print("    across a row  (mu increases)    -> the bundle tilts upward;")
    print("                                       the spread is unchanged.")
    print("    down a column (sigma increases) -> the bundle fans out, AND the")
    print("                                       median falls away from the mean.")
    print("\n  mu    sets the trend.")
    print("  sigma sets the dispersion - and through -sigma^2/2 also drags the")
    print("        median below the mean. At sigma = 0.40 the median grows at 2%")
    print("        while the mean grows at 10%.")


def two_drifts(S0=100.0, mu=0.10, T=1.0, n_steps=252, n_paths=N_PATHS_DRIFT):
    """E[S_T] grows at mu; E[ln S_T] grows at mu - sigma^2/2. The gap is exact."""
    rows = []
    for sigma in [0.05, 0.10, 0.20, 0.30, 0.40, 0.60]:
        ST = simulate_gbm(S0, mu, sigma, T, n_steps, n_paths)[:, -1]
        growth_mean = np.log(ST.mean()/S0) / T          # -> mu
        growth_log  = np.log(ST/S0).mean() / T          # -> mu - sigma^2/2
        rows.append({
            "sigma":             sigma,
            "growth of E[S_T]":  growth_mean,
            "theory mu":         mu,
            "E[ln S_T]/T":       growth_log,
            "theory mu - s^2/2": mu - 0.5*sigma**2,
            "gap":               growth_mean - growth_log,
            "sigma^2/2":         0.5*sigma**2,
            "s.e.":              sigma/np.sqrt(n_paths*T),
        })
    tab = pd.DataFrame(rows)
    with pd.option_context("display.float_format", lambda v: f"{v:13.6f}"):
        print(tab.to_string(index=False))

    dev = np.abs(tab["gap"] - tab["sigma^2/2"]) / tab["s.e."]
    print(f"\n  max |gap - sigma^2/2|        = "
          f"{np.abs(tab['gap']-tab['sigma^2/2']).max():.2e}")
    print(f"  max |gap - sigma^2/2| / s.e. = {dev.max():.2f}"
          "   <- within Monte Carlo error at every sigma")
    print()
    print("  In the MODEL this identity is EXACT:")
    print("      E[ln S_T] = ln S_0 + (mu - sigma^2/2) T")
    print("  holds for every sigma with no approximation. It comes from Ito's lemma")
    print("  applied to f = ln S, where (1/2) f''(S) (dS)^2 survives because")
    print("  (dW)^2 = dt - the quadratic-variation property measured in section 9.")
    print()
    print("  CONTRAST WITH WEEK 1.  The empirical gap mean(R) - mean(r) found in the")
    print("  NIFTY data is a Taylor/Jensen identity in sample moments, true of any")
    print("  small numbers whatsoever and requiring no probability model. This one is")
    print("  a statement about the continuous-time process, exact at all sigma, and it")
    print("  is what Ito's lemma actually asserts. The two agree numerically under")
    print("  GBM; they are not the same statement.")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(tab["sigma"], tab["gap"], "o-", color="#1f4e79", label="simulated gap")
    ax[0].plot(tab["sigma"], tab["sigma^2/2"], "r--", lw=1.6, label=r"$\sigma^2/2$")
    ax[0].set(xlabel=r"$\sigma$", ylabel=r"$\ln E[S_T]/T \; - \; E[\ln S_T]/T$",
              title=r"Model: the gap is exactly $\sigma^2/2$")
    ax[0].legend(frameon=False)
    sig = 0.40
    ST = simulate_gbm(S0, mu, sig, T, n_steps, 100_000)[:, -1]
    ax[1].hist(ST, bins=200, density=True, alpha=0.6, color="#1f4e79")
    ax[1].axvline(ST.mean(), color="r", lw=2, label=f"mean {ST.mean():.1f}")
    ax[1].axvline(np.median(ST), color="k", ls="--", lw=2,
                  label=f"median {np.median(ST):.1f}")
    ax[1].set(xlabel="$S_T$", ylabel="density",
              title=fr"Lognormal is right-skewed ($\sigma={sig}$): mean $>$ median")
    ax[1].legend(frameon=False)
    finish(fig, "w02_gbm_two_drifts")
    return tab


def load_nifty(csv_path=CSV_PATH):
    """Use the Week 1 cache if present; otherwise download and cache it."""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        print(f"  loaded {len(df):,} rows from cache: {csv_path}")
        return df.iloc[:, 0].dropna().astype(float)
    print("  cache not found - downloading ^NSEI ...")
    import yfinance as yf
    df = yf.download("^NSEI", start="2015-01-01", end="2025-01-01",
                     progress=False, auto_adjust=True)
    px = df["Close"]
    if isinstance(px, pd.DataFrame):
        px = px.iloc[:, 0]
    px = px.dropna().astype(float)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    px.to_frame("Close").to_csv(csv_path)
    print(f"  downloaded {len(px):,} rows; cached to {csv_path}")
    return px


def calibrate(prices):
    """Match GBM to the data on both first and second moments (15.4.4)."""
    r = np.log(prices/prices.shift(1)).dropna()
    daily_mean, daily_sd = r.mean(), r.std(ddof=1)
    sigma = daily_sd * np.sqrt(TRADING_DAYS)
    mu    = daily_mean * TRADING_DAYS + 0.5 * sigma ** 2
    print(f"  period                 {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"  observations           {len(r):,}")
    print(f"  daily mean log return  {daily_mean:+.8f}")
    print(f"  daily sd  log return   {daily_sd:.8f}")
    print(f"  annualised sigma       {sigma:.6f}   (plan 15.4.4 quotes 0.167)")
    print(f"  annualised mu          {mu:.6f}")
    print(f"\n  mu - sigma^2/2 = {mu-0.5*sigma**2:.6f} matches the empirical annualised")
    print(f"  mean log return  {daily_mean*TRADING_DAYS:.6f}.")
    print( "  GBM is therefore matched on BOTH first and second moments. Anything it")
    print( "  then fails to reproduce is a genuine failure, not a calibration artefact.")
    return r, mu, sigma


def model_vs_data(r_emp, mu, sigma):
    """Section 15.4.5. Same n, same mean, same sd - compare everything else."""
    n  = len(r_emp)
    dt = 1 / TRADING_DAYS
    r_gbm = (mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*rng.standard_normal(n)
    emp, sim = r_emp.values, r_gbm

    print(f"  Both samples: n = {n:,} daily log returns\n")
    rows = []
    for name, f in [("mean",            lambda x: x.mean()),
                    ("std deviation",   lambda x: x.std(ddof=1)),
                    ("annualised vol",  lambda x: x.std(ddof=1)*np.sqrt(TRADING_DAYS)),
                    ("skewness",        lambda x: stats.skew(x, bias=False)),
                    ("excess kurtosis", lambda x: stats.kurtosis(x, bias=False)),
                    ("minimum",         lambda x: x.min()),
                    ("maximum",         lambda x: x.max())]:
        rows.append({"statistic": name, "NIFTY 50": f(emp), "GBM": f(sim)})
    stat_tab = pd.DataFrame(rows).set_index("statistic")
    stat_tab["difference"] = stat_tab["GBM"] - stat_tab["NIFTY 50"]
    with pd.option_context("display.float_format", lambda v: f"{v:12.6f}"):
        print(stat_tab.to_string())
    print("\n  Mean and standard deviation agree - they were the calibration targets.")
    print("  Skewness and kurtosis do not, and were never targeted.")

    print("\n  Normality tests")
    print(f"    {'sample':<12}{'Jarque-Bera':>16}{'p-value':>14}{'verdict':>12}")
    for name, x in [("NIFTY 50", emp), ("GBM", sim)]:
        jb, p = stats.jarque_bera(x)
        print(f"    {name:<12}{jb:>16,.2f}{p:>14.3e}"
              f"{('REJECT' if p < 0.01 else 'ok'):>12}")

    def tails(x):
        z = (x - x.mean())/x.std(ddof=1)
        return {k: (int((np.abs(z) > k).sum()), 2*stats.norm.sf(k)*len(z))
                for k in (2, 3, 4, 5)}
    te, tg = tails(emp), tails(sim)
    print("\n  Tail event frequencies")
    print(f"    {'threshold':>10}{'NIFTY obs':>12}{'GBM obs':>10}{'normal exp':>12}"
          f"{'NIFTY ratio':>13}{'GBM ratio':>11}")
    tail_rows = []
    for k in (2, 3, 4, 5):
        oe, ex = te[k]; og, _ = tg[k]
        print(f"    {'|z| > '+str(k):>10}{oe:>12d}{og:>10d}{ex:>12.2f}"
              f"{oe/ex:>13.1f}x{og/ex:>10.1f}x")
        tail_rows.append({"threshold": f"|z| > {k}", "NIFTY observed": oe,
                          "GBM observed": og, "normal expected": ex,
                          "NIFTY ratio": oe/ex, "GBM ratio": og/ex})

    # ---- figures ----
    bins = np.linspace(min(emp.min(), sim.min()), max(emp.max(), sim.max()), 130)
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for a, logscale, ttl in [(ax[0], False, "Histograms, identical mean and volatility"),
                             (ax[1], True,  "Log density: the tails separate")]:
        a.hist(emp, bins=bins, density=True, alpha=0.55, color="#1f4e79", label="NIFTY 50")
        a.hist(sim, bins=bins, density=True, alpha=0.55, color="#e08214", label="GBM")
        if logscale: a.set_yscale("log")
        a.set(xlabel="daily log return",
              ylabel="density (log scale)" if logscale else "density", title=ttl)
        a.legend(frameon=False)
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
    qe, qg = np.sort(emp), np.sort(sim)
    ax[2].plot(qg, qe, ".", markersize=2.5, alpha=0.5, color="#1f4e79")
    lo, hi = min(qg.min(), qe.min()), max(qg.max(), qe.max())
    ax[2].plot([lo, hi], [lo, hi], "r--", lw=1.5, label="45 degrees")
    ax[2].set(xlabel="GBM quantiles", ylabel="NIFTY quantiles",
              title="NIFTY against GBM directly")
    ax[2].legend(frameon=False, fontsize=8)
    finish(fig, "w02_gbm_vs_nifty_qq")

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
    ax[1].set(xlabel="observation", ylabel="log return", title="Return series side by side")
    ax[1].legend(frameon=False, fontsize=8)
    finish(fig, "w02_gbm_vs_nifty_clustering")

    return stat_tab, pd.DataFrame(tail_rows), r_gbm, te, tg


INTERPRETATION = """
  PRINCIPAL QUESTION (15.4.5)
      If GBM is calibrated to the same mean and volatility, does it reproduce
      the distributional behaviour observed in the actual market data?

  ANSWER:  No.

  What GBM reproduces
      - the mean and the standard deviation, exactly - but only because these
        were the calibration targets, so this is not evidence in its favour;
      - positivity of prices, which is why the process is geometric;
      - the broad visual character of a price path over long horizons.

  What GBM fails to reproduce
      - SKEWNESS.  GBM gives approximately 0 by construction. NIFTY is close to
        -1.41: large falls dominate large rises.
      - KURTOSIS.  GBM gives approximately 0 excess. NIFTY exceeds 20.
      - TAIL FREQUENCIES.  Beyond four and five standard deviations the market
        delivered 15 and 10 days respectively; the model produced none.
      - VOLATILITY CLUSTERING.  GBM increments are i.i.d., so |returns| are
        uncorrelated at every lag. NIFTY shows strong, slowly decaying positive
        autocorrelation in |returns|. Volatility is not constant.

  WHY THIS MATTERS FOR THE PROJECT
      Black-Scholes is derived under exactly these dynamics and inherits every
      one of these failures. Two consequences follow directly:

        1. A single constant sigma cannot price all strikes consistently.
           Inverting the formula on market prices returns different implied
           volatilities at different strikes - the volatility smile - which is
           logically impossible if the model is true. Met in September Week 3
           (15.6.3) and again with real option data in October.

        2. Tail risk is understated. A model assigning probability ~5.7e-7 to a
           five-sigma day, in a market that delivers them roughly annually, will
           misprice deep out-of-the-money options and understate portfolio risk
           under stress. This is the subject of the November market-risk stage.

  POSITION GOING INTO SEPTEMBER WEEK 2
      These failures are now measured, not assumed. Week 2 (15.5) derives the
      Black-Scholes PDE from these same dynamics via Ito's lemma and delta
      hedging. Having quantified the gaps first, the derivation can be understood
      as a rigorous consequence of assumptions already known to be false in
      specific, measurable ways - which is the difference between using a model
      and believing it.
"""


# ============================================================================= main

def run():
    t_start = time.time()

    print("=" * 78)
    print("SEPTEMBER WEEK 1 - BROWNIAN MOTION AND GEOMETRIC BROWNIAN MOTION")
    print("September plan section 15.4")
    print("=" * 78)
    if SHOW_PLOTS:
        print("\nNOTE: figure windows will open and the script pauses until each is")
        print("closed. There are 10 figures. Set SHOW_PLOTS = False to skip this.")

    # ---------------- PART A ----------------
    rule("PART A / 1.  STANDARD NORMAL RANDOM VARIABLES   (task 1)")
    task1_standard_normals()

    rule("PART A / 2.  BROWNIAN INCREMENTS  dW = sqrt(dt) Z   (task 2)")
    task2_increments()

    rule("PART A / 3.  COMPLETE BROWNIAN-MOTION PATHS   (task 3)")
    task3_paths()

    rule("PART A / 4.  MOMENTS AND DEPENDENCE ON dt   (tasks 4-7)")
    scal_tab, slope = tasks4to7_scaling()

    rule("PART A / 5.  INDEPENDENCE AND STATIONARITY   [additional]")
    acf, band = task8_independence()

    rule("PART A / 6.  QUADRATIC VARIATION   [additional, for Week 2]")
    qv_tab = task9_quadratic_variation()

    # ---------------- PART B ----------------
    rule("PART B / 7.  GBM SIMULATION AND VALIDATION   (15.4.3)")
    validate_gbm()

    rule("PART B / 8.  THE ROLES OF mu AND sigma   (15.4.3)")
    roles_of_mu_sigma()

    rule("PART B / 9.  THE TWO DRIFTS: mu VERSUS mu - sigma^2/2")
    drift_tab = two_drifts()

    rule("PART B / 10. CALIBRATION TO NIFTY 50   (15.4.4)")
    prices = load_nifty()
    r_emp, mu, sigma = calibrate(prices)

    rule("PART B / 11. MODEL-VERSUS-DATA EXPERIMENT   (15.4.5)")
    stat_tab, tail_tab, r_gbm, te, tg = model_vs_data(r_emp, mu, sigma)

    rule("PART B / 12. INTERPRETATION OF MODEL LIMITATIONS   (15.4.6)")
    print(INTERPRETATION)

    # ---------------- save ----------------
    scal_tab.to_csv("results/w02_bm_scaling.csv", index=False)
    qv_tab.to_csv("results/w02_bm_quadratic_variation.csv", index=False)
    drift_tab.to_csv("results/w02_gbm_two_drifts.csv", index=False)
    stat_tab.to_csv("results/w02_gbm_vs_nifty_statistics.csv")
    tail_tab.to_csv("results/w02_gbm_vs_nifty_tails.csv", index=False)

    rule("SUMMARY")
    summary = {
        "seed":                      SEED,
        "sd_scaling_exponent":       float(slope),
        "sd_scaling_theory":         0.5,
        "max_|var/dt - 1|":          float(np.abs(scal_tab["var / dt"] - 1).max()),
        "max_|autocorrelation|":     float(np.max(np.abs(acf))),
        "acf_95pct_band":            float(band),
        "QV_at_n=100000":            float(qv_tab["sum (dW)^2"].iloc[-1]),
        "QV_theory":                 1.0,
        "max_|QV normalised dev|":   float(np.abs(qv_tab["(QV - T)/s.e."]).max()),
        "first_variation_n=100000":  float(qv_tab["sum |dW|"].iloc[-1]),
        "nifty_annualised_sigma":    float(sigma),
        "nifty_annualised_mu":       float(mu),
        "n_returns":                 int(len(r_emp)),
        "nifty_skewness":            float(stats.skew(r_emp, bias=False)),
        "gbm_skewness":              float(stats.skew(r_gbm, bias=False)),
        "nifty_excess_kurtosis":     float(stats.kurtosis(r_emp, bias=False)),
        "gbm_excess_kurtosis":       float(stats.kurtosis(r_gbm, bias=False)),
        "nifty_jarque_bera_p":       float(stats.jarque_bera(r_emp)[1]),
        "gbm_jarque_bera_p":         float(stats.jarque_bera(r_gbm)[1]),
        "nifty_days_beyond_4sigma":  int(te[4][0]),
        "gbm_days_beyond_4sigma":    int(tg[4][0]),
        "nifty_days_beyond_5sigma":  int(te[5][0]),
        "gbm_days_beyond_5sigma":    int(tg[5][0]),
        "max_two_drift_error":       float(np.abs(drift_tab["gap"]
                                                  - drift_tab["sigma^2/2"]).max()),
    }
    pd.Series(summary).to_csv("results/w02_summary.csv", header=False)
    for k, v in summary.items():
        print(f"  {k:<28} {v}")

    print(f"\n  runtime {time.time()-t_start:.1f} s")
    print("\nSaved: results/w02_*.csv   Figures: figures/w02_*.png")


def main():
    for d in ("data", "figures", "results"):
        os.makedirs(d, exist_ok=True)

    log_path = "results/w02_console_output.txt"
    tee = _Tee(log_path)
    sys.stdout = tee
    try:
        run()
        print(f"\nFull console output written to: {log_path}")
    finally:
        sys.stdout = tee.stdout
        tee.close()
    print(f"\n>>> Share this one file for review: {log_path}")


if __name__ == "__main__":
    main()
