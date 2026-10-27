"""
Week 1 - Financial Returns, Probability and Data Handling
=========================================================

Project : AI/ML-Augmented Black-Scholes Framework for Option Pricing and Market-Risk Analysis
Ref     : Work plan section 11.1 | Mathematics section 3.1
Text    : Carol Alexander, Market Risk Analysis Vol. I, sections I.3.1-I.3.3 and I.3.7.3
Asset   : NIFTY 50  (Yahoo Finance ticker ^NSEI)

Mathematical objective
----------------------
Understand why log returns are useful, and verify numerically that

        r_t ~= R_t        for sufficiently small returns.

Computational tasks (section 11.1)
----------------------------------
    1. import a sample asset-price time series
    2. compute simple returns
    3. compute log returns
    4. calculate mean, variance, skewness and kurtosis
    5. plot the price series
    6. plot the return series
    7. plot a return histogram

Usage
-----
    pip install numpy pandas scipy matplotlib yfinance
    python week01_returns_analysis.py

Outputs
-------
    data/nifty50_prices.csv                  cached price history
    figures/w01_*.png                        five figures
    results/w01_summary.csv                  key numbers
    results/w01_descriptive_statistics.csv   full statistics table
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

# --------------------------------------------------------------------------- config

TICKER   = "^NSEI"              # NIFTY 50
START    = "2015-01-01"
END      = "2025-01-01"
CSV_PATH = "data/nifty50_prices.csv"

SEED       = 20260825           # recorded for reproducibility (Deliverable 18)
SHOW_PLOTS = True               # set False to save figures without opening windows

rng = np.random.default_rng(SEED)
DATA_SOURCE = None              # "csv" | "yfinance" | "SIMULATED"

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
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


# ------------------------------------------------------- 1. price series (task 1)

def load_prices(ticker=TICKER, start=START, end=END, csv_path=CSV_PATH):
    """Return a pd.Series of closing prices indexed by date.

    Tries a cached CSV, then yfinance, then a clearly-flagged simulated series so
    the script always runs. Sets the module-level DATA_SOURCE flag.
    """
    global DATA_SOURCE

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        DATA_SOURCE = "csv"
        print(f"Loaded {len(df)} rows from cache: {csv_path}")
        return df.iloc[:, 0].dropna().astype(float)

    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if len(df) > 0:
            px = df["Close"]
            if isinstance(px, pd.DataFrame):          # yfinance >= 0.2.51 MultiIndex
                px = px.iloc[:, 0]
            px = px.dropna().astype(float)
            px.to_frame("Close").to_csv(csv_path)
            DATA_SOURCE = "yfinance"
            print(f"Downloaded {len(px)} rows for {ticker}; cached to {csv_path}")
            return px
        print("yfinance returned no rows.")
    except Exception as e:
        print(f"yfinance unavailable ({type(e).__name__}: {e}).")

    print("\n" + "!" * 78)
    print("!! FALLING BACK TO SIMULATED DATA - results below are NOT real market data.")
    print("!! Re-run with network access, or place a CSV at", csv_path)
    print("!" * 78)
    n = len(pd.bdate_range(start, end))
    mu, sigma, dt = 0.10, 0.17, 1 / 252
    log_path = np.cumsum((mu - 0.5 * sigma ** 2) * dt
                         + sigma * np.sqrt(dt) * rng.standard_normal(n))
    DATA_SOURCE = "SIMULATED"
    return pd.Series(8000 * np.exp(log_path), index=pd.bdate_range(start, end), name="Close")


def check_prices(prices):
    assert prices.notna().all(),                 "missing values in price series"
    assert (prices > 0).all(),                   "non-positive prices - log undefined"
    assert prices.index.is_monotonic_increasing, "index not sorted"
    assert not prices.index.duplicated().any(),  "duplicate dates"
    print("Price series checks passed.")
    print(f"  period       {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"  observations {len(prices)}")
    print(f"  min / max    {prices.min():,.2f} / {prices.max():,.2f}")


# ----------------------------------------------------- 2-3. returns (tasks 2, 3)

def compute_returns(prices):
    """Simple returns R_t and log returns r_t, with the exact identity verified."""
    R = prices.pct_change().dropna()
    r = np.log(prices / prices.shift(1)).dropna()

    assert np.allclose(R.values, (prices / prices.shift(1) - 1).dropna().values), \
        "pct_change disagrees with the definition"
    assert np.allclose(r.values, np.log1p(R.values)), "r = ln(1+R) violated"

    print("Exact identity  r_t = ln(1 + R_t)  verified to machine precision.")
    print(f"  simple returns: {len(R)} obs, range [{R.min():.6f}, {R.max():.6f}]")
    print(f"  log returns   : {len(r)} obs, range [{r.min():.6f}, {r.max():.6f}]")
    return R, r


# --------------------------------------------- 4. verify r ~= R (math objective)

def verify_approximation(R, r):
    """Confirm r - R = O(R^2) by fitting log|r-R| = a + b log|R| and checking b ~ 2."""
    diff = r - R

    print("Difference  r_t - R_t")
    print(f"  mean abs   {diff.abs().mean():.3e}")
    print(f"  max abs    {diff.abs().max():.3e}"
          f"   (on a return of {R.loc[diff.abs().idxmax()]:.4f})")

    rows = []
    for lo, hi in [(0.000, 0.005), (0.005, 0.010), (0.010, 0.020),
                   (0.020, 0.050), (0.050, 1.000)]:
        m = (R.abs() >= lo) & (R.abs() < hi)
        if m.sum() == 0:
            continue
        rows.append({"|R| band": f"{lo:.3f}-{hi:.3f}",
                     "n": int(m.sum()),
                     "mean |r-R|": diff[m].abs().mean(),
                     "rel err %": 100 * (diff[m].abs() / R[m].abs()).mean()})
    print("\nApproximation quality by return size")
    print(pd.DataFrame(rows).to_string(index=False))

    m = (R.abs() > 1e-5) & (diff.abs() > 1e-12)
    x, y = np.log(R[m].abs()), np.log(diff[m].abs())
    slope, intercept, rval, _, stderr = stats.linregress(x, y)

    print(f"\nScaling test:  log|r-R| = a + b log|R|")
    print(f"  slope b     {slope:.4f}  +/- {stderr:.4f}      (theory: 2)")
    print(f"  intercept a {intercept:.4f}                  (theory: ln(1/2) = {np.log(0.5):.4f})")
    print(f"  R^2         {rval**2:.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].scatter(R.abs(), diff.abs(), s=3, alpha=0.3, color="#1f4e79")
    grid = np.linspace(1e-4, R.abs().max(), 300)
    ax[0].plot(grid, 0.5 * grid ** 2, "r--", lw=1.6, label=r"theory: $|r-R| = R^2/2$")
    ax[0].set(xlabel="|R|", ylabel="|r - R|", title="Approximation error vs return size")
    ax[0].legend(frameon=False)
    ax[1].scatter(x, y, s=3, alpha=0.3, color="#1f4e79")
    ax[1].plot(x, intercept + slope * x, "r--", lw=1.6, label=f"fit, slope = {slope:.3f}")
    ax[1].set(xlabel=r"$\ln|R|$", ylabel=r"$\ln|r-R|$",
              title="Log-log: slope confirms second-order error")
    ax[1].legend(frameon=False)
    finish(fig, "w01_approximation_error")

    return slope, diff


# ------------------------------------------------------------- 5. time additivity

def verify_additivity(prices, R, r):
    """Log returns sum across time; simple returns do not."""
    total_log     = np.log(prices.iloc[-1] / prices.iloc[0])
    sum_of_logs   = r.sum()
    total_simple  = prices.iloc[-1] / prices.iloc[0] - 1
    prod_simple   = (1 + R).prod() - 1
    sum_simple    = R.sum()

    print("LOG RETURNS - additive")
    print(f"  ln(S_T / S_0)     {total_log:.8f}")
    print(f"  sum of r_t        {sum_of_logs:.8f}")
    print(f"  difference        {abs(total_log - sum_of_logs):.2e}   <- machine precision")
    print("\nSIMPLE RETURNS - multiplicative, NOT additive")
    print(f"  S_T / S_0 - 1     {total_simple:.8f}")
    print(f"  prod(1+R_t) - 1   {prod_simple:.8f}   <- correct")
    print(f"  sum of R_t        {sum_simple:.8f}   <- WRONG, off by "
          f"{abs(total_simple - sum_simple):.4f}")

    assert abs(total_log - sum_of_logs) < 1e-10
    assert abs(total_simple - prod_simple) < 1e-10


# ------------------------------------------------------- 6. statistics (task 4)

def describe(x, name, periods_per_year=252):
    x = np.asarray(x, dtype=float)
    mean, var = x.mean(), x.var(ddof=1)
    sd = np.sqrt(var)
    return {
        "series":            name,
        "n":                 len(x),
        "mean":              mean,
        "variance":          var,
        "std dev":           sd,
        "skewness":          stats.skew(x, bias=False),
        "excess kurtosis":   stats.kurtosis(x, fisher=True,  bias=False),
        "raw kurtosis":      stats.kurtosis(x, fisher=False, bias=False),
        "min":               x.min(),
        "max":               x.max(),
        "mean (annualised)": mean * periods_per_year,
        "vol (annualised)":  sd * np.sqrt(periods_per_year),
    }


def descriptive_statistics(R, r):
    """Mean, variance, skewness, kurtosis for both return definitions.

    Note: scipy.stats.kurtosis returns EXCESS kurtosis by default (normal -> 0).
    Raw (Pearson) kurtosis is reported alongside it (normal -> 3).
    """
    table = pd.DataFrame([describe(R, "simple return R_t"),
                          describe(r, "log return r_t")]).set_index("series").T
    table.to_csv("results/w01_descriptive_statistics.csv")
    print(table.to_string())

    gap = R.mean() - r.mean()
    print(f"\n  mean(R) - mean(r) = {gap:.8f}")
    print(f"  sigma^2 / 2       = {r.var(ddof=1) / 2:.8f}   <- the GBM correction term,")
    print( "                                        visible in the data before any model.")
    return table


# ------------------------------------------------------- 7. plots (tasks 5, 6, 7)

def plot_price_series(prices, note):
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    ax[0].plot(prices.index, prices.values, lw=0.9, color="#1f4e79")
    ax[0].set(ylabel="Index level", title=f"NIFTY 50 price series - {note}")
    ax[1].semilogy(prices.index, prices.values, lw=0.9, color="#c0504d")
    ax[1].set(ylabel="Index level (log scale)", xlabel="Date")
    ax[1].set_title("Log scale - a straight line means constant growth rate", fontsize=9)
    finish(fig, "w01_price_series")


def plot_return_series(R, r):
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, sharey=True)
    ax[0].plot(R.index, R.values, lw=0.5, color="#1f4e79")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set(ylabel="$R_t$", title="Daily simple returns")
    ax[1].plot(r.index, r.values, lw=0.5, color="#c0504d")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set(ylabel="$r_t$", xlabel="Date", title="Daily log returns")
    finish(fig, "w01_return_series")
    print("Volatility clustering is visible: large moves follow large moves.")
    print("Black-Scholes will assume volatility is constant. It plainly is not.")


def plot_return_histogram(r):
    mu_hat, sd_hat = r.mean(), r.std(ddof=1)
    grid = np.linspace(r.min(), r.max(), 600)
    pdf = stats.norm.pdf(grid, mu_hat, sd_hat)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    ax[0].hist(r, bins=120, density=True, alpha=0.65, color="#1f4e79", label="log returns")
    ax[0].plot(grid, pdf, "r-", lw=1.8, label=f"N({mu_hat:.5f}, {sd_hat:.5f}$^2$)")
    ax[0].set(xlabel="$r_t$", ylabel="density", title="Log-return histogram vs fitted normal")
    ax[0].legend(frameon=False)
    ax[1].hist(r, bins=120, density=True, alpha=0.65, color="#1f4e79")
    ax[1].plot(grid, pdf, "r-", lw=1.8)
    ax[1].set_yscale("log")
    ax[1].set(xlabel="$r_t$", ylabel="density (log scale)",
              title="Same, log density - the tails become visible")
    finish(fig, "w01_return_histogram")


# ---------------------------------------------------------------- 8. normality

def normality_analysis(r):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    stats.probplot(r, dist="norm", plot=ax[0])
    ax[0].get_lines()[0].set(markersize=2.5, alpha=0.5, color="#1f4e79")
    ax[0].get_lines()[1].set(color="red", lw=1.5)
    ax[0].set_title("QQ plot of log returns vs normal")

    normal_sample = rng.normal(r.mean(), r.std(ddof=1), size=len(r))
    ax[1].hist(r, bins=100, density=True, alpha=0.55, color="#1f4e79",
               label="empirical log returns")
    ax[1].hist(normal_sample, bins=100, density=True, alpha=0.55, color="#e08214",
               label="simulated normal, same $\\mu,\\sigma$")
    ax[1].set_yscale("log")
    ax[1].set(xlabel="return", ylabel="density (log)",
              title="Empirical vs normal, matched moments")
    ax[1].legend(frameon=False, fontsize=8)
    finish(fig, "w01_normality")

    jb_stat, jb_p = stats.jarque_bera(r)
    sw_stat, sw_p = stats.shapiro(r[:5000])

    print(f"  Jarque-Bera statistic  {jb_stat:>14,.2f}")
    print(f"  Jarque-Bera p-value    {jb_p:>14.3e}")
    print(f"  Shapiro-Wilk W         {sw_stat:>14.6f}   (first 5000 obs)")
    print(f"  Shapiro-Wilk p-value   {sw_p:>14.3e}")
    print(f"  skewness               {stats.skew(r, bias=False):>14.4f}   (normal: 0)")
    print(f"  excess kurtosis        {stats.kurtosis(r, bias=False):>14.4f}   (normal: 0)")
    print(f"\n  Normality at the 1% level: "
          f"{'REJECTED' if jb_p < 0.01 else 'not rejected'}")

    z = (r - r.mean()) / r.std(ddof=1)
    print("\n  TAIL EVENT FREQUENCIES")
    print(f"  {'threshold':>10} {'observed':>10} {'normal':>10} {'ratio':>8}")
    for k in (2, 3, 4, 5):
        obs = int((z.abs() > k).sum())
        exp = 2 * stats.norm.sf(k) * len(z)
        print(f"  {'|z| > ' + str(k):>10} {obs:>10d} {exp:>10.2f} "
              f"{obs / exp if exp > 0 else float('inf'):>8.1f}x")
    return jb_p


# --------------------------------------------------------------------- main

def main():
    for d in ("data", "figures", "results"):
        os.makedirs(d, exist_ok=True)

    rule("1. PRICE SERIES  (task 1)")
    prices = load_prices()
    prices.name = "Close"
    check_prices(prices)

    rule("2-3. SIMPLE AND LOG RETURNS  (tasks 2, 3)")
    R, r = compute_returns(prices)

    rule("4. MATHEMATICAL OBJECTIVE - verify  r_t ~= R_t")
    slope, diff = verify_approximation(R, r)

    rule("5. TIME-ADDITIVITY OF LOG RETURNS")
    verify_additivity(prices, R, r)

    rule("6. DESCRIPTIVE STATISTICS  (task 4)")
    descriptive_statistics(R, r)

    rule("7. MANDATORY PLOTS  (tasks 5, 6, 7)")
    note = ("SIMULATED DATA - not real market prices" if DATA_SOURCE == "SIMULATED"
            else f"{TICKER}  ({prices.index[0].date()} to {prices.index[-1].date()})")
    plot_price_series(prices, note)
    plot_return_series(R, r)
    plot_return_histogram(r)
    print("Saved: figures/w01_price_series.png, w01_return_series.png, "
          "w01_return_histogram.png")

    rule("8. NORMALITY OF DAILY LOG RETURNS")
    jb_p = normality_analysis(r)

    rule("WEEK 1 SUMMARY")
    summary = {
        "asset":                TICKER,
        "data_source":          DATA_SOURCE,
        "start":                str(prices.index[0].date()),
        "end":                  str(prices.index[-1].date()),
        "n_observations":       len(prices),
        "n_returns":            len(r),
        "seed":                 SEED,
        "mean_log_return":      float(r.mean()),
        "std_log_return":       float(r.std(ddof=1)),
        "annualised_vol":       float(r.std(ddof=1) * np.sqrt(252)),
        "skewness":             float(stats.skew(r, bias=False)),
        "excess_kurtosis":      float(stats.kurtosis(r, bias=False)),
        "jarque_bera_p":        float(jb_p),
        "max_abs_approx_error": float(diff.abs().max()),
        "approx_error_slope":   float(slope),
    }
    pd.Series(summary).to_csv("results/w01_summary.csv", header=False)
    for k, v in summary.items():
        print(f"  {k:<24} {v}")

    if DATA_SOURCE == "SIMULATED":
        print("\n*** WARNING: simulated data. Re-run with a real price series. ***")
    else:
        print("\nSaved: results/w01_summary.csv, results/w01_descriptive_statistics.csv")
        print("Figures in figures/")


if __name__ == "__main__":
    main()