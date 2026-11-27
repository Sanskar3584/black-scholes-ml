"""
build_real_dataset.py - clean quotes to a modelling dataset
============================================================

Project : AI/ML-Augmented Black-Scholes Framework
Ref     : master plan 11.9 data objective; ML_experiment_plan.md 4.1, 6.2

    python build_real_dataset.py

Reads `data/clean_options.csv` and produces `data/real_option_dataset.csv`
carrying, per master plan 11.9:

    S, K, T, r, C_market, m, sigma_hist, C_BS, epsilon_BS, sigma_IV

The volatility question, which is the whole difficulty
-------------------------------------------------------
Master plan 11.9 asks for both `sigma_IV` and `C_BS`, and its milestone
forbids using one to make the other:

    "The same option's implied volatility shall not be used to define the
     residual-learning benchmark."

So this file computes two different volatilities for two different purposes:

  sigma_IV        inverted from the option's own market price. A TARGET
                  quantity (Hybrid B learns it) and a diagnostic. Never a
                  feature, never the benchmark.

  sigma_bench     what C_BS is priced with. Must be computable without the
                  contract's own price. ML_experiment_plan.md 4.1 names two
                  admissible choices and this module implements the second,
                  `sigma_atm_prev`: the at-the-money implied volatility of the
                  SAME EXPIRY on the PREVIOUS trading date.

The first choice, `sigma_hist` (21-day realised), needs a daily NIFTY close
series covering the option window. The cached series ends 31 December 2024
and the options are November 2026, so it is written as blank and reported as
a blocker rather than silently filled.

Why sigma_atm_prev is legitimate and sigma_IV is not
-----------------------------------------------------
Both are implied volatilities, so the distinction is worth being precise
about. `sigma_atm_prev` is observable before the trading date it is used on,
is a single number per expiry rather than per contract, and does not involve
the contract being priced. A desk could quote with it at the open.
`sigma_IV` for a given contract cannot be known until that contract's price
is known - which is the thing being predicted.
"""

import os
import sys

import numpy as np
import pandas as pd

from black_scholes import bs_call
from implied_volatility import implied_vol
from clean_options import sigma_hist_status

__all__ = ["add_implied_vol", "atm_vol_by_expiry", "build"]

SRC = os.path.join("data", "clean_options.csv")
DEST = os.path.join("data", "real_option_dataset.csv")
ATM_BAND = 0.02          # |m| within which a quote counts as at the money


def add_implied_vol(df, verbose=True):
    """Invert each market price for its own implied volatility.

    Priced off the discounted forward with q = 0 rather than off spot with a
    dividend yield: the forward already contains the market's own dividend
    view (data_sources.md section 7), so going back to spot would double-count
    an assumption the forward has already made.
    """
    F_disc = (df["F_implied"] * np.exp(-df["r"] * df["T"])).to_numpy()
    out, method, unc = [], [], []
    for i, row in enumerate(df.itertuples()):
        try:
            s, info = implied_vol(row.C_market, F_disc[i], row.K, row.T,
                                  row.r, 0.0, option="call")
            out.append(float(s) if s is not None and np.isfinite(s) else np.nan)
            method.append(str(info.get("method", "")))
            unc.append(float(info.get("sigma_uncertainty", np.nan)))
        except Exception:
            out.append(np.nan)
            method.append("failed")
            unc.append(np.nan)
    df = df.copy()
    df["sigma_IV"] = out
    df["iv_method"] = method
    df["iv_uncertainty"] = unc
    if verbose:
        ok = np.isfinite(df["sigma_IV"])
        print("  implied volatility recovered on %d of %d rows (%.1f%%)"
              % (ok.sum(), len(df), 100 * ok.mean()))
        if ok.any():
            v = df.loc[ok, "sigma_IV"]
            print("    range %.1f%% to %.1f%%, median %.1f%%"
                  % (100 * v.min(), 100 * v.max(), 100 * v.median()))
        print("    solver: %s" % df["iv_method"].value_counts().to_dict())
    return df


def atm_vol_by_expiry(df):
    """At-the-money implied volatility per (trade_date, expiry).

    Averaged over the quotes inside the ATM band rather than taken from the
    single nearest strike, because one quote is one bid-ask spread away from
    its own true value and there are usually several inside the band.
    """
    a = df[np.abs(df["m"]) <= ATM_BAND]
    a = a[np.isfinite(a["sigma_IV"])]
    if len(a) == 0:
        return pd.DataFrame(columns=["trade_date", "expiry", "sigma_atm", "n"])
    g = (a.groupby(["trade_date", "expiry"])["sigma_IV"]
         .agg(sigma_atm="median", n="size").reset_index())
    return g


def build(src=SRC, dest=DEST, verbose=True):
    if not os.path.exists(src):
        raise SystemExit(
            "%s not found. Run clean_options.py first." % src)
    df = pd.read_csv(src)
    if verbose:
        print("=" * 78)
        print("BUILDING THE REAL-DATA MODELLING SET")
        print("=" * 78)
        print("  %d clean quotes, %d trading date(s), %d expiries"
              % (len(df), df["trade_date"].nunique(), df["expiry"].nunique()))

    df = add_implied_vol(df, verbose=verbose)

    # ---- the benchmark volatility: previous date, same expiry, at the money
    atm = atm_vol_by_expiry(df)
    dates = sorted(df["trade_date"].unique())
    prev = {d: dates[i - 1] for i, d in enumerate(dates) if i > 0}
    lookup = {(r.trade_date, r.expiry): r.sigma_atm for r in atm.itertuples()}

    df["sigma_atm_today"] = [lookup.get((d, e), np.nan)
                             for d, e in zip(df["trade_date"], df["expiry"])]
    df["sigma_bench"] = [lookup.get((prev.get(d), e), np.nan)
                         for d, e in zip(df["trade_date"], df["expiry"])]
    df["sigma_bench_source"] = np.where(
        np.isfinite(df["sigma_bench"]), "atm IV, previous trading date",
        "unavailable - no prior date for this expiry")

    # sigma_hist: 21-trading-day realised volatility of NIFTY log returns,
    # annualised. The primary benchmark volatility of ML_experiment_plan.md
    # 4.1, and the one immune to the sampling cadence - it comes from the
    # daily close series regardless of how the option files are sampled.
    st = sigma_hist_status(needed_from=pd.Timestamp(df["trade_date"].min()),
                           verbose=False)
    if st["ok"]:
        px = pd.read_csv(os.path.join("data", "nifty50_prices.csv"),
                         index_col=0, parse_dates=True).iloc[:, 0].dropna()
        lr = np.log(px / px.shift(1)).dropna()
        rv = lr.rolling(21).std() * np.sqrt(252)
        # shift(1): the volatility of the 21 days ENDING YESTERDAY, so nothing
        # from the trading date itself enters the feature.
        rv = rv.shift(1).dropna()
        key = pd.to_datetime(df["trade_date"])
        df["sigma_hist"] = key.map(rv).to_numpy()
        df["sigma_hist_status"] = np.where(np.isfinite(df["sigma_hist"]),
                                           "21-day realised, lagged 1 day",
                                           "no close data for this date")
        if verbose:
            ok_h = np.isfinite(df["sigma_hist"])
            print("\n  sigma_hist (21-day realised, lagged one day):")
            print("    available on %d of %d rows; range %.1f%% to %.1f%%"
                  % (ok_h.sum(), len(df),
                     100 * df.loc[ok_h, "sigma_hist"].min(),
                     100 * df.loc[ok_h, "sigma_hist"].max()))
    else:
        df["sigma_hist"] = np.nan
        df["sigma_hist_status"] = "unavailable"

    usable = np.isfinite(df["sigma_bench"])
    if verbose:
        print("\n  benchmark volatility (previous date, same expiry, ATM):")
        print("    available on %d of %d rows (%.1f%%)"
              % (usable.sum(), len(df), 100 * usable.mean()))
        for d in dates:
            k = df["trade_date"] == d
            print("      %s  %4d rows, benchmark on %4d"
                  % (d, int(k.sum()), int((k & usable).sum())))

    # ---- the benchmark price and the residual ----
    F_disc = df["F_implied"] * np.exp(-df["r"] * df["T"])
    df["C_BS"] = np.where(
        usable,
        bs_call(F_disc.to_numpy(), df["K"].to_numpy(), df["T"].to_numpy(),
                df["r"].to_numpy(),
                np.nan_to_num(df["sigma_bench"].to_numpy(), nan=0.2), 0.0),
        np.nan)
    df["epsilon_BS"] = df["C_market"] - df["C_BS"]
    df["y"] = df["C_market"] / df["K"]
    df["eps_scaled"] = df["epsilon_BS"] / df["K"]

    model_ready = usable & np.isfinite(df["sigma_IV"])
    df["model_ready"] = model_ready

    if verbose and model_ready.any():
        e = df.loc[model_ready, "epsilon_BS"]
        c = df.loc[model_ready, "C_market"]
        print("\n  Black-Scholes residual on the %d model-ready rows:"
              % int(model_ready.sum()))
        print("    mean %+.3f, mean |eps| %.3f, max |eps| %.3f index points"
              % (e.mean(), e.abs().mean(), e.abs().max()))
        print("    mean |eps| / price = %.2f%%"
              % (100 * (e.abs() / c).mean()))
        print("    This is the quantity Hybrid A learns. It is NOT zero,")
        print("    which is the point: yesterday's at-the-money volatility")
        print("    cannot price today's whole smile.")

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    df.to_csv(dest, index=False, float_format="%.10g")
    if verbose:
        print("\n  wrote %s  (%d rows, %d columns)"
              % (dest, len(df), df.shape[1]))
        if not st["ok"]:
            print("\n  sigma_hist is blank: %s" % st["reason"])
            print("  Fix with one line in week1.py (END_DATE) and a re-run,")
            print("  on a machine with network access.")
        n_dates = df["trade_date"].nunique()
        if n_dates < 12:
            print("\n  NOT YET A MODELLING SET. %d trading dates against the"
                  % n_dates)
            print("  15 to 20 consecutive daily files now specified.")
        else:
            print("\n  MODELLING SET READY. %d consecutive trading dates."
                  % n_dates)
            print("  The first date can never be model-ready, since the")
            print("  benchmark needs a prior date for the same expiry; over")
            print("  %d dates that costs under %.0f%% of the sample."
                  % (n_dates, 100.0 / n_dates + 1))
    return df


if __name__ == "__main__":
    build()
