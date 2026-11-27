"""
clean_options.py - raw bhavcopies to a modelling dataset
=========================================================

Project : AI/ML-Augmented Black-Scholes Framework
Ref     : post-midsem plan Week 7 row 28; master plan 11.9, 6.17-6.22
Spec    : data_sources.md section 4; ML_experiment_plan.md sections 4.1, 6.2

    python clean_options.py                    # everything in data/raw/fo_bhavcopy
    python clean_options.py <dir-or-file> ...

Produces `data/clean_options.csv` and a filter waterfall showing where every
dropped row went. The waterfall is the point: a cleaning script that reports
only its output row count is asking to be trusted rather than checked.

Order of operations, and why it is this order
----------------------------------------------
1. **Per file**, because every quantity below is a function of one trading
   day's prices. Files are concatenated only at the end.
2. **Implied forward per expiry** (nse_data), which refuses the expiring
   series and any expiry whose settlement column is degenerate - see
   data_sources.md 5.7 for what that trap cost.
3. **Maturity filter** before anything else that uses T, so the expiry-week
   distortion cannot contaminate a later step.
4. **Liquidity** before moneyness, because a stale quote in the wings is not
   evidence about where the wings are.
5. **OTM selection, then moneyness band.** This order is specified in
   ML_experiment_plan.md 6.2 and it matters: the band is measured against the
   implied forward F, not against spot, and the OTM/ITM boundary is also F.
6. **Parity conversion** of the retained puts to call-equivalent prices.
7. **No-arbitrage screening** last, on the converted prices, because that is
   what the model will see.

What is NOT done here
---------------------
No volatility feature. `sigma_hist` needs a daily NIFTY close series covering
the option window and the cached series in `data/nifty50_prices.csv` currently
ends 31 December 2024. See `sigma_hist_status()` below and the note it prints.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

import nse_data as nd

__all__ = ["clean_one", "clean_many", "sigma_hist_status", "FILTERS"]

# ---- filter parameters, from data_sources.md section 4 -------------------
MIN_T_DAYS = 7           # drop the expiry week: settlement there is distorted
MAX_ABS_M = 0.30         # |ln(F/K)| band, argued from the Week 3 vega result
MIN_VOLUME = 1           # contracts traded
MIN_OPEN_INTEREST = 1
R_DEFAULT = 0.065        # placeholder until the RBI series is downloaded

TICK = 0.05              # one tick on an NSE index option

# Minimum quoted premium, as observed - NOT as converted.
#
# The principle: a quote can pin implied volatility no better than its own
# granularity allows. One tick on a premium of Rs 5.00 is 1% of the quote, and
# below that the tick alone moves the recovered volatility by more than a
# volatility point. This is the Week 3 result - delta_sigma ~ residual / vega -
# restated in the one unit that is actually observable in the file.
#
# The two files to hand corroborate it. Implied volatility dispersion by
# quoted premium, over 1,410 cleaned rows:
#
#     premium     n     IV median    IV IQR     vega median
#     1 - 2      37      44.77%      7.36 pp         37
#     2 - 5     138      28.18%     10.55 pp         96
#     5 - 20    244      20.02%      5.85 pp        332
#     20 - 100  404      16.63%      3.09 pp       1102
#     > 100     548      14.43%      1.91 pp       2756
#
# The 1-5 band is not a smile, it is noise: a median of 28-45% against 14.4%
# for the liquid body. Applying the floor keeps 85% of rows and cuts the
# overall IQR from 5.15 to 3.61 percentage points.
#
# The threshold is set from the principle, not fitted to these two days -
# two trading dates are not enough to tune a filter on, and it is reviewed
# once the full window is in.
MIN_PREMIUM = 100 * TICK  # Rs 5.00: one tick is then at most 1% of the quote

FILTERS = ["raw NIFTY option rows", "expiry has a usable forward",
           "maturity >= %d days" % MIN_T_DAYS, "traded today",
           "open interest > 0", "out of the money",
           "|ln(F/K)| <= %.2f" % MAX_ABS_M,
           "premium >= Rs %.2f" % MIN_PREMIUM,
           "passes no-arbitrage screen"]


def sigma_hist_status(path=os.path.join("data", "nifty50_prices.csv"),
                      needed_from=None, verbose=True):
    """Report whether the cached close series can supply sigma_hist.

    This is a blocker, not a detail, so it is checked explicitly and reported
    rather than discovered as a column of NaNs later.
    """
    out = {"path": path, "ok": False, "reason": "", "last": None}
    if not os.path.exists(path):
        out["reason"] = "cache missing"
    else:
        d = pd.read_csv(path)
        col = next((c for c in d.columns if "date" in c.lower()), d.columns[0])
        last = pd.to_datetime(d[col]).max()
        out["last"] = last
        if needed_from is None or last >= needed_from:
            out["ok"] = True
        else:
            out["reason"] = ("cache ends %s, option data starts %s"
                             % (last.date(), pd.Timestamp(needed_from).date()))
    if verbose and not out["ok"]:
        print("\n  !! sigma_hist UNAVAILABLE: %s" % out["reason"])
        print("     ML_experiment_plan.md 4.1 makes 21-day realised volatility")
        print("     the baseline feature on market data. It needs a daily")
        print("     NIFTY close series covering the option window.")
        print("     Fix, on a machine with network access:")
        print("         edit week1.py: END_DATE = '2026-12-31'")
        print("         python week1.py        # re-downloads and re-caches")
        print("     The cleaned file is still written; the feature is left")
        print("     blank rather than filled with a fallback that would be")
        print("     silently wrong.")
    return out


# --------------------------------------------------------------------------

def clean_one(path, r=R_DEFAULT, verbose=True):
    """Clean one bhavcopy. Returns (DataFrame, waterfall dict)."""
    raw = nd.read_bhavcopy(path, verbose=False)
    df = nd.filter_nifty_options(raw, verbose=False)
    wf = {FILTERS[0]: len(df)}
    if len(df) == 0:
        return df, wf

    trade_date = df["trade_date"].dropna().iloc[0]
    spot = float(df["underlying"].dropna().iloc[0])

    fwd = nd.implied_forward(df, r=r, verbose=False)
    df = nd.attach_market_q(df, fwd, verbose=False)
    df = df[df["F_implied"].notna()]
    wf[FILTERS[1]] = len(df)

    df["trade_date"] = trade_date
    df["S"] = spot
    df["r"] = r

    # --- 3. maturity ---
    df = df[df["T"] >= MIN_T_DAYS / 365.0]
    wf[FILTERS[2]] = len(df)

    # --- 4. liquidity ---
    if "volume" in df:
        df = df[df["volume"].fillna(0) >= MIN_VOLUME]
    wf[FILTERS[3]] = len(df)
    if "open_int" in df:
        df = df[df["open_int"].fillna(0) >= MIN_OPEN_INTEREST]
    wf[FILTERS[4]] = len(df)

    # --- 5a. out of the money, measured against the IMPLIED FORWARD ---
    # Liquidity migrates to whichever side is out of the money, so this keeps
    # the traded side and discards the wide, stale, in-the-money quotes.
    F = df["F_implied"].to_numpy()
    K = df["strike"].to_numpy()
    is_call = (df["opt_type"] == "CE").to_numpy()
    otm = np.where(is_call, K >= F, K < F)
    df = df[otm]
    wf[FILTERS[5]] = len(df)

    # --- 5b. moneyness band, also against F ---
    df = df[np.abs(np.log(df["F_implied"] / df["strike"])) <= MAX_ABS_M]
    wf[FILTERS[6]] = len(df)

    # Floor on the RAW quote, before parity conversion. A deep out-of-the-money
    # put worth Rs 0.60 converts to a deep in-the-money call worth Rs 6,453 -
    # arithmetically exact, and still carrying only Rs 0.60 of information.
    # Filtering on the converted price would let all of them through.
    df = df[df["settle"] >= MIN_PREMIUM]
    wf[FILTERS[7]] = len(df)
    if len(df) == 0:
        return df, wf

    # --- 6. parity conversion: every retained put becomes a call ---
    #   C = P + e^{-rT}(F - K)
    # Recorded in source_side so any later result can be re-examined by side.
    T = df["T"].to_numpy()
    K = df["strike"].to_numpy()
    F = df["F_implied"].to_numpy()
    px = df["settle"].to_numpy()
    isc = (df["opt_type"] == "CE").to_numpy()
    disc = np.exp(-r * T)
    call_equiv = np.where(isc, px, px + disc * (F - K))

    out = pd.DataFrame({
        "trade_date": df["trade_date"].to_numpy(),
        "expiry": df["expiry"].to_numpy(),
        "T": T,
        "S": df["S"].to_numpy(),
        "K": K,
        "F_implied": F,
        "r": df["r"].to_numpy(),
        "q_market": df["q_market"].to_numpy(),
        "q_source": df["q_source"].to_numpy(),
        "source_side": np.where(isc, "call", "put->call"),
        "raw_price": px,
        "C_market": call_equiv,
        "volume": df.get("volume", pd.Series(np.nan, index=df.index)).to_numpy(),
        "open_int": df.get("open_int",
                           pd.Series(np.nan, index=df.index)).to_numpy(),
    })
    # features the model will use; m is measured against the FORWARD, which is
    # the scale-free variable the pricing problem is symmetric in
    out["m"] = np.log(out["F_implied"] / out["K"])
    out["y"] = out["C_market"] / out["K"]

    # --- 7. no-arbitrage screening, on the converted price ---
    lower = np.maximum(0.0, disc * (F - K))        # forward intrinsic
    upper = disc * F                               # a call cannot exceed e^-rT F
    keep = (out["C_market"] >= lower - 1e-9) & (out["C_market"] <= upper + 1e-9)
    out["arb_lower_breach"] = np.maximum(0.0, lower - out["C_market"])
    out["arb_upper_breach"] = np.maximum(0.0, out["C_market"] - upper)
    dropped = out[~keep]
    out = out[keep]
    wf[FILTERS[8]] = len(out)

    if verbose:
        print("  %s  spot %9.2f   %4d rows kept   (%d failed no-arbitrage)"
              % (pd.Timestamp(trade_date).date(), spot, len(out), len(dropped)))
    return out.reset_index(drop=True), wf


def clean_many(paths, r=R_DEFAULT, verbose=True):
    """Clean every file and concatenate. Returns (DataFrame, waterfall)."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "*.zip")))
            files += sorted(glob.glob(os.path.join(p, "*.csv")))
        else:
            files.append(p)
    if not files:
        raise SystemExit(
            "No bhavcopy files found.\n"
            "Download them per data_sources.md section 5 and put the ZIPs,\n"
            "unextracted, in data/raw/fo_bhavcopy/.")

    if verbose:
        print("=" * 78)
        print("CLEANING %d FILE%s" % (len(files), "" if len(files) == 1 else "S"))
        print("=" * 78)

    frames, total = [], {}
    for f in files:
        d, wf = clean_one(f, r=r, verbose=verbose)
        if len(d):
            frames.append(d)
        for k, v in wf.items():
            total[k] = total.get(k, 0) + v

    if not frames:
        raise SystemExit("every file cleaned to zero rows - check the filters")
    out = pd.concat(frames, ignore_index=True)

    if verbose:
        print("\n  FILTER WATERFALL")
        prev = None
        for k in FILTERS:
            n = total.get(k, 0)
            drop = "" if prev is None else "  -%d" % (prev - n)
            pct = 100.0 * n / max(total[FILTERS[0]], 1)
            print("    %-34s %7d  (%5.1f%%)%s" % (k, n, pct, drop))
            prev = n
        print("\n  kept %d of %d raw rows (%.1f%%)"
              % (len(out), total[FILTERS[0]],
                 100.0 * len(out) / max(total[FILTERS[0]], 1)))
        print("  %d from calls, %d converted from puts"
              % (int((out.source_side == "call").sum()),
                 int((out.source_side == "put->call").sum())))
        print("  %d trading dates, %d expiries, %d distinct strikes"
              % (out.trade_date.nunique(), out.expiry.nunique(),
                 out.K.nunique()))
    return out, total


# --------------------------------------------------------------------------

def main(argv):
    paths = argv[1:] or [os.path.join("data", "raw", "fo_bhavcopy")]
    out, wf = clean_many(paths)

    os.makedirs("data", exist_ok=True)
    dest = os.path.join("data", "clean_options.csv")
    out.to_csv(dest, index=False, float_format="%.10g")
    print("\n  wrote %s  (%d rows, %d columns)" % (dest, len(out), out.shape[1]))

    os.makedirs("results", exist_ok=True)
    pd.Series(wf).to_csv("results/w09_filter_waterfall.csv", header=False)

    # the volatility feature, reported as a blocker rather than a NaN column
    sigma_hist_status(needed_from=out["trade_date"].min())

    print("\n  SAMPLE READINESS")
    n_dates = out["trade_date"].nunique()
    print("    %d trading date%s in the cleaned file."
          % (n_dates, "" if n_dates == 1 else "s"))
    # Revised 27 Nov: the window is 15-20 CONSECUTIVE DAILY files, not twelve
    # months sampled weekly. Daily sampling is what keeps sigma_atm_prev one
    # day old instead of a week, which is the finding in the Week 8 report.
    if n_dates < 12:
        print("    The revised specification asks for 15 to 20 consecutive")
        print("    daily files. This is %d, which is too few to cut into" % n_dates)
        print("    train, validation and test chronologically.")
    else:
        print("    The revised specification asks for 15 to 20 consecutive")
        print("    daily files; this meets it. A chronological split is")
        print("    possible, holding out the latest dates for test.")
    return out




# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def _self_test(verbose=True):
    """Checks on a synthetic bhavcopy where the right answer is known."""
    import io
    import zipfile
    import tempfile
    from black_scholes import bs_call, bs_put

    if verbose:
        print("=" * 78)
        print("clean_options.py  SELF-TEST")
        print("=" * 78)
    checks = []

    def chk(name, cond, extra=""):
        checks.append(bool(cond))
        if verbose:
            print("  %-58s %s %s" % (name, "ok" if cond else "FAIL", extra))

    S, r, q, sig = 25000.0, 0.065, 0.015, 0.14
    trade = pd.Timestamp("2026-11-23")
    exps = [trade + pd.Timedelta(days=d) for d in (3, 21, 60)]
    rows = []
    for e in exps:
        T = (e - trade).days / 365.0
        F = S * np.exp((r - q) * T)
        for K in np.arange(20000, 30050, 50.0):
            c = float(bs_call(S, K, T, r, sig, q))
            p = float(bs_put(S, K, T, r, sig, q))
            vol = 5000 if abs(np.log(F / K)) < 0.35 else 0
            for tag, px in (("CE", c), ("PE", p)):
                rows.append({"TradDt": trade.strftime("%Y-%m-%d"),
                             "TckrSymb": "NIFTY", "FinInstrmTp": "IDO",
                             "XpryDt": e.strftime("%Y-%m-%d"), "StrkPric": K,
                             "OptnTp": tag, "SttlmPric": round(px, 2),
                             "ClsPric": round(px, 2), "OpnIntrst": 1000,
                             "TtlTradgVol": vol, "UndrlygPric": S})
    raw = pd.DataFrame(rows)
    tmp = tempfile.mktemp(suffix=".zip")
    buf = io.StringIO()
    raw.to_csv(buf, index=False)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("bhav.csv", buf.getvalue())

    out, wf = clean_one(tmp, r=r, verbose=False)
    chk("cleans without error", len(out) > 0, "(%d rows)" % len(out))

    # the 3-day expiry must be dropped by the maturity filter
    kept = set(pd.Timestamp(x).date() for x in out["expiry"])
    chk("expiry inside %d days is dropped" % MIN_T_DAYS,
        exps[0].date() not in kept)
    chk("the two longer expiries survive",
        exps[1].date() in kept and exps[2].date() in kept)

    # every retained row is out of the money against the forward
    isc = (out["source_side"] == "call").to_numpy()
    chk("retained calls are all K >= F",
        bool((out["K"].to_numpy()[isc] >= out["F_implied"].to_numpy()[isc]
              - 1e-9).all()))
    chk("retained puts are all K < F",
        bool((out["K"].to_numpy()[~isc] < out["F_implied"].to_numpy()[~isc]
              + 1e-9).all()))

    # THE parity check: a converted put must equal the true call price
    true_c = np.array([float(bs_call(S, k, t, r, sig, q))
                       for k, t in zip(out["K"], out["T"])])
    err = np.abs(out["C_market"].to_numpy() - true_c)
    chk("converted puts reproduce the true call price",
        float(err[~isc].max()) < 0.75,
        "(max %.4f index points on %d rows)" % (err[~isc].max(), (~isc).sum()))
    chk("native calls are untouched", float(err[isc].max()) < 0.75,
        "(max %.4f)" % err[isc].max())

    chk("moneyness band respected",
        float(np.abs(out["m"]).max()) <= MAX_ABS_M + 1e-9)
    chk("premium floor respected",
        float(out["raw_price"].min()) >= MIN_PREMIUM - 1e-9)
    chk("no no-arbitrage breaches survive",
        float(out["arb_lower_breach"].max() + out["arb_upper_breach"].max())
        < 1e-6)
    chk("waterfall is monotone non-increasing",
        all(wf[FILTERS[i]] >= wf[FILTERS[i + 1]]
            for i in range(len(FILTERS) - 1) if FILTERS[i + 1] in wf))
    chk("target y = C/K is consistent",
        float(np.abs(out["y"] - out["C_market"] / out["K"]).max()) < 1e-12)

    n_fail = sum(1 for c in checks if not c)
    if verbose:
        print("-" * 78)
        print("  %d checks, %d failed" % (len(checks), n_fail))
        print("  ALL CHECKS PASSED" if n_fail == 0
              else "  *** FAILURES ABOVE ***")
        print("=" * 78)
    return n_fail == 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        raise SystemExit(0 if _self_test() else 1)
    main(sys.argv)
