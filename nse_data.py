"""
nse_data.py - read the NSE F&O bhavcopy and run the data-quality checks
=========================================================================

Project : AI/ML-Augmented Black-Scholes Framework
Ref     : data_sources.md sections 2.1, 5; ML_experiment_plan.md 6.2
Plan ref: master plan 6.12 to 6.22

Purpose
-------
Turn a downloaded bhavcopy ZIP into a clean NIFTY option table, and run the
six-point sanity check from data_sources.md section 5.6 before any modelling
touches the data.

    python nse_data.py data/raw/fo_bhavcopy/fo_20261030.zip

Run this on the FIRST file, before downloading the other eleven. Finding a
layout problem now costs one download; finding it in December costs twelve.

!! UNTESTED AGAINST A REAL FILE !!
----------------------------------
This module was written without a real bhavcopy to hand, because NSE returns
HTTP 403 to programmatic requests. The UDiFF column names below come from the
NSE circular and from data_sources.md section 2.1, not from an inspected file.

Two consequences, both deliberate:

  1. Every column is resolved through COLUMN_ALIASES, so a name that differs
     from the guess can be fixed in one place rather than hunted through the
     code.
  2. The loader FAILS LOUDLY on an unexpected layout. It prints the columns it
     actually found and says which it could not resolve. It never guesses by
     position and never silently returns a partial table - a half-parsed
     options file that looks plausible is far worse than an exception.

If a column cannot be resolved, add the real name to the alias list and re-run.

The parity check is the point
------------------------------
Check 6 is put-call parity on matched (expiry, strike) pairs. This is the first
place in the project where parity will NOT hold exactly: bid-ask spread,
non-synchronous closes and stale settlement prices all break it. The size of
the violation is a measurement of data quality, so it is reported as a
distribution rather than passed or failed on a single tolerance.
"""

import os
import sys
import zipfile

import numpy as np
import pandas as pd

__all__ = [
    "COLUMN_ALIASES", "read_bhavcopy", "filter_nifty_options",
    "implied_forward", "run_quality_checks", "attach_market_q",
    "MIN_PAIRED_STRIKES",
]

# An expiry needs at least this many paired strikes before its implied
# forward is treated as a price rather than a mark. The 13 November file has
# 93 to 171 pairs on the liquid expiries and 13 to 15 on the far-dated ones,
# with single-digit volume; the gap is wide enough that the threshold is not
# a fine judgement.
MIN_PAIRED_STRIKES = 50

# ---------------------------------------------------------------------------
# column resolution
# ---------------------------------------------------------------------------
# canonical name -> candidate names, in order of preference.
# UDiFF names first (post July 2024), legacy bhavcopy names after.
COLUMN_ALIASES = {
    "symbol":     ["TckrSymb", "SYMBOL", "Symbol"],
    "instrument": ["FinInstrmTp", "INSTRUMENT", "FinInstrmNm"],
    "expiry":     ["XpryDt", "EXPIRY_DT", "ExpiryDate"],
    "strike":     ["StrkPric", "STRIKE_PR", "StrikePrice"],
    "opt_type":   ["OptnTp", "OPTION_TYP", "OptionType"],
    "settle":     ["SttlmPric", "SETTLE_PR", "SettlementPrice"],
    "close":      ["ClsPric", "CLOSE", "ClosePrice"],
    "open_int":   ["OpnIntrst", "OPEN_INT", "OpenInterest"],
    "volume":     ["TtlTradgVol", "TtlTrfVal", "CONTRACTS", "VOLUME"],
    "trade_date": ["TradDt", "TIMESTAMP", "TradeDate"],
    "underlying": ["UndrlygPric", "UNDERLYING_VALUE", "UnderlyingPrice"],
}

# Columns without which nothing downstream is possible.
REQUIRED = ["symbol", "expiry", "strike", "opt_type", "settle"]

# Check 6b compares the dividend yield the option prices imply against the one
# the project assumes (carried from Week 5). The tolerance is 1 percentage
# point: a genuine misestimate of q or r is a few tenths, and the bid-ask on
# the at-the-money pair contributes less again, so anything beyond a point
# means something structural is wrong with the prices rather than with the
# assumption.
Q_ASSUMED = 0.015
Q_TOLERANCE = 0.01


def _resolve(df):
    """Map the file's actual columns onto canonical names.

    Returns (mapping, missing). Matching is case-insensitive and ignores
    spaces and underscores, because NSE has been inconsistent about both.
    """
    def norm(s):
        return str(s).strip().lower().replace("_", "").replace(" ", "")

    present = {norm(c): c for c in df.columns}
    mapping, missing = {}, []
    for canon, cands in COLUMN_ALIASES.items():
        hit = next((present[norm(c)] for c in cands if norm(c) in present), None)
        if hit is not None:
            mapping[canon] = hit
        elif canon in REQUIRED:
            missing.append(canon)
    return mapping, missing


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def read_bhavcopy(path, verbose=True):
    """Read a bhavcopy from a ZIP or a bare CSV and return a canonical frame.

    Reads straight out of the ZIP: the downloaded archives stay untouched in
    data/raw/, so the parser can be re-run later without re-downloading.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            "%s not found.\\nFollow data_sources.md section 5.2 to download it, "
            "and keep the ZIP unextracted in data/raw/fo_bhavcopy/." % path)

    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not names:
                raise ValueError(
                    "%s contains no CSV. Members: %s" % (path, z.namelist()))
            if len(names) > 1 and verbose:
                print("  note: %d CSVs in the archive, reading %r"
                      % (len(names), names[0]))
            with z.open(names[0]) as fh:
                raw = pd.read_csv(fh, low_memory=False)
            inner = names[0]
    else:
        raw = pd.read_csv(path, low_memory=False)
        inner = os.path.basename(path)

    raw.columns = [str(c).strip() for c in raw.columns]
    mapping, missing = _resolve(raw)

    if missing:
        raise ValueError(
            "Could not resolve these required columns: %s\\n\\n"
            "Columns actually in %s:\\n  %s\\n\\n"
            "This module was written without a real bhavcopy to inspect, so the\\n"
            "expected names are educated guesses. Add the real name to\\n"
            "COLUMN_ALIASES in nse_data.py and re-run - nothing else needs to\\n"
            "change."
            % (", ".join(missing), inner, "\\n  ".join(map(str, raw.columns))))

    df = pd.DataFrame({canon: raw[col] for canon, col in mapping.items()})

    for c in ("strike", "settle", "close", "open_int", "volume", "underlying"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("expiry", "trade_date"):
        if c in df:
            # UDiFF writes ISO (2026-11-13); the legacy bhavcopy wrote
            # 13-NOV-2026. Try ISO first and only fall back to day-first, so a
            # date like 03-04-2026 is never silently read as the wrong month.
            parsed = pd.to_datetime(df[c], errors="coerce", format="ISO8601")
            if parsed.isna().all():
                parsed = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
            df[c] = parsed
    for c in ("symbol", "instrument", "opt_type"):
        if c in df:
            df[c] = df[c].astype(str).str.strip().str.upper()

    if verbose:
        print("  read %s: %d rows, %d columns resolved"
              % (inner, len(df), len(mapping)))
        unresolved = [k for k in COLUMN_ALIASES if k not in mapping]
        if unresolved:
            print("  optional columns not found: %s" % ", ".join(unresolved))
    return df


def filter_nifty_options(df, symbol="NIFTY", verbose=True):
    """Keep NIFTY index options only."""
    out = df[df["symbol"] == symbol]
    if "instrument" in out:
        tags = out["instrument"].unique()
        opt = [t for t in tags if "OPT" in t or t in ("CE", "PE", "STO", "IDO")]
        if opt:
            out = out[out["instrument"].isin(opt)]
    out = out[out["opt_type"].isin(["CE", "PE"])]
    if verbose:
        print("  %s options: %d rows (%d CE, %d PE), %d expiries"
              % (symbol, len(out), (out.opt_type == "CE").sum(),
                 (out.opt_type == "PE").sum(), out["expiry"].nunique()))
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# implied forward  (ML_experiment_plan.md section 6.2, step 1)
# ---------------------------------------------------------------------------

def implied_forward(df, r=0.065, verbose=True):
    """Per-expiry forward from the most liquid put-call pair, by parity.

        F = K* + e^{rT} (C* - P*)

    at the strike K* minimising |C - P|, which is the strike nearest the
    forward and the one where both sides are liquid.

    Taken from the market rather than from S e^{(r-q)T}, so no borrowed
    dividend series is trusted. The gap between the two is itself a
    data-quality measure, so both are returned where spot is available.
    """
    rows = []
    for (exp,), g in df.groupby(["expiry"]):
        ce = g[g.opt_type == "CE"].set_index("strike")["settle"]
        pe = g[g.opt_type == "PE"].set_index("strike")["settle"]
        common = ce.index.intersection(pe.index)
        if len(common) == 0:
            continue
        diff = (ce[common] - pe[common]).abs()
        k_star = float(diff.idxmin())

        if "trade_date" in df and df["trade_date"].notna().any():
            t0 = df["trade_date"].dropna().iloc[0]
            T = max((exp - t0).days, 0) / 365.0
        else:
            T = np.nan

        disc = np.exp(r * T) if np.isfinite(T) else 1.0
        F = k_star + disc * float(ce[k_star] - pe[k_star])

        rec = {"expiry": exp, "T": T, "k_star": k_star, "F_implied": F,
               "n_paired_strikes": int(len(common))}
        if "underlying" in g and g["underlying"].notna().any():
            S = float(g["underlying"].dropna().iloc[0])
            rec["S"] = S
            rec["F_over_S"] = F / S if S else np.nan
            # The dividend yield the market is actually pricing, per expiry.
            # Inverting F = S e^{(r-q)T}. Supervisor instruction, 17 Nov:
            # take q from the implied forward rather than from the 1.5%
            # constant, because Indian dividends are discrete and seasonal -
            # the 13 Nov file shows q running from 3.17% at ten days to
            # -0.12% at two and a half months.
            if np.isfinite(T) and T > 0 and S > 0 and F > 0:
                rec["q_implied"] = r - np.log(F / S) / T
            else:
                rec["q_implied"] = np.nan
        rows.append(rec)

    out = pd.DataFrame(rows).sort_values("expiry").reset_index(drop=True)
    if verbose and len(out):
        print("  implied forward by expiry:")
        for _, x in out.iterrows():
            extra = ("  F/S %.4f" % x["F_over_S"]) if "F_over_S" in x and \
                    np.isfinite(x.get("F_over_S", np.nan)) else ""
            print("    %s  T=%.3f  K*=%.0f  F=%.2f  (%d paired strikes)%s"
                  % (pd.Timestamp(x.expiry).date(), x["T"], x.k_star,
                     x.F_implied, x.n_paired_strikes, extra))
    return out


# ---------------------------------------------------------------------------
# the six-point check  (data_sources.md section 5.6)
# ---------------------------------------------------------------------------

def run_quality_checks(df, fwd=None, r=0.065, verbose=True):
    """Run the six checks and return a dict of results.

    Nothing here filters or repairs the data. The point is to look at it
    before deciding what to filter, not to quietly fix it.
    """
    res = {}

    def report(name, ok, detail):
        res[name] = {"ok": bool(ok), "detail": detail}
        if verbose:
            print("  %-34s %-7s %s" % (name, "ok" if ok else "LOOK",
                                       detail))

    if verbose:
        print("\n  Six-point check (data_sources.md 5.6), with check 6 split")
        print("  into 6a and 6b. 'LOOK' means inspect, not reject.\n")

    # 1 - volume of NIFTY rows
    n = len(df)
    report("1 NIFTY option rows present", n >= 1000,
           "%d rows (expect several thousand per day)" % n)

    # 2 - both sides present
    nce = int((df.opt_type == "CE").sum())
    npe = int((df.opt_type == "PE").sum())
    bal = min(nce, npe) / max(nce, npe) if max(nce, npe) else 0
    report("2 both CE and PE present", nce > 0 and npe > 0 and bal > 0.5,
           "%d CE, %d PE (ratio %.2f)" % (nce, npe, bal))

    # 3 - strike spacing near the money
    ks = np.sort(df["strike"].unique())
    if len(ks) > 3:
        mid = ks[len(ks) // 4: 3 * len(ks) // 4]
        gaps = np.diff(mid)
        common = pd.Series(gaps).mode()
        spacing = float(common.iloc[0]) if len(common) else float("nan")
        report("3 strike spacing near the money", spacing in (50.0, 100.0),
               "modal gap %.0f points over %d strikes" % (spacing, len(ks)))
    else:
        report("3 strike spacing near the money", False,
               "only %d distinct strikes" % len(ks))

    # 4 - settlement prices populated where there is activity
    if "volume" in df and df["volume"].notna().any():
        live = df[df["volume"].fillna(0) > 0]
    elif "open_int" in df and df["open_int"].notna().any():
        live = df[df["open_int"].fillna(0) > 0]
    else:
        live = df
    zero = int((live["settle"].fillna(0) <= 0).sum())
    frac = zero / max(len(live), 1)
    report("4 settle non-zero where traded", frac < 0.05,
           "%d of %d traded rows have settle <= 0 (%.1f%%)"
           % (zero, len(live), 100 * frac))

    # 5 - no negative prices or strikes, no missing settlement
    bad = int((df["settle"] < 0).sum() + (df["strike"] <= 0).sum()
              + df["settle"].isna().sum())
    report("5 no negative or missing values", bad == 0,
           "%d offending rows" % bad)

    # 6 - put-call parity on matched strikes. The interesting one.
    pairs = []
    for (exp,), g in df.groupby(["expiry"]):
        ce = g[g.opt_type == "CE"].set_index("strike")["settle"]
        pe = g[g.opt_type == "PE"].set_index("strike")["settle"]
        common = ce.index.intersection(pe.index)
        if not len(common):
            continue
        Frow = None
        if fwd is not None and len(fwd):
            hit = fwd[fwd["expiry"] == exp]
            if len(hit):
                Frow = hit.iloc[0]
        if Frow is None or not np.isfinite(Frow.get("T", np.nan)):
            continue
        T, F = float(Frow["T"]), float(Frow["F_implied"])
        K = common.to_numpy(dtype=float)
        # C - P = e^{-rT}(F - K)
        resid = (ce[common].to_numpy() - pe[common].to_numpy()
                 - np.exp(-r * T) * (F - K))
        blk = pd.DataFrame({"expiry": exp, "T": T, "strike": K,
                            "parity_resid": resid,
                            "m": np.log(F / K)})
        if "volume" in g:
            cv = g[g.opt_type == "CE"].set_index("strike")["volume"]
            pv = g[g.opt_type == "PE"].set_index("strike")["volume"]
            blk["both_traded"] = ((cv[common].fillna(0).to_numpy() > 0)
                                  & (pv[common].fillna(0).to_numpy() > 0))
        else:
            blk["both_traded"] = True
        pairs.append(blk)

    if pairs:
        pr = pd.concat(pairs, ignore_index=True)
        res["parity_table"] = pr

        # Report the FILTERED figure, not the raw one. The aggregate over all
        # pairs is dominated by contracts where one leg never traded, whose
        # settlement price is a theoretical mark rather than a quote. Judging
        # the data on that number would condemn a file that is in fact clean
        # exactly where the project intends to use it.
        live = pr[pr["both_traded"] & (pr["m"].abs() <= 0.3)]
        a_all = pr["parity_resid"].abs()
        if len(live) >= 20:
            a = live["parity_resid"].abs()
            report("6a parity shape (both legs traded, |m|<=0.3)",
                   a.median() < 10.0,
                   "median |resid| %.2f, 90th pct %.2f, max %.2f index points "
                   "over %d pairs" % (a.median(), a.quantile(0.9), a.max(),
                                      len(live)))
            if verbose:
                print("       unfiltered, all %d pairs: median %.2f, max %.2f"
                      % (len(pr), a_all.median(), a_all.max()))
                dead = pr[~pr["both_traded"]]
                if len(dead):
                    print("       one leg untraded, %d pairs: median %.2f"
                          % (len(dead), dead["parity_resid"].abs().median()))
                print("\n       by |m|, both legs traded:")
                s = live.copy()
                s["bin"] = pd.cut(s["m"].abs(), [0, 0.05, 0.1, 0.2, 0.3])
                for b, gg in s.groupby("bin", observed=True):
                    print("         %-12s n=%4d  median %6.2f  90pct %7.2f"
                          % (b, len(gg), gg["parity_resid"].abs().median(),
                             gg["parity_resid"].abs().quantile(0.9)))
        else:
            report("6a parity shape (both legs traded, |m|<=0.3)",
                   a_all.median() < 10.0,
                   "too few liquid pairs to filter; unfiltered median "
                   "|resid| %.2f over %d pairs" % (a_all.median(), len(pr)))

        if verbose:
            print("\n    Parity will NOT hold exactly on market data - bid-ask")
            print("    spread, non-synchronous closes and stale settlement")
            print("    prices all break it. The size of the violation is a")
            print("    measure of data quality, so record it rather than")
            print("    filtering it away silently. A few index points on a")
            print("    25,000 index is 0.01% and is fine; a median in the")
            print("    tens AFTER filtering means the prices are suspect.\n")
    else:
        report("6a parity shape (both legs traded, |m|<=0.3)", False,
               "no matched pairs with a usable maturity - check expiry "
               "and trade-date parsing")

    # 6b - the forward itself, against spot.
    #
    # Why 6a is not enough, and this was found by testing rather than by
    # thinking: F is ESTIMATED from the same put-call pair that 6a then
    # checks. Add a constant to every call price and F absorbs it exactly,
    # so 6a still reports a zero residual. 6a therefore detects only
    # STRIKE-DEPENDENT violations - the shape of the smile - and is blind to
    # a uniform offset on one side, which is precisely what a settlement
    # convention error or a wrong price column would look like.
    #
    # Comparing the implied forward against S e^{(r-q)T} catches it, because
    # spot comes from a different column entirely.
    # Expressed as an IMPLIED DIVIDEND YIELD rather than a relative gap in F.
    # Inverting F = S e^{(r-q)T} gives
    #
    #     q_implied = r - ln(F / S) / T
    #
    # which is the right way round for two reasons. It is interpretable - "the
    # options imply a 3.2% dividend yield against the 1.5% assumed" says
    # something, where "the forward is 0.16% off" does not. And the tolerance
    # becomes maturity-correct automatically: a fixed relative gap in F that is
    # alarming at one month is unremarkable at one year, because the same
    # price offset is divided by a longer T.
    # Restricted to LIQUID expiries. A far-dated NIFTY expiry with a dozen
    # paired strikes and single-digit volume has a forward that is a mark, not
    # a price; including it makes the implied yield look wild for reasons that
    # have nothing to do with dividends.
    if fwd is not None and len(fwd) and "F_over_S" in fwd:
        ok_rows = fwd[np.isfinite(fwd.get("F_over_S", np.nan))
                      & (fwd["T"] >= 7.0 / 365.0)
                      & (fwd["n_paired_strikes"] >= MIN_PAIRED_STRIKES)]
        if len(ok_rows) == 0:        # nothing liquid: fall back, and say so
            ok_rows = fwd[np.isfinite(fwd.get("F_over_S", np.nan))
                          & (fwd["T"] >= 7.0 / 365.0)]
        if len(ok_rows):
            T_ = ok_rows["T"].to_numpy()
            q_imp = r - np.log(ok_rows["F_over_S"].to_numpy()) / T_
            dev = np.abs(q_imp - Q_ASSUMED)
            worst = float(np.max(dev))
            report("6b implied dividend yield", worst < Q_TOLERANCE,
                   "implied q ranges %.2f%% to %.2f%% against %.2f%% assumed; "
                   "worst deviation %.2f pp over %d expiries"
                   % (100 * q_imp.min(), 100 * q_imp.max(),
                      100 * Q_ASSUMED, 100 * worst, len(ok_rows)))
            res["implied_q"] = pd.DataFrame(
                {"expiry": ok_rows["expiry"].to_numpy(), "T": T_,
                 "q_implied": q_imp})
            if verbose and worst >= Q_TOLERANCE:
                print("\n    An implied yield far from %.1f%% usually means the"
                      % (100 * Q_ASSUMED))
                print("    wrong price column, a settlement convention that is")
                print("    not a midpoint, or the wrong r. Unlike 6a this")
                print("    cannot be absorbed by the forward, because spot is")
                print("    read from a different column.\n")
        else:
            report("6b implied dividend yield", False,
                   "no expiry with spot and T >= 7 days - cannot compare")
    else:
        report("6b implied dividend yield", False,
               "underlying price column not found; 6a ALONE CANNOT DETECT a "
               "uniform offset on one side, so this gap matters")

    n_chk = sum(1 for k, v in res.items()
                if isinstance(v, dict) and "ok" in v)
    n_fail = sum(1 for k, v in res.items()
                 if isinstance(v, dict) and "ok" in v and not v["ok"])
    if verbose:
        print("  %d of %d checks flagged for inspection" % (n_fail, n_chk))
    return res


# ---------------------------------------------------------------------------

def attach_market_q(df, fwd, q_fallback=0.015, verbose=True):
    """Attach the per-expiry implied dividend yield and forward to each row.

    This is what the pricing stack should use on market data, in place of the
    1.5% constant carried through the synthetic stage. Two reasons, both
    visible in the first downloaded file:

      - Indian dividends are discrete and seasonal, not a continuous yield,
        so the yield the forward implies varies systematically with maturity;
      - taking q from the forward means no borrowed dividend series has to be
        trusted, which is the convention already fixed in
        ML_experiment_plan.md section 6.2.

    Expiries whose forward is a mark rather than a price - fewer than
    MIN_PAIRED_STRIKES paired strikes - fall back to q_fallback, and are
    flagged in the `q_source` column so nothing is silently substituted.
    """
    f = fwd.copy()
    if "q_implied" not in f:
        f["q_implied"] = np.nan
    liquid = (f["n_paired_strikes"] >= MIN_PAIRED_STRIKES) & \
             np.isfinite(f["q_implied"])
    f["q_market"] = np.where(liquid, f["q_implied"], q_fallback)
    f["q_source"] = np.where(liquid, "implied forward", "fallback constant")

    out = df.merge(f[["expiry", "T", "F_implied", "q_market", "q_source",
                      "n_paired_strikes"]], on="expiry", how="left")
    if verbose:
        use = f[liquid]
        print("  per-expiry q from the implied forward:")
        for _, x in f.iterrows():
            mark = "" if x["q_source"] == "implied forward" else "   (fallback)"
            print("    %s  T=%.3f  pairs=%3d   q = %+.2f%%%s"
                  % (pd.Timestamp(x["expiry"]).date(), x["T"],
                     x["n_paired_strikes"], 100 * x["q_market"], mark))
        if len(use):
            print("  %d of %d expiries use the market forward; q spans "
                  "%.2f%% to %.2f%%"
                  % (len(use), len(f), 100 * use["q_implied"].min(),
                     100 * use["q_implied"].max()))
    return out


def inspect(path, r=0.065):
    """Full pass over one downloaded file."""
    print("=" * 78)
    print("NSE F&O BHAVCOPY INSPECTION")
    print("=" * 78)
    print("  file: %s" % path)
    df = read_bhavcopy(path)
    opt = filter_nifty_options(df)
    if len(opt) == 0:
        raise SystemExit(
            "No NIFTY option rows found. Check the symbol and instrument "
            "columns printed above - the filter in filter_nifty_options() may "
            "need the real tag names.")
    fwd = implied_forward(opt, r=r)
    run_quality_checks(opt, fwd=fwd, r=r)
    print()
    opt = attach_market_q(opt, fwd)
    print("\n" + "=" * 78)
    print("Next: if all six look reasonable, download the other eleven months")
    print("(data_sources.md section 5.1 for the window). If any flagged, bring")
    print("the output above to the review before downloading anything else.")
    print("=" * 78)
    return opt, fwd


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nusage: python nse_data.py <path to bhavcopy zip or csv>")
        print("\nNothing downloaded yet? See data_sources.md section 5.")
        raise SystemExit(1)
    inspect(sys.argv[1])
