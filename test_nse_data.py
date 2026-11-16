"""
test_nse_data.py - exercise nse_data.py against a synthetic UDiFF fixture
==========================================================================

Project : AI/ML-Augmented Black-Scholes Framework
Ref     : data_sources.md section 5.6

Usage:  python test_nse_data.py          (from the repository root)

The real file cannot be fetched from here, so this builds a ZIP with the
column names nse_data.py expects, fills it with arbitrage-consistent prices,
and checks three things:

  1. a clean file parses and passes all six checks;
  2. each deliberately planted defect is actually caught;
  3. a file with the wrong column names fails LOUDLY, naming the columns it
     found, rather than returning a half-parsed table.

Point 3 is the one that matters most, because the column names in
nse_data.py are educated guesses.
"""
import io
import os
import sys
import zipfile

import numpy as np
import pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(
    os.path.abspath(__file__))
sys.path.insert(0, root)

from black_scholes import bs_call, bs_put          # noqa: E402
import nse_data                                     # noqa: E402

TMP = "/tmp/nse_fixture"
os.makedirs(TMP, exist_ok=True)
rng = np.random.default_rng(20260825)

TRADE = pd.Timestamp("2026-10-30")
# weeklies plus two monthlies, as in a real NIFTY chain
EXPIRIES = [pd.Timestamp(d) for d in
            ("2026-11-05", "2026-11-12", "2026-11-19", "2026-11-26",
             "2026-12-03", "2026-12-10", "2026-12-17", "2026-12-31")]
SPOT, R, Q, SIG = 24800.0, 0.065, 0.015, 0.13


def make_rows(defect=None):
    """Arbitrage-consistent NIFTY option rows in UDiFF-ish shape."""
    rows = []
    for exp in EXPIRIES:
        T = (exp - TRADE).days / 365.0
        for K in np.arange(23000, 26600, 50.0):
            c = float(bs_call(SPOT, K, T, R, SIG, Q))
            p = float(bs_put(SPOT, K, T, R, SIG, Q))
            # liquidity thins away from the money, as in the real thing
            far = abs(np.log(SPOT / K))
            vol = max(0, int(90000 * np.exp(-12 * far) + rng.normal(0, 40)))
            oi = max(0, int(300000 * np.exp(-9 * far)))
            for tag, px in (("CE", c), ("PE", p)):
                rows.append({
                    "TradDt": TRADE.strftime("%d-%m-%Y"),
                    "TckrSymb": "NIFTY",
                    "FinInstrmTp": "IDO",
                    "XpryDt": exp.strftime("%d-%m-%Y"),
                    "StrkPric": K,
                    "OptnTp": tag,
                    "SttlmPric": round(px, 2),
                    "ClsPric": round(px * (1 + rng.normal(0, 2e-4)), 2),
                    "OpnIntrst": oi,
                    "TtlTradgVol": vol,
                    "UndrlygPric": SPOT,
                })
    df = pd.DataFrame(rows)

    # a little non-NIFTY noise, which the filter must drop
    other = df.head(40).copy()
    other["TckrSymb"] = "BANKNIFTY"
    df = pd.concat([df, other], ignore_index=True)

    if defect == "parity":
        # shift every CE settle up: parity breaks by a fixed amount
        k = (df.TckrSymb == "NIFTY") & (df.OptnTp == "CE")
        df.loc[k, "SttlmPric"] = df.loc[k, "SttlmPric"] + 40.0
    elif defect == "zero_settle":
        k = (df.TckrSymb == "NIFTY") & (df.TtlTradgVol > 0)
        idx = df[k].sample(frac=0.30, random_state=1).index
        df.loc[idx, "SttlmPric"] = 0.0
    elif defect == "one_sided":
        df = df[~((df.TckrSymb == "NIFTY") & (df.OptnTp == "PE")
                  & (df.StrkPric > 24000))]
    elif defect == "negative":
        idx = df[df.TckrSymb == "NIFTY"].sample(5, random_state=2).index
        df.loc[idx, "SttlmPric"] = -1.0
    elif defect == "spacing":
        df = df[df.StrkPric % 500 == 0]
    return df


def zip_it(df, name, colmap=None):
    if colmap:
        df = df.rename(columns=colmap)
    p = os.path.join(TMP, name)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(name.replace(".zip", ".csv"), buf.getvalue())
    return p


passed = failed = 0


def expect(label, cond, extra=""):
    global passed, failed
    if cond:
        passed += 1
        print("  PASS  %s %s" % (label, extra))
    else:
        failed += 1
        print("  FAIL  %s %s" % (label, extra))


print("=" * 78)
print("nse_data.py  -  fixture tests")
print("=" * 78)

# ---------------------------------------------------------------- clean file
print("\n[1] clean file")
p = zip_it(make_rows(), "fo_clean.zip")
df = nse_data.read_bhavcopy(p, verbose=False)
opt = nse_data.filter_nifty_options(df, verbose=False)
expect("BANKNIFTY rows dropped", (opt.symbol == "NIFTY").all(),
       "(%d rows kept)" % len(opt))
expect("both sides present", {"CE", "PE"} == set(opt.opt_type.unique()))
expect("expiry parsed as dates", opt.expiry.notna().all())
expect("all expiries found", opt.expiry.nunique() == len(EXPIRIES))

fwd = nse_data.implied_forward(opt, r=R, verbose=False)
# the true forward is S e^{(r-q)T}
for _, x in fwd.iterrows():
    true_F = SPOT * np.exp((R - Q) * x["T"])
    expect("implied forward matches S e^{(r-q)T} for %s"
           % pd.Timestamp(x.expiry).date(),
           abs(x.F_implied - true_F) < 2.0,
           "(got %.1f, true %.1f)" % (x.F_implied, true_F))

res = nse_data.run_quality_checks(opt, fwd=fwd, r=R, verbose=False)
for k, v in res.items():
    if not (isinstance(v, dict) and "ok" in v):
        continue                      # parity_table / implied_q are data, not checks
    expect("clean: %s" % k, v["ok"], "- %s" % v["detail"])

pr = res["parity_table"]["parity_resid"].abs()
expect("parity residual is tiny on clean data", pr.max() < 1.0,
       "(max %.4f index points)" % pr.max())

# 6a is blind to a uniform offset: assert that explicitly, so the limitation is
# documented by a passing test rather than by a comment alone.
shifted = nse_data.filter_nifty_options(
    nse_data.read_bhavcopy(zip_it(make_rows("parity"), "fo_shift.zip"),
                           verbose=False), verbose=False)
fs = nse_data.implied_forward(shifted, r=R, verbose=False)
rs = nse_data.run_quality_checks(shifted, fwd=fs, r=R, verbose=False)
expect("6a is BLIND to a uniform offset (documented limitation)",
       rs["6a parity shape (both legs traded, |m|<=0.3)"]["ok"] is True,
       "- the forward absorbs it")
expect("6b catches the uniform offset", 
       rs["6b implied dividend yield"]["ok"] is False,
       "- %s" % rs["6b implied dividend yield"]["detail"])

# ------------------------------------------------------------- planted faults
print("\n[2] planted defects must be caught")
CASES = [
    # A UNIFORM offset on one side is absorbed by the implied forward, so 6a
    # cannot see it and 6b must. This was found by this test, not by design.
    ("parity", "6b implied dividend yield"),
    ("zero_settle", "4 settle non-zero where traded"),
    ("one_sided", "2 both CE and PE present"),
    ("negative", "5 no negative or missing values"),
    ("spacing", "3 strike spacing near the money"),
]
for defect, check in CASES:
    p = zip_it(make_rows(defect), "fo_%s.zip" % defect)
    d = nse_data.filter_nifty_options(
        nse_data.read_bhavcopy(p, verbose=False), verbose=False)
    f = nse_data.implied_forward(d, r=R, verbose=False)
    rr = nse_data.run_quality_checks(d, fwd=f, r=R, verbose=False)
    got = rr.get(check, {}).get("ok", None)
    expect("defect %-12s caught by %r" % (defect, check[:3]), got is False,
           "- %s" % rr.get(check, {}).get("detail", "check absent"))

# ------------------------------------------------------------- wrong layout
print("\n[3] a wrong layout must fail loudly, not silently")
bad = make_rows().rename(columns={"StrkPric": "SOME_OTHER_NAME",
                                  "OptnTp": "WHATEVER"})
p = zip_it(bad, "fo_badcols.zip")
try:
    nse_data.read_bhavcopy(p, verbose=False)
    expect("raises on unresolvable columns", False, "- it did NOT raise")
except ValueError as e:
    msg = str(e)
    expect("raises on unresolvable columns", True)
    expect("names the missing canonical columns",
           "strike" in msg and "opt_type" in msg)
    expect("prints the columns actually present", "SOME_OTHER_NAME" in msg)
    expect("says how to fix it", "COLUMN_ALIASES" in msg)
except Exception as e:                                   # noqa: BLE001
    expect("raises ValueError (not %s)" % type(e).__name__, False)

print("\n[4] alias resolution is tolerant of case and separators")
leg = make_rows().rename(columns={
    "TckrSymb": "SYMBOL", "XpryDt": "EXPIRY_DT", "StrkPric": "STRIKE_PR",
    "OptnTp": "OPTION_TYP", "SttlmPric": "settle pr", "ClsPric": "CLOSE",
    "OpnIntrst": "OPEN_INT", "TtlTradgVol": "CONTRACTS", "TradDt": "TIMESTAMP",
    "UndrlygPric": "UNDERLYING_VALUE", "FinInstrmTp": "INSTRUMENT"})
leg["INSTRUMENT"] = "OPTIDX"
p = zip_it(leg, "fo_legacy.zip")
try:
    d = nse_data.filter_nifty_options(
        nse_data.read_bhavcopy(p, verbose=False), verbose=False)
    expect("legacy column names resolve", len(d) > 100, "(%d rows)" % len(d))
except Exception as e:                                   # noqa: BLE001
    expect("legacy column names resolve", False, "- %s" % e)

print("\n" + "-" * 78)
print("  %d passed, %d failed" % (passed, failed))
print("-" * 78)
raise SystemExit(1 if failed else 0)
