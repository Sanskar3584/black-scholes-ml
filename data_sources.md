# Data sources for the real-market stage

**Ref:** Week 5 task sheet, Task 5 (1 hour, sources and access notes only, no analysis)
**Target:** 12 months of NIFTY 50 monthly options for the Weeks 6 to 9 stage
**Status:** sources identified; **download outstanding - see "Access notes" below**

---

## 1. What the pricing stack needs

Every model in this project takes five inputs. The table lists where each comes from.

| Input | Symbol | Source | Frequency |
|---|---|---|---|
| Option market price | `C_market` | NSE F&O bhavcopy | daily, per contract |
| Underlying spot | `S` | NSE NIFTY 50 close | daily |
| Strike | `K` | bhavcopy | per contract |
| Time to maturity | `T` | expiry date minus trade date | derived |
| Risk-free rate | `r` | RBI 91-day Treasury bill yield | weekly auction |
| Dividend yield | `q` | NSE NIFTY 50 index dividend yield | daily/monthly |

`q` matters here: NIFTY is a price index whose constituents pay dividends, so the
dividend-adjusted forward is $Se^{-qT}$ rather than $S$. All five pricers accept `q`
(verified in Week 5, Task 4), so the stack is ready for it.

---

## 2. Sources

### 2.1 NIFTY option prices - NSE F&O bhavcopy

- **What:** end-of-day record of every derivative contract traded.
- **Where:** NSE archives, "F&O Bhavcopy" under historical reports. The UDiFF format
  replaced the older CSV layout in July 2024, so a 12-month window spanning that change
  needs a parser for both.
- **Granularity:** one row per contract per day.
- **Cost:** free.

Expected fields in the F&O bhavcopy (names differ slightly between the legacy and UDiFF
layouts):

| Field | Meaning | Use |
|---|---|---|
| `INSTRUMENT` | OPTIDX for index options | filter to OPTIDX |
| `SYMBOL` | underlying, e.g. NIFTY | filter to NIFTY |
| `EXPIRY_DT` | contract expiry | gives `T` with the trade date |
| `STRIKE_PR` | strike price | `K` |
| `OPTION_TYP` | CE or PE | call or put |
| `OPEN`, `HIGH`, `LOW`, `CLOSE` | daily prices | see note below |
| `SETTLE_PR` | settlement price | preferred price field |
| `CONTRACTS`, `VAL_INLAKH` | volume | liquidity filter |
| `OPEN_INT`, `CHG_IN_OI` | open interest | liquidity filter |
| `TIMESTAMP` | trade date | gives `T` with the expiry |

**Which price to use.** The task sheet for later weeks asks for a defensible choice.
`CLOSE` is the last traded price and can be stale for illiquid strikes - a contract that
last traded hours before the close carries a price inconsistent with the closing spot.
`SETTLE_PR` is the exchange's settlement price and is defined for every contract. The
intention is to use `SETTLE_PR` as the primary field and to record `CLOSE` alongside it so
the two can be compared. A mid-quote would be better still, but bid-ask data is not in the
bhavcopy.

### 2.2 Risk-free rate - RBI 91-day Treasury bill

- **What:** weighted average yield at the weekly 91-day T-bill auction.
- **Where:** RBI Database on Indian Economy (DBIE), Treasury bill auction results; also
  published in the RBI Weekly Statistical Supplement.
- **Frequency:** weekly. Daily values obtained by forward-filling to the trade date.
- **Conversion:** quoted as an annualised yield in per cent; convert to a continuously
  compounded decimal rate, $r = \ln(1 + y/100)$, before use.

The 91-day tenor is shorter than some option maturities. For a 3-month window it is the
right instrument; for longer maturities the 364-day bill is closer. Which to use is a
decision to record in the Week 6 dataset documentation rather than leave implicit.

### 2.3 Dividend yield - NSE NIFTY 50 index

- **What:** trailing twelve-month dividend yield of the index.
- **Where:** NSE indices historical data, which publishes daily P/E, P/B and dividend
  yield alongside the index close.
- **Frequency:** daily.
- **Note:** a trailing yield is backward-looking, whereas pricing needs the dividend
  expected over the option's life. For a broad index over a 1 to 3 month horizon the
  difference is small, but it is an approximation and should be stated as one.

### 2.4 Underlying spot - NIFTY 50 close

- Already in the project: `data/nifty50_prices.csv`, 2,459 daily closes from 2 January
  2015 to 31 December 2024, downloaded via `yfinance` ticker `^NSEI` in August Week 1.
- For the option stage the spot must come from the **same source and timestamp** as the
  option prices, so the NSE index close is preferable to the Yahoo close. Mixing the two
  risks a small timing mismatch that would appear as a pricing error.

---

## 3. Access notes

**The download has not yet been done.** Two points to flag now, as the task sheet asks.

**NSE blocks programmatic access.** Direct requests to `nseindia.com` are rejected without
browser-like headers and a session cookie obtained from the homepage first. A plain
`requests.get` or `pandas.read_csv` on the archive URL returns 403. Options, in order of
preference:

1. Download the bhavcopy archives manually from the NSE site - reliable, and for 12 monthly
   expiries that is a manageable number of files.
2. Use a maintained wrapper library that handles the session handshake.
3. Scripted download with a browser session, which is fragile and tends to break when NSE
   changes its site.

Option 1 is the recommendation: it is a one-off cost of perhaps an hour, and it removes a
dependency that could fail silently later.

**The file format changed in July 2024.** NSE moved to the UDiFF format. A window of 12
months ending in late 2026 sits entirely after the change, so one parser suffices - but the
window should be chosen deliberately rather than by accident.

---

## 4. Proposed dataset specification

| Decision | Proposal | To confirm at review |
|---|---|---|
| Window | 12 consecutive months of monthly expiries | start month |
| Contract type | OPTIDX, NIFTY, monthly expiries only | exclude weeklies? |
| Price field | `SETTLE_PR`, with `CLOSE` retained for comparison | yes |
| Moneyness filter | retain $\lvert\ln(S/K)\rvert \le 0.3$ | threshold |
| Liquidity filter | drop contracts with zero volume or open interest | thresholds |
| Maturity filter | drop $T <$ 7 days | avoids the expiry-week distortion |
| Rate | RBI 91-day T-bill, forward-filled, continuously compounded | tenor choice |
| Dividend yield | NSE trailing index yield, daily | acceptable approximation? |

**Why the moneyness filter matters, and it is not arbitrary.** Week 3 established that
implied volatility is only as well determined as vega allows, since
$\delta\sigma \approx \text{residual}/\nu$, and that in the lowest vega band the recovery
error reached 4.4e-04 while 28.6% of cases needed the bisection fallback. Deep-wing
contracts therefore carry far less information than their quoted decimal places suggest.
Including them unfiltered would let the noisiest observations dominate any model fitted on
implied volatility. The filter is a consequence of a measured result, not a convention.

---

## 5. Download recipe

*Added 11 November 2026. The download is still outstanding and is now the only thing
blocking the real-data stage, so the steps are written out in full rather than left as
"download the bhavcopy".*

### 5.1 Choose the window first

Pick **twelve consecutive monthly expiries ending with the last expiry before the download
date**. Two constraints fix the choice:

- the window must sit **entirely after July 2024**, so that one UDiFF parser handles every
  file (section 3);
- it must end on a **completed** expiry, so that every contract in the sample has a known
  settlement.

A window of November 2025 to October 2026 satisfies both. Write the chosen start and end
months at the top of `data/raw/README.txt` so the sample is reproducible.

### 5.2 Get the option prices

NSE publishes a daily F&O bhavcopy. For twelve monthly expiries, what is needed is the
file for each **trading day** in the window, or - if that proves too many files - the file
for the last trading day of each week plus every expiry day.

1. Go to **nseindia.com -> Reports -> Derivatives -> Daily Reports -> F&O Bhavcopy**
   (currently `https://www.nseindia.com/all-reports-derivatives`).
2. Set the date and download the **"F&O - UDiFF Common Bhavcopy Final"** ZIP.
3. Save each ZIP **unextracted** into `data/raw/fo_bhavcopy/`, named by its date, for
   example `fo_20261030.zip`. Keeping the original archives means the parser can be
   re-run later without re-downloading anything.

Do **not** unzip into the repository. The parser will read straight from the ZIP.

> A browser session is required: a plain `requests.get` returns HTTP 403 (section 3).
> There is no way around this from a script without impersonating a browser, which is
> fragile and worth avoiding.

### 5.3 Get the risk-free rate

RBI publishes 91-day Treasury bill auction yields.

1. **rbi.org.in -> Statistics -> Weekly Statistical Supplement**, or the Database on
   Indian Economy at `dbie.rbi.org.in`.
2. Export the **91-day T-bill implicit yield at cut-off price**, weekly, for the same
   window plus one month either side (the extra month makes forward-filling safe at the
   edges).
3. Save as `data/raw/rbi_tbill_91d.csv`.

### 5.4 Get the dividend yield

1. **nseindia.com -> Market Data -> Indices -> NIFTY 50 -> Historical P/E, P/B and
   Dividend Yield.**
2. Export the daily series for the window.
3. Save as `data/raw/nifty_dividend_yield.csv`.

### 5.5 Get the underlying spot

Already available: `week1.py` caches NIFTY 50 closes to `data/nifty50_prices.csv`. If the
window extends past that cache, re-run `week1.py` with the end date advanced; it will
re-download and re-cache.

### 5.6 Sanity checks before any analysis

Run these on the first file, before downloading the other eleven. Finding a problem now
costs one download; finding it later costs twelve.

| Check | Expected |
|---|---|
| Column names match section 2.1 | `TckrSymb`, `XpryDt`, `StrkPric`, `OptnTp`, `SttlmPric`, `OpnIntrst`, `TtlTradgVol` or the equivalents |
| `TckrSymb == "NIFTY"` rows present | several thousand per day |
| Both `CE` and `PE` present | yes |
| Strike spacing | 50 points near the money |
| `SttlmPric` non-zero for liquid strikes | yes |
| Put-call parity on the same strike and expiry | holds to a few index points after the filters in section 4 |

That last check is the important one, and it is the first genuine test of the real data.
It is also the first point in the project where parity will **not** hold exactly - bid-ask
spread, non-synchronous quotes and stale settlement prices all break it. The size of the
violation is a useful measure of data quality, which is why it is worth recording rather
than filtering away silently.

---

## 6. Next actions

| Action | Owner | Status |
|---|---|---|
| Download one sample month of NIFTY option bhavcopy | Sanskar | **outstanding - blocks Week 8** |
| Run the section 5.6 checks on that month | Sanskar | after the download |
| Confirm field names against section 2.1 | Sanskar | after the download |
| Download RBI 91-day T-bill series | Sanskar | outstanding |
| Download NSE NIFTY 50 dividend-yield series | Sanskar | outstanding |
| Agree the dataset specification in section 4 | supervisor review | pending |
| Download the remaining 11 months | Sanskar | after the checks pass |

No cleaning, filtering or analysis has been done, in line with the Week 5 instruction that
this task is sources and access notes only.
