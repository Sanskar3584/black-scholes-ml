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

## 5. Next actions

| Action | Owner | By |
|---|---|---|
| Download one sample month of NIFTY option bhavcopy | Sanskar | 3 Nov |
| Confirm the field names present in that file against section 2.1 | Sanskar | 3 Nov |
| Download RBI 91-day T-bill series for the same window | Sanskar | Week 6 |
| Download NSE NIFTY 50 dividend-yield series | Sanskar | Week 6 |
| Agree the dataset specification in section 4 | supervisor review | Week 6 |
| Download the remaining 11 months | Sanskar | Week 6 |

No cleaning, filtering or analysis has been done, in line with the Week 5 instruction that
this task is sources and access notes only.
