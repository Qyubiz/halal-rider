# Panic Buyer refinement results (22 Oct 2026)

**Verdict: keep the live bot unchanged.** 105 variants across the section 8 ideas were tested. None beat the
live settings in both 2021–23 and 2024–26 without a deeper worst drop.

Data: Binance spot candles, hourly and 1-minute, 1 Jan 2021 to 22 Oct 2026, downloaded fresh from
data.binance.vision. Everything ran on GitHub Actions (`.github/workflows/panic-research.yml`); logs and
tables are in `results/`.

## The rule (fixed before any result was seen)

1. Each idea gets a small set of variants, written at the top of its script.
2. The variant is chosen on **2021–23 only**. It must beat the live settings there, stay above −5% in 2022, and
   keep its 2021–23 worst drop above −35%. The best return per unit of drop wins.
3. 2024–26 is checked once, for that pick. "Beats" = higher compounded return **and** a worst drop no more than
   1 point deeper, in both periods.
4. It must still beat the live settings with LUNA/FTT/SRM added and with every fill 0.2% worse.

## Live settings, reproduced

| | 2021–23 | 2024–26 |
|---|---|---|
| Multiple | ×5.75 | ×2.61 (RM6,000 → RM15,653) |
| Worst drop | −29% | −16% |
| 2022 | +19% | |

Per year: 2021 +262%, 2022 +19%, 2023 +33%, 2024 +71%, 2025 +28%, 2026 to date +19%. Since 2024: 884 trades,
93% wins, average +1.09%, worst −22%. Monthly returns match the handoff's `monthly_results.csv` to 0.06 points
on average (largest gaps 1.5 points, Jan and Apr 2021). The new engines reproduce `code/lab.py` exactly
(`check_engine.py`); the 1-minute engine gives ×5.74 / ×2.61 for the same bot.

## Each idea

| Idea | Variants | Best on 2021–23 | 2024–26 | Verdict |
|---|---|---|---|---|
| 2 Faster exits (take-profit 1/5/15 min after the fill, 1-minute candles) | 12 | none beat live | all lower: ×1.58–2.46 | rejected |
| 2b Slower first exit (follow-up: wait 2–6 hourly checks, or sell only at hourly checks) | 5 | none beat live (all deeper drops) | ×2.54–2.86, worst drop −20 to −21% | rejected |
| 3 Trigger depth by volatility or drop percentile | 40 | none beat live | all lower: ×1.34–2.61 | rejected |
| 4 Stress-aware sizing (BTC falling) | 20 | calm-deeper 2%/+4%: ×5.77 | ×2.14 | rejected |
| 5 Monthly top-M coins by volume (321 screened coins, delisted included) | 14 | 19 coins + top 10: ×6.22 | ×2.56 | rejected |
| 6 Exit variants (partial take-profit; shorter time stop in weak markets) | 14 | breadth ≥85% → 24h stop: ×6.42 | ×2.26 | rejected |

What the tests show:

- **Faster exits lose money.** 60% of take-profits fill at the first hourly check after the buy, at +2.1% on
  average instead of +1.5%: the bounce overshoots, and the live bot's sell, placed at the next hour, fills at
  the higher market price. Selling minutes after the fill caps that at +1.5%.
- **Every variant that earned more also dropped deeper.** 12 variants had higher returns in both periods
  (waiting longer to sell, fewer/bigger orders when BTC falls, partial take-profit with a second target), for
  example ×7.90 / ×2.86 for "no resting take-profit". All of them had worst drops of −31% to −40% (2021–23)
  and up to −21% (2024–26). None improved return per unit of drop. They are more risk, not a better bot; and
  this list was found by looking at both periods, so it is not a fair pick.
- **Volatility-scaled triggers** cut drawdowns but cut returns much more (best 2024–26 ×2.61 at a −20% drop).
- **Wider or rotating coin lists** add the small-coin crashes the handoff already found.

## Statistical checks on the live settings (`idea8_stats.py`)

- **Plateau, not a peak.** 91% of the 960 settings were profitable in 2024–26; all 90 that pass the 2021–23
  rule were, median +38%/year. The heatmaps (`results/heatmaps.png`) show a broad green area around 10–12%
  depth.
- **Overfitting probability (CSCV):** 13%. The best setting in one half of the data stays good in the other
  half (out-of-sample Sharpe above 0 in every split).
- **Deflated Sharpe:** 0.90 over 2021–26 (960 settings tried), 0.86 counting 1,000 more tries. On 2024–26
  alone it is only 0.26–0.34: under three years of data with fat-tailed days isn't enough on its own.
- **Yearly re-optimizing hurts.** Choosing each year's settings with the same rule on all earlier years picks
  concentrated settings (1–2 coins, 15% depth, +5%): 2022–26 ×5.51 with a −48% drop, against ×4.14 and −19%
  for the fixed live settings.
- **Next 12 months, bootstrapped from monthly returns (3-month blocks):** from the 2024–26 months, median +38%,
  5th percentile +19%; from the 2022–23 bear-market months only, median +25%, 5th percentile +4%, 2% chance of
  a losing year. Monthly data hides drops inside a month, so use the backtest's −16% / −29% as the drawdown guide.

## Costs (idea 9)

| | 2021–23 | 2024–26 |
|---|---|---|
| Live | ×5.75 | ×2.61 |
| BNB pays fees (0.075% a side) | ×6.33 | ×2.73 |
| Every fill 0.1% worse | ×4.79 | ×2.51 |
| Every fill 0.2% worse | ×3.98 (2022 −2%) | ×2.37 |
| Every fill 0.5% worse | ×2.23 (drop −39%) | ×1.80 |
| BNB fees + fills 0.2% worse | ×4.38 | ×2.48 |

The BNB discount helps in both periods with the same drawdown, but it is an account setting, not a strategy
change, and the bot would need a small fix first: it assumes the fee is taken in the coin, so with BNB fees it
would leave about 0.1% of every buy unsold as dust and slightly misstate P&L. Not changed.

Fill quality matters as much as anything tested: fills 0.2% worse take 2021–23 from ×5.75 to ×3.98 and 2024–26
from ×2.61 to ×2.37. The first live weeks (idea 1) will show what the real fills are.

## Not tested

- **Idea 1, live vs backtest:** needs live trades. Use `reconcile_live.py` after 2–4 weeks live.
- **Idea 7, guards (delisting schedule, monitoring tag, maintenance):** none of the 19 coins was delisted, so
  a backtest can't measure it. Worth adding as a safety feature before widening the coin list.
- **Idea 9, slippage from real order books:** no historical order-book data; covered by the worse-fill rows.

## Fixed along the way

Joining RENDER's old ticker (RNDR) to the new one used an unstable sort, which can mix the two price series
where the dates overlap. In the real data they don't overlap (RNDR ends 22 Jul 2024, RENDER starts
26 Jul 2024), so it didn't change the numbers: the reproduction matches the handoff's ×2.61.
