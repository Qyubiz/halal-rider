# Panic Buyer refinement results (22 Oct 2026)

**Verdict: keep the live bot unchanged.** Three rounds, about 200 variants (plus the 960-setting grid). None beat the live settings in
both 2021–23 and 2024–26 without a deeper worst drop. Round 1 is the section 8 ideas (below). Rounds 2–3, the
untouched 2018–20 holdout and a replay of the real bot code are summarised first, because they change what
to expect from the live bot.

## What to expect from the live bot (rounds 2–3, holdout, replay)

1. **2024–26 holds up for the real code.** `panic_buyer.py`, replayed hour by hour through the fake exchange
   over 2021–2026 on the 16 coins with complete data (`replay_bot_full.py`, `results_replay/`), made ×2.36
   in 2024–26 against the engine's ×2.38 on the same coins and candles (worst drop −14% both).
2. **2021–23 does not.** The same replay made ×3.73 (worst drop −33%) against the engine's ×5.36 (−29%).
   Two timing details explain most of it, and both matter mainly in fast crash markets like 2021:
   - After a market buy (coin already under its trigger at the open) the bot places its +1.5% sell at once.
     The backtest places it at the next hourly check, which lets the bounce overshoot. `engine.run(mtp_now=True)`
     models the bot: 19 coins ×4.73 / ×2.66 instead of ×5.75 / ×2.61.
   - The bot starts the 24h cooldown when it notices a sale (the next hourly check), one hour later than the
     backtest. On these 16 coins, engine with both details: ×3.89 / ×2.42, close to the bot's ×3.73 / ×2.36.
   - A fix for the first detail is in `patches/market_buy_tp_next_hour.diff`, **not applied**. Replayed, it gives
     ×4.52 with a −37% drop in 2021–23 and ×2.34 in 2024–26: more 2021–23 return, but a deeper drop and no
     2024–26 gain, so it doesn't pass the rule. On the 2018–20 holdout the engine version (A) beat the
     as-coded timing (B), ×2.26 vs ×2.14 with the same drop; that was the pre-registered tie-breaker
     (`ROUND3_PREREG.md`), but the full replay shows the real-code gain is smaller and riskier, so the bot was
     left as is.
3. **The 2018–20 holdout (never used before, looked at once): profitable, with a much deeper crash.** Live
   settings on the 16 coins that existed: ×2.26 (2018 +41%, 2019 +19%, 2020 +34%), 81% of months up. Worst
   drop **−45%**, all in the 12–13 March 2020 COVID crash: the bot bought 12% dips that fell another 20–26%,
   and the account took until September 2020 to recover. Outside that week, the worst drop was −23% (Jan
   2018). Plan for a −45% drop in a 2020-style crash; on RM6,000 that is about −RM2,700.
4. **Fill realism (1-minute data):** with stricter but plausible fills (limit buys need 0.3% through,
   take-profits 0.2% through, 1-minute delay), 2024–26 is ×2.31 instead of ×2.61. Realistic range for 2024–26:
   ×2.3 to ×2.7. The shallow fills a stricter rule removes are the best trades (+1.3% to +1.8% on average), so fill quality
   decides a lot of the result. `reconcile_live.py` will show the real fills after 2–4 weeks live.
5. **Noise.** Changes that shouldn't matter (depth 11.9% or 12.1%, fee ±0.005%, starting a few days later)
   move 2024–26 between ×2.52 and ×2.77 and the 2021–23 worst drop by up to 4 points. Every near miss in
   rounds 1–3 is inside that band, so they are ties, not improvements (`results_round3/noise/`).
6. **Corrected stress number.** FTT was halted on Binance from 15 Nov 2022 to 22 Sep 2023 (it reopened 32%
   lower). The backtest kept a position bought just before the halt frozen at the pre-halt price forever. Now
   it is sold when trading resumes. With LUNA/FTT/SRM added, 2024–26 is ×2.61 (RM15,653), not ×2.46 (the
   handoff's RM14,753); 2021–23 is ×5.22 with a −31% drop.

### Rounds 2–3: every idea and its verdict

Each idea: a grid of at most 8 variants written before running, chosen on 2021–23, checked on 2024–26, then
with LUNA/FTT/SRM and with fills 0.2% worse. Rounds 2 and 3 had adversarial verification and a completeness
critic (`results_round2/`, `results_round3/`, scripts `r2_*.py`, `r3_*.py`).

| Idea | Pick on 2021–23 | 2024–26 | Failed at |
|---|---|---|---|
| Break-even target after X hours | +0.5% after 6h: ×6.35 / −29% | ×2.55 | 2024–26 |
| Sell stale losers (3–10% down at 18–24h) | 18h: ×6.62 / −24% | ×2.19 | 2024–26 |
| Skip "falling knife" buys in weak markets | ×6.19 / −15% | ×2.06 | 2024–26 |
| Skip buys mid-cascade | ×6.26 / −29% | ×2.60 (tie) | 2024–26, stress checks |
| Extra second leg from free cash | 1× at 24%: ×7.51 / −30% | ×2.37 | 2024–26 |
| Split ladder 12% / 18%, 4 coins | ×9.07 / −23% | ×2.67 (tie) | LUNA/FTT/SRM: −49% drop |
| Same, capped at 25% per coin | ×8.85 / −23% | ×2.67 (tie) | LUNA/FTT/SRM: −44% drop |
| Trigger window 8–72h instead of 24h | 12h, 11%: ×6.25 / −20% | ×2.54 | 2024–26 |
| Cooldown 0–96h instead of 24h | 18h: ×6.56 / −30% | ×2.51 | 2024–26 |
| Take-profit relative to the 24h high | 0.10 × gap: ×6.65 / −29% | ×2.63 (tie) | fills 0.2% worse |
| Avoid UTC 00–07 / weekend dips | weekend 00–07 skip: ×7.49 / −24% | ×2.40 | 2024–26 |

Patterns: many filters cut the 2021–23 losers (a bear-market effect) but then give up 2024–26, where the same
conditions produced normal winners. Shorter cooldowns, larger orders and longer holds earn more only with
deeper drops. A hard stop-loss loses even in 2021–23: deep losers are fresh panics that bounce.

## Round 1: section 8 ideas

Data: Binance spot candles, hourly and 1-minute, 1 Jan 2021 to 22 Oct 2026, downloaded fresh from
data.binance.vision. Round 1 ran on GitHub Actions (`.github/workflows/panic-research.yml`); logs and
tables are in `results/`. This session can't reach Binance, so `.github/workflows/panic-data.yml` publishes
the candles as the pre-release `pb-data-v1` (about 820 MB, safe to delete when the research is done).

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
