# Round m20 pre-registration: smaller losing months at the same profit (written before any m20 run)

Goal (user request): cut 90% mode's losing months while keeping its profit. Prediction of single losers failed
(m10-m16); what did work in passing was betting less on buys that carry more risk (m12 sizing, m16 risk model).
This round tests that directly, with a rule a small bot can run.

Tool: 90% mode unchanged (same coin, same bid, same exits, so the same trades and win rate). When a risk flag is
on at the bid hour, the order uses a fraction f of the money (f = 0.5 or 0.33); the rest stays in USDT.
Flags (all from closed hourly candles before the bid; "no history yet" = flag off):
- A coin under its 7-day high: last close more than 15 / 20 / 25 / 30% under its highest close of 168h.
- B coin under its 30-day high: more than 30 / 40 / 50% under its highest close of 720h.
- C market volatility: median over the coins of the 168h volatility of hourly returns above its 80% / 90%
  quantile of the previous 365 days (needs 180 days of history).
- D breadth: 3 / 5 / 8 or more coins at least 10% under their 24h highest close.
- E BTC fall: BTC down 5 / 8% or more over 24h, or 10 / 15% or more over 72h.
- F calm coin: the coin's 7-day volatility (m10 `c_vol7`) at or under 7 / 9%.
- G gap: the coin's last close already 2 / 5% or more under its bid price.
- H (reference only) the m16 walk-forward risk model, score above its trailing 80 / 90% quantile. It needs
  LightGBM and monthly retraining in the bot, so a simple flag is preferred: H is picked only if no flag A-G
  qualifies.
- Pairs: flag X OR flag Y for the two best qualifying single flags from different groups (both floors).
At most 60 designs. Engine: `m16_riskmodel_lib.srun` (90% mode with a size table; equal to 90% mode at size 1).

## Selection (2021-23 only)
Each design and 90% mode are run in 5 nearby versions (base, cooldown -1h, +1h, bot timing, start 2021-01-15)
with normal fills and with every fill 0.2% worse; each version is compared with 90% mode run the same way.
- Profit kept: mean over versions of (design %/month - 90% mode %/month) >= -0.25 points, normal AND worse fills.
- Month loss = sum of the negative calendar-month returns. Score = mean over versions (normal fills) of the
  relative cut in month loss; the worst month and the worst drop must not be worse than 90% mode's on average.
- Neighbours (same flag at the adjacent thresholds, and the other floor): more than half must keep the profit.
- Pick = highest score among qualifying designs A-G and pairs (H only if none qualifies).

## One look and the pass rule (pick vs 90% mode, same engine)
1. Profit kept: %/month >= 90% mode - 0.25 points in 2024-26 and in 2018-20, normal fills and 0.2%-worse fills.
2. Losses cut in 2024-26 and in 2018-20 (normal fills): month loss at least 15% smaller, worst month not worse,
   worst drop not deeper.
3. Stressed (bot timing, worse fills, LUNA/FTT/SRM), 2021-23 and 2024-26: %/month >= 90% mode - 0.5, worst drop
   not deeper.
4. Neighbours: more than half of the pick's neighbours also meet 1 and 2.
5. Luck check (reported; decides the wording "proven" vs "plausible"): among 90% mode's trades in each test
   period, the mean return of the flagged trades against 10,000 random flag sets of the same size; one-sided
   p < 0.05 in 2024-26 and in 2018-20.
Decision fixed now: items 1-4 pass -> add the flag to the bot as an opt-in setting (off by default, live default
unchanged) with a paper bot; otherwise no tool. Disclosure: the flag family was inspired by m12/m16, which had
seen the test periods, so live paper trading stays the final test.
