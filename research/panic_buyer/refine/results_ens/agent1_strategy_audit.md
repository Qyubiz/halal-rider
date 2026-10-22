# Agent 1: Strategy and backtest audit (Panic Buyer, /home/user/halal-rider)

Every number below comes from repo files I cite or from scripts I ran. The two scripts (new files, nothing existing was edited, nothing committed) are:
- `/home/user/halal-rider/research/panic_buyer/refine/ens_audit_modes.py`: re-runs the three modes over all periods, under stress and on the holdout.
- `/home/user/halal-rider/research/panic_buyer/refine/ens_audit_overlap.py`: measures how much 90% mode and 10% mode overlap.

Outputs are in `/tmp/claude-0/-home-user-halal-rider/ec562e35-ea9e-5a74-ab92-4ff22c5eeb77/scratchpad/pb_out_ens/audit/`: `modes_audit.log`, `modes_audit.csv`, `overlap.log` and `monthly_returns_modes.csv`.

**Reproduction check.**
- The `normal` rows match `results_w90/check90.log` on every printed figure: LIVE, 10% mode and 90% mode, in 2021-23, 2024-26 and since 2025.
- The holdout rows match `results_w90/holdout_w90.log` exactly when computed the way that log does (`engine.metrics`): LIVE +2.30%/mo, 927 trades; 90% mode (F2) +10.88%/mo, 361 trades; 10% mode +11.09%/mo, 215 trades; fills 0.2% worse +1.70 / +9.59 / +10.16.
- `engine.run` with the crash guard applied as an smult mask gives the same equity curve and trade log as `w90_wr90_lib.wrun`. The script asserts this on every unstressed run.

## 1. Strategy inventory

| # | Strategy | Where | Status |
|---|---|---|---|
| A | **Halal pump-rider**: momentum breakout on 5-minute candles | `rider.py:1-13`, params `rider.py:44-53` | Live bot (separate service) |
| B | **Panic Buyer, Normal**: buys dips under the 24h high, sells the bounce | `panic_buyer.py:54-66` (DEFAULT_CFG) = `engine.py:33` (LIVE) | Live default |
| C | **Panic Buyer, 90% mode** (same code, config only) | `panic_config.90wr.example.json:2-8`; research `w90_wr90_final.py` SPEC "F2" | Opt-in |
| D | **Panic Buyer, 10% mode** (same code, config only) | `panic_config.10pct.example.json:2-8`; research `x10_*` "F3 resting bid" | Opt-in |
| E | Research families, rejected or not deployed | `HANDOFF.md` §4; `REFINE.md` lines 66-71, 96-146, 175-197 | Research only |

**A. Pump-rider** (`rider.py:1-13`, `44-53`):
- **When it trades:** only while BTC is above its 50-day EMA.
- **Signal:** a closed 5-minute candle up at least 2% over 15 minutes, with volume at least 3x the 4h average, on a green candle.
- **Entry:** a stop-buy 1% above the signal close, valid 60 minutes, with the limit at most 0.3% above the trigger.
- **Exits:** an exchange-side stop-loss 5% under entry, trailed to 8% under the high, and a 24h maximum hold.
- **Sizing:** at most 5 positions at 1/5 of capital each.
- **Halt:** no new trades after a 10% weekly loss (`weekly_halt_pct`).
- **Not backtested here:** the task scopes the backtests to the panic buyer.

**E. Research families** (all from the docs):
- **HANDOFF.md §4, failed:**
  - Indicator confluence: 63-78% win rate but negative expectancy.
  - EMA cross + ATR trailing stop.
  - Grid trading and rebalancing.
  - Momentum rotation (RM5.8k, -66%).
  - Breadth switch (-59%).
  - BTC-to-alt lead-lag.
  - Funding-rate signals.
  - 15-minute checks.
  - An 8% trigger.
- **REFINE.md rounds 1-3:** about 200 variants. None beat live in both periods without a deeper drop.
- **w90_* families:** wr90, watch, wider universe, tenh.
- **x10_* families:** cycle, idle, stack, universe, velocity.
- **hr_* high-return test:** concluded "10% a month: not reached in 2024-26 by any of 1,056 settings" (REFINE.md:105).

## 2. Exact logic of the panic-buyer modes (shared code path)

**Trigger** (`panic_buyer.py:192`; engine `engine.py:68,99,109`):
- trigger = highest hourly close of the last 24 hours × (1 − depth).

**Coin selection** (`panic_buyer.py:193-194, 313-315`; engine `100-104`):
- Coins are ranked by distance to their trigger, and the `n_orders` closest are taken.
- Coins already held, with an open order, or in cooldown are skipped.

**Crash guard** (`panic_buyer.py:186-191`; engine mask `w90_wr90_lib.py:175-185`):
- A coin is skipped if its last close is more than `crash_guard_pct` under the highest close of the last 168 hours.

**Sizing** (`panic_buyer.py:171-177, 318-320`; engine `106, 117`):
- per order = min(max_order_pct × bot money, free cash / number of selected coins).
- There is no volatility or stop-distance sizing.

**Entry** (`panic_buyer.py:327-342`; engine `111-114`):
- If the price is already under the trigger, the bot buys at market.
- Otherwise it places a GTC limit buy at the trigger, which is cancelled at the next hourly cycle (`check_orders`, `258-279`).

**Exit** (`panic_buyer.py:210-222, 281-308`; engine `124-133, 152-153`):
- **Take-profit:** a limit sell at entry × (1 + take_profit_pct).
  - After a market buy it goes up at once (`open_position`, `207`).
  - After a limit fill it goes up at the next hourly cycle.
- **Time stop:** after `max_hours` the position is sold at market (`301-302`).
- **There is no price stop-loss in any mode.** The only loss exit is the time stop. The 30% "catastrophe stop" in research variant F1 was never coded (`w90_wr90_final.py:6`).

**Cooldown** (`panic_buyer.py:127`; engine `159`):
- A coin is skipped for `cooldown_hours` after it is sold.

**Run cycle** (`panic_buyer.py:347-351, 486`):
- Every hour, 10 seconds after the candle close: check orders, then manage positions, then place orders.

**Risk controls that exist today:**
- `CAPITAL_USDT` cap (`budget()` `171-177`).
- Buy cash capped at 98% of free USDT (`319`).
- Crash guard.
- Time stop.
- Manual `pause`/`panic` commands (`370-395`).

**Risk controls that do not exist:** drawdown kill-switch, per-trade risk cap, loss-streak cooldown, spread, ATR or latency filter.

| Parameter | Normal | 90% mode | 10% mode |
|---|---|---|---|
| depth_pct (dip under 24h high) | 12 | 14 | 14 |
| n_orders × max_order_pct | 8 × 20% | 1 × 100% | 1 × 100% |
| take_profit_pct | 1.5 | 2.5 | 5.0 |
| max_hours (time stop) | 48 | 96 | 48 |
| cooldown_hours | 24 | 3 | 24 |
| crash_guard_pct | 0 (off) | 35 | 35 |
| Source | `panic_buyer.py:58-65` | `panic_config.90wr.example.json:2-8` | `panic_config.10pct.example.json:2-8` |

**90% mode and 10% mode use the same entry rule.** They differ only in take-profit, time stop and cooldown.
- 312 entries are on the same coin in the same hour. That is 75.7% of the 10% mode's 412 entries and 56.3% of the 90% mode's 554.
- Correlation of monthly returns between the two modes: 0.810 over 2021-26, 0.568 in 2024-26, 0.700 since 2025 (`overlap.log`).
- Both take all the money on one coin, so they compete for the same capital rather than diversifying each other.

## 3. Performance: engine re-run, 19 coins, backtest timing (`modes_audit.log`, "normal" rows)

**How to read the columns:**
- **PF (profit factor)** = gross profit / gross loss over trades, with each trade's P&L taken as a % of the account at entry.
  - The equal-weight per-trade PF is identical for the two 1 × 100% modes.
  - For Normal, the equal-weight per-trade PF is 2.32 / 2.99 / 2.71 for the three periods, and 1.76 on the holdout.
- **Worst trade** = the worst single trade's loss as a % of the account.
- **Trades** are assigned to a period by their entry hour.
- **Holdout** = 2018-01 to 2020-12 on the 16 coins that existed then. Its %/month follows the `holdout_w90.log` convention.

| Mode | Period | Win rate | %/month | Max drawdown | Profit factor | Trades | Worst trade (% acct) |
|---|---|---|---|---|---|---|---|
| Normal | 2018-20 holdout | 90.9% | +2.30 | −45.0% | 1.71 | 927 | −5.4% (BNB, Feb 2018) |
| Normal | 2021-23 | 93.6% | +4.98 | −29.2% | 2.25 | 1681 | −5.3% (RENDER, Nov 2022) |
| Normal | 2024-26 | 92.5% | +2.89 | −16.3% | 2.92 | 884 | −2.7% (HBAR, Apr 2024) |
| Normal | since 2025 | 91.5% | +1.96 | −15.5% | 2.30 | 492 | −2.0% (RENDER, Jan 2026) |
| **90% mode** | 2018-20 holdout | 92.5% | +10.88 | −66.3% | 2.01 | 361 | −37.3% (FET, Mar 2020) |
| **90% mode** | 2021-23 | 93.8% | +13.17 | −64.4% | 2.17 | 406 | **−50.3%** (DOGE, 29 Jan 2021) |
| **90% mode** | 2024-26 | 92.6% | +6.89 | −35.4% | 3.20 | 148 | −27.3% (HBAR, Apr 2024) |
| **90% mode** | since 2025 | 93.2% | +5.80 | −22.6% | 4.17 | 73 | −11.9% (BCH, Jun 2026) |
| **10% mode** | 2018-20 holdout | 80.5% | +11.09 | −61.3% | 2.10 | 215 | −30.0% (FET, May 2019) |
| **10% mode** | 2021-23 | 81.1% | +15.03 | −60.6% | 2.09 | 297 | **−54.1%** (DOGE, 29 Jan 2021) |
| **10% mode** | 2024-26 | 76.5% | +8.13 | −29.7% | 3.06 | 115 | −21.7% (HBAR, Apr 2024) |
| **10% mode** | since 2025 | 75.0% | +6.08 | −28.7% | 3.10 | 56 | −11.2% (NEAR, Nov 2025) |

**Average win and loss per trade (% of order), 2021-23 / 2024-26:**
- Normal: wins +1.84 / +1.77, losses −11.53 / −7.32.
- 90% mode: wins +2.70 / +2.52, losses −18.97 / −9.83.
- 10% mode: wins +4.84 / +4.73, losses −9.98 / −5.04.

**Hindsight:** the 90% and 10% modes were picked by looking at 2021-26. Only the 2018-20 holdout is a test they were not picked on (REFINE.md:43-76; `w90_summary`).

**Against the targets:**
- **About 90% win rate:** 90% mode holds it in every period (92.5-93.8%).
- **About 10% a month:** 10% mode reaches it only on the 2018-20 holdout (+11.09) and in 2021-23 (+15.03). It does not in 2024-26 (+8.13) or since 2025 (+6.08).

## 4. Stress results (`modes_audit.log`)

The stress conditions are:
- **Bot timing:** take-profit at once after a market buy (`mtp_now`), and cooldown +1h, as the code actually behaves.
- **Buys 0.2% worse:** `pen=0.002`.
- **Full stress:** buys and sells 0.2% worse, plus LUNA/FTT/SRM, sold at their last bar.

| Mode | Stress | 2021-23 (%/mo · WR · DD · PF · worst trade) | 2024-26 | since 2025 |
|---|---|---|---|---|
| Normal | bot timing + buys 0.2% worse | +3.10 · 92.6% · −35.4% · 1.70 · −5.2% | +2.66 · 92.0% · −16.2% · 2.83 | +1.72 · 90.4% · −15.7% · 2.12 |
| Normal | full stress | +1.72 · 92.7% · −42.7% · 1.35 · −20.0% (LUNA) | +2.11 · 92.0% · −16.4% · 2.41 | +1.24 · 90.4% · −15.9% · 1.79 |
| 90% | bot timing + buys 0.2% worse | +13.78 · 94.2% · −63.3% · 2.32 · −45.8% | +6.40 · 91.7% · −28.5% · 3.02 | +5.09 · 92.7% · −22.7% · 3.74 |
| 90% | full stress | +10.96 · 93.7% · −65.0% · 2.05 · −45.9% | +5.49 · 91.7% · −30.3% · 2.71 | +4.44 · 92.7% · −22.7% · 3.35 |
| 90% | +LUNA/FTT/SRM, **no crash guard** | −23.20 · 93.6% · **−100%** · 1.75 · **−100%** (LUNA) | +6.29 | +4.77 |
| 10% | bot timing + buys 0.2% worse | +14.39 · 79.9% · −64.0% · 2.03 · −44.4% | +7.19 · 76.1% · −29.1% · 2.63 | +5.14 · 73.2% · −28.8% · 2.46 |
| 10% | full stress | +12.20 · 79.9% · −74.5% · 1.80 · −44.5% | +6.47 · 76.1% · −29.8% · 2.44 | +4.60 · 73.2% · −29.0% · 2.27 |
| 10% | +LUNA/FTT/SRM, **no crash guard** | −10.25 · 81.8% · **−100%** · 1.70 · −99.98% (LUNA) | +7.75 | +5.51 |

**Holdout under stress (bot timing + buys 0.2% worse):**

| Mode | %/month | Win rate | Drawdown | PF |
|---|---|---|---|---|
| Normal | +1.46 | 89.9% | −45.1% | 1.43 |
| 90% mode | +8.39 | 91.7% | −68.7% | 1.80 |
| 10% mode | +10.19 | 79.9% | −61.6% | 2.01 |

**Real `panic_buyer.py` replayed** (16 coins; logs in the repo):
- 90% mode (`results_w90/replay_90wr_mode.log`):
  - Return +15.4 / +9.0 / +5.8%/mo, drawdown −49 / −28 / −28%.
  - Win rate 93.0% of 626 closed trades from 2021, 94.3% of 281 from 2024, 93.6% of 140 from 2025.
- 10% mode (`results_x10/replay_10pct_mode.log`): +15.7 / +9.6 / +6.5%/mo, drawdown −51 / −28 / −28%.

**Worst months** (monthly returns from Feb 2021, so the January 2021 DOGE month is not in this window; `overlap.log`):

| Mode | Worst month | Negative months (of 69) |
|---|---|---|
| Normal | −7.0% | 11 |
| 90% mode | −30.2% | 13 |
| 10% mode | −20.5% | 14 |

Both 90% and 10% mode were negative in the same month 7 times.

## 5. Findings for the synthesis and risk agents

1. **The high-return modes earn by concentrating money, not by a better entry.**
   - Both modes put one coin × 100% on the same trigger the live bot uses, only deeper (14%).
   - Profit factor in the period they were not picked on: 2.01 (90% mode) and 2.10 (10% mode). Live Normal: 1.71.
   - The cost is drawdown: about −61% to −66%, against −45% for Normal.
2. **Worst trades are 50-54% of the account** (DOGE, January 2021), because there is no price stop. The only loss exit is the 48h or 96h time stop.
   - REFINE.md:144-145 reports that a hard stop-loss lost money on 2021-23: deep losers bounced.
   - So a per-trade risk cap is really a position-size cap (stop distance × size), not a tight stop.
3. **The crash guard is what prevents total loss.** Without it, LUNA takes either mode to −100%. Any ensemble must keep it.
4. **A 90% win rate combined with 10% a month has never been shown.**
   - From `w90_summary.txt`: "10% a month is not reached at that win rate in any of 1,328 designs (best 2024-26 +7.3%, best since 2025 +5.85%)".
   - From REFINE.md:105: "not reached in 2024-26 by any of 1,056 settings".
   - Run as an ensemble on one bankroll, the two modes would mostly take the same entries (75.7% of the 10% mode's entries are shared).
