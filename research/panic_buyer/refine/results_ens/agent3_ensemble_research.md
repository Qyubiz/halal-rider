# Agent 3: ensemble and guardrail backtests (Panic Buyer, Binance spot)

**Bottom line:** I combined the two modes as two cash sleeves: 30% of the money runs 90% mode and 70% runs 10% mode, with free cash rebalanced monthly. On the 2021-23 rule this is the best design, but it only slightly beats the two modes alone. All the backtests I ran (406 runs) led to the same trade-off for the guardrails. The settings picked on 2021-23 are X=15%, Y=25%, S=25%, N=2, T=72h. They cut the 2021-23 max drop from -62% to -46%, and from -63% to -50% in the untouched 2018-20 holdout. But in 2024-26 they **made things worse**: the max drop went from -26% to -45% and the return from 7.9 to 3.6 %/mo. They also turned the Oct-2025 crash window from +22.9% into -11.7%. The cause is that the kill-switches and stops sell at the bottom of V-shaped crashes, and a dip-buyer earns its money on exactly those rebounds.

Every number below comes from scripts I ran. No existing file was edited and nothing was committed.

## Method
- **Engine:** `ens_lib.py` is a copy of `engine.gkernel` with these additions: capital sleeves, monthly rebalancing, fractional Kelly sizing, volatility-targeted sizing, a hard stop, a per-trade risk cap, daily and 30-day kill-switches, a consecutive-loss cooldown and an ATR-spike entry block.
- **Equivalence (`ens_check.py`, ALL CHECKS PASS):**
  - With the new switches off, the equity and trade log are bit-identical to `engine.run` for 6 cases on 19 and 22 coins: Normal, 90% mode, 10% mode, bot timing + 0.2% worse fills, and partial take-profit with tp2.
  - With collapse coins sold at their last bar and sells 0.2% worse, the result is identical to `w90_wr90_lib.wrun`.
  - Two sleeves without rebalancing equal the weighted sum of the single runs (max relative difference 4.9e-15).
  - Guardrails switched on but set so they never trigger change nothing.
- **Panels:** 19 coins, and 22 coins (adding LUNA/FTT/SRM, with any held position sold at the coin's last bar). The 35% crash guard of both modes is always on.
- **Metrics:** trades are assigned to a period by entry hour. Win rate counts positions still open at the end. Profit factor (PF) = sum of winning P&L ÷ |sum of losing P&L|, each trade's P&L taken as a % of the account at entry (the same definition as Agent 1, `ens_audit_modes.py`).
- **Selection rules, fixed before seeing results, 2021-23 only:**
  - Architecture: the best ratio of %/mo to |max drop|, using the worse of the two panels.
  - Guardrails: keep settings that hold at least 70% of the unguarded architecture's 2021-23 %/mo on both panels. Among those, take the shallowest max drop. If drops are within 1 point, take the higher %/mo.

## 1. Existing strategies vs the combined designs (19-coin panel, normal fills)
Each cell is WR % / %/mo / max drop % / PF.

| Design | 2021-23 | 2024-26 | since 2025 |
|---|---|---|---|
| 90% mode (14%, +2.5%, 96h, cd 3h) | 93.8 / 13.2 / -64.4 / 2.2 | 92.6 / 6.9 / -35.4 / 3.2 | 93.2 / 5.8 / -22.6 / 4.2 |
| 10% mode (14%, +5%, 48h, cd 24h) | 81.1 / 15.0 / -60.6 / 2.1 | 76.5 / 8.1 / -29.7 / 3.1 | 75.0 / 6.1 / -28.7 / 3.1 |
| Sleeves 50/50 (no rebalancing) | 88.5 / 14.2 / -62.5 / 2.1 | 85.6 / 7.7 / -25.8 / 3.1 | 85.3 / 6.0 / -25.8 / 3.3 |
| Sleeves 70/30 (no rebalancing) | 88.5 / 13.8 / -63.3 / 2.1 | 85.6 / 7.5 / -28.1 / 3.1 | 85.3 / 5.9 / -24.2 / 3.5 |
| **Chosen architecture: sleeves 30/70, monthly rebalance** | 88.5 / 14.6 / -61.7 / 2.1 | 85.6 / 7.9 / -25.7 / 3.1 | 85.3 / 6.0 / -25.7 / 3.3 |
| Single bot: sell 50% at +2.5%, rest at +5%, 96h, cd 3h | 84.3 / 13.4 / -60.8 / 2.1 | 81.7 / 5.9 / -35.6 / 2.5 | 80.9 / 4.9 / -23.6 / 2.9 |
| Single bot: sell 70% at +2.5%, rest at +5%, 96h, cd 24h | 85.9 / 12.5 / -60.6 / 2.3 | 86.0 / 6.6 / -29.0 / 3.1 | 86.8 / 5.3 / -22.7 / 3.5 |
| 90% mode, 0.5 Kelly (partial-loss formula) | 93.5 / 11.3 / -66.5 / 2.1 | 93.0 / 6.9 / -36.0 / 3.2 | 93.2 / 5.8 / -22.6 / 4.2 |
| 90% mode, 0.25 Kelly (textbook formula)* | 92.0 / 6.1 / -39.7 / 1.9 | 94.0 / 4.8 / -26.0 / 3.8 | 95.4 / 2.8 / -19.9 / 3.6 |
| 90% mode, vol target 4%/day | 92.7 / 7.7 / -43.3 / 2.1 | 93.6 / 5.4 / -23.8 / 3.8 | 94.0 / 4.1 / -22.7 / 3.7 |
| 50/50 sleeves, vol target 6%/day | 87.5 / 10.7 / -45.9 / 2.1 | 88.1 / 6.3 / -25.1 / 3.1 | 86.5 / 4.8 / -25.1 / 3.0 |
| Chosen architecture + picked guardrails (X15 Y25 S25 N2 T72) | 86.0 / 12.0 / -41.5 / 1.9 | 82.1 / 3.6 / -44.9 / 1.6 | 80.9 / 2.1 / -34.9 / 1.5 |
| Chosen architecture + strictest (X15 Y20 S25 Z10 N2 T72) | 84.5 / 8.0 / -36.1 / 1.7 | 81.0 / 2.6 / -36.4 / 1.5 | 79.4 / 1.8 / -32.5 / 1.5 |

\* On 22 coins this row's 2021-23 max drop is -64.4%.

**Stress run** (22 coins, real bot timing, every buy and sell 0.2% worse; %/mo / max drop):

| Design | 2021-23 | 2024-26 | since 2025 |
|---|---|---|---|
| 90% mode | 11.0 / -65.0 | 5.5 / -30.3 | 4.4 / -22.7 |
| 10% mode | 12.2 / -74.5 | 6.5 / -29.8 | 4.6 / -29.0 |
| Chosen architecture | 12.0 / -66.3 | 6.3 / -26.3 | 4.6 / -26.1 |
| Chosen architecture + picked guardrails | 10.1 / -56.7 | 2.0 / -46.3 | 0.7 / -35.4 |

**What the designs show:**
- **Sleeves:** the 30/70 split with monthly rebalancing wins the rule (score 0.237), but only just: 30/70 without rebalancing scores 0.236 and 50/50 scores 0.228. The weights hardly matter.
  - The real gain from sleeves is in single-trade damage: the worst trade is -39% of the account vs -50% (90% mode) and -54% (10% mode).
  - The 2024-26 max drop is -25.7% vs -35.4% and -29.7%.
  - The combined win rate is 85-89%, not 90%.
- **Single-bot confluence** (one bot taking part profit at +2.5% and the rest at +5%) is worse than sleeves in every period.
- **Kelly sizing:**
  - The partial-loss formula (f* = W/L - (1-W)/G) usually asks for more than 100%, so it is capped at 100% and stays near full size. It changes little except in 2021-23, where it costs return.
  - The textbook formula (W - (1-W)/R) and the vol targets cut the drop to about -35 to -46%, but they cut the return in proportion, because they just hold more cash.
  - The Kelly fraction uses only trades closed before each entry: the last 30 trades, at least 10, with a 0.2 floor.

## 2. Each guardrail alone on the chosen architecture
Changes vs no guardrail, worse of the two panels. Each cell is change in %/mo / change in max drop (points; positive = better).

| Guardrail | 2021-23 | 2024-26 | since 2025 |
|---|---|---|---|
| Daily kill-switch at 10% (fired 69 times) | -2.1 / +12.6 | -4.5 / -21.2 | -2.7 / -6.6 |
| Daily kill-switch at 15% | -1.1 / +5.8 | -2.5 / -14.4 | -2.0 / -6.6 |
| Daily kill-switch at 20% | -0.4 / +10.7 | -0.5 / -8.5 | 0 / 0 |
| 30-day kill-switch at 20% | -0.6 / +9.4 | -2.4 / -18.3 | -1.4 / -6.6 |
| 30-day kill-switch at 25% | -0.6 / +12.5 | -1.1 / -6.6 | -1.4 / -6.6 |
| 30-day kill-switch at 30% | -1.1 / +11.8 | 0 / 0 | 0 / 0 |
| Hard stop 10% (resting stop order) | -3.6 / +5.4 | -3.1 / -6.4 | -2.4 / +1.9 |
| Hard stop 15% | -2.5 / +0.5 | -2.6 / -10.6 | -1.6 / +6.5 |
| Hard stop 20% | -0.2 / -1.6 | -1.4 / -1.2 | -1.1 / +1.7 |
| Hard stop 25% | -1.9 / -2.4 | -1.8 / -8.9 | -1.8 / -3.0 |
| Stop 20% + risk cap Z 10% | -3.3 / +10.8 | -2.6 / +2.3 | -1.9 / +2.3 |
| Stop 15% + risk cap Z 5% | -6.9 / +12.8 | -4.3 / -4.7 | -3.1 / +5.4 |
| Stop 25%, checked at hourly close only | -0.5 / +4.7 | 0 / 0 | 0 / 0 |
| Cooldown: 2 losses → 24h | -0.2 / -1.1 | 0 / 0 | 0 / 0 |
| Cooldown: 2 losses → 72h | -0.9 / -1.1 | -0.2 / -0.1 | 0 / 0 |
| Cooldown: 3 losses (24h or 72h) | ≤-0.2 / 0 | 0 / 0 | 0 / 0 |
| ATR-spike block 3x | -0.8 / -0.7 | -0.9 / 0 | -1.5 / 0 |
| ATR-spike block 5x | -0.2 / -2.7 | -2.0 / -4.9 | -1.1 / 0 |

**Notes on the guardrails:**
- **Risk cap Z = 10%** changes nothing at S = 10%: each sleeve's order is at most about 70% of the account, so the order times the stop never exceeds 7% of the account.
- **Stops do not cap the loss in a gap:** with a 20% stop, the worst stopped trade returned -36.4%, because the price opened below the stop.
- **Same pattern on 90% mode alone** (`guard_single_90.csv`), with one exception: the ATR 3x block improved the 2021-23 drop by +7.2 and the 2024-26 drop by +8.7 at about zero cost, but on the chosen architecture it did not help.
- **What I could not backtest:** latency over N ms and spread widening, because there is no latency or order-book spread data.

## 3. Combined grid and the pick
I tested 135 combinations (270 runs): X in {10, 15, 20}%, Y in {20, 25, 30}%, S in {off, 15, 25}%, Z in {off, 10}% (Z only with a stop), and (N, T) in {off, (2, 72), (3, 24)}. 38 of them passed the 70%-of-return rule.

**Pick (2021-23 only):**

| Setting | Value |
|---|---|
| Daily kill-switch X | 15% (flatten everything, lock until the next UTC day) |
| Rolling 30-day kill-switch Y | 25% (flatten, lock 168h) |
| Hard stop S | 25% (resting stop order) |
| Risk cap Z | not binding. Implied by the sleeve sizes: up to about 0.7 × 25% ≈ 17.5% of the account. Observed worst trade: -20.0% of the account |
| Cooldown | N = 2 losses → T = 72h pause |
| ATR block | not adopted |

**Results for the pick:**

| Period | Max drop (worse panel) | %/mo | Share of unguarded return |
|---|---|---|---|
| 2021-23 | -61.8% → -46.1% | 15.5 → 11.8 (22 coins) | 76% |
| 2024-26, looked at once | -25.7% → **-44.9%** | 7.9 → **3.6** | |
| Since 2025 | -25.7% → -34.9% | 6.0 → 2.1 | |
| 2018-20 holdout | -62.7% → -49.9% | 11.2 → 8.0 | |

The kill-switches fired 18 to 19 times (daily) and 14 to 15 times (30-day) over 2021-26.

The strictest feasible setting (X15 Y20 S25 Z10 N2 T72) has the lowest 2021-23 drop of the whole grid, -36%, but keeps only 8 to 10 %/mo. It fails the 70% rule and also does badly in 2024-26: 2.6 %/mo with a -36% drop.

## 4. Worst historical regimes (`ens_validate.py`)
I took the 6 worst 30-day windows of an equal-weight index of the 19 coins, plus 14 days after each. Cells show return / max drop inside the window, 22 coins.

| Window (index fall) | 90% mode | 10% mode | Chosen architecture | + picked guardrails |
|---|---|---|---|---|
| May 2021 (-40.5%) | -16.6 / -50.6 | -52.2 / -70.2 | -42.7 / -60.7 | -37.8 / -44.6 |
| Dec 2021 – Jan 2022 (-36.8%) | -3.2 / -38.5 | -9.3 / -36.0 | -7.4 / -34.3 | -12.7 / -29.1 |
| Apr – May 2022, LUNA (-45.2%) | -26.2 / -53.7 | -0.1 / -47.8 | -8.3 / -47.0 | +1.3 / -25.4 |
| Jun 2022 (-32.2%) | -15.9 / -45.7 | -0.5 / -37.9 | -5.1 / -39.2 | -6.1 / -25.7 |
| Oct – Nov 2025 (-33.7%) | +32.6 / -13.9 | +18.9 / -13.9 | +22.9 / -13.9 | **-11.7 / -32.2** |
| Jan – Feb 2026 (-41.4%) | +9.5 / -15.4 | +15.1 / -15.4 | +13.4 / -15.4 | +0.4 / -16.3 |

**COVID crash** (2020-02-20 to 2020-04-01, holdout data): the chosen architecture made +11.8% with a -60.1% drop; with the picked guardrails, +8.8% with a -32.4% drop.

## Recommendation
1. **Architecture:** two sleeves, 30% on 90% mode and 70% on 10% mode, free cash rebalanced monthly. It is spot only, with no leverage.
2. **Guardrails:** the values that answer the brief are X=15%, Y=25%, S=25%, Z not binding (about 17.5% implied), N=2, T=72h. They work in long bear markets (2021-22 and the 2018-20 holdout) and fail in V-shaped crashes: in 2024-26 they doubled the max drop and cut the return by more than half.
3. **Deploying them is a judgment call.** Paying about 45% of the return for protection against a 2022-style bear market should be a deliberate, written decision. None of the tested guardrail sets improved both 2021-23 and 2024-26.
4. **Two post-hoc candidates** that looked consistent across 2021-26 both failed the 2018-20 holdout:
   - 30-day kill-switch at 30% alone: 7.5 vs 11.2 %/mo, drop -56 vs -63.
   - Stop 20% + Z 10%: 6.8 %/mo, drop -61.
5. **Small discrepancy:** check90.log reports 90% mode +LUNA/FTT/SRM at +15.9%/mo; I get +15.7% with positions sold at the collapse coin's last bar.

## Files
Scripts are in /home/user/halal-rider/research/panic_buyer/refine:
- ens_lib.py
- ens_check.py
- ens_designs.py
- ens_guard.py
- ens_validate.py

Outputs are in /tmp/claude-0/-home-user-halal-rider/ec562e35-ea9e-5a74-ab92-4ff22c5eeb77/scratchpad/pb_out_ens/research:
- ens_check.log
- designs.log, designs.csv
- guard.log, guard_single_ARCH.csv, guard_single_90.csv, guard_grid.csv, guard_none.csv, guard_pick.txt
- validate.log, validate.csv, validate_regimes.csv, validate_holdout.csv


KEY NUMBERS
Equivalence: ens_lib.erun == engine.run bit-identical (6 cases x 19/22 coins), == wrun with liquidate+pens; sleeves == weighted sum (max rel diff 4.9e-15). ALL CHECKS PASS.
19 coins, WR/%mo/DD/PF:
90% mode: 2021-23 93.8/13.2/-64.4/2.2; 2024-26 92.6/6.9/-35.4/3.2; since 2025 93.2/5.8/-22.6/4.2.
10% mode: 81.1/15.0/-60.6/2.1; 76.5/8.1/-29.7/3.1; 75.0/6.1/-28.7/3.1.
ARCH (sleeves 30% 90-mode / 70% 10-mode, monthly rebalance; picked by the 2021-23 %mo/|DD| rule, score 0.237 vs 0.228 for 50/50): 88.5/14.6/-61.7/2.1; 85.6/7.9/-25.7/3.1; 85.3/6.0/-25.7/3.3. Worst trade -39% of the account vs -50/-54%.
Best confluence (one bot, 50% at +2.5% / rest at +5%, 96h, cd 3h): 84.3/13.4/-60.8/2.1; 81.7/5.9/-35.6/2.5.
Guardrail pick on 2021-23 (135-config grid, 270 runs): X=15% daily, Y=25% rolling 30d (lock 7d), S=25% resting stop, Z not binding (implied ~17.5%), N=2, T=72h, ATR block not adopted.
ARCH+pick: 2021-23 max drop -61.8 -> -46.1 (worse panel), %mo 15.5 -> 11.8 (76%); 2024-26 DD -25.7 -> -44.9, %mo 7.9 -> 3.6; since 2025 6.0 -> 2.1 %mo, DD -34.9; 2018-20 holdout 11.2 -> 8.0 %mo, DD -62.7 -> -49.9.
Stress (22 coins, bot timing, fills 0.2% worse): ARCH 12.0/-66.3, 6.3/-26.3, 4.6/-26.1 (%mo/DD); ARCH+pick 10.1/-56.7, 2.0/-46.3, 0.7/-35.4.
Oct-Nov 2025 crash window: ARCH +22.9% vs ARCH+pick -11.7%. LUNA window Apr-May 2022: -8.3%/-47.0 vs +1.3%/-25.4.
Daily kill 10% fired 69 times and cost 4.5 %mo / 21 DD points in 2024-26. Stop 20%: worst stopped trade -36.4% (gaps).

SCRIPTS cd /home/user/halal-rider/research/panic_buyer/refine
export PB_DATA=/tmp/claude-0/-home-user-halal-rider/ec562e35-ea9e-5a74-ab92-4ff22c5eeb77/scratchpad/pb_data
export PB_HOLDOUT=/tmp/claude-0/-home-user-halal-rider/ec562e35-ea9e-5a74-ab92-4ff22c5eeb77/scratchpad/holdout
export PB_OUT=/tmp/claude-0/-home-user-halal-rider/ec562e35-ea9e-5a74-ab92-4ff22c5eeb77/scratchpad/pb_out_ens/research
python3 ens_check.py      # equivalence proof (~20 s)
python3 ens_designs.py    # 40 designs x 2 panels (~30 s)
python3 ens_guard.py      # single guardrails (2 bases) + 135-config grid (~100 s)
python3 ens_validate.py   # periods, stress, 6 data-picked worst regimes, 2018-20 holdout + COVID (~15 s)
Files: /home/user/halal-rider/research/panic_buyer/refine/ens_lib.py, ens_check.py, ens_designs.py, ens_guard.py, ens_validate.py (new, uncommitted)