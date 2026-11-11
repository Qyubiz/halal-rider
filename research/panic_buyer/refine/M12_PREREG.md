# Round m12 pre-registration (written before any m12 run; not to be changed afterwards)

Question: is there a setup with more profit per month than 90% mode, a similar or better win rate and a worst drop
no worse than -35%? Agents: A "sizing" (bet sizing to cut the drop), B "entry" (coin choice, bid level, joint
depth/target), C "verify" (independent check of the best A/B pick). Same costs and fill rules as m10/m11
(`M10.md`): Shariah spot only, fully funded, fee 0.1% per side (0.2% round trip), slippage as in the libs, resting
limits fill only when price trades 0.1% through, no lookahead.

## Baseline: 90% mode (1 coin x 100%, 24h highest close x 0.86, +2.5%, 96h, 3h cooldown, crash guard 35%)
Win rate / %/month / worst drop, from the m10/m11 engines (all reproduce it exactly):

| Run | 2021-23 | 2024-26 | since 2025 | 2018-20 holdout |
|---|---|---|---|---|
| normal fills | 93.8 / +13.17 / -64.4 | 92.6 / +6.89 / -35.4 | 93.2 / +5.80 / -22.6 | 92.5 / +10.82 / -66.3 |
| every fill 0.2% worse | 93.4 / +9.79 / -65.9 | 92.1 / +5.31 / -37.5 | 92.6 / +4.56 / -22.7 | 92.0 / +7.41 / -69.9 |
| stressed (bot timing, worse fills, LUNA/FTT/SRM) | 93.7 / +11.0 / -65 | 91.7 / +5.49 / -30.3 | | |

## Selection (each agent, 2021-23 data only)
Minimum bars in 2021-23, normal fills AND every fill 0.2% worse: win rate >= 92.5%, worst drop no deeper than
-64.4%, %/month above 90% mode's (13.17 normal, 9.79 worse). Each agent writes its ranking rule (it may add a
stricter drop target as a proxy for the -35% test limit) in its script header before running, scores every design
as the average over at least 4 nearby versions (cooldown +-1h, bot timing, start 2 weeks later) and requires the
median of its nearest parameter neighbours to pass the same bars. One pick per agent. Then one look at the tests.
No re-picking after that look; anything learned afterwards is labelled hindsight.

## Pass rule (one look; ALL must hold for "beats 90% mode")
1. 2024-26, normal fills: %/month > 6.89, win rate >= 92.5%, worst drop no deeper than -35.0%.
2. 2018-20 holdout, normal fills: %/month > 10.82, win rate >= 92.5%, worst drop no deeper than -66.3%.
3. Every fill 0.2% worse: %/month above 90% mode in 2024-26 (5.31) and 2018-20 (7.41), win rate >= 92.0%.
4. Stressed: 2024-26 %/month > 5.49; stressed 2021-23 never loses 95% or more (LUNA/FTT/SRM).
5. Agent C: the pick's nearby versions and parameter neighbours mostly pass 1-3 (more than half), not just the pick.
Since 2025 is reported but not part of the rule (it is inside 2024-26). If no candidate passes, the answer is
"nothing beats 90% mode" and no bot code or settings change.
