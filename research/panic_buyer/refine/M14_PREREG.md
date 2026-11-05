# Round m14 pre-registration (written before any m14 run; not to be changed afterwards)

Two parts, one family of candidates, one judge:
- **Grid** (`m14_grid.py`, run by the manager): about 9,000 settings of the m11 fusion engine (`m11_fusion_lib.fk`,
  proven equal to 90% mode with its switches off), covering everything earlier rounds touched: buy depth 12-16%,
  take-profit 2/2.5/3% or range-scaled, time stop 72/96/144h, cooldown after a win 0/3h, calm-coin filter
  off/7%, an optional second trigger (6h or 12h high, 10% or 13%, +2% or +3%, 48h), crash guard 30/35%, sizing
  1 x 100% or 2 x 50%.
- **Ideas** (10 Sonnet agents, 2 at a time, files `m14_<idea>_*.py`): each tests one idea not tried in m10-m13
  and hands in ONE pick chosen on 2021-23 only.

Costs and fills as in `M10.md`: Shariah spot only, fully funded, fee 0.1% per side, slippage as in the libs,
limit fills need 0.1% trade-through, no lookahead.

## Grid selection (2021-23 only)
1. Every setting: 2021-23 win rate >= 92.5% and worst drop no deeper than -64.4%, with normal fills and with every
   fill 0.2% worse; %/month above 13.17 (normal) and 9.79 (worse).
2. Settings passing 1: score = mean 2021-23 %/month over 5 nearby versions (base, cooldown -1h, +1h, bot timing,
   start 2021-01-15), from `m11_fusion_lib.design_scores`; all 5 versions must also pass the win-rate and drop bars.
3. Neighbours = the settings one grid step away in one parameter. The median of the neighbours' normal-fill
   2021-23 %/month, win rate and drop must pass bar 1.
4. The 20 best scores that pass 2 and 3 are the grid finalists (fewer if fewer pass).

## One look and the pass rule (grid finalists and agent picks alike, family size K = 30)
A candidate beats 90% mode only if ALL hold:
1-4. Items 1-4 of `M12_PREREG.md` (2024-26 and 2018-20 profit, win rate and drop bars; worse fills; stressed,
     incl. no 95% loss on LUNA/FTT/SRM).
5. Neighbours: more than half of its nearby versions and grid neighbours also pass items 1-3.
6. Luck check: monthly returns of candidate minus 90% mode (normal fills), separately for 2024-26 and for the
   2018-20 holdout; stationary block bootstrap (mean block 3 months, 20,000 resamples) of the mean difference;
   one-sided p-value below 0.05 / 30 = 0.0017 in BOTH periods (Bonferroni over the 30 candidates).
If nothing passes, the answer is "nothing beats 90% mode" and no bot code or settings change. Since 2025 is
reported only. Live paper trading remains the final test before real money for anything that passes.
