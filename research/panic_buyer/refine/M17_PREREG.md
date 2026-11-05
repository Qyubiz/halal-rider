# Round m17 pre-registration: the 20%/month target (written before any m17 run)

Target: 20%/month. Contestants: "bigtp" (bigger targets on volatile dips, the 10% mode family), "multi"
(more opportunities: several positions, more dip levels), "ceiling" (perfect-foresight upper bounds, to measure
what is available at all). Costs and fills as `M10.md` (Shariah spot only, fully funded, 0.2% round trip, 0.1%
trade-through, no lookahead). No win-rate floor this round.

Selection on 2021-23 only: highest mean %/month over 5 nearby versions (as `m11_fusion_lib.design_scores`) with
worst drop no deeper than -70% (normal and 0.2%-worse fills) and no 95% loss in the stressed 2021-23 run.
One pick per contestant, one look at 2024-26, since 2025, 2018-20, worse fills, stressed.

Reached 20%/month only if: %/month >= 20 in 2024-26 AND in 2018-20, normal and 0.2%-worse fills, stressed
2024-26 >= 20, worst drop no deeper than -70%, and the luck check of `m14_judge.py` vs 90% mode passes
(p < 0.0017 in both). Otherwise report the best out-of-sample %/month found, and the ceiling.
