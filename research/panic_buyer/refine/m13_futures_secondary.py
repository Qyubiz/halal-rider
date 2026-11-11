"""m13 futures, SECONDARY candidate (deviation, decided after the grid gave NO pick and BEFORE any look at
2024-26 / 2018-20). The pre-registered selection found nothing: 17 designs pass every bar except the worse-fill
2021-23 drop bar (-64.4%), which 90% mode itself fails (-65.9%). Here only that one bar is relaxed to 90% mode's own
worse-fill drop (-65.9%); everything else (score, walk-down of 8, neighbour median) is m13_futures_grid.py's rule.
The result is labelled a secondary candidate, NOT a pick under the pre-registered rule.
Writes $PB_OUT/pick.json (secondary) and secondary.log.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m13_futures_grid as G  # noqa: E402
import m13_futures_lib as F  # noqa: E402

DD_W = -0.6595


def bars2(s, pre=""):
    return [s["n_wr"] >= G.WR, s["wn_wr"] >= G.WR, s["n_dd"] >= G.DD, s["wn_dd"] >= DD_W, s["n_mo"] > G.MO_N,
            s["wn_mo"] > G.MO_W]


G.bars = bars2
G.LOG = open(os.path.join(F.OUT, "secondary.log"), "a")
G.main()
