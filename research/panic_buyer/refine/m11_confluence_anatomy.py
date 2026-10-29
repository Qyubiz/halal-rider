"""m11 confluence: vote anatomy on 2021-23 trades ONLY (design period) to fix each vote's direction before the grid.
For four kinds of candidate trade (90% mode; the 12h -13% second trigger; re-entries right after a win; extra coins
at a 24% dip) print the win rate and mean trade return with each vote on vs off."""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m11_confluence_lib as C  # noqa: E402

M = C.market("w")
T24 = np.searchsorted(M["idx"], C.T24)


def table(name, lg):
    lg = lg[(lg.eb < T24)]
    b = M["bits"][lg.eb.astype(int).values, lg.k.astype(int).values]
    w = (lg.ret > 0).values; r = lg.ret.values
    print(f"\n{name}: {len(lg)} trades, win {w.mean()*100:.1f}%, mean {r.mean()*100:+.2f}%")
    for i, v in enumerate(C.VOTES):
        on = ((b >> i) & 1).astype(bool)
        f = lambda m: f"{m.sum():4d} tr win {w[m].mean()*100:5.1f}% mean {r[m].mean()*100:+6.2f}%" if m.any() else "  -"  # noqa
        print(f"  {v:5s} on: {f(on)} | off: {f(~on)}")
    sc = b.astype(int)
    tot = sum(((sc >> i) & 1) for i in (0, 1, 3, 4))
    for s in range(5):
        m = tot == s
        if m.any():
            print(f"  score(vol,fast,btc,brd)={s}: {m.sum():4d} tr win {w[m].mean()*100:5.1f}% mean {r[m].mean()*100:+6.2f}%")


e, lg = C.run(M, {})
table("90% mode", lg)
# re-entries right after a win (velocity agent's re-arm): trades whose coin had a winning exit < 3h before
e, lg = C.run(M, dict(rak=0))
lg = lg.sort_values("eb")
prev = {}
re = []
for i, x in lg.iterrows():
    k = int(x.k)
    re.append(k in prev and prev[k][1] and x.eb - prev[k][0] < 3)
    prev[k] = (x.xb, x.ret > 0)
table("re-armed entries (rak 0)", lg[np.array(re)])
e, lg = C.run(M, dict(B=dict(w=12, dep=0.13, tp=0.03, hm=48)))
table("12h -13% trigger (tier B trades)", lg[lg.tier == 1])
e, lg = C.run(M, dict(X=dict(top=20, dep=0.24, tp=0.025, hm=96)))
table("extra coins top-20 at 24% (tier X trades)", lg[lg.tier == 1])
