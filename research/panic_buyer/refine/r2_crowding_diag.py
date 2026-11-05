"""Lens "crowding": post-hoc diagnostic of the H2 pick's failed LUNA/FTT/SRM check (run AFTER both grids; it
changes neither the grids, the pick nor the verdict). What did the collapse coins cost, and when was the drop?"""
import numpy as np

import engine as E
import study as S
from r2_crowding import H1, H2, crun

PICKS = {"LIVE SETTINGS": {}, "H1 pick": H1["extra leg 2: 1x at 24%"], "H2 pick": H2["split 50% at 12% / rest at 18%, 4 coins"]}
syms, idx, P = S.market(True)
for name, spec in PICKS.items():
    e, lg = crun(P, idx, **spec)
    r = E.score(e, lg, idx)
    d = e[e.index < E.DESIGN_END]
    d = d / d.iloc[0]
    dd = d / d.cummax() - 1
    trough = dd.idxmin()
    peak = d[:trough].idxmax()
    print(f"{name}: 21-23 drop {r['d_dd'] * 100:.1f}% from {peak} to {trough}")
    for s in E.DEAD:
        g = lg[lg.k.astype(int) == syms.index(s)]
        lead = f"  {s}: {len(g)} closed trades, sum ret*amt {(g.ret * g.amt).sum():+.3f}, worst {g.ret.min() * 100 if len(g) else np.nan:+.1f}%"
        if "leg2" in g:
            lead += f", with leg 2 {int(g.leg2.sum())}"
        # position still open when the coin's data ends (never sold): value lost vs cost
        print(lead)
    held = [s for s in E.DEAD]
    print(f"  equity at 2022-05-08 / 2022-05-14 / 2022-05-20: " + " / ".join(
        f"{e[e.index <= t].iloc[-1]:.3f}" for t in ("2022-05-08", "2022-05-14", "2022-05-20")))
