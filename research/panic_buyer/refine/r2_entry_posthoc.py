"""Round 2, lens "entry": post-hoc diagnostic, run AFTER both grids (not used for any choice). How the knife and
r1h>0 flags split the live bot's trades in 2021-23 vs 2024-26."""
import numpy as np
import pandas as pd

import engine as E
import study as S
from r2_entry_feats import flag

pd.set_option("display.width", 200)
syms, idx, P = S.market()
_, e, lg = S.evaluate({})
lg["per"] = np.where(idx[lg.eb.astype(int).values] < E.DESIGN_END, "2021-23", "2024-26")
lg["ts"] = (lg.why == 1).astype(int)
lg["contrib"] = lg.ret * lg.amt
eb, k = lg.eb.astype(int).values, lg.k.astype(int).values
for name, f in {"knife pre<=-3% br>=0.80": flag(P, pre=0.03, br=0.80), "knife pre<=-3% br>=0.90": flag(P, pre=0.03, br=0.90),
                "r1h>0": flag(P, r1=0.0), "r1h>0 & knife(-3%,0.90)": flag(P, r1=0.0, pre=0.03, br=0.90)}.items():
    lg["f"] = f[eb, k]
    g = lg.groupby(["per", "f"])
    print(f"\n--- {name}")
    print(pd.DataFrame({"n": g.size(), "ts_rate": g.ts.mean(), "avg_ret": g.ret.mean(), "sum_contrib": g.contrib.sum()}).round(4).to_string())
