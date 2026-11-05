"""Round 2, lens "entry", part 3 (design period only, run before the grids): sub-period splits that motivated the
knife and r1h filters. Reads entry_trades_2021_23.csv from r2_entry_analysis.py. No 2024-26 number is printed."""
import os

import numpy as np
import pandas as pd

import study as S

pd.set_option("display.width", 200)
lg = pd.read_csv(os.path.join(S.OUT, "entry_trades_2021_23.csv"), parse_dates=["bought"])
lg["yr"] = np.where(lg.bought.dt.year == 2021, "2021", "2022-23")


def s(m, name):
    g = lg.groupby([m, lg.yr])
    print(f"\n--- {name}")
    print(pd.DataFrame({"n": g.size(), "ts": g.ts.mean(), "avg": g.ret.mean(), "contrib": g.contrib.sum()}).unstack().round(4).to_string())


dt = lg.pre3d <= -0.03
s(np.where(dt, np.where(lg.dd24 <= -0.105, "down&deep-at-open", "down&shallow-at-open"), "not-down"), "pre3d<=-3% by dd24 at open")
s(np.where(dt, "down", "-") + np.where(lg.breadth7 >= 0.9, "|weak", "|ok"), "pre3d<=-3% x breadth7>=0.9")
s(pd.cut(lg.pre3d, [-9, -0.10, -0.06, -0.03, 0, 0.05, 9]), "pre3d bins")
s(pd.cut(lg.breadth7, [-0.01, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]), "breadth7 bins")
s(lg.hour // 3 * 3, "hour of day (3h bins)")
s(pd.cut(lg.r1h, [-1, -0.04, -0.02, -0.01, 0, 0.01, 0.03, 1]), "r1h bins")
s(pd.cut(lg.r4h, [-1, -0.06, -0.04, -0.02, -0.01, 0, 0.02, 1]), "r4h bins")
knife = dt & (lg.breadth7 >= 0.9)
s(np.where(knife, "knife", "-") + np.where(lg.r1h > 0, "|r1h>0", "|r1h<=0"), "knife x r1h>0")
print("knife trades:", int(knife.sum()), "distinct buy days:", lg[knife].bought.dt.floor("D").nunique())
