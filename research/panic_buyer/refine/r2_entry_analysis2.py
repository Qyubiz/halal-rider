"""Round 2, lens "entry", part 2 (design period only): stability of the strongest features across 2021 vs 2022-23,
2-D tables, and which features flag the worst time-stop clusters. Reads entry_trades_2021_23.csv written by
r2_entry_analysis.py. No 2024-26 number is printed."""
import os

import numpy as np
import pandas as pd

import study as S

pd.set_option("display.width", 200)
lg = pd.read_csv(os.path.join(S.OUT, "entry_trades_2021_23.csv"), parse_dates=["bought"])
lg["yr"] = np.where(lg.bought.dt.year == 2021, "2021", "2022-23")


def tab(mask_name, mask):
    g = lg.groupby([lg.yr, mask])
    t = pd.DataFrame({"n": g.size(), "ts_rate": g.ts.mean(), "avg": g.ret.mean(), "contrib": g.contrib.sum()})
    print(f"\n--- {mask_name}")
    print(t.round(4).to_string())


cuts = {
    "breadth7>=0.9": lg.breadth7 >= 0.9,
    "breadth7>=0.95": lg.breadth7 >= 0.95,
    "r3d<=-9%": lg.r3d <= -0.09,
    "pre3d<=-3%": lg.pre3d <= -0.03,
    "c_vs_7dma<=-8%": lg.c_vs_7dma <= -0.08,
    "r7d<=0": lg.r7d <= 0,
    "r1h>0": lg.r1h > 0,
    "hour<=4": lg.hour <= 4,
    "vol7d<=5.5%": lg.vol7d <= 0.055,
    "hits7d<=4": lg.hits7d <= 4,
}
for k, m in cuts.items():
    tab(k, m)


def two(a, b, qa=3, qb=3):
    A = pd.qcut(lg[a], qa, duplicates="drop")
    B = pd.qcut(lg[b], qb, duplicates="drop")
    g = lg.groupby([A, B], observed=True)
    print(f"\n=== {a} x {b}: avg ret / ts rate / n")
    print(g.ret.mean().unstack().round(4).to_string())
    print(g.ts.mean().unstack().round(3).to_string())
    print(g.size().unstack().to_string())


two("breadth7", "r3d")
two("breadth7", "c_vs_7dma")
two("r3d", "dd24")
two("breadth7", "idio")

# worst clusters: time stops grouped by exit day
ts = lg[lg.ts == 1].copy()
ts["day"] = ts.bought.dt.floor("D")
cl = ts.groupby("day").agg(n=("ret", "size"), contrib=("contrib", "sum"), coins=("coin", lambda s: " ".join(c[:-4] for c in s)),
                           breadth7=("breadth7", "mean"), r3d=("r3d", "mean"), c7=("c_vs_7dma", "mean"),
                           btc_r7d=("btc_r7d", "mean"), hour=("hour", "mean"))
print("\n=== worst time-stop buy days (2021-23)")
print(cl.sort_values("contrib").head(15).round(3).to_string())

# share of time-stop loss and of take-profit gain flagged by each cut
print("\n=== flagged share: trades, TP gains, TS losses, net contrib flagged")
for k, m in cuts.items():
    tpg = lg.contrib[(lg.ts == 0) & m].sum() / lg.contrib[lg.ts == 0].sum()
    tsl = lg.contrib[(lg.ts == 1) & m].sum() / lg.contrib[lg.ts == 1].sum()
    print(f"{k:18s} trades {m.mean():5.1%}  TP gain {tpg:5.1%}  TS loss {tsl:5.1%}  net {lg.contrib[m].sum():+.3f}")
