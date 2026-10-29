"""m10 grid: after the pick was fixed, the next four designs by the same 2021-23 rule on 2024-26 and since 2025,
to see whether the out-of-sample failure belongs to the pick or to the whole family. Information only, no re-pick."""
import os

import pandas as pd

import m10_grid_lib as G
from m10_grid_final import KEYS
from m10_grid_search import CSETS, DD_MIN, WR_MIN

df = pd.concat([pd.read_csv(os.path.join(G.OUT, f"search_{i}.csv")) for i in (1, 2, 3, 4)], ignore_index=True)
ok = df.drop_duplicates("name").query("wr >= @WR_MIN and dd >= @DD_MIN").sort_values("mo", ascending=False)
idx, P = G.market()
gc = {}
lines = []
for _, r in ok.iloc[:5].iterrows():
    d = {k: r[k] for k in KEYS if not pd.isna(r[k])}
    for k in ("L", "off", "tpk", "liq", "cd", "hm", "trail"):
        if k in d:
            d[k] = int(d[k])
    e, lg = G.run_any(idx, P, CSETS[d["cset"]], d, G.T_D0, None, gc)
    lines.append(r["name"])
    for k in ("t", "s"):
        lines.append("   " + G.fmt(k, G.period_stats(e, lg, idx, *G.PERIODS[k])))
print("\n".join(lines))
with open(os.path.join(G.OUT, "neighbours.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
