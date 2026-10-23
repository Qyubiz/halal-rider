"""Task "ladder", motivation only: 2021-23 design data (panels cut at 2024-01-01, nothing later is computed).

Question: in round 2's uncapped split ladder (cap 20%/w per coin, both legs), how often did a coin's budget go above
20% / 25% of equity, what did those trades contribute, and where did the 2021-23 drop come from, with the 19 live
coins and with LUNA/FTT/SRM added? Only settings already run in round 2 (live and the two r2 H2 families at d2 18%)
are run here; no capped variant is run before the grid is fixed.
"""
import numpy as np
import pandas as pd

import engine as E
import study as S
from r3_ladder import lrun


def design_panel(dead):
    syms, idx, P = S.market(dead)
    keep = idx < E.DESIGN_END
    return syms, idx[keep], {f: v[keep] for f, v in P.items()}


RUNS = {"LIVE SETTINGS": {},
        "r2 split 50%/18%, 4 coins, cap 40%": {"w": 0.5, "N": 4, "fmax": 0.40, "d2": 0.18},
        "r2 split 67%/18%, 5 coins, cap 30%": {"w": 2 / 3, "N": 5, "fmax": 0.30, "d2": 0.18}}

for dead in (False, True):
    syms, idx, P = design_panel(dead)
    print(f"\n==================== {'19 coins + LUNA/FTT/SRM' if dead else '19 live coins'}, 2021-23 only "
          f"({idx[0]} .. {idx[-1]})")
    for name, spec in RUNS.items():
        e, lg = lrun(P, idx, **spec)
        m = E.metrics(e)
        y = E.years(e)
        d = e / e.iloc[0]
        dd = d / d.cummax() - 1
        trough = dd.idxmin(); peak = d[:trough].idxmax()
        lg = lg.copy()
        for col in ("eb", "xb", "k"):
            lg[col] = lg[col].astype(int)
        eqe = np.where(lg.eb.values > 0, e.values[np.maximum(lg.eb.values - 1, 0)], 1.0)
        lg["bf"] = lg.bud.values / eqe                      # both legs' budget as a share of equity at the order
        lg["con"] = lg.ret.values * lg.amt.values / eqe     # crude contribution, share of equity
        print(f"\n{name}: x{m['mult']:.2f} DD {m['maxdd'] * 100:.1f}% ({peak} -> {trough}) 2022 {y.get(2022, np.nan) * 100:+.0f}% "
              f"| {len(lg)} closed trades")
        rows = []
        for lo, hi in ((0, 0.125001), (0.125001, 0.200001), (0.200001, 0.250001), (0.250001, 0.300001), (0.300001, 0.41)):
            g = lg[(lg.bf > lo) & (lg.bf <= hi)]
            if not len(g):
                continue
            rows.append(dict(budget=f"{lo * 100:.1f}-{hi * 100:.1f}%", trades=len(g), avg_ret=f"{g.ret.mean() * 100:+.2f}%",
                             time_stops=f"{(g.why == 1).mean() * 100:.1f}%", leg2=f"{g.leg2.mean() * 100:.0f}%",
                             sum_contrib=round(float(g.con.sum()), 3), worst=f"{g.ret.min() * 100:+.1f}%"))
        print(pd.DataFrame(rows).to_string(index=False))
        if spec:
            for cap in (0.20, 0.25):
                sc = np.minimum(1.0, cap / lg.bf.values)
                print(f"  crude: same trades scaled to a {cap:.0%} cap: sum contrib {float((lg.con * sc).sum()):+.3f} "
                      f"(uncapped {float(lg.con.sum()):+.3f}); {int((lg.bf > cap + 1e-6).sum())} trades above the cap "
                      f"contribute {float(lg.con[lg.bf > cap + 1e-6].sum()):+.3f}, of which time stops "
                      f"{float(lg.con[(lg.bf > cap + 1e-6) & (lg.why == 1)].sum()):+.3f}")
        w = lg[(idx[lg.eb.values] >= peak - pd.Timedelta(days=3)) & (idx[lg.eb.values] <= trough)]
        worst = w.sort_values("con").head(6)
        print(f"  worst drop window: {len(w)} trades entered, sum contrib {w.con.sum() * 100:+.1f}% of equity; largest losses:")
        for _, r in worst.iterrows():
            print(f"    {idx[int(r.eb)]} -> {idx[int(r.xb)]} {syms[int(r.k)]:9s} why {int(r.why)} leg2 {int(r.leg2)} budget "
                  f"{r.bf * 100:4.1f}% of equity, ret {r.ret * 100:+.1f}%, contrib {r.con * 100:+.1f}%")
        if dead:
            for s in E.DEAD:
                g = lg[lg.k == syms.index(s)]
                print(f"  {s}: {len(g)} trades, sum contrib {g.con.sum() * 100:+.1f}% of equity, max budget "
                      f"{g.bf.max() * 100:.1f}%, worst {g.ret.min() * 100:+.1f}%")
