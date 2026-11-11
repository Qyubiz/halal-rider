"""Task "ladder": post-hoc diagnostic, run AFTER the grid (r3_ladder.py) had run once. It changes neither the grid,
the pick nor the verdict. Exact numbers for the pick and its checks, and where the +LUNA/FTT/SRM drop comes from."""
import numpy as np
import pandas as pd

import engine as E
import study as S
from r3_ladder import GRID, lrun

PICK = "split 50% at 12% / rest at 18%, 4 coins, cap 25%"
R2 = {"w": 0.5, "N": 4, "fmax": 0.40, "d2": 0.18}            # round 2's uncapped pick, for comparison

for label, dead, pen in (("normal", False, 0.0), ("+LUNA/FTT/SRM", True, 0.0), ("fills 0.2% worse", False, 0.002)):
    syms, idx, P = S.market(dead)
    for name, spec in (("live", {}), ("pick", GRID[PICK])):
        e, lg = lrun(P, idx, pen=pen, **spec)
        r = E.score(e, lg, idx)
        print(f"{label:17s} {name}: d_mult {r['d_mult']:.4f} d_dd {r['d_dd']:.4f} y2022 {r['y2022']:+.4f} | "
              f"t_mult {r['t_mult']:.4f} t_dd {r['t_dd']:.4f}")

syms, idx, P = S.market(True)
for name, spec in (("live", {}), ("r2 uncapped pick (cap 40%)", R2), ("r3 pick (cap 25%)", GRID[PICK])):
    e, lg = lrun(P, idx, **spec)
    lg = lg.copy()
    for col in ("eb", "xb", "k"):
        lg[col] = lg[col].astype(int)
    eqe = np.where(lg.eb.values > 0, e.values[np.maximum(lg.eb.values - 1, 0)], 1.0)
    lg["bf"] = lg.bud.values / eqe
    lg["con"] = lg.ret.values * lg.amt.values / eqe
    d = e[e.index < E.DESIGN_END]
    d = d / d.iloc[0]
    dd = d / d.cummax() - 1
    trough = dd.idxmin(); peak = d[:trough].idxmax()
    print(f"\n+dead, {name}: 2021-23 drop {dd.min() * 100:.1f}% from {peak} to {trough}")
    # the deepest drops in 2021-23 outside that window too (separate episodes)
    ep = []
    dd2 = dd.copy()
    for _ in range(3):
        tr = dd2.idxmin()
        if dd2[tr] > -0.10:
            break
        pk = d[:tr].idxmax()
        rec = d[tr:][d[tr:] >= d[pk]]
        end = rec.index[0] if len(rec) else d.index[-1]
        ep.append((pk, tr, float(dd2[tr])))
        dd2[(dd2.index >= pk) & (dd2.index <= end)] = 0.0
    print("  drop episodes deeper than -10%: " + "; ".join(f"{p:%Y-%m-%d %H:%M} -> {t:%Y-%m-%d %H:%M} {v * 100:.1f}%"
                                                         for p, t, v in ep))
    w = lg[(idx[lg.eb.values] >= peak - pd.Timedelta(days=2)) & (idx[lg.eb.values] <= trough)]
    print(f"  trades entered in the worst window: {len(w)}, sum contrib {w.con.sum() * 100:+.1f}% of equity; largest losses:")
    for _, r in w.sort_values("con").head(6).iterrows():
        print(f"    {idx[int(r.eb)]} -> {idx[int(r.xb)]} {syms[int(r.k)]:9s} why {int(r.why)} leg2 {int(r.leg2)} budget "
              f"{r.bf * 100:4.1f}% of equity, ret {r.ret * 100:+.1f}%, contrib {r.con * 100:+.1f}%")
    dw = w[w.k.isin([syms.index(s) for s in E.DEAD])]
    print(f"  of which LUNA/FTT/SRM: {len(dw)} trades, sum contrib {dw.con.sum() * 100:+.1f}% of equity")
    g = lg[lg.k == syms.index("LUNAUSDT")]
    print(f"  LUNA: {len(g)} trades, sum contrib {g.con.sum() * 100:+.1f}%; the 4 worst: " + "; ".join(
        f"{idx[int(r.eb)]:%Y-%m-%d %H:%M} budget {r.bf * 100:.1f}% ret {r.ret * 100:+.1f}% contrib {r.con * 100:+.1f}%"
        for _, r in g.sort_values("con").head(4).iterrows()))
    # open positions at the LUNA cut (never sold: valued at LUNA's last price)
    cut = idx.get_loc(pd.Timestamp(E.DEAD["LUNAUSDT"]))
    print(f"  equity 2022-05-07 / 2022-05-14 / 2022-05-31: " + " / ".join(
        f"{e[e.index <= t].iloc[-1]:.3f}" for t in ("2022-05-07", "2022-05-14", "2022-05-31")) +
          f" | equity at the LUNA cut {e.iloc[cut]:.3f}")
