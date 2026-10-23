"""Task "stress", diagnostic: what the LUNA/FTT/SRM collapses did to each setting in hr_stress.py (hourly engine,
22 coins). Prints every trade in the three collapse coins (entry, exit, trade return, size and loss as % of equity),
the equity just before and after, and whether the account was left with a position it could never sell (a coin cut
off while held is valued at its last price and never sold, engine.DEAD) or with cash too small to trade (< 1e-9).
Measurement only."""
import os

import numpy as np
import pandas as pd

import engine as E
import study as S
from hr_stress import SETS

OUT = S.OUT

if __name__ == "__main__":
    log = open(os.path.join(OUT, "hr_stress_dead.log"), "w")

    def say(s=""):
        print(s, flush=True); log.write(s + "\n"); log.flush()

    syms, idx, P = S.market(True)
    for name, kw in SETS.items():
        e, lg = E.run(P, idx, **kw)
        eqb = np.r_[1.0, e.values[:-1]]
        say(f"\n{name}: 2021-23 x{e[e.index < E.DESIGN_END].iloc[-1]:.4f}  end x{e.iloc[-1]:.6g}")
        dead_k = [syms.index(s) for s in E.DEAD]
        g = lg[lg.k.astype(int).isin(dead_k)]
        for _, r in g.iterrows():
            eb, xb = int(r.eb), int(r.xb)
            say(f"  {syms[int(r.k)]:9s} bought {idx[eb]:%Y-%m-%d %H:%M} sold {idx[xb]:%Y-%m-%d %H:%M} why {int(r.why)} "
                f"trade {r.ret*100:+7.1f}%  size {r.amt/eqb[eb]*100:5.1f}% of equity -> {r.ret*r.amt/eqb[eb]*100:+6.1f}% "
                f"of equity | equity before x{eqb[eb]:.4g}, after sale x{e.iloc[xb]:.4g}")
        tin = idx[lg.eb.astype(int).values]
        last_t = idx[int(lg.xb.iloc[-1])] if len(lg) else None
        say(f"  trades: 2021-23 {int((tin < E.DESIGN_END).sum())}, 2024-26 {int((tin >= E.DESIGN_END).sum())}; last sale {last_t}; "
            f"equity min x{e.min():.3g} at {e.idxmin():%Y-%m-%d %H:%M}, at 2023-12-31 x{e[e.index < E.DESIGN_END].iloc[-1]:.4g}, "
            f"end x{e.iloc[-1]:.4g}")
        flat = e[e.index >= E.DESIGN_END]
        if flat.max() == flat.min():
            say(f"  2024-26 equity is flat at x{flat.iloc[0]:.6g}: the account is wiped / frozen (no usable cash or a "
                f"position in a coin whose data ended)")
        say(f"  trades logged in collapse coins: {len(g)}")
