"""Task "ten-wider": anatomy of a few designs (years, worst trades, P&L by coin outside the 19).
Usage: python w90_wider_anatomy.py  (designs listed in DESIGNS)."""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w90_wider_lib as L  # noqa: E402

T = L.TENMODE
DESIGNS = {
    "A v30 top30 @22% x0.5 N1": dict(T, rk="v30", M=30, dt=0.22, st=0.5, tt=0.05),
    "B v365 top30 @20% x1 N1": dict(T, rk="v365", M=30, dt=0.20, st=1.0, tt=0.05),
    "C v30 top30 @22% x1 N1": dict(T, rk="v30", M=30, dt=0.22, st=1.0, tt=0.05),
}


def anatomy(nm, d, W, say, stress=False):
    r, e, lg, syms = L.evaluate(d, stress=stress, W=W)
    idx = W["idx"]
    say(f"\n=== {nm}{' STRESSED' if stress else ''}: {L.name(d)}")
    say("   " + L.fmt(r))
    say("   years: " + " ".join(f"{y} {L.year_ret(e, y) * 100:+.0f}%" for y in range(2021, 2027)))
    lg2 = L.closed(lg).copy()
    lg2["when"] = idx[lg2.eb.astype(int).values]
    lg2["coin"] = [syms[int(k)][:-4] for k in lg2.k]
    lg2["acct"] = lg2.ret * lg2.amt / lg2.eq0
    w = lg2.sort_values("acct").head(8)
    say("   worst trades (% of account): " + "; ".join(f"{c} {t:%Y-%m-%d} {a * 100:+.0f}% (trade {x * 100:+.0f}%, why {int(y)})"
                                                     for c, t, a, x, y in zip(w.coin, w.when, w.acct, w.ret, w.why)))
    for tag, a, b in L.PERIODS:
        say(f"   [{tag}] trades {r[f'{tag}_n']} ({r[f'{tag}_n_out']} outside the 19), account P&L points "
            f"{r[f'{tag}_pnl'] * 100:+.0f} of which outside the 19 {r[f'{tag}_pnl_out'] * 100:+.0f}; "
            f"sold at a coin's last data hour: {r[f'{tag}_liq']}")
    bc = L.by_coin(lg, idx, syms, L.T24)
    bc = bc[[s not in L.E.H19 for s in bc.index]]
    say("   2024-26 outside-19 coins (P&L points, trades): " + ", ".join(f"{s[:-4]} {p * 100:+.0f} ({n})" for s, (p, n, _) in
                                                               bc.sort_values("pnl", ascending=False).iterrows()))
    bc = L.by_coin(lg, idx, syms, L.T21, L.T24)
    bc = bc[[s not in L.E.H19 for s in bc.index]]
    say("   2021-23 outside-19 coins: " + ", ".join(f"{s[:-4]} {p * 100:+.0f} ({n})" for s, (p, n, _) in
                                                     bc.sort_values("pnl", ascending=False).iterrows()))
    return r, e, lg


if __name__ == "__main__":
    W = L.build()
    out = open(os.path.join(L.OUT, "anatomy.log"), "a")

    def say(s=""):
        print(s, flush=True)
        out.write(s + "\n"); out.flush()
    for nm, d in DESIGNS.items():
        anatomy(nm, d, W, say)
    out.close()
