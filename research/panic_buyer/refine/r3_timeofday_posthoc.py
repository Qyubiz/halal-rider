"""Round 3, task "timeofday": POST-HOC diagnostics, written and run AFTER both grids in r3_timeofday.py had run once.
Nothing here changes a grid, a pick or a verdict; it only explains them.

1. Did the 2021-23 time-of-day effect persist? Live trades by window and year, 2021-26.
2. Why do the two picks have a -20/-21% 2024-26 drop against live -16%: peak/trough dates of the worst 2024-26 drop,
   and the positions held at the trough.
"""
import numpy as np
import pandas as pd

import engine as E
import study as S
from r3_timeofday import H1, H2, window

pd.set_option("display.width", 220)
syms, idx, P = S.market(False)
e0, lg0 = E.run(P, idx)
eb = lg0.eb.astype(int).values
yr = idx[eb].year
WIN = {"UTC 00-07": window("day", 0, 7), "UTC 03-06": window("day", 3, 6), "Sat-Sun all day": window("wkend", 0, 23),
       "Sat-Sun 00-07": window("wkend", 0, 7)}
print("1. live trades, inside vs outside each window, by year of the order (avg return, time-stop share, n)")
for w, f in WIN.items():
    m = f(P)[eb]
    cells = []
    for y in sorted(set(yr)):
        a = lg0[(yr == y) & m]; b = lg0[(yr == y) & ~m]
        cells.append(f"{y}: in {a.ret.mean() * 100:+.2f}% TS {(a.why == 1).mean() * 100:4.1f}% n {len(a):3d} / "
                     f"out {b.ret.mean() * 100:+.2f}% TS {(b.why == 1).mean() * 100:4.1f}%")
    print(f"  {w}:\n    " + "\n    ".join(cells))


def worst_dd(e, start):
    x = e[e.index >= start]
    x = x / x.iloc[0]
    dd = x / x.cummax() - 1
    tr = dd.idxmin()
    pk = x[:tr].idxmax()
    return pk, tr, float(dd.min())


print("\n2. worst 2024-26 drop: peak, trough, depth, and the positions open at the trough hour")
for name, spec in (("LIVE SETTINGS", {}), ("UTC 03-06 depth 14%", H1["UTC 03-06 depth 14%"]),
                   ("Sat-Sun 00-07 skip", H2["Sat-Sun 00-07 skip"])):
    kw = spec(P, syms) if callable(spec) else spec
    e, lg = E.run(P, idx, **kw)
    pk, tr, d = worst_dd(e, E.DESIGN_END)
    t = idx.get_loc(tr)
    op = lg[(lg.eb.astype(int) <= t) & (lg.xb.astype(int) >= t)]
    print(f"  {name:22s} peak {pk} trough {tr} ({tr:%a}) drop {d * 100:.1f}%  open at trough: {len(op)} positions, "
          f"{op.amt.sum() / e.iloc[t] * 100 if len(op) else 0:.0f}% of equity at cost; their final returns: "
          + ", ".join(f"{syms[int(k)][:-4]} bought {idx[int(b)]:%m-%d %H}h {r * 100:+.1f}%"
                      for k, b, r in zip(op.k, op.eb, op.ret)))
    # the same window for live, for comparison
    if name != "LIVE SETTINGS":
        print(f"    live over the same window: {(e0.loc[tr] / e0.loc[pk] - 1) * 100:+.1f}%")
