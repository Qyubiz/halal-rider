"""Lens "crowding", motivation part 3 (2021-23 only; nothing from 2024 onward is printed).

F. Timing of a second leg: does price reach ref*(1-d2) inside the leg-1 fill hour (only a leg 2 resting from the
   hour start catches that) or later (a leg 2 placed at the next hourly cycle also catches it)?
G. How much of the account the live bot actually holds in coins (the rest sits in resting buys that mostly do
   not fill), i.e. the room for extra leg-2 money.
"""
import numpy as np

import engine as E
import study as S

syms, idx, P = S.market(False)
e, lg = E.run(P, idx)
for col in ("eb", "xb", "k"):
    lg[col] = lg[col].astype(int)
d = lg[idx[lg.xb.values] < E.DESIGN_END].copy().reset_index(drop=True)
O, H, L, C, REF = (P[f].values for f in ("o", "h", "l", "cf", "ref"))
EPS = E.EPS

print("F. first hour at which the low trades through ref_fill*(1-d2) before the live exit (or 48h)")
for d2 in (0.16, 0.18, 0.20, 0.24):
    same = later = 0
    for t0, t1, k in zip(d.eb, d.xb, d.k):
        L2 = REF[t0, k] * (1 - d2)
        if O[t0, k] <= L2 or L[t0, k] <= L2 * (1 - EPS):
            same += 1
            continue
        for t in range(t0 + 1, t1 + 1):
            if L[t, k] <= L2 * (1 - EPS):
                later += 1
                break
    print(f"  d2={d2:.0%}: in the fill hour {same:3d} | later, before the exit {later:3d}")

des = idx < E.DESIGN_END
held = np.zeros(len(idx))
for t0, t1, k, a in zip(lg.eb, lg.xb, lg.k, lg.amt):
    held[t0:t1] += a                       # cost basis held from the fill hour to the hour before the exit
frac = held[des] / e.values[des]
print(f"\nG. 2021-23 share of equity held in coins (cost basis, hourly): mean {frac.mean() * 100:.0f}%, "
      f"median {np.median(frac) * 100:.0f}%, hours with nothing held {np.mean(frac == 0) * 100:.0f}%, "
      f"hours above 50% {np.mean(frac > 0.5) * 100:.0f}%, above 80% {np.mean(frac > 0.8) * 100:.0f}%")
