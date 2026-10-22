"""Lens "crowding", motivation only (2021-23 design period; nothing from 2024 onward is printed).

Only trades that entered AND exited before 2024-01-01 are used; the engine is causal, so they do not depend on
later data. Questions:
  A. Do fills that arrive together (same hour, or after a wave of fills in the previous 24h, or while many
     positions are already open) do worse than fills that arrive alone?
  B. Ladder: after the 12% leg fills, how often does price trade on down to a deeper level (16-24% under the
     24h high) before the exit, and what would a second leg there have done (crude per-trade replay, ignores
     re-use of cash, cooldown and re-entries)?
"""
import numpy as np
import pandas as pd

import engine as E
import study as S

syms, idx, P = S.market(False)
e, lg = E.run(P, idx)
lg = lg.copy()
for col in ("eb", "xb", "k"):
    lg[col] = lg[col].astype(int)
d = lg[idx[lg.xb.values] < E.DESIGN_END].copy().reset_index(drop=True)
eqv = e.values
O, H, L, C, REF = (P[f].values for f in ("o", "h", "l", "cf", "ref"))
print(f"2021-23 trades: {len(d)} | TP {int((d.why == 0).sum())} sum ret*amt {(d.ret * d.amt)[d.why == 0].sum():+.2f} "
      f"| time stops {int((d.why == 1).sum())} avg {d.ret[d.why == 1].mean() * 100:+.1f}% "
      f"sum {(d.ret * d.amt)[d.why == 1].sum():+.2f}")

# ------------------------------------------------------------------ A. crowding at entry
cnt_h = d.groupby("eb").size()
d["same_h"] = d.eb.map(cnt_h).values
ebs = np.sort(d.eb.values)
d["prev24"] = [int(np.searchsorted(ebs, t) - np.searchsorted(ebs, t - 24)) for t in d.eb.values]
# open positions at the start of the fill hour: entered earlier, not yet sold (exit hour >= t)
d["open0"] = [int(((d.eb.values < t) & (d.xb.values >= t)).sum()) for t in d.eb.values]
d["inv0"] = [float(d.amt.values[(d.eb.values < t) & (d.xb.values >= t)].sum() / eqv[t - 1]) for t in d.eb.values]


def grp(col, bins, labels):
    g = pd.cut(d[col], bins=bins, labels=labels, right=True)
    rows = []
    for lab in labels:
        s = d[g == lab]
        if not len(s):
            continue
        rows.append(dict(bucket=lab, trades=len(s), avg_ret=f"{s.ret.mean() * 100:+.2f}%",
                         ts_share=f"{(s.why == 1).mean() * 100:.1f}%", ts_avg=f"{s.ret[s.why == 1].mean() * 100:+.1f}%",
                         sum_ret_x_amt=round(float((s.ret * s.amt).sum()), 3), avg_amt=round(float(s.amt.mean()), 3)))
    print(f"\n{col}:")
    print(pd.DataFrame(rows).to_string(index=False))


grp("same_h", [0, 1, 2, 4, 6, 8, 99], ["1", "2", "3-4", "5-6", "7-8", "9+"])
grp("prev24", [-1, 0, 3, 7, 15, 99], ["0", "1-3", "4-7", "8-15", "16+"])
grp("open0", [-1, 0, 2, 4, 6, 8, 99], ["0", "1-2", "3-4", "5-6", "7-8", "9+"])
grp("inv0", [-1, 0.0001, 0.25, 0.5, 0.75, 9], ["0", "<25%", "25-50%", "50-75%", ">75%"])

print("\nhours with the most fills (2021-23):")
top = d.groupby("eb").agg(n=("ret", "size"), avg=("ret", "mean"), ts=("why", "sum"), s=("amt", lambda a: 0)).sort_values("n")
top["s"] = d.assign(x=d.ret * d.amt).groupby("eb").x.sum()
for t, r in top.tail(12).iloc[::-1].iterrows():
    print(f"  {idx[t]}  fills {int(r.n):2d}  avg {r.avg * 100:+5.1f}%  time stops {int(r.ts):2d}  sum ret*amt {r.s:+.3f}")

# per day: number of fills and the day's sum
dd = d.assign(day=idx[d.eb.values].floor("D"), x=d.ret * d.amt).groupby("day").agg(n=("x", "size"), s=("x", "sum"))
print("\nfills per calendar day vs that day's sum ret*amt:")
for lo, hi in ((1, 2), (3, 5), (6, 9), (10, 99)):
    m = (dd.n >= lo) & (dd.n <= hi)
    print(f"  {lo}-{hi} fills/day: {int(m.sum()):4d} days, sum ret*amt {dd.s[m].sum():+.3f}, "
          f"avg per day {dd.s[m].mean():+.4f}, losing days {(dd.s[m] < 0).mean() * 100:.0f}%")

# ------------------------------------------------------------------ B. second leg deeper
FEE, EPS, SL = E.FEE, E.EPS, E.SLIP_MAJOR


def ladder_trade(i, d2, w, tp=0.015, hm=48):
    """Crude replay of one trade with the money split w at the 12% leg and 1-w at ref*(1-d2)."""
    t0, k, en = d.eb[i], d.k[i], d.entry[i]
    L2 = REF[t0, k] * (1 - d2)
    a1, a2 = w, 1 - w
    u = a1 * (1 - FEE) / en
    f2 = -1.0
    if O[t0, k] <= L2:
        f2 = O[t0, k] * (1 + SL)
    elif L[t0, k] <= L2 * (1 - EPS):
        f2 = L2
    last = t0
    if f2 > 0:
        u += a2 * (1 - FEE) / f2
    for t in range(t0 + 1, len(idx)):
        if np.isnan(O[t, k]):
            continue
        if f2 < 0 and L[t, k] <= L2 * (1 - EPS):
            f2 = L2; u += a2 * (1 - FEE) / f2; last = t
        else:
            cost = a1 + (a2 if f2 > 0 else 0)
            avg = cost / (a1 / en + (a2 / f2 if f2 > 0 else 0))
            tpp = avg * (1 + tp)
            if t > last:
                if O[t, k] >= tpp:
                    return u * O[t, k] * (1 - FEE) - cost, f2 > 0, 0
                if H[t, k] >= tpp * (1 + EPS):
                    return u * tpp * (1 - FEE) - cost, f2 > 0, 0
        if t - t0 >= hm:
            cost = a1 + (a2 if f2 > 0 else 0)
            return u * C[t, k] * (1 - SL) * (1 - FEE) - cost, f2 > 0, 1
    return np.nan, f2 > 0, -1


print("\nB. second leg: share of 2021-23 trades whose price traded down to ref*(1-d2) before the exit, and a crude "
      "per-trade replay (profit per unit of the coin's budget, summed over trades x order size)")
print(f"  live (all money at 12%): sum ret*amt {(d.ret * d.amt).sum():+.3f}")
chk = np.array([ladder_trade(i, 0.99, 1.0)[0] for i in range(len(d))])
print(f"  replay check, w=1: sum {(chk * d.amt).sum():+.3f} (should be close to live; differences = replay simplifications)")
for d2 in (0.16, 0.18, 0.20, 0.24):
    for w in (0.5, 0.67):
        res = [ladder_trade(i, d2, w) for i in range(len(d))]
        pr = np.array([r[0] for r in res]); got2 = np.array([r[1] for r in res]); why = np.array([r[2] for r in res])
        s = (pr * d.amt.values)
        print(f"  d2={d2:.0%} w={w:.2f}: leg2 filled {got2.mean() * 100:5.1f}% of trades | sum {np.nansum(s):+.3f} | "
              f"time stops {int((why == 1).sum())} sum {np.nansum(s[why == 1]):+.3f} | TP sum {np.nansum(s[why == 0]):+.3f}"
              f" | worst trade {np.nanmin(pr) * 100:+.1f}% of budget")
