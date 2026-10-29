"""Lens "losers", motivation only: trade paths of the live settings in 2021-23 (design period).
Nothing from 2024 onward is printed: only trades that entered AND exited before 2024-01-01 are used, and the
engine is causal, so these trades do not depend on later data.

For every trade: hourly closes after the fill relative to the entry price (the fill hour's close is "hour 0"),
the worst hourly close before the exit (adverse excursion seen at the bot's hourly cycles), and how the trade
ended (take-profit or 48h time stop)."""
import numpy as np
import pandas as pd

import engine as E
import study as S

syms, idx, P = S.market(False)
e, lg = E.run(P, idx)
eb = lg.eb.astype(int).values; xb = lg.xb.astype(int).values
d = lg[(idx[xb] < E.DESIGN_END)].copy()
d["eb"] = d.eb.astype(int); d["xb"] = d.xb.astype(int); d["k"] = d.k.astype(int)
C = P["cf"].values
H = 49
path = np.full((len(d), H), np.nan)
for i, (a, b, k, en) in enumerate(zip(d.eb, d.xb, d.k, d.entry)):
    for j in range(0, min(H, b - a + 1)):
        path[i, j] = C[a + j, k] / en - 1.0
d["held"] = d.xb - d.eb
# worst hourly close strictly before the exit hour (what an hourly stop could have acted on)
mae = np.array([np.nanmin(path[i, :max(1, h)]) for i, h in enumerate(d.held.values)])
d["mae"] = mae
tp, ts = d[d.why == 0], d[d.why == 1]
print(f"2021-23 trades: {len(d)} | take-profits {len(tp)} (sum ret*amt {(tp.ret*tp.amt).sum():+.2f}) "
      f"| time stops {len(ts)} (avg {ts.ret.mean()*100:+.1f}%, sum ret*amt {(ts.ret*ts.amt).sum():+.2f})")
print("\nholding time of take-profits (hours after the fill hour):")
print(tp.held.describe(percentiles=[.5, .6, .75, .9, .95]).round(1).to_string())
print("\nworst hourly close before exit (vs entry), by outcome: share of trades whose worst close was <= -x")
for x in (0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20):
    print(f"  <= -{x:4.0%}: take-profits {np.mean(tp.mae <= -x)*100:5.1f}% ({int((tp.mae <= -x).sum()):4d})"
          f"  time stops {np.mean(ts.mae <= -x)*100:5.1f}% ({int((ts.mae <= -x).sum()):3d})")
print("\nhard stop at close <= -x (crude, ignores re-use of cash, cooldown, re-entries): trades hit, "
      "and value of the would-be exit vs actual outcome (sum of (exit at stop - actual) x amt)")
for x in (0.05, 0.07, 0.10, 0.12, 0.15, 0.20):
    gain = 0.0; nh = 0; ntp = 0
    for i in range(len(d)):
        h = d.held.values[i]
        p = path[i, :max(1, h)]
        hit = np.where(p <= -x)[0]
        if len(hit):
            j = hit[0]
            rs = (1 + p[j]) * (1 - 0.0002) * (1 - 0.001) * (1 - 0.001) - 1   # sell at that close, fees both sides
            gain += (rs - d.ret.values[i]) * d.amt.values[i]; nh += 1; ntp += d.why.values[i] == 0
    print(f"  -{x:4.0%}: {nh:4d} trades stopped ({ntp:3d} would have been take-profits), net {gain:+.3f}")
print("\nstill underwater at hour X (close vs entry), what happened next:")
for X in (6, 12, 18, 24, 30, 36):
    alive = d[d.held > X]
    pX = path[alive.index.map(lambda i: d.index.get_loc(i)), X]
    for lo, hi in ((-0.03, 0.0), (-0.06, -0.03), (-0.10, -0.06), (-1, -0.10)):
        m = (pX <= hi) & (pX > lo)
        s = alive[m]
        if len(s):
            rs = (1 + pX[m]) * (1 - 0.0002) * (1 - 0.001) * (1 - 0.001) - 1
            print(f"  h{X:2d} close in ({lo:+.0%},{hi:+.0%}]: {len(s):4d} trades, later TP {np.mean(s.why == 0)*100:5.1f}%, "
                  f"avg final {s.ret.mean()*100:+5.1f}% vs sell now {rs.mean()*100:+5.1f}%; sum (now-final)*amt "
                  f"{((rs - s.ret.values) * s.amt.values).sum():+.3f}")
print("\nbreak-even target from hour X: trades alive at X whose hourly high later reaches entry*(1+be)*(1+eps) "
      "(or open >= it) before their actual exit; crude net of selling there instead (fees both sides)")
O = P["o"].values; Hh = P["h"].values
for be in (0.002, 0.005):
    for X in (6, 12, 18, 24):
        alive = d[d.held > X]
        nbe = 0; nlose = 0; g = 0.0
        for a, b, k, en, rt, am in zip(alive.eb, alive.xb, alive.k, alive.entry, alive.ret, alive.amt):
            tg = en * (1 + be)
            for t in range(a + X + 1, b + 1):          # target placed at the cycle after hour X's close
                if O[t, k] >= tg:
                    px = O[t, k]
                elif Hh[t, k] >= tg * 1.001:
                    px = tg
                else:
                    continue
                if t == b and rt > 0:                  # actual take-profit in the same hour: no change
                    break
                nbe += 1; nlose += rt < 0
                g += (px / en * (1 - 0.001) * (1 - 0.001) - 1 - rt) * am
                break
        print(f"  be +{be:.1%} X={X:2d}: alive {len(alive):4d}, sold at break-even {nbe:4d} (of which {nlose:3d} were "
              f"actual losers), crude net {g:+.3f}")
print("\ntime-stop trades at 48h: what if held longer (close at hour 72 / 96 after the fill hour vs actual), by depth at 48h")
for lo, hi in ((-0.06, 1), (-0.12, -0.06), (-0.2, -0.12), (-1, -0.2)):
    s = ts[(ts.ret > lo) & (ts.ret <= hi)]
    out = []
    for Hx in (72, 96):
        v = []
        for a, k, en, rt, am in zip(s.eb, s.k, s.entry, s.ret, s.amt):
            if a + Hx < len(C) and idx[a + Hx] < E.DESIGN_END:
                v.append((C[a + Hx, k] / en * (1 - 0.0002) * 0.999 * 0.999 - 1 - rt) * am)
        out.append(f"h{Hx}: n={len(v)} net {np.sum(v):+.3f}")
    print(f"  final in ({lo:+.0%},{hi:+.0%}]: {len(s):3d} trades, " + " | ".join(out))
print("\nworst 2021-23 drawdown: dates")
ed = e[e.index < E.DESIGN_END]; ddc = ed / ed.cummax() - 1
print(f"  trough {ddc.idxmin()} {ddc.min()*100:.1f}%, peak before {ed[:ddc.idxmin()].idxmax()}")
print("  5 deepest troughs by month:", (ddc.resample('ME').min().nsmallest(5) * 100).round(1).to_dict())
