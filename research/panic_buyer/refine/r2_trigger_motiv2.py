"""Round 2, lens "trigger", motivation part 2 (2021-23 candles only, cut at 2024-01-01; nothing later is used).

D. Single-coin bots (live rules, one coin all-in) with different cooldowns after a sale (0/12/24/48/96h), and with
   a longer block only after a time stop (72/168h): is the coin's next trade after a quick re-entry worse?
E. Portfolio trades of the live settings (entered and exited before 2024): what the coins that had a time stop
   did over the next 24-168h (would a re-buy there have been a falling knife?).
"""
import numpy as np
import pandas as pd

import engine as E
import study as S
import r2_trigger_lib as L

syms, idx, P = S.market(False)
cut = idx < E.DESIGN_END
Pd = {f: P[f][cut] for f in ("o", "h", "l", "c", "cf", "ref")}

print("D. single-coin bots, 2021-23: totals over the 19 coins")
rows = []
for name, kw in {"cd 0h": dict(cd=0), "cd 12h": dict(cd=12), "cd 24h (live)": dict(cd=24), "cd 48h": dict(cd=48),
                 "cd 96h": dict(cd=96), "cd 24h, 72h after time stop": dict(cdts=72),
                 "cd 24h, 168h after time stop": dict(cdts=168)}.items():
    tr = []; mults = []
    for s in syms:
        Q = {g: Pd[g][[s]] for g in Pd}
        if Q["c"][s].notna().sum() < 500:
            continue
        ee, ll = L.trun(Q, Q["c"].index, N=1, fmax=1.0, **kw)
        tr.append(ll); mults.append(float(ee.iloc[-1]))
    t = pd.concat(tr)
    rows.append({"variant": name, "trades": len(t), "time-stop %": 100 * (t.why == 1).mean(),
                 "avg ret %": 100 * t.ret.mean(), "TS avg %": 100 * t.ret[t.why == 1].mean(),
                 "median coin multiple": np.median(mults), "coins up": int(np.sum(np.array(mults) > 1)),
                 "mean log multiple": np.mean(np.log(mults))})
pd.set_option("display.width", 160)
print(pd.DataFrame(rows).round(3).to_string(index=False))

# ---------------------------------------------------------------- E. after a time stop
e, lg = E.run(P, idx)
lg = lg.astype({"eb": int, "xb": int, "k": int})
d = lg[idx[lg.xb.values] < E.DESIGN_END]
ts = d[d.why == 1]
C = P["cf"].values; Lo = P["l"].values
print(f"\nE. {len(ts)} time stops in 2021-23: price path of the coin after the time-stop sale (close vs the sale-hour close)")
for H in (24, 48, 96, 168):
    r = []; mn = []
    for xb, k in zip(ts.xb, ts.k):
        if xb + H < len(idx) and idx[xb + H] < E.DESIGN_END:
            r.append(C[xb + H, k] / C[xb, k] - 1); mn.append(np.nanmin(Lo[xb + 1:xb + H + 1, k]) / C[xb, k] - 1)
    r = np.array(r); mn = np.array(mn)
    print(f"  +{H:3d}h: n {len(r)}, median close {np.median(r)*100:+.1f}%, mean {r.mean()*100:+.1f}%, "
          f"share lower {np.mean(r < 0)*100:.0f}%, median lowest low {np.median(mn)*100:+.1f}%, "
          f"share with a further -12% low {np.mean(mn <= -0.12)*100:.0f}%")
