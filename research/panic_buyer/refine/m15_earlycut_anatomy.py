"""m15 earlycut anatomy, 2021-23 ONLY: 90% mode's losing trades (count, share of loss, path at 6/12/24/48h vs
winners) and an oracle bound (every loser sold at its best hourly close during the hold; trade sequence kept)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import m15_earlycut_lib as L, m10_velocity_lib as V, engine as E
syms, idx, P = V.market(False)
e, lg, inv = L.erun(P, idx, L.OFF)
tb = idx[lg.eb.astype(int).values]
lg = lg[(tb >= L.T21) & (tb < L.T24)].reset_index(drop=True)
C = P["cf"].values; Lo = P["l"].values
mb, ma = L.mkt_index(P, 0), L.mkt_index(P, 1)
acct = lg.ret * lg.amt / lg.eq0
los = lg.ret <= 0
print(f"2021-23 trades {len(lg)}, losers {los.sum()} ({los.mean()*100:.1f}%), why of losers {lg.why[los].value_counts().to_dict()}")
gain = np.log1p(acct[~los]).sum(); loss = np.log1p(acct[los]).sum()
print(f"log-growth from winners {gain:+.3f}, from losers {loss:+.3f} (losers eat {-loss/gain*100:.0f}% of the winners' growth)")
print(f"avg loser {acct[los].mean()*100:+.2f}% acct, median {acct[los].median()*100:+.2f}, worst {acct.min()*100:+.2f}; avg winner {acct[~los].mean()*100:+.2f}")
rows = []
for i, r in lg.iterrows():
    eb, k, en = int(r.eb), int(r.k), r.entry
    d = dict(los=bool(r.ret <= 0))
    for H in (6, 12, 24, 48):
        t = min(eb + H, int(r.xb))
        d[f"c{H}"] = C[t, k] / en - 1
        d[f"mn{H}"] = np.nanmin(C[eb:t + 1, k]) / en - 1
        d[f"b{H}"] = np.exp(mb[t] - mb[eb]) - 1; d[f"a{H}"] = np.exp(ma[t] - ma[eb]) - 1
        d[f"open{H}"] = int(r.xb) > eb + H
    d["best"] = np.nanmax(C[eb:int(r.xb) + 1, k]) * (1 - E.SLIP_MAJOR) * (1 - E.FEE) ** 2 / en - 1
    rows.append(d)
A = pd.DataFrame(rows)
for H in (6, 12, 24, 48):
    for g, nm in ((A[A.los], "losers"), (A[~A.los & A[f"open{H}"]], "winners still open")):
        print(f"  {H:2d}h {nm:18s} n={len(g):3d} coin close {g[f'c{H}'].median()*100:+5.1f}% (q25 {g[f'c{H}'].quantile(.25)*100:+5.1f}) "
              f"BTC {g[f'b{H}'].median()*100:+5.1f}% idx19 {g[f'a{H}'].median()*100:+5.1f}% | "
              f"coin<=-5% & idx<=-3%: {((g[f'c{H}']<=-.05)&(g[f'a{H}']<=-.03)).sum()}")
    print(f"  {H:2d}h losers already exited (open{H}=False): {(~A[A.los][f'open{H}']).sum()}")
x = acct.values.copy(); m0 = np.prod(1 + x)
fr = (lg.amt / lg.eq0).values
best = A.best.values
xo = np.where(los, np.maximum(best, -1) * fr, x); xz = np.where(los, 0.0, x)
mo = lambda m: m ** (1 / 36) - 1
print(f"compounded trade product 2021-23: actual x{m0:.2f} ({mo(m0)*100:+.2f}%/mo); oracle best-close exit for losers "
      f"x{np.prod(1+xo):.1f} ({mo(np.prod(1+xo))*100:+.2f}%/mo); losers at 0 x{np.prod(1+xz):.1f} ({mo(np.prod(1+xz))*100:+.2f}%/mo)")
print("losers' best close during hold vs entry (after fees): median", f"{np.median(best[los])*100:+.2f}%", "share >0:", f"{(best[los]>0).mean()*100:.0f}%")
