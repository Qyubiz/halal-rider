"""Anatomy of 90% mode's LOSING trades, 2021-23 only (trades bought in the period)."""
import numpy as np, pandas as pd
import m15_memory_lib as L, m10_velocity_lib as V, engine as E
syms, idx, P = V.market(False)
e, lg, inv = L.mrun(P, idx)
tb = idx[lg.eb.astype(int).values]
x = lg[(tb >= V.T21) & (tb < V.T24)].reset_index(drop=True)
x["acct"] = x.ret * x.amt / x.eq0
C = P["cf"].values
los = x[x.ret <= 0]
print("trades", len(x), "losers", len(los), "winrate", 1 - len(los) / len(x))
lg_sum = np.log1p(x.acct).sum(); lg_los = np.log1p(los.acct).sum()
print("log-return of all trades %.3f, losers %.3f (winners %.3f); loser share of gross loss vs win gain: avg loser acct %.3f, avg winner acct %.4f"
      % (lg_sum, lg_los, lg_sum - lg_los, los.acct.mean(), x[x.ret > 0].acct.mean()))
print("why of losers:", los.why.value_counts().to_dict(), " worst 5 acct:", np.sort(los.acct.values)[:5].round(3))
print("loser acct quantiles", np.quantile(los.acct, [.1, .25, .5, .75, .9]).round(3))
# path: close/entry-1 at +6/12/24/48h after the entry bar (for losers), and share that were ever above entry
for H in (6, 12, 24, 48):
    v = []
    for r in los.itertuples():
        t = int(r.eb) + H
        if t <= int(r.xb): v.append(C[t, int(r.k)] / r.entry - 1)
    v = np.array(v)
    print(f"+{H}h: n alive {len(v)}  mean {v.mean()*100:+.1f}%  median {np.median(v)*100:+.1f}%  share under entry {np.mean(v<0)*100:.0f}%  under -5% {np.mean(v<-.05)*100:.0f}%")
# for winners same, to see separation
w = x[x.ret > 0]
for H in (6, 12, 24, 48):
    v = [C[int(r.eb) + H, int(r.k)] / r.entry - 1 for r in w.itertuples() if int(r.eb) + H <= int(r.xb)]
    print(f"winners +{H}h: n {len(v)} median {np.median(v)*100:+.1f}%  under entry {np.mean(np.array(v)<0)*100:.0f}%")
# oracle: cut each loser at best close after entry (before exit)
FEE, SL = E.FEE, E.SLIP_MAJOR
orc = []
for r in x.itertuples():
    if r.ret > 0: orc.append(r.acct); continue
    seg = C[int(r.eb):int(r.xb) + 1, int(r.k)]
    best = np.nanmax(seg * (1 - SL)) / r.entry * (1 - FEE) ** 2 - 1
    best = max(best, r.ret)
    orc.append(best * r.amt / r.eq0)
orc = np.array(orc)
mon = (V.T24 - V.T21).days / V.MD
print("2021-23 compounded (trade sequence held fixed): actual %.3fx (%.2f%%/mo)  oracle %.3fx (%.2f%%/mo)  all losers->0: %.3fx"
      % (np.prod(1 + x.acct), ((np.prod(1 + x.acct)) ** (1 / mon) - 1) * 100, np.prod(1 + orc), (np.prod(1 + orc) ** (1 / mon) - 1) * 100,
         np.prod(1 + np.where(x.ret <= 0, 0, x.acct))))
print("loser-only oracle avg loser acct %.4f" % np.mean(orc[x.ret.values <= 0]))
