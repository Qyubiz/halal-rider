"""m15 nobounce anatomy, 2021-23 ONLY. Losers of 90% mode: count, share of total loss, path in first 6/12/24/48h,
best-gain-in-first-N-hours for winners vs losers, oracle upper bound."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m15_nobounce_lib as L, m10_velocity_lib as V, engine as E
syms, idx, P = V.market(False)
e, lg, inv = L.frun(P, idx, L.OFF)
tb = idx[lg.eb.astype(int).values]
lg = lg[(tb >= V.T21) & (tb < V.T24)].reset_index(drop=True)
H, C, Lo = P["h"].values, P["cf"].values, P["l"].values
acct = (lg.ret * lg.amt / lg.eq0).values
win = lg.ret.values > 0
print("trades", len(lg), "losers", (~win).sum(), "winrate", win.mean().round(4))
print("loss total (sum acct ret) losers %.3f, winners %.3f" % (acct[~win].sum(), acct[win].sum()))
print("losers by exit why:", lg[~win].why.value_counts().to_dict(), "avg loser acct %.2f%%  worst %.1f%%" % (acct[~win].mean()*100, acct.min()*100))
def path(i, N):
    r = lg.iloc[i]; eb = int(r.eb); k = int(r.k); en = r.entry
    xb = int(r.xb); sl = slice(eb+1, min(eb+1+N, xb+1))
    return np.nanmax(H[sl, k])/en-1, C[min(eb+N, xb, len(C)-1), k]/en-1, np.nanmin(Lo[sl, k])/en-1
rows = {}
for N in (4, 6, 12, 24, 48):
    pw = np.array([path(i, N) for i in range(len(lg))])
    rows[N] = pw
    for nm, m in (("winners", win), ("losers", ~win)):
        q = np.percentile(pw[m, 0]*100, [10, 25, 50, 75, 90])
        print(f"N={N:2d} {nm:7s} best-high gain pct10/25/50/75/90: {np.round(q,2)}  close@N median {np.median(pw[m,1])*100:+.2f}%  min-low median {np.median(pw[m,2])*100:+.2f}%")
for N in (4, 6, 12, 24):
    for y in (0.005, 0.01, 0.015, 0.02):
        nb = rows[N][:, 0] < y
        print(f"N={N:2d} y={y*100:.1f}%: never-bounced {nb.sum():4d} of which losers {(nb&~win).sum():3d} winners {(nb&win).sum():4d} | share of losers caught {(nb&~win).sum()/(~win).sum():.2f}, winners hit {(nb&win).sum()/win.sum():.2f}")
# oracle: loser replaced by best hourly close exit (after fees+slip) during hold, floor actual
best = []
for i in np.where(~win)[0]:
    r = lg.iloc[i]; eb = int(r.eb); k = int(r.k); xb = int(r.xb)
    cs = C[eb+1:xb+1, k]
    b = (cs.max()*(1-E.SLIP_MAJOR)*(1-E.FEE)) / (r.entry/(1-E.FEE)) - 1 if len(cs) else r.ret
    best.append(max(b, r.ret))
ret = lg.ret.values.copy(); orc = ret.copy(); orc[np.where(~win)[0]] = best
scale = (lg.amt / lg.eq0).values
def mult(r): return np.prod(1 + r*scale)
months = 36.0
for nm, r in (("actual(trade-compounded)", ret), ("oracle all losers cut at best close", orc)):
    m = mult(r); print(nm, "mult %.2f  %%/mo %.2f" % (m, (m**(1/months)-1)*100), "winrate", (r>0).mean().round(3))
print("oracle avg best-close return of losers: %.2f%%" % (np.mean(best)*100))
