import glob, os, numpy as np, pandas as pd
import backtest as b, search as S
from profit_search import to_1h
d15 = {os.path.basename(p)[:-4]: pd.read_csv(p, index_col=0, parse_dates=True) for p in sorted(glob.glob("data/*.csv"))}
d1h = {s: to_1h(d) for s, d in d15.items()}
mid = next(iter(d1h.values())).index[len(next(iter(d1h.values())))//2]
RULE = ["mfi50-80","don mid","c>vwap","cmf>0","macd>sig"]
def sim(df, F, rules, rr_min=2.0, hold=48, tp_lb=48, sl_lb=20):
    o,h,l,c = df.o.values, df.h.values, df.l.values, df.c.values
    lowN = df.l.rolling(sl_lb).min().values; highN = df.h.rolling(tp_lb).max().values
    sig = F[rules].all(axis=1).values
    n=len(df); i=250; T=[]
    while i<n-2:
        if not sig[i]: i+=1; continue
        e=o[i+1]; sl=max(lowN[i]*0.999,e*0.96); sl=min(sl,e*0.996); tp=highN[i]
        if tp<=e*1.004 or (tp/e-1)/(1-sl/e) < rr_min: i+=1; continue
        px=None
        for j in range(i+1,min(i+1+hold,n)):
            if l[j]<=sl: px=sl;break
            if h[j]>=tp: px=tp;break
        if px is None: j=min(i+hold,n-1); px=c[j]
        T.append((df.index[i+1], px/e-1-b.COST)); i=j+1
    return pd.DataFrame(T, columns=["t","ret"])
F = {s: S.extra(d) for s,d in d1h.items()}
def st(t):
    if len(t)==0: return "no trades"
    r=t.ret.values; pf=r[r>0].sum()/max(-r[r<=0].sum(),1e-9); eq=np.cumsum(r)
    return f"n={len(r)} win={np.mean(r>0)*100:.0f}% avg={r.mean()*100:+.2f}% PF={pf:.2f} sum={r.sum()*100:+.0f}% maxDD={(eq-np.maximum.accumulate(eq)).min()*100:.0f}%"
allt = pd.concat([sim(d1h[s],F[s],RULE).assign(sym=s) for s in d1h])
print("RULE:", RULE, "| rr>=2 | 1h | one position per coin")
print("H1 (selection period):", st(allt[allt.t<mid]))
print("H2 (unseen)          :", st(allt[allt.t>=mid]))
print("ALL                  :", st(allt))
print("\nper coin (H2):")
for s in d1h: print(" ", s, st(allt[(allt.sym==s)&(allt.t>=mid)]))
print("\nper month:")
allt["m"]=allt.t.dt.to_period("M")
for m,g in allt.groupby("m"): print(" ", m, st(g))
print("\nsensitivity H2 (rr_min, hold):")
for rr in (1.5,2,2.5,3):
    for hold in (24,48,96):
        t=pd.concat([sim(d1h[s],F[s],RULE,rr_min=rr,hold=hold) for s in d1h]); print(f"  rr>={rr} hold={hold}:", st(t[t.t>=mid]))
print("\ndrop-one rule H2:")
for r in RULE:
    rules=[x for x in RULE if x!=r]; t=pd.concat([sim(d1h[s],F[s],rules) for s in d1h]); print(f"  without {r}:", st(t[t.t>=mid]))
allt.to_csv("trades_1h_rule.csv", index=False)
