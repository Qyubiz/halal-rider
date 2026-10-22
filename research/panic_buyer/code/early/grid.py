import glob, os, itertools, pandas as pd, numpy as np
import backtest as b
data = {os.path.basename(p)[:-4]: pd.read_csv(p, index_col=0, parse_dates=True) for p in sorted(glob.glob("data/*.csv"))}
scores = {s: b.indicators(d).sum(axis=1) for s, d in data.items()}
mid = next(iter(data.values())).index[8640]
rows=[]
for fee, mtp, tpl, slm, hold in itertools.product([0.001,0.00075],[0.004,0.008,0.012,0.016],[48,96],[0.015,0.025],[48,96]):
    b.FEE=fee; b.COST=2*(fee+b.SLIP); b.MIN_NET_TP=mtp; b.TP_LOOKBACK=tpl; b.SL_MAX=slm; b.MAX_HOLD=hold
    t=pd.concat([b.run(data[s],scores[s],10).assign(sym=s) for s in data])
    a=b.stats(t[t.entry_t<mid]); c=b.stats(t[t.entry_t>=mid])
    rows.append(dict(fee=fee,min_tp=mtp,tp_lb=tpl,sl_max=slm,hold=hold,n1=a["trades"],win1=a["win"],avg1=a["avg"],n2=c["trades"],win2=c["win"],avg2=c["avg"],tot2=c["total"]))
df=pd.DataFrame(rows).sort_values("avg1",ascending=False)
df.to_csv("grid.csv",index=False)
print(df.head(10).to_string())
print("configs with avg>0 in BOTH halves:", ((df.avg1>0)&(df.avg2>0)).sum(), "of", len(df))
print(df[(df.avg1>0)&(df.avg2>0)].to_string())
