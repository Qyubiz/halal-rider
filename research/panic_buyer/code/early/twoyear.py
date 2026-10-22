import glob, os, numpy as np, pandas as pd
import backtest as b, search as S
from backtest import ema, rma
D = {os.path.basename(p)[:-4]: pd.read_csv(p, index_col=0, parse_dates=True) for p in sorted(glob.glob("data1h/*.csv"))}
RULE = ["mfi50-80","don mid","c>vwap","cmf>0","macd>sig"]
F = {s: S.extra(d) for s, d in D.items()}
btc24 = D["BTCUSDT"].c.pct_change(24)
def feats(s):
    d=D[s]; c,h,l,v=d.c,d.h,d.l,d.v
    dd=c.diff(); rsi=100-100/(1+rma(dd.clip(lower=0),14)/rma((-dd).clip(lower=0),14))
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1); atr=rma(tr,14)
    X=pd.DataFrame(index=d.index)
    X["rsi"]=rsi; X["atr%"]=atr/c*100; X["vol_ratio"]=v/v.rolling(20).mean()
    X["chg6"]=c.pct_change(6)*100; X["chg24"]=c.pct_change(24)*100; X["chg72"]=c.pct_change(72)*100
    X["d_ema50"]=(c/ema(c,50)-1)*100; X["d_ema200"]=(c/ema(c,200)-1)*100
    X["hour"]=d.index.hour; X["dow"]=d.index.dayofweek
    X["btc24"]=btc24.reindex(d.index)*100
    X["range_pos"]=(c-l.rolling(48).min())/(h.rolling(48).max()-l.rolling(48).min())
    X["bar_pos"]=(c-l)/(h-l).replace(0,np.nan)
    return X
def sim(s, be_at=1.0, rr_min=2.0, hold=48):
    df=D[s]; o,h,l,c=df.o.values,df.h.values,df.l.values,df.c.values
    lowN=df.l.rolling(20).min().values; highN=df.h.rolling(48).max().values
    sig=F[s][RULE].all(axis=1).values; X=feats(s); n=len(df); i=250; R=[]
    while i<n-2:
        if not sig[i]: i+=1; continue
        e=o[i+1]; sl=max(lowN[i]*0.999,e*0.96); sl=min(sl,e*0.996); tp=highN[i]
        if tp<=e*1.004 or (tp/e-1)/(1-sl/e)<rr_min: i+=1; continue
        risk=e-sl; stop=sl; px=None
        for j in range(i+1,min(i+1+hold,n)):
            if l[j]<=stop: px=stop;break
            if h[j]>=tp: px=tp;break
            if be_at and h[j]>=e+be_at*risk: stop=max(stop,e*(1+b.COST))
        if px is None: j=min(i+hold,n-1); px=c[j]
        r=dict(t=df.index[i+1],xt=df.index[j],sym=s,ret=px/e-1-b.COST,sl_pct=(1-sl/e)*100,tp_pct=(tp/e-1)*100,rr=(tp/e-1)/(1-sl/e))
        r.update(X.iloc[i].to_dict()); R.append(r); i=j+1
    return pd.DataFrame(R)
def st(t):
    if len(t)==0: return "n=0"
    r=t.ret.values; pf=r[r>0].sum()/max(-r[r<=0].sum(),1e-9)
    return f"n={len(r):3d} win={np.mean(r>0)*100:3.0f}% avg={r.mean()*100:+.2f}% PF={pf:4.2f} sum={r.sum()*100:+4.0f}%"
T=pd.concat([sim(s) for s in D]).sort_values("t").reset_index(drop=True)
T.to_csv("trades_2y.csv",index=False)
print("RULE over 2 years, with breakeven stop (1h, rr>=2):", st(T))
T["half"]=T.t.dt.year.astype(str)+"H"+((T.t.dt.month>6)+1).astype(str)
for k,g in T.groupby("half"): print(" ",k,st(g))
mid=pd.Timestamp("2025-10-21")
tr,te=T[T.t<mid],T[T.t>=mid]
print("\nYear1:",st(tr)); print("Year2:",st(te))
print("\n--- losing vs winning trade features (year 1 = train) ---")
cols=["sl_pct","tp_pct","rr","rsi","atr%","vol_ratio","chg6","chg24","chg72","d_ema50","d_ema200","hour","dow","btc24","range_pos","bar_pos"]
W,Lz=tr[tr.ret>0],tr[tr.ret<=0]
print(pd.DataFrame({"win_median":W[cols].median(),"loss_median":Lz[cols].median()}).round(2).to_string())
print("\n--- single filters: pick best threshold on year 1, test on year 2 ---")
rows=[]
for cname in cols:
    for q in (0.2,0.33,0.5,0.67,0.8):
        th=tr[cname].quantile(q)
        for side in ("<=",">"):
            m1=(tr[cname]<=th) if side=="<=" else (tr[cname]>th); m2=(te[cname]<=th) if side=="<=" else (te[cname]>th)
            if m1.sum()<40 or m2.sum()<20: continue
            rows.append(dict(f=f"{cname} {side} {th:.2f}",n1=m1.sum(),avg1=tr[m1].ret.mean()*100,win1=(tr[m1].ret>0).mean()*100,n2=m2.sum(),avg2=te[m2].ret.mean()*100,win2=(te[m2].ret>0).mean()*100,sum2=te[m2].ret.sum()*100))
R=pd.DataFrame(rows).sort_values("avg1",ascending=False)
base1,base2=tr.ret.mean()*100,te.ret.mean()*100
print(f"baseline avg year1={base1:+.2f}% year2={base2:+.2f}%")
print(R.head(12).round(2).to_string(index=False))
good=R[(R.avg1>base1)&(R.avg2>base2)]
print(f"\nfilters better than baseline in BOTH years: {len(good)} of {len(R)}")
print(good.sort_values('avg2',ascending=False).head(8).round(2).to_string(index=False))
