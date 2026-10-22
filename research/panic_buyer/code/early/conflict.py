import glob, os, numpy as np, pandas as pd
import backtest as b
from backtest import ema, rma
from profit_search import to_1h
d15 = {os.path.basename(p)[:-4]: pd.read_csv(p, index_col=0, parse_dates=True) for p in sorted(glob.glob("data/*.csv"))}

def tri(df):
    """+1 bullish, -1 bearish, 0 neutral for each indicator."""
    c,h,l,v,o = df.c,df.h,df.l,df.v,df.o
    T = pd.DataFrame(index=df.index)
    sgn = lambda a,bb: np.sign(a-bb)
    T["ema9/21"]=sgn(ema(c,9),ema(c,21)); T["ema21/50"]=sgn(ema(c,21),ema(c,50)); T["c/ema200"]=sgn(c,ema(c,200))
    tp=(h+l+c)/3; day=df.index.date
    T["vwap"]=sgn(c,(tp*v).groupby(day).cumsum()/v.groupby(day).cumsum())
    d=c.diff(); rsi=100-100/(1+rma(d.clip(lower=0),14)/rma((-d).clip(lower=0),14))
    T["rsi"]=np.where(rsi>55,1,np.where(rsi<45,-1,0))
    macd=ema(c,12)-ema(c,26); sig=ema(macd,9); hist=macd-sig
    T["macd"]=sgn(macd,sig); T["hist"]=sgn(hist,hist.shift())
    ll,hh=l.rolling(14).min(),h.rolling(14).max(); k=(100*(c-ll)/(hh-ll)).rolling(3).mean(); T["stoch"]=sgn(k,k.rolling(3).mean())
    up,dn=h.diff(),-l.diff()
    pdm=pd.Series(np.where((up>dn)&(up>0),up,0.0),index=df.index); mdm=pd.Series(np.where((dn>up)&(dn>0),dn,0.0),index=df.index)
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1); atr=rma(tr,14)
    pdi=100*rma(pdm,14)/atr; mdi=100*rma(mdm,14)/atr; adx=rma(100*(pdi-mdi).abs()/(pdi+mdi),14)
    T["adx"]=np.where(adx<20,0,np.sign(pdi-mdi))
    obv=(np.sign(c.diff()).fillna(0)*v).cumsum(); T["obv"]=sgn(obv,ema(obv,20))
    tpm=tp.rolling(20).mean(); mad=tp.rolling(20).apply(lambda x:np.abs(x-x.mean()).mean(),raw=True); cci=(tp-tpm)/(0.015*mad)
    T["cci"]=np.where(cci>50,1,np.where(cci<-50,-1,0))
    clv=((c-l)-(h-c))/(h-l).replace(0,np.nan); cmf=(clv*v).rolling(20).sum()/v.rolling(20).sum(); T["cmf"]=np.sign(cmf)
    mf=tp*v; pos=mf.where(tp>tp.shift(),0).rolling(14).sum(); neg=mf.where(tp<tp.shift(),0).rolling(14).sum(); mfi=100-100/(1+pos/neg)
    T["mfi"]=np.where(mfi>55,1,np.where(mfi<45,-1,0))
    ao=((h+l)/2).rolling(5).mean()-((h+l)/2).rolling(34).mean(); T["ao"]=np.sign(ao)
    T["roc10"]=np.sign(c.pct_change(10))
    # supertrend
    a=rma(tr,10).values; hl2=((h+l)/2).values; cv=c.values; ub,lb=hl2+3*a,hl2-3*a; fub,flb=ub.copy(),lb.copy(); tr_=np.ones(len(cv),bool)
    for i in range(1,len(cv)):
        fub[i]=ub[i] if (ub[i]<fub[i-1] or cv[i-1]>fub[i-1]) else fub[i-1]
        flb[i]=lb[i] if (lb[i]>flb[i-1] or cv[i-1]<flb[i-1]) else flb[i-1]
        tr_[i]=False if (tr_[i-1] and cv[i]<flb[i]) else True if ((not tr_[i-1]) and cv[i]>fub[i]) else tr_[i-1]
    T["supertrend"]=np.where(tr_,1,-1)
    for rule in ("1h","4h"):
        rc=c.resample(rule).last().dropna(); t=np.sign(ema(rc,20)-ema(rc,50)).shift(1)
        T[f"{rule} trend"]=t.reindex(df.index,method="ffill")
    T=T.fillna(0).astype(int); T.iloc[:250]=0; return T

def sim(df, sig, rr_min=2.0, hold=48, tp_lb=48, sl_lb=20):
    o,h,l,c=df.o.values,df.h.values,df.l.values,df.c.values
    lowN=df.l.rolling(sl_lb).min().values; highN=df.h.rolling(tp_lb).max().values
    n=len(df); i=250; R=[]
    while i<n-2:
        if not sig[i]: i+=1; continue
        e=o[i+1]; sl=max(lowN[i]*0.999,e*0.96); sl=min(sl,e*0.996); tp=highN[i]
        if tp<=e*1.004 or (tp/e-1)/(1-sl/e)<rr_min: i+=1; continue
        px=None
        for j in range(i+1,min(i+1+hold,n)):
            if l[j]<=sl: px=sl;break
            if h[j]>=tp: px=tp;break
        if px is None: j=min(i+hold,n-1); px=c[j]
        R.append((df.index[i+1],px/e-1-b.COST)); i=j+1
    return pd.DataFrame(R,columns=["t","ret"])
def st(t):
    if len(t)==0: return "n=0"
    r=t.ret.values; pf=r[r>0].sum()/max(-r[r<=0].sum(),1e-9)
    return f"n={len(r):3d} win={np.mean(r>0)*100:3.0f}% avg={r.mean()*100:+.2f}% PF={pf:.2f}"
for name,data in (("1h",{s:to_1h(d) for s,d in d15.items()}),("15m",d15)):
    TT={s:tri(d) for s,d in data.items()}; ncol=TT[next(iter(TT))].shape[1]
    mid=next(iter(data.values())).index[len(next(iter(data.values())))//2]
    print(f"\n=== {name}: {ncol} indicators, rr>=2. Rule: zero bearish indicators AND at least K bullish")
    for K in (0,4,6,8,10,12):
        rows=[]
        for s,d in data.items():
            bull=(TT[s]==1).sum(axis=1).values; bear=(TT[s]==-1).sum(axis=1).values
            rows.append(sim(d,(bear==0)&(bull>=K)))
        t=pd.concat(rows); print(f" K>={K:2d}  H1: {st(t[t.t<mid])} | H2: {st(t[t.t>=mid])}")
    # looser: allow at most 1 / 2 bearish
    for mb in (1,2,3):
        rows=[]
        for s,d in data.items():
            bull=(TT[s]==1).sum(axis=1).values; bear=(TT[s]==-1).sum(axis=1).values
            rows.append(sim(d,(bear<=mb)&(bull>=8)))
        t=pd.concat(rows); print(f" bear<={mb}, bull>=8  H1: {st(t[t.t<mid])} | H2: {st(t[t.t>=mid])}")
