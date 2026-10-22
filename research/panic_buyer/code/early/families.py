import glob, os, numpy as np, pandas as pd
import backtest as b
from backtest import ema, rma
COST=b.COST
raw = {os.path.basename(p)[:-4]: pd.read_csv(p, index_col=0, parse_dates=True) for p in sorted(glob.glob("data1h/*.csv"))}
def tf(d, rule):
    return d if rule=="1h" else d.resample(rule).agg({"o":"first","h":"max","l":"min","c":"last","v":"sum"}).dropna()
def atr_(d,n=14):
    tr=pd.concat([d.h-d.l,(d.h-d.c.shift()).abs(),(d.l-d.c.shift()).abs()],axis=1).max(axis=1); return rma(tr,n)
def rsi_(c,n=14):
    dd=c.diff(); return 100-100/(1+rma(dd.clip(lower=0),n)/rma((-dd).clip(lower=0),n))

def sim(d, entry, exit_sig=None, mode="trail", k=3.0, maxhold=200, sl_k=2.5):
    o,h,l,c=d.o.values,d.h.values,d.l.values,d.c.values; a=atr_(d).values; n=len(d); i=250; R=[]
    while i<n-2:
        if not entry[i]: i+=1; continue
        e=o[i+1]; stop=e-sl_k*a[i] if mode!="trail" else e-k*a[i]; hh=e; px=None
        for j in range(i+1,min(i+1+maxhold,n)):
            if l[j]<=stop: px=min(stop,o[j]); break
            if mode=="trail":
                hh=max(hh,h[j]); stop=max(stop,hh-k*a[j])
            elif exit_sig is not None and exit_sig[j]:
                px=c[j]; break
        if px is None: j=min(i+maxhold,n-1); px=c[j]
        R.append((d.index[i+1],px/e-1-COST)); i=j+1
    return pd.DataFrame(R,columns=["t","ret"])

def families(d, rule):
    c,h,l,v=d.c,d.h,d.l,d.v; e200=ema(c,200); e50=ema(c,50); e20=ema(c,20); r=rsi_(c)
    mid=c.rolling(20).mean(); sd=c.rolling(20).std(); lowb=mid-2*sd
    F={}
    for N in (20,55):
        brk=(c>h.rolling(N).max().shift(1)).values
        F[f"breakout{N} trail3ATR"]=(brk, None, "trail", 3.0)
        F[f"breakout{N}+c>ema200 trail3ATR"]=(brk&(c>e200).values, None, "trail", 3.0)
    cross=((e20>e50)&(e20.shift()<=e50.shift())&(c>e200)).values
    F["ema20/50 cross trail3ATR"]=(cross,None,"trail",3.0)
    F["pullback rsi<35 in uptrend, exit rsi>55"]=(((e50>e200)&(r<35)).values,(r>55).values,"exit",None)
    F["pullback rsi<30 any trend, exit rsi>50"]=((r<30).values,(r>50).values,"exit",None)
    F["lower BB + c>ema200, exit mid"]=(((c<lowb)&(c>e200)).values,(c>mid).values,"exit",None)
    F["lower BB any trend, exit mid"]=((c<lowb).values,(c>mid).values,"exit",None)
    return F
def st(t):
    if len(t)==0: return "n=0"
    r=t.ret.values; pf=r[r>0].sum()/max(-r[r<=0].sum(),1e-9)
    return f"n={len(r):4d} win={np.mean(r>0)*100:3.0f}% avg={r.mean()*100:+.2f}% PF={pf:4.2f}"
mid=pd.Timestamp("2025-10-21"); out=[]
for rule in ("1h","4h"):
    data={s:tf(d,rule) for s,d in raw.items()}
    names=None; allr={}
    for s,d in data.items():
        for nm,(ent,ex,mode,k) in families(d,rule).items():
            kw=dict(mode=mode,maxhold=(200 if rule=="1h" else 120))
            if k: kw["k"]=k
            if mode=="exit": kw["maxhold"]=72 if rule=="1h" else 40
            allr.setdefault(nm,[]).append(sim(d,ent,ex,**kw))
    print(f"\n===== {rule} candles, 8 coins, 2 years, after 0.24% fees =====")
    for nm,lst in allr.items():
        t=pd.concat(lst); y1,y2=t[t.t<mid],t[t.t>=mid]
        flag="  <-- positive both years" if (y1.ret.mean()>0 and y2.ret.mean()>0) else ""
        print(f"{nm:42s} Y1 {st(y1)} | Y2 {st(y2)}{flag}")
        out.append((rule,nm,len(t),t.ret.mean()*100,y1.ret.mean()*100,y2.ret.mean()*100))
pd.DataFrame(out,columns=["tf","strategy","n","avg_all","avg_y1","avg_y2"]).to_csv("families.csv",index=False)
# buy & hold reference
bh=[(d.c.iloc[-1]/d.c.iloc[0]-1)*100 for d in raw.values()]; print("\nBuy&hold 2y per coin %:",[round(x) for x in bh],"avg",round(np.mean(bh)))
y1bh=[(d.c[d.index<mid].iloc[-1]/d.c.iloc[0]-1)*100 for d in raw.values()]; print("Buy&hold year1 avg %:",round(np.mean(y1bh)))
