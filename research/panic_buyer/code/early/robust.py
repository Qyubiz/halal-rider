import numpy as np, pandas as pd, itertools
from families import raw, tf, atr_, sim, ema, st
import families as fam
mid=pd.Timestamp("2025-10-21")
def run(rule, fast, slow, k, trend200=True, hold=120):
    R=[]
    for s,d0 in raw.items():
        d=tf(d0,rule); c=d.c; ef,es,e200=ema(c,fast),ema(c,slow),ema(c,200)
        ent=((ef>es)&(ef.shift()<=es.shift())&((c>e200) if trend200 else True)).values
        t=sim(d,ent,None,mode="trail",k=k,maxhold=hold); t["sym"]=s; R.append(t)
    return pd.concat(R)
print("--- neighbouring settings (4h unless noted); Y1 | Y2 ---")
for rule,fast,slow,k,t200 in [("4h",20,50,3,True),("4h",20,50,2.5,True),("4h",20,50,4,True),("4h",10,30,3,True),("4h",30,100,3,True),("4h",20,50,3,False),("4h",9,21,3,True),("6h",20,50,3,True),("12h",20,50,3,True),("2h",20,50,3,True)]:
    try: t=run(rule,fast,slow,k,t200,hold=120)
    except Exception as e: print(rule,e); continue
    y1,y2=t[t.t<mid],t[t.t>=mid]; flag=" +both" if (y1.ret.mean()>0 and y2.ret.mean()>0) else ""
    print(f"{rule:3s} ema{fast}/{slow} trail{k} {'c>ema200' if t200 else 'no200  '} Y1 {st(y1)} | Y2 {st(y2)}{flag}")
base=run("4h",20,50,3,True)
print("\n--- base: per coin (both years) ---")
for s,g in base.groupby("sym"): print(f"  {s}: {st(g)}")
base["q"]=base.t.dt.to_period("Q")
print("\n--- base: per quarter ---")
for q,g in base.groupby("q"): print(f"  {q}: {st(g)}")
r=base.ret.values; print("\nall:",st(base),"| avg win %.1f%% avg loss %.1f%%"%(r[r>0].mean()*100,r[r<=0].mean()*100))
# bootstrap: chance avg<=0
rng=np.random.default_rng(0); bs=[rng.choice(r,len(r)).mean() for _ in range(5000)]; print("bootstrap P(avg<=0) = %.3f"%np.mean(np.array(bs)<=0))
