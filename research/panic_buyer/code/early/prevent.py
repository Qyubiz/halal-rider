import glob, os, itertools, numpy as np, pandas as pd
import backtest as b, search as S
from backtest import ema, rma
from profit_search import to_1h
d15 = {os.path.basename(p)[:-4]: pd.read_csv(p, index_col=0, parse_dates=True) for p in sorted(glob.glob("data/*.csv"))}
D = {s: to_1h(d) for s, d in d15.items()}
RULE = ["mfi50-80","don mid","c>vwap","cmf>0","macd>sig"]
F = {s: S.extra(d) for s, d in D.items()}
idx = next(iter(D.values())).index; mid = idx[len(idx)//2]

def htf(df, rule, fn):
    rc = df.c.resample(rule).last().dropna(); return fn(rc).shift(1).reindex(df.index, method="ffill").fillna(False).values
btc_ok = htf(D["BTCUSDT"], "4h", lambda rc: rc > ema(rc, 20))
filters = {}
for s, d in D.items():
    filters[s] = dict(
        btc = btc_ok,
        daily = htf(d, "1D", lambda rc: rc > ema(rc, 10)),
        nopump = (d.c.pct_change(24) < 0.06).values,
        calm = (d.c.pct_change(24) > -0.08).values)

def sim(s, extra=(), be_at=None, rr_min=2.0, hold=48):
    df=D[s]; o,h,l,c=df.o.values,df.h.values,df.l.values,df.c.values
    lowN=df.l.rolling(20).min().values; highN=df.h.rolling(48).max().values
    sig = F[s][RULE].all(axis=1).values
    for k in extra: sig = sig & filters[s][k]
    n=len(df); i=250; R=[]
    while i<n-2:
        if not sig[i]: i+=1; continue
        e=o[i+1]; sl=max(lowN[i]*0.999,e*0.96); sl=min(sl,e*0.996); tp=highN[i]
        if tp<=e*1.004 or (tp/e-1)/(1-sl/e)<rr_min: i+=1; continue
        risk=e-sl; stop=sl; px=None; why="TIME"
        for j in range(i+1,min(i+1+hold,n)):
            if l[j]<=stop: px=stop; why="SL" if stop<=sl else "BE"; break
            if h[j]>=tp: px=tp; why="TP"; break
            if be_at and h[j]>=e+be_at*risk: stop=max(stop, e*(1+b.COST))   # move stop to breakeven for NEXT bar
        if px is None: j=min(i+hold,n-1); px=c[j]
        R.append((df.index[i+1], df.index[j], px/e-1-b.COST, why, s)); i=j+1
    return pd.DataFrame(R,columns=["t","xt","ret","why","sym"])

def post(t, cooldown=0, brk=0):
    t=t.sort_values("t").reset_index(drop=True); keep=[]; last_sl={}; hist=[]; pause_until=pd.Timestamp.min
    for r in t.itertuples():
        if r.t < pause_until: continue
        if cooldown and r.sym in last_sl and r.t < last_sl[r.sym]+pd.Timedelta(hours=cooldown): continue
        keep.append(r.Index)
        if r.why=="SL": last_sl[r.sym]=r.xt
    t=t.loc[keep]
    if brk:   # after `brk` consecutive losses (by exit time) pause 48h from the third loss's exit
        out=[]; streak=0; pause=pd.Timestamp.min; done=t.sort_values("xt")
        losses_end={}
        for r in done.itertuples():
            streak = streak+1 if r.ret<=0 else 0
            if streak>=brk: pause=r.xt+pd.Timedelta(hours=48); streak=0; losses_end[r.xt]=pause
        # drop trades that start inside any pause window
        wins=list(losses_end.items()); 
        def blocked(tt): return any(x < tt < p for x,p in wins)
        t=t[~t.t.apply(blocked)]
    return t
def stat(t):
    if len(t)==0: return "n=0"
    r=t.ret.values; pf=r[r>0].sum()/max(-r[r<=0].sum(),1e-9); eq=np.cumsum(r[np.argsort(t.xt.values)])
    return f"n={len(r):3d} win={np.mean(r>0)*100:3.0f}% avg={r.mean()*100:+.2f}% PF={pf:4.2f} sum={r.sum()*100:+4.0f}% DD={(eq-np.maximum.accumulate(eq)).min()*100:+4.0f}%"
def month_worst(t):
    m=t.groupby(t.xt.dt.to_period("M")).ret.sum()*100; return f"worst month {m.min():+.0f}%"
configs = [
 ("base",                      dict()),
 ("+BTC uptrend",              dict(extra=("btc",))),
 ("+own daily uptrend",        dict(extra=("daily",))),
 ("+skip if up>6% in 24h",     dict(extra=("nopump",))),
 ("+skip if down>8% in 24h",   dict(extra=("calm",))),
 ("breakeven stop at +1R",     dict(be_at=1.0)),
 ("breakeven stop at +1.5R",   dict(be_at=1.5)),
 ("cooldown 24h after stop",   dict(post=dict(cooldown=24))),
 ("pause 48h after 3 losses",  dict(post=dict(brk=3))),
 ("BTC+nopump",                dict(extra=("btc","nopump"))),
 ("BTC+nopump+BE1R",           dict(extra=("btc","nopump"), be_at=1.0)),
 ("BTC+daily+BE1R+cool24",     dict(extra=("btc","daily"), be_at=1.0, post=dict(cooldown=24))),
 ("BTC+nopump+BE1R+pause",     dict(extra=("btc","nopump"), be_at=1.0, post=dict(brk=3))),
]
res=[]
for name,cfg in configs:
    t=pd.concat([sim(s,cfg.get("extra",()),cfg.get("be_at")) for s in D])
    if "post" in cfg: t=post(t,**cfg["post"])
    print(f"{name:26s} H1 {stat(t[t.t<mid])}\n{'':26s} H2 {stat(t[t.t>=mid])} | {month_worst(t)}")
    res.append((name,t))
