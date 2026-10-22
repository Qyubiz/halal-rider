"""Realistic (hourly-check) version of the panic buyer: robustness, sizing caps, tail risk."""
import sys, itertools
sys.path.insert(0, "/home/claude/bt/lab2")
import numpy as np, pandas as pd
import lab
from lab import FEE, SLIP, DESIGN_END, metrics

data_dir, dead_dir = sys.argv[1], sys.argv[2]
D = lab.load(data_dir)
dead_cut = {"LUNAUSDT": "2022-05-13 23:00", "FTTUSDT": "2023-09-22 07:00", "SRMUSDT": "2022-11-28 23:00"}
for s, cut in dead_cut.items():
    d = pd.read_csv(f"{dead_dir}/{s}.csv", index_col=0, parse_dates=True); D[s] = d[d.index <= cut]
P = lab.panels(D); idx = P["c"].index; syms = list(P["c"].columns)
C = P["c"].ffill()
A = dict(o=P["o"].values, h=P["h"].values, l=P["l"].values, c=C.values, alive=np.ones(C.shape, np.bool_))
refs = {w: P["c"].rolling(w, min_periods=1).max().shift(1).values for w in (12, 24, 48)}

def run(dep=(0.10, 0.16, 0.24), f=0.15, tp=0.02, hm=96, sl=0.0, w=24, open_only=True, pen=0.0, fr=None):
    fr = np.array(fr if fr is not None else [f * (1 + 0.5 * i) for i in range(len(dep))])
    eq, lg = lab.wick2(A["o"], A["h"], A["l"], A["c"], refs[w], np.array(dep), fr, tp, hm, sl, 24, 0.001, FEE, SLIP, pen, open_only, A["alive"])
    return pd.Series(eq, idx), lg

def row(e):
    d, t = metrics(e, end=DESIGN_END), metrics(e, start=DESIGN_END)
    return d, t

print("== HOURLY-CHECK BOT, universe 21 coins + LUNA/FTT/SRM collapses ==")
base, lg = run()
d, t = row(base)
print(f"base 10/16/24% ladder, 15% size, +2%, 96h: 21-23 x{d['mult']:.2f} (CAGR {d['cagr']*100:+.0f}%) | 24-26 x{t['mult']:.2f} (CAGR {t['cagr']*100:+.0f}%) avg/mo {t['avg_m']*100:+.1f}% med/mo {t['med_m']*100:+.1f}% up-mo {t['pos_m']*100:.0f}% worst-mo {t['worst_m']*100:+.0f}% maxDD {t['maxdd']*100:.0f}%")
yr = base.resample("YE").last(); yr = pd.concat([pd.Series([1.0], [base.index[0]]), yr]).pct_change().dropna()
print("per year:", {str(k.year): f"{v*100:+.0f}%" for k, v in yr.items()})

L = pd.DataFrame(lg, columns=["eb", "xb", "k", "j", "ret", "amt", "gap"]); L["sym"] = [syms[int(k)] for k in L.k]
L["t"] = idx[L.eb.astype(int)]; L["pnl"] = L.ret * L.amt
print(f"trades {len(L)} ({len(L)/((idx[-1]-idx[0]).days/30.44):.1f}/month), win {np.mean(L.ret>0)*100:.0f}%, avg win {L[L.ret>0].ret.mean()*100:+.2f}%, avg loss {L[L.ret<=0].ret.mean()*100:+.2f}%")
print("worst 6 trades:"); print(L.nsmallest(6, "ret")[["t", "sym", "j", "ret", "amt"]].assign(ret=lambda x: (x.ret*100).round(1), amt=lambda x: (x.amt*100).round(1)).to_string(index=False))
dead = L[L.sym.isin(dead_cut)]
print("collapse coins:", dead.groupby("sym").agg(n=("ret", "size"), avg=("ret", lambda r: f"{r.mean()*100:+.1f}%"), worst=("ret", lambda r: f"{r.min()*100:+.0f}%"), pnl_pct_of_equity=("pnl", lambda p: f"{p.sum()*100:+.1f}")).to_string())

print("\n== robustness: nearby settings, hourly bot ==")
res = []
for dep, f, tp, hm, w in itertools.product([(0.08,), (0.10,), (0.12,), (0.08, 0.14, 0.20), (0.10, 0.16, 0.24), (0.12, 0.18, 0.26)],
                                           [0.10, 0.15, 0.25], [0.015, 0.02, 0.03, 0.05], [48, 96, 168], [12, 24, 48]):
    e, _ = run(dep=dep, f=f, tp=tp, hm=hm, w=w); d, t = row(e)
    res.append(dict(dep=str(dep), f=f, tp=tp, hm=hm, w=w, d=d["cagr"], t=t["cagr"], tdd=t["maxdd"], tpos=t["pos_m"], tavg=t["avg_m"]))
R = pd.DataFrame(res)
print(f"{len(R)} settings | design CAGR>0: {np.mean(R.d>0)*100:.0f}% | test CAGR>0: {np.mean(R.t>0)*100:.0f}% | test CAGR median {R.t.median()*100:+.0f}% (p10 {R.t.quantile(.1)*100:+.0f}%, p90 {R.t.quantile(.9)*100:+.0f}%)")
for col in ("dep", "f", "tp", "hm", "w"):
    print(f"  median test CAGR by {col}:", R.groupby(col).t.median().map(lambda x: f"{x*100:+.0f}%").to_dict())
best_d = R.sort_values("d", ascending=False).iloc[0]
print(f"best on DESIGN: {best_d[['dep','f','tp','hm','w']].to_dict()} -> design {best_d.d*100:+.0f}% test {best_d.t*100:+.0f}% CAGR, test avg/mo {best_d.tavg*100:+.1f}% maxDD {best_d.tdd*100:.0f}%")
R.to_csv("/home/claude/bt/out/hourly_neighbourhood.csv", index=False)

print("\n== risk caps on base ==")
for name, kw in [("size 10% (x1/1.5/2)", dict(f=0.10)), ("size 20%", dict(f=0.20)), ("size 25%", dict(f=0.25)),
                 ("single 10% level, 30% size", dict(dep=(0.10,), f=0.30)), ("stop -30%", dict(sl=0.30)), ("stop -40%", dict(sl=0.40))]:
    e, _ = run(**kw); d, t = row(e)
    print(f"{name:28s} 21-23 x{d['mult']:5.2f} maxDD {d['maxdd']*100:4.0f}% | 24-26 x{t['mult']:5.2f} avg/mo {t['avg_m']*100:+.1f}% up-mo {t['pos_m']*100:.0f}% worst-mo {t['worst_m']*100:+.0f}% maxDD {t['maxdd']*100:.0f}%")
base.to_pickle("/home/claude/bt/out/hourly_base.pkl"); L.to_csv("/home/claude/bt/out/hourly_trades.csv", index=False)
