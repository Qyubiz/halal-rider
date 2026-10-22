"""Same dynamic dip-buyer, but the bot runs every 15 minutes (15m candles). Includes LUNA/FTT/SRM collapses."""
import sys, itertools
sys.path.insert(0, "/home/claude/bt/lab2")
import numpy as np, pandas as pd
import lab
from lab import FEE, SLIP, DESIGN_END, metrics

d15, dead15 = sys.argv[1], sys.argv[2]
D = lab.load(d15)
for s, cut in {"LUNAUSDT": "2022-05-13 23:45", "FTTUSDT": "2023-09-22 07:45", "SRMUSDT": "2022-11-28 23:45"}.items():
    d = pd.read_csv(f"{dead15}/{s}.csv", index_col=0, parse_dates=True); D[s] = d[d.index <= cut]
idx = pd.date_range(min(d.index[0] for d in D.values()), max(d.index[-1] for d in D.values()), freq="15min")
P = {f: pd.DataFrame({s: d[f] for s, d in D.items()}).reindex(idx) for f in ("o", "h", "l", "c")}
del D
C = P["c"].ffill(); ref = P["c"].rolling(96, min_periods=1).max().shift(1).values
O, H, Lw, Cv = P["o"].values, P["h"].values, P["l"].values, C.values
print(f"15m panel: {len(idx)} bars x {O.shape[1]} coins")

def run(depth=0.10, N=8, fmax=0.20, tp=0.02, hm_h=96, pen=0.0):
    eq, lg = lab.dyn_kernel(O, H, Lw, Cv, ref, depth, N, fmax, tp, hm_h * 4, 96, 0.001, FEE, SLIP, pen)
    return pd.Series(eq, idx), lg

def show(name, e):
    d, t = metrics(e, end=DESIGN_END), metrics(e, start=DESIGN_END)
    y = e.resample("YE").last(); y = pd.concat([pd.Series([1.0], [e.index[0]]), y]).pct_change().dropna()
    print(f"{name:40s} 21-23 x{d['mult']:5.2f} DD {d['maxdd']*100:4.0f}% | 24-26 x{t['mult']:4.2f} CAGR {t['cagr']*100:+.0f}% avg/mo {t['avg_m']*100:+.1f}% med/mo {t['med_m']*100:+.1f}% up-mo {t['pos_m']*100:.0f}% worst-mo {t['worst_m']*100:+.0f}% DD {t['maxdd']*100:.0f}%")
    print(f"{'':40s} per year: " + " ".join(f"{k.year}:{v*100:+.0f}%" for k, v in y.items()))
    return d, t

e, lg = run(); show("15-MIN BOT, same settings as hourly pick", e)
e.to_pickle("/home/claude/bt/out/dyn15_pick.pkl")
L = pd.DataFrame(lg, columns=["eb", "xb", "k", "j", "ret", "amt"])
print(f"   trades {len(L)} ({len(L)/69.7:.0f}/month), win {np.mean(L.ret>0)*100:.0f}%, avg win {L[L.ret>0].ret.mean()*100:+.2f}% avg loss {L[L.ret<=0].ret.mean()*100:+.2f}% worst {L.ret.min()*100:+.0f}%")
for pen in (0.002, 0.005): show(f"  fills {pen:.1%} worse", run(pen=pen)[0])

rows = []
for depth, N, fmax, tp, hm in itertools.product([0.08, 0.10, 0.12], [3, 5, 8], [0.2, 0.33], [0.015, 0.02, 0.03], [48, 96]):
    e_, _ = run(depth, N, fmax, tp, hm)
    d, t = metrics(e_, end=DESIGN_END), metrics(e_, start=DESIGN_END)
    rows.append(dict(depth=depth, N=N, fmax=fmax, tp=tp, hm=hm, d=d["cagr"], ddd=d["maxdd"], t=t["cagr"], tavg=t["avg_m"], tdd=t["maxdd"], tpos=t["pos_m"]))
R = pd.DataFrame(rows); R.to_csv("/home/claude/bt/out/dyn15_grid.csv", index=False)
print(f"\nneighbourhood {len(R)} settings | design positive {np.mean(R.d>0)*100:.0f}% | test positive {np.mean(R.t>0)*100:.0f}% | test CAGR median {R.t.median()*100:+.0f}% p10 {R.t.quantile(.1)*100:+.0f}% p90 {R.t.quantile(.9)*100:+.0f}%")
for col in ("depth", "N", "fmax", "tp", "hm"):
    print(f"  median test CAGR by {col}:", R.groupby(col).t.median().map(lambda x: f"{x*100:+.0f}%").to_dict())
ok = R[R.ddd > -0.35]; b = ok.assign(s=ok.d / -ok.ddd).sort_values("s", ascending=False).iloc[0]
print(f"best on DESIGN (15m): depth {b.depth:.0%} N={int(b.N)} max/order {b.fmax:.0%} tp {b.tp:.1%} hold {int(b.hm)}h -> 21-23 CAGR {b.d*100:+.0f}% DD {b.ddd*100:.0f}% | 24-26 CAGR {b.t*100:+.0f}% avg/mo {b.tavg*100:+.1f}% up-mo {b.tpos*100:.0f}% DD {b.tdd*100:.0f}%")
