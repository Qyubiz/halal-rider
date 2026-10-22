"""Dynamic fully-funded resting bids (hourly bot). Settings chosen on 2021-23, tested 2024-26."""
import sys, itertools
sys.path.insert(0, "/home/claude/bt/lab2")
import numpy as np, pandas as pd
import lab
from lab import FEE, SLIP, DESIGN_END, metrics

data_dir, dead_dir = sys.argv[1], sys.argv[2]
D = lab.load(data_dir)
for s, cut in {"LUNAUSDT": "2022-05-13 23:00", "FTTUSDT": "2023-09-22 07:00", "SRMUSDT": "2022-11-28 23:00"}.items():
    d = pd.read_csv(f"{dead_dir}/{s}.csv", index_col=0, parse_dates=True); D[s] = d[d.index <= cut]
P = lab.panels(D); idx = P["c"].index
C = P["c"].ffill(); ref = P["c"].rolling(24, min_periods=1).max().shift(1).values
O, H, Lw, Cv = P["o"].values, P["h"].values, P["l"].values, C.values

def run(depth=0.10, N=3, fmax=0.33, tp=0.02, hm=96, pen=0.0):
    eq, lg = lab.dyn_kernel(O, H, Lw, Cv, ref, depth, N, fmax, tp, hm, 24, 0.001, FEE, SLIP, pen)
    return pd.Series(eq, idx), lg

rows = []
for depth, N, fmax, tp, hm in itertools.product([0.08, 0.10, 0.12, 0.15], [1, 2, 3, 5, 8], [0.2, 0.33, 0.5, 1.0], [0.015, 0.02, 0.03, 0.05], [48, 96, 168]):
    if fmax * N < 0.99 and fmax < 1.0 and N * fmax < 1: pass
    e, _ = run(depth, N, fmax, tp, hm)
    d, t = metrics(e, end=DESIGN_END), metrics(e, start=DESIGN_END)
    y22 = e[(e.index >= "2022-01-01") & (e.index < "2023-01-01")]
    rows.append(dict(depth=depth, N=N, fmax=fmax, tp=tp, hm=hm, d_cagr=d["cagr"], d_dd=d["maxdd"], t_cagr=t["cagr"], t_avg=t["avg_m"],
                     t_med=t["med_m"], t_pos=t["pos_m"], t_worst=t["worst_m"], t_dd=t["maxdd"], y2022=y22.iloc[-1] / y22.iloc[0] - 1))
R = pd.DataFrame(rows)
R.to_csv("/home/claude/bt/out/dyn_grid.csv", index=False)
print(f"{len(R)} settings | design positive {np.mean(R.d_cagr>0)*100:.0f}% | test positive {np.mean(R.t_cagr>0)*100:.0f}% | test CAGR median {R.t_cagr.median()*100:+.0f}% p10 {R.t_cagr.quantile(.1)*100:+.0f}% p90 {R.t_cagr.quantile(.9)*100:+.0f}%")
for col in ("depth", "N", "fmax", "tp", "hm"):
    print(f"  median test CAGR by {col}:", R.groupby(col).t_cagr.median().map(lambda x: f"{x*100:+.0f}%").to_dict())
# pick on DESIGN: best CAGR/|maxDD| with design maxDD better than -35%
ok = R[R.d_dd > -0.35]
sel = ok.assign(score=ok.d_cagr / -ok.d_dd).sort_values("score", ascending=False).head(5)
print("\nTop-5 on DESIGN (CAGR/maxDD, DD>-35%) and how they did on TEST:")
for _, r in sel.iterrows():
    print(f"  depth {r.depth:.0%} N={int(r.N)} max/order {r.fmax:.0%} tp {r.tp:.1%} hold {int(r.hm)}h | 21-23 CAGR {r.d_cagr*100:+.0f}% DD {r.d_dd*100:.0f}% 2022 {r.y2022*100:+.0f}% | 24-26 CAGR {r.t_cagr*100:+.0f}% avg/mo {r.t_avg*100:+.1f}% med/mo {r.t_med*100:+.1f}% up-mo {r.t_pos*100:.0f}% worst-mo {r.t_worst*100:+.0f}% DD {r.t_dd*100:.0f}%")
best = sel.iloc[0]
print("\nfill-quality sensitivity for the design pick:")
for pen in (0.0, 0.002, 0.005):
    e, lg = run(best.depth, int(best.N), best.fmax, best.tp, int(best.hm), pen)
    t = metrics(e, start=DESIGN_END); d = metrics(e, end=DESIGN_END)
    print(f"  fills {pen:.1%} worse: 21-23 x{d['mult']:.2f} | 24-26 x{t['mult']:.2f} avg/mo {t['avg_m']*100:+.1f}% up-mo {t['pos_m']*100:.0f}% DD {t['maxdd']*100:.0f}%")
e, lg = run(best.depth, int(best.N), best.fmax, best.tp, int(best.hm))
e.to_pickle("/home/claude/bt/out/dyn_best.pkl")
L = pd.DataFrame(lg, columns=["eb", "xb", "k", "j", "ret", "amt"])
print(f"trades {len(L)} ({len(L)/69.7:.0f}/month), win {np.mean(L.ret>0)*100:.0f}%, avg {L.ret.mean()*100:+.2f}%, worst {L.ret.min()*100:+.0f}%")
