import sys
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
R = pd.read_csv("/home/claude/bt/out/dyn_grid.csv")
print("12% / +1.5% / 48h across N and max-per-order:")
print(R[(R.depth==0.12)&(R.tp==0.015)&(R.hm==48)][["N","fmax","d_cagr","d_dd","y2022","t_cagr","t_worst","t_dd"]].round(3).to_string(index=False))
for name, (depth, tp, hm) in {"A aggressive 10%/+2%/96h": (0.10, 0.02, 96), "B steady 12%/+1.5%/48h": (0.12, 0.015, 48)}.items():
    print(f"\n===== {name} (8 orders, max 20% each) =====")
    for pen in (0.0, 0.002, 0.005):
        eq, lg = lab.dyn_kernel(O, H, Lw, Cv, ref, depth, 8, 0.20, tp, hm, 24, 0.001, FEE, SLIP, pen)
        e = pd.Series(eq, idx); t = metrics(e, start=DESIGN_END); d = metrics(e, end=DESIGN_END)
        print(f"fills {pen:.1%} worse: 21-23 x{d['mult']:.2f} DD {d['maxdd']*100:.0f}% | 24-26 x{t['mult']:.2f} RM6k->RM{6000*t['mult']:,.0f} per-mo(compounded) {(t['mult']**(1/33.7)-1)*100:+.2f}% up-mo {t['pos_m']*100:.0f}% worst-mo {t['worst_m']*100:+.0f}% DD {t['maxdd']*100:.0f}%")
        if pen == 0.0:
            e0, lg0 = e, lg
    y = e0.resample("YE").last(); y = pd.concat([pd.Series([1.0], [e0.index[0]]), y]).pct_change().dropna()
    print("per year:", " ".join(f"{k.year}:{v*100:+.0f}%" for k, v in y.items()))
    tt = e0[e0.index >= DESIGN_END]; m = tt.resample("ME").last(); m = pd.concat([pd.Series([tt.iloc[0]], [tt.index[0]]), m]).pct_change().dropna()
    print("2024-26 months:", " ".join(f"{k:%y%m}:{v*100:+.1f}" for k, v in m.items()))
    L = pd.DataFrame(lg0, columns=["eb", "xb", "k", "j", "ret", "amt"])
    print(f"trades {len(L)} ({len(L)/69.7:.0f}/month), win {np.mean(L.ret>0)*100:.0f}%, avg win {L[L.ret>0].ret.mean()*100:+.2f}%, avg loss {L[L.ret<=0].ret.mean()*100:+.2f}%, worst {L.ret.min()*100:+.0f}%")
    e0.to_pickle(f"/home/claude/bt/out/final_{name[0]}.pkl")
