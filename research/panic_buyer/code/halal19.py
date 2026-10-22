import sys
sys.path.insert(0, "/home/claude/bt/lab2")
import numpy as np, pandas as pd
import lab
from lab import FEE, SLIP, DESIGN_END, metrics
H19 = ["BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","BNBUSDT","DOGEUSDT","ADAUSDT","TRXUSDT","LINKUSDT","AVAXUSDT",
       "SUIUSDT","LTCUSDT","BCHUSDT","HBARUSDT","DOTUSDT","NEARUSDT","ARBUSDT","FETUSDT","RENDERUSDT"]
D = {}
for s in H19:
    for d_ in ("/home/claude/bt/data_long", "/home/claude/bt/data_halal"):
        try: D[s] = pd.read_csv(f"{d_}/{s}.csv", index_col=0, parse_dates=True); break
        except FileNotFoundError: pass
r1 = pd.read_csv("/home/claude/bt/data_halal/RNDRUSDT.csv", index_col=0, parse_dates=True)
D["RENDERUSDT"] = pd.concat([r1, D["RENDERUSDT"]]).sort_index()
D["RENDERUSDT"] = D["RENDERUSDT"][~D["RENDERUSDT"].index.duplicated()]
def run(Dx, depth, tp, hm, label):
    P = lab.panels(Dx); idx = P["c"].index
    C = P["c"].ffill(); ref = P["c"].rolling(24, min_periods=1).max().shift(1).values
    eq, lg = lab.dyn_kernel(P["o"].values, P["h"].values, P["l"].values, C.values, ref, depth, 8, 0.20, tp, hm, 24, 0.001, FEE, SLIP, 0.0)
    e = pd.Series(eq, idx); d, t = metrics(e, end=DESIGN_END), metrics(e, start=DESIGN_END)
    y = e.resample("YE").last(); y = pd.concat([pd.Series([1.0], [e.index[0]]), y]).pct_change().dropna()
    L = pd.DataFrame(lg, columns=["eb","xb","k","j","ret","amt"])
    print(f"{label:44s} 21-23 x{d['mult']:5.2f} DD {d['maxdd']*100:4.0f}% | 24-26 RM6k->RM{6000*t['mult']:,.0f} /mo {(t['mult']**(1/33.7)-1)*100:+.2f}% up-mo {t['pos_m']*100:.0f}% worst-mo {t['worst_m']*100:+.0f}% DD {t['maxdd']*100:.0f}% | trades/mo {len(L)/69.7:.0f} win {np.mean(L.ret>0)*100:.0f}%")
    print(f"{'':44s} per year: " + " ".join(f"{k.year}:{v*100:+.0f}%" for k, v in y.items()))
    return e
eB = run(D, 0.12, 0.015, 48, "HALAL-19, steady 12%/+1.5%/48h")
run(D, 0.10, 0.02, 96, "HALAL-19, aggressive 10%/+2%/96h")
Dd = dict(D)
for s, cut in {"LUNAUSDT": "2022-05-13 23:00", "FTTUSDT": "2023-09-22 07:00", "SRMUSDT": "2022-11-28 23:00"}.items():
    d = pd.read_csv(f"/home/claude/bt/data_dead/{s}.csv", index_col=0, parse_dates=True); Dd[s] = d[d.index <= cut]
run(Dd, 0.12, 0.015, 48, "HALAL-19 + LUNA/FTT/SRM, steady")
eB.to_pickle("/home/claude/bt/out/halal19_steady.pkl")
t = eB[eB.index >= "2024-01-01"]; m = t.resample("ME").last(); m = pd.concat([pd.Series([t.iloc[0]], [t.index[0]]), m]).pct_change().dropna()
print("2024-26 months:", " ".join(f"{k:%y%m}:{v*100:+.1f}" for k, v in m.items()))
# last 6 months only
r6 = eB[eB.index >= "2026-04-21"]; print(f"last 6 months: {(r6.iloc[-1]/r6.iloc[0]-1)*100:+.1f}%")
