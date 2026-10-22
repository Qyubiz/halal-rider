"""Implementable versions: (R1) fully-funded resting limit orders refreshed hourly, (R2) hourly market buys,
with bear-market filters chosen on 2021-23. Universe includes LUNA/FTT/SRM up to their collapse."""
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
P = lab.panels(D); idx = P["c"].index; syms = list(P["c"].columns); K = len(syms)
C = P["c"].ffill()
O, H, Lw, Cv = P["o"].values, P["h"].values, P["l"].values, C.values
alive = np.ones(C.shape, np.bool_)
ref24 = P["c"].rolling(24, min_periods=1).max().shift(1)

# entry gates (computed from data up to the previous day)
DC = C.resample("1D").last()
def hourly(g): return g.shift(1).astype("boolean").fillna(False).astype(bool).reindex(idx.floor("D")).fillna(False).values
btc = DC["BTCUSDT"]; bidx = (1 + DC.pct_change().mean(axis=1).fillna(0)).cumprod()
gates = {"none": np.ones(len(idx), bool),
         "BTC>SMA200d": hourly(btc > btc.rolling(200).mean()),
         "BTC>SMA100d": hourly(btc > btc.rolling(100).mean()),
         "basket>SMA100d": hourly(bidx > bidx.rolling(100).mean()),
         "BTC 7d ret>-10%": hourly(btc / btc.shift(7) - 1 > -0.10),
         "BTC 30d ret>-20%": hourly(btc / btc.shift(30) - 1 > -0.20)}

def run(dep=(0.10,), fr=(0.15,), tp=0.02, hm=96, open_only=False, gate="none", cols=None):
    ref = ref24.values.copy()
    ref[~gates[gate], :] = np.nan
    if cols is not None:
        mask = np.array([s in cols for s in syms]); ref[:, ~mask] = np.nan
    eq, lg = lab.wick2(O, H, Lw, Cv, ref, np.array(dep, float), np.array(fr, float), tp, hm, 0.0, 24, 0.001, FEE, SLIP, 0.0, open_only, alive)
    return pd.Series(eq, idx), lg

def show(name, e):
    d, t = metrics(e, end=DESIGN_END), metrics(e, start=DESIGN_END)
    y = e.resample("YE").last(); y = pd.concat([pd.Series([1.0], [e.index[0]]), y]).pct_change().dropna()
    print(f"{name:52s} 21-23 x{d['mult']:5.2f} DD {d['maxdd']*100:4.0f}% | 24-26 x{t['mult']:4.2f} avg/mo {t['avg_m']*100:+4.1f}% up-mo {t['pos_m']*100:3.0f}% worst-mo {t['worst_m']*100:+4.0f}% DD {t['maxdd']*100:4.0f}% | 2022 {y.iloc[1]*100:+.0f}%")
    return d, t

print("== R1: resting limit orders, fully funded (sum of all orders <= 100% of account) ==")
n_live = 21  # coins trading at any time is ~21-24
for dep, w in [((0.10,), (1,)), ((0.12,), (1,)), ((0.08,), (1,)), ((0.10, 0.16), (1, 1.5)), ((0.10, 0.16, 0.24), (1, 1.5, 2))]:
    fr = [x / sum(w) / n_live for x in w]
    for tp in (0.02, 0.03):
        show(f"levels {dep} tp {tp:.0%} (order size {fr[0]*100:.1f}%..)", run(dep=dep, fr=fr, tp=tp)[0])

# coin subset chosen on DESIGN: most profitable coins per trade for the 10% level
e, lg = run(dep=(0.10,), fr=(0.001,), tp=0.02)
L = pd.DataFrame(lg, columns=["eb", "xb", "k", "j", "ret", "amt", "gap"]); L["sym"] = [syms[int(k)] for k in L.k]
L = L[idx[L.eb.astype(int)] < DESIGN_END]
score = L.groupby("sym").ret.agg(["count", "mean"]); score["tot"] = score["count"] * score["mean"]
for N in (5, 8, 12):
    top = list(score.sort_values("tot", ascending=False).index[:N])
    fr = [1.0 / N]
    show(f"top-{N} coins by design P&L, 10% level, size {100/N:.0f}%", run(dep=(0.10,), fr=fr, tp=0.02, cols=top)[0])
    if N == 8: top8 = top
print("   top-8 picked on 2021-23:", top8)

print("\n== R2: hourly market buys at the open, with bear-market filters (filter picked on 2021-23) ==")
best = None
for g in gates:
    d, t = show(f"gate {g}", run(dep=(0.10, 0.16, 0.24), fr=(0.15, 0.225, 0.30), open_only=True, gate=g)[0])
    if best is None or d["cagr"] / -d["maxdd"] > best[1]: best = (g, d["cagr"] / -d["maxdd"])
print("   best filter on 2021-23 (CAGR/maxDD):", best[0])

print("\n== R1 with the same filters (10% level, all coins, fully funded) ==")
for g in gates:
    show(f"gate {g}", run(dep=(0.10,), fr=(1 / n_live,), tp=0.02, gate=g)[0])
