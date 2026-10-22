"""Futures funding rate as a crowd-positioning signal (spot trades only)."""
import sys, itertools
sys.path.insert(0, "/home/claude/bt/lab2")
import numpy as np, pandas as pd
import lab
from lab import FEE, SLIP, DESIGN_END, metrics, fmt

data_dir, dead_dir, fund_dir = sys.argv[1], sys.argv[2], sys.argv[3]
D = lab.load(data_dir)
P = lab.panels(D); idx = P["c"].index; syms = list(P["c"].columns)
C = P["c"].ffill(); ref = P["c"].rolling(24, min_periods=1).max().shift(1)
O, H, Lw, Cv = P["o"].values, P["h"].values, P["l"].values, C.values
F = pd.DataFrame({s: pd.read_csv(f"{fund_dir}/{s}.csv", index_col=0, parse_dates=True).rate for s in syms})
Fh = F.reindex(F.index.union(idx)).ffill().reindex(idx).shift(1)          # last settled rate known at each hour
F24 = F.resample("1D").sum()                                             # daily funding paid by longs
end_f = F.index[-1]

# (a) contrarian basket: hold equal-weight coins for H days after market-wide funding turns negative
rows = []
dc = C.resample("1D").last(); dr = dc.pct_change()
avgF = F24.mean(axis=1).reindex(dc.index).shift(1)  # yesterday's average daily funding across coins
for th, Hd in itertools.product([-0.0003, -0.0001, 0.0, 0.0001], [3, 7, 14, 30]):
    sig = (avgF < th).values; hold = np.zeros(len(dc), bool); left = 0
    for i in range(len(dc)):
        if sig[i]: left = Hd
        hold[i] = left > 0; left -= 1
    pos = pd.Series(hold, dc.index)
    ret = (dr.mean(axis=1) * pos) - pos.astype(int).diff().abs().fillna(0) * (FEE + SLIP)
    e = (1 + ret.fillna(0)).cumprod(); e = e[e.index <= end_f]
    rows.append((th, Hd, metrics(e, end=DESIGN_END), metrics(e, start=DESIGN_END), pos.mean()))
best = max(rows, key=lambda r: r[2]["cagr"] / max(1e-9, -r[2]["maxdd"]))
print(f"[7] funding contrarian: basket for {best[1]}d after avg funding < {best[0]*100:.2f}%/day (in market {best[4]*100:.0f}% of days)")
print(f"   DESIGN 21-23 {fmt(best[2])}\n   TEST   24-26 {fmt(best[3])}")
print(f"   all {len(rows)} settings: test CAGR median {np.median([r[3]['cagr'] for r in rows])*100:+.0f}%, positive {np.mean([r[3]['cagr']>0 for r in rows])*100:.0f}%")

# (b) funding as a filter for the dynamic dip-buyer (10% below 24h high, N=8, 20%/order, +2%, 96h)
def dyn(refm):
    eq, lg = lab.dyn_kernel(O, H, Lw, Cv, refm, 0.10, 8, 0.20, 0.02, 96, 24, 0.001, FEE, SLIP, 0.0)
    e = pd.Series(eq, idx); return e[e.index <= end_f]
print("\n[7b] dip-buyer filtered by the coin's latest funding rate (rate known before the dip):")
for name, mask in [("no filter", np.ones(Fh.shape, bool)),
                   ("only if funding > 0 (longs crowded)", (Fh > 0).values),
                   ("only if funding > 0.01%", (Fh > 0.0001).values),
                   ("only if funding <= 0.01%", (Fh <= 0.0001).values),
                   ("only if funding < 0 (shorts crowded)", (Fh < 0).values)]:
    r = ref.values.copy(); r[~mask] = np.nan
    e = dyn(r); d, t = metrics(e, end=DESIGN_END), metrics(e, start=DESIGN_END)
    print(f"   {name:38s} 21-23 x{d['mult']:5.2f} DD {d['maxdd']*100:4.0f}% | 24-26 x{t['mult']:4.2f} avg/mo {t['avg_m']*100:+.1f}% up-mo {t['pos_m']*100:.0f}% DD {t['maxdd']*100:.0f}%")
