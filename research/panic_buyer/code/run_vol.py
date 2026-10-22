"""Volatility-harvesting families. Settings picked on 2021-2023 only, then tested on 2024-2026."""
import sys, itertools, pickle
sys.path.insert(0, "/home/claude/bt/lab2")
import numpy as np, pandas as pd
import lab
from lab import FEE, SLIP, DESIGN_END, metrics, fmt, pick, robustness

data_dir, out_dir = sys.argv[1], sys.argv[2]
D = lab.load(data_dir)
P = lab.panels(D)
idx = P["c"].index
C = P["c"].ffill(); O = P["o"].ffill()
DC = lab.daily_close(D)
curves, picks = {}, {}

def both(eq):
    return dict(d=metrics(eq, end=DESIGN_END), t=metrics(eq, start=DESIGN_END))

def show(name, r, extra=""):
    print(f"{name}\n   DESIGN 21-23 {fmt(r['d'])}\n   TEST   24-26 {fmt(r['t'])} {extra}")

# ---------------- benchmarks
for name, eq in (("BTC buy&hold", C["BTCUSDT"]), ("Basket buy&hold (21 coins)", (C / C.iloc[0]).mean(axis=1))):
    show(f"[benchmark] {name}", both(eq)); curves[name] = eq / eq.iloc[0]

dr = DC.pct_change()
bidx = (1 + dr.mean(axis=1, skipna=True).fillna(0)).cumprod()
def basket_regime(N):
    g = (bidx > bidx.rolling(N).mean()).shift(1).astype("boolean").fillna(False).astype(bool)
    return g.reindex(idx.floor("D")).fillna(False).values.astype(bool)

# ---------------- 1. rebalancing premium (constant mix and trend-adaptive mix)
cost = FEE + SLIP
rows = []
o_, c_ = O.values, C.values
for wc, band in itertools.product([0.3, 0.5, 0.7, 1.0], [0.1, 0.25, 0.5]):
    eq, to = lab.rebalance_kernel(o_, c_, np.full(len(idx), wc), band, cost)
    e = pd.Series(eq, idx); rows.append(dict(p=f"const crypto {wc:.0%} band {band}", e=e.astype("float32"), **both(e)))
for (hi, lo), N, band in itertools.product([(1, 0), (1, 0.3), (0.8, 0.2), (0.7, 0)], [30, 50, 100, 200], [0.1, 0.25, 0.5]):
    g = basket_regime(N); wc = np.where(g, hi, lo).astype(float)
    eq, to = lab.rebalance_kernel(o_, c_, wc, band, cost)
    e = pd.Series(eq, idx); rows.append(dict(p=f"trend-mix {hi:.0%}/{lo:.0%} basket>SMA{N}d band {band}", e=e.astype("float32"), **both(e)))
pk = pick(rows); show(f"[1] Rebalance: {pk['p']}", pk, ""); print("   " + robustness(rows))
curves["rebalance"] = pk["e"]; picks["rebalance"] = pk["p"]
const_rows = [r for r in rows if r["p"].startswith("const")]
pc = pick(const_rows); show(f"[1b] Rebalance constant-mix only: {pc['p']}", pc); print("   " + robustness(const_rows))
curves["rebalance_const"] = pc["e"]

# ---------------- 2. trailing grid per coin
coin_gate = {}
for s in D:
    dc = D[s].c.resample("1D").last()
    for N in (50, 200):
        g = (dc > dc.rolling(N).mean()).shift(1).astype("boolean").fillna(False).astype(bool)
        coin_gate[(s, N)] = g.reindex(D[s].index.floor("D")).fillna(False).values.astype(bool)
bask100 = {s: (pd.Series(basket_regime(100), idx).reindex(D[s].index).fillna(False).values.astype(bool)) for s in D}
rows = []
gates = ["none", "coin>SMA50d", "coin>SMA200d", "basket>SMA100d"]
for sp, L, stop, gname, liq in itertools.product([0.01, 0.015, 0.02, 0.03, 0.05], [5, 10, 20], [0.0, 0.05, 0.15], gates, [False, True]):
    if gname == "none" and liq: continue
    sleeves = []
    for s, d in D.items():
        if gname == "none": g = np.ones(len(d), np.bool_)
        elif gname == "coin>SMA50d": g = coin_gate[(s, 50)]
        elif gname == "coin>SMA200d": g = coin_gate[(s, 200)]
        else: g = bask100[s]
        eq, rounds = lab.grid_kernel(d.o.values, d.h.values, d.l.values, d.c.values, g, sp, L, stop, 168, liq, 0.0005, FEE, SLIP)
        sleeves.append(pd.Series(eq, d.index).reindex(idx).ffill().fillna(1.0))
    e = sum(sleeves) / len(sleeves)
    rows.append(dict(p=f"grid step {sp:.1%} x{L} slots stop {stop:.0%} gate {gname}{' liquidate-when-off' if liq else ''}", e=e.astype("float32"), **both(e)))
pk = pick(rows); show(f"[2] Grid: {pk['p']}", pk); print("   " + robustness(rows))
curves["grid"] = pk["e"]; picks["grid"] = pk["p"]

# ---------------- 3. panic-wick limit ladder (shared cash across coins)
Hh, Ll, Oo = P["h"].values, P["l"].values, P["o"].values
refs = {"1h": C.shift(1).values, "24h": C.rolling(24).max().shift(1).values}
sets = {"1h": [(0.03,), (0.05,), (0.08,), (0.03, 0.06, 0.10), (0.05, 0.08, 0.12)],
        "24h": [(0.08,), (0.12,), (0.18,), (0.08, 0.14, 0.20), (0.10, 0.16, 0.24)]}
rows = []
for rname in ("1h", "24h"):
    for dep, f, tp, hm, sl in itertools.product(sets[rname], [0.05, 0.15], [0.02, 0.04, 0.08], [24, 96, 336], [0.0, 0.10, 0.25]):
        depths = np.array(dep); fr = np.array([f * (1 + 0.5 * i) for i in range(len(dep))])
        eq, lg = lab.wick_kernel(Oo, Hh, Ll, c_, refs[rname], depths, fr, tp, hm, sl, 24, 0.001, FEE, SLIP)
        e = pd.Series(eq, idx)
        rows.append(dict(p=f"wick ref {rname} depths {dep} size {f:.0%} tp {tp:.0%} hold<= {hm}h stop {sl:.0%}", e=e.astype("float32"), lg=lg,
                         arg=(rname, dep, f, tp, hm, sl), **both(e)))
pk = pick(rows)
lg = pk["lg"]; pnl = lg[:, 4] * lg[:, 5]; test_mask = idx[lg[:, 0].astype(int)] >= DESIGN_END
tp_ = pnl[test_mask]; top5 = np.sort(tp_)[-5:].sum() if len(tp_) >= 5 else tp_.sum()
show(f"[3] Wick ladder: {pk['p']}", pk,
     f"\n   test trades {test_mask.sum()}, win {np.mean(lg[test_mask,4]>0)*100:.0f}%, avg {np.mean(lg[test_mask,4])*100:+.2f}%/trade, best-5 trades = {top5/max(tp_.sum(),1e-9)*100:.0f}% of test P&L")
print("   " + robustness(rows))
curves["wick"] = pk["e"]; picks["wick"] = pk["p"]
pickle.dump(dict(curves=curves, picks=picks, wick_arg=pk["arg"]), open(f"{out_dir}/vol.pkl", "wb"))
