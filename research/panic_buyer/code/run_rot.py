"""Rotation / regime / lead-lag families. Universe includes LUNA, FTT, SRM until their collapse."""
import sys, itertools
sys.path.insert(0, "/home/claude/bt/lab2")
import numpy as np, pandas as pd
import lab
from lab import FEE, SLIP, DESIGN_END, metrics, fmt, pick, robustness

data_dir, dead_dir = sys.argv[1], sys.argv[2]
D = lab.load(data_dir)
for s, cut in {"LUNAUSDT": "2022-05-13 23:00", "FTTUSDT": "2023-09-22 07:00", "SRMUSDT": "2022-11-28 23:00"}.items():
    d = pd.read_csv(f"{dead_dir}/{s}.csv", index_col=0, parse_dates=True); D[s] = d[d.index <= cut]
DC = pd.DataFrame({s: d.c.resample("1D").last() for s, d in D.items()})
for s in ("LUNAUSDT", "FTTUSDT", "SRMUSDT"):            # no data after collapse/delisting
    last = D[s].index[-1].floor("D"); DC.loc[DC.index > last, s] = np.nan
cost = FEE + SLIP
both = lambda e: dict(d=metrics(e, end=DESIGN_END), t=metrics(e, start=DESIGN_END))

def portfolio(DC, weights_fn, reb=7):
    """weights_fn(i) -> dict sym->weight (decided at close of day i). Delisted holdings stay stuck at last price."""
    syms = list(DC.columns); X = DC.values; n, K = X.shape
    lastp = np.full(K, np.nan); units = np.zeros(K); cash = 1.0; eq = np.empty(n)
    for i in range(n):
        px = X[i]; ok = ~np.isnan(px); lastp[ok] = px[ok]
        V = cash + np.nansum(units * np.nan_to_num(lastp))
        eq[i] = V
        if i % reb == 0:
            w = weights_fn(i)
            tgt = np.zeros(K)
            for s, wt in w.items(): tgt[syms.index(s)] = wt
            tradable = ok
            stuck = units * np.nan_to_num(lastp) * (~tradable)
            free = V - stuck.sum()
            new_units = units.copy(); traded = 0.0
            for k in range(K):
                if not tradable[k]: continue
                nu = tgt[k] * free * (1 - cost) / px[k]
                traded += abs(nu - units[k]) * px[k]; new_units[k] = nu
            cash = V - stuck.sum() - np.nansum(new_units * np.where(tradable, px, 0)) - traded * cost
            units = new_units
    return pd.Series(eq, DC.index)

R = DC / DC.shift(1) - 1
btc = DC["BTCUSDT"]; btc_ok = (btc > btc.rolling(50).mean()).values

# ---------------- 4. cross-coin momentum rotation (dual momentum)
rows = []
for L, k, filt in itertools.product([7, 14, 28, 56, 90], [1, 2, 3, 5], ["abs", "abs+btc", "btc"]):
    mom = (DC / DC.shift(L) - 1).values
    def wf(i, mom=mom, k=k, filt=filt):
        if i < L: return {}
        m = mom[i]; valid = ~np.isnan(m)
        if filt in ("abs+btc", "btc") and not btc_ok[i]: return {}
        cand = [(m[j], DC.columns[j]) for j in range(len(m)) if valid[j] and (filt == "btc" or m[j] > 0)]
        cand.sort(reverse=True); top = cand[:k]
        return {s: 1.0 / len(top) for _, s in top} if top else {}
    e = portfolio(DC, wf); rows.append(dict(p=f"rotation: top-{k} by {L}d return, filter {filt}", e=e, **both(e)))
pk = pick(rows)
print(f"[4] {pk['p']}\n   DESIGN 21-23 {fmt(pk['d'])}\n   TEST   24-26 {fmt(pk['t'])}\n   {robustness(rows)}")
rot_e = pk["e"]

# ---------------- 5. breadth regime: whole basket when most coins trend up, else USDT
rows = []
for N, (on, off) in itertools.product([20, 50, 100], [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.5)]):
    above = (DC > DC.rolling(N).mean()); live = DC.notna()
    br = (above & live).sum(axis=1) / live.sum(axis=1)
    state = np.zeros(len(DC), bool); s_ = False
    for i, b in enumerate(br.values):
        if b >= on: s_ = True
        elif b <= off: s_ = False
        state[i] = s_
    def wf(i, state=state):
        if not state[i]: return {}
        live_s = [s for s in DC.columns if not np.isnan(DC.iloc[i][s])]
        return {s: 1.0 / len(live_s) for s in live_s}
    e = portfolio(DC, wf, reb=1 if False else 7)
    rows.append(dict(p=f"breadth: basket when >{on:.0%} of coins above SMA{N}d, exit <{off:.0%}", e=e, **both(e)))
pk = pick(rows)
print(f"[5] {pk['p']}\n   DESIGN 21-23 {fmt(pk['d'])}\n   TEST   24-26 {fmt(pk['t'])}\n   {robustness(rows)}")

# ---------------- 6. BTC lead-lag (hourly): BTC jumps, laggard alts catch up?
P = lab.panels({s: d for s, d in D.items()})
Cc, Oo = P["c"], P["o"]
r1 = Cc.pct_change(fill_method=None)
out = []
for x, gap, Hh in itertools.product([0.01, 0.015, 0.02], [0.005, 0.01], [1, 2, 4, 8]):
    ev = (r1["BTCUSDT"] > x).values
    rets = []
    for s in Cc.columns:
        if s == "BTCUSDT": continue
        lag = (r1[s] < r1["BTCUSDT"] - gap).values & ev
        ii = np.where(lag)[0]; ii = ii[ii + Hh + 1 < len(Cc)]
        ent = Oo[s].values[ii + 1]; ex = Cc[s].values[ii + Hh]
        rr = ex / ent * (1 - cost) ** 2 - 1
        t = Cc.index[ii]
        rets.append(pd.DataFrame({"t": t, "r": rr}).dropna())
    rr = pd.concat(rets)
    dsg, tst = rr[rr.t < DESIGN_END].r, rr[rr.t >= DESIGN_END].r
    out.append((x, gap, Hh, len(dsg), dsg.mean(), len(tst), tst.mean()))
o = pd.DataFrame(out, columns=["btc_jump", "gap", "hold_h", "n_d", "avg_d", "n_t", "avg_t"])
b = o.sort_values("avg_d", ascending=False).iloc[0]
print(f"[6] lead-lag best on design: BTC 1h > {b.btc_jump:.1%}, alt lagging by {b.gap:.1%}, hold {int(b.hold_h)}h -> "
      f"design {int(b.n_d)} trades avg {b.avg_d*100:+.2f}% | test {int(b.n_t)} trades avg {b.avg_t*100:+.2f}%  (after 0.24% costs) | "
      f"settings with positive test avg: {np.mean(o.avg_t>0)*100:.0f}%")
rot_e.to_pickle("/home/claude/bt/out/rotation.pkl")
