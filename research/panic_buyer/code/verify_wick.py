"""Independent checks of the panic-wick ladder before trusting it."""
import sys, itertools
sys.path.insert(0, "/home/claude/bt/lab2")
import numpy as np, pandas as pd
import lab
from lab import FEE, SLIP, DESIGN_END, metrics, fmt

data_dir, dead_dir = sys.argv[1], sys.argv[2]
D = lab.load(data_dir)
P = lab.panels(D); idx = P["c"].index; syms = list(P["c"].columns)

def arrays(Pp):
    C = Pp["c"].ffill()
    return dict(o=Pp["o"].values, h=Pp["h"].values, l=Pp["l"].values, c=C.values,
                ref=C.rolling(24).max().shift(1).values, alive=np.ones(C.shape, np.bool_), idx=Pp["c"].index)
A0 = arrays(P)

def run(A, dep=(0.10, 0.16, 0.24), f=0.15, tp=0.02, hm=96, sl=0.0, pen=0.0, open_only=False, fracs=None):
    depths = np.array(dep)
    fr = np.array(fracs) if fracs is not None else np.array([f * (1 + 0.5 * i) for i in range(len(dep))])
    eq, lg = lab.wick2(A["o"], A["h"], A["l"], A["c"], A["ref"], depths, fr, tp, hm, sl, 24, 0.001,
                       FEE if not open_only else FEE, SLIP, pen, open_only, A["alive"])
    return pd.Series(eq, A["idx"]), lg

def line(name, e):
    d, t = metrics(e, end=DESIGN_END), metrics(e, start=DESIGN_END)
    print(f"{name:44s} 21-23 x{d['mult']:7.2f} | 24-26 x{t['mult']:5.2f} avg/mo {t['avg_m']*100:+5.1f}% up-mo {t['pos_m']*100:3.0f}% worst-mo {t['worst_m']*100:+4.0f}% maxDD {t['maxdd']*100:4.0f}%")
    return t

# ---- 1. independent event study (plain python, single 10% level) vs kernel
def event_study(d, depth=0.10, tp=0.02, hm=96, cd=24):
    o, h, l, c = d.o.values, d.h.values, d.l.values, d.c.values
    ref = pd.Series(c).rolling(24).max().shift(1).values
    out = []; t = 24; n = len(c); next_ok = 0
    while t < n - 1:
        if t < next_ok or np.isnan(ref[t]): t += 1; continue
        lim = ref[t] * (1 - depth)
        if l[t] <= lim * 0.999:
            e = min(lim, o[t]); ex = None
            for u in range(t + 1, min(t + hm + 1, n)):
                if o[u] >= e * (1 + tp): ex = o[u]; break
                if h[u] >= e * (1 + tp) * 1.001: ex = e * (1 + tp); break
                if u - t >= hm: ex = c[u] * (1 - SLIP); break
            if ex is None: break
            out.append((d.index[t], ex * (1 - FEE) * (1 - FEE) / e - 1, o[t] <= lim)); next_ok = u + cd; t = u + 1
        else: t += 1
    return out
ev = []
for s, d in D.items(): ev += [(s,) + x for x in event_study(d)]
ev = pd.DataFrame(ev, columns=["sym", "t", "ret", "gap"])
_, lg1 = run(A0, dep=(0.10,), fracs=[0.0005])
print(f"[check] plain-python event study: {len(ev)} trades, win {np.mean(ev.ret>0)*100:.1f}%, avg {ev.ret.mean()*100:+.3f}%")
print(f"[check] kernel same rules         : {len(lg1)} trades, win {np.mean(lg1[:,4]>0)*100:.1f}%, avg {lg1[:,4].mean()*100:+.3f}%")
ev["yr"] = ev.t.dt.year
print("[check] event study by year (single 10% level, +2% target, 96h max):")
print(ev.groupby("yr").ret.agg(trades="count", win=lambda r: f"{np.mean(r>0)*100:.0f}%", avg=lambda r: f"{r.mean()*100:+.2f}%", worst=lambda r: f"{r.min()*100:+.0f}%").to_string())
print(f"[check] filled at candle open (gap, hourly bot can do it): {ev.gap.mean()*100:.0f}% of trades, avg {ev[ev.gap].ret.mean()*100:+.2f}% | intrabar wick fills avg {ev[~ev.gap].ret.mean()*100:+.2f}%")

# ---- 2. realism variants of the picked setting
print("\n[variants] picked: depths 10/16/24% below 24h high, size 15/22.5/30%, +2% target, 96h max hold")
base, lgb = run(A0); line("base (as tested)", base)
line("fully-funded resting orders (no overbooking)", run(A0, fracs=[x / len(syms) / 4.5 for x in (1, 1.5, 2)])[0])
for pen in (0.002, 0.005, 0.01): line(f"entry fill {pen:.1%} worse", run(A0, pen=pen)[0])
line("HOURLY BOT: buy at candle open only", run(A0, open_only=True)[0])
line("HOURLY BOT + 0.5% worse fills", run(A0, open_only=True, pen=0.005)[0])
for sl in (0.25, 0.35): line(f"with {sl:.0%} stop-loss", run(A0, sl=sl)[0])
# ---- 3. survivorship stress: add LUNA, FTT, SRM up to delisting/collapse
dead_cut = {"LUNAUSDT": "2022-05-13 23:00", "FTTUSDT": "2023-09-22 07:00", "SRMUSDT": "2022-11-28 23:00"}
D2 = dict(D)
for s, cut in dead_cut.items():
    d = pd.read_csv(f"{dead_dir}/{s}.csv", index_col=0, parse_dates=True); D2[s] = d[d.index <= cut]
P2 = lab.panels(D2); A2 = arrays(P2); A2["ref"] = P2["c"].rolling(24, min_periods=1).max().shift(1).values
line("+ LUNA/FTT/SRM collapses added", run(A2)[0])
line("+ collapses, hourly bot", run(A2, open_only=True)[0])
line("+ collapses, hourly bot, 35% stop", run(A2, open_only=True, sl=0.35)[0])

# ---- 4. neighbourhood on the test period (is the pick a lucky spike?)
res = []
for dep, f, tp, hm in itertools.product([(0.08, 0.14, 0.20), (0.10, 0.16, 0.24), (0.12, 0.18, 0.26), (0.10,), (0.15,)],
                                        [0.10, 0.15, 0.20], [0.015, 0.02, 0.03], [48, 96, 168]):
    e, _ = run(A0, dep=dep, f=f, tp=tp, hm=hm)
    t = metrics(e, start=DESIGN_END); res.append((dep, f, tp, hm, t["cagr"], t["maxdd"], t["pos_m"]))
r = pd.DataFrame(res, columns=["dep", "f", "tp", "hm", "cagr", "dd", "pos"])
print(f"\n[neighbourhood] {len(r)} nearby settings on 2024-26: CAGR min {r.cagr.min()*100:+.0f}% median {r.cagr.median()*100:+.0f}% max {r.cagr.max()*100:+.0f}%, positive {np.mean(r.cagr>0)*100:.0f}%, median maxDD {r.dd.median()*100:.0f}%")
print("median CAGR by target:", r.groupby("tp").cagr.median().map(lambda x: f"{x*100:+.0f}%").to_dict())
print("median CAGR by max hold:", r.groupby("hm").cagr.median().map(lambda x: f"{x*100:+.0f}%").to_dict())
print("median CAGR by depths:", r.groupby(r.dep.astype(str)).cagr.median().map(lambda x: f"{x*100:+.0f}%").to_dict())

# ---- 5. picked setting: per-year and per-coin
yr = base.resample("YE").last(); yr = pd.concat([pd.Series([1.0], [base.index[0]]), yr]).pct_change().dropna()
print("\n[per year]", {str(k.year): f"{v*100:+.0f}%" for k, v in yr.items()})
lg = pd.DataFrame(lgb, columns=["eb", "xb", "k", "j", "ret", "amt", "gap"]); lg["sym"] = [syms[int(k)] for k in lg.k]
lg["pnl"] = lg.ret * lg.amt; lg["test"] = idx[lg.eb.astype(int)] >= DESIGN_END
pc = lg[lg.test].groupby("sym").agg(trades=("ret", "size"), win=("ret", lambda x: np.mean(x > 0)), avg=("ret", "mean"))
print("[per coin, test] coins with negative avg:", [s for s, r_ in pc.iterrows() if r_.avg < 0], "| trades per coin min/max:", pc.trades.min(), pc.trades.max())
lt = lg[lg.test]
print(f"[trade shape, test] wins avg {lt[lt.ret>0].ret.mean()*100:+.2f}% | losses avg {lt[lt.ret<=0].ret.mean()*100:+.2f}% | worst {lt.ret.min()*100:+.0f}% | time-stopped losers {np.mean(lt.ret<=0)*100:.0f}%")
base.to_pickle("/home/claude/bt/out/wick_base.pkl")
