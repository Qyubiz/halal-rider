"""m11 stack: prove the combined 1-minute engine reproduces each signal alone.
  A only  == m10_velocity_lib.vrun(MODE90) on hourly candles built from the same minutes (trade log + equity), and the
            published 90% mode figures (hourly CSV candles) within rounding; also worse fills and the stressed run.
  E only  == m10_meanrev_lib.run(pick) on its own 5-minute bars (2021-26).
  F only  == m10_scalp_lib.srun(scalp(W60, 8%, +2%, 96h, 4 x 25%, guard 35%, wick 0.5%)) with its slippage panel.
Also prints, for the 2021-23 design period only, how far from its trigger the 90% mode's coin was at the open of each
fill hour (how much an A bid gated by nearA would miss)."""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m11_stack_lib as S  # noqa: E402
import m10_velocity_lib as V  # noqa: E402
import engine as E  # noqa: E402


def cmp(tag, lgA, lgB, eA, eB):
    same = len(lgA) == len(lgB) and np.allclose(lgA, lgB, rtol=1e-9, atol=1e-12)
    d = float(np.max(np.abs(np.asarray(eA) / np.asarray(eB) - 1)))
    print(f"  {tag:40s} trades {len(lgA)} vs {len(lgB)} log identical {same} max rel eq diff {d:.1e} "
          f"x{eA[-1]:.4f} vs x{eB[-1]:.4f}", flush=True)
    return same and d < 1e-9


ok = True
# ---------------------------------------------------------------- A only vs vk
for K, dead in ((19, False), (22, True)):
    D = S.data("main", K)
    Ph = {f: pd.DataFrame(D["Hd"][f], D["hidx"], D["syms"]) for f in ("o", "h", "l", "c")}
    Ph["cf"] = Ph["c"].ffill(); Ph["ref"] = Ph["c"].rolling(24, min_periods=1).max().shift(1)
    for name, d, kw in (("normal", S.MODE90, {}), ("worse", S.worse(S.MODE90), dict(pen=0.002, pens=0.002)),
                        ("stressed", S.stressed(S.MODE90), V.stress_kw(V.MODE90))):
        if dead != (name == "stressed"):
            continue
        t0 = time.time()
        e, lg, inv, lo = S.run(D, d)
        k = dict(V.MODE90, **kw) if name != "stressed" else kw
        k.setdefault("liquidate", dead)
        ev, lgv, _ = V.vrun(Ph, D["hidx"], **k)
        a = lg[["eb", "xb", "k", "why", "ret", "amt", "entry"]].values
        b = lgv[["eb", "xb", "k", "why", "ret", "amt", "entry"]].values
        ok &= cmp(f"A only {name} ({K} coins) vs vk", a, b, e.values, ev.values)
        print(f"     ({time.time()-t0:.0f}s)")
        for p, (x, y) in {"2021-23": (S.T21, S.T24), "2024-26": (S.T24, None), "since 2025": (S.T25, None)}.items():
            print("    " + S.fmt(f"{name} {p}", S.stats(e, lg, inv, lo, x, y)))
DH = S.data("hold")
e, lg, inv, lo = S.run(DH, S.MODE90)
print("    " + S.fmt("normal 2018-20 (1m data)", S.stats(e, lg, inv, lo, S.T18, S.T21)))
e, lg, inv, lo = S.run(DH, S.worse(S.MODE90))
print("    " + S.fmt("worse 2018-20 (1m data)", S.stats(e, lg, inv, lo, S.T18, S.T21)))
print("  published (hourly CSV candles): 2021-23 93.8/+13.17/-64.4, 2024-26 92.6/+6.89/-35.4, since 2025 93.2/+5.80/-22.6,"
      " 2018-20 92.5/+10.82/-66.3; worse 2024-26 +5.31; stressed 2024-26 91.7/+5.49/-30.3")

# ---------------------------------------------------------------- E only vs meanrev
import m10_meanrev_lib as L  # noqa: E402
L.OUT = os.path.join(S.SCR, "pb_out_m10", "meanrev")
B = L.load("main", C=5)
hb = lambda x: L.bars_per(B, x)   # noqa: E731
cfs = L.sh(B["cf"])
okE = (cfs >= L.rollmax(B, hb(24 * 7)) * 0.65) & (cfs >= L.rollmax(B, hb(24)) * 0.85)
lim = np.where(okE, L.rollmax(B, hb(1)) * 0.92, np.nan)
eqm, lgm = L.run(B, lim, nord=1, fmax=1.0, maxpos=1, tp=0.012, hm=hb(6), cd=hb(1))
D = S.data("main", 19)
assert B["t"][0] == D["hidx"][0], (B["t"][0], D["hidx"][0])
dE = dict(A=0, E=1, F=0, fsE=1.0, nslots=1)
e, lg, inv, lo = S.run(D, dE)
nh = min(len(e), len(eqm) // 12)
a = np.c_[lg.fm.values // 5, lg.xm.values // 5, lg[["k", "why", "ret", "amt", "entry"]].values]
a = a[(a[:, 3] != 9) & (a[:, 1] < 12 * nh - 1)]
a = a[a[:, 0].argsort(kind="stable")]
b = lgm[:, :7]
b = b[b[:, 1] < 12 * nh - 1]
b = b[b[:, 0].argsort(kind="stable")]
ok &= cmp("E only vs meanrev (5m bars)", a, b, e.values[:nh], eqm[11:12 * nh:12])
print("    " + S.fmt("E only 2021-23", S.stats(e, lg, inv, lo, S.T21, S.T24)))

# ---------------------------------------------------------------- F only vs scalp
import m10_scalp_lib as SC  # noqa: E402
A_, hidx, Ph, SLIP = SC.load(19)
gref = SC.guard_ref(Ph)
es, lgs, invs, eqlos, ntr = SC.srun(A_, hidx, Ph, SLIP, gref=gref,
                                    **SC.scalp(60, 0.08, 0.02, 5760, N=4, fmax=0.25, guard=0.35, wick=0.005))
dF = dict(A=0, E=0, F=1, fsF=0.25, nordF=4, nslots=99)
e, lg, inv, lo = S.run(D, dF, slip=SLIP)
a = lg[lg.why != 9][["eb", "xb", "k", "why", "ret", "amt", "entry"]].values
ok &= cmp("F only vs scalp (slip panel)", a, lgs[["eb", "xb", "k", "why", "ret", "amt", "entry"]].values,
          e.values, es.values)
print("    " + S.fmt("F only 2021-23 (slip panel)", S.stats(e, lg, inv, lo, S.T21, S.T24)))

# ---------------------------------------------------------------- design-period anatomy (2021-23 only)
e, lg, inv, lo = S.run(D, S.MODE90, end=S.T24)
dist = D["ho"][lg.eb.astype(int), lg.k.astype(int)] / (D["refA"][lg.eb.astype(int), lg.k.astype(int)] * 0.86) - 1
dist = np.maximum(dist, 0)
pnl = (lg.ret * lg.amt / lg.eq0).values
for x in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10):
    m = dist <= x
    print(f"  2021-23 90% mode fills whose hour opened <= {x*100:3.0f}% above the trigger: {m.mean()*100:5.1f}% of trades, "
          f"{np.log1p(pnl[m]).sum()/np.log1p(pnl).sum()*100:5.1f}% of the log-profit")
print("M11 STACK CHECK", "PASSED" if ok else "FAILED")
