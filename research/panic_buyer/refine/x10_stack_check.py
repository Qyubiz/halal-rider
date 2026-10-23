"""Code check for x10_stack_lib.skernel (no hypothesis results are printed, only booleans and differences).

1. New switches off (w=1, pens=0, tpfm=0): identical equity curve and trades to engine.run, for the live settings,
   concentrated settings, slower exits, no resting take-profit, partial take-profit, mtp_now, worse buy fills,
   per-hour N/fmax arrays, with and without LUNA/FTT/SRM.
2. Ladder on (pens=0, tpfm=0, release off): identical to r3_ladder.lrun (which is proven equal to round 2's crun).
3. tpfm=2 (every fill): identical to r3_tphigh.trun with the same tpf and floor.
4. Hand-made one-coin paths for pens, tpfm=1 (market buys only), tpf after a leg-2 fill and release.
"""
import sys

import numpy as np
import pandas as pd

import engine as E
import x10_stack_lib as X
import r3_ladder as RL
import r3_tphigh as RT

ok = True


def same(a, b, la, lb, cols=E.LOG_COLS):
    return bool(np.array_equal(a.values, b.values) and la[cols].reset_index(drop=True).equals(lb[cols].reset_index(drop=True)))


for dead in (False, True):
    syms, idx, P = X.market(dead)
    n = len(idx)
    bt = X.btc_dd24(P) <= -0.03
    cases = {
        "live": {},
        "N2 cap50 tp3 cd3": dict(N=2, fmax=0.5, tp=0.03, cd=3),
        "N1 cap100 tp3 cd0 tpwait3": dict(N=1, fmax=1.0, tp=0.03, cd=0, tpwait=3),
        "N2 cap50 no resting tp, tpwait4": dict(N=2, fmax=0.5, tp=0.02, tprest=False, tpwait=4),
        "N3 part .5 tp2 5%": dict(N=3, fmax=0.33, part=0.5, tp2=0.05),
        "N2 trail": dict(N=2, fmax=0.5, part=0.5, trail=0.05),
        "mtp_now pen .002": dict(N=2, fmax=0.5, mtp_now=True, pen=0.002),
        "BTC-falling arrays": dict(N=np.where(bt, 1, 3), fmax=np.where(bt, 1.0, 0.34), tp=0.03, cd=6),
        "slip vector": dict(N=2, fmax=0.5, slip=X.slip_vec(syms)),
    }
    for name, kw in cases.items():
        e0, l0 = E.run(P, idx, **kw)
        e1, l1 = X.srun(P, idx, **kw)
        s = same(e0, e1, l0, l1)
        ok &= s
        print(f"dead={dead!s:5} {name:34s} switches off == engine.run: {s}")
    lcases = {
        "ladder live 50/18": dict(w=0.5, d2=0.18),
        "ladder N2 cap50 67/16 cd2": dict(N=2, fmax=0.5, w=2 / 3, d2=0.16, cd=2, tp=0.03),
        "ladder N1 cap100 50/20 tpwait3": dict(N=1, fmax=1.0, w=0.5, d2=0.20, tpwait=3),
        "ladder N2 no rest mtp pen": dict(N=2, fmax=0.5, w=0.5, d2=0.18, tprest=False, tpwait=2, mtp_now=True,
                                          pen=0.002),
        "ladder part/tp2": dict(N=3, fmax=0.34, w=0.5, d2=0.18, part=0.5, tp2=0.04),
    }
    for name, kw in lcases.items():
        e0, l0 = RL.lrun(P, idx, **kw)
        e1, l1 = X.srun(P, idx, release=False, **kw)
        s = same(e0, e1, l0, l1, RL.LOG)
        ok &= s
        print(f"dead={dead!s:5} {name:34s} ladder == r3_ladder.lrun: {s}")
    for name, kw in {"tpf .10 floor 1.5%": dict(tpf=0.10), "tpf .20 no floor N2": dict(tp=-1.0, tpf=0.20, N=2, fmax=0.5),
                     "tpf .15 floor 3% cd2 tpwait2": dict(tp=0.03, tpf=0.15, cd=2, tpwait=2, N=2, fmax=0.5)}.items():
        e0, l0 = RT.trun(P, idx, **kw)
        e1, l1 = X.srun(P, idx, tpfm=2, **kw)
        s = same(e0, e1, l0, l1)
        ok &= s
        print(f"dead={dead!s:5} {name:34s} tpfm=2 == r3_tphigh.trun: {s}")


# ------------------------------------------------------------------ hand-made paths
def mk(o, h, l, c, ref):
    idx = pd.date_range("2021-01-01", periods=len(o), freq="1h")
    P = {f: pd.DataFrame({"A": np.asarray(v, float)}, index=idx) for f, v in
         (("o", o), ("h", h), ("l", l), ("c", c), ("cf", pd.Series(c).ffill().values), ("ref", ref))}
    return P, idx


fee, sl = E.FEE, E.SLIP_MAJOR
# A. pens: limit fill at 88, take-profit +1.5% at hour 2 -> sold at 89.32*(1-pens)
P, idx = mk([95, 88.5, 89, 89, 89], [95, 89, 90, 90, 90], [86.8, 87, 88, 88, 88], [88] * 5, [100.0] * 5)
e, lg = X.srun(P, idx, N=1, fmax=1.0, pens=0.002)
want = 1.015 * (1 - 0.002) * (1 - fee) ** 2 - 1
a = len(lg) == 1 and abs(lg.ret[0] - want) < 1e-12 and lg.xb[0] == 2
print(f"A pens: ret {lg.ret.tolist()} want {want} -> {a}")
ok &= a
# B. tpfm=1, market buy (open 85 < trigger 88): target max(85.017*1.015, 85.017 + .2*(100-85.017)) = 88.0136
P, idx = mk([95, 85, 86, 87, 88.5, 88.5], [95, 86, 87, 88.0, 89, 89], [94, 84, 85, 86, 88, 88], [95, 85.5, 86.5, 87.5, 88.5, 88.5],
            [100.0] * 6)
e, lg = X.srun(P, idx, N=1, fmax=1.0, tpf=0.2, tpfm=1)
f0 = 85 * (1 + sl)
tgt = f0 + 0.2 * (100 - f0)
b1 = len(lg) == 1 and lg.xb[0] == 4 and abs(lg.ret[0] - (88.5 / f0 * (1 - fee) ** 2 - 1)) < 1e-12 and lg.mkt[0] == 1
print(f"B tpfm=1 market buy: target {tgt:.4f}, sold at hour {lg.xb.tolist()} (open 88.5 >= target) -> {b1}")
ok &= b1
#    same path, limit fill (open 95 at hour 1, low 86.8): tpfm=1 leaves the +1.5% target
P, idx = mk([95, 95, 89, 89.5, 89.5], [95, 95, 90, 90, 90], [94, 86.8, 88, 88, 88], [95, 88, 89, 89, 89], [100.0] * 5)
e, lg = X.srun(P, idx, N=1, fmax=1.0, tpf=0.2, tpfm=1)
b2 = len(lg) == 1 and lg.xb[0] == 2 and abs(lg.ret[0] - (1.015 * (1 - fee) ** 2 - 1)) < 1e-12 and lg.mkt[0] == 0
print(f"B tpfm=1 limit fill keeps +1.5%: exit hour {lg.xb.tolist()} ret {lg.ret.tolist()} -> {b2}")
ok &= b2
# C. tpf after a leg-2 fill (tpfm=2): leg 1 at 88, leg 2 at 82 -> avg 84.87; target avg + .1*(100-avg) = 86.38
P, idx = mk([95, 87, 82.5, 86.5, 86, 86], [95, 87, 84, 86.5, 87, 87], [86.8, 81.0, 82, 85, 85, 85],
            [87, 82.5, 84, 86.5, 86, 86], [100.0] * 6)
e, lg = X.srun(P, idx, N=1, fmax=0.5, w=0.5, d2=0.18, tp=0.0, tpf=0.10, tpfm=2)
avg = 0.5 / (0.25 / 88 + 0.25 / 82)
t2 = avg + 0.1 * (100 - avg)
u = 0.25 * (1 - fee) / 88 + 0.25 * (1 - fee) / 82
c1 = len(lg) == 1 and lg.leg2[0] == 1 and lg.xb[0] == 3 and abs(lg.ret[0] - (u * 86.5 * (1 - fee) / 0.5 - 1)) < 1e-12
print(f"C tpf after leg 2: avg {avg:.3f} target {t2:.3f}; sold at hour {lg.xb.tolist()} at the 86.5 open -> {c1}")
ok &= c1
# D. release: leg 1 fills at 88, data ends at hour 2 with leg 2 resting -> leg-2 cash usable again from hour 3
idx = pd.date_range("2021-01-01", periods=6, freq="1h")
nan = np.nan
P = {"o": pd.DataFrame({"A": [95, 88.5, 88, nan, nan, nan], "B": [100, 100, 100, 95, 100, 100]}, index=idx),
     "h": pd.DataFrame({"A": [95, 88.6, 88, nan, nan, nan], "B": [100, 100, 100, 95, 100, 100]}, index=idx),
     "l": pd.DataFrame({"A": [86.8, 88, 87, nan, nan, nan], "B": [100, 100, 100, 86.0, 100, 100]}, index=idx),
     "c": pd.DataFrame({"A": [88, 88, 87, nan, nan, nan], "B": [100, 100, 100, 95, 100, 100]}, index=idx)}
P["cf"] = P["c"].ffill(); P["ref"] = pd.DataFrame({"A": [100.0] * 6, "B": [100.0] * 6}, index=idx)
e1, l1 = X.srun(P, idx, N=1, fmax=1.0, w=0.5, d2=0.18, release=True)
e0, l0 = X.srun(P, idx, N=1, fmax=1.0, w=0.5, d2=0.18, release=False)
# with release: hour 3 has 0.5 cash -> B bought (limit 88, low 86) with a1 = 0.25 (w x 0.5); without: no cash
d1 = (e1.iloc[2] == e0.iloc[2]) and len(l0) == 0
print(f"D release: equity hour 2 same {e1.iloc[2] == e0.iloc[2]}; B leg-1 money with release "
      f"{float(e1.iloc[3] - e1.iloc[2]):+.4f} vs without {float(e0.iloc[3] - e0.iloc[2]):+.4f}")
sys.stdout.flush()
# with release the B order is placed at hour 3 (0.5 cash): its leg-1 value change = 0.25*(1-fee)/88*95 - 0.25
dB = 0.25 * (1 - fee) / 88 * 95 - 0.25
d2_ = abs(float(e1.iloc[3] - e1.iloc[2]) - dB) < 1e-12 and abs(float(e0.iloc[3] - e0.iloc[2])) < 1e-12
print(f"D release: B bought only with release -> {d1 and d2_}")
ok &= d1 and d2_

print("ALL CHECKS PASS" if ok else "CHECK FAILED")
if not ok:
    sys.exit(1)
