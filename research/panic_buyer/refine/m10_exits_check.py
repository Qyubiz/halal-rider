"""m10 exits: with every exit switch off, m10_exits_lib.xrun == w90_wr90_lib.wrun exactly (equity and trade log), for
90% mode, the live settings, bot timing, fills 0.2% worse and LUNA/FTT/SRM with liquidation. Also checks that the
schedule with a1 beyond any hold and an unopened underwater branch change nothing.
Usage: PB_DATA=... PB_OUT=... python3 m10_exits_check.py"""
import numpy as np

import m10_exits_lib as X
import w90_wr90_lib as W

CASES = [dict(depth=0.14, N=1, fmax=1.0, tp=0.025, hm=96, cd=3, guard=0.35),
         dict(depth=0.12, N=8, fmax=0.20, tp=0.015, hm=48, cd=24, guard=0.0),
         dict(depth=0.14, N=1, fmax=1.0, tp=0.025, hm=96, cd=4, guard=0.35, mtp_now=True, pen=0.002, pens=0.002),
         dict(depth=0.13, N=2, fmax=0.5, tp=0.02, hm=72, cd=6, guard=0.25)]
ok_all = True
for dead in (False, True):
    syms, idx, P = W.market(dead)
    for kw in CASES:
        k = dict(kw, liquidate=dead)
        e1, l1 = W.wrun(P, idx, **k)
        e2, l2 = X.xrun(P, idx, **k)
        same = np.array_equal(e1.values, e2.values) and np.array_equal(
            l1[["eb", "xb", "k", "why", "ret", "amt", "entry", "eq0"]].values,
            l2[["eb", "xb", "k", "why", "ret", "amt", "entry", "eq0"]].values)
        ok_all &= same
        print(f"{'22' if dead else '19'} coins {kw}: wrun x{e1.iloc[-1]:.6f} xrun x{e2.iloc[-1]:.6f} "
              f"trades {len(l1)}/{len(l2)} {'IDENTICAL' if same else 'DIFFERENT'}", flush=True)
syms, idx, P = W.market(False)
e1, _ = X.xrun(P, idx, **CASES[0])
e2, _ = X.xrun(P, idx, **CASES[0], a1=5000, be=0.0)
e3, _ = X.xrun(P, idx, **CASES[0], uw=0.5, uwmode=0, hm2=96)      # hm2 <= hm: the kept branch never opens
same = np.array_equal(e1.values, e2.values) and np.array_equal(e1.values, e3.values)
ok_all &= same
print(f"inactive schedule / underwater switches change nothing: {same}")
e4, _ = X.xrun(P, idx, **CASES[0], part=0.999999, hmr=1)
print(f"part=0.999999, dust sold at the next hourly close: x{e4.iloc[-1]:.4f} vs x{e1.iloc[-1]:.4f} (close: the coin is "
      f"blocked one more hour)")
e5, _ = X.xrun(P, idx, **CASES[0], part=0.999999, hmr=96)
print(f"INFO part=0.999999, dust held to the 96h time stop: x{e5.iloc[-1]:.4f}: a held rest blocks re-buying the same "
      f"coin, which costs a lot (staged exits pay this)")
print("ALL CHECKS PASS" if ok_all else "SOME CHECK FAILED")
