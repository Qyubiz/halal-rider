"""Proof that ens_lib.erun == engine.run (gkernel) and == w90_wr90_lib.wrun with every new switch off, and that two
sleeves without rebalancing == the weighted sum of the two single-sleeve runs.

Usage (from research/panic_buyer/refine): PB_DATA=... PB_OUT=... python3 ens_check.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("PB_OUT", os.path.expanduser("~/pb_out_ens/research"))
import engine as E  # noqa: E402
import ens_lib as L  # noqa: E402
import w90_wr90_lib as W  # noqa: E402

COLS = ["eb", "xb", "k", "why", "ret", "amt", "entry"]
CASES = {
    "LIVE": (dict(depth=0.12, N=8, fmax=0.20, tp=0.015, part=1.0, tp2=0.0, hm=48, cd=24), 0.0, {}),
    "90% mode": (L.MODE90, 0.35, {}),
    "10% mode": (L.MODE10, 0.35, {}),
    "90% mode bot timing + pen": (dict(L.MODE90, cd=4), 0.35, dict(mtp_now=True, pen=0.002)),
    "confluence 14%/+2.5% half/+5% 96h": (dict(L.MODE90, part=0.5, tp2=0.05), 0.35, {}),
    "2 coins x50% partial 0.7": (dict(depth=0.14, N=2, fmax=0.5, tp=0.025, part=0.7, tp2=0.06, hm=72, cd=12), 0.35, {}),
}
ok_all = True
for dead in (False, True):
    syms, idx, P = L.market(dead)
    for name, (sv, g, over) in CASES.items():
        sm = W.guard_mask(P, g) if g > 0 else 1.0
        kw = {k: v for k, v in sv.items()}
        kw["N"] = sv["N"]
        e1, l1 = E.run(P, idx, smult=sm, depth=sv["depth"], N=sv["N"], fmax=sv["fmax"], tp=sv["tp"], part=sv["part"],
                       tp2=sv["tp2"], hm=sv["hm"], cd=sv["cd"], **over)
        e2, l2, _, _ = L.erun(P, idx, [sv], guard=g, **over)
        l2c = l2[l2.why != 9]
        ok = np.array_equal(e1.values, e2.values) and np.array_equal(l1[COLS].values, l2c[COLS].values)
        ok_all &= ok
        print(f"{'22' if dead else '19'} coins {name:38s} engine.run x{e1.iloc[-1]:10.4f} erun x{e2.iloc[-1]:10.4f} "
              f"trades {len(l1)}/{len(l2c)} {'IDENTICAL' if ok else 'DIFFERENT'}", flush=True)
        if dead and sv["part"] == 1.0:
            e3, l3 = W.wrun(P, idx, depth=sv["depth"], N=sv["N"], fmax=sv["fmax"], tp=sv["tp"], hm=sv["hm"], cd=sv["cd"],
                            guard=g, liquidate=True, pens=0.002, **over)
            e4, l4, _, _ = L.erun(P, idx, [sv], guard=g, liquidate=True, pens=0.002, **over)
            ok = np.array_equal(e3.values, e4.values) and np.array_equal(l3[COLS].values, l4[COLS].values)
            ok_all &= ok
            print(f"   + liquidate, pens 0.2%: wrun x{e3.iloc[-1]:10.4f} erun x{e4.iloc[-1]:10.4f} "
                  f"{'IDENTICAL' if ok else 'DIFFERENT'}", flush=True)
    # sleeves: no rebalance == weighted sum of the single runs
    ea, _, _, _ = L.erun(P, idx, [L.MODE90])
    eb, _, _, _ = L.erun(P, idx, [L.MODE10])
    for w in (0.5, 0.7, 0.3):
        es, ls, _, _ = L.erun(P, idx, [L.MODE90, L.MODE10], weights=[w, 1 - w])
        ref = w * ea.values + (1 - w) * eb.values
        err = float(np.max(np.abs(es.values / ref - 1)))
        ok = err < 1e-9
        ok_all &= ok
        print(f"{'22' if dead else '19'} coins sleeves {w:.1f}/{1-w:.1f} vs weighted sum: max rel diff {err:.1e} "
              f"{'OK' if ok else 'DIFFERENT'}", flush=True)
    # guardrails "on" but never binding change nothing
    e5, _, _, _ = L.erun(P, idx, [L.MODE90], killd=0.999, killm=0.999, cln=10**6, clT=1, zcap=10.0)
    ok = np.array_equal(ea.values, e5.values)
    ok_all &= ok
    print(f"{'22' if dead else '19'} coins non-binding guardrails: {'IDENTICAL' if ok else 'DIFFERENT'}", flush=True)
print("ALL CHECKS PASS" if ok_all else "SOME CHECK FAILED")
