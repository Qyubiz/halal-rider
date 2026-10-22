"""Proof that w90_tenh_lib.trun equals engine.run with the new switches off, and reproduction of 10% mode.

Cases on the 19 coins and on the 22-coin stress panel: live settings, 10% mode (guard 35%/168h), mtp_now with
cd 25, buy penalty, per-hour N/fmax arrays, partial take-profit with trail, tpwait/tprest. For each: identical
trade logs (all engine columns) and max relative equity difference. Also: cdw=cd gives the same as cdw=-1, and
liq=1 changes nothing on the 19 coins (no coin's data ends early).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402
import w90_tenh_lib as T  # noqa: E402

out = open(os.path.join(T.OUT, "check.log"), "w")


def say(s=""):
    print(s, flush=True)
    out.write(s + "\n"); out.flush()


ok_all = True
for dead in (False, True):
    syms, idx, P = T.market(dead)
    n = len(idx)
    gm = T.guard(P, 0.35, 168)
    rng = np.random.default_rng(1)
    Narr = rng.integers(1, 4, n)
    farr = rng.choice([0.33, 0.5, 1.0], n)
    cases = {
        "live": dict(),
        "10% mode": dict(depth=0.14, N=1, fmax=1.0, tp=0.05, hm=48, cd=24, smult=gm),
        "10% mode bot timing": dict(depth=0.14, N=1, fmax=1.0, tp=0.05, hm=48, cd=25, smult=gm, mtp_now=True),
        "10% mode pen 0.2%": dict(depth=0.14, N=1, fmax=1.0, tp=0.05, hm=48, cd=25, smult=gm, mtp_now=True, pen=0.002),
        "N1 tp3 cd0 hm96": dict(depth=0.12, N=1, fmax=1.0, tp=0.03, hm=96, cd=0),
        "per-hour N/fmax": dict(depth=0.13, N=Narr, fmax=farr, tp=0.04, hm=48, cd=12),
        "partial + trail": dict(depth=0.12, N=2, fmax=0.5, tp=0.03, part=0.5, trail=0.03, hm=72, cd=6),
        "tpwait 3 no rest": dict(depth=0.12, N=2, fmax=0.5, tp=0.03, tpwait=3, tprest=False, hm=48, cd=24),
    }
    for name, kw in cases.items():
        e0, l0 = E.run(P, idx, **kw)
        e1, l1 = T.trun(P, idx, slip=E.SLIP_MAJOR, **kw)
        e2, l2 = T.trun(P, idx, slip=E.SLIP_MAJOR, cdw=int(kw.get("cd", 24)), **kw)
        same_log = len(l0) == len(l1) and np.array_equal(l0.values, l1[E.LOG_COLS].values)
        rel = float(np.max(np.abs(e1.values / e0.values - 1)))
        same2 = len(l2) == len(l1) and np.array_equal(l2.values, l1.values) and np.array_equal(e2.values, e1.values)
        ok = same_log and rel < 1e-12 and same2
        line = f"{'22' if dead else '19'} coins | {name:22s} trades {len(l0):5d} logs identical {same_log} max rel eq diff {rel:.1e} cdw=cd identical {same2}"
        if not dead:
            e3, l3 = T.trun(P, idx, slip=E.SLIP_MAJOR, liq=1, **kw)
            same3 = np.array_equal(e3.values, e1.values) and np.array_equal(l3.values, l1.values)
            line += f" liq=1 identical {same3}"
            ok = ok and same3
        ok_all &= ok
        say(line + ("  OK" if ok else "  FAIL"))

say(f"\nALL CHECKS {'PASS' if ok_all else 'FAIL'}\n")

# reproduction of the 10% mode rows of results_x10/tenpct_mode.log
syms, idx, P = T.market(False)
S10 = dict(depth=0.14, N=1, fmax=1.0, tp=0.05, hm=48, cd=24, g=0.35, gL=168)
for name, over, dead in (("10% mode", {}, False), ("10% mode, real bot timing", dict(mtp_now=True, cd=25), False),
                         ("bot timing + buys 0.2% worse", dict(mtp_now=True, cd=25, pen=0.002), False),
                         ("bot timing + buys and sells 0.2% worse", dict(mtp_now=True, cd=25, pen=0.002, pens=0.002), False),
                         ("+LUNA/FTT/SRM (frozen, engine)", {}, True),
                         ("+LUNA/FTT/SRM (sold at last hour)", dict(liq=1), True)):
    r, e, lg = T.evaluate(S10, dead=dead, **over)
    say(T.fmt(name, r) + f" | 2022 {r['y2022']*100:+.0f}%")
r, e, lg = T.evaluate(S10)
rs, _, _ = T.evaluate(S10, stress=True)
say("\n" + T.fmt("10% mode + full stressed version", r, rs))
out.close()
