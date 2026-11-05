"""m10 ladder: code checks only (no design is judged here).

1. One leg, every new switch off: m10_ladder_lib.lrun == engine.run exactly (equity and trade log) for the live
   settings, 90% mode (with the crash guard, == w90_wr90_lib.wrun) and two more settings incl. mtp_now and pen; on the
   19 coins and with LUNA/FTT/SRM.
2. Two legs, reserve mode, no holdings cap: lrun == r3_ladder.lrun exactly (equity and trade log).
3. Hand-made one-coin paths for the new parts: 3-leg fills and the average-cost take-profit, the time stop, the
   max-loss exit, the break-even target, lazy funding, the holdings cap and the open-at-end mark.
Usage: PB_DATA=... PB_OUT=... python3 m10_ladder_check.py
"""
import numpy as np
import pandas as pd

import engine as E
import m10_ladder_lib as M
import w90_wr90_lib as W

FEE, SL, EPS = E.FEE, E.SLIP_MAJOR, E.EPS
COLS = ["eb", "xb", "k", "why", "ret", "amt", "entry"]


def same(e1, l1, e2, l2):
    l2 = l2[l2.why != 9]
    l1 = l1[l1.why != 9] if "why" in l1 else l1
    return np.array_equal(e1.values, e2.values) and np.array_equal(l1[COLS].values, l2[COLS].values)


def check_engine():
    ok = True
    cases = [(dict(E.LIVE), dict(legs=((0.12, 1.0),), H=999, N=8, fmax=0.2, tp=0.015, hm=48, cd=24, guard=0.0), 0.0),
             (dict(depth=0.14, N=1, fmax=1.0, tp=0.025, hm=96, cd=3),
              dict(legs=((0.14, 1.0),), H=999, N=1, fmax=1.0, tp=0.025, hm=96, cd=3, guard=0.35), 0.35),
             (dict(depth=0.12, N=2, fmax=0.5, tp=0.02, hm=96, cd=6, mtp_now=True, pen=0.002),
              dict(legs=((0.12, 1.0),), H=999, N=2, fmax=0.5, tp=0.02, hm=96, cd=6, guard=0.0, mtp_now=True, pen=0.002), 0.0),
             (dict(depth=0.16, N=4, fmax=0.25, tp=0.01, hm=168, cd=12, tpwait=2),
              dict(legs=((0.16, 1.0),), H=999, N=4, fmax=0.25, tp=0.01, hm=168, cd=12, guard=0.0, tpwait=2), 0.0)]
    for dead in (False, True):
        syms, idx, P = M.market(dead)
        for ekw, mkw, g in cases:
            sm = W.guard_mask(P, g) if g > 0 else 1.0
            e1, l1 = E.run(P, idx, smult=sm, **ekw)
            e2, l2 = M.lrun(P, idx, **mkw)
            s = same(e1, l1, e2, l2)
            ew, lw = W.wrun(P, idx, guard=g, **ekw)
            s2 = np.array_equal(ew.values, e2.values) and np.array_equal(lw[COLS].values, l2[COLS].values)
            ok &= s and s2
            print(f"{'22' if dead else '19'} coins {ekw}: engine x{e1.iloc[-1]:.6f} m10 x{e2.iloc[-1]:.6f} "
                  f"trades {len(l1)}/{len(l2[l2.why != 9])} -> {'IDENTICAL' if s else 'DIFFERENT'}; "
                  f"wrun incl. open-at-end rows: {'IDENTICAL' if s2 else 'DIFFERENT'}", flush=True)
    return ok


def check_r3():
    import r3_ladder as R3
    ok = True
    for dead, pen, mtp in ((False, 0.0, False), (True, 0.0, False), (False, 0.002, True)):
        syms, idx, P = M.market(dead)
        for w, nn, cap, d2 in ((0.5, 4, 0.25, 0.18), (2 / 3, 5, 0.20, 0.20)):
            e1, l1 = R3.lrun(P, idx, N=nn, fmax=cap, w=w, d2=d2, pen=pen, mtp_now=mtp)
            e2, l2 = M.lrun(P, idx, legs=((0.12, w), (d2, 1 - w)), H=999, N=nn, fmax=cap, tp=0.015, hm=48, cd=24,
                            guard=0.0, pen=pen, mtp_now=mtp)
            l2c = l2[l2.why != 9]
            # w=2/3: r3 sizes leg 2 as a - a*w, m10 as a*(1/3): equal up to floating rounding (1e-15 relative)
            exact = np.array_equal(e1.values, e2.values) and np.array_equal(l1[COLS].values, l2c[COLS].values)
            dmax = float(np.max(np.abs(e1.values / e2.values - 1)))
            s = (len(l1) == len(l2c) and dmax < 1e-9 and np.allclose(l1[COLS].values, l2c[COLS].values, rtol=1e-9, atol=1e-12)
                 and np.array_equal(l1.leg2.values + 1, l2c.nf.values))
            ok &= s
            print(f"r3_ladder {'22' if dead else '19'} coins w{w:.2f} N{nn} cap{cap} d2 {d2} pen {pen} mtp {mtp}: "
                  f"r3 x{e1.iloc[-1]:.6f} m10 x{e2.iloc[-1]:.6f} trades {len(l1)}/{len(l2c)} max equity gap {dmax:.1e} -> "
                  f"{'IDENTICAL' if exact else ('SAME up to rounding' if s else 'DIFFERENT')}", flush=True)
    return ok


def _mk(o, h, l, c, ref, cols=("A",)):
    idx = pd.date_range("2021-01-01", periods=len(o), freq="1h")
    P = {f: pd.DataFrame({cc: np.asarray(v, float) for cc in cols}, index=idx) for f, v in
         (("o", o), ("h", h), ("l", l), ("c", c), ("cf", c), ("ref", ref))}
    return P, idx


def synthetic():
    ok = True
    legs3 = ((0.10, 1.0), (0.20, 1.0), (0.30, 1.0))
    # 1. leg 1 at 90 (hour 0), leg 2 at 80 (hour 1), leg 3 at 70 (hour 2); hour 3 high 80 >= avg x 1.02 -> sold at tp
    o = [95, 88, 78, 72, 75, 75]; h = [95, 88, 78, 82, 75, 75]; l = [89.8, 79.8, 69.8, 71, 75, 75]
    c = [88, 78, 72, 75, 75, 75]; ref = [100.0] * 6
    P, idx = _mk(o, h, l, c, ref)
    e, lg = M.lrun(P, idx, legs=legs3, H=1, fmax=1.0, tp=0.02, hm=999, cd=3, guard=0)
    b = 1 / 3
    u = b * (1 - FEE) * (1 / 90 + 1 / 80 + 1 / 70)
    avg = 1.0 / (b / 90 + b / 80 + b / 70)
    want = u * avg * 1.02 * (1 - FEE) - 1
    ok1 = len(lg) == 1 and lg.why[0] == 0 and lg.xb[0] == 3 and lg.nf[0] == 3 and abs(lg.ret[0] - want) < 1e-12 \
        and abs(e.iloc[-1] - (1 + want)) < 1e-12 and abs(lg.entry[0] - avg) < 1e-12
    print(f"  1 three legs, tp on the average: exit hour {lg.xb.tolist()} legs {lg.nf.tolist()} ret {lg.ret.tolist()} "
          f"(want {want:.10f}) -> {'ok' if ok1 else 'WRONG'}")
    ok &= ok1
    # 1b. equity in hour 0 (reserve mode): two legs' cash still locked counts at face value
    eh0 = 1 - b + b * (1 - FEE) / 90 * 88
    ok1b = abs(e.iloc[0] - eh0) < 1e-12
    print(f"  1b reserve-mode equity after leg 1: {e.iloc[0]:.12f} (want {eh0:.12f}) -> {'ok' if ok1b else 'WRONG'}")
    ok &= ok1b
    # 2. time stop: same path but price stays at 72 -> sold at hour hm=4's close with slippage
    o2 = [95, 88, 78, 72, 72, 72, 72]; h2 = [95, 88, 78, 73, 73, 73, 73]; l2 = [89.8, 79.8, 69.8, 71, 71, 71, 71]
    c2 = [88, 78, 72, 72, 72, 72, 72]
    P2, idx2 = _mk(o2, h2, l2, c2, [100.0] * 7)
    e, lg = M.lrun(P2, idx2, legs=legs3, H=1, fmax=1.0, tp=0.02, hm=4, cd=3, guard=0)
    want2 = u * 72 * (1 - SL) * (1 - FEE) - 1
    ok2 = len(lg) == 1 and lg.why[0] == 1 and lg.xb[0] == 4 and abs(lg.ret[0] - want2) < 1e-12
    print(f"  2 time stop from the first fill: exit hour {lg.xb.tolist()} why {lg.why.tolist()} -> {'ok' if ok2 else 'WRONG'}")
    ok &= ok2
    # 3. max-loss 10% under the average (avg ~79.3): close 71 is <= 71.4 -> sold at hour 3's close
    e, lg = M.lrun(P2, idx2, legs=legs3, H=1, fmax=1.0, tp=0.02, hm=999, cd=3, guard=0, ml=0.09)
    ok3 = lg.why[0] == 4 and lg.xb[0] == 2
    print(f"  3 max-loss 9% under the average: exit hour {lg.xb.tolist()} why {lg.why.tolist()} -> {'ok' if ok3 else 'WRONG'}")
    ok &= ok3
    # 4. break-even target after 4h: avg x 1.003 (~79.5) is hit at hour 6 (high 80) though the 2% target (~80.9) is not
    o4 = [95, 88, 78, 72, 75, 78, 79, 79]; h4 = [95, 88, 78, 74, 76, 79, 80, 80]; l4 = [89.8, 79.8, 69.8, 71, 74, 77, 78, 78]
    c4 = [88, 78, 72, 74, 76, 79, 79, 79]
    P4, idx4 = _mk(o4, h4, l4, c4, [100.0] * 8)
    e, lg = M.lrun(P4, idx4, legs=legs3, H=1, fmax=1.0, tp=0.02, hm=999, cd=3, guard=0, beh=4, be=0.003)
    e0, lg0 = M.lrun(P4, idx4, legs=legs3, H=1, fmax=1.0, tp=0.02, hm=999, cd=3, guard=0)
    want4 = u * avg * 1.003 * (1 - FEE) - 1
    ok4 = len(lg) == 1 and lg.why[0] == 0 and lg.xb[0] == 6 and abs(lg.ret[0] - want4) < 1e-12 and lg0.why[0] == 9
    print(f"  4 break-even target after 4h: exit hour {lg.xb.tolist()} ret {lg.ret.tolist()} (want {want4:.6f}); "
          f"without it still open (why {lg0.why.tolist()}) -> {'ok' if ok4 else 'WRONG'}")
    ok &= ok4
    # 5. lazy funding: while A holds (leg 1 only), B triggers; in lazy mode A's unfilled legs are funded only for the
    #    hour, so B gets the cash cap allows (H=2, fmax 50%: B's budget 50% of equity in both modes)
    idx = pd.date_range("2021-01-01", periods=5, freq="1h")
    oA = [95, 89, 89, 89, 89]; hA = [95, 89.5, 89.5, 89.5, 89.5]; lA = [89.8, 88.5, 88.5, 88.5, 88.5]; cA = [89.0] * 5
    oB = [99, 99, 95, 89, 89]; hB = [99, 99, 95, 89.5, 89.5]; lB = [98, 98, 89.8, 88.5, 88.5]; cB = [99, 99, 89, 89, 89]
    P5 = {f: pd.DataFrame({"A": a, "B": bb}, index=idx) for f, a, bb in
          (("o", oA, oB), ("h", hA, hB), ("l", lA, lB), ("c", cA, cB), ("cf", cA, cB), ("ref", [100.0] * 5, [100.0] * 5))}
    e, lg = M.lrun(P5, idx, legs=legs3, H=2, fmax=0.5, tp=0.02, hm=999, cd=3, guard=0, lazy=1)
    eR, lgR = M.lrun(P5, idx, legs=legs3, H=2, fmax=0.5, tp=0.02, hm=999, cd=3, guard=0, lazy=0)
    # lazy: A's unused legs (1/3) come back after hour 0, its next leg (1/6) is locked each hour, so cash 2/3 is free
    # and B gets min(50% x equity, 2/3) = 50% x equity; reserve: A's legs stay locked, cash 1/2, B gets the same
    okl = (len(lg) == 2 and abs(lg.bud.values[1] - 0.5 * e.iloc[1]) < 1e-12 and len(lgR) == 2
           and abs(lgR.bud.values[1] - 0.5 * eR.iloc[1]) < 1e-12)
    print(f"  5 two coins, B's budget lazy {lg.bud.values[1]:.6f} (want {0.5 * e.iloc[1]:.6f}) reserve {lgR.bud.values[1]:.6f} "
          f"(want {0.5 * eR.iloc[1]:.6f}) -> {'ok' if okl else 'WRONG'}")
    ok &= okl
    # 6. holdings cap: H=1 -> B gets no order while A is held
    e, lg = M.lrun(P5, idx, legs=legs3, H=1, fmax=1.0, tp=0.02, hm=999, cd=3, guard=0)
    ok6 = len(lg) == 1 and int(lg.k[0]) == 0
    print(f"  6 holdings cap H=1: positions {lg.k.tolist()} -> {'ok' if ok6 else 'WRONG'}")
    ok &= ok6
    # 7. lazy mode: H=2, fmax=1, both coins bid in hour 0 -> A's budget 50% (leg 1 = 1/6), its unfilled legs' cash comes
    #    back after hour 0; hour 1 A's next leg (1/6) is locked first; hour 2 B fills with the free 2/3; hour 3 the 1/6
    #    released at hour 2's end funds A's leg 2, which fills at 80.
    oA = [95, 89, 89, 81, 81]; hA = [95, 89.5, 89.5, 81, 81]; lA = [89.8, 88.5, 88.5, 79.8, 80]; cA = [89, 89, 89, 81, 81]
    P7 = {f: pd.DataFrame({"A": a, "B": bb}, index=idx) for f, a, bb in
          (("o", oA, oB), ("h", hA, hB), ("l", lA, lB), ("c", cA, cB), ("cf", cA, cB), ("ref", [100.0] * 5, [100.0] * 5))}
    e, lg = M.lrun(P7, idx, legs=legs3, H=2, fmax=1.0, tp=0.02, hm=999, cd=3, guard=0, lazy=1)
    a_ = lg[lg.k == 0].iloc[0]; b_ = lg[lg.k == 1].iloc[0]
    okz = (len(lg) == 2 and int(a_.nf) == 2 and abs(a_.bud - 0.5) < 1e-12 and abs(a_.amt - 1 / 3) < 1e-12
           and abs(b_.bud - 2 / 3) < 1e-12)
    print(f"  7 lazy: A legs {int(a_.nf)} budget {a_.bud:.4f} spent {a_.amt:.4f}; B budget {b_.bud:.4f} -> {'ok' if okz else 'WRONG'}")
    ok &= okz
    return ok


if __name__ == "__main__":
    a = check_engine()
    b = check_r3()
    c = synthetic()
    print("ALL CHECKS PASS" if (a and b and c) else "SOME CHECK FAILED")
