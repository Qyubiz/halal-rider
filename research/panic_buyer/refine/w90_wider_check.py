"""Proof that w90_wider_lib.wkernel (pens=0, liq=0) reproduces engine.run on the 19 coins (and the 22-coin stress
panel) and x10_universe_lib.urun (bid mode) on the wider panel exactly; plus the 10% mode baseline in both cost
models. Run: PB_DATA=... PB_OUT=... python w90_wider_check.py"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402
import w90_wider_lib as L  # noqa: E402
import x10_universe_lib as X  # noqa: E402

SETTINGS = [
    dict(),
    dict(N=2, fmax=0.5, tp=0.03, cd=12, depth=0.10),
    dict(N=1, fmax=1.0, tp=0.05, cd=24, depth=0.14, hm=48),
    dict(N=1, fmax=1.0, tp=0.05, cd=25, depth=0.14, mtp_now=True, pen=0.002),
    dict(N=3, fmax=0.33, tp=0.015, cd=24, mtp_now=True, tpwait=2),
    dict(N=4, fmax=0.25, tp=0.02, cd=6, tprest=False, hm=96),
]


def same(ea, la, eb, lb):
    lb = lb[lb.why != 9][E.LOG_COLS].values
    return np.array_equal(ea.values, eb.values) and np.array_equal(la[E.LOG_COLS].values, lb)


if __name__ == "__main__":
    out = open(os.path.join(L.OUT, "check.log"), "w")

    def say(s=""):
        print(s, flush=True)
        out.write(s + "\n"); out.flush()

    ok = True
    for dead in (False, True):
        syms, idx, P = E.panel(E.load_h19(dead=dead))
        for kw in SETTINGS:
            ea, la = E.run(P, idx, **{"depth": 0.12, "N": 8, "fmax": 0.2, **kw})
            eb, lb = L.wrun(P, idx, **{"depth": 0.12, "N": 8, "fmax": 0.2, "tp": 0.015, **kw}, slip=E.SLIP_MAJOR,
                            liq=0)
            r = same(ea, la, eb, lb)
            ok &= r
            say(f"engine.run vs wrun, {len(syms)} coins {kw}: {'IDENTICAL' if r else 'DIFFERENT'} "
                f"(x{ea.iloc[-1]:.4f} vs x{eb.iloc[-1]:.4f}, {len(la)} trades)")
        # 10% mode crash-guard mask through engine.run's smult
        g = L.guard_mask(P, 0.35)
        ea, la = E.run(P, idx, depth=0.14, N=1, fmax=1.0, tp=0.05, hm=48, cd=24, smult=g)
        eb, lb = L.wrun(P, idx, depth=0.14, N=1, fmax=1.0, tp=0.05, hm=48, cd=24, smult=g, slip=E.SLIP_MAJOR, liq=0)
        r = same(ea, la, eb, lb)
        ok &= r
        say(f"10% mode with crash guard, {len(syms)} coins: {'IDENTICAL' if r else 'DIFFERENT'}")
        if not dead:
            s = L.summarize(eb, lb, idx, np.ones(len(syms), bool))
            say("10% mode, engine.run on the 19 coins (pb_data/1h, slip 0.02%): " + L.fmt(s))
            sp = L.portable_spec(dict(L.TENMODE))
            ec, lc = L.runner(P, idx, **sp(P, syms), slip=E.SLIP_MAJOR, liq=0)
            r = np.array_equal(ec.values, ea.values)
            ok &= r
            say(f"portable_spec/runner (M=0) on the 19-coin panel = engine.run 10% mode: {r}")

    W = L.build()
    say(f"\nwider panel: {len(W['syms'])} coins, {len(W['idx'])} hours {W['idx'][0]} .. {W['idx'][-1]}")
    for dd in (dict(M=50, rk="v30", dt=0.20, st=0.5), dict(M=30, rk="v365", dt=0.18, st=1.0, tt=0.08)):
        d = {**L.TENMODE, **dd}
        sm, dep, tp = L.design_arrays(W, d)
        cols = np.where((sm > 0).any(axis=0))[0]
        P = L.subpanel(W, list(cols))
        for kw in (dict(N=1, fmax=1.0, cd=24), dict(N=2, fmax=1.0, cd=12, mtp_now=True, pen=0.002)):
            ea, la = X.urun(P, W["idx"], W["slip"][:, cols], depth=dep[:, cols], smult=sm[:, cols], tp=tp[:, cols],
                            hm=48, **kw)
            eb, lb = L.wrun(P, W["idx"], depth=dep[:, cols], smult=sm[:, cols], tp=tp[:, cols], hm=48,
                            slip=W["slip"][:, cols], liq=0, **kw)
            r = np.array_equal(ea.values, eb.values) and np.array_equal(la.values, lb.values)
            ok &= r
            say(f"x10 urun (bid) vs wrun, {len(cols)} coins {dd} {kw}: {'IDENTICAL' if r else 'DIFFERENT'} "
                f"(x{ea.iloc[-1]:.4g}, {len(la)} trades)")
    say("\nALL IDENTICAL" if ok else "\nMISMATCH")

    lb_ = L.last_bars(W["P"])
    n = len(W["idx"])
    dead = [(W["syms"][k], str(W["idx"][lb_[k]])) for k in range(len(W["syms"])) if lb_[k] <= n]
    say(f"coins whose data ends before the panel end ({len(dead)}): " + ", ".join(f"{s[:-4]} {t[:10]}" for s, t in dead))
    for nm, dd in (("10% mode, 19 only, universe slippage, sold at last data hour", {}),):
        r, e, lg, syms = L.evaluate({**L.TENMODE, **dd})
        say(f"{nm}: " + L.fmt(r))
        rs, e, lg, syms = L.evaluate({**L.TENMODE, **dd}, stress=True)
        say(f"   stressed: " + L.fmt(rs))
    out.close()
