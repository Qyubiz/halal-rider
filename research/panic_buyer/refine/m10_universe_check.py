"""m10 "universe": checks that the new kernel and panel reproduce the existing engines.

1. mkernel (core only, slots off) == engine.run on the 19-coin panel: same equity and trade log, several settings
   incl. 90% mode with its crash guard.
2. mkernel == w90_wider_lib.wkernel on the wide panel with a tier (per-group depth / take-profit / size, liq on).
3. 90% mode on the wide panel, 19 coins only, constant 0.02% slippage, no last-hour sale: same %/month and win rate
   as the 19-coin engine (published: 2021-23 +13.17% WR 93.8%, 2024-26 +6.89% 92.6%, since 2025 +5.80% 93.2%).
4. slots=True with N=1 == slots=False with N=1.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m10_universe_lib as M  # noqa: E402
from m10_universe_lib import E, L  # noqa: E402


def main():
    out = open(os.path.join(M.OUT, "check.log"), "w")

    def say(s=""):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    ok_all = True
    # 1. engine.run on the 19 coins
    syms, idx, P = E.panel(E.load_h19())
    n, K = P["o"].shape
    for kw in (dict(depth=0.14, N=1, fmax=1.0, tp=0.025, hm=96, cd=3, g=0.35),
               dict(depth=0.12, N=8, fmax=0.2, tp=0.015, hm=48, cd=24, g=0.0),
               dict(depth=0.14, N=2, fmax=0.5, tp=0.025, hm=96, cd=3, g=0.35, mtp_now=True, pen=0.002)):
        k = dict(kw); g = k.pop("g")
        sm = L.guard_mask(P, g) if g > 0 else np.ones((n, K))
        e1, l1 = E.run(P, idx, smult=sm, **k)
        el = sm.astype(bool)
        eq, lg = M.mkernel(P["o"].values, P["h"].values, P["l"].values, P["cf"].values, P["ref"].values, el,
                           np.ones(K, np.bool_), k["depth"], k["depth"], k["tp"], k["tp"], 1.0, 1.0, k["N"], k["fmax"],
                           False, -1, k["hm"], k["cd"], E.EPS, E.FEE, np.full((n, K), E.SLIP_MAJOR),
                           k.get("pen", 0.0), 0.0, 1, k.get("mtp_now", False), np.full(K, n + 1, np.int64), 0)
        lg = pd.DataFrame(lg, columns=M.LOGC)
        lg = lg[lg.why != 9]
        same = np.array_equal(e1.values, eq) and np.array_equal(l1[["eb", "xb", "k", "why", "ret", "amt"]].values,
                                                                lg[["eb", "xb", "k", "why", "ret", "amt"]].values)
        ok_all &= same
        say(f"1. engine.run vs mkernel {kw}: x{e1.iloc[-1]:.6f} vs x{eq[-1]:.6f}, trades {len(l1)}/{len(lg)} "
            f"{'IDENTICAL' if same else 'DIFFERENT'}")
    # 2. w90_wider_lib.wkernel with a tier on the wide panel
    t0 = time.time()
    W = M.build()
    say(f"wide panel: {len(W['syms'])} coins, {len(W['idx'])} hours ({W['idx'][0]} - {W['idx'][-1]}), "
        f"loaded in {time.time() - t0:.0f}s")
    for d in (dict(M.MODE90, M=30, rk="v365", dt=0.26, st=0.5, tt=0.03, slots=False),
              dict(M.MODE90, M=60, rk="v30", dt=0.20, st=1.0, N=2, fmax=0.5, slots=False, lmin=5e6)):
        el, core = M.masks(W, d)
        cols = np.where(el.any(axis=0))[0]
        Pw = {f: pd.DataFrame(W[f if f != "cf" else "c"][:, cols], W["idx"]) for f in ("o", "h", "l", "cf", "ref")}
        sm = np.where(core[cols][None, :], 1.0, d["st"]) * el[:, cols]
        dep = np.where(core[cols][None, :], d["dc"], d["dt"]) * np.ones((len(W["idx"]), 1))
        tp = np.where(core[cols][None, :], d["tp"], d.get("tt", d["tp"])) * np.ones((len(W["idx"]), 1))
        e1, l1 = L.wrun(Pw, W["idx"], depth=dep, N=d["N"], fmax=d.get("fmax", 1.0), smult=sm, tp=tp, hm=d["hm"],
                        cd=d["cd"], slip=W["slip"][:, cols].astype(np.float64), liq=1)
        e2, l2, *_ = M.run_design(W, d)
        l1 = l1.assign(k=cols[l1.k.astype(int).values].astype(float))       # subpanel column -> panel column
        same = np.array_equal(e1.values, e2.values) and np.array_equal(l1.values, l2.values)
        ok_all &= same
        say(f"2. w90_wider_lib.wkernel vs mkernel, {M.name(d)}: x{e1.iloc[-1]:.4f} vs x{e2.iloc[-1]:.4f}, trades "
            f"{len(l1)}/{len(l2)} {'IDENTICAL' if same else 'DIFFERENT'}")
    # 3. 90% mode reproduction
    r, e, lg, _ = M.evaluate(M.MODE90, W=W, slip19=E.SLIP_MAJOR, liq=0)
    say("3. 90% mode on the wide panel, 19 coins, 0.02% slippage: " + M.fmt(r))
    sm = L.guard_mask(P, 0.35)
    e1, l1 = E.run(P, idx, depth=0.14, N=1, fmax=1.0, tp=0.025, hm=96, cd=3, smult=sm)
    rr = M.summarize(e1, l1.assign(eq0=1.0), idx, np.ones(K, bool))
    say("   same on the 19-coin engine panel (worst trade n/a):    " + M.fmt(rr))
    gap = max(abs(r[f"{p}_mo"] - rr[f"{p}_mo"]) for p in ("d", "t", "s"))
    same = gap < 1e-5 and all(r[f"{p}_win"] == rr[f"{p}_win"] and r[f"{p}_n"] == rr[f"{p}_n"] for p in ("d", "t", "s"))
    ok_all &= same
    say(f"   {'SAME' if same else 'DIFFERENT'} trades and win rates; largest %/month gap {gap:.2e} (the two panels end "
        f"{e.index[-1]} / {e1.index[-1]})")
    r2, *_ = M.evaluate(M.MODE90, W=W)
    say("   90% mode, rank-based slippage + last-hour sale (the m10 reference): " + M.fmt(r2))
    # 4. slots with N=1
    a, *_ = M.evaluate(dict(M.MODE90, M=30, rk="v30", dt=0.22, slots=True), W=W)
    b, *_ = M.evaluate(dict(M.MODE90, M=30, rk="v30", dt=0.22, slots=False), W=W)
    same = all(a[k_] == b[k_] or (np.isnan(a[k_]) and np.isnan(b[k_])) for k_ in a)
    ok_all &= same
    say(f"4. slots on/off with N=1: {'IDENTICAL' if same else 'DIFFERENT'}")
    say("ALL CHECKS PASS" if ok_all else "SOME CHECK FAILED")


if __name__ == "__main__":
    main()
