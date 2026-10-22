"""Round 2, lens "crowding": market-wide cascades fill many coins at once and they keep falling.

Motivation (2021-23 only: r2_crowding_motiv.py, r2_crowding_motiv2.py, r2_crowding_motiv3.py; logs in $PB_OUT):
  * Crowded hours are worse: hours with 7-8 of the bot's buys filling average +0.27% per trade with 14% time
    stops (sum of return x order size -0.18), against +1.12% and 5% for buys that fill alone; days with 10+
    fills lost money in total (-0.18).
  * But inside a crowded hour it is the FIRST fills that keep falling. Ordering fills by the minute they happened
    (1-minute data, r2_crowding_fm.py), in hours with 5+ fills: ranks 1-2 average -0.07% (13% time stops),
    ranks 3-4 +0.25%, ranks 5-6 +1.57%, ranks 7-8 +1.82%. A cap on fills per hour (cancel the other resting buys
    once F have filled) would skip exactly the good late fills: for F=2..6 the skipped trades average +1.08% to
    +1.82% against +0.94% to +0.97% for the kept ones. Rolling caps (fills after F others in the last 6/12/24h)
    show no damage either (those trades average +0.79% to +1.25%). So NO cap grid: the evidence goes the other way.
  * What does help the first fills is averaging down. A second buy at ref*(1-d2) (ref = the 24h high that set the
    12% trigger), sold with the position, returns +2.33% / +3.21% / +3.47% / +5.00% on its own money for
    d2 = 16/18/20/24% (86-90% winners), on exactly the trades whose first leg did worst (live result there -0.95%
    to -6.01%, 20-50% time stops). Of those second-leg chances, 297/159/97/50 come inside the fill hour and
    251/222/165/103 later, before the exit.
  * The live bot holds coins only 5% of the time-weighted equity on average (84% of hours hold nothing); the rest
    sits in resting buys. So money for a second leg is usually free.
  * Splitting each coin's budget (w at 12%, 1-w deeper) on the same 8 coins lowers the crude per-trade total in
    all 8 combinations tried (+3.70 to +4.28 vs +4.75 live) even though time-stop losses shrink (-1.75 to -3.59 vs
    -5.09): half the first-leg money is idle in the 70-90% of trades that never reach leg 2. Keeping the first
    leg's size means fewer coins (the 960 grid on 2021-23: 5 orders at 12% give x9.25, drop -30.4%).

Two hypotheses, both need new entry logic, so ckernel below is engine.gkernel copied with a ladder (all off by
default; with it off it reproduces engine.run exactly, proven in prove_equivalence()):
  H1 extra leg 2 (x > 0): unchanged first leg. From the hourly cycle after the first leg fills, a fully funded
     limit buy of x times the first leg's money rests at ref_fill*(1-d2); these orders are funded FIRST from free
     cash (scaled down together if cash is short), the rest goes to the usual 8 first-leg orders. At the cycle,
     an open already at/below the level buys at market (like the live rule for coins under their trigger).
  H2 split ladder (w < 1): each selected coin's budget is split, w at the 12% trigger and 1-w at ref*(1-d2), both
     resting from the hour start (both can fill in the same hour); after the first leg fills the second keeps
     resting at that price with its cash locked until it fills or the position is sold. To keep the first leg
     about the live size the bot uses fewer coins: w=1/2 on 4 coins, w=2/3 on 5 coins, cap fmax = 20%/w per coin.
  In both: after leg 2 fills, the take-profit is the average cost x (1+1.5%), placed at the next hourly cycle
  (none in the hour leg 2 fills); the 48h time stop and 24h cooldown count from the first leg's fill.
Everything is decided at hourly cycles from closed candles (ref = 24h max close shifted 1); nothing looks ahead.
"""
import sys

import numpy as np
import pandas as pd
from numba import njit

import engine as E
import study as S

L = E.LIVE
LOG = E.LOG_COLS + ["leg2", "a1"]


@njit(cache=False)
def ckernel(o, h, l, c, ref, depth, nord, fmax, smult, tpa, part, tp2, trail, hm, cd, eps, fee, slip, pen,
            tpwait, tprest, d2, w, x):
    """engine.gkernel plus a second entry leg at ref_fill*(1-d2): split (w<1, resting from the hour start, cash
    locked) or extra (x>0, x times the first leg's money, placed from the next hourly cycle, funded first)."""
    n, K = c.shape
    units = np.zeros(K); entry = np.zeros(K); amt0 = np.zeros(K); got = np.zeros(K); tpp = np.zeros(K)
    stage = np.zeros(K, np.int64); sbar = np.zeros(K, np.int64); peak = np.zeros(K)
    ebar = np.full(K, -1, np.int64); okbar = np.zeros(K, np.int64)
    lastc = np.full(K, np.nan); dist = np.empty(K)
    # ladder state
    lp2 = np.zeros(K)                       # leg-2 limit price (0: none pending)
    res2 = np.zeros(K)                      # split: cash locked in the resting leg 2
    lock = np.zeros(K)                      # extra: cash locked this hour in a leg-2 order
    a1 = np.zeros(K); e1 = np.zeros(K); tpr = np.zeros(K); leg2 = np.zeros(K)
    lbar = np.full(K, -1, np.int64)         # hour of the latest fill (take-profit from lbar+tpwait)
    cash = 1.0; eq = np.empty(n); eq_prev = 1.0
    maxlog = 600000; lg = np.zeros((maxlog, 9)); nl = 0
    for t in range(n):
        # extra leg 2: fund the pending orders first
        if x > 0.0:
            need = 0.0
            for k in range(K):
                lock[k] = 0.0
                if units[k] > 0.0 and lp2[k] > 0.0 and ebar[k] < t:
                    need += x * a1[k]
            if need > 0.0:
                sc = min(1.0, max(cash, 0.0) / need)
                for k in range(K):
                    if units[k] > 0.0 and lp2[k] > 0.0 and ebar[k] < t:
                        lock[k] = x * a1[k] * sc
                        cash -= lock[k]
        for k in range(K):
            dist[k] = 1e9
            if units[k] > 0.0 or t < okbar[k] or np.isnan(o[t, k]) or np.isnan(ref[t, k]) or smult[t, k] <= 0.0:
                continue
            dist[k] = o[t, k] / (ref[t, k] * (1.0 - depth[t, k])) - 1.0
        order = np.argsort(dist)
        nsel = 0
        for q in range(K):
            if dist[order[q]] < 1e8 and nsel < nord[t]:
                nsel += 1
        if nsel > 0 and cash > 1e-9:
            per = min(fmax[t] * eq_prev, cash / nsel)
            for q in range(nsel):
                k = order[q]
                lim = ref[t, k] * (1.0 - depth[t, k])
                fill = -1.0
                if o[t, k] <= lim:
                    fill = o[t, k] * (1.0 + slip[k]) * (1.0 + pen)
                elif l[t, k] <= lim * (1.0 - eps):
                    fill = lim * (1.0 + pen)
                if fill > 0.0 and cash > 1e-9:
                    a = min(per * smult[t, k], cash)
                    b1 = a * w
                    b2 = a - b1
                    cash -= a; units[k] = b1 * (1.0 - fee) / fill
                    entry[k] = fill; amt0[k] = b1; ebar[k] = t; got[k] = 0.0
                    tpp[k] = fill * (1.0 + tpa[t, k]); stage[k] = 0; peak[k] = fill
                    lbar[k] = t; a1[k] = b1; e1[k] = fill; tpr[k] = tpa[t, k]; leg2[k] = 0.0
                    lp2[k] = 0.0; res2[k] = 0.0
                    if d2 > 0.0 and (b2 > 0.0 or x > 0.0):
                        lp2[k] = ref[t, k] * (1.0 - d2)
                    if b2 > 0.0:                                  # split: leg 2 rests from the hour start
                        f2 = -1.0
                        if o[t, k] <= lp2[k]:
                            f2 = o[t, k] * (1.0 + slip[k]) * (1.0 + pen)
                        elif l[t, k] <= lp2[k] * (1.0 - eps):
                            f2 = lp2[k] * (1.0 + pen)
                        if f2 > 0.0:
                            units[k] += b2 * (1.0 - fee) / f2; amt0[k] = b1 + b2
                            entry[k] = amt0[k] / (b1 / fill + b2 / f2); tpp[k] = entry[k] * (1.0 + tpr[k])
                            leg2[k] = 1.0; lp2[k] = 0.0
                        else:
                            res2[k] = b2
        for k in range(K):
            if units[k] > 0.0 and ebar[k] < t and not np.isnan(o[t, k]):
                # leg 2: split order resting since the first fill, or this hour's extra order
                if lp2[k] > 0.0 and (res2[k] > 0.0 or lock[k] > 0.0):
                    f2 = -1.0
                    if res2[k] > 0.0:
                        b2 = res2[k]
                        if l[t, k] <= lp2[k] * (1.0 - eps):
                            f2 = lp2[k] * (1.0 + pen)
                    else:
                        b2 = lock[k]
                        if o[t, k] <= lp2[k]:
                            f2 = o[t, k] * (1.0 + slip[k]) * (1.0 + pen)
                        elif l[t, k] <= lp2[k] * (1.0 - eps):
                            f2 = lp2[k] * (1.0 + pen)
                    if f2 > 0.0:
                        units[k] += b2 * (1.0 - fee) / f2; amt0[k] = a1[k] + b2
                        entry[k] = amt0[k] / (a1[k] / e1[k] + b2 / f2); tpp[k] = entry[k] * (1.0 + tpr[k])
                        leg2[k] = 1.0; lp2[k] = 0.0; res2[k] = 0.0; lock[k] = 0.0; lbar[k] = t
                px = -1.0; why = 0.0
                if stage[k] == 0:
                    if t - lbar[k] < tpwait:
                        pass
                    elif o[t, k] >= tpp[k]:
                        px = o[t, k]
                    elif tprest and h[t, k] >= tpp[k] * (1.0 + eps):
                        px = tpp[k]
                    if px > 0.0 and part < 1.0:
                        q = units[k] * part
                        p1 = q * px * (1.0 - fee)
                        cash += p1; got[k] += p1; units[k] -= q
                        stage[k] = 1; sbar[k] = t; peak[k] = max(px, h[t, k])
                        px = -1.0
                elif sbar[k] < t:
                    if tp2 > 0.0:
                        p2 = entry[k] * (1.0 + tp2)
                        if o[t, k] >= p2:
                            px = o[t, k]; why = 3.0
                        elif h[t, k] >= p2 * (1.0 + eps):
                            px = p2; why = 3.0
                    if px < 0.0 and trail > 0.0:
                        if h[t, k] > peak[k]:
                            peak[k] = h[t, k]
                        if c[t, k] <= peak[k] * (1.0 - trail):
                            px = c[t, k] * (1.0 - slip[k]); why = 2.0
                if px < 0.0 and t - ebar[k] >= hm[t]:
                    px = c[t, k] * (1.0 - slip[k]); why = 1.0
                if px > 0.0:
                    proceeds = units[k] * px * (1.0 - fee); cash += proceeds; got[k] += proceeds
                    if nl < maxlog:
                        lg[nl, 0] = ebar[k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = why
                        lg[nl, 4] = got[k] / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]
                        lg[nl, 7] = leg2[k]; lg[nl, 8] = a1[k]; nl += 1
                    units[k] = 0.0; okbar[k] = t + cd
                    cash += res2[k]; res2[k] = 0.0; lp2[k] = 0.0      # cancel an unfilled split leg 2
        if x > 0.0:                                                   # unfilled extra orders are cancelled
            for k in range(K):
                if lock[k] > 0.0:
                    cash += lock[k]; lock[k] = 0.0
        v = cash
        for k in range(K):
            v += res2[k]
            if not np.isnan(c[t, k]):
                lastc[k] = c[t, k]
            if units[k] > 0.0:
                v += units[k] * lastc[k]
        eq[t] = v; eq_prev = v
    return eq, lg[:nl].copy()


_orig_run = E.run


def crun(P, idx, depth=L["depth"], N=L["N"], fmax=L["fmax"], smult=1.0, tp=L["tp"], part=1.0, tp2=0.0, trail=0.0,
         hm=L["hm"], cd=L["cd"], eps=E.EPS, fee=E.FEE, slip=E.SLIP_MAJOR, pen=0.0, tpwait=1, tprest=True,
         d2=0.0, w=1.0, x=0.0):
    """engine.run with the ladder; same arguments and defaults otherwise."""
    assert not (w < 1.0 and x > 0.0), "split and extra leg 2 are separate hypotheses"
    O = P["o"].values; n, K = O.shape
    sl = np.full(K, slip) if np.ndim(slip) == 0 else np.asarray(slip, np.float64)
    eq, lg = ckernel(O, P["h"].values, P["l"].values, P["cf"].values, P["ref"].values,
                     E.full(n, K, depth), E.full(n, 0, N, np.int64), E.full(n, 0, fmax), E.full(n, K, smult),
                     E.full(n, K, tp), float(part), float(tp2), float(trail), E.full(n, 0, hm, np.int64), int(cd), eps,
                     fee, sl, pen, int(tpwait), bool(tprest), float(d2), float(w), float(x))
    return pd.Series(eq, idx), pd.DataFrame(lg, columns=LOG)


# ------------------------------------------------------------------ the grids (fixed before any result is seen)
H1 = {}
for xx in (0.5, 1.0):
    for dd in (0.16, 0.18, 0.20, 0.24):
        H1[f"extra leg 2: {xx:g}x at {dd:.0%}"] = {"x": xx, "d2": dd}

H2 = {}
for ww, nn in ((0.5, 4), (2 / 3, 5)):
    for dd in (0.16, 0.18, 0.20, 0.24):
        H2[f"split {ww:.0%} at 12% / rest at {dd:.0%}, {nn} coins"] = {"w": ww, "N": nn, "fmax": 0.20 / ww, "d2": dd}


# ------------------------------------------------------------------ code checks (no results)
def synthetic_check():
    """Hand-made one-coin paths: the ladder must fill and exit exactly as described."""
    def mk(o, h, l, c, ref):
        idx = pd.date_range("2021-01-01", periods=len(o), freq="1h")
        P = {f: pd.DataFrame({"A": np.asarray(v, float)}, index=idx) for f, v in
             (("o", o), ("h", h), ("l", l), ("c", c), ("cf", c), ("ref", ref))}
        return P, idx
    fee, sl = E.FEE, E.SLIP_MAJOR
    n = 6
    ref = [100.0] * n
    # hour 0: low 86.8 fills leg 1 at 88; hour 1: low 81.0 fills leg 2 at 82; hour 2: high 87 >= tp of the average (86.17)
    o = [95, 87, 82.5, 84, 84, 84]; h = [95, 87, 87, 85, 85, 85]; l = [86.8, 81.0, 82, 83, 83, 83]
    c = [87, 82.5, 84, 84, 84, 84]
    P, idx = mk(o, h, l, c, ref)
    # extra leg 2 (x=1 at 18%): leg 1 = 20% of equity (1 coin -> fmax binds), leg 2 = the same money at 82
    e, lg = crun(P, idx, N=1, x=1.0, d2=0.18)
    a = 0.2
    u = a * (1 - fee) / 88 + a * (1 - fee) / 82
    avg = 2 * a / (a / 88 + a / 82)
    want = u * avg * 1.015 * (1 - fee) / (2 * a) - 1
    ok1 = len(lg) == 1 and lg.leg2[0] == 1 and lg.xb[0] == 2 and abs(lg.ret[0] - want) < 1e-12 and abs(lg.amt[0] - 0.4) < 1e-12
    print(f"  extra leg 2: trades {len(lg)}, leg2 {lg.leg2.tolist()}, exit hour {lg.xb.tolist()}, ret {lg.ret.tolist()} "
          f"(want {want}) -> {'ok' if ok1 else 'WRONG'}")
    # split (w=0.5 at 18%): both legs rest from hour 0, leg 2 fills in hour 1, same exit
    e, lg = crun(P, idx, N=1, w=0.5, fmax=0.4, d2=0.18)
    b = 0.2
    u = b * (1 - fee) / 88 + b * (1 - fee) / 82
    want2 = u * avg * 1.015 * (1 - fee) / (2 * b) - 1
    ok2 = len(lg) == 1 and lg.leg2[0] == 1 and lg.xb[0] == 2 and abs(lg.ret[0] - want2) < 1e-12
    print(f"  split leg 2: trades {len(lg)}, leg2 {lg.leg2.tolist()}, exit hour {lg.xb.tolist()}, ret {lg.ret.tolist()} "
          f"(want {want2}) -> {'ok' if ok2 else 'WRONG'}")
    # split, both legs in the fill hour (low 81 in hour 0); TP at hour 1 open (>= avg*1.015)
    o3 = [95, 86.5, 86, 86, 86, 86]; h3 = [95, 87, 87, 87, 87, 87]; l3 = [81.0, 85, 85, 85, 85, 85]; c3 = [85] * 6
    P3, idx3 = mk(o3, h3, l3, c3, ref)
    e, lg = crun(P3, idx3, N=1, w=0.5, fmax=0.4, d2=0.18)
    want3 = u * 86.5 * (1 - fee) / (2 * b) - 1
    ok3 = len(lg) == 1 and lg.leg2[0] == 1 and lg.xb[0] == 1 and abs(lg.ret[0] - want3) < 1e-12
    print(f"  split, both legs in the fill hour: exit hour {lg.xb.tolist()}, ret {lg.ret.tolist()} (want {want3}) "
          f"-> {'ok' if ok3 else 'WRONG'}")
    # extra leg 2 never reached: leg 1 sells at +1.5%, all locked cash comes back (equity = cash after the exit)
    o4 = [95, 88.5, 89, 89, 89, 89]; h4 = [95, 89, 90, 90, 90, 90]; l4 = [86.8, 87, 88, 88, 88, 88]; c4 = [88] * 6
    P4, idx4 = mk(o4, h4, l4, c4, ref)
    e, lg = crun(P4, idx4, N=1, x=1.0, d2=0.18)
    e0, lg0 = _orig_run(P4, idx4, N=1)
    ok4 = len(lg) == 1 and lg.leg2[0] == 0 and np.allclose(e.values, e0.values, rtol=0, atol=1e-15)
    print(f"  extra leg 2 not reached: same equity as the live engine {np.allclose(e.values, e0.values, rtol=0, atol=1e-15)} "
          f"-> {'ok' if ok4 else 'WRONG'}")
    # time stop counts from the first leg; leg 2 market buy at the cycle when the open is already under its level
    o5 = [95, 80, 80, 80, 80, 80]; h5 = [95, 81, 81, 81, 81, 81]; l5 = [86.8, 79.5, 79.5, 79.5, 79.5, 79.5]; c5 = [80] * 6
    P5, idx5 = mk(o5, h5, l5, c5, ref)
    e, lg = crun(P5, idx5, N=1, x=1.0, d2=0.18, hm=3)
    f2 = 80 * (1 + sl)
    u = a * (1 - fee) / 88 + a * (1 - fee) / f2
    want5 = u * 80 * (1 - sl) * (1 - fee) / (2 * a) - 1
    ok5 = len(lg) == 1 and lg.why[0] == 1 and lg.xb[0] == 3 and abs(lg.ret[0] - want5) < 1e-12
    print(f"  leg 2 at market at the cycle + time stop from leg 1: exit hour {lg.xb.tolist()} why {lg.why.tolist()} "
          f"ret {lg.ret.tolist()} (want {want5}) -> {'ok' if ok5 else 'WRONG'}")
    return ok1 and ok2 and ok3 and ok4 and ok5


def prove_equivalence():
    """crun with the ladder off must give exactly engine.run's equity curve and trades."""
    ok = True
    for dead, pen in ((False, 0.0), (True, 0.0), (False, 0.002), (True, 0.002)):
        syms, idx, P = S.market(dead)
        e0, l0 = _orig_run(P, idx, pen=pen)
        e1, l1 = crun(P, idx, pen=pen)
        same = np.array_equal(e0.values, e1.values) and l0.equals(l1[E.LOG_COLS])
        ok &= same
        print(E.line(f"crun, ladder off, dead={dead} pen={pen}", E.score(e1, l1, idx)) + f"  identical to engine.run: {same}")
    ok &= synthetic_check()
    if not ok:
        raise SystemExit("ckernel does not reproduce engine.run / the hand-made cases - stop")


def anatomy(name, spec):
    syms, idx, P = S.market(False)
    e, lg = crun(P, idx, **spec)
    per = np.where(idx[lg.xb.astype(int).values] < E.DESIGN_END, "21-23", "24-26")
    for p in ("21-23", "24-26"):
        g = lg[per == p]
        parts = [f"{wn}: {int((g.why == cc).sum())} ({(g.ret[g.why == cc] * g.amt[g.why == cc]).sum():+.2f}, "
                 f"avg {g.ret[g.why == cc].mean() * 100:+.1f}%)" for cc, wn in ((0, "TP"), (1, "time stop")) if (g.why == cc).any()]
        l2 = g[g.leg2 == 1]
        print(f"  anatomy {name} {p}: {len(g)} trades | " + " | ".join(parts) +
              f" | with leg 2: {len(l2)} (avg {l2.ret.mean() * 100 if len(l2) else np.nan:+.1f}%, "
              f"time stops {int((l2.why == 1).sum())}) | worst {g.ret.min() * 100:+.1f}%")


if __name__ == "__main__":
    prove_equivalence()
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        sys.exit(0)
    E.run = crun        # in memory only: study.study() now runs every variant (and the baseline) through crun
    R1, v1 = S.study("r2 crowding H1: extra leg 2, funded first from free cash", H1, "r2_crowding_h1_extra")
    S.save_verdict(v1)
    R2, v2 = S.study("r2 crowding H2: split ladder on fewer coins", H2, "r2_crowding_h2_split")
    S.save_verdict(v2)
    # anatomy of each pick, printed after both grids have run
    anatomy("LIVE SETTINGS", {})
    for v, V in ((v1, H1), (v2, H2)):
        if v["pick"]:
            anatomy(v["pick"], V[v["pick"]])
