"""Round 3, task "ladder": the split ladder of round 2 (r2_crowding.py H2) with the coin's total capped.

STATUS OF THIS FAMILY (read first): the split-ladder family has already been looked at on 2024-26. In round 2,
r2_crowding.py H2 ran 8 uncapped split variants (cap 20%/w per coin, i.e. up to 40% / 30% of equity in one coin)
and printed their 2024-26 results: 3 of 8 beat live there, and its pick (50% at 12% / 50% at 18%, 4 coins)
beat live in 2021-23 and 2024-26 but failed the LUNA/FTT/SRM check. This capped grid was written AFTER those
2024-26 numbers were known, so a pass here is NOT clean out-of-sample evidence: 2024-26 is contaminated for this
family. Only the reserved 2018-20 holdout (evaluated once, later, by someone else) can decide it.

Ladder (as round 2 H2): each selected coin's budget a = min(fmax x equity, cash / nsel) is split, w at the usual
12% trigger and 1-w at ref*(1-d2) (ref = the 24h high close at the leg-1 order hour), both resting from the hour
start, both fully funded (cash for leg 2 is locked until it fills or the position is sold). After leg 1 fills,
leg 2 keeps resting at that fixed price. After leg 2 fills, the take-profit is the average cost x 1.015,
placed at the next hourly cycle (none in the hour leg 2 fills); the 48h time stop and the 24h cooldown count
from the first leg's fill. When the position is sold, an unfilled leg 2 is cancelled and its cash comes back.
NEW HERE: fmax caps the coin's TOTAL (both legs) at 20% or 25% of equity, instead of round 2's 20%/w.

lkernel is engine.gkernel (the current version, with mtp_now) copied with the split ladder added; with w=1 it
reproduces engine.run exactly (prove_equivalence(), all four market/fill combinations and mtp_now=True), and with
the ladder on and mtp_now off it is bit-identical to round 2's r2_crowding.crun (r3_ladder_xcheck.py, 96 runs).
Everything is decided at hourly cycles from closed candles (ref = 24h max close shifted 1); nothing looks ahead.

MOTIVATION (2021-23 only; r3_ladder_motiv.py runs on panels cut at 2024-01-01, log motiv.log in $PB_OUT):
  * Round 2's uncapped 4-coin ladder (50% at 12% / 50% at 18%, cap 40%) on the 19 coins: 818 of 1,633 trades had a
    budget of 20-25% of equity (cash/4 at full cash) and only 40 went above 25% (crude contribution +0.147 of
    +2.262). A 25% cap would barely touch it (same trades scaled: +2.228); a 20% cap would trim 858 trades
    (scaled: +1.912).
  * With LUNA/FTT/SRM added (2021-23 x4.82, drop -49.0%, live x5.22 / -31.4%): the trades above 25% LOST money
    (34 trades, -0.094; scaled to a 25% cap the crude total rises from +1.758 to +1.803). The drop is May 2021:
    one LUNA trade at 40.0% of equity lost 47.4% (-19.0% of equity), SOL at 23.8% lost 43.3% (-10.3%).
    LUNA's trades sum to -28.6% of equity against +0.7% for the live settings, where no coin exceeds 20%.
  * The 5-coin family (67% at 12%, cap 30%) rarely used more than 20% (50 of 1,657 trades, +0.010), but its
    2021-23 drop on the 19 coins is already -33.8% (vs live -29.2%), mostly from 20%-budget trades (DOT, LINK,
    XRP -30..-33% on 21-23 May 2021) plus NEAR at 30.0% (-7.1% of equity).
  * Leg-2 depth (round 2's 2021-23 numbers): d2 = 16% gave the deepest 2021-23 drops in both families (-41% and
    -34%), so it is left out; 18% and 20% gave x9.07 / -23% and x7.79 / -22% (4 coins), x6.74 / -34% and
    x6.61 / -30% (5 coins); 24% had the lowest 4-coin return (x6.17). A second leg's own money returned
    +3.21% (18%) / +3.47% (20%) on 381 / 262 of the 1,681 live 2021-23 trades (r2_crowding_motiv.py).

GRID (fixed here, before any capped variant was run): first-leg share w {1/2, 2/3} x (coins N, cap) {(4, 25%),
(5, 20%)} x d2 {18%, 20%} = 8 variants. The cap is 1/N, so at full cash the N coins can use all the money
(like live's 8 x 12.5%) and no coin ever holds more than 25% / 20% of equity, both legs together.
First leg at full cash: 12.5% (w 1/2, 4 coins; the live first-leg size), 16.7% (w 2/3, 4 coins), 10% (w 1/2,
5 coins), 13.3% (w 2/3, 5 coins).

RESULT (added after the single run; log r3_ladder.log in $PB_OUT): REJECTED. Pick on 2021-23: 50% at 12% / rest at
18%, 4 coins, cap 25% (x8.85, drop -22.6%, 2022 +39%); 2024-26 x2.67 / -15.1% beats live (identical to round 2's
uncapped pick: no 2024-26 budget ever exceeded 25%, so the cap only acts in 2021-23); fills 0.2% worse ok
(x7.41 / x2.49); +LUNA/FTT/SRM FAILS: 2021-23 x5.16, drop -44.2% (live x5.22 / -31.4%). No holdout candidate.

Usage: python r3_ladder.py check   (equivalence + hand-made paths only)
       python r3_ladder.py         (checks, then the grid once through study.study())
"""
import sys

import numpy as np
import pandas as pd
from numba import njit

import engine as E
import study as S

L = E.LIVE
LOG = E.LOG_COLS + ["leg2", "a1", "bud"]      # leg2: 1 if leg 2 filled; a1: leg-1 money; bud: both legs' budget


@njit(cache=False)
def lkernel(o, h, l, c, ref, depth, nord, fmax, smult, tpa, part, tp2, trail, hm, cd, eps, fee, slip, pen,
            tpwait, tprest, mtp_now, d2, w):
    """engine.gkernel plus a split ladder: w of each coin's budget at the trigger, 1-w resting at ref*(1-d2)."""
    n, K = c.shape
    units = np.zeros(K); entry = np.zeros(K); amt0 = np.zeros(K); got = np.zeros(K); tpp = np.zeros(K)
    stage = np.zeros(K, np.int64); sbar = np.zeros(K, np.int64); peak = np.zeros(K)
    ebar = np.full(K, -1, np.int64); okbar = np.zeros(K, np.int64); mkt = np.zeros(K, np.int64)
    lastc = np.full(K, np.nan); dist = np.empty(K)
    # ladder state
    lp2 = np.zeros(K)                       # leg-2 limit price (0: none resting)
    res2 = np.zeros(K)                      # cash locked in the resting leg 2
    a1 = np.zeros(K); e1 = np.zeros(K); tpr = np.zeros(K); leg2 = np.zeros(K); bud = np.zeros(K)
    lbar = np.full(K, -1, np.int64)         # hour of the latest fill (take-profit from lbar+tpwait)
    cash = 1.0; eq = np.empty(n); eq_prev = 1.0
    maxlog = 600000; lg = np.zeros((maxlog, 10)); nl = 0
    for t in range(n):
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
                fill = -1.0; ismkt = 0
                if o[t, k] <= lim:
                    fill = o[t, k] * (1.0 + slip[k]) * (1.0 + pen); ismkt = 1
                elif l[t, k] <= lim * (1.0 - eps):
                    fill = lim * (1.0 + pen)
                if fill > 0.0 and cash > 1e-9:
                    mkt[k] = ismkt if mtp_now else 0
                    a = min(per * smult[t, k], cash)
                    b1 = a * w
                    b2 = a - b1
                    cash -= a; units[k] = b1 * (1.0 - fee) / fill
                    entry[k] = fill; amt0[k] = b1; ebar[k] = t; got[k] = 0.0
                    tpp[k] = fill * (1.0 + tpa[t, k]); stage[k] = 0; peak[k] = fill
                    lbar[k] = t; a1[k] = b1; e1[k] = fill; tpr[k] = tpa[t, k]; leg2[k] = 0.0; bud[k] = a
                    lp2[k] = 0.0; res2[k] = 0.0
                    if b2 > 0.0:                                  # leg 2 rests from the hour start
                        lp2[k] = ref[t, k] * (1.0 - d2)
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
            if units[k] > 0.0 and (ebar[k] < t or mkt[k] == 1) and not np.isnan(o[t, k]):
                # leg 2 resting since an earlier hour (a resting limit: fills at its price when the low trades through)
                if ebar[k] < t and res2[k] > 0.0 and l[t, k] <= lp2[k] * (1.0 - eps):
                    b2 = res2[k]
                    f2 = lp2[k] * (1.0 + pen)
                    units[k] += b2 * (1.0 - fee) / f2; amt0[k] = a1[k] + b2
                    entry[k] = amt0[k] / (a1[k] / e1[k] + b2 / f2); tpp[k] = entry[k] * (1.0 + tpr[k])
                    leg2[k] = 1.0; lp2[k] = 0.0; res2[k] = 0.0; lbar[k] = t
                px = -1.0; why = 0.0
                if stage[k] == 0:
                    if ebar[k] == t:                     # market buy, take-profit placed at once (mtp_now)
                        if leg2[k] == 0.0 and h[t, k] >= tpp[k] * (1.0 + eps):     # not if leg 2 filled this hour
                            px = tpp[k]
                    elif t - lbar[k] < tpwait:
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
                        cash += res2[k]; res2[k] = 0.0; lp2[k] = 0.0      # first sale cancels an unfilled leg 2
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
                if px < 0.0 and t - ebar[k] >= hm[t] - mkt[k]:
                    px = c[t, k] * (1.0 - slip[k]); why = 1.0
                if px > 0.0:
                    proceeds = units[k] * px * (1.0 - fee); cash += proceeds; got[k] += proceeds
                    if nl < maxlog:
                        lg[nl, 0] = ebar[k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = why
                        lg[nl, 4] = got[k] / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]
                        lg[nl, 7] = leg2[k]; lg[nl, 8] = a1[k]; lg[nl, 9] = bud[k]; nl += 1
                    units[k] = 0.0; okbar[k] = t + cd
                    cash += res2[k]; res2[k] = 0.0; lp2[k] = 0.0      # cancel an unfilled leg 2
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


def lrun(P, idx, depth=L["depth"], N=L["N"], fmax=L["fmax"], smult=1.0, tp=L["tp"], part=1.0, tp2=0.0, trail=0.0,
         hm=L["hm"], cd=L["cd"], eps=E.EPS, fee=E.FEE, slip=E.SLIP_MAJOR, pen=0.0, tpwait=1, tprest=True,
         mtp_now=False, d2=0.0, w=1.0):
    """engine.run plus the split ladder (w < 1 needs d2 deeper than the trigger); same arguments and defaults.
    fmax caps the coin's total budget (both legs). Works on any panel P with o, h, l, cf, ref."""
    assert 0.0 < w <= 1.0, "w is the first leg's share of the coin's budget"
    assert w == 1.0 or float(d2) > float(np.max(depth)), "leg 2 must rest deeper than the trigger"
    O = P["o"].values; n, K = O.shape
    sl = np.full(K, slip) if np.ndim(slip) == 0 else np.asarray(slip, np.float64)
    eq, lg = lkernel(O, P["h"].values, P["l"].values, P["cf"].values, P["ref"].values,
                     E.full(n, K, depth), E.full(n, 0, N, np.int64), E.full(n, 0, fmax), E.full(n, K, smult),
                     E.full(n, K, tp), float(part), float(tp2), float(trail), E.full(n, 0, hm, np.int64), int(cd), eps,
                     fee, sl, pen, int(tpwait), bool(tprest), bool(mtp_now), float(d2), float(w))
    return pd.Series(eq, idx), pd.DataFrame(lg, columns=LOG)


# ------------------------------------------------------------------ code checks (no hypothesis results)
def _mk(o, h, l, c, ref):
    idx = pd.date_range("2021-01-01", periods=len(o), freq="1h")
    P = {f: pd.DataFrame({"A": np.asarray(v, float)}, index=idx) for f, v in
         (("o", o), ("h", h), ("l", l), ("c", c), ("cf", c), ("ref", ref))}
    return P, idx


def synthetic_check():
    """Hand-made one-coin paths: the capped ladder must fill, size and exit exactly as described."""
    fee, sl = E.FEE, E.SLIP_MAJOR
    ok = True
    ref = [100.0] * 6
    # 1. hour 0: low 86.8 fills leg 1 at 88; hour 1: low 81.0 fills leg 2 at 82; hour 2: high 87 >= avg x 1.015.
    #    One coin, cap 25%: budget = 25% of equity, 12.5% per leg (the cap binds: cash/nsel = 100%).
    o = [95, 87, 82.5, 84, 84, 84]; h = [95, 87, 87, 85, 85, 85]; l = [86.8, 81.0, 82, 83, 83, 83]
    c = [87, 82.5, 84, 84, 84, 84]
    P, idx = _mk(o, h, l, c, ref)
    e, lg = lrun(P, idx, N=1, w=0.5, fmax=0.25, d2=0.18)
    b = 0.125
    u = b * (1 - fee) / 88 + b * (1 - fee) / 82
    avg = 2 * b / (b / 88 + b / 82)
    want = u * avg * 1.015 * (1 - fee) / (2 * b) - 1
    eq_end = 1 - 2 * b + u * avg * 1.015 * (1 - fee)
    ok1 = (len(lg) == 1 and lg.leg2[0] == 1 and lg.xb[0] == 2 and abs(lg.ret[0] - want) < 1e-12
           and abs(lg.amt[0] - 0.25) < 1e-12 and abs(lg.bud[0] - 0.25) < 1e-12 and abs(e.iloc[-1] - eq_end) < 1e-12)
    print(f"  1 split, cap 25%: trades {len(lg)}, leg2 {lg.leg2.tolist()}, exit hour {lg.xb.tolist()}, budget "
          f"{lg.bud.tolist()}, ret {lg.ret.tolist()} (want {want}) -> {'ok' if ok1 else 'WRONG'}")
    ok &= ok1
    # 2. both legs in the fill hour (low 81 in hour 0); take-profit at hour 1's open (>= avg x 1.015)
    o3 = [95, 86.5, 86, 86, 86, 86]; h3 = [95, 87, 87, 87, 87, 87]; l3 = [81.0, 85, 85, 85, 85, 85]; c3 = [85] * 6
    P3, idx3 = _mk(o3, h3, l3, c3, ref)
    e, lg = lrun(P3, idx3, N=1, w=0.5, fmax=0.25, d2=0.18)
    want3 = u * 86.5 * (1 - fee) / (2 * b) - 1
    ok2 = len(lg) == 1 and lg.leg2[0] == 1 and lg.xb[0] == 1 and abs(lg.ret[0] - want3) < 1e-12
    print(f"  2 both legs in the fill hour: exit hour {lg.xb.tolist()}, ret {lg.ret.tolist()} (want {want3}) "
          f"-> {'ok' if ok2 else 'WRONG'}")
    ok &= ok2
    # 3. leg 2 never reached: leg 1 (w=2/3 of a 20% budget) sells at +1.5% in hour 2, the locked leg-2 cash comes back
    o4 = [95, 88.5, 89, 89, 89, 89]; h4 = [95, 89, 90, 90, 90, 90]; l4 = [86.8, 87, 88, 88, 88, 88]; c4 = [88] * 6
    P4, idx4 = _mk(o4, h4, l4, c4, ref)
    e, lg = lrun(P4, idx4, N=1, w=2 / 3, fmax=0.20, d2=0.18)
    b1 = 0.20 * (2 / 3)
    want4 = 1.015 * (1 - fee) ** 2 - 1
    eq4 = 1 - b1 + b1 * (1 - fee) / 88 * 88 * 1.015 * (1 - fee)
    ok3 = (len(lg) == 1 and lg.leg2[0] == 0 and lg.xb[0] == 2 and abs(lg.ret[0] - want4) < 1e-12
           and abs(lg.amt[0] - b1) < 1e-12 and abs(e.iloc[2] - eq4) < 1e-12 and abs(e.iloc[0] - (1 - b1 + b1 * (1 - fee))) < 1e-12)
    print(f"  3 leg 2 not reached: exit hour {lg.xb.tolist()}, amt {lg.amt.tolist()} (want {b1}), equity after "
          f"{e.iloc[2]:.12f} (want {eq4:.12f}) -> {'ok' if ok3 else 'WRONG'}")
    ok &= ok3
    # 4. time stop counts from leg 1; leg 2 fills at its limit 2 hours later; equity counts the locked leg-2 cash
    o5 = [95, 87, 84, 80, 80, 80]; h5 = [95, 87, 84, 81, 81, 81]; l5 = [86.8, 85, 83, 79.5, 79.5, 79.5]; c5 = [87, 85, 83, 80, 80, 80]
    P5, idx5 = _mk(o5, h5, l5, c5, ref)
    e, lg = lrun(P5, idx5, N=1, w=0.5, fmax=0.25, d2=0.18, hm=4)
    u5 = b * (1 - fee) / 88 + b * (1 - fee) / 82
    want5 = u5 * 80 * (1 - sl) * (1 - fee) / (2 * b) - 1
    eq_h1 = 1 - 2 * b + b + b * (1 - fee) / 88 * 85          # hour 1: leg 2 cash still locked, counted as cash
    ok4 = (len(lg) == 1 and lg.why[0] == 1 and lg.xb[0] == 4 and lg.leg2[0] == 1 and abs(lg.ret[0] - want5) < 1e-12
           and abs(e.iloc[1] - eq_h1) < 1e-12)
    print(f"  4 leg 2 at hour 3 + time stop from leg 1: exit hour {lg.xb.tolist()} why {lg.why.tolist()} ret "
          f"{lg.ret.tolist()} (want {want5}) -> {'ok' if ok4 else 'WRONG'}")
    ok &= ok4
    # 5. the cap binds with several coins: 2 coins, N=4, cap 20%, cash 100% -> per = min(20%, 50%) = 20% each
    idx = pd.date_range("2021-01-01", periods=4, freq="1h")
    P6 = {f: pd.DataFrame({"A": v, "B": v}, index=idx) for f, v in
          (("o", [95.0, 89, 89, 89]), ("h", [95.0, 90, 90, 90]), ("l", [86.8, 88, 88, 88]), ("c", [88.0] * 4),
           ("cf", [88.0] * 4), ("ref", [100.0] * 4))}
    e, lg = lrun(P6, idx, N=4, w=0.5, fmax=0.20, d2=0.18)
    ok5 = len(lg) == 2 and np.allclose(lg.bud.values, 0.2, atol=1e-12) and np.allclose(lg.a1.values, 0.1, atol=1e-12)
    print(f"  5 two coins, cap 20%: budgets {lg.bud.tolist()}, leg-1 money {lg.a1.tolist()} -> {'ok' if ok5 else 'WRONG'}")
    ok &= ok5
    # 6. mtp_now: market buy at hour 0's open (under the trigger); leg 2 not reached; take-profit fills in hour 0
    o7 = [87.0, 90, 90]; h7 = [90.0, 90, 90]; l7 = [86.0, 89, 89]; c7 = [89.0, 90, 90]
    P7, idx7 = _mk(o7, h7, l7, c7, [100.0] * 3)
    e, lg = lrun(P7, idx7, N=1, w=0.5, fmax=0.25, d2=0.18, mtp_now=True)
    f7 = 87 * (1 + sl)
    want7 = 1.015 * (1 - fee) ** 2 - 1
    ok6 = len(lg) == 1 and lg.xb[0] == 0 and lg.leg2[0] == 0 and abs(lg.ret[0] - want7) < 1e-12 and abs(lg.entry[0] - f7) < 1e-12
    print(f"  6 mtp_now market buy, take-profit in the buy hour: exit hour {lg.xb.tolist()} ret {lg.ret.tolist()} "
          f"(want {want7}) -> {'ok' if ok6 else 'WRONG'}")
    ok &= ok6
    # 7. mtp_now: market buy and leg 2 in the same hour -> no take-profit that hour (average placed next cycle)
    o8 = [87.0, 90, 90]; h8 = [90.0, 90, 90]; l8 = [81.0, 89, 89]; c8 = [85.0, 90, 90]
    P8, idx8 = _mk(o8, h8, l8, c8, [100.0] * 3)
    e, lg = lrun(P8, idx8, N=1, w=0.5, fmax=0.25, d2=0.18, mtp_now=True)
    ok7 = len(lg) == 1 and lg.xb[0] == 1 and lg.leg2[0] == 1
    print(f"  7 mtp_now, both legs in the buy hour: exit hour {lg.xb.tolist()} leg2 {lg.leg2.tolist()} -> {'ok' if ok7 else 'WRONG'}")
    ok &= ok7
    return ok


def prove_equivalence():
    """lrun with the ladder off (w=1) must give exactly engine.run's equity curve and trades."""
    ok = True
    for dead, pen, mtp in ((False, 0.0, False), (True, 0.0, False), (False, 0.002, False), (True, 0.002, False),
                           (False, 0.0, True), (True, 0.0, True)):
        syms, idx, P = S.market(dead)
        e0, l0 = _orig_run(P, idx, pen=pen, mtp_now=mtp)
        e1, l1 = lrun(P, idx, pen=pen, mtp_now=mtp)
        same = np.array_equal(e0.values, e1.values) and l0.equals(l1[E.LOG_COLS])
        ok &= same
        print(E.line(f"lrun, ladder off, dead={dead} pen={pen} mtp={mtp}", E.score(e1, l1, idx))
              + f"  identical to engine.run: {same}")
    # w=1 switches the ladder off whatever d2 is
    syms, idx, P = S.market(False)
    e0, l0 = _orig_run(P, idx)
    e2, l2 = lrun(P, idx, d2=0.999, w=1.0)
    same2 = np.array_equal(e0.values, e2.values)
    print(f"lrun with d2=99.9% and w=1 identical: {same2}")
    ok &= same2
    ok &= synthetic_check()
    if not ok:
        raise SystemExit("lkernel does not reproduce engine.run / the hand-made cases - stop")


# ------------------------------------------------------------------ the grid (fixed before any capped variant ran)
GRID = {}
for ww, wl in ((0.5, "50%"), (2 / 3, "67%")):
    for nn, cap in ((4, 0.25), (5, 0.20)):
        for dd in (0.18, 0.20):
            GRID[f"split {wl} at 12% / rest at {dd:.0%}, {nn} coins, cap {cap:.0%}"] = {"w": ww, "N": nn, "fmax": cap, "d2": dd}
assert len(GRID) == 8


def anatomy(name, spec, dead=False):
    """Exit reasons, leg-2 use and the largest coin budget, per period (printed only after the grid has run)."""
    syms, idx, P = S.market(dead)
    e, lg = lrun(P, idx, **spec)
    lg = lg.copy()
    for col in ("eb", "xb", "k"):
        lg[col] = lg[col].astype(int)
    eqe = np.where(lg.eb.values > 0, e.values[np.maximum(lg.eb.values - 1, 0)], 1.0)
    lg["bf"] = lg.bud.values / eqe
    lg["con"] = lg.ret.values * lg.amt.values / eqe
    per = np.where(idx[lg.xb.values] < E.DESIGN_END, "21-23", "24-26")
    for p in ("21-23", "24-26"):
        g = lg[per == p]
        parts = [f"{wn}: {int((g.why == cc).sum())} (sum {(g.ret[g.why == cc] * g.amt[g.why == cc]).sum():+.2f}, "
                 f"avg {g.ret[g.why == cc].mean() * 100:+.1f}%)" for cc, wn in ((0, "TP"), (1, "time stop")) if (g.why == cc).any()]
        l2 = g[g.leg2 == 1]
        print(f"  anatomy {name}{' +dead' if dead else ''} {p}: {len(g)} trades | " + " | ".join(parts) +
              f" | with leg 2: {len(l2)} (avg {l2.ret.mean() * 100 if len(l2) else np.nan:+.1f}%, time stops "
              f"{int((l2.why == 1).sum())}) | max budget {g.bf.max() * 100:.1f}% of equity | worst {g.ret.min() * 100:+.1f}%")
    if dead:
        for s in E.DEAD:
            g = lg[lg.k == syms.index(s)]
            print(f"    {s}: {len(g)} closed trades, sum contrib {g.con.sum() * 100:+.1f}% of equity, max budget "
                  f"{g.bf.max() * 100:.1f}%, worst {g.ret.min() * 100:+.1f}%")


if __name__ == "__main__":
    prove_equivalence()
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        sys.exit(0)
    E.run = lrun        # in memory only: study.study() now runs every variant (and the baseline) through lrun
    R, v = S.study("r3 ladder: split ladder, coin's total capped at 1/N (25% on 4 coins, 20% on 5)", GRID, "r3_ladder")
    S.save_verdict(v)
    # anatomy, printed after the grid has run (changes neither the grid, the pick nor the verdict)
    anatomy("LIVE SETTINGS", {})
    if v["pick"]:
        anatomy(v["pick"], GRID[v["pick"]])
        anatomy("LIVE SETTINGS", {}, dead=True)
        anatomy(v["pick"], GRID[v["pick"]], dead=True)
