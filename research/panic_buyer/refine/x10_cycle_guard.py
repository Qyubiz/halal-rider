"""Lever "cycle", phase-2 kernel: ckernel2 (fast trigger + optional slow trigger) plus three collapse guards.

Why: phase 1 showed the fast bot re-buys a collapsing coin again and again (LUNA May 2022: -24%, -81%, -98% of the
account in three consecutive all-in trades). All guards use only information known at the check:
  lcd   loss cooldown: after an exit at a loss the coin is blocked for lcd bars (instead of cd). 0 = off.
  stop  emergency exit: at a bar close at or below entry x (1 - stop) the position is sold at that close
        (market, - slippage). Checked only if the take-profit did not fill in that bar. 0 = off.
  skip  crash skip: no bid on a coin whose open is below ref3 x (1 - skip), ref3 = highest close of the last
        7 days (known at the check). 0 = off.
  lastb (per coin, -1 = off): the last bar with data of a collapse coin cut from the panel (LUNA/FTT/SRM). A position
        still held there is sold at that bar's close (- slippage), even if bought in that bar. Without it a coin bought
        in its last data hour stays frozen at that price forever; with 1-2 coins x 50-100% that freezes the whole
        account (seen in phase 2: FTT bought at its Sep 2023 reopen, the panel's cut). The coins kept trading in reality.
With lcd = stop = skip = 0 and lastb = -1 the kernel is ckernel2 exactly (main() proves it).
"""
import os
import sys

import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402
import x10_cycle_ext as X  # noqa: E402
import x10_cycle_lib as L  # noqa: E402


@njit(cache=True)
def ckernel3(o, h, l, c, ref, depth, ref2, depth2, ref3, skip, lcd, stop, nord, fmax, tp, part, tp2, trail, hm, cd,
             eps, fee, slip, pen, pens, tpwait, tprest, mtp_now, lastb):
    n, K = c.shape
    units = np.zeros(K); entry = np.zeros(K); amt0 = np.zeros(K); got = np.zeros(K); tpp = np.zeros(K)
    stage = np.zeros(K, np.int64); sbar = np.zeros(K, np.int64); peak = np.zeros(K); eq0 = np.zeros(K)
    ebar = np.full(K, -1, np.int64); okbar = np.zeros(K, np.int64); mkt = np.zeros(K, np.int64)
    lastc = np.full(K, np.nan); dist = np.empty(K); lims = np.empty(K)
    cash = 1.0; eq = np.empty(n); eq_prev = 1.0
    maxlog = 2000000; lg = np.zeros((maxlog, 8)); nl = 0
    for t in range(n):
        anyfill = False
        for k in range(K):
            dist[k] = 1e9
            if units[k] > 0.0 or t < okbar[k] or np.isnan(o[t, k]) or np.isnan(ref[t, k]):
                continue
            if skip > 0.0 and not np.isnan(ref3[t, k]) and o[t, k] < ref3[t, k] * (1.0 - skip):
                continue
            lim = ref[t, k] * (1.0 - depth)
            if depth2 > 0.0 and not np.isnan(ref2[t, k]):
                lim2 = ref2[t, k] * (1.0 - depth2)
                if lim2 > lim:
                    lim = lim2
            lims[k] = lim
            dist[k] = o[t, k] / lim - 1.0
            if o[t, k] <= lim or l[t, k] <= lim * (1.0 - eps):
                anyfill = True
        nsel = 0
        if anyfill and cash > 1e-9:
            order = np.argsort(dist)
            for q in range(K):
                if dist[order[q]] < 1e8 and nsel < nord:
                    nsel += 1
        if nsel > 0 and cash > 1e-9:
            per = min(fmax * eq_prev, cash / nsel)
            for q in range(nsel):
                k = order[q]
                lim = lims[k]
                fill = -1.0; ismkt = 0
                if o[t, k] <= lim:
                    fill = o[t, k] * (1.0 + slip[k]) * (1.0 + pen); ismkt = 1
                elif l[t, k] <= lim * (1.0 - eps):
                    fill = lim * (1.0 + pen)
                if fill > 0.0 and cash > 1e-9:
                    mkt[k] = ismkt if mtp_now else 0
                    a = min(per * 1.0, cash)
                    cash -= a; units[k] = a * (1.0 - fee) / fill
                    entry[k] = fill; amt0[k] = a; ebar[k] = t; got[k] = 0.0; eq0[k] = eq_prev
                    tpp[k] = fill * (1.0 + tp); stage[k] = 0; peak[k] = fill
        for k in range(K):
            if units[k] > 0.0 and (ebar[k] < t or mkt[k] == 1) and not np.isnan(o[t, k]):
                px = -1.0; why = 0.0
                if stage[k] == 0:
                    if ebar[k] == t:
                        if h[t, k] >= tpp[k] * (1.0 + eps):
                            px = tpp[k]
                    elif t - ebar[k] < tpwait:
                        pass
                    elif o[t, k] >= tpp[k]:
                        px = o[t, k]
                    elif tprest and h[t, k] >= tpp[k] * (1.0 + eps):
                        px = tpp[k]
                    if px > 0.0 and part < 1.0:
                        q = units[k] * part
                        p1 = q * px * (1.0 - pens) * (1.0 - fee)
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
                if px < 0.0 and t - ebar[k] >= hm - mkt[k]:
                    px = c[t, k] * (1.0 - slip[k]); why = 1.0
                if px < 0.0 and stop > 0.0 and ebar[k] < t and c[t, k] <= entry[k] * (1.0 - stop):
                    px = c[t, k] * (1.0 - slip[k]); why = 4.0
                if px > 0.0:
                    proceeds = units[k] * px * (1.0 - pens) * (1.0 - fee); cash += proceeds; got[k] += proceeds
                    if nl < maxlog:
                        lg[nl, 0] = ebar[k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = why
                        lg[nl, 4] = got[k] / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]
                        lg[nl, 7] = eq0[k]; nl += 1
                    units[k] = 0.0; okbar[k] = t + cd
                    if lcd > 0 and got[k] < amt0[k]:
                        okbar[k] = t + max(cd, lcd)
        for k in range(K):
            if units[k] > 0.0 and lastb[k] == t:
                proceeds = units[k] * c[t, k] * (1.0 - slip[k]) * (1.0 - pens) * (1.0 - fee)
                cash += proceeds; got[k] += proceeds
                if nl < maxlog:
                    lg[nl, 0] = ebar[k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = 5.0
                    lg[nl, 4] = got[k] / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]
                    lg[nl, 7] = eq0[k]; nl += 1
                units[k] = 0.0; okbar[k] = t + cd
        v = cash
        for k in range(K):
            if not np.isnan(c[t, k]):
                lastc[k] = c[t, k]
            if units[k] > 0.0:
                v += units[k] * lastc[k]
        eq[t] = v; eq_prev = v
    return eq, lg[:nl].copy()


def crun3(B, ref, cf, depth, N, fmax, tp, hm, cd, ref2=None, depth2=0.0, ref3=None, skip=0.0, lcd=0, stop=0.0,
          part=1.0, tp2=0.0, trail=0.0, eps=E.EPS, fee=E.FEE, slip=None, pen=0.0, pens=0.0, tpwait=1, tprest=True,
          mtp_now=False, liquidate=False):
    sl = L.slip_vec(B["syms"]) if slip is None else (np.full(len(B["syms"]), slip) if np.ndim(slip) == 0
                                                     else np.asarray(slip, np.float64))
    ref2 = ref if ref2 is None else ref2
    ref3 = ref if ref3 is None else ref3
    lastb = last_bars(B) if liquidate else np.full(len(B["syms"]), -1, np.int64)
    return ckernel3(B["o"], B["h"], B["l"], cf, ref, float(depth), ref2, float(depth2), ref3, float(skip), int(lcd),
                    float(stop), int(N), float(fmax), float(tp), float(part), float(tp2), float(trail), int(hm),
                    int(cd), float(eps), float(fee), sl, float(pen), float(pens), int(tpwait), bool(tprest),
                    bool(mtp_now), lastb)


def last_bars(B):
    """Last bar with data for the collapse coins (LUNA/FTT/SRM), -1 for the others."""
    out = np.full(len(B["syms"]), -1, np.int64)
    for j, s in enumerate(B["syms"]):
        if s in E.DEAD:
            v = np.flatnonzero(~np.isnan(B["o"][:, j]))
            out[j] = int(v[-1])
    return out


def main():
    ok_all = True
    cases = [dict(depth=0.06, N=1, fmax=1.0, tp=0.01, hm=12, cd=0, W=1),
             dict(depth=0.08, N=1, fmax=1.0, tp=0.03, hm=12, cd=0, W=3, depth2=0.12),
             dict(depth=0.08, N=2, fmax=0.5, tp=0.03, hm=6, cd=2, W=2, part=0.0, trail=0.01, depth2=0.12),
             dict(depth=0.10, N=3, fmax=1 / 3, tp=0.02, hm=6, cd=0, W=6, tpwait=4, pen=0.002, pens=0.002)]
    for C in (5, 15):
        for dead in (False, True):
            B = L.load_bars(C, dead); cf = L.ffill_close(B); bph = 60 // C
            r24 = L.make_ref(B, 24 * bph); r168 = L.make_ref(B, 168 * bph)
            for kw in cases:
                kw = dict(kw); W = kw.pop("W")
                ref = L.make_ref(B, W * bph)
                for k in ("hm", "cd"):
                    kw[k] = kw[k] * bph
                e1, l1 = X.crun2(B, ref, cf, ref2=r24, **kw)
                e2, l2 = crun3(B, ref, cf, ref2=r24, ref3=r168, skip=0.0, lcd=0, stop=0.0, **kw)
                ok = np.array_equal(e1, e2) and np.array_equal(l1, l2)
                if not dead:            # no collapse coin: liquidation switch on must change nothing
                    e3, l3 = crun3(B, ref, cf, ref2=r24, ref3=r168, liquidate=True, **kw)
                    ok = ok and np.array_equal(e1, e3) and np.array_equal(l1, l3)
                else:                   # with collapse coins: report what the liquidation changes
                    e3, l3 = crun3(B, ref, cf, ref2=r24, ref3=r168, liquidate=True, **kw)
                    print(f"      liquidation on: x{e3[-1]:.4f}, {int((l3[:, 3] == 5).sum())} forced sales")
                ok_all &= ok
                print(f"{C}m{' +dead' if dead else ''} {kw} W{W}: x{e1[-1]:.4f} vs x{e2[-1]:.4f} "
                      f"trades {len(l1)}/{len(l2)} {'IDENTICAL' if ok else 'DIFFERENT'}")
            del B, cf, r24, r168
    print("ALL CHECKS PASS" if ok_all else "SOME CHECK FAILED")


if __name__ == "__main__":
    main()
