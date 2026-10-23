"""m11 "stack": ONE account, ONE cash pot, three dip signals in one 1-minute engine (Binance spot, fully funded).

Signals (each exactly as published; only the way they share cash is searched):
  A  90% mode, hourly: at each hour's open, bid on the coin closest to (24h highest hourly close) x 0.86, crash guard
     (last hourly close >= 0.65 x highest close of the 168 closed hours), take-profit +2.5% from the next hourly check
     (an hour opening at/above it sells at that open), time stop 96h (hour close), 3h cooldown per coin after a sale.
  E  5-minute "panic wick" (m10 meanrev pick): at each 5-minute open, bid on the coin closest to (highest 5m close of
     the last 12 closed 5m bars) x 0.92, skipping coins whose last 5m close is >15% under the 24h max or >35% under the
     7-day max of 5m closes; take-profit +1.2% from the next 5m bar, time stop 72 bars (6h, bar close), 1h cooldown.
  F  1-minute flash dip (m10 scalp pick): every minute, bids on the `nordF` coins closest to (highest 1m close of the
     last 60 closed minutes) x 0.92, crash guard (minute open >= 0.65 x highest hourly close of the 168 closed hours),
     take-profit +2% from the minute after the fill, time stop 5760 minutes (96h), no cooldown; fills tested on
     1-minute candles with wicks trimmed to 0.5% beyond the candle body (wickF), as published.

Combination switches (the search):
  fs[s]    order size of signal s as a fraction of the account (always <= free cash, i.e. cash not held by resting bids)
  nordF    F bids per minute; nmax[s] most positions per signal; nslots most positions in total
  nearA    A places its hourly bid only if its closest coin opens at most nearA above the trigger (1e9 = always, the
           90% mode). A bid reserves its cash for the hour, so with nearA = 1e9 and fs[A] = 1, E and F never trade while
           flat; a finite nearA lends that idle cash to E/F in hours when no coin is near its A trigger.
  gEF      E/F place no new bid while any A-eligible coin opens within gEF above its A trigger (keep cash for A).
           -1 = off. When gated, resting E/F bids are cancelled.
  pre      0: A acts only through its hourly bid (90% mode). 1: A also watches every minute: a coin whose minute open is
           at/under its A trigger (and has no A bid) is bought at that open + slippage with free cash. 2: ... and if free
           cash is short of fs[A] x account, E/F positions in OTHER coins are sold at that minute's open (- slippage) first;
           an A limit fill that got less than its size because E/F held cash is topped up the same way at the next
           minute's open if the price is still at/under the A limit (once). 3: ... E/F positions in the same coin are
           sold too (and re-bought as an A position: full fees and slippage both ways, conservative).
  lots     0: one position per coin over all signals (a signal never bids on a coin another signal holds or bids on).
           1: different signals may hold the same coin as separate lots (separate take-profit orders).
Priority: inside a minute A (hour open) then A watch, then E (5m open), then F; resting bids reserve cash, so earlier
signals get cash first; resting fills are processed A, E, F.

Costs and rules: fee 0.1% a side; market buys and time-stop sells pay slippage slip[hour, coin] (0.02% for the 19 coins,
0.1% for LUNA/FTT/SRM in the stressed run, as m10_velocity_lib); limit buys and take-profits fill only when the price
trades eps = 0.1% through; pen / pens make every buy / sell fill worse (stress 0.002). mtp_now (the real bot after a
MARKET buy for A): take-profit at once, time stop counted from the hour's start. Cash from a sale is usable from the next
minute (pre-emption sells and buys in the same minute: spot settles at once). Every reference is computed from candles
closed before the decision (no lookahead). Collapse coins (lastb >= 0) are sold at their last hour's close.

With only A on, nslots = 1, nearA = 1e9, pre = 0 the kernel reproduces m10_velocity_lib.vrun(MODE90) on hourly candles
built from the same minutes; with only E on it reproduces m10_meanrev_lib.run(pick); with only F on it reproduces
m10_scalp_lib.srun(scalp(...)) (m11_stack_check.py).
Log columns: eb, xb (hour of entry / exit), k, why (0 tp, 1 time stop, 5 collapse cut, 6 sold to fund A, 9 open at end),
ret, amt, entry, eq0 (account value the order was sized on), sig (0 A, 1 E, 2 F), fm, xm (fill / exit minute).
"""
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402

SCR = "/tmp/claude-0/-home-user-halal-rider/ec562e35-ea9e-5a74-ab92-4ff22c5eeb77/scratchpad"
OUT = os.environ.get("PB_OUT", os.path.join(SCR, "pb_out_m11", "stack"))
os.makedirs(OUT, exist_ok=True)
XCACHE = os.path.join(SCR, "pb_out_x10", "velocity", "cache")      # 2021-26, 22 coins (H19 + LUNA FTT SRM)
HCACHE = os.path.join(SCR, "pb_out_m10", "scalp", "hcache")         # 2017-12 .. 2020-12, 16 coins
SYMS = E.H19 + list(E.DEAD)
T18, T21, T24, T25 = (pd.Timestamp(x) for x in ("2018-01-01", "2021-01-01", "2024-01-01", "2025-01-01"))
LOGC = ["eb", "xb", "k", "why", "ret", "amt", "entry", "eq0", "sig", "fm", "xm"]
MD = 365.25 / 12


# ------------------------------------------------------------------ kernel
@njit(cache=True)
def _log(lg, nl, s, k, why, t, i, proceeds, amt0, entry, eq0, ebh, fmn):
    if nl < lg.shape[0]:
        lg[nl, 0] = ebh[s, k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = why
        lg[nl, 4] = proceeds / amt0[s, k] - 1.0; lg[nl, 5] = amt0[s, k]; lg[nl, 6] = entry[s, k]
        lg[nl, 7] = eq0[s, k]; lg[nl, 8] = s; lg[nl, 9] = fmn[s, k]; lg[nl, 10] = i
    return nl + 1


@njit(cache=True)
def ck(o1, h1, l1, c1, ho, hcf, refA, gm, gref, limE, slip, lastb,
       on, fs, tp, nmax, nslots, hmA, hmE, hmF, cdA, cdE, cdF, depA, depF, W, gF, wickF, nordF,
       nearA, gEF, pre, lots, mtp_now, eps, fee, pen, pens, maxlog):
    nm, K = c1.shape
    nh = nm // 60
    units = np.zeros((3, K)); entry = np.zeros((3, K)); amt0 = np.zeros((3, K)); tpp = np.zeros((3, K))
    eq0 = np.zeros((3, K)); ebh = np.zeros((3, K), np.int64); fmn = np.zeros((3, K), np.int64)
    act = np.zeros((3, K), np.int64); mkt = np.zeros((3, K), np.int64)
    resv = np.zeros((3, K)); rlim = np.zeros((3, K)); rsum = 0.0
    okA = np.zeros(K, np.int64); okE = np.zeros(K, np.int64); okF = np.zeros(K, np.int64)
    alim = np.zeros(K); topped = np.zeros(K, np.int64)
    lastm = np.full(K, np.nan); dist = np.empty(K); order = np.zeros(K, np.int64)
    cap = W + 2
    dq = np.zeros((K, cap), np.int64); dh = np.zeros(K, np.int64); dn = np.zeros(K, np.int64)
    cash = 1.0; eqm = 1.0; eq_prev = 1.0
    eq = np.empty(nh); eqlo = np.empty(nh); inv = np.empty(nh)
    lg = np.zeros((maxlog, 11)); nl = 0
    lo_t = 1e300
    held = np.zeros(3, np.int64)
    for i in range(nm):
        t = i // 60; j = i - 60 * t; b = i // 5; jb = i - 5 * b
        if j == 0:
            lo_t = 1e300
        for s in range(3):
            held[s] = 0
            for k in range(K):
                if units[s, k] > 0.0:
                    held[s] += 1
        nheld = held[0] + held[1] + held[2]
        # ---------------- 1. A hourly cycle
        if j == 0:
            for k in range(K):
                if resv[0, k] > 0.0:
                    rsum -= resv[0, k]; resv[0, k] = 0.0
            if on[0] == 1 and held[0] < nmax[0] and nheld < nslots and cash - rsum > 1e-9:
                best = -1; bd = 1e9
                for k in range(K):
                    if units[0, k] > 0.0 or t < okA[k] or np.isnan(ho[t, k]) or np.isnan(refA[t, k]) or gm[t, k] <= 0.0:
                        continue
                    if lots == 0 and (units[1, k] > 0.0 or units[2, k] > 0.0):
                        continue
                    d = ho[t, k] / (refA[t, k] * (1.0 - depA)) - 1.0
                    if d < bd:
                        bd = d; best = k
                if best >= 0 and bd <= nearA:
                    k = best
                    lim = refA[t, k] * (1.0 - depA)
                    per = min(fs[0] * eq_prev, cash - rsum)
                    if lots == 0:                       # an E/F bid on this coin gives way to A
                        for s in range(1, 3):
                            if resv[s, k] > 0.0:
                                rsum -= resv[s, k]; resv[s, k] = 0.0
                        per = min(fs[0] * eq_prev, cash - rsum)
                    if ho[t, k] <= lim:
                        fill = o1[i, k] * (1.0 + slip[t, k]) * (1.0 + pen)
                        a = per
                        cash -= a; units[0, k] = a * (1.0 - fee) / fill; entry[0, k] = fill; amt0[0, k] = a
                        ebh[0, k] = t; fmn[0, k] = i; eq0[0, k] = eq_prev; tpp[0, k] = fill * (1.0 + tp[0])
                        mkt[0, k] = 1 if mtp_now == 1 else 0
                        act[0, k] = i if mtp_now == 1 else 60 * (t + 1)
                        alim[k] = lim; topped[k] = 0
                        held[0] += 1; nheld += 1
                    else:
                        resv[0, k] = per; rlim[0, k] = lim; rsum += per; eq0[0, k] = eq_prev
        # ---------------- 2. A watch / pre-emption
        if on[0] == 1 and pre >= 1:
            want = -1; kind = 0
            if held[0] < nmax[0]:
                bd = 1e9
                for k in range(K):
                    if units[0, k] > 0.0 or resv[0, k] > 0.0 or t < okA[k] or np.isnan(o1[i, k]) or np.isnan(refA[t, k]) \
                            or gm[t, k] <= 0.0:
                        continue
                    lim = refA[t, k] * (1.0 - depA)
                    if o1[i, k] > lim:
                        continue
                    if lots == 0 and pre < 3 and (units[1, k] > 0.0 or units[2, k] > 0.0):
                        continue
                    d = o1[i, k] / lim - 1.0
                    if d < bd:
                        bd = d; want = k
                kind = 1
            if want < 0 and pre >= 2:              # top-up of an under-funded A limit fill
                for k in range(K):
                    if units[0, k] > 0.0 and topped[k] == 0 and ebh[0, k] == t and fmn[0, k] < i \
                            and amt0[0, k] < 0.98 * fs[0] * eq0[0, k] and not np.isnan(o1[i, k]) and o1[i, k] <= alim[k]:
                        want = k; kind = 2
                        break
            if want >= 0:
                k = want
                target = fs[0] * eqm if kind == 1 else fs[0] * eq0[0, k] - amt0[0, k]
                if kind == 1:
                    for kk in range(K):                 # A's resting hourly bid (another coin) is cancelled
                        if resv[0, kk] > 0.0:
                            rsum -= resv[0, kk]; resv[0, kk] = 0.0
                if pre >= 2 and cash - rsum < 0.99 * target:
                    for s in range(1, 3):
                        for kk in range(K):
                            if resv[s, kk] > 0.0:
                                rsum -= resv[s, kk]; resv[s, kk] = 0.0
                    for pas in range(2):               # same coin first (pre 3), then the others
                        for s in range(1, 3):
                            for kk in range(K):
                                if units[s, kk] <= 0.0 or np.isnan(o1[i, kk]) or cash - rsum >= 0.99 * target:
                                    continue
                                if (pas == 0 and (kk != k or pre < 3)) or (pas == 1 and kk == k):
                                    continue
                                px = o1[i, kk] * (1.0 - slip[t, kk])
                                proceeds = units[s, kk] * px * (1.0 - pens) * (1.0 - fee); cash += proceeds
                                nl = _log(lg, nl, s, kk, 6.0, t, i, proceeds, amt0, entry, eq0, ebh, fmn)
                                units[s, kk] = 0.0
                                if s == 1:
                                    okE[kk] = b + cdE
                                else:
                                    okF[kk] = i + cdF
                                nheld -= 1; held[s] -= 1
                ok = True
                if lots == 0 and (units[1, k] > 0.0 or units[2, k] > 0.0):
                    ok = False
                if kind == 1 and nheld >= nslots and pre < 2:
                    ok = False
                a = min(target, cash - rsum)
                if ok and a > 1e-3 * eqm:
                    if lots == 0:
                        for s in range(1, 3):
                            if resv[s, k] > 0.0:
                                rsum -= resv[s, k]; resv[s, k] = 0.0
                    fill = o1[i, k] * (1.0 + slip[t, k]) * (1.0 + pen)
                    cash -= a
                    if kind == 1:
                        units[0, k] = a * (1.0 - fee) / fill; entry[0, k] = fill; amt0[0, k] = a
                        ebh[0, k] = t; fmn[0, k] = i; eq0[0, k] = eqm; tpp[0, k] = fill * (1.0 + tp[0])
                        mkt[0, k] = 1 if mtp_now == 1 else 0
                        act[0, k] = i + 1 if mtp_now == 1 else 60 * (t + 1)
                        alim[k] = refA[t, k] * (1.0 - depA); topped[k] = 1
                        held[0] += 1; nheld += 1
                    else:
                        units[0, k] += a * (1.0 - fee) / fill; amt0[0, k] += a
                        entry[0, k] = amt0[0, k] * (1.0 - fee) / units[0, k]
                        tpp[0, k] = entry[0, k] * (1.0 + tp[0]); topped[k] = 1
        # ---------------- 3. gate: an A trigger is close -> no new E/F bids
        gated = False
        if gEF >= 0.0 and on[0] == 1 and held[0] < nmax[0]:
            for k in range(K):
                if units[0, k] > 0.0 or t < okA[k] or np.isnan(o1[i, k]) or np.isnan(refA[t, k]) or gm[t, k] <= 0.0:
                    continue
                if o1[i, k] <= refA[t, k] * (1.0 - depA) * (1.0 + gEF):
                    gated = True
                    break
        if gated:
            for s in range(1, 3):
                for k in range(K):
                    if resv[s, k] > 0.0:
                        rsum -= resv[s, k]; resv[s, k] = 0.0
        # ---------------- 4. E cycle (5-minute open)
        if jb == 0 and on[1] == 1:
            for k in range(K):
                if resv[1, k] > 0.0:
                    rsum -= resv[1, k]; resv[1, k] = 0.0
            if not gated and held[1] < nmax[1] and nheld < nslots and cash - rsum > 1e-9:
                best = -1; bd = 1e9
                for k in range(K):
                    if units[1, k] > 0.0 or b < okE[k] or np.isnan(o1[i, k]) or not (limE[b, k] > 0.0):
                        continue
                    if lots == 0 and (units[0, k] > 0.0 or units[2, k] > 0.0 or resv[0, k] > 0.0):
                        continue
                    d = o1[i, k] / limE[b, k] - 1.0
                    if d < bd:
                        bd = d; best = k
                if best >= 0:
                    k = best; lim = limE[b, k]
                    per = min(fs[1] * eqm, cash - rsum)
                    if o1[i, k] <= lim:
                        fill = o1[i, k] * (1.0 + slip[t, k]) * (1.0 + pen)
                        cash -= per; units[1, k] = per * (1.0 - fee) / fill; entry[1, k] = fill; amt0[1, k] = per
                        ebh[1, k] = t; fmn[1, k] = i; eq0[1, k] = eqm; tpp[1, k] = fill * (1.0 + tp[1])
                        act[1, k] = 5 * (b + 1)
                        held[1] += 1; nheld += 1
                    else:
                        resv[1, k] = per; rlim[1, k] = lim; rsum += per; eq0[1, k] = eqm
        # ---------------- 5. F cycle (every minute)
        if on[2] == 1:
            for k in range(K):
                if resv[2, k] > 0.0:
                    rsum -= resv[2, k]; resv[2, k] = 0.0
            if not gated and held[2] < nmax[2] and nheld < nslots and cash - rsum > 1e-9:
                nel = 0
                for k in range(K):
                    dist[k] = 1e9
                    if units[2, k] > 0.0 or i < okF[k] or np.isnan(o1[i, k]) or dn[k] == 0:
                        continue
                    if lots == 0 and (units[0, k] > 0.0 or units[1, k] > 0.0 or resv[0, k] > 0.0 or resv[1, k] > 0.0):
                        continue
                    if gF > 0.0 and not np.isnan(gref[t, k]) and o1[i, k] < gref[t, k] * (1.0 - gF):
                        continue
                    dist[k] = o1[i, k] / (c1[dq[k, dh[k]], k] * (1.0 - depF)) - 1.0
                    nel += 1
                nord = min(nordF, nmax[2] - held[2], nslots - nheld)
                if nel <= nord:
                    nsel = 0
                    for k in range(K):
                        if dist[k] < 1e8:
                            order[nsel] = k; nsel += 1
                elif nord == 1:
                    order[0] = np.argmin(dist); nsel = 1
                else:
                    order = np.argsort(dist)
                    nsel = nord
                if nsel > 0:
                    per = min(fs[2] * eqm, (cash - rsum) / nsel)
                    for q in range(nsel):
                        k = order[q]
                        lim = c1[dq[k, dh[k]], k] * (1.0 - depF)
                        if o1[i, k] <= lim:
                            if cash - rsum > 1e-9:
                                a = min(per, cash - rsum)
                                fill = o1[i, k] * (1.0 + slip[t, k]) * (1.0 + pen)
                                cash -= a; units[2, k] = a * (1.0 - fee) / fill; entry[2, k] = fill; amt0[2, k] = a
                                ebh[2, k] = t; fmn[2, k] = i; eq0[2, k] = eqm; tpp[2, k] = fill * (1.0 + tp[2])
                                act[2, k] = i + 1
                                held[2] += 1; nheld += 1
                        else:
                            resv[2, k] = per; rlim[2, k] = lim; rsum += per; eq0[2, k] = eqm
        # ---------------- 6. resting bids that trade eps through their limit (A, E, F)
        if rsum > 0.0:
            for s in range(3):
                for k in range(K):
                    if resv[s, k] <= 0.0 or np.isnan(l1[i, k]):
                        continue
                    lo = l1[i, k]
                    if s == 2 and wickF > 0.0:
                        fl = min(o1[i, k], c1[i, k]) * (1.0 - wickF)
                        if fl > lo:
                            lo = fl
                    if lo <= rlim[s, k] * (1.0 - eps):
                        fill = rlim[s, k] * (1.0 + pen)
                        a = min(resv[s, k], cash)
                        rsum -= resv[s, k]; resv[s, k] = 0.0
                        if a > 1e-9:
                            cash -= a; units[s, k] = a * (1.0 - fee) / fill; entry[s, k] = fill; amt0[s, k] = a
                            ebh[s, k] = t; fmn[s, k] = i; tpp[s, k] = fill * (1.0 + tp[s]); mkt[s, k] = 0
                            if s == 0:
                                act[0, k] = 60 * (t + 1); alim[k] = rlim[0, k]; topped[k] = 0
                            elif s == 1:
                                act[1, k] = 5 * (b + 1)
                            else:
                                act[2, k] = i + 1
        # ---------------- 7. exits
        for s in range(3):
            for k in range(K):
                if units[s, k] <= 0.0 or np.isnan(o1[i, k]):
                    continue
                px = -1.0; why = 0.0
                if s == 0:
                    if i >= act[0, k]:
                        if j == 0 and ebh[0, k] < t and o1[i, k] >= tpp[0, k]:
                            px = o1[i, k]
                        elif h1[i, k] >= tpp[0, k] * (1.0 + eps):
                            px = tpp[0, k]
                    if px < 0.0 and j == 59 and (ebh[0, k] < t or mkt[0, k] == 1) and t - ebh[0, k] + mkt[0, k] >= hmA:
                        px = hcf[t, k] * (1.0 - slip[t, k]); why = 1.0
                elif s == 1:
                    if i >= act[1, k]:
                        T = tpp[1, k]
                        if jb == 0 and o1[i, k] >= T:
                            px = o1[i, k] if i == act[1, k] else T
                        elif h1[i, k] >= T * (1.0 + eps):
                            px = T
                    if px < 0.0 and jb == 4 and b - fmn[1, k] // 5 >= hmE:
                        px = c1[i, k] * (1.0 - slip[t, k]); why = 1.0
                else:
                    if i >= act[2, k]:
                        hi = h1[i, k]
                        if wickF > 0.0:
                            fh = max(o1[i, k], c1[i, k]) * (1.0 + wickF)
                            if fh < hi:
                                hi = fh
                        if hi >= tpp[2, k] * (1.0 + eps):
                            px = tpp[2, k]
                    if px < 0.0 and not np.isnan(c1[i, k]) and i - fmn[2, k] >= hmF:
                        px = c1[i, k] * (1.0 - slip[t, k]); why = 1.0
                if px > 0.0:
                    proceeds = units[s, k] * px * (1.0 - pens) * (1.0 - fee); cash += proceeds
                    nl = _log(lg, nl, s, k, why, t, i, proceeds, amt0, entry, eq0, ebh, fmn)
                    units[s, k] = 0.0
                    if s == 0:
                        okA[k] = t + cdA
                    elif s == 1:
                        okE[k] = b + cdE
                    else:
                        okF[k] = i + cdF
        # ---------------- 8. collapse coins: sold at their last hour's close
        if j == 59:
            for k in range(K):
                if lastb[k] == t:
                    for s in range(3):
                        if resv[s, k] > 0.0:
                            rsum -= resv[s, k]; resv[s, k] = 0.0
                        if units[s, k] > 0.0:
                            proceeds = units[s, k] * hcf[t, k] * (1.0 - slip[t, k]) * (1.0 - pens) * (1.0 - fee)
                            cash += proceeds
                            nl = _log(lg, nl, s, k, 5.0, t, i, proceeds, amt0, entry, eq0, ebh, fmn)
                            units[s, k] = 0.0
                            okA[k] = t + cdA; okE[k] = b + cdE; okF[k] = i + cdF
        # ---------------- 9. F reference deque, minute mark, hourly mark
        for k in range(K):
            x = c1[i, k]
            if np.isnan(x):
                continue
            lastm[k] = x
            if on[2] == 1:
                while dn[k] > 0 and c1[dq[k, (dh[k] + dn[k] - 1) % cap], k] <= x:
                    dn[k] -= 1
                dq[k, (dh[k] + dn[k]) % cap] = i
                dn[k] += 1
                while dn[k] > 0 and dq[k, dh[k]] <= i - W:
                    dh[k] = (dh[k] + 1) % cap; dn[k] -= 1
        v = cash; w = 0.0
        for s in range(3):
            for k in range(K):
                if units[s, k] > 0.0:
                    x = lastm[k] if not np.isnan(lastm[k]) else entry[s, k]
                    v += units[s, k] * x; w += units[s, k] * x
        eqm = v
        if v < lo_t:
            lo_t = v
        if j == 59:
            eq[t] = v; eq_prev = v; inv[t] = w / v if v > 0.0 else 0.0; eqlo[t] = lo_t
    for s in range(3):
        for k in range(K):
            if units[s, k] > 0.0:
                proceeds = units[s, k] * lastm[k] * (1.0 - slip[nh - 1, k]) * (1.0 - pens) * (1.0 - fee)
                nl = _log(lg, nl, s, k, 9.0, nh - 1, nm - 1, proceeds, amt0, entry, eq0, ebh, fmn)
    return eq, lg[:min(nl, maxlog)].copy(), inv, eqlo, nl


# ------------------------------------------------------------------ data
def _hourly(Hd, syms):
    """Hourly inputs known at each hour's open, from hourly candles built from the minutes."""
    hc = pd.DataFrame(Hd["c"], columns=syms)
    cf = hc.ffill()
    refA = hc.rolling(24, min_periods=1).max().shift(1)
    hi = cf.rolling(168, min_periods=1).max()
    gm = ((cf / hi - 1) >= -0.35).shift(1, fill_value=True).astype(float)
    gref = hc.rolling(168, min_periods=1).max().shift(1)
    return {"ho": np.ascontiguousarray(Hd["o"]), "hcf": np.ascontiguousarray(cf.values),
            "refA": np.ascontiguousarray(refA.values), "gm": np.ascontiguousarray(gm.values),
            "gref": np.ascontiguousarray(gref.values)}


def _lim5(c1):
    """E's bid on the 5-minute grid (m10_meanrev_final.pick): NaN when the coin is filtered out."""
    c5 = pd.DataFrame(np.asarray(c1[4::5], dtype=np.float64))
    cf = c5.ffill()
    sh = lambda x: x.shift(1)   # noqa: E731
    cfs = sh(cf)
    ok = (cfs >= sh(cf.rolling(2016, min_periods=1).max()) * 0.65) & (cfs >= sh(cf.rolling(288, min_periods=1).max()) * 0.85)
    lim = sh(cf.rolling(12, min_periods=1).max()) * 0.92
    return np.ascontiguousarray(np.where(ok.values, lim.values, np.nan))


_D = {}


def data(which="main", K=19):
    """Minute arrays (memory-mapped float32 views), hourly arrays, E's 5m bid, hidx, syms, lastb for the run.
    which 'main' (2021-26; K 19, or 22 with LUNA/FTT/SRM) or 'hold' (2018-20 holdout, the 16 coins that existed)."""
    key = (which, K)
    if key in _D:
        return _D[key]
    src = XCACHE if which == "main" else HCACHE
    M = {f: np.load(os.path.join(src, f"m_{f}.npy"), mmap_mode="r") for f in ("o", "h", "l", "c")}
    hidx = pd.DatetimeIndex(pd.to_datetime(np.load(os.path.join(src, "hidx.npy")), unit="s"))
    if which == "main":
        syms = SYMS[:K]
        M = {f: v[:, :K] for f, v in M.items()}
        Hd = {f: np.load(os.path.join(src, f"h_{f}.npy"))[:, :K] for f in ("o", "h", "l", "c")}
    else:
        syms = list(pd.read_csv(os.path.join(src, "syms.csv")).iloc[:, 0])
        nh = len(hidx)
        Hd = {"o": np.asarray(M["o"][::60], np.float64), "c": np.asarray(M["c"][59::60], np.float64),
              "h": np.fmax.reduce(np.asarray(M["h"]).reshape(nh, 60, -1), axis=1).astype(np.float64),
              "l": np.fmin.reduce(np.asarray(M["l"]).reshape(nh, 60, -1), axis=1).astype(np.float64)}
    D = dict(M=M, hidx=hidx, syms=syms, Hd=Hd, **_hourly(Hd, syms), limE=_lim5(M["c"]))
    lastb = np.full(len(syms), -1, np.int64)
    for j, s in enumerate(syms):
        if s in E.DEAD:
            v = np.flatnonzero(~np.isnan(Hd["o"][:, j]))
            lastb[j] = int(v[-1])
    D["lastb"] = lastb
    sl = np.array([E.SLIP_MAJOR if s in E.H19 else E.SLIP_OTHER for s in syms])
    D["slip"] = np.ascontiguousarray(np.tile(sl, (len(hidx), 1)))
    _D[key] = D
    return D


# ------------------------------------------------------------------ designs
SIG_A = dict(depA=0.14, tpA=0.025, hmA=96, cdA=3)
SIG_E = dict(tpE=0.012, hmE=72, cdE=12)
SIG_F = dict(depF=0.08, tpF=0.02, hmF=5760, cdF=0, W=60, gF=0.35, wickF=0.005)
DEF = dict(A=1, E=0, F=0, fsA=1.0, fsE=1.0, fsF=0.25, nmaxA=1, nmaxE=1, nmaxF=99, nslots=1, nordF=4, nearA=1e9,
           gEF=-1.0, pre=0, lots=0, mtp_now=0, cd_add=0, pen=0.0, pens=0.0, **SIG_A, **SIG_E, **SIG_F)
MODE90 = dict(DEF)


def run(D, design, start=None, end=None, slip=None, maxlog=400000):
    """Run a design on hours [start, end) of D. cd_add (hours) is added to A's and E's cooldowns (F's only when > 0).
    Returns (equity Series, log DataFrame, invested Series, eqlo Series)."""
    p = dict(DEF); p.update(design)
    bad = set(design) - set(DEF)
    assert not bad, f"unknown settings {bad}"
    hidx = D["hidx"]
    h0 = 0 if start is None else int(hidx.searchsorted(pd.Timestamp(start)))
    h1 = len(hidx) if end is None else int(hidx.searchsorted(pd.Timestamp(end)))
    M = D["M"]
    m = lambda f: M[f][60 * h0:60 * h1]   # noqa: E731
    hs = lambda x: x[h0:h1]               # noqa: E731
    sl = D["slip"] if slip is None else slip
    cda = int(p["cd_add"])
    eq, lg, inv, eqlo, nl = ck(m("o"), m("h"), m("l"), m("c"), hs(D["ho"]), hs(D["hcf"]), hs(D["refA"]), hs(D["gm"]),
                               hs(D["gref"]), D["limE"][12 * h0:12 * h1], np.ascontiguousarray(sl[h0:h1]),
                               np.where(D["lastb"] >= 0, D["lastb"] - h0, -1).astype(np.int64),
                               np.array([p["A"], p["E"], p["F"]], np.int64),
                               np.array([p["fsA"], p["fsE"], p["fsF"]]), np.array([p["tpA"], p["tpE"], p["tpF"]]),
                               np.array([p["nmaxA"], p["nmaxE"], p["nmaxF"]], np.int64), int(p["nslots"]),
                               int(p["hmA"]), int(p["hmE"]), int(p["hmF"]), max(0, int(p["cdA"]) + cda),
                               max(0, int(p["cdE"]) + 12 * cda), max(0, int(p["cdF"]) + 60 * max(cda, 0)),
                               float(p["depA"]), float(p["depF"]), int(p["W"]), float(p["gF"]), float(p["wickF"]),
                               int(p["nordF"]), float(p["nearA"]), float(p["gEF"]), int(p["pre"]), int(p["lots"]),
                               int(p["mtp_now"]), E.EPS, E.FEE, float(p["pen"]), float(p["pens"]), int(maxlog))
    assert nl <= maxlog, "trade log overflow"
    hx = hidx[h0:h1]
    return pd.Series(eq, hx), pd.DataFrame(lg, columns=LOGC), pd.Series(inv, hx), pd.Series(eqlo, hx)


def worse(d):
    return dict(d, pen=0.002, pens=0.002)


def timing(d):
    """The real bot's timing (m10_velocity_lib.stress_kw without the costs): take-profit at once after an A market buy,
    cooldowns counted from the next hourly check."""
    return dict(d, mtp_now=1, cd_add=int(d.get("cd_add", 0)) + 1)


def stressed(d):
    return worse(timing(d))


# ------------------------------------------------------------------ metrics (m10_velocity_lib conventions)
def stats(e, lg, inv, eqlo, a, b):
    """Window [a, b): %/month compounded from the equity just before a, multiple, worst drop on hourly closes (and on
    1-minute marks), trades bought in the window, win rate (open positions marked at the end count), trades/month,
    worst trade as % of the account it was sized on, per-signal trades and win rates, share of hours with no position."""
    idx = e.index
    i0 = int(np.searchsorted(idx.values, np.datetime64(a)))
    i1 = len(e) if b is None else int(np.searchsorted(idx.values, np.datetime64(b)))
    start = e.values[i0 - 1] if i0 > 0 else 1.0
    x = np.concatenate([[1.0], e.values[i0:i1] / start])
    months = ((idx[i1 - 1] + pd.Timedelta(hours=1)) - idx[i0]).total_seconds() / 86400 / MD
    mult = float(x[-1])
    pk = np.maximum.accumulate(x)
    lo = eqlo.values[i0:i1] / start
    r = dict(mo=mult ** (1 / months) - 1 if mult > 0 else -1.0, mult=mult, dd=float((x / pk - 1).min()),
             dd1=float(min((lo / pk[:-1] - 1).min(), (x / pk - 1).min())), months=months)
    tb = idx[lg.eb.astype(int).values] if len(lg) else pd.DatetimeIndex([])
    sel = np.asarray((tb >= a) & ((tb < b) if b is not None else True), bool)
    g = lg[sel]
    acct = (g.ret * g.amt / g.eq0).values
    r.update(n=len(g), tpm=len(g) / months, wr=float((g.ret > 0).mean()) if len(g) else np.nan,
             wt=float(acct.min()) if len(g) else np.nan, avg=float(g.ret.mean()) if len(g) else np.nan,
             flat=float((inv.values[i0:i1] <= 1e-12).mean()), npre=int((g.why == 6).sum()))
    for s, nm in enumerate("AEF"):
        gs = g[g.sig == s]
        r[f"n{nm}"] = len(gs)
        r[f"wr{nm}"] = float((gs.ret > 0).mean()) if len(gs) else np.nan
    return r


def fmt(tag, r):
    return (f"{tag:28s} win {r['wr']*100:5.1f}% | {r['mo']*100:+6.2f}%/mo | drop {r['dd']*100:5.1f}% (1-min "
            f"{r['dd1']*100:5.1f}%) | worst trade {r['wt']*100:+6.1f}% acct | {r['tpm']:5.1f} tr/mo | "
            f"A {r['nA']} {r['wrA']*100:4.1f}% E {r['nE']} {r['wrE']*100:4.1f}% F {r['nF']} {r['wrF']*100:4.1f}% | "
            f"sold-for-A {r['npre']} | flat {r['flat']*100:4.1f}%")
