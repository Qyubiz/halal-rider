"""Round 2, lens "realism": engine.mkernel copied as rkernel with switches for stricter / later fills.
All switches off (beps = seps = 0.001, bmin = 0, mpx = 0, lat = 0, mdelay = 0, mtsh = 0) it is engine.mkernel with
delay = 0, i.e. the live bot on 1-minute candles (proved in r2_realism.py: identical equity curve and trade log).

Switches (nothing here changes a signal; only how/when an order the bot already decided on gets filled):
  beps    a resting limit BUY fills only when a minute's low trades beps through the limit (live 0.1%)
  bmin    ... and the price has been at/below the limit for bmin consecutive minutes of that hour (0/1 = off)
  bmode   what "at/below" means for bmin: 0 = the minute's low <= limit (it traded there), 1 = its close <= limit
  seps    the take-profit SELL fills when a minute's high trades seps through it (live 0.1%)
  mpx     market buys at the hour open (coin already under its trigger) fill at: 0 the hour's open (live engine),
          1 the close of the hour's first minute, 2 the high of the first minute (worst price in it)
  lat     orders the bot places at the hourly cycle exist only from minute `lat` of the hour: a limit buy whose
          price is already through the limit then fills at that minute's open (taker, + slippage, never above the
          limit); a take-profit placed at the cycle sells at that minute's open if already above target; a time
          stop sells at the close of minute lat-1 of the next hour instead of this hour's close
  mdelay  >0: the take-profit of a MARKET buy is placed mdelay minutes after it (the live bot places it at once;
          the engine only at the next hour). Limit-buy take-profits stay at the next hourly cycle
  mtsh    the time stop of a market buy comes mtsh hours earlier (the live bot counts 48h from the cycle at the
          start of the buy hour, the engine from the end of it)
Extra log columns: fm (fill minute), lim (limit price), mkt (0 resting limit, 1 market at the open, 2 marketable
limit at placement because of lat), xo (1 = the take-profit sold at an open above target).
"""
import numpy as np
import pandas as pd
from numba import njit

import engine as E

RLOG = E.LOG_COLS + ["xm", "fm", "lim", "mkt", "xo"]


@njit(cache=False)
def rkernel(o1, h1, l1, c1, ref, depth, nord, fmax, tpa, hm, cd, fee, slip, pen, delay,
            beps, bmin, bmode, seps, mpx, lat, mdelay, mtsh):
    nm, K = c1.shape
    nh = nm // 60
    units = np.zeros(K); entry = np.zeros(K); amt0 = np.zeros(K); tpp = np.zeros(K)
    ebar = np.full(K, -1, np.int64); okbar = np.zeros(K, np.int64); act = np.zeros(K, np.int64)
    lim0 = np.zeros(K); fm0 = np.zeros(K, np.int64); mk0 = np.zeros(K, np.int64)
    lastc = np.full(K, np.nan); dist = np.empty(K)
    cash = 1.0; eq = np.empty(nh); eq_prev = 1.0
    maxlog = 600000; lg = np.zeros((maxlog, 12)); nl = 0
    for t in range(nh):
        m0 = 60 * t
        for k in range(K):
            dist[k] = 1e9
            if units[k] > 0.0 or t < okbar[k] or np.isnan(o1[m0, k]) or np.isnan(ref[t, k]):
                continue
            dist[k] = o1[m0, k] / (ref[t, k] * (1.0 - depth[t, k])) - 1.0
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
                fill = -1.0; fm = -1; mkt = 0
                if o1[m0, k] <= lim:
                    px0 = o1[m0, k]
                    if mpx == 1:
                        px0 = c1[m0, k]
                    elif mpx == 2:
                        px0 = h1[m0, k]
                    fill = px0 * (1.0 + slip[k]) * (1.0 + pen); fm = m0; mkt = 1
                else:
                    ms = m0 + lat
                    if lat > 0 and o1[ms, k] <= lim:
                        fill = min(lim, o1[ms, k] * (1.0 + slip[k])) * (1.0 + pen); fm = ms; mkt = 2
                    else:
                        run = 0; thr = False
                        for i in range(ms, m0 + 60):
                            if l1[i, k] <= lim * (1.0 - beps):
                                thr = True
                            if bmin > 1:
                                x = l1[i, k] if bmode == 0 else c1[i, k]
                                if x <= lim:
                                    run += 1
                                else:
                                    run = 0
                                if thr and run >= bmin:
                                    fill = lim * (1.0 + pen); fm = i
                                    break
                            elif thr:
                                fill = lim * (1.0 + pen); fm = i
                                break
                if fill > 0.0 and cash > 1e-9:
                    a = min(per, cash)
                    cash -= a; units[k] = a * (1.0 - fee) / fill
                    entry[k] = fill; amt0[k] = a; ebar[k] = t; tpp[k] = fill * (1.0 + tpa[t, k])
                    if mkt == 1 and mdelay > 0:
                        act[k] = fm + mdelay
                    elif delay > 0:
                        act[k] = fm + delay
                    else:
                        act[k] = m0 + 60 + lat
                    lim0[k] = lim; fm0[k] = fm; mk0[k] = mkt
        for k in range(K):
            if units[k] <= 0.0:
                continue
            px = -1.0; why = 0.0; xm = -1; xo = 0.0
            s = act[k] if act[k] > m0 else m0
            for i in range(s, m0 + 60):
                if np.isnan(o1[i, k]):
                    continue
                if (i == act[k] or i == m0) and o1[i, k] >= tpp[k]:
                    px = o1[i, k]; xm = i; xo = 1.0
                    break
                if h1[i, k] >= tpp[k] * (1.0 + seps):
                    px = tpp[k]; xm = i
                    break
            hmk = hm[t] - mtsh if mk0[k] == 1 else hm[t]
            if px < 0.0 and ebar[k] < t and t - ebar[k] >= hmk and not np.isnan(c1[m0 + 59, k]):
                j = m0 + 59 + lat
                if lat > 0 and j < nm and not np.isnan(c1[j, k]):
                    px = c1[j, k] * (1.0 - slip[k]); xm = j
                else:
                    px = c1[m0 + 59, k] * (1.0 - slip[k]); xm = m0 + 59
                why = 1.0
            if px > 0.0:
                proceeds = units[k] * px * (1.0 - fee); cash += proceeds
                if nl < maxlog:
                    lg[nl, 0] = ebar[k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = why
                    lg[nl, 4] = proceeds / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]; lg[nl, 7] = xm
                    lg[nl, 8] = fm0[k]; lg[nl, 9] = lim0[k]; lg[nl, 10] = mk0[k]; lg[nl, 11] = xo
                    nl += 1
                units[k] = 0.0; okbar[k] = t + cd
        v = cash
        for k in range(K):
            if not np.isnan(c1[m0 + 59, k]):
                lastc[k] = c1[m0 + 59, k]
            if units[k] > 0.0:
                v += units[k] * lastc[k]
        eq[t] = v; eq_prev = v
    return eq, lg[:nl].copy()


def rrun(A, hidx, Ph, depth=E.LIVE["depth"], N=E.LIVE["N"], fmax=E.LIVE["fmax"], tp=E.LIVE["tp"], hm=E.LIVE["hm"],
         cd=E.LIVE["cd"], fee=E.FEE, slip=E.SLIP_MAJOR, pen=0.0, delay=0, beps=E.EPS, bmin=0, bmode=0, seps=E.EPS,
         mpx=0, lat=0, mdelay=0, mtsh=0):
    nh, K = Ph["c"].shape
    sl = np.full(K, slip) if np.ndim(slip) == 0 else np.asarray(slip, np.float64)
    eq, lg = rkernel(A["o"], A["h"], A["l"], A["c"], Ph["ref"].values, E.full(nh, K, depth),
                     E.full(nh, 0, N, np.int64), E.full(nh, 0, fmax), E.full(nh, K, tp), E.full(nh, 0, hm, np.int64),
                     int(cd), fee, sl, pen, int(delay), float(beps), int(bmin), int(bmode), float(seps), int(mpx),
                     int(lat), int(mdelay), int(mtsh))
    return pd.Series(eq, hidx), pd.DataFrame(lg, columns=RLOG)
