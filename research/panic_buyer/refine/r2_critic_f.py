"""Round 2 critic: independent HOURLY cross-check of the realism lens's row (f), the gap between panic_buyer.py and
the backtest for market buys (coin already under its trigger at the hour open).

panic_buyer.py: open_position() calls place_tp() right after a market buy, so the +1.5% sell rests from ~10 s into
the buy hour (backtest: from the next hourly cycle). opened_ms = the cycle time at the start of the buy hour, so the
48h time stop fires at the cycle 48h later = the close of hour ebar+47 (backtest: close of hour ebar+48).

fkernel = engine.gkernel (full take-profit path only) plus two switches, both off = engine.run exactly:
  mtp   1: a market buy's take-profit may fill in the buy hour when the hour's high trades eps through it
        (it cannot fill at the open: the open is the buy price, below the target)
  mtsh  a market buy's time stop comes mtsh hours earlier
Approximation vs the realism lens's 1-minute rkernel: the hourly high includes minute 0 (slightly optimistic).
This is a check of an existing claim, not a new hypothesis: nothing here is proposed for the live bot.
"""
import numpy as np
import pandas as pd
from numba import njit

import engine as E
import study as S


@njit(cache=False)
def fkernel(o, h, l, c, ref, depth, nord, fmax, tpa, hm, cd, eps, fee, slip, pen, mtp, mtsh):
    n, K = c.shape
    units = np.zeros(K); entry = np.zeros(K); amt0 = np.zeros(K); tpp = np.zeros(K)
    ebar = np.full(K, -1, np.int64); okbar = np.zeros(K, np.int64); mk = np.zeros(K, np.int64)
    lastc = np.full(K, np.nan); dist = np.empty(K)
    cash = 1.0; eq = np.empty(n); eq_prev = 1.0
    maxlog = 600000; lg = np.zeros((maxlog, 8)); nl = 0
    for t in range(n):
        for k in range(K):
            dist[k] = 1e9
            if units[k] > 0.0 or t < okbar[k] or np.isnan(o[t, k]) or np.isnan(ref[t, k]):
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
                fill = -1.0; m = 0
                if o[t, k] <= lim:
                    fill = o[t, k] * (1.0 + slip[k]) * (1.0 + pen); m = 1
                elif l[t, k] <= lim * (1.0 - eps):
                    fill = lim * (1.0 + pen)
                if fill > 0.0 and cash > 1e-9:
                    a = min(per, cash)
                    cash -= a; units[k] = a * (1.0 - fee) / fill
                    entry[k] = fill; amt0[k] = a; ebar[k] = t; mk[k] = m
                    tpp[k] = fill * (1.0 + tpa[t, k])
        for k in range(K):
            if units[k] <= 0.0 or np.isnan(o[t, k]):
                continue
            px = -1.0; why = 0.0
            if ebar[k] < t:
                if o[t, k] >= tpp[k]:
                    px = o[t, k]
                elif h[t, k] >= tpp[k] * (1.0 + eps):
                    px = tpp[k]
                hmk = hm[t] - mtsh if mk[k] == 1 else hm[t]
                if px < 0.0 and t - ebar[k] >= hmk:
                    px = c[t, k] * (1.0 - slip[k]); why = 1.0
            elif mtp == 1 and mk[k] == 1 and h[t, k] >= tpp[k] * (1.0 + eps):
                px = tpp[k]
            if px > 0.0:
                proceeds = units[k] * px * (1.0 - fee); cash += proceeds
                if nl < maxlog:
                    lg[nl, 0] = ebar[k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = why
                    lg[nl, 4] = proceeds / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]
                    lg[nl, 7] = mk[k]; nl += 1
                units[k] = 0.0; okbar[k] = t + cd
        v = cash
        for k in range(K):
            if not np.isnan(c[t, k]):
                lastc[k] = c[t, k]
            if units[k] > 0.0:
                v += units[k] * lastc[k]
        eq[t] = v; eq_prev = v
    return eq, lg[:nl].copy()


def frun(P, idx, mtp=0, mtsh=0, pen=0.0):
    L = E.LIVE
    O = P["o"].values; n, K = O.shape
    eq, lg = fkernel(O, P["h"].values, P["l"].values, P["cf"].values, P["ref"].values, E.full(n, K, L["depth"]),
                     E.full(n, 0, L["N"], np.int64), E.full(n, 0, L["fmax"]), E.full(n, K, L["tp"]),
                     E.full(n, 0, L["hm"], np.int64), int(L["cd"]), E.EPS, E.FEE, np.full(K, E.SLIP_MAJOR), pen,
                     int(mtp), int(mtsh))
    return pd.Series(eq, idx), pd.DataFrame(lg, columns=E.LOG_COLS + ["mkt"])


if __name__ == "__main__":
    for dead, pen in ((False, 0.0), (True, 0.0), (False, 0.002)):
        syms, idx, P = S.market(dead)
        e0, l0 = E.run(P, idx, pen=pen)
        rows = {"backtest (engine.run)": (e0, l0)}
        e1, l1 = frun(P, idx, pen=pen)
        same = np.array_equal(e0.values, e1.values) and np.array_equal(l0.values, l1[E.LOG_COLS].values)
        print(f"--- dead={dead} pen={pen}: fkernel switches off identical to engine.run: {same}")
        assert same
        rows["TP right after market buy"] = frun(P, idx, mtp=1, pen=pen)
        rows["time stop 1h earlier (mkt)"] = frun(P, idx, mtsh=1, pen=pen)
        rows["live bot timing (both) = f"] = frun(P, idx, mtp=1, mtsh=1, pen=pen)
        for name, (e, lg) in rows.items():
            print(E.line(name, E.score(e, lg, idx)))
        lg = rows["live bot timing (both) = f"][1]
        d = lg[(idx[lg.eb.astype(int).values] < E.DESIGN_END) & (lg.mkt == 1)]
        same_hour = (d.xb == d.eb).sum()
        print(f"    f: 2021-23 market buys {len(d)}, sold in the buy hour {same_hour}, avg {d.ret.mean() * 100:+.2f}%")
