"""x10 critic: stress tools for the 1-minute designs. `xkernel` is x10_velocity_lib.vkernel copied, plus four switches.
With lat=0, mlat=0, watch=0, wslip=0 it is vkernel exactly (x10_critic_check.py proves it).

  lat    1: a resting buy bid can only fill from the minute AFTER it was placed (a bid re-placed on the same coin at the
         same limit keeps its age). Removes every limit fill inside the minute whose open was used to place the bid.
  mlat   1: one minute of execution latency on market orders. A market buy decided at a cycle minute's open fills at the
         NEXT minute's open (+ slippage); a trailing sale decided on a minute close fills one minute later than in
         vkernel (open of the minute after next). Cash for a pending market buy is set aside at the decision.
  watch  1: minute-level "watch mode" (x10_universe_lib.ukernel watch=True, on 1-minute candles): every eligible coin
         is watched; at a cycle minute a coin under its trigger at the open is bought at market; inside a minute, a
         coin whose low trades eps through its trigger is bought at trigger*(1+slippage+wslip). Buys are min(fmax *
         equity, cash), open positions <= nord; coins are served closest-to-trigger-at-the-cycle first (the order in
         which coins cross inside ONE minute is unknown; it matters far less than inside one hour).
Everything else (costs, exits, cooldowns, flash trigger, re-entry, loss cooldown) is vkernel's.
"""
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402
import x10_velocity_lib as V  # noqa: E402

VEL_OUT = "/tmp/claude-0/-home-user-halal-rider/ec562e35-ea9e-5a74-ab92-4ff22c5eeb77/scratchpad/pb_out_x10/velocity"
V.CACHE = os.path.join(VEL_OUT, "cache")          # reuse the velocity agent's float32 minute cache (read-only)
OUT = os.environ.get("PB_OUT", os.path.expanduser("~/pb_out_x10_critic"))
os.makedirs(OUT, exist_ok=True)


@njit(cache=False)
def xkernel(o1, h1, l1, c1, ref, depth, nord, fmax, tpa, hm, cd, eps, fee, slip, pen, delay,
            cyc, rearm, cdm, near, xmode, arm, trail, cap, spen, ref2, depth2, tpf, redip, rewin, cdl,
            lat, mlat, watch, wslip):
    nm, K = c1.shape
    nh = nm // 60
    units = np.zeros(K); entry = np.zeros(K); amt0 = np.zeros(K); tpp = np.zeros(K); capp = np.zeros(K)
    ebar = np.full(K, -1, np.int64); okm = np.zeros(K, np.int64); act = np.zeros(K, np.int64)
    fmin = np.zeros(K, np.int64); eqb = np.zeros(K); lastm = np.zeros(K)
    armed = np.zeros(K, np.int64); peak = np.zeros(K); pend = np.zeros(K, np.int64)
    resv = np.zeros(K); rlim = np.zeros(K); nres = 0; rsum = 0.0
    rex = np.zeros(K); rexu = np.full(K, -1, np.int64); rsrc = np.zeros(K, np.int64)
    limk = np.zeros(K); srck = np.zeros(K, np.int64)
    rfrom = np.zeros(K, np.int64); plim = np.full(K, -1.0)          # lat: first fillable minute, previous limit
    mq = np.zeros(K); mqsrc = np.zeros(K, np.int64); mqeq = np.zeros(K)   # mlat: pending market buys
    wl = np.zeros(K); wsrc = np.zeros(K, np.int64); wok = np.zeros(K, np.int64); word = np.arange(K); wper = 0.0
    lastc = np.full(K, np.nan); dist = np.empty(K)
    cash = 1.0; eq = np.empty(nh); inv = np.empty(nh); eq_prev = 1.0
    maxlog = 3000000; lg = np.zeros((maxlog, 11)); nl = 0
    due = -1
    for t in range(nh):
        m0 = 60 * t
        for i in range(m0, m0 + 60):
            j = i - m0
            # ---- 0. pending market buys (mlat): fill at this minute's open
            for k in range(K):
                if mq[k] > 0.0 and not np.isnan(o1[i, k]):
                    fill = o1[i, k] * (1.0 + slip[t, k]) * (1.0 + pen)
                    a = mq[k]; mq[k] = 0.0
                    units[k] = a * (1.0 - fee) / fill
                    entry[k] = fill; amt0[k] = a; ebar[k] = t; fmin[k] = i; eqb[k] = mqeq[k]
                    tpp[k] = fill * (1.0 + (tpf if mqsrc[k] == 1 else tpa[t, k])); capp[k] = fill * (1.0 + cap)
                    act[k] = i + delay if delay > 0 else m0 + 60
                    armed[k] = 0; pend[k] = 0; lastm[k] = fill
            # ---- 1. buy cycle
            if j % cyc == 0 or i == due:
                if nres > 0:
                    for k in range(K):
                        plim[k] = rlim[k] if resv[k] > 0.0 else -1.0
                        resv[k] = 0.0
                    nres = 0; rsum = 0.0
                else:
                    for k in range(K):
                        plim[k] = -1.0
                for k in range(K):
                    wok[k] = 0
                if cash > 1e-9:
                    for k in range(K):
                        dist[k] = 1e9
                        if units[k] > 0.0 or mq[k] > 0.0 or i < okm[k] or np.isnan(o1[i, k]) or np.isnan(ref[t, k]):
                            continue
                        lim_ = ref[t, k] * (1.0 - depth[t, k]); s_ = 0
                        if depth2 > 0.0 and not np.isnan(ref2[t, k]):
                            l2_ = ref2[t, k] * (1.0 - depth2)
                            if l2_ > lim_:
                                lim_ = l2_; s_ = 1
                        if redip > 0.0 and i < rexu[k] and rex[k] > lim_:
                            lim_ = rex[k]; s_ = 2
                        limk[k] = lim_; srck[k] = s_
                        d_ = o1[i, k] / lim_ - 1.0
                        if d_ <= near:
                            dist[k] = d_
                    order = np.argsort(dist)
                    if j == 0:
                        eqm = eq_prev
                    else:
                        eqm = cash
                        for k in range(K):
                            if units[k] > 0.0:
                                eqm += units[k] * lastm[k]
                            eqm += mq[k]
                    if watch == 1:
                        npos = 0
                        for k in range(K):
                            if units[k] > 0.0 or mq[k] > 0.0:
                                npos += 1
                        wper = fmax[t] * eqm
                        for q in range(K):
                            k = order[q]
                            word[q] = k
                            if dist[k] >= 1e8:
                                continue
                            lim = limk[k]
                            if o1[i, k] <= lim:
                                if npos < nord[t] and cash > 1e-9:
                                    a = min(wper, cash)
                                    if mlat == 1:
                                        cash -= a; mq[k] = a; mqsrc[k] = srck[k]; mqeq[k] = eqm; npos += 1
                                    else:
                                        fill = o1[i, k] * (1.0 + slip[t, k]) * (1.0 + pen)
                                        cash -= a; units[k] = a * (1.0 - fee) / fill
                                        entry[k] = fill; amt0[k] = a; ebar[k] = t; fmin[k] = i; eqb[k] = eqm
                                        tpp[k] = fill * (1.0 + (tpf if srck[k] == 1 else tpa[t, k])); capp[k] = fill * (1.0 + cap)
                                        act[k] = i + delay if delay > 0 else m0 + 60
                                        armed[k] = 0; pend[k] = 0; lastm[k] = fill; npos += 1
                            else:
                                wok[k] = 1; wl[k] = lim; wsrc[k] = srck[k]
                    else:
                        nsel = 0
                        for q in range(K):
                            if dist[order[q]] < 1e8 and nsel < nord[t]:
                                nsel += 1
                        if nsel > 0:
                            per = min(fmax[t] * eqm, cash / nsel)
                            for q in range(nsel):
                                k = order[q]
                                lim = limk[k]
                                if o1[i, k] <= lim:
                                    if cash - rsum > 1e-9:
                                        a = min(per, cash - rsum)
                                        if mlat == 1:
                                            cash -= a; mq[k] = a; mqsrc[k] = srck[k]; mqeq[k] = eqm
                                        else:
                                            fill = o1[i, k] * (1.0 + slip[t, k]) * (1.0 + pen)
                                            cash -= a; units[k] = a * (1.0 - fee) / fill
                                            entry[k] = fill; amt0[k] = a; ebar[k] = t; fmin[k] = i; eqb[k] = eqm
                                            tpp[k] = fill * (1.0 + (tpf if srck[k] == 1 else tpa[t, k])); capp[k] = fill * (1.0 + cap)
                                            act[k] = i + delay if delay > 0 else m0 + 60
                                            armed[k] = 0; pend[k] = 0; lastm[k] = fill
                                else:
                                    resv[k] = per; rlim[k] = lim; rsum += per; nres += 1; rsrc[k] = srck[k]
                                    eqb[k] = eqm
                                    if not (plim[k] == lim and rfrom[k] <= i + lat):
                                        rfrom[k] = i + lat
                due = -1
            # ---- 2. resting bids that trade eps through their limit this minute
            if nres > 0:
                for k in range(K):
                    if resv[k] > 0.0 and i >= rfrom[k] and l1[i, k] <= rlim[k] * (1.0 - eps):
                        fill = rlim[k] * (1.0 + pen)
                        a = min(resv[k], cash)
                        rsum -= resv[k]; resv[k] = 0.0; nres -= 1
                        if a > 1e-9:
                            cash -= a; units[k] = a * (1.0 - fee) / fill
                            entry[k] = fill; amt0[k] = a; ebar[k] = t; fmin[k] = i
                            tpp[k] = fill * (1.0 + (tpf if rsrc[k] == 1 else tpa[t, k])); capp[k] = fill * (1.0 + cap)
                            act[k] = i + delay if delay > 0 else m0 + 60
                            armed[k] = 0; pend[k] = 0; lastm[k] = fill
            # ---- 2w. watch mode: buy the coins that cross their trigger this minute
            if watch == 1:
                npos = 0
                for k in range(K):
                    if units[k] > 0.0 or mq[k] > 0.0:
                        npos += 1
                for q in range(K):
                    k = word[q]
                    if wok[k] == 0 or units[k] > 0.0 or mq[k] > 0.0:
                        continue
                    if npos >= nord[t] or cash <= 1e-9:
                        break
                    if i >= okm[k] and l1[i, k] <= wl[k] * (1.0 - eps):
                        fill = wl[k] * (1.0 + slip[t, k] + wslip) * (1.0 + pen)
                        a = min(wper, cash)
                        cash -= a; units[k] = a * (1.0 - fee) / fill
                        entry[k] = fill; amt0[k] = a; ebar[k] = t; fmin[k] = i; eqb[k] = wper / fmax[t]
                        tpp[k] = fill * (1.0 + (tpf if wsrc[k] == 1 else tpa[t, k])); capp[k] = fill * (1.0 + cap)
                        act[k] = i + delay if delay > 0 else m0 + 60
                        armed[k] = 0; pend[k] = 0; lastm[k] = fill; npos += 1; wok[k] = 0
            # ---- 3. exits
            for k in range(K):
                if units[k] <= 0.0:
                    continue
                px = -1.0; why = 0.0
                if i >= act[k] and not np.isnan(o1[i, k]):
                    if xmode == 0:
                        if (i == act[k] or j == 0) and o1[i, k] >= tpp[k]:
                            px = o1[i, k]
                        elif h1[i, k] >= tpp[k] * (1.0 + eps):
                            px = tpp[k]
                    else:
                        if pend[k] == 1:
                            px = o1[i, k] * (1.0 - slip[t, k]); why = 2.0
                        else:
                            if pend[k] >= 2:
                                pend[k] -= 1
                            if cap > 0.0 and (i == act[k] or j == 0) and o1[i, k] >= capp[k]:
                                px = o1[i, k]
                            elif cap > 0.0 and h1[i, k] >= capp[k] * (1.0 + eps):
                                px = capp[k]
                            elif pend[k] == 0:
                                cc = c1[i, k]
                                if armed[k] == 0:
                                    if cc >= entry[k] * (1.0 + arm):
                                        armed[k] = 1; peak[k] = cc
                                elif cc > peak[k]:
                                    peak[k] = cc
                                elif (trail > 0.0 and cc <= peak[k] * (1.0 - trail)) or (trail <= 0.0 and cc < peak[k]):
                                    pend[k] = 1 + mlat
                if px < 0.0 and j == 59 and ebar[k] < t and t - ebar[k] >= hm[t] and not np.isnan(c1[i, k]):
                    px = c1[i, k] * (1.0 - slip[t, k]); why = 1.0
                if px > 0.0:
                    proceeds = units[k] * px * (1.0 - spen) * (1.0 - fee); cash += proceeds
                    if nl < maxlog:
                        lg[nl, 0] = ebar[k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = why
                        lg[nl, 4] = proceeds / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]
                        lg[nl, 7] = i; lg[nl, 8] = fmin[k]; lg[nl, 9] = eqb[k]; lg[nl, 10] = px
                        nl += 1
                    units[k] = 0.0; pend[k] = 0; armed[k] = 0
                    okm[k] = (t + cd) * 60 if cdm < 0 else i + cdm
                    if cdl >= 0 and proceeds < amt0[k]:
                        okm[k] = i + cdl
                    if redip > 0.0 and proceeds > amt0[k]:
                        rex[k] = px * (1.0 - redip); rexu[k] = i + rewin
                    if rearm == 1:
                        due = i + 1
                elif not np.isnan(c1[i, k]):
                    lastm[k] = c1[i, k]
        v = cash; w = 0.0
        for k in range(K):
            if not np.isnan(c1[m0 + 59, k]):
                lastc[k] = c1[m0 + 59, k]
            if units[k] > 0.0:
                v += units[k] * lastc[k]; w += units[k] * lastc[k]
            v += mq[k]
        eq[t] = v; eq_prev = v; inv[t] = w / v if v > 0 else 0.0
    return eq, lg[:nl].copy(), inv


XDEF = dict(V.DEFAULTS, lat=0, mlat=0, watch=0, wslip=0.0)
LOGX = V.LOGC + ["px"]


def xrun(A, hidx, Ph, SLIP, K=None, ref=None, **kw):
    """vrun with the critic switches. ref: optional (n,K) trigger reference replacing Ph['ref'] (NaN = coin not
    eligible that hour, e.g. a crash guard); the flash reference is masked the same way."""
    p = dict(XDEF); p.update(kw)
    bad = set(kw) - set(XDEF)
    assert not bad, f"unknown settings {bad}"
    K = K or A["c"].shape[1]
    nh = len(hidx)
    sl = SLIP[:, :K] if np.ndim(SLIP) == 2 else np.full((nh, K), float(SLIP))
    o1, h1, l1, c1 = (A[f][:, :K] if A[f].shape[1] != K else A[f] for f in ("o", "h", "l", "c"))
    r1 = np.ascontiguousarray(Ph["ref"].values[:, :K] if ref is None else ref[:, :K])
    r2 = V.ref2_(Ph, p["look2"], K)
    if ref is not None:
        r2 = np.where(np.isnan(r1), np.nan, r2)
    eq, lg, inv = xkernel(np.asarray(o1), np.asarray(h1), np.asarray(l1), np.asarray(c1), r1,
                          E.full(nh, K, p["depth"]), E.full(nh, 0, p["N"], np.int64), E.full(nh, 0, p["fmax"]),
                          E.full(nh, K, p["tp"]), E.full(nh, 0, p["hm"], np.int64), int(p["cd"]), float(p["eps"]),
                          float(p["fee"]), np.ascontiguousarray(sl, np.float64), float(p["pen"]), int(p["delay"]),
                          int(p["cyc"]), int(p["rearm"]), int(p["cdm"]), float(p["near"]), int(p["xmode"]),
                          float(p["arm"]), float(p["trail"]), float(p["cap"]), float(p["spen"]),
                          np.ascontiguousarray(r2), float(p["depth2"]), float(p["tpf"]), float(p["redip"]),
                          int(p["rewin"]), int(p["cdl"]), int(p["lat"]), int(p["mlat"]), int(p["watch"]),
                          float(p["wslip"]))
    return pd.Series(eq, hidx), pd.DataFrame(lg, columns=LOGX), pd.Series(inv, hidx)


def guard_ref(Ph, g, K):
    """Trigger reference with the universe agent's crash guard: a coin is skipped for the hour when its last close is
    more than g below its highest close of the previous 168 hours (x10_universe_push2.guard_mask, known at the open)."""
    c = Ph["cf"].iloc[:, :K]
    hi = c.rolling(168, min_periods=1).max()
    ok = ((c / hi - 1) >= -g).shift(1, fill_value=True).values
    r = Ph["ref"].values[:, :K].copy()
    r[~ok] = np.nan
    return np.ascontiguousarray(r)
