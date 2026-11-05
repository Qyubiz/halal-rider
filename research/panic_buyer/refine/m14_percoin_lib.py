"""m14 percoin lib: per-coin walk-forward depth / take-profit. ek2 = m12 ek with a per-hour per-coin take-profit table TP
(fixed at the entry hour) and a per-hour per-coin bid table LIM. Per-coin parameters come from a coin's OWN past
standalone trades (exit before the refit date), see schedule(). With default depth/tp everywhere ek2 == 90% mode."""
import os, sys
import numpy as np, pandas as pd
from numba import njit
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import engine as E
import m10_velocity_lib as V
import m12_entry_lib as M

LOGC = V.LOGC
DEPTHS = [0.12, 0.13, 0.14, 0.15, 0.16, 0.17]
TPS = [0.02, 0.025, 0.03]
CANDS = [(d, t) for d in DEPTHS for t in TPS]
DEF = (0.14, 0.025)


@njit(cache=True)
def ek2(o, h, l, c, LIM, SC, usesc, near, lastb, TP, hm, cd, eps, fee, slip, pen, pens, mtp_now, t0):
    n, K = c.shape
    units = np.zeros(K); entry = np.zeros(K); amt0 = np.zeros(K); tpp = np.zeros(K); eq0 = np.zeros(K)
    ebar = np.full(K, -1, np.int64); mkt = np.zeros(K, np.int64); okc = np.zeros(K, np.int64)
    lastc = np.full(K, np.nan)
    cash = 1.0; eq = np.ones(n); inv = np.zeros(n); eq_prev = 1.0
    maxlog = 200000; lg = np.zeros((maxlog, 9)); nl = 0
    key = np.empty(K); ck = np.empty(K, np.int64); clim = np.empty(K)
    for t in range(t0):
        for k in range(K):
            if not np.isnan(c[t, k]):
                lastc[k] = c[t, k]
    for t in range(t0, n):
        nheld = 0
        for k in range(K):
            if units[k] > 0.0:
                nheld += 1
        m = 0
        if nheld == 0:
            for k in range(K):
                if t < okc[k] or np.isnan(o[t, k]) or np.isnan(LIM[t, k]):
                    continue
                d = o[t, k] / LIM[t, k] - 1.0
                if usesc == 1:
                    if d <= near and not np.isnan(SC[t, k]):
                        key[m] = SC[t, k] - 1e6
                    else:
                        key[m] = d
                else:
                    key[m] = d
                ck[m] = k; clim[m] = LIM[t, k]; m += 1
        if m > 0 and cash > 1e-9:
            j = np.argsort(key[:m])[0]
            k = ck[j]; lim = clim[j]
            per = min(eq_prev, cash)
            fill = -1.0; ismkt = 0
            if o[t, k] <= lim:
                fill = o[t, k] * (1.0 + slip[k]) * (1.0 + pen); ismkt = 1
            elif l[t, k] <= lim * (1.0 - eps):
                fill = lim * (1.0 + pen)
            if fill > 0.0:
                mkt[k] = ismkt if mtp_now else 0
                a = min(per, cash)
                cash -= a; units[k] = a * (1.0 - fee) / fill
                entry[k] = fill; amt0[k] = a; ebar[k] = t; eq0[k] = eq_prev
                tpp[k] = fill * (1.0 + TP[t, k])
        for k in range(K):
            if units[k] > 0.0 and (ebar[k] < t or mkt[k] == 1) and not np.isnan(o[t, k]):
                px = -1.0; why = 0.0
                if ebar[k] == t:
                    if h[t, k] >= tpp[k] * (1.0 + eps):
                        px = tpp[k]
                elif o[t, k] >= tpp[k]:
                    px = o[t, k]
                elif h[t, k] >= tpp[k] * (1.0 + eps):
                    px = tpp[k]
                age = t - ebar[k] + mkt[k]
                if px < 0.0 and age >= hm:
                    px = c[t, k] * (1.0 - slip[k]); why = 1.0
                if px > 0.0:
                    proceeds = units[k] * px * (1.0 - pens) * (1.0 - fee); cash += proceeds
                    if nl < maxlog:
                        lg[nl, 0] = ebar[k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = why
                        lg[nl, 4] = proceeds / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]
                        lg[nl, 7] = eq0[k]; nl += 1
                    units[k] = 0.0
                    okc[k] = t + cd
        for k in range(K):
            if units[k] > 0.0 and lastb[k] == t:
                proceeds = units[k] * c[t, k] * (1.0 - slip[k]) * (1.0 - pens) * (1.0 - fee); cash += proceeds
                if nl < maxlog:
                    lg[nl, 0] = ebar[k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = 5.0
                    lg[nl, 4] = proceeds / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]
                    lg[nl, 7] = eq0[k]; nl += 1
                units[k] = 0.0
                okc[k] = t + cd
        for k in range(K):
            if not np.isnan(c[t, k]):
                lastc[k] = c[t, k]
        v = cash; w = 0.0
        for k in range(K):
            if units[k] > 0.0:
                v += units[k] * lastc[k]; w += units[k] * lastc[k]
        eq[t] = v; eq_prev = v; inv[t] = w / v if v > 0.0 else 0.0
    for k in range(K):
        if units[k] > 0.0 and nl < maxlog:
            proceeds = units[k] * lastc[k] * (1.0 - slip[k]) * (1.0 - pens) * (1.0 - fee)
            lg[nl, 0] = ebar[k]; lg[nl, 1] = n - 1; lg[nl, 2] = k; lg[nl, 3] = 9.0
            lg[nl, 4] = proceeds / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]; lg[nl, 7] = eq0[k]; nl += 1
    return eq, lg[:nl].copy(), inv




@njit(cache=True)
def solo(o, h, l, c, LIM, tpa, hm, cd, eps, fee):
    """Each coin alone, one trade at a time: returns (coin, exit hour, net return) of every closed trade."""
    n, K = c.shape
    mx = 4000
    out = np.zeros((K * mx, 3)); m = 0
    for k in range(K):
        held = False; ent = 0.0; eb = 0; ok = 0; cnt = 0
        for t in range(n):
            if np.isnan(o[t, k]):
                continue
            if not held and t >= ok and not np.isnan(LIM[t, k]):
                lim = LIM[t, k]
                if o[t, k] <= lim:
                    ent = o[t, k]; held = True; eb = t
                elif l[t, k] <= lim * (1.0 - eps):
                    ent = lim; held = True; eb = t
            if held:
                tp = ent * (1.0 + tpa); px = -1.0
                if h[t, k] >= tp * (1.0 + eps) and t > eb:
                    px = tp
                elif t == eb and h[t, k] >= tp * (1.0 + eps):
                    px = tp
                if px < 0.0 and t - eb >= hm:
                    px = c[t, k]
                if px > 0.0 and cnt < mx:
                    held = False; ok = t + cd
                    out[m, 0] = k; out[m, 1] = t; out[m, 2] = px / ent * (1 - fee) * (1 - fee) - 1.0
                    m += 1; cnt += 1
    return out[:m]


def trade_store(P, idx):
    """{cand index: DataFrame(coin, time(ns), ret)} from standalone replays on this panel (normal fills)."""
    A = M.anchor(P, "c"); g = M.guard(P, 0.35)
    st = {}
    o, h, l, cf = P["o"].values, P["h"].values, P["l"].values, P["cf"].values
    tv = idx.values
    for ci, (d, t) in enumerate(CANDS):
        LIM = np.ascontiguousarray(np.where(g > 0, A * (1.0 - d), np.nan))
        r = solo(o, h, l, cf, LIM, float(t), 96, 3, float(E.EPS), float(E.FEE))
        st[ci] = (r[:, 0].astype(int), tv[r[:, 1].astype(int)], r[:, 2])
    return st


def merge_stores(stores):
    """stores: list of (cols, store) -> {symbol: {cand: (time ns array, ret array)}}"""
    out = {}
    for cols, st in stores:
        for ci, (kk, tt, rr) in st.items():
            for j, s in enumerate(cols):
                m = kk == j
                out.setdefault(s, {}).setdefault(ci, []).append((tt[m], rr[m]))
    res = {}
    for s, dd in out.items():
        res[s] = {}
        for ci, lst in dd.items():
            tt = np.concatenate([a for a, b in lst]); rr = np.concatenate([b for a, b in lst])
            o = np.argsort(tt, kind="stable"); res[s][ci] = (tt[o], rr[o])
    return res


def first_bar(P):
    """first hour with data per coin (datetime64)"""
    return {s: P["o"][s].first_valid_index() for s in P["o"].columns}


def schedule(P, idx, store, des, firsts):
    """Per-hour per-coin (depth, tp) tables. des: win (0 expanding / months), R (months), minT, m, metric, cset."""
    cols = list(P["o"].columns); n = len(idx); K = len(cols)
    DEP = np.full((n, K), DEF[0]); TP = np.full((n, K), DEF[1])
    dstart = pd.Timestamp("2018-01-01")
    dates = []
    y, mo = 2018, 1
    while pd.Timestamp(y, mo, 1) <= idx[-1]:
        dates.append(pd.Timestamp(y, mo, 1)); mo += des["R"]
        if mo > 12: mo -= 12; y += 1
    defi = CANDS.index(DEF)
    for di, D in enumerate(dates):
        if D < idx[0] and (di + 1 < len(dates) and dates[di + 1] <= idx[0]):
            continue
        D1 = dates[di + 1] if di + 1 < len(dates) else idx[-1] + pd.Timedelta(days=1)
        a = np.searchsorted(idx.values, np.datetime64(max(D, idx[0]))); b = np.searchsorted(idx.values, np.datetime64(D1))
        if b <= a: continue
        lo = np.datetime64(D - pd.DateOffset(months=des["win"])) if des["win"] else np.datetime64("1990-01-01")
        hi = np.datetime64(D)
        for j, s in enumerate(cols):
            if s not in store: continue
            f0 = firsts.get(s)
            months = max((D - max(f0, D - pd.DateOffset(months=des["win"]) if des["win"] else f0)).days / 30.4, 6.0) if f0 is not None else 6.0
            def stat(ci):
                tt, rr = store[s][ci]
                i0 = np.searchsorted(tt, lo); i1 = np.searchsorted(tt, hi)
                r = rr[i0:i1]
                if des["metric"] == "mean":
                    return len(r), (r.mean() if len(r) else 0.0)
                return len(r), r.sum() / months
            nd, sd = stat(defi)
            if nd < des["minT"]: continue
            best, bd = defi, 0.0
            for ci, (d, t) in enumerate(CANDS):
                if ci == defi: continue
                if des["cset"] == "depth" and t != DEF[1]: continue
                nc, sc = stat(ci)
                if nc < des["minT"]: continue
                ne = min(nc, nd)
                delta = (sc - sd) * ne / (ne + des["m"])
                if delta > bd: bd, best = delta, ci
            DEP[a:b, j], TP[a:b, j] = CANDS[best]
    return DEP, TP


def prun(P, idx, DEP, TP, pen=0.0, pens=0.0, mtp_now=False, cd_add=0, liquidate=False, t0=None):
    O = P["o"].values; K = O.shape[1]
    A = M.anchor(P, "c"); g = M.guard(P, 0.35)
    LIM = np.ascontiguousarray(np.where(g > 0, A * (1.0 - DEP), np.nan))
    sl = np.array([E.SLIP_MAJOR if s in E.H19 else E.SLIP_OTHER for s in P["o"].columns]) if liquidate else np.full(K, E.SLIP_MAJOR)
    lastb = V.last_bars(P) if liquidate else np.full(K, -1, np.int64)
    i0 = 0 if t0 is None else int(np.searchsorted(idx.values, np.datetime64(t0)))
    eq, lg, inv = ek2(O, P["h"].values, P["l"].values, P["cf"].values, LIM, np.zeros((1, 1)), 0, 0.0, lastb,
                      np.ascontiguousarray(TP), 96, max(3 + cd_add, 0), float(E.EPS), float(E.FEE), sl,
                      float(pen), float(pens), bool(mtp_now), i0)
    return pd.Series(eq, idx), pd.DataFrame(lg, columns=LOGC), pd.Series(inv, idx)
