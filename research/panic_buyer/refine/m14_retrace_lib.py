"""m14 "retrace": take-profit from the size of the fall. At the fill, H = 24h highest close the bid was based on,
F = fill price; target = F + r x (H - F), i.e. gain r x (H/F - 1), clipped to [glo, ghi]; fixed once at the fill.
r = 0 -> identical to m11 fk (90% mode). Everything else (entry, 96h stop, cooldown, guard) as fk."""
import os, sys
import numpy as np
import pandas as pd
from numba import njit
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m11_fusion_lib as F  # noqa: E402
import m10_velocity_lib as V  # noqa: E402
import engine as E  # noqa: E402
LOGC = V.LOGC


@njit(cache=True)
def fkr(o, h, l, c, gmS, R, rid, lastb, dep, tpS, hm, cdw, cdl, fs, nslots, nbid, eps, fee, slip, pen, pens, mtp_now, t0,
       hs, zcap, killm, killw, klock, cln, clT, rr, glo, ghi):
    n, K = c.shape
    S = dep.shape[0]
    units = np.zeros((S, K)); entry = np.zeros((S, K)); amt0 = np.zeros((S, K)); tpp = np.zeros((S, K))
    eq0 = np.zeros((S, K)); ebar = np.full((S, K), -1, np.int64); mkt = np.zeros((S, K), np.int64)
    okc = np.zeros(K, np.int64)
    lastc = np.full(K, np.nan)
    cash = 1.0; eq = np.ones(n); inv = np.zeros(n); eq_prev = 1.0
    maxlog = 400000; lg = np.zeros((maxlog, 9)); nl = 0
    cdist = np.empty(K); cs = np.empty(K, np.int64); ck = np.empty(K, np.int64); clim = np.empty(K); clh = np.empty(K)
    bs = np.empty(K, np.int64); bk = np.empty(K, np.int64); blim = np.empty(K); bh = np.empty(K)
    lock = 0; streak = 0; rel = t0
    for t in range(t0):
        for k in range(K):
            if not np.isnan(c[t, k]):
                lastc[k] = c[t, k]
    for t in range(t0, n):
        nheld = 0
        for s in range(S):
            for k in range(K):
                if units[s, k] > 0.0:
                    nheld += 1
        m = 0
        if t >= lock:
            for k in range(K):
                busy = False
                for s in range(S):
                    if units[s, k] > 0.0:
                        busy = True
                if busy or t < okc[k] or np.isnan(o[t, k]):
                    continue
                best = -1.0; bsel = -1
                for s in range(S):
                    if gmS[s, t, k] <= 0.0:
                        continue
                    v = R[rid[s], t, k]
                    if np.isnan(v):
                        continue
                    lim = v * (1.0 - dep[s])
                    if lim > best:
                        best = lim; bsel = s
                if bsel >= 0:
                    cdist[m] = o[t, k] / best - 1.0; cs[m] = bsel; ck[m] = k; clim[m] = best; clh[m] = best / (1.0 - dep[bsel]); m += 1
        nb = 0
        if m > 0 and cash > 1e-9:
            order = np.argsort(cdist[:m])
            free = min(nslots - nheld, nbid)
            for q in range(m):
                if nb >= free:
                    break
                j = order[q]
                bs[nb] = cs[j]; bk[nb] = ck[j]; blim[nb] = clim[j]; bh[nb] = clh[j]; nb += 1
        if nb > 0:
            cash0 = cash
            for q in range(nb):
                s = bs[q]; k = bk[q]; lim = blim[q]; Hh = bh[q]
                per = min(fs[s] * eq_prev, cash0 / nb)
                if zcap > 0.0:
                    risk = hs if hs > 0.0 else 1.0
                    per = min(per, zcap * eq_prev / risk)
                fill = -1.0; ismkt = 0
                if o[t, k] <= lim:
                    fill = o[t, k] * (1.0 + slip[k]) * (1.0 + pen); ismkt = 1
                elif l[t, k] <= lim * (1.0 - eps):
                    fill = lim * (1.0 + pen)
                if fill > 0.0 and cash > 1e-9:
                    mkt[s, k] = ismkt if mtp_now else 0
                    a = min(per, cash)
                    cash -= a; units[s, k] = a * (1.0 - fee) / fill
                    entry[s, k] = fill; amt0[s, k] = a; ebar[s, k] = t; eq0[s, k] = eq_prev
                    if rr[s] > 0.0:
                        g = rr[s] * (Hh / fill - 1.0)
                        if g < glo[s]:
                            g = glo[s]
                        if g > ghi[s]:
                            g = ghi[s]
                        tpp[s, k] = fill * (1.0 + g)
                    else:
                        tpp[s, k] = fill * (1.0 + tpS[s, t, k])
        # ---------------- exits
        for s in range(S):
            for k in range(K):
                if units[s, k] > 0.0 and not np.isnan(o[t, k]):
                    px = -1.0; why = 0.0
                    if ebar[s, k] < t or mkt[s, k] == 1:
                        if ebar[s, k] == t:
                            if h[t, k] >= tpp[s, k] * (1.0 + eps):
                                px = tpp[s, k]
                        elif o[t, k] >= tpp[s, k]:
                            px = o[t, k]
                        elif h[t, k] >= tpp[s, k] * (1.0 + eps):
                            px = tpp[s, k]
                        age = t - ebar[s, k] + mkt[s, k]
                        if px < 0.0 and age >= hm[s]:
                            px = c[t, k] * (1.0 - slip[k]); why = 1.0
                    if px < 0.0 and hs > 0.0 and c[t, k] <= entry[s, k] * (1.0 - hs):
                        px = c[t, k] * (1.0 - slip[k]); why = 4.0
                    if px > 0.0:
                        proceeds = units[s, k] * px * (1.0 - pens) * (1.0 - fee); cash += proceeds
                        if nl < maxlog:
                            lg[nl, 0] = ebar[s, k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = why
                            lg[nl, 4] = proceeds / amt0[s, k] - 1.0; lg[nl, 5] = amt0[s, k]; lg[nl, 6] = entry[s, k]
                            lg[nl, 7] = eq0[s, k]; lg[nl, 8] = s; nl += 1
                        units[s, k] = 0.0
                        win = proceeds > amt0[s, k]
                        okc[k] = t + (cdw[s] if win else cdl[s])
                        if win:
                            streak = 0
                        else:
                            streak += 1
                            if cln > 0 and streak >= cln:
                                lock = max(lock, t + 1 + clT); streak = 0
            for k in range(K):
                if units[s, k] > 0.0 and lastb[k] == t:
                    proceeds = units[s, k] * c[t, k] * (1.0 - slip[k]) * (1.0 - pens) * (1.0 - fee); cash += proceeds
                    if nl < maxlog:
                        lg[nl, 0] = ebar[s, k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = 5.0
                        lg[nl, 4] = proceeds / amt0[s, k] - 1.0; lg[nl, 5] = amt0[s, k]; lg[nl, 6] = entry[s, k]
                        lg[nl, 7] = eq0[s, k]; lg[nl, 8] = s; nl += 1
                    units[s, k] = 0.0
                    okc[k] = t + cdl[s]
                    if proceeds > amt0[s, k]:
                        streak = 0
                    else:
                        streak += 1
                        if cln > 0 and streak >= cln:
                            lock = max(lock, t + 1 + clT); streak = 0
        # ---------------- hourly mark
        for k in range(K):
            if not np.isnan(c[t, k]):
                lastc[k] = c[t, k]
        v = cash; w = 0.0
        for k in range(K):
            for s in range(S):
                if units[s, k] > 0.0:
                    v += units[s, k] * lastc[k]; w += units[s, k] * lastc[k]
        # ---------------- 30-day drawdown kill-switch (guardrails monthly_dd_pct)
        if killm > 0.0:
            pk = v
            for j in range(max(t - killw + 1, rel), t):
                if eq[j] > pk:
                    pk = eq[j]
            if v < pk * (1.0 - killm):
                for s in range(S):
                    for k in range(K):
                        if units[s, k] > 0.0:
                            proceeds = units[s, k] * lastc[k] * (1.0 - slip[k]) * (1.0 - pens) * (1.0 - fee)
                            cash += proceeds
                            if nl < maxlog:
                                lg[nl, 0] = ebar[s, k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = 7.0
                                lg[nl, 4] = proceeds / amt0[s, k] - 1.0; lg[nl, 5] = amt0[s, k]
                                lg[nl, 6] = entry[s, k]; lg[nl, 7] = eq0[s, k]; lg[nl, 8] = s; nl += 1
                            units[s, k] = 0.0
                            okc[k] = t + cdl[s]
                v = cash; w = 0.0
                lock = max(lock, t + 1 + klock); rel = t
        eq[t] = v; eq_prev = v; inv[t] = w / v if v > 0.0 else 0.0
    for s in range(S):
        for k in range(K):
            if units[s, k] > 0.0 and nl < maxlog:
                proceeds = units[s, k] * lastc[k] * (1.0 - slip[k]) * (1.0 - pens) * (1.0 - fee)
                lg[nl, 0] = ebar[s, k]; lg[nl, 1] = n - 1; lg[nl, 2] = k; lg[nl, 3] = 9.0
                lg[nl, 4] = proceeds / amt0[s, k] - 1.0; lg[nl, 5] = amt0[s, k]; lg[nl, 6] = entry[s, k]
                lg[nl, 7] = eq0[s, k]; lg[nl, 8] = s; nl += 1
    return eq, lg[:nl].copy(), inv


def frun_r(P, idx, d, pen=0.0, pens=0.0, mtp_now=False, cd_add=0, liquidate=False, t0=None):
    T = [dict(F.TIERA, **t) for t in d["tiers"]]
    r = dict(F.OFFRISK); r.update({k: v for k, v in d.items() if k != "tiers"})
    O = P["o"].values
    n, K = O.shape
    refs = sorted({t["ref"] for t in T})
    R = np.ascontiguousarray(np.stack([F.ref(P, w) for w in refs]))
    rid = np.array([refs.index(t["ref"]) for t in T], np.int64)
    sl = np.array([E.SLIP_MAJOR if s in E.H19 else E.SLIP_OTHER for s in P["o"].columns]) if liquidate \
        else np.full(K, E.SLIP_MAJOR)
    lastb = V.last_bars(P) if liquidate else np.full(K, -1, np.int64)
    g = F.guard(P, r["guard"])
    gmS = np.ascontiguousarray(np.stack([g * F.vol_mask(P, t["vol"]) for t in T]))
    tpS = np.ascontiguousarray(np.stack([F.tp_table(P, t["tp"]) for t in T]))
    f = lambda key, dt=np.float64: np.array([t[key] for t in T], dt)   # noqa: E731
    rrv = np.array([r.get("rr", 0.0)] * len(T)); lo = np.array([r.get("glo", 0.015)] * len(T)); hi = np.array([r.get("ghi", 0.06)] * len(T))
    i0 = 0 if t0 is None else int(np.searchsorted(idx.values, np.datetime64(t0)))
    eq, lg, inv = fkr(O, P["h"].values, P["l"].values, P["cf"].values, gmS, R, rid, lastb, f("depth"), tpS,
                      f("hm", np.int64), np.maximum(f("cdw", np.int64) + cd_add, 0),
                      np.maximum(f("cdl", np.int64) + cd_add, 0), f("fs"), int(r["nslots"]), int(r["nbid"]), float(E.EPS),
                      float(E.FEE), sl, float(pen), float(pens), bool(mtp_now), i0, float(r["hs"]), float(r["zcap"]),
                      float(r["killm"]), int(r["killw"]), int(r["klock"]), int(r["cln"]), int(r["clT"]), rrv, lo, hi)
    return pd.Series(eq, idx), pd.DataFrame(lg, columns=LOGC), pd.Series(inv, idx)


def design_scores_r(d):
    """2021-23 only: n, w, cm, cp, bt, ls (same as F.design_scores)."""
    syms, idx, P = V.market(False)
    out = {}
    runs = {"n": {}, "w": dict(pen=F.PEN, pens=F.PEN), "cm": dict(cd_add=-1), "cp": dict(cd_add=1),
            "bt": dict(mtp_now=True, cd_add=1), "ls": dict(t0=F.LATE)}
    for tag, kw in runs.items():
        e, lg, inv = frun_r(P, idx, d, **kw)
        per = (("d", F.LATE, F.T24),) if tag == "ls" else F.DPER
        r = V.summarize(e, lg, inv, idx, per)
        out[f"{tag}_wr"], out[f"{tag}_mo"], out[f"{tag}_dd"] = r["d_wr"], r["d_mo"], r["d_dd"]
        if tag == "n":
            out["n_tpm"] = r["d_tpm"]
    out["robust"] = float(np.mean([out[f"{t}_mo"] for t in ("n", "cm", "cp", "bt", "ls")]))
    out["rmin"] = float(np.min([out[f"{t}_mo"] for t in ("n", "cm", "cp", "bt", "ls")]))
    return out


def evaluate_r(d, worse=False, stress=False, holdout=False):
    dead = stress
    syms, idx, P = V.market_holdout(dead) if holdout else V.market(dead)
    kw = {}
    if worse or stress:
        kw.update(pen=F.PEN, pens=F.PEN)
    if stress:
        kw.update(mtp_now=True, cd_add=1)
    if dead:
        kw["liquidate"] = True
    e, lg, inv = frun_r(P, idx, d, **kw)
    return V.summarize(e, lg, inv, idx, V.HOLD if holdout else V.MAIN), e, lg
