"""Ensemble + guardrail engine (agent 3). A copy of engine.gkernel extended with capital sleeves, dynamic sizing and
execution-level circuit breakers. Every new switch is OFF by default, and then ekernel == engine.gkernel exactly
(equity and trade log, see ens_check.py); with liquidate it == w90_wr90_lib.wrun.

Sleeves (S >= 1): each sleeve runs its own rules (depth, N, fmax, tp, part, tp2, hm, cd) on its own cash and its own
positions; the account equity is the sum. A coin may be held by two sleeves at once. w0 = starting weights.
  rb=1  monthly rebalance: at the first hour of each month, free cash is moved between sleeves toward the target
        weights (only cash moves; open positions are never sold for it). rb=0: no rebalance (weights drift).
Confluence (one sleeve): part < 1 sells `part` at +tp, the rest at +tp2, anything left at the time stop (gkernel).

Dynamic sizing (multiplies the gkernel order size; never above 100% of the sleeve's money):
  kf > 0  fractional Kelly, computed ONLY from the sleeve's last kwin closed trades before each entry hour
          (trades that close in hour t are added after hour t's entries). W = win share, G = avg win, L = |avg loss|
          (per-trade returns). kmode 0 ("proportional loss" Kelly, the right one for a stake that loses L, not all of
          it): f* = W/L - (1-W)/G.  kmode 1 (textbook bet that loses the whole stake): f* = W - (1-W)/(G/L).
          size = clip(kf*f*, kfloor, 1); fewer than kmin closed trades -> kwarm. No loss yet -> 1; no win yet -> kfloor.
  vmult   (n,K) size multiplier precomputed outside (vol target: min(1, tv / trailing 7d vol), known at the open).
Guardrails:
  hs      hard stop-loss per trade, as a fraction of the entry price. hsmode 1 (default, a resting stop order on the
          exchange): from the hour after the fill, sold at min(open, stop) when the hour's low touches the stop
          (gaps fill at the open, - slippage); checked BEFORE the take-profit in the same hour (worst case). In the
          fill hour itself: sold at the close if the close is at/below the stop. hsmode 0: hourly close check only.
  zcap    per-trade risk cap: order money x hs <= zcap x account equity (orders are cut down to fit; with hs off,
          the full order money counts as at risk, so the order is cut to zcap x equity).
  killd   daily kill-switch: at each hourly close, if account equity < (1-killd) x the highest equity of the current
          UTC day (incl. the previous day's last close), every position of every sleeve is sold at that close
          (- slippage) and no order is placed until the next UTC day.
  killm   rolling kill-switch: same, vs the highest hourly close of the last killw hours (after the last release);
          locks entries for klock hours; after the release the reference peak restarts from the release hour.
  cln,clT consecutive-loss cooldown: after cln losing trades in a row in a sleeve, that sleeve places no new order
          for clT hours (positions it holds are still managed). The streak restarts after a pause or a win.
  gm      (n,K) entry mask (crash guard x ATR-spike block), 0 = no order on that coin that hour.
Log columns: eb, xb, k, why, ret, amt, entry, eq0 (account equity when sized), sleeve.
why: 0 take-profit, 1 time stop, 3 second target, 4 hard stop, 5 collapse coin cut, 7 kill-switch flatten, 9 open at end.
"""
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402

LOGC = ["eb", "xb", "k", "why", "ret", "amt", "entry", "eq0", "sl"]


@njit(cache=True)
def _close(lg, nl, eb, t, k, why, got, amt0, entry, eq0, s):
    if nl < lg.shape[0]:
        lg[nl, 0] = eb; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = why
        lg[nl, 4] = got / amt0 - 1.0; lg[nl, 5] = amt0; lg[nl, 6] = entry; lg[nl, 7] = eq0; lg[nl, 8] = s
        return nl + 1
    return nl


@njit(cache=True)
def ekernel(o, h, l, c, ref, gm, vmult, hod, mstart, lastb,
            depth, nord, fmax, tpa, part, tp2, hm, cd, w0, rb,
            eps, fee, slip, pen, pens, tpwait, mtp_now,
            kf, kmode, kwin, kmin, kwarm, kfloor,
            hs, hsmode, zcap, killd, killm, killw, klock, cln, clT):
    n, K = c.shape
    S = depth.shape[0]
    units = np.zeros((S, K)); entry = np.zeros((S, K)); amt0 = np.zeros((S, K)); got = np.zeros((S, K))
    tpp = np.zeros((S, K)); eq0 = np.zeros((S, K))
    stage = np.zeros((S, K), np.int64); sbar = np.zeros((S, K), np.int64)
    ebar = np.full((S, K), -1, np.int64); okbar = np.zeros((S, K), np.int64); mkt = np.zeros((S, K), np.int64)
    cash = np.empty(S)
    for s in range(S):
        cash[s] = w0[s]
    seqp = cash.copy()                     # sleeve equity at the previous close
    rbuf = np.zeros((S, max(kwin, 1))); rcnt = np.zeros(S, np.int64)
    streak = np.zeros(S, np.int64); pause = np.zeros(S, np.int64)
    lastc = np.full(K, np.nan); dist = np.empty(K)
    eq = np.empty(n); eq_prev = 0.0
    for s in range(S):
        eq_prev += cash[s]
    eqh = np.empty(n)
    kill_until = 0; rel = 0; dpk = eq_prev; nkd = 0; nkm = 0
    maxlog = 600000; lg = np.zeros((maxlog, 9)); nl = 0
    for t in range(n):
        if hod[t] == 0:
            dpk = eq_prev
        # ---------------- monthly rebalance of free cash between sleeves
        if rb == 1 and S > 1 and mstart[t] == 1:
            need = np.zeros(S); give = np.zeros(S); G = 0.0; NE = 0.0
            for s in range(S):
                d = w0[s] * eq_prev - seqp[s]
                if d < 0.0:
                    give[s] = min(cash[s], -d); G += give[s]
                else:
                    need[s] = d; NE += need[s]
            moved = min(G, NE)
            if moved > 1e-12:
                for s in range(S):
                    if give[s] > 0.0:
                        cash[s] -= give[s] * moved / G
                    if need[s] > 0.0:
                        cash[s] += need[s] * moved / NE
        # ---------------- entries
        for s in range(S):
            if t < kill_until or t < pause[s]:
                continue
            for k in range(K):
                dist[k] = 1e9
                if units[s, k] > 0.0 or t < okbar[s, k] or np.isnan(o[t, k]) or np.isnan(ref[t, k]) or gm[t, k] <= 0.0:
                    continue
                dist[k] = o[t, k] / (ref[t, k] * (1.0 - depth[s])) - 1.0
            order = np.argsort(dist)
            nsel = 0
            for q in range(K):
                if dist[order[q]] < 1e8 and nsel < nord[s]:
                    nsel += 1
            if nsel > 0 and cash[s] > 1e-9:
                kel = 1.0
                if kf > 0.0:
                    m = min(rcnt[s], kwin)
                    if m < kmin:
                        kel = kwarm
                    else:
                        nw = 0; sw = 0.0; sl_ = 0.0
                        for j in range(m):
                            r = rbuf[s, j]
                            if r > 0.0:
                                nw += 1; sw += r
                            else:
                                sl_ += r
                        W = nw / m
                        if nw == 0:
                            fs = 0.0
                        elif nw == m or sl_ >= 0.0:
                            fs = 1e9
                        else:
                            G_ = sw / nw; L_ = -sl_ / (m - nw)
                            if kmode == 0:
                                fs = W / L_ - (1.0 - W) / G_
                            else:
                                fs = W - (1.0 - W) / (G_ / L_)
                        kel = min(1.0, max(kfloor, kf * fs))
                per = min(fmax[s] * seqp[s], cash[s] / nsel)
                for q in range(nsel):
                    k = order[q]
                    lim = ref[t, k] * (1.0 - depth[s])
                    fill = -1.0; ismkt = 0
                    if o[t, k] <= lim:
                        fill = o[t, k] * (1.0 + slip[k]) * (1.0 + pen); ismkt = 1
                    elif l[t, k] <= lim * (1.0 - eps):
                        fill = lim * (1.0 + pen)
                    if fill > 0.0 and cash[s] > 1e-9:
                        mkt[s, k] = ismkt if mtp_now else 0
                        a = min(per * gm[t, k] * vmult[t, k] * kel, cash[s])
                        if zcap > 0.0:
                            risk = hs if hs > 0.0 else 1.0
                            a = min(a, zcap * eq_prev / risk)
                        cash[s] -= a; units[s, k] = a * (1.0 - fee) / fill
                        entry[s, k] = fill; amt0[s, k] = a; ebar[s, k] = t; got[s, k] = 0.0; eq0[s, k] = eq_prev
                        tpp[s, k] = fill * (1.0 + tpa[s]); stage[s, k] = 0; sbar[s, k] = 0
        # ---------------- exits
        for s in range(S):
            for k in range(K):
                if units[s, k] <= 0.0 or np.isnan(o[t, k]):
                    continue
                px = -1.0; why = 0.0
                if hs > 0.0:
                    sp = entry[s, k] * (1.0 - hs)
                    if ebar[s, k] < t and hsmode == 1:
                        if o[t, k] <= sp:
                            px = o[t, k] * (1.0 - slip[k]); why = 4.0
                        elif l[t, k] <= sp:
                            px = sp * (1.0 - slip[k]); why = 4.0
                if px < 0.0 and (ebar[s, k] < t or mkt[s, k] == 1):
                    if stage[s, k] == 0:
                        if ebar[s, k] == t:
                            if h[t, k] >= tpp[s, k] * (1.0 + eps):
                                px = tpp[s, k]
                        elif t - ebar[s, k] < tpwait:
                            pass
                        elif o[t, k] >= tpp[s, k]:
                            px = o[t, k]
                        elif h[t, k] >= tpp[s, k] * (1.0 + eps):
                            px = tpp[s, k]
                        if px > 0.0 and part[s] < 1.0:
                            qq = units[s, k] * part[s]
                            p1 = qq * px * (1.0 - pens) * (1.0 - fee)
                            cash[s] += p1; got[s, k] += p1; units[s, k] -= qq
                            stage[s, k] = 1; sbar[s, k] = t
                            px = -1.0
                    elif sbar[s, k] < t:
                        if tp2[s] > 0.0:
                            p2 = entry[s, k] * (1.0 + tp2[s])
                            if o[t, k] >= p2:
                                px = o[t, k]; why = 3.0
                            elif h[t, k] >= p2 * (1.0 + eps):
                                px = p2; why = 3.0
                    if px < 0.0 and t - ebar[s, k] >= hm[s] - mkt[s, k]:
                        px = c[t, k] * (1.0 - slip[k]); why = 1.0
                if px < 0.0 and hs > 0.0 and (hsmode == 0 or ebar[s, k] == t) and c[t, k] <= entry[s, k] * (1.0 - hs):
                    px = c[t, k] * (1.0 - slip[k]); why = 4.0
                if px > 0.0:
                    pr = units[s, k] * px * (1.0 - pens) * (1.0 - fee); cash[s] += pr; got[s, k] += pr
                    nl = _close(lg, nl, ebar[s, k], t, k, why, got[s, k], amt0[s, k], entry[s, k], eq0[s, k], s)
                    units[s, k] = 0.0; okbar[s, k] = t + cd[s]
                    r = got[s, k] / amt0[s, k] - 1.0
                    rbuf[s, rcnt[s] % max(kwin, 1)] = r; rcnt[s] += 1
                    if r <= 0.0:
                        streak[s] += 1
                        if cln > 0 and streak[s] >= cln:
                            pause[s] = t + 1 + clT; streak[s] = 0
                    else:
                        streak[s] = 0
            for k in range(K):
                if units[s, k] > 0.0 and lastb[k] == t:
                    pr = units[s, k] * c[t, k] * (1.0 - slip[k]) * (1.0 - pens) * (1.0 - fee)
                    cash[s] += pr; got[s, k] += pr
                    nl = _close(lg, nl, ebar[s, k], t, k, 5.0, got[s, k], amt0[s, k], entry[s, k], eq0[s, k], s)
                    units[s, k] = 0.0; okbar[s, k] = t + cd[s]
                    r = got[s, k] / amt0[s, k] - 1.0
                    rbuf[s, rcnt[s] % max(kwin, 1)] = r; rcnt[s] += 1
                    if r <= 0.0:
                        streak[s] += 1
                        if cln > 0 and streak[s] >= cln:
                            pause[s] = t + 1 + clT; streak[s] = 0
                    else:
                        streak[s] = 0
        for k in range(K):
            if not np.isnan(c[t, k]):
                lastc[k] = c[t, k]
        v = 0.0
        for s in range(S):
            sv = cash[s]
            for k in range(K):
                if units[s, k] > 0.0:
                    sv += units[s, k] * lastc[k]
            seqp[s] = sv; v += sv
        # ---------------- kill-switches (account level, at the hourly close)
        if killd > 0.0 or killm > 0.0:
            if v > dpk:
                dpk = v
            fire = 0
            if killd > 0.0 and v < dpk * (1.0 - killd):
                fire = 1
            pk = v
            if killm > 0.0:
                j0 = max(t - killw + 1, rel)
                for j in range(j0, t):
                    if eqh[j] > pk:
                        pk = eqh[j]
                if v < pk * (1.0 - killm):
                    fire = 2 if fire == 0 else 3
            if fire > 0:
                v = 0.0
                for s in range(S):
                    for k in range(K):
                        if units[s, k] > 0.0:
                            pxk = lastc[k] * (1.0 - slip[k])
                            pr = units[s, k] * pxk * (1.0 - pens) * (1.0 - fee); cash[s] += pr; got[s, k] += pr
                            nl = _close(lg, nl, ebar[s, k], t, k, 7.0, got[s, k], amt0[s, k], entry[s, k], eq0[s, k], s)
                            units[s, k] = 0.0; okbar[s, k] = t + cd[s]
                            r = got[s, k] / amt0[s, k] - 1.0
                            rbuf[s, rcnt[s] % max(kwin, 1)] = r; rcnt[s] += 1
                            if r <= 0.0:
                                streak[s] += 1
                                if cln > 0 and streak[s] >= cln:
                                    pause[s] = t + 1 + clT; streak[s] = 0
                            else:
                                streak[s] = 0
                    seqp[s] = cash[s]; v += cash[s]
                if fire == 1 or fire == 3:
                    kill_until = max(kill_until, t + 24 - hod[t]); nkd += 1
                if fire >= 2:
                    kill_until = max(kill_until, t + 1 + klock); nkm += 1
                    rel = kill_until
                dpk = v
        eq[t] = v; eqh[t] = v; eq_prev = v
    for s in range(S):
        for k in range(K):
            if units[s, k] > 0.0:
                pr = units[s, k] * lastc[k] * (1.0 - slip[k]) * (1.0 - pens) * (1.0 - fee)
                nl = _close(lg, nl, ebar[s, k], n - 1, k, 9.0, got[s, k] + pr, amt0[s, k], entry[s, k], eq0[s, k], s)
    return eq, lg[:nl].copy(), nkd, nkm


# ------------------------------------------------------------------ masks / features (known at each hour's open)
_cache = {}


def guard_mask(P, g):
    """Crash guard (w90_wr90_lib.guard_mask): last close no more than g below the highest close of the last 168h."""
    c = P["cf"]
    hi = c.rolling(168, min_periods=1).max()
    return ((c / hi - 1) >= -g).shift(1, fill_value=True).values.astype(float)


def atr_mask(P, k, win=720):
    """ATR-spike block: 0 when the last closed hour's range (h-l)/c > k x the median hourly range of the previous
    `win` hours (30 days). Unknown median (first 10 days of a coin) -> allowed."""
    key = (id(P["c"]), "rng", win)
    if key not in _cache:
        rng = (P["h"] - P["l"]) / P["cf"]
        last = rng.shift(1)
        med = rng.rolling(win, min_periods=240).median().shift(2)    # the 30 days before the last hour
        _cache[key] = (last, med)
    last, med = _cache[key]
    blk = (last > k * med).values
    return np.where(blk, 0.0, 1.0)


def vol_mult(P, tv, w=168):
    """Volatility-adjusted size: min(1, tv / daily vol of the last w closed hours); unknown vol -> 1."""
    r = np.log(P["cf"]).diff()
    vol = (r.rolling(w, min_periods=w // 2).std() * np.sqrt(24)).shift(1)
    m = (tv / vol).clip(upper=1.0)
    return np.ascontiguousarray(m.fillna(1.0).values)


def last_bars(P):
    o = P["o"]
    out = np.full(o.shape[1], -1, np.int64)
    for j, s in enumerate(o.columns):
        if s in E.DEAD:
            v = np.flatnonzero(~np.isnan(o[s].values))
            out[j] = int(v[-1])
    return out


# ------------------------------------------------------------------ runner
MODE90 = dict(depth=0.14, N=1, fmax=1.0, tp=0.025, part=1.0, tp2=0.0, hm=96, cd=3)
MODE10 = dict(depth=0.14, N=1, fmax=1.0, tp=0.05, part=1.0, tp2=0.0, hm=48, cd=24)


def erun(P, idx, sleeves, weights=None, rb=0, guard=0.35, atr_k=0.0, tv=0.0, eps=E.EPS, fee=E.FEE,
         slip=E.SLIP_MAJOR, pen=0.0, pens=0.0, tpwait=1, mtp_now=False, cd_add=0, liquidate=False,
         kf=0.0, kmode=0, kwin=30, kmin=10, kwarm=1.0, kfloor=0.2,
         hs=0.0, hsmode=1, zcap=0.0, killd=0.0, killm=0.0, killw=720, klock=168, cln=0, clT=0, smult=None):
    """sleeves: list of dicts (depth, N, fmax, tp, part, tp2, hm, cd). Returns (equity, log, n daily kills, n rolling kills)."""
    O = P["o"].values; n, K = O.shape
    S = len(sleeves)
    w = np.full(S, 1.0 / S) if weights is None else np.asarray(weights, np.float64)
    if np.ndim(slip) == 0:
        sl = np.array([slip if s in E.H19 else E.SLIP_OTHER for s in P["o"].columns]) if liquidate else np.full(K, slip)
    else:
        sl = np.asarray(slip, np.float64)
    gm = np.ones((n, K))
    if guard > 0:
        gm = gm * guard_mask(P, guard)
    if atr_k > 0:
        gm = gm * atr_mask(P, atr_k)
    if smult is not None:
        gm = gm * E.full(n, K, smult)
    gm = np.ascontiguousarray(gm)
    vm = vol_mult(P, tv) if tv > 0 else np.ones((n, K))
    hod = np.asarray(idx.hour, np.int64)
    ms = np.asarray((idx.day == 1) & (idx.hour == 0), np.int64)
    lastb = last_bars(P) if liquidate else np.full(K, -1, np.int64)
    a = lambda key, dt=np.float64: np.array([sv[key] for sv in sleeves], dt)  # noqa: E731
    eq, lg, nkd, nkm = ekernel(O, P["h"].values, P["l"].values, P["cf"].values, P["ref"].values, gm, vm, hod, ms, lastb,
                               a("depth"), a("N", np.int64), a("fmax"), a("tp"), a("part"), a("tp2"),
                               a("hm", np.int64), a("cd", np.int64) + int(cd_add), w, int(rb),
                               float(eps), float(fee), sl, float(pen), float(pens), int(tpwait), bool(mtp_now),
                               float(kf), int(kmode), int(kwin), int(kmin), float(kwarm), float(kfloor),
                               float(hs), int(hsmode), float(zcap), float(killd), float(killm), int(killw), int(klock),
                               int(cln), int(clT))
    return pd.Series(eq, idx), pd.DataFrame(lg, columns=LOGC), int(nkd), int(nkm)


# ------------------------------------------------------------------ metrics
PER = (("21-23", pd.Timestamp("2021-01-01"), pd.Timestamp("2024-01-01")),
       ("24-26", pd.Timestamp("2024-01-01"), None),
       ("25-", pd.Timestamp("2025-01-01"), None))


def period(e, a, b):
    """Compounded %/month and worst drop from a to b, starting from the equity just before a (w90_wr90_lib.period)."""
    i0 = int(np.searchsorted(e.index.values, np.datetime64(a), side="left"))
    i1 = len(e) if b is None else int(np.searchsorted(e.index.values, np.datetime64(b), side="left"))
    start = e.values[i0 - 1] if i0 > 0 else 1.0
    x = np.concatenate([[1.0], e.values[i0:i1] / start])
    yrs = ((e.index[i1 - 1] + pd.Timedelta(hours=1)) - e.index[i0]).total_seconds() / (365.25 * 86400)
    mult = float(x[-1])
    mo = mult ** (1.0 / (12 * yrs)) - 1.0 if mult > 0 else -1.0
    dd = float((x / np.maximum.accumulate(x) - 1.0).min())
    seg = pd.Series(x[1:], e.index[i0:i1])
    mret = seg.resample("ME").last().pct_change()
    mret.iloc[0] = seg.resample("ME").last().iloc[0] / 1.0 - 1.0
    dret = seg.resample("D").last()
    dret = pd.concat([pd.Series([1.0]), dret]).pct_change().dropna()
    return mult, mo, dd, float(mret.min()), float(dret.min())


def pf(x):
    g, l_ = x[x > 0].sum(), -x[x <= 0].sum()
    return float(g / l_) if l_ > 0 else float("inf")


def summarize(e, lg, idx):
    """Per period: mult, %/mo, max drop, worst month, worst day, trades (by entry hour), win rate (open positions
    marked at the end count), profit factor on account P&L (ret x money / account equity at entry, as
    ens_audit_modes.py), worst trade as % of the account."""
    r = {}
    tb = idx[lg.eb.astype(int).values] if len(lg) else pd.DatetimeIndex([])
    acct = (lg.ret * lg.amt / lg.eq0).values if len(lg) else np.zeros(0)
    for tag, a, b in PER:
        m, mo, dd, wm, wd = period(e, a, b)
        sel = (tb >= a) & ((tb < b) if b is not None else True)
        x = lg[sel]
        r.update({f"{tag} mult": m, f"{tag} %mo": mo * 100, f"{tag} DD": dd * 100, f"{tag} worstmo": wm * 100,
                  f"{tag} worstday": wd * 100, f"{tag} n": int(len(x)),
                  f"{tag} WR": float((x.ret > 0).mean()) * 100 if len(x) else np.nan,
                  f"{tag} PF": pf(acct[sel]) if sel.any() else np.nan,
                  f"{tag} worsttr": float(acct[sel].min()) * 100 if sel.any() else np.nan,
                  f"{tag} size": float((x.amt / x.eq0).mean()) * 100 if len(x) else np.nan})
    return r


def market(dead=False):
    import study as ST   # study.market (dead=True adds LUNA/FTT/SRM cut at their collapse)
    return ST.market(dead)


PANELS = (("19", False), ("22", True))      # 22 = + LUNA/FTT/SRM, held positions sold at the coin's last bar


def run_both(name, sleeves, **kw):
    """Run one design on the 19-coin panel and on the 22-coin collapse panel (liquidate=True). Returns rows."""
    rows = []
    for tag, dead in PANELS:
        syms, idx, P = market(dead)
        e, lg, nkd, nkm = erun(P, idx, sleeves, liquidate=dead, **kw)
        r = summarize(e, lg, idx)
        r.update(design=name, panel=tag, kills_d=nkd, kills_m=nkm)
        rows.append(r)
    return rows


SHOW = ["%mo", "WR", "DD", "PF", "worstmo", "worsttr", "n", "size"]


def table(df, periods=("21-23", "24-26", "25-"), cols=SHOW):
    """Markdown-ish fixed-width table: one line per design x panel."""
    out = []
    head = f"{'design':44s} {'pan':3s} " + " | ".join(
        f"{p}: " + " ".join(f"{c:>7s}" for c in cols) for p in periods)
    out.append(head)
    for _, r in df.iterrows():
        out.append(f"{r['design'][:44]:44s} {r['panel']:3s} " + " | ".join(
            f"{p}: " + " ".join(f"{r[f'{p} {c}']:7.1f}" if c != "n" else f"{int(r[f'{p} {c}']):7d}" for c in cols)
            for p in periods))
    return "\n".join(out)
