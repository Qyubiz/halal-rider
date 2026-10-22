"""m10 "ladder": win almost every trade by averaging down within strict limits (spot only, fully funded).

lkernel = engine.gkernel (full take-profit only) with an m-leg buy ladder per coin:
  legs      leg j rests at ref x (1 - dl[j]), ref = the coin's 24h high close known at the leg-1 order hour
            (dl ascending, e.g. 10/15/20/25/30%), money = wl[j] x the coin's budget (wl sums to 1).
  budget    a = min(fmax x equity, cash / nsel) when leg 1 is placed (fmax caps the coin's TOTAL, all legs).
  H         at most H coins held at once (new entries per hour <= min(N, H - held)).
  lazy      0 = "reserve": the whole budget is locked in resting orders when leg 1 fills; unfilled legs keep
                resting at their fixed prices until they fill or the position is sold (then the cash comes back).
            1 = "lazy": in the entry hour all legs rest (funded by the budget); from the next hour only the NEXT
                unfilled leg is re-placed each hour from FREE cash (held ladders first, then new entries), locked
                for that hour only; if there is not enough free cash the leg is smaller. A level already under at
                the hour open is bought at market. Every order is fully funded when placed.
  take-profit  sell everything at avg x (1 + tp), avg = money spent / coins bought (gross of fees), placed at the
            hourly cycle after the latest fill (none in a leg's fill hour, the bot re-places it at the next check).
  beh, be   after beh hours (0 = off) the target drops to avg x (1 + be) (placed at the next check): a slow loser
            that comes back to its average still exits in profit.
  ml        max-loss exit: sold at an hourly close <= avg x (1 - ml) (market, - slippage). mlall=1: only once all
            legs are filled. 0 = off.
  hm        time stop in hours from the first fill (sold at that close). Huge = off.
  cd        cooldown after any exit.
  gm        (n, K) entry mask (crash guard): 0 = no new ladder on that coin that hour (held ladders unaffected).
  lastb     last data hour of a collapse coin (LUNA/FTT/SRM), -1 = none: a position still held is sold at that close.
Costs: 0.1% fee per side, market buys/sells 0.02% slippage, limit fills need the low (high) to trade 0.1% through,
pen / pens = extra penalty on every buy / sell fill ("fills 0.2% worse" = pen = pens = 0.002).
Nothing looks ahead: orders are decided at the hour open from closed candles (ref is shifted one hour).
Positions still open at the end are logged with why=9 at their mark (last close - slippage - fee), so an open loser
counts as a loss. why: 0 take-profit, 1 time stop, 4 max-loss, 5 collapse-coin cut, 9 open at end.

With m=1 (one leg, wl=[1]), H large, lazy=0, ml=0, beh=0 it is engine.run / w90_wr90_lib.wrun exactly; with two legs
in reserve mode it is r3_ladder.lrun exactly (m10_ladder_check.py proves both).
"""
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402

OUT = os.environ.get("PB_OUT", os.path.expanduser("~/pb_out_m10/ladder"))
os.makedirs(OUT, exist_ok=True)
T_D0, T_T0, T_S0, T_END = (pd.Timestamp("2021-01-01"), pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01"), None)
LOGC = ["eb", "xb", "k", "why", "ret", "amt", "entry", "eq0", "nf", "bud", "mkt"]
STRESS_PEN = 0.002


@njit(cache=True)
def lkernel(o, h, l, c, ref, gm, dl, wl, H, N, fmax, tp, hm, cd, eps, fee, slip, pen, pens, tpwait, mtp_now,
            lazy, ml, mlall, beh, be, lastb):
    n, K = c.shape
    m = dl.shape[0]
    units = np.zeros(K); amt0 = np.zeros(K); inv = np.zeros(K); nf = np.zeros(K, np.int64)
    refe = np.zeros(K); bud = np.zeros(K); res = np.zeros(K); lk = np.zeros(K)
    tpp = np.zeros(K); eq0 = np.zeros(K); rawm = np.zeros(K); avgp = np.zeros(K)
    ebar = np.full(K, -1, np.int64); lbar = np.full(K, -1, np.int64); okbar = np.zeros(K, np.int64)
    mkt = np.zeros(K, np.int64); lowered = np.zeros(K, np.int64)
    lastc = np.full(K, np.nan); dist = np.empty(K)
    cash = 1.0; eq = np.empty(n); eq_prev = 1.0
    maxlog = 600000; lg = np.zeros((maxlog, 11)); nl = 0
    for t in range(n):
        # ---------------- lazy mode: fund the deeper legs of held ladders from free cash for this hour
        if lazy == 1:
            for k in range(K):
                if units[k] > 0.0 and nf[k] < m and ebar[k] < t and not np.isnan(o[t, k]):
                    x = min(wl[nf[k]] * bud[k], cash)          # the NEXT leg only
                    if x > 1e-12:
                        lk[k] = x; cash -= x
        # ---------------- entries (leg 1, with the deeper legs resting from the same hour)
        nheld = 0
        for k in range(K):
            if units[k] > 0.0:
                nheld += 1
        slots = min(N, H - nheld)
        for k in range(K):
            dist[k] = 1e9
            if units[k] > 0.0 or t < okbar[k] or np.isnan(o[t, k]) or np.isnan(ref[t, k]) or gm[t, k] <= 0.0:
                continue
            dist[k] = o[t, k] / (ref[t, k] * (1.0 - dl[0])) - 1.0
        order = np.argsort(dist)
        nsel = 0
        for q in range(K):
            if dist[order[q]] < 1e8 and nsel < slots:
                nsel += 1
        if nsel > 0 and cash > 1e-9:
            per = min(fmax * eq_prev, cash / nsel)
            for q in range(nsel):
                k = order[q]
                lim = ref[t, k] * (1.0 - dl[0])
                fill = -1.0; ismkt = 0
                if o[t, k] <= lim:
                    fill = o[t, k] * (1.0 + slip[k]) * (1.0 + pen); ismkt = 1
                elif l[t, k] <= lim * (1.0 - eps):
                    fill = lim * (1.0 + pen)
                if fill > 0.0 and cash > 1e-9:
                    a = min(per, cash)
                    cash -= a
                    b0 = a * wl[0]
                    units[k] = b0 * (1.0 - fee) / fill; inv[k] = b0 / fill; amt0[k] = b0; nf[k] = 1
                    bud[k] = a; refe[k] = ref[t, k]; ebar[k] = t; lbar[k] = t; eq0[k] = eq_prev
                    mkt[k] = ismkt if mtp_now else 0; rawm[k] = ismkt; lowered[k] = 0; avgp[k] = fill
                    for j in range(1, m):                       # deeper legs rest from the hour start
                        lj = refe[k] * (1.0 - dl[j])
                        fj = -1.0
                        if o[t, k] <= lj:
                            fj = o[t, k] * (1.0 + slip[k]) * (1.0 + pen)
                        elif l[t, k] <= lj * (1.0 - eps):
                            fj = lj * (1.0 + pen)
                        if fj < 0.0:
                            break
                        bj = a * wl[j]
                        units[k] += bj * (1.0 - fee) / fj; inv[k] += bj / fj; amt0[k] += bj; nf[k] += 1
                    left = a - amt0[k]
                    if left < 1e-15:
                        left = 0.0
                    if lazy == 1:
                        lk[k] = left                          # locked this hour only, released at the hour's end
                    else:
                        res[k] = left
                    if nf[k] > 1:
                        avgp[k] = amt0[k] / inv[k]
                    tpp[k] = avgp[k] * (1.0 + tp)
        # ---------------- held ladders: deeper legs, then exits
        for k in range(K):
            if units[k] > 0.0 and (ebar[k] < t or mkt[k] == 1) and not np.isnan(o[t, k]):
                if ebar[k] < t and nf[k] < m:
                    avail = lk[k] if lazy == 1 else res[k]
                    got = False
                    for j in range(nf[k], m):
                        if avail <= 1e-12:
                            break
                        lj = refe[k] * (1.0 - dl[j])
                        fj = -1.0
                        if lazy == 1 and o[t, k] <= lj:          # re-placed at the hour open: already under -> market
                            fj = o[t, k] * (1.0 + slip[k]) * (1.0 + pen)
                        elif l[t, k] <= lj * (1.0 - eps):        # resting limit, traded through
                            fj = lj * (1.0 + pen)
                        if fj < 0.0:
                            break
                        bj = min(wl[j] * bud[k], avail)
                        avail -= bj
                        units[k] += bj * (1.0 - fee) / fj; inv[k] += bj / fj; amt0[k] += bj; nf[k] += 1
                        got = True
                    if lazy == 1:
                        lk[k] = avail
                    else:
                        res[k] = avail
                    if got:
                        lbar[k] = t
                        avgp[k] = amt0[k] / inv[k]
                        tpp[k] = avgp[k] * (1.0 + (be if lowered[k] == 1 else tp))
                avg = avgp[k]
                px = -1.0; why = 0.0
                if ebar[k] == t:                         # market buy, take-profit placed at once (mtp_now)
                    if nf[k] == 1 and h[t, k] >= tpp[k] * (1.0 + eps):
                        px = tpp[k]
                elif t - lbar[k] < tpwait:
                    pass
                elif o[t, k] >= tpp[k]:
                    px = o[t, k]
                elif h[t, k] >= tpp[k] * (1.0 + eps):
                    px = tpp[k]
                age = t - ebar[k] + mkt[k]
                if px < 0.0 and ml > 0.0 and ebar[k] < t and (mlall == 0 or nf[k] == m) and c[t, k] <= avg * (1.0 - ml):
                    px = c[t, k] * (1.0 - slip[k]); why = 4.0
                if px < 0.0 and age >= hm:
                    px = c[t, k] * (1.0 - slip[k]); why = 1.0
                if px < 0.0 and beh > 0 and age >= beh and lowered[k] == 0:
                    lowered[k] = 1                       # next hourly check: target -> avg x (1 + be)
                    b = avg * (1.0 + be)
                    if b < tpp[k]:
                        tpp[k] = b
                if px > 0.0:
                    proceeds = units[k] * px * (1.0 - pens) * (1.0 - fee); cash += proceeds
                    if nl < maxlog:
                        lg[nl, 0] = ebar[k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = why
                        lg[nl, 4] = proceeds / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = avg
                        lg[nl, 7] = eq0[k]; lg[nl, 8] = nf[k]; lg[nl, 9] = bud[k]; lg[nl, 10] = rawm[k]; nl += 1
                    units[k] = 0.0; okbar[k] = t + cd
                    cash += res[k]; res[k] = 0.0
        # ---------------- collapse coins: sold at their last data hour
        for k in range(K):
            if units[k] > 0.0 and lastb[k] == t:
                proceeds = units[k] * c[t, k] * (1.0 - slip[k]) * (1.0 - pens) * (1.0 - fee); cash += proceeds
                if nl < maxlog:
                    lg[nl, 0] = ebar[k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = 5.0
                    lg[nl, 4] = proceeds / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = avgp[k]
                    lg[nl, 7] = eq0[k]; lg[nl, 8] = nf[k]; lg[nl, 9] = bud[k]; lg[nl, 10] = rawm[k]; nl += 1
                units[k] = 0.0; okbar[k] = t + cd
                cash += res[k]; res[k] = 0.0
        v = 0.0
        for k in range(K):
            if lk[k] > 0.0:                              # lazy: this hour's unfilled leg orders are cancelled
                cash += lk[k]; lk[k] = 0.0
        v = cash
        for k in range(K):
            v += res[k]
            if not np.isnan(c[t, k]):
                lastc[k] = c[t, k]
            if units[k] > 0.0:
                v += units[k] * lastc[k]
        eq[t] = v; eq_prev = v
    for k in range(K):                                   # still open at the end: marked, logged why=9
        if units[k] > 0.0 and nl < maxlog:
            proceeds = units[k] * lastc[k] * (1.0 - slip[k]) * (1.0 - pens) * (1.0 - fee)
            lg[nl, 0] = ebar[k]; lg[nl, 1] = n - 1; lg[nl, 2] = k; lg[nl, 3] = 9.0
            lg[nl, 4] = proceeds / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = avgp[k]
            lg[nl, 7] = eq0[k]; lg[nl, 8] = nf[k]; lg[nl, 9] = bud[k]; lg[nl, 10] = rawm[k]; nl += 1
    return eq, lg[:nl].copy()


# ------------------------------------------------------------------ inputs
_gm = {}


def guard_mask(P, g, L=168):
    """(n, K) 1 = allowed: last close no more than g under the highest close of the previous L hours (known at the
    open); identical to w90_wr90_lib.guard_mask / the live bot's crash_guard_pct for L=168. g <= 0: all ones."""
    c = P["cf"]
    if g <= 0:
        return np.ones(c.shape)
    key = (id(c), g, L)
    hit = _gm.get(key)
    if hit is None or hit[0] is not c:
        hi = c.rolling(L, min_periods=1).max()
        ok = ((c / hi - 1) >= -g).shift(1, fill_value=True)
        hit = _gm[key] = (c, ok.values.astype(float))
    return hit[1]


def last_bars(P, liquidate):
    o = P["o"]
    out = np.full(o.shape[1], -1, np.int64)
    if liquidate:
        for j, s in enumerate(o.columns):
            if s in E.DEAD:
                v = np.flatnonzero(~np.isnan(o[s].values))
                out[j] = int(v[-1])
    return out


def lrun(P, idx, legs=((0.14, 1.0),), H=999, N=None, fmax=1.0, tp=0.025, hm=96, cd=3, guard=0.35, lazy=0, ml=0.0,
         mlall=0, beh=0, be=0.003, eps=E.EPS, fee=E.FEE, slip=E.SLIP_MAJOR, pen=0.0, pens=0.0, tpwait=1, mtp_now=False,
         liquidate=False):
    """legs: ((depth, weight), ...) with depths ascending; weights are normalised to sum 1. N defaults to H."""
    dl = np.array([float(d) for d, _ in legs])
    wl = np.array([float(w) for _, w in legs]); wl = wl / wl.sum()
    assert np.all(np.diff(dl) > 0), "leg depths must increase"
    O = P["o"].values; n, K = O.shape
    if np.ndim(slip) == 0:
        sl = np.array([slip if s in E.H19 else E.SLIP_OTHER for s in P["o"].columns]) if liquidate else np.full(K, slip)
    else:
        sl = np.asarray(slip, np.float64)
    gm = np.ascontiguousarray(guard_mask(P, guard)) if guard > 0 else np.ones((n, K))
    Nn = H if N is None else N
    eq, lg = lkernel(O, P["h"].values, P["l"].values, P["cf"].values, P["ref"].values, gm, dl, wl, int(H), int(Nn),
                     float(fmax), float(tp), int(hm), int(cd), float(eps), float(fee), sl, float(pen), float(pens),
                     int(tpwait), bool(mtp_now), int(lazy), float(ml), int(mlall), int(beh), float(be),
                     last_bars(P, liquidate))
    return pd.Series(eq, idx), pd.DataFrame(lg, columns=LOGC)


# ------------------------------------------------------------------ data
_mk = {}


def market(dead=False, end=None):
    """(syms, idx, P) for the 19 coins from 2021 (optionally + LUNA/FTT/SRM). end: cut the panel there (no data at or
    after `end` is loaded), used for the 2021-23 design runs."""
    key = (dead, end)
    if key not in _mk:
        D = E.load_h19(dead=dead)
        if end is not None:
            D = {s: d[d.index < end] for s, d in D.items()}
            D = {s: d for s, d in D.items() if len(d)}
        _mk[key] = E.panel(D)
    return _mk[key]


def holdout_market(dead=False):
    key = ("pre", dead)
    if key not in _mk:
        _mk[key] = E.panel(E.load_pre(dead=dead))
    return _mk[key]


# ------------------------------------------------------------------ metrics
def period(e, a, b):
    """Compounded mult, %/month and worst drop of equity e from a to b (from the value just before a)."""
    i0 = int(np.searchsorted(e.index.values, np.datetime64(a), side="left"))
    i1 = len(e) if b is None else int(np.searchsorted(e.index.values, np.datetime64(b), side="left"))
    start = e.values[i0 - 1] if i0 > 0 else 1.0
    x = np.concatenate([[1.0], e.values[i0:i1] / start])
    yrs = ((e.index[i1 - 1] + pd.Timedelta(hours=1)) - e.index[i0]).total_seconds() / (365.25 * 86400)
    mult = float(x[-1])
    mo = mult ** (1.0 / (12 * yrs)) - 1.0 if mult > 0 else -1.0
    dd = float((x / np.maximum.accumulate(x) - 1.0).min())
    return mult, mo, dd, 12 * yrs


def stats(e, lg, idx, a, b, tag=""):
    """Metrics for ladders BOUGHT in [a, b): win rate (strict: a win is a take-profit exit with ret > 0, or a position
    still open at the end marked above cost; time stops, max-loss exits and collapse cuts are losses whatever their
    return), the plain ret > 0 win rate, %/month, worst drop, worst trade as % of the account, trades per month,
    hold days (mean, capital-weighted mean, max), share of hours with a position held over 30 days."""
    m, mo, dd, months = period(e, a, b)
    r = {f"{tag}mult": m, f"{tag}mo": mo, f"{tag}dd": dd}
    tb = idx[lg.eb.astype(int).values] if len(lg) else pd.DatetimeIndex([])
    sel = (tb >= a) & ((tb < b) if b is not None else True)
    x = lg[sel]
    hold = (x.xb - x.eb).values / 24.0
    win = ((x.why == 0) & (x.ret > 0)) | ((x.why == 9) & (x.ret > 0))
    acct = (x.ret * x.amt / x.eq0).values
    frac = (x.amt / x.eq0).values
    r.update({f"{tag}n": int(len(x)), f"{tag}wr": float(win.mean()) if len(x) else np.nan,
              f"{tag}wr_pos": float((x.ret > 0).mean()) if len(x) else np.nan,
              f"{tag}tpm": len(x) / months, f"{tag}wt": float(acct.min()) if len(x) else np.nan,
              f"{tag}avg": float(x.ret.mean()) if len(x) else np.nan,
              f"{tag}hold": float(hold.mean()) if len(x) else np.nan,
              f"{tag}hold_cw": float((hold * frac).sum() / frac.sum()) if len(x) and frac.sum() > 0 else np.nan,
              f"{tag}hold_max": float(hold.max()) if len(x) else np.nan,
              f"{tag}open": int((x.why == 9).sum()), f"{tag}ts": int((x.why == 1).sum()),
              f"{tag}mlx": int((x.why == 4).sum()), f"{tag}cut": int((x.why == 5).sum()),
              f"{tag}nleg_full": float((x.nf == x.nf.max()).mean()) if len(x) else np.nan})
    # hours (inside [a, b)) with at least one position older than 30 days
    i0 = int(np.searchsorted(idx.values, np.datetime64(a), side="left"))
    i1 = len(idx) if b is None else int(np.searchsorted(idx.values, np.datetime64(b), side="left"))
    old = np.zeros(len(idx), bool)
    for eb_, xb_ in zip(lg.eb.astype(int).values, lg.xb.astype(int).values):
        if xb_ - eb_ > 720:
            old[eb_ + 720:xb_ + 1] = True
    r[f"{tag}stuck30"] = float(old[i0:i1].mean()) if i1 > i0 else np.nan
    return r


def evaluate_design(spec, stress=False):
    """2021-23 only: the panel is cut at 2024-01-01, so nothing after the design period is loaded."""
    syms, idx, P = market(False, end=T_T0)
    kw = dict(spec)
    if stress:
        kw.update(mtp_now=True, cd=int(kw.get("cd", 3)) + 1, pen=STRESS_PEN, pens=STRESS_PEN)
    e, lg = lrun(P, idx, **kw)
    r = stats(e, lg, idx, T_D0, T_T0, "d_")
    r["y2021"], r["y2022"], r["y2023"] = (period(e, pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y + 1}-01-01"))[0] - 1
                                          for y in (2021, 2022, 2023))
    return r


def legs_name(legs):
    ws = np.array([w for _, w in legs], float); ws = ws / ws.sum()
    return "/".join(f"{d*100:g}" for d, _ in legs) + " w" + ":".join(f"{w*100:.0f}" for w in ws)


def name(spec):
    s = spec
    parts = [legs_name(s["legs"]), f"H{s.get('H', 999)} cap{s.get('fmax', 1.0)*100:.0f}%", f"tp{s.get('tp', .025)*100:g}",
             f"hm{s.get('hm', 96)}", f"cd{s.get('cd', 3)}", f"g{s.get('guard', .35)*100:g}"]
    if s.get("ml", 0) > 0:
        parts.append(f"ml{s['ml']*100:g}{'all' if s.get('mlall', 0) else ''}")
    if s.get("beh", 0) > 0:
        parts.append(f"be{s.get('be', .003)*100:g}@{s['beh']}h")
    if s.get("lazy", 0):
        parts.append("lazy")
    return " ".join(parts)
