"""Task "wr90": the highest monthly return with a win rate >= 90% in both 2021-23 and 2024-26.

wkernel = engine.gkernel (full take-profit only: part=1, no trail/tp2) plus these switches, all off by default:
  pens   sell-side fill penalty (engine's `pen` only touches buys). "Every fill 0.2% worse" = pen=pens=0.002.
  lastb  per coin, -1 = off: the last bar with data of a collapse coin cut from the panel (LUNA/FTT/SRM). A position
         still held there is sold at that bar's close (- slippage), so it is not frozen at the cut price forever
         (x10_cycle_guard `liquidate`).
  pmode  1 = "profit-only time stop": once a position has been held hm hours it is NOT sold at the close; instead the
         take-profit is lowered to entry x (1 + be) (a resting limit placed at the next hourly check, like the live
         bot re-places its order), so it only sells at or above break-even. Realized losses then come only from
         hs / hmax / lastb.
  hs     hard stop: a position is sold at an hourly close at or below entry x (1 - hs) (market, - slippage). 0 = off.
  hsa    the hard stop is only active once the position is at least hsa hours old (0 = from the first hour).
  hmax   absolute maximum hold in hours (pmode 1): sold at that close whatever the price. 0 = never.
Logs also carry the account equity when the order was sized (for the worst trade as % of the account).

The crash guard (crash_guard_pct in panic_buyer.py) is the smult mask from guard_mask(): a coin is skipped for the
hour when its last close is more than g below its highest close of the previous 168 hours (known at the open), exactly
x10_universe_push2.guard_mask.

With every switch off, wrun == engine.run exactly (check()).
Positions still open at the end of the data are marked at the last close (- slippage, - fee, - pens) and logged with
why=9, so an open loser counts as a loss in the win rate.
"""
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402

OUT = os.environ.get("PB_OUT", os.path.expanduser("~/pb_out_w90/wr90"))
os.makedirs(OUT, exist_ok=True)
T_D0, T_T0, T_S0 = pd.Timestamp("2021-01-01"), pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")
LOGC = ["eb", "xb", "k", "why", "ret", "amt", "entry", "eq0"]   # why 0 tp, 1 time stop, 4 hard stop, 5 cut, 6 hmax, 9 open
STRESS_PEN = 0.002


@njit(cache=True)
def wkernel(o, h, l, c, ref, depth, nord, fmax, smult, tpa, hm, cd, eps, fee, slip, pen, pens, tpwait, tprest,
            mtp_now, lastb, pmode, be, hs, hsa, hmax):
    n, K = c.shape
    units = np.zeros(K); entry = np.zeros(K); amt0 = np.zeros(K); tpp = np.zeros(K); eq0 = np.zeros(K)
    ebar = np.full(K, -1, np.int64); okbar = np.zeros(K, np.int64); mkt = np.zeros(K, np.int64)
    lowered = np.zeros(K, np.int64)
    lastc = np.full(K, np.nan); dist = np.empty(K)
    cash = 1.0; eq = np.empty(n); eq_prev = 1.0
    maxlog = 600000; lg = np.zeros((maxlog, 8)); nl = 0
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
                    cash -= a; units[k] = a * (1.0 - fee) / fill
                    entry[k] = fill; amt0[k] = a; ebar[k] = t; eq0[k] = eq_prev; lowered[k] = 0
                    tpp[k] = fill * (1.0 + tpa[t, k])
        for k in range(K):
            if units[k] > 0.0 and (ebar[k] < t or mkt[k] == 1) and not np.isnan(o[t, k]):
                px = -1.0; why = 0.0
                if ebar[k] == t:                     # market buy, take-profit placed at once (mtp_now)
                    if h[t, k] >= tpp[k] * (1.0 + eps):
                        px = tpp[k]
                elif t - ebar[k] < tpwait:
                    pass
                elif o[t, k] >= tpp[k]:
                    px = o[t, k]
                elif tprest and h[t, k] >= tpp[k] * (1.0 + eps):
                    px = tpp[k]
                age = t - ebar[k] + mkt[k]
                if px < 0.0:
                    if pmode == 0:
                        if age >= hm[t]:
                            px = c[t, k] * (1.0 - slip[k]); why = 1.0
                    else:
                        if hmax > 0 and age >= hmax:
                            px = c[t, k] * (1.0 - slip[k]); why = 6.0
                        elif age >= hm[t] and lowered[k] == 0:
                            lowered[k] = 1                       # next hourly check: target -> break-even
                            b = entry[k] * (1.0 + be)
                            if b < tpp[k]:
                                tpp[k] = b
                if px < 0.0 and hs > 0.0 and ebar[k] < t and age >= hsa and c[t, k] <= entry[k] * (1.0 - hs):
                    px = c[t, k] * (1.0 - slip[k]); why = 4.0
                if px > 0.0:
                    proceeds = units[k] * px * (1.0 - pens) * (1.0 - fee); cash += proceeds
                    if nl < maxlog:
                        lg[nl, 0] = ebar[k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = why
                        lg[nl, 4] = proceeds / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]
                        lg[nl, 7] = eq0[k]; nl += 1
                    units[k] = 0.0; okbar[k] = t + cd
        for k in range(K):
            if units[k] > 0.0 and lastb[k] == t:
                proceeds = units[k] * c[t, k] * (1.0 - slip[k]) * (1.0 - pens) * (1.0 - fee); cash += proceeds
                if nl < maxlog:
                    lg[nl, 0] = ebar[k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = 5.0
                    lg[nl, 4] = proceeds / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]
                    lg[nl, 7] = eq0[k]; nl += 1
                units[k] = 0.0; okbar[k] = t + cd
        v = cash
        for k in range(K):
            if not np.isnan(c[t, k]):
                lastc[k] = c[t, k]
            if units[k] > 0.0:
                v += units[k] * lastc[k]
        eq[t] = v; eq_prev = v
    # positions still open at the end: marked at the last close (not sold; equity above is unchanged)
    for k in range(K):
        if units[k] > 0.0 and nl < maxlog:
            proceeds = units[k] * lastc[k] * (1.0 - slip[k]) * (1.0 - pens) * (1.0 - fee)
            lg[nl, 0] = ebar[k]; lg[nl, 1] = n - 1; lg[nl, 2] = k; lg[nl, 3] = 9.0
            lg[nl, 4] = proceeds / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]; lg[nl, 7] = eq0[k]
            nl += 1
    return eq, lg[:nl].copy()


def last_bars(P):
    """Last bar with data for the collapse coins (LUNA/FTT/SRM), -1 for the others."""
    o = P["o"]
    out = np.full(o.shape[1], -1, np.int64)
    for j, s in enumerate(o.columns):
        if s in E.DEAD:
            v = np.flatnonzero(~np.isnan(o[s].values))
            out[j] = int(v[-1])
    return out


def wrun(P, idx, depth=0.12, N=8, fmax=0.20, smult=1.0, tp=0.015, hm=48, cd=24, eps=E.EPS, fee=E.FEE,
         slip=E.SLIP_MAJOR, pen=0.0, pens=0.0, tpwait=1, tprest=True, mtp_now=False, liquidate=False, pmode=0,
         be=0.0025, hs=0.0, hsa=0, hmax=0, guard=0.0):
    """engine.run's signature (full take-profit only) + the switches above. guard>0 multiplies smult by the crash
    guard mask. Returns (equity Series, log DataFrame incl. open positions with why=9)."""
    O = P["o"].values; n, K = O.shape
    if np.ndim(slip) == 0:
        sl = np.array([slip if s in E.H19 else E.SLIP_OTHER for s in P["o"].columns]) if liquidate else np.full(K, slip)
    else:
        sl = np.asarray(slip, np.float64)
    sm = E.full(n, K, smult)
    if guard > 0:
        sm = np.ascontiguousarray(sm * guard_mask(P, guard))
    lastb = last_bars(P) if liquidate else np.full(K, -1, np.int64)
    eq, lg = wkernel(O, P["h"].values, P["l"].values, P["cf"].values, P["ref"].values,
                     E.full(n, K, depth), E.full(n, 0, N, np.int64), E.full(n, 0, fmax), sm, E.full(n, K, tp),
                     E.full(n, 0, hm, np.int64), int(cd), float(eps), float(fee), sl, float(pen), float(pens),
                     int(tpwait), bool(tprest), bool(mtp_now), lastb, int(pmode), float(be), float(hs), int(hsa),
                     int(hmax))
    return pd.Series(eq, idx), pd.DataFrame(lg, columns=LOGC)


_gm = {}


def guard_mask(P, g):
    """(n, K) 1 = allowed: last close no more than g below the highest close of the previous 168 hours."""
    c = P["cf"]
    key = (id(c), g)
    hit = _gm.get(key)
    if hit is None or hit[0] is not c:      # the cache keeps a reference, so an id can't be reused by another panel
        hi = c.rolling(168, min_periods=1).max()
        ok = ((c / hi - 1) >= -g).shift(1, fill_value=True)
        hit = _gm[key] = (c, ok.values.astype(float))
    return hit[1]


# ------------------------------------------------------------------ data
_mk = {}


def market(dead=False):
    """(syms, idx, P) for the 19 coins, optionally + LUNA/FTT/SRM cut at their collapse (engine.load_h19)."""
    if dead not in _mk:
        _mk[dead] = E.panel(E.load_h19(dead=dead))
    return _mk[dead]


# ------------------------------------------------------------------ metrics
PERIODS = (("d", T_D0, T_T0), ("t", T_T0, None), ("s", T_S0, None))


def period(e, a, b):
    """Compounded %/month and worst drop on equity e from a to b, starting from the value just before a."""
    i0 = int(np.searchsorted(e.index.values, np.datetime64(a), side="left"))
    i1 = len(e) if b is None else int(np.searchsorted(e.index.values, np.datetime64(b), side="left"))
    start = e.values[i0 - 1] if i0 > 0 else 1.0
    x = np.concatenate([[1.0], e.values[i0:i1] / start])
    yrs = ((e.index[i1 - 1] + pd.Timedelta(hours=1)) - e.index[i0]).total_seconds() / (365.25 * 86400)
    mult = float(x[-1])
    mo = mult ** (1.0 / (12 * yrs)) - 1.0 if mult > 0 else -1.0
    dd = float((x / np.maximum.accumulate(x) - 1.0).min())
    return mult, mo, dd


def summarize(e, lg, idx):
    """Per period (d 2021-23, t 2024-26, s since 2025): mult, %/month, worst drop, trades bought in the period,
    win rate (ret > 0, open positions marked at the end), avg win/loss, worst trade as % of the account, open."""
    r = {}
    tb = idx[lg.eb.astype(int).values] if len(lg) else pd.DatetimeIndex([])
    acct = (lg.ret * lg.amt / lg.eq0).values if len(lg) else np.zeros(0)
    for tag, a, b in PERIODS:
        m, mo, dd = period(e, a, b)
        r[f"{tag}_mult"], r[f"{tag}_mo"], r[f"{tag}_dd"] = m, mo, dd
        sel = (tb >= a) & ((tb < b) if b is not None else True)
        x = lg[sel]
        r[f"{tag}_n"] = int(len(x))
        r[f"{tag}_wr"] = float((x.ret > 0).mean()) if len(x) else np.nan
        r[f"{tag}_avgw"] = float(x.ret[x.ret > 0].mean()) if (x.ret > 0).any() else np.nan
        r[f"{tag}_avgl"] = float(x.ret[x.ret <= 0].mean()) if (x.ret <= 0).any() else np.nan
        r[f"{tag}_wt"] = float(acct[sel].min()) if sel.any() else np.nan
        r[f"{tag}_open"] = int((x.why == 9).sum())
    y = E.years(e)
    r["y2022"] = y.get(2022, np.nan)
    r["minmo"] = min(r["d_mo"], r["t_mo"], r["s_mo"])
    r["mindd"] = min(r["d_dd"], r["t_dd"], r["s_dd"])
    r["minwr"] = min(r["d_wr"], r["t_wr"])
    r["hold_max_h"] = float((lg.xb - lg.eb).max()) if len(lg) else 0.0
    return r


def stressed_kw(kw):
    """Real bot timing (market-buy take-profit at once, cooldown +1h), every fill 0.2% worse (buys and sells),
    LUNA/FTT/SRM added (run on market(dead=True) with liquidate=True)."""
    k = dict(kw)
    k.update(mtp_now=True, cd=int(kw.get("cd", 24)) + 1, pen=STRESS_PEN, pens=STRESS_PEN, liquidate=True)
    return k


def evaluate(kw, stress=False, runner=None):
    runner = runner or wrun
    syms, idx, P = market(dead=stress)
    k = stressed_kw(kw) if stress else dict(kw)
    e, lg = runner(P, idx, **k)
    return summarize(e, lg, idx), e, lg


def fmt(name, r):
    s = (f"{name}\n"
         f"   2021-23 WR {r['d_wr']*100:5.1f}% ({r['d_n']:4d} tr) {r['d_mo']*100:+6.2f}%/mo drop {r['d_dd']*100:5.1f}% "
         f"worst trade {r['d_wt']*100:+6.1f}% acct | avg win {r['d_avgw']*100:+.2f}% loss {r['d_avgl']*100:+.2f}%\n"
         f"   2024-26 WR {r['t_wr']*100:5.1f}% ({r['t_n']:4d} tr) {r['t_mo']*100:+6.2f}%/mo drop {r['t_dd']*100:5.1f}% "
         f"worst trade {r['t_wt']*100:+6.1f}% acct | avg win {r['t_avgw']*100:+.2f}% loss {r['t_avgl']*100:+.2f}%\n"
         f"   since-25 WR {r['s_wr']*100:5.1f}% ({r['s_n']:4d} tr) {r['s_mo']*100:+6.2f}%/mo drop {r['s_dd']*100:5.1f}% "
         f"worst trade {r['s_wt']*100:+6.1f}% acct | open at end {r['s_open']}")
    return s


def check():
    """Every new switch off: wrun == engine.run exactly (equity and trade log), on the 19 coins and with the
    collapse coins, for several settings incl. mtp_now, pen and a guard mask; liquidate on with no collapse coin and
    pmode 1 with hm beyond any hold also change nothing."""
    ok_all = True
    cases = [dict(E.LIVE), dict(depth=0.14, N=1, fmax=1.0, tp=0.05, hm=48, cd=24),
             dict(depth=0.12, N=2, fmax=0.5, tp=0.02, hm=96, cd=6, mtp_now=True, pen=0.002),
             dict(depth=0.16, N=4, fmax=0.25, tp=0.01, hm=168, cd=12, tpwait=2)]
    for dead in (False, True):
        syms, idx, P = market(dead)
        for kw in cases:
            for g in (0.0, 0.35):
                sm = guard_mask(P, g) if g > 0 else 1.0
                e1, l1 = E.run(P, idx, smult=sm, **kw)
                e2, l2 = wrun(P, idx, guard=g, **kw)
                l2c = l2[l2.why != 9]
                ok = np.array_equal(e1.values, e2.values) and np.array_equal(
                    l1[["eb", "xb", "k", "why", "ret", "amt", "entry"]].values,
                    l2c[["eb", "xb", "k", "why", "ret", "amt", "entry"]].values)
                if not dead:
                    e3, l3 = wrun(P, idx, guard=g, liquidate=True, **kw)
                    ok = ok and np.array_equal(e1.values, e3.values)
                ok_all &= ok
                print(f"{'22 coins' if dead else '19 coins'} g{g} {kw}: engine x{e1.iloc[-1]:.6f} wrun "
                      f"x{e2.iloc[-1]:.6f} trades {len(l1)}/{len(l2c)} {'IDENTICAL' if ok else 'DIFFERENT'}", flush=True)
    print("ALL CHECKS PASS" if ok_all else "SOME CHECK FAILED")
    return ok_all


if __name__ == "__main__":
    check()
