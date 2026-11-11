"""Lever "stack": the return boosters found so far, stacked on concentrated settings (N 2-4 bids, cap 25-50%).

skernel is r3_ladder.lkernel (= engine.gkernel + the split ladder) copied with four extra switches:
  pens     every SELL fill `pens` worse (engine's `pen` only touches buys), for "every fill 0.2% worse".
  tpf/tpfm take-profit at max(entry*(1+tp), entry + tpf*(ref_at_entry - entry)); tpfm 0 = off, 1 = only for
           market buys (coin already under its trigger at the hour's open), 2 = every fill. After a ladder leg-2
           fill the same rule is applied to the new average entry.
  release  cash locked in a resting leg 2 of a coin whose data has ended (LUNA/FTT/SRM after their cut) goes
           back to cash (the order would be cancelled); without it the cash stays locked forever.
  eq0      the account value the order was sized on, logged per trade (worst trade as % of the account).
Per-hour N (bids per hour) and fmax (cap per coin, both legs) arrays give "fewer/bigger orders when BTC falls".
With w=1, pens=0, tpfm=0 it is engine.run exactly; with the ladder on it is r3_ladder.lrun exactly
(x10_stack_check.py proves both). Everything is decided at hourly cycles from closed candles; no lookahead.

Costs: 0.1% fee per side; market buys at the open and time-stop sells pay 0.02% slippage for the 19 live coins
(all in the top 50 of the screened universe by volume), 0.1% for LUNA/FTT/SRM (conservative, stress only);
limit buys fill only if the low trades 0.1% through. Orders are fully funded: each order <= cash at placement,
leg-2 cash is locked until it fills or the position is sold.
"""
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402

OUT = os.environ.get("PB_OUT", os.path.expanduser("~/pb_out_x10_stack"))
os.makedirs(OUT, exist_ok=True)
T_TEST0, T_25 = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")
LOG = E.LOG_COLS + ["leg2", "a1", "bud", "eq0", "mkt"]
L = E.LIVE


@njit(cache=False)
def skernel(o, h, l, c, ref, depth, nord, fmax, smult, tpa, part, tp2, trail, hm, cd, eps, fee, slip, pen,
            tpwait, tprest, mtp_now, d2, w, pens, tpf, tpfm, lastbar):
    n, K = c.shape
    units = np.zeros(K); entry = np.zeros(K); amt0 = np.zeros(K); got = np.zeros(K); tpp = np.zeros(K)
    stage = np.zeros(K, np.int64); sbar = np.zeros(K, np.int64); peak = np.zeros(K)
    ebar = np.full(K, -1, np.int64); okbar = np.zeros(K, np.int64); mkt = np.zeros(K, np.int64)
    lastc = np.full(K, np.nan); dist = np.empty(K)
    lp2 = np.zeros(K); res2 = np.zeros(K)
    a1 = np.zeros(K); e1 = np.zeros(K); tpr = np.zeros(K); leg2 = np.zeros(K); bud = np.zeros(K)
    lbar = np.full(K, -1, np.int64)
    tf = np.zeros(K); refe = np.zeros(K); eq0 = np.zeros(K); rawm = np.zeros(K)
    cash = 1.0; eq = np.empty(n); eq_prev = 1.0
    maxlog = 600000; lg = np.zeros((maxlog, 12)); nl = 0
    for t in range(n):
        for k in range(K):
            if res2[k] > 0.0 and t > lastbar[k]:          # coin's data has ended: cancel the resting leg 2
                cash += res2[k]; res2[k] = 0.0; lp2[k] = 0.0
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
                    b1 = a * w
                    b2 = a - b1
                    cash -= a; units[k] = b1 * (1.0 - fee) / fill
                    entry[k] = fill; amt0[k] = b1; ebar[k] = t; got[k] = 0.0
                    tpp[k] = fill * (1.0 + tpa[t, k]); stage[k] = 0; peak[k] = fill
                    lbar[k] = t; a1[k] = b1; e1[k] = fill; tpr[k] = tpa[t, k]; leg2[k] = 0.0; bud[k] = a
                    lp2[k] = 0.0; res2[k] = 0.0
                    eq0[k] = eq_prev; rawm[k] = ismkt; refe[k] = ref[t, k]
                    tf[k] = 0.0
                    if tpfm == 2 or (tpfm == 1 and ismkt == 1):
                        tf[k] = tpf
                    if tf[k] > 0.0:
                        alt = fill + tf[k] * (refe[k] - fill)
                        if alt > tpp[k]:
                            tpp[k] = alt
                    if b2 > 0.0:
                        lp2[k] = ref[t, k] * (1.0 - d2)
                        f2 = -1.0
                        if o[t, k] <= lp2[k]:
                            f2 = o[t, k] * (1.0 + slip[k]) * (1.0 + pen)
                        elif l[t, k] <= lp2[k] * (1.0 - eps):
                            f2 = lp2[k] * (1.0 + pen)
                        if f2 > 0.0:
                            units[k] += b2 * (1.0 - fee) / f2; amt0[k] = b1 + b2
                            entry[k] = amt0[k] / (b1 / fill + b2 / f2); tpp[k] = entry[k] * (1.0 + tpr[k])
                            if tf[k] > 0.0:
                                alt = entry[k] + tf[k] * (refe[k] - entry[k])
                                if alt > tpp[k]:
                                    tpp[k] = alt
                            leg2[k] = 1.0; lp2[k] = 0.0
                        else:
                            res2[k] = b2
        for k in range(K):
            if units[k] > 0.0 and (ebar[k] < t or mkt[k] == 1) and not np.isnan(o[t, k]):
                if ebar[k] < t and res2[k] > 0.0 and l[t, k] <= lp2[k] * (1.0 - eps):
                    b2 = res2[k]
                    f2 = lp2[k] * (1.0 + pen)
                    units[k] += b2 * (1.0 - fee) / f2; amt0[k] = a1[k] + b2
                    entry[k] = amt0[k] / (a1[k] / e1[k] + b2 / f2); tpp[k] = entry[k] * (1.0 + tpr[k])
                    if tf[k] > 0.0:
                        alt = entry[k] + tf[k] * (refe[k] - entry[k])
                        if alt > tpp[k]:
                            tpp[k] = alt
                    leg2[k] = 1.0; lp2[k] = 0.0; res2[k] = 0.0; lbar[k] = t
                px = -1.0; why = 0.0
                if stage[k] == 0:
                    if ebar[k] == t:
                        if leg2[k] == 0.0 and h[t, k] >= tpp[k] * (1.0 + eps):
                            px = tpp[k]
                    elif t - lbar[k] < tpwait:
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
                        cash += res2[k]; res2[k] = 0.0; lp2[k] = 0.0
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
                if px < 0.0 and t - ebar[k] >= hm[t] - mkt[k]:
                    px = c[t, k] * (1.0 - slip[k]); why = 1.0
                if px > 0.0:
                    proceeds = units[k] * px * (1.0 - pens) * (1.0 - fee); cash += proceeds; got[k] += proceeds
                    if nl < maxlog:
                        lg[nl, 0] = ebar[k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = why
                        lg[nl, 4] = got[k] / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]
                        lg[nl, 7] = leg2[k]; lg[nl, 8] = a1[k]; lg[nl, 9] = bud[k]; lg[nl, 10] = eq0[k]
                        lg[nl, 11] = rawm[k]; nl += 1
                    units[k] = 0.0; okbar[k] = t + cd
                    cash += res2[k]; res2[k] = 0.0; lp2[k] = 0.0
        v = cash
        for k in range(K):
            v += res2[k]
            if not np.isnan(c[t, k]):
                lastc[k] = c[t, k]
            if units[k] > 0.0:
                v += units[k] * lastc[k]
        eq[t] = v; eq_prev = v
    return eq, lg[:nl].copy()


def slip_vec(syms):
    return np.array([E.SLIP_MAJOR if s in E.H19 else E.SLIP_OTHER for s in syms])


def last_bars(P, release=True):
    O = P["o"].values
    n, K = O.shape
    if not release:
        return np.full(K, n + 1, np.int64)
    ok = ~np.isnan(O)
    return np.array([np.nonzero(ok[:, k])[0][-1] if ok[:, k].any() else -1 for k in range(K)], np.int64)


def srun(P, idx, depth=L["depth"], N=L["N"], fmax=L["fmax"], smult=1.0, tp=L["tp"], part=1.0, tp2=0.0, trail=0.0,
         hm=L["hm"], cd=L["cd"], eps=E.EPS, fee=E.FEE, slip=E.SLIP_MAJOR, pen=0.0, tpwait=1, tprest=True,
         mtp_now=False, d2=0.0, w=1.0, pens=0.0, tpf=0.0, tpfm=0, release=True):
    """engine.run + ladder (w, d2) + pens + tpf/tpfm + release. slip: scalar, per-coin array, or 'auto'
    (0.02% for the 19 live coins, 0.1% for the others)."""
    assert 0.0 < w <= 1.0
    assert w == 1.0 or float(d2) > float(np.max(depth)), "leg 2 must rest deeper than the trigger"
    O = P["o"].values; n, K = O.shape
    if isinstance(slip, str):
        sl = slip_vec(list(P["o"].columns))
    else:
        sl = np.full(K, slip) if np.ndim(slip) == 0 else np.asarray(slip, np.float64)
    eq, lg = skernel(O, P["h"].values, P["l"].values, P["cf"].values, P["ref"].values,
                     E.full(n, K, depth), E.full(n, 0, N, np.int64), E.full(n, 0, fmax), E.full(n, K, smult),
                     E.full(n, K, tp), float(part), float(tp2), float(trail), E.full(n, 0, hm, np.int64), int(cd),
                     eps, fee, sl, pen, int(tpwait), bool(tprest), bool(mtp_now), float(d2), float(w), float(pens),
                     float(tpf), int(tpfm), last_bars(P, release))
    return pd.Series(eq, idx), pd.DataFrame(lg, columns=LOG)


# ------------------------------------------------------------------ metrics
def period(e, a, b=None):
    """Multiple, %/month and worst drop on [a, b): start = account value at the last hour before a (1.0 if none).
    %/month = multiple ** (1 / months) - 1, months = (last hour + 1h - first hour) / (365.25/12 days)."""
    idx = e.index
    ia = int(np.searchsorted(idx, a, side="left"))
    ib = len(idx) if b is None else int(np.searchsorted(idx, b, side="left"))
    v = e.values
    s = v[ia - 1] if ia > 0 else 1.0
    x = np.concatenate([[1.0], v[ia:ib] / s])
    yrs = ((idx[ib - 1] + pd.Timedelta(hours=1)) - idx[ia]).total_seconds() / (365.25 * 86400)
    mult = float(x[-1])
    mo = mult ** (1.0 / (12 * yrs)) - 1.0 if mult > 0 else -1.0
    dd = float((x / np.maximum.accumulate(x) - 1.0).min())
    return mult, mo, dd


def year_ret(e, y):
    a = e[e.index < pd.Timestamp(f"{y}-01-01")]
    b = e[e.index < pd.Timestamp(f"{y + 1}-01-01")]
    if len(b) == 0 or len(b) == len(a):
        return np.nan
    s = a.iloc[-1] if len(a) else 1.0
    return float(b.iloc[-1] / s - 1.0)


def summarize(e, lg, idx):
    r = {}
    for tag, a, b in (("d", idx[0], T_TEST0), ("t", T_TEST0, None), ("s", T_25, None)):
        r[f"{tag}_mult"], r[f"{tag}_mo"], r[f"{tag}_dd"] = period(e, a, b)
    for y in range(2021, 2027):
        r[f"y{y}"] = year_ret(e, y)
    r["trades"] = len(lg)
    if len(lg):
        acct = (lg.ret * lg.amt / lg.eq0).values
        eb = idx[lg.eb.astype(int).values]
        tt = eb >= T_TEST0
        r["worst_acct"] = float(acct.min())
        r["worst_acct_d"] = float(acct[~tt].min()) if (~tt).any() else np.nan
        r["worst_acct_t"] = float(acct[tt].min()) if tt.any() else np.nan
        r["worst_ret"] = float(lg.ret.min())
        r["t_trades"] = int(tt.sum())
        r["t_win"] = float((lg.ret[tt] > 0).mean()) if tt.any() else np.nan
        r["t_avg"] = float(lg.ret[tt].mean()) if tt.any() else np.nan
        r["max_bud"] = float((lg.bud / lg.eq0).max())
    return r


def fmt4(name, r, rs=None):
    s = (f"{name}\n  (1) 2024-26  {r['t_mo']*100:+6.2f}%/mo  x{r['t_mult']:7.2f}  drop {r['t_dd']*100:5.1f}%\n"
         f"  (2) 2025-26  {r['s_mo']*100:+6.2f}%/mo  x{r['s_mult']:7.2f}  drop {r['s_dd']*100:5.1f}%\n"
         f"  (3) 2021-23  {r['d_mo']*100:+6.2f}%/mo  x{r['d_mult']:7.2f}  drop {r['d_dd']*100:5.1f}%  2022 {r['y2022']*100:+5.1f}%")
    if "worst_acct" in r:
        s += (f"\n      trades {r['trades']}, worst trade {r['worst_acct']*100:+.1f}% of account (2021-23 "
              f"{r['worst_acct_d']*100:+.1f}%, 2024-26 {r['worst_acct_t']*100:+.1f}%), worst trade ret "
              f"{r['worst_ret']*100:+.1f}%; 2024-26 win {r['t_win']*100:.0f}% avg {r['t_avg']*100:+.2f}%")
    if rs is not None:
        s += (f"\n  (4) fills 0.2% worse (buys and sells) + LUNA/FTT/SRM: 2024-26 {rs['t_mo']*100:+6.2f}%/mo drop "
              f"{rs['t_dd']*100:5.1f}% | 2025-26 {rs['s_mo']*100:+6.2f}%/mo drop {rs['s_dd']*100:5.1f}% | 2021-23 "
              f"{rs['d_mo']*100:+6.2f}%/mo drop {rs['d_dd']*100:5.1f}% 2022 {rs['y2022']*100:+5.1f}%")
        if "worst_acct" in rs:
            s += f" | worst trade {rs['worst_acct']*100:+.1f}% of account"
    return s


_mk = {}


def market(dead=False):
    """(syms, idx, P) for the 19 live coins (+ LUNA/FTT/SRM): engine.panel(engine.load_h19), cached."""
    if dead not in _mk:
        _mk[dead] = E.panel(E.load_h19(dead=dead))
    return _mk[dead]


def btc_dd24(P):
    """BTC's last close vs its highest close of the last 24h (<= 0), known at the hour's start."""
    cf = P["cf"]["BTCUSDT"]
    return (cf / cf.rolling(24, min_periods=1).max() - 1).shift(1).fillna(0.0).values


def spec_kw(P, s):
    """Turn a design dict into srun kwargs. Keys: depth N fmax tp hm cd tpwait tprest w d2 tpf tpfm
    and the BTC rule bx (BTC under its 24h high by >= bx -> N=bN bids, cap bf)."""
    kw = {k: v for k, v in s.items() if k not in ("bx", "bN", "bf", "name")}
    if s.get("bx", 0) > 0:
        st = btc_dd24(P) <= -s["bx"]
        kw["N"] = np.where(st, s["bN"], s["N"]).astype(np.int64)
        kw["fmax"] = np.where(st, s["bf"], s["fmax"]).astype(np.float64)
    return kw


def evaluate(s, stress=False):
    """Normal: 19 coins, normal costs. stress: + LUNA/FTT/SRM and every fill (buys and sells) 0.2% worse."""
    syms, idx, P = market(stress)
    kw = spec_kw(P, s)
    if stress:
        kw.update(pen=0.002, pens=0.002)
    e, lg = srun(P, idx, slip="auto", **kw)
    return summarize(e, lg, idx), e, lg
