"""Task "ten-hourly" (w90 round): make 10%/month hold in 2024-26 AND since 2025 with the hourly bot mechanism.

tkernel is x10_stack_lib.skernel (= engine.gkernel + split ladder + sell penalty + tp relative to the 24h high +
release of a dead coin's leg-2 cash) copied with three extra switches:
  cdw   cooldown after a WINNING exit, in hours (-1 = use cd, the engine's single cooldown). cd stays the cooldown
        after a losing exit. cdw=1 = "re-arm the coin at the next hourly check after a profitable sale".
  near  bid only on coins whose open is at most `near` above the trigger (1e8 = off, the engine). With N=2 and
        fmax=1 this puts 100% on one coin when only one is near its trigger and 50/50 when two are.
  liq   1 = a position still open at the coin's last data hour (LUNA/FTT/SRM cuts) is sold at that hour's close
        (with market slippage and the sell penalty) instead of staying frozen at its last price forever.
  eq0   (already in skernel) the account value the order was sized on, for "worst trade as % of the account".
Everything else is skernel unchanged. With cdw=-1, liq=0, near=1e8, w=1, pens=0, tpfm=0 it is engine.run exactly
(w90_tenh_check.py proves it on the 19 coins and the 22-coin stress panel, with and without the switches that
engine.run has).

Inputs that need no kernel change (built here from closed candles only, known at each hour's open):
  guard(P, g, L)       crash guard mask: skip a coin whose last close is more than g under its highest close of
                       the previous L hours (the live bot's crash_guard_pct uses L=168).
  vol(P, W)            each coin's own daily volatility from its last W closed hourly candles.
  tiers(...)           per-coin depth (and take-profit) from that volatility.

Costs as everywhere: 0.1% fee per side, market buys/time-stop sells 0.02% slippage (0.1% for LUNA/FTT/SRM),
limit buys only when the low trades 0.1% through, all orders fully funded from cash (spot, no leverage).
"""
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402

OUT = os.environ.get("PB_OUT", os.path.expanduser("~/pb_out_w90_tenh"))
os.makedirs(OUT, exist_ok=True)
T21, T24, T25 = pd.Timestamp("2021-01-01"), pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")
LOG = E.LOG_COLS + ["leg2", "a1", "bud", "eq0", "mkt"]
TENMODE = dict(depth=0.14, N=1, fmax=1.0, tp=0.05, hm=48, cd=24, g=0.35, gL=168)


@njit(cache=True)
def tkernel(o, h, l, c, ref, depth, nord, fmax, smult, tpa, part, tp2, trail, hm, cd, eps, fee, slip, pen,
            tpwait, tprest, mtp_now, d2, w, pens, tpf, tpfm, lastbar, cdw, liq, near):
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
            if dist[k] > near:                 # bid only on coins whose open is within `near` of the trigger
                dist[k] = 1e9
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
                if px < 0.0 and liq == 1 and t == lastbar[k]:      # last data hour of a dead coin: sell it
                    px = c[t, k] * (1.0 - slip[k]); why = 4.0
                if px > 0.0:
                    proceeds = units[k] * px * (1.0 - pens) * (1.0 - fee); cash += proceeds; got[k] += proceeds
                    if nl < maxlog:
                        lg[nl, 0] = ebar[k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = why
                        lg[nl, 4] = got[k] / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]
                        lg[nl, 7] = leg2[k]; lg[nl, 8] = a1[k]; lg[nl, 9] = bud[k]; lg[nl, 10] = eq0[k]
                        lg[nl, 11] = rawm[k]; nl += 1
                    units[k] = 0.0
                    if cdw >= 0 and got[k] > amt0[k]:
                        okbar[k] = t + cdw
                    else:
                        okbar[k] = t + cd
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


def last_bars(P):
    """Last hour with data per coin; coins that trade to the end of the panel get n+1 (never liquidated)."""
    O = P["o"].values
    n, K = O.shape
    ok = ~np.isnan(O)
    lb = np.array([np.nonzero(ok[:, k])[0][-1] if ok[:, k].any() else -1 for k in range(K)], np.int64)
    lb[lb >= n - 1] = n + 1
    return lb


def trun(P, idx, depth=0.12, N=8, fmax=0.2, smult=1.0, tp=0.015, part=1.0, tp2=0.0, trail=0.0, hm=48, cd=24,
         eps=E.EPS, fee=E.FEE, slip="auto", pen=0.0, tpwait=1, tprest=True, mtp_now=False, d2=0.0, w=1.0, pens=0.0,
         tpf=0.0, tpfm=0, cdw=-1, liq=0, near=1e8):
    """engine.run's signature plus d2/w (ladder), pens, tpf/tpfm, cdw, liq. slip 'auto' = 0.02% for the 19 coins,
    0.1% for others (engine.run default is 0.02% for all; identical on the 19 coins)."""
    assert 0.0 < w <= 1.0
    assert w == 1.0 or float(d2) > float(np.max(depth)), "leg 2 must rest deeper than the trigger"
    O = P["o"].values; n, K = O.shape
    if isinstance(slip, str):
        sl = slip_vec(list(P["o"].columns))
    else:
        sl = np.full(K, slip) if np.ndim(slip) == 0 else np.asarray(slip, np.float64)
    eq, lg = tkernel(O, P["h"].values, P["l"].values, P["cf"].values, P["ref"].values,
                     E.full(n, K, depth), E.full(n, 0, N, np.int64), E.full(n, 0, fmax), E.full(n, K, smult),
                     E.full(n, K, tp), float(part), float(tp2), float(trail), E.full(n, 0, hm, np.int64), int(cd),
                     eps, fee, sl, pen, int(tpwait), bool(tprest), bool(mtp_now), float(d2), float(w), float(pens),
                     float(tpf), int(tpfm), last_bars(P), int(cdw), int(liq), float(near))
    return pd.Series(eq, idx), pd.DataFrame(lg, columns=LOG)


# ------------------------------------------------------------------ features (closed candles only)
_fc = {}


def _key(P, *a):
    return (id(P["cf"]), len(P["cf"])) + a


def guard(P, g=0.35, L=168):
    """1.0 = allowed: last close no more than g under the highest close of the previous L hours (known at the
    open). L=168 is the live bot's crash_guard_pct. g<=0: no guard."""
    if g <= 0:
        return 1.0
    k = _key(P, "g", g, L)
    if k not in _fc:
        cf = P["cf"]
        hi = cf.rolling(L, min_periods=1).max()
        _fc[k] = ((cf / hi - 1) >= -g).shift(1, fill_value=True).values.astype(float)
    return _fc[k]


def vol(P, W=720):
    """Daily volatility (std of hourly log returns x sqrt 24) over the last W closed hours, known at the open."""
    k = _key(P, "v", W)
    if k not in _fc:
        r = np.log(P["cf"]).diff()
        _fc[k] = (r.rolling(W, min_periods=W // 4).std() * np.sqrt(24)).shift(1)
    return _fc[k]


def vol_rank(P, W=720):
    """Each coin's volatility rank among the coins trading that hour, 0 (calmest) .. 1 (wildest)."""
    k = _key(P, "vr", W)
    if k not in _fc:
        v = vol(P, W)
        _fc[k] = v.rank(axis=1, pct=True)
    return _fc[k]


def depth_tiers(P, base, lo=0.0, hi=0.0, W=720, cut=(1 / 3, 2 / 3)):
    """Per-coin depth by volatility rank: calmest third base+lo, middle base, wildest third base+hi.
    Coins without enough history get the base depth."""
    if lo == 0.0 and hi == 0.0:
        return base
    r = vol_rank(P, W).values
    d = np.full(r.shape, base)
    d[r <= cut[0]] = base + lo
    d[r > cut[1]] = base + hi
    return d


def depth_scaled(P, m, dmin, dmax, W=720):
    """Per-coin depth = m x the coin's own daily volatility, clipped to [dmin, dmax]."""
    v = vol(P, W).values
    d = np.clip(m * v, dmin, dmax)
    return np.where(np.isnan(d), dmax, d)


# ------------------------------------------------------------------ metrics
def period(e, a, b=None):
    """Multiple, %/month and worst drop on [a, b) (start = value at the last hour before a, 1.0 if none);
    %/month = multiple ** (1 / months) - 1, months = hours covered / (365.25/12 days). Same as x10_stack_lib."""
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


PERIODS = (("d", T21, T24), ("t", T24, None), ("s", T25, None))


def summarize(e, lg, idx):
    """Per period d=2021-23, t=2024-26, s=since 2025: multiple, %/month, worst drop, trades bought in the period,
    win rate, average trade, worst trade as % of the account (trade return x order size / account value)."""
    r = {}
    eb = idx[lg.eb.astype(int).values] if len(lg) else pd.DatetimeIndex([])
    acct = (lg.ret * lg.amt / lg.eq0).values if len(lg) else np.array([])
    for tag, a, b in PERIODS:
        r[f"{tag}_mult"], r[f"{tag}_mo"], r[f"{tag}_dd"] = period(e, a, b)
        m = (eb >= a) & ((eb < b) if b is not None else True)
        m = np.asarray(m, bool)
        r[f"{tag}_n"] = int(m.sum())
        r[f"{tag}_win"] = float((lg.ret.values[m] > 0).mean()) if m.any() else np.nan
        r[f"{tag}_avg"] = float(lg.ret.values[m].mean()) if m.any() else np.nan
        r[f"{tag}_avgw"] = float(lg.ret.values[m][lg.ret.values[m] > 0].mean()) if (m & (lg.ret.values > 0)).any() else np.nan
        r[f"{tag}_avgl"] = float(lg.ret.values[m][lg.ret.values[m] <= 0].mean()) if (m & (lg.ret.values <= 0)).any() else np.nan
        r[f"{tag}_wacct"] = float(acct[m].min()) if m.any() else np.nan
    r["y2022"] = year_ret(e, 2022)
    r["min_mo"] = min(r["d_mo"], r["t_mo"], r["s_mo"])
    r["min_dd"] = min(r["d_dd"], r["t_dd"], r["s_dd"])
    r["min_win"] = min(r["d_win"], r["t_win"])
    return r


_mk = {}


def market(dead=False):
    """(syms, idx, P) for the 19 live coins (+ LUNA/FTT/SRM): engine.panel(engine.load_h19), cached."""
    if dead not in _mk:
        _mk[dead] = E.panel(E.load_h19(dead=dead))
    return _mk[dead]


# ------------------------------------------------------------------ design -> kwargs
def spec_kw(P, s):
    """Design dict -> trun kwargs. Extra keys (not trun args):
      g, gL           crash guard threshold and lookback (smult mask)
      dlo, dhi, vW    per-coin depth tiers by volatility rank (calm third depth+dlo, wild third depth+dhi)
      vm, vmin, vmax  depth = vm x own daily vol, clipped (overrides depth when vm>0)
      tlo, thi        per-coin take-profit tiers by the same rank (calm tp+tlo, wild tp+thi)
      tpr             take-profit = tpr x the coin's depth that hour (overrides tp when > 0)
      refW            trigger reference = highest close of the last refW hours (24 = live bot; evaluate only)."""
    kw = {k: v for k, v in s.items() if k not in ("g", "gL", "dlo", "dhi", "vW", "vm", "vmin", "vmax", "tlo", "thi",
                                                   "tpr", "refW", "name", "sub")}
    W = s.get("vW", 720)
    if s.get("vm", 0) > 0:
        kw["depth"] = depth_scaled(P, s["vm"], s["vmin"], s["vmax"], W)
    elif s.get("dlo", 0) or s.get("dhi", 0):
        kw["depth"] = depth_tiers(P, s["depth"], s.get("dlo", 0.0), s.get("dhi", 0.0), W)
    if s.get("tlo", 0) or s.get("thi", 0):
        kw["tp"] = depth_tiers(P, s["tp"], s.get("tlo", 0.0), s.get("thi", 0.0), W)
    if s.get("tpr", 0) > 0:
        kw["tp"] = s["tpr"] * (np.asarray(kw["depth"], float) if np.ndim(kw["depth"]) else
                               np.full(P["o"].shape, float(kw["depth"])))
    kw["smult"] = guard(P, s.get("g", 0.0), s.get("gL", 168))
    return kw


def stress_over(s):
    """The stressed version: real bot timing (take-profit placed at once after a market buy, cooldown counted from
    the next hourly check, i.e. cd+1 and cdw+1), every buy AND sell fill 0.2% worse, LUNA/FTT/SRM added (sold at
    their last data hour)."""
    o = dict(mtp_now=True, cd=int(s.get("cd", 24)) + 1, pen=0.002, pens=0.002, liq=1)
    if s.get("cdw", -1) >= 0:
        o["cdw"] = int(s["cdw"]) + 1
    return o


def with_ref(P, W):
    """Panel copy whose trigger reference is the highest close of the last W closed hours (live bot: 24)."""
    if W == 24:
        return P
    k = _key(P, "ref", W)
    if k not in _fc:
        Q = dict(P)
        Q["ref"] = P["c"].rolling(W, min_periods=1).max().shift(1)
        _fc[k] = Q
    return _fc[k]


def evaluate(s, stress=False, dead=None, **over):
    """Run one design. stress=True: the stressed version on the 22-coin panel."""
    dead = stress if dead is None else dead
    syms, idx, P = market(dead)
    P = with_ref(P, s.get("refW", 24))
    kw = spec_kw(P, s)
    if stress:
        kw.update(stress_over(s))
    kw.update(over)
    if s.get("sub"):                      # two independent sub-accounts: (weight, design) list
        return evaluate_sub(s, stress, dead, **over)
    e, lg = trun(P, idx, **kw)
    return summarize(e, lg, idx), e, lg


def evaluate_sub(s, stress=False, dead=False, **over):
    """s['sub'] = [(weight, design), ...]: separate bot instances with their own money, never rebalanced.
    Account = weighted sum of the sub-account curves. Trade stats pooled (trade % of the account scaled by weight)."""
    syms, idx, P = market(dead)
    E_, L_ = 0.0, []
    for wgt, d in s["sub"]:
        kw = spec_kw(P, d)
        if stress:
            kw.update(stress_over(d))
        kw.update(over)
        e, lg = trun(P, idx, **kw)
        E_ = E_ + wgt * e
        lg = lg.copy()
        lg["eq0"] = lg["eq0"] / wgt          # trade size vs the whole account (approx: sub-account share at start)
        L_.append(lg)
    lg = pd.concat(L_, ignore_index=True)
    return summarize(E_, lg, idx), E_, lg


def fmt(name, r, rs=None):
    s = (f"{name}\n"
         f"   2021-23 {r['d_mo']*100:+6.2f}%/mo dd {r['d_dd']*100:4.0f}% win {r['d_win']*100:4.1f}% n {r['d_n']:4d} | "
         f"2024-26 {r['t_mo']*100:+6.2f}%/mo dd {r['t_dd']*100:4.0f}% win {r['t_win']*100:4.1f}% n {r['t_n']:4d} | "
         f"since 2025 {r['s_mo']*100:+6.2f}%/mo dd {r['s_dd']*100:4.0f}% win {r['s_win']*100:4.1f}% | "
         f"worst trade (acct) {r['d_wacct']*100:+.0f}% / {r['t_wacct']*100:+.0f}% / {r['s_wacct']*100:+.0f}%")
    if rs is not None:
        s += (f"\n   STRESSED 2021-23 {rs['d_mo']*100:+6.2f}% dd {rs['d_dd']*100:4.0f}% | 2024-26 {rs['t_mo']*100:+6.2f}% "
              f"dd {rs['t_dd']*100:4.0f}% | since 2025 {rs['s_mo']*100:+6.2f}% dd {rs['s_dd']*100:4.0f}% | "
              f"win {rs['d_win']*100:.1f}/{rs['t_win']*100:.1f}/{rs['s_win']*100:.1f}% | worst trade "
              f"{rs['d_wacct']*100:+.0f}% / {rs['t_wacct']*100:+.0f}% / {rs['s_wacct']*100:+.0f}%")
    return s


def row(name, s, r, rs=None):
    out = {"name": name, **{k: v for k, v in s.items() if k != "sub"}, **r}
    if rs is not None:
        out.update({f"S_{k}": v for k, v in rs.items()})
    return out
