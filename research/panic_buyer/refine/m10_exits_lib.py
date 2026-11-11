"""m10 "exits": keep 90% mode's entry (1 coin x 100%, 14% under the 24h-high close, crash guard 35%, 3h cooldown) and
search only the exit.

xkernel = w90_wr90_lib.wkernel (= engine.gkernel with full take-profit) restricted to the live tp timing (take-profit
placed at the first hourly check after the fill, resting limit, needs 0.1% trade-through) plus these exit switches.
All of them are evaluated at hourly checks (the bot runs once an hour), using only what is known at that moment.

  Entry target   tp_e = clip(tpa[entry hour, k] + tpd * extra_depth, tpmin, tpmax). tpa is precomputed from data known at
                 the entry hour's open (fixed %, or scaled by volatility / drop size, see tp_table). extra_depth =
                 lim/fill - 1 > 0 only for market buys below the trigger at the open.
  Schedule       age < a1: target tp_e. From a1 on the target is re-placed each hourly check: linear from tp_e down to
                 entry*(1+be) at a2 (a2 <= a1: a step to be at a1). be can be negative (accept a small loss).
                 a1 >= 10**8 = off (90% mode).
  Time stop      at age >= hm the position is sold at the hourly close, unless the underwater rule keeps it:
                 uwmode 0: keep it if the close is more than uw under the entry (deep losers: hold for the bounce);
                 uwmode 1: keep it if the close is no more than uw under the entry (shallow losers).
                 A kept position gets the target entry*(1+be2) and is sold at the close at age hm2.
                 uw >= 1 with uwmode 0 = plain time stop (90% mode).
  Staged exit    when the ORIGINAL target fills (not a lowered one): sell `part` of the coins there (part 1 = 90% mode).
                 The rest (part 0 = nothing sold, the target only "arms" the rest) then exits at
                   tp2  resting limit at entry*(1+tp2) (0 = off),
                   a stop at max(entry*(1+fl), peak*(1-tr)) where peak is the highest high since the first target
                        (tr 0 = no trailing; fl <= -1 = no floor),
                        stopmode 0: checked at the hourly close, sold at the close (- slippage) [bot can do this hourly];
                        stopmode 1: a resting stop order: an open at/below it sells at the open, else a low at/below it
                                    sells at the stop (- slippage); checked BEFORE tp2 in the same hour (conservative),
                   time stop at age >= hmr (0: same hm as the full position).
Logs: eb, xb, k, why (0 tp, 1 time stop, 2 stop/trail, 3 tp2, 5 collapse-coin cut, 6 kept-branch stop at hm2, 9 open at
end), ret (whole position incl. partial sales, net of fees), amt, entry, eq0 (equity when sized), tp_e.
With every exit switch off, xrun == w90_wr90_lib.wrun exactly (m10_exits_check.py).
"""
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.environ.get("PB_OUT", os.path.expanduser("~/pb_out_m10/exits"))
os.makedirs(OUT, exist_ok=True)
os.environ["PB_OUT"] = OUT            # w90_wr90_lib makes its own output dir from PB_OUT at import
import engine as E  # noqa: E402
import w90_wr90_lib as W  # noqa: E402

LOGC = ["eb", "xb", "k", "why", "ret", "amt", "entry", "eq0", "tpe"]
BASE90 = dict(depth=0.14, N=1, fmax=1.0, cd=3, guard=0.35, tp=0.025, hm=96)     # panic_config.90wr.example.json
OFF = dict(tpd=0.0, tpmin=0.0, tpmax=1.0, a1=10 ** 9, a2=0, be=0.0025, uw=2.0, uwmode=0, hm2=0, be2=0.0025,
           part=1.0, tp2=0.0, tr=0.0, fl=-2.0, stopmode=0, hmr=0)
STRESS_PEN = 0.002


@njit(cache=True)
def xkernel(o, h, l, c, ref, depth, nord, fmax, smult, tpa, cd, eps, fee, slip, pen, pens, mtp_now, lastb,
            tpd, tpmin, tpmax, a1, a2, be, hm, uw, uwmode, hm2, be2, part, tp2, tr, fl, stopmode, hmr):
    n, K = c.shape
    units = np.zeros(K); entry = np.zeros(K); amt0 = np.zeros(K); got = np.zeros(K); eq0 = np.zeros(K)
    tpe = np.zeros(K); lim0 = np.zeros(K)
    stage = np.zeros(K, np.int64); sbar = np.zeros(K, np.int64); peak = np.zeros(K); branch = np.zeros(K, np.int64)
    ebar = np.full(K, -1, np.int64); okbar = np.zeros(K, np.int64); mkt = np.zeros(K, np.int64)
    lastc = np.full(K, np.nan); dist = np.empty(K)
    cash = 1.0; eq = np.empty(n); eq_prev = 1.0
    maxlog = 600000; lg = np.zeros((maxlog, 9)); nl = 0
    for t in range(n):
        # ---------------- buys (identical to wkernel)
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
                    entry[k] = fill; amt0[k] = a; ebar[k] = t; eq0[k] = eq_prev; got[k] = 0.0
                    x = tpa[t, k]
                    if tpd != 0.0 and ismkt == 1:
                        x += tpd * (lim / o[t, k] - 1.0)
                    x = min(max(x, tpmin), tpmax)
                    tpe[k] = x; lim0[k] = lim
                    stage[k] = 0; branch[k] = 0; peak[k] = fill
        # ---------------- exits
        for k in range(K):
            if units[k] > 0.0 and (ebar[k] < t or mkt[k] == 1) and not np.isnan(o[t, k]):
                px = -1.0; why = 0.0
                age = t - ebar[k] + mkt[k]
                if stage[k] == 0:
                    # target re-placed at this hourly check (decided from the age at the check)
                    lowered = 0
                    if branch[k] == 1:
                        g = be2; lowered = 1
                    elif age >= a1:
                        lowered = 1
                        if a2 > a1:
                            w = (age - a1) / (a2 - a1)
                            if w > 1.0:
                                w = 1.0
                            g = tpe[k] + (be - tpe[k]) * w
                        else:
                            g = be
                        if g > tpe[k]:
                            g = tpe[k]
                    else:
                        g = tpe[k]
                    tpp = entry[k] * (1.0 + g)
                    if ebar[k] == t:                     # market buy, take-profit placed at once (mtp_now)
                        if h[t, k] >= tpp * (1.0 + eps):
                            px = tpp
                    elif o[t, k] >= tpp:
                        px = o[t, k]
                    elif h[t, k] >= tpp * (1.0 + eps):
                        px = tpp
                    if px > 0.0 and lowered == 0 and part < 1.0:
                        q = units[k] * part
                        p1 = q * px * (1.0 - pens) * (1.0 - fee)
                        cash += p1; got[k] += p1; units[k] -= q
                        stage[k] = 1; sbar[k] = t
                        peak[k] = px if px > h[t, k] else h[t, k]
                        px = -1.0
                    elif px < 0.0:
                        if branch[k] == 1:
                            if age >= hm2:
                                px = c[t, k] * (1.0 - slip[k]); why = 6.0
                        elif age >= hm:
                            under = c[t, k] / entry[k] - 1.0
                            keep = 0
                            if uwmode == 0 and under < -uw:
                                keep = 1
                            if uwmode == 1 and under >= -uw:
                                keep = 1
                            if keep == 1 and hm2 > age:
                                branch[k] = 1
                            else:
                                px = c[t, k] * (1.0 - slip[k]); why = 1.0
                elif sbar[k] < t:
                    S = -1.0
                    if fl > -1.0:
                        S = entry[k] * (1.0 + fl)
                    if tr > 0.0:
                        s2 = peak[k] * (1.0 - tr)
                        if s2 > S:
                            S = s2
                    if stopmode == 1 and S > 0.0:
                        if o[t, k] <= S:
                            px = o[t, k] * (1.0 - slip[k]); why = 2.0
                        elif l[t, k] <= S:
                            px = S * (1.0 - slip[k]); why = 2.0
                    if px < 0.0 and tp2 > 0.0:
                        p2 = entry[k] * (1.0 + tp2)
                        if o[t, k] >= p2:
                            px = o[t, k]; why = 3.0
                        elif h[t, k] >= p2 * (1.0 + eps):
                            px = p2; why = 3.0
                    if px < 0.0 and stopmode == 0 and S > 0.0 and c[t, k] <= S:
                        px = c[t, k] * (1.0 - slip[k]); why = 2.0
                    if px < 0.0:
                        lim_h = hmr if hmr > 0 else hm
                        if age >= lim_h:
                            px = c[t, k] * (1.0 - slip[k]); why = 1.0
                    if h[t, k] > peak[k]:
                        peak[k] = h[t, k]
                if px > 0.0:
                    proceeds = units[k] * px * (1.0 - pens) * (1.0 - fee); cash += proceeds; got[k] += proceeds
                    if nl < maxlog:
                        lg[nl, 0] = ebar[k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = why
                        lg[nl, 4] = got[k] / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]
                        lg[nl, 7] = eq0[k]; lg[nl, 8] = tpe[k]; nl += 1
                    units[k] = 0.0; okbar[k] = t + cd
        for k in range(K):
            if units[k] > 0.0 and lastb[k] == t:
                proceeds = units[k] * c[t, k] * (1.0 - slip[k]) * (1.0 - pens) * (1.0 - fee); cash += proceeds
                got[k] += proceeds
                if nl < maxlog:
                    lg[nl, 0] = ebar[k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = 5.0
                    lg[nl, 4] = got[k] / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]
                    lg[nl, 7] = eq0[k]; lg[nl, 8] = tpe[k]; nl += 1
                units[k] = 0.0; okbar[k] = t + cd
        v = cash
        for k in range(K):
            if not np.isnan(c[t, k]):
                lastc[k] = c[t, k]
            if units[k] > 0.0:
                v += units[k] * lastc[k]
        eq[t] = v; eq_prev = v
    for k in range(K):
        if units[k] > 0.0 and nl < maxlog:
            proceeds = units[k] * lastc[k] * (1.0 - slip[k]) * (1.0 - pens) * (1.0 - fee)
            lg[nl, 0] = ebar[k]; lg[nl, 1] = n - 1; lg[nl, 2] = k; lg[nl, 3] = 9.0
            lg[nl, 4] = (got[k] + proceeds) / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]
            lg[nl, 7] = eq0[k]; lg[nl, 8] = tpe[k]; nl += 1
    return eq, lg[:nl].copy()


# ------------------------------------------------------------------ target tables (known at the entry hour's open)
_feat = {}


def features(P):
    """(n, K) features, each using only closes/highs/lows of hours BEFORE the current hour (shifted by one):
    vol7  = std of hourly log returns over the previous 168 hours x sqrt(24) (daily volatility)
    rng24 = (highest high - lowest low) / last close over the previous 24 hours
    dd7   = 1 - last close / highest close of the previous 168 hours (how deep the coin is under its 7-day high)
    """
    key = id(P["cf"])
    if key in _feat and _feat[key][0] is P["cf"]:
        return _feat[key][1]
    c = P["cf"]
    lr = np.log(c).diff()
    vol7 = (lr.rolling(168, min_periods=48).std() * np.sqrt(24)).shift(1)
    hi = P["h"].rolling(24, min_periods=6).max(); lo = P["l"].rolling(24, min_periods=6).min()
    rng24 = ((hi - lo) / c).shift(1)
    dd7 = (1 - c / c.rolling(168, min_periods=1).max()).shift(1)
    F = {"vol7": vol7.values, "rng24": rng24.values, "dd7": dd7.values}
    _feat[key] = (P["cf"], F)
    return F


def tp_table(P, mode="fix", tp=0.025, ref=None, alpha=1.0, lo=0.01, hi=0.08):
    """Per (hour, coin) take-profit known at the hour's open.
    fix: tp.  vol7 / rng24 / dd7: tp * (feature / ref) ** alpha, clipped to [lo, hi]; ref = a fixed reference level of
    the feature (set from 2021-23 entries only, see REF), NaN features fall back to tp."""
    ref = REF.get(mode) if ref is None else ref
    n, K = P["o"].shape
    if mode == "fix":
        return np.full((n, K), tp)
    f = features(P)[mode]
    x = tp * np.power(np.where(np.isfinite(f) & (f > 0), f, ref) / ref, alpha)
    return np.ascontiguousarray(np.clip(x, lo, hi))


# median of each feature at 90% mode entries bought in 2021-23 (m10_exits_anatomy.py), fixed before the grid
REF = {"vol7": 0.1090, "rng24": 0.2074, "dd7": 0.1441}


def xrun(P, idx, depth=0.14, N=1, fmax=1.0, smult=1.0, tp=0.025, tpa=None, cd=3, guard=0.35, eps=E.EPS, fee=E.FEE,
         slip=E.SLIP_MAJOR, pen=0.0, pens=0.0, mtp_now=False, liquidate=False, hm=96, **ex):
    """Run xkernel. tpa: (n,K) target table (overrides tp). ex: exit switches (OFF keys)."""
    O = P["o"].values; n, K = O.shape
    x = dict(OFF); x.update(ex)
    bad = set(ex) - set(OFF)
    assert not bad, f"unknown switches {bad}"
    if np.ndim(slip) == 0:
        sl = np.array([slip if s in E.H19 else E.SLIP_OTHER for s in P["o"].columns]) if liquidate else np.full(K, slip)
    else:
        sl = np.asarray(slip, np.float64)
    sm = E.full(n, K, smult)
    if guard > 0:
        sm = np.ascontiguousarray(sm * W.guard_mask(P, guard))
    lastb = W.last_bars(P) if liquidate else np.full(K, -1, np.int64)
    ta = np.full((n, K), float(tp)) if tpa is None else np.ascontiguousarray(tpa, dtype=np.float64)
    eq, lg = xkernel(O, P["h"].values, P["l"].values, P["cf"].values, P["ref"].values,
                     E.full(n, K, depth), E.full(n, 0, N, np.int64), E.full(n, 0, fmax), sm, ta,
                     int(cd), float(eps), float(fee), sl, float(pen), float(pens), bool(mtp_now), lastb,
                     float(x["tpd"]), float(x["tpmin"]), float(x["tpmax"]), int(x["a1"]), int(x["a2"]), float(x["be"]),
                     int(hm), float(x["uw"]), int(x["uwmode"]), int(x["hm2"]), float(x["be2"]), float(x["part"]),
                     float(x["tp2"]), float(x["tr"]), float(x["fl"]), int(x["stopmode"]), int(x["hmr"]))
    return pd.Series(eq, idx), pd.DataFrame(lg, columns=LOGC)


# ------------------------------------------------------------------ data
_hold = {}


def holdout_market():
    """2018-20 holdout panel (engine.load_pre: the 16 live coins that existed then), hourly."""
    if "p" not in _hold:
        D = E.load_pre()
        _hold["p"] = E.panel(D)
    return _hold["p"]


# ------------------------------------------------------------------ metrics
PERIODS = (("d", pd.Timestamp("2021-01-01"), pd.Timestamp("2024-01-01")),
           ("t", pd.Timestamp("2024-01-01"), None),
           ("s", pd.Timestamp("2025-01-01"), None))
HPERIOD = (("h", E.H_START, E.H_END),)


def summarize(e, lg, idx, periods=PERIODS):
    """Per period: %/month (compounded), worst drop, trades bought in the period (open ones marked at the last close
    count, as in w90_wr90_lib), win rate (ret > 0 net of fees), share of 'tiny' wins (0 < ret < 0.5%), worst trade as
    % of the account, trades per month, average win / loss."""
    r = {}
    tb = idx[lg.eb.astype(int).values] if len(lg) else pd.DatetimeIndex([])
    acct = (lg.ret * lg.amt / lg.eq0).values if len(lg) else np.zeros(0)
    for tag, a, b in periods:
        m, mo, dd = W.period(e, a, b)
        b_ = idx[-1] + pd.Timedelta(hours=1) if b is None else b
        months = (b_ - a).total_seconds() / (365.25 / 12 * 86400)
        sel = (tb >= a) & ((tb < b) if b is not None else True)
        x = lg[sel]
        rr = x.ret.values
        r[f"{tag}_mult"], r[f"{tag}_mo"], r[f"{tag}_dd"] = m, mo, dd
        r[f"{tag}_n"] = int(len(x))
        r[f"{tag}_tpm"] = len(x) / months
        r[f"{tag}_wr"] = float((rr > 0).mean()) if len(x) else np.nan
        r[f"{tag}_tiny"] = float(((rr > 0) & (rr < 0.005)).mean()) if len(x) else np.nan
        r[f"{tag}_avgw"] = float(rr[rr > 0].mean()) if (rr > 0).any() else np.nan
        r[f"{tag}_avgl"] = float(rr[rr <= 0].mean()) if (rr <= 0).any() else np.nan
        r[f"{tag}_wt"] = float(acct[sel].min()) if sel.any() else np.nan
        r[f"{tag}_open"] = int((x.why == 9).sum())
        r[f"{tag}_hold"] = float((x.xb - x.eb).max()) if len(x) else np.nan
    return r


def evaluate(kw, stress=False, holdout=False):
    """kw: xrun keywords (tpa may be a callable P -> table). stress: every fill 0.2% worse (buys and sells)."""
    syms, idx, P = holdout_market() if holdout else W.market(False)
    k = dict(kw)
    if callable(k.get("tpa")):
        k["tpa"] = k["tpa"](P)
    if stress:
        k.update(pen=STRESS_PEN, pens=STRESS_PEN)
    e, lg = xrun(P, idx, **k)
    return summarize(e, lg, idx, HPERIOD if holdout else PERIODS), e, lg


def fmt_row(name, r, tags=("d", "t", "s")):
    out = [name]
    lab = {"d": "2021-23", "t": "2024-26", "s": "since-25", "h": "2018-20"}
    for tg in tags:
        out.append(f"   {lab[tg]:8s} WR {r[f'{tg}_wr']*100:5.1f}% (tiny {r[f'{tg}_tiny']*100:4.1f}%) "
                   f"{r[f'{tg}_mo']*100:+6.2f}%/mo drop {r[f'{tg}_dd']*100:5.1f}% worst trade {r[f'{tg}_wt']*100:+6.1f}% acct "
                   f"| {r[f'{tg}_tpm']:4.1f} tr/mo ({r[f'{tg}_n']}) avg win {r[f'{tg}_avgw']*100:+.2f}% "
                   f"loss {r[f'{tg}_avgl']*100:+.2f}% maxhold {r[f'{tg}_hold']:.0f}h open {r[f'{tg}_open']}")
    return "\n".join(out)
