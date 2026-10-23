"""m10 "velocity" angle: put the Panic Buyer's idle cash to work at a 90%+ win rate with several triggers at once.

vk = w90_wr90_lib.wkernel (hourly, full take-profit, crash-guard mask, sell penalty, collapse-coin liquidation, open
positions logged as why=9) generalised to TIERS. A tier is one trigger rule with its own exit and size:
  ref     trigger reference: >0 = the highest hourly close of the last `ref` closed hours (24 = the live bot's 24h high;
          3..12 = a fast "flash dip" reference); -1 = re-entry: the coin's last winning exit price (valid `rewin` hours)
  depth   bid at reference x (1 - depth)
  tp, hm  take-profit (resting limit from the next hourly check) and time stop in hours (sold at that close)
  cdw,cdl cooldown in hours after a winning / losing exit (cdw 0..1 = "re-arm the coin at the next hourly check")
  fs      largest order as a fraction of the account (sized on the previous hourly close, like the engine)
  near    bid only when the hour's open is at most `near` above the trigger (1e8 = off)
  nmax    most positions this tier may hold at once (excl=0 only)
Account switches:
  nslots  most positions held at once over all tiers
  nbid    most new bids per hour (the engine's N; 99 = off)
  excl    1 = one position per coin: each coin gets ONE resting bid, at the highest of its tiers' trigger prices (price
          falls through that one first), and that tier's tp/hm/fs apply; cooldowns are per coin.
          0 = tiers are independent sleeves sharing one cash pool: a coin can be held by two tiers (a ladder); cooldowns
          per tier and coin.
  dyn     sizing by how many signals are active. 0: order = min(fs x account, free cash / number of bids this hour)
          (engine). 1: also at most account / (positions held + bids this hour) (equal weight). 2: fs is the size for a
          LONE signal; with m bids this hour each gets min(fs, 1/m) of the account, and each later signal gets what is free.
  redip   re-entry tier price = winning exit price x (1 - redip)
Costs (all as the refine engines): fee 0.1% a side, market buys (open already under the trigger) and time-stop sells
0.02% slippage (0.1% for LUNA/FTT/SRM when liquidate), limit buys fill only when the low trades 0.1% through, a
resting take-profit only when the high trades 0.1% through, all orders funded from free cash (spot, no leverage).
pen / pens = every buy / sell fill worse by that fraction (stress 0.002). mtp_now = the real bot's timing after a
market buy (take-profit placed at once). Cash from a sale is usable at the next hourly check.
Only information known at each hour's open is used: references and the crash guard come from closed candles.

With one tier (ref 24) and nslots=1, excl=1, dyn=0 the kernel is w90_wr90_lib.wrun(N=1, fmax=fs, ...) exactly
(m10_velocity_check.py). The 90% mode is TIER90 below.
"""
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402

OUT = os.environ.get("PB_OUT", os.path.expanduser("~/pb_out_m10/velocity"))
os.makedirs(OUT, exist_ok=True)
T21, T24, T25 = pd.Timestamp("2021-01-01"), pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")
H18, H21 = pd.Timestamp("2018-01-01"), pd.Timestamp("2021-01-01")
LOGC = ["eb", "xb", "k", "why", "ret", "amt", "entry", "eq0", "tier"]   # why 0 tp, 1 time stop, 5 cut, 9 open at end
TIER90 = dict(ref=24, depth=0.14, tp=0.025, hm=96, cdw=3, cdl=3, fs=1.0, near=1e8, nmax=19)
MODE90 = dict(tiers=[TIER90], nslots=1, excl=1, dyn=0, guard=0.35)
STRESS_PEN = 0.002


@njit(cache=True)
def vk(o, h, l, c, gm, R, rid, lastb, dep, tpa, hm, cdw, cdl, fs, near, nmaxs, nslots, nbid, excl, dyn,
       eps, fee, slip, pen, pens, mtp_now, redip, rewin):
    n, K = c.shape
    S = dep.shape[0]
    units = np.zeros((S, K)); entry = np.zeros((S, K)); amt0 = np.zeros((S, K)); tpp = np.zeros((S, K))
    eq0 = np.zeros((S, K)); ebar = np.full((S, K), -1, np.int64); mkt = np.zeros((S, K), np.int64)
    okt = np.zeros((S, K), np.int64)            # cooldown per tier and coin (excl=0) ...
    okc = np.zeros(K, np.int64)                 # ... or per coin (excl=1)
    rex = np.zeros(K); rexu = np.full(K, -1, np.int64)
    lastc = np.full(K, np.nan)
    cash = 1.0; eq = np.empty(n); inv = np.empty(n); eq_prev = 1.0
    maxlog = 800000; lg = np.zeros((maxlog, 9)); nl = 0
    nc = S * K
    cdist = np.empty(nc); cs = np.empty(nc, np.int64); ck = np.empty(nc, np.int64); clim = np.empty(nc)
    held_s = np.zeros(S, np.int64); take_s = np.zeros(S, np.int64)
    bs = np.empty(nc, np.int64); bk = np.empty(nc, np.int64); blim = np.empty(nc)
    for t in range(n):
        # ---------------- candidate bids
        nheld = 0
        for s in range(S):
            held_s[s] = 0
            for k in range(K):
                if units[s, k] > 0.0:
                    held_s[s] += 1
            nheld += held_s[s]
        m = 0
        if excl == 1:
            for k in range(K):
                busy = False
                for s in range(S):
                    if units[s, k] > 0.0:
                        busy = True
                if busy or t < okc[k] or np.isnan(o[t, k]) or gm[t, k] <= 0.0:
                    continue
                best = -1.0; bsel = -1
                for s in range(S):
                    r = rid[s]
                    if r < 0:
                        if t >= rexu[k]:
                            continue
                        lim = rex[k] * (1.0 - redip)
                    else:
                        if np.isnan(R[r, t, k]):
                            continue
                        lim = R[r, t, k] * (1.0 - dep[s])
                    if o[t, k] / lim - 1.0 > near[s]:
                        continue
                    if lim > best:
                        best = lim; bsel = s
                if bsel >= 0:
                    cdist[m] = o[t, k] / best - 1.0; cs[m] = bsel; ck[m] = k; clim[m] = best; m += 1
        else:
            for s in range(S):
                r = rid[s]
                for k in range(K):
                    if units[s, k] > 0.0 or t < okt[s, k] or np.isnan(o[t, k]) or gm[t, k] <= 0.0:
                        continue
                    if r < 0:
                        if t >= rexu[k]:
                            continue
                        lim = rex[k] * (1.0 - redip)
                    else:
                        if np.isnan(R[r, t, k]):
                            continue
                        lim = R[r, t, k] * (1.0 - dep[s])
                    d = o[t, k] / lim - 1.0
                    if d > near[s]:
                        continue
                    cdist[m] = d; cs[m] = s; ck[m] = k; clim[m] = lim; m += 1
        nb = 0
        if m > 0 and cash > 1e-9:
            order = np.argsort(cdist[:m])
            free = min(nslots - nheld, nbid)
            for s in range(S):
                take_s[s] = 0
            for q in range(m):
                if nb >= free:
                    break
                j = order[q]
                s = cs[j]
                if excl == 0 and held_s[s] + take_s[s] >= nmaxs[s]:
                    continue
                take_s[s] += 1
                bs[nb] = s; bk[nb] = ck[j]; blim[nb] = clim[j]; nb += 1
        if nb > 0:
            cash0 = cash
            for q in range(nb):
                s = bs[q]; k = bk[q]; lim = blim[q]
                if dyn == 2:
                    per = min(fs[s], 1.0 / nb) * eq_prev
                    per = min(per, cash0 / nb)
                else:
                    per = min(fs[s] * eq_prev, cash0 / nb)
                    if dyn == 1:
                        per = min(per, eq_prev / (nheld + nb))
                fill = -1.0; ismkt = 0
                if o[t, k] <= lim:
                    fill = o[t, k] * (1.0 + slip[k]) * (1.0 + pen); ismkt = 1
                elif l[t, k] <= lim * (1.0 - eps):
                    fill = lim * (1.0 + pen)
                if fill > 0.0 and cash > 1e-9:
                    mkt[s, k] = ismkt if mtp_now else 0
                    a = min(per * gm[t, k], cash)
                    cash -= a; units[s, k] = a * (1.0 - fee) / fill
                    entry[s, k] = fill; amt0[s, k] = a; ebar[s, k] = t; eq0[s, k] = eq_prev
                    tpp[s, k] = fill * (1.0 + tpa[s])
        # ---------------- exits
        for s in range(S):
            for k in range(K):
                if units[s, k] > 0.0 and (ebar[s, k] < t or mkt[s, k] == 1) and not np.isnan(o[t, k]):
                    px = -1.0; why = 0.0
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
                    if px > 0.0:
                        proceeds = units[s, k] * px * (1.0 - pens) * (1.0 - fee); cash += proceeds
                        if nl < maxlog:
                            lg[nl, 0] = ebar[s, k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = why
                            lg[nl, 4] = proceeds / amt0[s, k] - 1.0; lg[nl, 5] = amt0[s, k]; lg[nl, 6] = entry[s, k]
                            lg[nl, 7] = eq0[s, k]; lg[nl, 8] = s; nl += 1
                        units[s, k] = 0.0
                        win = proceeds > amt0[s, k]
                        ok = t + (cdw[s] if win else cdl[s])
                        if excl == 1:
                            okc[k] = ok
                        else:
                            okt[s, k] = ok
                        if win and rewin > 0:
                            rex[k] = px; rexu[k] = t + 1 + rewin
            for k in range(K):
                if units[s, k] > 0.0 and lastb[k] == t:
                    proceeds = units[s, k] * c[t, k] * (1.0 - slip[k]) * (1.0 - pens) * (1.0 - fee); cash += proceeds
                    if nl < maxlog:
                        lg[nl, 0] = ebar[s, k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = 5.0
                        lg[nl, 4] = proceeds / amt0[s, k] - 1.0; lg[nl, 5] = amt0[s, k]; lg[nl, 6] = entry[s, k]
                        lg[nl, 7] = eq0[s, k]; lg[nl, 8] = s; nl += 1
                    units[s, k] = 0.0
                    okc[k] = t + cdl[s]; okt[s, k] = t + cdl[s]
        # ---------------- hourly mark
        for k in range(K):
            if not np.isnan(c[t, k]):
                lastc[k] = c[t, k]
        v = cash; w = 0.0
        for k in range(K):
            for s in range(S):
                if units[s, k] > 0.0:
                    v += units[s, k] * lastc[k]; w += units[s, k] * lastc[k]
        eq[t] = v; eq_prev = v; inv[t] = w / v if v > 0.0 else 0.0
    for s in range(S):
        for k in range(K):
            if units[s, k] > 0.0 and nl < maxlog:
                proceeds = units[s, k] * lastc[k] * (1.0 - slip[k]) * (1.0 - pens) * (1.0 - fee)
                lg[nl, 0] = ebar[s, k]; lg[nl, 1] = n - 1; lg[nl, 2] = k; lg[nl, 3] = 9.0
                lg[nl, 4] = proceeds / amt0[s, k] - 1.0; lg[nl, 5] = amt0[s, k]; lg[nl, 6] = entry[s, k]
                lg[nl, 7] = eq0[s, k]; lg[nl, 8] = s; nl += 1
    return eq, lg[:nl].copy(), inv


# ------------------------------------------------------------------ inputs from closed candles
def guard_mask(P, g, L=168):
    """1 = allowed: the coin's last close is no more than g under its highest close of the previous L hours (known at
    the hour's open; panic_buyer.py crash_guard_pct). g <= 0: all allowed. Uncached, safe for any panel."""
    if g <= 0:
        return np.ones(P["o"].shape)
    cf = P["cf"]
    hi = cf.rolling(L, min_periods=1).max()
    return ((cf / hi - 1) >= -g).shift(1, fill_value=True).values.astype(float)


def ref_panel(P, W):
    """Highest hourly close of the last W closed hours (W=24 is P['ref'], the engine's trigger reference)."""
    if W == 24:
        return P["ref"].values
    return P["c"].rolling(W, min_periods=1).max().shift(1).values


def last_bars(P):
    o = P["o"]
    out = np.full(o.shape[1], -1, np.int64)
    for j, s in enumerate(o.columns):
        if s in E.DEAD:
            v = np.flatnonzero(~np.isnan(o[s].values))
            out[j] = int(v[-1])
    return out


TIER_DEF = dict(ref=24, depth=0.14, tp=0.025, hm=96, cdw=3, cdl=3, fs=1.0, near=1e8, nmax=19)


def vrun(P, idx, tiers, nslots=1, nbid=99, excl=1, dyn=0, guard=0.35, gL=168, pen=0.0, pens=0.0, mtp_now=False,
         liquidate=False, cd_add=0, redip=0.0, rewin=0, eps=E.EPS, fee=E.FEE, slip=E.SLIP_MAJOR):
    """Run the tiers on panel P. cd_add adds hours to every cooldown (the real bot counts them from the next hourly
    check: stress uses 1). Returns (equity Series, log DataFrame, invested-fraction Series)."""
    T = [dict(TIER_DEF, **t) for t in tiers]
    O = P["o"].values
    n, K = O.shape
    refs = sorted({t["ref"] for t in T if t["ref"] > 0})
    R = np.stack([ref_panel(P, w) for w in refs]) if refs else np.full((1, n, K), np.nan)
    rid = np.array([refs.index(t["ref"]) if t["ref"] > 0 else -1 for t in T], np.int64)
    sl = np.array([slip if s in E.H19 else E.SLIP_OTHER for s in P["o"].columns]) if liquidate else np.full(K, slip)
    lastb = last_bars(P) if liquidate else np.full(K, -1, np.int64)
    gm = np.ascontiguousarray(guard_mask(P, guard, gL))
    f = lambda key, dt=np.float64: np.array([t[key] for t in T], dt)   # noqa: E731
    eq, lg, inv = vk(O, P["h"].values, P["l"].values, P["cf"].values, gm, np.ascontiguousarray(R), rid, lastb,
                     f("depth"), f("tp"), f("hm", np.int64), f("cdw", np.int64) + cd_add, f("cdl", np.int64) + cd_add,
                     f("fs"), f("near"), f("nmax", np.int64), int(nslots), int(nbid), int(excl), int(dyn), float(eps), float(fee),
                     sl, float(pen), float(pens), bool(mtp_now), float(redip), int(rewin))
    return pd.Series(eq, idx), pd.DataFrame(lg, columns=LOGC), pd.Series(inv, idx)


# ------------------------------------------------------------------ data
_mk = {}


def market(dead=False):
    """(syms, idx, P) 2021-26: the 19 coins (+ LUNA/FTT/SRM cut at their collapse)."""
    key = ("main", dead)
    if key not in _mk:
        _mk[key] = E.panel(E.load_h19(dead=dead))
    return _mk[key]


def market_holdout(dead=False):
    """(syms, idx, P) 2018-20 holdout: the 16 live coins that existed (+ LUNA/FTT/SRM if dead)."""
    key = ("hold", dead)
    if key not in _mk:
        _mk[key] = E.panel(E.load_pre(dead=dead))
    return _mk[key]


# ------------------------------------------------------------------ metrics
MD = 365.25 / 12


def period(e, a, b):
    """Multiple, compounded %/month, worst drop on [a, b), starting from the value just before a (w90 convention)."""
    i0 = int(np.searchsorted(e.index.values, np.datetime64(a), side="left"))
    i1 = len(e) if b is None else int(np.searchsorted(e.index.values, np.datetime64(b), side="left"))
    start = e.values[i0 - 1] if i0 > 0 else 1.0
    x = np.concatenate([[1.0], e.values[i0:i1] / start])
    months = ((e.index[i1 - 1] + pd.Timedelta(hours=1)) - e.index[i0]).total_seconds() / 86400 / MD
    mult = float(x[-1])
    mo = mult ** (1.0 / months) - 1.0 if mult > 0 else -1.0
    dd = float((x / np.maximum.accumulate(x) - 1.0).min())
    return mult, mo, dd, months


MAIN = (("d", T21, T24), ("t", T24, None), ("s", T25, None))
HOLD = (("h", H18, H21),)


def summarize(e, lg, inv, idx, periods=MAIN):
    """Per period: mult, %/month, worst drop, trades bought in the period, win rate (open positions marked at the
    end count), trades/month, worst trade as % of the account it was sized on, % of hours with no position,
    average share of the account in cash, and the same win rate per tier."""
    r = {}
    tb = idx[lg.eb.astype(int).values] if len(lg) else pd.DatetimeIndex([])
    acct = (lg.ret * lg.amt / lg.eq0).values if len(lg) else np.zeros(0)
    for tag, a, b in periods:
        mult, mo, dd, months = period(e, a, b)
        r.update({f"{tag}_mult": mult, f"{tag}_mo": mo, f"{tag}_dd": dd})
        sel = np.asarray((tb >= a) & ((tb < b) if b is not None else True), bool)
        x = lg[sel]
        r[f"{tag}_n"] = int(len(x))
        r[f"{tag}_tpm"] = len(x) / months
        r[f"{tag}_wr"] = float((x.ret > 0).mean()) if len(x) else np.nan
        r[f"{tag}_avg"] = float(x.ret.mean()) if len(x) else np.nan
        r[f"{tag}_wt"] = float(acct[sel].min()) if sel.any() else np.nan
        iv = inv[(inv.index >= a) & ((inv.index < b) if b is not None else True)]
        r[f"{tag}_cashh"] = float((iv.values <= 1e-12).mean())
        r[f"{tag}_cash"] = float(1.0 - iv.values.mean())
        for s in sorted(set(lg.tier.astype(int))) if len(lg) else []:
            xs = x[x.tier == s]
            r[f"{tag}_n{s}"] = int(len(xs))
            r[f"{tag}_wr{s}"] = float((xs.ret > 0).mean()) if len(xs) else np.nan
    if periods is MAIN:
        r["y2022"] = float(e[e.index < pd.Timestamp("2023-01-01")].iloc[-1] / e[e.index < pd.Timestamp("2022-01-01")].iloc[-1] - 1)
    return r


def stress_kw(kw):
    """Real bot timing (take-profit at once after a market buy, cooldowns from the next hourly check), every buy AND
    sell fill 0.2% worse, collapse coins sold at their last hour (use with dead=True panels)."""
    k = dict(kw)
    k.update(mtp_now=True, cd_add=1, pen=STRESS_PEN, pens=STRESS_PEN, liquidate=True)
    return k


def evaluate(kw, stress=False, worse=False, holdout=False, dead=None):
    """worse=True: every buy and sell fill 0.2% worse, nothing else changed. stress=True: stress_kw on +LUNA/FTT/SRM."""
    dead = stress if dead is None else dead
    syms, idx, P = market_holdout(dead) if holdout else market(dead)
    k = stress_kw(kw) if stress else dict(kw)
    if worse:
        k.update(pen=STRESS_PEN, pens=STRESS_PEN)
    if dead and not stress:
        k["liquidate"] = True
    e, lg, inv = vrun(P, idx, **k)
    return summarize(e, lg, inv, idx, HOLD if holdout else MAIN), e, lg, inv


def fmt(name, r, periods=("d", "t", "s")):
    lab = {"d": "2021-23", "t": "2024-26", "s": "since-25", "h": "2018-20"}
    out = [name]
    for p in periods:
        out.append(f"   {lab[p]} WR {r[f'{p}_wr']*100:5.1f}% {r[f'{p}_mo']*100:+6.2f}%/mo drop {r[f'{p}_dd']*100:5.1f}% "
                   f"worst trade {r[f'{p}_wt']*100:+6.1f}% acct | {r[f'{p}_tpm']:5.1f} trades/mo | no position "
                   f"{r[f'{p}_cashh']*100:4.1f}% of hours, avg cash {r[f'{p}_cash']*100:4.1f}%")
    return "\n".join(out)
