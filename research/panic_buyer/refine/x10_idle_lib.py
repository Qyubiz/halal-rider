"""Lever "idle cash": park the dip bot's idle cash in a second halal SPOT engine and sell it when dip bids need money.

ikernel = engine.gkernel (full take-profit, no partial/trail/tp2, tpwait 1, resting take-profit, backtest timing)
copied, plus a parking engine. With pk_on=False (or all parking weights 0) it is engine.run exactly
(x10_idle_check.py proves it). All decisions use the hour's open and closed candles only (no lookahead).

The dip bot is unchanged: each hour it wants fully funded bids on the N coins closest to their trigger, sized
per = min(fmax * equity, liquid / nsel), where liquid = cash + what the parking would fetch if sold now. Parking:

  armed     a wanted bid is "armed" when the coin's open is within armx of its trigger (dist <= armx; dist =
            open / trigger - 1). The cash for armed bids (Rq) must be in USDT at the open.
  target    parking value per asset m = pw[t, m] * (liquid - Rq), in the asset(s) the weight builder picked
            (BTC, ETH, both, the 19-coin momentum leader...). pw is decided from candles closed before the hour.
  open      at each hourly open (market orders at the open, fee 0.1% + slippage pslip, both ways):
            - cash < Rq ("forced"): sell parking down to target (raises exactly the reserve); asset with weight 0
              is sold completely;
            - otherwise sell an asset whose value exceeds its target by > band*equity (or whose weight is 0);
            - buy parking with cash above Rq when the gap to target exceeds band*equity, but not within
              rebuy_wait hours after a forced or watcher sale.
            Then the bids are placed exactly as engine.gkernel (closest first, a = min(per, cash)).
  mode 0    unarmed coins get a bid only from leftover cash (normally none while parked): fills that would come
            from a fall of more than armx inside one hour are missed.
  mode 1    "watcher": a 1-minute watcher sells parking as soon as an unarmed bid coin trades within armx of its
            trigger. Modelled conservatively on hourly candles: if the coin's hourly low reaches trigger*(1+armx)
            (it must have crossed that level before any fill, since it opened above it), the missing cash is raised
            by selling parking at the parking asset's LOW of that hour (worst price of the hour), then the bid is
            placed and fills as usual if the low trades through it.
Costs: dip fills as engine (0.1% fee, 0.02% slippage for market buys/time stops on the 19 coins, 0.1% for
LUNA/FTT/SRM, limit buys need 0.1% through). Stress: pen (buys) and pens (sells) make every fill worse, including
parking fills. Fully funded: every bid <= cash at placement; parking is bought only with cash above the reserve.
"""
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402

OUT = os.environ.get("PB_OUT", os.path.expanduser("~/pb_out_x10_idle"))
os.makedirs(OUT, exist_ok=True)
T_TEST0, T_25 = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")
LOG = E.LOG_COLS + ["eq0", "mkt"]
L = E.LIVE
BIG = 1e9


@njit(cache=True)
def ikernel(o, h, l, c, ref, depth, nord, fmax, smult, tpa, hm, cd, eps, fee, slip, pen, pens,
            po, pl, pc, pw, pslip, armx, mode, band, rebuy_wait, pk_on, pfee):
    n, K = c.shape
    M = po.shape[1]
    units = np.zeros(K); entry = np.zeros(K); amt0 = np.zeros(K); tpp = np.zeros(K)
    ebar = np.full(K, -1, np.int64); okbar = np.zeros(K, np.int64)
    lastc = np.full(K, np.nan); dist = np.empty(K)
    eq0 = np.zeros(K); rawm = np.zeros(K)
    pu = np.zeros(M); plast = np.full(M, np.nan); tgt = np.zeros(M); vn = np.zeros(M)
    last_forced = -10**9
    cash = 1.0; eq = np.empty(n); eq_prev = 1.0
    cashv = np.empty(n); parkv = np.empty(n); resv = np.zeros(n); pflow = np.zeros(n)
    stats = np.zeros(8)   # 0 forced sells, 1 band sells, 2 buys, 3 watcher sells, 4 turnover (x equity), 5 costs (x eq)
    maxlog = 600000; lg = np.zeros((maxlog, 9)); nl = 0
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
        # ---------------- parking at the open
        liquid = cash
        if pk_on:
            for m in range(M):
                vn[m] = 0.0
                if pu[m] > 0.0:
                    px = po[t, m] if not np.isnan(po[t, m]) else plast[m]
                    vn[m] = pu[m] * px * (1.0 - pslip[m]) * (1.0 - pens) * (1.0 - pfee)
                    liquid += vn[m]
        per = 0.0
        if nsel > 0:
            per = min(fmax[t] * eq_prev, liquid / nsel)
        if pk_on:
            rq = 0.0
            for q in range(nsel):
                k = order[q]
                if dist[k] <= armx:
                    rq += per * smult[t, k]
            if rq > liquid:
                rq = liquid
            resv[t] = rq
            avail = liquid - rq
            if avail < 0.0:
                avail = 0.0
            forced = cash < rq * (1.0 - 1e-12)
            for m in range(M):
                tgt[m] = pw[t, m] * avail
            # sells
            for m in range(M):
                if pu[m] <= 0.0 or np.isnan(po[t, m]):
                    continue
                if pw[t, m] <= 0.0 or (forced and vn[m] > tgt[m]) or vn[m] - tgt[m] > band * eq_prev:
                    if pw[t, m] <= 0.0 or tgt[m] <= 0.0:
                        frac = 1.0
                    else:
                        frac = 1.0 - tgt[m] / vn[m]
                    got = vn[m] * frac
                    gross = pu[m] * frac * po[t, m]
                    cash += got; pu[m] -= pu[m] * frac; pflow[t] -= got
                    if frac >= 1.0:
                        pu[m] = 0.0
                    stats[4] += gross / eq_prev; stats[5] += (gross - got) / eq_prev
                    if forced:
                        stats[0] += 1.0; last_forced = t
                    else:
                        stats[1] += 1.0
            # buys
            exc = cash - rq
            if exc > band * eq_prev and t - last_forced >= rebuy_wait:
                need = 0.0
                for m in range(M):
                    if pw[t, m] > 0.0 and not np.isnan(po[t, m]) and tgt[m] > vn[m]:
                        need += tgt[m] - vn[m]
                if need > band * eq_prev:
                    scale = min(1.0, exc / need)
                    for m in range(M):
                        if pw[t, m] > 0.0 and not np.isnan(po[t, m]) and tgt[m] > vn[m]:
                            b = (tgt[m] - vn[m]) * scale
                            fillp = po[t, m] * (1.0 + pslip[m]) * (1.0 + pen)
                            pu[m] += b * (1.0 - pfee) / fillp
                            cash -= b; pflow[t] += b
                            stats[2] += 1.0; stats[4] += b / eq_prev
                            stats[5] += b * (1.0 - (1.0 - pfee) * po[t, m] / fillp) / eq_prev
        # ---------------- bids (engine.gkernel)
        if nsel > 0 and (cash > 1e-9 or pk_on):
            for q in range(nsel):
                k = order[q]
                lim = ref[t, k] * (1.0 - depth[t, k])
                want = per * smult[t, k]
                if pk_on and mode == 1 and dist[k] > armx and cash < want and l[t, k] <= lim * (1.0 + armx):
                    # watcher: sell parking at the hour's low (worst case) to fund this bid
                    needc = want - cash
                    for m in range(M):
                        if needc <= 0.0:
                            break
                        if pu[m] <= 0.0 or np.isnan(pl[t, m]):
                            continue
                        unitnet = pl[t, m] * (1.0 - pslip[m]) * (1.0 - pens) * (1.0 - pfee)
                        su = min(pu[m], needc / unitnet)
                        got = su * unitnet
                        cash += got; pu[m] -= su; needc -= got; pflow[t] -= got
                        if pu[m] * unitnet < 1e-12:
                            pu[m] = 0.0
                        stats[3] += 1.0; stats[4] += su * pl[t, m] / eq_prev
                        stats[5] += (su * pl[t, m] - got) / eq_prev
                        last_forced = t
                fill = -1.0; ismkt = 0
                if o[t, k] <= lim:
                    fill = o[t, k] * (1.0 + slip[k]) * (1.0 + pen); ismkt = 1
                elif l[t, k] <= lim * (1.0 - eps):
                    fill = lim * (1.0 + pen)
                if fill > 0.0 and cash > 1e-9:
                    a = min(want, cash)
                    if pk_on and a < 1e-6 * eq_prev:
                        continue
                    cash -= a; units[k] = a * (1.0 - fee) / fill
                    entry[k] = fill; amt0[k] = a; ebar[k] = t
                    tpp[k] = fill * (1.0 + tpa[t, k]); eq0[k] = eq_prev; rawm[k] = ismkt
        # ---------------- exits (engine.gkernel, full take-profit)
        for k in range(K):
            if units[k] > 0.0 and ebar[k] < t and not np.isnan(o[t, k]):
                px = -1.0; why = 0.0
                if o[t, k] >= tpp[k]:
                    px = o[t, k]
                elif h[t, k] >= tpp[k] * (1.0 + eps):
                    px = tpp[k]
                if px < 0.0 and t - ebar[k] >= hm[t]:
                    px = c[t, k] * (1.0 - slip[k]); why = 1.0
                if px > 0.0:
                    proceeds = units[k] * px * (1.0 - pens) * (1.0 - fee); cash += proceeds
                    if nl < maxlog:
                        lg[nl, 0] = ebar[k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = why
                        lg[nl, 4] = proceeds / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]
                        lg[nl, 7] = eq0[k]; lg[nl, 8] = rawm[k]; nl += 1
                    units[k] = 0.0; okbar[k] = t + cd
        v = cash
        for k in range(K):
            if not np.isnan(c[t, k]):
                lastc[k] = c[t, k]
            if units[k] > 0.0:
                v += units[k] * lastc[k]
        pv = 0.0
        for m in range(M):
            if not np.isnan(pc[t, m]):
                plast[m] = pc[t, m]
            if pu[m] > 0.0:
                pv += pu[m] * plast[m]
        v += pv
        eq[t] = v; eq_prev = v; cashv[t] = cash; parkv[t] = pv
    return eq, lg[:nl].copy(), cashv, parkv, resv, stats, pflow


def slip_vec(syms):
    return np.array([E.SLIP_MAJOR if s in E.H19 else E.SLIP_OTHER for s in syms])


_mk = {}


def market(dead=False):
    """(syms, idx, P): engine.panel(engine.load_h19(dead)), cached (19 coins, + LUNA/FTT/SRM if dead)."""
    if dead not in _mk:
        _mk[dead] = E.panel(E.load_h19(dead=dead))
    return _mk[dead]


def irun(P, idx, pw=None, depth=L["depth"], N=L["N"], fmax=L["fmax"], smult=1.0, tp=L["tp"], hm=L["hm"],
         cd=L["cd"], eps=E.EPS, fee=E.FEE, slip="auto", pen=0.0, pens=0.0, armx=BIG, mode=0, band=0.05,
         rebuy_wait=0, pfee=None, pslip=None):
    """pw: None (no parking) or an (n, K) DataFrame/array of parking weights over P's columns (sum <= 1).
    pfee / pslip: fee and slippage of PARKING trades (default: same as the dip bot). Diagnostic only:
    pfee=0, pslip=0 gives the frictionless upper bound of the lever."""
    O = P["o"].values; n, K = O.shape
    syms = list(P["o"].columns)
    sl = slip_vec(syms) if isinstance(slip, str) else (np.full(K, slip) if np.ndim(slip) == 0 else np.asarray(slip))
    pk_on = pw is not None
    W = np.zeros((n, K)) if pw is None else np.ascontiguousarray(np.asarray(pw, np.float64))
    eq, lg, cashv, parkv, resv, st, pflow = ikernel(
        O, P["h"].values, P["l"].values, P["cf"].values, P["ref"].values,
        E.full(n, K, depth), E.full(n, 0, N, np.int64), E.full(n, 0, fmax), E.full(n, K, smult), E.full(n, K, tp),
        E.full(n, 0, hm, np.int64), int(cd), eps, fee, sl, pen, pens,
        O, P["l"].values, P["cf"].values, W,
        (sl if pslip is None else np.full(K, float(pslip))).astype(np.float64), float(armx), int(mode), float(band),
        int(rebuy_wait), bool(pk_on), float(fee if pfee is None else pfee))
    aux = pd.DataFrame({"cash": cashv, "park": parkv, "res": resv, "equity": eq, "pflow": pflow}, idx)
    stats = dict(zip(["forced", "band_sells", "buys", "watch_sells", "turnover", "costs"], st[:6]))
    return pd.Series(eq, idx), pd.DataFrame(lg, columns=LOG), aux, stats


# ------------------------------------------------------------------ parking weight builders (no lookahead)
def daily_hold(sig, idx):
    """Evaluate a signal only at 00:00 UTC hours and hold it for the day (cuts churn)."""
    s = pd.DataFrame(sig, index=idx).copy()
    s[s.index.hour != 0] = np.nan
    return s.ffill().fillna(0.0)


def w_fixed(P, idx, assets, filt_days=0):
    """Fixed weights {sym: w}; filt_days>0: an asset's weight is 0 on days its last close (23:00 UTC) is under
    its filt_days-day average of hourly closes."""
    cols = list(P["o"].columns)
    W = pd.DataFrame(0.0, idx, cols)
    for s, w in assets.items():
        W[s] = w
    if filt_days:
        cf = P["cf"]
        on = (cf > cf.rolling(24 * filt_days, min_periods=24 * filt_days).mean()).astype(float).shift(1)
        W = W * daily_hold(on, idx)
    return W.fillna(0.0).values


def w_mom(P, idx, days=30, top=1, absmom=True, filt_days=0, cands=None, weekly=False):
    """Each day at 00:00 UTC: the `top` coins with the best `days`-day return (closes up to 23:00), equal
    weight; absmom: only coins with a positive return; filt_days: and close above their filt_days average.
    cands: candidate symbols (default all columns of P, incl. LUNA/FTT/SRM when loaded)."""
    cf = P["cf"]
    cols = list(cf.columns)
    cands = cols if cands is None else cands
    mom = (cf / cf.shift(24 * days) - 1).shift(1)
    ok = mom.notna() & P["o"].notna()
    if absmom:
        ok &= mom > 0
    if filt_days:
        ok &= (cf > cf.rolling(24 * filt_days, min_periods=24 * filt_days).mean()).shift(1).fillna(False).astype(bool)
    m = mom.where(ok).copy()
    for s in cols:
        if s not in cands:
            m[s] = np.nan
    m = m[m.index.hour == 0]
    if weekly:                                   # rebalance only on Mondays 00:00 UTC
        m = m[m.index.dayofweek == 0]
    rank = m.rank(axis=1, ascending=False, method="first")
    sel = (rank <= top).astype(float)
    sel = sel.div(top)
    W = sel.reindex(idx).ffill().fillna(0.0)
    # a coin whose data ended (LUNA/FTT/SRM) cannot be held: weight 0 when its open is missing
    W = W.where(P["o"].notna(), 0.0)
    return W.values


# ------------------------------------------------------------------ metrics (same conventions as x10_stack_lib)
def period(e, a, b=None):
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


def summarize(e, lg, idx, aux=None, stats=None):
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
        r["t_win"] = float((lg.ret[tt] > 0).mean()) if tt.any() else np.nan
        r["t_avg"] = float(lg.ret[tt].mean()) if tt.any() else np.nan
    if aux is not None:
        sh = aux.cash / aux.equity
        pk = aux.park / aux.equity
        for tag, msk in (("d", aux.index < T_TEST0), ("t", aux.index >= T_TEST0)):
            r[f"{tag}_cash"] = float(sh[msk].mean())
            r[f"{tag}_park"] = float(pk[msk].mean())
    if stats is not None:
        r.update({f"pk_{k}": float(v) for k, v in stats.items()})
        if aux is not None and "pflow" in aux:
            pp = aux.park.diff().fillna(aux.park.iloc[0]) - aux.pflow          # parking P&L per hour, after costs
            eqp = aux.equity.shift(1).fillna(1.0)
            day = pp.resample("D").sum() / eqp.resample("D").first()
            r["pk_worst_day"] = float(day.min())
            held = (aux.park > 1e-12) | (aux.park.shift(1).fillna(0.0) > 1e-12)
            rid = (held != held.shift()).cumsum()
            g = pd.DataFrame({"pp": pp, "e0": eqp, "rid": rid})[held.values]
            if len(g):
                runs = g.groupby("rid").agg(pp=("pp", "sum"), e0=("e0", "first"))
                r["pk_worst_run"] = float((runs.pp / runs.e0).min())
    return r


def fmt4(name, r, rs=None):
    s = (f"{name}\n  (1) 2024-26  {r['t_mo']*100:+6.2f}%/mo  x{r['t_mult']:7.2f}  drop {r['t_dd']*100:5.1f}%\n"
         f"  (2) 2025-26  {r['s_mo']*100:+6.2f}%/mo  x{r['s_mult']:7.2f}  drop {r['s_dd']*100:5.1f}%\n"
         f"  (3) 2021-23  {r['d_mo']*100:+6.2f}%/mo  x{r['d_mult']:7.2f}  drop {r['d_dd']*100:5.1f}%  2022 {r['y2022']*100:+5.1f}%")
    if "worst_acct" in r:
        s += (f"\n      trades {r['trades']}, worst trade {r['worst_acct']*100:+.1f}% of account (2021-23 "
              f"{r['worst_acct_d']*100:+.1f}%, 2024-26 {r['worst_acct_t']*100:+.1f}%)")
    if "t_cash" in r:
        s += (f"\n      avg cash share 21-23 {r['d_cash']*100:.0f}% / 24-26 {r['t_cash']*100:.0f}%, parked 21-23 "
              f"{r['d_park']*100:.0f}% / 24-26 {r['t_park']*100:.0f}%")
    if "pk_turnover" in r:
        s += (f"; parking trades: {r['pk_forced']:.0f} forced, {r['pk_watch_sells']:.0f} watcher, "
              f"{r['pk_band_sells']:.0f} other sells, {r['pk_buys']:.0f} buys, turnover x{r['pk_turnover']:.0f}, "
              f"costs {r['pk_costs']*100:.1f}% of equity (sum)")
    if "pk_worst_run" in r:
        s += (f"\n      parking: worst day {r['pk_worst_day']*100:+.1f}% of account, worst single parked stretch "
              f"{r['pk_worst_run']*100:+.1f}% of account")
    if rs is not None:
        s += (f"\n  (4) fills 0.2% worse (all fills) + LUNA/FTT/SRM: 2024-26 {rs['t_mo']*100:+6.2f}%/mo drop "
              f"{rs['t_dd']*100:5.1f}% | 2025-26 {rs['s_mo']*100:+6.2f}%/mo drop {rs['s_dd']*100:5.1f}% | 2021-23 "
              f"{rs['d_mo']*100:+6.2f}%/mo drop {rs['d_dd']*100:5.1f}% 2022 {rs['y2022']*100:+5.1f}%")
        if "worst_acct" in rs:
            s += f" | worst trade {rs['worst_acct']*100:+.1f}% of account"
    return s


# ------------------------------------------------------------------ design specs
BASES = {
    "live 8x20% +1.5%": dict(depth=0.12, N=8, fmax=0.20, tp=0.015, hm=48, cd=24),
    "2x50% +5%": dict(depth=0.12, N=2, fmax=0.50, tp=0.05, hm=48, cd=24),
    "2x50% +3%": dict(depth=0.12, N=2, fmax=0.50, tp=0.03, hm=48, cd=24),
    "1x100% +3%": dict(depth=0.12, N=1, fmax=1.00, tp=0.03, hm=48, cd=24),
}
BOT_KEYS = ("depth", "N", "fmax", "tp", "hm", "cd")
PK_KEYS = ("armx", "mode", "band", "rebuy_wait")


def weights_for(P, idx, park):
    """park: None | dict(kind='fixed', assets={sym: w}, filt=days) | dict(kind='mom', days, top, absmom, filt).
    Options (round 2): renorm=True (fixed: weights of the assets whose filter is on are rescaled to sum 1),
    pf (scale all weights, e.g. 0.5 = park half of the idle cash), calm=x (weight 0 in hours when BTC's last
    close is more than x under its highest close of the last 24h; decided at each open from closed candles)."""
    if park is None:
        return None
    if park["kind"] == "fixed":
        W = w_fixed(P, idx, park["assets"], park.get("filt", 0))
        if park.get("renorm"):
            s = W.sum(axis=1, keepdims=True)
            W = np.where(s > 0, W / np.where(s > 0, s, 1.0), 0.0)
    elif park["kind"] == "mom":
        W = w_mom(P, idx, park.get("days", 30), park.get("top", 1), park.get("absmom", True), park.get("filt", 0),
                  park.get("cands"), park.get("weekly", False))
    else:
        raise ValueError(park)
    W = W * float(park.get("pf", 1.0))
    if park.get("calm", 0) > 0:
        cf = P["cf"]["BTCUSDT"]
        dd = (cf / cf.rolling(24, min_periods=1).max() - 1).shift(1).fillna(0.0).values
        W = W * (dd > -park["calm"])[:, None]
    return np.ascontiguousarray(W, dtype=np.float64)


def evaluate(bot, park=None, pk=None, stress=False):
    """bot: dip-bot settings; park: weight spec; pk: armx/mode/band/rebuy_wait. stress: + LUNA/FTT/SRM and
    every fill (dip and parking, buys and sells) 0.2% worse."""
    syms, idx, P = market(stress)
    kw = {k: bot[k] for k in BOT_KEYS if k in bot}
    kw.update(pk or {})
    if stress:
        kw.update(pen=0.002, pens=0.002)
    pw = weights_for(P, idx, park)
    e, lg, aux, st = irun(P, idx, pw=pw, **kw)
    return summarize(e, lg, idx, aux, st if pw is not None else None), e, lg, aux
