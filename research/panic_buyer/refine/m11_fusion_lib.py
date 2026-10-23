"""m11 "fusion": ONE hourly single-account bot that fuses the best m10 entry pieces, exit pieces and risk controls.

fk = m10_velocity_lib.vk (excl=1: one position per coin, each coin ONE resting bid at the highest of its tiers'
trigger prices, that tier's exit and size apply; dyn=0 sizing) plus:
  nbid / nslots        new bids per hour (the live bot's n_orders; 1 = 90% mode) / most positions held (99 = no cap,
                       as the live bot: n_orders limits bids per hour, not positions, e.g. a risk-capped order leaves
                       cash for another coin next hour)
  per-tier entry mask  gmS[s, t, k] = crash guard x optional "calm coin" filter (skip the coin for that tier when
                       its 7-day daily volatility c_vol7 <= vol; m10_regime_lib definition, long warmed-up panel)
  per-tier target      tpS[s, t, k] known at the entry hour's open: fixed tp, or the m10 exits range-scaled target
                       tp x (rng24 / 0.2074) ** alpha clipped [lo, hi] (m10_exits_lib.features / REF)
  t0                   start hour (the "start 2 weeks later" robustness version; equity 1.0 before t0)
  guardrails.py risk limits, account level, all evaluated at the hourly close like the live bot:
    hs      stop_loss_pct: a position whose hourly candle closes hs or more under its entry is sold at that close
            (- slippage), unless its take-profit filled in that hour. Also checked on the fill hour's close.
    zcap    max_trade_risk_pct: order money <= zcap x equity / hs (hs off: <= zcap x equity)
    killm   monthly_dd_pct: equity under (1-killm) x its highest hourly value of the last killw (720) hours since the
            last trip -> sell everything at that close, no new buys for klock hours (monthly_lock_hours)
    cln,clT loss_streak / loss_pause_hours: after cln losing trades in a row (account), no new buys for clT hours
Costs/timing exactly as vk (fee 0.1%/side, 0.02% slippage on market buys and time-stop sells, 0.1% for LUNA/FTT/SRM
when liquidate, limit fills need 0.1% trade-through, pen/pens worse fills, mtp_now bot timing, cash from a sale
usable at the next hourly check). Log why: 0 tp, 1 time stop, 4 stop-loss, 5 collapse-coin cut, 7 kill-switch,
9 open at end. With one tier and every new switch off, fk == vk == 90% mode (m11_fusion_check.py).
"""
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("PB_OUT", os.path.expanduser("~/pb_out_m11/fusion"))
OUT = os.environ["PB_OUT"]
os.makedirs(OUT, exist_ok=True)
import engine as E  # noqa: E402
import m10_velocity_lib as V  # noqa: E402
import m10_exits_lib as X  # noqa: E402

LOGC = V.LOGC
T21, T24, T25 = V.T21, V.T24, V.T25
TIERA = dict(ref=24, depth=0.14, tp=0.025, hm=96, cdw=3, cdl=3, fs=1.0, vol=0.0)
OFFRISK = dict(nbid=1, nslots=99, guard=0.35, hs=0.0, zcap=0.0, killm=0.0, killw=720, klock=168, cln=0, clT=0)
MODE90 = dict(OFFRISK, tiers=[TIERA])
PEN = 0.002


@njit(cache=True)
def fk(o, h, l, c, gmS, R, rid, lastb, dep, tpS, hm, cdw, cdl, fs, nslots, nbid, eps, fee, slip, pen, pens, mtp_now, t0,
       hs, zcap, killm, killw, klock, cln, clT):
    n, K = c.shape
    S = dep.shape[0]
    units = np.zeros((S, K)); entry = np.zeros((S, K)); amt0 = np.zeros((S, K)); tpp = np.zeros((S, K))
    eq0 = np.zeros((S, K)); ebar = np.full((S, K), -1, np.int64); mkt = np.zeros((S, K), np.int64)
    okc = np.zeros(K, np.int64)
    lastc = np.full(K, np.nan)
    cash = 1.0; eq = np.ones(n); inv = np.zeros(n); eq_prev = 1.0
    maxlog = 400000; lg = np.zeros((maxlog, 9)); nl = 0
    cdist = np.empty(K); cs = np.empty(K, np.int64); ck = np.empty(K, np.int64); clim = np.empty(K)
    bs = np.empty(K, np.int64); bk = np.empty(K, np.int64); blim = np.empty(K)
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
                    cdist[m] = o[t, k] / best - 1.0; cs[m] = bsel; ck[m] = k; clim[m] = best; m += 1
        nb = 0
        if m > 0 and cash > 1e-9:
            order = np.argsort(cdist[:m])
            free = min(nslots - nheld, nbid)
            for q in range(m):
                if nb >= free:
                    break
                j = order[q]
                bs[nb] = cs[j]; bk[nb] = ck[j]; blim[nb] = clim[j]; nb += 1
        if nb > 0:
            cash0 = cash
            for q in range(nb):
                s = bs[q]; k = bk[q]; lim = blim[q]
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


# ------------------------------------------------------------------ features (known at each hour's open)
_vol = {}


def vol7_long():
    """c_vol7 exactly as m10_regime_lib.features(): std of hourly log returns of the forward-filled close over the
    previous 168 hours (min 84) x sqrt(24), shifted one hour, on the Dec 2017 - Oct 2026 joined panel (+dead coins)."""
    if "v" not in _vol:
        pre = E.load_pre(dead=True)
        post = E.load_h19(dead=True, start="2017-01-01")
        D = {}
        for s in set(pre) | set(post):
            parts = [x for x in (pre.get(s), post.get(s)) if x is not None]
            d = pd.concat([x[["c"]] for x in parts])
            D[s] = d[~d.index.duplicated(keep="last")].sort_index()
        syms = sorted(D)
        idx = pd.date_range(min(d.index[0] for d in D.values()), max(d.index[-1] for d in D.values()), freq="1h")
        C = pd.DataFrame({s: D[s]["c"] for s in syms}).reindex(idx)
        lr = np.log(C.ffill()).diff()
        _vol["v"] = lr.rolling(168, min_periods=84).std().shift(1) * np.sqrt(24)
    return _vol["v"]


_cache = {}


def _key(P, *a):
    return (id(P["o"]),) + a


def vol_mask(P, v):
    """1 = allowed: c_vol7 > v (NaN never blocks). v <= 0: all allowed."""
    if v <= 0:
        return np.ones(P["o"].shape)
    k = _key(P, "vm", v)
    if k not in _cache:
        x = vol7_long().reindex(index=P["o"].index, columns=P["o"].columns).values
        with np.errstate(invalid="ignore"):
            _cache[k] = np.where(np.isfinite(x) & (x <= v), 0.0, 1.0)
    return _cache[k]


def guard(P, g):
    k = _key(P, "g", g)
    if k not in _cache:
        _cache[k] = V.guard_mask(P, g)
    return _cache[k]


def tp_table(P, tp):
    """tp: a number (fixed) or ('rng', base, alpha, lo, hi): m10 exits range-scaled target."""
    if not isinstance(tp, tuple):
        return np.full(P["o"].shape, float(tp))
    k = _key(P, "tp", tp)
    if k not in _cache:
        _, base, alpha, lo, hi = tp
        _cache[k] = X.tp_table(P, mode="rng24", tp=base, alpha=alpha, lo=lo, hi=hi)
    return _cache[k]


def ref(P, W):
    k = _key(P, "ref", W)
    if k not in _cache:
        _cache[k] = V.ref_panel(P, W)
    return _cache[k]


# ------------------------------------------------------------------ runner
def frun(P, idx, d, pen=0.0, pens=0.0, mtp_now=False, cd_add=0, liquidate=False, t0=None):
    T = [dict(TIERA, **t) for t in d["tiers"]]
    r = dict(OFFRISK); r.update({k: v for k, v in d.items() if k != "tiers"})
    O = P["o"].values
    n, K = O.shape
    refs = sorted({t["ref"] for t in T})
    R = np.ascontiguousarray(np.stack([ref(P, w) for w in refs]))
    rid = np.array([refs.index(t["ref"]) for t in T], np.int64)
    sl = np.array([E.SLIP_MAJOR if s in E.H19 else E.SLIP_OTHER for s in P["o"].columns]) if liquidate \
        else np.full(K, E.SLIP_MAJOR)
    lastb = V.last_bars(P) if liquidate else np.full(K, -1, np.int64)
    g = guard(P, r["guard"])
    gmS = np.ascontiguousarray(np.stack([g * vol_mask(P, t["vol"]) for t in T]))
    tpS = np.ascontiguousarray(np.stack([tp_table(P, t["tp"]) for t in T]))
    f = lambda key, dt=np.float64: np.array([t[key] for t in T], dt)   # noqa: E731
    i0 = 0 if t0 is None else int(np.searchsorted(idx.values, np.datetime64(t0)))
    eq, lg, inv = fk(O, P["h"].values, P["l"].values, P["cf"].values, gmS, R, rid, lastb, f("depth"), tpS,
                     f("hm", np.int64), np.maximum(f("cdw", np.int64) + cd_add, 0),
                     np.maximum(f("cdl", np.int64) + cd_add, 0), f("fs"), int(r["nslots"]), int(r["nbid"]), float(E.EPS),
                     float(E.FEE), sl, float(pen), float(pens), bool(mtp_now), i0, float(r["hs"]), float(r["zcap"]),
                     float(r["killm"]), int(r["killw"]), int(r["klock"]), int(r["cln"]), int(r["clT"]))
    return pd.Series(eq, idx), pd.DataFrame(lg, columns=LOGC), pd.Series(inv, idx)


DPER = (("d", T21, T24),)
LATE = pd.Timestamp("2021-01-15")


def design_scores(d):
    """2021-23 ONLY (the selection data): normal, worse fills, and the 5 nearby versions used for the robust score:
    base, cooldown -1h, cooldown +1h, bot timing (take-profit at once after a market buy + cooldown +1h), start 2
    weeks later. Never computes anything after 2023."""
    syms, idx, P = V.market(False)
    out = {}
    runs = {"n": {}, "w": dict(pen=PEN, pens=PEN), "cm": dict(cd_add=-1), "cp": dict(cd_add=1),
            "bt": dict(mtp_now=True, cd_add=1), "ls": dict(t0=LATE)}
    for tag, kw in runs.items():
        e, lg, inv = frun(P, idx, d, **kw)
        per = (("d", LATE, T24),) if tag == "ls" else DPER
        r = V.summarize(e, lg, inv, idx, per)
        out[f"{tag}_wr"], out[f"{tag}_mo"], out[f"{tag}_dd"] = r["d_wr"], r["d_mo"], r["d_dd"]
        if tag == "n":
            out["n_tpm"], out["n_wt"] = r["d_tpm"], r["d_wt"]
    out["robust"] = float(np.mean([out[f"{t}_mo"] for t in ("n", "cm", "cp", "bt", "ls")]))
    out["rmin"] = float(np.min([out[f"{t}_mo"] for t in ("n", "cm", "cp", "bt", "ls")]))
    return out


WR_BAR, DD_BAR = 0.9375, -0.645     # 90% mode 2021-23: 93.8% (rounded) and -64% (rounded; exact -64.4%)


def passes(s):
    return (s["n_wr"] >= WR_BAR and s["n_dd"] > DD_BAR and s["w_wr"] >= WR_BAR and s["w_dd"] > DD_BAR)


# ------------------------------------------------------------------ full evaluation (final look only)
def evaluate(d, worse=False, stress=False, holdout=False):
    dead = stress
    syms, idx, P = V.market_holdout(dead) if holdout else V.market(dead)
    kw = {}
    if worse or stress:
        kw.update(pen=PEN, pens=PEN)
    if stress:
        kw.update(mtp_now=True, cd_add=1)
    if dead:
        kw["liquidate"] = True
    e, lg, inv = frun(P, idx, d, **kw)
    return V.summarize(e, lg, inv, idx, V.HOLD if holdout else V.MAIN), e, lg


def describe(d):
    parts = []
    for i, t in enumerate(d["tiers"]):
        t = dict(TIERA, **t)
        tp = t["tp"]
        tps = f"+{tp*100:.1f}%" if not isinstance(tp, tuple) else f"rng{tp[1]*100:.1f}%^{tp[2]:g}[{tp[3]*100:.0f}-{tp[4]*100:.0f}]"
        parts.append(f"{t['ref']}h-{t['depth']*100:.0f}% {tps} {t['hm']}h cd{t['cdw']}/{t['cdl']}"
                     + (f" vol>{t['vol']:.2f}" if t["vol"] > 0 else "") + (f" fs{t['fs']:g}" if t["fs"] != 1 else ""))
    r = dict(OFFRISK); r.update({k: v for k, v in d.items() if k != "tiers"})
    risk = []
    if r["nbid"] != 1 or r["nslots"] != 99:
        risk.append(f"bids{r['nbid']}" + (f"/max{r['nslots']}" if r["nslots"] != 99 else ""))
    if r["guard"] != 0.35:
        risk.append(f"guard{r['guard']:.2f}")
    if r["hs"]:
        risk.append(f"stop{r['hs']*100:.0f}%")
    if r["zcap"]:
        risk.append(f"cap{r['zcap']*100:.0f}%")
    if r["killm"]:
        risk.append(f"kill30d{r['killm']*100:.0f}%/{r['klock']}h")
    if r["cln"]:
        risk.append(f"streak{r['cln']}/{r['clT']}h")
    return " OR ".join(parts) + (" | " + " ".join(risk) if risk else "")
