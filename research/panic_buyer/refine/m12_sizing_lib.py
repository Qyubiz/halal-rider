"""m12 "sizing" (agent A): 90% mode's entry and exit unchanged, only the bet size of each trade changes.

sk = m11_fusion_lib.fk with one tier (24h highest hourly close x 0.86, +2.5% take-profit, 96h time stop, 3h cooldown,
crash guard 35% over 168h closes) and the risk switches removed, plus a SIZE MULTIPLIER m computed at each bid from
information known before that hour (results of trades closed in earlier hours, hourly equity marks of earlier hours,
features from closed candles). Order money = min(min(1, fs x m) x equity_prev, free cash / bids this hour).
Unused cash stays in USDT (earns nothing). Switches (all off = 90% mode exactly, m12_sizing_check.py):
  fs        base fraction of the account per trade (1.0 = 90% mode)
  nslots    most positions held at once (1 = 90% mode; 2 = a second coin may use the cash a smaller bet leaves)
  nbid      most new bids per hour (1)
  streak    ln, lmul, lmin, wstep: after ln losing trades in a row, level = max(lmin, level x lmul) at each further
            loss; each win: level = min(1, level + wstep). m *= level.   (ln = 0: off)
  drawdown  d0, dk, dmin, pw: dd = 1 - equity_prev / peak (peak = highest hourly mark since start, or of the last pw
            hours if pw > 0); m *= max(dmin, 1 - dk x (dd - d0)) when dd > d0.   (dk = 0: off)
  volatility vt, vsrc, vmin: vol = feature vsrc (std of hourly log returns of the forward-filled close over the last
            24h / 168h of closed candles: coin "c24"/"c168", or median over coins "m24"/"m168");
            m *= clip(vt / vol, vmin, 1).   (vt = 0: off)
  guard depth g1, gf: gd = the coin's last close / its highest close of the previous 168h - 1 (the crash-guard
            measure, known at the open); m *= gf when gd < -g1.   (g1 = 0: off)
  kelly     kN, kc, kmin: over the last kN closed trades, p = win share, aW = mean win, aL = mean |loss|;
            f* = p / aL - (1 - p) / aW (binary Kelly fraction); m *= clip(kc x f*, kmin, 1); no losses -> 1.
            (kN = 0: off; needs kN trades first, m = 1 before)
  deep fill dx, dm: a market buy whose open is at least dx under the limit (deeper fill) gets m *= dm (may be > 1 when
            fs < 1; still capped at 100% of the account).   (dx = 0: off)
Costs/timing exactly as fk: fee 0.1%/side, 0.02% slippage on market buys and time-stop sells (0.1% for LUNA/FTT/SRM
when liquidate), limit fills need 0.1% trade-through, pen/pens worse fills, mtp_now bot timing, cd_add, t0 start,
cash from a sale usable at the next hourly check. Spot only, fully funded.
"""
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("PB_OUT", os.path.expanduser("~/pb_out_m12/sizing"))
OUT = os.environ["PB_OUT"]
os.makedirs(OUT, exist_ok=True)
import engine as E  # noqa: E402
import m10_velocity_lib as V  # noqa: E402

T21, T24, T25 = V.T21, V.T24, V.T25
LATE = pd.Timestamp("2021-01-15")
PEN = 0.002
OFF = dict(fs=1.0, nslots=1, nbid=1, ln=0, lmul=0.5, lmin=0.25, wstep=1.0, d0=0.0, dk=0.0, dmin=0.25, pw=0,
           vt=0.0, vsrc="c168", vmin=0.25, g1=0.0, gf=1.0, kN=0, kc=0.2, kmin=0.25, dx=0.0, dm=1.0)
VSRC = {"c24": 0, "c168": 1, "m24": 2, "m168": 3}


@njit(cache=True)
def sk(o, h, l, c, gm, ref, lastb, depth, tpa, hm, cdw, cdl, nslots, nbid, eps, fee, slip, pen, pens, mtp_now, t0,
       fs, ln, lmul, lmin, wstep, d0, dk, dmin, pw, vt, vol, vmin, g1, gd, gf, kN, kc, kmin, dx, dm):
    n, K = c.shape
    units = np.zeros(K); entry = np.zeros(K); amt0 = np.zeros(K); tpp = np.zeros(K)
    eq0 = np.zeros(K); ebar = np.full(K, -1, np.int64); mkt = np.zeros(K, np.int64); szm = np.zeros(K)
    okc = np.zeros(K, np.int64)
    lastc = np.full(K, np.nan)
    cash = 1.0; eq = np.ones(n); inv = np.zeros(n); eq_prev = 1.0; peak = 1.0
    maxlog = 200000; lg = np.zeros((maxlog, 10)); nl = 0
    cdist = np.empty(K); ck = np.empty(K, np.int64); clim = np.empty(K)
    bk = np.empty(K, np.int64); blim = np.empty(K)
    lvl = 1.0; lstreak = 0
    nr = 0; ring = np.zeros(max(kN, 1))
    for t in range(t0):
        for k in range(K):
            if not np.isnan(c[t, k]):
                lastc[k] = c[t, k]
    for t in range(t0, n):
        nheld = 0
        for k in range(K):
            if units[k] > 0.0:
                nheld += 1
        m = 0
        for k in range(K):
            if units[k] > 0.0 or t < okc[k] or np.isnan(o[t, k]) or gm[t, k] <= 0.0:
                continue
            v = ref[t, k]
            if np.isnan(v):
                continue
            lim = v * (1.0 - depth)
            cdist[m] = o[t, k] / lim - 1.0; ck[m] = k; clim[m] = lim; m += 1
        nb = 0
        if m > 0 and cash > 1e-9:
            order = np.argsort(cdist[:m])
            free = min(nslots - nheld, nbid)
            for q in range(m):
                if nb >= free:
                    break
                j = order[q]
                bk[nb] = ck[j]; blim[nb] = clim[j]; nb += 1
        if nb > 0:
            # ---- account-level size multiplier (information from hours < t only)
            mult = 1.0
            if ln > 0:
                mult *= lvl
            if dk > 0.0:
                pk = peak
                if pw > 0:
                    pk = eq_prev
                    for j in range(max(t - pw, t0), t):
                        if eq[j] > pk:
                            pk = eq[j]
                dd = 1.0 - eq_prev / pk
                if dd > d0:
                    mult *= max(dmin, 1.0 - dk * (dd - d0))
            if kN > 0 and nr >= kN:
                nw = 0; sw = 0.0; sl_ = 0.0
                for j in range(kN):
                    if ring[j] > 0.0:
                        nw += 1; sw += ring[j]
                    else:
                        sl_ -= ring[j]
                if nw < kN and nw > 0:
                    p = nw / kN
                    aW = sw / nw; aL = sl_ / (kN - nw)
                    if aL > 0.0:
                        f = kc * (p / aL - (1.0 - p) / aW)
                        mult *= min(1.0, max(kmin, f))
                elif nw == 0:
                    mult *= kmin
            cash0 = cash
            for q in range(nb):
                k = bk[q]; lim = blim[q]
                mk = mult
                if vt > 0.0:
                    vv = vol[t, k]
                    if vv > 0.0 and not np.isnan(vv):
                        mk *= min(1.0, max(vmin, vt / vv))
                if g1 > 0.0 and gd[t, k] < -g1:
                    mk *= gf
                fill = -1.0; ismkt = 0
                if o[t, k] <= lim:
                    fill = o[t, k] * (1.0 + slip[k]) * (1.0 + pen); ismkt = 1
                    if dx > 0.0 and o[t, k] <= lim * (1.0 - dx):
                        mk *= dm
                elif l[t, k] <= lim * (1.0 - eps):
                    fill = lim * (1.0 + pen)
                frac = min(1.0, fs * mk)
                per = min(frac * eq_prev, cash0 / nb)
                if fill > 0.0 and cash > 1e-9:
                    mkt[k] = ismkt if mtp_now else 0
                    a = min(per, cash)
                    cash -= a; units[k] = a * (1.0 - fee) / fill
                    entry[k] = fill; amt0[k] = a; ebar[k] = t; eq0[k] = eq_prev; szm[k] = frac
                    tpp[k] = fill * (1.0 + tpa)
        # ---------------- exits
        for k in range(K):
            px = -1.0; why = 0.0
            if units[k] > 0.0 and not np.isnan(o[t, k]):
                if ebar[k] < t or mkt[k] == 1:
                    if ebar[k] == t:
                        if h[t, k] >= tpp[k] * (1.0 + eps):
                            px = tpp[k]
                    elif o[t, k] >= tpp[k]:
                        px = o[t, k]
                    elif h[t, k] >= tpp[k] * (1.0 + eps):
                        px = tpp[k]
                    age = t - ebar[k] + mkt[k]
                    if px < 0.0 and age >= hm:
                        px = c[t, k] * (1.0 - slip[k]); why = 1.0
            if px < 0.0 and units[k] > 0.0 and lastb[k] == t:
                px = c[t, k] * (1.0 - slip[k]); why = 5.0
            if px > 0.0:
                proceeds = units[k] * px * (1.0 - pens) * (1.0 - fee); cash += proceeds
                ret = proceeds / amt0[k] - 1.0
                if nl < maxlog:
                    lg[nl, 0] = ebar[k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = why
                    lg[nl, 4] = ret; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]
                    lg[nl, 7] = eq0[k]; lg[nl, 8] = 0; lg[nl, 9] = szm[k]; nl += 1
                units[k] = 0.0
                win = proceeds > amt0[k]
                okc[k] = t + (cdw if (win and why != 5.0) else cdl)
                if win:
                    lstreak = 0
                    lvl = min(1.0, lvl + wstep)
                else:
                    lstreak += 1
                    if ln > 0 and lstreak >= ln:
                        lvl = max(lmin, lvl * lmul)
                if kN > 0:
                    ring[nr % kN] = ret; nr += 1
        # ---------------- hourly mark
        for k in range(K):
            if not np.isnan(c[t, k]):
                lastc[k] = c[t, k]
        v = cash; w = 0.0
        for k in range(K):
            if units[k] > 0.0:
                v += units[k] * lastc[k]; w += units[k] * lastc[k]
        eq[t] = v; eq_prev = v; inv[t] = w / v if v > 0.0 else 0.0
        if v > peak:
            peak = v
    for k in range(K):
        if units[k] > 0.0 and nl < maxlog:
            proceeds = units[k] * lastc[k] * (1.0 - slip[k]) * (1.0 - pens) * (1.0 - fee)
            lg[nl, 0] = ebar[k]; lg[nl, 1] = n - 1; lg[nl, 2] = k; lg[nl, 3] = 9.0
            lg[nl, 4] = proceeds / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]
            lg[nl, 7] = eq0[k]; lg[nl, 8] = 0; lg[nl, 9] = szm[k]; nl += 1
    return eq, lg[:nl].copy(), inv


LOGC = V.LOGC + ["size"]
_cache = {}


def feats(P):
    """Features known at each hour's open (closed candles only): vol panels and crash-guard depth."""
    key = id(P["o"])
    if key not in _cache:
        cf = P["cf"]
        r = np.log(cf).diff()
        c24 = r.rolling(24, min_periods=12).std().shift(1)
        c168 = r.rolling(168, min_periods=48).std().shift(1)
        m24 = c24.median(axis=1); m168 = c168.median(axis=1)
        K = cf.shape[1]
        vols = np.stack([c24.values, c168.values, np.repeat(m24.values[:, None], K, 1),
                         np.repeat(m168.values[:, None], K, 1)])
        hi = cf.rolling(168, min_periods=1).max()
        gd = (cf / hi - 1).shift(1).fillna(0.0).values
        _cache[key] = (np.ascontiguousarray(np.nan_to_num(vols, nan=0.0)), np.ascontiguousarray(gd),
                       np.ascontiguousarray(V.guard_mask(P, 0.35)), np.ascontiguousarray(P["ref"].values),
                       V.last_bars(P))
    return _cache[key]


def srun(P, idx, d, pen=0.0, pens=0.0, mtp_now=False, cd_add=0, liquidate=False, t0=None, n_end=None):
    """Run design d (dict of OFF keys) on panel P. n_end: stop after that many hours (selection runs on 2021-23)."""
    r = dict(OFF); r.update(d)
    vols, gd, gm, ref, lb = feats(P)
    O = P["o"].values
    n, K = O.shape
    ne = n if n_end is None else n_end
    sl = np.array([E.SLIP_MAJOR if s in E.H19 else E.SLIP_OTHER for s in P["o"].columns]) if liquidate \
        else np.full(K, E.SLIP_MAJOR)
    lastb = lb if liquidate else np.full(K, -1, np.int64)
    i0 = 0 if t0 is None else int(np.searchsorted(idx.values, np.datetime64(t0)))
    a = lambda x: np.ascontiguousarray(x[:ne])   # noqa: E731
    eq, lg, inv = sk(a(O), a(P["h"].values), a(P["l"].values), a(P["cf"].values), a(gm), a(ref), lastb, 0.14, 0.025,
                     96, max(3 + cd_add, 0), max(3 + cd_add, 0), int(r["nslots"]), int(r["nbid"]), float(E.EPS),
                     float(E.FEE), sl, float(pen), float(pens), bool(mtp_now), i0, float(r["fs"]), int(r["ln"]),
                     float(r["lmul"]), float(r["lmin"]), float(r["wstep"]), float(r["d0"]), float(r["dk"]),
                     float(r["dmin"]), int(r["pw"]), float(r["vt"]), np.ascontiguousarray(vols[VSRC[r["vsrc"]]][:ne]),
                     float(r["vmin"]), float(r["g1"]), a(gd), float(r["gf"]), int(r["kN"]), float(r["kc"]),
                     float(r["kmin"]), float(r["dx"]), float(r["dm"]))
    ix = idx[:ne]
    return pd.Series(eq, ix), pd.DataFrame(lg, columns=LOGC), pd.Series(inv, ix)


DPER = (("d", T21, T24),)


def design_scores(d):
    """2021-23 ONLY (data cut at 2024-01-01 before the run): normal, worse fills and the nearby versions:
    cooldown 2h, cooldown 4h, bot timing (take-profit at once after a market buy + cooldown +1h), start 2021-01-15."""
    syms, idx, P = V.market(False)
    ne = int(np.searchsorted(idx.values, np.datetime64(T24)))
    out = {}
    runs = {"n": {}, "w": dict(pen=PEN, pens=PEN), "cm": dict(cd_add=-1), "cp": dict(cd_add=1),
            "bt": dict(mtp_now=True, cd_add=1), "ls": dict(t0=LATE)}
    for tag, kw in runs.items():
        e, lg, inv = srun(P, idx, d, n_end=ne, **kw)
        per = (("d", LATE, T24),) if tag == "ls" else DPER
        r = V.summarize(e, lg, inv, idx[:ne], per)
        out[f"{tag}_wr"], out[f"{tag}_mo"], out[f"{tag}_dd"] = r["d_wr"], r["d_mo"], r["d_dd"]
        if tag == "n":
            out["n_tpm"], out["n_wt"] = r["d_tpm"], r["d_wt"]
            out["n_size"] = float(lg.loc[lg.eb < ne, "size"].mean()) if len(lg) else np.nan
    vs = [out[f"{t}_mo"] for t in ("n", "cm", "cp", "bt", "ls")]
    out["robust"] = float(np.mean(vs)); out["rmin"] = float(np.min(vs))
    return out


def evaluate(d, worse=False, stress=False, holdout=False):
    """Full-period look (final script only)."""
    dead = stress
    syms, idx, P = V.market_holdout(dead) if holdout else V.market(dead)
    kw = {}
    if worse or stress:
        kw.update(pen=PEN, pens=PEN)
    if stress:
        kw.update(mtp_now=True, cd_add=1)
    if dead:
        kw["liquidate"] = True
    e, lg, inv = srun(P, idx, d, **kw)
    return V.summarize(e, lg, inv, idx, V.HOLD if holdout else V.MAIN), e, lg


def describe(d):
    r = dict(OFF); r.update(d)
    p = []
    if r["fs"] != 1 or r["nslots"] != 1 or r["nbid"] != 1:
        p.append(f"fs{r['fs']:g} slots{r['nslots']} bids{r['nbid']}")
    if r["ln"]:
        p.append(f"streak{r['ln']} x{r['lmul']:g} min{r['lmin']:g} +{r['wstep']:g}/win")
    if r["dk"]:
        p.append(f"dd>{r['d0']:g} k{r['dk']:g} min{r['dmin']:g}" + (f" pw{r['pw']}" if r["pw"] else ""))
    if r["vt"]:
        p.append(f"vol {r['vsrc']} vt{r['vt']:g} min{r['vmin']:g}")
    if r["g1"]:
        p.append(f"guard<-{r['g1']:g} x{r['gf']:g}")
    if r["kN"]:
        p.append(f"kelly{r['kN']} c{r['kc']:g} min{r['kmin']:g}")
    if r["dx"]:
        p.append(f"deep{r['dx']:g} x{r['dm']:g}")
    return " | ".join(p) or "90% mode"
