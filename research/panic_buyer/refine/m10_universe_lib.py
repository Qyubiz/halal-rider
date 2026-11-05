"""m10 "universe": 90% mode on the user's 19 coins plus a monthly top-N tier of other screened halal coins.

Goal: more dips to buy (more trades) while keeping a >= 90% win rate.

Universe (point in time, no lookahead, delisted coins included):
  core = the user's 19 coins (RENDER joined to its old ticker RNDR), always eligible.
  tier = every OTHER screened coin (code/universe.py pass_map, $PB_DATA/univ) that is in the monthly top M of the
         non-19 screened coins by trailing quote volume: "v30" (30 days) or "v365" (365 days, as much as exists
         since Jan 2021). Ranks are fixed at each month's first hour from data before it (min 30 days of history,
         traded in the previous hour); they come from w90_wider_lib.ranks_all() (ranked over ALL screened coins).
  Optional tier filters, all known at the open of the hour:
    lmin  liquidity floor: trailing 30-day average daily quote volume (USDT) >= lmin.
    amin  age floor: >= amin days of trading history. The data start on 1 Jan 2021, so a coin already trading then
          has an unknown listing date; it is treated as old enough (assumption, affects early 2021 only).
    gt    crash guard for tier coins (default: same as core g).
    tcap  at most tcap tier positions at a time (-1: no cap).
  Crash guard (panic_buyer.py crash_guard_pct): skip a coin whose last close is more than g under its highest
  close of the previous 168 hours.

Mechanism: the live bot's resting bids (engine.gkernel): each hour fully funded limit bids sit at
24h-high-close x (1 - depth) on the eligible coins closest to their trigger (open / trigger - 1, smallest first);
a coin already under its trigger at the open is bought at market. Order = min(fmax x equity, cash / bids) x size.
  slots=False: N bids every hour among coins not held (live bot: n_orders).
  slots=True : "max positions" = N - open positions bids (a small code change), so with fmax = 1/N each position
               gets a full 1/N of the money and at most N are held.
Exit: limit take-profit at entry x (1 + tp) (placed at the next hourly check; mtp_now=True: at once after a market
buy, as panic_buyer.py does), else sold at the close after hm hours; cooldown cd hours per coin after a sale.
liq=1: a position still open at its coin's last data hour (delisting, ticker migration, LUNA/FTT/SRM cuts) is sold
at that hour's close with market slippage (why 4) instead of staying frozen at its last price.

Costs: 0.1% fee per side; market fills (buys at the open, time-stop sells, last-hour sales) pay slippage 0.02% for
coins in the monthly top 50 of all screened coins by 30-day volume, else 0.1%; limit buys fill only when the low
trades 0.1% through the limit (take-profits: the high 0.1% through). pen / pens make every buy / sell fill worse.

mkernel = w90_wider_lib.wkernel with per-group (core / tier) depth, take-profit and size given as scalars plus a
boolean eligibility mask (keeps memory low with ~300 coins), and the slots / tcap switches. With slots=False,
tcap=-1 it gives exactly w90_wider_lib.wkernel's result (m10_universe_check.py) and, on the 19 coins with constant
slippage and liq=0, exactly engine.gkernel's.
"""
import os
import shutil
import sys

import numpy as np
import pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "code"))
OUT = os.environ.get("PB_OUT", os.path.expanduser("~/pb_out_m10_universe"))
os.makedirs(OUT, exist_ok=True)
os.environ["PB_OUT"] = OUT                       # w90_wider_lib caches ranks_all.pkl in PB_OUT
import engine as E  # noqa: E402
import x10_universe_lib as X  # noqa: E402
import w90_wider_lib as L  # noqa: E402

T21, T24, T25 = pd.Timestamp("2021-01-01"), pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")
PERIODS = (("d", T21, T24), ("t", T24, None), ("s", T25, None))
LOGC = ["eb", "xb", "k", "why", "ret", "amt", "entry", "eq0"]   # why 0 tp, 1 time stop, 4 last data hour, 9 open
STRESS_COINS = ("LUNAUSDT", "FTTUSDT", "SRMUSDT")
MODE90 = dict(dc=0.14, tp=0.025, hm=96, cd=3, N=1, fmax=1.0, g=0.35, M=0)    # panic_config.90wr.example.json
MAXM = 100
OLD_H = 10 * 365 * 24          # age given to coins already trading on 1 Jan 2021 (listing date unknown)


# ------------------------------------------------------------------ kernel
@njit(cache=True)
def mkernel(o, h, l, c, ref, elig, core, dep_c, dep_t, tp_c, tp_t, sz_c, sz_t, N, fmax, slots, tcap, hm, cd, eps,
            fee, slip, pen, pens, tpwait, mtp_now, lastbar, liq):
    n, K = c.shape
    units = np.zeros(K); entry = np.zeros(K); amt0 = np.zeros(K); got = np.zeros(K); tpp = np.zeros(K)
    ebar = np.full(K, -1, np.int64); okbar = np.zeros(K, np.int64); mkt = np.zeros(K, np.int64)
    eq0 = np.zeros(K)
    lastc = np.full(K, np.nan); dist = np.empty(K); sel = np.empty(K, np.int64)
    cash = 1.0; eq = np.empty(n); eq_prev = 1.0
    maxlog = 600000; lg = np.zeros((maxlog, 8)); nl = 0
    for t in range(n):
        npos = 0; ntier = 0
        for k in range(K):
            if units[k] > 0.0:
                npos += 1
                if not core[k]:
                    ntier += 1
        for k in range(K):
            dist[k] = 1e9
            if units[k] > 0.0 or t < okbar[k] or np.isnan(o[t, k]) or np.isnan(ref[t, k]) or not elig[t, k]:
                continue
            d = dep_c if core[k] else dep_t
            dist[k] = o[t, k] / (ref[t, k] * (1.0 - d)) - 1.0
        order = np.argsort(dist)
        nb = N - npos if slots else N
        nsel = 0; tsel = ntier
        for q in range(K):
            k = order[q]
            if dist[k] >= 1e8 or nsel >= nb:
                break
            if (not core[k]) and tcap >= 0 and tsel >= tcap:
                continue
            sel[nsel] = k; nsel += 1
            if not core[k]:
                tsel += 1
        if nsel > 0 and cash > 1e-9:
            per = min(fmax * eq_prev, cash / nsel)
            for q in range(nsel):
                k = sel[q]
                d = dep_c if core[k] else dep_t
                lim = ref[t, k] * (1.0 - d)
                fill = -1.0; ismkt = 0
                if o[t, k] <= lim:
                    fill = o[t, k] * (1.0 + slip[t, k]) * (1.0 + pen); ismkt = 1
                elif l[t, k] <= lim * (1.0 - eps):
                    fill = lim * (1.0 + pen)
                if fill > 0.0 and cash > 1e-9:
                    mkt[k] = ismkt if mtp_now else 0
                    a = min(per * (sz_c if core[k] else sz_t), cash)
                    cash -= a; units[k] = a * (1.0 - fee) / fill
                    entry[k] = fill; amt0[k] = a; ebar[k] = t; got[k] = 0.0; eq0[k] = eq_prev
                    tpp[k] = fill * (1.0 + (tp_c if core[k] else tp_t))
        for k in range(K):
            if units[k] > 0.0 and (ebar[k] < t or mkt[k] == 1) and not np.isnan(o[t, k]):
                px = -1.0; why = 0.0
                if ebar[k] == t:
                    if h[t, k] >= tpp[k] * (1.0 + eps):
                        px = tpp[k]
                elif t - ebar[k] < tpwait:
                    pass
                elif o[t, k] >= tpp[k]:
                    px = o[t, k]
                elif h[t, k] >= tpp[k] * (1.0 + eps):
                    px = tpp[k]
                if px < 0.0 and t - ebar[k] >= hm - mkt[k]:
                    px = c[t, k] * (1.0 - slip[t, k]); why = 1.0
                if px < 0.0 and liq == 1 and t == lastbar[k]:
                    px = c[t, k] * (1.0 - slip[t, k]); why = 4.0
                if px > 0.0:
                    proceeds = units[k] * px * (1.0 - pens) * (1.0 - fee); cash += proceeds; got[k] += proceeds
                    if nl < maxlog:
                        lg[nl, 0] = ebar[k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = why
                        lg[nl, 4] = got[k] / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]
                        lg[nl, 7] = eq0[k]; nl += 1
                    units[k] = 0.0; okbar[k] = t + cd
        v = cash
        for k in range(K):
            if not np.isnan(c[t, k]):
                lastc[k] = c[t, k]
            if units[k] > 0.0:
                v += units[k] * lastc[k]
        eq[t] = v; eq_prev = v
    for k in range(K):                       # still open at the end: marked at the last close, logged why 9
        if units[k] > 0.0 and nl < maxlog:
            lg[nl, 0] = ebar[k]; lg[nl, 1] = -1.0; lg[nl, 2] = k; lg[nl, 3] = 9.0
            lg[nl, 4] = (got[k] + units[k] * lastc[k]) / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]
            lg[nl, 7] = eq0[k]; nl += 1
    return eq, lg[:nl].copy()


# ------------------------------------------------------------------ panel
def _ranks():
    """w90_wider_lib.ranks_all() (cached in OUT; copied from the w90 cache when present: same code, same data)."""
    mine = os.path.join(OUT, "ranks_all.pkl")
    if not os.path.exists(mine):
        src = os.path.join(os.path.dirname(os.path.dirname(OUT)), "pb_out_w90", "wider", "ranks_all.pkl")
        if os.path.exists(src):
            shutil.copy(src, mine)
    L.OUT = OUT
    return L.ranks_all()


_W = {}


def build():
    """Panel of the 19 coins, every other screened coin ever in the monthly top MAXM non-19 coins by 30- or
    365-day volume, LUNA/SRM and FTT (FTT: stress runs only). float64 prices (o, h, l, c = forward-filled close,
    ref = highest close of the previous 24 hours), float32 adv (trailing 30-day average daily quote volume), int32
    age (hours of history before the hour), float32 ranks, float64 slippage. Cached as plain .npy files (no pickle),
    memory-mapped read-only when loaded."""
    if "W" in _W:
        return _W["W"]
    cp = os.path.join(OUT, "m10_panel.npz")
    dp = os.path.join(OUT, "m10_panel")
    if not os.path.exists(os.path.join(dp, "done")) and not os.path.exists(cp):
        R = _ranks()
        idx = R["idx"]; s_all = list(R["syms"])
        ever = ((np.nan_to_num(R["r30o"], nan=1e9).min(0) <= MAXM) |
                (np.nan_to_num(R["r365o"], nan=1e9).min(0) <= MAXM))
        keep = [s for j, s in enumerate(s_all) if ever[j] or s in E.H19 or s in ("LUNAUSDT", "SRMUSDT")]
        keep = sorted(set(keep)) + ["FTTUSDT"]
        n, K = len(idx), len(keep)
        A = {f: np.full((n, K), np.nan) for f in ("o", "h", "l", "c", "ref")}
        adv = np.zeros((n, K), np.float32); age = np.zeros((n, K), np.int32)
        for i0 in range(0, K, 40):
            chunk = keep[i0:i0 + 40]
            D = X.load_univ("FTTUSDT" in chunk, only=set(chunk))
            for s in chunk:
                j = keep.index(s)
                d = D[s].reindex(idx)
                for f in ("o", "h", "l"):
                    A[f][:, j] = d[f].values
                cs = d["c"]
                A["c"][:, j] = cs.ffill().values
                A["ref"][:, j] = cs.rolling(24, min_periods=1).max().shift(1).values
                adv[:, j] = (d["qv"].fillna(0.0).rolling(720, min_periods=1).sum().shift(1).fillna(0.0) / 30.0).values
                ag = cs.notna().cumsum().shift(1).fillna(0).values
                if cs.first_valid_index() == idx[0]:
                    ag = ag + OLD_H
                age[:, j] = np.minimum(ag, 2 ** 31 - 1).astype(np.int32)
            del D
        col = [s_all.index(s) if s in s_all else -1 for s in keep]

        def pick(M_, fill):
            out = np.full((n, K), fill, np.float32)
            for j, cj in enumerate(col):
                if cj >= 0:
                    out[:, j] = M_[:, cj]
            return out
        srank = pick(R["srank"], np.nan)
        slip = np.where(srank <= 50, E.SLIP_MAJOR, E.SLIP_OTHER).astype(np.float32)
        np.savez(cp, t=idx.values.astype("datetime64[s]").astype(np.int64), syms=np.array(keep), o=A["o"], h=A["h"],
                 l=A["l"], c=A["c"], ref=A["ref"], adv=adv, age=age, r30o=pick(R["r30o"], np.nan),
                 r365o=pick(R["r365o"], np.nan), slip=slip)
        del A, R
    F = ("o", "h", "l", "c", "ref", "adv", "age", "r30o", "r365o", "slip")
    if not os.path.exists(os.path.join(dp, "done")):         # one .npy per array, memory-mapped when loaded
        os.makedirs(dp, exist_ok=True)
        z = np.load(cp, allow_pickle=False)
        for f in F + ("t", "syms"):
            a = z[f]
            if f == "slip":
                a = np.where(a < 0.0005, E.SLIP_MAJOR, E.SLIP_OTHER)       # exact float64 values
            np.save(os.path.join(dp, f + ".npy"), a)
        open(os.path.join(dp, "done"), "w").close()
        os.remove(cp)
    ld = {f: np.asarray(np.load(os.path.join(dp, f + ".npy"), mmap_mode="r")) for f in F}
    syms = [str(s) for s in np.load(os.path.join(dp, "syms.npy"))]
    W = dict(idx=pd.to_datetime(np.load(os.path.join(dp, "t.npy")), unit="s"), syms=syms, **ld)
    W["is19"] = np.array([s in E.H19 for s in syms])
    O = W["o"]; ok = ~np.isnan(O)
    lb = np.array([np.nonzero(ok[:, k])[0][-1] if ok[:, k].any() else -1 for k in range(len(syms))], np.int64)
    lb[lb >= len(W["idx"]) - 3] = len(W["idx"]) + 1          # trades to the end: never liquidated
    W["lastbar"] = lb
    _W["W"] = W
    return W


_gm = {}


def guard(W, g, L_=168):
    """(n, K) bool, True = allowed: last close no more than g under the highest close of the previous L_ hours
    (known at the open). Same rule as w90_wider_lib.guard_mask / panic_buyer.py crash_guard_pct."""
    if g <= 0:
        return None
    key = (g, L_)
    if key not in _gm:
        c = W["c"]
        out = np.ones(c.shape, bool)
        for j0 in range(0, c.shape[1], 50):
            cf = pd.DataFrame(c[:, j0:j0 + 50])
            hi = cf.rolling(L_, min_periods=1).max()
            out[:, j0:j0 + 50] = ((cf / hi - 1) >= -g).shift(1, fill_value=True).values
        _gm[key] = out
    return _gm[key]


def masks(W, d, stress=False):
    """Design dict -> (elig (n, K) bool, core (K,) bool). Keys: M tier size (0 = none), rk 'v30'/'v365', lmin, amin
    (days), g core guard, gt tier guard, excl (coins never bought), xlast (days: no buys in a coin's last xlast days of
    data, a lookahead stand-in for reading Binance's delisting notices; diagnostic only)."""
    is19 = W["is19"]
    syms = W["syms"]
    core = is19.copy()
    if stress:
        for s in STRESS_COINS:
            core[syms.index(s)] = True
    n, K = W["o"].shape
    el = np.zeros((n, K), bool)
    el[:, core] = True
    M = d.get("M", 0)
    if M > 0:
        R = W["r30o"] if d.get("rk", "v30") == "v30" else W["r365o"]
        tier = (R <= M)                      # NaN compares False
        tier[:, core] = False
        if d.get("lmin", 0) > 0:
            tier &= W["adv"] >= d["lmin"]
        if d.get("amin", 0) > 0:
            tier &= W["age"] >= int(d["amin"] * 24)
        el |= tier
    if not stress:
        el[:, syms.index("FTTUSDT")] = False                 # FTT: not screened
    gc = guard(W, d.get("g", 0.35))
    gt = guard(W, d.get("gt", d.get("g", 0.35)))
    if gc is not None or gt is not None:
        if gc is gt:
            el &= gc
        else:
            if gc is not None:
                el[:, core] &= gc[:, core]
            if gt is not None:
                el[:, ~core] &= gt[:, ~core]
    for s in d.get("excl", ()):
        el[:, syms.index(s)] = False
    if d.get("xlast", 0) > 0:
        for k, lb in enumerate(W["lastbar"]):
            if lb <= n:
                el[max(0, lb - int(d["xlast"] * 24)):, k] = False
    return el, core


def run_design(W, d, stress=False, start=None, pen=0.0, pens=0.0, mtp_now=False, eps=E.EPS, slip19=None, liq=1,
               block=None):
    """Run design d. stress=True: real bot timing (mtp_now, cooldown +1h), every buy and sell fill 0.2% worse,
    LUNA/FTT/SRM forced in as core coins. pen/pens: fills worse without the rest. slip19: constant slippage for the
    19 coins (reproduces the 19-coin engine runs). start: fresh account from this time. block: [(a, b)] no buys.
    All coins are passed to the kernel (ineligible ones are skipped by the mask), so nothing is copied."""
    el, core = masks(W, d, stress)
    if block:
        for a, b in block:
            el[(W["idx"] >= pd.Timestamp(a)) & (W["idx"] < pd.Timestamp(b))] = False
    i0 = 0 if start is None else int(np.searchsorted(W["idx"], pd.Timestamp(start)))
    sl = W["slip"][i0:]
    if slip19 is not None:
        sl = np.array(sl)
        sl[:, W["is19"]] = slip19
    cd = d.get("cd", 3); mt = mtp_now
    if stress:
        cd += 1; mt = True; pen = max(pen, 0.002); pens = max(pens, 0.002)
    N = d.get("N", 1)
    lb = W["lastbar"] - i0
    eq, lg = mkernel(W["o"][i0:], W["h"][i0:], W["l"][i0:], W["c"][i0:], W["ref"][i0:], el[i0:], core,
                     float(d["dc"]), float(d.get("dt", d["dc"])), float(d["tp"]), float(d.get("tt", d["tp"])), 1.0,
                     float(d.get("st", 1.0)), int(N), float(d.get("fmax", 1.0 / N)), bool(d.get("slots", True)),
                     int(d.get("tcap", -1)), int(d.get("hm", 96)), int(cd), float(eps), E.FEE, sl, float(pen),
                     float(pens), 1, bool(mt), lb.astype(np.int64), int(liq))
    return pd.Series(eq, W["idx"][i0:]), pd.DataFrame(lg, columns=LOGC), W["syms"], W["is19"]


# ------------------------------------------------------------------ metrics
def summarize(e, lg, idx, is19):
    """Per period d=2021-23, t=2024-26, s=since 2025 (trades = bought in the period, closed incl. last-hour sales;
    positions open at the end of the data are left out): multiple, %/month, worst drop, trades, trades/month, win
    rate, avg win/loss, worst trade as % of the account, account-level P&L points (ret x order / account at entry,
    summed) in total and from coins outside the 19, losses outside the 19."""
    lg = lg[lg.why != 9]
    r = {}
    eb = idx[lg.eb.astype(int).values] if len(lg) else pd.DatetimeIndex([])
    ret = lg.ret.values
    acct = (lg.ret * lg.amt / lg.eq0).values
    out19 = ~is19[lg.k.astype(int).values] if len(lg) else np.zeros(0, bool)
    for tag, a, b in PERIODS:
        r[f"{tag}_mult"], r[f"{tag}_mo"], r[f"{tag}_dd"] = L.period(e, a, b)
        ia = int(np.searchsorted(e.index, a)); ib = len(e) if b is None else int(np.searchsorted(e.index, b))
        months = (ib - ia) / (365.25 / 12 * 24)
        m = np.asarray((eb >= a) & ((eb < b) if b is not None else True), bool)
        r[f"{tag}_n"] = int(m.sum())
        r[f"{tag}_tpm"] = m.sum() / months
        r[f"{tag}_win"] = float((ret[m] > 0).mean()) if m.any() else np.nan
        r[f"{tag}_avgw"] = float(ret[m & (ret > 0)].mean()) if (m & (ret > 0)).any() else np.nan
        r[f"{tag}_avgl"] = float(ret[m & (ret <= 0)].mean()) if (m & (ret <= 0)).any() else np.nan
        r[f"{tag}_wacct"] = float(acct[m].min()) if m.any() else np.nan
        r[f"{tag}_pnl"] = float(acct[m].sum()) if m.any() else 0.0
        r[f"{tag}_pnl_out"] = float(acct[m & out19].sum()) if m.any() else 0.0
        r[f"{tag}_n_out"] = int((m & out19).sum())
        r[f"{tag}_loss"] = int((m & (ret <= 0)).sum())
        r[f"{tag}_loss_out"] = int((m & out19 & (ret <= 0)).sum())
        r[f"{tag}_lossp_out"] = float(acct[m & out19 & (ret <= 0)].sum()) if m.any() else 0.0
        r[f"{tag}_liq"] = int((m & (lg.why.values == 4)).sum())
    r["y2022"] = L.year_ret(e, 2022)
    return r


def evaluate(d, W=None, stress=False, **kw):
    W = build() if W is None else W
    e, lg, syms, i19 = run_design(W, d, stress, **kw)
    return summarize(e, lg, e.index, i19), e, lg, syms


def name(d):
    if d.get("M", 0) == 0:
        t = "19 only"
    else:
        f = []
        if d.get("lmin", 0) > 0:
            f.append(f"vol>={d['lmin'] / 1e6:g}M/day")
        if d.get("amin", 0) > 0:
            f.append(f"age>={d['amin']}d")
        if d.get("tcap", -1) >= 0:
            f.append(f"tiercap{d['tcap']}")
        if "gt" in d:
            f.append(f"tierguard{d['gt'] * 100:.0f}")
        t = (f"19 + top{d['M']} {d.get('rk', 'v30')} @{d.get('dt', d['dc']) * 100:.0f}% x{d.get('st', 1.0):g} "
             f"tp{d.get('tt', d['tp']) * 100:g}%" + (" " + " ".join(f) if f else ""))
    return (f"{t} | core {d['dc'] * 100:.0f}% tp{d['tp'] * 100:g}% N{d.get('N', 1)}"
            f"{'' if d.get('slots', True) else 'bids'} hm{d.get('hm', 96)} cd{d.get('cd', 3)} g{d.get('g', 0.35) * 100:.0f}")


def fmt(r, pre=""):
    def p(t):
        return (f"WR {r[pre + t + '_win'] * 100:4.1f}% {r[pre + t + '_mo'] * 100:+6.2f}%/mo dd {r[pre + t + '_dd'] * 100:4.0f}% "
                f"{r[pre + t + '_tpm']:4.1f}tr/mo wt {r[pre + t + '_wacct'] * 100:+4.0f}%")
    return f"21-23 {p('d')} | 24-26 {p('t')} | 25-26 {p('s')}"


def by_coin(lg, idx, syms, a, b=None):
    lg = lg[lg.why != 9]
    eb = idx[lg.eb.astype(int).values]
    m = np.asarray((eb >= a) & ((eb < b) if b is not None else True), bool)
    t = lg[m]
    acct = t.ret * t.amt / t.eq0
    df = pd.DataFrame({"coin": [syms[int(k)] for k in t.k], "pnl": acct.values, "win": (t.ret > 0).values})
    return df.groupby("coin").agg(pnl=("pnl", "sum"), n=("pnl", "size"), win=("win", "mean")).sort_values("pnl")


# ------------------------------------------------------------------ portable runner (2018-20 holdout: 19 coins)
def run_panel(P, idx, d, pen=0.0, pens=0.0, mtp_now=False, extra_core=()):
    """Run the CORE of design d on any engine.panel P (e.g. the 2018-20 holdout, which has only the 19 coins):
    every coin in P is a core coin; crash guard g; slippage 0.02% (19 coins) / 0.1% (others)."""
    syms = list(P["o"].columns)
    n, K = P["o"].shape
    g = d.get("g", 0.35)
    el = L.guard_mask(P, g).astype(bool) if g > 0 else np.ones((n, K), bool)
    sl = np.array([E.SLIP_MAJOR if s in E.H19 else E.SLIP_OTHER for s in syms])
    sl = np.repeat(sl[None, :], n, axis=0)
    N = d.get("N", 1)
    eq, lg = mkernel(P["o"].values, P["h"].values, P["l"].values, P["cf"].values, P["ref"].values,
                     np.ascontiguousarray(el), np.ones(K, np.bool_), float(d["dc"]), float(d["dc"]), float(d["tp"]),
                     float(d["tp"]), 1.0, 1.0, int(N), float(d.get("fmax", 1.0 / N)), bool(d.get("slots", True)), -1,
                     int(d.get("hm", 96)), int(d.get("cd", 3)), E.EPS, E.FEE, np.ascontiguousarray(sl), float(pen),
                     float(pens), 1, bool(mtp_now), np.full(K, n + 1, np.int64), 0)
    return pd.Series(eq, idx), pd.DataFrame(lg, columns=LOGC)
