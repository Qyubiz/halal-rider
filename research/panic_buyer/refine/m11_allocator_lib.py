"""m11 allocator: ONE spot account whose money a walk-forward meta-allocator splits across the candidate strategies
(sleeves) using only their past results.

Inputs: the standalone unit curves of m11_allocator_sleeves.py (each strategy run alone on 100% of a unit account,
with its own m10 engine, costs and fill rules).

How the one account works (exact, no approximation):
  * The account = a CASH pool + one sub-balance per sleeve. Each sleeve trades only its own sub-balance, with its
    own unchanged rules (so every order is fully funded and the sum of all orders never exceeds the account's cash).
  * At each rebalance time (first hour of the month by default; or weekly, Monday 00:00) the allocator sets new
    weights w_s for every sleeve, from trailing results known before that hour; a sleeve's target is w_s x the
    account equity at the last close before the hour in which its balance is adjusted.
  * A sleeve's sub-balance is changed ONLY at the start of an hour in which it holds no position (it is all cash
    then): a sleeve above target releases the excess to the pool; a sleeve below target takes what the pool has (it
    stays pending, and tops up at its next flat hour, until the next rebalance). Open positions are never sold for
    a rebalance and nothing is bought for one, so a rebalance has no trading cost.
  * Because every sleeve sizes its orders as fractions of its own balance, a sleeve whose all-cash balance is
    multiplied by lambda behaves exactly as lambda x its standalone run from that hour on (same decisions). The
    combined account is therefore pool + sum lambda_s(t) x U_s(t), and a trade of sleeve s entered in hour eb is
    taken with money lambda_s(eb) x amt (taken only if lambda_s(eb) > 0). Sleeves with no money keep running on paper
    (the live bot would do the same, to know their trailing results and cooldown state).
  * 100% weight on one sleeve reproduces that sleeve's standalone run exactly.

Allocator inputs at a rebalance time tau: each sleeve's stitched standalone history (2018-20 holdout run joined to
the 2021-26 run of the same scenario) strictly before tau: trailing L-month return, daily volatility, worst drop,
daily mean/variance (Kelly), win rate of trades closed in the window, and its last 5 closed trades. With less than
L months of history (start of the holdout run) the account is 100% 90% mode.
"""
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.environ.get("PB_OUT", os.path.expanduser("~/pb_out_m11/allocator"))
SDIR = os.path.join(OUT, "sleeves")
NAMES = ["m90", "m10", "norm", "vel", "reg", "mr", "scalp", "trend"]
UNIV = {"A": NAMES, "B": ["m90", "vel", "reg", "mr", "scalp"], "C": ["m90", "vel", "reg"]}
HIST = {"m": "h", "mw": "hw", "ms": "hw", "h": "h", "hw": "hw"}    # holdout scenario used as earlier history
MD = 365.25 / 12
T18, T21, T24, T25 = (pd.Timestamp(x) for x in ("2018-01-01", "2021-01-01", "2024-01-01", "2025-01-01"))
PERIODS = {"2018-20": (T18, T21), "2021-23": (T21, T24), "2024-26": (T24, None), "since 2025": (T25, None)}
MINW = 0.10


# ------------------------------------------------------------------ data
_cache = {}


def load(scen):
    """(idx, U (n,S), NF (n,S) bool 'holds a position into hour t', trades DataFrame with column s)."""
    if scen in _cache:
        return _cache[scen]
    Us, NFs, T = [], [], []
    idx = None
    for s, name in enumerate(NAMES):
        z = np.load(os.path.join(SDIR, f"{name}_{scen}.npz"))
        ix = pd.DatetimeIndex(z["t"].astype("datetime64[h]"))
        if idx is None:
            idx = ix
        assert len(ix) == len(idx) and (ix == idx).all(), name
        n = len(idx)
        U = z["U"]
        d = np.zeros(n + 1, np.int64)
        eb, xb = z["eb"], z["xb"]
        cross = xb > eb                       # held at the end of hours eb..xb-1 -> not flat at hours eb+1..xb
        np.add.at(d, eb[cross] + 1, 1)
        np.add.at(d, np.minimum(xb[cross] + 1, n), -1)
        Us.append(U); NFs.append(np.cumsum(d)[:n] > 0)
        T.append(pd.DataFrame(dict(s=s, eb=eb, xb=xb, ret=z["ret"], amt=z["amt"])))
    out = (idx, np.ascontiguousarray(np.column_stack(Us)), np.ascontiguousarray(np.column_stack(NFs)),
           pd.concat(T, ignore_index=True))
    _cache[scen] = out
    return out


# ------------------------------------------------------------------ trailing statistics on the stitched history
_hist = {}


def history(scen):
    """Stitched daily equity (days x S, at 00:00 UTC = the close of the previous day) and closed trades (exit time,
    s, ret) of every sleeve: holdout scenario, then (for main scenarios) the main scenario from 2021-01-01."""
    if scen in _hist:
        return _hist[scen]
    hidx, hU, _, hT = load(HIST[scen])
    parts = [(hidx, hU, hT)]
    if not scen.startswith("h"):
        midx, mU, _, mT = load(scen)
        parts.append((midx, mU, mT))
    days, eqs, trs = [], [], []
    base = np.ones(len(NAMES))
    for ix, U, T in parts:
        e = pd.DataFrame(U * base, ix)
        base = base * U[-1]
        d = e.resample("D").last()
        d.index = d.index + pd.Timedelta(days=1)                # value known at 00:00 of the next day
        days.append(d)
        trs.append(pd.DataFrame(dict(xt=ix[T.xb.values] + pd.Timedelta(hours=1), s=T.s.values, ret=T.ret.values)))
    D = pd.concat(days)
    D = D[~D.index.duplicated(keep="last")]
    first = parts[0][0][0]
    D = pd.concat([pd.DataFrame([np.ones(len(NAMES))], [first.floor("D")]), D])
    D = D[~D.index.duplicated(keep="last")].sort_index()
    TR = pd.concat(trs, ignore_index=True).sort_values("xt", kind="stable")
    _hist[scen] = (D, TR, first)
    return _hist[scen]


def rebal_times(idx, freq, day, t_start):
    """Rebalance hours on idx: 'M' = 00:00 of day `day` of each month, 'W' = 00:00 of weekday `day` (0 = Monday),
    plus the account's first hour t_start."""
    if freq == "M":
        m = (idx.day == day) & (idx.hour == 0)
    else:
        m = (idx.dayofweek == day) & (idx.hour == 0)
    pos = np.flatnonzero(m)
    pos = pos[pos > t_start]
    return np.concatenate([[t_start], pos])


def trailing(scen, taus, L):
    """Per rebalance time (Timestamps) and sleeve: ret, vol (daily), mdd, mu/var (daily), wr, ntr, streak (losers in
    the last 5 closed trades), ok (enough history). Only data strictly before tau."""
    D, TR, first = history(scen)
    dv = D.values; di = D.index
    R = len(taus); S = len(NAMES)
    out = {k: np.full((R, S), np.nan) for k in ("ret", "vol", "mdd", "mu", "var", "wr", "ntr", "streak")}
    ok = np.zeros(R, bool)
    xt = TR.xt.values; ts = TR.s.values; tr = TR.ret.values
    for r, tau in enumerate(taus):
        a = tau - pd.Timedelta(days=L * MD)
        ok[r] = a >= first.floor("D")
        i1 = int(np.searchsorted(di, tau, side="right"))          # values known at or before tau
        i0 = int(np.searchsorted(di, a, side="left"))
        seg = dv[max(i0 - 1, 0):i1]
        if len(seg) < 3:
            continue
        lr = np.diff(np.log(seg), axis=0)
        out["ret"][r] = seg[-1] / seg[0] - 1
        out["vol"][r] = lr.std(axis=0)
        out["mu"][r] = (seg[1:] / seg[:-1] - 1).mean(axis=0)
        out["var"][r] = (seg[1:] / seg[:-1] - 1).var(axis=0)
        out["mdd"][r] = (seg / np.maximum.accumulate(seg, axis=0) - 1).min(axis=0)
        k1 = int(np.searchsorted(xt, np.datetime64(tau), side="right"))
        k0 = int(np.searchsorted(xt, np.datetime64(a), side="left"))
        for s in range(S):
            w = tr[k0:k1][ts[k0:k1] == s]
            out["ntr"][r, s] = len(w)
            out["wr"][r, s] = (w > 0).mean() if len(w) else np.nan
            last = tr[:k1][ts[:k1] == s][-5:]
            out["streak"][r, s] = (last <= 0).sum()
    out["ok"] = ok
    return out


def weights(st, d):
    """Design d -> (R, S) target weights (sum <= 1; the rest stays in the cash pool)."""
    R, S = st["ret"].shape
    W = np.zeros((R, S))
    uni = np.array([n in UNIV[d["univ"]] for n in NAMES])
    for r in range(R):
        if not st["ok"][r]:
            W[r, 0] = 1.0
            continue
        ret = st["ret"][r]; cand = uni & np.isfinite(ret)
        if d.get("streak"):
            cand &= ~(st["streak"][r] >= 2)
        w = np.zeros(S)
        rule = d["rule"]
        if rule == "topk":
            c = np.flatnonzero(cand)
            c = c[np.argsort(-ret[c], kind="stable")][:d["k"]]
            if d["absf"]:
                c = c[ret[c] > 0]
            w[c] = 1.0 / d["k"]
        elif rule == "topwr":
            c = np.flatnonzero(cand & (ret > 0) & (st["ntr"][r] >= 10))
            c = c[np.argsort(-st["wr"][r][c], kind="stable")][:d["k"]]
            w[c] = 1.0 / d["k"]
        elif rule in ("ivol", "idd"):
            m = cand & (ret > 0) if d["absf"] else cand
            x = st["vol"][r] if rule == "ivol" else np.abs(st["mdd"][r])
            x = np.maximum(np.nan_to_num(x, nan=1e9), 0.02 if rule == "idd" else 1e-4)
            w[m] = 1.0 / x[m]
            if w.sum() > 0:
                w /= w.sum()
        elif rule == "kelly":
            k = d["f"] * st["mu"][r] / np.maximum(st["var"][r], 1e-12)
            w[cand] = np.clip(np.nan_to_num(k[cand]), 0.0, 1.0)
            if w.sum() > 1:
                w /= w.sum()
        elif rule == "wrgate":
            c = np.flatnonzero(cand & (st["ntr"][r] >= 10) & (st["wr"][r] >= d["g"]) & (ret > 0))
            if len(c):
                w[c[np.argmax(ret[c])]] = 1.0
        elif rule == "anchor":
            w[0] = d["b"]
            c = np.flatnonzero(cand & (ret > 0))
            if len(c):
                w[c[np.argmax(ret[c])]] += 1.0 - d["b"]
        tot = w.sum()
        w[w < MINW - 1e-12] = 0.0
        if w.sum() > 0:
            w *= tot / w.sum()
        W[r] = w
    return W


# ------------------------------------------------------------------ the one-account kernel
@njit(cache=True)
def akernel(U, NF, rbt, Wt, t0):
    """U (n,S) unit curves, NF (n,S) 'holds a position into hour t', rbt (n,) row of Wt applied at hour t (-1 none),
    t0 first hour of the account. Returns account equity (n,), lambda (n,S), pool (n,)."""
    n, S = U.shape
    lam = np.zeros(S); pend = np.zeros(S, np.bool_); tgt = np.zeros(S)
    row = 0
    pool = 1.0
    eq = np.ones(n); LAM = np.zeros((n, S)); PO = np.ones(n)
    for t in range(t0, n):
        if rbt[t] >= 0:
            row = rbt[t]
            for s in range(S):
                pend[s] = True
        anyp = False
        for s in range(S):
            if pend[s]:
                anyp = True
        if anyp:                                            # target = weight x account equity at the last close
            A = pool
            for s in range(S):
                up = U[t - 1, s] if t > 0 else 1.0
                A += lam[s] * up
            for s in range(S):
                tgt[s] = Wt[row, s] * A
        for s in range(S):                                  # releases first
            if pend[s] and not NF[t, s]:
                up = U[t - 1, s] if t > 0 else 1.0
                cur = lam[s] * up
                if cur >= tgt[s]:
                    pool += cur - tgt[s]
                    lam[s] = tgt[s] / up
                    pend[s] = False
        for s in range(S):                                  # then top-ups from the pool
            if pend[s] and not NF[t, s]:
                up = U[t - 1, s] if t > 0 else 1.0
                cur = lam[s] * up
                add = min(tgt[s] - cur, pool)
                if add > 0.0:
                    pool -= add
                    lam[s] = (cur + add) / up
                if tgt[s] - cur - add <= 1e-12 * (1.0 + tgt[s]):
                    pend[s] = False
        v = pool
        for s in range(S):
            v += lam[s] * U[t, s]
            LAM[t, s] = lam[s]
        eq[t] = v; PO[t] = pool
    return eq, LAM, PO


_trail = {}


def run(scen, design, freq=None, day=None, start=None):
    """One account on scenario scen. freq/day override the design's calendar; start = first hour (Timestamp)."""
    idx, U, NF, T = load(scen)
    freq = freq or design.get("freq", "M")
    day = day if day is not None else (1 if freq == "M" else 0)
    t0 = 0 if start is None else int(np.searchsorted(idx, start))
    pos = rebal_times(idx, freq, day, t0)
    taus = idx[pos]
    if design.get("static") is not None:
        Wt = np.tile(np.asarray(design["static"], np.float64), (len(pos), 1))
    else:
        key = (scen, freq, day, t0, design["L"])
        if key not in _trail:
            _trail[key] = trailing(scen, taus, design["L"])
        Wt = weights(_trail[key], design)
    rbt = np.full(len(idx), -1, np.int64)
    rbt[pos] = np.arange(len(pos))
    eq, LAM, PO = akernel(U, NF, rbt, np.ascontiguousarray(Wt), t0)
    lam_e = LAM[T.eb.values, T.s.values]
    tk = T[lam_e > 0].copy()
    tk["money"] = lam_e[lam_e > 0] * tk.amt.values
    ebm1 = np.maximum(tk.eb.values - 1, 0)
    tk["acct"] = tk.money.values * tk.ret.values / eq[ebm1]
    return pd.Series(eq, idx), tk, LAM, Wt, taus


def stats(e, tk, a, b):
    idx = e.index
    i0 = int(np.searchsorted(idx, a)); i1 = len(idx) if b is None else int(np.searchsorted(idx, b))
    start = e.values[i0 - 1] if i0 > 0 else 1.0
    x = np.concatenate([[1.0], e.values[i0:i1] / start])
    months = ((idx[i1 - 1] + pd.Timedelta(hours=1)) - idx[i0]).total_seconds() / 86400 / MD
    tb = idx[tk.eb.values]
    sel = np.asarray((tb >= a) & ((tb < b) if b is not None else True), bool)
    t = tk[sel]
    return dict(wr=100 * float((t.ret > 0).mean()) if len(t) else np.nan,
                mo=100 * (x[-1] ** (1 / months) - 1) if x[-1] > 0 else -100.0,
                dd=100 * float((x / np.maximum.accumulate(x) - 1).min()),
                wt=100 * float(t.acct.min()) if len(t) else np.nan, tpm=len(t) / months, n=len(t))


def fmt(r):
    return (f"WR {r['wr']:5.1f}% | {r['mo']:+6.2f}%/mo | drop {r['dd']:6.1f}% | worst trade {r['wt']:+6.1f}% acct | "
            f"{r['tpm']:5.1f} tr/mo")


def dname(d):
    if d.get("static") is not None:
        return "static " + "/".join(f"{n}{w:.0%}" for n, w in zip(NAMES, d["static"]) if w > 0)
    extra = {"topk": f"k{d.get('k')} abs{d.get('absf')}", "topwr": f"k{d.get('k')}", "ivol": f"abs{d.get('absf')}",
             "idd": f"abs{d.get('absf')}", "kelly": f"f{d.get('f')}", "wrgate": f"g{d.get('g')}",
             "anchor": f"b{d.get('b')}"}[d["rule"]]
    return (f"{d['univ']} {d['rule']} {extra} L{d['L']} {d.get('freq', 'M')}"
            + (" streak" if d.get("streak") else ""))
