"""m15 memory: 90% mode (m11_fusion_lib.fk) + coin-level memory switches. All switches off == 90% mode.
 bl   blacklist days D: after a LOSING trade the coin is barred D days (cdl = 24*D hours, per-coin cooldown in fk)
 dk,dM repeat-dip: skip coin if >= dk separate 24h-trigger episodes (low <= ref24*0.86, no hit in the 24h before) began in
       [t-dM days, t-24h) (known at the hour's open)
 tW    trend A: skip if last close < lowest close of the 'tW' days ending tW days ago
 tx    trend B: skip if last close < tx * highest close of previous 90 days
"""
import numpy as np, pandas as pd
import m11_fusion_lib as F, m10_velocity_lib as V, engine as E

_c = {}


def extra_mask(P, dk=0, dM=0, tW=0, tx=0.0):
    n, K = P["o"].shape
    m = np.ones((n, K))
    key0 = id(P["o"])
    if dk > 0:
        k = (key0, "dip")
        if k not in _c:
            ref = P["ref"].values; lo = P["l"].values
            with np.errstate(invalid="ignore"):
                hit = np.where(np.isfinite(ref) & np.isfinite(lo), lo <= ref * 0.86, False)
            hp = pd.DataFrame(hit.astype(float)).rolling(24, min_periods=1).max().shift(1).fillna(0).values > 0
            _c[k] = np.cumsum(hit & ~hp, axis=0)
        cs = _c[k]
        for_t = np.arange(n)
        a = np.clip(for_t - 24 - 1, 0, n - 1); b = np.clip(for_t - dM * 24 - 1, 0, n - 1)
        cnt = cs[a] - cs[b]
        cnt[for_t - 25 < 0] = 0
        m *= (cnt < dk)
    cf = P["cf"]
    if tW > 0:
        k = (key0, "tA", tW)
        if k not in _c:
            w = tW * 24
            low = cf.rolling(w, min_periods=w // 2).min().shift(w)
            _c[k] = ((cf.shift(1) < low) & low.notna()).values
        m *= ~_c[k]
    if tx > 0:
        k = (key0, "tB", tx)
        if k not in _c:
            hi = cf.rolling(2160, min_periods=1).max().shift(1)
            _c[k] = (cf.shift(1) < tx * hi).values
        m *= ~_c[k]
    return m


def mrun(P, idx, bl=0, dk=0, dM=0, tW=0, tx=0.0, pen=0.0, pens=0.0, mtp_now=False, cd_add=0, liquidate=False, t0=None):
    T = F.TIERA
    O = P["o"].values; n, K = O.shape
    R = np.ascontiguousarray(np.stack([F.ref(P, 24)]))
    rid = np.zeros(1, np.int64)
    sl = np.array([E.SLIP_MAJOR if s in E.H19 else E.SLIP_OTHER for s in P["o"].columns]) if liquidate else np.full(K, E.SLIP_MAJOR)
    lastb = V.last_bars(P) if liquidate else np.full(K, -1, np.int64)
    g = F.guard(P, 0.35) * extra_mask(P, dk, dM, tW, tx)
    gmS = np.ascontiguousarray(g[None])
    tpS = np.ascontiguousarray(np.full((1, n, K), 0.025))
    cdl = max(3, int(round(bl * 24)))
    i0 = 0 if t0 is None else int(np.searchsorted(idx.values, np.datetime64(t0)))
    eq, lg, inv = F.fk(O, P["h"].values, P["l"].values, P["cf"].values, gmS, R, rid, lastb, np.array([0.14]), tpS,
                       np.array([96], np.int64), np.array([max(3 + cd_add, 0)], np.int64), np.array([max(cdl + cd_add, 0)], np.int64),
                       np.array([1.0]), 99, 1, float(E.EPS), float(E.FEE), sl, float(pen), float(pens), bool(mtp_now), i0,
                       0.0, 0.0, 0.0, 720, 168, 0, 0)
    return pd.Series(eq, idx), pd.DataFrame(lg, columns=V.LOGC), pd.Series(inv, idx)


PEN = 0.002
LATE = F.LATE
RUNS = {"n": {}, "w": dict(pen=PEN, pens=PEN), "cm": dict(cd_add=-1), "cp": dict(cd_add=1),
        "bt": dict(mtp_now=True, cd_add=1), "ls": dict(t0=LATE)}


def scores(d):
    syms, idx, P = V.market(False)
    out = {}
    for tag, kw in RUNS.items():
        e, lg, inv = mrun(P, idx, **d, **kw)
        per = (("d", LATE, V.T24),) if tag == "ls" else F.DPER
        r = V.summarize(e, lg, inv, idx, per)
        out[f"{tag}_wr"], out[f"{tag}_mo"], out[f"{tag}_dd"] = r["d_wr"], r["d_mo"], r["d_dd"]
    out["robust"] = float(np.mean([out[f"{t}_mo"] for t in ("n", "cm", "cp", "bt", "ls")]))
    return out


def full(d, worse=False, stress=False, holdout=False):
    dead = stress
    syms, idx, P = V.market_holdout(dead) if holdout else V.market(dead)
    kw = {}
    if worse or stress: kw.update(pen=PEN, pens=PEN)
    if stress: kw.update(mtp_now=True, cd_add=1)
    if dead: kw["liquidate"] = True
    e, lg, inv = mrun(P, idx, **d, **kw)
    return V.summarize(e, lg, inv, idx, V.HOLD if holdout else V.MAIN), e, lg, idx


def loser_stats(lg, idx, periods):
    tb = idx[lg.eb.astype(int).values]
    acct = lg.ret * lg.amt / lg.eq0
    out = {}
    for tag, a, b in periods:
        s = (tb >= a) & ((tb < b) if b is not None else True)
        x = acct[s]; L = x[x <= 0]
        out[tag] = (float(L.mean()) if len(L) else 0.0, float(x.min()) if len(x) else 0.0)
    return out
