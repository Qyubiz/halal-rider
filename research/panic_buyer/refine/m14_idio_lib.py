"""m14 idio lib: two-regime bid. breadth[t] = number of the 19 coins whose last closed hour is >=10% under their 24h
highest close (known at hour open). breadth >= kk: CRASH regime (tier 1: depth dc, coins = majors or all or none);
else SINGLE-coin regime (tier 0: depth ds, any coin). Everything else = 90% mode via m11 fk."""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m11_fusion_lib as F, m10_velocity_lib as V, engine as E
MAJ = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
OUT = F.OUT
OFF = dict(kk=99, ds=0.14, dc=0.14, cs="all")
_c = {}


def breadth(P):
    k = ("b", id(P["o"]))
    if k not in _c:
        cols = [s for s in P["o"].columns if s in E.H19]
        cl = P["cf"][cols].shift(1).values; ref = P["ref"][cols].values
        with np.errstate(invalid="ignore"):
            dn = np.where(np.isfinite(cl) & np.isfinite(ref), cl <= 0.9 * ref, False)
        _c[k] = dn.sum(1)
    return _c[k]


def frun(P, idx, d, **kw):
    kk = d["kk"]
    O = P["o"].values; n, K = O.shape
    g = F.guard(P, 0.35)
    crash = (breadth(P) >= kk)[:, None] & np.ones((1, K), bool)
    if d["cs"] == "maj":
        cm = np.array([s in MAJ for s in P["o"].columns])[None, :]
    elif d["cs"] == "all":
        cm = np.ones((1, K), bool)
    else:
        cm = np.zeros((1, K), bool)
    m0 = g * (~crash); m1 = g * (crash & cm)
    gm = np.ascontiguousarray(np.stack([m0, m1]))
    tp = F.tp_table(P, 0.025)
    tpS = np.ascontiguousarray(np.stack([tp, tp]))
    refs = np.ascontiguousarray(np.stack([F.ref(P, 24)]))
    rid = np.zeros(2, np.int64)
    liquidate = kw.get("liquidate", False)
    sl = np.array([E.SLIP_MAJOR if s in E.H19 else E.SLIP_OTHER for s in P["o"].columns]) if liquidate else np.full(K, E.SLIP_MAJOR)
    lastb = V.last_bars(P) if liquidate else np.full(K, -1, np.int64)
    cd_add = kw.get("cd_add", 0); t0 = kw.get("t0")
    i0 = 0 if t0 is None else int(np.searchsorted(idx.values, np.datetime64(t0)))
    f = lambda v, dt=np.float64: np.array(v, dt)
    cdw = f([3, 3], np.int64) + cd_add
    eq, lg, inv = F.fk(O, P["h"].values, P["l"].values, P["cf"].values, gm, refs, rid, lastb, f([d["ds"], d["dc"]]), tpS,
                       f([96, 96], np.int64), np.maximum(cdw, 0), np.maximum(cdw, 0), f([1.0, 1.0]), 99, 1, float(E.EPS),
                       float(E.FEE), sl, float(kw.get("pen", 0.0)), float(kw.get("pens", 0.0)), bool(kw.get("mtp_now", False)),
                       i0, 0.0, 0.0, 0.0, 720, 168, 0, 0)
    return pd.Series(eq, idx), pd.DataFrame(lg, columns=V.LOGC), pd.Series(inv, idx)


DPER = (("d", V.T21, V.T24),)


def scores(d):
    syms, idx, P = V.market(False)
    out = {}
    runs = {"n": {}, "w": dict(pen=F.PEN, pens=F.PEN), "cm": dict(cd_add=-1), "cp": dict(cd_add=1),
            "bt": dict(mtp_now=True, cd_add=1), "ls": dict(t0=F.LATE)}
    for tag, kw in runs.items():
        e, lg, inv = frun(P, idx, d, **kw)
        per = (("d", F.LATE, V.T24),) if tag == "ls" else DPER
        r = V.summarize(e, lg, inv, idx, per)
        out[f"{tag}_wr"], out[f"{tag}_mo"], out[f"{tag}_dd"] = r["d_wr"], r["d_mo"], r["d_dd"]
        if tag == "n": out["n_tpm"] = r["d_tpm"]
    out["robust"] = float(np.mean([out[f"{t}_mo"] for t in ("n", "cm", "cp", "bt", "ls")]))
    return out


def ok_s(s, ver=("n", "w")):
    return all(s[f"{t}_wr"] >= 0.925 and s[f"{t}_dd"] > -0.644 - 1e-9 for t in ver) and s["n_mo"] > 0.1317 and s["w_mo"] > 0.0979


def evaluate(d, worse=False, stress=False, holdout=False):
    dead = stress
    syms, idx, P = V.market_holdout(dead) if holdout else V.market(dead)
    kw = {}
    if worse or stress: kw.update(pen=F.PEN, pens=F.PEN)
    if stress: kw.update(mtp_now=True, cd_add=1, liquidate=True)
    e, lg, inv = frun(P, idx, d, **kw)
    return V.summarize(e, lg, inv, idx, V.HOLD if holdout else V.MAIN), e, lg
