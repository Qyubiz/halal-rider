"""m14 "sleeves": N slightly different copies of 90% mode, each in its OWN account (1/N of the money, own trades).
Each sleeve = m11_fusion_lib.fk with one tier (proven == 90% mode with the switches off, see m14_sleeves_grid.py
check). The account is the weighted sum of the sleeve equity curves: rb=0 never rebalanced (each sleeve keeps its own
money), rb=1 reset to equal weights at the first hour of each month (idealised: assumes position value can be moved;
the bot version would use free cash only, so rb=1 is an optimistic upper bound for rebalancing)."""
import numpy as np, pandas as pd
import m11_fusion_lib as F, m10_velocity_lib as V

_c = {}


def sleeve_params(N, dc, ds, ts, cs, tpc=0.025):
    offs = {3: [-1, 0, 1], 5: [-2, -1, 0, 1, 2]}[N]
    cdo = [0, -1, 1, -1, 1][:N] if cs else [0] * N
    sl = []
    for j in range(N):
        sl.append(dict(depth=round(dc + offs[j] * ds, 4), tp=round(tpc + offs[(j + 1) % N] * ts, 4),
                       cd=max(3 + cdo[(j + 2) % N] * cs, 0)))
    return sl


def one(P, idx, tag, sp, kw):
    k = (tag, sp["depth"], sp["tp"], sp["cd"], tuple(sorted(kw.items(), key=str)))
    if k not in _c:
        d = dict(F.MODE90, tiers=[dict(F.TIERA, depth=sp["depth"], tp=sp["tp"], cdw=sp["cd"], cdl=sp["cd"])])
        _c[k] = F.frun(P, idx, d, **kw)
    return _c[k]


def combine(res, idx, rb):
    eqs = np.stack([r[0].values for r in res]); N = len(res)
    if rb == 0:
        e = eqs.mean(0)
    else:
        e = np.empty(eqs.shape[1]); per = idx.to_period("M"); st = 1.0
        brk = np.flatnonzero(np.r_[True, per[1:] != per[:-1]])
        for bi, a in enumerate(brk):
            b = brk[bi + 1] if bi + 1 < len(brk) else len(idx)
            prev = eqs[:, a - 1] if a > 0 else np.ones(N)
            e[a:b] = st * (eqs[:, a:b] / prev[:, None]).mean(0)
            st = e[b - 1]
    lgs = []
    for i, r in enumerate(res):
        l = r[1].copy(); l["tier"] = i; lgs.append(l)
    lg = pd.concat(lgs, ignore_index=True)
    inv = sum(r[2].values for r in res) / N
    return pd.Series(e, idx), lg, pd.Series(inv, idx)


def run(P, idx, tag, sl, rb, kw):
    return combine([one(P, idx, tag, sp, kw) for sp in sl], idx, rb)


KWS = {"n": {}, "w": dict(pen=F.PEN, pens=F.PEN), "cm": dict(cd_add=-1), "cp": dict(cd_add=1),
       "bt": dict(mtp_now=True, cd_add=1), "ls": dict(t0=F.LATE)}


def scores(sl, rb, P, idx):
    out = {}
    for tg, kw in KWS.items():
        e, lg, inv = run(P, idx, "main", sl, rb, kw)
        per = (("d", F.LATE, V.T24),) if tg == "ls" else F.DPER
        r = V.summarize(e, lg, inv, idx, per)
        out[f"{tg}_wr"], out[f"{tg}_mo"], out[f"{tg}_dd"] = r["d_wr"], r["d_mo"], r["d_dd"]
        if tg == "n":
            out["n_tpm"] = r["d_tpm"]
    out["robust"] = float(np.mean([out[f"{t}_mo"] for t in ("n", "cm", "cp", "bt", "ls")]))
    return out
