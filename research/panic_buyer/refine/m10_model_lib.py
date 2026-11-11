"""m10 "model": a walk-forward machine-learning bounce predictor for the 90% mode dips.

Setting predicted (90% mode, w90_wr90_final F2): 1 coin x 100%, buy limit at 14% under the 24h high close, take-profit
+2.5% (resting limit from the next hourly check), 96h time stop, 3h cooldown, crash guard 35%.

Candidate = (hour t, coin k) whose 14% bid would fill in hour t (open <= limit, or low <= limit x (1 - 0.1%)).
Label     = 1 if the +2.5% take-profit is reached (high >= target x 1.001, or an open >= target) in hours t+1..t+96,
            else 0 (the trade would be sold by the 96h time stop).
Features  = known at the OPEN of hour t: closed candles up to t-1 plus the open of t (list in FEATS).
Training  = expanding window, retrained at the start of every month on candidates whose 96h outcome was fully known
            before that month started (entry hour + 96 < first hour of the month). Never any later data.
The predictions for every (hour, coin) become an smult mask for engine/w90_wr90_lib (0 = skip the coin that hour).

One continuous panel Dec 2017 - Oct 2026: 2017-12..2020-12 from the holdout files (16 coins that existed), 2021+
from pb_data (19 coins, RENDER joined to RNDR). Engine runs use the same panels as the published 90% mode numbers
(engine.load_pre / engine.load_h19), so the baseline reproduces them.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402

OUT = os.environ.get("PB_OUT", os.path.expanduser("~/pb_out_m10/model"))
os.makedirs(OUT, exist_ok=True)

DEPTH, TP, HM, EPS = 0.14, 0.025, 96, E.EPS
F2 = dict(N=1, fmax=1.0, depth=DEPTH, tp=TP, hm=HM, cd=3, guard=0.35)
WF_START = pd.Timestamp("2018-06-01")


# ------------------------------------------------------------------ data
def load_all(dead=False):
    """{sym: hourly DataFrame o,h,l,c,v,qv,tb} from Dec 2017 (where the coin existed) to Oct 2026."""
    pre = E.load_pre(dead=dead)
    main = E.load_h19(dead=dead)
    out = {}
    for s in main:
        parts = []
        if s in pre:
            parts.append(pre[s][pre[s].index < "2021-01-01"])
        parts.append(main[s])
        d = pd.concat(parts)
        d = d[~d.index.duplicated(keep="last")].sort_index()
        out[s] = d[["o", "h", "l", "c", "v", "qv", "tb"]].astype(float)
    return out


def big_panel(D):
    syms = list(D)
    idx = pd.date_range(min(d.index[0] for d in D.values()), max(d.index[-1] for d in D.values()), freq="1h")
    P = {f: pd.DataFrame({s: D[s][f] for s in syms}).reindex(idx) for f in ("o", "h", "l", "c", "v", "qv", "tb")}
    P["cf"] = P["c"].ffill()
    P["ref"] = P["cf"].rolling(24, min_periods=1).max().shift(1)
    return syms, idx, P


# ------------------------------------------------------------------ features (all known at the open of hour t)
FEATS = ["open_gap", "fill_depth", "c1_ref", "hrs_since_high", "low24_ref",
         "r1", "r3", "r6", "r12", "r24", "pre3d", "r7d", "r30d", "dd7", "dd30", "dd90", "up30low",
         "vol24", "vol168", "vol_ratio", "qv1", "qv24", "taker1", "taker24", "clv1", "rng1",
         "btc_r1", "btc_r6", "btc_r24", "btc_c1ref", "btc_r7d", "breadth7", "mkt_c1ref", "frac_deep", "rel24",
         "hour", "dow", "age", "ndips30"]


def _hours_since_max(c, w=24):
    """Hours since the highest close of the previous w closes (0 = the last closed candle), shape (n, K)."""
    a = c.values
    n, K = a.shape
    out = np.full((n, K), np.nan)
    from numpy.lib.stride_tricks import sliding_window_view
    for k in range(K):
        x = a[:, k]
        if n < w:
            continue
        win = sliding_window_view(np.where(np.isnan(x), -np.inf, x), w)        # win[i] = x[i:i+w]
        am = w - 1 - np.argmax(win[:, ::-1], axis=1)                           # last occurrence of the max
        hs = (w - 1 - am).astype(float)                                       # hours before x[i+w-1]
        out[w:, k] = hs[:-1]                                                  # at the open of t: window t-w..t-1
    return out


def features(P, depth=DEPTH):
    """Dict of (n, K) arrays, each known at the open of hour t."""
    C = P["cf"]
    c1 = C.shift(1)
    o = P["o"]
    ref = P["ref"]
    lim = ref * (1 - depth)
    lr = np.log(C / C.shift(1))
    F = {}
    F["open_gap"] = o / lim - 1
    F["fill_depth"] = np.minimum(o, lim) / ref - 1
    F["c1_ref"] = c1 / ref - 1
    F["hrs_since_high"] = pd.DataFrame(_hours_since_max(C), C.index, C.columns)
    F["low24_ref"] = P["l"].rolling(24, min_periods=1).min().shift(1) / ref - 1
    for n_ in (1, 3, 6, 12, 24):
        F[f"r{n_}"] = c1 / C.shift(1 + n_) - 1
    F["pre3d"] = C.shift(25) / C.shift(97) - 1
    F["r7d"] = c1 / C.shift(169) - 1
    F["r30d"] = c1 / C.shift(721) - 1
    F["dd7"] = c1 / C.rolling(168, min_periods=24).max().shift(1) - 1
    F["dd30"] = c1 / C.rolling(720, min_periods=24).max().shift(1) - 1
    F["dd90"] = c1 / C.rolling(2160, min_periods=24).max().shift(1) - 1
    F["up30low"] = c1 / C.rolling(720, min_periods=24).min().shift(1) - 1
    v24 = lr.rolling(24, min_periods=12).std().shift(1)
    v168 = lr.rolling(168, min_periods=48).std().shift(1)
    v720 = lr.rolling(720, min_periods=168).std().shift(1)
    F["vol24"], F["vol168"], F["vol_ratio"] = v24, v168, v24 / v720
    qv = P["qv"]
    F["qv1"] = qv.shift(1) / qv.rolling(168, min_periods=24).mean().shift(1)
    F["qv24"] = qv.rolling(24, min_periods=12).mean().shift(1) / qv.rolling(720, min_periods=168).mean().shift(1)
    F["taker1"] = (P["tb"] / P["v"]).shift(1)
    F["taker24"] = (P["tb"].rolling(24, min_periods=12).sum() / P["v"].rolling(24, min_periods=12).sum()).shift(1)
    rng = (P["h"] - P["l"])
    F["clv1"] = ((P["c"] - P["l"]) / rng.where(rng > 0)).shift(1)
    F["rng1"] = (rng / P["c"]).shift(1)
    b = "BTCUSDT"
    for name, val in (("btc_r1", F["r1"][b]), ("btc_r6", F["r6"][b]), ("btc_r24", F["r24"][b]),
                      ("btc_c1ref", F["c1_ref"][b]), ("btc_r7d", F["r7d"][b])):
        F[name] = pd.DataFrame(np.repeat(val.values[:, None], C.shape[1], 1), C.index, C.columns)
    live = P["c"].shift(1).rolling(24, min_periods=1).count() > 0
    below = (c1 < C.rolling(168, min_periods=84).mean().shift(1)) & live
    nlive = live.sum(axis=1).clip(lower=1)
    rep = lambda s: pd.DataFrame(np.repeat(s.values[:, None], C.shape[1], 1), C.index, C.columns)  # noqa: E731
    F["breadth7"] = rep(below.sum(axis=1) / nlive)
    cr = F["c1_ref"].where(live)
    F["mkt_c1ref"] = rep(cr.mean(axis=1))
    F["frac_deep"] = rep((cr < -0.10).sum(axis=1) / nlive)
    F["rel24"] = F["r24"].sub(F["r24"].where(live).median(axis=1), axis=0)
    F["hour"] = rep(pd.Series(C.index.hour, C.index).astype(float))
    F["dow"] = rep(pd.Series(C.index.dayofweek, C.index).astype(float))
    first = P["c"].apply(lambda s: s.first_valid_index())
    age = np.stack([(C.index - (first[s] if first[s] is not None else C.index[0])).total_seconds() / 86400
                    for s in C.columns], 1)
    F["age"] = pd.DataFrame(np.log1p(np.clip(age, 0, None)), C.index, C.columns)
    cand_prev = (P["l"] <= lim * (1 - EPS)) | (o <= lim)
    F["ndips30"] = cand_prev.astype(float).rolling(720, min_periods=1).sum().shift(1)
    return {k: np.asarray(v.values if hasattr(v, "values") else v, np.float64) for k, v in F.items()}


# ------------------------------------------------------------------ candidates and labels
def candidates(P, depth=DEPTH, tp=TP, hm=HM, slip=E.SLIP_MAJOR, fee=E.FEE):
    """(n, K) bool fill-in-hour-t, fill price, label (take-profit reached within hm hours), realized return
    (take-profit or the close of hour t+hm), and the hour the outcome is known (exit hour)."""
    o, h, l, cf = P["o"].values, P["h"].values, P["l"].values, P["cf"].values
    ref = P["ref"].values
    lim = ref * (1 - depth)
    n, K = o.shape
    with np.errstate(invalid="ignore"):
        mkt = o <= lim
        cand = (mkt | (l <= lim * (1 - EPS))) & ~np.isnan(o) & ~np.isnan(ref)
    fill = np.where(mkt, o * (1 + slip), lim)
    y = np.full((n, K), np.nan); ret = np.full((n, K), np.nan); xh = np.full((n, K), -1, np.int64)
    hh = np.where(np.isnan(h), -np.inf, h)
    oo = np.where(np.isnan(o), -np.inf, o)
    for k in range(K):
        ts = np.flatnonzero(cand[:, k])
        for t in ts:
            tgt = fill[t, k] * (1 + tp)
            e = min(t + hm, n - 1)
            if e <= t:
                continue
            hit = np.flatnonzero((hh[t + 1:e + 1, k] >= tgt * (1 + EPS)) | (oo[t + 1:e + 1, k] >= tgt))
            if len(hit):
                j = t + 1 + hit[0]
                px = max(tgt, o[j, k]) if o[j, k] >= tgt else tgt
                y[t, k] = 1.0; xh[t, k] = j
                ret[t, k] = px / fill[t, k] * (1 - fee) ** 2 - 1
            elif t + hm <= n - 1:
                y[t, k] = 0.0; xh[t, k] = t + hm
                ret[t, k] = cf[t + hm, k] * (1 - slip) / fill[t, k] * (1 - fee) ** 2 - 1
            # else: unresolved at the end of the data -> stays NaN
    return cand, fill, y, ret, xh


# ------------------------------------------------------------------ walk-forward
def month_starts(idx, start=WF_START):
    m = pd.date_range(start, idx[-1], freq="MS")
    return [int(np.searchsorted(idx.values, np.datetime64(x))) for x in m], m


def make_model(kind="lgb", seed=0):
    if kind == "lgb":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                                  n_jobs=1, random_state=seed, verbose=-1)
    if kind == "logit":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.1, max_iter=2000))
    raise ValueError(kind)


def walk_forward(X, cand, y, xh, train_mask=None, kind="lgb", start=WF_START, idx=None, retrain="MS",
                 log=print, hm=HM):
    """X: (n, K, F). Returns pred (n, K) for every hour from `start` (NaN before), each month predicted by a model
    trained only on candidates bought at least hm+1 hours before the month's first hour, so every one of them (winner
    or loser) was resolved before the month started and recent losers are not under-represented. train_mask (n, K)
    bool limits the training rows (e.g. the 19 live coins)."""
    n, K, nf = X.shape
    pred = np.full((n, K), np.nan)
    m = pd.date_range(start, idx[-1], freq=retrain)
    bounds = [int(np.searchsorted(idx.values, np.datetime64(x))) for x in m] + [n]
    tr_ok = cand & ~np.isnan(y) & (xh >= 0)
    if train_mask is not None:
        tr_ok &= train_mask
    rows_t, rows_k = np.nonzero(tr_ok)
    t0 = time.time()
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        sel = rows_t + hm < a                  # bought >= hm+1 hours before: outcome known before the month starts
        Xt = X[rows_t[sel], rows_k[sel]]; yt = y[rows_t[sel], rows_k[sel]]
        if len(yt) < 200 or yt.min() == yt.max():
            continue
        mdl = make_model(kind)
        mdl.fit(Xt, yt)
        blk = X[a:b].reshape(-1, nf)
        pred[a:b] = mdl.predict_proba(blk)[:, 1].reshape(b - a, K)
        if i % 12 == 0:
            log(f"   retrain {idx[a]:%Y-%m}: {len(yt)} training candidates (win rate {yt.mean()*100:.1f}%), "
                f"{time.time()-t0:.0f}s", flush=True)
    return pred


def walk_forward2(X, rows_t, rows_k, target, idx, feat=None, weight=None, kind="lgb", params=None,
                  start=WF_START, retrain="MS", hm=HM, log=print, Xrows=None):
    """General walk-forward. Training rows are given explicitly: (rows_t, rows_k) entry hour/coin, target per row
    (binary -> classifier P(1); float -> regressor), optional per-row weight, optional extra rows Xrows=(Xr, tr, yr, wr)
    from other coins (features (m, F), entry hour index on the same idx, target, weight). feat: feature indices.
    A row trains month M's model only if its entry hour + hm < M's first hour."""
    n, K, nf = X.shape
    feat = np.arange(nf) if feat is None else np.asarray(feat)
    pred = np.full((n, K), np.nan)
    m = pd.date_range(start, idx[-1], freq=retrain)
    bounds = [int(np.searchsorted(idx.values, np.datetime64(x))) for x in m] + [n]
    Xa = X[rows_t, rows_k][:, feat]; ya = np.asarray(target, np.float64)
    wa = np.ones(len(ya)) if weight is None else np.asarray(weight, np.float64)
    ta = np.asarray(rows_t)
    if Xrows is not None:
        Xr, tr, yr, wr = Xrows
        Xa = np.vstack([Xa, Xr[:, feat]]); ya = np.concatenate([ya, yr]); ta = np.concatenate([ta, tr])
        wa = np.concatenate([wa, wr])
    binary = set(np.unique(ya)) <= {0.0, 1.0}
    t0 = time.time()
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        sel = ta + hm < a
        if sel.sum() < 200 or (binary and len(np.unique(ya[sel])) < 2):
            continue
        if kind == "lgb":
            import lightgbm as lgb
            pp = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100, subsample=0.8,
                      subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, n_jobs=1, random_state=0, verbose=-1)
            pp.update(params or {})
            mdl = lgb.LGBMClassifier(**pp) if binary else lgb.LGBMRegressor(**pp)
            mdl.fit(Xa[sel], ya[sel], sample_weight=wa[sel])
        else:
            mdl = make_model(kind)
            mdl.fit(Xa[sel], ya[sel], **({"logisticregression__sample_weight": wa[sel]} if kind == "logit" else {}))
        blk = X[a:b][:, :, feat].reshape(-1, len(feat))
        out = mdl.predict_proba(blk)[:, 1] if binary else mdl.predict(blk)
        pred[a:b] = out.reshape(b - a, K)
        if i % 24 == 0:
            log(f"   retrain {idx[a]:%Y-%m}: {int(sel.sum())} rows, {time.time()-t0:.0f}s", flush=True)
    return pred


def trailing_threshold(pred, cand, idx, q, start=WF_START, look_days=365):
    """(n,) threshold per hour: for each month M, the q-quantile of the out-of-sample predictions made on candidate
    hours in the look_days before M (all made by earlier models, so known at M's start). -inf when none exist yet."""
    n = len(idx)
    thr = np.full(n, -np.inf)
    m = pd.date_range(start, idx[-1], freq="MS")
    bounds = [int(np.searchsorted(idx.values, np.datetime64(x))) for x in m] + [n]
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        a0 = int(np.searchsorted(idx.values, np.datetime64(idx[a] - pd.Timedelta(days=look_days))))
        v = pred[a0:a][cand[a0:a]]
        v = v[~np.isnan(v)]
        if len(v) >= 50:
            thr[a:b] = np.quantile(v, q)
    return thr


def auc(yv, pv):
    from sklearn.metrics import roc_auc_score
    ok = ~np.isnan(yv) & ~np.isnan(pv)
    if ok.sum() < 20 or len(np.unique(yv[ok])) < 2:
        return np.nan
    return float(roc_auc_score(yv[ok], pv[ok]))


def build(dead=False, depth=DEPTH, tp=TP, hm=HM):
    """Everything the experiments need, cached on disk: big panel, feature tensor, candidates and labels."""
    tag = f"d{depth}_tp{tp}_hm{hm}{'_dead' if dead else ''}"
    p = os.path.join(OUT, f"build_{tag}.npz")
    D = load_all(dead=dead)
    syms, idx, P = big_panel(D)
    if os.path.exists(p):
        z = np.load(p)
        return syms, idx, P, z["X"], z["cand"], z["fill"], z["y"], z["ret"], z["xh"]
    F = features(P, depth)
    X = np.stack([F[f] for f in FEATS], axis=2).astype(np.float32)
    X[~np.isfinite(X)] = np.nan
    cand, fill, y, ret, xh = candidates(P, depth, tp, hm)
    np.savez(p, X=X, cand=cand, fill=fill, y=y, ret=ret, xh=xh)
    return syms, idx, P, X, cand, fill, y, ret, xh


# ------------------------------------------------------------------ masks onto the engine panels
def to_panel(arr, idx_big, syms_big, idx, syms, fill=np.nan):
    """Reindex an (n_big, K_big) array onto an engine panel's (idx, syms)."""
    df = pd.DataFrame(arr, idx_big, syms_big).reindex(index=idx, columns=syms)
    return df.fillna(fill).values if not (isinstance(fill, float) and np.isnan(fill)) else df.values
