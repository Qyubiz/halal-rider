import glob, os, json
import numpy as np, pandas as pd
import backtest as b
from backtest import ema, rma

def extra(df):
    c, h, l, v, o = df.c, df.h, df.l, df.v, df.o
    f = b.indicators(df)
    e200 = ema(c, 200)
    f["ema50>200"] = ema(c, 50) > e200
    f["c>ema200"] = c > e200
    f["c>ema20"] = c > ema(c, 20)
    d = c.diff()
    rsi = 100 - 100 / (1 + rma(d.clip(lower=0), 14) / rma((-d).clip(lower=0), 14))
    f["rsi>50"] = rsi > 50
    f["rsi rising"] = rsi > rsi.shift(3)
    f["rsi<65"] = rsi < 65
    f["roc10>0"] = c.pct_change(10) > 0
    f["roc3>0"] = c.pct_change(3) > 0
    hh, ll = h.rolling(14).max(), l.rolling(14).min()
    f["willr>-50"] = (-100 * (hh - c) / (hh - ll)) > -50
    tp = (h + l + c) / 3
    mf = tp * v
    pos = mf.where(tp > tp.shift(), 0).rolling(14).sum(); neg = mf.where(tp < tp.shift(), 0).rolling(14).sum()
    mfi = 100 - 100 / (1 + pos / neg)
    f["mfi50-80"] = (mfi > 50) & (mfi < 80)
    ar = lambda n: 100 * h.rolling(n + 1).apply(lambda x: n - np.argmax(x), raw=True) / n
    f["aroon up>70"] = (100 * h.rolling(26).apply(lambda x: np.argmax(x), raw=True) / 25) > 70
    clv = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    cmf = (clv * v).rolling(20).sum() / v.rolling(20).sum()
    f["cmf>0"] = cmf > 0
    f["force>0"] = ema(c.diff() * v, 13) > 0
    ao = ((h + l) / 2).rolling(5).mean() - ((h + l) / 2).rolling(34).mean()
    f["AO>0"] = ao > 0
    f["AO rising"] = ao > ao.shift()
    trix = ema(ema(ema(c, 15), 15), 15).pct_change()
    f["trix>0"] = trix > 0
    f["vol>sma20"] = v > v.rolling(20).mean()
    f["taker buy>50%"] = (df.tb / v) > 0.5
    f["don mid"] = c > (h.rolling(20).max() + l.rolling(20).min()) / 2
    atr = rma(pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1), 14)
    f["kelt>mid"] = c > ema(c, 20)
    f["atr not spiking"] = atr < atr.rolling(50).mean() * 1.5
    f["higher close x3"] = (c > c.shift()) & (c.shift() > c.shift(2))
    f["green candle"] = c > o
    # ichimoku cloud
    sa = ((h.rolling(9).max() + l.rolling(9).min()) / 2 + (h.rolling(26).max() + l.rolling(26).min()) / 2) / 2
    sb = (h.rolling(52).max() + l.rolling(52).min()) / 2
    f["above cloud"] = (c > sa.shift(26)) & (c > sb.shift(26))
    # higher timeframe trend (shifted to avoid lookahead)
    for rule, nm in (("1h", "1h"), ("4h", "4h")):
        rc = c.resample(rule).last().dropna()
        t = (ema(rc, 20) > ema(rc, 50)).shift(1)
        f[f"{nm} ema20>50"] = t.reindex(df.index, method="ffill")
        f[f"{nm} close>ema20"] = (rc > ema(rc, 20)).shift(1).reindex(df.index, method="ffill")
    f = f.fillna(False).astype(bool)
    f.iloc[:250] = False
    return f

def labels(df):
    """Per-bar outcome if we entered at next open with the same exit rules."""
    o, h, l, c = df.o.values, df.h.values, df.l.values, df.c.values
    lowN = df.l.rolling(b.SL_LOOKBACK).min().values
    highN = df.h.rolling(b.TP_LOOKBACK).max().values
    n = len(df); ret = np.full(n, np.nan)
    for i in range(250, n - 2):
        entry = o[i + 1]
        sl = max(lowN[i] * 0.999, entry * (1 - b.SL_MAX)); sl = min(sl, entry * (1 - b.SL_MIN))
        tp = highN[i]
        if (tp - entry) / entry < b.MIN_NET_TP: continue
        px = None
        for j in range(i + 1, min(i + 1 + b.MAX_HOLD, n)):
            if l[j] <= sl: px = sl; break
            if h[j] >= tp: px = tp; break
        if px is None: px = c[min(i + b.MAX_HOLD, n - 1)]
        ret[i] = px / entry - 1 - b.COST
    return ret

if __name__ == "__main__":
    data = {os.path.basename(p)[:-4]: pd.read_csv(p, index_col=0, parse_dates=True) for p in sorted(glob.glob("data/*.csv"))}
    F = {s: extra(d) for s, d in data.items()}
    R = {s: labels(d) for s, d in data.items()}
    names = list(next(iter(F.values())).columns)
    print("indicator pool:", len(names))
    mid = next(iter(data.values())).index[8640]

    def split(half):
        X, y, sym = [], [], []
        for s, d in data.items():
            m = (d.index < mid) if half == 1 else (d.index >= mid)
            m &= ~np.isnan(R[s])
            X.append(F[s].values[m]); y.append(R[s][m])
        return np.vstack(X), np.concatenate(y)
    X1, y1 = split(1); X2, y2 = split(2)
    print("bars with valid setup  H1:", len(y1), " H2:", len(y2), "| baseline win H1/H2:",
          round((y1 > 0).mean() * 100, 1), round((y2 > 0).mean() * 100, 1))

    def rep(mask1, mask2):
        out = {}
        for k, (m, y) in {"H1": (mask1, y1), "H2": (mask2, y2)}.items():
            r = y[m]
            out[k] = dict(n=int(m.sum()), win=round(float((r > 0).mean() * 100), 1) if len(r) else 0,
                          avg=round(float(r.mean() * 100), 3) if len(r) else 0)
        return out

    # greedy forward selection of AND-conjunction, chosen on H1 only
    sel, m1, m2, hist = [], np.ones(len(y1), bool), np.ones(len(y2), bool), []
    for step in range(len(names)):
        best = None
        for j, nm in enumerate(names):
            if j in sel: continue
            mm = m1 & X1[:, j]
            if mm.sum() < 150: continue
            w = (y1[mm] > 0).mean()
            if best is None or w > best[0]: best = (w, j)
        if best is None: break
        sel.append(best[1]); m1 &= X1[:, best[1]]; m2 &= X2[:, best[1]]
        r = rep(m1, m2); r["n_ind"] = len(sel); r["added"] = names[best[1]]
        hist.append(r)
        print(f"{len(sel):2d} +{names[best[1]]:<18} H1 n={r['H1']['n']:5d} win={r['H1']['win']:5.1f}% avg={r['H1']['avg']:+.3f}% | H2 n={r['H2']['n']:5d} win={r['H2']['win']:5.1f}% avg={r['H2']['avg']:+.3f}%")

    # k-of-N across whole pool
    S1, S2 = X1.sum(1), X2.sum(1)
    print("\nk-of-%d pool" % len(names))
    kk = []
    for k in range(18, len(names) + 1):
        r = rep(S1 >= k, S2 >= k); r["k"] = k; kk.append(r)
        if r["H1"]["n"] > 0: print(k, r)
    json.dump(dict(names=names, greedy=hist, kofn=kk), open("search.json", "w"), indent=1)
