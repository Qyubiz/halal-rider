import glob, os, json
import numpy as np, pandas as pd
import backtest as b
import search as S

def to_1h(d):
    return d.resample("1h").agg({"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum", "qv": "sum", "tb": "sum"}).dropna()

def labels(df, tp_lb, sl_lb, hold):
    o, h, l, c = df.o.values, df.h.values, df.l.values, df.c.values
    lowN = df.l.rolling(sl_lb).min().values; highN = df.h.rolling(tp_lb).max().values
    n = len(df); ret = np.full(n, np.nan); sld = np.full(n, np.nan); tpd = np.full(n, np.nan)
    for i in range(250, n - 2):
        e = o[i + 1]
        sl = max(lowN[i] * 0.999, e * (1 - 0.04)); sl = min(sl, e * (1 - 0.004)); tp = highN[i]
        if tp <= e * 1.004: continue
        px = None
        for j in range(i + 1, min(i + 1 + hold, n)):
            if l[j] <= sl: px = sl; break
            if h[j] >= tp: px = tp; break
        if px is None: px = c[min(i + hold, n - 1)]
        ret[i] = px / e - 1 - b.COST; sld[i] = 1 - sl / e; tpd[i] = tp / e - 1
    return ret, sld, tpd

def run_tf(name, data, tp_lb, sl_lb, hold, runup_bars):
    F = {s: S.extra(d) for s, d in data.items()}
    L = {s: labels(d, tp_lb, sl_lb, hold) for s, d in data.items()}
    mid_idx = len(next(iter(data.values()))) // 2
    mid = next(iter(data.values())).index[mid_idx]
    names = list(next(iter(F.values())).columns) + ["rr>=1", "rr>=1.5", "rr>=2", "tp>=1%", "runup<1%", "runup<2%", "runup<3%"]
    Xs = {1: [], 2: []}; ys = {1: [], 2: []}; syms = {1: [], 2: []}
    for s, d in data.items():
        ret, sld, tpd = L[s]
        ru = d.c.pct_change(runup_bars).values
        rr = tpd / sld
        ext = np.column_stack([rr >= 1, rr >= 1.5, rr >= 2, tpd >= 0.01, ru < 0.01, ru < 0.02, ru < 0.03])
        X = np.hstack([F[s].values, ext])
        for half in (1, 2):
            m = (d.index < mid) if half == 1 else (d.index >= mid)
            m = m & ~np.isnan(ret)
            Xs[half].append(X[m]); ys[half].append(ret[m]); syms[half].append(np.full(m.sum(), s))
    X1, X2 = np.vstack(Xs[1]), np.vstack(Xs[2]); y1, y2 = np.concatenate(ys[1]), np.concatenate(ys[2])
    sym2 = np.concatenate(syms[2])
    print(f"\n=== {name}  setups H1={len(y1)} H2={len(y2)}  baseline avg H1={y1.mean()*100:+.3f}% H2={y2.mean()*100:+.3f}%")
    sel, m1, m2, best_hist = [], np.ones(len(y1), bool), np.ones(len(y2), bool), []
    for step in range(8):
        best = None
        for j in range(len(names)):
            if j in sel: continue
            mm = m1 & X1[:, j]
            if mm.sum() < 150: continue
            a = y1[mm].mean()
            if best is None or a > best[0]: best = (a, j)
        if best is None: break
        sel.append(best[1]); m1 &= X1[:, best[1]]; m2 &= X2[:, best[1]]
        r1, r2 = y1[m1], y2[m2]
        rec = dict(tf=name, k=len(sel), added=names[best[1]], n1=int(m1.sum()), win1=round((r1 > 0).mean() * 100, 1), avg1=round(r1.mean() * 100, 3),
                   n2=int(m2.sum()), win2=round((r2 > 0).mean() * 100, 1) if len(r2) else 0, avg2=round(r2.mean() * 100, 3) if len(r2) else 0,
                   total2=round(r2.sum() * 100, 1) if len(r2) else 0)
        best_hist.append(rec)
        print(f"{rec['k']} +{rec['added']:<16} H1 n={rec['n1']:5d} win={rec['win1']:5.1f}% avg={rec['avg1']:+.3f}% | H2 n={rec['n2']:5d} win={rec['win2']:5.1f}% avg={rec['avg2']:+.3f}% total={rec['total2']:+.1f}%")
    return best_hist, [names[j] for j in sel]

if __name__ == "__main__":
    d15 = {os.path.basename(p)[:-4]: pd.read_csv(p, index_col=0, parse_dates=True) for p in sorted(glob.glob("data/*.csv"))}
    d1h = {s: to_1h(d) for s, d in d15.items()}
    out = []
    h, sel = run_tf("15m", d15, 48, 20, 48, 8); out += h; print("selected:", sel)
    h, sel = run_tf("1h", d1h, 48, 20, 48, 2); out += h; print("selected:", sel)
    json.dump(out, open("profit_search.json", "w"), indent=1)
