"""Round 2, lens "entry": which buys become time-stop losers?

Runs the live settings, keeps ONLY trades bought before 2024 (design period), and attaches features that the bot
could compute at the start of the hour the order was placed (closed candles only: everything uses data up to the
previous close, i.e. shift(1)). Prints how each feature separates time-stop (why=1) trades from take-profit trades.
No 2024-26 number is printed by this script.
"""
import os

import numpy as np
import pandas as pd

import engine as E
import study as S

pd.set_option("display.width", 200)
syms, idx, P = S.market()
base, e, lg = S.evaluate({})
lg = lg.copy()
lg["bought"] = idx[lg.eb.astype(int).values]
lg = lg[lg.bought < E.DESIGN_END].reset_index(drop=True)          # design-period trades only
lg["coin"] = [syms[int(k)] for k in lg.k]
lg["ts"] = (lg.why == 1).astype(int)
lg["contrib"] = lg.ret * lg.amt                                   # return x order size (equity units)
print(f"2021-23 trades: {len(lg)}  time stops: {lg.ts.sum()}  "
      f"sum ret x size: TP {lg.contrib[lg.ts == 0].sum():+.2f}  TS {lg.contrib[lg.ts == 1].sum():+.2f}")


def feats(P, syms):
    """Dict of (n,K) DataFrames, each value known at the start of hour t (uses closes up to t-1)."""
    cf = P["cf"]
    c1 = cf.shift(1)                                               # last closed candle
    F = {}
    F["dd24"] = c1 / P["ref"] - 1                                  # how far under the 24h high (trigger at -12%)
    F["r1h"] = c1 / cf.shift(2) - 1
    F["r4h"] = c1 / cf.shift(5) - 1                                # speed of the fall
    F["r3d"] = c1 / cf.shift(73) - 1
    F["r7d"] = c1 / cf.shift(169) - 1
    F["pre3d"] = cf.shift(25) / cf.shift(97) - 1                   # 3d return BEFORE the last 24h
    F["pre7d"] = cf.shift(25) / cf.shift(193) - 1
    F["d30h"] = c1 / cf.rolling(720, min_periods=240).max().shift(1) - 1   # distance from the 30d high
    F["d7h"] = c1 / cf.rolling(168, min_periods=84).max().shift(1) - 1
    # hours since the 24h high close
    hi_age = cf.rolling(24, min_periods=1).apply(lambda a: len(a) - 1 - np.argmax(a), raw=True).shift(1)
    F["hi_age"] = hi_age
    qv = P["qv"].fillna(0)
    F["qv24_30d"] = (qv.rolling(24).sum() / (qv.rolling(720, min_periods=240).sum() / 30)).shift(1)
    F["qv4_30d"] = (qv.rolling(4).sum() / (qv.rolling(720, min_periods=240).sum() / 180)).shift(1)
    F["vol7d"] = S.roll_vol(P, 168)
    F["vol1d"] = S.roll_vol(P, 24)
    F["vratio"] = F["vol1d"] / F["vol7d"]
    b = cf["BTCUSDT"]
    bc1 = b.shift(1)
    bcast = lambda s: pd.DataFrame(np.repeat(s.values[:, None], len(syms), 1), s.index, syms)
    F["btc_dd24"] = bcast(S.btc_dd24(P))
    F["btc_r4h"] = bcast(bc1 / b.shift(5) - 1)
    F["btc_r7d"] = bcast(bc1 / b.shift(169) - 1)
    F["btc_d30h"] = bcast(bc1 / b.rolling(720, min_periods=240).max().shift(1) - 1)
    live = c1.notna()
    near = (F["dd24"] <= -0.08) & live
    F["n_near8"] = bcast(near.sum(axis=1).astype(float))           # coins within 4 points of trigger
    F["n_under"] = bcast(((F["dd24"] <= -0.12) & live).sum(axis=1).astype(float))
    F["med_dd24"] = bcast(F["dd24"].median(axis=1))                # market-wide drop
    F["idio"] = F["dd24"] - F["med_dd24"]                          # coin drop beyond the market's
    below7 = (c1 < cf.rolling(168, min_periods=84).mean().shift(1)) & live
    F["breadth7"] = bcast(below7.sum(axis=1) / live.sum(axis=1).clip(lower=1))
    F["c_vs_7dma"] = c1 / cf.rolling(168, min_periods=84).mean().shift(1) - 1
    F["c_vs_30dma"] = c1 / cf.rolling(720, min_periods=240).mean().shift(1) - 1
    # open of the hour vs the trigger (open is known at the hour start; < 0 = market buy)
    F["open_vs_trig"] = P["o"] / (P["ref"] * 0.88) - 1
    # previous 12% trigger events: did this coin trade 12% under its 24h high during the last 3 / 7 days?
    hit = (P["l"] <= P["ref"] * 0.88 * (1 - E.EPS)).astype(float)
    F["hits3d"] = hit.rolling(72, min_periods=1).sum().shift(25)   # ignoring the last 24h (current dip)
    F["hits7d"] = hit.rolling(168, min_periods=1).sum().shift(25)
    return F


F = feats(P, syms)
eb = lg.eb.astype(int).values
kk = lg.k.astype(int).values
for name, df in F.items():
    lg[name] = df.values[eb, kk]
lg["hour"] = lg.bought.dt.hour
lg["dow"] = lg.bought.dt.dayofweek
lg["mkt_buy"] = (lg.open_vs_trig <= 0).astype(int)

# path-dependent: did this coin's previous trade end in a time stop, and how long ago?
full = e  # noqa
prev_ts, prev_gap = [], []
last = {}
for i, r in lg.sort_values("eb").iterrows():
    k = int(r.k)
    if k in last:
        pxb, pwhy = last[k]
        prev_ts.append((i, int(pwhy == 1), r.eb - pxb))
    else:
        prev_ts.append((i, 0, np.nan))
    last[k] = (r.xb, r.why)
pt = pd.DataFrame(prev_ts, columns=["i", "prev_ts", "prev_gap"]).set_index("i")
lg = lg.join(pt)
lg["prev_ts7d"] = ((lg.prev_ts == 1) & (lg.prev_gap <= 168)).astype(int)
lg["prev_ts14d"] = ((lg.prev_ts == 1) & (lg.prev_gap <= 336)).astype(int)

lg.to_csv(os.path.join(S.OUT, "entry_trades_2021_23.csv"), index=False)

print("\nmeans by exit type (TP=0 / TS=1), and AUC of each feature for predicting a time stop")
rows = []
for c in list(F) + ["hour", "mkt_buy", "prev_ts7d", "prev_ts14d"]:
    x = lg[c]
    ok = x.notna()
    a, y = x[ok], lg.ts[ok]
    # AUC via ranks
    rk = a.rank()
    n1, n0 = y.sum(), (1 - y).sum()
    auc = (rk[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan
    rows.append(dict(feat=c, tp_mean=a[y == 0].mean(), ts_mean=a[y == 1].mean(), tp_med=a[y == 0].median(),
                     ts_med=a[y == 1].median(), auc=auc, nan=int((~ok).sum())))
A = pd.DataFrame(rows).set_index("feat")
A["sep"] = (A.auc - 0.5).abs()
print(A.sort_values("sep", ascending=False).round(4).to_string())


def buckets(c, q=5):
    x = lg[c]
    try:
        b = pd.qcut(x, q, duplicates="drop")
    except ValueError:
        b = x
    g = lg.groupby(b, observed=True)
    return pd.DataFrame({"n": g.size(), "ts_rate": g.ts.mean(), "avg_ret": g.ret.mean(), "sum_contrib": g.contrib.sum(),
                         "ts_contrib": g.apply(lambda d: d.contrib[d.ts == 1].sum(), include_groups=False)})


for c in A.sort_values("sep", ascending=False).index[:14]:
    print(f"\n--- {c} (quintiles, 2021-23 trades)")
    print(buckets(c).round(4).to_string())

print("\n--- coins (2021-23)")
g = lg.groupby("coin")
print(pd.DataFrame({"n": g.size(), "ts": g.ts.sum(), "ts_rate": g.ts.mean(), "avg": g.ret.mean(),
                    "contrib": g.contrib.sum()}).sort_values("avg").round(4).to_string())
print("\n--- year")
g = lg.groupby(lg.bought.dt.year)
print(pd.DataFrame({"n": g.size(), "ts": g.ts.sum(), "ts_rate": g.ts.mean(), "avg": g.ret.mean(),
                    "contrib": g.contrib.sum()}).round(4).to_string())
