"""
Backtest: long-only spot, 15m candles. Enter only when >= N of 12 indicators are bullish
at the candle close. Entry at next candle open. SL/TP from price-action support/resistance.
Fees + slippage included. Conservative intrabar rule: if SL and TP both touch, SL fills.
"""
import sys, glob, os, json
import numpy as np, pandas as pd

FEE = 0.001        # Binance spot 0.10% per side (no BNB discount assumed)
SLIP = 0.0002      # 0.02% slippage per side
COST = 2 * (FEE + SLIP)
MIN_NET_TP = 0.004 # skip if resistance target is less than +0.4% away (must beat costs)
SL_LOOKBACK = 20   # support = lowest low of last 20 bars
TP_LOOKBACK = 48   # resistance = highest high of last 48 bars (12h)
MAX_HOLD = 48      # bars
SL_MIN, SL_MAX = 0.004, 0.025

def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rma(s, n): return s.ewm(alpha=1/n, adjust=False).mean()

def indicators(df):
    c, h, l, v, o = df.c, df.h, df.l, df.v, df.o
    out = pd.DataFrame(index=df.index)
    # 1-2 EMA trend stack
    e9, e21, e50 = ema(c, 9), ema(c, 21), ema(c, 50)
    out["ema9>21"] = e9 > e21
    out["ema21>50"] = e21 > e50
    # 3 close above daily-reset VWAP
    tp = (h + l + c) / 3
    day = df.index.date
    vw = (tp * v).groupby(day).cumsum() / v.groupby(day).cumsum()
    out["c>vwap"] = c > vw
    # 4 RSI 50-70
    d = c.diff()
    rsi = 100 - 100 / (1 + rma(d.clip(lower=0), 14) / rma((-d).clip(lower=0), 14))
    out["rsi50-70"] = (rsi > 50) & (rsi < 70)
    # 5-6 MACD
    macd = ema(c, 12) - ema(c, 26); sig = ema(macd, 9); hist = macd - sig
    out["macd>sig"] = macd > sig
    out["hist rising"] = hist > hist.shift(1)
    # 7 Stochastic
    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    k = 100 * (c - ll) / (hh - ll); dd = k.rolling(3).mean(); k = k.rolling(3).mean()
    out["stoch k>d<80"] = (k > dd) & (k < 80)
    # 8 ADX / DI
    up, dn = h.diff(), -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0); mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = rma(tr, 14)
    pdi = 100 * rma(pd.Series(pdm, index=df.index), 14) / atr
    mdi = 100 * rma(pd.Series(mdm, index=df.index), 14) / atr
    adx = rma(100 * (pdi - mdi).abs() / (pdi + mdi), 14)
    out["adx>20&+DI"] = (adx > 20) & (pdi > mdi)
    # 9 Bollinger: above mid, below upper
    m, sd = c.rolling(20).mean(), c.rolling(20).std()
    out["bb mid-upper"] = (c > m) & (c < m + 2 * sd)
    # 10 OBV above its EMA
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    out["obv>ema"] = obv > ema(obv, 20)
    # 11 CCI
    tpm = tp.rolling(20).mean(); mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - tpm) / (0.015 * mad)
    out["cci>0"] = cci > 0
    # 12 Supertrend(10,3)
    n, mult = 10, 3.0
    a = rma(tr, n).values; hl2 = ((h + l) / 2).values; cv = c.values
    ub, lb = hl2 + mult * a, hl2 - mult * a
    fub, flb = ub.copy(), lb.copy(); trend = np.ones(len(cv), dtype=bool)
    for i in range(1, len(cv)):
        fub[i] = ub[i] if (ub[i] < fub[i-1] or cv[i-1] > fub[i-1]) else fub[i-1]
        flb[i] = lb[i] if (lb[i] > flb[i-1] or cv[i-1] < flb[i-1]) else flb[i-1]
        if trend[i-1] and cv[i] < flb[i]: trend[i] = False
        elif (not trend[i-1]) and cv[i] > fub[i]: trend[i] = True
        else: trend[i] = trend[i-1]
    out["supertrend"] = trend
    out = out.fillna(False).astype(bool)
    out.iloc[:60] = False   # warm-up
    return out

def run(df, score, thresh, rng=None, random_entry_rate=None):
    o, h, l, c = df.o.values, df.h.values, df.l.values, df.c.values
    lowN = df.l.rolling(SL_LOOKBACK).min().values
    highN = df.h.rolling(TP_LOOKBACK).max().values
    sc = score.values
    trades, i, n = [], 61, len(df)
    while i < n - 2:
        if random_entry_rate is not None:
            go = rng.random() < random_entry_rate
        else:
            go = sc[i] >= thresh
        if not go: i += 1; continue
        entry = o[i + 1]
        sl = max(lowN[i] * 0.999, entry * (1 - SL_MAX)); sl = min(sl, entry * (1 - SL_MIN))
        tp = highN[i]
        if (tp - entry) / entry < MIN_NET_TP:     # resistance too close / already above -> skip
            i += 1; continue
        exit_px, reason, j = None, None, i + 1
        for j in range(i + 1, min(i + 1 + MAX_HOLD, n)):
            if l[j] <= sl: exit_px, reason = sl, "SL"; break      # SL first if both touch
            if h[j] >= tp: exit_px, reason = tp, "TP"; break
        if exit_px is None:
            j = min(i + MAX_HOLD, n - 1); exit_px, reason = c[j], "TIME"
        ret = exit_px / entry - 1 - COST
        trades.append((df.index[i + 1], df.index[j], entry, exit_px, ret, reason, (j - i)))
        i = j + 1
    return pd.DataFrame(trades, columns=["entry_t", "exit_t", "entry", "exit", "ret", "reason", "bars"])

def stats(t):
    if len(t) == 0: return dict(trades=0, win=0, avg=0, pf=0, total=0, maxdd=0)
    r = t.ret.values; w = r[r > 0].sum(); ls = -r[r <= 0].sum()
    eq = np.cumsum(r); dd = (eq - np.maximum.accumulate(eq)).min()
    return dict(trades=len(r), win=round((r > 0).mean() * 100, 1), avg=round(r.mean() * 100, 3),
                pf=round(w / ls, 2) if ls > 0 else 99.0, total=round(r.sum() * 100, 1), maxdd=round(dd * 100, 1))

if __name__ == "__main__":
    data_dir = sys.argv[1]
    data = {os.path.basename(p)[:-4]: pd.read_csv(p, index_col=0, parse_dates=True) for p in sorted(glob.glob(f"{data_dir}/*.csv"))}
    ind = {s: indicators(d) for s, d in data.items()}
    scores = {s: ind[s].sum(axis=1) for s in data}
    names = list(next(iter(ind.values())).columns)
    mid = next(iter(data.values())).index[len(next(iter(data.values()))) // 2]
    res = {"indicators": names}

    # threshold sweep (all coins, full period + in-sample / out-of-sample halves)
    sweep = []
    for th in range(6, 13):
        allt = pd.concat([run(data[s], scores[s], th).assign(sym=s) for s in data])
        row = {"min_indicators": th, "ALL": stats(allt),
               "first_half": stats(allt[allt.entry_t < mid]), "second_half": stats(allt[allt.entry_t >= mid])}
        sweep.append(row)
    res["sweep"] = sweep

    # main config: >=10 of 12, per coin
    main = pd.concat([run(data[s], scores[s], 10).assign(sym=s) for s in data])
    res["main_per_coin"] = {s: stats(main[main.sym == s]) for s in data}
    res["main_exit_reasons"] = main.reason.value_counts().to_dict()
    res["main_avg_bars"] = round(main.bars.mean(), 1) if len(main) else 0
    main.to_csv("trades_main_ge10.csv", index=False)

    # signal frequency
    res["signal_bars_pct_ge10"] = {s: round((scores[s] >= 10).mean() * 100, 2) for s in data}

    # random-entry baseline with same exits and same trade count (50 seeds)
    base = []
    for seed in range(50):
        rng = np.random.default_rng(seed)
        rate = max(len(main) / sum(len(d) for d in data.values()) * 3, 0.002)
        bt = pd.concat([run(data[s], scores[s], 0, rng=rng, random_entry_rate=rate) for s in data])
        base.append(stats(bt))
    b = pd.DataFrame(base)
    res["random_baseline_mean"] = {k: round(float(b[k].mean()), 3) for k in b.columns}
    res["random_baseline_win_range"] = [float(b.win.min()), float(b.win.max())]

    # which indicators matter: win rate when each is True among trades >=10? (approx: avg score contribution)
    json.dump(res, open("results.json", "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))
