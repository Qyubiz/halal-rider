"""m10 meanrev, stage 1: which intraday oversold entries have an edge? 2021-23 ONLY (bars after 2023 are cut off).

Entry families on 5-minute (and 15-minute) bars, all computed from closed bars:
  DIP  W,d : resting bid at highest close of the last W hours * (1-d)
  BB   n,k : resting bid at SMA_n - k*STD_n (Bollinger lower band, n bars)
  KC   n,k : resting bid at EMA_n - k*ATR_n (Keltner lower band)
  SIG      : market buy at the open after a closed-bar signal (RSI(2/14) extreme, z-score, red-candle run + drop)
Exits: fixed take-profit tp with time stop hm hours, or the moving mean (the family's SMA/EMA) floored at +0.3%.
Regime: the 90% mode crash guard (skip a coin whose last close is >35% under its 7-day high) on every design here;
other regime filters are stage 2.
Sizing: 'ind' = every coin on its own (19 bids x 1/19 of equity: the signal's quality) and 'all' = one coin with all
the money (the 90% mode sizing).
Output: OUT/grid1_<C>m.csv (2021-23 metrics only).
"""
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m10_meanrev_lib as L  # noqa: E402

C = int(sys.argv[1]) if len(sys.argv) > 1 else 5
B = L.cut(L.load("main", C=C), L.T21, L.T24)
hb = lambda x: L.bars_per(B, x)   # noqa: E731
cfs = L.sh(B["cf"])
guard = cfs >= L.rollmax(B, hb(24 * 7)) * (1 - 0.35)

entries = {}


def add(name, fn):
    entries[name] = fn


for W, d in itertools.product((1, 4, 12, 24), (0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12)):
    if W <= 4 and d > 0.08:
        continue
    add(f"DIP W{W}h d{d*100:.0f}", lambda W=W, d=d: (L.rollmax(B, hb(W)) * (1 - d), None))
for n, k in itertools.product((20, 48, 144, 288), (2.5, 3, 3.5, 4, 5)):
    add(f"BB n{n} k{k}", lambda n=n, k=k: (lambda m, s: (m - k * s, m))(L.sma(B, n), L.std(B, n)))
for n, k in itertools.product((20, 48, 144), (3, 4, 5, 6, 8)):
    add(f"KC n{n} k{k}", lambda n=n, k=k: (lambda m, a: (m - k * a, m))(L.ema(B, n), L.atr(B, n)))
INF = np.inf


def sig(mask):
    return np.where(mask, INF, np.nan)


for x in (2, 5, 10):
    add(f"SIG rsi2<{x}", lambda x=x: (sig(L.rsi(B, 2) < x), None))
for x in (15, 20, 25):
    add(f"SIG rsi14<{x}", lambda x=x: (sig(L.rsi(B, 14) < x), None))
for n, z in itertools.product((48, 288), (3, 4, 5)):
    add(f"SIG z{n}<-{z}", lambda n=n, z=z: (sig((cfs - L.sma(B, n)) / L.std(B, n) < -z), L.sma(B, n)))
for r, d in itertools.product((4, 6, 8), (0.02, 0.04)):
    add(f"SIG red{r}+drop{d*100:.0f}", lambda r=r, d=d: (sig((L.red_run(B) >= r) & (L.ret_over(B, r) < -d)), None))

EXITS = [("tp", tp, h) for tp in (0.005, 0.008, 0.012, 0.02, 0.03) for h in (6, 24, 96)] + \
        [("mean", 0.003, h) for h in (6, 24, 96)]
SIZES = {"ind": dict(nord=19, fmax=1 / 19, maxpos=19), "all": dict(nord=1, fmax=1.0, maxpos=1)}

rows = []
t0 = time.time()
for name, fn in entries.items():
    lim, mean = fn()
    lim = np.where(guard, lim, np.nan)
    for (kind, tp, h), (sz, kw) in itertools.product(EXITS, SIZES.items()):
        if kind == "mean" and mean is None:
            continue
        eq, lg = L.run(B, lim, tp=tp, tpmax=0.10, tgt=mean if kind == "mean" else None, hm=hb(h), cd=hb(1), **kw)
        r = L.stats(B, eq, lg, L.T21, L.T24)
        y = L.yearly(B, eq)
        rows.append({"entry": name, "exit": kind, "tp": tp, "hm": h, "size": sz, **r, "y2022": y.get(2022, np.nan)})
    print(f"{name:22s} done ({time.time()-t0:.0f}s)", flush=True)
    del lim, mean
df = pd.DataFrame(rows)
df.to_csv(os.path.join(L.OUT, f"grid1_{C}m.csv"), index=False)
pd.set_option("display.width", 250)
cols = ["entry", "exit", "tp", "hm", "size", "mo", "dd", "win", "avg", "tpm", "worst", "worst_acct", "y2022"]
for sz in ("ind", "all"):
    d = df[df["size"] == sz]
    print(f"\n=== {sz}: top 25 by %/month, win >= 90%")
    print(d[d.win >= 0.90].sort_values("mo", ascending=False)[cols].head(25).to_string(index=False))
    print(f"\n=== {sz}: top 15 by win rate, >= 10 trades/month")
    print(d[d.tpm >= 10].sort_values("win", ascending=False)[cols].head(15).to_string(index=False))
