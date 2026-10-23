"""Round 2, lens "entry": entry-filter features, all known at the start of hour t (closed candles only).

  pre3d    the coin's 3-day return BEFORE the last 24h: close[t-25] / close[t-97] - 1  ("already falling before the dip")
  r1h      the last closed hour's return: close[t-1] / close[t-2] - 1                ("<= 0: a cascade in progress")
  breadth  share of live coins whose last close is under their 7-day average close     ("most of the market is weak")

Motivation (2021-23 trades of the live bot only, r2_entry_analysis*.py): buys on coins with pre3d <= -3% while
breadth >= 0.9 ("knife", 455 of 1,681 trades) averaged +0.01% (2021) and +0.15% (2022-23), time-stop rate 11%/10%,
against +1.5%/+1.0% for the rest; among them, those whose last hour was up averaged -2.7%/-0.7% (time stops 23%/16%).
"""
import numpy as np


def feats(P):
    cf = P["cf"]
    c1 = cf.shift(1)
    pre3d = cf.shift(25) / cf.shift(97) - 1
    r1h = c1 / cf.shift(2) - 1
    live = P["c"].shift(1).rolling(24, min_periods=1).count() > 0   # a real close in the last 24h: outage hours
                                                                     # keep the last prices, cut coins drop out
    below = (c1 < cf.rolling(168, min_periods=84).mean().shift(1)) & live
    breadth = below.sum(axis=1) / live.sum(axis=1).clip(lower=1)
    return pre3d.values, r1h.values, breadth.values


def flag(P, pre=None, br=None, r1=None):
    """Boolean (n,K): pre3d <= -pre (if given) & breadth >= br (if given) & r1h > r1 (if given). NaN -> not flagged."""
    pre3d, r1h, breadth = feats(P)
    f = np.ones(pre3d.shape, bool)
    with np.errstate(invalid="ignore"):
        if pre is not None:
            f &= pre3d <= -pre
        if br is not None:
            f &= (breadth >= br)[:, None]
        if r1 is not None:
            f &= r1h > r1
    return f


def act(f, how):
    """'skip' -> smult 0 on flagged coins/hours; a float -> that trigger depth on flagged coins/hours (live 12%)."""
    if how == "skip":
        return {"smult": np.where(f, 0.0, 1.0)}
    return {"depth": np.where(f, float(how), 0.12)}


def variant(how, **cond):
    def spec(P, syms):
        return act(flag(P, **cond), how)
    return spec
