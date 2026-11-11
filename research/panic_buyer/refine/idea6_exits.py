"""Idea 6: exit variants.
  partial    sell `part` at +1.5%, the rest on a trailing drop from its high or at a second target (48h time stop kept)
  breadth    shorter time stop while the market is weak: share of coins below their 7-day average >= thr
  btc-weak   shorter time stop while BTC is below its 7-day average"""
import numpy as np

import study as S


def partial(part, trail=0.0, tp2=0.0):
    return {"part": part, "trail": trail, "tp2": tp2}


def weak_breadth(thr, hm_short):
    def f(P, syms):
        below = (P["cf"] < P["cf"].rolling(168, min_periods=84).mean()).shift(1, fill_value=False)
        live = P["cf"].shift(1).notna()
        share = (below & live).sum(axis=1) / live.sum(axis=1).clip(lower=1)
        return {"hm": np.where(share.values >= thr, hm_short, 48)}
    return f


def weak_btc(hm_short):
    def f(P, syms):
        b = P["cf"]["BTCUSDT"]
        weak = (b < b.rolling(168, min_periods=84).mean()).shift(1, fill_value=False).values
        return {"hm": np.where(weak, hm_short, 48)}
    return f


V = {}
for part in (0.5, 0.75):
    for trail in (0.02, 0.04):
        V[f"partial {part:.0%} + trail {trail:.0%}"] = partial(part, trail=trail)
    for tp2 in (0.03, 0.05):
        V[f"partial {part:.0%} + 2nd target {tp2:.0%}"] = partial(part, tp2=tp2)
for thr in (0.7, 0.85):
    for hs in (12, 24):
        V[f"breadth>={thr:.0%} -> {hs}h stop"] = weak_breadth(thr, hs)
for hs in (12, 24):
    V[f"BTC<7d avg -> {hs}h stop"] = weak_btc(hs)

if __name__ == "__main__":
    R, v = S.study("idea 6: exit variants", V, "idea6_exits")
    S.save_verdict(v)
