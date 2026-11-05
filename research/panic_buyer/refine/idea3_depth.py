"""Idea 3: adaptive trigger depth per coin instead of a fixed 12%.
  vol   depth = k x daily volatility (last W hours), clipped to 8-25%
  pctl  depth = the 24h drop exceeded only q of the time over the last W hours, clipped to 8-25%
  floor depth = max(12%, k x daily volatility): only deeper when a coin is unusually wild"""
import numpy as np

import study as S


def vol_depth(k, w, lo=0.08, hi=0.25, floor=None):
    def f(P, syms):
        d = (k * S.roll_vol(P, w)).clip(lo, hi)
        if floor is not None:
            d = np.maximum(d, floor)
        return {"depth": d.fillna(0.12).values}
    return f


def pctl_depth(q, w, lo=0.08, hi=0.25):
    def f(P, syms):
        dd = S.dd24(P)
        d = (-dd.rolling(w, min_periods=w // 2).quantile(q)).clip(lo, hi)
        return {"depth": d.fillna(0.12).values}
    return f


V = {}
for w in (168, 720, 2160):
    for k in (2.0, 2.5, 3.0, 3.5, 4.0, 5.0):
        V[f"vol k={k} W={w // 24}d"] = vol_depth(k, w)
for w in (720, 2160, 4320):
    for q in (0.005, 0.01, 0.02, 0.05):
        V[f"pctl q={q:.1%} W={w // 24}d"] = pctl_depth(q, w)
for w in (168, 720):
    for k in (2.5, 3.0, 3.5, 4.0, 5.0):
        V[f"floor12 k={k} W={w // 24}d"] = vol_depth(k, w, floor=0.12)

if __name__ == "__main__":
    R, v = S.study("idea 3: adaptive trigger depth", V, "idea3_depth")
    S.save_verdict(v)
