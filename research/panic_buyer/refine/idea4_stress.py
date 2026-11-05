"""Idea 4: stress-aware sizing. 'Stress' = BTC's last close at least x under its 24h high (known at the hour's start,
like everything the live bot uses).
  calm-size   in calm hours other coins get m x the normal order (0 = no order), BTC always normal
  calm-deeper in calm hours other coins' trigger is 12% + extra
  stress-N    in stress hours the bot spreads its cash over N orders (fewer = bigger orders), max 25% each"""
import numpy as np

import study as S


def calm_size(x, m):
    def f(P, syms):
        calm = (S.btc_dd24(P) > -x).values
        sm = np.ones(P["o"].shape)
        sm[calm] = m
        sm[:, syms.index("BTCUSDT")] = 1.0
        return {"smult": sm}
    return f


def calm_deeper(x, extra):
    def f(P, syms):
        calm = (S.btc_dd24(P) > -x).values
        d = np.full(P["o"].shape, 0.12)
        d[calm] += extra
        d[:, syms.index("BTCUSDT")] = 0.12
        return {"depth": d}
    return f


def stress_n(x, n_stress, fmax_stress=0.25):
    def f(P, syms):
        st = (S.btc_dd24(P) <= -x).values
        return {"N": np.where(st, n_stress, 8), "fmax": np.where(st, fmax_stress, 0.20)}
    return f


V = {}
for x in (0.02, 0.03, 0.05):
    for m in (0.0, 0.5):
        V[f"calm-size x={x:.0%} m={m}"] = calm_size(x, m)
for x in (0.02, 0.03, 0.05):
    for extra in (0.02, 0.04):
        V[f"calm-deeper x={x:.0%} +{extra:.0%}"] = calm_deeper(x, extra)
for x in (0.03, 0.05):
    for n in (4, 5, 12, 19):
        V[f"stress-N x={x:.0%} N={n}"] = stress_n(x, n)

if __name__ == "__main__":
    R, v = S.study("idea 4: stress-aware sizing", V, "idea4_stress")
    S.save_verdict(v)
