"""m10 grid, design search on 2021-23 ONLY (the simulation stops at 2024-01-01, later candles are never read).

Usage: python3 m10_grid_search.py STAGE      (stage 1: broad map, stage 2: refinement around the stage-1 leaders)
Writes $PB_OUT/search_STAGE.csv and prints the leaders under the pre-registered rule:
    round-trip win rate >= 90% and 2021-23 max drawdown no deeper than -35%; then the best 2021-23 %/month.
"""
import itertools
import os
import sys
import time

import pandas as pd

import m10_grid_lib as G

CSETS = {
    "maj5": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"],
    "maj10": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "TRXUSDT", "LINKUSDT",
              "LTCUSDT"],
    "all19": G.H19,
    "btceth": ["BTCUSDT", "ETHUSDT"],
}
WR_MIN, DD_MIN = 0.90, -0.35


def designs(stage):
    out = []
    if stage == 1:
        exits = [dict(), dict(stop=0.03, cd=24), dict(hm=72), dict(gate="btc100", liq=1),
                 dict(gate="coin50", liq=1), dict(gate="btc100", liq=1, stop=0.03, cd=24)]
        for cs, s, L, d0, ex in itertools.product(["maj5", "maj10", "all19"], [0.005, 0.01, 0.02], [5, 10, 20],
                                                 [0.0, 0.05], exits):
            d = dict(cset=cs, s=s, L=L, off=int(round(d0 / s)), gate="none")
            d.update(ex)
            out.append(d)
    if stage == 2:
        # sleeves around the stage-1 leaders (2% steps, regime exit), wider steps, deeper first level, 2-step target
        for s, L, d0, tpk, g in itertools.product([0.02, 0.03, 0.04], [5, 10, 20], [0.0, 0.04, 0.08], [1, 2],
                                                 ["btc100", "coin50", "btc100&coin50"]):
            out.append(dict(cset="maj5", s=s, L=L, off=int(round(d0 / s)), tpk=tpk, gate=g, liq=1))
        # one shared cash pool: lot = equity / Ltot, orders on the nearest levels across coins
        for cs, s, L, Lt, d0, g in itertools.product(["maj5", "maj10", "all19"], [0.02, 0.03], [5, 10], [5, 10, 20],
                                                    [0.0, 0.04], ["btc100", "coin50"]):
            out.append(dict(cset=cs, s=s, L=L, Ltot=Lt, off=int(round(d0 / s)), gate=g, liq=1))
    if stage == 3:
        # around the stage-2 leaders: steps 2% (first level 4% down) or 3% (3% down), 5 or 10 levels
        base = [(0.02, 2), (0.03, 1)]
        for (s, off), L in itertools.product(base, [5, 10]):
            b = dict(cset="maj5", s=s, L=L, off=off)
            for g in ["btc50", "btc100", "btc200", "btc100&coin50", "btc100&coin20", "btc100&r168_10",
                      "btc50&coin50", "btc100&er168_0.3"]:
                out.append(dict(b, gate=g, liq=1))
            for g, hs in itertools.product(["btc100", "btc100&coin50"], [0.10, 0.20]):
                out.append(dict(b, gate=g, liq=0, hs=hs))
            out.append(dict(b, gate="btc100&coin50", liq=1, trail=1))
            out.append(dict(b, gate="btc100&coin50", liq=1, stop=0.10, cd=24))
        for cs, (s, off), Lt in itertools.product(["maj5", "all19"], base, [5, 10]):
            b = dict(cset=cs, s=s, L=5, Ltot=Lt, off=off)
            for g in ["btc100", "btc100&coin50", "btc100&r168_10"]:
                out.append(dict(b, gate=g, liq=1))
            for g in ["btc100", "btc100&coin50"]:
                out.append(dict(b, gate=g, liq=0, hs=0.15))
    if stage == 4:
        # the 200-day BTC filter family found in stage 3
        for s, L, d0, g in itertools.product([0.02, 0.03, 0.04], [5, 10], [0.0, 0.035],
                                            ["btc200", "btc200&coin50", "btc200&coin100"]):
            out.append(dict(cset="maj5", s=s, L=L, off=int(round(d0 / s)), gate=g, liq=1))
        for cs, s, Lt, g in itertools.product(["maj5", "all19"], [0.02, 0.03], [5, 10], ["btc200", "btc200&coin50"]):
            out.append(dict(cset=cs, s=s, L=5, Ltot=Lt, off=int(round(0.035 / s)), gate=g, liq=1))
    return out


def main():
    stage = int(sys.argv[1])
    idx, P = G.market()
    gc = {}
    rows = []
    t = time.time()
    for i, d in enumerate(designs(stage)):
        e, lg = G.run_any(idx, P, CSETS[d["cset"]], d, G.T_D0, G.T_T0, gc)
        r = G.period_stats(e, lg, idx, G.T_D0, G.T_T0)
        y = e.resample("YE").last()
        y = pd.concat([pd.Series([1.0], [e.index[0]]), y]).pct_change().dropna()
        rows.append(dict(name=G.dname(d), **{k: d.get(k) for k in ("cset", "s", "L", "Ltot", "off", "tpk", "gate", "liq",
                                                                  "stop", "cd", "hm", "hs", "trail")},
                         **r, y2021=y.iloc[0], y2022=y.iloc[1], y2023=y.iloc[2]))
    print(f"{len(rows)} designs in {time.time()-t:.0f}s")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(G.OUT, f"search_{stage}.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
    cols = ["name", "mo", "dd", "worst_m", "wr", "rt_m", "avg", "y2021", "y2022", "y2023"]
    ok = df[(df.wr >= WR_MIN) & (df.dd >= DD_MIN)].sort_values("mo", ascending=False)
    print(f"\n{len(ok)} pass the rule (WR>={WR_MIN:.0%}, DD>={DD_MIN:.0%}); top 25 by 2021-23 %/month:")
    print(ok[cols].head(25).to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nTop 15 by 2021-23 %/month, any drawdown / win rate:")
    print(df.sort_values("mo", ascending=False)[cols].head(15).to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nTop 10 by %/month / |drawdown|:")
    df["calmar"] = df.mo / (-df.dd).clip(lower=1e-6)
    print(df.sort_values("calmar", ascending=False)[cols].head(10).to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
