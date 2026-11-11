"""m10 meanrev, final report (run once): the stage-2 pick (m10_meanrev_grid2.py, chosen on 2021-23 only) on 2021-23,
2024-26, since Jan 2025 and the untouched 2018-20 holdout (16 coins that existed), with normal fills and with every
fill (buys and sells) 0.1% worse. Reference: the 90% mode (1 coin x 100%, 14% under the 24h high, +2.5%, 96h, cd 3h,
crash guard 35%) through the same kernel on hourly bars built from the same minutes (it reproduces
+13.17 / +6.89 / +5.80 %/month, win 93.8 / 92.6 / 93.2%).

Pick: 5-minute checks; resting bid at (highest 5m close of the last hour) x (1 - 8%); skip a coin whose last close is
>35% under its 7-day high or >15% under its 24h high; best 1 coin (closest to its bid) with all the money; take-profit
+1.2% placed at the next 5-minute check after the fill; time stop 6h (sell at market); 1h cooldown per coin.
Drops: "dd" on 5-minute closes, "dd_h" on hourly closes (the 90% mode's figures are hourly).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m10_meanrev_lib as L  # noqa: E402

PERIODS = {"main": [("2021-23", L.T21, L.T24), ("2024-26", L.T24, L.TEND), ("since 2025", L.T25, L.TEND)],
           "hold": [("2018-20 holdout", L.H18, L.H21)]}


def pick(B):
    hb = lambda x: L.bars_per(B, x)   # noqa: E731
    cfs = L.sh(B["cf"])
    ok = (cfs >= L.rollmax(B, hb(24 * 7)) * 0.65) & (cfs >= L.rollmax(B, hb(24)) * 0.85)
    lim = np.where(ok, L.rollmax(B, hb(1)) * 0.92, np.nan)
    return lim, dict(nord=1, fmax=1.0, maxpos=1, tp=0.012, hm=hb(6), cd=hb(1))


def mode90(B):
    hb = lambda x: L.bars_per(B, x)   # noqa: E731
    cfs = L.sh(B["cf"])
    lim = np.where(cfs >= L.rollmax(B, hb(24 * 7)) * 0.65, L.rollmax(B, hb(24)) * 0.86, np.nan)
    return lim, dict(nord=1, fmax=1.0, maxpos=1, tp=0.025, hm=hb(96), cd=hb(3))


rows = []
for which in ("main", "hold"):
    for name, fn, C in (("PICK meanrev 5m", pick, 5), ("90% mode (hourly)", mode90, 60)):
        B = L.load(which, C=C)
        lim, kw = fn(B)
        for fills, pen in (("normal", 0.0), ("0.1% worse", 0.001)):
            eq, lg = L.run(B, lim, pen=pen, pens=pen, **kw)
            ys = L.yearly(B, eq)
            for per, a, b in PERIODS[which]:
                r = L.stats(B, eq, lg, a, b)
                rows.append({"design": name, "fills": fills, "period": per, **r})
                print(f"{name:18s} {fills:10s} {L.fmt(per, r)} dd_h {r['dd_h']*100:5.1f}%", flush=True)
            yr = {k: v for k, v in ys.items() if (which == "hold" and 2018 <= k <= 2020) or (which == "main" and k >= 2021)}
            print(f"{'':29s} years: " + "  ".join(f"{k} {v*100:+.0f}%" for k, v in yr.items()), flush=True)
            if name.startswith("PICK") and fills == "normal":
                t = B["t"]
                d = pd.DataFrame(lg, columns=L.LOG)
                d["sym"] = [B["syms"][int(k)] for k in d.k]
                d["t_in"] = t[d.eb.astype(int).values]
                d["acct"] = d.ret * d.amt / d.eq0
                d.to_csv(os.path.join(L.OUT, f"final_trades_{which}.csv"), index=False)
                print(f"{'':29s} worst 5 trades: " + "; ".join(f"{r.sym[:-4]} {r.t_in:%Y-%m-%d %H:%M} {r.acct*100:+.1f}%"
                                                              for r in d.nsmallest(5, "acct").itertuples()), flush=True)
                m = pd.Series(eq, t).resample("ME").last()
                mr = m / np.concatenate([[1.0], m.values[:-1]]) - 1
                for per, a, b in PERIODS[which]:
                    x = mr[(mr.index >= a) & (mr.index < b)]
                    print(f"{'':29s} {per}: months up {(x > 0).mean()*100:.0f}%, worst month {x.min()*100:+.1f}% "
                          f"({x.idxmin():%Y-%m}), median {x.median()*100:+.1f}%", flush=True)
        del B, lim
pd.DataFrame(rows).to_csv(os.path.join(L.OUT, "final.csv"), index=False)
