"""Signal study, 2021-23 ONLY (the design period; later periods are not printed or saved).

Every trigger rule is traded with every signal taken: no capital limit (19 slots x 1/19 of the account), one position
per coin, crash guard 35% / 168h, cooldown 3h, all costs. Per rule: trades per month, win rate, average trade, and the
per-trade log growth if the whole account went into each trade, g = mean(log(1 + ret)), with n x g = the monthly log
growth an all-in bot would get if it could take every one of these trades in turn (an upper bound: trades overlap).

Rules: reference = highest close of the last W closed hours (W = 3, 4, 6, 12 = flash dips; 24 = the live trigger;
48), depth, take-profit, time stop 48h / 96h.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m10_velocity_lib as V  # noqa: E402

DEPTHS = {3: [0.05, 0.06, 0.07, 0.08, 0.10, 0.12], 4: [0.05, 0.06, 0.07, 0.08, 0.10, 0.12],
          6: [0.06, 0.07, 0.08, 0.10, 0.12, 0.14], 12: [0.08, 0.10, 0.12, 0.14],
          24: [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16], 48: [0.14, 0.16, 0.18]}
TPS = [0.015, 0.02, 0.025, 0.03]
HMS = [48, 96]


def main():
    syms, idx, P = V.market(False)
    rows = []
    for W, ds in DEPTHS.items():
        for d in ds:
            for tp in TPS:
                for hm in HMS:
                    tier = dict(ref=W, depth=d, tp=tp, hm=hm, cdw=3, cdl=3, fs=1 / 19)
                    e, lg, inv = V.vrun(P, idx, tiers=[tier], nslots=99, excl=1, guard=0.35)
                    tb = idx[lg.eb.astype(int).values]
                    x = lg[(tb >= V.T21) & (tb < V.T24)]
                    rows.append(dict(W=W, depth=d, tp=tp, hm=hm, n_mo=len(x) / 36.0, wr=(x.ret > 0).mean(),
                                     avg=x.ret.mean(), avgl=x.ret[x.ret <= 0].mean(), worst=x.ret.min(),
                                     g=np.log1p(x.ret).mean(), mg=len(x) / 36.0 * np.log1p(x.ret).mean()))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(V.OUT, "signals_2021_23.csv"), index=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)
    for W in DEPTHS:
        print(f"\n=== reference = {W}h high; 2021-23, every signal taken ===")
        x = R[R.W == W].copy()
        x["wr"] = (x.wr * 100).round(1); x["avg"] = (x.avg * 100).round(2); x["avgl"] = (x.avgl * 100).round(1)
        x["worst"] = (x.worst * 100).round(0); x["g"] = (x.g * 100).round(2); x["mg"] = (x.mg * 100).round(1)
        x["n_mo"] = x.n_mo.round(1)
        print(x.to_string(index=False))
    print("\nRules with win rate >= 92%, ranked by n x g (all-in monthly log growth bound, %):")
    q = R[R.wr >= 0.92].sort_values("mg", ascending=False).head(40)
    print(q.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
