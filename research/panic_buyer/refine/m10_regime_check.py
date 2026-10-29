"""m10 regime, step 1: reproduce 90% mode (check90.log / holdout_w90.log) through m10_regime_lib, check that an
all-ones mask changes nothing, and write every trade with its regime features at the entry hour (for the 2021-23
loser anatomy; 2024-26 and holdout trades are written too but NOT used to choose filters).

Output: $PB_OUT/check.log, $PB_OUT/trades_<panel>.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m10_regime_lib as L  # noqa: E402
import w90_wr90_lib as W  # noqa: E402

FEATS = ["c_dip24", "c_r1", "c_r4", "c_r24", "c_pre3d", "c_pre7d", "c_r7d", "c_r30d", "c_dd30", "c_sma7", "c_sma30",
         "c_rv", "c_qv", "btc_r1", "btc_r4", "btc_r12", "btc_r24", "btc_r72", "btc_r168", "btc_dd7", "btc_dd30",
         "btc_sma50", "btc_sma200", "btc_rv", "btc_rv72", "btc_qv", "mkt_qv", "brd_dip5", "brd_dip8", "brd_dip10",
         "brd_dip14", "brd_r24_5", "brd_r24_10", "brd_r24_15", "brd_sma7", "brd_sma30", "mkt_r24", "mkt_r72",
         "mkt_r7d", "mkt_rv", "btc_red", "mkt_red", "since_b10", "since_b7", "since_brd50", "since_brd30"]


def trades_with_feats(which, lg, idx):
    syms, _, P = L.market(which)
    eb = lg.eb.astype(int).values
    k = lg.k.astype(int).values
    T = pd.DataFrame({"entry_time": idx[eb], "exit_time": idx[lg.xb.astype(int).values],
                      "sym": [syms[j] for j in k], "ret": lg.ret.values, "why": lg.why.values,
                      "acct": (lg.ret * lg.amt / lg.eq0).values, "hold": (lg.xb - lg.eb).values})
    for f in FEATS:
        a = L.aligned(f, P)
        T[f] = a[eb, k] if a.ndim == 2 else a[eb]
    return T


def main():
    t0 = time.time()
    log = open(os.path.join(L.OUT, "check.log"), "w")

    def p(s):
        print(s, flush=True)
        log.write(s + "\n")

    p("90% mode = " + str(L.MODE90))
    for which in ("d", "h"):
        r, e, lg, idx = L.run(which)
        p(f"[{which}] normal          " + L.fmt(r))
        syms, _, P = L.market(which)
        r1, e1, lg1, _ = L.run(which, rules=[[("c_dip24", ">=", 9.0)]])      # never true -> all-ones mask
        same = np.array_equal(e.values, e1.values) and len(lg) == len(lg1)
        p(f"[{which}] never-true rule gives identical equity: {same}")
        rs, _, _, _ = L.run(which, **L.stress_kw())
        p(f"[{which}] fills 0.2% worse " + L.fmt(rs))
        T = trades_with_feats(which, lg, idx)
        T.to_csv(os.path.join(L.OUT, f"trades_{which}.csv"), index=False)
        # cross-check against w90_wr90_lib.evaluate (2021-26 only)
        if which == "d":
            rw, _, _ = W.evaluate(dict(L.MODE90))
            p(f"[d] w90_wr90_lib.evaluate: WR {rw['d_wr']*100:.1f}/{rw['t_wr']*100:.1f}/{rw['s_wr']*100:.1f} "
              f"%/mo {rw['d_mo']*100:+.2f}/{rw['t_mo']*100:+.2f}/{rw['s_mo']*100:+.2f} "
              f"DD {rw['d_dd']*100:.0f}/{rw['t_dd']*100:.0f}/{rw['s_dd']*100:.0f}")
            rx, _, _, _ = L.run("dx", **L.bot_kw())
            p("[dx] bot timing + fills 0.2% worse + LUNA/FTT/SRM " + L.fmt(rx))
    p(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
