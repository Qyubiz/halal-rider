"""m10 regime: anatomy of 90% mode's losing trades in every period (diagnostic; explains the ceiling of the angle).

Each loser is classed by the market regime at its entry hour (known at the open):
  CRASH  at least 30% of the 19 coins >= 10% under their 24h high (brd_dip10 >= 0.3), or BTC 24h <= -7%, or within
         72h after such an hour (since_brd30 <= 72 or since_b7 <= 72)
  WEAK   not CRASH, but >= 80% of the coins under their 7-day average (brd_sma7 >= 0.8)
  SOLO   the rest: the coin dumped on its own in a calm market
Also: the same split for winners, and each class's share of the money lost (sum of loss as % of the account).

Output: $PB_OUT/anatomy.log
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m10_regime_lib as L  # noqa: E402

PERS = (("2018-20", "h", "2018-01-01", "2021-01-01"), ("2021-23", "d", "2021-01-01", "2024-01-01"),
        ("2024-26", "d", "2024-01-01", "2100-01-01"), ("since25", "d", "2025-01-01", "2100-01-01"))


def classify(which, lg, idx):
    syms, _, P = L.market(which)
    eb = lg.eb.astype(int).values
    f = {n: L.aligned(n, P)[eb] for n in ("brd_dip10", "btc_r24", "since_brd30", "since_b7", "brd_sma7")}
    crash = (f["brd_dip10"] >= 0.3) | (f["btc_r24"] <= -0.07) | (f["since_brd30"] <= 72) | (f["since_b7"] <= 72)
    weak = ~crash & (f["brd_sma7"] >= 0.8)
    cls = np.where(crash, "CRASH", np.where(weak, "WEAK", "SOLO"))
    return pd.DataFrame({"t": idx[eb], "sym": [syms[k] for k in lg.k.astype(int)], "ret": lg.ret.values,
                         "acct": (lg.ret * lg.amt / lg.eq0).values, "cls": cls})


def main():
    log = open(os.path.join(L.OUT, "anatomy.log"), "w")

    def p(s):
        print(s, flush=True)
        log.write(s + "\n")

    T = {}
    for which in ("h", "d"):
        _, e, lg, idx = L.run(which)
        T[which] = classify(which, lg, idx)
    p("90% mode losers by market regime at entry (CRASH / WEAK / SOLO, see docstring)")
    p(f"{'period':8s} {'trades':>6s} {'losers':>6s} | {'losers CRASH/WEAK/SOLO':>24s} | {'loss % acct CRASH/WEAK/SOLO':>30s} |"
      f" {'win rate in CRASH/WEAK/SOLO':>28s} | {'share of trades':>18s}")
    for nm, w, a, b in PERS:
        X = T[w][(T[w].t >= a) & (T[w].t < b)]
        Lr = X[X.ret <= 0]
        cnt = [int((Lr.cls == c).sum()) for c in ("CRASH", "WEAK", "SOLO")]
        loss = [float(-np.log1p(Lr.acct[Lr.cls == c]).sum()) * 100 for c in ("CRASH", "WEAK", "SOLO")]
        wr = [float((X.ret[X.cls == c] > 0).mean()) * 100 if (X.cls == c).any() else np.nan for c in ("CRASH", "WEAK", "SOLO")]
        sh = [float((X.cls == c).mean()) * 100 for c in ("CRASH", "WEAK", "SOLO")]
        p(f"{nm:8s} {len(X):6d} {len(Lr):6d} | {cnt[0]:7d} {cnt[1]:7d} {cnt[2]:7d} | {loss[0]:9.0f} {loss[1]:9.0f} {loss[2]:9.0f}  |"
          f" {wr[0]:8.1f} {wr[1]:8.1f} {wr[2]:8.1f} | {sh[0]:5.0f} {sh[1]:5.0f} {sh[2]:5.0f}")
    p("(loss % acct = sum of -log(1 + loss as a share of the account), in log-points x 100)")
    p("\nEvery loser (period, entry, coin, trade %, account %, class):")
    for nm, w, a, b in PERS[:3]:
        X = T[w][(T[w].t >= a) & (T[w].t < b)]
        for _, r in X[X.ret <= 0].iterrows():
            p(f"  {nm} {r.t:%Y-%m-%d %H:%M} {r.sym[:-4]:7s} {r.ret*100:+6.1f}% acct {r.acct*100:+6.1f}% {r.cls}")


if __name__ == "__main__":
    main()
