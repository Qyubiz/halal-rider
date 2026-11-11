"""m11 confluence: the ONE look. The pick was fixed by m11_confluence_grid.py's rule on 2021-23 only:
    PICK = kA4@0.20 tpScaled>=5 noB norearm   (key (4, 0.2, 5, None, None, None))
In words: score = how many of 6 votes agree (coin 7-day volatility > 8%; last hourly close down; 24h range >= 20%;
fewer than 25% of the 19 coins 10%+ under their 24h high; 24h high more than 12h old; open not already 2+ points under
the 14% trigger). One position, all the money. Score >= 4: bid 14% under the 24h high close (90% mode); score <= 3:
bid 20% under instead. Take-profit +2.5%, or 3% x sqrt(24h range / 0.2074) clipped 1-6% when score >= 5.
96h time stop, 3h cooldown, crash guard 35%.
Rows: normal fills, every fill 0.2% worse, stressed (bot timing mtp_now + cooldown +1h + fills 0.2% worse +
LUNA/FTT/SRM), for 2021-23, 2024-26, since 2025 and the 2018-20 holdout; 90% mode rows from the same engine."""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m11_confluence_lib as C  # noqa: E402
import m11_confluence_grid as G  # noqa: E402

PICK = (4, 0.2, 5, None, None, None)
DESIGNS = {"90% mode": {}, "PICK " + G.name(PICK): G.design(*PICK)}


def main():
    for nm, d in DESIGNS.items():
        print(f"\n{nm}")
        for tag, mk, hk, kw in (("normal", "w", "h", {}), ("worse", "w", "h", dict(pen=0.002, pens=0.002)),
                                ("stressed", "ws", "hs", dict(pen=0.002, pens=0.002, mtp_now=True, cd_add=1))):
            M = C.market(mk)
            e, lg = C.run(M, d, **kw)
            r = C.summ(e, lg, C.PER_MAIN)
            H = C.market(hk)
            eh, lgh = C.run(H, d, **kw)
            rh = C.summ(eh, lgh, C.PER_HOLD)
            for p in ("d", "t", "s"):
                print(C.line(tag, p, r))
            print(C.line(tag, "h", rh))
            if tag == "normal" and d:
                for lab, L, idx, a, b in (("2021-23", lg, e.index, C.T21, C.T24), ("2024-26", lg, e.index, C.T24, None),
                                          ("holdout", lgh, eh.index, C.H18, C.H21)):
                    tb = idx[L.eb.astype(int).values]
                    x = L[(tb >= a) & ((tb < b) if b is not None else True)]
                    deep = x.entry.values < 0  # placeholder
                    hi = x.sc >= 4
                    print(f"     {lab}: score>=4 (14% bid) {hi.sum()} trades win {(x.ret[hi] > 0).mean()*100:.1f}% | "
                          f"score<=3 (20% bid) {(~hi).sum()} trades win {(x.ret[~hi] > 0).mean()*100:.1f}% | "
                          f"scaled TP (score>=5) {(x.sc >= 5).sum()} trades")
            print(f"  {tag} years: " + " ".join(
                f"{y} {C.V.period(e, pd.Timestamp(f'{y}-01-01'), pd.Timestamp(f'{y+1}-01-01'))[0]*100-100:+.0f}%"
                for y in range(2021, 2027)))


if __name__ == "__main__":
    main()
