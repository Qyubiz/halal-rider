"""Final table for the velocity lever (x10 study): the four reports for the key designs, plus a strict-fill row (limit
orders - buys and the cap sell - need 0.3% through instead of 0.1%). A measurement only; the designs come from the
grids: REF live, the round-1 pick (honest), the round-2 pick (honest inside its grid; round 2 was started after
round 1 missed 10%), the round-3 pick (2021-23 rule inside a grid located by hindsight) and two hindsight designs."""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import x10_velocity_lib as V  # noqa: E402
import x10_velocity_grid as G1  # noqa: E402
import x10_velocity_grid2 as G2  # noqa: E402
import x10_velocity_grid3 as G3  # noqa: E402

DESIGNS = [
    ("REF live settings", G1.REFS["REF live (N8 f0.2 tp1.5 hourly cd24h)"]),
    ("R1 PICK (honest): S2 N2 | C4 1min | cd24h | E5 trail a3 t0 c8", G1.GRID["S2 N2 | C4 1min | cd24h | E5 trail a3 t0 c8"]),
    ("R2 PICK: N2 | 15min | trail a3 | hm48 | flash3h8 | cd24h", G2.GRID["N2 | 15min | trail a3 | hm48 | flash3h8 | cd24h"]),
    ("R3 PICK (hindsight-located grid): N1 f0.7 | trail a3 | hm24 | flash3h8 | redip3/60m | lossCD 72h",
     G3.GRID["N1 f0.7 | trail a3 | hm24 | flash3h8 | redip3/60m | lossCD 72h"]),
    ("HINDSIGHT BEST: N1 f1 | trail a3 | hm24 | flash3h8 | redip3/60m | lossCD 24h",
     G3.GRID["N1 f1 | trail a3 | hm24 | flash3h8 | redip3/60m | lossCD 24h"]),
    ("HINDSIGHT >=10% least hurt by LUNA: N1 f1 | trail a3 | hm12 | flash3h8 | redip3/60m | lossCD 168h",
     G3.GRID["N1 f1 | trail a3 | hm12 | flash3h8 | redip3/60m | lossCD 168h"]),
]
ROWS = [("(1-3) 19 coins", 19, {}), ("+LUNA/FTT/SRM", 22, {}), ("fills 0.2% worse", 19, dict(pen=0.002, spen=0.002)),
        ("(4) fills 0.2% worse +LUNA/FTT/SRM", 22, dict(pen=0.002, spen=0.002)), ("strict limit fills (0.3% through)", 19, dict(eps=0.003))]


def main():
    out = []
    for name, kw in DESIGNS:
        print(f"\n{name}\n  settings {kw}")
        for lab, K, over in ROWS:
            e, lg, inv = V.vrun(G1.A, G1.hidx, G1.Ph, G1.SLIP, K=K, **{**kw, **over})
            r = V.report(e, lg, G1.hidx)
            r.update(design=name, row=lab)
            out.append(r)
            wt = (f"{r['t_worst_trade']*100:+.1f}%" if r.get("t_trades", 0) else "n/a")
            print(f"  {lab:36s} {V.fmt(r)} | worst trade %acct 24-26 {wt}, 21-23 {r['d_worst_trade']*100:+.1f}% "
                  f"({r['d_worst_trade_coin']}) | 2024-26 x{r['t_mult']:.2f}, trades {r.get('t_trades', 0)}", flush=True)
    pd.DataFrame(out).to_csv(os.path.join(V.OUT, "summary.csv"), index=False)


if __name__ == "__main__":
    main()
