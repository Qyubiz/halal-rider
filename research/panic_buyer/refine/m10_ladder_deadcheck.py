"""m10 ladder: LUNA/FTT/SRM safety check on 2021-23 ONLY (panel cut at 2024-01-01) for the stage 2-3 leaders.
Added after the stage-3 ranking (before any 2024-26 or holdout run): the rule's literal pick has no crash guard, and
the collapse coins (LUNA May 2022, FTT/SRM Nov 2022) are inside the design period. Extra filter: worst drop with
LUNA/FTT/SRM (sold at their last trading hour) >= -65%, normal and bot timing + fills 0.2% worse. Among the stage 2-3
designs passing the original rule AND this filter: highest median %/month (ties within 0.3: higher min WR)."""
import os
import pandas as pd
import m10_ladder_lib as M
import m10_ladder_grid2 as G

R = pd.concat([pd.read_csv(os.path.join(M.OUT, f"grid{s}.csv")) for s in (2, 3)], ignore_index=True)
Q = G.rank(R, 0.97, -0.65).head(10)
syms, idx, P = M.market(True, end=M.T_T0)
rows = []
for _, q in Q.iterrows():
    spec = eval(q.spec)
    out = {}
    for lab, over in (("dead", dict(liquidate=True)),
                      ("dead_x", dict(liquidate=True, mtp_now=True, cd=spec["cd"] + 1, pen=0.002, pens=0.002))):
        e, lg = M.lrun(P, idx, **{**spec, **over})
        r = M.stats(e, lg, idx, M.T_D0, M.T_T0)
        out.update({f"{lab}_mo": r["mo"], f"{lab}_dd": r["dd"], f"{lab}_wr": r["wr"], f"{lab}_wt": r["wt"]})
    rows.append({"name": q["name"], "spec": q.spec, "med_mo": q.med_mo, "min_wr": q.min_wr, "min_dd": q.min_dd, **out})
    print(f"{q['name'][:62]:62s} med {q.med_mo*100:+.2f} minWR {q.min_wr*100:.1f} | +LUNA/FTT/SRM {out['dead_mo']*100:+.2f}%/mo "
          f"dd {out['dead_dd']*100:.0f}% WR {out['dead_wr']*100:.1f} worst trade {out['dead_wt']*100:+.0f}% | stressed "
          f"{out['dead_x_mo']*100:+.2f} dd {out['dead_x_dd']*100:.0f}%", flush=True)
D = pd.DataFrame(rows)
ok = D[(D.dead_dd >= -0.65) & (D.dead_x_dd >= -0.65)].sort_values("med_mo", ascending=False)
best = ok.iloc[0]
near = ok[ok.med_mo >= best.med_mo - 0.003].sort_values("min_wr", ascending=False)
print(f"\nFINAL PICK (2021-23 only): {near.iloc[0]['name']}\n  spec {near.iloc[0].spec}")
