"""Diagnostics: each parking engine ALONE (no dip bot, N=0), with fees, on 19 coins and on 19 + LUNA/FTT/SRM.
Shows what the second engine can add at most (it earns this on the share of the account it holds), and checks
the research history (HANDOFF section 4: momentum rotation alone lost)."""
import os

import pandas as pd

import x10_idle_grid as G
import x10_idle_lib as X

rows = []
for stress in (False, True):
    syms, idx, P = X.market(stress)
    for name, park in G.PARKS.items():
        pw = X.weights_for(P, idx, park)
        kw = dict(N=0, band=0.0)
        if stress:
            kw.update(pen=0.002, pens=0.002)
        e, lg, aux, st = X.irun(P, idx, pw=pw, **kw)
        r = X.summarize(e, lg, idx, aux, st)
        r.update(park=name, stress=stress)
        rows.append(r)
        print(f"{'stress ' if stress else ''}{name:12s} 2021-23 {r['d_mo']*100:+5.2f}%/mo drop {r['d_dd']*100:5.1f}% "
              f"2022 {r['y2022']*100:+5.1f}% | 2024-26 {r['t_mo']*100:+5.2f}%/mo drop {r['t_dd']*100:5.1f}% | "
              f"since 2025 {r['s_mo']*100:+5.2f}%/mo drop {r['s_dd']*100:5.1f}% | invested 21-23 {r['d_park']*100:.0f}% "
              f"24-26 {r['t_park']*100:.0f}% | turnover x{r['pk_turnover']:.0f} costs {r['pk_costs']*100:.1f}%")
pd.DataFrame(rows).to_csv(os.path.join(X.OUT, "parking_alone.csv"), index=False)
