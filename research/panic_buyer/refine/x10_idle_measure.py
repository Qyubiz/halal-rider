"""Measurement only (dip bot alone, no parking): how much of the account sits in cash, and how far above its
trigger a coin opened in the hour it filled (dist = open / trigger - 1; <= 0 = market buy at the open).
Answers: how much capital a parking engine could use, and how many fills an 'armed only within x%' rule
would miss. Design decisions use the 2021-23 rows; 2024-26 rows are descriptive."""
import os

import numpy as np
import pandas as pd

import x10_idle_lib as X

rows = []
syms, idx, P = X.market(False)
for name, b in X.BASES.items():
    e, lg, aux, st = X.irun(P, idx, **b)
    sh = aux.cash / aux.equity
    inpos = 1 - sh
    eb = lg.eb.astype(int).values; k = lg.k.astype(int).values
    trig = P["ref"].values[eb, k] * (1 - b["depth"])
    dist = P["o"].values[eb, k] / trig - 1
    per = idx[eb] >= X.T_TEST0
    for tag, msk_h, msk_f in (("2021-23", aux.index < X.T_TEST0, ~per), ("2024-26", aux.index >= X.T_TEST0, per)):
        d = dist[msk_f]
        r = lg.ret.values[msk_f] * lg.amt.values[msk_f] / lg.eq0.values[msk_f]
        row = dict(base=name, period=tag, cash_mean=float(sh[msk_h].mean()), cash_median=float(sh[msk_h].median()),
                   hours_all_cash=float((sh[msk_h] > 0.999).mean()), hours_cash_lt50=float((sh[msk_h] < 0.5).mean()),
                   fills=int(msk_f.sum()), mkt_fills=float((d <= 0).mean()))
        for x in (0.01, 0.02, 0.03, 0.05, 0.08, 0.12):
            far = d > x
            row[f"fills_dist>{x:.0%}"] = float(far.mean())
            row[f"pnl_share_dist>{x:.0%}"] = float(r[far].sum() / r.sum()) if r.sum() != 0 else np.nan
        rows.append(row)
R = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(R.round(3).to_string(index=False))
R.to_csv(os.path.join(X.OUT, "measure_cash_and_fill_distance.csv"), index=False)

# how far the N-th closest coin usually is (live sizing reserves cash for all N bids)
e, lg, aux, st = X.irun(P, idx, **X.BASES["live 8x20% +1.5%"])
O = P["o"].values; ref = P["ref"].values
dist = O / (ref * (1 - 0.12)) - 1
ds = np.sort(np.where(np.isnan(dist), np.inf, dist), axis=1)
m = idx < X.T_TEST0
print("\n2021-23 median distance to trigger of the j-th closest coin at the open (all coins, ignores holdings):")
print("  " + "  ".join(f"#{j+1}: {np.median(ds[m, j])*100:4.1f}%" for j in range(8)))
for x in (0.02, 0.03, 0.05):
    print(f"  hours with at least one coin within {x:.0%} of its trigger: {np.mean(ds[m, 0] <= x)*100:.1f}% (2021-23), "
          f"{np.mean(ds[~m, 0] <= x)*100:.1f}% (2024-26)")
