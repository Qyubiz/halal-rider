"""m10 meanrev, stage 2 (2021-23 ONLY; bars after 2023 are cut off). Stage 1 (m10_meanrev_grid1.py, grid1_5m/15m.csv)
found that Bollinger / Keltner / RSI / z-score entries and exits at the mean have ~zero or negative expectancy after
costs on 5m and 15m bars; only deep %-drop entries ("panic wicks") have an edge, and 5m beats 15m. Here: those entries
on 5m bars x take-profit x time stop x sizing x regime filter.

Regime filters (all from closed bars; every design keeps the 90% mode crash guard: skip a coin >35% under its 7-day high):
  G   : crash guard only
  BTC : also skip all buys while BTC's last close is >5% under its 24h high (market-wide crash)
  KNF : also skip a coin whose last close is >15% under its own 24h high (falling knife already)
  LB  : also pause all buys for 72h after any losing trade
PRE-REGISTERED PICK (written before running): among designs with 2021-23 win rate >= 93%, >= 10 trades/month, each of
2021, 2022, 2023 positive and 2021-23 worst drop (5-minute closes) no deeper than -40%, the highest 2021-23 %/month.
If none qualifies: the same with -50%.
Output: OUT/grid2.csv, OUT/grid2_pick.json
"""
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m10_meanrev_lib as L  # noqa: E402

B = L.cut(L.load("main", C=5), L.T21, L.T24)
hb = lambda x: L.bars_per(B, x)   # noqa: E731
cfs = L.sh(B["cf"])
guard = cfs >= L.rollmax(B, hb(24 * 7)) * (1 - 0.35)
h24 = L.rollmax(B, hb(24))
ib = B["syms"].index("BTCUSDT")
btc_ok = (cfs[:, ib] >= h24[:, ib] * 0.95)[:, None]
knf_ok = cfs >= h24 * 0.85
REG = {"G": guard, "BTC": guard & btc_ok, "KNF": guard & knf_ok, "LB": guard}

ENTRIES = {}
for W, d in ((1, 0.05), (1, 0.06), (1, 0.08), (4, 0.08), (12, 0.10), (12, 0.12), (24, 0.12)):
    ENTRIES[f"DIP W{W}h d{d*100:.0f}"] = lambda W=W, d=d: L.rollmax(B, hb(W)) * (1 - d)
ENTRIES["SIG red4+drop4"] = lambda: np.where((L.red_run(B) >= 4) & (L.ret_over(B, 4) < -0.04), np.inf, np.nan)
TPS, HMS = (0.008, 0.012, 0.02, 0.03), (6, 24, 96)
SIZES = {"1x100": (1, 1.0), "2x50": (2, 0.5), "3x33": (3, 1 / 3), "4x25": (4, 0.25)}

rows = []
t0 = time.time()
for name, fn in ENTRIES.items():
    raw = fn()
    for reg, mask in REG.items():
        lim = np.where(mask, raw, np.nan)
        for tp, h, (sz, (N, f)) in itertools.product(TPS, HMS, SIZES.items()):
            eq, lg = L.run(B, lim, nord=N, fmax=f, maxpos=N, tp=tp, hm=hb(h), cd=hb(1),
                           lossblock=hb(72) if reg == "LB" else 0)
            r = L.stats(B, eq, lg, L.T21, L.T24)
            y = L.yearly(B, eq)
            rows.append({"entry": name, "reg": reg, "tp": tp, "hm": h, "size": sz, **r,
                         **{f"y{k}": y.get(k, np.nan) for k in (2021, 2022, 2023)}})
        del lim
    print(f"{name:18s} done ({time.time()-t0:.0f}s)", flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(L.OUT, "grid2.csv"), index=False)

pd.set_option("display.width", 250)
cols = ["entry", "reg", "tp", "hm", "size", "mo", "dd", "dd_h", "win", "avg", "tpm", "worst_acct", "y2021", "y2022", "y2023"]
base = df[(df.win >= 0.93) & (df.tpm >= 10) & (df.y2021 > 0) & (df.y2022 > 0) & (df.y2023 > 0)]
pick = None
for lim_dd in (-0.40, -0.50):
    q = base[base.dd >= lim_dd]
    print(f"\nqualifying with drop >= {lim_dd:.0%}: {len(q)}")
    print(q.sort_values("mo", ascending=False)[cols].head(15).to_string(index=False))
    if len(q):
        pick = q.sort_values("mo", ascending=False).iloc[0]
        break
print("\nPICK (2021-23 only):", pick[cols].to_dict() if pick is not None else None)
if pick is not None:
    json.dump({k: (v.item() if hasattr(v, "item") else v) for k, v in pick.to_dict().items()},
              open(os.path.join(L.OUT, "grid2_pick.json"), "w"), indent=1)

print("\nfrontier (2021-23): best %/month at each win-rate floor, >= 10 trades/month")
for w in (0.90, 0.93, 0.95, 0.97, 0.98):
    q = df[(df.win >= w) & (df.tpm >= 10)]
    for dd in (-0.30, -0.40, -0.60):
        z = q[q.dd >= dd]
        if len(z):
            r = z.sort_values("mo", ascending=False).iloc[0]
            print(f"  win>={w:.0%} drop>={dd:.0%}: {r.entry} {r.reg} tp{r.tp} hm{r.hm} {r['size']}: {r.mo*100:+.2f}%/mo "
                  f"dd {r.dd*100:.0f}% win {r.win*100:.1f}% {r.tpm:.0f}/mo")
