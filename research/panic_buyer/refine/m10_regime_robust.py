"""m10 regime, ROUND 2 (written AFTER round 1's pick failed out of sample and after the hindsight table and the loser
anatomy were seen; disclosed as such). Lower complexity and a stricter, year-by-year 2021-23 selection rule.

Designs: one rule each (no OR-stacking) = the 94 round-1 singles + new ones from the anatomy:
  WEAK-not-CRASH  block when >= x of the coins are under their 7-day average AND no crash hour in the last 72h
                  (brd_sma7 >= x & since_brd30 > 72 & since_b7 > 72), x in {0.7, 0.8, 0.9}
  calm-coin dip   block when the coin's 7-day (30-day) daily volatility is <= v: v7 {6, 7, 8%}, v30 {6, 7%}
  vol depth       bid at max(14%, k x the coin's 7-day daily vol), k in {1.5, 2.0}
Selection (2021-23 ONLY): qualifies if, in EACH of 2021, 2022 and 2023 separately, its win rate >= 90% mode's that
year and its %/month >= 90% mode's that year - 1 point; and over 2021-23 its worst drop is no more than 1 point deeper.
Pick = highest 2021-23 win rate; ties -> higher 2021-23 %/month. Then that one design is run on 2018-20, 2024-26, since
2025, with fills 0.2% worse and with bot timing + fills 0.2% worse + LUNA/FTT/SRM.

Output: $PB_OUT/robust.log, robust.csv
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m10_regime_final as FN  # noqa: E402
import m10_regime_grid as G  # noqa: E402
import m10_regime_hindsight as HS  # noqa: E402
import m10_regime_lib as L  # noqa: E402

NEW = {}
for x in (0.7, 0.8, 0.9):
    NEW[f"WEAK-not-CRASH brd_sma7>={x}"] = ([[("brd_sma7", ">=", x), ("since_brd30", ">", 72), ("since_b7", ">", 72)]],
                                            None)
for v in (0.06, 0.07, 0.08):
    NEW[f"calm coin c_vol7<={v}"] = ([[("c_vol7", "<=", v)]], None)
for v in (0.06, 0.07):
    NEW[f"calm coin c_vol30<={v}"] = ([[("c_vol30", "<=", v)]], None)


def vdepth(k):
    def f(P):
        v = L.aligned("c_vol7", P)
        return np.where(np.isnan(v), 0.14, np.maximum(0.14, k * v))
    return f


for k in (1.5, 2.0):
    NEW[f"vol depth max(14%, {k} x vol7)"] = ([], vdepth(k))


def designs():
    D = {"90% mode": ([], None)}
    for n, r in G.SINGLES.items():
        D[n] = (r, None)
    D.update(NEW)
    return D


def main():
    log = open(os.path.join(L.OUT, "robust.log"), "w")

    def p(s):
        print(s, flush=True)
        log.write(s + "\n")

    rows = []
    D = designs()
    for name, (rules, dfn) in D.items():
        r, e, lg, idx = L.run("d", rules=rules, depth_fn=dfn)
        rows.append({"name": name, **{k: v for k, v in r.items() if k.startswith("d_")}, **HS.year_stats(e, lg, idx)})
    R = pd.DataFrame(rows)
    b = R.iloc[0]
    ok = np.ones(len(R), bool)
    for y in ("2021", "2022", "2023"):
        ok &= (R[f"y{y}_wr"] >= b[f"y{y}_wr"]) & (R[f"y{y}_mo"] >= b[f"y{y}_mo"] - 0.01)
    ok &= R.d_dd >= b.d_dd - 0.01
    R["qualifies"] = ok
    R.to_csv(os.path.join(L.OUT, "robust.csv"), index=False)
    p(f"{len(R) - 1} single-rule designs; 90% mode per year WR {b.y2021_wr*100:.1f}/{b.y2022_wr*100:.1f}/{b.y2023_wr*100:.1f}%"
      f" %/mo {b.y2021_mo*100:+.2f}/{b.y2022_mo*100:+.2f}/{b.y2023_mo*100:+.2f}")
    Q = R[ok & (R.name != "90% mode")].sort_values(["d_wr", "d_mo"], ascending=False)
    p(f"qualify (year-by-year rule, 2021-23 only): {len(Q)}")
    for _, r in Q.iterrows():
        p(f"  {r['name']:40s} 2021-23 WR {r.d_wr*100:5.1f}% {r.d_mo*100:+6.2f}%/mo DD {r.d_dd*100:4.0f}% | per year WR "
          f"{r.y2021_wr*100:4.1f}/{r.y2022_wr*100:4.1f}/{r.y2023_wr*100:4.1f} %/mo {r.y2021_mo*100:+5.2f}/"
          f"{r.y2022_mo*100:+5.2f}/{r.y2023_mo*100:+5.2f}")
    if not len(Q):
        p("nothing qualifies")
        return
    pick = Q.iloc[0]["name"]
    rules, dfn = D[pick]
    p(f"\nROUND-2 PICK: {pick}")
    res = {}
    for lab, over in (("normal", {}), ("fills 0.2% worse", L.stress_kw()), ("bot+0.2%+LUNA", L.bot_kw())):
        x = "x" if lab == "bot+0.2%+LUNA" else ""
        rd, _, _, _ = L.run("d" + x, rules=rules, depth_fn=dfn, **over)
        rh, _, _, _ = L.run("h" + x, rules=rules, depth_fn=dfn, **over)
        res[lab] = {**rh, **rd}
    FN.table(p, f"ROUND-2 PICK: {pick}", rules, None, res)
    pd.DataFrame([{"design": pick, "test": k, **v} for k, v in res.items()]).to_csv(
        os.path.join(L.OUT, "robust_pick.csv"), index=False)


if __name__ == "__main__":
    main()
