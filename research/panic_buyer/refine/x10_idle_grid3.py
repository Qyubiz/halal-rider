"""Lever "idle cash", round 3: slower / narrower rotation parking. Pre-registered, pick on 2021-23 only.

PRE-REGISTRATION (written 22 Oct 2026 after rounds 1-2 and the hindsight checks were seen, before this script was
run; not changed after the run). The grid's location is informed by everything seen so far (incl. 2024-26: ETH
did best there); only the choice inside it, and the final union pick, are honest.

GRID: 4 bases (as before) x (no parking + 3 parking engines x 4 mechanics) = 52 runs, 19 coins, band 5%.
  parking engines:
     BTC/ETH dual f50     daily: of BTC and ETH, the one with the higher 30-day return, if that return > 0 and it
                          is above its 50-day average; else cash
     top1-90 f50 weekly   Mondays 00:00 UTC: the coin of the panel with the best 90-day return (> 0, above its
                          50-day average), held for the week
     top3-90 f50 weekly   the same, top 3, equal weight
  mechanics: (mode 1, arm 2%, wait 168h) (mode 0, arm 5%, wait 168h) (mode 0, arm 5%, wait 24h) (mode 0, arm 7%, wait 24h)

SELECTION RULE (unchanged; 2021-23 columns only): 2022 >= 0% and 2021-23 worst drop >= -50%; highest 2021-23
%/month. FINAL HONEST PICK of the lever = the same rule applied to the union of rounds 1, 2 and 3 (all
pre-registered runs). One look at 2024-26 / since 2025 / stress for the round-3 pick and the final pick.
"""
import os

import numpy as np
import pandas as pd

import x10_idle_lib as X

PARKS = {
    "BTC/ETH dual f50": dict(kind="mom", days=30, top=1, absmom=True, filt=50, cands=["BTCUSDT", "ETHUSDT"]),
    "top1-90 f50 weekly": dict(kind="mom", days=90, top=1, absmom=True, filt=50, weekly=True),
    "top3-90 f50 weekly": dict(kind="mom", days=90, top=3, absmom=True, filt=50, weekly=True),
}
MECH = [(1, 0.02, 168), (0, 0.05, 168), (0, 0.05, 24), (0, 0.07, 24)]
MIN_2022, MAX_DD = 0.0, -0.50


def full_report(tag, b, park, pk, lab):
    r, e, lg, aux = X.evaluate(b, park, pk)
    rs, _, _, _ = X.evaluate(b, park, pk, stress=True)
    print(X.fmt4(f"{tag}: {lab}", r, rs))
    print("      years: " + " ".join(f"{y} {r[f'y{y}']*100:+.0f}%" for y in range(2021, 2027)))


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    syms, idx, P = X.market(False)
    W = {p: X.weights_for(P, idx, s) for p, s in PARKS.items()}
    rows = []
    for bname, b in X.BASES.items():
        for cmb in [None] + [(p, m) for p in PARKS for m in MECH]:
            kw = {k: b[k] for k in X.BOT_KEYS}
            if cmb:
                p, (mo, ax, wt) = cmb
                kw.update(mode=mo, armx=ax, band=0.05, rebuy_wait=wt)
            e, lg, aux, st = X.irun(P, idx, pw=None if cmb is None else W[cmb[0]], **kw)
            r = X.summarize(e, lg, idx, aux, st if cmb else None)
            r.update(base=bname, park=cmb[0] if cmb else "none", mode=cmb[1][0] if cmb else -1,
                     armx=cmb[1][1] if cmb else np.nan, rebuy_wait=cmb[1][2] if cmb else -1)
            rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(X.OUT, "grid3.csv"), index=False)
    D = R[["base", "park", "mode", "armx", "rebuy_wait", "d_mo", "d_dd", "y2022"]]
    ok = D[(D.y2022 >= MIN_2022) & (D.d_dd >= MAX_DD)]
    print(f"round 3: {len(R)} runs; qualify on 2021-23: {len(ok)}")
    print(ok.sort_values(["d_mo", "d_dd"], ascending=[False, False]).head(8).round(4).to_string(index=False))
    if len(ok):
        pk3 = R.loc[ok.sort_values(["d_mo", "d_dd"], ascending=[False, False]).index[0]]
        b = X.BASES[pk3.base]
        if pk3.park == "none":
            full_report("ROUND-3 PICK", b, None, None, f"{pk3.base} no parking")
        else:
            full_report("ROUND-3 PICK", b, PARKS[pk3.park], dict(mode=int(pk3["mode"]), armx=float(pk3.armx), band=0.05,
                        rebuy_wait=int(pk3.rebuy_wait)), f"{pk3.base} | {pk3.park} | m{int(pk3['mode'])} "
                        f"arm {pk3.armx:.0%} wait {int(pk3.rebuy_wait)}h")

    # union of the three pre-registered rounds (2021-23 columns only)
    U = []
    for f, rnd in (("grid.csv", 1), ("grid2.csv", 2), ("grid3.csv", 3)):
        G = pd.read_csv(os.path.join(X.OUT, f))
        G["round"] = rnd
        U.append(G[["round", "base", "park", "mode", "armx", "rebuy_wait"] + (["pf"] if "pf" in G else []) +
                   ["d_mo", "d_dd", "y2022"]])
    U = pd.concat(U, ignore_index=True)
    okU = U[(U.y2022 >= MIN_2022) & (U.d_dd >= MAX_DD)].sort_values(["d_mo", "d_dd"], ascending=[False, False])
    print(f"\nUNION of rounds 1-3: {len(U)} runs, {len(okU)} qualify on 2021-23. Top 5 by 2021-23 %/month:")
    print(okU.head(5).round(4).to_string(index=False))

    print("\nround-3 table (all bases):")
    show = ["base", "park", "mode", "armx", "rebuy_wait", "d_mo", "d_dd", "y2022", "t_mo", "t_dd", "s_mo", "s_dd",
            "pk_costs"]
    print(R[show].round(4).to_string(index=False))
    print(f"\nMAX 2024-26 %/month of any round-3 run (hindsight): {R.t_mo.max()*100:.2f}%")
