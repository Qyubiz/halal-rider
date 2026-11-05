"""Final numbers for the idle-cash lever (all four reports for every design named in the result).

1. The live bot's idle cash (from x10_idle_measure.py logic).
2. FINAL HONEST PICK = selection rule (2022 >= 0%, 2021-23 drop >= -50%, best 2021-23 %/month) on the union of the
   pre-registered rounds 1-3, and its base without parking.
3. Per-base honest picks (the same rule applied within each base; the live base included).
4. HINDSIGHT: best of the pre-registered grids by 2024-26, and the best 2024-26 of any variant run in this lever
   (x10_idle_hindsight.py plateau: ETH 30-day filter).
5. Diagnostic upper bound: the same designs with FREE parking trades (no fee, no slippage on the parking legs).
"""
import os

import numpy as np
import pandas as pd

import x10_idle_grid as G1
import x10_idle_grid2 as G2
import x10_idle_grid3 as G3
import x10_idle_lib as X

MIN_2022, MAX_DD = 0.0, -0.50
PARKS = {**G1.PARKS, **G3.PARKS}


def park_of(row):
    if row["park"] == "none":
        return None
    p = dict(PARKS.get(row["park"], G2.PARKS.get(row["park"])) if row["park"] in PARKS else G2.PARKS[row["park"]])
    if "pf" in row and not pd.isna(row["pf"]):
        p["pf"] = float(row["pf"])
    return p


def pk_of(row):
    if row["park"] == "none":
        return None
    return dict(mode=int(row["mode"]), armx=float(row["armx"]), band=0.05, rebuy_wait=int(row["rebuy_wait"]))


def lab(row):
    if row["park"] == "none":
        return f"{row['base']} | no parking"
    pf = "" if pd.isna(row.get("pf", np.nan)) else f" x{row['pf']:.0%}"
    return (f"{row['base']} | park {row['park']}{pf} | mode {int(row['mode'])} arm {row['armx']:.0%} | "
            f"re-buy wait {int(row['rebuy_wait'])}h (round {int(row['round'])})")


def four(tag, b, park, pk, name, **extra):
    r, e, lg, aux = X.evaluate(b, park, pk) if not extra else X.evaluate(b, park, pk)
    rs, _, _, _ = X.evaluate(b, park, pk, stress=True)
    print(X.fmt4(f"{tag}: {name}", r, rs))
    print("      years: " + " ".join(f"{y} {r[f'y{y}']*100:+.0f}%" for y in range(2021, 2027)))
    return r, rs


def free(b, park, pk):
    syms, idx, P = X.market(False)
    kw = {k: b[k] for k in X.BOT_KEYS}; kw.update(pk)
    e, lg, aux, st = X.irun(P, idx, pw=X.weights_for(P, idx, park), pfee=0.0, pslip=0.0, **kw)
    return X.summarize(e, lg, idx, aux, st)


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    syms, idx, P = X.market(False)
    print("######## 1. Idle cash of the dip bot alone (cash / account, hourly average)")
    for bn, b in X.BASES.items():
        e, lg, aux, st = X.irun(P, idx, **{k: b[k] for k in X.BOT_KEYS})
        sh = aux.cash / aux.equity
        m = aux.index < X.T_TEST0
        print(f"  {bn:18s} 2021-23 {sh[m].mean()*100:.1f}% (all-cash hours {(sh[m] > 0.999).mean()*100:.0f}%) | "
              f"2024-26 {sh[~m].mean()*100:.1f}% (all-cash hours {(sh[~m] > 0.999).mean()*100:.0f}%)")

    U = []
    for f, rnd in (("grid.csv", 1), ("grid2.csv", 2), ("grid3.csv", 3)):
        g = pd.read_csv(os.path.join(X.OUT, f)); g["round"] = rnd
        if "pf" not in g:
            g["pf"] = np.nan
        U.append(g)
    U = pd.concat(U, ignore_index=True)
    ok = U[(U.y2022 >= MIN_2022) & (U.d_dd >= MAX_DD)].sort_values(["d_mo", "d_dd"], ascending=[False, False])
    print(f"\n######## 2. FINAL HONEST PICK (rule on 2021-23 only, union of {len(U)} pre-registered runs, "
          f"{len(ok)} qualify)")
    pick = ok.iloc[0]
    b = X.BASES[pick.base]
    rp, rps = four("HONEST PICK", b, park_of(pick), pk_of(pick), lab(pick))
    four("ITS BASE WITHOUT PARKING", b, None, None, f"{pick.base} | no parking")
    four("LIVE BOT (reference)", X.BASES["live 8x20% +1.5%"], None, None, "live 8x20% +1.5% | no parking")

    print("\n######## 3. Per-base honest picks (same rule within each base; 'none' = no parking won)")
    for bn in X.BASES:
        okb = ok[ok.base == bn]
        if not len(okb):
            print(f"  {bn}: no run of this base passes the rule (2021-23 drop deeper than -50% or 2022 < 0%)")
            continue
        row = okb.iloc[0]
        four(f"PER-BASE PICK {bn}", X.BASES[bn], park_of(row), pk_of(row), lab(row))

    print("\n######## 4. HINDSIGHT")
    hb = U.sort_values("t_mo", ascending=False).iloc[0]
    four("HINDSIGHT best of the pre-registered grids by 2024-26", X.BASES[hb.base], park_of(hb), pk_of(hb), lab(hb))
    eth30 = dict(kind="fixed", assets={"ETHUSDT": 1.0}, filt=30)
    hpk = dict(mode=0, armx=0.07, band=0.05, rebuy_wait=24)
    rh, rhs = four("HINDSIGHT best 2024-26 of any variant run (plateau neighbour)", X.BASES["1x100% +3%"], eth30, hpk,
                   "1x100% +3% | park ETH, 30-day filter | mode 0 arm 7% | re-buy wait 24h")

    print("\n######## 5. Diagnostic upper bound: parking legs free (no fee, no slippage); not achievable on Binance")
    for name, bb, park, pk in (("honest pick", b, park_of(pick), pk_of(pick)),
                               ("hindsight ETH f50 1x100%", X.BASES["1x100% +3%"],
                                dict(kind="fixed", assets={"ETHUSDT": 1.0}, filt=50), hpk)):
        r = free(bb, park, pk)
        print(f"  {name:26s} free parking: 2024-26 {r['t_mo']*100:+.2f}%/mo drop {r['t_dd']*100:.1f}% | since 2025 "
              f"{r['s_mo']*100:+.2f}%/mo drop {r['s_dd']*100:.1f}% | 2021-23 {r['d_mo']*100:+.2f}%/mo drop "
              f"{r['d_dd']*100:.1f}%")
    print(f"\nall pre-registered runs: {len(U)}; max 2024-26 of them {U.t_mo.max()*100:.2f}%/mo; "
          f"runs >= 10%/mo in 2024-26: {(U.t_mo >= 0.10).sum()}")
