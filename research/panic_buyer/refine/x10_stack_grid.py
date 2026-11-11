"""Lever "stack": grid of stacked return boosters on concentrated settings, honest pick on 2021-23, one look at 2024+.

PRE-REGISTRATION (written 22 Oct 2026 after x10_stack_motiv.py, which printed 2021-23 numbers only; nothing below
is changed after the run)
--------------------------------------------------------------------------------------------------------------
Caveat: concentrated settings were chosen as the base because the 960-setting grid showed they did best in
2024-26, so the grid's location is hindsight-informed; only the choice inside it is honest.

MAIN GRID (288 designs; trigger 12% under the 24h high close, 48h time stop, everything else as engine.run):
    sizing   (N bids per hour, cap per coin incl. both ladder legs): (2, 50%), (3, 33%), (4, 25%)
    cd       cooldown after a sale: 2h, 6h
    ladder   off | 50% at 12% + 50% at 18% | 67% at 12% + 33% at 18%
    exit     take-profit resting from the next hourly check (live rule) | no resting take-profit, sell at an
             hourly open at or above the target from the 3rd hourly check on (tprest=False, tpwait=3)
    tp       +3% | +5% (average entry after a leg-2 fill)
    tpf      none | market buys: target max(tp, 0.20 x (24h high - entry))
    BTC      none | BTC >= 5% under its 24h high close at the hour's start: 1 bid, cap min(100%, 2 x cap)
Each design runs twice: normal (19 coins, normal costs) and stressed (+LUNA/FTT/SRM, every buy AND sell fill
0.2% worse). LUNA/FTT/SRM pay 0.1% market slippage, the 19 coins 0.02%.

SELECTION RULE (2021-23 columns only, normal and stressed runs):
    eligible = normal 2021-23 worst drop >= -50% AND stressed 2021-23 worst drop >= -55%;
    pick = highest STRESSED 2021-23 %/month (the run with the collapse coins a dip bot would really have bought
    and worse fills); tie -> shallower stressed 2021-23 drop.
    If nothing is eligible: highest stressed 2021-23 CAGR / |stressed drop|.
    Then 2024-26 and since-2025 are printed once for the pick (four reports + worst trade as % of the account).

HINDSIGHT (labelled, not a pick): the main-grid design with the highest normal 2024-26 %/month, and the top 10.

CEILING PROBE (hindsight only, not eligible for the pick): 1 bid x 100% of the account with the same boosters,
    cd {2, 6} x ladder (3) x exit (2) x tp (2) x tpf (2) = 48 designs, to measure how far stacking can push
    2024-26 at all.

Outputs in $PB_OUT: grid.csv (every design, both runs), ceiling.csv, grid.log (stdout).
"""
import itertools
import json
import os

import numpy as np
import pandas as pd

import x10_stack_lib as X

SIZING = [(2, 0.50), (3, 1 / 3), (4, 0.25)]
CDS = [2, 6]
LADDERS = {"off": {}, "50/18": dict(w=0.5, d2=0.18), "67/18": dict(w=2 / 3, d2=0.18)}
EXITS = {"rest": {}, "norest3": dict(tprest=False, tpwait=3)}
TPS = [0.03, 0.05]
TPFS = {"none": {}, "mkt.20": dict(tpf=0.20, tpfm=1)}
BTCS = ["none", "btc5"]
MAX_DD_N, MAX_DD_S = -0.50, -0.55


def design(N, cap, cd, lad, ex, tp, tpf, btc):
    s = dict(depth=0.12, N=N, fmax=cap, tp=tp, hm=48, cd=cd, **LADDERS[lad], **EXITS[ex], **TPFS[tpf])
    if btc == "btc5":
        s.update(bx=0.05, bN=1, bf=min(1.0, 2 * cap))
    name = f"N{N} cap{cap*100:.0f} cd{cd} lad {lad} {ex} tp{tp*100:.0f} tpf {tpf} {btc}"
    return name, s


def run_all(designs):
    rows = []
    for name, s in designs:
        r, _, _ = X.evaluate(s)
        rs, _, _ = X.evaluate(s, stress=True)
        row = {"name": name, "spec": json.dumps(s), **r, **{f"S_{k}": v for k, v in rs.items()}}
        rows.append(row)
    return pd.DataFrame(rows)


def short(r, p=""):
    return (f"21-23 {r[p+'d_mo']*100:+6.2f}%/mo DD {r[p+'d_dd']*100:5.1f}% | 24-26 {r[p+'t_mo']*100:+6.2f}%/mo "
            f"DD {r[p+'t_dd']*100:5.1f}% | 25-26 {r[p+'s_mo']*100:+6.2f}%/mo DD {r[p+'s_dd']*100:5.1f}%")


def four(row):
    r = {k: row[k] for k in row.index if not k.startswith("S_")}
    rs = {k[2:]: row[k] for k in row.index if k.startswith("S_")}
    return X.fmt4(row["name"], r, rs)


if __name__ == "__main__":
    out = X.OUT
    grid = [design(N, cap, cd, lad, ex, tp, tpf, btc) for (N, cap), cd, lad, ex, tp, tpf, btc in
            itertools.product(SIZING, CDS, LADDERS, EXITS, TPS, TPFS, BTCS)]
    assert len(grid) == 288
    G = run_all(grid)
    G.to_csv(os.path.join(out, "grid.csv"), index=False)

    # ---------------- honest pick: the selection sees 2021-23 columns only
    D = G[["name", "d_mo", "d_dd", "y2022", "S_d_mo", "S_d_dd", "S_y2022"]].copy()
    elig = D[(D.d_dd >= MAX_DD_N) & (D.S_d_dd >= MAX_DD_S)]
    print(f"######## SELECTION (2021-23 only): {len(elig)} of {len(D)} designs eligible "
          f"(normal drop >= {MAX_DD_N:.0%}, stressed drop >= {MAX_DD_S:.0%})")
    if len(elig):
        top = elig.sort_values(["S_d_mo", "S_d_dd"], ascending=[False, False])
        pick = top.iloc[0]["name"]
        rule = "highest stressed 2021-23 %/month among eligible"
    else:
        top = D.assign(sc=D.S_d_mo / D.S_d_dd.abs().clip(lower=1e-9)).sort_values("sc", ascending=False)
        pick = top.iloc[0]["name"]
        rule = "fallback: highest stressed 2021-23 return / |drop|"
    for _, r in top.head(10).iterrows():
        print(f"  {r['name']:60s} normal 21-23 {r.d_mo*100:+6.2f}%/mo DD {r.d_dd*100:5.1f}% 2022 {r.y2022*100:+5.0f}% | "
              f"stressed {r.S_d_mo*100:+6.2f}%/mo DD {r.S_d_dd*100:5.1f}% 2022 {r.S_y2022*100:+5.0f}%")
    print(f"PICK ({rule}): {pick}")

    # ---------------- one look at 2024-26 / since 2025 for the pick
    print("\n######## HONEST PICK, four reports (2024-26 and since 2025 looked at once)")
    prow = G[G.name == pick].iloc[0]
    print(four(prow))
    for y in range(2021, 2027):
        print(f"    {y}: normal {prow[f'y{y}']*100:+7.1f}%  stressed {prow[f'S_y{y}']*100:+7.1f}%")

    # ---------------- reference rows (already published settings, same metric code)
    print("\n######## REFERENCE (same metric code)")
    for name, s in (("LIVE 8 coins x 20%, +1.5%, cd 24", {}),
                    ("2 coins x 50%, +3%, cd 24", dict(N=2, fmax=0.5, tp=0.03)),
                    ("2 coins x 50%, +5%, cd 24 (hr pick)", dict(N=2, fmax=0.5, tp=0.05)),
                    ("1 coin x 100%, +3%, cd 24", dict(N=1, fmax=1.0, tp=0.03))):
        r, _, _ = X.evaluate(s)
        rs, _, _ = X.evaluate(s, stress=True)
        print(X.fmt4(name, r, rs))

    # ---------------- hindsight
    print("\n######## HINDSIGHT (chosen by 2024-26, not a fair pick)")
    H = G.sort_values("t_mo", ascending=False)
    print(four(H.iloc[0]))
    print("  top 10 by normal 2024-26 %/month:")
    for _, r in H.head(10).iterrows():
        print(f"  {r['name']:60s} {short(r)} | stressed 21-23 {r.S_d_mo*100:+6.2f}%/mo DD {r.S_d_dd*100:5.1f}%")
    print(f"  grid 2024-26 %/month: max {G.t_mo.max()*100:.2f}, median {G.t_mo.median()*100:.2f}; designs >= 10%/mo: "
          f"{int((G.t_mo >= 0.10).sum())}; since 2025 max {G.s_mo.max()*100:.2f}")
    print(f"  pick's rank by 2024-26 %/month: {int((G.t_mo > prow.t_mo).sum()) + 1} of {len(G)}")

    # ---------------- factor marginals (all designs, means of 2024-26 and stressed 2021-23 %/month)
    print("\n######## FACTOR MEANS over the grid (2021-23 normal | stressed 2021-23 | 2024-26 | since 2025, %/month)")
    sp = G.spec.map(json.loads)
    F = pd.DataFrame({"sizing": sp.map(lambda s: f"N{s['N']}"), "cd": sp.map(lambda s: s["cd"]),
                      "ladder": G.name.str.extract(r"lad (\S+)")[0], "exit": G.name.str.extract(r"(rest|norest3)")[0],
                      "tp": sp.map(lambda s: s["tp"]), "tpf": G.name.str.extract(r"tpf (\S+)")[0],
                      "btc": G.name.str.split().str[-1]})
    for col in F.columns:
        m = pd.concat([F[col], G[["d_mo", "S_d_mo", "t_mo", "s_mo", "t_dd"]]], axis=1).groupby(col).mean()
        print(f"  {col}: " + " | ".join(f"{k}: {v.d_mo*100:+.2f} {v.S_d_mo*100:+.2f} {v.t_mo*100:+.2f} {v.s_mo*100:+.2f} "
                                         f"(24-26 DD {v.t_dd*100:.0f}%)" for k, v in m.iterrows()))

    # ---------------- ceiling probe (hindsight only)
    print("\n######## CEILING PROBE: 1 bid x 100% of the account + boosters (hindsight, not eligible)")
    ceil = [(f"N1 cap100 cd{cd} lad {lad} {ex} tp{tp*100:.0f} tpf {tpf}",
             dict(depth=0.12, N=1, fmax=1.0, tp=tp, hm=48, cd=cd, **LADDERS[lad], **EXITS[ex], **TPFS[tpf]))
            for cd, lad, ex, tp, tpf in itertools.product(CDS, LADDERS, EXITS, TPS, TPFS)]
    assert len(ceil) == 48
    C = run_all(ceil)
    C.to_csv(os.path.join(out, "ceiling.csv"), index=False)
    Cs = C.sort_values("t_mo", ascending=False)
    print(four(Cs.iloc[0]))
    for _, r in Cs.head(8).iterrows():
        print(f"  {r['name']:60s} {short(r)} | stressed 21-23 {r.S_d_mo*100:+6.2f}%/mo DD {r.S_d_dd*100:5.1f}%")
    print(f"  ceiling 2024-26 %/month: max {C.t_mo.max()*100:.2f}, median {C.t_mo.median()*100:.2f}; >= 10%/mo: "
          f"{int((C.t_mo >= 0.10).sum())}; since 2025 max {C.s_mo.max()*100:.2f}")
