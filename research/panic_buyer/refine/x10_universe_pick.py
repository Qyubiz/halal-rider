"""x10 lever "universe", step 2: apply the pre-registered selection rules (x10_universe_grid.py and
x10_universe_round2.py docstrings) using 2021-23 columns only, then report each pick once: 2024-26, since 2025,
2021-23, and the stress run (fills 0.2% worse + FTT added + LUNA/FTT/SRM always eligible). The hindsight best
(highest 2024-26 %/month) is reported separately and labelled hindsight.
Usage: python x10_universe_pick.py [1|2]   (round; default 1)"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import x10_universe_grid as G  # noqa: E402
import x10_universe_lib as X  # noqa: E402
import x10_universe_round2 as G2  # noqa: E402
import x10_universe_push as G3  # noqa: E402

KEYS = ["univ", "mode", "N", "fmax", "depth", "tp", "cd", "hm"]
SEL_COLS = KEYS + ["21-23_mo", "21-23_dd", "y2022"]           # the only columns the selection code sees
FORCE = ("LUNAUSDT", "FTTUSDT", "SRMUSDT")


def pick(D, univs, ddmin):
    ok = D[D.univ.isin(univs) & (D.y2022 >= 0) & (D["21-23_dd"] >= ddmin)]
    if not len(ok):
        return None
    return ok.sort_values(["21-23_mo", "21-23_dd"], ascending=[False, False]).iloc[0]


def name(r):
    return (f"{r['univ']} {r['mode']} N{int(r['N'])}x{r['fmax'] * 100:.0f}% depth {r['depth'] * 100:.0f}% "
            f"tp +{r['tp'] * 100:.1f}% cd {int(r['cd'])}h hm {int(r['hm'])}h")


def run_spec(U_, r, cache, force=(), **kw):
    """Run one grid row's settings on panel U_ (round 1 or round 2 universe). Returns e, lg, i19, symbols."""
    univ = r["univ"]
    args = (univ, r["mode"], int(r["N"]), float(r["fmax"]), float(r["depth"]), float(r["tp"]), int(r["cd"]))
    if univ in G2.UNIVS2:
        e, lg, i19, cols = G2.run_one2(U_, *args, force=force, hm=int(r["hm"]), **kw)
    elif univ.startswith("L"):
        e, lg, i19, cols = G3.run_one3(U_, *args, force=force, hm=int(r["hm"]), **kw)
    else:
        key = (r["depth"], r["tp"], r["cd"], int(r["hm"]))
        if univ not in ("U19", "T20", "T30", "T50") and key not in cache:
            cache[key] = X.standalone_scores(U_, r["depth"], r["tp"], int(r["hm"]), int(r["cd"]))
        sc = cache.get(key)
        e, lg, i19 = G.run_one(U_, *args, sc, force=force, hm=int(r["hm"]), **kw)
        cols = X.sub(U_, G.umask(U_, univ, sc, force))[4]
    return e, lg, i19, [U_["syms"][c] for c in cols]


def runs(U_, specs, tag, **kw):
    out, curves, cache = {}, {}, {}
    for label, r in specs.items():
        e, lg, i19, syms = run_spec(U_, r, cache, **kw)
        out[label] = X.report(e, lg, U_["idx"], i19)
        curves[label] = e.resample("1D").last()
        lg2 = lg.copy()
        lg2["sym"] = [syms[int(k)] for k in lg.k]
        lg2["entry_time"] = U_["idx"][lg.eb.astype(int).values]
        lg2["acct"] = lg.ret * lg.amt / lg.eq0
        lg2.to_csv(os.path.join(X.OUT, f"trades_{tag}_{label}.csv"), index=False)
    pd.DataFrame(curves).to_csv(os.path.join(X.OUT, f"equity_{tag}.csv"))
    return out


def show(label, r, rep, st=None, adv=None, u19=None):
    print(f"\n=== {label}: {name(r)}")
    for p, desc in (("24-26", "(1) 2024-01..2026-10"), ("25-26", "(2) since 2025-01-01"), ("21-23", "(3) 2021-23")):
        sh = (rep.get(p + "_pnl_out19", 0) / rep[p + "_pnl_sum"] * 100) if rep.get(p + "_pnl_sum") else float("nan")
        print(f"  {desc:24s} {rep[p + '_mo'] * 100:+6.2f}%/month  worst drop {rep[p + '_dd'] * 100:5.1f}%  "
              f"x{rep[p + '_x']:.2f}  trades {rep.get(p + '_n', 0)}  worst trade "
              f"{rep.get(p + '_worst_acct', np.nan) * 100:+.1f}% of account  account P&L from coins outside the 19: "
              f"{rep.get(p + '_pnl_out19', 0) * 100:+.0f} of {rep.get(p + '_pnl_sum', 0) * 100:+.0f} points ({sh:.0f}%)")
    print("  years: " + " ".join(f"{y} {rep[f'y{y}'] * 100:+.0f}%" for y in range(2021, 2027)))
    if st is not None:
        print("  (4) fills 0.2% worse + FTT added + LUNA/FTT/SRM always eligible:")
        for p in ("24-26", "25-26", "21-23"):
            print(f"      {p}: {st[p + '_mo'] * 100:+6.2f}%/month  worst drop {st[p + '_dd'] * 100:5.1f}%  "
                  f"x{st[p + '_x']:.3g}  worst trade {st.get(p + '_worst_acct', np.nan) * 100:+.1f}% of account  "
                  f"stuck positions {st.get(p + '_stuck', 0)}")
    if adv is not None:
        print("  watch mode, adversarial in-hour order + 0.3% extra slippage on watch fills: " + X.fmt(adv))
    if u19 is not None:
        print("  same settings on the 19 live coins only: " + X.fmt(u19))


if __name__ == "__main__":
    rnd = sys.argv[1] if len(sys.argv) > 1 else "1"
    Gd = pd.read_csv(os.path.join(X.OUT, {"1": "grid.csv", "2": "grid2.csv", "3": "grid3.csv"}[rnd]))
    D = Gd[SEL_COLS].copy()
    if rnd == "1":
        non19 = [u for u in G.UNIVS if u != "U19"]
        picks = {"A_lever_pick": pick(D, non19, -0.50), "B_safer_pick": pick(D, non19, -0.35),
                 "C_control_19": pick(D, ["U19"], -0.50)}
    elif rnd == "2":
        picks = {"A2_pick": pick(D, G2.UNIVS2, -0.50), "B2_safer_pick": pick(D, G2.UNIVS2, -0.35)}
    else:       # round 3: the grid was placed with hindsight, so this "pick" is NOT honest
        picks = {"A3_rule_on_hindsight_grid": pick(D, list(Gd.univ.unique()), -0.50)}
    print(f"ROUND {rnd} PICKS (2021-23 columns only):")
    for k, r in picks.items():
        print(f"  {k}: {name(r) if r is not None else 'none qualifies'}")
    H = Gd.loc[Gd["24-26_mo"].idxmax()]
    specs = {k: r for k, r in picks.items() if r is not None}
    specs[f"H{rnd}_hindsight_best"] = H
    print(f"  HINDSIGHT best 2024-26 of the grid: {name(H)}")
    best = Gd.sort_values("24-26_mo", ascending=False).groupby(["univ", "mode"]).head(1)
    print("\nHINDSIGHT: best 2024-26 %/month per universe and mode (normal costs)")
    for _, r in best.sort_values(["univ", "mode"]).iterrows():
        print(f"  {r['univ']:9s} {r['mode']:5s} {r['24-26_mo'] * 100:+6.2f}%/mo DD {r['24-26_dd'] * 100:4.0f}% | "
              f"25-26 {r['25-26_mo'] * 100:+5.2f}% | 21-23 {r['21-23_mo'] * 100:+6.2f}% DD {r['21-23_dd'] * 100:4.0f}%  "
              f"({name(r)})")
    print("\nRULE-A PICK PER UNIVERSE (2021-23 rule applied within each universe; then its 2024-26)")
    for u in sorted(Gd.univ.unique()):
        r = pick(D, [u], -0.50)
        if r is None:
            print(f"  {u:9s} none qualifies")
            continue
        g = Gd.loc[r.name]
        print(f"  {u:9s} 21-23 {g['21-23_mo'] * 100:+6.2f}% DD {g['21-23_dd'] * 100:4.0f}% -> 24-26 "
              f"{g['24-26_mo'] * 100:+6.2f}% DD {g['24-26_dd'] * 100:4.0f}% | 25-26 {g['25-26_mo'] * 100:+5.2f}%  ({name(g)})")
    U_ = X.build() if rnd == "1" else G2.build2()
    base = runs(U_, specs, f"r{rnd}_base")
    adv = runs(U_, {k: r for k, r in specs.items() if r["mode"] == "watch"}, f"r{rnd}_adversarial",
               wslip=0.003, worder=1)
    u19 = runs(U_, {k: pd.Series({**r.to_dict(), "univ": "U19"}) for k, r in specs.items() if r["univ"] != "U19"},
               f"r{rnd}_same_on_19")
    del U_
    US = X.build(stress=True) if rnd == "1" else G2.build2(stress=True)
    st = runs(US, specs, f"r{rnd}_stress", force=FORCE, pen=0.002)
    rows = []
    for k, r in specs.items():
        show(k + (" (HINDSIGHT)" if k.startswith("H") else ""), r, base[k], st[k], adv.get(k), u19.get(k))
        rows.append({"label": k, **{c: r[c] for c in KEYS}, **base[k],
                     **{f"stress_{a}": b for a, b in st[k].items()}})
    pd.DataFrame(rows).to_csv(os.path.join(X.OUT, f"picks_r{rnd}.csv"), index=False)
