"""m10 exits, step 2: exit grids on 2021-23 ONLY (the panel is cut at 2024-01-01, so nothing later is seen).

Entry fixed at 90% mode (1 coin x 100%, 14% under the 24h-high close, crash guard 35%, cooldown 3h).
Single-coin all-in results are very path dependent (m10_exits_check.py: delaying one cooldown by an hour moves 2021-26
from x807 to x1409), so every design is run on 5 entry paths -- the pick's own path (cd3, 14%) and four near copies
(cooldown 2h / 4h, depth 13.9% / 14.1%) -- and scored on the average. Each path also runs with every fill 0.2% worse.

SELECTION RULE (written before any grid result was seen):
  eligible = path-average 2021-23 win rate >= 90% mode's path-average win rate,
             path-average worst drop no more than 2 points deeper than 90% mode's,
             path-average %/month with fills 0.2% worse >= 90% mode's (same stress);
  pick     = the eligible design with the highest path-average 2021-23 %/month.
Usage: PB_DATA=... PB_OUT=... python3 m10_exits_grid.py g1|g2
"""
import itertools
import sys
import time

import numpy as np
import pandas as pd

import m10_exits_lib as X
import w90_wr90_lib as W

T_END = pd.Timestamp("2024-01-01")
PATHS = [dict(cd=3, depth=0.14), dict(cd=2, depth=0.14), dict(cd=4, depth=0.14), dict(cd=3, depth=0.139),
         dict(cd=3, depth=0.141)]
DPER = (("d", pd.Timestamp("2021-01-01"), T_END),)


def design_panel():
    syms, idx, P = W.market(False)
    keep = idx < T_END
    Pd = {k: (v[keep] if isinstance(v, pd.DataFrame) else v) for k, v in P.items()}
    return syms, idx[keep], Pd


def g1():
    D = [("base", "90% mode", {})]
    for a1, be in itertools.product((6, 12, 24, 36, 48, 72), (-0.01, -0.005, 0.0, 0.0025, 0.005, 0.01)):
        D.append(("A_bestep", f"after {a1}h target -> {be*100:+.2f}%", dict(a1=a1, be=be)))
    for a1, span, be in itertools.product((1, 6, 12, 24), (24, 48, 72), (-0.01, 0.0025, 0.01)):
        D.append(("B_decay", f"from {a1}h linear to {be*100:+.2f}% at {a1+span}h", dict(a1=a1, a2=a1 + span, be=be)))
    for hm in (48, 72, 120, 144, 192):
        D.append(("C_hm", f"plain time stop {hm}h", dict(hm=hm)))
    for hm, uw, hm2, be2 in itertools.product((72, 96), (0.05, 0.10, 0.15, 0.20), (168, 240), (0.0025, 0.01, 0.025)):
        D.append(("C_deep", f"at {hm}h keep if >{uw*100:.0f}% under, target {be2*100:+.2f}%, stop {hm2}h",
                  dict(hm=hm, uw=uw, uwmode=0, hm2=hm2, be2=be2)))
    for hm, uw, hm2, be2 in itertools.product((48, 72, 96), (0.03, 0.06, 0.10), (144, 240), (0.0025, 0.01)):
        D.append(("C_shallow", f"at {hm}h keep if <={uw*100:.0f}% under, target {be2*100:+.2f}%, stop {hm2}h",
                  dict(hm=hm, uw=uw, uwmode=1, hm2=hm2, be2=be2)))
    for part, tr, fl, sm, hmr in itertools.product((0.25, 0.5, 0.75), (0.02, 0.04, 0.08), (0.0, 0.01), (0, 1), (24, 96)):
        D.append(("D_parttrail", f"sell {part:.0%} at +2.5%, rest trail {tr:.0%} floor {fl*100:+.0f}% "
                  f"{'stop order' if sm else 'hourly close'} max {hmr}h",
                  dict(part=part, tr=tr, fl=fl, stopmode=sm, hmr=hmr)))
    for tr, fl, sm, hmr in itertools.product((0.005, 0.01, 0.02, 0.03, 0.05), (0.01, 0.015, 0.02), (0, 1), (24, 96)):
        D.append(("D_armtrail", f"+2.5% arms trail {tr*100:.1f}% floor {fl*100:+.1f}% "
                  f"{'stop order' if sm else 'hourly close'} max {hmr}h",
                  dict(part=0.0, tr=tr, fl=fl, stopmode=sm, hmr=hmr)))
    for part, tp2, fl, hmr in itertools.product((0.25, 0.5, 0.75), (0.04, 0.06, 0.10), (-2.0, 0.0), (48, 96)):
        D.append(("D_parttp2", f"sell {part:.0%} at +2.5%, rest at +{tp2:.0%} "
                  f"{'floor 0% stop order' if fl > -1 else 'no floor'} max {hmr}h",
                  dict(part=part, tp2=tp2, fl=fl, stopmode=1, hmr=hmr)))
    for mode, base, alpha in itertools.product(("vol7", "rng24", "dd7"), (0.02, 0.025, 0.03), (0.5, 1.0)):
        D.append(("E_scaled", f"tp {base*100:.1f}% x ({mode}/median)^{alpha} in [1%, 6%]",
                  dict(tpspec=(mode, base, alpha, 0.01, 0.06))))
    for tpd in (0.25, 0.5, 1.0):
        D.append(("E_mktdepth", f"tp 2.5% + {tpd} x extra depth of a market buy", dict(tpd=tpd)))
    for tp in (0.02, 0.0225, 0.0275, 0.03):
        D.append(("E_fix", f"fixed tp {tp*100:.2f}%", dict(tp=tp)))
    return D


def g2():
    """Written after g1 (g1 pick: at 72h keep a trade that is no more than 10% under, target +0.25%, stop 240h).
    The neighbourhood of that rule, and its combinations with the other g1 families' best ideas."""
    D = [("base", "90% mode", {})]
    for hm, uw, hm2, be2 in itertools.product((60, 72, 84), (0.08, 0.10, 0.12, 0.15), (168, 240, 336), (0.005, 0.01, 0.02)):
        D.append(("G_shallow", f"at {hm}h keep if <={uw*100:.0f}% under, target {be2*100:+.2f}%, stop {hm2}h",
                  dict(hm=hm, uw=uw, uwmode=1, hm2=hm2, be2=be2)))
    cores = {"72/10/240/1": dict(hm=72, uw=0.10, uwmode=1, hm2=240, be2=0.01),
             "72/10/240/0.25": dict(hm=72, uw=0.10, uwmode=1, hm2=240, be2=0.0025),
             "72/12/240/1": dict(hm=72, uw=0.12, uwmode=1, hm2=240, be2=0.01),
             "84/10/240/1": dict(hm=84, uw=0.10, uwmode=1, hm2=240, be2=0.01)}
    tps = {"rng24 2.5%^0.5": ("rng24", 0.025, 0.5), "rng24 2.75%^0.5": ("rng24", 0.0275, 0.5),
           "rng24 3%^0.5": ("rng24", 0.03, 0.5), "dd7 2%^1": ("dd7", 0.02, 1.0), "vol7 2.5%^0.5": ("vol7", 0.025, 0.5)}
    for (cn, c), (tn, (m, b, a)) in itertools.product(cores.items(), tps.items()):
        D.append(("H_core+tp", f"core {cn} + tp {tn}", dict(c, tpspec=(m, b, a, 0.01, 0.06))))
    for (cn, c), tp in itertools.product(cores.items(), (0.0225, 0.0275, 0.03)):
        D.append(("H_core+fix", f"core {cn} + fixed tp {tp*100:.2f}%", dict(c, tp=tp)))
    for (cn, c), (a1, be) in itertools.product(list(cores.items())[:2], ((24, 0.005), (24, 0.01), (48, 0.005), (48, 0.01))):
        D.append(("H_core+sched", f"core {cn} + after {a1}h target {be*100:+.2f}%", dict(c, a1=a1, be=be)))
    for (cn, c), (part, tr) in itertools.product(list(cores.items())[:2], ((0.75, 0.02), (0.75, 0.04), (0.5, 0.02))):
        D.append(("H_core+trail", f"core {cn} + sell {part:.0%} at tp, rest trail {tr:.0%} floor +1% stop order 24h",
                  dict(c, part=part, tr=tr, fl=0.01, stopmode=1, hmr=24)))
    for (cn, c), tpd in itertools.product(list(cores.items())[:2], (0.25, 0.5)):
        D.append(("H_core+mkt", f"core {cn} + tp + {tpd} x market-buy extra depth", dict(c, tpd=tpd)))
    return D


GRIDS = {"g1": g1, "g2": g2}


def run_design(P, idx, kw, path, stress):
    k = dict(X.BASE90)
    k.update(path)
    kw = dict(kw)
    spec = kw.pop("tpspec", None)
    if spec is not None:
        mode, base, alpha, lo, hi = spec
        kw["tpa"] = X.tp_table(P, mode, base, alpha=alpha, lo=lo, hi=hi)
    k.update(kw)
    if stress:
        k.update(pen=X.STRESS_PEN, pens=X.STRESS_PEN)
    e, lg = X.xrun(P, idx, **k)
    return X.summarize(e, lg, idx, DPER)


def main(name):
    syms, idx, P = design_panel()
    print(f"design panel {idx[0]} -> {idx[-1]} ({len(idx)} hours, {len(syms)} coins)", flush=True)
    designs = GRIDS[name]()
    print(f"{name}: {len(designs)} designs x {len(PATHS)} paths x (normal, fills 0.2% worse)", flush=True)
    rows = []
    t0 = time.time()
    for i, (fam, lab, kw) in enumerate(designs):
        R = [run_design(P, idx, kw, p, False) for p in PATHS]
        S = [run_design(P, idx, kw, p, True) for p in PATHS]
        rows.append(dict(fam=fam, design=lab, kw=repr(kw),
                         mo=np.mean([r["d_mo"] for r in R]), mo_min=min(r["d_mo"] for r in R),
                         wr=np.mean([r["d_wr"] for r in R]), wr_min=min(r["d_wr"] for r in R),
                         dd=np.mean([r["d_dd"] for r in R]), dd_min=min(r["d_dd"] for r in R),
                         tpm=np.mean([r["d_tpm"] for r in R]), tiny=np.mean([r["d_tiny"] for r in R]),
                         wt=np.mean([r["d_wt"] for r in R]),
                         s_mo=np.mean([r["d_mo"] for r in S]), s_wr=np.mean([r["d_wr"] for r in S]),
                         s_dd=np.mean([r["d_dd"] for r in S]),
                         p0_mo=R[0]["d_mo"], p0_wr=R[0]["d_wr"], p0_dd=R[0]["d_dd"], p0_wt=R[0]["d_wt"],
                         p0_n=R[0]["d_n"]))
        if i % 25 == 0:
            print(f"  {i}/{len(designs)} {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{X.OUT}/grid_{name}.csv", index=False)
    base = df[df.fam == "base"].iloc[0] if (df.fam == "base").any() else None
    report(df, base, name)


def eligible(df, base):
    """g1 rule; from g2 on also: the win rate must hold with fills 0.2% worse (stressed path-average win rate >= 90%
    mode's), so tiny +0.05% 'wins' that a worse fill turns into losses can't carry the pick (added before g2 ran)."""
    ok = (df.wr >= base.wr) & (df.dd >= base.dd - 0.02) & (df.s_mo >= base.s_mo) & (df.fam != "base")
    return df[ok & (df.s_wr >= base.s_wr)]


def report(df, base, name):
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70); pd.set_option("display.max_rows", 400)
    cols = ["fam", "design", "mo", "mo_min", "wr", "wr_min", "dd", "s_mo", "s_wr", "tpm", "tiny", "wt", "p0_mo", "p0_wr",
            "p0_dd"]
    f = lambda v: f"{v:.4f}"  # noqa: E731
    print(f"\nBASE 90% mode (path average): {base.mo*100:+.2f}%/mo (min path {base.mo_min*100:+.2f}), win {base.wr*100:.2f}%, "
          f"drop {base.dd*100:.1f}%, stressed {base.s_mo*100:+.2f}%/mo; own path {base.p0_mo*100:+.2f}%/mo "
          f"win {base.p0_wr*100:.1f}% drop {base.p0_dd*100:.1f}%")
    print("\nbest 3 per family by path-average %/month:")
    print(df.sort_values("mo", ascending=False).groupby("fam").head(3).sort_values(["fam", "mo"], ascending=[True, False])
          [cols].to_string(index=False, float_format=f))
    print("\nbest 3 per family by path-average win rate:")
    print(df.sort_values("wr", ascending=False).groupby("fam").head(3).sort_values(["fam", "wr"], ascending=[True, False])
          [cols].to_string(index=False, float_format=f))
    el = eligible(df, base)
    print(f"\nELIGIBLE under the rule: {len(el)} of {len(df)-1}")
    print(el.sort_values("mo", ascending=False).head(30)[cols].to_string(index=False, float_format=f))
    # Pareto front win rate vs %/mo (path averages)
    d = df.sort_values("wr", ascending=False)
    front, best = [], -9
    for r in d.itertuples():
        if r.mo > best:
            front.append(r.Index); best = r.mo
    print("\nPareto front (path-average win rate vs %/month):")
    print(df.loc[front][cols].to_string(index=False, float_format=f))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "g1")
