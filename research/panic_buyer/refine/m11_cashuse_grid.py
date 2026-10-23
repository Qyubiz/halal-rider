"""m11 cashuse search on 2021-23 ONLY (no 2024-26, no since-2025, no holdout number is computed here).

SELECTION RULE (written before any result of this search was seen):
  Required in 2021-23 (19 coins, normal fills): pooled trade win rate >= 93.8% (90% mode: 93.8%) and worst drop no
  deeper than -64% (rounded to whole percent; 90% mode -64.4%). The same must hold with every buy and sell fill
  0.2% worse (literal reading: win rate >= 93.8%, drop >= -64%). If no design passes the literal reading, the
  fallback (also fixed now) is "no worse than 90% mode with worse fills": win rate >= 93.4%, drop >= -65.9%.
  Among the designs that pass: highest 2021-23 %/month averaged over 5 versions: as designed, all cooldowns -1h,
  all cooldowns +1h, the real bot timing (mtp_now: take-profit at once after a market buy), start 2 weeks later
  (2021-01-15). Ties (within 0.05 %/mo): fewer parameters, then the higher worse-fill win rate.
  Then ONE look at 2024-26, since 2025, the 2018-20 holdout, worse fills and the stressed run (m11_cashuse_final.py).
  No re-picking after that look.
Stages (at most ~300 designs in all, counted in the log):
  A  second engine alone (core off): which S2 rules have a high win rate on their own;
  B  core + S2 with the cash rules (armx, pm, watch, size, pause after a core loss, cut after an S2 loss);
  C  the 5 versions of every design that passes the requirements.
Usage: python m11_cashuse_grid.py A|B|C
"""
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import m10_velocity_lib as V
import m11_cashuse_lib as C

OUT = C.OUT
BIG5 = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
BIG2 = ["BTCUSDT", "ETHUSDT"]
D_END = pd.Timestamp("2024-01-01")


def run(P, idx, kw, worse=False):
    k = dict(kw)
    if worse:
        k.update(pen=C.STRESS_PEN, pens=C.STRESS_PEN)
    e, lg, inv = C.crun(P, idx, **k)
    e = e[e.index < D_END]
    lg = lg[idx[lg.eb.astype(int).values] < D_END]
    inv = inv[inv.index < D_END]
    r = C.summarize(e, lg, inv, idx[idx < D_END], (("d", C.T21, None),))
    return r


def row(name, kw, r, rw=None):
    x = dict(name=name, kw=json.dumps(kw), wr=r["d_wr"], mo=r["d_mo"], dd=r["d_dd"], wt=r["d_wt"], n=r["d_n"],
             n1=r.get("d_n1", 0), wr1=r.get("d_wr1", np.nan), n0=r.get("d_n0", 0), wr0=r.get("d_wr0", np.nan))
    if rw is not None:
        x.update(wr_w=rw["d_wr"], mo_w=rw["d_mo"], dd_w=rw["d_dd"])
    return x


def passes(x, literal=True):
    a = x["wr"] >= 0.9375 and round(x["dd"] * 100) >= -64
    if literal:
        return a and x["wr_w"] >= 0.9375 and round(x["dd_w"] * 100) >= -64
    return a and x["wr_w"] >= 0.9335 and x["dd_w"] >= -0.6595


def stage_a():
    """S2 alone (core off), 2021-23."""
    syms, idx, P = V.market(False)
    rows = []
    fam = []
    for d2, tp2, hm2 in itertools.product((0.06, 0.08, 0.10, 0.12), (0.01, 0.015, 0.02), (48, 96)):
        fam.append(("A1 24h all", dict(ref=24, d2=d2, tp2=tp2, hm2=hm2, cd2=24, fs2=0.2, n2=5)))
    for d2, tp2, hm2 in itertools.product((0.03, 0.05, 0.07), (0.01, 0.015), (48, 168)):
        fam.append(("A2 24h big5", dict(ref=24, d2=d2, tp2=tp2, hm2=hm2, cd2=24, fs2=0.2, n2=5, coins=BIG5)))
    for d2, tp2 in itertools.product((0.06, 0.08, 0.10), (0.015, 0.02)):
        fam.append(("A3 6h all", dict(ref=6, d2=d2, tp2=tp2, hm2=48, cd2=24, fs2=0.2, n2=5)))
    for d2, tp2 in itertools.product((0.06, 0.08, 0.10), (0.01, 0.015)):
        fam.append(("A4 24h all sma200", dict(ref=24, d2=d2, tp2=tp2, hm2=48, cd2=24, fs2=0.2, n2=5, sma=200)))
    for d2, tp2 in itertools.product((0.06, 0.08, 0.10), (0.01, 0.015)):
        fam.append(("A5 24h all BTC calm 3%", dict(ref=24, d2=d2, tp2=tp2, hm2=48, cd2=24, fs2=0.2, n2=5,
                                                   calm=("BTCUSDT", 0.03, 24))))
    for name, s2 in fam:
        kw = dict(s2=s2, core_on=0)
        r = run(P, idx, kw)
        rows.append(row(name, s2, r))
        x = rows[-1]
        print(f"{name:24s} {json.dumps({k: v for k, v in s2.items() if k not in ('coins',)}):110s} WR {x['wr']*100:5.1f}% "
              f"{x['mo']*100:+6.2f}%/mo dd {x['dd']*100:6.1f}% worst {x['wt']*100:+6.1f}% n {x['n']}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "grid_a.csv"), index=False)
    print(f"stage A designs: {len(df)}")


S2C = {
    "S2a 24h-12% +1% 96h": dict(ref=24, d2=0.12, tp2=0.01, hm2=96, cd2=24),
    "S2b 24h-12% +1% 48h": dict(ref=24, d2=0.12, tp2=0.01, hm2=48, cd2=24),
    "S2c 24h-10% +1% 96h": dict(ref=24, d2=0.10, tp2=0.01, hm2=96, cd2=24),
    "S2d 24h-10% +1% 48h": dict(ref=24, d2=0.10, tp2=0.01, hm2=48, cd2=24),
    "S2e 6h-10% +1.5% 48h": dict(ref=6, d2=0.10, tp2=0.015, hm2=48, cd2=24),
    "S2f 24h-12% +1.5% 96h": dict(ref=24, d2=0.12, tp2=0.015, hm2=96, cd2=24),
}
NS = 0


def show(x, ok):
    print(f"{x['name']:58s} WR {x['wr']*100:5.1f}% {x['mo']*100:+6.2f}%/mo dd {x['dd']*100:6.1f}% wt {x['wt']*100:+6.1f}% "
          f"| worse WR {x['wr_w']*100:5.1f}% {x['mo_w']*100:+6.2f}%/mo dd {x['dd_w']*100:6.1f}% | core {x['n0']} "
          f"{x['wr0']*100:4.1f}% S2 {x['n1']} {x['wr1']*100:4.1f}% {'PASS' if ok else ''}", flush=True)


def evalrow(P, idx, name, kw):
    r = run(P, idx, kw)
    rw = run(P, idx, kw, worse=True)
    x = row(name, kw, r, rw)
    x["pass"] = passes(x); x["pass_fb"] = passes(x, literal=False)
    show(x, x["pass"])
    return x


def stage_b1():
    syms, idx, P = V.market(False)
    rows = []
    for (sn, s2), pm, armx, (fs2, n2) in itertools.product(S2C.items(), (0, 1, 2), (0.02, 0.05), ((0.1, 5), (0.2, 5))):
        s = dict(s2, fs2=fs2, n2=n2)
        kw = dict(s2=s, pm=pm, armx=armx, watch=0)
        rows.append(evalrow(P, idx, f"B1 {sn} pm{pm} armx{armx} {fs2}x{n2}", kw))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "grid_b1.csv"), index=False)
    print(f"stage B1 designs: {len(df)}, pass literal {int(df['pass'].sum())}, pass fallback {int(df['pass_fb'].sum())}")


def stage_b2():
    """Small refinement around the B1 passers (cash rules only; S2 rules from stage A)."""
    syms, idx, P = V.market(False)
    c, d, a = S2C["S2c 24h-10% +1% 96h"], S2C["S2d 24h-10% +1% 48h"], S2C["S2a 24h-12% +1% 96h"]
    L = [("S2c pm0 armx0.08 0.2x5", dict(s2=dict(c, fs2=0.2, n2=5), pm=0, armx=0.08)),
         ("S2c pm0 armx0.05 0.2x5 watch", dict(s2=dict(c, fs2=0.2, n2=5), pm=0, armx=0.05, watch=1)),
         ("S2c pm0 armx0.02 0.2x5 watch", dict(s2=dict(c, fs2=0.2, n2=5), pm=0, armx=0.02, watch=1)),
         ("S2c pm0 armx0.05 0.25x4", dict(s2=dict(c, fs2=0.25, n2=4), pm=0, armx=0.05)),
         ("S2c pm0 armx0.05 0.15x6", dict(s2=dict(c, fs2=0.15, n2=6), pm=0, armx=0.05)),
         ("S2c pm0 armx0.05 0.33x3", dict(s2=dict(c, fs2=0.33, n2=3), pm=0, armx=0.05)),
         ("S2c pm0 armx0.05 0.2x3", dict(s2=dict(c, fs2=0.2, n2=3), pm=0, armx=0.05)),
         ("S2c pm0 armx0.05 0.2x5 pause48", dict(s2=dict(c, fs2=0.2, n2=5), pm=0, armx=0.05, pause_cl=48)),
         ("S2c pm0 armx0.05 0.2x5 cut0.5", dict(s2=dict(c, fs2=0.2, n2=5), pm=0, armx=0.05, cut_s2=0.5)),
         ("S2c pm1 armx0.08 0.2x5", dict(s2=dict(c, fs2=0.2, n2=5), pm=1, armx=0.08)),
         ("S2d pm0 armx0.08 0.2x5", dict(s2=dict(d, fs2=0.2, n2=5), pm=0, armx=0.08)),
         ("S2a pm0 armx0.08 0.2x5", dict(s2=dict(a, fs2=0.2, n2=5), pm=0, armx=0.08))]
    rows = [evalrow(P, idx, "B2 " + nm, kw) for nm, kw in L]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "grid_b2.csv"), index=False)
    print(f"stage B2 designs: {len(df)}, pass literal {int(df['pass'].sum())}, pass fallback {int(df['pass_fb'].sum())}")


VERSIONS = (("as designed", {}), ("cooldowns -1h", dict(cd_add=-1)), ("cooldowns +1h", dict(cd_add=1)),
            ("bot timing", dict(mtp_now=True)), ("start 2021-01-15", dict(t0=pd.Timestamp("2021-01-15"))))


def stage_c():
    """5 versions of every design that passes (literal rule; fallback rule only if none passes literally)."""
    syms, idx, P = V.market(False)
    df = pd.concat([pd.read_csv(os.path.join(OUT, f)) for f in ("grid_b1.csv", "grid_b2.csv")], ignore_index=True)
    print(f"designs searched: A {len(pd.read_csv(os.path.join(OUT, 'grid_a.csv')))} + B {len(df)}")
    ok = df[df["pass"]]
    rule = "literal"
    if ok.empty:
        ok, rule = df[df["pass_fb"]], "fallback"
    print(f"{len(ok)} designs pass ({rule} rule)")
    out = []
    for _, x in ok.iterrows():
        kw = json.loads(x["kw"])
        mos = []
        for vn, v in VERSIONS:
            r = run(P, idx, dict(kw, **v))
            mos.append(r["d_mo"])
        out.append(dict(name=x["name"], kw=x["kw"], mo=x["mo"], wr=x["wr"], wr_w=x["wr_w"], dd=x["dd"], dd_w=x["dd_w"],
                        avg5=float(np.mean(mos)), versions=" ".join(f"{m*100:+.2f}" for m in mos)))
        print(f"{x['name']:52s} avg of 5 versions {np.mean(mos)*100:+6.2f}%/mo [{out[-1]['versions']}] WR {x['wr']*100:.1f} "
              f"worse WR {x['wr_w']*100:.1f} dd {x['dd']*100:.1f} worse dd {x['dd_w']*100:.1f}", flush=True)
    res = pd.DataFrame(out).sort_values("avg5", ascending=False)
    res.to_csv(os.path.join(OUT, "grid_c.csv"), index=False)
    print("PICK:", res.iloc[0]["name"], res.iloc[0]["kw"])


if __name__ == "__main__":
    t = time.time()
    {"A": stage_a, "B1": stage_b1, "B2": stage_b2, "C": stage_c}[sys.argv[1]]()
    print(f"{time.time()-t:.0f}s")
