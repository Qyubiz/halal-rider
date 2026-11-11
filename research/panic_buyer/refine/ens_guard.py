"""Agent 3 (b): execution-level guardrails on the architecture picked by ens_designs.py (2021-23 rule): two sleeves,
30% on 90% mode and 70% on 10% mode, free cash rebalanced monthly. Same guardrails also on 90% mode alone, for reference.

Step 1, one guardrail at a time (cost/benefit vs no guardrail):
  kill-switch daily X in {10,15,20}% (lock to the next UTC day) / rolling 30-day Y in {20,25,30}% (lock 7 days);
  hard stop S in {10,15,20,25}% (resting stop, gaps fill at the open; S 15/25 also as an hourly-close check);
  per-trade risk cap Z in {5,10}% of the account with each S (orders cut so money x S <= Z x equity);
  consecutive-loss cooldown N in {2,3} losses -> T in {24,72} h pause; ATR-spike entry block k in {3,5}.
Step 2, combined grid on the architecture: X {10,15,20} x Y {20,25,30} x S {off,15,25} x Z {off,10%} (Z only with S)
  x (N,T) {off,(2,72),(3,24)} = 135 configs.
Pick rule, fixed before any result was seen (2021-23 only): keep configs whose 2021-23 %/month is >= 70% of the
unguarded architecture's on BOTH panels; among them the shallowest 2021-23 max drop (worse panel); drops within 1
point -> higher %/month (worse panel). 2024-26 / since 2025 then reported once, plus a stress run (real bot timing,
every fill 0.2% worse, 22 coins).

Usage: PB_DATA=... PB_OUT=... python3 ens_guard.py
"""
import itertools
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.environ.setdefault("PB_OUT", os.path.expanduser("~/pb_out_ens/research"))
os.makedirs(OUT, exist_ok=True)
import ens_lib as L  # noqa: E402

ARCH = ([L.MODE90, L.MODE10], dict(weights=[0.3, 0.7], rb=1))
BASES = {"ARCH 30/70": ARCH, "90% mode": ([L.MODE90], {})}

SINGLE = {"none": {}}
for x in (0.10, 0.15, 0.20):
    SINGLE[f"kill daily {x*100:.0f}%"] = dict(killd=x)
for y in (0.20, 0.25, 0.30):
    SINGLE[f"kill 30d {y*100:.0f}%"] = dict(killm=y)
for s in (0.10, 0.15, 0.20, 0.25):
    SINGLE[f"stop {s*100:.0f}%"] = dict(hs=s)
    for z in (0.05, 0.10):
        SINGLE[f"stop {s*100:.0f}% + risk cap {z*100:.0f}%"] = dict(hs=s, zcap=z)
for s in (0.15, 0.25):
    SINGLE[f"stop {s*100:.0f}% close-check"] = dict(hs=s, hsmode=0)
for n in (2, 3):
    for tt in (24, 72):
        SINGLE[f"cooldown {n} losses -> {tt}h"] = dict(cln=n, clT=tt)
for k in (3, 5):
    SINGLE[f"ATR block {k}x"] = dict(atr_k=k)


def run_set(base_name, configs, tag):
    sv, kw0 = BASES[base_name]
    rows = []
    for name, kw in configs.items():
        for r in L.run_both(name, sv, **dict(kw0, **kw)):
            r["base"] = base_name
            r["cfg"] = repr(kw)
            rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, f"{tag}.csv"), index=False)
    return R


def costben(R):
    """Per guardrail: change vs 'none' in %/mo and in max drop (points), per period, worse of the two panels."""
    out = []
    for name, g in R.groupby("design", sort=False):
        row = {"guardrail": name}
        for p in ("21-23", "24-26", "25-"):
            mo = g[f"{p} %mo"].min(); dd = g[f"{p} DD"].min(); wm = g[f"{p} worstmo"].min()
            row.update({f"{p} %mo": mo, f"{p} DD": dd, f"{p} worstmo": wm,
                        f"{p} WR": g[f"{p} WR"].min(), f"{p} PF": g[f"{p} PF"].min()})
        row["kills"] = int(g.kills_d.max() + g.kills_m.max())
        out.append(row)
    C = pd.DataFrame(out)
    b = C.iloc[0]
    for p in ("21-23", "24-26", "25-"):
        C[f"{p} d%mo"] = C[f"{p} %mo"] - b[f"{p} %mo"]
        C[f"{p} dDD"] = C[f"{p} DD"] - b[f"{p} DD"]
    return C


def show_cb(C):
    cols = []
    for p in ("21-23", "24-26", "25-"):
        cols += [f"{p} %mo", f"{p} WR", f"{p} DD", f"{p} PF", f"{p} worstmo", f"{p} d%mo", f"{p} dDD"]
    return C[["guardrail"] + cols + ["kills"]].to_string(index=False, float_format=lambda v: f"{v:6.1f}")


pd.set_option("display.width", 400)
print("=== Step 1: one guardrail at a time. Each cell = WORSE of the 19-coin and 22-coin panels; d = change vs none")
for b in BASES:
    R = run_set(b, SINGLE, f"guard_single_{b.split()[0].replace('%', '')}")
    print(f"\n--- base: {b}")
    print(show_cb(costben(R)))
    print("\nper panel:")
    print(L.table(R))

print("\n=== Step 2: combined grid on ARCH 30/70")
GRID = {}
for x, y, s, z, nt in itertools.product((0.10, 0.15, 0.20), (0.20, 0.25, 0.30), (0.0, 0.15, 0.25), (0.0, 0.10),
                                         ((0, 0), (2, 72), (3, 24))):
    if s == 0.0 and z > 0:
        continue
    kw = dict(killd=x, killm=y, hs=s, zcap=z, cln=nt[0], clT=nt[1])
    GRID[f"X{x*100:.0f} Y{y*100:.0f} S{s*100:.0f} Z{z*100:.0f} N{nt[0]} T{nt[1]}"] = kw
R0 = run_set("ARCH 30/70", {"none": {}}, "guard_none")
RG = run_set("ARCH 30/70", GRID, "guard_grid")
w = lambda R, c: R.groupby("design", sort=False)[c].min()  # noqa: E731
base_mo = R0.set_index("panel")["21-23 %mo"]
ok = RG.groupby("design", sort=False).apply(
    lambda g: all(g.set_index("panel").loc[p, "21-23 %mo"] >= 0.7 * base_mo[p] for p in ("19", "22")))
S = pd.DataFrame({"mo": w(RG, "21-23 %mo"), "dd": w(RG, "21-23 DD"), "ok": ok})
print(f"{len(GRID)} configs x 2 panels = {len(RG)} runs; unguarded 2021-23 %/mo: 19 coins {base_mo['19']:.2f}, "
      f"22 coins {base_mo['22']:.2f}; feasible (>= 70% of it on both): {int(S.ok.sum())}")
F = S[S.ok].copy()
best_dd = F.dd.max()
cand = F[F.dd >= best_dd - 1.0]
pick = cand.mo.idxmax()
print("\nfeasible configs, 2021-23 (worse panel), top 15 by drop:")
print(F.sort_values(["dd", "mo"], ascending=False).head(15).to_string(float_format=lambda v: f"{v:6.2f}"))
print(f"\nPICK (2021-23 only): {pick}  {GRID[pick]}")
P = RG[RG.design == pick]
print(L.table(pd.concat([R0, P])))
with open(os.path.join(OUT, "guard_pick.txt"), "w") as f:
    f.write(f"{pick}\n{GRID[pick]}\n")
