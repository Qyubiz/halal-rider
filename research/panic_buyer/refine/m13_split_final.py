"""m13 split final look. The pre-written rule produced NO pick (see grid log). This runs, labelled FALLBACK (not a
pass candidate), the single design that met the bars without the neighbour test, next to 90% mode, plus the share of
its grid neighbours passing items 1-3 (hindsight-labelled)."""
import json
import os
import m13_split_lib as S
import m13_split_grid as G

pick = json.load(open(os.path.join(S.M.OUT, "pick.json")))
d = pick["pick"] or pick["fallback"][0]
print("design (FALLBACK)" if not pick["pick"] else "design (PICK)", json.dumps(d))
B = S.full(((0.14, 1.0),), 0.025)
R = S.full(G.legs(d), d["tp"])
for run in ("normal", "worse", "stress"):
    for p, pl in (("d", "2021-23"), ("t", "2024-26"), ("s", "since 2025"), ("h", "2018-20")):
        if (run, p) not in R:
            continue
        f = lambda v: f"{v['wr']*100:5.1f} {v['mo']*100:+6.2f} {v['dd']*100:6.1f} {v['wt']*100:6.1f} {v['tpm']:5.1f}"
        print(f"{run:7s} {pl:10s} split {f(R[(run, p)])} | 90% {f(B[(run, p)])}")
print("items 1,2,3 (pick):", S.passes(R))
st = R[("stress", "d")]; print("item4: stress 24-26 mo", R[("stress", "t")]["mo"] * 100, "stress 21-23 mult", st["mult"])
grid = json.load(open(os.path.join(S.M.OUT, "grid.json")))
nb = G.neighbours(d, grid)
c = 0
for e in nb:
    p = S.passes(S.full(G.legs(e), e["tp"])); c += all(p); print(e, p)
print(f"neighbours passing items 1-3: {c}/{len(nb)}")
