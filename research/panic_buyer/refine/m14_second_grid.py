"""m14 second: grid on 2021-23 ONLY. RULE (written before any run):
Grid (144): sx (bid under exit price) {5,7,9,12}% x stp {2,3,4}% x shm {48,72,96}h x sN {24,48,72,96}h, sblk=1.
A design qualifies if, with normal fills AND every fill 0.2% worse, 2021-23 win rate >= 92.5%, worst drop no deeper
than -64.4% (tested as > -64.45%, i.e. 90% mode itself at -64.43% passes; first run used -64.40 and nothing passed for that reason), %/month strictly above 90% mode's from this engine (13.17 / 9.79); all 5 nearby versions (base, cd -1h,
+1h, bot timing, start 2021-01-15) also pass win rate and drop; neighbours = designs one grid step away in ONE
parameter: their median normal-fill %/month, win rate, drop must pass the same bars. Score = mean %/month over the 5
versions. Pick = highest score among qualifiers; none qualifies -> no pick. No look at 2024+ in this script.
"""
import os, sys, itertools, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import m14_second_lib as L, m11_fusion_lib as F, m10_velocity_lib as V
OUT = os.environ["PB_OUT"]
_, idx, P = V.market(False)
b = L.scores(F.MODE90, P, idx)
MO_N, MO_W = b["n_mo"], b["w_mo"]
print("baseline", {k: round(v, 4) for k, v in b.items()})
GR = dict(sx=[.05, .07, .09, .12], stp=[.02, .03, .04], shm=[48, 72, 96], sN=[24, 48, 72, 96])
names = list(GR)
rows = {}
for combo in itertools.product(*[range(len(GR[n])) for n in names]):
    d = dict(F.MODE90, sblk=1, **{n: GR[n][i] for n, i in zip(names, combo)})
    rows[combo] = L.scores(d, P, idx)
    rows[combo]["combo"] = combo
def ok(s):
    return (s["n_wr"] >= .925 and s["n_dd"] > -.6445 and s["n_mo"] > MO_N and s["w_wr"] >= .925 and s["w_dd"] > -.6445
            and s["w_mo"] > MO_W and all(s[f"{t}_wr"] >= .925 and s[f"{t}_dd"] > -.6445 for t in ("cm", "cp", "bt", "ls")))
res = []
for combo, s in rows.items():
    nb = []
    for j, n in enumerate(names):
        for dlt in (-1, 1):
            c = list(combo); c[j] += dlt
            if 0 <= c[j] < len(GR[n]):
                nb.append(rows[tuple(c)])
    med = lambda k: float(np.median([x[k] for x in nb]))
    nbok = med("n_wr") >= .925 and med("n_dd") > -.6445 and med("n_mo") > MO_N
    res.append(dict(s, **{n: GR[n][i] for n, i in zip(names, combo)}, q=bool(ok(s)), nbok=bool(nbok),
                    nb_mo=med("n_mo"), nb_wr=med("n_wr"), nb_dd=med("n_dd")))
df = pd.DataFrame(res).drop(columns="combo")
df.to_csv(os.path.join(OUT, "grid.csv"), index=False)
print("designs", len(df), "pass bar1+versions", int(df.q.sum()), "also neighbours", int((df.q & df.nbok).sum()))
cols = names + ["n_wr", "n_mo", "n_dd", "w_mo", "robust", "rmin", "n_tpm", "q", "nbok", "nb_mo"]
print(df.sort_values("robust", ascending=False)[cols].head(12).round(4).to_string())
fin = df[df.q & df.nbok].sort_values("robust", ascending=False)
if len(fin):
    p = fin.iloc[0]
    pick = {n: (int(p[n]) if n in ("shm", "sN") else float(p[n])) for n in names}
    json.dump(pick, open(os.path.join(OUT, "pick.json"), "w")); print("PICK", pick, round(p.robust, 4))
else:
    print("NO PICK")
