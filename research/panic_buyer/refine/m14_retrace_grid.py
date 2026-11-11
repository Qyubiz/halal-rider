"""m14 retrace grid. RULE (written before any run), 2021-23 data only:
 design = (r, glo, ghi); r in .10,.15,.20,.25,.30,.35,.40,.45 ; glo in 1.5,2.0,2.5% ; ghi in 3,4,5,6% (ghi>=glo). Entry fixed (24h x 0.86).
 1. normal and every-fill-0.2%-worse 2021-23: win rate >= 92.5%, worst drop > -64.4%, %/month > 13.17 normal / 9.79 worse.
 2. score = mean %/month over 5 nearby versions (base, cd-1, cd+1, bot timing, start 2021-01-15); all 5 pass wr>=92.5, dd>-64.4.
 3. neighbours (one grid step in one parameter): median normal wr, dd, %/month pass bar 1 (normal).
 Pick = highest score among those passing 1-3. None if none pass.
"""
import os, sys, json, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import m14_retrace_lib as L, m11_fusion_lib as F

RS = [.10, .15, .20, .25, .30, .35, .40, .45]; LOS = [.015, .02, .025]; HIS = [.03, .04, .05, .06]
grid = [(r, lo, hi) for r in RS for lo in LOS for hi in HIS if hi >= lo]
res = {}
for g in grid:
    d = dict(F.MODE90, rr=g[0], glo=g[1], ghi=g[2])
    res[g] = L.design_scores_r(d)
def ok(s):
    return (s["n_wr"] >= .925 and s["n_dd"] > -.644 and s["w_wr"] >= .925 and s["w_dd"] > -.644 and s["n_mo"] > .1317 and s["w_mo"] > .0979)
def ok5(s):
    return all(s[f"{t}_wr"] >= .925 and s[f"{t}_dd"] > -.644 for t in ("n", "cm", "cp", "bt", "ls"))
def nbrs(g):
    r, lo, hi = g; out = []
    for (A, i) in ((RS, 0), (LOS, 1), (HIS, 2)):
        j = A.index(g[i])
        for jj in (j - 1, j + 1):
            if 0 <= jj < len(A):
                h = list(g); h[i] = A[jj]; h = tuple(h)
                if h in res: out.append(h)
    return out
rows = []
for g in grid:
    s = res[g]; nb = nbrs(g)
    med = lambda k: float(np.median([res[h][k] for h in nb]))
    nbok = med("n_wr") >= .925 and med("n_dd") > -.644 and med("n_mo") > .1317
    p = ok(s) and ok5(s) and nbok
    rows.append((g, s["robust"], p, ok(s), s["n_wr"], s["n_mo"], s["n_dd"], s["w_mo"]))
rows.sort(key=lambda x: -x[1])
for x in rows[:25]:
    print(("PASS " if x[2] else "     "), x[0], f"robust {x[1]*100:.2f} bar1 {x[3]} wr {x[4]*100:.1f} mo {x[5]*100:.2f} dd {x[6]*100:.1f} worse {x[7]*100:.2f}")
print("n grid", len(grid), "pass bar1", sum(x[3] for x in rows), "pass all", sum(x[2] for x in rows))
pk = [x for x in rows if x[2]]
json.dump(dict(pick=pk[0][0] if pk else None, npass=len(pk)), open(os.path.join(os.environ["PB_OUT"], "pick.json"), "w"))
print("--- candidates with n_wr>=92.5 (why they fail)")
for x in sorted([x for x in rows if x[4] >= .925], key=lambda x: -x[1])[:14]:
    s = res[x[0]]
    print(x[0], f"robust {x[1]*100:.2f} n {s['n_wr']*100:.1f}/{s['n_mo']*100:.2f}/{s['n_dd']*100:.1f} w {s['w_wr']*100:.1f}/{s['w_mo']*100:.2f}/{s['w_dd']*100:.1f} ok5 {ok5(s)}",
          " ".join(f"{t}:{s[t+'_wr']*100:.1f}/{s[t+'_dd']*100:.1f}" for t in ("cm", "cp", "bt", "ls")))
