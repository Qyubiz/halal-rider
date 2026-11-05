"""m14 longwin: dip measured from a LONGER reference high (highest hourly close of last W=36/48/72/120 h), alone or
combined (OR, bid at the higher price) with the 90%-mode 24h-14% trigger. Engine: m11_fusion_lib.frun with tiers.
RULE (written before any run; 2021-23 data only):
 designs: mode alone|combo(24h-14% + W tier) x W{36,48,72,120} x depth{14,17,20,24}% x tp{2,3,4}% at hm96 (96 designs);
          then hm {72,144} for the 15 best of those (30 designs). Total 126.
 bar 1:  all 5 versions (n,cm,cp,bt,ls) win rate>=92.5% and drop > -64.4%; worse fills wr>=92.5%, drop>-64.4%;
         n_mo>13.17 and w_mo>9.79.
 score:  robust = mean %/month over the 5 versions. Neighbour check: one step in depth, W, tp, or hm (same mode);
         median of neighbours' normal-fill 2021-23 mo, wr, dd must pass bar 1 (neighbours evaluated by design_scores 'n').
 pick:   highest robust among those passing bar 1 and neighbours. None if none. Also log how often crash guard blocks.
"""
import json, os, sys, itertools
import numpy as np, pandas as pd
import m11_fusion_lib as F, m10_velocity_lib as V

WR, DD, MON, MOW = 0.925, -0.644, 0.1317, 0.0979
WS, DEPS, TPS, HMS = [36, 48, 72, 120], [.14, .17, .20, .24], [.02, .03, .04], [72, 96, 144]

def design(mode, W, dep, tp, hm):
    t = dict(ref=W, depth=dep, tp=tp, hm=hm)
    if mode == "alone":
        tiers = [t]
    else:
        tiers = [dict(ref=24, depth=.14, tp=tp, hm=hm), t]
    return dict(tiers=tiers)

def key(x): return json.dumps(x)

def main():
    rows = []
    cache = {}
    def score(p):
        k = key(p)
        if k not in cache:
            cache[k] = F.design_scores(design(*p))
        return cache[k]
    def ok(s):
        return all(s[t + "_wr"] >= WR and s[t + "_dd"] > DD for t in ("n", "cm", "cp", "bt", "ls", "w")) \
            and s["n_mo"] > MON and s["w_mo"] > MOW
    stage1 = list(itertools.product(["alone", "combo"], WS, DEPS, TPS, [96]))
    for p in stage1:
        s = score(p); rows.append(dict(mode=p[0], W=p[1], dep=p[2], tp=p[3], hm=p[4], **s))
    df = pd.DataFrame(rows).sort_values("robust", ascending=False)
    top = df.head(15)
    for _, r in top.iterrows():
        for hm in (72, 144):
            p = (r["mode"], int(r.W), r.dep, r.tp, hm); s = score(p)
            rows.append(dict(mode=p[0], W=p[1], dep=p[2], tp=p[3], hm=hm, **s))
    df = pd.DataFrame(rows)
    df["bar1"] = df.apply(ok, axis=1)
    df.to_csv(os.path.join(F.OUT, "grid.csv"), index=False)
    print("designs", len(df), "bar1 pass", int(df.bar1.sum()))
    print(df.sort_values("robust", ascending=False).head(12)[["mode", "W", "dep", "tp", "hm", "n_wr", "n_mo", "n_dd", "w_wr", "w_mo", "w_dd", "robust", "rmin", "bar1"]].to_string())
    cand = df[df.bar1].sort_values("robust", ascending=False)
    pick = None
    for _, r in cand.iterrows():
        p = (r["mode"], int(r.W), r.dep, r.tp, int(r.hm))
        nb = []
        for (a, vals, cur) in (("W", WS, p[1]), ("d", DEPS, p[2]), ("tp", TPS, p[3]), ("hm", HMS, p[4])):
            i = vals.index(cur) if cur in vals else None
            if i is None: continue
            for j in (i - 1, i + 1):
                if 0 <= j < len(vals):
                    q = list(p); q[{"W": 1, "d": 2, "tp": 3, "hm": 4}[a]] = vals[j]; nb.append(score(tuple(q)))
        med = lambda k: float(np.median([x[k] for x in nb]))
        npass = med("n_wr") >= WR and med("n_dd") > DD and med("n_mo") > MON
        print("cand", p, "robust %.4f" % r.robust, "nbr med wr %.3f mo %.4f dd %.3f" % (med("n_wr"), med("n_mo"), med("n_dd")), npass)
        if npass:
            pick = p; break
    print("PICK", pick)
    if pick:
        json.dump(dict(mode=pick[0], W=pick[1], dep=pick[2], tp=pick[3], hm=pick[4]), open(os.path.join(F.OUT, "pick_params.json"), "w"))

if __name__ == "__main__":
    main()
