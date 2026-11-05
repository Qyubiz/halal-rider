"""m17 bigtp grid. RULE (written before any run), 2021-23 data only (scores() never touches later data):
  score = mean %/month over 5 nearby versions (base, cooldown -1h, +1h, bot timing, start 2 weeks later);
  eligible: worst drop > -70% in ALL versions incl. every-fill-0.2%-worse (ddmin), robust score computed as in
  m11_fusion_lib.design_scores. Ranking value = min(own robust, median robust of its one-step neighbours) so a pick must
  sit in a plateau. Neighbours of the top 5 per stage are evaluated and counted in the design total (<=200).
  Stage A: 96 single-step designs  depth{12,14,16,20} x tp{3,5,7,10} x hm{48,120,240} x cd{0,24}  (guard .35, no vol)
  Stage B: top 6 of A (own robust, eligible) x guard{.30,.40}, vol>7%, range-scaled tp base=tp alpha.5 clip[3,10]   (<=~24)
  Stage C: two-step (sell half at tp1, rest at tp2) on top 4 of A/B: tp1{2.5,3.5,5} x tp2{6,8,10}, tp1<tp2 (<=~36)
  Then neighbours of the top 5 overall (one step in depth/tp/hm/cd), pick = best ranking value. ONE pick.
"""
import itertools, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m17_bigtp_lib as L
import m11_fusion_lib as F

OUT = os.environ["PB_OUT"]
R = {}


def key(d):
    return json.dumps(d, sort_keys=True, default=list)


def sc(d):
    k = key(d)
    if k not in R:
        R[k] = (d, L.scores(d))
    return R[k][1]


def ok(s):
    return s["ddmin"] > -0.70


def show(d, s):
    return (f"{d['depth']*100:.0f}% tp{d['tp'] if isinstance(d['tp'],tuple) else round(d['tp']*100,1)} "
            f"p{d['part']:g}/{d['tp2']*100:.0f} hm{d['hm']} cd{d['cd']} g{d['guard']:.2f} v{d['vol']:.2f} | "
            f"rob {s['robust']*100:+.2f} n {s['n_mo']*100:+.2f} w {s['w_mo']*100:+.2f} wr {s['n_wr']*100:.1f} "
            f"dd {s['ddmin']*100:.0f} tpm {s['n_tpm']:.1f}")


A = [L.D(dp, tp, hm, cd) for dp in (.12, .14, .16, .20) for tp in (.03, .05, .07, .10) for hm in (48, 120, 240)
     for cd in (0, 24)]
for d in A:
    sc(d)
rank = sorted(((sc(d)["robust"], key(d)) for d in A if ok(sc(d))), reverse=True)
topA = [R[k][0] for _, k in rank[:6]]
print("A top:"); [print(" ", show(*R[key(d)])) for d in topA]
B = []
for d in topA:
    for g in (.30, .40):
        B.append(dict(d, guard=g))
    B.append(dict(d, vol=.07))
    B.append(dict(d, tp=("rng", d["tp"], .5, .03, .10)))
for d in B:
    sc(d)
allr = sorted(((sc(R[k][0])["robust"], k) for k in R if ok(R[k][1])), reverse=True)
top4 = [R[k][0] for _, k in allr[:4]]
Cc = []
for d in top4:
    if isinstance(d["tp"], tuple):
        continue
    for t1 in (.025, .035, .05):
        for t2 in (.06, .08, .10):
            Cc.append(dict(d, tp=t1, part=0.5, tp2=t2))
Cc = Cc[:40]
for d in Cc:
    sc(d)
allr = sorted(((sc(R[k][0])["robust"], k) for k in R if ok(R[k][1])), reverse=True)
print("designs so far", len(R))
print("top overall:"); [print(" ", show(*R[k])) for _, k in allr[:10]]


def neigh(d):
    out = []
    steps = dict(depth=(.12, .14, .16, .20), hm=(48, 120, 240), cd=(0, 24))
    for f, vals in steps.items():
        if d[f] in vals:
            i = vals.index(d[f])
            for j in (i - 1, i + 1):
                if 0 <= j < len(vals):
                    out.append(dict(d, **{f: vals[j]}))
    if not isinstance(d["tp"], tuple):
        for f in ("tp", "tp2") if d["part"] < 1 else ("tp",):
            if d[f] > 0:
                for m in (0.8, 1.25):
                    out.append(dict(d, **{f: round(d[f] * m, 4)}))
    return out


final = []
for _, k in allr[:5]:
    d = R[k][0]
    ns = [sc(x) for x in neigh(d)]
    med = float(np.median([s["robust"] for s in ns])) if ns else np.nan
    pas = float(np.mean([ok(s) for s in ns])) if ns else np.nan
    final.append((min(R[k][1]["robust"], med), d, med, pas))
print("designs total", len(R))
final.sort(key=lambda x: -x[0])
for v, d, med, pas in final:
    print(f"  rank {v*100:+.2f} nb-median {med*100:+.2f} nb-ok {pas:.2f} ::", show(d, sc(d)))
pick = final[0][1]
json.dump(pick, open(os.path.join(OUT, "pick.json"), "w"), default=list)
print("PICK", json.dumps(pick, default=list))
