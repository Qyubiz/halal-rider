"""m15 earlycut grid (2021-23 ONLY). Rule written before any grid run:

DESIGNS (132): 90% mode + one early exit (m15_earlycut_lib):
  A continuation: market mi in {BTC, 19-coin index} x coin drop ax in {3,5,7,10%} x window aN in {6,12,24,48h}
    x market drop ay in {2,4,6%}                                                     (96)
  B failed bounce: window bN in {12,24,48h} x bounce bb in {0.5,1,2%} x under fill-hour low by bz in {0,2,4%} (27)
  C limit at a small loss: arm after cN in {24,48,72h} under entry, limit at entry x cl in {0.98,0.99,0.995} (9)
BAR 1 (normal AND every fill 0.2% worse, 2021-23): win rate >= 92.5%, worst drop no deeper than -64.45% (90% mode's
  -64.4% at one decimal), %/month > 13.17 (normal) and > 9.79 (worse).
BAR 2: all 5 nearby versions (base, cooldown -1h, +1h, bot timing, start 2021-01-15) win rate >= 92.5% and drop
  no deeper than -64.45%. Score = mean %/month over the 5 versions.
BAR 3: grid neighbours = same family, one step in one parameter (mi toggled counts as a step); the median of the
  neighbours' normal %/month, win rate and drop (each separately) must pass the normal part of bar 1.
PICK: highest score among designs passing bars 1-3; none passing -> no pick. Also reported for the pick and the top
  designs: how many trades the early exit cut, and how many of those 90% mode had closed as winners (same entry bar
  and coin in the 90% mode log).
"""
import itertools, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import m15_earlycut_lib as L

GA = dict(mi=[0, 1], ax=[.03, .05, .07, .10], aN=[6, 12, 24, 48], ay=[.02, .04, .06])
GB = dict(bN=[12, 24, 48], bb=[.005, .01, .02], bz=[0.0, .02, .04])
GC = dict(cN=[24, 48, 72], cl=[.98, .99, .995])
FAMS = {1: GA, 2: GB, 3: GC}


def designs():
    out = []
    for fam, G in FAMS.items():
        ks = list(G)
        for vals in itertools.product(*[G[k] for k in ks]):
            out.append(dict(fam=fam, **dict(zip(ks, vals))))
    return out


def neighbours(d):
    G = FAMS[d["fam"]]; nb = []
    for k, vs in G.items():
        i = vs.index(d[k])
        for j in (i - 1, i + 1):
            if 0 <= j < len(vs):
                nb.append(dict(d, **{k: vs[j]}))
    return nb


def key(d):
    return json.dumps({k: d[k] for k in sorted(d)})


def bar1n(s):
    return s["n_wr"] >= .925 and s["n_dd"] >= -.6445 and s["n_mo"] > .1317


def main():
    import m10_velocity_lib as V
    base = L.design_scores(L.OFF)
    bl = base["lg"]; bmap = {(int(a), int(b)): r for a, b, r in zip(bl.eb, bl.k, bl.ret)}
    print("90% mode", {k: round(v, 4) for k, v in base.items() if k != "lg"})
    S = {}
    for d in designs():
        s = L.design_scores(d)
        lg = s.pop("lg")
        _, idx, _ = V.market(False)
        tb = idx[lg.eb.astype(int).values]
        cut = lg[(lg.why >= 11) & (tb >= L.T21) & (tb < L.T24)]
        m = [bmap.get((int(a), int(b))) for a, b in zip(cut.eb, cut.k)]
        s["ncut"] = len(cut); s["cut_match"] = sum(x is not None for x in m); s["cut_win"] = sum(x is not None and x > 0 for x in m)
        s["b1"] = bar1n(s) and s["w_wr"] >= .925 and s["w_dd"] >= -.6445 and s["w_mo"] > .0979
        s["b2"] = all(s[f"{t}_wr"] >= .925 and s[f"{t}_dd"] >= -.6445 for t in ("n", "cm", "cp", "bt", "ls"))
        S[key(d)] = (d, s)
    rows = []
    for kk, (d, s) in S.items():
        nb = [S[key(x)][1] for x in neighbours(d)]
        med = {f: float(np.median([x[f] for x in nb])) for f in ("n_mo", "n_wr", "n_dd")}
        s["b3"] = bar1n(med)
        s["nb_mo"] = med["n_mo"]
        rows.append(dict(d=kk, **{k: v for k, v in s.items()}))
    R = pd.DataFrame(rows).sort_values("robust", ascending=False)
    R.to_csv(os.path.join(L.OUT, "grid.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
    cols = ["d", "n_wr", "n_mo", "n_dd", "w_mo", "robust", "nb_mo", "ncut", "cut_match", "cut_win", "b1", "b2", "b3"]
    print("counts: bar1", int(R.b1.sum()), "bar1+2", int((R.b1 & R.b2).sum()), "bar1+2+3", int((R.b1 & R.b2 & R.b3).sum()))
    for fam in (1, 2, 3):
        x = R[R.d.str.contains(f'"fam": {fam}')]
        print(f"family {fam}: best normal %/mo {x.n_mo.max()*100:.2f}, median {x.n_mo.median()*100:.2f}, bar1 {int(x.b1.sum())}/{len(x)}")
    print(R[cols].head(25).to_string(index=False))
    ok = R[R.b1 & R.b2 & R.b3]
    if len(ok):
        pick = json.loads(ok.iloc[0].d)
        print("PICK", json.dumps(pick))
        json.dump(pick, open(os.path.join(L.OUT, "earlycut_pick.json"), "w"))
    else:
        print("NO PICK under the rule")


if __name__ == "__main__":
    main()
