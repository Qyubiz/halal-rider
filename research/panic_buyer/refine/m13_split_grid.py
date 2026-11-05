"""m13 split grid (2021-23 ONLY). RULE written before any run:
Designs: 2 legs d1 in {12,13,14}%, d2 in {15,16,17}% (both under the 24h highest close), f = weight of the shallow leg
in {0.3,0.5,0.7}, tp in {2,2.5,3}% (81 designs) + 6 three-leg designs (13/15/17, w3=0.3, f in{.3,.5,.7}, tp 2/2.5%?
-> d3=17 d2=15 d1=13 only, tp in {2.5,3}). Fixed: 96h from first fill, 3h cooldown, 35% guard, one coin, fully funded.
Each design scored as the mean over 5 versions (as designed, cooldown -1h, +1h, bot timing, start 2 weeks later), for
normal and every-fill-0.2%-worse fills; win rate = mean over versions, drop = median over versions.
Qualify: normal AND worse: win >= 92.5%, worst drop >= -64.4%, %/month > 13.17 (normal) / 9.79 (worse); extra drop
target proxy: normal drop >= -55%. Also the median of the design's grid neighbours (one step in one parameter) must pass
the same bars. Rank qualifiers by the mean of (normal %/month, worse %/month). One pick = rank 1. If none qualifies,
say so (a labelled fallback = best-ranked design that passes the bars without the neighbour test, else none).
"""
import itertools
import json
import os
import time

import numpy as np
import m13_split_lib as S

D1, D2, F, TP = (0.12, 0.13, 0.14), (0.15, 0.16, 0.17), (0.3, 0.5, 0.7), (0.02, 0.025, 0.03)
BARS = dict(wr=.925, dd=-.644, mo=.1317, wmo=.0979)


def designs():
    out = [dict(d1=a, d2=b, f=f, tp=t) for a, b, f, t in itertools.product(D1, D2, F, TP)]
    out += [dict(d1=.13, d2=.15, d3=.17, f=f, w3=.3, tp=t) for f in F for t in (0.025, 0.03)]
    return out


def key(d):
    return json.dumps(d, sort_keys=True)


def legs(d):
    return S.legs_of(d["d1"], d["d2"], d["f"], d.get("d3"), d.get("w3", 0.0))


def ok(s):
    return s["wr"] >= BARS["wr"] and s["wwr"] >= BARS["wr"] and s["dd"] >= BARS["dd"] and s["wdd"] >= BARS["dd"] \
        and s["mo"] > BARS["mo"] and s["wmo"] > BARS["wmo"]


def neighbours(d, allk):
    """2-leg designs differing in one parameter by one grid step."""
    if "d3" in d:
        return []
    out = []
    for name, grid in (("d1", D1), ("d2", D2), ("f", F), ("tp", TP)):
        i = grid.index(d[name])
        for j in (i - 1, i + 1):
            if 0 <= j < len(grid):
                e = dict(d, **{name: grid[j]})
                if key(e) in allk:
                    out.append(e)
    return out


def main():
    path = os.path.join(S.M.OUT, "grid.json")
    res = {}
    t = time.time()
    for d in designs():
        res[key(d)] = S.design_scores(legs(d), d["tp"])
        print(len(res), f"{time.time()-t:.0f}s", flush=True)
    json.dump(res, open(path, "w"))
    base = S.design_scores(((0.14, 1.0),), 0.025)
    print("baseline 90% mode (5 versions avg)", {k: round(v, 4) for k, v in base.items() if k in ("mo", "wr", "dd", "wmo", "wwr", "wdd")})
    rows = []
    for d in designs():
        s = res[key(d)]
        nb = neighbours(d, res)
        q = ok(s)
        npass = None
        if nb:
            med = {k: float(np.median([res[key(e)][k] for e in nb])) for k in ("wr", "wwr", "dd", "wdd", "mo", "wmo")}
            npass = ok(med)
        rows.append((q, npass, (s["mo"] + s["wmo"]) / 2, d, s))
    rows.sort(key=lambda r: -r[2])
    print("\nTOP 15 by rank score (q=qualifies, nb=neighbour median passes)")
    for q, nb, sc, d, s in rows[:15]:
        print(f"q={int(q)} nb={nb} score {sc*100:.2f} mo {s['mo']*100:.2f}/{s['wmo']*100:.2f} wr {s['wr']*100:.1f}/{s['wwr']*100:.1f} "
              f"dd {s['dd']*100:.1f}/{s['wdd']*100:.1f} {d}")
    good = [r for r in rows if r[0] and r[1]]
    print(f"\nqualify: {sum(r[0] for r in rows)}  qualify+neighbours: {len(good)}")
    pick = good[0][3] if good else None
    print("PICK", pick)
    json.dump(dict(pick=pick, fallback=[r[3] for r in rows if r[0]][:1]), open(os.path.join(S.M.OUT, "pick.json"), "w"))


if __name__ == "__main__":
    main()
