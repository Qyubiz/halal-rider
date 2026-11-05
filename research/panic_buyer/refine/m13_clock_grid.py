"""m13 clock grid (2021-23 ONLY). RULE WRITTEN BEFORE ANY RUN.
Designs (42, header count corrected after run: 39 windows+types, 3 weekend): clock window of len 4/6/8 UTC hours at starts that are multiples of len (6+4+3 = 13 windows) x type {off: no bid in window, shal: depth 12% in window, deep: depth 16% in window} = 39,
times nothing else; plus weekend (Sat+Sun) off/shal/deep = 3 (weekend designs are anatomy only: no neighbours, never
picked). Everything else is 90% mode. Time stop by hour not tested.
Bars (2021-23, normal AND every fill 0.2% worse, as designed): win rate >= 92.5%, worst drop no deeper than -64.4%,
%/month > 13.17 normal and > 9.79 worse. Rank bar-passers by mean %/month over the 5 nearby versions (as designed,
cooldown -1h/+1h, bot timing, start 2 weeks later), normal fills. Neighbours: window shifted by -1h and +1h, and
width one step (4<->6<->8) at the same start; MEDIAN over neighbours of normal/worse win rate, drop, %/month must pass
the same bars. Walk down the ranking (max 8); first pass = THE pick. None -> nothing passes. One look: m13_clock_final.py.
"""
import json, os, sys, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m13_clock_lib as C, m12_entry_lib as L

WR, DD, MO_N, MO_W = 0.925, -0.6445, 0.1317, 0.0979
LOG = open(os.path.join(L.OUT, "clock_grid.log"), "a")
SEEN = {}


def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def fails(s):
    return int(s["n_wr"] < WR) + int(s["w_wr"] < WR) + int(s["n_dd"] < DD) + int(s["w_dd"] < DD) + \
        int(s["n_mo"] <= MO_N) + int(s["w_mo"] <= MO_W)


def score(c):
    k = json.dumps(c, sort_keys=True)
    if k not in SEEN:
        s = C.design_scores(c); s["fails"] = fails(s); s["c"] = c
        SEEN[k] = s
        say(f"{k:55s} score {s['robust']*100:+6.2f} n {s['n_wr']*100:4.1f}/{s['n_mo']*100:+6.2f}/{s['n_dd']*100:5.1f} "
            f"w {s['w_wr']*100:4.1f}/{s['w_mo']*100:+6.2f}/{s['w_dd']*100:5.1f} tpm {s['n_tpm']:.1f} fails {s['fails']}")
    return SEEN[k]


def neighbours(c):
    out = []
    for sh in (-1, 1):
        out.append(dict(c, start=(c["start"] + sh) % 24))
    ln = {4: [6], 6: [4, 8], 8: [6]}[c["len"]]
    out += [dict(c, len=l) for l in ln]
    return out


def main():
    t0 = time.time()
    designs = [dict(typ=t, start=s, len=w, wk="none") for w in (4, 6, 8) for s in range(0, 24, w) for t in ("off", "shal", "deep")]
    designs += [dict(typ="none", start=0, len=4, wk=t) for t in ("off", "shal", "deep")]
    score(dict(typ="none", start=0, len=4, wk="none"))
    for c in designs: score(c)
    cand = [s for s in SEEN.values() if s["fails"] == 0 and s["c"]["typ"] != "none"]
    cand.sort(key=lambda s: -s["robust"])
    say(f"{len(SEEN)} scored in {time.time()-t0:.0f}s; bar-passers (excl. weekend/none): {len(cand)}")
    pick = None
    for s in cand[:8]:
        S = [score(x) for x in neighbours(s["c"])]
        med = {k: float(np.median([q[k] for q in S])) for k in ("n_wr", "w_wr", "n_dd", "w_dd", "n_mo", "w_mo")}
        ok = fails(med) == 0
        say(f"try {json.dumps(s['c'])} score {s['robust']*100:+.2f}: nb median n {med['n_wr']*100:.1f}/{med['n_mo']*100:+.2f}/{med['n_dd']*100:.1f} w {med['w_wr']*100:.1f}/{med['w_mo']*100:+.2f} -> {'PASS' if ok else 'fail'}")
        if ok: pick = s["c"]; break
    pd.DataFrame([{k: v for k, v in s.items() if k != "c"} | {"c": json.dumps(s["c"])} for s in SEEN.values()]).to_csv(os.path.join(L.OUT, "clock_grid.csv"), index=False)
    say("PICK: " + (json.dumps(pick) if pick else "none (nothing passes the 2021-23 selection)"))
    json.dump({"pick": pick}, open(os.path.join(L.OUT, "clock_pick.json"), "w"))


if __name__ == "__main__":
    main()
