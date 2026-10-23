"""m11 "fusion": search ONE single-account hourly bot that fuses entry pieces, exit pieces and risk controls.

SELECTION RULE (written before any result was looked at; uses 2021-23 ONLY, the search never computes later data)
  Bars, all on 2021-23:  win rate >= 93.75% (90% mode's 93.8% as printed) and worst drop shallower than -64.5%
                         (90% mode's -64%), with normal fills AND with every buy and sell fill 0.2% worse.
  Score:                 mean 2021-23 %/month over 5 nearby versions: as designed, every cooldown -1h, every
                         cooldown +1h, bot timing (take-profit at once after a market buy, cooldowns +1h), and
                         starting 2 weeks later (2021-01-15).
  Robustness:            the design's nearest parameter neighbours (each switched-on numeric setting moved one notch
                         up / down on the grids in NB below, one at a time): the MEDIAN over neighbours of each of
                         normal win rate, normal worst drop, worse-fills win rate, worse-fills worst drop must pass
                         the same bars.
  Pick:                  rank every design that passes the bars by score; walk down the list (at most 8 tries) and
                         take the first whose neighbours pass. Then ONE look at 2024-26, since 2025, the 2018-20
                         holdout, worse fills and the stressed run (m11_fusion_final.py). No re-picking after it.
SEARCH (stage-wise, ~320 designs + neighbour checks; stage winners chosen by the same rule: passers by score, then
non-passers by fewest failed bars, then score):
  Stage 1 entries (168): tier A = 90% mode (24h-14%, +2.5%, 96h) always; optional tier B from the m10 velocity
          picks chosen on 2021-23 (12h-13% +3% 48h, 6h-10% +2% 48h) and their neighbours; cooldown after a win
          0 (re-arm) or 3h, after a loss 3h; calm-coin filter (skip a coin when its 7-day daily volatility <= v,
          v 7/8/9%) on all tiers, tier B only or tier A only.
  Stage 2 exits on the best 4 (<= 96): tier A target fixed 2.5% or range-scaled 3% x sqrt(rng24/0.2074) clipped
          1-6%; tier A time stop 72/96/120h; tier B target as stage 1 or range-scaled; tier B time stop 48/96h.
  Stage 3 risk and sizing on the best 3 (~57): one at a time stop-loss at the hourly close 15/20/25/30%, 30-day
          drawdown kill-switch 25/35/45% (lock 168h), losing-streak pause 2/24h, 2/72h, 3/72h, sizing 2 bids x 50%
          or 2 bids with A 100% / B 50%, crash guard 30/40%; then greedily combine the options that raised the score
          while passing (one per category, best category first, kept only if the score rises and the bars pass).
Usage: PB_DATA=... PB_HOLDOUT=... PB_OUT=... NUMBA_NUM_THREADS=1 python3 m11_fusion_search.py
Writes $PB_OUT/search.csv, $PB_OUT/search.log, $PB_OUT/pick.json.
"""
import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m11_fusion_lib as F  # noqa: E402

RNG = ("rng", 0.03, 0.5, 0.01, 0.06)
A0 = dict(ref=24, depth=0.14, tp=0.025, hm=96, cdw=3, cdl=3, fs=1.0, vol=0.0)
BS = [(12, 0.13, 0.03), (6, 0.10, 0.02), (12, 0.13, 0.025), (12, 0.12, 0.03), (8, 0.13, 0.025), (8, 0.12, 0.025),
      (6, 0.11, 0.02), (6, 0.10, 0.025)]
NB = dict(depthA=[0.13, 0.14, 0.15], tpA=[0.02, 0.025, 0.03], hm=[48, 72, 96, 120, 144], cd=[0, 2, 3, 4, 6],
          refB=[4, 6, 8, 12, 16], depthB=[0.09, 0.10, 0.11, 0.12, 0.13, 0.14], tpB=[0.015, 0.02, 0.025, 0.03, 0.035],
          vol=[0.06, 0.07, 0.08, 0.09, 0.10], rbase=[0.025, 0.03, 0.035], rhi=[0.05, 0.06, 0.07],
          hs=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35], killm=[0.15, 0.25, 0.35, 0.45, 0.55], clT=[12, 24, 48, 72, 120],
          cln=[1, 2, 3, 4], guard=[0.30, 0.35, 0.40], fs=[0.25, 0.5, 0.75, 1.0])

LOG = open(os.path.join(F.OUT, "search.log"), "a")
ROWS = []
SEEN = {}


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n"); LOG.flush()


def key(d):
    return json.dumps(d, sort_keys=True)


def score(d, stage):
    k = key(d)
    if k in SEEN:
        return SEEN[k]
    s = F.design_scores(d)
    s["fails"] = int(s["n_wr"] < F.WR_BAR) + int(s["n_dd"] <= F.DD_BAR) + int(s["w_wr"] < F.WR_BAR) + \
        int(s["w_dd"] <= F.DD_BAR)
    s["pass"] = s["fails"] == 0
    s["stage"] = stage; s["design"] = k; s["name"] = F.describe(d)
    SEEN[k] = s
    ROWS.append(s)
    return s


def rank(designs):
    sc = [(SEEN[key(d)], d) for d in designs]
    sc.sort(key=lambda x: (x[0]["fails"], -x[0]["robust"]))
    return sc


def line(s):
    return (f"{'PASS' if s['pass'] else 'fail'} score {s['robust']*100:+6.2f} (min {s['rmin']*100:+6.2f}) | "
            f"n {s['n_wr']*100:5.1f}% {s['n_mo']*100:+6.2f} {s['n_dd']*100:5.1f} | w {s['w_wr']*100:5.1f}% "
            f"{s['w_mo']*100:+6.2f} {s['w_dd']*100:5.1f} | {s['n_tpm']:4.1f} tr/mo | {s['name']}")


def mk(tiers, **risk):
    return dict(tiers=tiers, **risk)


# ------------------------------------------------------------------ neighbours
def _steps(grid, v):
    vv = [round(x, 6) for x in grid]
    v = round(v, 6)
    if v not in vv:
        return []
    i = vv.index(v)
    return [grid[j] for j in (i - 1, i + 1) if 0 <= j < len(grid)]


def neighbours(d):
    out = []
    for i, t in enumerate(d["tiers"]):
        t = dict(A0, **t)
        isA = i == 0

        def put(field, val, t=t, i=i):
            x = copy.deepcopy(d); x["tiers"][i] = dict(t, **{field: val}); out.append(x)

        for v in _steps(NB["depthA" if isA else "depthB"], t["depth"]):
            put("depth", v)
        if not isA:
            for v in _steps(NB["refB"], t["ref"]):
                put("ref", v)
        if isinstance(t["tp"], tuple):
            for v in _steps(NB["rbase"], t["tp"][1]):
                put("tp", (t["tp"][0], v) + t["tp"][2:])
            for v in _steps(NB["rhi"], t["tp"][4]):
                put("tp", t["tp"][:4] + (v,))
        else:
            for v in _steps(NB["tpA" if isA else "tpB"], t["tp"]):
                put("tp", v)
        for v in _steps(NB["hm"], t["hm"]):
            put("hm", v)
        for f in ("cdw", "cdl"):
            for v in _steps(NB["cd"], t[f]):
                put(f, v)
        if t["vol"] > 0:
            for v in _steps(NB["vol"], t["vol"]):
                put("vol", v)
        if t["fs"] != 1.0:
            for v in _steps(NB["fs"], t["fs"]):
                put("fs", v)
    r = dict(F.OFFRISK); r.update({k: v for k, v in d.items() if k != "tiers"})
    for f, g in (("hs", "hs"), ("killm", "killm"), ("guard", "guard")):
        if r[f] > 0:
            for v in _steps(NB[g], r[f]):
                x = copy.deepcopy(d); x[f] = v; out.append(x)
    if r["cln"] > 0:
        for f in ("cln", "clT"):
            for v in _steps(NB[f], r[f]):
                x = copy.deepcopy(d); x[f] = v; out.append(x)
    return out


def neighbour_test(d):
    nbs = neighbours(d)
    S = [score(x, "nb") for x in nbs]
    med = {m: float(np.median([s[m] for s in S])) for m in ("n_wr", "n_dd", "w_wr", "w_dd", "robust")}
    ok = med["n_wr"] >= F.WR_BAR and med["n_dd"] > F.DD_BAR and med["w_wr"] >= F.WR_BAR and med["w_dd"] > F.DD_BAR
    return ok, med, len(nbs), S


def main():
    t00 = time.time()
    say(f"=== m11 fusion search {pd.Timestamp.now()} (2021-23 only) ===")
    # stage 0: reproductions (2021-23 only)
    for name, d in (("90% mode", mk([A0])), ("m10 regime pick (calm coin vol7<=8%)", mk([dict(A0, vol=0.08)])),
                    ("m10 exits range-scaled tp, 96h", mk([dict(A0, tp=RNG)]))):
        s = score(d, "s0")
        say(f"[s0] {name}: {line(s)}")
    # stage 1: entries
    st1 = []
    for B in [None] + BS:
        for cdw in (0, 3):
            vopts = [(0.0, "-")] + [(v, sc) for v in (0.07, 0.08, 0.09) for sc in (("all", "B", "A") if B else ("all",))]
            for v, sc in vopts:
                A = dict(A0, cdw=cdw, vol=v if sc in ("all", "A") else 0.0)
                tiers = [A]
                if B:
                    tiers.append(dict(ref=B[0], depth=B[1], tp=B[2], hm=48, cdw=cdw, cdl=3, fs=1.0,
                                      vol=v if sc in ("all", "B") else 0.0))
                st1.append(mk(tiers))
    for d in st1:
        score(d, "s1")
    r1 = rank(st1)
    say(f"[s1] {len(st1)} designs, {sum(s['pass'] for s, _ in r1)} pass the bars ({time.time()-t00:.0f}s). Top 15:")
    for s, _ in r1[:15]:
        say("   " + line(s))
    # stage 2: exits on the best 4
    st2 = []
    for s, d in r1[:4]:
        B = d["tiers"][1] if len(d["tiers"]) > 1 else None
        for tpA in (0.025, RNG):
            for hmA in (72, 96, 120):
                for tpB in ((B["tp"], RNG) if B else (None,)):
                    for hmB in ((48, 96) if B else (None,)):
                        tiers = [dict(d["tiers"][0], tp=tpA, hm=hmA)]
                        if B:
                            tiers.append(dict(B, tp=tpB, hm=hmB))
                        st2.append(mk(tiers))
    for d in st2:
        score(d, "s2")
    r2 = rank(st2)
    say(f"[s2] {len(st2)} designs, {sum(s['pass'] for s, _ in r2)} pass ({time.time()-t00:.0f}s). Top 15:")
    for s, _ in r2[:15]:
        say("   " + line(s))
    # stage 3: risk and sizing on the best 3
    st3 = []
    for s0, d in r2[:3]:
        twoB = len(d["tiers"]) > 1
        cats = {"hs": [dict(hs=v) for v in (0.15, 0.20, 0.25, 0.30)],
                "kill": [dict(killm=v) for v in (0.25, 0.35, 0.45)],
                "streak": [dict(cln=2, clT=24), dict(cln=2, clT=72), dict(cln=3, clT=72)],
                "size": [dict(nbid=2, fs=(0.5, 0.5))] + ([dict(nbid=2, fs=(1.0, 0.5))] if twoB else []),
                "guard": [dict(guard=0.30), dict(guard=0.40)]}

        def apply(base, opt):
            x = copy.deepcopy(base)
            for k, v in opt.items():
                if k == "fs":
                    for i, t in enumerate(x["tiers"]):
                        t["fs"] = v[min(i, len(v) - 1)]
                else:
                    x[k] = v
            return x

        best = {}
        for c, opts in cats.items():
            for o in opts:
                x = apply(d, o)
                st3.append(x)
                s = score(x, "s3")
                if s["pass"] and s["robust"] > s0["robust"] and (c not in best or s["robust"] > best[c][0]):
                    best[c] = (s["robust"], o)
        cur, cs = d, s0
        for c, (sc, o) in sorted(best.items(), key=lambda z: -z[1][0]):
            x = cur
            for k2, v2 in o.items():
                x = apply(x, {k2: v2})
            if key(x) == key(cur):
                continue
            st3.append(x)
            s = score(x, "s3c")
            if s["pass"] and s["robust"] > cs["robust"]:
                cur, cs = x, s
        say(f"[s3] base {line(s0)}\n      -> greedy combo {line(cs)}")
    r3 = rank(st3)
    say(f"[s3] {len(st3)} designs, {sum(s['pass'] for s, _ in r3)} pass ({time.time()-t00:.0f}s). Top 10:")
    for s, _ in r3[:10]:
        say("   " + line(s))
    # pick
    allr = [(s, json.loads(s["design"])) for s in ROWS if s["stage"] != "nb" and s["pass"]]
    allr.sort(key=lambda x: -x[0]["robust"])
    pick = None
    for s, d0 in allr[:8]:
        d = json.loads(s["design"], object_hook=None)
        for t in d["tiers"]:
            if isinstance(t["tp"], list):
                t["tp"] = tuple(t["tp"])
        ok, med, nn, S = neighbour_test(d)
        say(f"[pick] try {line(s)}\n       {nn} neighbours: median n_wr {med['n_wr']*100:.1f} n_dd {med['n_dd']*100:.1f} "
            f"w_wr {med['w_wr']*100:.1f} w_dd {med['w_dd']*100:.1f} score {med['robust']*100:+.2f} -> "
            f"{'PASS' if ok else 'fail'}")
        for x in S:
            say("         nb " + line(x))
        if ok:
            pick = d
            break
    n_designs = sum(1 for s in ROWS)
    say(f"designs evaluated (incl. neighbours): {n_designs} in {time.time()-t00:.0f}s")
    pd.DataFrame(ROWS).to_csv(os.path.join(F.OUT, "search.csv"), index=False)
    if pick is None:
        say("NO PICK: no design in the top 8 has neighbours that pass the bars")
        return
    say(f"PICK: {F.describe(pick)}")
    with open(os.path.join(F.OUT, "pick.json"), "w") as f:
        json.dump(pick, f, indent=1)


if __name__ == "__main__":
    main()
