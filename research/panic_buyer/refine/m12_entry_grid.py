"""m12 entry (agent B): stage-wise search over WHICH coin 90% mode bids on and AT WHAT PRICE. 2021-23 ONLY.

RANKING RULE (written before any run; M12_PREREG.md selection, plus a stricter drop proxy)
  Bars, 2021-23, normal fills AND every buy and sell fill 0.2% worse (as designed, from design_scores):
      win rate >= 92.5%; worst drop no deeper than -64.4%; %/month above 90% mode's (13.17% normal, 9.79% worse).
  Drop proxy for the -35% test limit: the MEDIAN worst drop over the 5 nearby versions (normal fills) no deeper than
      -60% (90% mode: -62.6%). If no design meets bars + proxy, the proxy is dropped (bars only).
  Score: mean 2021-23 %/month over 5 nearby versions (as designed, cooldown -1h, cooldown +1h, bot timing =
      take-profit at once after a market buy + cooldown +1h, start 2021-01-15), normal fills.
  Robustness: nearest parameter neighbours (each numeric setting one notch up/down on NB, one at a time); the
      MEDIAN over neighbours of normal and worse-fill win rate, worst drop and %/month must pass the bars.
  Pick: rank designs meeting bars (+proxy) by score; walk down at most 8; the first whose neighbours pass is THE
      pick. None -> "nothing passes the 2021-23 selection". Then ONE look (m12_entry_final.py).
  Stage winners (to carry forward): order by (meets bars+proxy, meets bars, fewest failed bars, score).
SEARCH (~230 designs + neighbour checks)
  S1  depth 12-17% x take-profit 1.5-4% x time stop 48/96/144h, cooldown 3h, anchor = 24h highest close, coin =
      closest to trigger (108).
  S1b on the best 4: time stop 72/120h, cooldown 0/1/2/4/6h, one at a time (28).
  S2  on the best 3 so far: anchor highest high (depth +0/+1pt) or 24h mean close (depth -2/-3/-4pt); coin choice
      among coins within 1% / 3% of their trigger by liquidity, coin size, own walk-forward bounce rate, deepest /
      shallowest vs the 7-day high (15 each).
  S3  on the best 3 so far: majors (BTC ETH BNB SOL XRP) depth -2..+2pt vs the rest; volatility depth
      k x 30-day median hourly range, k 9/11/12/13/15/17 clipped 10-25%, and k 11/13/15 clipped 12-25% (13 each).
  S4  combine: the best design of S3 with the best S2 coin-choice option and the best S2 anchor option (<= 3).
Usage: PB_DATA=... PB_HOLDOUT=... PB_OUT=... NUMBA_NUM_THREADS=1 python3 m12_entry_grid.py
Writes $PB_OUT/grid.csv, grid.log, pick.json.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m12_entry_lib as L  # noqa: E402

WR, DD, MO_N, MO_W, PROXY = 0.925, -0.6445, 0.1317, 0.0979, -0.60
NB = dict(depth=[0.12, 0.13, 0.14, 0.15, 0.16, 0.17], dmaj=[0.12, 0.13, 0.14, 0.15, 0.16, 0.17],
          tp=[0.015, 0.02, 0.025, 0.03, 0.035, 0.04], hm=[48, 72, 96, 120, 144], cd=[0, 1, 2, 3, 4, 6],
          near=[0.01, 0.02, 0.03, 0.05], vk=[9, 11, 12, 13, 15, 17], vlo=[0.08, 0.10, 0.12], vhi=[0.20, 0.25, 0.30])
LOG = open(os.path.join(L.OUT, "grid.log"), "a")
SEEN, ROWS = {}, []
USE_PROXY = [True]
SHIFT = {}


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def fails(s):
    return int(s["n_wr"] < WR) + int(s["w_wr"] < WR) + int(s["n_dd"] < DD) + int(s["w_dd"] < DD) + \
        int(s["n_mo"] <= MO_N) + int(s["w_mo"] <= MO_W)


def score(d, stage):
    k = L.key(d)
    if k not in SEEN:
        s = L.design_scores(d)
        s["fails"] = fails(s); s["bars"] = s["fails"] == 0; s["proxy"] = s["rdd_med"] >= PROXY
        s.update(stage=stage, design=k, name=L.describe(d))
        SEEN[k] = s; ROWS.append(s)
        say(f"[{stage}] {s['name']:58s} score {s['robust']*100:+6.2f} | n {s['n_wr']*100:4.1f}/{s['n_mo']*100:+6.2f}/"
            f"{s['n_dd']*100:5.1f} w {s['w_wr']*100:4.1f}/{s['w_mo']*100:+6.2f}/{s['w_dd']*100:5.1f} "
            f"ddmed {s['rdd_med']*100:5.1f} fails {s['fails']}{' PASS' if s['bars'] else ''}{'+P' if s['bars'] and s['proxy'] else ''}")
    return SEEN[k]


def order(rows):
    return sorted(rows, key=lambda s: (not (s["bars"] and s["proxy"]), not s["bars"], s["fails"], -s["robust"]))


def best(stage_rows, n):
    return [json.loads(s["design"]) for s in order(stage_rows)[:n]]


def steps(grid, v):
    if v not in grid:
        return []
    i = grid.index(v)
    return [grid[j] for j in (i - 1, i + 1) if 0 <= j < len(grid)]


def neighbours(d):
    d = dict(L.BASE, **d); out = []
    fields = ["tp", "hm", "cd"]
    if d["vk"] > 0:
        fields += ["vk", "vlo", "vhi"]
    else:
        fields += ["depth"] + (["dmaj"] if d["dmaj"] > 0 else [])
    if d["rank"] != "dist":
        fields.append("near")
    for f in fields:
        if f == "depth" and d["anchor"] != "c":      # shifted depths of other anchors: +-1pt
            for v in (round(d["depth"] - 0.01, 3), round(d["depth"] + 0.01, 3)):
                out.append(dict(d, depth=v))
            continue
        for v in steps(NB[f], d[f]):
            out.append(dict(d, **{f: v}))
    return out


def neighbours_pass(d):
    S = [score(x, "nb") for x in neighbours(d)]
    med = {k: float(np.median([s[k] for s in S])) for k in ("n_wr", "w_wr", "n_dd", "w_dd", "n_mo", "w_mo")}
    return fails(med) == 0, med, len(S)


def run(stage, designs):
    return [score(d, stage) for d in designs]


def main():
    t0 = time.time()
    s1 = run("S1", [dict(depth=dp, tp=tp, hm=hm) for dp in NB["depth"] for tp in NB["tp"] for hm in (48, 96, 144)])
    top = best(s1, 4)
    s1b = []
    for d in top:
        s1b += run("S1b", [dict(d, hm=h) for h in (72, 120)] + [dict(d, cd=c) for c in (0, 1, 2, 4, 6)])
    top = best(s1 + s1b, 3)
    s2 = []
    for d in top:
        dp = d["depth"]
        ds = []
        for an, x in (("h", 0.0), ("h", 0.01), ("m", -0.02), ("m", -0.03), ("m", -0.04)):
            x2 = dict(d, anchor=an, depth=round(dp + x, 3)); SHIFT[L.key(x2)] = x; ds.append(x2)
        ds += [dict(d, rank=r, near=nr) for r in ("liq", "size", "bounce", "d7deep", "d7shal") for nr in (0.01, 0.03)]
        s2 += run("S2", ds)
    top = best(s1 + s1b + s2, 3)
    s3 = []
    for d in top:
        dp = d["depth"]
        ds = [dict(d, dmaj=round(dp + x, 3)) for x in (-0.02, -0.01, 0.01, 0.02)]
        ds += [dict(d, vk=v, vlo=0.10, vhi=0.25) for v in (9, 11, 12, 13, 15, 17)]
        ds += [dict(d, vk=v, vlo=0.12, vhi=0.25) for v in (11, 13, 15)]
        s3 += run("S3", ds)
    allr = s1 + s1b + s2 + s3
    b3 = best(allr, 1)[0]
    r2 = [s for s in s2 if json.loads(s["design"])["rank"] != "dist"]
    a2 = [s for s in s2 if json.loads(s["design"])["anchor"] != "c"]
    s4 = []
    if r2 and a2:
        br, ba = json.loads(order(r2)[0]["design"]), json.loads(order(a2)[0]["design"])
        c1 = dict(b3, rank=br["rank"], near=br["near"])
        c2 = dict(b3, anchor=ba["anchor"], depth=round(b3["depth"] + SHIFT[L.key(ba)], 3)) if b3["vk"] == 0 else None
        if b3["dmaj"] > 0 and c2:
            c2["dmaj"] = round(b3["dmaj"] + SHIFT[L.key(ba)], 3)
        s4 += run("S4", [c1] + ([c2, dict(c2, rank=br["rank"], near=br["near"])] if c2 else []))
    allr = s1 + s1b + s2 + s3 + s4
    say(f"\n{len(SEEN)} designs scored ({time.time()-t0:.0f}s). Passing bars: {sum(s['bars'] for s in allr)}, "
        f"bars+proxy: {sum(s['bars'] and s['proxy'] for s in allr)}")
    cand = [s for s in allr if s["bars"] and s["proxy"]]
    if not cand:
        say("No design meets bars + drop proxy: proxy dropped (pre-registered fallback).")
        USE_PROXY[0] = False
        cand = [s for s in allr if s["bars"]]
    cand = sorted(cand, key=lambda s: -s["robust"])
    pick = None
    for s in cand[:8]:
        d = json.loads(s["design"])
        ok, med, nn = neighbours_pass(d)
        say(f"try {s['name']} score {s['robust']*100:+.2f}: {nn} neighbours median n {med['n_wr']*100:.1f}/"
            f"{med['n_mo']*100:+.2f}/{med['n_dd']*100:.1f} w {med['w_wr']*100:.1f}/{med['w_mo']*100:+.2f}/"
            f"{med['w_dd']*100:.1f} -> {'PASS' if ok else 'fail'}")
        if ok:
            pick = d; break
    pd.DataFrame(ROWS).to_csv(os.path.join(L.OUT, "grid.csv"), index=False)
    say(f"total designs incl. neighbours: {len(SEEN)}")
    if pick is None:
        say("PICK: none (nothing passes the 2021-23 selection)")
    else:
        say("PICK: " + L.describe(pick) + "  " + json.dumps(dict(L.BASE, **pick), sort_keys=True))
    json.dump({"pick": dict(L.BASE, **pick) if pick else None, "proxy_used": USE_PROXY[0]},
              open(os.path.join(L.OUT, "pick.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
