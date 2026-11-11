"""m13 futures: search over futures-flush signals that set 90% mode's bid depth. 2021-23 ONLY.

RANKING RULE (written before any run; M12_PREREG.md selection)
  Bars, 2021-23, as designed, normal fills AND every buy and sell fill 0.2% worse: win rate >= 92.5%; worst drop
      no deeper than -64.4%; %/month above 90% mode (13.17% normal, 9.79% worse).
  Drop proxy for the -35% test limit: median worst drop over the 5 nearby versions (normal fills) no deeper than
      90% mode's same median. If no design meets bars + proxy, the proxy is dropped (bars only).
  Score: mean 2021-23 %/month over 5 nearby versions (as designed, cooldown -1h, +1h, bot timing, start
      2021-01-15), normal fills. Ties: higher mean worse-fill %/month.
  Robustness: nearest parameter neighbours (x, N, d_on, d_off each one notch up/down on NB, one at a time); the
      MEDIAN over neighbours of normal and worse-fill win rate, worst drop and %/month (as designed) must pass bars.
  Pick: rank designs meeting bars (+proxy) by score; walk down at most 8; the first whose neighbours pass is THE
      pick. None -> "nothing passes the 2021-23 selection". Then ONE look (m13_futures_final.py).
  Coin-hours with no futures data for the signal bid like 90% mode (14%), decided before any run.
  Signal coverage inside 2021-23: oi from ~2021-12 (so 2022-23); funding / premium from 2021-01 for most coins.
SEARCH (95 designs + neighbours)
  oi    N 4/12/24h x drop 3/6/10% x (d_on, d_off) in PAIRS                         45
  fund  last funding <= -0.01% / 0% / +0.005% x PAIRS                              15
  fdrop funding fell >= 0.01 / 0.02 / 0.04 points below its 72h high x PAIRS       15
  prem  premium index <= -0.05 / -0.1 / -0.2 / -0.4% x PAIRS                       20
  PAIRS: bid 12% when on / 14% off; 14% / 16%; 14% / none; 12% / 16%; 12% / none.
Usage: PB_DATA=... PB_HOLDOUT=... PB_OUT=... PB_FUT=... NUMBA_NUM_THREADS=1 python3 m13_futures_grid.py
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
import m13_futures_lib as F  # noqa: E402

WR, DD, MO_N, MO_W = 0.925, -0.6445, 0.1317, 0.0979
PAIRS = [(0.12, 0.14), (0.14, 0.16), (0.14, None), (0.12, 0.16), (0.12, None)]
XS = dict(oi=[0.03, 0.06, 0.10], fund=[-0.0001, 0.0, 0.00005], fdrop=[0.0001, 0.0002, 0.0004],
          prem=[0.0005, 0.001, 0.002, 0.004])
NB = dict(N=[2, 4, 8, 12, 24, 48], d_on=[0.10, 0.11, 0.12, 0.13, 0.14, 0.15], d_off=[0.14, 0.15, 0.16, 0.18, None],
          x=dict(oi=[0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.15],
                 fund=[-0.0003, -0.0002, -0.0001, -0.00005, 0.0, 0.00005, 0.0001],
                 fdrop=[0.00005, 0.0001, 0.0002, 0.0004, 0.0008],
                 prem=[0.00025, 0.0005, 0.001, 0.002, 0.004, 0.008]))
LOG = open(os.path.join(F.OUT, "grid.log"), "a")
SEEN = {}


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def norm(d):
    d = dict(F.BASE, **d)
    if d["sig"] in ("fund", "fdrop"):
        d["N"] = 0
    return d


def score(d):
    d = norm(d)
    k = F.key(d)
    if k not in SEEN:
        s = F.design_scores(d)
        s.update(d)
        SEEN[k] = s
    return SEEN[k]


def bars(s, pre=""):
    """pre '' = as designed."""
    return [s["n_wr"] >= WR, s["wn_wr"] >= WR, s["n_dd"] >= DD, s["wn_dd"] >= DD, s["n_mo"] > MO_N,
            s["wn_mo"] > MO_W]


def neighbours(d):
    d = norm(d)
    out = []
    for p in ("x", "N", "d_on", "d_off"):
        if p == "N" and d["sig"] not in ("oi",):
            continue
        g = NB["x"][d["sig"]] if p == "x" else NB[p]
        if d[p] not in g:
            continue
        i = g.index(d[p])
        for j in (i - 1, i + 1):
            if 0 <= j < len(g):
                nd = dict(d, **{p: g[j]})
                if nd["d_off"] is not None and nd["d_on"] == nd["d_off"]:
                    continue
                out.append(nd)
    return out


def nb_median_pass(d):
    S = [score(n) for n in neighbours(d)]
    if not S:
        return False, {}
    med = {k: float(np.median([s[k] for s in S])) for k in ("n_wr", "wn_wr", "n_dd", "wn_dd", "n_mo", "wn_mo")}
    return all(bars(med)), med


def fmt(s):
    off = "none" if s["d_off"] is None else f"{s['d_off']*100:.0f}"
    return (f"{s['sig']:5s} N{s['N']:<3d} x{s['x']:+.5f} on{s['d_on']*100:.0f} off{off:>4s} | n {s['n_wr']*100:5.1f} "
            f"{s['n_mo']*100:+6.2f} {s['n_dd']*100:6.1f} | w {s['wn_wr']*100:5.1f} {s['wn_mo']*100:+6.2f} "
            f"{s['wn_dd']*100:6.1f} | robust {s['robust']*100:+6.2f} w {s['wrobust']*100:+6.2f} rdd {s['rdd']*100:6.1f}")


def main():
    t0 = time.time()
    base = score(dict(sig="none"))
    say("90% mode:", fmt(base))
    designs = []
    for sig, xs in XS.items():
        for N in ([4, 12, 24] if sig == "oi" else [0]):
            for x in xs:
                for on, off in PAIRS:
                    designs.append(dict(sig=sig, N=N, x=x, d_on=on, d_off=off))
    say(f"{len(designs)} designs")
    rows = []
    for d in designs:
        s = score(d)
        rows.append(s)
        say(fmt(s), "BARS" if all(bars(s)) else "")
    say(f"grid done {time.time()-t0:.0f}s")
    pd.DataFrame(rows).to_csv(os.path.join(F.OUT, "grid.csv"), index=False)
    ok = [s for s in rows if all(bars(s))]
    prox = [s for s in ok if s["rdd"] >= base["rdd"]]
    say(f"meet bars {len(ok)}, bars + proxy {len(prox)} (90% mode rdd {base['rdd']*100:.1f})")
    cand = prox if prox else ok
    if not prox:
        say("no design meets bars + proxy: proxy dropped (pre-registered)")
    cand.sort(key=lambda s: (s["robust"], s["wrobust"]), reverse=True)
    pick = None
    for s in cand[:8]:
        p, med = nb_median_pass(s)
        say("  candidate", fmt(s), "| nb median", {k: round(v * 100, 2) for k, v in med.items()}, "PASS" if p else "fail")
        if p:
            pick = s
            break
    if pick is None:
        say("NO PICK: nothing passes the 2021-23 selection")
    else:
        d = {k: pick[k] for k in F.BASE}
        say("PICK", json.dumps(d))
        json.dump(d, open(os.path.join(F.OUT, "pick.json"), "w"))
    say(f"done {time.time()-t0:.0f}s, {len(SEEN)} designs scored")


if __name__ == "__main__":
    main()
