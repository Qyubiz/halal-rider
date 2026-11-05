"""w90 ten-watch, step 2: light tuning of the realistic minute-level watch bot (w90_watch_lib mode 1: market buy at
the next minute's open + 0.05% after a minute's low crosses the trigger, take-profit 0.1% through, wicks trimmed to
1% beyond the body, 19 coins, fees 0.1%/side). Everything here is in-sample on BOTH periods (the selection rule looks
at 2021-23 and 2024-26), so any pick from it is found by tuning on 2024-26 too.

Grid A (10% hunt, 1 coin x 100%): depth 12/13/14/15/16/18% x tp 3/4/5/6/8% x guard off/25/35/45% x time stop
         24/48/96h; cooldown 24h.                                                                   360 runs
Grid B (win-rate frontier): tp 1/1.5/2/2.5% x depth 10/12/14% x sizing 1x100% / 2x50% / 3x33% / 8x20%
         x guard off/35%; time stop 48h, cooldown 24h.                                               96 runs
Grid C (written after seeing A and B: the best >=90%-win design sat at the grid edge, time stop 96h): 1 coin x 100%,
         tp 2.5/3/3.5% x depth 13/14/15% x time stop 72/96/120/168h x guard off/35/45%.                108 runs
Grid D (written after seeing A and B, lower-drop >=90%-win designs): 2x50% / 3x33%, tp 2/2.5/3% x depth 12/13/14%
         x time stop 48/96h x guard off/35%.                                                         72 runs
usage: python w90_watch_grid.py A|B|C|D [part nparts]
"""
import itertools
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w90_watch_lib as W  # noqa: E402


def grid(which):
    G = []
    if which == "A":
        for d, tp, g, hm in itertools.product((0.12, 0.13, 0.14, 0.15, 0.16, 0.18), (0.03, 0.04, 0.05, 0.06, 0.08),
                                              (0.0, 0.25, 0.35, 0.45), (24, 48, 96)):
            G.append((f"A N1 d{d*100:.0f} tp{tp*100:g} g{g*100:.0f} hm{hm}",
                      dict(N=1, fmax=1.0, depth=d, tp=tp, guard=g, hm=hm, cd=24)))
    elif which == "C":
        for tp, d, hm, g in itertools.product((0.025, 0.03, 0.035), (0.13, 0.14, 0.15), (72, 96, 120, 168),
                                              (0.0, 0.35, 0.45)):
            G.append((f"C N1 d{d*100:.0f} tp{tp*100:g} g{g*100:.0f} hm{hm}",
                      dict(N=1, fmax=1.0, depth=d, tp=tp, guard=g, hm=hm, cd=24)))
    elif which == "D":
        for (n, f), tp, d, hm, g in itertools.product(((2, 0.5), (3, 1 / 3)), (0.02, 0.025, 0.03), (0.12, 0.13, 0.14),
                                                      (48, 96), (0.0, 0.35)):
            G.append((f"D N{n} d{d*100:.0f} tp{tp*100:g} g{g*100:.0f} hm{hm}",
                      dict(N=n, fmax=f, depth=d, tp=tp, guard=g, hm=hm, cd=24)))
    else:
        for tp, d, (n, f), g in itertools.product((0.01, 0.015, 0.02, 0.025), (0.10, 0.12, 0.14),
                                                  ((1, 1.0), (2, 0.5), (3, 1 / 3), (8, 0.2)), (0.0, 0.35)):
            G.append((f"B N{n} d{d*100:.0f} tp{tp*100:g} g{g*100:.0f} hm48",
                      dict(N=n, fmax=f, depth=d, tp=tp, guard=g, hm=48, cd=24)))
    return G


if __name__ == "__main__":
    which = sys.argv[1]
    part, nparts = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
    G = grid(which)[part::nparts]
    ctx = W.Ctx()
    rows = []
    t0 = time.time()
    path = os.path.join(W.OUT, f"grid{which}_{part}.csv")
    for n, (name, kw) in enumerate(G):
        r = ctx.rep(19, **kw)
        r.update(name=name, **{f"p_{k}": v for k, v in kw.items()})
        rows.append(r)
        print(f"[{n+1}/{len(G)} {time.time()-t0:.0f}s] {name:34s} {W.line(r)}", flush=True)
        if (n + 1) % 20 == 0 or n + 1 == len(G):
            pd.DataFrame(rows).to_csv(path, index=False)
    print(f"done {len(G)} runs in {time.time()-t0:.0f}s -> {path}")
