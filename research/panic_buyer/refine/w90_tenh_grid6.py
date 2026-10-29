"""Grid 6 (written after grids 1-5, before running this): fill in the win-rate frontier between the live bot
(8 coins x 20%) and the all-in designs, crash guard 35%/168h.
N {1, 2, 3, 4} with fmax 1/N x depth {12, 13, 14}% x take-profit {1.5, 2, 3}% x time stop {48, 96}h x cooldown
24h with or without re-arm after a win = 144 designs, plus the live settings with and without the guard.
Same fixed selection rule (HINDSIGHT)."""
import itertools
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402
import w90_tenh_lib as T  # noqa: E402
import w90_tenh_report as R  # noqa: E402


def designs():
    out = {"LIVE (N8 x20% d12 tp1.5 hm48 cd24)": dict(E.LIVE), "LIVE + guard 35/168": dict(E.LIVE, g=0.35, gL=168)}
    for N, d, tp, hm, w in itertools.product([1, 2, 3, 4], [0.12, 0.13, 0.14], [0.015, 0.02, 0.03], [48, 96], [-1, 1]):
        out[f"N{N} x{100/N:.0f}% d{d*100:.0f} tp{tp*100:g} hm{hm} cd24{' w1' if w == 1 else ''}"] = dict(
            N=N, fmax=round(1.0 / N, 4), depth=d, tp=tp, hm=hm, cd=24, cdw=w, g=0.35, gL=168)
    return out


if __name__ == "__main__":
    t0 = time.time()
    D = designs()
    rows = []
    for name, s in D.items():
        r, _, _ = T.evaluate(s)
        rows.append(T.row(name, s, r))
    G = pd.DataFrame(rows)
    R.report(G, "grid6", D, stress_top=4, extra=["LIVE (N8 x20% d12 tp1.5 hm48 cd24)"])
    print(f"{len(G)} designs, {time.time()-t0:.0f}s")
