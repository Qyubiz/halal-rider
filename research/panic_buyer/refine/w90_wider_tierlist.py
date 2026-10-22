"""Task "ten-wider": which coins outside the 19 the finalists' tiers contain (ever, and the current month), with their
approximate-screen category (code/universe.py pass_map). All of them need the user's own halal screener verdict."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w90_wider_lib as L  # noqa: E402  (puts code/ on the path)
import universe as U  # noqa: E402

if __name__ == "__main__":
    R = L.ranks_all()
    pm = U.pass_map()
    out = open(os.path.join(L.OUT, "tierlist.log"), "w")
    for lab, key, M in (("F1/F2: top 30 non-19 by 365-day volume", "r365o", 30), ("F3: top 20 non-19 by 30-day volume", "r30o", 20)):
        A = np.nan_to_num(R[key], nan=1e9)
        ever = [s for j, s in enumerate(R["syms"]) if (A[:, j] <= M).any()]
        now = sorted([(A[-1, j], s) for j, s in enumerate(R["syms"]) if A[-1, j] <= M])
        months = (A <= M)[~R["idx"].to_period("M").duplicated()].sum(0)
        s = (f"{lab}\n  in the tier at some point 2021-26: {len(ever)} coins\n  " +
             ", ".join(f"{x[:-4]}({int(months[R['syms'].index(x)])}mo)" for x in sorted(ever)) +
             f"\n  tier for {R['idx'][-1]:%B %Y} ({len(now)}): " +
             ", ".join(f"{x[:-4]} [{pm.get(x[:-4].replace('LUNA2', 'LUNA'), '?')}]" for _, x in now))
        print(s); out.write(s + "\n\n")
    out.close()
