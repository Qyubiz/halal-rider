"""m11 cashuse: the ONE look at 2024-26, since 2025, the 2018-20 holdout, worse fills and the stressed run for the
design picked on 2021-23 by m11_cashuse_grid.py (rule in its header; pick = best average of 5 versions, grid_c.log).
No re-picking after this run. Rows: normal fills; every buy and sell 0.2% worse; stressed = real bot timing
(take-profit at once after a market buy, cooldowns +1h) + fills 0.2% worse + LUNA/FTT/SRM sold at their last hour."""
import json
import os

import pandas as pd

import m11_cashuse_lib as C

PICK = json.loads(pd.read_csv(os.path.join(C.OUT, "grid_c.csv")).iloc[0]["kw"])
for name, kw in (("90% MODE (second engine off)", {}), ("PICK " + json.dumps(PICK), PICK)):
    print(name)
    for tag, opt in (("normal", {}), ("worse", dict(worse=True)), ("stressed", dict(stress=True))):
        r, *_ = C.evaluate(kw, **opt)
        for p in ("d", "t", "s"):
            print(C.fmt(tag, p, r))
        r, *_ = C.evaluate(kw, holdout=True, **opt)
        print(C.fmt(tag, "h", r))
    print()
