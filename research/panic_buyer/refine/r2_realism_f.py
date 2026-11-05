"""Round 2, lens "realism", POST-HOC diagnostic (written after r2_realism.py ran; changes none of its numbers).
Row (f) "live-bot timing for market buys" moved 2021-23 from x5.74 to x4.72 but 2024-26 from x2.61 to x2.66.
Split it into its two parts (take-profit placed 1 minute after a market buy; time stop 1 hour earlier for market
buys) and show per-year returns and what happened to the market-buy trades themselves."""
import os

import numpy as np
import pandas as pd

import engine as E
import study as S
import r2_realism_lib as R

D = {"LIVE (1-minute engine)": {}, "(f) both parts": dict(mdelay=1, mtsh=1),
     "(f1) market-buy take-profit 1 min after the buy only": dict(mdelay=1),
     "(f2) market-buy time stop 1h earlier only": dict(mtsh=1)}

if __name__ == "__main__":
    out = open(os.path.join(S.OUT, "r2_realism_f.log"), "w")

    def say(s):
        print(s); out.write(s + "\n"); out.flush()

    idx, A, hidx, Ph = E.minute_panel(E.H19)
    logs = {}
    for name, kw in D.items():
        e, lg = R.rrun(A, hidx, Ph, **kw)
        r = E.score(e, lg, hidx); logs[name] = lg
        say(E.line(name, r))
        say("    per year: " + " ".join(f"{y}:{v*100:+.1f}%" for y, v in E.years(e).items()))
        p = np.where(hidx[lg.eb.astype(int).values] < E.DESIGN_END, "2021-23", "2024-26")
        for tag in ("2021-23", "2024-26"):
            m = lg[(p == tag) & (lg.mkt == 1)]
            say(f"    {tag} market buys {len(m)}: avg {m.ret.mean()*100:+.2f}%, sold in the buy hour {int((m.eb == m.xb).sum())}, "
                f"take-profits {int((m.why == 0).sum())} (avg {m[m.why == 0].ret.mean()*100:+.2f}%), time stops "
                f"{int((m.why == 1).sum())} (avg {m[m.why == 1].ret.mean()*100:+.2f}%), sum ret x size {float((m.ret*m.amt).sum()):+.2f}")
