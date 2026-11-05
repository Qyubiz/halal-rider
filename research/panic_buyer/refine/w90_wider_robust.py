"""Task "ten-wider": extra robustness for the finalists (after w90_wider_final.py).
  1. Fine tier-depth scan around F2 (24-28% in 0.5-point steps): how narrow is the 10% peak?
  2. Leave-coin-out: F1/F2/F3 without their biggest outside-19 contributor(s) in 2024-26.
  3. The 19-coin core's own depth scan (13-15%) with the F2 tier, to separate core sensitivity from tier sensitivity.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w90_wider_final as F  # noqa: E402
import w90_wider_lib as L  # noqa: E402

if __name__ == "__main__":
    W = L.build()
    out = open(os.path.join(L.OUT, "robust.log"), "w")

    def say(s=""):
        print(s, flush=True)
        out.write(s + "\n"); out.flush()
    say("1. F2 tier-depth fine scan (everything else F2)")
    res = []
    for dt in np.arange(0.24, 0.2801, 0.005):
        r, *_ = L.evaluate(dict(F.FINALISTS["F2"], dt=round(float(dt), 4)), W=W)
        res.append(r)
        say(f"   tier depth {dt * 100:4.1f}%: " + L.fmt(r) + f" | worst-of-three {r['min_mo'] * 100:.2f}")
    m = np.array([x["min_mo"] for x in res]); t = np.array([x["t_mo"] for x in res])
    say(f"   mean over the scan: worst-of-three {m.mean() * 100:.2f}%/month, 2024-26 {t.mean() * 100:.2f}%, "
        f"{sum(x['ten'] for x in res)} of {len(res)} qualify")
    say("\n   same scan for F3 (v30 top20 x0.75), tier depth 18-22%")
    res = []
    for dt in np.arange(0.18, 0.2201, 0.005):
        r, *_ = L.evaluate(dict(F.FINALISTS["F3"], dt=round(float(dt), 4)), W=W)
        res.append(r)
        say(f"   tier depth {dt * 100:4.1f}%: " + L.fmt(r) + f" | worst-of-three {r['min_mo'] * 100:.2f}")
    m = np.array([x["min_mo"] for x in res]); t = np.array([x["t_mo"] for x in res])
    say(f"   mean over the scan: worst-of-three {m.mean() * 100:.2f}%/month, 2024-26 {t.mean() * 100:.2f}%, "
        f"{sum(x['ten'] for x in res)} of {len(res)} qualify")

    say("\n2. leave-coin-out (2024-26 biggest outside-19 contributors)")
    for k, ex in (("F1", ["HEMIUSDT"]), ("F2", ["HEMIUSDT"]), ("F2", ["HEMIUSDT", "ZECUSDT", "MOVEUSDT", "MIRAUSDT"]),
                  ("F3", ["HEMIUSDT"]), ("F3", ["HEMIUSDT", "PROMUSDT", "DASHUSDT", "ZECUSDT"])):
        r, *_ = L.evaluate(dict(F.FINALISTS[k], excl=tuple(ex)), W=W)
        say(f"   {k} without {', '.join(s[:-4] for s in ex)}: " + L.fmt(r))

    say("\n3. core depth scan with the F2 tier (tier 26%) vs the 19 alone")
    for dc in (0.13, 0.135, 0.14, 0.145, 0.15):
        r, *_ = L.evaluate(dict(F.FINALISTS["F2"], dc=dc), W=W)
        r0, *_ = L.evaluate(dict(F.FINALISTS["F2"], dc=dc, M=0), W=W)
        say(f"   core {dc * 100:4.1f}%: with tier {L.fmt(r)}")
        say(f"               19 only   {L.fmt(r0)}")
    out.close()
