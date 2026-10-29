"""x10 critic: dependence of the 1-minute designs on single-minute wicks. Re-runs the designs on a 'de-wicked' minute
panel where each minute's high is capped at max(open, close) * (1 + w) and its low floored at min(open, close) * (1 - w)
(the body is kept, only the part of the wick beyond w is removed), for w = 1% and 0.5%. Resting limit orders (dip bids,
take-profits, the +8% cap) then cannot fill on prints far outside the minute's open/close. 19 coins, normal costs."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import x10_critic_lib as X  # noqa: E402
import x10_critic_velocity as CV  # noqa: E402
import x10_velocity_lib as V  # noqa: E402

A, hidx, Ph, SLIP = V.load()
CV.X_hidx = hidx
K = 19
o = np.ascontiguousarray(A["o"][:, :K]); c = np.ascontiguousarray(A["c"][:, :K])
hi_body = np.fmax(o, c); lo_body = np.fmin(o, c)
designs = CV.DESIGNS + [("H4 core 19, 1-min watch hourly cycle", dict(N=1, fmax=1.0, depth=0.14, tp=0.05, hm=48, cd=24,
                                                                       watch=1, cyc=60, _guard=True))]
for w in (None, 0.01, 0.005):
    if w is None:
        B = {"o": o, "c": c, "h": np.ascontiguousarray(A["h"][:, :K]), "l": np.ascontiguousarray(A["l"][:, :K])}
    else:
        B = {"o": o, "c": c, "h": np.fmin(A["h"][:, :K], (hi_body * np.float32(1 + w)).astype(np.float32)),
             "l": np.fmax(A["l"][:, :K], (lo_body * np.float32(1 - w)).astype(np.float32))}
    print(f"\n######## {'original minutes' if w is None else f'wicks beyond {w*100:.1f}% of the candle body removed'}", flush=True)
    for name, kw in designs:
        kw = dict(kw)
        ref = X.guard_ref(Ph, 0.35, K) if kw.pop("_guard", False) else None
        e, lg, _ = X.xrun(B, hidx, Ph, SLIP, K=K, ref=ref, **kw)
        r = V.report(e, lg, hidx)
        print(f"  {name:40s} {CV.line(r)}", flush=True)
