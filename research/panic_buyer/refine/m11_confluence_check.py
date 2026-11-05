"""m11 confluence: reproduce 90% mode through ck with every vote off (must match m10_velocity_lib.vk / the published
numbers within rounding): 2021-26 on the universe panel (core coins only), 2018-20 holdout, worse fills, stressed."""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m11_confluence_lib as C  # noqa: E402
import m10_velocity_lib as V  # noqa: E402

D90 = dict()          # every switch at its default = 90% mode


def rows(d, tag=""):
    out = {}
    t0 = time.time()
    M = C.market("w")
    e, lg = C.run(M, d); out["normal"] = C.summ(e, lg, C.PER_MAIN)
    e, lg = C.run(M, d, pen=0.002, pens=0.002); out["worse"] = C.summ(e, lg, C.PER_MAIN)
    Ms = C.market("ws")
    e, lg = C.run(Ms, d, pen=0.002, pens=0.002, mtp_now=True, cd_add=1); out["stressed"] = C.summ(e, lg, C.PER_MAIN)
    H = C.market("h")
    e, lg = C.run(H, d); out["h_normal"] = C.summ(e, lg, C.PER_HOLD)
    e, lg = C.run(H, d, pen=0.002, pens=0.002); out["h_worse"] = C.summ(e, lg, C.PER_HOLD)
    Hs = C.market("hs")
    e, lg = C.run(Hs, d, pen=0.002, pens=0.002, mtp_now=True, cd_add=1); out["h_stressed"] = C.summ(e, lg, C.PER_HOLD)
    print(f"{tag} ({time.time() - t0:.0f}s)")
    for k in ("normal", "worse", "stressed"):
        for p in ("d", "t", "s"):
            print(C.line(k, p, out[k]))
    for k in ("h_normal", "h_worse", "h_stressed"):
        print(C.line(k[2:], "h", out[k]))
    return out


if __name__ == "__main__":
    mine = rows(D90, "ck, all votes off (90% mode)")
    print("\nreference: m10_velocity_lib.vk MODE90")
    ref = {}
    for tag, opt in (("normal", {}), ("worse", dict(worse=True)), ("stressed", dict(stress=True))):
        r, *_ = V.evaluate(V.MODE90, **opt); rh, *_ = V.evaluate(V.MODE90, holdout=True, **opt)
        ref[tag] = r; ref["h_" + tag] = rh
    bad = 0
    for k in mine:
        for p in (("h",) if k.startswith("h_") else ("d", "t", "s")):
            for m in ("wr", "mo", "dd", "wt", "tpm"):
                a, b = mine[k][f"{p}_{m}"], ref[k][f"{p}_{m}"]
                if not np.isclose(a, b, rtol=1e-9, atol=1e-12):
                    bad += 1
                    print(f"  DIFF {k} {p} {m}: ck {a:.6f} vk {b:.6f}")
    print("MATCH: every metric identical to vk" if bad == 0 else f"{bad} metrics differ")
