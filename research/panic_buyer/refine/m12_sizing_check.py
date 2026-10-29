"""m12 sizing: check that sk (every sizing switch off) reproduces 90% mode exactly (m10_velocity_lib.vk) under normal
fills, worse fills, stressed (+LUNA/FTT/SRM) and the 2018-20 holdout; that the 2021-23 cut run equals the full run on
2021-23; prints 90% mode's table (compare M12_PREREG.md). Usage: source m12env; python3 m12_sizing_check.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m12_sizing_lib as S  # noqa: E402
import m10_velocity_lib as V  # noqa: E402


def main():
    ok = True
    for lab, extra, hold, dead in (("normal", {}, False, False), ("worse", dict(pen=S.PEN, pens=S.PEN), False, False),
                                   ("stressed", None, False, True), ("holdout", {}, True, False),
                                   ("holdout worse", dict(pen=S.PEN, pens=S.PEN), True, False)):
        syms, idx, P = V.market_holdout(dead) if hold else V.market(dead)
        k = V.stress_kw(V.MODE90) if extra is None else dict(V.MODE90, **extra)
        e1, l1, _ = V.vrun(P, idx, **k)
        sk = {kk: k[kk] for kk in ("pen", "pens", "mtp_now", "cd_add", "liquidate") if kk in k}
        e2, l2, i2 = S.srun(P, idx, {}, **sk)
        same = np.allclose(e1.values, e2.values, rtol=1e-12, atol=0) and len(l1) == len(l2) and \
            np.allclose(l1.values, l2.values[:, :9], rtol=1e-12)
        ok &= same
        r = V.summarize(e2, l2, i2, idx, V.HOLD if hold else V.MAIN)
        print(f"{lab:14s} identical={same}  final {e1.values[-1]:.6g} vs {e2.values[-1]:.6g}")
        print(V.fmt("  90% mode", r, ("h",) if hold else ("d", "t", "s")))
    syms, idx, P = V.market(False)
    ne = int(np.searchsorted(idx.values, np.datetime64(S.T24)))
    e3, _, _ = S.srun(P, idx, {}, n_end=ne)
    e2, _, _ = S.srun(P, idx, {})
    cut = np.allclose(e3.values, e2.values[:ne], rtol=1e-12)
    print("2021-23 cut run == full run on 2021-23:", cut)
    ds = S.design_scores({})
    print("design_scores(90% mode):", {k: round(v, 4) for k, v in ds.items()})
    print("ALL OK" if ok and cut else "MISMATCH")


if __name__ == "__main__":
    main()
