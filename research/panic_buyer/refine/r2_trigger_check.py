"""Round 2, lens "trigger": code check. trun / gkernel_t with the switches at the live setting must give exactly the
same equity curve and trade log as engine.run (live x5.75 / x2.61), with and without LUNA/FTT/SRM and with fills
0.2% worse. Only live-setting numbers are printed."""
import numpy as np

import engine as E
import study as S
import r2_trigger_lib as L

for dead, pen in ((False, 0.0), (True, 0.0), (False, 0.002)):
    syms, idx, P = S.market(dead)
    e0, l0 = L._ORIG_RUN(P, idx, pen=pen)
    cases = {"trun()": L.trun(P, idx, pen=pen),
             "trun(ref=ref_close 24h)": L.trun(P, idx, pen=pen, ref=L.ref_close(P, 24).values),
             "trun(cdts=24)": L.trun(P, idx, pen=pen, cdts=24),
             "trun(cdts=24, tsloss=-0.05)": L.trun(P, idx, pen=pen, cdts=24, tsloss=-0.05)}
    print(E.line(f"engine.run dead={dead} pen={pen}", E.score(e0, l0, idx)))
    for name, (e1, l1) in cases.items():
        same = np.array_equal(e0.values, e1.values) and np.array_equal(l0.values, l1.values)
        print(f"   {name:32s} identical equity and trade log: {same}")
        assert same
print("OK: the lens-trigger engine reproduces engine.run exactly with its switches off")
