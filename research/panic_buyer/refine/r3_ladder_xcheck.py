# Cross-check (prints only True/False, no metrics): lrun == r2_crowding.crun for split-ladder settings, mtp_now off.
import sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import numpy as np
import study as S
import engine as E
from r3_ladder import lrun, LOG
from r2_crowding import crun
specs = []
for w, n, fm in ((0.5, 4, 0.4), (2 / 3, 5, 0.3), (0.5, 4, 0.25), (2 / 3, 5, 0.20), (0.5, 5, 0.2), (2 / 3, 4, 0.25)):
    for d2 in (0.16, 0.18, 0.20, 0.24):
        specs.append(dict(w=w, N=n, fmax=fm, d2=d2))
allok = True
for dead in (False, True):
    for pen in (0.0, 0.002):
        syms, idx, P = S.market(dead)
        for sp in specs:
            e1, l1 = lrun(P, idx, pen=pen, **sp)
            e2, l2 = crun(P, idx, pen=pen, **sp)
            same = np.array_equal(e1.values, e2.values) and l1[LOG[:-1]].equals(l2)
            allok &= same
            if not same:
                print("MISMATCH", dead, pen, sp)
print("lrun identical to r2 crun on", 2 * 2 * len(specs), "ladder runs:", allok)
