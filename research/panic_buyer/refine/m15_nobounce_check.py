import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m15_nobounce_lib as L, m10_velocity_lib as V
for ho in (False, True):
    for w in (False, True):
        r, e, lg = L.evaluate(L.OFF, worse=w, holdout=ho)
        ks = ("h",) if ho else ("d", "t", "s")
        print("hold" if ho else "main", "worse" if w else "norm", [(k, round(r[k+"_wr"]*100,1), round(r[k+"_mo"]*100,2), round(r[k+"_dd"]*100,1)) for k in ks])
