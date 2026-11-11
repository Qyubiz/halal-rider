"""m16 riskmodel, step 2: walk-forward risk scores for one (target, feature set): python3 m16_riskmodel_train.py mae fut"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m16_riskmodel_lib as R  # noqa: E402

target, fs = sys.argv[1], sys.argv[2]
syms, idx, P, Z = R.build()
if target == "cov":
    ok = Z["cand"] & Z["guard"]
    yr = np.asarray(idx.year)
    for Y in range(2018, 2027):
        m = ok & (yr == Y)[:, None]
        x = Z["X"][m][:, len(R.M.FEATS):]
        print(Y, int(m.sum()), "has funding", f"{np.mean(~np.isnan(x[:, 0])):.2f}", "premium",
              f"{np.mean(~np.isnan(x[:, 3])):.2f}", "oi", f"{np.mean(~np.isnan(x[:, 6])):.2f}",
              "big-loss rate", f"{np.nanmean(Z['ret'][m] < -0.10):.3f}", "mean mae", f"{np.nanmean(Z['mae'][m]):.3f}")
    sys.exit()
R.train(target, fs, Z, idx, log=lambda *a, **k: print(target, fs, *a, flush=True))
print("done", target, fs)
