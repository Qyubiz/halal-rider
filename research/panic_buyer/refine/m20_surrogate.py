"""m20 surrogate: can the m16 risk model's flag (score in the top 10% of the last 365 days) be reduced to a few simple
rules? A shallow decision tree is fitted to the MODEL'S FLAG (not to trade outcomes) on candidate coin-hours before
2021 only, then applied everywhere and run through the engine like any other flag (bet 50% when flagged).
Hindsight description: all periods are shown; nothing here is a pre-registered pick."""
import os, sys, json
import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m10_velocity_lib as V, m10_model_lib as M, m16_riskmodel_lib as R, m20_flags as G

B = G.big()
Z, idx_b, syms_b = B["Z"], B["idx"], B["syms"]
X = Z["X"][:, :, :len(M.FEATS)]
ok = Z["cand"] & Z["guard"]
risk = R.train("big", "base", Z, idx_b)
out = {}
for q in (0.9, 0.8):
    thr = R.thresholds(risk, Z, idx_b, q, ("big", "base"))
    flag = risk > thr[:, None]
    rt, rk = np.nonzero(ok & np.isfinite(risk) & np.isfinite(thr)[:, None])
    Xr = np.nan_to_num(X[rt, rk], nan=-9.0)
    y = flag[rt, rk].astype(int)
    tr = idx_b[rt] < pd.Timestamp("2021-01-01")
    print(f"\n=== model flag top {100*(1-q):.0f}%: {len(y)} candidate hours with a score, {y.mean()*100:.1f}% flagged; "
          f"fit on {tr.sum()} rows before 2021")
    for depth in (1, 2, 3):
        t = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=100, class_weight="balanced", random_state=0)
        t.fit(Xr[tr], y[tr])
        for lab, m in (("before 2021", tr), ("2021-23", (idx_b[rt] >= V.T21) & (idx_b[rt] < V.T24)), ("2024-26", idx_b[rt] >= V.T24)):
            p = t.predict(Xr[m]); yy = y[m]
            print(f"  depth {depth} {lab}: flags {p.mean()*100:.0f}% of rows; catches {((p==1)&(yy==1)).sum()/max(yy.sum(),1)*100:.0f}% "
                  f"of the model's flags; {((p==1)&(yy==1)).sum()/max(p.sum(),1)*100:.0f}% of its flags are the model's")
        print(export_text(t, feature_names=list(M.FEATS), decimals=4))
        out[(q, depth)] = t
import pickle
pickle.dump(out, open(os.path.join(R.OUT, "m20_surrogates.pkl"), "wb"))
