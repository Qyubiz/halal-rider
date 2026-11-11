"""m16 riskmodel, step 3: the design grid, 2021-23 ONLY (nothing after 2023-12-31 or before 2021 is computed).

Designs (60): target {mae, big} x features {base, fut} x q {0.5, 0.6, 0.7, 0.8, 0.9} x floor {0.25, 0.375, 0.5}.
  Size = floor when the trade's risk score is above the q-quantile of the previous 365 days' out-of-sample scores
  of guard-passing candidates, else 100%.

SELECTION RULE (written before any run of this script; M14_PREREG.md bars):
  1. As designed, 2021-23, normal fills AND every fill 0.2% worse: win rate >= 92.5%, worst drop no deeper than
     -64.4%, %/month above 13.17 (normal) and 9.79 (worse).
  2. All 5 nearby versions (as designed, cooldown -1h, +1h, bot timing, start 2021-01-15), normal and worse fills,
     pass the win-rate and drop bars. Score = mean normal-fill 2021-23 %/month over the 5 versions.
  3. Neighbours = same target and features, q one step away or floor one step away. The median of the neighbours'
     normal-fill 2021-23 %/month, win rate and drop must pass bar 1 (normal).
  4. PICK = the highest score passing 1-3. None pass -> no pick (report the best by score as hindsight only).
Also printed: the four models' skill on 90% mode's own 2021-23 buys (Spearman of risk score vs the 96h worst close,
AUC for a realized loss below -10%).
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m16_riskmodel_lib as R  # noqa: E402
import m10_velocity_lib as V  # noqa: E402
import m10_model_lib as M  # noqa: E402

QS = [0.5, 0.6, 0.7, 0.8, 0.9]
FLS = [0.25, 0.375, 0.5]
DES = [dict(target=t, fs=f, q=q, floor=fl) for t, f, q, fl in itertools.product(["mae", "big"], ["base", "fut"], QS, FLS)]


def dkey(d):
    return f"{d['target']}_{d['fs']}_q{d['q']}_f{d['floor']}"


def main():
    syms_b, idx_b, P_b, Z = R.build()
    syms, idx, P = V.market(False)
    risks = {(t, f): R.train(t, f, Z, idx_b) for t in ("mae", "big") for f in ("base", "fut")}
    # skill on 90% mode's own buys, 2021-23
    e0, lg0, _ = R.srun(P, idx)
    tb = idx[lg0.eb.astype(int).values]
    sel = (tb >= V.T21) & (tb < V.T24)
    bi = idx_b.get_indexer(tb[sel]); bk = np.array([syms_b.index(syms[int(k)]) for k in lg0.k[sel]])
    mae = Z["mae"][bi, bk]; big = (lg0.ret[sel].values < -0.10).astype(float)
    print(f"90% mode 2021-23 buys: {sel.sum()}, losses < -10%: {int(big.sum())}")
    for (t, f), r in risks.items():
        s = r[bi, bk]
        print(f"skill 2021-23 {t:4s} {f:5s}: spearman(risk, -mae) {spearmanr(s, -mae, nan_policy='omit')[0]:+.3f}  "
              f"AUC(loss<-10%) {M.auc(big, s):.3f}", flush=True)
    rows = []
    for d in DES:
        sz = R.to_panel(R.size_big(risks[(d["target"], d["fs"])], Z, idx_b, d["q"], d["floor"],
                                   (d["target"], d["fs"])), idx_b, syms_b, idx, syms)
        o = dict(d, key=dkey(d))
        for tag, kw in R.VERS.items():
            for fl, fk in (("", {}), ("w", dict(pen=R.PEN, pens=R.PEN))):
                e, lg, inv = R.srun(P, idx, sz, **kw, **fk)
                per = (("d", R.L.LATE if tag == "ls" else V.T21, V.T24),)
                r = V.summarize(e, lg, inv, idx, per)
                o[f"{fl}{tag}_wr"], o[f"{fl}{tag}_mo"], o[f"{fl}{tag}_dd"] = r["d_wr"], r["d_mo"], r["d_dd"]
                if fl + tag == "n":
                    o["avgloss"] = float((lg.ret * lg.amt / lg.eq0)[np.asarray(sel) & (lg.ret.values < 0) if len(lg) == len(sel) else (lg.ret < 0)].mean())
                    o["wt"] = r["d_wt"]
        o["score"] = float(np.mean([o[f"{t}_mo"] for t in R.VERS]))
        rows.append(o)
    T = pd.DataFrame(rows)
    b1 = lambda wr, mo, dd, w=False: (wr >= 0.925) & (dd >= -0.644) & (mo > (0.0979 if w else 0.1317))  # noqa: E731
    T["p1"] = b1(T.n_wr, T.n_mo, T.n_dd) & b1(T.wn_wr, T.wn_mo, T.wn_dd, True)
    T["p2"] = np.all([(T[f"{fl}{t}_wr"] >= 0.925) & (T[f"{fl}{t}_dd"] >= -0.644) for t in R.VERS for fl in ("", "w")], 0)
    p3 = []
    for d in DES:
        nb = [x for x in DES if x["target"] == d["target"] and x["fs"] == d["fs"] and (
            (x["floor"] == d["floor"] and abs(QS.index(x["q"]) - QS.index(d["q"])) == 1) or
            (x["q"] == d["q"] and abs(FLS.index(x["floor"]) - FLS.index(d["floor"])) == 1))]
        N = T[T.key.isin([dkey(x) for x in nb])]
        p3.append(bool(b1(N.n_wr.median(), N.n_mo.median(), N.n_dd.median())))
    T["p3"] = p3
    T["ok"] = T.p1 & T.p2 & T.p3
    T.to_csv(os.path.join(R.OUT, "grid_2021_23.csv"), index=False)
    T = T.sort_values("score", ascending=False)
    print(f"\nbase 90% mode 2021-23: 93.8 / +13.17 / -64.4 (worse 93.4 / +9.79 / -65.9)")
    print(f"designs {len(T)}, pass1 {int(T.p1.sum())}, pass1+2 {int((T.p1 & T.p2).sum())}, all {int(T.ok.sum())}")
    for _, r in T.head(15).iterrows():
        print(f"{r.key:28s} score {r.score*100:+6.2f}  n {r.n_wr*100:.1f}/{r.n_mo*100:+.2f}/{r.n_dd*100:.1f}  "
              f"w {r.wn_wr*100:.1f}/{r.wn_mo*100:+.2f}/{r.wn_dd*100:.1f}  wt {r.wt*100:+.1f}  p {int(r.p1)}{int(r.p2)}{int(r.p3)}")
    print("drop range over all designs (normal):", f"{T.n_dd.min()*100:.1f} .. {T.n_dd.max()*100:.1f}",
          "mo range", f"{T.n_mo.min()*100:+.2f} .. {T.n_mo.max()*100:+.2f}")
    ok = T[T.ok]
    pick = None if ok.empty else {k: ok.iloc[0][k] for k in ("target", "fs", "q", "floor")}
    print("PICK:", json.dumps(pick))
    json.dump(pick, open(os.path.join(R.OUT, "pick.json"), "w"))


if __name__ == "__main__":
    main()
