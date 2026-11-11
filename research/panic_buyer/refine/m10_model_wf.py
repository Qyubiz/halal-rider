"""m10 model, step 1: walk-forward predictions and out-of-sample AUC.

Monthly expanding-window retraining from 2018-06 (training data from Dec 2017), LightGBM (small: 300 trees, 15 leaves,
n_jobs=1) and a logistic-regression reference. Every prediction for month M comes from a model fitted only on
candidates bought at least 97 hours before M's first hour. Saves preds_<kind>.npy (n, K) in PB_OUT.

AUC per calendar year on: all candidate hours; first hours of a dip (coin not a candidate the hour before); and the
trades the unfiltered 90% mode actually bought (engine run on the same panels as the published numbers).
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402
import m10_model_lib as M  # noqa: E402
import w90_wr90_lib as W  # noqa: E402


def bot_entries(idx_big, syms_big):
    """(n_big, K) bool: hours/coins where the unfiltered 90% mode bought (holdout panel 2018-20 + main panel 2021+)."""
    out = np.zeros((len(idx_big), len(syms_big)), bool)
    for P_src in ("pre", "main"):
        D = E.load_pre() if P_src == "pre" else E.load_h19()
        syms, idx, P = E.panel(D)
        e, lg = W.wrun(P, idx, **M.F2)
        for _, r in lg.iterrows():
            t = idx[int(r.eb)]; s = syms[int(r.k)]
            i = idx_big.get_loc(t); j = syms_big.index(s)
            out[i, j] = True
    return out


def main():
    syms, idx, P, X, cand, fill, y, ret, xh = M.build()
    print(f"panel {idx[0]} .. {idx[-1]}, {len(syms)} coins, {int(cand.sum())} candidate hours, "
          f"take-profit rate {np.nanmean(y[cand])*100:.1f}%", flush=True)
    for kind in ("lgb", "logit"):
        p = os.path.join(M.OUT, f"preds_{kind}.npy")
        if not os.path.exists(p):
            print(f"walk-forward {kind}", flush=True)
            pred = M.walk_forward(X, cand, y, xh, kind=kind, idx=idx)
            np.save(p, pred)
    first = cand & ~np.vstack([np.zeros((1, cand.shape[1]), bool), cand[:-1]])
    bot = bot_entries(idx, syms)
    np.save(os.path.join(M.OUT, "bot_entries.npy"), bot)
    yr = np.asarray(idx.year)
    rows = []
    for kind in ("lgb", "logit"):
        pred = np.load(os.path.join(M.OUT, f"preds_{kind}.npy"))
        for Y in list(range(2018, 2027)) + ["2018-20", "2021-23", "2024-26", "2025-26"]:
            if isinstance(Y, str):
                a, b = int(Y[:4]), 2000 + int(Y[-2:])
                m = (yr >= a) & (yr <= b)
            else:
                m = yr == Y
            m = m[:, None] & (idx >= M.WF_START)[:, None]
            r = {"model": kind, "year": Y}
            for name, sel in (("all", cand), ("first", first), ("bot", bot & cand)):
                s = sel & m
                r[f"auc_{name}"] = M.auc(y[s], pred[s])
                r[f"n_{name}"] = int(s.sum())
                r[f"lose_{name}"] = int((y[s] == 0).sum())
            rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(M.OUT, "auc_by_year.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\nOut-of-sample AUC (prediction = P(take-profit within 96h); 0.5 = no skill). n = candidates, lose = losers")
    print(T.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    # feature importance of the last model (trained on everything before the last month)
    import lightgbm  # noqa: F401
    a = int(np.searchsorted(idx.values, np.datetime64(pd.Timestamp(idx[-1]).to_period("M").start_time)))
    rt, rk = np.nonzero(cand & ~np.isnan(y))
    sel = rt + M.HM < a
    mdl = M.make_model("lgb").fit(X[rt[sel], rk[sel]], y[rt[sel], rk[sel]])
    imp = pd.Series(mdl.booster_.feature_importance("gain"), M.FEATS).sort_values(ascending=False)
    print("\nfeature importance (gain, last model):\n" + (imp / imp.sum()).round(3).to_string())


if __name__ == "__main__":
    main()
