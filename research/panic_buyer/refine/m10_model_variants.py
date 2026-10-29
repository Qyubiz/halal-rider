"""m10 model, step 2: model variants, judged on 2021-23 ONLY (this script prints nothing from 2024+).

Variants (fixed before running; all walk-forward, monthly expanding window from 2018-06, LightGBM n_jobs=1):
  V0 lgb_all      step 1 model: all candidate hours, 39 features, P(take-profit)
  V1 lgb_first    only the first hour of each dip (coin not a candidate the hour before)
  V2 lgb_guard    only candidates the 35% crash guard lets the bot buy (dd7 >= -35%): the traded distribution
  V3 lgb_guard_w  V2 with losers weighted by their loss (|return| / 2.5%, capped at 20): focus on the deep losers
  V4 lgb_small    V2 with 17 regime features (depth, trends, drawdowns, volatility, BTC, breadth, volume)
  V5 lgb_reg      V2 as a regression on the trade return (clipped at -30%): predicted return as the score
  V6 lgb_slow     V2 heavily regularized (150 trees, 7 leaves, 300 rows per leaf)
  V7 logit_guard  V2 with logistic regression
Judged by: AUC on guard-passing first-hour candidates and on the unfiltered 90% mode's own buys, 2021-23; and the 90%
mode with the model's mask, blocking dips scored under the q-quantile (q = 10/20/30%) of the previous 12 months'
out-of-sample candidate scores (trailing_threshold), 2021-23 win rate, %/month, worst drop.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m10_model_lib as M  # noqa: E402
import w90_wr90_lib as W  # noqa: E402

SMALL = ["fill_depth", "r24", "pre3d", "r7d", "r30d", "dd7", "dd30", "vol168", "vol_ratio", "btc_r24", "btc_r7d",
         "btc_c1ref", "breadth7", "mkt_c1ref", "frac_deep", "rel24", "qv24"]
QS = (0.1, 0.2, 0.3)


def variant_preds(name, X, cand, y, ret, idx):
    p = os.path.join(M.OUT, f"preds_{name}.npy")
    if os.path.exists(p):
        return np.load(p)
    if name == "lgb_all":
        return np.load(os.path.join(M.OUT, "preds_lgb.npy"))
    fi = {f: i for i, f in enumerate(M.FEATS)}
    first = cand & ~np.vstack([np.zeros((1, cand.shape[1]), bool), cand[:-1]])
    guard = X[:, :, fi["dd7"]] >= -0.35
    ok = cand & ~np.isnan(y)
    rows = {"lgb_first": ok & first}.get(name, ok & guard)
    rt, rk = np.nonzero(rows)
    yy, rr = y[rt, rk], ret[rt, rk]
    kw = {}
    if name == "lgb_guard_w":
        kw["weight"] = np.where(yy == 0, np.clip(np.abs(rr) / 0.025, 1, 20), 1.0)
    if name == "lgb_small":
        kw["feat"] = [fi[f] for f in SMALL]
    if name == "lgb_slow":
        kw["params"] = dict(n_estimators=150, num_leaves=7, min_child_samples=300)
    if name == "logit_guard":
        kw["kind"] = "logit"
    tgt = np.clip(rr, -0.30, None) if name == "lgb_reg" else yy
    print(f"walk-forward {name}", flush=True)
    pred = M.walk_forward2(X, rt, rk, tgt, idx, **kw)
    np.save(p, pred)
    return pred


def main():
    syms_b, idx_b, P_b, X, cand, fill, y, ret, xh = M.build()
    fi = {f: i for i, f in enumerate(M.FEATS)}
    first = cand & ~np.vstack([np.zeros((1, cand.shape[1]), bool), cand[:-1]])
    guard = X[:, :, fi["dd7"]] >= -0.35
    bot = np.load(os.path.join(M.OUT, "bot_entries.npy"))
    yr = np.asarray(idx_b.year)
    d = ((yr >= 2021) & (yr <= 2023))[:, None]
    syms, idx, P = W.market(False)
    names = ["lgb_all", "lgb_first", "lgb_guard", "lgb_guard_w", "lgb_small", "lgb_reg", "lgb_slow", "logit_guard"]
    if len(sys.argv) > 1:
        names = sys.argv[1:]
    base = W.summarize(*W.wrun(P, idx, **M.F2), idx)
    print(f"no model (90% mode): 2021-23 WR {base['d_wr']*100:.1f}% ({base['d_n']} trades) {base['d_mo']*100:+.2f}%/mo "
          f"drop {base['d_dd']*100:.0f}%", flush=True)
    rows = []
    for nm in names:
        pred = variant_preds(nm, X, cand, y, ret, idx_b)
        r = {"variant": nm,
             "auc_first_guard": M.auc(y[cand & first & guard & d], pred[cand & first & guard & d]),
             "auc_bot": M.auc(y[bot & cand & d], pred[bot & cand & d])}
        for q in QS:
            thr = M.trailing_threshold(pred, cand & guard, idx_b, q)
            keep = np.where(np.isnan(pred), 1.0, (pred >= thr[:, None]).astype(float))
            sm = M.to_panel(keep, idx_b, syms_b, idx, syms, fill=1.0)
            s = W.summarize(*W.wrun(P, idx, smult=sm, **M.F2), idx)
            r[f"q{int(q*100)}"] = f"WR {s['d_wr']*100:4.1f}% n{s['d_n']:3d} {s['d_mo']*100:+5.2f}%/mo dd {s['d_dd']*100:3.0f}%"
        rows.append(r)
        print(" | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r.values()), flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(M.OUT, "variants_2021_23.csv"), index=False)
    pd.set_option("display.width", 250)
    print("\n2021-23 only:\n" + T.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
