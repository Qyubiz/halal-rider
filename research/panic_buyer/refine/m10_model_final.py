"""m10 model, step 3: the picks, fixed on 2021-23 (variants_2021_23.csv, m10_model_variants.py), on every period.

Pick rule, fixed before any 2024+ or holdout strategy result of a model variant was looked at: among the 24
(variant, q) cells of m10_model_variants.py, the highest 2021-23 %/month with a 2021-23 win rate >= the unfiltered
90% mode's (93.8%) and a worst drop no more than 1 point deeper (>= -65.4%).
  PICK  lgb_reg, q = 20%: LightGBM regression on the trade return, block a dip when its predicted return is under the
        20% quantile of the previous 12 months' out-of-sample candidate scores. 2021-23: +15.56%/mo, WR 94.6%, -64%.
  WR    (second, for the win-rate goal) the best 2021-23 win rate: logit_guard, q = 30%: 95.5%, +13.55%/mo, -64%.
Base: 90% mode (1 coin x 100%, 14%, +2.5%, 96h, cd 3h, crash guard 35%) without a model.

Periods: 2018-06..2020-12 holdout (16 coins, no trading before 2018-06 for every row, models trained only on earlier
data from Dec 2017), 2021-23, 2024-26, since 2025. "Fills 0.2% worse": every buy and every sell 0.2% worse.
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

PICKS = {"PICK lgb_reg q20": ("lgb_reg", 0.2), "WR logit_guard q30": ("logit_guard", 0.3)}
PER = (("2018-06..20 holdout", "2018-06-01", "2021-01-01", "pre"), ("2021-23", "2021-01-01", "2024-01-01", "main"),
       ("2024-26", "2024-01-01", None, "main"), ("since 2025", "2025-01-01", None, "main"))


def stats(e, lg, idx, a, b):
    a = pd.Timestamp(a); b = pd.Timestamp(b) if b else None
    mult, mo, dd = W.period(e, a, b)
    tb = idx[lg.eb.astype(int).values]
    sel = (tb >= a) & ((tb < b) if b is not None else True)
    x = lg[sel]
    end = b if b is not None else idx[-1] + pd.Timedelta(hours=1)
    months = (end - a).total_seconds() / (30.4375 * 86400)
    acct = (x.ret * x.amt / x.eq0)
    return dict(wr=float((x.ret > 0).mean()), mo=mo, dd=dd, wt=float(acct.min()), tpm=len(x) / months, n=len(x))


def main():
    syms_b, idx_b, P_b, X, cand, fill, y, ret, xh = M.build()
    fi = {f: i for i, f in enumerate(M.FEATS)}
    guard = X[:, :, fi["dd7"]] >= -0.35
    bot = np.load(os.path.join(M.OUT, "bot_entries.npy"))
    panels = {"main": W.market(False)}
    D = E.load_pre()
    panels["pre"] = E.panel(D)
    keeps = {"Base 90% mode": None}
    preds = {}
    for name, (v, q) in PICKS.items():
        pred = np.load(os.path.join(M.OUT, f"preds_{v}.npy"))
        preds[name] = pred
        thr = M.trailing_threshold(pred, cand & guard, idx_b, q)
        keeps[name] = np.where(np.isnan(pred), 1.0, (pred >= thr[:, None]).astype(float))
    rows = []
    for name, keep in keeps.items():
        for fills, pen in (("normal", 0.0), ("fills 0.2% worse", 0.002)):
            runs = {}
            for pk, (syms, idx, P) in panels.items():
                sm = np.ones((len(idx), len(syms))) if keep is None else M.to_panel(keep, idx_b, syms_b, idx, syms, 1.0)
                if pk == "pre":
                    sm[idx < M.WF_START] = 0.0          # no trading before the first model exists (all rows)
                runs[pk] = W.wrun(P, idx, smult=sm, pen=pen, pens=pen, **M.F2) + (idx,)
            for lab, a, b, pk in PER:
                e, lg, idx = runs[pk]
                rows.append(dict(rule=name, fills=fills, period=lab, **stats(e, lg, idx, a, b)))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(M.OUT, "final.csv"), index=False)
    for fills in ("normal", "fills 0.2% worse"):
        print(f"\n=== {fills} ===")
        for _, r in T[T.fills == fills].iterrows():
            print(f"{r.rule:20s} {r.period:20s} WR {r.wr*100:5.1f}%  {r.mo*100:+6.2f}%/mo  drop {r.dd*100:5.1f}%  "
                  f"worst trade {r.wt*100:+6.1f}% acct  {r.tpm:4.1f} trades/mo ({r.n})")
    # out-of-sample AUC per year for the picks: all candidate hours, and the unfiltered bot's own buys
    yr = np.asarray(idx_b.year)
    print("\nOOS AUC per year (all candidates / unfiltered 90% mode buys)")
    for name, pred in preds.items():
        out = []
        for Y in range(2018, 2027):
            m = ((yr == Y) & (idx_b >= M.WF_START))[:, None]
            out.append(f"{Y}: {M.auc(y[cand & m], pred[cand & m]):.2f}/{M.auc(y[bot & cand & m], pred[bot & cand & m]):.2f}")
        print(f"{name:20s} " + "  ".join(out))


if __name__ == "__main__":
    main()
