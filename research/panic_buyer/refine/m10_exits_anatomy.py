"""m10 exits, step 1: anatomy of 90% mode's trades on 2021-23 ONLY (the design period), to motivate the exit grid.
For every trade bought in 2021-23: hours to exit, where the take-profit filled (first hourly check or later, at the
open = overshoot or at the target), the best price reached before the exit (MFE), the worst (MAE), and for losers what
the price did after the 96h time stop. Also the feature levels (vol7, rng24, dd7) at entry, whose 2021-23 medians become
the fixed reference levels for volatility / drop-scaled targets (REF in m10_exits_lib).
Usage: PB_DATA=... PB_OUT=... python3 m10_exits_anatomy.py"""
import numpy as np
import pandas as pd

import m10_exits_lib as X
import w90_wr90_lib as W

pd.set_option("display.width", 200)
syms, idx, P = W.market(False)
e, lg = X.xrun(P, idx, **X.BASE90)
tb = idx[lg.eb.astype(int).values]
d = lg[(tb >= pd.Timestamp("2021-01-01")) & (tb < pd.Timestamp("2024-01-01"))].copy()
H, L, C, O = P["h"].values, P["l"].values, P["cf"].values, P["o"].values
F = X.features(P)
rows = []
for r in d.itertuples():
    eb, xb, k = int(r.eb), int(r.xb), int(r.k)
    hh = H[eb + 1:xb + 1, k]; ll = L[eb + 1:xb + 1, k]
    mfe = np.nanmax(hh) / r.entry - 1 if len(hh) else np.nan
    mae = np.nanmin(np.r_[L[eb, k], ll]) / r.entry - 1
    first_open = O[eb + 1, k] / r.entry - 1
    after = {}
    for hz in (24, 48, 96, 168):
        j = min(xb + hz, len(idx) - 1)
        after[f"max_next{hz}"] = np.nanmax(H[xb + 1:j + 1, k]) / r.entry - 1
        after[f"close_next{hz}"] = C[j, k] / r.entry - 1
    rows.append(dict(eb=eb, k=syms[k], why=r.why, ret=r.ret, hold=xb - eb, mfe=mfe, mae=mae, first_open=first_open,
                     vol7=F["vol7"][eb, k], rng24=F["rng24"][eb, k], dd7=F["dd7"][eb, k], **after))
A = pd.DataFrame(rows)
A.to_csv(f"{X.OUT}/anatomy_2021_23.csv", index=False)
win, los = A[A.ret > 0], A[A.ret <= 0]
print(f"90% mode, trades bought 2021-23: {len(A)}, win rate {len(win)/len(A)*100:.1f}%, avg win {win.ret.mean()*100:+.2f}%, "
      f"avg loss {los.ret.mean()*100:+.2f}%  (per-trade expectancy {A.ret.mean()*100:+.2f}%)")
print("\nhours to exit (winners):", win.hold.describe(percentiles=[.25, .5, .75, .9]).round(1).to_dict())
fc = win[win.hold == 1]
print(f"winners sold at the first hourly check: {len(fc)} ({len(fc)/len(win)*100:.0f}%), avg ret {fc.ret.mean()*100:+.2f}%; "
      f"of these opened above the target (overshoot): {(fc.first_open >= 0.025).mean()*100:.0f}%")
print("\nwinners: how much higher price went in the next 24/48h after the take-profit (max high vs entry):")
print(win[["max_next24", "max_next48", "close_next24", "close_next48"]].describe(percentiles=[.1, .25, .5, .75, .9]).round(4))
print("\nlosers (time stop at 96h):")
print(los[["k", "ret", "mfe", "mae", "max_next24", "max_next96", "max_next168", "close_next96", "close_next168"]]
      .round(3).to_string())
print(f"\nlosers whose MFE before the stop reached: >=+1%: {(los.mfe >= .01).mean()*100:.0f}%  >=+0.5%: "
      f"{(los.mfe >= .005).mean()*100:.0f}%  >=+0.25%: {(los.mfe >= .0025).mean()*100:.0f}%  >=0%: {(los.mfe >= 0).mean()*100:.0f}%")
print(f"losers reaching entry+0.5% within 168h after the stop: {(los.max_next168 >= .005).mean()*100:.0f}%; "
      f"+2.5%: {(los.max_next168 >= .025).mean()*100:.0f}%")
print("\nMAE of winners (deepest dip before the take-profit):", win.mae.describe(percentiles=[.1, .5, .9, .99]).round(3).to_dict())
print("\nhold time of winners vs MFE: winners held >= 24h:", (win.hold >= 24).sum(), "; >= 48h:", (win.hold >= 48).sum())
for f_ in ("vol7", "rng24", "dd7"):
    print(f"{f_}: median at 2021-23 entries {A[f_].median():.4f}; winners {win[f_].median():.4f}, losers {los[f_].median():.4f}")
# does a bigger feature predict a bigger post-entry max? (Spearman, 2021-23 only)
A["mfe96"] = [np.nanmax(H[int(eb) + 1:min(int(eb) + 97, len(idx)), syms.index(k)]) / en - 1
              for eb, k, en in zip(d.eb, A.k, d.entry)]
print("\nSpearman rank corr. of the entry feature with the best price in the 96h after the buy, and with ret:")
for f_ in ("vol7", "rng24", "dd7"):
    print(f"  {f_}: vs mfe96 {A[f_].corr(A.mfe96, method='spearman'):+.3f}  vs loss (ret<=0) "
          f"{A[f_].corr((A.ret <= 0).astype(float), method='spearman'):+.3f}")
print("\nquantiles of mfe96 by vol7 tercile:")
A["vq"] = pd.qcut(A.vol7, 3, labels=["low", "mid", "high"])
print(A.groupby("vq", observed=True).mfe96.describe(percentiles=[.1, .25, .5, .75]).round(4))
