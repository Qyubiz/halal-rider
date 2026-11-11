"""m18: two checks nobody ran before.

1. Audit of 90% mode itself (it was the winner of ~200 early variants): stationary block bootstrap of its own
   monthly returns in the periods it was NOT picked on (2024-26, 2018-20): 90% interval of the compounded %/month,
   chance that the true rate is <= 0, and <= 3%/month.
2. Idle cash in BTC (optimistic estimate): while 90% mode is flat, no coin is within `near` of its trigger and BTC's
   last daily close is above its 200-day average, hold BTC; switch costs 0.12% each way. The decision uses only data before the hour. A dip
   that fills in an hour when the cash is in BTC is still counted as bought (in reality the bid could not rest),
   so this is an upper bound on what the idea can add.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E, m10_velocity_lib as V, m11_fusion_lib as F, m14_judge as J

def boot(m, n=20000, block=3.0, seed=0):
    rng = np.random.default_rng(seed); T = len(m); lg = np.log1p(m); out = np.empty(n)
    for b in range(n):
        i = rng.integers(T); s = 0.0
        for _ in range(T):
            s += lg[i]; i = rng.integers(T) if rng.random() < 1 / block else (i + 1) % T
        out[b] = np.expm1(s / T)
    return out

def btc_long():
    parts = [x for x in (E.load_pre().get("BTCUSDT"), E.load_h19(start="2017-01-01").get("BTCUSDT")) if x is not None]
    c = pd.concat([x["c"] for x in parts]); c = c[~c.index.duplicated(keep="last")].sort_index()
    d = c.resample("1D").last()
    up = (d > d.rolling(200).mean()).shift(1)          # yesterday's close vs its 200-day average
    return c, up

def overlay(P, idx, e, inv, near, use_trend=True):
    c, up = btc_long()
    cb = c.reindex(idx).ffill().values
    upd = up.reindex(idx.normalize()).values if use_trend else np.ones(len(idx))
    C = P["cf"].values; R = V.ref_panel(P, 24)
    import warnings
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dist = np.nanmin(C[:-1] / (R[1:] * 0.86) - 1.0, axis=1)       # known at the open of hour t
    dist = np.concatenate([[np.nan], dist])
    flat = (inv.values <= 1e-12); flat_prev = np.concatenate([[False], flat[:-1]])
    hold = flat_prev & (dist > near) & (upd == 1)          # decided at the open: nothing from hour t itself
    r = np.zeros(len(idx)); r[1:] = cb[1:] / cb[:-1] - 1.0
    ov = np.where(hold, r, 0.0)
    sw = np.abs(np.diff(np.concatenate([[0], hold.astype(int)])))
    ov = (1 + ov) * (1 - 0.0012 * sw) - 1
    return pd.Series(e.values * np.cumprod(1 + ov), idx), hold.mean()

def main():
    print("## 1. Audit of 90% mode (periods it was not picked on)\n")
    cur = {}
    for lab, hold, a, b in (("2024-26", False, J.T24, None), ("2018-20", True, J.H18, J.H21)):
        syms, idx, P = V.market_holdout(False) if hold else V.market(False)
        e, lg, inv = F.frun(P, idx, F.MODE90)
        cur[lab] = (P, idx, e, inv, a, b)
        m = J.monthly(e, a, b); bs = boot(m) * 100
        print(f"{lab}: {len(m)} months, compounded {np.expm1(np.log1p(m).mean())*100:+.2f}%/month; 90% interval "
              f"{np.percentile(bs,5):+.1f} to {np.percentile(bs,95):+.1f}; chance true rate <= 0: {(bs<=0).mean()*100:.1f}%, "
              f"<= 3%/month: {(bs<=3).mean()*100:.1f}%; months > 0: {(m>0).mean()*100:.0f}%")
    print("\n## 2. Idle cash in BTC while flat (optimistic upper bound), compounded %/month and worst drop\n")
    print("| rule | 2021-23 | 2024-26 | 2018-20 | share of hours in BTC (2024-26) |\n|---|---|---|---|---|")
    P, idx, e, inv, _, _ = cur["2024-26"]; Ph, ih, eh, invh, _, _ = cur["2018-20"]
    def cell(x, a, b):
        mult, mo, dd, _ = V.period(x, a, b); return f"{mo*100:+.2f} / {dd*100:.0f}"
    print(f"| 90% mode | {cell(e, V.T21, V.T24)} | {cell(e, V.T24, None)} | {cell(eh, J.H18, J.H21)} | 0% |")
    for near in (0.03, 0.05, 0.08):
        for tr in (True, False):
            x, sh = overlay(P, idx, e, inv, near, tr); xh, _ = overlay(Ph, ih, eh, invh, near, tr)
            print(f"| BTC when no coin within {near*100:.0f}% of its trigger{', BTC above 200-day avg' if tr else ''} | "
                  f"{cell(x, V.T21, V.T24)} | {cell(x, V.T24, None)} | {cell(xh, J.H18, J.H21)} | {sh*100:.0f}% (all years) |")

if __name__ == "__main__":
    main()
