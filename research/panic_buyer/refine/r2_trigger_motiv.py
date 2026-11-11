"""Round 2, lens "trigger", motivation only. Uses 2021-23 data only: trades of the live settings that entered AND
exited before 2024-01-01 (the engine is causal, so they do not depend on later data), and single-coin event
studies run on candles cut at 2024-01-01. Nothing from 2024 onward is printed.

A. Re-entries: how a trade does depending on how the coin's previous trade ended (take-profit / time stop) and
   how long ago (cooldown and "block after a time stop" questions).
B. Shape of the dip at entry: age of the 24h highest close, and how far the coin already was under its 72h
   high (reference-window question).
C. Single-coin bots (one coin, all-in, live rules) with the reference = highest close of the last w hours
   (w = 12/24/36/48/72) or the highest hourly high of 24h, 2021-23 only.
"""
import numpy as np
import pandas as pd

import engine as E
import study as S
import r2_trigger_lib as L

syms, idx, P = S.market(False)
e, lg = E.run(P, idx)
lg = lg.astype({"eb": int, "xb": int, "k": int})
d = lg[idx[lg.xb.values] < E.DESIGN_END].copy()
d["pnl"] = d.ret * d.amt
d["ts"] = (d.why == 1).astype(int)


def summary(g):
    return pd.Series({"n": len(g), "time-stop %": 100 * g.ts.mean(), "avg ret %": 100 * g.ret.mean(),
                      "TS avg %": 100 * g.ret[g.ts == 1].mean() if g.ts.sum() else np.nan,
                      "sum ret*amt": g.pnl.sum()})


print(f"2021-23 live trades (entered and exited before 2024): {len(d)}, time stops {d.ts.sum()}, "
      f"sum ret*amt {d.pnl.sum():+.2f}")

# ---------------------------------------------------------------- A. previous trade of the same coin
prev_why = np.full(len(d), -1); gap = np.full(len(d), np.nan); prev_ret = np.full(len(d), np.nan)
last = {}
for i, (k, eb, xb, why, r) in enumerate(zip(d.k, d.eb, d.xb, d.why, d.ret)):
    if k in last:
        pxb, pwhy, pr = last[k]
        prev_why[i], gap[i], prev_ret[i] = pwhy, eb - pxb, pr
    last[k] = (xb, why, r)
d["prev"] = np.where(prev_why == -1, "first", np.where(prev_why == 1, "after time stop", "after take-profit"))
d["gap"] = pd.cut(gap, [0, 48, 96, 168, 336, 1e9], right=False,
                  labels=["24-47h", "48-95h", "96-167h", "168-335h", ">=336h"])
pd.set_option("display.width", 160)
print("\nA. by the coin's previous trade and the hours since it was sold")
print(d.groupby(["prev", "gap"], observed=True).apply(summary, include_groups=False).round(2).to_string())
print("\n   overall by previous outcome")
print(d.groupby("prev").apply(summary, include_groups=False).round(2).to_string())
ats = d[d.prev == "after time stop"]
ats = ats.assign(prev_loss=pd.cut(prev_ret[d.prev.values == "after time stop"], [-1, -0.10, -0.05, 0, 1],
                                  labels=["prev TS <= -10%", "-10..-5%", "-5..0%", "> 0%"]))
print("\n   re-entries after a time stop, by that time stop's return")
print(ats.groupby("prev_loss", observed=True).apply(summary, include_groups=False).round(2).to_string())
for H in (48, 72, 96, 168, 336):
    w = ats[gap[d.prev.values == "after time stop"] < H]
    print(f"   re-entries within {H:3d}h of a time stop: {len(w):3d} trades, time-stop rate {100*w.ts.mean():5.1f}%, "
          f"avg {100*w.ret.mean():+.2f}%, sum ret*amt {w.pnl.sum():+.3f}")
atp = d[d.prev == "after take-profit"]
for H in (36, 48, 72, 96):
    w = atp[gap[d.prev.values == "after take-profit"] < H]
    print(f"   re-entries within {H:3d}h of a take-profit: {len(w):3d} trades, time-stop rate {100*w.ts.mean():5.1f}%, "
          f"avg {100*w.ret.mean():+.2f}%, sum ret*amt {w.pnl.sum():+.3f}")

# ---------------------------------------------------------------- B. dip shape at entry
C = P["c"]
age = np.full(len(d), np.nan); pre72 = np.full(len(d), np.nan); hc = np.full(len(d), np.nan)
cv = C.values; hv = P["h"].values
for i, (k, eb) in enumerate(zip(d.k, d.eb)):
    w = cv[max(0, eb - 24):eb, k]
    if np.isfinite(w).any():
        age[i] = 24 - np.nanargmax(w)                       # 1 = the max close was the last closed hour
    w72 = cv[max(0, eb - 72):eb, k]
    pre72[i] = np.nanmax(w) / np.nanmax(w72) - 1.0          # <= 0: 24h high already under the 72h high
    hc[i] = np.nanmax(hv[max(0, eb - 24):eb, k]) / np.nanmax(w) - 1.0
d["age"] = pd.cut(age, [0, 6, 12, 18, 25], labels=["1-6h", "7-12h", "13-18h", "19-24h"])
d["pre72"] = pd.cut(pre72, [-1, -0.10, -0.05, -1e-12, 1], labels=["<=-10%", "-10..-5%", "-5..0%", "0 (24h high is 72h high)"])
d["hc"] = pd.cut(hc, [-1e-9, 0.01, 0.02, 0.04, 10], labels=["0-1%", "1-2%", "2-4%", ">4%"])
print("\nB1. hours since the 24h highest close at entry")
print(d.groupby("age", observed=True).apply(summary, include_groups=False).round(2).to_string())
print("\nB2. 24h highest close vs 72h highest close at entry")
print(d.groupby("pre72", observed=True).apply(summary, include_groups=False).round(2).to_string())
print("\nB3. 24h highest high vs 24h highest close at entry (how much a high-based reference would lift the trigger)")
print(d.groupby("hc", observed=True).apply(summary, include_groups=False).round(2).to_string())

# ---------------------------------------------------------------- C. single-coin bots, 2021-23 candles only
cut = idx < E.DESIGN_END
Pd = {f: P[f][cut] for f in ("o", "h", "l", "c", "cf")}
refs = {f"close {w}h": (lambda w: (lambda Q: L.ref_close(Q, w)))(w) for w in (12, 24, 36, 48, 72)}
refs["high 24h"] = lambda Q: L.ref_high(Q, 24)
print("\nC. single-coin bots (live rules, one coin all-in), 2021-23: totals over the 19 coins")
rows = []
for name, f in refs.items():
    R = f(Pd)
    tr = []; mults = []
    for j, s in enumerate(syms):
        Q = {g: Pd[g][[s]] for g in Pd}
        if Q["c"][s].notna().sum() < 500:
            continue
        ee, ll = L.trun(Q, Q["c"].index, N=1, fmax=1.0, ref=R[[s]].values)
        tr.append(ll); mults.append(float(ee.iloc[-1]))
    t = pd.concat(tr)
    rows.append({"reference": name, "trades": len(t), "time-stop %": 100 * (t.why == 1).mean(),
                 "avg ret %": 100 * t.ret.mean(), "TS avg %": 100 * t.ret[t.why == 1].mean(),
                 "median coin multiple": np.median(mults), "coins up": int(np.sum(np.array(mults) > 1)),
                 "mean log multiple": np.mean(np.log(mults))})
print(pd.DataFrame(rows).round(3).to_string(index=False))
