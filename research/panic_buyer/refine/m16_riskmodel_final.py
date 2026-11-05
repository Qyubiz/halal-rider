"""m16 riskmodel, step 4: the ONE look. The grid (m16_riskmodel_grid.py) gave NO pick (0 of 60 designs pass bars
1-3), so this reports, labelled "fallback, not a pick", the highest 2021-23 score (big_base_q0.9_f0.5, which passes
bars 1-2 but not the neighbour bar) on every period, next to 90% mode, plus the four models' out-of-sample skill per
year on 90% mode's own buys. Curves saved as fallback_eq_main.csv / fallback_eq_hold.csv (time,equity)."""
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

FB = dict(target="big", fs="base", q=0.9, floor=0.5)
NB = [dict(FB, q=0.8), dict(FB, floor=0.375)]
syms_b, idx_b, P_b, Z = R.build()
risks = {(t, f): R.train(t, f, Z, idx_b) for t in ("mae", "big") for f in ("base", "fut")}


def sz_for(d, syms, idx):
    if d is None:
        return None
    return R.to_panel(R.size_big(risks[(d["target"], d["fs"])], Z, idx_b, d["q"], d["floor"], (d["target"], d["fs"])),
                      idx_b, syms_b, idx, syms)


def runs(d):
    out = {}
    for hold in (False, True):
        for name, kw in (("normal", {}), ("worse", dict(pen=R.PEN, pens=R.PEN)),
                         ("stress", dict(pen=R.PEN, pens=R.PEN, mtp_now=True, cd_add=1, liquidate=True))):
            syms, idx, P = V.market_holdout(name == "stress") if hold else V.market(name == "stress")
            e, lg, inv = R.srun(P, idx, sz_for(d, syms, idx), **kw)
            r = V.summarize(e, lg, inv, idx, V.HOLD if hold else V.MAIN)
            tb = idx[lg.eb.astype(int).values]
            acct = lg.ret * lg.amt / lg.eq0
            for tag, a, b in (V.HOLD if hold else V.MAIN):
                s = (tb >= a) & ((tb < b) if b is not None else True)
                r[f"{tag}_avgl"] = float(acct[s & (lg.ret.values < 0)].mean())
            out[(hold, name)] = (r, e, lg, idx, syms)
    return out


def items(o):
    g = lambda h, n, k: o[(h, n)][0][k]  # noqa: E731
    i1 = g(False, "normal", "t_mo") > 0.0689 and g(False, "normal", "t_wr") >= 0.925 and g(False, "normal", "t_dd") >= -0.35
    i2 = g(True, "normal", "h_mo") > 0.1082 and g(True, "normal", "h_wr") >= 0.925 and g(True, "normal", "h_dd") >= -0.663
    i3 = g(False, "worse", "t_mo") > 0.0531 and g(True, "worse", "h_mo") > 0.0741 and g(False, "worse", "t_wr") >= 0.92 \
        and g(True, "worse", "h_wr") >= 0.92
    i4 = g(False, "stress", "t_mo") > 0.0549 and o[(False, "stress")][1].min() > 0.05
    return [i1, i2, i3, i4]


base, fb = runs(None), runs(FB)
LAB = (("2021-23", False, "d"), ("2024-26", False, "t"), ("since 2025", False, "s"), ("2018-20", True, "h"))
print("rows: win rate / %/month / worst drop / trades per month / avg losing trade / worst trade (% of account)")
for fills in ("normal", "worse", "stress"):
    for lab, hold, p in LAB:
        if fills == "stress" and p in ("s", "h"):
            continue
        line = f"{fills:6s} {lab:10s}"
        for nm, o in (("90%", base), ("fallback", fb)):
            r = o[(hold, fills)][0]
            line += (f" | {nm} {r[p+'_wr']*100:.1f} / {r[p+'_mo']*100:+.2f} / {r[p+'_dd']*100:.1f} / {r[p+'_tpm']:.1f}"
                     f" / {r[p+'_avgl']*100:+.1f} / {r[p+'_wt']*100:+.1f}")
        print(line)
print("fallback items 1-4:", items(fb), " stressed 2021-23 min equity", f"{fb[(False, 'stress')][1].min():.3f}")
nbp = [all(items(runs(d))[:3]) for d in NB]
print("fallback grid neighbours passing items 1-3:", sum(nbp), "of", len(nbp))
for hold, fn in ((False, "fallback_eq_main.csv"), (True, "fallback_eq_hold.csv")):
    e = fb[(hold, "normal")][1]
    pd.DataFrame({"time": e.index, "equity": e.values}).to_csv(os.path.join(R.OUT, fn), index=False)

# skill on 90% mode's own buys (normal fills; 2018-20 holdout run + 2021-26 run)
rows = []
for hold in (True, False):
    r, e, lg, idx, syms = base[(hold, "normal")]
    tb = idx[lg.eb.astype(int).values]
    bi = idx_b.get_indexer(tb); bk = np.array([syms_b.index(syms[int(k)]) for k in lg.k])
    for t, b, s, rt in zip(tb, bi, bk, lg.ret.values):
        if (hold and t >= pd.Timestamp("2021-01-01")) or (not hold and t < pd.Timestamp("2021-01-01")):
            continue
        rows.append(dict(year=t.year, mae=Z["mae"][b, s], big=float(rt < -0.10),
                         **{f"{a}_{c}": risks[(a, c)][b, s] for a, c in risks}))
S = pd.DataFrame(rows)
print("\nOOS skill on 90% mode's buys: spearman(risk, worst 96h close under entry) / AUC(loss > 10%)  [n buys, n big losses]")
for (a, c) in risks:
    out = []
    for Y, g in S.groupby("year"):
        ok = g[f"{a}_{c}"].notna()
        rho = spearmanr(g[f"{a}_{c}"][ok], -g.mae[ok])[0] if ok.sum() > 10 else np.nan
        out.append(f"{Y}: {rho:+.2f}/{M.auc(g.big.values, g[f'{a}_{c}'].values):.2f}")
    print(f"{a}_{c:5s} " + "  ".join(out))
print("counts " + "  ".join(f"{Y}: {len(g)}/{int(g.big.sum())}" for Y, g in S.groupby("year")))
