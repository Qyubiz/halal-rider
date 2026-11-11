"""m20: smaller losing months at the same profit. 90% mode's trades are untouched; a buy made while a risk flag is on
uses a fraction f of the money. Rule, flag family, selection and pass rule: M20_PREREG.md (written before any run).

  python3 m20_flags.py select    2021-23 only: every design in 5 nearby versions, normal and 0.2%-worse fills,
                                 against 90% mode run the same way -> $PB_OUT/m20_select.csv, m20_pick.json
  python3 m20_flags.py final     the ONE look at 2024-26, 2018-20 and the stressed runs for the pick and its
                                 neighbours, the permutation luck check, and the losing months of 2024-26

Engine: m16_riskmodel_lib.srun (m12 `ek` with a size table; equal to 90% mode at size 1). Flags use closed hourly
candles before the bid hour only (every table is shifted one hour); they are computed on the long Dec 2017 - Oct 2026
panel so that 7-day / 30-day / 365-day windows have history at the start of each test panel.
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m10_velocity_lib as V  # noqa: E402
import m11_fusion_lib as F  # noqa: E402
import m12_entry_lib as L  # noqa: E402
import m14_judge as J  # noqa: E402
import m16_riskmodel_lib as R  # noqa: E402

OUT = R.OUT
PEN = R.PEN
T21, T24, T25 = V.T21, V.T24, V.T25
H18, H21 = J.H18, J.H21
FLOORS = (0.5, 0.33)
GROUPS = {                                    # group -> thresholds (ordered, neighbours are adjacent entries)
    "A": (0.15, 0.20, 0.25, 0.30),            # coin's last close more than x under its 168h highest close
    "B": (0.30, 0.40, 0.50),                  # ... under its 720h highest close
    "C": (0.80, 0.90),                        # market volatility above its q-quantile of the previous 365 days
    "D": (3, 5, 8),                           # k or more coins at least 10% under their 24h highest close
    "E24": (0.05, 0.08),                      # BTC down x or more over 24h
    "E72": (0.10, 0.15),                      # BTC down x or more over 72h
    "F": (0.07, 0.09),                        # coin's 7-day volatility at or under x
    "G": (0.02, 0.05),                        # last close already x or more under the bid price
    "Hbig": (0.80, 0.90),                     # reference: m16 model, P(loss > 10%), above its trailing q-quantile
    "Hmae": (0.80, 0.90),                     # reference: m16 model, predicted worst close
}
SIMPLE = [g for g in GROUPS if not g.startswith("H")]

_big = {}


def big():
    """Feature tables on the long panel (index idx_b, columns syms_b); row t uses closes up to t-1."""
    if _big:
        return _big
    syms_b, idx_b, P_b, Z = R.build()
    C = P_b["cf"]
    assert list(C.columns) == list(syms_b) and C.index.equals(idx_b)
    c1 = C.shift(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _big.update(
            syms=syms_b, idx=idx_b, Z=Z, c1=c1,
            dd7=1 - c1 / C.rolling(168, min_periods=24).max().shift(1),
            dd30=1 - c1 / C.rolling(720, min_periods=168).max().shift(1),
            vol168=np.log(C).diff().rolling(168, min_periods=84).std().shift(1),
            dip24=(c1 <= 0.9 * C.rolling(24, min_periods=1).max().shift(1)),
            b24=C["BTCUSDT"].shift(1) / C["BTCUSDT"].shift(25) - 1,
            b72=C["BTCUSDT"].shift(1) / C["BTCUSDT"].shift(73) - 1,
            cvol7=F.vol7_long(),
        )
    return _big


_ft = {}


def flag(P, idx, group, x):
    """Boolean (n, K) table for one flag on panel P."""
    key = (id(P["o"]), group, x)
    if key in _ft:
        return _ft[key]
    B = big()
    cols = list(P["o"].columns)
    n, K = P["o"].shape

    def pan(df):
        return df.reindex(index=idx, columns=cols).values

    def ser(s):
        return np.repeat(s.reindex(idx).values[:, None], K, axis=1)

    with np.errstate(invalid="ignore"):
        if group == "A":
            f = pan(B["dd7"]) > x
        elif group == "B":
            f = pan(B["dd30"]) > x
        elif group == "C":
            mv = B["vol168"][cols].median(axis=1)
            thr = mv.rolling(8760, min_periods=4320).quantile(x)
            f = ser((mv > thr).astype(float)) > 0
        elif group == "D":
            f = ser(B["dip24"][cols].sum(axis=1)) >= x
        elif group == "E24":
            f = ser(B["b24"]) <= -x
        elif group == "E72":
            f = ser(B["b72"]) <= -x
        elif group == "F":
            f = pan(B["cvol7"]) <= x
        elif group == "G":
            lim = L.lim_table(P, L.BASE)
            f = P["cf"].shift(1).values <= lim * (1 - x)
        elif group == "ALL":
            f = np.ones((n, K), bool)
        elif group in ("Hbig", "Hmae"):
            tgt = group[1:]
            risk = R.train(tgt, "base", B["Z"], B["idx"])
            sz = R.to_panel(R.size_big(risk, B["Z"], B["idx"], x, 0.5, (tgt, "base")), B["idx"], B["syms"], idx, cols)
            f = sz < 1
        else:
            raise ValueError(group)
    _ft[key] = np.asarray(f, bool)
    return _ft[key]


def size_table(P, idx, d):
    f = np.zeros(P["o"].shape, bool)
    for g, x in d["flags"]:
        f |= flag(P, idx, g, x)
    return np.where(f, d["f"], 1.0), f


def name(d):
    lab = {"A": "coin >{:.0f}% under 7-day high", "B": "coin >{:.0f}% under 30-day high",
           "C": "market volatility above its {:.0f}% level", "D": "{:.0f}+ coins in a 10% dip",
           "E24": "BTC -{:.0f}% in 24h", "E72": "BTC -{:.0f}% in 72h", "F": "coin 7-day volatility <= {:.0f}%",
           "G": "close {:.0f}%+ under the bid", "Hbig": "model P(big loss) top {:.0f}%",
           "Hmae": "model worst-close top {:.0f}%", "ALL": "every buy"}
    parts = []
    for g, x in d["flags"]:
        v = x if g == "D" else (100 * (1 - x) if g.startswith("H") else 100 * x)
        parts.append(lab[g].format(v))
    return " OR ".join(parts) + f" -> bet {d['f']*100:.0f}%"


def stats(e, a, b, a_mo=None):
    mult, mo, dd, _ = V.period(e, a_mo or a, b)
    m = J.monthly(e, a, b)
    return dict(mo=mo, dd=dd, worst=float(m.min()), mloss=float(-m[m < 0].sum()), nlos=int((m < 0).sum()), nm=len(m))


# ------------------------------------------------------------------ selection (2021-23 ONLY)
_base_sel = {}


def sel_runs(P, idx, SZ):
    out = {}
    for fill, pk in (("n", {}), ("w", dict(pen=PEN, pens=PEN))):
        for v, kw in R.VERS.items():
            e, lg, inv = R.srun(P, idx, SZ, **dict(kw, **pk))
            out[(fill, v)] = stats(e, T21, T24, L.LATE if v == "ls" else None)
    return out


def sel_score(P, idx, d):
    if not _base_sel:
        _base_sel.update(sel_runs(P, idx, None))
    SZ, f = size_table(P, idx, d)
    r = sel_runs(P, idx, SZ)
    vs = list(R.VERS)
    g = lambda fill, k: float(np.mean([r[(fill, v)][k] - _base_sel[(fill, v)][k] for v in vs]))   # noqa: E731
    cut = float(np.mean([1 - r[("n", v)]["mloss"] / _base_sel[("n", v)]["mloss"] for v in vs]))
    return dict(dmo_n=g("n", "mo"), dmo_w=g("w", "mo"), cut=cut, dworst=g("n", "worst"), ddd=g("n", "dd"),
                mo=r[("n", "n")]["mo"], worst=r[("n", "n")]["worst"], dd=r[("n", "n")]["dd"],
                mloss=r[("n", "n")]["mloss"], nlos=r[("n", "n")]["nlos"])


def keeps(s):
    return s["dmo_n"] >= -0.0025 and s["dmo_w"] >= -0.0025


def qualifies(s):
    return keeps(s) and s["dworst"] >= -1e-9 and s["ddd"] >= -1e-9


def neighbours(d):
    out = []
    other = [x for x in FLOORS if x != d["f"]][0]
    out.append(dict(flags=d["flags"], f=other))
    for i, (g, x) in enumerate(d["flags"]):
        th = GROUPS[g]
        j = th.index(x)
        for jj in (j - 1, j + 1):
            if 0 <= jj < len(th):
                fl = list(d["flags"])
                fl[i] = (g, th[jj])
                out.append(dict(flags=fl, f=d["f"]))
    return out


def key(d):
    return json.dumps([[g, x] for g, x in d["flags"]]) + f"|{d['f']}"


def select():
    syms, idx, P = V.market(False)
    S = {}

    def score(d):
        k = key(d)
        if k not in S:
            S[k] = dict(sel_score(P, idx, d), d=d)
        return S[k]

    singles = [dict(flags=[(g, x)], f=f) for g in GROUPS for x in GROUPS[g] for f in FLOORS]
    for d in singles:
        score(d)
    b = _base_sel[("n", "n")]
    print(f"90% mode 2021-23: {b['mo']*100:+.2f}%/month, worst month {b['worst']*100:+.1f}%, month loss "
          f"{b['mloss']*100:.0f} pts in {b['nlos']} losing months, worst drop {b['dd']*100:.1f}%\n")

    def nb_ok(d):
        nb = [score(x) for x in neighbours(d)]
        return np.mean([keeps(x) for x in nb]) > 0.5

    def table(ds, title):
        print(f"\n{title}\n| design | profit change normal / worse fills (pts/month) | month loss cut | worst month | "
              f"worst drop | qualifies | neighbours |\n|---|---|---|---|---|---|---|")
        for d in sorted(ds, key=lambda x: -score(x)["cut"]):
            s = score(d)
            print(f"| {name(d)} | {s['dmo_n']*100:+.2f} / {s['dmo_w']*100:+.2f} | {s['cut']*100:.0f}% | "
                  f"{s['worst']*100:+.1f}% | {s['dd']*100:.1f}% | {'yes' if qualifies(s) else 'no'} | "
                  f"{'ok' if nb_ok(d) else 'fail'} |", flush=True)

    table(singles, "Single flags (2021-23, mean over 5 versions)")
    okS = [d for d in singles if d["flags"][0][0] in SIMPLE and qualifies(score(d)) and nb_ok(d)]
    best = {}
    for d in sorted(okS, key=lambda x: -score(x)["cut"]):
        best.setdefault(d["flags"][0][0], d)
    top = list(best.values())[:2]
    pairs = []
    if len(top) == 2:
        pairs = [dict(flags=[top[0]["flags"][0], top[1]["flags"][0]], f=f) for f in FLOORS]
        for d in pairs:
            score(d)
        table(pairs, "Pairs (flag X OR flag Y)")
    cand = [d for d in okS + pairs if qualifies(score(d)) and nb_ok(d)]
    if not cand:
        cand = [d for d in singles if d["flags"][0][0] not in SIMPLE and qualifies(score(d)) and nb_ok(d)]
    pick = max(cand, key=lambda x: score(x)["cut"]) if cand else None
    rows = [dict(design=name(v["d"]), key=k, **{a: v[a] for a in v if a != "d"}) for k, v in S.items()]
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "m20_select.csv"), index=False)
    json.dump(pick, open(os.path.join(OUT, "m20_pick.json"), "w"))
    print(f"\n{len(S)} designs scored. PICK: {name(pick) if pick else 'none'}")
    if pick:
        print(json.dumps(pick))


# ------------------------------------------------------------------ the one look
def look(d, label, base=None):
    """All test-period numbers for one design. Returns dict of stats per (run, period)."""
    out = {}
    for hold, dead, tag, kw, pers in (
            (False, False, "n", {}, (("2021-23", T21, T24), ("2024-26", T24, None), ("since 2025", T25, None))),
            (True, False, "n", {}, (("2018-20", H18, H21),)),
            (False, False, "w", dict(pen=PEN, pens=PEN), (("2024-26", T24, None),)),
            (True, False, "w", dict(pen=PEN, pens=PEN), (("2018-20", H18, H21),)),
            (False, True, "s", dict(pen=PEN, pens=PEN, mtp_now=True, cd_add=1, liquidate=True),
             (("2021-23", T21, T24), ("2024-26", T24, None)))):
        syms, idx, P = V.market_holdout(dead) if hold else V.market(dead)
        SZ = None if d is None else size_table(P, idx, d)[0]
        e, lg, inv = R.srun(P, idx, SZ, **kw)
        r = V.summarize(e, lg, inv, idx, V.HOLD if hold else V.MAIN)
        for pl, a, b in pers:
            s = stats(e, a, b)
            t = {"2021-23": "d", "2024-26": "t", "since 2025": "s", "2018-20": "h"}[pl]
            s.update(wr=r[f"{t}_wr"], wt=r[f"{t}_wt"])
            out[(tag, pl)] = s
    return out


def items(p, b):
    i1 = all(p[k]["mo"] >= b[k]["mo"] - 0.0025 for k in (("n", "2024-26"), ("n", "2018-20"), ("w", "2024-26"),
                                                         ("w", "2018-20")))
    i2 = all(p[k]["mloss"] <= 0.85 * b[k]["mloss"] and p[k]["worst"] >= b[k]["worst"] - 1e-9
             and p[k]["dd"] >= b[k]["dd"] - 1e-9 for k in (("n", "2024-26"), ("n", "2018-20")))
    i3 = all(p[k]["mo"] >= b[k]["mo"] - 0.005 and p[k]["dd"] >= b[k]["dd"] - 1e-9
             for k in (("s", "2021-23"), ("s", "2024-26")))
    return i1, i2, i3


def perm(d, n=10000, seed=0):
    """Mean return of the flagged trades of 90% mode vs random flag sets of the same size, per test period."""
    rng = np.random.default_rng(seed)
    out = {}
    for hold, pl, a, b in ((False, "2024-26", T24, None), (True, "2018-20", H18, H21), (False, "2021-23", T21, T24)):
        syms, idx, P = V.market_holdout(False) if hold else V.market(False)
        e, lg, inv = R.srun(P, idx, None)
        fl = size_table(P, idx, d)[1]
        tb = idx[lg.eb.astype(int).values]
        sel = np.asarray((tb >= a) & ((tb < b) if b is not None else True))
        ret = lg.ret.values[sel]
        fg = fl[lg.eb.astype(int).values[sel], lg.k.astype(int).values[sel]]
        k = int(fg.sum())
        if k == 0 or k == len(ret):
            out[pl] = dict(n=len(ret), k=k, p=np.nan, mf=np.nan, mu=np.nan, lossf=np.nan)
            continue
        obs = ret[fg].mean()
        sims = np.array([ret[rng.choice(len(ret), k, replace=False)].mean() for _ in range(n)])
        out[pl] = dict(n=len(ret), k=k, p=float((np.sum(sims <= obs) + 1) / (n + 1)), mf=float(obs),
                       mu=float(ret[~fg].mean()), lossf=float(ret[fg][ret[fg] < 0].sum() / ret[ret < 0].sum()),
                       wrf=float((ret[fg] > 0).mean()), wru=float((ret[~fg] > 0).mean()))
    return out


def final():
    d = json.load(open(os.path.join(OUT, "m20_pick.json")))
    if d is None:
        print("no pick")
        return
    d["flags"] = [tuple(x) for x in d["flags"]]
    print(f"PICK: {name(d)}\n")
    b = look(None, "90% mode")
    p = look(d, "pick")
    print("| run | 90% mode: %/month / wins / worst month / losing months / month loss / worst drop / worst trade | "
          "with the tool |\n|---|---|---|")
    for k in b:
        f = lambda s: (f"{s['mo']*100:+.2f} / {s['wr']*100:.1f} / {s['worst']*100:+.1f} / {s['nlos']} of {s['nm']} / "   # noqa: E731
                       f"{s['mloss']*100:.0f} / {s['dd']*100:.1f} / {s['wt']*100:.1f}")
        print(f"| {dict(n='normal', w='worse fills', s='stressed')[k[0]]} {k[1]} | {f(b[k])} | {f(p[k])} |")
    i1, i2, i3 = items(p, b)
    print(f"\nitem 1 profit kept: {'PASS' if i1 else 'FAIL'}; item 2 losses cut: {'PASS' if i2 else 'FAIL'}; "
          f"item 3 stressed: {'PASS' if i3 else 'FAIL'}")
    nb = neighbours(d)
    ok = []
    print("\nNeighbours (2024-26 and 2018-20, normal fills: profit change pts/month, month-loss cut):")
    for x in nb:
        q = look(x, "nb")
        a1, a2, _ = items(q, b)
        ok.append(a1 and a2)
        print(f"  {name(x)}: 2024-26 {(q[('n','2024-26')]['mo']-b[('n','2024-26')]['mo'])*100:+.2f}, "
              f"{(1-q[('n','2024-26')]['mloss']/b[('n','2024-26')]['mloss'])*100:.0f}%; 2018-20 "
              f"{(q[('n','2018-20')]['mo']-b[('n','2018-20')]['mo'])*100:+.2f}, "
              f"{(1-q[('n','2018-20')]['mloss']/b[('n','2018-20')]['mloss'])*100:.0f}% -> items 1,2 "
              f"{'pass' if ok[-1] else 'fail'}")
    print(f"item 4 neighbours: {sum(ok)} of {len(ok)} pass -> {'PASS' if np.mean(ok) > 0.5 else 'FAIL'}")
    pm = perm(d)
    print("\nLuck check (flagged trades of 90% mode vs random flags of the same size):")
    for pl, r in pm.items():
        print(f"  {pl}: {r['k']} of {r['n']} trades flagged; mean return flagged {r['mf']*100:+.2f}% vs others "
              f"{r['mu']*100:+.2f}%; win rate flagged {r.get('wrf', np.nan)*100:.0f}% vs {r.get('wru', np.nan)*100:.0f}%; "
              f"share of all losses in flagged trades {r['lossf']*100:.0f}%; p = {r['p']:.4f}")
    print(f"item 5 luck check: {'PASS' if pm['2024-26']['p'] < 0.05 and pm['2018-20']['p'] < 0.05 else 'not proven'}")
    # the losing months of 2024-26, trade by trade
    syms, idx, P = V.market(False)
    e0, lg, _ = R.srun(P, idx, None)
    SZ, fl = size_table(P, idx, d)
    e1, _, _ = R.srun(P, idx, SZ)
    m0 = e0.resample("1ME").last().pct_change()
    m1 = e1.resample("1ME").last().pct_change()
    print("\nLosing months of 90% mode since 2024 (month: 90% mode -> with the tool; its losing trades):")
    for t in m0.index[(m0.index >= T24) & (m0 < 0)]:
        a, z = t - pd.offsets.MonthBegin(1), t + pd.Timedelta(days=1)
        tr = lg[(idx[lg.xb.astype(int).values] >= a) & (idx[lg.xb.astype(int).values] < z) & (lg.ret < 0)]
        ts = "; ".join(f"{syms[int(r.k)].replace('USDT','')} bought {idx[int(r.eb)]:%d %b} {r.ret*100:+.1f}%"
                       f"{' (flagged)' if fl[int(r.eb), int(r.k)] else ''}" for r in tr.itertuples())
        print(f"  {t:%b %Y}: {m0[t]*100:+.1f}% -> {m1[t]*100:+.1f}%; {ts}")


def menu():
    """HINDSIGHT description (all periods shown side by side, no selection): what each kind of bet cut costs in
    profit and saves in losing months, next to a plain smaller bet on every buy."""
    D = [dict(flags=[("ALL", 0)], f=0.5),
         dict(flags=[("A", 0.15)], f=0.5), dict(flags=[("A", 0.15)], f=0.33), dict(flags=[("A", 0.20)], f=0.5),
         dict(flags=[("B", 0.30)], f=0.33), dict(flags=[("B", 0.40)], f=0.33),
         dict(flags=[("Hbig", 0.90)], f=0.5), dict(flags=[("Hbig", 0.80)], f=0.5), dict(flags=[("Hbig", 0.80)], f=0.33),
         dict(flags=[("G", 0.02)], f=0.5)]
    b = look(None, "base")
    ks = (("n", "2021-23"), ("n", "2024-26"), ("n", "2018-20"))
    print("Per period: profit kept (% of 90% mode's %/month) / month loss cut / worst month / worst drop\n")
    print("| tool | " + " | ".join(k[1] for k in ks) + " | worse fills 24-26, 18-20: profit kept | stressed 21-23, 24-26: "
          "profit kept | luck p 21-23 / 24-26 / 18-20 |\n|---|---|---|---|---|---|---|")
    print("| 90% mode | " + " | ".join(f"{b[k]['mo']*100:+.2f}%/mo / {b[k]['mloss']*100:.0f} pts / {b[k]['worst']*100:+.1f}% / "
                                        f"{b[k]['dd']*100:.0f}%" for k in ks) + " | | | |")
    for d in D:
        p = look(d, "x")
        pm = perm(d, n=4000) if d["flags"][0][0] != "ALL" else None
        cells = [f"{p[k]['mo']/b[k]['mo']*100:.0f}% / {(1-p[k]['mloss']/b[k]['mloss'])*100:.0f}% / {p[k]['worst']*100:+.1f}% / "
                 f"{p[k]['dd']*100:.0f}%" for k in ks]
        w = " , ".join(f"{p[k]['mo']/b[k]['mo']*100:.0f}%" for k in (("w", "2024-26"), ("w", "2018-20")))
        st = " , ".join(f"{p[k]['mo']/b[k]['mo']*100:.0f}%" for k in (("s", "2021-23"), ("s", "2024-26")))
        lp = " / ".join(f"{pm[x]['p']:.3f}" for x in ("2021-23", "2024-26", "2018-20")) if pm else ""
        print(f"| {name(d)} | " + " | ".join(cells) + f" | {w} | {st} | {lp} |", flush=True)


if __name__ == "__main__":
    {"select": select, "final": final, "menu": menu}[sys.argv[1]]()
