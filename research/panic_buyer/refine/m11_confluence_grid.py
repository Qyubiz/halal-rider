"""m11 confluence: the grid, scored on 2021-23 ONLY. Rule written before any grid result was seen.

VOTES (fixed before the grid; directions from m11_confluence_anatomy.py, 2021-23 90% mode trades only; the
thresholds are the round numbers set in m11_confluence_lib.py and are not searched):
   vol    coin 7-day daily volatility > 8%                  (95.0% vs 90.6% win in 2021-23)
   red1   the last hourly close was down                     (95.8% vs 88.0%)
   rng    24h high-low range >= 20% of the last close        (94.8% vs 92.8%)
   brd    < 25% of the core coins are 10%+ under their 24h high (94.4% vs 91.0%)
   -fast  the 24h high close is MORE than 12 hours old      (94.3% vs 92.7%)
   -deep  the open is NOT already 2+ points under the 14% trigger (94.1% vs 90.9%)
   (btc dropped: its 2021-23 split was the wrong way round and small.)  score = number of these 6 that are on (0..6).

ONE combined bot (1 position, all the money, as 90% mode), switches searched:
   kA, dlo  the 24h -14% bid needs score >= kA (0, 2, 3, 4); below kA the coin is skipped (dlo = skip) or bid deeper
            at 24h high x (1 - 0.20) (dlo = 0.20)
   kT       take-profit for score >= kT: 3% x sqrt(rng24 / 0.2074) clipped 1-6% (exits agent); below: +2.5% (off, 5, 4)
   kB       second, faster trigger 12h highest close -13% (+3%, 48h, velocity agent) only when score >= kB (off, 5, 4)
   rak      re-arm a coin right after a winning exit when its score >= rak (off, 4, 0; 0 = always, velocity agent)
   X        stage 2, on the 5 best stage-1 designs only: extra coins = monthly top 20 screened coins by 30-day volume,
            bid at 24% under the 24h high only when score >= kX (5, 4)  (universe agent)
   4 x 3 x 3 x 3 (kA/dlo 7 combos) = 189 stage-1 designs + 10 stage-2 designs <= 300.

SELECTION RULE (2021-23 only):
 1. Bar: 2021-23 win rate >= 93.8% and worst drop >= -64.5% (no deeper than -64% after rounding), with normal fills
    AND with every fill 0.2% worse. If NO design passes the worse-fills part literally, the worse-fills bar becomes
    90% mode's own worse-fills result (win rate >= 93.4%, drop >= -65.9%), i.e. like for like.
 2. Robust %/month = mean 2021-23 %/month over 5 versions: as is, cooldown +1h, cooldown -1h, the bot's take-profit
    timing (mtp_now), start 2021-01-15.
 3. Neighbours = every design one step away in ONE switch (kA in 0,2,3,4 keeping dlo; dlo skip <-> 0.20; kT, kB, rak
    one step along their lists; X: the other kX and X off). The MEDIAN of the neighbours' win rate and drop (normal
    and worse fills) must also pass the bar of step 1.
 4. Pick = the design passing 1 and 3 with the highest robust %/month. Ties: fewer switches on.
 Then ONE look (m11_confluence_final.py) at 2024-26, since 2025, the 2018-20 holdout, worse fills and the stressed run.
"""
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m11_confluence_lib as C  # noqa: E402

VOTES = ["vol", "red1", "rng", "brd", "-fast", "-deep"]
KA = [0, 2, 3, 4]
KT = [None, 5, 4]
KB = [None, 5, 4]
RAK = [None, 4, 0]
END = "2024-01-06"          # every trade bought before 2024 closes by then (96h time stop)
PERIOD = (("d", C.T21, C.T24),)
BAR_WR, BAR_DD = 0.938, -0.645


def design(kA, dlo, kT, kB, rak, kX=None):
    d = dict(votes=VOTES, kA=kA, dlo=dlo)
    if kT is not None:
        d["kT"] = kT
    if kB is not None:
        d["B"] = dict(w=12, dep=0.13, tp=0.03, hm=48, k=kB)
    if rak is not None:
        d["rak"] = rak
    if kX is not None:
        d["X"] = dict(top=20, dep=0.24, tp=0.025, hm=96, k=kX)
    return d


def key(kA, dlo, kT, kB, rak, kX=None):
    return (kA, dlo, kT, kB, rak, kX)


def name(k):
    kA, dlo, kT, kB, rak, kX = k
    s = [f"kA{kA}" + ("" if kA == 0 else ("skip" if dlo is None else f"@{dlo:.2f}"))]
    s.append("tp2.5" if kT is None else f"tpScaled>={kT}")
    s.append("noB" if kB is None else f"B12h>={kB}")
    s.append("norearm" if rak is None else f"rearm>={rak}")
    if kX is not None:
        s.append(f"X20@24%>={kX}")
    return " ".join(s)


def evaluate(M, k):
    d = design(*k)
    out = {}
    for tag, kw in (("n", {}), ("w", dict(pen=0.002, pens=0.002)), ("cp", dict(cd_add=1)), ("cm", dict(cd_add=-1)),
                    ("mt", dict(mtp_now=True)), ("st", dict(start="2021-01-15"))):
        e, lg = C.run(M, d, end=END, **kw)
        r = C.summ(e, lg, PERIOD)
        out[f"{tag}_wr"], out[f"{tag}_mo"], out[f"{tag}_dd"] = r["d_wr"], r["d_mo"], r["d_dd"]
        if tag == "n":
            out["n_tpm"], out["n_wt"] = r["d_tpm"], r["d_wt"]
    out["rob_mo"] = np.mean([out[f"{t}_mo"] for t in ("n", "cp", "cm", "mt", "st")])
    return out


def neighbours(k, have):
    kA, dlo, kT, kB, rak, kX = k
    nb = []
    i = KA.index(kA)
    for j in (i - 1, i + 1):
        if 0 <= j < len(KA):
            nb.append((KA[j], None if KA[j] == 0 else (dlo if kA != 0 else None), kT, kB, rak, kX))
            if kA == 0 and KA[j] != 0:
                nb.append((KA[j], 0.20, kT, kB, rak, kX))
    if kA != 0:
        nb.append((kA, 0.20 if dlo is None else None, kT, kB, rak, kX))
    for pos, L in ((2, KT), (3, KB), (4, RAK)):
        i = L.index(k[pos])
        for j in (i - 1, i + 1):
            if 0 <= j < len(L):
                x = list(k); x[pos] = L[j]; nb.append(tuple(x))
    if kX is not None:
        for v in (None, 5 if kX == 4 else 4):
            x = list(k); x[5] = v; nb.append(tuple(x))
    return [x for x in dict.fromkeys(nb) if x in have and x != k]


def passes(r, wbar):
    return (r["n_wr"] >= BAR_WR - 1e-9 and r["n_dd"] >= BAR_DD and r["w_wr"] >= wbar[0] - 1e-9 and r["w_dd"] >= wbar[1])


def main():
    t0 = time.time()
    M = C.market("w")
    print(f"market ready {time.time() - t0:.0f}s", flush=True)
    res = {}
    keys = []
    for kA in KA:
        for dlo in ([None] if kA == 0 else [None, 0.20]):
            for kT, kB, rak in itertools.product(KT, KB, RAK):
                keys.append(key(kA, dlo, kT, kB, rak))
    for i, k in enumerate(keys):
        res[k] = evaluate(M, k)
        if i % 20 == 0:
            print(f"  {i}/{len(keys)} {time.time() - t0:.0f}s", flush=True)

    def table(res):
        return pd.DataFrame([dict(name=name(k), **v) for k, v in res.items()], index=list(res.keys()))

    def select(res):
        df = table(res)
        lit = [k for k in res if passes(res[k], (BAR_WR, BAR_DD))]
        wbar = (BAR_WR, BAR_DD) if lit else (0.934, -0.6595)
        el = []
        for k in res:
            if not passes(res[k], wbar):
                continue
            nb = neighbours(k, res)
            if not nb:
                continue
            med = {m: np.median([res[x][m] for x in nb]) for m in ("n_wr", "n_dd", "w_wr", "w_dd")}
            if passes(med, wbar):
                el.append(k)
        return df, lit, wbar, el

    df, lit, wbar, el = select(res)
    # stage 2: extra coins on the 5 best stage-1 designs (by robust %/month among the step-1/3 eligible; if fewer
    # than 5 are eligible, the best by robust %/month that pass step 1)
    pool = sorted(el, key=lambda k: -res[k]["rob_mo"])[:5]
    if len(pool) < 5:
        p1 = [k for k in res if passes(res[k], wbar) and k not in pool]
        pool += sorted(p1, key=lambda k: -res[k]["rob_mo"])[:5 - len(pool)]
    for k in pool:
        for kX in (5, 4):
            kk = k[:5] + (kX,)
            res[kk] = evaluate(M, kk)
    df, lit, wbar, el = select(res)
    df = df.sort_values("rob_mo", ascending=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
    cols = ["name", "n_wr", "n_mo", "n_dd", "w_wr", "w_mo", "w_dd", "rob_mo", "n_tpm", "n_wt"]
    df.to_csv(os.path.join(C.OUT, "grid.csv"))
    print(f"\n{len(res)} designs, {time.time() - t0:.0f}s. Literal bar passed by {len(lit)}; worse-fills bar used: "
          f"win >= {wbar[0]*100:.1f}%, drop >= {wbar[1]*100:.2f}%. Eligible after the neighbour test: {len(el)}")
    print("\nTop 25 by robust 2021-23 %/month (all designs):")
    print(df[cols].head(25).to_string(float_format=lambda x: f"{x:.4f}"))
    base = res[key(0, None, None, None, None)]
    print(f"\n90% mode in this grid: win {base['n_wr']*100:.2f}% {base['n_mo']*100:+.2f}%/mo drop {base['n_dd']*100:.1f}% | "
          f"worse {base['w_wr']*100:.2f}% {base['w_mo']*100:+.2f}% {base['w_dd']*100:.1f}% | robust {base['rob_mo']*100:+.2f}%")
    if el:
        el = sorted(el, key=lambda k: (-res[k]["rob_mo"], sum(x is not None and x != 0 for x in k)))
        print("\nEligible designs (pass bar + neighbour median), best first:")
        for k in el[:15]:
            r = res[k]
            print(f"  {name(k):55s} win {r['n_wr']*100:5.2f}% {r['n_mo']*100:+6.2f}%/mo drop {r['n_dd']*100:5.1f}% | "
                  f"worse {r['w_wr']*100:5.2f}% {r['w_mo']*100:+6.2f}% {r['w_dd']*100:5.1f}% | robust {r['rob_mo']*100:+6.2f}%"
                  f" | neighbours {len(neighbours(k, res))}")
        print(f"\nPICK: {name(el[0])}  key={el[0]}")
        with open(os.path.join(C.OUT, "pick.txt"), "w") as f:
            f.write(repr(el[0]) + "\n")
    else:
        print("\nNo design passes the rule.")


if __name__ == "__main__":
    main()
