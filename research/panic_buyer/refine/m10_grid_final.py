"""m10 grid, final: pick on 2021-23 with the pre-registered rule over all search stages, then report it ONCE on
2024-26, since Jan 2025 and the 2018-20 holdout, normal and with every fill 0.1% worse.

Rule (m10_grid_search.py): round-trip win rate >= 90% and max drawdown >= -35% in 2021-23, then best 2021-23 %/month.
The 2021-26 run is one continuous simulation from 1 Jan 2021; the holdout is a separate run from 1 Jan 2018 to
1 Jan 2021 (coins enter when listed; a sleeve stays in cash until its coin lists). Writes $PB_OUT/final.txt.
"""
import os

import pandas as pd

import m10_grid_lib as G
from m10_grid_search import CSETS, DD_MIN, WR_MIN

KEYS = ("cset", "s", "L", "Ltot", "off", "tpk", "gate", "liq", "stop", "cd", "hm", "hs", "trail")


def pick():
    df = pd.concat([pd.read_csv(os.path.join(G.OUT, f"search_{i}.csv")) for i in (1, 2, 3, 4)], ignore_index=True)
    df = df.drop_duplicates("name")
    ok = df[(df.wr >= WR_MIN) & (df.dd >= DD_MIN)].sort_values("mo", ascending=False)
    r = ok.iloc[0]
    d = {k: r[k] for k in KEYS if k in r and not pd.isna(r[k])}
    for k in ("L", "off", "tpk", "liq", "cd", "hm", "trail"):
        if k in d:
            d[k] = int(d[k])
    if "Ltot" in d:
        d["Ltot"] = float(d["Ltot"])
    return d, len(df), len(ok), ok.head(5)


def main():
    out = []
    say = lambda x="": (print(x), out.append(x))
    d, n_all, n_ok, top = pick()
    say(f"{n_all} distinct designs searched on 2021-23; {n_ok} pass the rule. Pick: {G.dname(d)}")
    say("Next four on 2021-23: " + " | ".join(f"{x['name'][:60]} {x.mo*100:+.1f}%/mo DD {x.dd*100:.0f}%"
                                               for _, x in top.iloc[1:].iterrows()))
    idx, P = G.market()
    coins = CSETS[d["cset"]]
    gc = {}
    for pen, tag in ((0.0, "normal fills"), (0.001, "every fill 0.1% worse")):
        say(f"\n== {tag} ==")
        e, lg = G.run_any(idx, P, coins, d, G.T_D0, None, gc, pen=pen)
        for k, lab in (("d", "2021-23 (design)"), ("t", "2024-26"), ("s", "since Jan 2025")):
            a, b = G.PERIODS[k]
            say(G.fmt(f"{lab:17s}", G.period_stats(e, lg, idx, a, b)))
        eh, lh = G.run_any(idx, P, coins, d, G.T_H0, G.T_D0, gc, pen=pen)
        say(G.fmt(f"{'2018-20 holdout':17s}", G.period_stats(eh, lh, idx, G.T_H0, G.T_D0)))
        if pen == 0.0:
            y = e.resample("YE").last(); y = pd.concat([pd.Series([1.0], [e.index[0]]), y]).pct_change().dropna()
            yh = eh.resample("YE").last(); yh = pd.concat([pd.Series([1.0], [eh.index[0]]), yh]).pct_change().dropna()
            say("per year: " + " ".join(f"{t.year} {v*100:+.0f}%" for t, v in pd.concat([yh, y]).items()))
            for nm, x in (("2021-26", lg), ("holdout", lh)):
                say(f"exits {nm}: " + ", ".join(f"{w}: {int((x.why == c).sum())} (win {((x.ret > 0) & (x.why == c)).sum()})"
                                                for c, w in ((0, "grid sell"), (3, "regime exit"), (9, "open at end"))))
            gate = G.gates_for(P, idx, d["gate"], gc)["BTCUSDT"]
            gs = pd.Series(gate, idx)
            say("share of hours the filter allows buying: " + " ".join(
                f"{y_}: {gs[gs.index.year == y_].mean()*100:.0f}%" for y_ in range(2018, 2027)))
            # coins held vs. buy-and-hold of the same 5 coins (equal weight), for context
            for nm, a, b in (("2024-26", G.T_T0, None), ("since 2025", G.T_S0, None), ("2018-20", G.T_H0, G.T_D0)):
                c = P["c"][coins].ffill()
                c = c[(c.index >= a) & ((c.index < b) if b is not None else True)]
                bh = (c / c.bfill().iloc[0]).fillna(1.0).mean(axis=1)
                m = (bh.iloc[-1]) ** (1 / (len(bh) / (24 * 365.25 / 12))) - 1
                dd = (bh / bh.cummax() - 1).min()
                say(f"buy & hold same 5 coins {nm}: {m*100:+.2f}%/mo, drop {dd*100:.0f}%")
    with open(os.path.join(G.OUT, "final.txt"), "w") as f:
        f.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
