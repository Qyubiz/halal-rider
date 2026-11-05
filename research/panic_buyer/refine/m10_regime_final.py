"""m10 regime: out-of-sample report for the designs m10_regime_grid.py chose on 2021-23 only.

For PICK and BAND95/96/97 (and 90% mode): per period 2018-20 holdout, 2021-23, 2024-26, since 2025: win rate,
%/month, worst drop, worst trade (% of account), trades/month; normal fills, every fill 0.2% worse (buys and sells),
and real bot timing + fills 0.2% worse + LUNA/FTT/SRM.
Then (diagnostic only, NOT used to choose): the 2024-26 / since-2025 / holdout spread of the 25 best designs by
the grid's 2021-23 ranking, to see whether the 2021-23 gain carries over at all.

Usage: python m10_regime_final.py  (needs $PB_OUT/grid.csv)  -> final.log, final.csv
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m10_regime_grid as G  # noqa: E402
import m10_regime_lib as L  # noqa: E402


def parse(rules_s):
    """'(a>=1) OR (b<=2 & c>=3)' -> rules list."""
    if rules_s == "none":
        return []
    out = []
    for part in rules_s.split(" OR "):
        cond = []
        for term in part.strip("()").split(" & "):
            for op in (">=", "<=", ">", "<"):
                if op in term:
                    f, v = term.split(op)
                    cond.append((f, op, float(v)))
                    break
        out.append(cond)
    return out


def full_eval(rules, deep):
    deep = deep if deep and deep > 0 else None
    res = {}
    for lab, over in (("normal", {}), ("fills 0.2% worse", L.stress_kw()), ("bot+0.2%+LUNA", L.bot_kw())):
        rd, _, _, _ = L.run("dx" if lab == "bot+0.2%+LUNA" else "d", rules=rules, deep=deep, **over)
        rh, _, _, _ = L.run("hx" if lab == "bot+0.2%+LUNA" else "h", rules=rules, deep=deep, **over)
        res[lab] = {**rh, **rd}
    return res


def table(p, name, rules, deep, res):
    p(f"\n{name}\n  rules: {L.rules_str(rules)}" + (f"  -> blocked hours bid at {deep*100:.0f}% instead of skipping"
                                                    if deep else "  -> blocked hours: no bid"))
    p(f"  {'':18s} {'period':8s} {'win%':>6s} {'%/mo':>7s} {'maxDD':>6s} {'worst tr':>8s} {'tr/mo':>6s} {'losers':>6s}")
    for lab, r in res.items():
        for t, nm in (("h", "2018-20"), ("d", "2021-23"), ("t", "2024-26"), ("s", "since25")):
            p(f"  {lab:18s} {nm:8s} {r[f'{t}_wr']*100:6.1f} {r[f'{t}_mo']*100:+7.2f} {r[f'{t}_dd']*100:6.1f} "
              f"{r[f'{t}_wt']*100:+8.1f} {r[f'{t}_tpm']:6.1f} {r[f'{t}_nl']:6d}")


def main():
    log = open(os.path.join(L.OUT, "final.log"), "w")

    def p(s):
        print(s, flush=True)
        log.write(s + "\n")

    R = pd.read_csv(os.path.join(L.OUT, "grid.csv"))
    base = R[R.stage == "base"].iloc[0]
    ok = R[(R.d_mo >= 0.9 * base["d_mo"]) & (R.d_dd >= base["d_dd"] - 0.01)]
    sel = {"90% mode": base, "PICK": ok.sort_values(["d_wr", "d_mo"], ascending=False).iloc[0]}
    for k in (0.95, 0.96, 0.97):
        B = R[(R.d_wr >= k) & (R.d_n >= 100)]
        if len(B):
            sel[f"BAND{int(k*100)}"] = B.sort_values(["d_mo", "d_wr"], ascending=False).iloc[0]
    rows = []
    done = {}
    for tag, r in sel.items():
        key = (r["rules"], r["deep"])
        rules = parse(r["rules"])
        res = done.get(key) or full_eval(rules, r["deep"])
        done[key] = res
        table(p, f"{tag}: {r['name']}", rules, r["deep"] if r["deep"] > 0 else None, res)
        for lab, x in res.items():
            rows.append({"tag": tag, "design": r["name"], "test": lab, **x})
    pd.DataFrame(rows).to_csv(os.path.join(L.OUT, "final.csv"), index=False)

    p("\nDIAGNOSTIC (not used to choose): top 25 designs by the grid's 2021-23 ranking, out of sample")
    p(f"  {'design':80s} {'21-23 WR/%mo':>14s} | {'24-26 WR/%mo/DD':>18s} | {'since25 WR/%mo':>15s} | {'2018-20 WR/%mo/DD':>18s}")
    top = G.rank(R, base).head(25)
    hold = []
    for _, r in pd.concat([R[R.stage == "base"], top]).iterrows():
        rh, _, _, _ = L.run("h", rules=parse(r["rules"]), deep=r["deep"] if r["deep"] > 0 else None)
        hold.append(rh)
        p(f"  {r['name'][:80]:80s} {r['d_wr']*100:5.1f}/{r['d_mo']*100:+6.2f} | {r['t_wr']*100:5.1f}/{r['t_mo']*100:+5.2f}/"
          f"{r['t_dd']*100:4.0f} | {r['s_wr']*100:5.1f}/{r['s_mo']*100:+5.2f} | {rh['h_wr']*100:5.1f}/{rh['h_mo']*100:+6.2f}/"
          f"{rh['h_dd']*100:4.0f}")
    H = pd.DataFrame(hold[1:])
    p(f"  top-25 medians: 2024-26 WR {top.t_wr.median()*100:.1f}% {top.t_mo.median()*100:+.2f}%/mo | since25 WR "
      f"{top.s_wr.median()*100:.1f}% {top.s_mo.median()*100:+.2f}%/mo | 2018-20 WR {H.h_wr.median()*100:.1f}% "
      f"{H.h_mo.median()*100:+.2f}%/mo; share beating 90% mode on WR: 2024-26 {(top.t_wr > base['t_wr']).mean()*100:.0f}%, "
      f"since25 {(top.s_wr > base['s_wr']).mean()*100:.0f}%, 2018-20 {(H.h_wr > hold[0]['h_wr']).mean()*100:.0f}%; on %/mo: "
      f"2024-26 {(top.t_mo > base['t_mo']).mean()*100:.0f}%, since25 {(top.s_mo > base['s_mo']).mean()*100:.0f}%, "
      f"2018-20 {(H.h_mo > hold[0]['h_mo']).mean()*100:.0f}%")


if __name__ == "__main__":
    main()
