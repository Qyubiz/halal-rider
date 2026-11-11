"""m11 stack, final report: the ONE look at 2024-26, since Jan 2025 and the untouched 2018-20 holdout for the pick of
m11_stack_grid.py (chosen on 2021-23 only, rule in that file's header; pick.json). No re-pick after this run.

Rows, pick and the 90% mode through the same engine: normal fills; every buy and sell fill 0.2% worse; stressed
(real bot timing: take-profit at once after an A market buy and cooldowns +1h, fills 0.2% worse, + LUNA/FTT/SRM sold at
their last hour). Fixed before the look as extra checks: the grid's runner-up (best design WITH F, not the pick), the
pick with the 0.02%/0.1% top-50 slippage panel, and its 4 nearby versions in every period.
Holdout: hourly candles built from 1-minute data (the published 92.5 / +10.8 uses hourly CSV candles: different data).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m11_stack_lib as S  # noqa: E402

PK = json.load(open(os.path.join(S.OUT, "pick.json")))
PICK = PK["params"]
RUNNER = dict(A=1, E=1, F=1, nslots=1, fsE=1.0, fsF=1.0, nordF=1, nmaxF=1, nearA=0.06, pre=0, gEF=0.03)
PER = (("2021-23", S.T21, S.T24), ("2024-26", S.T24, None), ("since 2025", S.T25, None))


def show(name, d, D, DH, D22, extra=()):
    print(f"\n{name}: {json.dumps(d)}", flush=True)
    out = {}
    for tag, dd in (("normal", d), ("worse", S.worse(d))):
        e, lg, inv, lo = S.run(D, dd)
        for p, a, b in PER:
            r = S.stats(e, lg, inv, lo, a, b); out[(tag, p)] = r
            print("  " + S.fmt(f"{tag} {p}", r), flush=True)
        if tag == "normal" and name == "PICK":
            lg2 = lg.copy(); lg2["sym"] = [D["syms"][int(k)][:-4] for k in lg2.k]
            lg2["t_in"] = D["hidx"][lg2.eb.astype(int).values]; lg2["acct"] = lg2.ret * lg2.amt / lg2.eq0
            lg2.to_csv(os.path.join(S.OUT, "final_trades_pick.csv"), index=False)
            w = lg2.nsmallest(6, "acct")
            print("    worst trades: " + "; ".join(f"{'AEF'[int(r.sig)]} {r.sym} {r.t_in:%Y-%m-%d %H:%M} {r.acct*100:+.1f}%"
                                               for r in w.itertuples()))
            y = e.resample("YE").last(); y = y / np.concatenate([[1.0], y.values[:-1]]) - 1
            print("    years: " + "  ".join(f"{k.year} {v*100:+.0f}%" for k, v in y.items()))
        e, lg, inv, lo = S.run(DH, dd)
        r = S.stats(e, lg, inv, lo, S.T18, S.T21); out[(tag, "2018-20")] = r
        print("  " + S.fmt(f"{tag} 2018-20 holdout", r), flush=True)
    e, lg, inv, lo = S.run(D22, S.stressed(d))
    for p, a, b in PER:
        r = S.stats(e, lg, inv, lo, a, b); out[("stressed", p)] = r
        print("  " + S.fmt(f"stressed {p}", r), flush=True)
    for tag, dd, kw in extra:
        e, lg, inv, lo = S.run(D, dd, **kw)
        for p, a, b in PER:
            print("  " + S.fmt(f"{tag} {p}", S.stats(e, lg, inv, lo, a, b)), flush=True)
    return out


def main():
    D, DH, D22 = S.data("main", 19), S.data("hold"), S.data("main", 22)
    slip_panel = np.load(os.path.join(S.XCACHE, "slip.npy"))[:, :19]
    print(f"PICK (rule, 2021-23 only): {PK['design']}{PK['note']}")
    res = {"PICK": show("PICK", PICK, D, DH, D22, extra=(
        ("slip panel", PICK, dict(slip=slip_panel)), ("cd-1h", dict(PICK, cd_add=-1), {}),
        ("cd+1h", dict(PICK, cd_add=1), {}), ("bot timing", S.timing(PICK), {}),
        ("start 2021-01-15", PICK, dict(start=pd.Timestamp("2021-01-15")))))}
    res["MODE90"] = show("MODE90 (90% mode, same engine)", S.MODE90, D, DH, D22)
    res["RUNNER"] = show("RUNNER-UP with F (AEF-s1-near6-pre0-g3; NOT the pick)", RUNNER, D, DH, D22)
    print("\nVERDICT (pick vs 90% mode, same engine and data): win rate and %/month")
    for tag, p in (("normal", "2024-26"), ("normal", "since 2025"), ("normal", "2018-20"), ("worse", "2024-26"),
                   ("worse", "since 2025"), ("worse", "2018-20"), ("stressed", "2024-26")):
        a, b = res["PICK"][(tag, p)], res["MODE90"][(tag, p)]
        print(f"  {tag:8s} {p:10s} win {a['wr']*100:5.1f} vs {b['wr']*100:5.1f} ({'beats' if a['wr'] > b['wr'] else 'NO'}) | "
              f"%/mo {a['mo']*100:+6.2f} vs {b['mo']*100:+6.2f} ({'beats' if a['mo'] > b['mo'] else 'NO'}) | drop "
              f"{a['dd']*100:5.1f} vs {b['dd']*100:5.1f}")


if __name__ == "__main__":
    main()
