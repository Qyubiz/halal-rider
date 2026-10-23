"""m11 fusion: the ONE look at the pick chosen by m11_fusion_search.py ($PB_OUT/pick.json) outside 2021-23.

Prints win rate / %/month / worst drop / worst trade (% of the account) / trades per month for 2021-23, 2024-26,
since 2025 and the 2018-20 holdout, with normal fills, every fill 0.2% worse, and the stressed run (bot timing:
take-profit at once after a market buy + cooldowns +1h, fills 0.2% worse, LUNA/FTT/SRM sold when they stop
trading), for the pick and for 90% mode through the same engine. Then the verdict. The section after the verdict
is HINDSIGHT (each fused piece removed one at a time): it explains the result and is never used to re-pick.
Usage: PB_DATA=... PB_HOLDOUT=... PB_OUT=... NUMBA_NUM_THREADS=1 python3 m11_fusion_final.py
"""
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m11_fusion_lib as F  # noqa: E402

PL = (("d", "2021-23"), ("t", "2024-26"), ("s", "since 2025"), ("h", "2018-20 holdout"))


def load_pick():
    d = json.load(open(os.path.join(F.OUT, "pick.json")))
    for t in d["tiers"]:
        if isinstance(t.get("tp"), list):
            t["tp"] = tuple(t["tp"])
    return d


def table(d):
    out = {}
    for lab, kw in (("normal", {}), ("worse 0.2%", dict(worse=True)), ("stressed", dict(stress=True))):
        r, _, lg = F.evaluate(d, **kw)
        rh, _, lgh = F.evaluate(d, holdout=True, **kw)
        r.update(rh)
        out[lab] = r
    return out


def show(name, T):
    print(f"\n{name}\n  fills       period            win%   %/mo   drop  worst-trade  tr/mo")
    for lab, r in T.items():
        for p, pl in PL:
            print(f"  {lab:11s} {pl:16s} {r[f'{p}_wr']*100:5.1f} {r[f'{p}_mo']*100:+6.2f} {r[f'{p}_dd']*100:6.1f} "
                  f"{r[f'{p}_wt']*100:+8.1f}% {r[f'{p}_tpm']:6.1f}")
        per = [f"{pl}: " + " / ".join(f"tier{s} {r.get(f'{p}_n{s}', 0)} tr {r.get(f'{p}_wr{s}', float('nan'))*100:.1f}%"
                                      for s in range(3) if f"{p}_n{s}" in r) for p, pl in PL]
        print(f"  {lab:11s} per trigger: " + "; ".join(per))


def main():
    pick = load_pick()
    print(f"PICK (chosen on 2021-23 by m11_fusion_search.py): {F.describe(pick)}")
    print(json.dumps(pick))
    P = table(pick)
    B = table(F.MODE90)
    show("PICK", P)
    show("90% mode (same engine)", B)
    print("\nVERDICT (beats = higher win rate AND higher %/month than 90% mode)")
    for lab in P:
        for p, pl in PL[1:]:
            a, b = P[lab], B[lab]
            w = a[f"{p}_wr"] > b[f"{p}_wr"]; m = a[f"{p}_mo"] > b[f"{p}_mo"]
            print(f"  {lab:11s} {pl:16s} win {a[f'{p}_wr']*100:5.1f} vs {b[f'{p}_wr']*100:5.1f} "
                  f"{'yes' if w else 'NO '} | %/mo {a[f'{p}_mo']*100:+6.2f} vs {b[f'{p}_mo']*100:+6.2f} "
                  f"{'yes' if m else 'NO '} | {'BEATS' if w and m else 'does not beat'}")
    # ---------------- HINDSIGHT: remove one fused piece at a time (not used to choose anything)
    print("\nHINDSIGHT ablation (normal fills; win% %/mo drop per period) -- explanation only, no re-picking")
    abl = []
    if len(pick["tiers"]) > 1:
        x = copy.deepcopy(pick); x["tiers"] = x["tiers"][:1]; abl.append(("without the second trigger", x))
    for i, t in enumerate(pick["tiers"]):
        if t.get("vol", 0) > 0:
            x = copy.deepcopy(pick); x["tiers"][i]["vol"] = 0.0; abl.append((f"tier{i} without calm-coin filter", x))
        if isinstance(t.get("tp"), tuple):
            x = copy.deepcopy(pick); x["tiers"][i]["tp"] = 0.025 if i == 0 else 0.03
            abl.append((f"tier{i} fixed target", x))
        if t.get("cdw", 3) != 3:
            x = copy.deepcopy(pick); x["tiers"][i]["cdw"] = 3; abl.append((f"tier{i} 3h cooldown after a win", x))
    for k, off in (("hs", 0.0), ("killm", 0.0), ("cln", 0), ("nbid", 1), ("guard", 0.35)):
        if k in pick and pick[k] != off:
            x = copy.deepcopy(pick); x[k] = off
            if k == "nbid":
                for t in x["tiers"]:
                    t["fs"] = 1.0
            abl.append((f"without {k}", x))
    for name, x in [("pick", pick)] + abl:
        r, _, _ = F.evaluate(x)
        r.update(F.evaluate(x, holdout=True)[0])
        print(f"  {name:34s} " + " | ".join(f"{pl[:7]} {r[f'{p}_wr']*100:5.1f} {r[f'{p}_mo']*100:+6.2f} "
                                            f"{r[f'{p}_dd']*100:4.0f}" for p, pl in PL))


if __name__ == "__main__":
    main()
