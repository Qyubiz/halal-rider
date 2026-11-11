"""m12 entry (agent B): the ONE look at the tests for the pick in $PB_OUT/pick.json (or a JSON design as argv[1]).
Prints win rate / %/month / worst drop / worst trade as % of the account / trades per month for 90% mode and the
pick: 2021-23, 2024-26, since 2025, 2018-20 holdout, normal fills and every fill 0.2% worse, stressed 2021-23 and
2024-26 (+LUNA/FTT/SRM, bot timing, worse fills); then pass-rule items 1-4 of M12_PREREG.md.
Usage: PB_DATA=... PB_HOLDOUT=... PB_OUT=... NUMBA_NUM_THREADS=1 python3 m12_entry_final.py [design-json]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m12_entry_lib as L  # noqa: E402


def table(d):
    R = {}
    for lab, kw in (("normal", {}), ("worse", dict(worse=True)), ("stressed", dict(stress=True))):
        r, e, lg = L.evaluate(d, **kw)
        if lab != "stressed":
            rh, _, _ = L.evaluate(d, holdout=True, **kw); r.update(rh)
        R[lab] = r
        for p, pl in (("d", "2021-23"), ("t", "2024-26"), ("s", "since 2025"), ("h", "2018-20")):
            if f"{p}_wr" in r and not (lab == "stressed" and p in ("s", "h")):
                print(f"  {lab:9s} {pl:10s} {r[f'{p}_wr']*100:5.1f} {r[f'{p}_mo']*100:+6.2f} {r[f'{p}_dd']*100:6.1f} "
                      f"{r[f'{p}_wt']*100:6.1f} {r[f'{p}_tpm']:5.1f}  x{r[f'{p}_mult']:.4g}")
    return R


def main():
    d = json.loads(sys.argv[1]) if len(sys.argv) > 1 else json.load(open(os.path.join(L.OUT, "pick.json")))["pick"]
    print("90% mode (win / %/month / worst drop / worst trade % acct / trades/mo / multiple)")
    table({})
    print(f"\nPICK {L.describe(d)}  {json.dumps(d, sort_keys=True)}")
    R = table(d)
    n, w, s = R["normal"], R["worse"], R["stressed"]
    i1 = n["t_mo"] > 0.0689 and n["t_wr"] >= 0.925 and n["t_dd"] >= -0.350
    i2 = n["h_mo"] > 0.1082 and n["h_wr"] >= 0.925 and n["h_dd"] >= -0.663
    i3 = w["t_mo"] > 0.0531 and w["h_mo"] > 0.0741 and w["t_wr"] >= 0.920 and w["h_wr"] >= 0.920
    i4 = s["t_mo"] > 0.0549 and s["d_mult"] > 0.05 and (s["d_dd"] > -0.95)
    for i, ok in enumerate((i1, i2, i3, i4), 1):
        print(f"pass-rule item {i}: {'PASS' if ok else 'FAIL'}")
    print("BEATS 90% MODE" if all((i1, i2, i3, i4)) else "does NOT beat 90% mode")


if __name__ == "__main__":
    main()
