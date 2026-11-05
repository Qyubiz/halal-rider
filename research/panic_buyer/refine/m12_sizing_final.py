"""m12 sizing (agent A): the ONE look at the tests for the pick in $PB_OUT/pick.json (chosen by m12_sizing_grid.py on
2021-23). Prints the pick and 90% mode (same engine) for 2021-23, 2024-26, since 2025, 2018-20 holdout with normal and
0.2%-worse fills, plus the stressed run (bot timing, worse fills, +LUNA/FTT/SRM), then M12_PREREG.md pass items 1-4.
No re-picking after this. Usage: source m12env.sh; python3 m12_sizing_final.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m12_sizing_lib as S  # noqa: E402
import m10_velocity_lib as V  # noqa: E402


def rows(d):
    out = {}
    for lab, kw in (("normal", {}), ("worse", dict(worse=True)), ("stressed", dict(stress=True))):
        r, e, lg = S.evaluate(d, **kw)
        out[lab] = r
        if lab == "stressed":
            x = e[e.index < S.T24]
            out["stress_min21"] = float((x / x.cummax() - 1).min())
        if lab != "stressed":
            out[lab].update(S.evaluate(d, holdout=True, **kw)[0])
    return out


def line(name, R):
    lab = {"d": "2021-23", "t": "2024-26", "s": "since-25", "h": "2018-20"}
    for run in ("normal", "worse", "stressed"):
        for p in (("d", "t", "s", "h") if run != "stressed" else ("d", "t", "s")):
            r = R[run]
            print(f"{name:9s} {run:8s} {lab[p]:8s} WR {r[p+'_wr']*100:5.1f}  {r[p+'_mo']*100:+6.2f}%/mo  drop "
                  f"{r[p+'_dd']*100:6.1f}%  worst trade {r[p+'_wt']*100:+6.1f}% acct  {r[p+'_tpm']:5.1f} trades/mo")


def main():
    pk = json.load(open(os.path.join(S.OUT, "pick.json")))
    d = pk["design"]
    print("PICK:", S.describe(d), json.dumps(d))
    P, B = rows(d), rows({})
    line("pick", P); line("90% mode", B)
    n, w, s = P["normal"], P["worse"], P["stressed"]
    i1 = n["t_mo"] > 0.0689 and n["t_wr"] >= 0.925 and n["t_dd"] >= -0.350
    i2 = n["h_mo"] > 0.1082 and n["h_wr"] >= 0.925 and n["h_dd"] >= -0.663
    i3 = w["t_mo"] > 0.0531 and w["h_mo"] > 0.0741 and w["t_wr"] >= 0.92 and w["h_wr"] >= 0.92
    i4 = s["t_mo"] > 0.0549 and P["stress_min21"] > -0.95
    print(f"stressed 2021-23 worst drop (whole run to 2023-12-31): {P['stress_min21']*100:.1f}%")
    for i, ok in enumerate((i1, i2, i3, i4), 1):
        print(f"pass item {i}: {'PASS' if ok else 'FAIL'}")
    print("BEATS 90% MODE (items 1-4)" if all((i1, i2, i3, i4)) else "DOES NOT BEAT 90% MODE")


if __name__ == "__main__":
    main()
