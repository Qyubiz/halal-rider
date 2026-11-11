"""m13 clock: ONE look for the pick in $PB_OUT/clock_pick.json. Table for 90% mode (clock off, same engine) and pick,
pass-rule items 1-4 (M12_PREREG.md), and item 5 proxy: share of the pick's neighbours (window shift -1/+1h, width step,
and 2-hour shifts) passing items 1-3 each (tested on 2024-26 and 2018-20, so this is part of the look)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m13_clock_lib as C, m12_entry_lib as L
OFF = {"typ": "none", "start": 0, "len": 4, "wk": "none"}


def table(c, show=True):
    R = {}
    for lab, kw in (("normal", {}), ("worse", dict(worse=True)), ("stressed", dict(stress=True))):
        r, e, lg = C.evaluate(c, **kw)
        if lab != "stressed":
            r.update(C.evaluate(c, holdout=True, **kw)[0])
        R[lab] = r
        if show:
            for p, pl in (("d", "2021-23"), ("t", "2024-26"), ("s", "since 2025"), ("h", "2018-20")):
                if f"{p}_wr" in r and not (lab == "stressed" and p in ("s", "h")):
                    print(f"  {lab:9s} {pl:10s} {r[f'{p}_wr']*100:5.1f} {r[f'{p}_mo']*100:+6.2f} {r[f'{p}_dd']*100:6.1f} {r[f'{p}_wt']*100:6.1f} {r[f'{p}_tpm']:5.1f}")
    return R


def items(R):
    n, w, s = R["normal"], R["worse"], R["stressed"]
    i1 = n["t_mo"] > 0.0689 and n["t_wr"] >= 0.925 and n["t_dd"] >= -0.350
    i2 = n["h_mo"] > 0.1082 and n["h_wr"] >= 0.925 and n["h_dd"] >= -0.663
    i3 = w["t_mo"] > 0.0531 and w["h_mo"] > 0.0741 and w["t_wr"] >= 0.920 and w["h_wr"] >= 0.920
    i4 = s["t_mo"] > 0.0549 and s["d_mult"] > 0.05
    return i1, i2, i3, i4


def main():
    c = json.load(open(os.path.join(L.OUT, "clock_pick.json")))["pick"]
    print("90% mode, clock off (win / %/mo / worst drop / worst trade % acct / trades/mo)"); table(OFF)
    print("PICK", json.dumps(c)); R = table(c)
    it = items(R)
    for i, ok in enumerate(it, 1): print(f"pass-rule item {i}: {'PASS' if ok else 'FAIL'}")
    print("BEATS 90% MODE" if all(it) else "does NOT beat 90% mode")
    nb = [dict(c, start=(c["start"] + s) % 24) for s in (-2, -1, 1, 2)] + [dict(c, len=l) for l in {4: [6], 6: [4, 8], 8: [6]}[c["len"]]]
    ok = 0
    for x in nb:
        R2 = table(x, False); i = items(R2)[:3]; ok += all(i)
        print("  neighbour", json.dumps(x), "items1-3", i, f"t {R2['normal']['t_mo']*100:+.2f}/{R2['normal']['t_dd']*100:.1f} h {R2['normal']['h_mo']*100:+.2f}")
    print(f"neighbours passing items 1-3: {ok}/{len(nb)}")


if __name__ == "__main__":
    main()
