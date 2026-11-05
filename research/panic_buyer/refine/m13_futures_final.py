"""m13 futures: the ONE look at the pick from m13_futures_grid.py (pick.json): 2021-23, 2024-26, since 2025, 2018-20
holdout (futures-covered part only; uncovered coin-hours bid as 90% mode), normal / 0.2%-worse fills, stressed;
next to 90% mode; M12_PREREG.md pass items 1-4; share of the pick's parameter neighbours passing items 1-3.
Usage: PB_DATA=... PB_HOLDOUT=... PB_OUT=... PB_FUT=... NUMBA_NUM_THREADS=1 python3 m13_futures_final.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m13_futures_lib as F  # noqa: E402
import m13_futures_grid as G  # noqa: E402


def table(d):
    r = {}
    for lab, kw in (("n", {}), ("w", dict(worse=True)), ("x", dict(stress=True))):
        a = F.evaluate(d, **kw)[0]
        if lab != "x":
            a.update(F.evaluate(d, holdout=True, **kw)[0])
        r.update({f"{lab}{k}": v for k, v in a.items()})
    return r


def items(r):
    i1 = r["nt_mo"] > 0.0689 and r["nt_wr"] >= 0.925 and r["nt_dd"] >= -0.350
    i2 = r["nh_mo"] > 0.1082 and r["nh_wr"] >= 0.925 and r["nh_dd"] >= -0.663
    i3 = r["wt_mo"] > 0.0531 and r["wh_mo"] > 0.0741 and r["wt_wr"] >= 0.92 and r["wh_wr"] >= 0.92
    i4 = r["xt_mo"] > 0.0549 and r["xd_dd"] > -0.95
    return i1, i2, i3, i4


def main():
    out = open(os.path.join(F.OUT, "final.log"), "a")

    def say(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); out.write(s + "\n"); out.flush()
    pick = json.load(open(os.path.join(F.OUT, "pick.json")))
    say("pick", json.dumps(pick))
    R = {"90% mode": table({}), "pick": table(pick)}
    say("win / %/month / worst drop / worst trade % acct / trades per month")
    for lab, pl in (("n", "normal"), ("w", "0.2% worse"), ("x", "stressed")):
        for p, pn in (("d", "2021-23"), ("t", "2024-26"), ("s", "since 2025"), ("h", "2018-20")):
            k = f"{lab}{p}"
            if f"{k}_wr" not in R["pick"]:
                continue
            say(f"{pl:10s} {pn:10s} " + " || ".join(
                f"{nm}: {r[k+'_wr']*100:5.1f} {r[k+'_mo']*100:+6.2f} {r[k+'_dd']*100:6.1f} {r[k+'_wt']*100:6.1f} "
                f"{r[k+'_tpm']:5.1f}" for nm, r in R.items()))
    say("pass items 1-4:", items(R["pick"]))
    nbs = G.neighbours(pick)
    ok = 0
    for nd in nbs:
        r = table(nd)
        it = items(r)
        ok += all(it[:3])
        say("  nb", G.fmt(dict(G.score(nd))), "| items", it)
    say(f"neighbours passing items 1-3: {ok}/{len(nbs)}")


if __name__ == "__main__":
    main()
