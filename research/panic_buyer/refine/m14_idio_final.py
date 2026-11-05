"""m14 idio ONE look for the pick in $PB_OUT/idio_pick.json."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m14_idio_lib as L
def table(c, show=True):
    R = {}
    for lab, kw in (("normal", {}), ("worse", dict(worse=True)), ("stressed", dict(stress=True))):
        r, e, lg = L.evaluate(c, **kw)
        if lab != "stressed": r.update(L.evaluate(c, holdout=True, **kw)[0])
        R[lab] = r
        if show:
            for p, pl in (("d", "2021-23"), ("t", "2024-26"), ("s", "since25"), ("h", "2018-20")):
                if f"{p}_wr" in r and not (lab == "stressed" and p in ("s", "h")):
                    print(f"  {lab:9s} {pl:8s} {r[f'{p}_wr']*100:5.1f} {r[f'{p}_mo']*100:+6.2f} {r[f'{p}_dd']*100:6.1f} {r[f'{p}_tpm']:5.1f}")
    return R
def items(R):
    n, w, s = R["normal"], R["worse"], R["stressed"]
    return (n["t_mo"] > .0689 and n["t_wr"] >= .925 and n["t_dd"] >= -.35,
            n["h_mo"] > .1082 and n["h_wr"] >= .925 and n["h_dd"] >= -.663,
            w["t_mo"] > .0531 and w["h_mo"] > .0741 and w["t_wr"] >= .92 and w["h_wr"] >= .92,
            s["t_mo"] > .0549 and s["d_mult"] > .05)
c = json.load(open(os.path.join(L.OUT, "idio_pick.json")))
print("90% mode"); table(L.OFF); print("PICK", json.dumps(c)); R = table(c); it = items(R)
for i, ok in enumerate(it, 1): print("item", i, "PASS" if ok else "FAIL")
print("worst stressed d mult", R["stressed"]["d_mult"])
for tag, h in (("main", False), ("hold", True)):
    r, e, lg = L.evaluate(c, holdout=h)
    e.rename("equity").rename_axis("time").to_csv(os.path.join(L.OUT, f"pick_eq_{tag}.csv"), header=True)
KK = list(range(4, 11)); DS = [.13, .14, .15]; DC = [.14, .16, .18]
k, ds, dc = c["kk"], c["ds"], c["dc"]; nbs = []
for a in (-1, 1):
    if k + a in KK: nbs.append(dict(c, kk=k + a))
    i = DS.index(ds) + a
    if 0 <= i < 3: nbs.append(dict(c, ds=DS[i]))
    j = DC.index(dc) + a
    if 0 <= j < 3: nbs.append(dict(c, dc=DC[j]))
ok = 0
for x in nbs:
    i = items(table(x, False))[:3]; ok += all(i); print(" nb", x, i)
print("neighbours passing 1-3:", ok, "/", len(nbs))
