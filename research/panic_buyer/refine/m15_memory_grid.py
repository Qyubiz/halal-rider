"""m15 memory GRID. RULE (written before any run): 2021-23 only. Designs: blacklist-after-loss D days (bl), repeat-dip
counter (dk episodes in dM days), trend A (close under the tW-day low of tW days ago), trend B (close under tx x 90d high),
singles and pairs. Pass bars (normal AND every fill 0.2% worse): win rate >= 92.5%, drop no deeper than -64.4%, %/month
> 13.17 normal and > 9.79 worse; all 5 nearby versions (base, cd-1, cd+1, bot timing, start 2021-01-15) must pass wr and
drop bars; score = mean %/month over the 5 versions; neighbours (one step in one parameter, in the lists below) median of
normal-fill %/month, win rate, drop must pass bar 1. Pick = highest score passing everything. One pick or none."""
import itertools, json, sys
import numpy as np
import m15_memory_lib as L

BL = [2, 3, 5, 7, 10, 14, 21, 30]
DK = [2, 3, 4]; DM = [7, 14, 21, 30]
TW = [30, 60]; TX = [0.4, 0.5, 0.6, 0.7]
LISTS = dict(bl=BL, dk=DK, dM=DM, tW=TW, tx=TX)


def designs():
    D = [dict(bl=b) for b in BL]
    D += [dict(dk=k, dM=m) for k in DK for m in DM]
    D += [dict(tW=w) for w in TW] + [dict(tx=x) for x in TX]
    dips = [dict(dk=k, dM=m) for k, m in ((2, 14), (3, 14), (2, 30), (3, 30))]
    for b in (3, 7, 14):
        D += [dict(bl=b, **p) for p in dips]
        D += [dict(bl=b, tW=w) for w in TW] + [dict(bl=b, tx=x) for x in (0.5, 0.6, 0.7)]
    for p in dips:
        D += [dict(p, tW=w) for w in TW] + [dict(p, tx=x) for x in (0.5, 0.6, 0.7)]
    return D


def neighbours(d):
    out = []
    for key, val in d.items():
        lst = LISTS[key]; i = lst.index(val)
        for j in (i - 1, i + 1):
            if 0 <= j < len(lst):
                out.append(dict(d, **{key: lst[j]}))
    return out


def ok_bar1(s):
    return (s["n_wr"] >= .925 and s["n_dd"] >= -.6445 and s["n_mo"] > .1317 and s["w_wr"] >= .925 and s["w_dd"] >= -.6445 and s["w_mo"] > .0979)


def ok_wd(s):
    return all(s[f"{t}_wr"] >= .925 and s[f"{t}_dd"] >= -.6445 for t in ("n", "w", "cm", "cp", "bt", "ls"))


cache = {}
def sc(d):
    k = json.dumps(d, sort_keys=True)
    if k not in cache:
        cache[k] = L.scores(d)
    return cache[k]


if __name__ == "__main__":
    rows = []
    for d in designs():
        s = sc(d)
        p1 = ok_bar1(s) and ok_wd(s)
        row = dict(d=d, s=s, p1=p1)
        if p1:
            ns = [sc(n) for n in neighbours(d)]
            med = {k: float(np.median([n[k] for n in ns])) for k in ("n_wr", "n_dd", "n_mo")} if ns else None
            row["nb_ok"] = bool(ns) and med["n_wr"] >= .925 and med["n_dd"] >= -.6445 and med["n_mo"] > .1317
            row["med"] = med
        rows.append(row)
        print(json.dumps(d), "wr %.3f mo %.4f dd %.3f | w mo %.4f | robust %.4f %s %s" % (
            s["n_wr"], s["n_mo"], s["n_dd"], s["w_mo"], s["robust"], "P1" if p1 else "-", "NB" if row.get("nb_ok") else ""), flush=True)
    fin = sorted([r for r in rows if r["p1"] and r["nb_ok"]], key=lambda r: -r["s"]["robust"])
    print("\nFINALISTS", len(fin))
    for r in fin[:10]:
        print(json.dumps(r["d"]), "robust %.4f" % r["s"]["robust"], "med", r["med"])
    json.dump(dict(pick=fin[0]["d"] if fin else None, n_designs=len(rows)), open(L.F.OUT + "/pick.json", "w"))
