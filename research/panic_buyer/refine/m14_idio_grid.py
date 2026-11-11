"""m14 idio grid. RULE (written before any run), 2021-23 only:
Designs: kk (crash if >=kk of 19 coins >=10% under 24h high close) 4..10; single-coin depth ds {13,14,15}%; crash depth
dc {14,16,18}%; crash coins {majors, all}; plus 'none' (no bids in a crash; dc irrelevant). 147 designs.
Pass bars (normal AND every fill 0.2% worse): win rate>=92.5%, drop no deeper than -64.4%, %/month>13.17 / 9.79; all 5
nearby versions (base, cd-1, cd+1, bot timing, late start) must meet win rate & drop bars. Neighbours (one step in kk,
ds or dc, same crash coin set): median normal-fill %/month, win rate, drop pass the bars. Pick = highest robust score
(mean %/month over the 5 versions) among passing designs; none passing -> no pick."""
import sys, os, json, itertools, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m14_idio_lib as L
KK = list(range(4, 11)); DS = [.13, .14, .15]; DC = [.14, .16, .18]
D = {}
for kk, ds, cs in itertools.product(KK, DS, ("maj", "all", "none")):
    for dc in (DC if cs != "none" else [.14]):
        D[(kk, ds, dc, cs)] = dict(kk=kk, ds=ds, dc=dc, cs=cs)
S = {k: L.scores(d) for k, d in D.items()}
def bars(s, vers): return all(s[f"{t}_wr"] >= .925 and s[f"{t}_dd"] > -.644 for t in vers)
def nb(k):
    kk, ds, dc, cs = k; out = []
    for a in (-1, 1):
        if kk + a in KK: out.append((kk + a, ds, dc, cs))
        i = DS.index(ds) + a
        if 0 <= i < 3: out.append((kk, DS[i], dc, cs))
        if cs != "none":
            j = DC.index(dc) + a
            if 0 <= j < 3: out.append((kk, ds, DC[j], cs))
    return out
rows = []
for k, s in S.items():
    p1 = L.ok_s(s) and bars(s, ("cm", "cp", "bt", "ls"))
    ns = [S[x] for x in nb(k)]
    p3 = np.median([x["n_mo"] for x in ns]) > .1317 and np.median([x["n_wr"] for x in ns]) >= .925 and np.median([x["n_dd"] for x in ns]) > -.644
    rows.append((k, s["robust"], p1, p3, s["n_wr"], s["n_mo"], s["n_dd"], s["n_tpm"]))
rows.sort(key=lambda r: -r[1])
print("90%% mode off check: robust %.4f" % L.scores(L.OFF)["robust"])
for r in rows[:25]: print(r[0], "rob %.4f p1 %s p3 %s wr %.3f mo %.4f dd %.3f tpm %.1f" % r[1:])
print("p1 pass", sum(r[2] for r in rows), "p1&p3", sum(r[2] and r[3] for r in rows), "of", len(rows))
ok = [r for r in rows if r[2] and r[3]]
if ok:
    k = ok[0][0]; print("PICK", k); json.dump(D[k], open(os.path.join(L.OUT, "idio_pick.json"), "w"))
else: print("NO PICK")
