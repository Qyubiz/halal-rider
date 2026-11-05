"""m14 speed grid. RULE (written before any run), 2021-23 only:
Designs: fast = hours since the 24h highest close <= A, A in {2,4,8,12,16}; fast-tier depth df and slow-tier depth ds each in
{12,14,16,17%, off(no bid)}, not both off (24 combos x 5 = 120 designs). Pass bars (normal AND every fill 0.2% worse):
win rate>=92.5%, drop no deeper than -64.4%, %/month>13.17 / 9.79; all 5 nearby versions (base, cd-1, cd+1, bot timing, late
start) meet win rate & drop bars. Neighbours (one step in A, df or ds along the lists): median normal-fill %/month, win rate,
drop pass the bars. Pick = highest robust score (mean %/month over the 5 versions) among passing designs; none -> no pick."""
import sys, os, json, itertools, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m14_speed_lib as L
AS = [2, 4, 8, 12, 16]; DP = [.12, .14, .16, .17, None]
D = {(a, x, y): dict(A=a, df=x, ds=y) for a, x, y in itertools.product(AS, DP, DP) if not (x is None and y is None)}
if __name__ == "__main__":
    b = L.scores(L.OFF); print("90%% mode off check: n_wr %.4f n_mo %.4f n_dd %.4f" % (b["n_wr"], b["n_mo"], b["n_dd"]))
    S = {k: L.scores(d) for k, d in D.items()}
    def bars(s, vers): return all(s[f"{t}_wr"] >= .925 and s[f"{t}_dd"] > -.6445 for t in vers)
    def nb(k):
        a, x, y = k; out = []
        for s in (-1, 1):
            i = AS.index(a) + s
            if 0 <= i < 5: out.append((AS[i], x, y))
            i = DP.index(x) + s
            if 0 <= i < 5 and (AS[0:0] or True): out.append((a, DP[i], y))
            i = DP.index(y) + s
            if 0 <= i < 5: out.append((a, x, DP[i]))
        return [q for q in out if q in S]
    rows = []
    for k, s in S.items():
        p1 = L.ok_s(s) and bars(s, ("cm", "cp", "bt", "ls"))
        ns = [S[q] for q in nb(k)]
        p3 = bool(ns) and np.median([q["n_mo"] for q in ns]) > .1317 and np.median([q["n_wr"] for q in ns]) >= .925 and np.median([q["n_dd"] for q in ns]) > -.6445
        rows.append((k, s["robust"], p1, p3, s["n_wr"], s["n_mo"], s["n_dd"], s["n_tpm"]))
    rows.sort(key=lambda r: -r[1])
    for r in rows[:25]: print(r[0], "rob %.4f p1 %s p3 %s wr %.3f mo %.4f dd %.3f tpm %.1f" % r[1:])
    print("p1 pass", sum(r[2] for r in rows), "p1&p3", sum(r[2] and r[3] for r in rows), "of", len(rows))
    ok = [r for r in rows if r[2] and r[3]]
    if ok:
        k = ok[0][0]; print("PICK", k); json.dump(D[k], open(os.path.join(L.OUT, "speed_pick.json"), "w"))
