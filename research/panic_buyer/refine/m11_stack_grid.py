"""m11 stack, grid (design period 2021-01-01 .. 2024-01-01 ONLY: every run is cut at 2024-01-01, nothing later is
simulated). Usage: python m11_stack_grid.py 1   (stage 1: all designs, normal fills + every fill 0.2% worse)
                   python m11_stack_grid.py 2   (stage 2: nearby versions of the qualifiers, then the pick)

Designs (152 + the 90% mode reference). A, E, F keep their published rules; A always sizes at all free cash (<= 100%).
  signals   AE, AF, AEF
  sizing    s1 one pot, one position: E and F all free cash, 1 position in total
            s2 E and F 50% of the account each, F 1 bid per minute and <= 2 positions, <= 3 positions in total
            s3 E and F 25% of the account each, F 4 bids per minute (the published F), no position cap
  nearA     A bids at the hour's open only if its coin opens <= 3% / 6% above the trigger (else its cash is lent to E/F)
  pre       0 A only through its hourly bid; 1 + A watches every minute (market buy with free cash when a coin opens
            under its A trigger); 2 + sells E/F in other coins to fund A (and tops up an under-funded A fill);
            3 + also sells E/F in the same coin
  gEF       -1 off; 0.03: no new E/F bid while an A-eligible coin opens within 3% of its A trigger
  + lots=1 (A may hold a coin E/F hold, as a separate lot) for AEF s2/s3, nearA 3%/6%, pre 0/2, gEF off (8 designs)

SELECTION RULE (written before any result was seen; 2021-23 only):
  1. Qualify: 2021-23 win rate >= 93.8% and worst drop (hourly closes) >= -64.5% (i.e. -64% rounded, the 90% mode's
     -64.4%), with normal fills AND with every buy and sell fill 0.2% worse.
  2. For the qualifiers (if more than 12: the 12 with the highest normal-fill 2021-23 %/month) run 4 nearby versions:
     A and E cooldown -1h, +1h, the bot timing switch (take-profit at once after an A market buy, cooldowns +1h), and
     the start moved to 2021-01-15. Score = mean 2021-23 %/month over the base run and the 4 versions.
  3. Pick = highest score. If nothing qualifies, the design with the highest score among those meeting step 1 with
     normal fills only is shown as "best non-qualifying" and the verdict is that nothing beats the 90% mode.
  The pick is then run ONCE on 2024-26, since 2025, the 2018-20 holdout, worse fills and the stressed run
  (m11_stack_final.py); no re-pick after that.
"""
import itertools
import json
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m11_stack_lib as S  # noqa: E402

SIGS = {"AE": dict(E=1, F=0), "AF": dict(E=0, F=1), "AEF": dict(E=1, F=1)}
SIZ = {"s1": dict(nslots=1, fsE=1.0, fsF=1.0, nordF=1, nmaxF=1),
       "s2": dict(nslots=3, fsE=0.5, fsF=0.5, nordF=1, nmaxF=2),
       "s3": dict(nslots=99, fsE=0.25, fsF=0.25, nordF=4, nmaxF=99)}


def designs():
    out = {"MODE90": dict(S.MODE90)}
    for sg, sz, na, pr, g in itertools.product(SIGS, SIZ, (0.03, 0.06), (0, 1, 2, 3), (-1.0, 0.03)):
        out[f"{sg}-{sz}-near{int(na*100)}-pre{pr}-g{'off' if g < 0 else int(g*100)}"] = dict(
            A=1, **SIGS[sg], **SIZ[sz], nearA=na, pre=pr, gEF=g)
    for sz, na, pr in itertools.product(("s2", "s3"), (0.03, 0.06), (0, 2)):
        out[f"AEF-{sz}-near{int(na*100)}-pre{pr}-goff-lots"] = dict(A=1, E=1, F=1, **SIZ[sz], nearA=na, pre=pr, lots=1)
    return out


def r21(D, d, start=S.T21):
    e, lg, inv, lo = S.run(D, d, start=start, end=S.T24)
    return S.stats(e, lg, inv, lo, start, S.T24)


def main(stage):
    D = S.data("main", 19)
    DS = designs()
    f1 = os.path.join(S.OUT, "grid1.csv")
    if stage == 1:
        rows = []
        for i, (name, d) in enumerate(DS.items()):
            t0 = time.time()
            rn, rw = r21(D, d), r21(D, S.worse(d))
            rows.append({"design": name, **{f"n_{k}": v for k, v in rn.items()}, **{f"w_{k}": v for k, v in rw.items()}})
            print(f"{i:3d} {name:36s} normal win {rn['wr']*100:5.1f}% {rn['mo']*100:+6.2f}%/mo dd {rn['dd']*100:5.1f}% "
                  f"wt {rn['wt']*100:+5.1f}% {rn['tpm']:5.1f}/mo (A {rn['nA']} E {rn['nE']} F {rn['nF']} pre {rn['npre']}) | "
                  f"worse win {rw['wr']*100:5.1f}% {rw['mo']*100:+6.2f}%/mo dd {rw['dd']*100:5.1f}%  ({time.time()-t0:.0f}s)",
                  flush=True)
            pd.DataFrame(rows).to_csv(f1, index=False)
        return
    G = pd.read_csv(f1)
    q = G[(G.n_wr >= 0.938) & (G.n_dd >= -0.645) & (G.w_wr >= 0.938) & (G.w_dd >= -0.645)]
    note = ""
    if q.empty:
        q = G[(G.n_wr >= 0.938) & (G.n_dd >= -0.645)]
        note = " (NOTHING qualifies with worse fills: best non-qualifying, normal fills only)"
    print(f"qualifiers: {len(q)} of {len(G)}{note}")
    q = q.sort_values("n_mo", ascending=False).head(12)
    rows = []
    for _, r in q.iterrows():
        d = DS[r.design]
        v = [r.n_mo, r21(D, dict(d, cd_add=-1))["mo"], r21(D, dict(d, cd_add=1))["mo"], r21(D, S.timing(d))["mo"],
             r21(D, d, start=pd.Timestamp("2021-01-15"))["mo"]]
        rows.append({"design": r.design, "score": sum(v) / len(v), "base": v[0], "cd-1": v[1], "cd+1": v[2],
                     "timing": v[3], "start+2w": v[4], "win": r.n_wr, "dd": r.n_dd, "w_win": r.w_wr, "w_dd": r.w_dd,
                     "w_mo": r.w_mo})
        print("  " + "  ".join(f"{k} {x*100:+.2f}" if isinstance(x, float) and k not in ("design",) else f"{x}"
                               for k, x in rows[-1].items()), flush=True)
    R = pd.DataFrame(rows).sort_values("score", ascending=False)
    R.to_csv(os.path.join(S.OUT, "grid2.csv"), index=False)
    pick = R.iloc[0].design
    print(f"\nPICK{note}: {pick}  score {R.iloc[0].score*100:+.2f}%/mo\n{json.dumps(DS[pick])}")
    json.dump({"design": pick, "params": DS[pick], "note": note}, open(os.path.join(S.OUT, "pick.json"), "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]))
