"""m10 ladder, stages 2-3: robust selection on 2021-23 ONLY (panel cut at 2024-01-01).

Why: stage 1 (m10_ladder_grid.py) showed one-coin all-in ladders are path-chaotic: fills 0.1% worse, or the bot's
market-buy timing, reshuffle the whole trade sequence and move 2021-23 by +-3 %/month. So every design here runs on
six versions of 2021-23 and is scored on all of them:
  P0 normal | P1 every fill 0.1% worse | P2 every fill 0.2% worse | P3 bot timing (market-buy take-profit at once,
  cooldown +1h) | P4 bot timing + every fill 0.2% worse | P5 start two weeks later (15 Jan 2021).
Score: median %/month over P0-P5; also min strict win rate, worst drop, worst trade (% of account) over P0-P5.

SELECTION RULE (fixed before stage 2 ran): among designs with min strict WR >= 97% and worst drop >= -65% over P0-P5,
the highest median %/month. Stage 3 adds option switches to the stage-2 leaders and the rule is applied to stage 2 + 3
together. Ties within 0.3 %/month: the higher min WR.

Stage 2 grid (168): legs {12/20 w1:2, 13/21 w1:2, 12/22 w1:2, 14/22 w1:1, 12/18/25 w2:3:5, 12/17/22/28 w1:1:2:2,
10/15/20/25/30 equal} x holdings {1 coin x 100%, 2 coins x 50%} x take-profit {2, 2.5, 3, 4%} x time stop {7, 14, 30 days};
cooldown 3h, crash guard 35%, reserve funding, no max-loss.
Stage 3: the 8 best stage-2 designs (by the rule's score, WR/drop filters relaxed to 96% / -70% to keep 8) x options
{max-loss 30% under the average, break-even target after 7 days, cooldown 24h, crash guard 25%, no crash guard,
2 coins x 75% lazy, 3 coins x 50% lazy}, one at a time (8 x 7 = 56 designs).

Usage: PB_DATA=... PB_OUT=... python3 m10_ladder_grid2.py 2|3
"""
import os
import sys
import time

import numpy as np
import pandas as pd

import m10_ladder_lib as M

PERT = (("P0", {}, None), ("P1", dict(pen=0.001, pens=0.001), None), ("P2", dict(pen=0.002, pens=0.002), None),
        ("P3", "bot", None), ("P4", "botpen", None), ("P5", {}, pd.Timestamp("2021-01-15")))

LEGS2 = {
    "12/20 w1:2": ((0.12, 1), (0.20, 2)),
    "13/21 w1:2": ((0.13, 1), (0.21, 2)),
    "12/22 w1:2": ((0.12, 1), (0.22, 2)),
    "14/22 w1:1": ((0.14, 1), (0.22, 1)),
    "12/18/25 w2:3:5": ((0.12, 2), (0.18, 3), (0.25, 5)),
    "12/17/22/28 w1:1:2:2": ((0.12, 1), (0.17, 1), (0.22, 2), (0.28, 2)),
    "10/15/20/25/30": ((0.10, 1), (0.15, 1), (0.20, 1), (0.25, 1), (0.30, 1)),
}


def stage2():
    G = []
    for legs in LEGS2.values():
        for H, f in ((1, 1.0), (2, 0.5)):
            for tp in (0.02, 0.025, 0.03, 0.04):
                for hm in (168, 336, 720):
                    G.append(dict(legs=legs, H=H, fmax=f, tp=tp, hm=hm, cd=3, guard=0.35))
    return G


OPTS3 = {"ml30": dict(ml=0.30), "be7d": dict(beh=168, be=0.003), "cd24": dict(cd=24), "g25": dict(guard=0.25),
         "g0": dict(guard=0.0), "2x75 lazy": dict(H=2, fmax=0.75, lazy=1), "3x50 lazy": dict(H=3, fmax=0.5, lazy=1)}


def _sub(P, idx, start):
    i0 = int(np.searchsorted(idx.values, np.datetime64(start)))
    return {f: v.iloc[i0:] for f, v in P.items()}, idx[i0:]


def robust(spec):
    syms, idx, P = M.market(False, end=M.T_T0)
    out = {}
    for tag, over, start in PERT:
        kw = dict(spec)
        if over == "bot":
            kw.update(mtp_now=True, cd=int(kw.get("cd", 3)) + 1)
        elif over == "botpen":
            kw.update(mtp_now=True, cd=int(kw.get("cd", 3)) + 1, pen=0.002, pens=0.002)
        else:
            kw.update(over)
        PP, ii = (P, idx) if start is None else _sub(P, idx, start)
        e, lg = M.lrun(PP, ii, **kw)
        r = M.stats(e, lg, ii, M.T_D0 if start is None else start, M.T_T0)
        for k in ("mo", "wr", "dd", "wt", "tpm", "hold", "hold_cw", "hold_max", "ts", "mlx", "stuck30", "n"):
            out[f"{tag}_{k}"] = r[k]
        if tag == "P0":
            out["y2022"] = M.period(e, pd.Timestamp("2022-01-01"), pd.Timestamp("2023-01-01"))[0] - 1
    mos = [out[f"{t}_mo"] for t, _, _ in PERT]
    out.update(med_mo=float(np.median(mos)), min_mo=float(np.min(mos)), min_wr=min(out[f"{t}_wr"] for t, _, _ in PERT),
               min_dd=min(out[f"{t}_dd"] for t, _, _ in PERT), min_wt=min(out[f"{t}_wt"] for t, _, _ in PERT))
    return out


def run(G, tag):
    rows = []
    t0 = time.time()
    for i, spec in enumerate(G):
        rows.append({"name": M.name(spec), "spec": repr(spec), **robust(spec)})
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(G)} designs, {time.time() - t0:.0f}s", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(M.OUT, f"grid{tag}.csv"), index=False)
    return R


def rank(R, wr=0.97, dd=-0.65):
    Q = R[(R.min_wr >= wr) & (R.min_dd >= dd)].copy()
    Q = Q.sort_values(["med_mo", "min_wr"], ascending=False)
    return Q


def show(R, title, n=20):
    print(f"\n{title}")
    print(f"{'design':66s} {'med%/mo':>7s} {'min':>6s} {'P0':>6s} {'P4':>6s} {'minWR':>6s} {'P0WR':>5s} {'drop':>5s} "
          f"{'wt%':>4s} {'tr/mo':>5s} {'hold':>5s} {'hcw':>5s} {'hmax':>4s} {'ts':>3s} {'2022':>5s}")
    for _, r in R.head(n).iterrows():
        print(f"{r['name'][:66]:66s} {r.med_mo*100:+7.2f} {r.min_mo*100:+6.2f} {r.P0_mo*100:+6.2f} {r.P4_mo*100:+6.2f} "
              f"{r.min_wr*100:6.1f} {r.P0_wr*100:5.1f} {r.min_dd*100:5.0f} {r.min_wt*100:+4.0f} {r.P0_tpm:5.1f} "
              f"{r.P0_hold:5.1f} {r.P0_hold_cw:5.1f} {r.P0_hold_max:4.0f} {int(r.P0_ts):3d} {r.y2022*100:+5.0f}")


def pick(R):
    for wr, dd in ((0.97, -0.65), (0.96, -0.70)):
        Q = rank(R, wr, dd)
        if len(Q):
            best = Q.iloc[0]
            near = Q[Q.med_mo >= best.med_mo - 0.003]
            return near.sort_values("min_wr", ascending=False).iloc[0], (wr, dd)
    return None, None


if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else "2"
    if st == "2":
        R = run(stage2(), "2")
    else:
        R2 = pd.read_csv(os.path.join(M.OUT, "grid2.csv"))
        top = rank(R2, 0.96, -0.70).head(8)
        G = []
        for s in top.spec:
            base = eval(s)
            for o in OPTS3.values():
                G.append({**base, **o})
        R3 = run(G, "3")
        R = pd.concat([R2, R3], ignore_index=True)
        show(rank(R3, 0.0, -1.0), "STAGE 3 alone, all 56 by median %/month", 56)
    show(rank(R, 0.0, -1.0), f"STAGE {st}: top 20 by median %/month (no filter)")
    show(rank(R, 0.97, -0.65), f"STAGE {st}: top 20 passing min WR >= 97%, worst drop >= -65% (P0-P5)")
    show(rank(R, 0.98, -0.65), f"STAGE {st}: top 10 passing min WR >= 98%, worst drop >= -65%", 10)
    show(rank(R, 0.97, -0.50), f"STAGE {st}: top 10 passing min WR >= 97%, worst drop >= -50%", 10)
    p, rule = pick(R)
    if p is not None:
        print(f"\nRULE PICK ({'min WR >= %.0f%%, drop >= %.0f%%' % (rule[0] * 100, rule[1] * 100)}): {p['name']}\n  spec {p['spec']}")
        print("  per perturbation (2021-23): " + " | ".join(
            f"{t} {p[f'{t}_mo']*100:+.2f}%/mo WR {p[f'{t}_wr']*100:.1f} dd {p[f'{t}_dd']*100:.0f}" for t, _, _ in PERT))
