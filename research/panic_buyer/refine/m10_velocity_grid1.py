"""Grid 1 (2021-23 ONLY; later periods are neither printed nor saved): the 90% mode plus a second, faster trigger.

Tier A = the 90% mode's trigger (24h high -14%, +2.5%, 96h) or a neighbour; tier B = a flash dip (highest close of the
last 3/4/6/12 hours minus 8..13%) with its own take-profit and time stop. One position per coin: each coin's bid sits
at the higher of its two trigger prices. One slot, all the free money (as the 90% mode). Cooldown after a win 0h
(re-arm at the next hourly check) or 3h; after a loss 3h or 24h. Crash guard 35% / 168h.
Each design is run with normal fills and with every buy and sell fill 0.2% worse (stress "worse").

Fixed rule (written before the run): eligible = 2021-23 win rate >= 92% with normal fills and >= 90% with worse fills,
and worst drop no deeper than -65%; ranked by the 2021-23 %/month with worse fills.
"""
import itertools
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m10_velocity_lib as V  # noqa: E402

TA = [dict(ref=24, depth=0.14, tp=0.025, hm=96), dict(ref=24, depth=0.15, tp=0.025, hm=96)]
TB = [None] + [dict(ref=w, depth=d, tp=tp, hm=hm) for w, d, tp, hm in itertools.product(
    (3, 4, 6, 12), (0.08, 0.10, 0.11, 0.12, 0.13), (0.02, 0.025, 0.03), (48, 96))]
CDS = [(3, 3), (0, 3), (0, 24)]


def dcols(r, pre=""):
    return {pre + k: v for k, v in r.items() if k.startswith("d_") or k == "y2022"}


def run_grid(designs, name):
    syms, idx, P = V.market(False)
    rows = []
    for i, (label, kw) in enumerate(designs):
        e, lg, inv = V.vrun(P, idx, **kw)
        r = V.summarize(e, lg, inv, idx)
        e2, lg2, inv2 = V.vrun(P, idx, **kw, pen=V.STRESS_PEN, pens=V.STRESS_PEN)
        r2 = V.summarize(e2, lg2, inv2, idx)
        rows.append({"label": label, **dcols(r), **dcols(r2, "w_")})
        if i % 50 == 0:
            print(f"{i}/{len(designs)}", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(V.OUT, f"{name}.csv"), index=False)
    return R


def rank(R):
    ok = R[(R.d_wr >= 0.92) & (R.w_d_wr >= 0.90) & (R.d_dd >= -0.65)]
    return ok.sort_values("w_d_mo", ascending=False)


def show(R, n=30):
    cols = ["label", "d_mo", "w_d_mo", "d_wr", "w_d_wr", "d_dd", "w_d_dd", "d_wt", "d_tpm", "d_cashh", "y2022",
            "d_wr0", "d_n0", "d_wr1", "d_n1"]
    cols = [c for c in cols if c in R.columns]
    x = R[cols].head(n).copy()
    for c in cols[1:]:
        if c not in ("d_tpm", "d_n0", "d_n1"):
            x[c] = (x[c] * 100).round(1)
        else:
            x[c] = x[c].round(1)
    print(x.to_string(index=False))


def designs():
    out = []
    for ta, tb, (cdw, cdl) in itertools.product(TA, TB, CDS):
        tiers = [dict(ta, cdw=cdw, cdl=cdl, fs=1.0)]
        if tb is not None:
            tiers.append(dict(tb, cdw=cdw, cdl=cdl, fs=1.0))
        lab = (f"A{ta['depth']*100:.0f}/{ta['tp']*100:g}" +
               (f" B{tb['ref']}h-{tb['depth']*100:.0f}/{tb['tp']*100:g}/{tb['hm']}h" if tb else " noB") + f" cd{cdw}/{cdl}")
        out.append((lab, dict(tiers=tiers, nslots=1, excl=1, dyn=0, guard=0.35)))
    return out


def main():
    D = designs()
    print(f"{len(D)} designs", flush=True)
    R = run_grid(D, "grid1_2021_23")
    print("\nbaselines (no tier B):")
    show(R[R.label.str.contains("noB")])
    print(f"\nRULE: eligible {len(rank(R))} of {len(R)}; top 30 by 2021-23 %/month with worse fills")
    show(rank(R))
    print("\nTop 15 by normal-fill 2021-23 %/month among eligible:")
    show(rank(R).sort_values("d_mo", ascending=False), 15)


if __name__ == "__main__":
    main()
