"""Lever "stack": extra checks on the three designs that matter (printed after x10_stack_grid.py / _ceiling.py ran;
they change no pick):
  HONEST PICK   N2 x 50%, cd 6, ladder 50% at 12% / 50% at 18%, +5%, tpf .2 on market buys (x10_stack_grid.py)
  GRID HINDSIGHT N2 x 50%, cd 6, no resting take-profit from the 3rd check, +3%, tpf .2 market buys
  CEILING (hindsight, tuned on 2024-26) N1 x 100%, cd 24, +3% from the 3rd hourly check, tpf .2 market buys
For each: the stresses one at a time (LUNA/FTT/SRM only; fills 0.2% worse only), the real bot's timing
(mtp_now + cooldown one hour longer), months up / worst month in 2024-26, and the honest pick's one-step grid
neighbours. Log: $PB_OUT/report.log
"""
import json

import numpy as np
import pandas as pd

import x10_stack_lib as X
import x10_stack_grid as G

DESIGNS = {
    "HONEST PICK": dict(depth=0.12, N=2, fmax=0.5, tp=0.05, hm=48, cd=6, w=0.5, d2=0.18, tpf=0.2, tpfm=1),
    "GRID HINDSIGHT": dict(depth=0.12, N=2, fmax=0.5, tp=0.03, hm=48, cd=6, tprest=False, tpwait=3, tpf=0.2, tpfm=1),
    "CEILING (hindsight)": dict(depth=0.12, N=1, fmax=1.0, tp=0.03, hm=48, cd=24, tpwait=3, tpf=0.2, tpfm=1),
}


def run(s, dead, pen, pens=None, **over):
    syms, idx, P = X.market(dead)
    kw = X.spec_kw(P, s)
    kw.update(over)
    e, lg = X.srun(P, idx, slip="auto", pen=pen, pens=pen if pens is None else pens, **kw)
    return X.summarize(e, lg, idx), e


def ln(lab, r):
    return (f"  {lab:34s} 21-23 {r['d_mo']*100:+6.2f}%/mo DD {r['d_dd']*100:6.1f}% | 24-26 {r['t_mo']*100:+6.2f}%/mo DD "
            f"{r['t_dd']*100:5.1f}% | 25-26 {r['s_mo']*100:+6.2f}%/mo DD {r['s_dd']*100:5.1f}% | worst trade "
            f"{r['worst_acct']*100:+6.1f}% of acct")


if __name__ == "__main__":
    for dn, s in DESIGNS.items():
        print(f"\n######## {dn}: {s}")
        r, e = run(s, False, 0.0)
        print(ln("normal", r))
        print(ln("+LUNA/FTT/SRM only", run(s, True, 0.0)[0]))
        print(ln("fills 0.2% worse (buys+sells) only", run(s, False, 0.002)[0]))
        print(ln("fills 0.2% worse (buys only)", run(s, False, 0.002, pens=0.0)[0]))
        print(ln("both (report 4)", run(s, True, 0.002)[0]))
        print(ln("bot timing: mtp_now, cd+1", run(s, False, 0.0, mtp_now=True, cd=s["cd"] + 1)[0]))
        m = e[e.index >= X.T_TEST0].resample("ME").last()
        prev = np.concatenate([[e[e.index < X.T_TEST0].iloc[-1]], m.values[:-1]])
        mr = pd.Series(m.values / prev - 1, m.index)
        print(f"  2024-26 months: {len(mr)}, up {int((mr > 0).sum())}, worst {mr.min()*100:+.1f}% ({mr.idxmin():%Y-%m}), "
              f"best {mr.max()*100:+.1f}%, median {mr.median()*100:+.2f}%, months >= +10%: {int((mr >= 0.10).sum())}")
        y = {yy: r[f"y{yy}"] for yy in range(2021, 2027)}
        print("  per year: " + " ".join(f"{k} {v*100:+.0f}%" for k, v in y.items()))

    # one-step neighbours of the honest pick inside the registered grid
    T = pd.read_csv(f"{X.OUT}/grid.csv")
    pick = "N2 cap50 cd6 lad 50/18 rest tp5 tpf mkt.20 none"
    parts = pick.split()
    alts = {1: ["cap50", "cap33", "cap25"], 2: ["cd2", "cd6"], 4: ["off", "50/18", "67/18"], 5: ["rest", "norest3"],
            6: ["tp3", "tp5"], 8: ["none", "mkt.20"], 9: ["none", "btc5"]}
    nb = []
    for i, vals in alts.items():
        for v in vals:
            if v != parts[i]:
                q = list(parts); q[i] = v
                if i == 1:
                    q[0] = {"cap50": "N2", "cap33": "N3", "cap25": "N4"}[v]
                nb.append(" ".join(q))
    R = T[T.name.isin(nb)]
    print(f"\n######## HONEST PICK's one-step grid neighbours ({len(R)})")
    for _, r in R.iterrows():
        print(f"  {r['name']:52s} 21-23 {r.d_mo*100:+6.2f}%/mo DD {r.d_dd*100:5.1f}% | stressed 21-23 {r.S_d_mo*100:+6.2f}%/mo "
              f"DD {r.S_d_dd*100:5.1f}% | 24-26 {r.t_mo*100:+6.2f}%/mo DD {r.t_dd*100:5.1f}% | 25-26 {r.s_mo*100:+6.2f}%/mo")
    print(f"  neighbours 2024-26 %/month: median {R.t_mo.median()*100:.2f}, min {R.t_mo.min()*100:.2f}, max {R.t_mo.max()*100:.2f}")
