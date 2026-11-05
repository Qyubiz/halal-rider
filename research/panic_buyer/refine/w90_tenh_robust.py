"""Robustness of the ten-hourly leaders (measurement only, no new picks):
  1. one-step neighbours: depth +-0.5/+-1 pt, take-profit one step up/down on {1.5,2,3,4,5,6,8}%, time stop one step
     on {24,48,72,96}h, guard threshold +-5 pts, the other cooldown schemes -> median / min of worst-of-three
     %/month, 2024-26 %/month and win rates over the neighbours;
  2. the stressed version and its parts (bot timing only; buys+sells 0.2% worse only; LUNA/FTT/SRM only;
     the REFINE.md reference "bot timing + buys 0.2% worse");
  3. fresh starts (all cash) on each of the first 30 days of 2024 (2024-26 %/month) and of 2025 (since-2025);
  4. calendar years, months >= +10%, losing months, worst month; the 5 worst trades as % of the account.
Candidates: C1 d14 tp5 hm48 cdw1 g35 | C2 d15 tp6 hm48 cdw1 g35 | C3 d15 tp4 hm96 cdw1 g35 | C4 d13 tp1.5 hm96 cdw1 g35
| C5 d15 tp6 hm48 cd24 g40 (config only) | C0 10% mode (d14 tp5 hm48 cd24 g35), reference."""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w90_tenh_lib as T  # noqa: E402

out = open(os.path.join(T.OUT, "robust.log"), "w")


def say(s=""):
    print(s, flush=True)
    out.write(s + "\n"); out.flush()


B = dict(N=1, fmax=1.0, gL=168)
CANDS = {
    "C0 10% mode d14 tp5 hm48 cd24 g35": dict(B, depth=0.14, tp=0.05, hm=48, cd=24, g=0.35),
    "C1 d14 tp5 hm48 cd24 re-arm g35": dict(B, depth=0.14, tp=0.05, hm=48, cd=24, cdw=1, g=0.35),
    "C2 d15 tp6 hm48 cd24 re-arm g35": dict(B, depth=0.15, tp=0.06, hm=48, cd=24, cdw=1, g=0.35),
    "C3 d15 tp4 hm96 cd24 re-arm g35": dict(B, depth=0.15, tp=0.04, hm=96, cd=24, cdw=1, g=0.35),
    "C4 d13 tp1.5 hm96 cd24 re-arm g35": dict(B, depth=0.13, tp=0.015, hm=96, cd=24, cdw=1, g=0.35),
    "C5 d15 tp6 hm48 cd24 g40": dict(B, depth=0.15, tp=0.06, hm=48, cd=24, g=0.40),
}
TPS = [0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08]
HMS = [24, 48, 72, 96]
CDS = [dict(cd=24, cdw=-1), dict(cd=24, cdw=1), dict(cd=12, cdw=1), dict(cd=6, cdw=-1)]


def neighbours(s):
    nb = []
    for dd in (-0.01, -0.005, 0.005, 0.01):
        nb.append(dict(s, depth=round(s["depth"] + dd, 4)))
    i = TPS.index(s["tp"])
    for j in (i - 1, i + 1):
        if 0 <= j < len(TPS):
            nb.append(dict(s, tp=TPS[j]))
    i = HMS.index(s["hm"])
    for j in (i - 1, i + 1):
        if 0 <= j < len(HMS):
            nb.append(dict(s, hm=HMS[j]))
    for dg in (-0.05, 0.05):
        nb.append(dict(s, g=round(s["g"] + dg, 3)))
    for c in CDS:
        if not (c["cd"] == s["cd"] and c["cdw"] == s.get("cdw", -1)):
            nb.append(dict(s, **c))
    return nb


def fresh(s, start, end=None):
    """Fresh all-cash start at `start` (features keep their warm-up): %/month to the end."""
    syms, idx, P = T.market(False)
    P = T.with_ref(P, s.get("refW", 24))
    kw = T.spec_kw(P, s)
    i0 = int(np.searchsorted(idx, start))
    Q = {k: v.iloc[i0:] for k, v in P.items()}
    kw = {k: (v[i0:] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(idx) else v) for k, v in kw.items()}
    e, lg = T.trun(Q, idx[i0:], **kw)
    return T.period(e, start, end)[1]


for name, s in CANDS.items():
    r, e, lg = T.evaluate(s)
    rs, es, _ = T.evaluate(s, stress=True)
    say("=" * 140)
    say(T.fmt(name, r, rs))
    parts = {"bot timing only": dict(mtp_now=True, cd=s["cd"] + 1, **({"cdw": s["cdw"] + 1} if s.get("cdw", -1) >= 0 else {})),
             "buys+sells 0.2% worse only": dict(pen=0.002, pens=0.002),
             "REFINE ref: bot timing + buys 0.2% worse": dict(mtp_now=True, cd=s["cd"] + 1, pen=0.002,
                                                              **({"cdw": s["cdw"] + 1} if s.get("cdw", -1) >= 0 else {}))}
    for pn, po in parts.items():
        rp, _, _ = T.evaluate(s, **po)
        say(f"   {pn:42s} 2021-23 {rp['d_mo']*100:+6.2f}% dd {rp['d_dd']*100:4.0f}% | 2024-26 {rp['t_mo']*100:+6.2f}% dd "
            f"{rp['t_dd']*100:4.0f}% | since 2025 {rp['s_mo']*100:+6.2f}% | win {rp['d_win']*100:.1f}/{rp['t_win']*100:.1f}/{rp['s_win']*100:.1f}%")
    rd, _, _ = T.evaluate(s, dead=True, liq=1)
    say(f"   {'LUNA/FTT/SRM only (sold at last hour)':42s} 2021-23 {rd['d_mo']*100:+6.2f}% dd {rd['d_dd']*100:4.0f}% | 2022 "
        f"{rd['y2022']*100:+.0f}% | worst trade {rd['d_wacct']*100:+.0f}% of account")
    # neighbours
    NB = []
    for s2 in neighbours(s):
        r2, _, _ = T.evaluate(s2)
        NB.append(r2)
    N = pd.DataFrame(NB)
    say(f"   {len(N)} neighbours: worst-of-three median {N.min_mo.median()*100:+.2f}% min {N.min_mo.min()*100:+.2f}% max "
        f"{N.min_mo.max()*100:+.2f}% | 2024-26 median {N.t_mo.median()*100:+.2f}% (min {N.t_mo.min()*100:+.2f}) | since 2025 "
        f"median {N.s_mo.median()*100:+.2f}% | win (lower of 21-23/24-26) median {N.min_win.median()*100:.1f}% min "
        f"{N.min_win.min()*100:.1f}% | 21-23 dd median {N.d_dd.median()*100:.0f}%")
    # fresh starts
    f24 = [fresh(s, T.T24 + pd.Timedelta(days=d)) for d in range(30)]
    f25 = [fresh(s, T.T25 + pd.Timedelta(days=d)) for d in range(30)]
    say(f"   fresh starts 2024-01-01+0..29d: 2024-26 {min(f24)*100:+.2f}..{max(f24)*100:+.2f}%/mo (median {np.median(f24)*100:+.2f}) | "
        f"2025-01-01+0..29d: since 2025 {min(f25)*100:+.2f}..{max(f25)*100:+.2f}%/mo (median {np.median(f25)*100:+.2f})")
    # years and months
    ys = " ".join(f"{y} {T.year_ret(e, y)*100:+.0f}%" for y in range(2021, 2027))
    mr = e.resample("ME").last().pct_change()
    mr.iloc[0] = e.resample("ME").last().iloc[0] - 1
    m24 = mr[mr.index >= T.T24]
    say(f"   years: {ys} | 2024-26 months: {len(m24)}, >=+10%: {(m24 >= 0.10).sum()}, losing: {(m24 < 0).sum()}, worst "
        f"{m24.min()*100:+.0f}%, median {m24.median()*100:+.1f}%")
    syms = list(T.market(False)[2]["o"].columns)
    idx = T.market(False)[1]
    acct = lg.ret * lg.amt / lg.eq0
    w = lg.assign(acct=acct).sort_values("acct").head(5)
    say("   worst trades (% of account): " + "; ".join(
        f"{syms[int(x.k)][:-4]} {idx[int(x.eb)].strftime('%Y-%m-%d')} {x.acct*100:+.0f}%" for x in w.itertuples()))
out.close()
