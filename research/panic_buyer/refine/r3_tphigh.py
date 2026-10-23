"""Round 3, task "tphigh": a take-profit set part of the way back to the reference 24h high.

Rule tested: sell at entry + f x (ref - entry), where ref = the highest close of the 24 closed hourly candles before
the entry hour (the number the trigger is computed from) and entry = the actual fill price; optionally never below
the live +1.5% (floor). Everything else is the live bot: the sell is placed at the next hourly cycle (fills at that
open if already above), 48h time stop, 24h cooldown, same costs and fill rules.

Motivation, 2021-23 ONLY (r3_tphigh_motiv.py, panel cut at 2024-01-01; log r3_tphigh_motiv.log):
  * 1,681 live trades closed in 2021-23. 1,506 are limit fills at exactly 12% under ref, i.e. 13.64% below ref as a
    fraction of entry, so for them the new rule is just a fixed take-profit of f x 13.64% (f = 0.11 is +1.5%).
    The two rules differ only on the 175 market buys (open already under the trigger): median 18.7% under ref
    (p25 16.0%, p75 24.8%, p90 37.3%).
  * Deep entries bounce back in proportion to the drop: the share of trades whose 48h high reaches
    entry + 0.10 x (ref - entry) is 95% for limit fills, 86-100% in every deeper bucket, while a fixed +3% is reached
    by 86% of limit fills but 93-100% of entries more than 20% under ref.
  * Standalone replay of each 2021-23 trade (no cash/cooldown interaction): live +1.5% avg +0.98%/trade, 6.4h held,
    15.35 bp per hour held. Market buys: +1.21% live vs +1.71% with f = 0.10 and a 1.5% floor (7.1h -> 7.5h), and
    +1.48% with f = 0.08 floored. By year the floored f = 0.10 gains on market buys in all of 2021-23
    (2021 +2.05% -> +2.41%, 2022 -0.46% -> +0.45%, 2023 -1.13% -> -0.98%).
    Unfloored: f = 0.08 / 0.10 cut the limit-fill target to 1.09% / 1.36% and give the highest return per hour held
    (17.41 / 17.27 bp); f = 0.15 / 0.20 / 0.25 earn more per trade (+1.10% / +1.31% / +1.50%) but hold far longer
    (8.7 / 11.8 / 14.5h, 12.60 / 11.12 / 10.34 bp per hour), and in 2022 their limit fills earn less than live.
  So two families: (A) the pure rule f x (ref - entry) without a floor, f from 0.08 to 0.25 (limit-fill target
  1.09% to 3.41%); (B) the live +1.5% for limit fills with more for deep market buys: max(+1.5%, f x gap) with
  f = 0.06, 0.08, 0.10 (the floor binds only for f < 0.11, so floored f >= 0.11 would duplicate family A).

Engine: tkernel is engine.gkernel copied with one extra per-entry input tpf[t, k]. At a fill the take-profit price is
  tpp = max(fill x (1 + tp[t, k]), fill + tpf[t, k] x (ref[t, k] - fill))  (the second term only when tpf > 0),
so tp acts as the floor; tp = -1 means no floor. With tpf = 0 it is gkernel line for line (proved below:
identical equity curves and trade logs to engine.run). The target uses only the fill price and ref at the entry
hour (closed candles); nothing looks ahead. The protocol is study.study() unchanged, with engine.run pointed at
trun in memory only.
"""
import numpy as np
import pandas as pd
from numba import njit

import engine as E
import study as S


@njit(cache=False)
def tkernel(o, h, l, c, ref, depth, nord, fmax, smult, tpa, part, tp2, trail, hm, cd, eps, fee, slip, pen,
            tpwait, tprest, mtp_now, tpf):
    """engine.gkernel plus tpf: take-profit at max(fill*(1+tpa[t,k]), fill + tpf[t,k]*(ref[t,k]-fill))."""
    n, K = c.shape
    units = np.zeros(K); entry = np.zeros(K); amt0 = np.zeros(K); got = np.zeros(K); tpp = np.zeros(K)
    stage = np.zeros(K, np.int64); sbar = np.zeros(K, np.int64); peak = np.zeros(K)
    ebar = np.full(K, -1, np.int64); okbar = np.zeros(K, np.int64); mkt = np.zeros(K, np.int64)
    lastc = np.full(K, np.nan); dist = np.empty(K)
    cash = 1.0; eq = np.empty(n); eq_prev = 1.0
    maxlog = 600000; lg = np.zeros((maxlog, 7)); nl = 0
    for t in range(n):
        for k in range(K):
            dist[k] = 1e9
            if units[k] > 0.0 or t < okbar[k] or np.isnan(o[t, k]) or np.isnan(ref[t, k]) or smult[t, k] <= 0.0:
                continue
            dist[k] = o[t, k] / (ref[t, k] * (1.0 - depth[t, k])) - 1.0
        order = np.argsort(dist)
        nsel = 0
        for q in range(K):
            if dist[order[q]] < 1e8 and nsel < nord[t]:
                nsel += 1
        if nsel > 0 and cash > 1e-9:
            per = min(fmax[t] * eq_prev, cash / nsel)
            for q in range(nsel):
                k = order[q]
                lim = ref[t, k] * (1.0 - depth[t, k])
                fill = -1.0; ismkt = 0
                if o[t, k] <= lim:
                    fill = o[t, k] * (1.0 + slip[k]) * (1.0 + pen); ismkt = 1
                elif l[t, k] <= lim * (1.0 - eps):
                    fill = lim * (1.0 + pen)
                if fill > 0.0 and cash > 1e-9:
                    mkt[k] = ismkt if mtp_now else 0
                    a = min(per * smult[t, k], cash)
                    cash -= a; units[k] = a * (1.0 - fee) / fill
                    entry[k] = fill; amt0[k] = a; ebar[k] = t; got[k] = 0.0
                    tpp[k] = fill * (1.0 + tpa[t, k]); stage[k] = 0; peak[k] = fill
                    if tpf[t, k] > 0.0:                                   # NEW: part of the way back to ref
                        alt = fill + tpf[t, k] * (ref[t, k] - fill)
                        if alt > tpp[k]:
                            tpp[k] = alt
        for k in range(K):
            if units[k] > 0.0 and (ebar[k] < t or mkt[k] == 1) and not np.isnan(o[t, k]):
                px = -1.0; why = 0.0
                if stage[k] == 0:
                    if ebar[k] == t:                     # market buy, take-profit placed at once (mtp_now)
                        if h[t, k] >= tpp[k] * (1.0 + eps):
                            px = tpp[k]
                    elif t - ebar[k] < tpwait:
                        pass
                    elif o[t, k] >= tpp[k]:
                        px = o[t, k]
                    elif tprest and h[t, k] >= tpp[k] * (1.0 + eps):
                        px = tpp[k]
                    if px > 0.0 and part < 1.0:
                        q = units[k] * part
                        p1 = q * px * (1.0 - fee)
                        cash += p1; got[k] += p1; units[k] -= q
                        stage[k] = 1; sbar[k] = t; peak[k] = max(px, h[t, k])
                        px = -1.0
                elif sbar[k] < t:
                    if tp2 > 0.0:
                        p2 = entry[k] * (1.0 + tp2)
                        if o[t, k] >= p2:
                            px = o[t, k]; why = 3.0
                        elif h[t, k] >= p2 * (1.0 + eps):
                            px = p2; why = 3.0
                    if px < 0.0 and trail > 0.0:
                        if h[t, k] > peak[k]:
                            peak[k] = h[t, k]
                        if c[t, k] <= peak[k] * (1.0 - trail):
                            px = c[t, k] * (1.0 - slip[k]); why = 2.0
                if px < 0.0 and t - ebar[k] >= hm[t] - mkt[k]:
                    px = c[t, k] * (1.0 - slip[k]); why = 1.0
                if px > 0.0:
                    proceeds = units[k] * px * (1.0 - fee); cash += proceeds; got[k] += proceeds
                    if nl < maxlog:
                        lg[nl, 0] = ebar[k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = why
                        lg[nl, 4] = got[k] / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]; nl += 1
                    units[k] = 0.0; okbar[k] = t + cd
        v = cash
        for k in range(K):
            if not np.isnan(c[t, k]):
                lastc[k] = c[t, k]
            if units[k] > 0.0:
                v += units[k] * lastc[k]
        eq[t] = v; eq_prev = v
    return eq, lg[:nl].copy()


_orig_run = E.run
L = E.LIVE


def trun(P, idx, depth=L["depth"], N=L["N"], fmax=L["fmax"], smult=1.0, tp=L["tp"], part=1.0, tp2=0.0, trail=0.0,
         hm=L["hm"], cd=L["cd"], eps=E.EPS, fee=E.FEE, slip=E.SLIP_MAJOR, pen=0.0, tpwait=1, tprest=True,
         mtp_now=False, tpf=0.0):
    """engine.run plus tpf (scalar, (n,) or (n,K)): take-profit = max(entry*(1+tp), entry + tpf*(ref - entry)).
    tp = -1 means no floor. Same arguments and defaults as engine.run otherwise; any coin list, any dates."""
    O = P["o"].values; n, K = O.shape
    sl = np.full(K, slip) if np.ndim(slip) == 0 else np.asarray(slip, np.float64)
    eq, lg = tkernel(O, P["h"].values, P["l"].values, P["cf"].values, P["ref"].values,
                     E.full(n, K, depth), E.full(n, 0, N, np.int64), E.full(n, 0, fmax), E.full(n, K, smult),
                     E.full(n, K, tp), float(part), float(tp2), float(trail), E.full(n, 0, hm, np.int64), int(cd), eps,
                     fee, sl, pen, int(tpwait), bool(tprest), bool(mtp_now), E.full(n, K, tpf))
    return pd.Series(eq, idx), pd.DataFrame(lg, columns=E.LOG_COLS)


# ------------------------------------------------------------------ the grid (fixed before any result is seen)
GRID = {}
for f in (0.08, 0.10, 0.15, 0.20, 0.25):                  # A: pure rule, no floor (limit-fill target f x 13.64%)
    GRID[f"tp = {f:.2f} x (ref - entry), no floor"] = {"tp": -1.0, "tpf": f}
for f in (0.06, 0.08, 0.10):                              # B: live +1.5% floor, more for deep market buys
    GRID[f"tp = max(1.5%, {f:.2f} x (ref - entry))"] = {"tp": 0.015, "tpf": f}
assert len(GRID) <= 8


def prove_equivalence():
    """trun with tpf = 0 must give exactly engine.run's equity curve and trades. Prints only booleans and the
    live settings' own (already published) numbers."""
    ok = True
    for dead, pen, mtp in ((False, 0.0, False), (True, 0.0, False), (False, 0.002, False), (False, 0.0, True)):
        syms, idx, P = S.market(dead)
        e0, l0 = _orig_run(P, idx, pen=pen, mtp_now=mtp)
        e1, l1 = trun(P, idx, pen=pen, mtp_now=mtp)
        same = np.array_equal(e0.values, e1.values) and l0.equals(l1)
        ok &= same
        r = E.score(e1, l1, idx)
        print(E.line(f"trun tpf=0, dead={dead} pen={pen} mtp={mtp}", r) + f"  identical to engine.run: {same}")
    # cross-check against engine.run's own per-entry tp array (pen = 0, so the fill price is known in advance:
    # market buy o*(1+slip), limit fill lim): tp[t,k] = max(floor, f*(ref/fill - 1)) must give the same target.
    syms, idx, P = S.market(False)
    O, R = P["o"].values, P["ref"].values
    lim = R * (1 - L["depth"])
    xfill = np.where(O <= lim, O * (1 + E.SLIP_MAJOR), lim)
    for name, spec in GRID.items():
        tpa = np.maximum(spec["tp"], spec["tpf"] * (R / xfill - 1))
        tpa = np.where(np.isfinite(tpa), tpa, L["tp"])
        e0, l0 = _orig_run(P, idx, tp=tpa)
        e1, l1 = trun(P, idx, **spec)
        rel = float(np.nanmax(np.abs(e1.values / e0.values - 1)))
        same_trades = len(l0) == len(l1) and np.array_equal(l0[["eb", "xb", "k", "why"]].values,
                                                             l1[["eb", "xb", "k", "why"]].values)
        print(f"  {name:40s} vs engine.run with a per-entry tp array: same trades {same_trades}, "
              f"max equity rel. diff {rel:.1e}")
        ok &= same_trades and rel < 1e-9
    # the target really is entry + f x (ref - entry): implied sell price of take-profits that filled at the limit
    spec = GRID["tp = 0.20 x (ref - entry), no floor"]
    e1, l1 = trun(P, idx, **spec)
    tg = l1[l1.why == 0]
    ref_e = R[tg.eb.astype(int).values, tg.k.astype(int).values]
    want = tg.entry.values + spec["tpf"] * (ref_e - tg.entry.values)
    px = (tg.ret.values + 1) / (1 - E.FEE) ** 2 * tg.entry.values
    at_lim = np.abs(px / want - 1) < 1e-9
    print(f"  f=0.20 take-profits: {at_lim.mean() * 100:.0f}% sold exactly at entry + 0.2 x (ref - entry), "
          f"the rest above it (open gaps): min implied/target - 1 = {np.min(px / want - 1):.1e}")
    ok &= bool(np.min(px / want - 1) > -1e-9)
    if not ok:
        raise SystemExit("tkernel does not reproduce engine.run - stop")


def anatomy(spec, P, idx, label):
    e, lg = trun(P, idx, **spec)
    per = np.where(idx[lg.xb.astype(int).values] < E.DESIGN_END, "21-23", "24-26")
    R, O = P["ref"].values, P["o"].values
    ebi, ki = lg.eb.astype(int).values, lg.k.astype(int).values
    mkt = O[ebi, ki] <= R[ebi, ki] * (1 - L["depth"])
    for p in ("21-23", "24-26"):
        m = per == p
        g = lg[m]
        s = (f"  {label:40s} {p}: trades {len(g)} (market buys {int(mkt[m].sum())}), TP {int((g.why == 0).sum())} "
             f"({(g.ret[g.why == 0] * g.amt[g.why == 0]).sum():+.2f}, avg {g.ret[g.why == 0].mean() * 100:+.2f}%), "
             f"time stops {int((g.why == 1).sum())} ({(g.ret[g.why == 1] * g.amt[g.why == 1]).sum():+.2f}, "
             f"avg {g.ret[g.why == 1].mean() * 100:+.1f}%), avg held {(g.xb - g.eb).mean():.1f}h, "
             f"market-buy avg ret {g.ret[mkt[m]].mean() * 100:+.2f}%")
        print(s)


if __name__ == "__main__":
    prove_equivalence()
    E.run = trun        # in memory only: study.study() now runs every variant (and the baseline) through trun
    R1, v1 = S.study("r3 tphigh: take-profit part of the way back to the reference 24h high", GRID, "r3_tphigh")
    S.save_verdict(v1)
    print("\nExit anatomy (printed after the grid ran; sums are return x order size as a fraction of equity):")
    syms, idx, P = S.market(False)
    anatomy({}, P, idx, "LIVE SETTINGS")
    if v1["pick"]:
        anatomy(GRID[v1["pick"]], P, idx, v1["pick"])
