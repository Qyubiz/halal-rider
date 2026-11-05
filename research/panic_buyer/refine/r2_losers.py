"""Round 2, lens "losers": cut the 48h time-stop losers without killing the take-profit winners.

Motivation (2021-23 only, r2_losers_motiv.py; trades that entered and exited before 2024):
  * 1,681 trades; 108 time stops at -11.5% on average take back about half of the take-profit gains.
  * A hard stop-loss from entry does not help: at every level from -5% to -20% (hourly close, market sell) the
    stopped trades would on net have done better if held (crude sums -0.71 to -1.88 x order size), because
    positions that are deep under water are in a fresh panic and bounce (that is the bot's own thesis), and
    15% of the take-profit winners closed 3%+ under water at some hour first. So no hard stop-loss grid.
  * H1 break-even: of the trades still open 6-24 hours after the fill, many trade back up to the entry price
    but then fall into the time stop. Selling at entry +0.2% (about break-even after fees) or +0.5% once the
    position is X hours old: crude net +0.05 (X=24) to +0.54 (X=6), positive for all X tested.
  * H2 stale loser: at the close 18-30 hours after the fill, positions 3-10% under water mostly kept sliding
    (later take-profit only 18-21% at h24, 24-31% at h18; selling then instead of waiting: +0.76 at h24,
    +0.41 at h18),
    while positions more than 10% under water bounced (selling them: -0.32 at h24, -0.12 at h18).

Both need new exit logic, so lkernel below is engine.gkernel copied with two extra exits (all off by default):
  beh/bev  break-even target: from the hourly cycle after the close of the beh-th hour after the fill hour,
           the take-profit limit becomes entry*(1+bev) (if lower than the normal one). Same fill rules as the
           take-profit: an open already at/above the target sells at the open, else the high must trade eps through.
  sx/su/sd stale-loser exit: once, at the close of the sx-th hour after the fill hour (the first hour with a
           candle from then on), sell at market (close x (1-slip), like the time stop) if
           entry*(1-sd) < close <= entry*(1-su). sd >= 1 means no lower bound.
Everything is decided at hourly cycles from closed candles; nothing looks ahead.

The protocol is study.study() itself, unchanged: engine.run is pointed at lrun (in memory only) after proving
that lrun with the new exits off gives exactly the same equity curves as engine.run.
"""
import numpy as np
import pandas as pd
from numba import njit

import engine as E
import study as S


@njit(cache=False)
def lkernel(o, h, l, c, ref, depth, nord, fmax, smult, tpa, part, tp2, trail, hm, cd, eps, fee, slip, pen,
            tpwait, tprest, beh, bev, sx, su, sd):
    """engine.gkernel plus the break-even target (beh, bev) and the stale-loser exit (sx, su, sd).
    why codes in the log: 0 take-profit, 1 time stop, 2 trail, 3 tp2, 4 stale-loser exit, 5 break-even target."""
    n, K = c.shape
    units = np.zeros(K); entry = np.zeros(K); amt0 = np.zeros(K); got = np.zeros(K); tpp = np.zeros(K)
    stage = np.zeros(K, np.int64); sbar = np.zeros(K, np.int64); peak = np.zeros(K)
    ebar = np.full(K, -1, np.int64); okbar = np.zeros(K, np.int64); chk = np.zeros(K, np.int64)
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
                fill = -1.0
                if o[t, k] <= lim:
                    fill = o[t, k] * (1.0 + slip[k]) * (1.0 + pen)
                elif l[t, k] <= lim * (1.0 - eps):
                    fill = lim * (1.0 + pen)
                if fill > 0.0 and cash > 1e-9:
                    a = min(per * smult[t, k], cash)
                    cash -= a; units[k] = a * (1.0 - fee) / fill
                    entry[k] = fill; amt0[k] = a; ebar[k] = t; got[k] = 0.0
                    tpp[k] = fill * (1.0 + tpa[t, k]); stage[k] = 0; peak[k] = fill; chk[k] = 0
        for k in range(K):
            if units[k] > 0.0 and ebar[k] < t and not np.isnan(o[t, k]):
                px = -1.0; why = 0.0
                if stage[k] == 0:
                    tg = tpp[k]
                    if beh >= 0 and t - ebar[k] > beh and entry[k] * (1.0 + bev) < tg:
                        tg = entry[k] * (1.0 + bev); why = 5.0
                    if t - ebar[k] < tpwait:
                        pass
                    elif o[t, k] >= tg:
                        px = o[t, k]
                    elif tprest and h[t, k] >= tg * (1.0 + eps):
                        px = tg
                    if px < 0.0:
                        why = 0.0
                    if px > 0.0 and part < 1.0:
                        q = units[k] * part
                        p1 = q * px * (1.0 - fee)
                        cash += p1; got[k] += p1; units[k] -= q
                        stage[k] = 1; sbar[k] = t; peak[k] = max(px, h[t, k])
                        px = -1.0; why = 0.0
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
                if px < 0.0 and t - ebar[k] >= hm[t]:
                    px = c[t, k] * (1.0 - slip[k]); why = 1.0
                if px < 0.0 and sx >= 0 and chk[k] == 0 and t - ebar[k] >= sx:
                    chk[k] = 1
                    r = c[t, k] / entry[k] - 1.0
                    if r <= -su and (sd >= 1.0 or r > -sd):
                        px = c[t, k] * (1.0 - slip[k]); why = 4.0
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


def lrun(P, idx, depth=L["depth"], N=L["N"], fmax=L["fmax"], smult=1.0, tp=L["tp"], part=1.0, tp2=0.0, trail=0.0,
         hm=L["hm"], cd=L["cd"], eps=E.EPS, fee=E.FEE, slip=E.SLIP_MAJOR, pen=0.0, tpwait=1, tprest=True,
         beh=-1, bev=0.002, sx=-1, su=0.0, sd=9.0):
    """engine.run with the two new exits; same arguments and defaults otherwise."""
    O = P["o"].values; n, K = O.shape
    sl = np.full(K, slip) if np.ndim(slip) == 0 else np.asarray(slip, np.float64)
    eq, lg = lkernel(O, P["h"].values, P["l"].values, P["cf"].values, P["ref"].values,
                     E.full(n, K, depth), E.full(n, 0, N, np.int64), E.full(n, 0, fmax), E.full(n, K, smult),
                     E.full(n, K, tp), float(part), float(tp2), float(trail), E.full(n, 0, hm, np.int64), int(cd), eps,
                     fee, sl, pen, int(tpwait), bool(tprest), int(beh), float(bev), int(sx), float(su), float(sd))
    return pd.Series(eq, idx), pd.DataFrame(lg, columns=E.LOG_COLS)


# ------------------------------------------------------------------ the grids (fixed before any result is seen)
H1 = {}
for X in (6, 12, 18, 24):
    for be in (0.002, 0.005):
        H1[f"break-even +{be:.1%} after {X}h"] = {"beh": X, "bev": be}

H2 = {}
for X in (18, 24):
    for u in (0.0, 0.03):
        for dlo in (0.10, 9.0):
            band = f"{u:.0%}..{dlo:.0%} under" if dlo < 1 else f">={u:.0%} under"
            H2[f"stale exit at {X}h if {band}"] = {"sx": X, "su": u, "sd": dlo}


def prove_equivalence():
    """lrun with the new exits off must give exactly engine.run's equity curve and trades."""
    ok = True
    for dead, pen in ((False, 0.0), (True, 0.0), (False, 0.002)):
        syms, idx, P = S.market(dead)
        e0, l0 = _orig_run(P, idx, pen=pen)
        e1, l1 = lrun(P, idx, pen=pen)
        same = np.array_equal(e0.values, e1.values) and l0.equals(l1)
        ok &= same
        r = E.score(e1, l1, idx)
        print(E.line(f"lrun, exits off, dead={dead} pen={pen}", r) + f"  identical to engine.run: {same}")
    # switches that never trigger must also change nothing
    syms, idx, P = S.market(False)
    e0, _ = _orig_run(P, idx)
    e2, _ = lrun(P, idx, beh=100000, sx=100000)
    print(f"lrun with beh=sx=100000 identical: {np.array_equal(e0.values, e2.values)}")
    ok &= np.array_equal(e0.values, e2.values)
    if not ok:
        raise SystemExit("lkernel does not reproduce engine.run - stop")


if __name__ == "__main__":
    prove_equivalence()
    E.run = lrun        # in memory only: study.study() now runs every variant (and the baseline) through lrun
    R1, v1 = S.study("r2 losers H1: break-even target after X hours", H1, "r2_losers_h1_breakeven")
    S.save_verdict(v1)
    R2, v2 = S.study("r2 losers H2: stale-loser exit at hour X", H2, "r2_losers_h2_stale")
    S.save_verdict(v2)
    # anatomy of each pick (exit reasons, sum of return x order size), printed after the grids have run
    syms, idx, P = S.market(False)
    for v, V in ((v1, H1), (v2, H2)):
        for name, spec in ((("LIVE SETTINGS", {}),) + (((v["pick"], V[v["pick"]]),) if v["pick"] else ())):
            e, lg = lrun(P, idx, **spec)
            per = np.where(idx[lg.xb.astype(int).values] < E.DESIGN_END, "21-23", "24-26")
            for p in ("21-23", "24-26"):
                g = lg[per == p]
                parts = [f"{w}: {int((g.why == c).sum())} ({(g.ret[g.why == c] * g.amt[g.why == c]).sum():+.2f}, "
                         f"avg {g.ret[g.why == c].mean() * 100:+.1f}%)" for c, w in ((0, "TP"), (5, "BE"), (4, "stale"),
                                                                                (1, "time stop")) if (g.why == c).any()]
                print(f"  anatomy {name} {p}: {len(g)} trades | " + " | ".join(parts))
