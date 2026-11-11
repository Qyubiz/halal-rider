"""m17 ceiling: HINDSIGHT upper bounds (perfect foresight on purpose; none of these is tradable).
1 90%-mode entries, perfect exit at highest hourly high x(1-0.1%) before the next baseline entry (also capped at 96h)
2 90%-mode winners only   3 perfect dip buyer (buy 24h dip>=10% at the bar low, sell at highest high of next 96h,
non-overlapping: greedy-by-gain and optimal DP)   4 best single coin buy&hold each calendar month (open->close)
Plus: count of 14% 24h trigger hits per month and avg gain to the 96h high after them. Fees 0.1%/side."""
import bisect
import sys
import numpy as np
import pandas as pd
import m10_velocity_lib as V
E = V.E
FEE, EPS = E.FEE, E.EPS
F2 = (1 - FEE) ** 2


def months(idx, a, b):
    b = idx[-1] + pd.Timedelta(hours=1) if b is None else b
    return (b - a).total_seconds() / 86400 / V.MD


def mo(logsum, m):
    return float(np.exp(logsum / m) - 1)


def dp_best(cands, a_i, b_i):
    """cands: (start,end,gain). Optimal non-overlapping set maximizing sum log1p(gain) (start > previous end)."""
    c = sorted((x for x in cands if x[0] >= a_i and x[1] < b_i and x[2] > 0), key=lambda x: x[1])
    ends = [x[1] for x in c]
    best = [0.0] * (len(c) + 1)
    for i, (s, e, g) in enumerate(c):
        j = bisect.bisect_left(ends, s)          # intervals with end < s
        best[i + 1] = max(best[i], best[j] + np.log1p(g))
    return best[-1], len(c)


def greedy(cands, a_i, b_i):
    c = sorted((x for x in cands if x[0] >= a_i and x[1] < b_i and x[2] > 0), key=lambda x: -x[2])
    taken = []   # sorted starts
    ls = 0.0; n = 0; iv = []
    for s, e, g in c:
        ok = all(e < s2 or s > e2 for s2, e2 in iv)
        if ok:
            iv.append((s, e)); ls += np.log1p(g); n += 1
    return ls, n


def run_period(tag, syms, idx, P, periods, holdout):
    O, H, L = P["o"].values, P["h"].values, P["l"].values
    C = P["cf"].values; ref = P["ref"].values
    n, K = O.shape
    ev, lg, inv = V.vrun(P, idx, **V.MODE90)
    lg = lg.copy(); lg["eb"] = lg.eb.astype(int); lg["xb"] = lg.xb.astype(int); lg["k"] = lg.k.astype(int)
    lg = lg.sort_values("eb").reset_index(drop=True)
    # bound 1
    nxt = np.append(lg.eb.values[1:], n)
    r1 = []; r1c = []
    for i, r in lg.iterrows():
        k, eb = int(r.k), int(r.eb)
        w_end = nxt[i]
        for lst, end in ((r1, w_end), (r1c, min(w_end, eb + 96))):
            seg = H[eb + 1:max(end, eb + 2), k]
            seg = seg[~np.isnan(seg)]
            px = seg.max() * (1 - EPS) if len(seg) else 0
            lst.append(max(F2 * px / r.entry - 1, r.ret))
    lg["r1"], lg["r1c"] = r1, r1c
    # dip candidates
    out = {}
    cand = {}
    for thr in (0.10, 0.14):
        cl = []
        W = 96
        for k in range(K):
            hk = pd.Series(H[:, k]).rolling(W, min_periods=1).max().shift(-W).values   # max h[t+1..t+W]
            for t in np.flatnonzero(L[:, k] <= ref[:, k] * (1 - thr)):
                if t + W >= n or np.isnan(hk[t]):
                    continue
                # argmax time of high in t+1..t+W
                j = t + 1 + int(np.nanargmax(H[t + 1:t + W + 1, k]))
                cl.append((t, j, F2 * hk[t] * (1 - EPS) / L[t, k] - 1))
        cand[thr] = cl
    # 14% trigger hits (episode starts) and gross gain to 96h high from the limit price
    k14 = []
    for k in range(K):
        hit = L[:, k] <= ref[:, k] * 0.86
        hit[np.isnan(ref[:, k])] = False
        st = np.flatnonzero(hit & ~np.roll(hit, 1))
        for t in st:
            if t + 96 >= n:
                continue
            lim = ref[t, k] * 0.86
            buy = min(O[t, k], lim)
            k14.append((t, np.nanmax(H[t + 1:t + 97, k]) / buy - 1))
    k14 = np.array(k14)
    res = []
    for name, a, b in periods:
        m = months(idx, a, b)
        ai = int(np.searchsorted(idx.values, np.datetime64(a)))
        bi = n if b is None else int(np.searchsorted(idx.values, np.datetime64(b)))
        s = lg[(lg.eb >= ai) & (lg.eb < bi)]
        row = dict(period=name, months=m, trades90=len(s), base_mo=mo(np.log1p(s.ret).sum(), m))
        row["b1_mo"] = mo(np.log1p(s.r1).sum(), m)
        row["b1_cap96_mo"] = mo(np.log1p(s.r1c).sum(), m)
        row["b2_mo"] = mo(np.log1p(s.ret[s.ret > 0]).sum(), m)
        for thr in (0.10, 0.14):
            ls, nn = greedy(cand[thr], ai, bi); row[f"b3g{int(thr*100)}_mo"] = mo(ls, m); row[f"b3g{int(thr*100)}_n"] = nn / m
            ls, nn = dp_best(cand[thr], ai, bi); row[f"b3d{int(thr*100)}_mo"] = mo(ls, m); row[f"b3d{int(thr*100)}_n"] = nn / m
        # bound 4
        ls4 = 0.0; nm = 0; picks = []
        cur = pd.Timestamp(a).to_period("M")
        endp = (idx[bi - 1]).to_period("M")
        while cur <= endp:
            ms = max(cur.start_time, a); me = cur.end_time
            i0 = int(np.searchsorted(idx.values, np.datetime64(ms)))
            i1 = int(np.searchsorted(idx.values, np.datetime64(cur.start_time + pd.offsets.MonthBegin(1))))
            i1 = min(i1, bi)
            if i1 - i0 > 600:
                o0 = O[i0]; c1 = C[i1 - 1]
                g = F2 * c1 / o0 - 1
                g = np.where(np.isnan(g), -1, g)
                picks.append(g.max()); ls4 += np.log1p(g.max()); nm += 1
            cur += 1
        row["b4_mo"] = mo(ls4, m) if nm else np.nan
        row["b4_months"] = nm; row["b4_median_month"] = float(np.median(picks))
        sel = (k14[:, 0] >= ai) & (k14[:, 0] < bi) if len(k14) else np.array([])
        row["hits14_per_mo"] = float(sel.sum() / m); row["avg_gain96_14"] = float(k14[sel, 1].mean()) if sel.any() else np.nan
        row["med_gain96_14"] = float(np.median(k14[sel, 1])) if sel.any() else np.nan
        res.append(row)
    # reproduction of 90% mode
    sm = V.summarize(ev, V.vrun(P, idx, **V.MODE90)[1], inv, idx, V.HOLD if holdout else V.MAIN)
    return res, sm


if __name__ == "__main__":
    allrows = []
    syms, idx, P = V.market(False)
    rows, sm = run_period("main", syms, idx, P, (("2021-23", V.T21, V.T24), ("2024-26", V.T24, None), ("since-25", V.T25, None)), False)
    print("repro 90% mode:", {k: round(v * 100, 2) for k, v in sm.items() if k.endswith("_mo")}, {k: round(v * 100, 1) for k, v in sm.items() if k.endswith("_wr")})
    allrows += rows
    syms, idx, P = V.market_holdout(False)
    rows, sm = run_period("hold", syms, idx, P, (("2018-20", V.H18, V.H21),), True)
    print("repro hold:", {k: round(v * 100, 2) for k, v in sm.items() if k.endswith("_mo")})
    allrows += rows
    df = pd.DataFrame(allrows).set_index("period")
    pd.set_option("display.width", 250)
    print((df.drop(columns=["months"]).T).round(3).to_string())
