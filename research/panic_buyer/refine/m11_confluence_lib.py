"""m11 "confluence": ONE single-account Panic Buyer that combines the round-m10 ingredients as VOTES, not hard gates.

Every candidate buy (hour t, coin k) gets a score = number of independent "votes" that agree (each vote is a yes/no
fact known at the OPEN of hour t, i.e. from closed candles up to t-1). The score then decides, per trigger tier,
whether the coin is bid at all, how deep, how large and with which take-profit; and whether a coin may be re-armed
right after a winning exit.

ck = m10_velocity_lib.vk with excl=1 (one position per coin, ONE resting bid per coin at the highest of its tiers'
trigger prices), extended so that a tier's depth / size / take-profit are looked up from the candidate's score:
  tier s:  ref   tref[s] 0 = 24h highest close (the live bot's reference), 1 = the second, faster reference R2
                 (highest close of the last w2 closed hours, w2 = 12 or 8)
           group tg[s] 0 = the user's core coins, 1 = the extra coins (top-M screened coins by 30-day volume that month)
           dep[s, score]  bid at ref x (1 - dep); dep >= 1 = this tier does not bid at that score
           tpb[s, score], tpm[s, score]  take-profit: tpm 0 = fixed tpb; 1 = tpb x sqrt(rng24 / 0.2074) clipped to
                 [tplo, tphi] (m10 exits agent's volatility-scaled target; rng24 = 24h high-low range / last close)
           fsz[s, score]  order = min(fsz x account, free cash / bids this hour)
           hm[s] time stop (hours, sold at the hourly close), cdw[s] / cdl[s] cooldown after a win / loss (hours)
  re-arm:  after a WIN, the coin may be bid again from t + cdr (instead of t + cdw) in any hour where its score >= rak
           (rak > 7 = off; the m10 velocity agent's "re-arm after a win" is cdr 0, rak 0)
  nslots   most positions held at once (1 = 90% mode: all the money in one coin)
Costs as every refine engine: fee 0.1% a side; market buys (open already under the trigger) and time-stop sells pay
slippage (0.02% for the 19 coins, 0.1% for LUNA/FTT/SRM in the stressed run, rank-based for extra coins as in
m10_universe_lib); a limit buy fills only when the low trades 0.1% through, a resting take-profit only when the high
trades 0.1% through; pen / pens = every buy / sell 0.2% worse in the stress rows; mtp_now = the real bot's timing
(take-profit placed at once after a market buy); cd_add = cooldowns counted from the next hourly check (+1h).
Coins whose data end (LUNA/FTT/SRM collapse, delistings) are sold at their last hour's close (why 5).

With one tier (24h, 14%, +2.5%, 96h, cooldown 3h, size 1, every score) and re-arm off, ck == m10_velocity_lib.vk on
MODE90 exactly (m11_confluence_check.py).

VOTES (bit i of the uint8 'bits' array; all from closed candles before hour t; NaN = vote 0):
  0 vol    coin 7-day daily volatility (std of hourly log returns x sqrt 24) > 8%   (m10 regime agent)
  1 fast   the 24h high close is at most 12 hours old (the 12h trigger would sit at the same reference)
  2 deep   the open is already >= 2 points under the 14% trigger (a deeper dip at order time)
  3 btc    BTC's last 24h move is above -3% (the dip is the coin's own, not a market crash)
  4 brd    fewer than 25% of the core coins are >= 10% under their 24h high (no broad crash)
  5 rng    the coin's 24h high-low range / last close >= 0.20 (a big, volatile day)
  6 red1   the last hourly close was down (the coin is still falling into the bid)
Directions/thresholds: see m11_confluence_anatomy.py (2021-23 trades only) and the grid header.
"""
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.environ.get("PB_OUT", os.path.expanduser("~/pb_out_m11/confluence"))
os.makedirs(OUT, exist_ok=True)
import engine as E  # noqa: E402
import m10_velocity_lib as V  # noqa: E402

T21, T24, T25 = pd.Timestamp("2021-01-01"), pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")
H18, H21 = pd.Timestamp("2018-01-01"), pd.Timestamp("2021-01-01")
LOGC = ["eb", "xb", "k", "why", "ret", "amt", "entry", "eq0", "tier", "sc"]
UPAN = os.environ.get("PB_UPANEL", os.path.join(os.path.dirname(os.path.dirname(OUT)), "pb_out_m10", "universe",
                                                "m10_panel"))
NV = 7
VOTES = ["vol", "fast", "deep", "btc", "brd", "rng", "red1"]
RNG_REF = 0.2074
STRESS_PEN = 0.002


# ------------------------------------------------------------------ kernel
@njit(cache=True)
def ck(o, h, l, c, gm, R24, R2, SC, RNG, CORE, EX, lastb, slip,
       tref, tg, dep, tpb, tpm, tplo, tphi, fsz, hm, cdw, cdl, cdr, rak, nslots,
       eps, fee, pen, pens, mtp_now):
    n, K = c.shape
    S = tref.shape[0]
    units = np.zeros(K); entry = np.zeros(K); amt0 = np.zeros(K); tpp = np.zeros(K); eq0 = np.zeros(K)
    ebar = np.full(K, -1, np.int64); mkt = np.zeros(K, np.int64); tier = np.zeros(K, np.int64)
    sc0 = np.zeros(K, np.int64)
    okc = np.zeros(K, np.int64); okw = np.full(K, 2 ** 62, np.int64)
    lastc = np.full(K, np.nan)
    cash = 1.0; eq = np.empty(n); eq_prev = 1.0
    maxlog = 400000; lg = np.zeros((maxlog, 10)); nl = 0
    cdist = np.empty(K); cs = np.empty(K, np.int64); ck_ = np.empty(K, np.int64); clim = np.empty(K)
    for t in range(n):
        nheld = 0
        for k in range(K):
            if units[k] > 0.0:
                nheld += 1
        m = 0
        if nheld < nslots and cash > 1e-9:
            for k in range(K):
                if units[k] > 0.0 or np.isnan(o[t, k]) or gm[t, k] == 0:
                    continue
                sc = SC[t, k]
                if t < okc[k]:
                    if not (t >= okw[k] and sc >= rak):
                        continue
                best = -1.0; bsel = -1
                for s in range(S):
                    if tg[s] == 0:
                        if not CORE[k]:
                            continue
                    elif not EX[t, k]:
                        continue
                    d = dep[s, sc]
                    if d >= 1.0:
                        continue
                    r = R24[t, k] if tref[s] == 0 else R2[t, k]
                    if np.isnan(r):
                        continue
                    lim = r * (1.0 - d)
                    if lim > best:
                        best = lim; bsel = s
                if bsel >= 0:
                    cdist[m] = o[t, k] / best - 1.0; cs[m] = bsel; ck_[m] = k; clim[m] = best; m += 1
        if m > 0:
            order = np.argsort(cdist[:m])
            nb = min(m, nslots - nheld)
            cash0 = cash
            for q in range(nb):
                j = order[q]
                s = cs[j]; k = ck_[j]; lim = clim[j]; sc = SC[t, k]
                per = min(fsz[s, sc] * eq_prev, cash0 / nb)
                fill = -1.0; ismkt = 0
                if o[t, k] <= lim:
                    fill = o[t, k] * (1.0 + slip[t, k]) * (1.0 + pen); ismkt = 1
                elif l[t, k] <= lim * (1.0 - eps):
                    fill = lim * (1.0 + pen)
                if fill > 0.0 and cash > 1e-9:
                    mkt[k] = ismkt if mtp_now else 0
                    a = min(per, cash)
                    cash -= a; units[k] = a * (1.0 - fee) / fill
                    entry[k] = fill; amt0[k] = a; ebar[k] = t; eq0[k] = eq_prev; tier[k] = s; sc0[k] = sc
                    tp = tpb[s, sc]
                    if tpm[s, sc] == 1:
                        x = RNG[t, k]
                        if np.isfinite(x) and x > 0.0:
                            tp = tp * np.sqrt(x / 0.2074)
                        tp = min(max(tp, tplo), tphi)
                    tpp[k] = fill * (1.0 + tp)
        # ---------------- exits
        for k in range(K):
            if units[k] > 0.0 and (ebar[k] < t or mkt[k] == 1) and not np.isnan(o[t, k]):
                s = tier[k]
                px = -1.0; why = 0.0
                if ebar[k] == t:
                    if h[t, k] >= tpp[k] * (1.0 + eps):
                        px = tpp[k]
                elif o[t, k] >= tpp[k]:
                    px = o[t, k]
                elif h[t, k] >= tpp[k] * (1.0 + eps):
                    px = tpp[k]
                age = t - ebar[k] + mkt[k]
                if px < 0.0 and age >= hm[s]:
                    px = c[t, k] * (1.0 - slip[t, k]); why = 1.0
                if px > 0.0:
                    proceeds = units[k] * px * (1.0 - pens) * (1.0 - fee); cash += proceeds
                    if nl < maxlog:
                        lg[nl, 0] = ebar[k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = why
                        lg[nl, 4] = proceeds / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]
                        lg[nl, 7] = eq0[k]; lg[nl, 8] = s; lg[nl, 9] = sc0[k]; nl += 1
                    units[k] = 0.0
                    if proceeds > amt0[k]:
                        okc[k] = t + cdw[s]; okw[k] = t + cdr
                    else:
                        okc[k] = t + cdl[s]; okw[k] = 2 ** 62
        for k in range(K):
            if units[k] > 0.0 and lastb[k] == t:
                s = tier[k]
                proceeds = units[k] * c[t, k] * (1.0 - slip[t, k]) * (1.0 - pens) * (1.0 - fee); cash += proceeds
                if nl < maxlog:
                    lg[nl, 0] = ebar[k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = 5.0
                    lg[nl, 4] = proceeds / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]
                    lg[nl, 7] = eq0[k]; lg[nl, 8] = s; lg[nl, 9] = sc0[k]; nl += 1
                units[k] = 0.0
                okc[k] = t + cdl[s]; okw[k] = 2 ** 62
        # ---------------- hourly mark
        for k in range(K):
            if not np.isnan(c[t, k]):
                lastc[k] = c[t, k]
        v = cash
        for k in range(K):
            if units[k] > 0.0:
                v += units[k] * lastc[k]
        eq[t] = v; eq_prev = v
    for k in range(K):
        if units[k] > 0.0 and nl < maxlog:
            proceeds = units[k] * lastc[k] * (1.0 - slip[n - 1, k]) * (1.0 - pens) * (1.0 - fee)
            lg[nl, 0] = ebar[k]; lg[nl, 1] = n - 1; lg[nl, 2] = k; lg[nl, 3] = 9.0
            lg[nl, 4] = proceeds / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]
            lg[nl, 7] = eq0[k]; lg[nl, 8] = tier[k]; lg[nl, 9] = sc0[k]; nl += 1
    return eq, lg[:nl].copy()


@njit(cache=True)
def _hours_since_high(c, W):
    """Hours since the highest close among the W closes before hour t (most recent max on ties); 0 if none."""
    n, K = c.shape
    out = np.zeros((n, K), np.float32)
    for k in range(K):
        for t in range(n):
            best = -1.0; bj = -1
            for j in range(max(0, t - W), t):
                x = c[j, k]
                if not np.isnan(x) and x >= best:
                    best = x; bj = j
            out[t, k] = np.nan if bj < 0 else t - bj
    return out


# ------------------------------------------------------------------ markets
_MK = {}


def _bits_and_rng(o, h, l, cf, ref, core, btc_j, chunk=40):
    """Vote bits (uint8, bit i = VOTES[i]) and rng24 (float32), computed in column chunks to keep memory low."""
    n, K = cf.shape
    bits = np.zeros((n, K), np.uint8)
    rng = np.full((n, K), np.nan, np.float32)
    # market features from the core coins (and BTC)
    C = pd.DataFrame(np.asarray(cf[:, core], np.float64))
    R = pd.DataFrame(np.asarray(ref[:, core], np.float64))
    live = pd.DataFrame(np.asarray(o[:, core], np.float64)).notna().shift(1).rolling(24, min_periods=1).sum() > 0
    dip_c = C.shift(1) / R - 1
    nlive = live.sum(axis=1).clip(lower=1)
    brd = (((dip_c <= -0.10) & live).sum(axis=1) / nlive).values
    b = pd.Series(np.asarray(cf[:, btc_j], np.float64))
    btc24 = (b.shift(1) / b.shift(25) - 1).values
    m_btc = np.nan_to_num(btc24, nan=-1.0) > -0.03
    m_brd = brd < 0.25
    for j0 in range(0, K, chunk):
        sl = slice(j0, min(K, j0 + chunk))
        c = pd.DataFrame(np.asarray(cf[:, sl], np.float64))
        lr = np.log(c).diff()
        vol7 = (lr.rolling(168, min_periods=84).std() * np.sqrt(24)).shift(1).values
        hi = pd.DataFrame(np.asarray(h[:, sl], np.float64)).rolling(24, min_periods=6).max()
        lo = pd.DataFrame(np.asarray(l[:, sl], np.float64)).rolling(24, min_periods=6).min()
        r24 = ((hi - lo) / c).shift(1).values
        hsh = _hours_since_high(np.ascontiguousarray(c.values), 24)
        oo = np.asarray(o[:, sl], np.float64); rf = np.asarray(ref[:, sl], np.float64)
        r1 = (c.shift(1) / c.shift(2) - 1).values
        with np.errstate(invalid="ignore"):
            v = [vol7 > 0.08, hsh <= 12, oo <= rf * 0.86 * 0.98, np.repeat(m_btc[:, None], c.shape[1], 1),
                 np.repeat(m_brd[:, None], c.shape[1], 1), r24 >= 0.20, r1 < 0]
        bb = np.zeros(c.shape, np.uint8)
        for i, x in enumerate(v):
            bb |= (np.nan_to_num(x, nan=0).astype(bool).astype(np.uint8) << i)
        bits[:, sl] = bb
        rng[:, sl] = r24.astype(np.float32)
    return bits, rng


def market(which):
    """which: 'w' 2021-26 universe panel (19 core + screened coins; LUNA/FTT/SRM as extra-coin rows only),
    'ws' the same for the stressed run (LUNA/FTT/SRM forced in as core coins), 'h' the 2018-20 holdout (16 coins),
    'hs' holdout + LUNA/FTT/SRM rows (none of them traded before 2019-20; same as m10). Returns a dict of arrays."""
    if which in _MK:
        return _MK[which]
    if which in ("w", "ws"):
        F = ("o", "h", "l", "c", "ref", "r30o", "slip")
        A = {f: np.load(os.path.join(UPAN, f + ".npy"), mmap_mode="r") for f in F}
        syms = [str(s) for s in np.load(os.path.join(UPAN, "syms.npy"))]
        idx = pd.to_datetime(np.load(os.path.join(UPAN, "t.npy")), unit="s")
        n, K = A["o"].shape
        core = np.array([s in E.H19 for s in syms])
        if which == "ws":
            for s in E.DEAD:
                core[syms.index(s)] = True
        ok = ~np.isnan(np.asarray(A["o"]))
        lb = np.array([np.nonzero(ok[:, k])[0][-1] if ok[:, k].any() else -1 for k in range(K)], np.int64)
        lb[lb >= n - 3] = n + 1
        del ok
        slip = np.array(A["slip"], np.float64)
        slip[:, np.array([s in E.H19 for s in syms])] = E.SLIP_MAJOR
        for s in E.DEAD:
            slip[:, syms.index(s)] = E.SLIP_OTHER
        if which == "w":
            # FTT is not a screened coin: never an extra coin outside the stressed run (m10_universe_lib)
            pass
        M = dict(idx=idx, syms=syms, o=A["o"], h=A["h"], l=A["l"], c=A["c"], ref=A["ref"], r30o=A["r30o"],
                 core=core, lastb=lb, slip=slip)
    else:
        syms_, idx, P = V.market_holdout(which == "hs")
        syms = list(P["o"].columns)
        n, K = P["o"].shape
        core = np.ones(K, bool)
        lb = V.last_bars(P) if which == "hs" else np.full(K, -1, np.int64)
        sl = np.array([E.SLIP_MAJOR if s in E.H19 else E.SLIP_OTHER for s in syms]) if which == "hs" \
            else np.full(K, E.SLIP_MAJOR)
        M = dict(idx=idx, syms=syms, o=P["o"].values, h=P["h"].values, l=P["l"].values, c=P["cf"].values,
                 ref=P["ref"].values, r30o=None, core=core, lastb=lb,
                 slip=np.ascontiguousarray(np.repeat(sl[None, :], n, 0)))
    o, h, l, c, ref = M["o"], M["h"], M["l"], M["c"], M["ref"]
    M["gm"] = (V.guard_mask({"cf": pd.DataFrame(np.asarray(c))}, 0.35) > 0).astype(np.uint8)
    h19 = np.array([s in E.H19 for s in M["syms"]])      # market votes (breadth) from the 19 coins only
    M["bits"], M["rng"] = _bits_and_rng(o, h, l, c, ref, h19, M["syms"].index("BTCUSDT"))
    M["R2"] = {}
    _MK[which] = M
    return M


def r2(M, w):
    if w not in M["R2"]:
        cc = M["c"]
        out = np.empty(cc.shape)
        raw = M["o"]   # closes are forward-filled; use the raw-close definition (NaN where no candle) like ref
        for j0 in range(0, cc.shape[1], 40):
            sl = slice(j0, min(cc.shape[1], j0 + 40))
            x = np.asarray(cc[:, sl], np.float64).copy()
            x[np.isnan(np.asarray(raw[:, sl], np.float64))] = np.nan
            out[:, sl] = pd.DataFrame(x).rolling(w, min_periods=1).max().shift(1).values
        M["R2"][w] = out
    return M["R2"][w]


def extra_mask(M, top, stress):
    """(n, K) bool: coin is an extra coin this hour (monthly top-`top` non-core screened coin by 30-day volume)."""
    if M["r30o"] is None or top <= 0:
        return np.zeros((1, 1), np.bool_), False
    ex = np.asarray(M["r30o"]) <= top
    ex[:, M["core"]] = False
    if not stress:
        ex[:, M["syms"].index("FTTUSDT")] = False
    return ex, True


# ------------------------------------------------------------------ designs
BASE = dict(dep=0.14, tp=0.025, hm=96, cd=3)


def score(M, votes):
    """Number of the listed votes that are on, (n, K) int8."""
    b = M["bits"]
    s = np.zeros(b.shape, np.int8)
    for v in votes:                      # "-name" = the vote is ON when the fact is FALSE
        x = ((b >> VOTES.index(v.lstrip("-"))) & 1).astype(np.int8)
        s += (1 - x) if v.startswith("-") else x
    return s


def tables(d):
    """Design dict -> per-tier arrays. Keys (all optional; defaults = 90% mode):
      votes  list of vote names (score = how many agree); 'kA' min score for the 24h tier (0 = always);
      'dlo'  depth for the 24h tier when score < kA (None = no bid); 'fz' size when score < kF (kF, fz);
      'tpA' fixed TP or 'scaled' (when score >= kT; tpA0 below kT); 'B' second trigger dict(w, dep, tp, hm, k);
      'X' extra-coin tier dict(top, dep, tp, hm, k); 'rak' re-arm min score (None = off); cd cooldown."""
    tiers = []
    sz = np.zeros(8); sz[:] = 1.0
    if d.get("kF") is not None:
        sz[:d["kF"]] = d.get("fz", 0.5)
    # tier A: 24h reference, 14%
    dA = np.full(8, d.get("depA", 0.14))
    if d.get("kA", 0) > 0:
        dA[:d["kA"]] = d["dlo"] if d.get("dlo") is not None else 2.0
    tA = np.full(8, d.get("tpA", 0.025)); mA = np.zeros(8, np.int64)
    if d.get("kT") is not None:                         # scaled TP from score kT up (fixed 2.5% below)
        tA[d["kT"]:] = 0.03; mA[d["kT"]:] = 1
    tiers.append(dict(tref=0, tg=0, dep=dA, tpb=tA, tpm=mA, fsz=sz.copy(), hm=d.get("hm", 96)))
    if d.get("B"):
        B = d["B"]
        dB = np.full(8, B["dep"]); dB[:B.get("k", 0)] = 2.0
        tiers.append(dict(tref=1, tg=0, dep=dB, tpb=np.full(8, B["tp"]), tpm=np.zeros(8, np.int64), fsz=sz.copy(),
                          hm=B["hm"]))
    if d.get("X"):
        X = d["X"]
        dX = np.full(8, X["dep"]); dX[:X.get("k", 0)] = 2.0
        tiers.append(dict(tref=0, tg=1, dep=dX, tpb=np.full(8, X.get("tp", 0.025)), tpm=np.zeros(8, np.int64),
                          fsz=sz.copy(), hm=X.get("hm", 96)))
    return tiers


def run(M, d, pen=0.0, pens=0.0, mtp_now=False, cd_add=0, start=None, end=None):
    """Run design d on market M (from `start`, up to `end` exclusive). Returns (equity Series, log DataFrame)."""
    tiers = tables(d)
    S = len(tiers)
    cd = d.get("cd", 3)
    i0 = 0 if start is None else int(np.searchsorted(M["idx"], pd.Timestamp(start)))
    i1 = None if end is None else int(np.searchsorted(M["idx"], pd.Timestamp(end)))
    sl = slice(i0, i1)
    sc = np.ascontiguousarray(score(M, d.get("votes", []))[sl])
    stress = bool(M["core"][M["syms"].index("LUNAUSDT")]) if "LUNAUSDT" in M["syms"] else False
    X = d.get("X")
    if X:
        ex, _ = extra_mask(M, X["top"], stress)
        ex = np.ascontiguousarray(ex[sl])
    else:
        ex = np.zeros((1, 1), np.bool_)
    w2 = d["B"]["w"] if d.get("B") else 24
    R24 = np.asarray(M["ref"])[sl]
    R2 = r2(M, w2)[sl] if d.get("B") else R24
    rak = d["rak"] if d.get("rak") is not None else 99
    lb = M["lastb"] - i0
    f = lambda key, dt=np.float64: np.array([t[key] for t in tiers], dt)   # noqa: E731
    eq, lg = ck(np.asarray(M["o"])[sl], np.asarray(M["h"])[sl], np.asarray(M["l"])[sl], np.asarray(M["c"])[sl],
                M["gm"][sl], R24, R2, sc, M["rng"][sl], M["core"], ex, lb, M["slip"][sl],
                f("tref", np.int64), f("tg", np.int64), np.stack([t["dep"] for t in tiers]),
                np.stack([t["tpb"] for t in tiers]), np.stack([t["tpm"] for t in tiers]),
                float(d.get("tplo", 0.01)), float(d.get("tphi", 0.06)), np.stack([t["fsz"] for t in tiers]),
                f("hm", np.int64), np.full(S, cd + cd_add, np.int64), np.full(S, cd + cd_add, np.int64),
                int(d.get("cdr", 0) + cd_add), int(rak), int(d.get("nslots", 1)),
                E.EPS, E.FEE, float(pen), float(pens), bool(mtp_now))
    return pd.Series(eq, M["idx"][sl]), pd.DataFrame(lg, columns=LOGC)


# ------------------------------------------------------------------ metrics
def summ(e, lg, periods):
    """Per period: %/month, worst drop, win rate (incl. open positions marked at the end), trades/month, worst trade
    as % of the account (m10_velocity_lib.summarize conventions)."""
    lg = lg.rename(columns={"sc": "_sc"})
    r = V.summarize(e, lg, pd.Series(np.zeros(len(e)), e.index), e.index, periods)
    return r


PER_MAIN = V.MAIN
PER_HOLD = V.HOLD
LAB = {"d": "2021-23", "t": "2024-26", "s": "since 2025", "h": "2018-20 holdout"}


def line(tag, p, r):
    return (f"  {tag:9s} {LAB[p]:16s} win {r[f'{p}_wr']*100:5.1f}% | {r[f'{p}_mo']*100:+6.2f}%/mo | max drop "
            f"{r[f'{p}_dd']*100:6.1f}% | worst trade {r[f'{p}_wt']*100:+6.1f}% of acct | {r[f'{p}_tpm']:5.1f} trades/mo")
