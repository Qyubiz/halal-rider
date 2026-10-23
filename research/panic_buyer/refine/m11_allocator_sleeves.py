"""m11 allocator, step 1: every candidate strategy run ALONE on 100% of a unit account (its standalone "unit curve"),
for every scenario, saved as hourly arrays so the allocator (m11_allocator_lib.py) can combine them exactly.

Sleeves (each run with its own m10 engine, exactly as its m10 final script ran it):
  m90   90% mode            m10_velocity_lib.vrun(MODE90): 1 coin x 100%, 24h high -14%, +2.5%, 96h, cd 3h, guard 35%
  m10   10% mode            vrun: 1 coin x 100%, 24h high -14%, +5%, 48h, cd 24h, guard 35% (ens_lib.MODE10)
  norm  Normal mode         vrun: up to 8 bids/hour x 20%, 24h high -12%, +1.5%, 48h, cd 24h, no guard (engine.LIVE)
  vel   velocity pick U1    vrun: 90% mode OR 12h high -13% (+3%, 48h), re-arm after a win (m10_velocity_final)
  reg   regime pick         m10_regime_lib.run: 90% mode, no bid on a coin with 7-day daily volatility <= 8%
  mr    meanrev pick        m10_meanrev_lib: 5-min checks, 8% under the 1h high, +1.2%, 6h stop, 1h cd
  scalp scalp pick          m10_scalp_lib: 1-min, 8% under the 60-min high, 4 x 25%, +2%, 96h stop, guard 35%
  trend BTC/ETH trend       m10_portfolio_lib.prun(w=[0,0,1,0]): 50/50 BTC+ETH held while close > SMA200

Scenarios: m = 2021-26 normal fills; mw = every buy and sell fill 0.2% worse; ms = stressed (real bot timing where
the engine has it, fills 0.2% worse, + LUNA/FTT/SRM); h = 2018-20 holdout (Dec 2017 - Dec 2020) normal; hw = holdout
with fills 0.2% worse.

Saved per (sleeve, scenario): U[t] = unit equity at the close of hour t on the scenario's hourly index (main panel
2021-01-01.. or the holdout panel 2017-12-01..), trades (eb, xb = hour of entry fill / exit, ret, amt = money of the
order in unit-account terms). A sleeve holds a position across the boundary into hour t iff eb <= t-1 < xb.
Prints only the 2021-23 numbers of each sleeve + the 90% mode reproduction (all periods, published baseline).

Usage: PB_OUT=.../pb_out_m11/allocator python m11_allocator_sleeves.py [sleeve ...]
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.environ.get("PB_OUT", os.path.expanduser("~/pb_out_m11/allocator"))
SDIR = os.path.join(OUT, "sleeves")
os.makedirs(SDIR, exist_ok=True)
SCR = "/tmp/claude-0/-home-user-halal-rider/ec562e35-ea9e-5a74-ab92-4ff22c5eeb77/scratchpad"

import engine as E  # noqa: E402

PEN = 0.002
SCEN = ("m", "mw", "ms", "h", "hw")


def main_idx():
    return E.panel(E.load_h19())[1]


def hold_idx():
    return E.panel(E.load_pre())[1]


def save(name, scen, idx_target, U, eb, xb, ret, amt):
    np.savez(os.path.join(SDIR, f"{name}_{scen}.npz"), t=idx_target.values.astype("datetime64[h]").astype(np.int64),
             U=np.asarray(U, np.float64), eb=np.asarray(eb, np.int64), xb=np.asarray(xb, np.int64),
             ret=np.asarray(ret, np.float64), amt=np.asarray(amt, np.float64))


def to_target(e_idx, U, eb, xb, idx_t):
    """Re-index an hourly run onto the target index (identical hours expected; checked)."""
    pos = idx_t.get_indexer(e_idx)
    late = (pos < 0) & (e_idx > idx_t[-1])          # the 1-minute cache runs 1 hour past the hourly panel
    pos = np.where(late, len(idx_t) - 1, pos)
    assert (pos >= 0).all(), "run hours outside the target index"
    Ut = pd.Series(np.asarray(U), e_idx).reindex(idx_t).ffill().fillna(1.0).values
    return Ut, pos[np.asarray(eb, np.int64)], pos[np.asarray(xb, np.int64)]


# ------------------------------------------------------------------ hourly dip family through vrun
def vel_specs():
    import m10_velocity_lib as V
    A14 = dict(ref=24, depth=0.14, tp=0.025, hm=96)
    return {
        "m90": V.MODE90,
        "m10": dict(tiers=[dict(ref=24, depth=0.14, tp=0.05, hm=48, cdw=24, cdl=24)], nslots=1, excl=1, dyn=0,
                    guard=0.35),
        "norm": dict(tiers=[dict(ref=24, depth=0.12, tp=0.015, hm=48, cdw=24, cdl=24, fs=0.20)], nslots=99, nbid=8,
                     excl=1, dyn=0, guard=0.0),
        "vel": dict(tiers=[dict(A14, cdw=0, cdl=3), dict(ref=12, depth=0.13, tp=0.03, hm=48, cdw=0, cdl=3)],
                    nslots=1, excl=1, dyn=0, guard=0.35),
    }


def run_vel_family(name, idxm, idxh):
    import m10_velocity_lib as V
    kw = vel_specs()[name]
    for scen in SCEN:
        hold = scen.startswith("h")
        e, lg, inv = V.evaluate(kw, holdout=hold, worse=scen in ("mw", "hw"), stress=scen == "ms")[1:]
        U, eb, xb = to_target(e.index, e.values, lg.eb.values, lg.xb.values, idxh if hold else idxm)
        save(name, scen, idxh if hold else idxm, U, eb, xb, lg.ret.values, lg.amt.values)


def check_vrun_vs_wrun():
    """10% mode and Normal through vrun == w90_wr90_lib.wrun (the engine the live modes were audited with)."""
    import m10_velocity_lib as V
    import w90_wr90_lib as W
    syms, idx, P = V.market(False)
    out = []
    for name, wk in (("m10", dict(depth=0.14, N=1, fmax=1.0, tp=0.05, hm=48, cd=24, guard=0.35)),
                     ("norm", dict(depth=0.12, N=8, fmax=0.20, tp=0.015, hm=48, cd=24, guard=0.0))):
        e1, l1, _ = V.vrun(P, idx, **vel_specs()[name])
        e2, l2 = W.wrun(P, idx, **wk)
        out.append(f"{name}: vrun vs wrun max |equity diff| {np.abs(e1.values / e2.values - 1).max():.2e}, "
                   f"trades {len(l1)} vs {len(l2)}")
    return out


def run_regime(idxm, idxh):
    import m10_regime_lib as R
    rules = [[("c_vol7", "<=", 0.08)]]
    for scen in SCEN:
        which = {"m": "d", "mw": "d", "ms": "dx", "h": "h", "hw": "h"}[scen]
        over = R.bot_kw() if scen == "ms" else (R.stress_kw() if scen in ("mw", "hw") else {})
        r, e, lg, idx = R.run(which, rules=rules, **over)
        hold = scen.startswith("h")
        U, eb, xb = to_target(e.index, e.values, lg.eb.values, lg.xb.values, idxh if hold else idxm)
        save("reg", scen, idxh if hold else idxm, U, eb, xb, lg.ret.values, lg.amt.values)


def run_trend(idxm, idxh):
    import m10_portfolio_lib as PL
    for scen in SCEN:
        hold = scen.startswith("h")
        kw = PL.STRESS if scen in ("mw", "ms", "hw") else {}
        e, sq, lg = PL.prun(hold, [0.0, 0.0, 1.0, 0.0], **kw)
        lg = lg[lg.sl == 2]
        U, eb, xb = to_target(e.index, e.values, lg.eb.values, lg.xb.values, idxh if hold else idxm)
        save("trend", scen, idxh if hold else idxm, U, eb, xb, lg.ret.values, lg.amt.values)


# ------------------------------------------------------------------ meanrev (5-minute engine)
def _mr_dead_bars():
    """22-coin 5-minute bars (19 + LUNA/FTT/SRM cut at their collapse hour, engine.DEAD), as m10_meanrev_lib.load."""
    import m10_meanrev_lib as M
    path = os.path.join(SDIR, "bars5_dead.npz")
    if not os.path.exists(path):
        orig = M._coin

        def cut_coin(src, s, joins):
            d = orig(src, s, joins)
            if d is not None and s in E.DEAD:
                d = d[d.index < pd.Timestamp(E.DEAD[s]) + pd.Timedelta(hours=1)]
            return d
        M._coin = cut_coin
        M._build(os.path.join(E.DATA, "1m"), list(E.H19) + list(E.DEAD), {"RENDERUSDT": "RNDRUSDT"}, M.T21, M.TEND, 5,
                 path)
        M._coin = orig
    z = np.load(path, allow_pickle=False)
    B = {"syms": [str(x) for x in z["syms"]], "t": pd.to_datetime(z["t"], unit="s"),
         **{f: np.ascontiguousarray(z[f]) for f in ("o", "h", "l", "c")}}
    B["C"] = 5
    B["cf"] = pd.DataFrame(B["c"]).ffill().values
    return B


def _mr_pick(B):
    import m10_meanrev_lib as M
    hb = lambda x: M.bars_per(B, x)   # noqa: E731
    cfs = M.sh(B["cf"])
    ok = (cfs >= M.rollmax(B, hb(24 * 7)) * 0.65) & (cfs >= M.rollmax(B, hb(24)) * 0.85)
    lim = np.where(ok, M.rollmax(B, hb(1)) * 0.92, np.nan)
    return lim, dict(nord=1, fmax=1.0, maxpos=1, tp=0.012, hm=hb(6), cd=hb(1))


def _bars_to_hours(B, eq, lg, idx_t):
    t = B["t"]
    hr = t.floor("1h")
    U = pd.Series(eq, t).groupby(hr).last().reindex(idx_t).ffill().fillna(1.0).values
    pos = idx_t.get_indexer(hr)
    eb = pos[lg[:, 0].astype(np.int64)]
    xb = pos[lg[:, 1].astype(np.int64)]
    assert (eb >= 0).all() and (xb >= 0).all()
    return U, eb, xb


def run_meanrev(idxm, idxh):
    import m10_meanrev_lib as M
    M.OUT = os.path.join(SCR, "pb_out_m10", "meanrev")      # read the m10 5-minute bar caches
    for which, scens in (("main", ("m", "mw")), ("hold", ("h", "hw")), ("dead", ("ms",))):
        B = _mr_dead_bars() if which == "dead" else M.load(which, C=5)
        lim, kw = _mr_pick(B)
        for scen in scens:
            pen = PEN if scen in ("mw", "hw", "ms") else 0.0
            eq, lg = M.run(B, lim, pen=pen, pens=pen, **kw)
            idx_t = idxh if scen.startswith("h") else idxm
            U, eb, xb = _bars_to_hours(B, eq, lg, idx_t)
            save("mr", scen, idx_t, U, eb, xb, lg[:, 4], lg[:, 5])
        del B, lim


# ------------------------------------------------------------------ scalp (1-minute engine)
SCALP = dict(W=60, depth=0.08, tp=0.02, hmm=5760, N=4, fmax=0.25, guard=0.35)


def run_scalp(idxm, idxh):
    import m10_scalp_lib as S
    for K, scens in ((19, ("m", "mw")), (22, ("ms",))):
        A, hidx, Ph, SLIP = S.load(K)
        g = S.guard_ref(Ph)
        for scen in scens:
            pen = PEN if scen != "m" else 0.0
            e, lg, inv, lo, ntr = S.srun(A, hidx, Ph, SLIP, gref=g, **S.scalp(**SCALP, wick=0.005, pen=pen, spen=pen))
            U, eb, xb = to_target(e.index, e.values, lg.eb.values, lg.xb.values, idxm)
            save("scalp", scen, idxm, U, eb, xb, lg.ret.values, lg.amt.values)
        del A, Ph
    HA, hhidx, HPh, HSLIP, hsyms = S.load_holdout()
    hg = S.guard_ref(HPh)
    for scen in ("h", "hw"):
        pen = PEN if scen == "hw" else 0.0
        e, lg, inv, lo, ntr = S.srun(HA, hhidx, HPh, HSLIP, gref=hg, **S.scalp(**SCALP, wick=0.005, pen=pen, spen=pen))
        # the holdout minute panel may start before the hourly holdout panel: keep the hours of the hourly panel
        e = e[e.index >= idxh[0]]
        sh = int(np.searchsorted(hhidx, idxh[0]))
        keep = lg.eb.values >= sh
        lg = lg[keep]
        U, eb, xb = to_target(e.index, e.values, lg.eb.values - sh, lg.xb.values - sh, idxh)
        save("scalp", scen, idxh, U, eb, xb, lg.ret.values, lg.amt.values)


# ------------------------------------------------------------------ quick 2021-23 view + 90% mode reproduction
def load_sleeve(name, scen):
    z = np.load(os.path.join(SDIR, f"{name}_{scen}.npz"))
    idx = pd.DatetimeIndex(z["t"].astype("datetime64[h]"))
    return idx, z["U"], pd.DataFrame({k: z[k] for k in ("eb", "xb", "ret", "amt")})


def quick(name, scen, a, b):
    idx, U, lg = load_sleeve(name, scen)
    i0 = int(np.searchsorted(idx, a)); i1 = len(idx) if b is None else int(np.searchsorted(idx, b))
    x = np.concatenate([[1.0], U[i0:i1] / (U[i0 - 1] if i0 > 0 else 1.0)])
    months = ((idx[i1 - 1] + pd.Timedelta(hours=1)) - idx[i0]).total_seconds() / 86400 / (365.25 / 12)
    tb = idx[lg.eb.values]
    sel = (tb >= a) & ((tb < b) if b is not None else True)
    t = lg[sel]
    return (f"WR {100 * (t.ret > 0).mean():5.1f}% {100 * (x[-1] ** (1 / months) - 1):+6.2f}%/mo "
            f"DD {100 * (x / np.maximum.accumulate(x) - 1).min():6.1f}% {len(t) / months:5.1f} tr/mo")


def main():
    want = sys.argv[1:] or ["m90", "m10", "norm", "vel", "reg", "trend", "mr", "scalp"]
    idxm, idxh = main_idx(), hold_idx()
    for name in want:
        t0 = time.time()
        if name in ("m90", "m10", "norm", "vel"):
            run_vel_family(name, idxm, idxh)
        elif name == "reg":
            run_regime(idxm, idxh)
        elif name == "trend":
            run_trend(idxm, idxh)
        elif name == "mr":
            run_meanrev(idxm, idxh)
        elif name == "scalp":
            run_scalp(idxm, idxh)
        print(f"{name:6s} done in {time.time() - t0:5.0f}s | 2021-23 normal "
              f"{quick(name, 'm', pd.Timestamp('2021-01-01'), pd.Timestamp('2024-01-01'))} | worse "
              f"{quick(name, 'mw', pd.Timestamp('2021-01-01'), pd.Timestamp('2024-01-01'))}", flush=True)
    if "m90" in want:
        print("90% mode reproduction (published baseline):")
        for scen, lab in (("h", "holdout normal"), ("m", "normal"), ("mw", "worse"), ("ms", "stressed")):
            pers = [("2018-20", pd.Timestamp("2018-01-01"), pd.Timestamp("2021-01-01"))] if scen == "h" else [
                ("2021-23", pd.Timestamp("2021-01-01"), pd.Timestamp("2024-01-01")),
                ("2024-26", pd.Timestamp("2024-01-01"), None), ("since 2025", pd.Timestamp("2025-01-01"), None)]
            for p, a, b in pers:
                print(f"  {lab:15s} {p:10s} {quick('m90', scen, a, b)}")
        for line in check_vrun_vs_wrun():
            print("  " + line)


if __name__ == "__main__":
    main()
