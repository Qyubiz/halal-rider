"""Proof that the cycle kernel is the engine with the new switches off.

1. Hourly candles (study.market): x10_cycle_lib.ckernel == engine.run, equity and trade log, bit for bit, for the
   live settings and settings that exercise every switch (N, fmax, tp, hm, cd, part/tp2, trail, tpwait, tprest,
   mtp_now, pen, per-coin slippage, LUNA/FTT/SRM).
2. 1-minute data: the 60-minute bars built by x10_cycle_lib.build_bars equal engine.minute_panel's hourly candles,
   and ckernel on them == engine.mrun(delay=0) (6 coins, to keep memory low).
3. The 5-minute and 15-minute bars aggregate back to the 60-minute bars exactly.
4. Clock metrics vs lab.metrics for the live settings.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402
import study  # noqa: E402
import x10_cycle_lib as L  # noqa: E402
from lab import metrics  # noqa: E402


def same(a, b):
    return a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=True))


def k_on_panel(P, slip, depth, N, fmax, tp, hm, cd, part=1.0, tp2=0.0, trail=0.0, pen=0.0, tpwait=1, tprest=True,
               mtp_now=False):
    K = P["o"].shape[1]
    sl = np.full(K, slip) if np.ndim(slip) == 0 else np.asarray(slip, np.float64)
    return L.ckernel(P["o"].values, P["h"].values, P["l"].values, P["cf"].values, P["ref"].values, float(depth),
                     int(N), float(fmax), float(tp), float(part), float(tp2), float(trail), int(hm), int(cd), E.EPS,
                     E.FEE, sl, float(pen), 0.0, int(tpwait), bool(tprest), bool(mtp_now))


CASES = {
    "live": dict(depth=0.12, N=8, fmax=0.20, tp=0.015, hm=48, cd=24),
    "2 coins x50% tp5% hm96 cd12": dict(depth=0.12, N=2, fmax=0.5, tp=0.05, hm=96, cd=12),
    "1 coin x100% d15 tp3%": dict(depth=0.15, N=1, fmax=1.0, tp=0.03, hm=48, cd=24),
    "part 0.5 + tp2 4%": dict(depth=0.10, N=3, fmax=0.33, tp=0.015, hm=48, cd=0, part=0.5, tp2=0.04),
    "part 0.5 + trail 3%": dict(depth=0.12, N=4, fmax=0.25, tp=0.02, hm=72, cd=3, part=0.5, trail=0.03),
    "tpwait 3, no resting": dict(depth=0.12, N=8, fmax=0.20, tp=0.015, hm=48, cd=24, tpwait=3, tprest=False),
    "mtp_now, pen 0.2%": dict(depth=0.12, N=2, fmax=0.5, tp=0.03, hm=48, cd=25, mtp_now=True, pen=0.002),
}


def main():
    ok_all = True
    print("1. hourly candles: engine.run vs ckernel")
    for dead in (False, True):
        syms, idx, P = study.market(dead)
        slip = np.array([E.SLIP_MAJOR if s in E.H19 else E.SLIP_OTHER for s in syms])
        for name, kw in CASES.items():
            e, lg = E.run(P, idx, slip=slip, **kw)
            eq, lg2 = k_on_panel(P, slip, **kw)
            ok = same(e.values, eq) and same(lg.values, lg2[:, :7])
            ok_all &= ok
            print(f"   {'dead ' if dead else ''}{name:32s} engine x{e.iloc[-1]:8.4f} cycle x{eq[-1]:8.4f} "
                  f"trades {len(lg)}/{len(lg2)} {'IDENTICAL' if ok else 'DIFFERENT'}")

    print("2. 1-minute data: build_bars(60) vs engine.minute_panel, ckernel vs engine.mrun(delay=0)")
    syms6 = ["BTCUSDT", "SOLUSDT", "DOGEUSDT", "AVAXUSDT", "FETUSDT", "RENDERUSDT"]
    idx1, A, hidx, Ph = E.minute_panel(syms6)
    B = L.load_bars(60)
    cols = [B["syms"].index(s) for s in syms6]
    tb = B["t"]
    okt = len(tb) == len(hidx) and bool((tb == hidx).all())
    okb = okt and all(same(B[f][:, cols], Ph[f].values) for f in ("o", "h", "l", "c"))
    print(f"   bars: index {'same' if okt else 'DIFFERENT'}, o/h/l/c {'IDENTICAL' if okb else 'DIFFERENT'}")
    ok_all &= okb
    Bs = {f: np.ascontiguousarray(B[f][:, cols]) for f in ("o", "h", "l", "c")}
    Bs.update(syms=syms6, t=tb)
    ref = L.make_ref(Bs, 24); cf = L.ffill_close(Bs)
    okr = same(ref, Ph["ref"].values) and same(cf, Ph["cf"].values)
    print(f"   ref (24-bar high close) and ffilled close {'IDENTICAL' if okr else 'DIFFERENT'}")
    ok_all &= okr
    for name, kw in (("live", dict(depth=0.12, N=8, fmax=0.20, tp=0.015, hm=48, cd=24)),
                     ("2 coins x50% tp3%", dict(depth=0.12, N=2, fmax=0.5, tp=0.03, hm=48, cd=12)),
                     ("1 coin x100% d10 tp2% cd0", dict(depth=0.10, N=1, fmax=1.0, tp=0.02, hm=24, cd=0))):
        em, lgm = E.mrun(A, hidx, Ph, 0, **kw)
        eq, lg2 = L.crun(Bs, ref, cf, slip=E.SLIP_MAJOR, **kw)
        ok = same(em.values, eq) and same(lgm.values[:, :7], lg2[:, :7])
        ok_all &= ok
        print(f"   {name:32s} mrun x{em.iloc[-1]:8.4f} cycle x{eq[-1]:8.4f} trades {len(lgm)}/{len(lg2)} "
              f"{'IDENTICAL' if ok else 'DIFFERENT'}")
    del idx1, A, Ph

    print("3. 5m/15m bars aggregate to the 60m bars")
    for C in (5, 15):
        for dead in (False, True):
            b = L.load_bars(C, dead); b60 = L.load_bars(60, dead)
            m = 60 // C; nb = b["o"].shape[0] // m
            agg = {"o": b["o"][::m], "c": b["c"][m - 1::m],
                   "h": np.fmax.reduce(b["h"].reshape(nb, m, -1), axis=1),
                   "l": np.fmin.reduce(b["l"].reshape(nb, m, -1), axis=1)}
            ok = all(same(agg[f], b60[f]) for f in agg) and b["syms"] == b60["syms"]
            ok_all &= ok
            print(f"   {C}m{' +dead' if dead else ''}: {'IDENTICAL' if ok else 'DIFFERENT'}")
            del b, b60

    print("4. metrics: Clock vs lab.metrics, live settings on hourly candles")
    syms, idx, P = study.market(False)
    e, lg = E.run(P, idx)
    clk = L.Clock(idx)
    r = L.summarize(clk, e.values)
    d, t = metrics(e, end=E.DESIGN_END), metrics(e, start=E.DESIGN_END)
    print(f"   2021-23 lab x{d['mult']:.4f} DD {d['maxdd']*100:.2f}% | Clock x{r['d_mult']:.4f} DD {r['d_dd']*100:.2f}%")
    print(f"   2024-26 lab x{t['mult']:.4f} DD {t['maxdd']*100:.2f}% | Clock x{r['t_mult']:.4f} DD {r['t_dd']*100:.2f}% "
          f"{r['t_mo']*100:+.2f}%/mo; since 2025 {r['s_mo']*100:+.2f}%/mo DD {r['s_dd']*100:.2f}%")
    print("ALL CHECKS PASS" if ok_all else "SOME CHECK FAILED")


if __name__ == "__main__":
    main()
