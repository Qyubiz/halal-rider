"""Code check for x10_velocity_lib.vkernel (not a result): with every new switch off it must reproduce engine.mkernel
(engine.mrun) - same trades, same exit minutes and prices, equity equal up to float rounding - on synthetic
crash-and-bounce minutes (check_engine.synthetic) and on the real 19-coin minute panel, for several settings
(live, concentrated, take-profit delay, cooldown 0). Also prints engine.run (hourly) on the hourly candles built from
the same minutes, which engine.mrun(delay 0) matches (check_engine.py). Then sanity checks of the new switches."""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402
import check_engine as CE  # noqa: E402
import x10_velocity_lib as V  # noqa: E402

SETS = [dict(), dict(N=2, fmax=0.5, tp=0.03), dict(N=1, fmax=1.0, tp=0.03, hm=96), dict(delay=5),
        dict(N=2, fmax=0.5, tp=0.05, cd=0), dict(N=3, fmax=0.33, tp=0.02, delay=1, cd=1)]


def same(name, m_eq, m_lg, v_eq, v_lg):
    a = m_lg[["eb", "xb", "k", "why", "entry", "xm"]].values
    b = v_lg[["eb", "xb", "k", "why", "entry", "xm"]].values
    # trade order inside an hour can differ (the minute kernel logs in time order); compare sorted
    sa = a[np.lexsort(a.T[::-1])]; sb = b[np.lexsort(b.T[::-1])]
    ok_log = a.shape == b.shape and np.array_equal(sa, sb)
    d = float(np.nanmax(np.abs(m_eq.values - v_eq.values)))
    rel = float(np.nanmax(np.abs(m_eq.values / v_eq.values - 1)))
    print(f"  {name:48s} trades {len(m_lg):5d} vs {len(v_lg):5d}  log identical: {ok_log}  "
          f"max |eq diff| {d:.2e} (rel {rel:.2e})  final x{m_eq.iloc[-1]:.6f} vs x{v_eq.iloc[-1]:.6f}")
    assert ok_log and rel < 1e-9, name
    return rel


def main():
    print("######## synthetic minutes (check_engine.synthetic)")
    A = CE.synthetic()
    Hh = CE.hourly(A)
    idx = pd.date_range("2021-01-01", periods=len(Hh["c"]), freq="1h")
    ph = {f: pd.DataFrame(v, idx) for f, v in Hh.items()}
    ph["cf"] = ph["c"].ffill(); ph["ref"] = ph["c"].rolling(24, min_periods=1).max().shift(1)
    K = A["c"].shape[1]
    for s in SETS:
        kw = {k: v for k, v in s.items() if k != "delay"}
        m_eq, m_lg = E.mrun(A, idx, ph, delay=s.get("delay", 0), **kw)
        v_eq, v_lg, _ = V.vrun(A, idx, ph, E.SLIP_MAJOR, K=K, **s)
        same(f"synthetic {s}", m_eq, m_lg, v_eq, v_lg)

    print("\n######## real 1-minute panel, 19 live coins (float32 cache, memory-mapped)")
    t0 = time.time()
    A, hidx, Ph, SLIP = V.load()
    print(f"loaded {A['c'].shape} in {time.time()-t0:.0f}s, hours {hidx[0]} .. {hidx[-1]}")
    A19 = {f: v[:, :19] for f, v in A.items()}
    P19 = {f: v.iloc[:, :19] for f, v in Ph.items()}
    for s in SETS:
        kw = {k: v for k, v in s.items() if k != "delay"}
        t1 = time.time()
        m_eq, m_lg = E.mrun(A19, hidx, P19, delay=s.get("delay", 0), **kw)
        t2 = time.time()
        v_eq, v_lg, _ = V.vrun(A, hidx, Ph, E.SLIP_MAJOR, K=19, **s)
        t3 = time.time()
        same(f"real {s}", m_eq, m_lg, v_eq, v_lg)
        print(f"  {'':48s} mkernel {t2-t1:.1f}s, vkernel {t3-t2:.1f}s")
    h_eq, h_lg = E.run(P19, hidx)
    m_eq, _ = E.mrun(A19, hidx, P19, delay=0)
    print(f"  engine.run (hourly candles from the same minutes), live settings: final x{h_eq.iloc[-1]:.6f}, "
          f"engine.mrun delay 0: x{m_eq.iloc[-1]:.6f}, max rel diff {np.nanmax(np.abs(h_eq.values/m_eq.values-1)):.2e}")

    print("\n######## new switches: sanity (whole period, normal 0.02% slippage, 19 coins)")
    for s in (dict(cyc=1), dict(rearm=1), dict(cdm=0), dict(near=0.02), dict(xmode=1, arm=0.015, trail=0.005, delay=1),
              dict(xmode=1, arm=0.015, trail=0.0, cap=0.05, delay=1), dict(spen=0.002), dict(pen=0.002, spen=0.002)):
        e, lg, inv = V.vrun(A, hidx, Ph, E.SLIP_MAJOR, K=19, **s)
        print(f"  {str(s):60s} final x{e.iloc[-1]:10.4f} trades {len(lg):6d} exits {lg.why.value_counts().to_dict()} "
              f"invested {inv.mean()*100:.1f}% of hours-equity")
    e, lg, inv = V.vrun(A, hidx, Ph, SLIP, K=19)
    print(f"  live settings with the volume-rank slippage panel: final x{e.iloc[-1]:.4f}; share of coin-hours at 0.1%: "
          f"{(SLIP[:, :19] > 0.0005).mean()*100:.1f}% (19 coins), {(SLIP[:, 19:] > 0.0005).mean()*100:.1f}% (LUNA/FTT/SRM)")
    print("VELOCITY KERNEL CHECK PASSED")


if __name__ == "__main__":
    main()
