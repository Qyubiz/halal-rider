"""Proof that x10_idle_lib.ikernel reproduces engine.run exactly with parking off (and with parking on but all
weights 0), on the 19 coins and with LUNA/FTT/SRM; plus a sanity check of the parking leg alone (N=0: buy and
hold BTC must equal BTC's price change minus one buy's costs)."""
import numpy as np

import engine as E
import x10_idle_lib as X

ok_all = True
for dead in (False, True):
    syms, idx, P = X.market(dead)
    sl = X.slip_vec(syms)
    for name, b in X.BASES.items():
        for pen in (0.0, 0.002):
            e0, l0 = E.run(P, idx, slip=sl, pen=pen, **b)
            e1, l1, aux, st = X.irun(P, idx, pw=None, slip=sl, pen=pen, **b)
            W0 = np.zeros(P["o"].shape)
            e2, l2, _, _ = X.irun(P, idx, pw=W0, slip=sl, pen=pen, **b)
            d1 = float(np.max(np.abs(e0.values - e1.values)))
            d2 = float(np.max(np.abs(e0.values - e2.values)))
            same_log = len(l0) == len(l1) and np.array_equal(l0[E.LOG_COLS].values, l1[E.LOG_COLS].values)
            ok = d1 == 0.0 and d2 < 1e-12 and same_log and len(l2) == len(l0)
            ok_all &= ok
            print(f"dead={dead!s:5s} {name:18s} pen={pen}: engine x{e0.iloc[-1]:.4f} idle-off x{e1.iloc[-1]:.4f} "
                  f"(max diff {d1:.1e}) idle-zero-weights max diff {d2:.1e}; trades {len(l0)}/{len(l1)}/{len(l2)} "
                  f"logs identical {same_log} -> {'OK' if ok else 'MISMATCH'}")

syms, idx, P = X.market(False)
W = X.w_fixed(P, idx, {"BTCUSDT": 1.0})
e, lg, aux, st = X.irun(P, idx, pw=W, N=0, band=0.0)
o0 = P["o"]["BTCUSDT"].iloc[0]; c1 = P["cf"]["BTCUSDT"].iloc[-1]
expect = (1 - E.FEE) / (1 + E.SLIP_MAJOR) * c1 / o0
print(f"parking alone, BTC buy and hold: x{e.iloc[-1]:.6f} vs expected x{expect:.6f}, trades {st}")
ok_all &= abs(e.iloc[-1] - expect) < 1e-9
print("ALL OK" if ok_all else "SOME CHECK FAILED")
