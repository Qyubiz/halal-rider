"""Code check for m10_scalp_lib.skernel (not a result).
1. With every new switch off it must reproduce x10_velocity_lib.vkernel (xmode 0), itself proven equal to engine.mkernel:
   same trade log, equity equal up to float rounding, on the real 19-coin minute panel 2021-26, for several settings.
2. No lookahead in the W-minute reference: every logged entry must equal (max of the W closes BEFORE the fill minute)
   x (1 - depth) for a resting fill, or the fill minute's open x (1 + slippage) for a market buy at or under the trigger.
3. Fill rules: no take-profit in the fill minute; every take-profit minute's trimmed high traded eps through the target;
   every resting buy minute's trimmed low traded eps through the limit.
4. Huge wick allowance == no trimming."""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402
import m10_scalp_lib as S  # noqa: E402
import x10_velocity_lib as V  # noqa: E402

A, hidx, Ph, SLIP = S.load(19)
print(f"panel {A['c'].shape}, {hidx[0]} .. {hidx[-1]}", flush=True)

# ---- 1. equivalence with vkernel
SETS = [dict(), dict(N=1, fmax=1.0, tp=0.025, hm=96, cd=3), dict(cyc=1, delay=1, cdm=0, N=2, fmax=0.5, tp=0.01),
        dict(cyc=5, rearm=1, near=0.02, delay=3, tp=0.02), dict(pen=0.002, spen=0.002)]
for s in SETS:
    t0 = time.time()
    ve, vl, vi = V.vrun({f: A[f] for f in A}, hidx, Ph, SLIP, K=19, **s)
    t1 = time.time()
    se, sl_, si, slo, ntr = S.srun(A, hidx, Ph, SLIP, **s)
    t2 = time.time()
    a = vl[["eb", "xb", "k", "why", "entry", "xm", "fm"]].values
    b = sl_[["eb", "xb", "k", "why", "entry", "xm", "fm"]].values
    ok = a.shape == b.shape and np.array_equal(a, b)
    rel = float(np.nanmax(np.abs(ve.values / se.values - 1)))
    print(f"  {str(s):60s} trades {len(vl)} vs {len(sl_)} log identical {ok} max rel eq diff {rel:.1e} "
          f"x{ve.iloc[-1]:.4f} vs x{se.iloc[-1]:.4f}  ({t1-t0:.0f}s / {t2-t1:.0f}s)", flush=True)
    assert ok and rel < 1e-9
    assert (slo.values <= se.values + 1e-12).all()

# ---- 2-4 on 2024 (one year) for a minute scalper
i0 = hidx.searchsorted(pd.Timestamp("2024-01-01")); i1 = hidx.searchsorted(pd.Timestamp("2025-01-01"))
o = np.asarray(A["o"][60 * i0:60 * i1]); h = np.asarray(A["h"][60 * i0:60 * i1])
low = np.asarray(A["l"][60 * i0:60 * i1]); c = np.asarray(A["c"][60 * i0:60 * i1])
for W, depth, wick in ((60, 0.02, 0.005), (240, 0.03, 0.0), (15, 0.01, 0.01)):
    kw = S.scalp(W=W, depth=depth, tp=0.007, hmm=24 * 60, N=4, fmax=0.25, wick=wick)
    e, lg, inv, lo, ntr = S.srun(A, hidx, Ph, SLIP, i0=i0, i1=i1, **kw)
    ref = pd.DataFrame(c).rolling(W, min_periods=1).max().shift(1).values
    fm = lg.fm.astype(int).values; k = lg.k.astype(int).values; xm = lg.xm.astype(int).values
    lim = ref[fm, k] * (1 - depth)
    sl_ = SLIP[i0:i1][fm // 60, k]
    rest = np.isclose(lg.entry.values, lim, rtol=1e-12)
    mkt = np.isclose(lg.entry.values, o[fm, k].astype(np.float64) * (1 + sl_), rtol=1e-12) & (o[fm, k] <= lim * (1 + 1e-12))
    lo_t = np.maximum(low[fm, k], np.minimum(o[fm, k], c[fm, k]) * np.float32(1 - wick)) if wick else low[fm, k]
    through = lo_t <= lim * (1 - E.EPS) * (1 + 1e-9)
    tp = lg.why.values == 0
    tgt = lg.entry.values * 1.007
    hi_t = np.minimum(h[xm, k], np.maximum(o[xm, k], c[xm, k]) * np.float32(1 + wick)) if wick else h[xm, k]
    tp_ok = hi_t[tp] >= tgt[tp] * (1 + E.EPS) * (1 - 1e-9)
    print(f"  W={W} depth={depth} wick={wick}: {len(lg)} trades; entries at the lagged W-max trigger {rest.sum()}, "
          f"market at/under trigger {mkt.sum()}, other {(~rest & ~mkt).sum()}; resting fills with trimmed low through "
          f"{through[rest].mean()*100:.1f}%; exits in the fill minute {(xm == fm).sum()}; take-profits with trimmed "
          f"high through {tp_ok.mean()*100:.1f}%", flush=True)
    assert (rest | mkt).all() and through[rest].all() and (xm > fm).all() and tp_ok.all()
kw = S.scalp(W=60, depth=0.02, tp=0.007, hmm=1440, N=4, fmax=0.25, wick=0.0)
e0, *_ = S.srun(A, hidx, Ph, SLIP, i0=i0, i1=i1, **kw)
kw["wick"] = 1e6
e1, *_ = S.srun(A, hidx, Ph, SLIP, i0=i0, i1=i1, **kw)
print(f"  wick 0 vs wick 1e6: x{e0.iloc[-1]:.6f} vs x{e1.iloc[-1]:.6f}")
assert abs(e0.iloc[-1] / e1.iloc[-1] - 1) < 1e-12
print("M10 SCALP KERNEL CHECK PASSED")
