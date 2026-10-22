"""w90 ten-watch: proofs for w90_watch_lib.wkernel.

1. mode=0 (switches off) == x10_critic_lib.xkernel minute watch mode: identical trade logs and equity, 19 and 22 coins,
   original and wick-trimmed minutes, several settings (sizing, cycle, re-arm, TP delay, worse fills, guard on/off).
2. mode=1 (the new watch bot) audited trade by trade against the raw minute arrays, independently of the kernel:
   entry = next minute's open x (1 + 0.05%); the decision minute's low traded through the trigger; no eligible coin
   crossed earlier while the slot was free (so the bot buys the FIRST crossing); take-profits fill at the limit on the
   first minute (after the fill minute) whose high is 0.1% through; time stops at the first hourly close >= hm hours;
   cooldowns respected; one position at a time; trade return = exit/entry x (1-fee)^2 - 1.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w90_watch_lib as W  # noqa: E402
import x10_critic_lib as X  # noqa: E402

A, hidx, Ph, SLIP = W.load()
print(f"minutes {A['c'].shape}, hours {len(hidx)} {hidx[0]} .. {hidx[-1]}", flush=True)
ok_all = True

# ------------------------------------------------------------------ 1. mode 0 == xkernel watch
CASES = [("N1 f1 d14 tp5 cyc60 g35", 0.35, dict(N=1, fmax=1.0, depth=0.14, tp=0.05, hm=48, cd=24, cyc=60)),
         ("N1 f1 d14 tp5 cyc1 rearm g35", 0.35, dict(N=1, fmax=1.0, depth=0.14, tp=0.05, hm=48, cd=24, cyc=1, rearm=1)),
         ("N3 f.33 d12 tp1.5 cyc60 nog worse", None, dict(N=3, fmax=0.33, depth=0.12, tp=0.015, hm=48, cd=24, cyc=60,
                                                           pen=0.002, spen=0.002, wslip=0.003)),
         ("N1 f1 d14 tp5 cyc1 rearm delay1 g35", 0.35, dict(N=1, fmax=1.0, depth=0.14, tp=0.05, hm=24, cd=24, cyc=1,
                                                             rearm=1, delay=1))]
for K in (19, 22):
    for w in (None, 0.01):
        B = W.wick(A, K, w)
        for name, g, kw in CASES:
            ref = W.guard_ref(Ph, g, K)
            t0 = time.time()
            ex, lx, _ = X.xrun(B, hidx, Ph, SLIP, K=K, ref=None if g is None else X.guard_ref(Ph, g, K), watch=1, **kw)
            t1 = time.time()
            ew, lw, _, _ = W.wrun(A, hidx, Ph, SLIP, K=K, B=B, ref=ref, mode=0, **kw)
            t2 = time.time()
            same = len(lx) == len(lw) and np.array_equal(lx.values, lw.values)
            d = float(np.nanmax(np.abs(ew.values / ex.values - 1)))
            good = same and d < 1e-12
            ok_all &= good
            print(f"K={K} wick={w} {name:38s} trades {len(lx):5d} vs {len(lw):5d} log identical {same} max rel eq diff "
                  f"{d:.1e} final x{ex.iloc[-1]:.6g} vs x{ew.iloc[-1]:.6g} ({t1-t0:.1f}s / {t2-t1:.1f}s) "
                  f"{'OK' if good else 'MISMATCH'}", flush=True)
        del B
# the critic's own guard_ref and this lib's are the same mask
for g in (0.25, 0.35, 0.45):
    a, b = X.guard_ref(Ph, g, 22), W.guard_ref(Ph, g, 22)
    same = np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(np.nan_to_num(a), np.nan_to_num(b), rtol=0, atol=0)
    ok_all &= same
    print(f"guard_ref {g}: identical to x10_critic_lib.guard_ref {same}")


# ------------------------------------------------------------------ 2. audit of mode 1
def audit(K, w, kw, label):
    global ok_all
    p = dict(W.WDEF); p.update(kw)
    B = W.wick(A, K, w)
    ref = W.guard_ref(Ph, p["guard"], K)
    e, lg, _, _ = W.wrun(A, hidx, Ph, None, K=K, B=B, ref=ref, **kw)
    o, h, l, c = B["o"], B["h"], B["l"], B["c"]
    lim_h = ref * (1 - p["depth"])                                        # (hours, K) trigger, NaN = guarded
    cdmin = p["cd"] * 60 + p["cdx"] if p["cdmin"] is None else p["cdmin"] + p["cdx"]
    eps, fee, ms, tp = p["eps"], p["fee"], p["mslip"], p["tp"]
    bad = []
    lg = lg.sort_values("fm").reset_index(drop=True)
    okm = np.zeros(K, np.int64)                                            # first minute a coin may be BOUGHT
    free_from = 0                                                          # first decision minute with a free slot
    for r in lg.itertuples():
        k, fm, xm, eb, xb = int(r.k), int(r.fm), int(r.xm), int(r.eb), int(r.xb)
        dm = fm - 1
        tag = f"trade fm={hidx[fm // 60]}+{fm % 60}m k={W.SYMS[k]}"
        # entry price and decision
        if not np.isclose(r.entry, float(o[fm, k]) * (1 + ms) * (1 + p["pen"]), rtol=1e-12, atol=0):
            bad.append(f"{tag}: entry {r.entry} != next open")
        lim = lim_h[dm // 60, k]
        hit = (l[dm, k] <= lim * (1 - p["deps"])) if p["det"] == 0 else (c[dm, k] <= lim)
        if not (np.isfinite(lim) and hit):
            bad.append(f"{tag}: decision minute did not cross the trigger")
        if fm < okm[k]:
            bad.append(f"{tag}: bought inside cooldown")
        if dm < free_from:
            bad.append(f"{tag}: bought while the slot was busy")
        # first crossing: no eligible coin crossed in [free_from, dm-1]
        if dm > free_from:
            seg = slice(free_from, dm)
            mins = np.arange(free_from, dm)
            L = lim_h[mins // 60]                                           # (m, K)
            X_ = (l[seg] <= L * (1 - p["deps"])) if p["det"] == 0 else (c[seg] <= L)
            X_ &= np.isfinite(L) & ~np.isnan(c[seg])
            X_ &= (mins[:, None] + 1) >= okm[None, :]
            if X_.any():
                mi, ki = np.argwhere(X_)[0]
                bad.append(f"{tag}: {W.SYMS[ki]} crossed earlier at minute {mins[mi]} while the slot was free")
        # exit
        tpp = r.entry * (1 + tp)
        seg = h[fm + p["tpd"]:xm + 1, k]
        cross = np.flatnonzero(seg >= tpp * (1 + eps))
        if r.why == 0:
            if not (np.isclose(r.px, tpp, rtol=1e-12) and len(cross) and fm + p["tpd"] + cross[0] == xm):
                bad.append(f"{tag}: take-profit not at the first 0.1%-through minute")
        else:
            late = np.arange(eb + p["hm"], xb) * 60 + 59                  # earlier hourly closes must be missing
            tsm_ok = (xm % 60 == 59 and xb - eb >= p["hm"] and np.isnan(c[late, k]).all()
                      and np.isclose(r.px, float(c[xm, k]) * (1 - ms), rtol=1e-12) and len(cross) == 0)
            if xb - eb > p["hm"] and tsm_ok:
                print(f"    note: {tag} held {xb-eb}h: no trading at the {p['hm']}h mark (halt), sold when it resumed")
            if not tsm_ok:
                bad.append(f"{tag}: time stop wrong (held {xb-eb}h, crosses {len(cross)})")
        if not np.isclose(r.ret, r.px / r.entry * (1 - fee) ** 2 * (1 - p["spen"]) - 1, rtol=1e-9, atol=1e-12):
            bad.append(f"{tag}: return inconsistent")
        okm[k] = xm + 1 + cdmin
        free_from = xm
    good = not bad
    ok_all &= good
    print(f"audit {label}: {len(lg)} trades, {len(bad)} violations {'OK' if good else 'FAIL'}", flush=True)
    for b in bad[:10]:
        print("   ", b)
    return e, lg


audit(19, 0.01, dict(), "10% mode watch, K19, wicks 1%")
audit(19, None, dict(tp=0.03, depth=0.12, hm=24), "tp3 d12 hm24, K19, original wicks")
audit(22, 0.01, dict(guard=0.0), "no guard, K22, wicks 1%")
audit(19, 0.01, dict(det=1, cdx=60), "close detection, cd+1h, K19")
print("WATCH CHECK PASSED" if ok_all else "WATCH CHECK FAILED")
