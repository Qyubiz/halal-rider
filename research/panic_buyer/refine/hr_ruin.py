"""Task "ruin": what would RM6,000 funded from a credit card go through in the next 12 months?

A measurement only: nothing is picked or tuned. Settings (fixed by the task; depth 12%, cooldown 24h):
  LIVE  N=8 fmax 0.20 tp 1.5% 48h  (engine.LIVE)
  A     N=1 fmax 1.00 tp 3%   48h
  C     N=2 fmax 0.50 tp 3%   48h
  E     N=2 fmax 0.50 tp 5%   48h
  G     N=3 fmax 0.33 tp 3%   48h
Method:
  1. engine.run on the 19-coin hourly panel (study.market), the same run the 960-setting grid used (self-checked
     against results/grid960_19coins.csv). Daily returns = close-to-close of the hourly equity at each UTC day's last
     hour (start = 1.0 before the first hour). The last day (2026-10-22, 2 hours of data) is dropped.
  2. Pools of daily returns: 2021-01-01..2026-10-21 and 2022-01-01..2023-12-31 (the task's two), plus two extra pools
     for context: 2024-01-01..2026-10-21 (recent market only) and 2022 alone (the deepest bear year).
  3. Circular block bootstrap: 13 blocks of 30 consecutive days (start drawn uniformly, wrapping from the pool's end
     to its start), cut to 365 days; 10,000 paths. The same block draws (one seed per pool) are used for every
     setting, so settings are compared on the same simulated markets.
  4. Per path: 12-month result = product(1 + r) - 1; worst drop = deepest fall of the daily equity below its running
     peak, the starting RM6,000 included; lowest balance = lowest daily close vs the RM6,000 put in (P(ever below
     RM4,200 / RM3,000), 5th percentile of the lowest balance). Drops inside a day are not seen (daily closes), so the true worst drops
     are somewhat deeper.
  5. Reported: median, 5th and 95th percentile of the 12-month result; P(12-month loss); P(worst drop > 30%),
     P(worst drop > 50%); P(12-month result >= 1.05**12 - 1 = +79.6%, i.e. an average of 5%/month or more) and
     >= 1.10**12 - 1 (10%/month); P(result < +18%) as a reference for an 18%/year card rate (assumed rate, not
     data); median and 5th-percentile worst drop; RM values on RM6,000.
  6. Sensitivities on the same pools: +LUNA/FTT/SRM (22-coin panel, collapse coins cut as in engine.DEAD) and the
     real bot's timing (mtp_now=True, cd=25, REFINE.md). A second seed checks the Monte Carlo noise.
  7. For comparison without resampling: every actual 365-day window inside each pool (start on any day).
Only information known at each hour's start is used (the engine is unchanged).
"""
import os
import time

import numpy as np
import pandas as pd

import engine as E
import study as S

OUT = S.OUT
HERE = os.path.dirname(os.path.abspath(__file__))
RM0 = 6000.0
NPATH, H, B = 10_000, 365, 30
SEED, SEED2 = 20261022, 777
TARGET5, TARGET10, CARD = 1.05 ** 12 - 1, 1.10 ** 12 - 1, 0.18

SETS = {
    "LIVE N8 f0.20 tp1.5%": dict(E.LIVE),
    "A N1 f1.00 tp3%": dict(depth=0.12, N=1, fmax=1.0, tp=0.03, hm=48, cd=24),
    "C N2 f0.50 tp3%": dict(depth=0.12, N=2, fmax=0.5, tp=0.03, hm=48, cd=24),
    "E N2 f0.50 tp5%": dict(depth=0.12, N=2, fmax=0.5, tp=0.05, hm=48, cd=24),
    "G N3 f0.33 tp3%": dict(depth=0.12, N=3, fmax=0.33, tp=0.03, hm=48, cd=24),
}
POOLS = {  # name: (first day, last day) inclusive
    "2021-26": ("2021-01-01", "2026-10-21"),
    "2022-23": ("2022-01-01", "2023-12-31"),
    "2024-26": ("2024-01-01", "2026-10-21"),
    "2022": ("2022-01-01", "2022-12-31"),
}
VARIANTS = {  # name: (dead coins?, extra engine settings)
    "engine (19 coins)": (False, {}),
    "+LUNA/FTT/SRM": (True, {}),
    "bot timing": (False, dict(mtp_now=True, cd=25)),
}

_log = None


def say(s=""):
    print(s, flush=True)
    _log.write(s + "\n"); _log.flush()


def daily_returns(e):
    """Close-to-close daily returns of the hourly equity; days with fewer than 24 hours (the last one) dropped."""
    cnt = e.resample("D").count()
    last = e.resample("D").last()
    prev = last.shift(1)
    prev.iloc[0] = 1.0                                   # the account starts at 1.0 before the first hour
    r = (last / prev - 1.0)[cnt == 24]
    return r


def block_index(T, seed):
    rng = np.random.default_rng(seed)
    nb = -(-H // B)
    st = rng.integers(0, T, size=(NPATH, nb))
    return ((st[:, :, None] + np.arange(B)[None, None, :]) % T).reshape(NPATH, nb * B)[:, :H]


def path_stats(paths):
    """paths: (n, days) daily returns -> 12-month result and worst drop per path."""
    eq = np.cumprod(1.0 + paths, axis=1)
    eq0 = np.concatenate([np.ones((len(eq), 1)), eq], axis=1)
    peak = np.maximum.accumulate(eq0, axis=1)
    dd = (eq0 / peak - 1.0).min(axis=1)
    return eq[:, -1] - 1.0, dd, eq0.min(axis=1)


def summarize(tot, dd, low):
    q = lambda a, p: float(np.quantile(a, p))  # noqa: E731
    return dict(median=q(tot, 0.5), p5=q(tot, 0.05), p95=q(tot, 0.95), p_loss=float((tot < 0).mean()),
                p_dd30=float((dd < -0.30).mean()), p_dd50=float((dd < -0.50).mean()),
                p_5pm=float((tot >= TARGET5).mean()), p_10pm=float((tot >= TARGET10).mean()),
                p_below_card=float((tot < CARD).mean()), dd_median=q(dd, 0.5), dd_p5=q(dd, 0.05),
                rm_p5=RM0 * (1 + q(tot, 0.05)), rm_median=RM0 * (1 + q(tot, 0.5)), rm_p95=RM0 * (1 + q(tot, 0.95)),
                p_low30=float((low < 0.70).mean()), p_low50=float((low < 0.50).mean()), rm_low_p5=RM0 * q(low, 0.05),
                rm_low_median=RM0 * q(low, 0.5))


def actual_windows(r):
    """Every real 365-day window in the pool (no wrap): 12-month result and worst drop."""
    a = r.values
    T = len(a)
    if T < H:
        return None
    st = np.arange(T - H + 1)
    tot, dd, low = path_stats(a[st[:, None] + np.arange(H)[None, :]])
    return dict(n=len(st), median=float(np.median(tot)), min=float(tot.min()), max=float(tot.max()),
                p_loss=float((tot < 0).mean()), p_5pm=float((tot >= TARGET5).mean()), worst_dd=float(dd.min()),
                p_dd30=float((dd < -0.30).mean()), p_dd50=float((dd < -0.50).mean()), lowest=float(low.min()) - 1.0)


def self_test_grid(kw, sc):
    g = pd.read_csv(os.path.join(HERE, "results", "grid960_19coins.csv"))
    m = g[np.isclose(g.depth, kw["depth"]) & (g.N == kw["N"]) & np.isclose(g.fmax, kw["fmax"]) &
          np.isclose(g.tp, kw["tp"]) & (g.hm == kw["hm"])]
    if len(m) != 1:
        raise SystemExit(f"setting {kw} not in grid960 ({len(m)} rows)")
    m = m.iloc[0]
    for k in ("d_mult", "d_dd", "t_mult", "t_dd"):
        if abs(sc[k] - m[k]) > 1e-6 * max(1.0, abs(m[k])):
            raise SystemExit(f"{kw}: {k} {sc[k]} does not reproduce grid960 {m[k]}")
    return f"grid960 reproduced: 21-23 x{m.d_mult:.2f} DD {m.d_dd*100:.0f}%, 24-26 x{m.t_mult:.2f} DD {m.t_dd*100:.0f}%"


def line(s):
    return (f"median {s['median']*100:+7.1f}%  p5 {s['p5']*100:+7.1f}%  p95 {s['p95']*100:+8.1f}% | P(loss) "
            f"{s['p_loss']*100:5.1f}%  P(drop>30%) {s['p_dd30']*100:5.1f}%  P(drop>50%) {s['p_dd50']*100:5.1f}%  "
            f"P(>=5%/mo) {s['p_5pm']*100:5.1f}%  P(>=10%/mo) {s['p_10pm']*100:5.1f}%  P(<+18%) {s['p_below_card']*100:5.1f}% | "
            f"drop median {s['dd_median']*100:5.1f}% p5 {s['dd_p5']*100:5.1f}% | balance ever <RM4,200 {s['p_low30']*100:5.1f}%, "
            f"<RM3,000 {s['p_low50']*100:5.1f}%, lowest balance p5 RM{s['rm_low_p5']:,.0f} | RM6,000 -> p5 RM{s['rm_p5']:,.0f}  "
            f"median RM{s['rm_median']:,.0f}  p95 RM{s['rm_p95']:,.0f}")


def main():
    global _log
    os.makedirs(OUT, exist_ok=True)
    _log = open(os.path.join(OUT, "hr_ruin.log"), "w")
    t0 = time.time()
    say(f"hr_ruin  PB_DATA={E.DATA}  PB_OUT={OUT}")
    say(f"bootstrap: {NPATH} paths x {H} days, circular blocks of {B} days, seeds {SEED} (main) / {SEED2} (noise check)")
    say(f"targets: 5%/month over 12 months = {TARGET5*100:+.1f}%, 10%/month = {TARGET10*100:+.1f}%; card-rate "
        f"reference {CARD*100:.0f}%/year (assumed)")
    rets, rows, wins, noise = {}, [], [], []
    for vname, (dead, extra) in VARIANTS.items():
        syms, idx, P = S.market(dead)
        say(f"\n=== {vname}: {len(syms)} coins, {idx[0]} .. {idx[-1]}")
        for sname, kw in SETS.items():
            e, lg = E.run(P, idx, **{**kw, **extra})
            sc = E.score(e, lg, idx)
            chk = self_test_grid(kw, sc) if vname == "engine (19 coins)" else ""
            r = daily_returns(e)
            rets[(vname, sname)] = r
            say(f"  {sname:22s} 21-23 x{sc['d_mult']:.2f} DD {sc['d_dd']*100:.0f}% | 24-26 x{sc['t_mult']:.2f} "
                f"DD {sc['t_dd']*100:.0f}% | daily returns {r.index[0]:%Y-%m-%d}..{r.index[-1]:%Y-%m-%d} ({len(r)} days), "
                f"worst day {r.min()*100:+.1f}% ({r.idxmin():%Y-%m-%d})  {chk}")
    for pname, (a, b) in POOLS.items():
        T = None
        for vname in VARIANTS:
            for sname in SETS:
                r = rets[(vname, sname)]
                rp = r[(r.index >= a) & (r.index <= b)]
                if T is None:
                    T = len(rp)
                    I1, I2 = block_index(T, SEED), block_index(T, SEED2)
                    say(f"\n######## pool {pname}: {rp.index[0]:%Y-%m-%d}..{rp.index[-1]:%Y-%m-%d}, {T} days; "
                        f"pool's own result x{np.prod(1 + rp.values):.3g}")
                assert len(rp) == T
                tot, dd, low = path_stats(rp.values[I1])
                s = summarize(tot, dd, low)
                s.update(pool=pname, variant=vname, setting=sname, days=T)
                rows.append(s)
                if vname == "engine (19 coins)":
                    tot2, dd2, low2 = path_stats(rp.values[I2])
                    s2 = summarize(tot2, dd2, low2)
                    noise.append(dict(pool=pname, setting=sname,
                                      **{k: s2[k] - s[k] for k in ("median", "p5", "p95", "p_loss", "p_dd30",
                                                                  "p_dd50", "p_5pm")}))
                w = actual_windows(rp)
                if w:
                    w.update(pool=pname, variant=vname, setting=sname)
                    wins.append(w)
                say(f"  [{vname:17s}] {sname:22s} {line(s)}")
            say("")
    R = pd.DataFrame(rows)
    cols = ["pool", "variant", "setting", "days"] + [c for c in R.columns if c not in ("pool", "variant", "setting", "days")]
    R[cols].to_csv(os.path.join(OUT, "hr_ruin_bootstrap.csv"), index=False)
    W = pd.DataFrame(wins)
    W[["pool", "variant", "setting"] + [c for c in W.columns if c not in ("pool", "variant", "setting")]].to_csv(
        os.path.join(OUT, "hr_ruin_actual_windows.csv"), index=False)
    N = pd.DataFrame(noise)
    N.to_csv(os.path.join(OUT, "hr_ruin_seed_noise.csv"), index=False)

    say("\n######## actual 365-day windows (no resampling; start on every day of the pool)")
    for _, w in W.iterrows():
        say(f"  {w.pool:8s} [{w.variant:17s}] {w.setting:22s} {int(w.n):4d} windows | median {w['median']*100:+7.1f}%  "
            f"min {w['min']*100:+7.1f}%  max {w['max']*100:+8.1f}% | P(loss) {w.p_loss*100:5.1f}%  P(>=5%/mo) "
            f"{w.p_5pm*100:5.1f}% | worst drop {w.worst_dd*100:5.1f}%  P(drop>30%) {w.p_dd30*100:5.1f}%  "
            f"P(drop>50%) {w.p_dd50*100:5.1f}%")
    say("\n######## Monte Carlo noise: second seed minus first, engine (19 coins), largest absolute difference")
    for k in ("median", "p5", "p95", "p_loss", "p_dd30", "p_dd50", "p_5pm"):
        say(f"  {k:7s} max |diff| {N[k].abs().max()*100:.2f} points")

    say("\n######## headline (engine, 19 coins): the two pools the task asked for")
    for pname in ("2021-26", "2022-23"):
        say(f"--- pool {pname}")
        for sname in SETS:
            s = R[(R.pool == pname) & (R.variant == "engine (19 coins)") & (R.setting == sname)].iloc[0]
            say(f"  {sname:22s} {line(s)}")
    say(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
