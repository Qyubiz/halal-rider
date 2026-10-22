"""Idea 8: statistical rigor for the live settings, on the 19 coins.
  - the original 960-setting grid re-run on the 19 coins (depth x orders x order cap x take-profit x time stop)
  - anchored yearly walk-forward: each year's settings chosen with the original rule on all earlier years only
  - parameter heatmaps (2021-23 and 2024-26)
  - block-bootstrap Monte Carlo of monthly returns (12-month outcomes)
  - deflated Sharpe ratio (Bailey & Lopez de Prado 2014) for the number of settings tried
  - probability of backtest overfitting by combinatorially symmetric cross-validation (Bailey et al. 2015)"""
import itertools
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

import engine as E
import study as S

GRID = list(itertools.product([0.08, 0.10, 0.12, 0.15], [1, 2, 3, 5, 8], [0.2, 0.33, 0.5, 1.0],
                              [0.015, 0.02, 0.03, 0.05], [48, 96, 168]))
LIVE_KEY = (0.12, 8, 0.2, 0.015, 48)
TRIALS_EXTRA = 1000   # earlier families + refinement variants (sections 4 and 8): a deliberately generous count


def grid_daily():
    syms, idx, P = S.market()
    cols, daily, rows = [], [], []
    for g in GRID:
        depth, N, fmax, tp, hm = g
        e, _ = E.run(P, idx, depth=depth, N=N, fmax=fmax, tp=tp, hm=hm)
        r = E.score(e); r.update(depth=depth, N=N, fmax=fmax, tp=tp, hm=hm)
        rows.append(r)
        daily.append(e.resample("1D").last().pct_change().fillna(0.0).values)
        cols.append(g)
    didx = e.resample("1D").last().index
    return pd.DataFrame(rows), pd.DataFrame(np.array(daily).T, index=didx, columns=pd.MultiIndex.from_tuples(cols))


def walk_forward(Rd):
    """Rd: daily returns per setting. Each year Y: choose on 2021..Y-1 (DD > -35%, 2022 > -5% when 2022 is in the
    window, best CAGR/|DD|), then take that setting's returns in Y."""
    eq = (1 + Rd).cumprod()
    out, picks = [], []
    for Y in range(2022, eq.index[-1].year + 1):
        ins = eq[eq.index < f"{Y}-01-01"]
        best, best_s = None, -1e9
        for col in eq.columns:
            e = ins[col] / ins[col].iloc[0]
            dd = float((e / e.cummax() - 1).min())
            yrs = (e.index[-1] - e.index[0]).days / 365.25
            cagr = e.iloc[-1] ** (1 / yrs) - 1
            if dd < -0.35:
                continue
            if Y > 2022:
                y22 = ins[col][(ins.index >= "2022-01-01") & (ins.index < "2023-01-01")]
                if y22.iloc[-1] / y22.iloc[0] - 1 < -0.05:
                    continue
            s = cagr / max(1e-9, -dd)
            if s > best_s:
                best, best_s = col, s
        oos = Rd.loc[(Rd.index >= f"{Y}-01-01") & (Rd.index < f"{Y + 1}-01-01"), best]
        out.append(oos); picks.append((Y, best))
    return pd.concat(out), picks


def heatmaps(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    views = [("depth", "tp", dict(N=8, fmax=0.2, hm=48)), ("N", "fmax", dict(depth=0.12, tp=0.015, hm=48)),
             ("depth", "hm", dict(N=8, fmax=0.2, tp=0.015))]
    fig, ax = plt.subplots(len(views), 2, figsize=(11, 4.2 * len(views)), layout="constrained")
    for i, (a, b, fix) in enumerate(views):
        sub = R
        for k, v in fix.items():
            sub = sub[np.isclose(sub[k], v)]
        for j, (col, lab) in enumerate((("d_cagr", "2021-23 CAGR"), ("t_cagr", "2024-26 CAGR"))):
            M = sub.pivot(index=a, columns=b, values=col)
            M.round(3).to_csv(os.path.join(S.OUT, f"heat_{a}_{b}_{col}.csv"))
            im = ax[i, j].imshow(M.values * 100, cmap="RdYlGn", aspect="auto", vmin=-50, vmax=100)
            ax[i, j].set_xticks(range(len(M.columns)), [f"{x:g}" for x in M.columns])
            ax[i, j].set_yticks(range(len(M.index)), [f"{x:g}" for x in M.index])
            ax[i, j].set_xlabel(b); ax[i, j].set_ylabel(a)
            ax[i, j].set_title(f"{lab} ({', '.join(f'{k}={v:g}' for k, v in fix.items())})", fontsize=9)
            for (y, x), v in np.ndenumerate(M.values):
                ax[i, j].text(x, y, f"{v*100:.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.6, label="CAGR %")
    fig.savefig(os.path.join(S.OUT, "heatmaps.png"), dpi=110, bbox_inches="tight")


def bootstrap(monthly, horizon=12, block=3, n=20000, seed=1):
    rng = np.random.default_rng(seed)
    r = np.asarray(monthly); T = len(r)
    paths = np.empty((n, horizon))
    for i in range(n):
        out = []
        while len(out) < horizon:
            s = rng.integers(0, T)
            out.extend(r[(s + np.arange(block)) % T])          # circular blocks keep runs of bad months together
        paths[i] = out[:horizon]
    eq = np.cumprod(1 + paths, axis=1)
    tot = eq[:, -1] - 1
    dd = (eq / np.maximum.accumulate(np.concatenate([np.ones((n, 1)), eq], axis=1)[:, 1:]) - 1).min(axis=1)
    return {"median": np.median(tot), "p5": np.quantile(tot, 0.05), "p25": np.quantile(tot, 0.25),
            "p95": np.quantile(tot, 0.95), "p_loss": float((tot < 0).mean()), "p_dd20": float((dd < -0.20).mean())}


def deflated_sharpe(r, sr_all, n_trials):
    """r: daily returns of the chosen setting; sr_all: daily Sharpe of every setting tried (for their variance)."""
    sr = r.mean() / r.std(ddof=1)
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    V = np.var(sr_all, ddof=1)
    em = 0.5772156649
    sr0 = math.sqrt(V) * ((1 - em) * norm.ppf(1 - 1 / n_trials) + em * norm.ppf(1 - 1 / (n_trials * math.e)))
    T = len(r)
    z = (sr - sr0) * math.sqrt(T - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    return {"sr_daily": sr, "sr_annual": sr * math.sqrt(365), "sr0_daily": sr0, "dsr": float(norm.cdf(z)),
            "skew": g3, "kurt": g4, "T": T}


def pbo_cscv(Rd, S_=16):
    """Probability that the in-sample best setting is below the median out of sample."""
    X = Rd.values; T, N = X.shape
    T = T - T % S_
    B = X[:T].reshape(S_, T // S_, N)
    s1, s2, cnt = B.sum(1), (B ** 2).sum(1), T // S_
    lam, deg = [], []
    for c in itertools.combinations(range(S_), S_ // 2):
        m = np.zeros(S_, bool); m[list(c)] = True
        def sharpe(mask):
            n = cnt * mask.sum(); mu = s1[mask].sum(0) / n
            return mu / np.sqrt(np.maximum(s2[mask].sum(0) / n - mu ** 2, 1e-18))
        is_, oos = sharpe(m), sharpe(~m)
        best = int(np.argmax(is_))
        w = (oos < oos[best]).sum() / (N + 1) + 0.5 / (N + 1)
        lam.append(math.log(w / (1 - w))); deg.append((is_[best], oos[best]))
    lam = np.array(lam); deg = np.array(deg)
    return {"pbo": float((lam <= 0).mean()), "combos": len(lam), "is_best_sr": float(deg[:, 0].mean()),
            "oos_of_is_best_sr": float(deg[:, 1].mean()), "oos_below_zero": float((deg[:, 1] <= 0).mean())}


if __name__ == "__main__":
    print("\n######## idea 8: statistical rigor (19 coins)")
    R, Rd = grid_daily()
    R.to_csv(os.path.join(S.OUT, "grid960_19coins.csv"), index=False)
    live = R[(R.depth == 0.12) & (R.N == 8) & (R.fmax == 0.2) & (R.tp == 0.015) & (R.hm == 48)].iloc[0]
    print(E.line("live settings", live))
    ok = R[(R.y2022 >= -0.05) & (R.d_dd >= -0.35)]
    print(f"grid: {len(R)} settings | positive in 2024-26: {(R.t_cagr > 0).mean()*100:.0f}% | of the {len(ok)} that pass "
          f"the 2021-23 rule: {(ok.t_cagr > 0).mean()*100:.0f}% positive, median 2024-26 CAGR {ok.t_cagr.median()*100:+.0f}%")
    pick = ok.assign(s=ok.d_cagr / -ok.d_dd).sort_values("s", ascending=False).iloc[0]
    print(f"original rule on the 19 coins picks: depth {pick.depth:.0%} N={int(pick.N)} cap {pick.fmax:.0%} tp {pick.tp:.1%} "
          f"{int(pick.hm)}h -> 24-26 x{pick.t_mult:.2f} DD {pick.t_dd*100:.0f}%")
    heatmaps(R)

    wf, picks = walk_forward(Rd)
    wf_eq = (1 + wf).cumprod()
    fixed = Rd[LIVE_KEY][Rd.index >= "2022-01-01"]
    fx_eq = (1 + fixed).cumprod()
    print("\nanchored walk-forward (each year chosen on earlier years only):")
    for Y, k in picks:
        y_wf = (1 + wf[wf.index.year == Y]).prod() - 1; y_fx = (1 + fixed[fixed.index.year == Y]).prod() - 1
        print(f"  {Y}: chose depth {k[0]:.0%} N={k[1]} cap {k[2]:.0%} tp {k[3]:.1%} {k[4]}h -> {y_wf*100:+.0f}%  (live settings {y_fx*100:+.0f}%)")
    print(f"  2022-26 chained: walk-forward x{wf_eq.iloc[-1]:.2f} DD {(wf_eq/wf_eq.cummax()-1).min()*100:.0f}% | "
          f"live settings x{fx_eq.iloc[-1]:.2f} DD {(fx_eq/fx_eq.cummax()-1).min()*100:.0f}%")

    base_e = (1 + Rd[LIVE_KEY]).cumprod()
    m = E.lab.month_returns(base_e)
    print("\nblock bootstrap of monthly returns (3-month blocks), next 12 months:")
    for lab_, mm in (("from 2024-26 months", m[m.index >= "2024-01-01"]), ("from all 2021-26 months", m),
                     ("from 2022-23 months only", m[(m.index >= "2022-01-01") & (m.index < "2024-01-01")])):
        b = bootstrap(mm.values)
        print(f"  {lab_:26s} median {b['median']*100:+.0f}%  5th pct {b['p5']*100:+.0f}%  95th {b['p95']*100:+.0f}%  "
              f"P(loss) {b['p_loss']*100:.1f}%  P(drop >20%) {b['p_dd20']*100:.1f}%")

    sr_all = (Rd.mean() / Rd.std(ddof=1)).values
    for lab_, mask in (("full 2021-26", slice(None)), ("2024-26 only", Rd.index >= "2024-01-01")):
        r = Rd[LIVE_KEY][mask]
        sra = (Rd[mask].mean() / Rd[mask].std(ddof=1)).values
        for n_tr in (len(GRID), len(GRID) + TRIALS_EXTRA):
            d = deflated_sharpe(r.values, sra, n_tr)
            print(f"deflated Sharpe {lab_}, {n_tr} trials: Sharpe {d['sr_annual']:.2f}/yr, DSR {d['dsr']:.3f} "
                  f"(skew {d['skew']:.1f}, kurtosis {d['kurt']:.0f})")
    p = pbo_cscv(Rd)
    print(f"PBO (CSCV, 16 blocks, {p['combos']} splits): {p['pbo']*100:.0f}% | in-sample best daily Sharpe "
          f"{p['is_best_sr']:.3f} -> out of sample {p['oos_of_is_best_sr']:.3f}; out-of-sample Sharpe <= 0 in "
          f"{p['oos_below_zero']*100:.0f}% of splits")
