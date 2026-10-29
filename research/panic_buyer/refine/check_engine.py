"""Code check (not a result): on synthetic crash-and-bounce prices, the new engines must reproduce the validated
lab.dyn_kernel exactly when every new switch is at the live setting."""
import numpy as np
import pandas as pd

import engine as E
import lab


def synthetic(K=10, days=240, seed=7):
    rng = np.random.default_rng(seed)
    nm = days * 1440
    r = rng.normal(0, 0.0012, (nm, K))
    for k in range(K):                                  # crashes: -15..-25% over 20-90 minutes, then a bounce
        for s in rng.choice(nm - 600, size=days // 6, replace=False):
            d, m = rng.uniform(0.15, 0.25), int(rng.integers(20, 90))
            r[s:s + m, k] += np.log(1 - d) / m
            r[s + m:s + m + 240, k] -= np.log(1 - d) * rng.uniform(0.2, 0.9) / 240
    c = 100 * np.exp(np.cumsum(r, axis=0))
    o = np.vstack([c[:1], c[:-1]])
    w = np.abs(rng.normal(0, 0.0008, (nm, K)))
    h = np.maximum(o, c) * (1 + w); l = np.minimum(o, c) * (1 - w)
    c[: 1440 * 3, 0] = np.nan; o[: 1440 * 3, 0] = np.nan; h[: 1440 * 3, 0] = np.nan; l[: 1440 * 3, 0] = np.nan
    return {"o": o, "h": h, "l": l, "c": c}


def hourly(A):
    nh = A["c"].shape[0] // 60
    return {"o": A["o"][::60], "c": A["c"][59::60],
            "h": np.fmax.reduce(A["h"].reshape(nh, 60, -1), axis=1),
            "l": np.fmin.reduce(A["l"].reshape(nh, 60, -1), axis=1)}


def main():
    A = synthetic()
    Hh = hourly(A)
    idx = pd.date_range("2021-01-01", periods=len(Hh["c"]), freq="1h")
    P = {f: pd.DataFrame(v, idx) for f, v in Hh.items()}
    P["cf"] = P["c"].ffill(); P["ref"] = P["c"].rolling(24, min_periods=1).max().shift(1)
    n, K = P["c"].shape
    L = E.LIVE
    ref_eq, ref_lg = lab.dyn_kernel(P["o"].values, P["h"].values, P["l"].values, P["cf"].values, P["ref"].values,
                                    L["depth"], L["N"], L["fmax"], L["tp"], L["hm"], L["cd"], E.EPS, E.FEE, E.SLIP_MAJOR, 0.0)
    eq, lg = E.run(P, idx)
    assert len(ref_lg) > 50, "synthetic data should produce trades"
    assert np.allclose(eq.values, ref_eq, rtol=0, atol=1e-12), "gkernel != lab.dyn_kernel"
    assert len(lg) == len(ref_lg) and np.allclose(lg.ret.values, ref_lg[:, 4])
    print(f"gkernel == lab.dyn_kernel: x{ref_eq[-1]:.4f}, {len(ref_lg)} trades")

    # partial take-profit switches are inert at part=1
    eq2, _ = E.run(P, idx, tp2=0.04, trail=0.03)
    assert np.allclose(eq2.values, ref_eq, atol=1e-12)
    eq3, lg3 = E.run(P, idx, part=0.5, trail=0.03)
    print(f"partial take-profit runs: x{eq3.iloc[-1]:.4f}, exits by reason {lg3.why.value_counts().to_dict()}")

    # the minute engine with the take-profit placed at the next hour == hourly engine on the same candles
    ph = {f: pd.DataFrame(v, idx) for f, v in Hh.items()}
    ph["cf"] = ph["c"].ffill(); ph["ref"] = ph["c"].rolling(24, min_periods=1).max().shift(1)
    m_eq, m_lg = E.mrun(A, idx, ph, delay=0)
    assert np.allclose(m_eq.values, ref_eq, atol=1e-12), "mkernel(delay=0) != hourly engine"
    print(f"mkernel(delay=0) == hourly engine: x{m_eq.iloc[-1]:.4f}, {len(m_lg)} trades")
    for d in (5, 1):
        e_, lg_ = E.mrun(A, idx, ph, delay=d)
        same_hour = int((lg_.eb == lg_.xb).sum())
        print(f"mkernel(delay={d}): x{e_.iloc[-1]:.4f}, {len(lg_)} trades, {same_hour} sold in the hour they were bought")
        assert same_hour > 0
    print("ENGINE CHECK PASSED")


if __name__ == "__main__":
    main()
