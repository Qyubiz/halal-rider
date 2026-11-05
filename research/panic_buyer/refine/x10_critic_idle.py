"""x10 critic: re-run the idle-cash honest pick (2x50% +5%, BTC 50-day-filter parking, watcher at 2%, 7-day re-buy
wait) and its base, normal and stressed, with the idle agent's own library (numbers in pb_out_x10/idle/report.log)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import x10_idle_lib as X  # noqa: E402

b = X.BASES["2x50% +5%"]
park = dict(kind="fixed", assets={"BTCUSDT": 1.0}, filt=50, pf=1.0)
pk = dict(mode=1, armx=0.02, band=0.05, rebuy_wait=168)
for name, pa, p in (("idle honest pick", park, pk), ("its base, no parking", None, None)):
    r, e, lg, aux = X.evaluate(b, pa, p)
    rs, _, _, _ = X.evaluate(b, pa, p, stress=True)
    print(X.fmt4(name, r, rs), flush=True)
