"""w90 ten-watch: smoke test of the holdout interface on the 2021-26 data. The 1-minute runners must reproduce the
base rows of w90_watch_final.py (F1-F3), and the hourly fallbacks its "same settings, HOURLY resting bid" rows."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402
import w90_watch_final as F  # noqa: E402
import w90_watch_lib as W  # noqa: E402

A, hidx, Ph, SLIP = W.load()
A19 = {f: A[f][:, :19] for f in ("o", "h", "l", "c")}
Ph19 = {f: v.iloc[:, :19] for f, v in Ph.items()}
for name, spec, runner in F.holdout_candidates():
    e, lg = runner(A19, hidx, Ph19, **spec)
    print(f"1-min  {name:48s} {W.line(W.report(e, lg, hidx))}", flush=True)
    s = dict(spec); s.update(W.STRESS)
    print(f"  stressed spec (19 coins here) keys ok: {sorted(s)}")
syms, idx, P = E.panel(E.load_h19())
for name, spec, runner in F.holdout_candidates_hourly():
    e, lg = runner(P, idx, **spec(P, syms))
    lg = lg.copy(); lg["eqb"] = e.shift(1).fillna(1.0).values[lg.eb.astype(int).values]
    print(f"hourly {name:48s} {W.line(W.report(e, lg, idx, syms=syms))}", flush=True)
name, spec, runner = F.holdout_candidate()
print(f"holdout_candidate() -> {name}: {spec}, runner {runner.__name__}")
