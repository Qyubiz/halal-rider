"""Round 2, lens "trigger", hypothesis 1: a shorter reference window for the trigger.

Live: trigger = highest close of the last 24 closed hourly candles x (1 - 12%). Thesis: the liquidation overshoot
the bot buys is a FAST drop; a 12% fall spread over a day is more often a coin repricing. Motivation, 2021-23 only
(r2_trigger_motiv.log):
  - live trades by hours since the 24h highest close at entry: 1-6h avg +2.30% (152 trades), 7-12h +1.06%,
    13-18h +0.51% (time stops 9.2%), 19-24h +0.98%;
  - live trades whose 24h high was already >=10% under the 72h high (multi-day slide): avg 0.00%, time stops 9.2%
    averaging -18.1% -> longer windows (which buy such slides earlier) should be worse;
  - single-coin bots (live rules), reference window 12/24/36/48/72h: avg trade +1.45/+1.04/+0.84/+0.64/+0.40%,
    time stops 5.3/6.4/8.0/9.2/11.3%, 12h with 1,180 trades vs 1,723 (mean log multiple 0.851 vs 0.853).
A shorter window means fewer, better trades; the portfolio question is whether the freed cash is better used.
The 12h window with an 11% depth recovers some of the lost trades. Longer windows (36/48/72h) are the
counter-check. Reference from hourly highs was not motivated (single-coin: avg +0.79%, log multiple 0.777).

Grid fixed before any 2024-26 number was printed (8 variants). Engine: r2_trigger_lib.trun (= engine.run plus a
custom `ref`; reproduces engine.run exactly, r2_trigger_check.log)."""
import study as S
import r2_trigger_lib as L


def window(w, depth=None):
    def f(P, syms):
        kw = {"ref": L.ref_close(P, w).values}
        if depth is not None:
            kw["depth"] = depth
        return kw
    return f


V = {
    "ref = highest close 8h": window(8),
    "ref = highest close 12h": window(12),
    "ref = highest close 16h": window(16),
    "ref = highest close 20h": window(20),
    "ref = highest close 36h": window(36),
    "ref = highest close 48h": window(48),
    "ref = highest close 72h": window(72),
    "ref = highest close 12h, depth 11%": window(12, 0.11),
}

if __name__ == "__main__":
    L.use_trun()
    R, v = S.study("r2 trigger H1: reference window (highest close of the last w hours)", V, "r2_trigger_h1_window")
    S.save_verdict(v)
