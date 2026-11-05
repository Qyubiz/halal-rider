# Panic Buyer refinement (HANDOFF.md section 8)

Test harness for the section 8 ideas. The live bot (`panic_buyer.py`) stays unchanged unless a change beats
the live settings in **both** 2021–23 and 2024–26.

## Rule every idea is held to

1. A small grid of variants is written down before any result is seen (top of each `idea*.py`).
2. The variant is chosen on **2021–23 only**. It must beat the live settings there, keep 2022 above −5% and
   the 2021–23 drawdown above −35%. The best 2021–23 CAGR / |max drawdown| wins (`engine.pick_on_design`).
3. 2024–26 is looked at once, for that pick. "Beats" means a higher compounded return **and** a max drawdown
   no more than 1 point deeper, in each period (`engine.beats`).
4. The pick must still beat the live settings with LUNA/FTT/SRM added and with every fill 0.2% worse.

Only then is it a candidate for the live bot.

## Files

| File | What it does |
|---|---|
| `fetch_vision.py`, `fetch_all.sh` | Download 1h and 1m candles from data.binance.vision (+ data-api.binance.vision for the last hours) into `$PB_DATA` (default `~/pb_data`) |
| `engine.py` | `gkernel`: the validated live engine plus switches for every idea. `mkernel`: the same bot on 1-minute candles. Scoring and the "beats" rule |
| `check_engine.py` | Code check: both new engines reproduce `code/lab.py:dyn_kernel` exactly with the live settings |
| `study.py` | The protocol above, shared by the idea scripts |
| `baseline.py` | Live settings on fresh data vs the handoff numbers; idea 9 (BNB fee discount, worse fills, slippage) |
| `idea2_fast_exit.py` | Take-profit placed 1/5/15 min after the fill (1-minute candles) |
| `idea3_depth.py` | Trigger depth scaled by each coin's volatility or drop percentile |
| `idea4_stress.py` | Order size/count/depth depending on whether BTC itself is falling |
| `idea5_universe.py` | Monthly top-M screened coins by trailing volume, delisted coins included |
| `idea6_exits.py` | Partial take-profit + trailing/second target; shorter time stop in weak markets |
| `idea8_stats.py` | 960-setting grid on the 19 coins, walk-forward, heatmaps, block bootstrap, deflated Sharpe, PBO |
| `reconcile_live.py` | Idea 1: live `panic_trades.csv` vs the backtest on the same weeks, trade by trade |
| `run_all.sh` | Everything in order, logs to `$PB_OUT` (default `~/pb_out`) |

## Run

```bash
pip install numpy pandas numba scipy matplotlib
bash fetch_all.sh          # ~1.5 GB; UNIVERSE=0 skips the 322-coin download
bash run_all.sh
```
