"""Replay the actual paper-bot code over historical 15m candles (one run per 15 min) and compare with the backtest kernel."""
import sys, os, importlib.util
sys.path.insert(0, "/home/claude/bt/lab2")
import numpy as np, pandas as pd
import lab

bot_path, data_dir, start, end, work = sys.argv[1], sys.argv[2], pd.Timestamp(sys.argv[3]), pd.Timestamp(sys.argv[4]), sys.argv[5]
os.makedirs(work, exist_ok=True); os.chdir(work)
for f in ("pb_state.json", "pb_trades.csv", "pb_log.txt"):
    if os.path.exists(f): os.remove(f)
spec = importlib.util.spec_from_file_location("pb", bot_path); pb = importlib.util.module_from_spec(spec); spec.loader.exec_module(pb)
coins = pb.CONFIG["coins"]
EPOCH = pd.Timestamp("1970-01-01")
arr, full = {}, {}
for s in coins:
    d = pd.read_csv(f"{data_dir}/{s}.csv", index_col=0, parse_dates=True)
    full[s] = d
    d = d[(d.index >= start - pd.Timedelta(days=3)) & (d.index < end)]
    ot = ((d.index - EPOCH) // pd.Timedelta(milliseconds=1)).to_numpy().astype(np.int64)
    arr[s] = (ot, d.o.values, d.h.values, d.l.values, d.c.values)

SIM = [0]
pb.now_ms = lambda: SIM[0]
def fake_klines(sym, start_ms):
    ot, o, h, l, c = arr[sym]
    i0 = np.searchsorted(ot, start_ms); i1 = min(np.searchsorted(ot, SIM[0] - pb.BAR, side="right"), i0 + 1000)
    return [(int(ot[i]), float(o[i]), float(h[i]), float(l[i]), float(c[i])) for i in range(i0, i1)]
pb.klines = fake_klines
pb.log = lambda msg: None

t0 = int((start - EPOCH) // pd.Timedelta(milliseconds=1)); t1 = int((end - EPOCH) // pd.Timedelta(milliseconds=1))
eqs = []
for T in range(t0, t1, pb.BAR):
    SIM[0] = T + 5000            # run 5 seconds after each 15m candle closes
    pb.main()
import json
st = json.load(open("pb_state.json"))
lastc = {s: arr[s][4][-1] for s in coins}
bot_eq = st["cash"] + sum(o["amount"] for o in st["orders"].values()) + sum(p["qty"] * lastc[s] for s, p in st["positions"].items())
tr = pd.read_csv("pb_trades.csv") if os.path.exists("pb_trades.csv") else pd.DataFrame()

# kernel on the same window
bph = 3600000 // pb.BAR
idx = pd.date_range(start - pd.Timedelta(days=3), end - pd.Timedelta(milliseconds=pb.BAR), freq=f"{pb.BAR//60000}min")
Pp = {f: pd.DataFrame({s: full[s][f] for s in coins}).reindex(idx) for f in ("o", "h", "l", "c")}
ref = Pp["c"].rolling(pb.WINDOW, min_periods=pb.WINDOW).max().shift(1)
m = idx >= start
cfg = pb.CONFIG
eq, lg = lab.dyn_kernel(Pp["o"].values[m], Pp["h"].values[m], Pp["l"].values[m], Pp["c"].ffill().values[m], ref.values[m],
                        cfg["depth"], cfg["n_orders"], cfg["max_per_order"], cfg["take_profit"], cfg["max_hold_h"] * bph,
                        cfg["cooldown_h"] * bph, cfg["through"], cfg["fee"], cfg["slip"], 0.0)
print(f"window {start:%Y-%m-%d} -> {end:%Y-%m-%d}")
print(f"BOT CODE : equity {bot_eq/cfg['start_cash']:.4f}x  trades {len(tr)}  win {np.mean(tr.return_pct>0)*100 if len(tr) else 0:.0f}%  avg {tr.return_pct.mean() if len(tr) else 0:+.2f}%")
print(f"BACKTEST : equity {eq[-1]:.4f}x  trades {len(lg)}  win {np.mean(lg[:,4]>0)*100:.0f}%  avg {lg[:,4].mean()*100:+.2f}%")
