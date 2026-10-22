"""Fetch Binance spot klines. Usage: fetch_long.py INTERVAL START SYM1,SYM2,... OUTDIR"""
import sys, os, time, json, urllib.request
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

BASE = "https://data-api.binance.vision/api/v3/klines"

def get(url):
    last = None
    for k in range(6):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read())
        except Exception as e:
            last = e
            time.sleep(1.5 * (k + 1))
    raise RuntimeError(f"{url}: {last}")

def fetch(sym, interval, start, out):
    s = int(pd.Timestamp(start).timestamp() * 1000)
    end = int(time.time() * 1000)
    rows = []
    while s < end:
        try:
            k = get(f"{BASE}?symbol={sym}&interval={interval}&startTime={s}&limit=1000")
        except Exception as e:
            return f"{sym}: FAILED {str(e)[:80]}"
        if not k:
            break
        rows += k
        s = k[-1][0] + 1
        if len(k) < 1000:
            break
    if not rows:
        return f"{sym}: no data"
    df = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v", "ct", "qv", "n", "tb", "tq", "x"])
    df = df[["t", "o", "h", "l", "c", "v", "qv", "tb"]].astype(float)
    df["t"] = pd.to_datetime(df["t"], unit="ms")
    df = df.drop_duplicates("t").set_index("t")
    df.to_csv(os.path.join(out, f"{sym}.csv"))
    return f"{sym}: {len(df)} bars {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}"

if __name__ == "__main__":
    interval, start, syms, out = sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4]
    os.makedirs(out, exist_ok=True)
    with ThreadPoolExecutor(int(os.environ.get("THREADS", "6"))) as ex:
        for r in ex.map(lambda s: fetch(s, interval, start, out), syms):
            print(r, flush=True)
