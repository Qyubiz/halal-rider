"""Fetch Binance USD-M perpetual funding-rate history (monthly zips) — used only as a sentiment signal for spot.
Usage: fetch_funding.py SYM1,SYM2 OUTDIR"""
import sys, os, io, time, zipfile, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

syms, out = sys.argv[1].split(","), sys.argv[2]
months = [str(p) for p in pd.period_range("2021-01", "2026-09", freq="M")]

def one(sym, m):
    url = f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{m}.zip"
    for k in range(4):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                b = r.read()
            z = zipfile.ZipFile(io.BytesIO(b))
            raw = pd.read_csv(z.open(z.namelist()[0]), header=None)
            if not str(raw.iloc[0, 0]).strip().isdigit():
                raw = raw.iloc[1:]
            df = pd.DataFrame({"t": pd.to_datetime(raw.iloc[:, 0].astype("int64"), unit="ms"),
                               "rate": raw.iloc[:, -1].astype(float)})
            return df
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(1 + k)
        except Exception:
            time.sleep(1 + k)
    return None

def fetch(sym):
    parts = [one(sym, m) for m in months]
    parts = [p for p in parts if p is not None]
    if not parts:
        return f"{sym}: none"
    df = pd.concat(parts).drop_duplicates("t").sort_values("t").set_index("t")
    df.to_csv(os.path.join(out, f"{sym}.csv"))
    return f"{sym}: {len(df)} rows {df.index[0]:%Y-%m} -> {df.index[-1]:%Y-%m}"

os.makedirs(out, exist_ok=True)
with ThreadPoolExecutor(4) as ex:
    for r in ex.map(fetch, syms):
        print(r, flush=True)
