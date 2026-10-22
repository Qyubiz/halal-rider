"""Download Binance spot candles from the public bulk archive (data.binance.vision), no API key.

Usage: python3 fetch_vision.py INTERVAL START SYM1,SYM2,... OUTDIR [--api-tail]

Monthly zips up to last month, daily zips for the current month. Files from 2025 on store times in
microseconds; both units are handled. --api-tail fills the hours the archive hasn't published yet from
data-api.binance.vision. Output: OUTDIR/SYM.pkl (1m) or OUTDIR/SYM.csv (other intervals), columns
o h l c v qv tb indexed by the candle open time (UTC), same as code/fetch_klines.py
(1m: o h l c only, float32). SKIP_EXISTING=1 skips symbols already downloaded.
"""
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

ROOT = "https://data.binance.vision/data/spot"
API = "https://data-api.binance.vision/api/v3/klines"
COLS = ["t", "o", "h", "l", "c", "v", "ct", "qv", "n", "tb", "tq", "x"]


def get(url, tries=6):
    last = None
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None                         # month/day not in the archive (coin not listed then)
            last = e
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(1.5 * (k + 1))
    raise RuntimeError(f"{url}: {last}")


def read_zip(blob):
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw = z.read(z.namelist()[0])
    df = pd.read_csv(io.BytesIO(raw), header=None, names=COLS)
    if not str(df.t.iloc[0]).lstrip("-").isdigit():  # some files carry a header row
        df = df.iloc[1:]
    return df


def to_frame(df):
    df = df[["t", "o", "h", "l", "c", "v", "qv", "tb"]].astype(float)
    t = df.t.astype("int64")
    t = t.where(t < 10**14, t // 1000)              # microseconds (2025+) -> milliseconds
    df["t"] = pd.to_datetime(t, unit="ms")
    return df.drop_duplicates("t").set_index("t").sort_index()


def api_tail(sym, interval, since_ms):
    rows, s, end = [], since_ms, int(time.time() * 1000)
    while s < end:
        try:
            k = json.loads(get(f"{API}?symbol={sym}&interval={interval}&startTime={s}&limit=1000", tries=3) or b"[]")
        except RuntimeError:
            break                                   # delisted symbol: the archive is all there is
        if not k:
            break
        rows += k
        s = k[-1][0] + 1
        if len(k) < 1000:
            break
    if not rows:
        return None
    return pd.DataFrame([r[:12] for r in rows], columns=COLS)


def fetch(sym, interval, start, out, tail):
    dest = os.path.join(out, f"{sym}.{'pkl' if interval == '1m' else 'csv'}")
    if os.environ.get("SKIP_EXISTING") == "1" and os.path.exists(dest):
        return f"{sym}: already downloaded"
    start = pd.Timestamp(start).to_period("M")
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    this_month = today.to_period("M")
    parts = []
    days_from = this_month.start_time
    for m in pd.period_range(start, this_month - 1, freq="M"):
        b = get(f"{ROOT}/monthly/klines/{sym}/{interval}/{sym}-{interval}-{m:%Y-%m}.zip")
        if b:
            parts.append(read_zip(b))
        elif m == this_month - 1:
            days_from = m.start_time                # last month not archived yet: use its daily files
    for d in pd.date_range(days_from, today - pd.Timedelta(days=1), freq="D"):
        b = get(f"{ROOT}/daily/klines/{sym}/{interval}/{sym}-{interval}-{d:%Y-%m-%d}.zip")
        if b:
            parts.append(read_zip(b))
    if not parts:
        return f"{sym}: no data"
    df = to_frame(pd.concat(parts))
    if tail:
        more = api_tail(sym, interval, int(df.index[-1].timestamp() * 1000) + 1)
        if more is not None:
            df = pd.concat([df, to_frame(more)])
            df = df[~df.index.duplicated()].sort_index()
            now = pd.Timestamp.utcnow().tz_localize(None)
            step = pd.Timedelta(interval.replace("m", "min"))
            df = df[df.index + step <= now]       # drop the candle still forming
    if interval == "1m":
        df[["o", "h", "l", "c"]].astype("float32").to_pickle(dest)     # ~50 MB per coin
    else:
        df.to_csv(dest)
    return f"{sym}: {len(df)} bars {df.index[0]:%Y-%m-%d %H:%M} -> {df.index[-1]:%Y-%m-%d %H:%M}"


if __name__ == "__main__":
    interval, start, syms, out = sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4]
    tail = "--api-tail" in sys.argv[5:]
    os.makedirs(out, exist_ok=True)
    with ThreadPoolExecutor(int(os.environ.get("THREADS", "6"))) as ex:
        for r in ex.map(lambda s: fetch(s, interval, start, out, tail), syms):
            print(r, flush=True)
