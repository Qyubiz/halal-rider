#!/usr/bin/env bash
# Download everything the refinement scripts need into $PB_DATA (default ~/pb_data). Public data, no API key.
# Needs data.binance.vision (and data-api.binance.vision for the last hours) to be reachable.
set -euo pipefail
cd "$(dirname "$0")"
OUT="${PB_DATA:-$HOME/pb_data}"
CORE="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,BNBUSDT,DOGEUSDT,ADAUSDT,TRXUSDT,LINKUSDT,AVAXUSDT,SUIUSDT,LTCUSDT,BCHUSDT,HBARUSDT,DOTUSDT,NEARUSDT,ARBUSDT,FETUSDT,RENDERUSDT,RNDRUSDT,LUNAUSDT,FTTUSDT,SRMUSDT"
export THREADS="${THREADS:-8}"

echo "== 1h candles, 19 coins + RNDR + LUNA/FTT/SRM"
python3 fetch_vision.py 1h 2021-01-01 "$CORE" "$OUT/1h" --api-tail

echo "== 1m candles, same coins (about 1 GB)"
python3 fetch_vision.py 1m 2021-01-01 "$CORE" "$OUT/1m" --api-tail

if [ "${UNIVERSE:-1}" = "1" ]; then
  echo "== 1h candles, the 322 screened coins (dynamic-universe test)"
  UNIV=$(python3 -c "import pandas as pd; print(','.join(pd.read_csv('../results/coins_used.csv').symbol))")
  python3 fetch_vision.py 1h 2021-01-01 "$UNIV" "$OUT/univ" --api-tail
fi
echo "done: $OUT"
