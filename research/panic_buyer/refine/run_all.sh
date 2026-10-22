#!/usr/bin/env bash
# Run every refinement test in order; logs and CSVs go to $PB_OUT (default ~/pb_out).
set -euo pipefail
cd "$(dirname "$0")"
OUT="${PB_OUT:-$HOME/pb_out}"; mkdir -p "$OUT"
python3 check_engine.py | tee "$OUT/check_engine.log"
for s in baseline idea3_depth idea4_stress idea6_exits idea2_fast_exit idea5_universe idea8_stats; do
  echo "== $s"
  python3 "$s.py" 2>&1 | tee "$OUT/$s.log"
done
echo "verdicts: $OUT/verdicts.csv"
