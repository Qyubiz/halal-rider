"""Convert the 1-minute pickles to plain numpy files (no pickle) for sharing: t (unix seconds), o h l c (float32).
Usage: python3 export_npz.py SRC_DIR DST_DIR"""
import glob
import os
import sys

import numpy as np
import pandas as pd

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
for p in sorted(glob.glob(os.path.join(src, "*.pkl"))):
    d = pd.read_pickle(p)
    t = (d.index - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
    np.savez_compressed(os.path.join(dst, os.path.basename(p)[:-4] + ".npz"), t=np.asarray(t, np.int64),
                        **{f: d[f].to_numpy(np.float32) for f in ("o", "h", "l", "c")})
    print(os.path.basename(p), len(d), flush=True)
