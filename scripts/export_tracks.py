"""Dump a tracks.parquet cache to CSV (for sharing / inspecting without pyarrow).

    python scripts/export_tracks.py outputs/adesanya_pereira_2/tracks.parquet
"""
import sys
from pathlib import Path

import pandas as pd

p = Path(sys.argv[1])
out = p.with_name("tracks_export.csv")
pd.read_parquet(p).to_csv(out, index=False)
print(f"wrote {out}")
