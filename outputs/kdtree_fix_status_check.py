"""Compare current 02-deduped-assets parquets to pre-fix counts (where known)."""
from __future__ import annotations
import os, pandas as pd

REPO = r"C:\Users\j.guthrie\Downloads\RESEARCH\infra_fm"
PRE_DIR  = os.path.join(REPO, "data", "PIPELINE", "01-extracted-assets")
POST_DIR = os.path.join(REPO, "data", "PIPELINE", "02-deduped-assets")

# Pre-fix counts captured before the dedup rerun (from earlier audit run today).
PRE_FIX_DEDUPED = {
    "maine":             372,
    "central-america":   1807,
    "australia-oceania": 5583,
    "south-america":     17899,
    "africa":            11044,
    "asia":              127244,
    "north-america":     77906,
    "europe":            505951,
}

print(f"{'region':<20} {'input':>8} {'old_kept':>10} {'new_kept':>10} {'delta':>8} "
      f"{'old_pct':>8} {'new_pct':>8}")
print("-" * 80)
for region, old_n in PRE_FIX_DEDUPED.items():
    pre  = os.path.join(PRE_DIR,  f"{region}_all_assets_substations.parquet")
    post = os.path.join(POST_DIR, f"{region}_deduped_assets_substations.parquet")
    pre_n = len(pd.read_parquet(pre,  columns=["asset_id"]))
    new_n = len(pd.read_parquet(post, columns=["asset_id"]))
    delta = new_n - old_n
    old_pct = 100 * (pre_n - old_n) / max(pre_n, 1)
    new_pct = 100 * (pre_n - new_n) / max(pre_n, 1)
    marker = "(stale - not re-run)" if new_n == old_n and region == "europe" else ""
    print(f"{region:<20} {pre_n:>8,} {old_n:>10,} {new_n:>10,} {delta:>+8,} "
          f"{old_pct:>7.2f}% {new_pct:>7.2f}%  {marker}")
