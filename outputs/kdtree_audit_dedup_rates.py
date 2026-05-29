"""
KDTree latitude-bug audit: report dedup rates per region and their median
absolute latitudes. The bug under-suppresses at high latitudes (cos(lat)
factor missing), so we expect higher-latitude regions to show LOWER dedup
rates than they should — but raw dedup rate alone doesn't prove the bug,
since real duplicate density varies by region.

This script just collates inputs:
  - pre-dedup row count from 01-extracted-assets/<region>_all_assets_substations.parquet
  - post-dedup row count from 02-deduped-assets/<region>_deduped_assets_substations.parquet
  - median |lat| of the input data per region
"""
from __future__ import annotations
import os
import math
import pandas as pd

REPO = r"C:\Users\j.guthrie\Downloads\RESEARCH\infra_fm"

PRE_DIR  = os.path.join(REPO, "data", "PIPELINE", "01-extracted-assets")
POST_DIR = os.path.join(REPO, "data", "PIPELINE", "02-deduped-assets")

# Sector pairs to audit (substations are most consistent across regions).
REGIONS = [
    "maine",
    "central-america",
    "australia-oceania",
    "south-america",
    "africa",
    "asia",
    "north-america",
    "europe",
]


def cos_lat(lat_deg: float) -> float:
    return math.cos(math.radians(lat_deg))


rows = []
for region in REGIONS:
    pre_path  = os.path.join(PRE_DIR,  f"{region}_all_assets_substations.parquet")
    post_path = os.path.join(POST_DIR, f"{region}_deduped_assets_substations.parquet")

    if not os.path.exists(pre_path) or not os.path.exists(post_path):
        rows.append({"region": region, "note": "missing parquet(s)"})
        continue

    pre_df  = pd.read_parquet(pre_path,  columns=["lat", "lon", "asset_type"])
    post_df = pd.read_parquet(post_path, columns=["asset_id"])

    pre_n  = len(pre_df)
    post_n = len(post_df)
    removed = pre_n - post_n
    dedup_pct = 100 * removed / pre_n if pre_n else 0.0

    median_abs_lat = pre_df["lat"].abs().median()
    p10 = pre_df["lat"].abs().quantile(0.10)
    p90 = pre_df["lat"].abs().quantile(0.90)
    # cos(lat) effective at median lat — this is the fraction by which the
    # east-west threshold is too small. e.g. 0.5 means E-W radius is half
    # of intended.
    cos_med = cos_lat(median_abs_lat)

    rows.append({
        "region": region,
        "pre_n":   pre_n,
        "post_n":  post_n,
        "removed": removed,
        "dedup_pct":      round(dedup_pct, 2),
        "median_abs_lat": round(median_abs_lat, 1),
        "lat_p10":        round(p10, 1),
        "lat_p90":        round(p90, 1),
        "cos(median_lat)": round(cos_med, 3),
        "expected_ratio_lost": round(1 - cos_med, 2),
    })

df = pd.DataFrame(rows)
# Sort by median |lat| ascending so the most-affected (high-lat) appear at the bottom
df = df.sort_values("median_abs_lat")
# Pretty-print
print(df.to_string(index=False))
print()
print("Interpretation:")
print("  - 'cos(median_lat)' is the fraction of intended E-W radius the bug delivers.")
print("    1.0  = at the equator, no bug effect.")
print("    0.5  = E-W radius is half of intended (e.g. ~60 degrees latitude).")
print("    0.25 = E-W radius is one quarter of intended (~75 degrees).")
print()
print("  - 'expected_ratio_lost' is the fraction of intended threshold *missing*")
print("    along the E-W axis at the median latitude.")
print()
print("  - dedup_pct should be re-computed after the cos(lat) fix to see how")
print("    much it grows. The further the cos(med_lat) is from 1.0, the more")
print("    a region is expected to gain in dedup_pct from a correct fix.")
