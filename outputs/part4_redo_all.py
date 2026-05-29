"""
Post-fix verification: re-run all three sectors that have parquet outputs
to confirm corrected counts after removing the idx parameter from _run().
Also re-save the corrected parquets.
"""
from __future__ import annotations
import os, sys, time, json, random

REPO = r"C:\Users\j.guthrie\Downloads\RESEARCH\infra_fm"

POWER_ONLY = (
    r"C:\Users\j.guthrie\Downloads\RESEARCH\infra_fm\data\pbf\power_only"
    r"\central-america-latest.osm_power_only.osm.pbf"
)
WATER_ONLY     = r"D:\central-america-260526.osm_water_only.osm.pbf"
TRANSPORT_ONLY = r"D:\central-america-260526.osm_transport_only.osm.pbf"

sys.path.insert(0, os.path.join(REPO, "curation"))
sys.path.insert(0, os.path.join(REPO, "curation", "utils"))
os.chdir(REPO)

from sources import GeoFabrikSource
from ontology import get_class_by_name


OUT_DIR = os.path.join(REPO, "data", "PIPELINE", "01-extracted-assets")
os.makedirs(OUT_DIR, exist_ok=True)


def run(path: str, preset: str, out_basename: str, expected_baseline: int | None = None):
    print("=" * 70)
    print(f"PRESET: {preset}    on    {path}")
    if expected_baseline is not None:
        print(f"  (expected ~= {expected_baseline:,})")
    print("=" * 70)
    src = GeoFabrikSource(path, min_confidence="medium", filter_preset=preset, pre_filter=False)
    t = time.time()
    df = src.extract_all()
    elapsed = time.time() - t
    print(f"  wall time: {elapsed:.1f}s   matched={len(df):,}")
    if len(df):
        print(df["asset_type"].value_counts().to_string())
    print()
    if expected_baseline is not None:
        diff = len(df) - expected_baseline
        pct = 100 * diff / expected_baseline if expected_baseline else 0
        print(f"  baseline diff: {diff:+d} ({pct:+.2f}%)")
    out_path = os.path.join(OUT_DIR, out_basename)
    df.drop(columns=["osm_tags"], errors="ignore").to_parquet(out_path, index=False)
    sz = os.path.getsize(out_path)
    print(f"  saved -> {out_path}  ({sz:,} bytes)")
    print()
    return df


# 1. Substation — May 26 baseline was 1,920
df_sub = run(POWER_ONLY, "substation",
             "central-america_all_assets_substations.parquet",
             expected_baseline=1920)

# 2. Water — pre-fix run gave 1,927
df_water = run(WATER_ONLY, "water",
               "central-america_all_assets_water.parquet",
               expected_baseline=1927)

# 3. Transport — pre-fix run gave 423 (buggy); diagnostic Run B gave 675
df_trans = run(TRANSPORT_ONLY, "transport",
               "central-america_all_assets_transport.parquet",
               expected_baseline=675)

# Detailed area + spot-check for transport (the one we actually care about)
print("=" * 70)
print("Transport area + spot-check (post-fix)")
print("=" * 70)

import osmium
from shapely import wkb as shapely_wkb
from pyproj import Geod
GEOD = Geod(ellps="WGS84"); WKBF = osmium.geom.WKBFactory()

df_areas = df_trans[df_trans["asset_id"].str.startswith(("osm_way_", "osm_relation_"))]
targets = set()
for _, row in df_areas.iterrows():
    aid = row["asset_id"]
    if aid.startswith("osm_way_"):
        targets.add(("way", int(aid.split("_")[2])))
    else:
        targets.add(("relation", int(aid.split("_")[2])))
print(f"way/relation matches needing area lookup: {len(targets):,}")

area_lookup: dict = {}
class S(osmium.SimpleHandler):
    def area(self, a):
        k = ("way" if a.from_way() else "relation", a.orig_id())
        if k not in targets: return
        try:
            poly = shapely_wkb.loads(bytes.fromhex(WKBF.create_multipolygon(a)))
            area_lookup[k] = abs(GEOD.geometry_area_perimeter(poly)[0])
        except Exception:
            pass
S().apply_file(TRANSPORT_ONLY, locations=True)

def get_area(aid):
    if aid.startswith("osm_way_"):
        return area_lookup.get(("way", int(aid.split("_")[2])))
    if aid.startswith("osm_relation_"):
        return area_lookup.get(("relation", int(aid.split("_")[2])))
    return None
df_trans["area_m2"] = df_trans["asset_id"].apply(get_area)

print()
print("Per-class area distribution:")
for cls_name in sorted(df_trans["asset_type"].unique()):
    cls = get_class_by_name(cls_name)
    sub = df_trans[df_trans["asset_type"] == cls_name]
    with_area = sub[sub["area_m2"].notna()]
    print(f"  {cls_name}: {len(sub)} rows, {len(with_area)} with area")
    if cls.min_area_m2:
        print(f"    min_area_m2 filter: {cls.min_area_m2:,.0f}")
    if len(with_area):
        a = with_area["area_m2"]
        print(f"    area min/med/max (ha): {a.min()/1e4:.1f} / {a.median()/1e4:.1f} / {a.max()/1e4:.1f}")
        if cls.min_area_m2:
            below = (a < cls.min_area_m2).sum()
            print(f"    below min_area: {below} (should be 0)")
    print()

random.seed(42)
for cls_name in sorted(df_trans["asset_type"].unique()):
    sub = df_trans[df_trans["asset_type"] == cls_name]
    n = min(5, len(sub))
    sample = sub.sample(n=n, random_state=42).reset_index(drop=True)
    print(f"--- {cls_name} (showing {n}/{len(sub)} random) ---")
    for i, row in sample.iterrows():
        a = row["area_m2"]
        astr = f"{a:,.0f} m^2 ({a/1e4:.1f} ha)" if a is not None else "n/a"
        name = row["name"] if row["name"] else "(no name)"
        tags_str = json.dumps(row["osm_tags"], ensure_ascii=False)
        if len(tags_str) > 240: tags_str = tags_str[:240] + "..."
        print(f"  [{i+1}] {row['asset_id']}  lat={row['lat']:.5f}, lon={row['lon']:.5f}  area={astr}")
        print(f"      name: {name}")
        print(f"      tags: {tags_str}")
    print()
