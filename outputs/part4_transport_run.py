"""
Part 4: transport-sector verification on central-america.

Runs pre-filter + extract + area-validation re-scan + spot-check.
Outputs:
  - data/PIPELINE/01-extracted-assets/central-america_all_assets_transport.parquet
  - stdout report
"""
from __future__ import annotations

import sys
import os
import time
import json
import random

REPO = r"C:\Users\j.guthrie\Downloads\RESEARCH\infra_fm"
SRC  = r"D:\central-america-260526.osm.pbf"

sys.path.insert(0, os.path.join(REPO, "curation"))
sys.path.insert(0, os.path.join(REPO, "curation", "utils"))
os.chdir(REPO)

from sources import GeoFabrikSource
from ontology import get_classes_for_sector, get_class_by_name


def main() -> None:
    print("=" * 70)
    print("Part 4: transport-sector verification on central-america")
    print("=" * 70)
    print(f"Source PBF: {SRC}")
    src_size = os.path.getsize(SRC)
    print(f"Source size: {src_size:,} bytes ({src_size/1_048_576:.1f} MB)")
    print()
    print("Active transport classes:")
    for c in get_classes_for_sector("transport"):
        print(
            f"  {c.name:32s} geom={c.require_geometry}  "
            f"area>={int(c.min_area_m2 or 0):>8d}  tags={list(c.tags)}"
        )
    print()

    # ---- Step 1: pre-filter --------------------------------------------------
    print("=" * 70)
    print('STEP 1: pre-filter (sector_name="transport", '
          'sector_tag_keys=["aeroway","harbour","railway"])')
    print("=" * 70)
    src = GeoFabrikSource(
        SRC, min_confidence="medium", filter_preset="transport", pre_filter=True,
    )
    t0 = time.time()
    transport_only_path = src._pre_filter_pbf_by_tags(
        sector_name="transport",
        sector_tag_keys=["aeroway", "harbour", "railway"],
    )
    prefilter_elapsed = time.time() - t0
    out_size = os.path.getsize(transport_only_path)
    print()
    print("PRE-FILTER RESULT")
    print(f"  Output path     : {transport_only_path}")
    print(f"  Output size     : {out_size:,} bytes ({out_size/1_048_576:.1f} MB)")
    print(f"  Size ratio      : {out_size/src_size:.4f}")
    print(f"  Wall time       : {prefilter_elapsed:.1f}s")
    print()

    # ---- Step 2: extract -----------------------------------------------------
    print("=" * 70)
    print("STEP 2: extract transport sector from transport-only PBF")
    print("=" * 70)
    src2 = GeoFabrikSource(
        transport_only_path,
        min_confidence="medium",
        filter_preset="transport",
        pre_filter=False,
    )
    t1 = time.time()
    df = src2.extract_all()
    extract_elapsed = time.time() - t1
    print()
    print("EXTRACT RESULT")
    print(f"  Rows extracted  : {len(df):,}")
    print(f"  Wall time       : {extract_elapsed:.1f}s")
    print()
    if len(df):
        print("Counts by asset type:")
        print(df["asset_type"].value_counts().to_string())
        print()

    # Save parquet (drop osm_tags for parity with other 01-extracted-assets parquets)
    out_pq = os.path.join(
        REPO, "data", "PIPELINE", "01-extracted-assets",
        "central-america_all_assets_transport.parquet",
    )
    os.makedirs(os.path.dirname(out_pq), exist_ok=True)
    df.drop(columns=["osm_tags"], errors="ignore").to_parquet(out_pq, index=False)
    print(f"Saved -> {out_pq}")
    print()

    if not len(df):
        print("No rows; stopping before validation.")
        return

    # ---- Step 3: area validation re-scan ------------------------------------
    print("=" * 70)
    print("STEP 3: area validation re-scan + spot-check (5 random per class)")
    print("=" * 70)

    import osmium
    from shapely import wkb as shapely_wkb
    from pyproj import Geod

    df_areas = df[df["asset_id"].str.startswith(("osm_way_", "osm_relation_"))].copy()
    targets: set = set()
    for _, row in df_areas.iterrows():
        aid = row["asset_id"]
        if aid.startswith("osm_way_"):
            targets.add(("way", int(aid.split("_")[2])))
        else:
            targets.add(("relation", int(aid.split("_")[2])))
    print(f"Targets for area lookup: {len(targets):,} way/relation matches")

    GEOD = Geod(ellps="WGS84")
    WKBF = osmium.geom.WKBFactory()
    area_lookup: dict = {}

    class AreaScanner(osmium.SimpleHandler):
        def area(self, a):
            otype = "way" if a.from_way() else "relation"
            key = (otype, a.orig_id())
            if key not in targets:
                return
            try:
                wkb_hex = WKBF.create_multipolygon(a)
                poly = shapely_wkb.loads(bytes.fromhex(wkb_hex))
                area_signed, _ = GEOD.geometry_area_perimeter(poly)
                area_lookup[key] = abs(area_signed)
            except Exception:
                area_lookup[key] = None

    t2 = time.time()
    AreaScanner().apply_file(transport_only_path, locations=True)
    area_scan_elapsed = time.time() - t2
    print(
        f"Area re-scan wall: {area_scan_elapsed:.1f}s, "
        f"{len(area_lookup)}/{len(targets)} areas computed"
    )
    print()

    def get_area(asset_id):
        if asset_id.startswith("osm_way_"):
            return area_lookup.get(("way", int(asset_id.split("_")[2])))
        if asset_id.startswith("osm_relation_"):
            return area_lookup.get(("relation", int(asset_id.split("_")[2])))
        return None

    df["area_m2"] = df["asset_id"].apply(get_area)

    # Per-class histogram
    print("Per-class area distribution (where computed):")
    for cls_name in sorted(df["asset_type"].unique()):
        cls = get_class_by_name(cls_name)
        sub = df[df["asset_type"] == cls_name]
        with_area = sub[sub["area_m2"].notna()]
        print(f"  {cls_name}:")
        print(f"    rows               : {len(sub)}")
        print(f"    with computed area : {len(with_area)}")
        if cls.min_area_m2:
            print(f"    min_area_m2 filter : {cls.min_area_m2:,.0f}")
        if len(with_area):
            areas = with_area["area_m2"]
            print(
                f"    area min / med / max (m²): "
                f"{areas.min():,.0f} / {areas.median():,.0f} / {areas.max():,.0f}"
            )
            if cls.min_area_m2:
                below = (areas < cls.min_area_m2).sum()
                pct = 100 * below / len(with_area)
                marker = "<-- should be 0 if filter is correct" if below else ""
                print(f"    rows below min_area: {below} ({pct:.1f}%)  {marker}")
        print()

    # Spot-check
    random.seed(42)
    for cls_name in sorted(df["asset_type"].unique()):
        sub = df[df["asset_type"] == cls_name]
        n = min(5, len(sub))
        sample = sub.sample(n=n, random_state=42).reset_index(drop=True)
        print(f"--- {cls_name} (showing {n}/{len(sub)} random) ---")
        for i, row in sample.iterrows():
            if row["area_m2"] is not None:
                area_str = f"{row['area_m2']:,.0f} m²"
            else:
                area_str = "n/a (node or area not computed)"
            name = row["name"] if row["name"] else "(no name)"
            tags_str = json.dumps(row["osm_tags"], ensure_ascii=False)
            if len(tags_str) > 240:
                tags_str = tags_str[:240] + "..."
            print(
                f"  [{i+1}] {row['asset_id']}  "
                f"lat={row['lat']:.5f}, lon={row['lon']:.5f}  area={area_str}"
            )
            print(f"      name: {name}")
            print(f"      tags: {tags_str}")
        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Pre-filter wall  : {prefilter_elapsed:.1f}s")
    print(f"Extract wall     : {extract_elapsed:.1f}s")
    print(f"Area re-scan wall: {area_scan_elapsed:.1f}s")
    print(f"Source size      : {src_size/1_048_576:.1f} MB")
    print(f"Transport-only   : {out_size/1_048_576:.1f} MB  "
          f"(ratio {out_size/src_size:.4f})")
    print(f"Total assets     : {len(df):,}")


if __name__ == "__main__":
    main()
