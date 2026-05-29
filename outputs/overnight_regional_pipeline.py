"""
overnight_regional_pipeline.py
------------------------------
Replicate the central-america pipeline (substation, water, transport,
telecom — everything except STAC imagery fetch) for the 6 remaining
regions. For each (region, sector):

  - pre-filter source PBF (or two-pass direct scan for telecom)
  - extract via GeoFabrikSource
  - dedup with per-class thresholds + BallTree+haversine
  - save parquets to data/PIPELINE/01-extracted-assets and 02-deduped-assets

Then for substations specifically, delete the buggy OLD
data/pbf/power_only/<region>-<old-date>.osm_power_only.osm.pbf to recover
disk space.

Region order: smallest-source-first so value accrues early.

Per-region per-sector outputs are logged. Per-sector failures do not
abort the run; other sectors and regions still proceed.
"""
from __future__ import annotations
import os, sys, time, traceback
from datetime import datetime

REPO = r"C:\Users\j.guthrie\Downloads\RESEARCH\infra_fm"
sys.path.insert(0, os.path.join(REPO, "curation"))
sys.path.insert(0, os.path.join(REPO, "curation", "utils"))
os.chdir(REPO)

import pandas as pd

# Region order: smallest source first.
REGIONS = [
    ("australia-oceania", r"D:\australia-oceania-260526.osm.pbf"),
    ("south-america",     r"D:\south-america-260526.osm.pbf"),
    ("africa",            r"D:\africa-260526.osm.pbf"),
    ("asia",              r"D:\asia-260526.osm.pbf"),
    ("north-america",     r"D:\north-america-latest.osm.pbf"),
    ("europe",            r"D:\europe-latest.osm.pbf"),
]

# Filename roots of OLD buggy power_only files to delete after the new
# substation pre-filter for each region succeeds.
OLD_POWER_ONLY_DIR = os.path.join(REPO, "data", "pbf", "power_only")
OLD_POWER_ONLY_FILES = {
    "australia-oceania": "australia-oceania-260408.osm_power_only.osm.pbf",
    "south-america":     "south-america-260410.osm_power_only.osm.pbf",
    "africa":            "africa-260408.osm_power_only.osm.pbf",
    "asia":              "asia-260408.osm_power_only.osm.pbf",
    "north-america":     "north-america-latest.osm_power_only.osm.pbf",
    "europe":            "europe-latest.osm_power_only.osm.pbf",
    # Also clean up the obsolete extras left over from earlier experiments:
    "central-america":   "central-america-260408.osm_power_only.osm.pbf",
    "maine":             "maine-latest.osm_power_only.osm_power_only.osm.pbf",
}

# Sector -> (preset_name, sector_name_for_prefilter, sector_tag_keys).
# Telecom is handled separately via two-pass scan (no pre-filter file).
#
# The "energy" preset covers ALL 7 energy ontology classes (substations,
# power plants, solar/wind farms). The pre-filter anchor stays "power" —
# every energy class uses power=* as its required tag — so the
# *_power_only.osm.pbf file is unchanged from a substation-only run.
PREFILTER_SECTORS = [
    ("energy",     "power",     ["power"]),
    ("water",      "water",     ["man_made"]),
    ("transport",  "transport", ["aeroway", "harbour", "railway"]),
]

OUT_DIR_EXTRACT = os.path.join(REPO, "data", "PIPELINE", "01-extracted-assets")
OUT_DIR_DEDUP   = os.path.join(REPO, "data", "PIPELINE", "02-deduped-assets")
os.makedirs(OUT_DIR_EXTRACT, exist_ok=True)
os.makedirs(OUT_DIR_DEDUP,   exist_ok=True)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def run_prefilter_sector(region: str, source_pbf: str,
                         preset: str, sector_name: str,
                         sector_tag_keys: list[str]) -> bool:
    """Pre-filter + extract + dedup for one (region, sector)."""
    from sources import GeoFabrikSource
    from deduplication import Deduplicator

    src_extract = os.path.join(OUT_DIR_EXTRACT,
                               f"{region}_all_assets_{preset}.parquet")
    src_dedup   = os.path.join(OUT_DIR_DEDUP,
                               f"{region}_deduped_assets_{preset}.parquet")

    # Skip if both output parquets already exist (resume optimization).
    # IMPORTANT: do NOT `import pandas as pd` inside this if-block —
    # that turns `pd` into a local variable that shadows the module-level
    # import, then UnboundLocalError fires later in the function when the
    # if-block didn't run. Use the module-level pd directly.
    if os.path.exists(src_extract) and os.path.exists(src_dedup):
        n_extract = len(pd.read_parquet(src_extract, columns=["asset_id"]))
        n_dedup   = len(pd.read_parquet(src_dedup,   columns=["asset_id"]))
        log(f"  [{preset}] outputs already exist "
            f"({n_extract:,} extracted, {n_dedup:,} deduped); skipping")
        return True

    log(f"  [{preset}] pre-filter...")
    t = time.time()
    src = GeoFabrikSource(source_pbf, min_confidence="medium",
                          filter_preset=preset, pre_filter=True)
    pre_filtered = src._pre_filter_pbf_by_tags(sector_name=sector_name,
                                               sector_tag_keys=sector_tag_keys)
    log(f"  [{preset}] pre-filter done in {time.time()-t:.0f}s -> "
        f"{os.path.getsize(pre_filtered)/1_048_576:.1f} MB")

    log(f"  [{preset}] extract...")
    t = time.time()
    src2 = GeoFabrikSource(pre_filtered, min_confidence="medium",
                           filter_preset=preset, pre_filter=False)
    df = src2.extract_all()
    log(f"  [{preset}] extract done in {time.time()-t:.0f}s -> {len(df):,} rows")
    if len(df) == 0:
        log(f"  [{preset}] no rows — skipping dedup")
        return True

    df.drop(columns=["osm_tags"], errors="ignore").to_parquet(src_extract, index=False)
    log(f"  [{preset}] saved -> {src_extract}")

    log(f"  [{preset}] dedup...")
    t = time.time()
    dedup = Deduplicator()
    clean, _ = dedup.run(df)
    log(f"  [{preset}] dedup done in {time.time()-t:.0f}s -> "
        f"{len(clean):,} kept ({100*(len(df)-len(clean))/max(len(df),1):.1f}% removed)")
    clean.drop(columns=["osm_tags"], errors="ignore").to_parquet(src_dedup, index=False)
    log(f"  [{preset}] saved -> {src_dedup}")
    return True


def run_telecom(region: str, source_pbf: str) -> bool:
    """Telecom direct two-pass scan + dedup. 'building' is too broad an
    anchor for pre-filter to be useful."""
    import osmium
    from deduplication import Deduplicator

    extract_path = os.path.join(OUT_DIR_EXTRACT,
                                f"{region}_all_assets_telecom.parquet")
    dedup_path   = os.path.join(OUT_DIR_DEDUP,
                                f"{region}_deduped_assets_telecom.parquet")

    # Skip the two-pass scan entirely if both outputs already exist.
    # Telecom's 2x full-source scan is hours on the big regions — the
    # most important place to have this resume optimization.
    # Use module-level pd; don't re-import inside the if-block (would
    # shadow it as a local and break the later `df = pd.DataFrame(...)`).
    if os.path.exists(extract_path) and os.path.exists(dedup_path):
        n_extract = len(pd.read_parquet(extract_path, columns=["asset_id"]))
        n_dedup   = len(pd.read_parquet(dedup_path,   columns=["asset_id"]))
        log(f"  [telecom] outputs already exist "
            f"({n_extract:,} extracted, {n_dedup:,} deduped); skipping")
        return True

    log(f"  [telecom] pass 1 (find building=data_center)...")
    t = time.time()

    matched_nodes = []   # [(id, lat, lon, tags)]
    matched_ways  = []   # [(id, [node_ids], tags)]
    matched_rels  = []   # [(id, [(type, ref)], tags)]
    target_node_ids: set = set()

    class P1(osmium.SimpleHandler):
        def node(self, n):
            if n.tags.get("building") == "data_center":
                if n.location.valid():
                    matched_nodes.append(
                        (n.id, n.location.lat, n.location.lon, dict(n.tags))
                    )
        def way(self, w):
            if w.tags.get("building") == "data_center":
                node_ids = [nd.ref for nd in w.nodes]
                target_node_ids.update(node_ids)
                matched_ways.append((w.id, node_ids, dict(w.tags)))
        def relation(self, r):
            if r.tags.get("building") == "data_center":
                members = [(m.type, m.ref) for m in r.members]
                for mt, mr in members:
                    if mt == 'n':
                        target_node_ids.add(mr)
                matched_rels.append((r.id, members, dict(r.tags)))

    P1().apply_file(source_pbf)
    log(f"  [telecom] pass 1 done in {time.time()-t:.0f}s -> "
        f"{len(matched_nodes)} nodes / {len(matched_ways)} ways / "
        f"{len(matched_rels)} relations")
    if not (matched_nodes or matched_ways or matched_rels):
        log(f"  [telecom] no features; empty parquet skipped")
        return True

    log(f"  [telecom] pass 2 (gather {len(target_node_ids):,} member locations)...")
    t = time.time()
    locs: dict = {}

    class P2(osmium.SimpleHandler):
        def node(self, n):
            if n.id in target_node_ids and n.location.valid():
                locs[n.id] = (n.location.lat, n.location.lon)

    if target_node_ids:
        P2().apply_file(source_pbf)
    log(f"  [telecom] pass 2 done in {time.time()-t:.0f}s -> "
        f"{len(locs):,}/{len(target_node_ids):,} resolved")

    rows = []
    for nid, lat, lon, tags in matched_nodes:
        rows.append({
            "asset_id":   f"osm_node_{nid}",
            "asset_type": "telecom.data_center",
            "lat": lat, "lon": lon,
            "name": tags.get("name", ""), "source": "osm_geofabrik",
        })
    for wid, node_ids, tags in matched_ways:
        coords = [locs[nid] for nid in node_ids if nid in locs]
        if not coords:
            continue
        lat = sum(c[0] for c in coords) / len(coords)
        lon = sum(c[1] for c in coords) / len(coords)
        rows.append({
            "asset_id":   f"osm_way_{wid}",
            "asset_type": "telecom.data_center",
            "lat": lat, "lon": lon,
            "name": tags.get("name", ""), "source": "osm_geofabrik",
        })
    for rid, members, tags in matched_rels:
        node_refs = [ref for (mt, ref) in members if mt == 'n']
        coords = [locs[nid] for nid in node_refs if nid in locs]
        if not coords:
            continue
        lat = sum(c[0] for c in coords) / len(coords)
        lon = sum(c[1] for c in coords) / len(coords)
        rows.append({
            "asset_id":   f"osm_relation_{rid}",
            "asset_type": "telecom.data_center",
            "lat": lat, "lon": lon,
            "name": tags.get("name", ""), "source": "osm_geofabrik",
        })

    df = pd.DataFrame(rows)
    log(f"  [telecom] extracted {len(df)} rows")
    if len(df) == 0:
        return True

    df.to_parquet(extract_path, index=False)
    log(f"  [telecom] saved -> {extract_path}")

    # Dedup (trivial for small counts)
    dedup = Deduplicator()
    clean, _ = dedup.run(df)
    log(f"  [telecom] dedup -> {len(clean)} kept")
    clean.to_parquet(dedup_path, index=False)
    log(f"  [telecom] saved -> {dedup_path}")
    return True


def delete_old_power_only(region: str) -> None:
    fname = OLD_POWER_ONLY_FILES.get(region)
    if fname is None:
        return
    path = os.path.join(OLD_POWER_ONLY_DIR, fname)
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / 1_048_576
        try:
            os.remove(path)
            log(f"  deleted old buggy power_only file ({size_mb:.0f} MB): {fname}")
        except OSError as e:
            log(f"  could not delete {fname}: {e}")


def main():
    run_start = time.time()
    log(f"=== overnight pipeline start ===")
    log(f"Regions: {[r for r, _ in REGIONS]}")
    log(f"Sectors per region: {[s[0] for s in PREFILTER_SECTORS]} + telecom")

    region_results: dict = {}

    for region, source_pbf in REGIONS:
        log("")
        log(f"==================== {region} ====================")
        log(f"Source: {source_pbf}")
        if not os.path.exists(source_pbf):
            log(f"  MISSING source PBF; skipping region")
            region_results[region] = "MISSING_SOURCE"
            continue

        region_start = time.time()
        sector_status: dict = {}

        for preset, sector_name, tag_keys in PREFILTER_SECTORS:
            try:
                run_prefilter_sector(region, source_pbf, preset, sector_name, tag_keys)
                sector_status[preset] = "ok"
            except Exception as e:
                sector_status[preset] = f"FAIL: {e!r}"
                log(f"  [{preset}] FAILED: {e}")
                traceback.print_exc()

        # Telecom (no pre-filter)
        try:
            run_telecom(region, source_pbf)
            sector_status["telecom"] = "ok"
        except Exception as e:
            sector_status["telecom"] = f"FAIL: {e!r}"
            log(f"  [telecom] FAILED: {e}")
            traceback.print_exc()

        # Disk cleanup: drop the buggy old power_only file if substation succeeded
        if sector_status.get("substation") == "ok":
            delete_old_power_only(region)

        elapsed_min = (time.time() - region_start) / 60
        log(f"--- {region} done in {elapsed_min:.1f} min: {sector_status}")
        region_results[region] = sector_status

    total_min = (time.time() - run_start) / 60
    log("")
    log(f"=== overnight pipeline end (total {total_min:.1f} min) ===")
    for region, status in region_results.items():
        log(f"  {region}: {status}")

    # Also clean up the two stray old buggy power_only files for
    # central-america + maine (not regions in this pipeline, but old
    # files clutter the power_only/ dir).
    for stray in ("central-america", "maine"):
        delete_old_power_only(stray)


if __name__ == "__main__":
    main()
