"""
missing_cells_extract.py
-----------------------
Extract + dedup + sample the 5 v1 cells that were missing as of the
scope-reset turn, plus europe energy (replaces the substations-only
fallback) as a final bonus step.

For each cell, in order:
  1. Pre-filter (or 2-pass direct scan for telecom) -> small sector PBF
  2. Extract via GeoFabrikSource              -> _all_assets_<sector>.parquet
  3. Dedup with per-class thresholds          -> _deduped_assets_<sector>.parquet
  4. Run the v1 sampler on the new parquet    -> _<sector>_v1_sample.parquet

Outputs land in the same directories the previous pipeline used. Each
cell's parquets persist as soon as it completes, so the run is naturally
checkpointed if interrupted.

After all cells finish, the script re-runs the full v1 sampler one more
time to rebuild the summary JSON against the final on-disk state.
"""
from __future__ import annotations
import os
import sys
import time
import traceback
from datetime import datetime

import osmium
import pandas as pd

REPO = r"C:\Users\j.guthrie\Downloads\RESEARCH\infra_fm"
sys.path.insert(0, os.path.join(REPO, "curation"))
sys.path.insert(0, os.path.join(REPO, "curation", "utils"))
os.chdir(REPO)

from sources import GeoFabrikSource
from deduplication import Deduplicator
import sample_v1


OUT_DIR_EXTRACT = os.path.join(REPO, "data", "PIPELINE", "01-extracted-assets")
OUT_DIR_DEDUP   = os.path.join(REPO, "data", "PIPELINE", "02-deduped-assets")

# (region, sector, source_pbf, preset, sector_tag_keys)
# The five required cells, followed by europe energy as the bonus tail.
CELLS = [
    ("north-america", "transport",
     r"D:\north-america-latest.osm.pbf",
     "transport", ["aeroway", "harbour", "railway"]),

    ("north-america", "telecom",
     r"D:\north-america-latest.osm.pbf",
     "telecom",   None),    # telecom uses 2-pass direct scan, no pre-filter

    ("europe",        "water",
     r"D:\europe-latest.osm.pbf",
     "water",     ["man_made"]),

    ("europe",        "transport",
     r"D:\europe-latest.osm.pbf",
     "transport", ["aeroway", "harbour", "railway"]),

    ("europe",        "telecom",
     r"D:\europe-latest.osm.pbf",
     "telecom",   None),

    # Bonus tail: replaces the substations-only fallback europe energy
    # with a proper energy extract from the same source PBF.
    ("europe",        "energy",
     r"D:\europe-latest.osm.pbf",
     "energy",    ["power"]),
]

V1_TARGET    = 10_000
V1_THRESHOLD = 10_000   # simple 10k rule: sample any cell with pop > 10k to 10k
V1_SEED      = 42


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def run_prefilter_cell(region: str, source_pbf: str, preset: str,
                       sector_tag_keys: list) -> None:
    """Standard pre-filter + extract + dedup for energy/water/transport."""
    extract_path = os.path.join(OUT_DIR_EXTRACT,
                                f"{region}_all_assets_{preset}.parquet")
    dedup_path   = os.path.join(OUT_DIR_DEDUP,
                                f"{region}_deduped_assets_{preset}.parquet")

    # Resume optimization: if both expected outputs already exist and are
    # readable, skip pre-filter+extract+dedup entirely. The v1 sample step
    # always runs at the call site after this returns, so the sample stays
    # in sync regardless. We use try/except in case a parquet is corrupted
    # (e.g. 0-byte file from a prior kill mid-write) — in that case we fall
    # through and redo the work.
    if os.path.exists(extract_path) and os.path.exists(dedup_path):
        try:
            n_extract = len(pd.read_parquet(extract_path, columns=["asset_id"]))
            n_dedup   = len(pd.read_parquet(dedup_path,   columns=["asset_id"]))
            log(f"  [{preset}] outputs already exist "
                f"({n_extract:,} extracted, {n_dedup:,} deduped); skipping")
            return
        except Exception as e:
            log(f"  [{preset}] outputs exist but unreadable ({e!r}); will redo")

    # Sector_name for the output PBF filename: "power" for energy preset to
    # keep the *_power_only.osm.pbf convention; else use the preset name.
    sector_name = "power" if preset == "energy" else preset

    log(f"  pre-filter (sector={sector_name}, tags={sector_tag_keys})...")
    t = time.time()
    src = GeoFabrikSource(source_pbf, min_confidence="medium",
                          filter_preset=preset, pre_filter=True)
    pre_filtered = src._pre_filter_pbf_by_tags(
        sector_name=sector_name,
        sector_tag_keys=sector_tag_keys,
    )
    log(f"  pre-filter done in {time.time()-t:.0f}s -> "
        f"{os.path.getsize(pre_filtered)/1_048_576:.1f} MB")

    log("  extract...")
    t = time.time()
    src2 = GeoFabrikSource(pre_filtered, min_confidence="medium",
                           filter_preset=preset, pre_filter=False)
    df = src2.extract_all()
    log(f"  extract done in {time.time()-t:.0f}s -> {len(df):,} rows")
    if len(df) == 0:
        log("  no rows; skipping dedup + sample")
        # Write empty parquet so downstream skip-if-exists works correctly
        df.drop(columns=["osm_tags"], errors="ignore").to_parquet(
            extract_path, index=False)
        return
    df.drop(columns=["osm_tags"], errors="ignore").to_parquet(
        extract_path, index=False)
    log(f"  extract saved -> {extract_path}")

    log("  dedup...")
    t = time.time()
    dedup = Deduplicator()
    clean, _ = dedup.run(df)
    log(f"  dedup done in {time.time()-t:.0f}s -> {len(clean):,} kept "
        f"({100*(len(df)-len(clean))/max(len(df),1):.1f}% removed)")
    clean.drop(columns=["osm_tags"], errors="ignore").to_parquet(
        dedup_path, index=False)
    log(f"  dedup saved -> {dedup_path}")


def run_telecom_cell(region: str, source_pbf: str) -> None:
    """Two-pass direct scan for telecom + dedup (no pre-filter file)."""
    extract_path = os.path.join(OUT_DIR_EXTRACT,
                                f"{region}_all_assets_telecom.parquet")
    dedup_path   = os.path.join(OUT_DIR_DEDUP,
                                f"{region}_deduped_assets_telecom.parquet")

    # Resume optimization: skip the 2-pass direct scan if both outputs
    # already exist. Telecom's full-source scan is hours on big regions, so
    # this is the most valuable place to have the skip. v1 sample still
    # runs at the call site after this returns.
    if os.path.exists(extract_path) and os.path.exists(dedup_path):
        try:
            n_extract = len(pd.read_parquet(extract_path, columns=["asset_id"]))
            n_dedup   = len(pd.read_parquet(dedup_path,   columns=["asset_id"]))
            log(f"  [telecom] outputs already exist "
                f"({n_extract:,} extracted, {n_dedup:,} deduped); skipping")
            return
        except Exception as e:
            log(f"  [telecom] outputs exist but unreadable ({e!r}); will redo")

    matched_nodes = []
    matched_ways  = []
    matched_rels  = []
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
                    if mt == "n":
                        target_node_ids.add(mr)
                matched_rels.append((r.id, members, dict(r.tags)))

    log("  pass 1 (find building=data_center)...")
    t = time.time()
    P1().apply_file(source_pbf)
    log(f"  pass 1 done in {time.time()-t:.0f}s -> "
        f"{len(matched_nodes)} nodes / {len(matched_ways)} ways / "
        f"{len(matched_rels)} relations")
    if not (matched_nodes or matched_ways or matched_rels):
        log("  no data_center features; writing empty parquet")
        empty = pd.DataFrame(columns=["asset_id", "asset_type", "lat",
                                      "lon", "name", "source"])
        empty.to_parquet(extract_path, index=False)
        empty.to_parquet(dedup_path, index=False)
        return

    log(f"  pass 2 (gather {len(target_node_ids):,} member node locations)...")
    t = time.time()
    locs: dict = {}

    class P2(osmium.SimpleHandler):
        def node(self, n):
            if n.id in target_node_ids and n.location.valid():
                locs[n.id] = (n.location.lat, n.location.lon)

    if target_node_ids:
        P2().apply_file(source_pbf)
    log(f"  pass 2 done in {time.time()-t:.0f}s -> "
        f"{len(locs):,}/{len(target_node_ids):,} resolved")

    rows = []
    for nid, lat, lon, tags in matched_nodes:
        rows.append({"asset_id": f"osm_node_{nid}",
                     "asset_type": "telecom.data_center",
                     "lat": lat, "lon": lon,
                     "name": tags.get("name", ""), "source": "osm_geofabrik"})
    for wid, node_ids, tags in matched_ways:
        coords = [locs[nid] for nid in node_ids if nid in locs]
        if not coords:
            continue
        lat = sum(c[0] for c in coords) / len(coords)
        lon = sum(c[1] for c in coords) / len(coords)
        rows.append({"asset_id": f"osm_way_{wid}",
                     "asset_type": "telecom.data_center",
                     "lat": lat, "lon": lon,
                     "name": tags.get("name", ""), "source": "osm_geofabrik"})
    for rid, members, tags in matched_rels:
        node_refs = [ref for (mt, ref) in members if mt == "n"]
        coords = [locs[nid] for nid in node_refs if nid in locs]
        if not coords:
            continue
        lat = sum(c[0] for c in coords) / len(coords)
        lon = sum(c[1] for c in coords) / len(coords)
        rows.append({"asset_id": f"osm_relation_{rid}",
                     "asset_type": "telecom.data_center",
                     "lat": lat, "lon": lon,
                     "name": tags.get("name", ""), "source": "osm_geofabrik"})

    df = pd.DataFrame(rows)
    df.to_parquet(extract_path, index=False)
    log(f"  extract saved -> {extract_path} ({len(df)} rows)")

    dedup = Deduplicator()
    clean, _ = dedup.run(df)
    clean.to_parquet(dedup_path, index=False)
    log(f"  dedup saved -> {dedup_path} ({len(clean)} kept)")


def sample_cell_now(region: str, sector: str) -> None:
    """Run the v1 sampler on a single cell and write its parquet."""
    r = sample_v1.sample_cell(
        region=region, sector=sector,
        target=V1_TARGET, sample_threshold=V1_THRESHOLD, seed=V1_SEED,
    )
    if r["status"] == "ok":
        tag = " (fallback)" if r["fallback"] else ""
        log(f"  v1 sample: {r['input_rows']:,} -> {r['sample_rows']:,} -> "
            f"{r['output_path']}{tag}")
    else:
        log(f"  v1 sample FAILED: {r['status']}")


def main():
    log("=== missing_cells_extract.py start ===")
    log(f"Cells to process: {len(CELLS)} (5 required + europe energy bonus)")
    log("")

    run_start = time.time()
    results: list = []

    for i, (region, sector, source_pbf, preset, sector_tag_keys) in enumerate(CELLS, 1):
        log(f"========== [{i}/{len(CELLS)}] {region} / {sector} ==========")
        log(f"Source: {source_pbf}")
        if not os.path.exists(source_pbf):
            log(f"  MISSING source PBF; skipping")
            results.append((region, sector, "MISSING_SOURCE", 0))
            continue

        cell_start = time.time()
        try:
            if sector == "telecom":
                run_telecom_cell(region, source_pbf)
            else:
                run_prefilter_cell(region, source_pbf, preset, sector_tag_keys)
            sample_cell_now(region, sector)
            results.append((region, sector, "ok",
                            (time.time() - cell_start) / 60))
            log(f"--- {region}/{sector} done in "
                f"{(time.time() - cell_start)/60:.1f} min")
        except Exception as e:
            log(f"  FAILED: {e}")
            traceback.print_exc()
            results.append((region, sector, f"FAIL: {e!r}",
                            (time.time() - cell_start) / 60))
        log("")

    # Rebuild the full summary against the final on-disk state.
    log("Rebuilding full v1 sample summary against on-disk parquets...")
    sample_v1.main = lambda: None  # no-op shim; we call sample_cell directly below
    full_results = []
    for region in sample_v1.V1_REGIONS:
        for sector in sample_v1.SECTORS:
            r = sample_v1.sample_cell(
                region=region, sector=sector,
                target=V1_TARGET, sample_threshold=V1_THRESHOLD, seed=V1_SEED,
            )
            full_results.append(r)

    import json
    summary_path = os.path.join(REPO, "data", "PIPELINE", "03-v1-samples",
                                "_v1_sample_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "target_per_cell":  V1_TARGET,
            "sample_threshold": V1_THRESHOLD,
            "seed":             V1_SEED,
            "regions":          sample_v1.V1_REGIONS,
            "sectors":          sample_v1.SECTORS,
            "results":          full_results,
        }, f, indent=2, default=str)
    log(f"Summary updated -> {summary_path}")

    total_min = (time.time() - run_start) / 60
    log("")
    log(f"=== missing_cells_extract.py end ({total_min:.1f} min) ===")
    for region, sector, status, dur in results:
        log(f"  {region:<16} / {sector:<10}  {dur:6.1f} min  {status}")


if __name__ == "__main__":
    main()
