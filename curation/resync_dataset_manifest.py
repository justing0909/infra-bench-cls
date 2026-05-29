"""
resync_dataset_manifest.py
--------------------------
Reconcile an existing curated imagery dataset against an updated deduped
asset parquet, without refetching tiles.

Three buckets after reconciliation:
  - kept    : tile exists in images/ AND asset_id is in the new parquet
  - orphan  : tile exists in images/ but asset_id is NOT in the new parquet
              (happens when the corrected dedup collapsed that asset away)
  - missing : asset_id is in the new parquet but no tile exists.
              IMPORTANT: this set typically mixes two very different
              populations:
                (a) assets recovered by post-hoc bug fixes that were
                    never in the original imagery scope,
                (b) assets that the original dataset intentionally
                    sampled out (e.g. the 25k-tile-per-region cap used
                    in the May 2026 STAC pass).
              For sample-capped datasets, (b) dominates and the
              `missing_from_dataset.csv` output is informational, NOT
              an actionable refetch list.

The script rewrites summary.csv + manifest.json against only the `kept`
set and writes:
  - missing_from_dataset.csv — every asset in the new parquet that has
                               no tile on disk. Inspect manually before
                               feeding into any refetch step.

By default, orphan .npy files are left on disk for safety. Use
`--delete-orphans` to remove them.

Usage:
    python -m curation.resync_dataset_manifest \\
        --dataset-dir data/curated_datasets/dataset_central-america_stac_v1 \\
        --deduped-parquet data/PIPELINE/02-deduped-assets/central-america_deduped_assets_substations.parquet

Or call `resync_dataset()` directly from Python.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

import pandas as pd


# Filename pattern: <asset_id>_<source-tag>.npy where asset_id is e.g.
# osm_way_12345 or osm_node_12345 or osm_relation_12345.
_FILENAME_RE = re.compile(r"^(osm_(?:node|way|relation)_\d+)_.+\.npy$")


def _asset_id_from_filename(name: str) -> str | None:
    m = _FILENAME_RE.match(name)
    return m.group(1) if m else None


def resync_dataset(
    dataset_dir:      str,
    deduped_parquet:  str,
    delete_orphans:   bool = False,
    write_manifest:   bool = True,
) -> dict:
    """
    Reconcile dataset_dir against deduped_parquet. Returns a stats dict.

    Side effects (when write_manifest=True):
      - Overwrites <dataset_dir>/summary.csv with kept-only rows.
      - Overwrites <dataset_dir>/manifest.json with updated counts.
      - Writes <dataset_dir>/to_refetch.csv listing missing asset_ids.
      - When delete_orphans=True, removes orphan .npy files in images/.
    """
    dataset_dir = os.path.abspath(dataset_dir)
    images_dir  = os.path.join(dataset_dir, "images")
    summary_path  = os.path.join(dataset_dir, "summary.csv")
    manifest_path = os.path.join(dataset_dir, "manifest.json")
    refetch_path  = os.path.join(dataset_dir, "missing_from_dataset.csv")

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    if not os.path.exists(deduped_parquet):
        raise FileNotFoundError(f"Deduped parquet not found: {deduped_parquet}")

    deduped_df = pd.read_parquet(deduped_parquet)
    valid_ids: set[str] = set(deduped_df["asset_id"].tolist())

    # Catalogue files in images/ by asset_id (multiple tile variants OK).
    files_by_asset: dict[str, list[str]] = {}
    skipped: list[str] = []
    for fn in os.listdir(images_dir):
        if not fn.endswith(".npy"):
            continue
        aid = _asset_id_from_filename(fn)
        if aid is None:
            skipped.append(fn)
            continue
        files_by_asset.setdefault(aid, []).append(fn)

    on_disk_ids = set(files_by_asset.keys())
    kept_ids    = on_disk_ids & valid_ids
    orphan_ids  = on_disk_ids - valid_ids
    missing_ids = valid_ids - on_disk_ids

    stats = {
        "dataset_dir":        dataset_dir,
        "deduped_parquet":    deduped_parquet,
        "parquet_rows":       len(deduped_df),
        "files_on_disk":      sum(len(v) for v in files_by_asset.values()),
        "assets_on_disk":     len(on_disk_ids),
        "skipped_filenames":  len(skipped),
        "kept_assets":        len(kept_ids),
        "orphan_assets":      len(orphan_ids),
        "missing_assets":     len(missing_ids),
    }

    if not write_manifest:
        return stats

    # Rewrite summary.csv against the new kept set. If an old summary
    # exists, preserve its columns and only retain kept-asset rows; else
    # fall back to the parquet's columns.
    if os.path.exists(summary_path):
        old_summary = pd.read_csv(summary_path)
        new_summary = old_summary[old_summary["asset_id"].isin(kept_ids)].copy()
    else:
        new_summary = deduped_df[deduped_df["asset_id"].isin(kept_ids)].copy()
    new_summary.to_csv(summary_path, index=False)

    # Rewrite manifest.json — preserve existing keys, update counts.
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {}
    manifest["resynced_at"] = datetime.now(timezone.utc).isoformat()
    manifest["resynced_against"] = os.path.basename(deduped_parquet)
    manifest["n_tiles_before_resync"] = stats["files_on_disk"]
    manifest["n_tiles"]               = len(new_summary)
    manifest["n_assets_kept"]         = stats["kept_assets"]
    manifest["n_assets_orphan"]       = stats["orphan_assets"]
    manifest["n_assets_to_refetch"]   = stats["missing_assets"]
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # missing_from_dataset.csv — rows from the deduped parquet whose tiles
    # are absent. NOTE: this typically includes both (a) post-fix recovered
    # assets and (b) deliberately sampled-out assets. Treat as informational.
    missing_df = deduped_df[deduped_df["asset_id"].isin(missing_ids)].copy()
    missing_df.to_csv(refetch_path, index=False)

    # Optional cleanup of orphan tiles.
    if delete_orphans and orphan_ids:
        for aid in orphan_ids:
            for fn in files_by_asset.get(aid, ()):
                try:
                    os.remove(os.path.join(images_dir, fn))
                except OSError:
                    pass

    return stats


def _format_stats(s: dict) -> str:
    return (
        f"  parquet rows         : {s['parquet_rows']:>10,}\n"
        f"  tiles on disk        : {s['files_on_disk']:>10,}  "
        f"({s['assets_on_disk']:,} unique assets)\n"
        f"  kept (intersection)  : {s['kept_assets']:>10,}\n"
        f"  orphan (tile only)   : {s['orphan_assets']:>10,}\n"
        f"  missing (parquet only): {s['missing_assets']:>10,}\n"
        f"  unrecognized files   : {s['skipped_filenames']:>10,}"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir",     required=True,
                   help="Path to dataset_<region>_<version>/")
    p.add_argument("--deduped-parquet", required=True,
                   help="Path to <region>_deduped_assets_<sector>.parquet")
    p.add_argument("--delete-orphans",  action="store_true",
                   help="Remove orphan .npy files from images/. Default: keep.")
    p.add_argument("--dry-run",         action="store_true",
                   help="Report counts but do not write any files.")
    args = p.parse_args()

    stats = resync_dataset(
        dataset_dir     = args.dataset_dir,
        deduped_parquet = args.deduped_parquet,
        delete_orphans  = args.delete_orphans,
        write_manifest  = not args.dry_run,
    )
    print(f"Resync against {os.path.basename(stats['deduped_parquet'])}:")
    print(_format_stats(stats))
    if args.dry_run:
        print("\n(dry run; no files written)")
    else:
        print(f"\n  -> updated summary.csv, manifest.json, missing_from_dataset.csv")
        if stats["missing_assets"] > 500:
            print(f"  NOTE: 'missing' count {stats['missing_assets']:,} is large — "
                  f"likely dominated by deliberately sampled-out assets, not bug recovery.")
        if args.delete_orphans and stats["orphan_assets"]:
            print(f"  -> removed {stats['orphan_assets']} orphan tile sets from images/")


if __name__ == "__main__":
    main()
