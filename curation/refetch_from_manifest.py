"""
refetch_from_manifest.py
------------------------
Rebuild a curated dataset from the exact STAC scenes its manifest names.

A normal fetch asks Planetary Computer for the least-cloudy scene in the date
window. That is a question whose answer can change: the archive gets
reprocessed, item ids and pixels move, and a rerun months later can quietly
produce different imagery with nothing in the output to show it happened.

This reads the `scenes` block each manifest record carries and pins every tile
to those item ids, so the rebuild either reproduces the dataset or fails loudly
on the scenes that are gone.

Manifests written before `scenes` existed carry no item ids. Those tiles cannot
be pinned and are reported as unpinnable rather than silently refetched with a
fresh search; pass --allow-unpinned to fetch them by search anyway.

Usage (from the repo root):
    python -m curation.refetch_from_manifest \\
        --dataset-dir data/curated_datasets/dataset_maine_stac_v1 \\
        --output-dir  data/curated_datasets/dataset_maine_stac_v1_refetch

    python -m curation.refetch_from_manifest --dataset-dir <dir> --report-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import pandas as pd

from .paths import CHECKPOINTS_DIR


def load_pins(manifest_path: str) -> tuple[dict, list, list]:
    """read a manifest into ({asset_id: {modality: item_id}}, rows, unpinnable).

    rows carries the asset_id/asset_type/lat/lon the fetcher needs, so the
    original asset table is not required.
    """
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    pins, rows, unpinnable = {}, [], []
    for rec in manifest.get("records", []):
        asset_id = rec["asset_id"]
        rows.append({
            "asset_id":   asset_id,
            "asset_type": rec["asset_type"],
            "lat":        rec["lat"],
            "lon":        rec["lon"],
        })
        scenes = rec.get("scenes") or {}
        ids = {mod: meta.get("item_id") for mod, meta in scenes.items()
               if isinstance(meta, dict) and meta.get("item_id")}
        if ids:
            pins[asset_id] = ids
        else:
            unpinnable.append(asset_id)
    return pins, rows, unpinnable


def report(manifest_path: str) -> dict:
    """summarise how much of a manifest can be pinned, without fetching."""
    pins, rows, unpinnable = load_pins(manifest_path)
    per_modality = Counter()
    for ids in pins.values():
        per_modality.update(ids.keys())

    print(f"manifest: {manifest_path}")
    print(f"  tiles           : {len(rows):,}")
    print(f"  pinnable        : {len(pins):,}")
    print(f"  unpinnable      : {len(unpinnable):,}")
    if per_modality:
        print("  item ids per modality:")
        for mod, n in sorted(per_modality.items()):
            print(f"    {mod:20s} {n:,}")
    if unpinnable:
        print("\n  this manifest predates scene provenance, wholly or in part.")
        print("  those tiles can only be refetched by search, which does not")
        print("  guarantee the same imagery. rerun with --allow-unpinned to")
        print("  do that anyway.")
    return {"n_tiles": len(rows), "n_pinned": len(pins),
            "n_unpinnable": len(unpinnable)}


def refetch(dataset_dir: str, output_dir: str, allow_unpinned: bool = False,
            modalities: list | None = None, buffer_m: int = 300,
            max_workers: int = 16) -> dict:
    from .stac_imagery import STACImageryFetcher
    from .dataset import DatasetAssembler

    manifest_path = os.path.join(dataset_dir, "manifest.json")
    pins, rows, unpinnable = load_pins(manifest_path)

    if unpinnable and not allow_unpinned:
        raise SystemExit(
            f"{len(unpinnable):,} of {len(rows):,} tiles carry no scene ids, so "
            f"they cannot be reproduced exactly.\nRun with --report-only to "
            f"inspect, or --allow-unpinned to refetch them by search."
        )

    with open(manifest_path, encoding="utf-8") as f:
        modalities = modalities or json.load(f).get(
            "modalities", ["sentinel2_ms", "sentinel1"])

    df = pd.DataFrame(rows)
    if not allow_unpinned:
        df = df[df["asset_id"].isin(pins)].reset_index(drop=True)

    print(f"refetching {len(df):,} tiles pinned to their recorded scenes")
    print(f"  modalities: {modalities}")
    print(f"  output    : {output_dir}")

    fetcher = STACImageryFetcher(
        buffer_m        = buffer_m,
        modalities      = list(modalities),
        temporal_stack  = False,
        scene_pins      = pins,
        checkpoint_path = str(CHECKPOINTS_DIR /
                              f"{os.path.basename(output_dir)}_refetch.pkl"),
        adaptive_concurrency = True,
        start_workers   = min(8, max_workers),
        max_workers     = max_workers,
    )
    tiles = fetcher.fetch_all(df)

    ok = [t for t in tiles if t.status == "ok" and t.image is not None]
    print(f"\n  refetched {len(ok):,} / {len(tiles):,}")

    # report scenes that no longer resolve -- the whole point of pinning is
    # that this surfaces rather than being papered over with a fresh search
    lost = [t.asset_id for t in tiles if t.status != "ok" and t.asset_id in pins]
    if lost:
        print(f"  {len(lost):,} pinned scenes no longer resolve, for example:")
        for a in lost[:5]:
            print(f"    {a}: {pins[a]}")

    summary = DatasetAssembler(output_dir).assemble(ok, None)
    return {"n_requested": len(df), "n_refetched": len(ok),
            "n_lost": len(lost), "n_assembled": len(summary)}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", required=True,
                    help="curated dataset directory holding manifest.json")
    ap.add_argument("--output-dir",
                    help="where to write the rebuilt dataset "
                         "(required unless --report-only)")
    ap.add_argument("--report-only", action="store_true",
                    help="report how much of the manifest is pinnable, fetch nothing")
    ap.add_argument("--allow-unpinned", action="store_true",
                    help="also refetch tiles with no recorded scene, by search")
    ap.add_argument("--buffer-m", type=int, default=300)
    ap.add_argument("--max-workers", type=int, default=16)
    args = ap.parse_args()

    manifest = os.path.join(args.dataset_dir, "manifest.json")
    if not os.path.exists(manifest):
        raise SystemExit(f"no manifest.json in {args.dataset_dir}")

    if args.report_only:
        report(manifest)
        return
    if not args.output_dir:
        raise SystemExit("--output-dir is required unless --report-only")
    if os.path.abspath(args.output_dir) == os.path.abspath(args.dataset_dir):
        raise SystemExit("--output-dir must differ from --dataset-dir")

    result = refetch(args.dataset_dir, args.output_dir,
                     allow_unpinned=args.allow_unpinned,
                     buffer_m=args.buffer_m, max_workers=args.max_workers)
    print("\n" + json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
