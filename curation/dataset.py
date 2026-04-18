"""
dataset.py
----------
Assembles accepted tiles into a structured training dataset.

Output format is COCO-inspired — a directory of image files plus a JSON
manifest describing each record. This is the standard format expected by
most training pipelines including YOLO, torchvision, and HuggingFace datasets.

Directory structure produced:
    dataset/
    ├── images/
    │   ├── osm_node_123456_sentinel2.npy    # raw numpy arrays
    │   ├── osm_node_123456_naip.npy
    │   └── ...
    ├── manifest.json                         # full metadata per tile
    └── summary.csv                           # one row per tile, no image data

Each record in manifest.json contains:
    asset_id, asset_type, source, lat, lon, bbox,
    image_date, confidence, image_file, image_shape

Usage:
    from dataset import DatasetAssembler
    assembler = DatasetAssembler("data/dataset_maine_v1")
    assembler.assemble(accepted_tiles, triage_results)
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Optional, Dict
from legacy.imagery import TileResult
from triage import TriageResult


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DatasetAssembler:
    """
    Assembles accepted tiles into a structured training dataset.

    Parameters
    ----------
    output_dir : str
        Root directory for the dataset.
        !! CHANGE THIS for each new dataset version to avoid overwriting.
        Convention: data/dataset_<region>_<version>/
        Examples:
          "data/dataset_maine_v1"
          "data/dataset_northeast_usa_v1"
          "data/dataset_global_energy_v1"
    """

    def __init__(self, output_dir: str):
        self.output_dir  = output_dir
        self.images_dir  = os.path.join(output_dir, "images")
        self.manifest_path = os.path.join(output_dir, "manifest.json")
        self.summary_path  = os.path.join(output_dir, "summary.csv")

    def assemble(
        self,
        accepted_tiles: List[TileResult],
        triage_results: Optional[List[TriageResult]] = None,
    ) -> pd.DataFrame:
        """
        Saves images to disk and writes manifest.json + summary.csv.

        Parameters
        ----------
        accepted_tiles  : list of TileResult — from triager.filter_accepted()
        triage_results  : list of TriageResult — optional, used to attach
                          confidence and reason to each record

        Returns
        -------
        summary DataFrame
        """
        os.makedirs(self.images_dir, exist_ok=True)

        # Build a lookup from asset_id+source to triage result
        triage_lookup: Dict[str, TriageResult] = {}
        if triage_results:
            for r in triage_results:
                key = f"{r.asset_id}__{r.source}"
                triage_lookup[key] = r

        records  = []
        n_saved  = 0
        n_failed = 0

        for i, tile in enumerate(accepted_tiles):
            if tile.image is None:
                n_failed += 1
                continue

            # Build filename
            safe_id   = tile.asset_id.replace("/", "_").replace(":", "_")
            filename  = f"{safe_id}_{tile.source}.npy"
            filepath  = os.path.join(self.images_dir, filename)

            # Save image as numpy array
            try:
                np.save(filepath, tile.image)
                n_saved += 1
            except Exception as e:
                print(f"  Warning: could not save {filename}: {e}")
                n_failed += 1
                continue

            # Retrieve triage metadata
            triage_key = f"{tile.asset_id}__{tile.source}"
            triage     = triage_lookup.get(triage_key)

            record = {
                "asset_id":    tile.asset_id,
                "asset_type":  tile.asset_type,
                "source":      tile.source,
                "lat":         tile.lat,
                "lon":         tile.lon,
                "bbox":        list(tile.bbox),
                "image_date":  tile.image_date,
                "confidence":  triage.confidence if triage else "high",
                "triage_reason": triage.reason if triage else "",
                "image_file":  filename,
                "image_shape": list(tile.image.shape),
            }
            records.append(record)

            if (i + 1) % 1000 == 0 or (i + 1) == len(accepted_tiles):
                print(f"  [{i+1}/{len(accepted_tiles)}] "
                      f"saved={n_saved} failed={n_failed}")

        # Write manifest.json
        manifest = {
            "created_at":   datetime.utcnow().isoformat() + "Z",
            "n_tiles":      len(records),
            "asset_types":  list({r["asset_type"] for r in records}),
            "sources":      list({r["source"] for r in records}),
            "records":      records,
        }
        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # Write summary.csv (no image data — lightweight)
        summary_df = pd.DataFrame([
            {k: v for k, v in r.items() if k not in ("bbox", "image_shape")}
            for r in records
        ])
        summary_df.to_csv(self.summary_path, index=False)

        print(f"Dataset assembled: {n_saved} tiles saved, {n_failed} failed")
        print(f"  Images:   {self.images_dir}")
        print(f"  Manifest: {self.manifest_path}")
        print(f"  Summary:  {self.summary_path}")

        if not summary_df.empty:
            print(f"\nTiles by asset type:")
            print(summary_df["asset_type"].value_counts().to_string())
            print(f"\nTiles by source:")
            print(summary_df["source"].value_counts().to_string())

        return summary_df

    def load_manifest(self) -> dict:
        """Loads the manifest.json for this dataset."""
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(
                f"No manifest found at {self.manifest_path}. "
                "Run assemble() first."
            )
        with open(self.manifest_path) as f:
            return json.load(f)

    def load_tile(self, record: dict) -> np.ndarray:
        """
        Loads a single tile image from disk given a manifest record.
        Returns a (3, rows, cols) uint8 numpy array.
        """
        filepath = os.path.join(self.images_dir, record["image_file"])
        return np.load(filepath)

    def stats(self) -> None:
        """Prints dataset statistics from the manifest."""
        manifest = self.load_manifest()
        print(f"=== Dataset: {self.output_dir} ===")
        print(f"Created:     {manifest['created_at']}")
        print(f"Total tiles: {manifest['n_tiles']}")
        print(f"Asset types: {', '.join(manifest['asset_types'])}")
        print(f"Sources:     {', '.join(manifest['sources'])}")

        df = pd.DataFrame([
            {k: v for k, v in r.items() if k not in ("bbox", "image_shape")}
            for r in manifest["records"]
        ])
        print(f"\nTiles by asset type:")
        print(df["asset_type"].value_counts().to_string())
        print(f"\nTiles by source:")
        print(df["source"].value_counts().to_string())
        print(f"\nTiles by confidence:")
        print(df["confidence"].value_counts().to_string())


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from curation.sources import GeoFabrikSource
    from curation.legacy.imagery import ImageryFetcher
    from curation.qc import QualityChecker
    from curation.triage import RuleBasedTriager

    # !! CHANGE THIS for each new dataset version
    OUTPUT_DIR = "data/dataset_us-northeast_v1"
    INPUT_CSV  = "data/us-northeast_deduped_assets.csv"

    if not os.path.exists(INPUT_CSV):
        print(f"No CSV at {INPUT_CSV} — run sources.py and deduplication.py first.")
    else:
        df = pd.read_csv(INPUT_CSV)

        # Sample for demo
        df_sample = (
            df.groupby("asset_type", group_keys=False)
              .apply(lambda g: g.sample(min(len(g), 2), random_state=42))
              .reset_index(drop=True)
        )

        print(f"Fetching tiles for {len(df_sample)} assets...")
        fetcher = ImageryFetcher(buffer_m=150, sources=["sentinel2"])
        tiles   = fetcher.fetch_all(df_sample)

        print("\nRunning QC...")
        checker    = QualityChecker()
        qc_results = checker.check_all(tiles)
        clean      = checker.filter_ok(qc_results)

        print("\nRunning triage...")
        triager        = RuleBasedTriager()
        triage_results = triager.triage_all(clean)
        accepted       = triager.filter_accepted(triage_results)

        print(f"\nAssembling dataset from {len(accepted)} accepted tiles...")
        assembler = DatasetAssembler(OUTPUT_DIR)
        summary   = assembler.assemble(accepted, triage_results)

        print()
        assembler.stats()