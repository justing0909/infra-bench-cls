"""
dataset.py
----------
assembles accepted tiles into a structured training dataset.

handles multimodal tiles (sentinel2_ms, sentinel1, landsat_thermal, naip)
and, when a temporal stack is present, saves image_stack as
<asset_id>_temporal.npy (T,C,H,W) alongside the single best composite
<asset_id>.npy (C,H,W). modality and band metadata go into manifest.json.
output directories carry a _stac_v1 suffix; the _sentinel_v1 suffix belongs
to the retired Google Earth Engine path.

directory structure:
    dataset_<region>_stac_v1/
    ├── images/
    │   ├── osm_node_123_stac_sentinel2_ms+sentinel1.npy       (C, H, W)
    │   ├── osm_node_123_stac_sentinel2_ms+sentinel1_temporal.npy  (T, C, H, W)
    │   └── ...
    ├── manifest.json
    └── summary.csv

manifest.json records per tile:
    asset_id, asset_type, source, lat, lon, bbox,
    image_date, image_dates, confidence, image_file, temporal_file,
    image_shape, stack_shape, modalities, n_bands, n_timesteps, scenes

scenes maps each modality to the STAC item its pixels came from
(collection, item_id, datetime, cloud_cover).

Usage:
    from curation.dataset import DatasetAssembler
    assembler = DatasetAssembler("data/curated_datasets/dataset_central-america_stac_v1")
    assembler.assemble(accepted_tiles, triage_results)
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Optional, Dict

from .helpers.tile_types import TileResult
from .triage import TriageResult


# ---------------------------------------------------------------------------
# version suffix constants
# ---------------------------------------------------------------------------

VERSION_SUFFIX_STAC = "stac_v1"
VERSION_SUFFIX_GEE  = "sentinel_v1"


def dataset_output_dir(region: str, use_stac: bool = True,
                        root: str = "data/curated_datasets") -> str:
    """
    returns the canonical output directory path for a region's dataset.

    Examples
    --------
    dataset_output_dir("central-america", use_stac=True)
    -> "data/curated_datasets/dataset_central-america_stac_v1"

    dataset_output_dir("central-america", use_stac=False)
    -> "data/curated_datasets/dataset_central-america_sentinel_v1"
    """
    suffix = VERSION_SUFFIX_STAC if use_stac else VERSION_SUFFIX_GEE
    return os.path.join(root, f"dataset_{region}_{suffix}")


# ---------------------------------------------------------------------------
# main class
# ---------------------------------------------------------------------------

class DatasetAssembler:
    """
    assembles accepted tiles into a structured training dataset.

    Parameters
    ----------
    output_dir : str
        root directory for the dataset.
        convention: data/curated_datasets/dataset_<region>_stac_v1/
        use dataset_output_dir() to generate this consistently.
    """

    def __init__(self, output_dir: str):
        self.output_dir    = output_dir
        self.images_dir    = os.path.join(output_dir, "images")
        self.manifest_path = os.path.join(output_dir, "manifest.json")
        self.summary_path  = os.path.join(output_dir, "summary.csv")

    def assemble(
        self,
        accepted_tiles  : List[TileResult],
        triage_results  : Optional[List[TriageResult]] = None,
    ) -> pd.DataFrame:
        """
        saves images to disk and writes manifest.json + summary.csv.

        for tiles with a temporal stack (image_stack is not None), saves:
          - <asset_id>_<source>.npy            — single best composite (C, H, W)
          - <asset_id>_<source>_temporal.npy   — full stack (T, C, H, W)

        returns summary DataFrame.
        """
        os.makedirs(self.images_dir, exist_ok=True)

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

            safe_id  = tile.asset_id.replace("/", "_").replace(":", "_")
            filename = f"{safe_id}_{tile.source}.npy"
            filepath = os.path.join(self.images_dir, filename)

            # save primary image (C, H, W)
            try:
                np.save(filepath, tile.image)
                n_saved += 1
            except Exception as e:
                print(f"  Warning: could not save {filename}: {e}")
                n_failed += 1
                continue

            # save temporal stack (T, C, H, W) if present
            temporal_filename = None
            if tile.image_stack is not None:
                temporal_filename = f"{safe_id}_{tile.source}_temporal.npy"
                temporal_filepath = os.path.join(self.images_dir, temporal_filename)
                try:
                    np.save(temporal_filepath, tile.image_stack)
                except Exception as e:
                    print(f"  Warning: could not save temporal stack "
                          f"{temporal_filename}: {e}")
                    temporal_filename = None

            triage_key = f"{tile.asset_id}__{tile.source}"
            triage     = triage_lookup.get(triage_key)

            modalities  = getattr(tile, "modalities",  ["sentinel2_rgb"])
            n_bands     = getattr(tile, "n_bands",     tile.image.shape[0])
            n_timesteps = getattr(tile, "n_timesteps", 1)
            image_dates = getattr(tile, "image_dates", [])

            record = {
                "asset_id":          tile.asset_id,
                "asset_type":        tile.asset_type,
                "source":            tile.source,
                "lat":               tile.lat,
                "lon":               tile.lon,
                "bbox":              list(tile.bbox),
                "image_date":        tile.image_date,
                "image_dates":       image_dates,
                # getattr so manifests can still be reassembled from tiles
                # produced before scenes existed
                "scenes":            getattr(tile, "scenes", {}) or {},
                "confidence":        triage.confidence if triage else "high",
                "triage_reason":     triage.reason if triage else "",
                "image_file":        filename,
                "temporal_file":     temporal_filename,
                "image_shape":       list(tile.image.shape),
                "stack_shape":       list(tile.image_stack.shape)
                                     if tile.image_stack is not None else [],
                "modalities":        modalities,
                "n_bands":           n_bands,
                "n_timesteps":       n_timesteps,
            }
            records.append(record)

            if (i + 1) % 1000 == 0 or (i + 1) == len(accepted_tiles):
                print(f"  [{i+1}/{len(accepted_tiles)}] "
                      f"saved={n_saved} failed={n_failed}")

        # collect per-modality tile counts for timing log
        modality_counts: Dict[str, int] = {}
        for r in records:
            key = "+".join(r["modalities"])
            modality_counts[key] = modality_counts.get(key, 0) + 1

        # write manifest.json
        manifest = {
            "created_at":      datetime.utcnow().isoformat() + "Z",
            "n_tiles":         len(records),
            "asset_types":     list({r["asset_type"] for r in records}),
            "sources":         list({r["source"] for r in records}),
            "modalities":      list({m for r in records
                                     for m in r["modalities"]}),
            "modality_counts": modality_counts,
            "has_temporal":    any(r["temporal_file"] for r in records),
            "records":         records,
        }
        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # write summary.csv (lightweight — no large array fields)
        _skip = {"bbox", "image_shape", "stack_shape", "image_dates", "scenes"}
        summary_df = pd.DataFrame([
            {k: ("+".join(v) if k == "modalities" else v)
             for k, v in r.items() if k not in _skip}
            for r in records
        ])
        summary_df.to_csv(self.summary_path, index=False)

        print(f"\nDataset assembled: {n_saved} tiles saved, {n_failed} failed")
        print(f"  Images:   {self.images_dir}")
        print(f"  Manifest: {self.manifest_path}")
        print(f"  Summary:  {self.summary_path}")

        if not summary_df.empty:
            print(f"\nTiles by asset type:")
            print(summary_df["asset_type"].value_counts().to_string())
            print(f"\nTiles by source/modality:")
            print(summary_df["source"].value_counts().to_string())
            if "n_timesteps" in summary_df.columns:
                temporal = summary_df[summary_df["n_timesteps"] > 1]
                print(f"\nTemporal tiles: {len(temporal)} / {len(summary_df)}")

        return summary_df

    def load_manifest(self) -> dict:
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(
                f"No manifest found at {self.manifest_path}. "
                "Run assemble() first."
            )
        with open(self.manifest_path) as f:
            return json.load(f)

    def load_tile(self, record: dict) -> np.ndarray:
        """loads the primary (C, H, W) tile for a manifest record."""
        filepath = os.path.join(self.images_dir, record["image_file"])
        return np.load(filepath)

    def load_temporal_tile(self, record: dict) -> Optional[np.ndarray]:
        """loads the temporal (T, C, H, W) stack for a manifest record, or None."""
        if not record.get("temporal_file"):
            return None
        filepath = os.path.join(self.images_dir, record["temporal_file"])
        if not os.path.exists(filepath):
            return None
        return np.load(filepath)

    def stats(self) -> None:
        manifest = self.load_manifest()
        print(f"=== Dataset: {self.output_dir} ===")
        print(f"Created:      {manifest['created_at']}")
        print(f"Total tiles:  {manifest['n_tiles']}")
        print(f"Asset types:  {', '.join(manifest['asset_types'])}")
        print(f"Modalities:   {', '.join(manifest.get('modalities', []))}")
        print(f"Has temporal: {manifest.get('has_temporal', False)}")

        df = pd.DataFrame([
            {k: ("+".join(v) if k == "modalities" else v)
             for k, v in r.items()
             if k not in ("bbox", "image_shape", "stack_shape", "image_dates",
                          "scenes")}
            for r in manifest["records"]
        ])
        print(f"\nTiles by asset type:")
        print(df["asset_type"].value_counts().to_string())
        print(f"\nTiles by source:")
        print(df["source"].value_counts().to_string())
        if "n_timesteps" in df.columns:
            print(f"\nTimestep distribution:")
            print(df["n_timesteps"].value_counts().to_string())
        print(f"\nModality combination counts:")
        if "modalities" in df.columns:
            print(df["modalities"].value_counts().to_string())