from __future__ import annotations

# Changes from original:
#   - NPY_PATTERN replaced with a more flexible regex that handles both the
#     legacy GEE naming (sentinel2_gee) and the newer STAC naming
#     (stac_sentinel2_ms+sentinel1, stac_sentinel2_ms+sentinel1_temporal, etc.)
#   - parse_asset_id_from_filename updated to strip any source suffix cleanly
#     regardless of modality combination, so asset IDs stay stable across runs.
#   - Added parse_band_indices here (was duplicated in asset_classification/datasets.py)

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# Matches both legacy GEE filenames and new STAC filenames, e.g.:
#   osm_node_123456_sentinel2_gee.npy
#   osm_way_789_stac_sentinel2_ms+sentinel1.npy
#   osm_node_123456_stac_sentinel2_ms+sentinel1_temporal.npy
#   inferred_solar_cluster_42_sentinel2_gee.npy
NPY_PATTERN = re.compile(
    r"^(?P<prefix>osm_node|osm_way|osm_relation|inferred_solar_cluster)"
    r"_(?P<id>.+?)"
    r"_(?:stac_|)(?:sentinel2_gee|sentinel2_ms|sentinel2_rgb|sentinel1|landsat_thermal|naip)"
    r"(?:[+\w]*)?"          # additional modalities joined with +
    r"(?:_temporal)?"       # optional temporal suffix
    r"\.npy$"
)


def resolve_path(dataset_root: str | Path, maybe_relative: str | Path) -> Path:
    p = Path(maybe_relative)
    if p.exists():
        return p
    return (Path(dataset_root) / p).resolve()


def parse_asset_id_from_filename(filename: str) -> str:
    """
    Extracts a stable asset_id from an npy filename regardless of source suffix.

    Examples
    --------
    osm_node_123456_sentinel2_gee.npy          -> osm_node_123456
    osm_way_789_stac_sentinel2_ms+sentinel1.npy -> osm_way_789
    inferred_solar_cluster_42_sentinel2_gee.npy -> inferred_solar_cluster_42
    unknown_format.npy                          -> unknown_format  (stem fallback)
    """
    match = NPY_PATTERN.match(Path(filename).name)
    if not match:
        return Path(filename).stem
    prefix = match.group("prefix")
    value = match.group("id")
    return f"{prefix}_{value}"


def parse_band_indices(text: str) -> list[int]:
    """Parses a comma-separated band index string, e.g. '0,1,2' -> [0, 1, 2]."""
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def infer_sector_from_asset_type(asset_type: str) -> str:
    if not asset_type:
        return "unknown"
    return asset_type.split(".", 1)[0]


def load_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    raise ValueError(f"Unsupported table type: {p}")


def find_default_summary(dataset_root: str | Path) -> Optional[Path]:
    root = Path(dataset_root)
    for name in ["summary.csv", "summary.parquet"]:
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


def percentile_normalize(image: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    img = image.astype(np.float32)
    low = np.percentile(img, lo)
    high = np.percentile(img, hi)
    if high <= low:
        return np.clip(img / 255.0, 0.0, 1.0)
    return np.clip((img - low) / (high - low), 0.0, 1.0)