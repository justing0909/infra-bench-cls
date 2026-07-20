"""
Dataset wrapper for the curated .npy tile files.

`NpyInfrastructureDataset` reads a manifest / summary CSV produced by
the curation/ pipeline, resolves each tile path, applies per-tile band
selection and percentile normalization, and yields (image, label)
tuples suitable for DataLoader.

Used by both ResNet-18 baselines. Foundation-model notebooks use their
own per-FM Dataset wrappers because they need FM-specific normalization
and band mappings.
"""

from __future__ import annotations

# Changes from original:
#   - parse_band_indices import moved to io.py (single source of truth);
#     the duplicate definition here is removed.
#   - _load_image now emits a one-time warning when the tile has more bands
#     than the requested band_indices — catches silent RGB-only training on
#     7-band sentinel2_ms tiles.
#   - _filter_records warning messages now include the failure reason so
#     dropped tiles are easier to diagnose.

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .io import (
    find_default_summary,
    infer_sector_from_asset_type,
    load_table,
    parse_asset_id_from_filename,
    parse_band_indices,
    percentile_normalize,
    resolve_path,
)


@dataclass
class SampleRecord:
    path: Path
    asset_id: str
    asset_type: str | None = None
    sector: str | None = None


class NpyInfrastructureDataset(Dataset):
    """Generic dataset for infrastructure `.npy` crops.

    Shared across pretraining and all downstream tasks. Assumes as little
    as possible about sector so it stays reusable as the project expands
    beyond power (water, transport, telecom, ...).
    """

    def __init__(
        self,
        dataset_root: str | Path,
        transform: Optional[Callable] = None,
        metadata_file: str | Path | None = None,
        image_column: str = "image_path",
        asset_id_column: str = "asset_id",
        asset_type_column: str = "asset_type",
        band_indices: str = "0,1,2",
        min_valid_size: int = 16,
        max_images: int | None = None,
        require_labels: bool = False,
        allowed_asset_types: list[str] | None = None,
    ):
        self.dataset_root = Path(dataset_root)
        self.transform = transform
        self.band_indices = parse_band_indices(band_indices)
        self.min_valid_size = min_valid_size
        self.require_labels = require_labels
        self.allowed_asset_types = set(allowed_asset_types) if allowed_asset_types else None
        self._band_warning_issued = False

        self.records = self._build_records(
            metadata_file, image_column, asset_id_column, asset_type_column
        )
        self.records = self._filter_records(self.records)
        if max_images is not None:
            self.records = self.records[:max_images]
        if not self.records:
            raise RuntimeError("No valid imagery samples were found.")

    def _discover_npy_files(self) -> list[Path]:
        images_dir = self.dataset_root / "images"
        search_root = images_dir if images_dir.exists() else self.dataset_root
        return sorted(p for p in search_root.rglob("*.npy") if p.is_file())

    def _build_records(
        self,
        metadata_file: str | Path | None,
        image_column: str,
        asset_id_column: str,
        asset_type_column: str,
    ) -> list[SampleRecord]:
        summary_path = (
            resolve_path(self.dataset_root, metadata_file)
            if metadata_file
            else find_default_summary(self.dataset_root)
        )
        file_paths = self._discover_npy_files()
        path_index = {p.name: p for p in file_paths}
        records: list[SampleRecord] = []

        if summary_path and summary_path.exists():
            df = load_table(summary_path)
            possible_image_cols = [
                c for c in [image_column, "image_path", "crop_path", "path", "filename"]
                if c in df.columns
            ]
            image_col = possible_image_cols[0] if possible_image_cols else None
            for _, row in df.iterrows():
                if image_col:
                    raw_path = str(row[image_col])
                    path = resolve_path(self.dataset_root, raw_path)
                    if not path.exists() and Path(raw_path).name in path_index:
                        path = path_index[Path(raw_path).name]
                else:
                    raw_asset_id = str(row.get(asset_id_column, ""))
                    path = next(
                        (p for p in file_paths if raw_asset_id and raw_asset_id in p.stem),
                        None,
                    )
                if path is None or not path.exists() or path.suffix.lower() != ".npy":
                    continue
                asset_id = str(
                    row.get(asset_id_column, parse_asset_id_from_filename(path.name))
                )
                asset_type = row.get(asset_type_column, None)
                asset_type = None if pd.isna(asset_type) else str(asset_type)
                sector = infer_sector_from_asset_type(asset_type) if asset_type else None
                records.append(
                    SampleRecord(path=path, asset_id=asset_id, asset_type=asset_type, sector=sector)
                )
        else:
            for p in file_paths:
                records.append(
                    SampleRecord(path=p, asset_id=parse_asset_id_from_filename(p.name))
                )
        return records

    @staticmethod
    def _infer_hwc(shape: tuple[int, ...]) -> tuple[int, int, int]:
        if len(shape) != 3:
            raise ValueError(f"Expected 3D array, got shape {shape}")
        if shape[0] <= 16 and shape[1] > 16 and shape[2] > 16:
            return shape[1], shape[2], shape[0]
        return shape[0], shape[1], shape[2]

    def _load_image(self, path: Path) -> np.ndarray:
        arr = np.load(path)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D array, got {arr.shape} at {path}")
        # Normalise to HWC
        if arr.shape[0] <= 16 and arr.shape[1] > 16 and arr.shape[2] > 16:
            arr = np.transpose(arr, (1, 2, 0))
        # Warn once if we're silently dropping bands the user might care about
        if not self._band_warning_issued and arr.shape[2] > len(self.band_indices):
            warnings.warn(
                f"Tile has {arr.shape[2]} bands but band_indices={self.band_indices} "
                f"selects only {len(self.band_indices)}. Pass --band-indices explicitly "
                f"to use all bands (e.g. 0,1,2,3,4,5,6 for sentinel2_ms).",
                stacklevel=2,
            )
            self._band_warning_issued = True
        arr = arr[..., self.band_indices]
        arr = percentile_normalize(arr)
        return arr

    def _filter_records(self, records: list[SampleRecord]) -> list[SampleRecord]:
        valid: list[SampleRecord] = []
        n_dropped = {"no_label": 0, "asset_type_filtered": 0, "load_error": 0,
                     "too_small": 0, "too_few_bands": 0}
        for record in records:
            if self.require_labels and not record.asset_type:
                n_dropped["no_label"] += 1
                continue
            if self.allowed_asset_types and record.asset_type not in self.allowed_asset_types:
                n_dropped["asset_type_filtered"] += 1
                continue
            try:
                arr = np.load(record.path, mmap_mode="r")
                h, w, c = self._infer_hwc(arr.shape)
                if h < self.min_valid_size or w < self.min_valid_size:
                    n_dropped["too_small"] += 1
                    continue
                if c < max(self.band_indices) + 1:
                    n_dropped["too_few_bands"] += 1
                    continue
                valid.append(record)
            except Exception:
                n_dropped["load_error"] += 1
                continue
        total_dropped = sum(n_dropped.values())
        if total_dropped > 0:
            reasons = ", ".join(f"{k}={v}" for k, v in n_dropped.items() if v > 0)
            print(f"  NpyInfrastructureDataset: dropped {total_dropped} records ({reasons})")
        return valid

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        image = self._load_image(record.path)
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        return {
            "image": image,
            "path": str(record.path),
            "asset_id": record.asset_id,
            "asset_type": record.asset_type,
            "sector": record.sector,
        }