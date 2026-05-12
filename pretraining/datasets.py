from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from downstream.common.io import resolve_path


VALID_SUFFIXES = {".npy"}


def discover_npy_images(dataset_root: str | Path) -> list[Path]:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    images_dir = root / "images"
    search_root = images_dir if images_dir.exists() else root
    return sorted([p for p in search_root.rglob("*.npy") if p.is_file()])


def parse_band_indices(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


class InfrastructureImageDataset(Dataset):
    """
    Loader for infrastructure crop datasets stored as .npy arrays.

    Works with samples like:
      - osm_way_<id>_sentinel2_gee.npy
      - osm_node_<id>_sentinel2_gee.npy
      - inferred_solar_cluster_<n>_sentinel2_gee.npy

    Assumptions:
      - arrays are either HWC or CHW
      - arrays contain at least 3 channels after band selection
      - default band selection is 0,1,2 unless overridden
    """

    def __init__(
        self,
        dataset_root: str | Path,
        transform=None,
        metadata_file: Optional[str | Path] = None,
        image_column: str = "image_path",
        max_images: Optional[int] = None,
        band_indices: str = "0,1,2",
        min_valid_size: int = 16,
    ):
        self.dataset_root = Path(dataset_root)
        self.transform = transform
        self.image_column = image_column
        self.band_indices = parse_band_indices(band_indices)
        self.min_valid_size = min_valid_size
        self.samples = self._build_samples(metadata_file=metadata_file, image_column=image_column)
        self.samples = self._filter_valid_samples(self.samples)
        if max_images is not None:
            self.samples = self.samples[:max_images]
        if not self.samples:
            raise RuntimeError("No valid .npy imagery samples were found.")

    def _build_samples(self, metadata_file: Optional[str | Path], image_column: str) -> list[Path]:
        if metadata_file:
            metadata_path = resolve_path(self.dataset_root, metadata_file)
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            if metadata_path.suffix.lower() == ".csv":
                df = pd.read_csv(metadata_path)
            elif metadata_path.suffix.lower() in {".parquet", ".pq"}:
                df = pd.read_parquet(metadata_path)
            elif metadata_path.suffix.lower() == ".json":
                data = json.loads(metadata_path.read_text())
                df = pd.DataFrame(data)
            else:
                raise ValueError("metadata_file must be CSV, Parquet, or JSON")

            if image_column not in df.columns:
                raise ValueError(
                    f"Column '{image_column}' not found in metadata. Available columns: {list(df.columns)}"
                )
            samples = []
            for raw_path in df[image_column].dropna().astype(str).tolist():
                p = resolve_path(self.dataset_root, raw_path)
                if p.exists() and p.suffix.lower() in VALID_SUFFIXES:
                    samples.append(p)
            return samples

        return discover_npy_images(self.dataset_root)

    def _filter_valid_samples(self, samples: list[Path]) -> list[Path]:
        valid = []
        for p in samples:
            try:
                arr = np.load(p, mmap_mode="r")
                h, w, c = self._infer_hwc(arr.shape)
                if h >= self.min_valid_size and w >= self.min_valid_size and c >= max(self.band_indices) + 1:
                    valid.append(p)
            except Exception:
                continue
        return valid

    @staticmethod
    def _infer_hwc(shape: tuple[int, ...]) -> tuple[int, int, int]:
        if len(shape) == 2:
            h, w = shape
            return h, w, 1
        if len(shape) != 3:
            raise ValueError(f"Unsupported array shape: {shape}")
        a, b, c = shape
        if c <= 16 and a > 16 and b > 16:
            return a, b, c
        if a <= 16 and b > 16 and c > 16:
            return b, c, a
        return a, b, c

    @staticmethod
    def _to_hwc(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            arr = arr[..., None]
        elif arr.ndim != 3:
            raise ValueError(f"Unsupported array shape: {arr.shape}")
        if arr.shape[-1] <= 16 and arr.shape[0] > 16 and arr.shape[1] > 16:
            return arr
        if arr.shape[0] <= 16 and arr.shape[1] > 16 and arr.shape[2] > 16:
            return np.transpose(arr, (1, 2, 0))
        return arr

    @staticmethod
    def _normalize_to_unit(arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(np.float32)
        finite = np.isfinite(arr)
        if not finite.any():
            return np.zeros_like(arr, dtype=np.float32)
        arr = np.where(finite, arr, 0.0)
        lo = np.percentile(arr, 2)
        hi = np.percentile(arr, 98)
        if hi <= lo:
            lo = float(arr.min())
            hi = float(arr.max())
        if hi <= lo:
            return np.zeros_like(arr, dtype=np.float32)
        arr = np.clip(arr, lo, hi)
        arr = (arr - lo) / (hi - lo)
        return arr.astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path = self.samples[idx]
        arr = np.load(image_path)
        arr = self._to_hwc(arr)

        if arr.shape[-1] <= max(self.band_indices):
            raise RuntimeError(
                f"Band indices {self.band_indices} incompatible with sample {image_path} shape {arr.shape}"
            )

        arr = arr[..., self.band_indices]
        arr = self._normalize_to_unit(arr)
        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

        if self.transform is not None:
            x = self.transform(x)

        return {
            "image": x,
            "path": str(image_path),
            "filename": image_path.name,
        }
