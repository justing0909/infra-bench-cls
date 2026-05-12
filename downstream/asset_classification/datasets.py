from __future__ import annotations

# Changes from original:
#   - parse_band_indices removed; now imported from downstream.common.io
#     (single definition, shared with NpyInfrastructureDataset and future modules).
#   - No other logic changes — LabelSpace and AssetClassificationDataset are unchanged.

from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import Dataset

from downstream.common.comm_datasets import NpyInfrastructureDataset


@dataclass
class LabelSpace:
    classes: list[str]

    @property
    def class_to_idx(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.classes)}


def build_label_space(records, min_count: int = 1) -> LabelSpace:
    counts: dict[str, int] = {}
    for r in records:
        if r.asset_type:
            counts[r.asset_type] = counts.get(r.asset_type, 0) + 1
    classes = sorted([k for k, v in counts.items() if v >= min_count])
    return LabelSpace(classes=classes)


class AssetClassificationDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        transform,
        metadata_file: Optional[str] = None,
        band_indices: str = "0,1,2",
        max_images: int | None = None,
        min_valid_size: int = 16,
        label_space: Optional[LabelSpace] = None,
        min_class_count: int = 1,
    ):
        self.base = NpyInfrastructureDataset(
            dataset_root=dataset_root,
            transform=transform,
            metadata_file=metadata_file,
            band_indices=band_indices,
            max_images=max_images,
            min_valid_size=min_valid_size,
            require_labels=True,
        )
        self.label_space = label_space or build_label_space(
            self.base.records, min_count=min_class_count
        )
        self.class_to_idx = self.label_space.class_to_idx
        self.indices = [
            i for i, r in enumerate(self.base.records)
            if r.asset_type in self.class_to_idx
        ]
        if not self.indices:
            raise RuntimeError("No labeled samples remained after label-space filtering.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        base_item = self.base[self.indices[idx]]
        y = self.class_to_idx[base_item["asset_type"]]
        base_item["label"] = torch.tensor(y, dtype=torch.long)
        return base_item