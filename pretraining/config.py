from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from downstream.common.utils import save_json


@dataclass
class TrainConfig:
    dataset_root: str
    output_dir: str
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 25
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    temperature: float = 0.2
    projection_dim: int = 128
    backbone_name: str = "resnet18"
    pretrained_backbone: bool = False
    mixed_precision: bool = True
    seed: int = 42
    save_every: int = 5
    device: Optional[str] = None
    metadata_file: Optional[str] = None
    image_column: str = "image_path"
    max_images: Optional[int] = None
    band_indices: str = "0,1,2"
    min_valid_size: int = 16

    def save(self, path: str | Path) -> None:
        save_json(path, self)
