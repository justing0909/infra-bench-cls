from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T


class MultimodalResize:
    """Resizes an HWC numpy array of any channel count without going through PIL.
    Handles multimodal tiles (10+ bands) that PIL cannot process.
    """
    def __init__(self, size: int):
        self.size = size

    def __call__(self, array: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(array, (self.size, self.size),
                             interpolation=cv2.INTER_LINEAR)
        if resized.ndim == 2:
            resized = resized[..., np.newaxis]
        return torch.from_numpy(np.transpose(resized, (2, 0, 1))).float()


class MultimodalAugment:
    def __init__(self, size: int, min_crop_scale: float = 0.75):
        self.size = size
        self.min_crop_scale = min_crop_scale

    def __call__(self, array: np.ndarray) -> torch.Tensor:
        h, w = array.shape[:2]

        # Random crop — valid for all modalities
        scale = np.random.uniform(self.min_crop_scale, 1.0)
        crop_h = int(h * scale)
        crop_w = int(w * scale)
        top  = np.random.randint(0, max(1, h - crop_h))
        left = np.random.randint(0, max(1, w - crop_w))
        array = array[top:top + crop_h, left:left + crop_w, :]

        # Resize back to target
        resized = cv2.resize(array, (self.size, self.size),
                             interpolation=cv2.INTER_LINEAR)
        if resized.ndim == 2:
            resized = resized[..., np.newaxis]

        # Spatial flips and rotations — valid for all modalities
        if np.random.random() > 0.5:
            resized = np.fliplr(resized)
        if np.random.random() > 0.5:
            resized = np.flipud(resized)
        k = np.random.randint(0, 4)
        if k > 0:
            resized = np.rot90(resized, k)

        # Per-band Gaussian noise on optical bands only (0-6)
        # Simulates sensor noise and atmospheric variation
        if np.random.random() > 0.7:
            noise = np.random.normal(0, 0.01, resized[..., :7].shape).astype(np.float32)
            resized[..., :7] = np.clip(resized[..., :7] + noise, 0, 1)

        return torch.from_numpy(
            np.ascontiguousarray(np.transpose(resized, (2, 0, 1)))
        ).float()


class TwoCropTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return self.base_transform(x), self.base_transform(x)


def build_eval_transform(image_size: int = 224):
    return MultimodalResize(image_size)


def build_simclr_transform(image_size: int = 224):
    return TwoCropTransform(MultimodalAugment(image_size))