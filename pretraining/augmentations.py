from __future__ import annotations

# Changes from original:
#   - Removed hard assertion that x.shape[0] == 3. Pretraining operates on
#     all bands (e.g. 10 for sentinel2_ms + sentinel1 + landsat_thermal).
#   - _maybe_grayscale removed — grayscale is meaningless for SAR/thermal bands
#     and would corrupt non-optical channels. Replaced with per-band noise on
#     optical bands only (indices 0-6 for sentinel2_ms).
#   - Color jitter now applied to optical bands only (0:n_optical), leaving
#     SAR and thermal untouched.
#   - n_optical parameter added (default 7 for sentinel2_ms) to control which
#     bands receive value-based augmentations.

import torch
import torch.nn.functional as F


class TwoCropTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x: torch.Tensor):
        return self.base_transform(x), self.base_transform(x)


class TensorSimCLRTransform:
    """
    SimCLR augmentation pipeline for multimodal satellite tiles.

    Spatial transforms (crop, flip) are applied to all bands.
    Value transforms (jitter, noise) are applied to optical bands only.

    Parameters
    ----------
    image_size    : int   — output spatial size
    jitter_strength : float — brightness/contrast jitter magnitude
    hflip_p       : float — horizontal flip probability
    vflip_p       : float — vertical flip probability
    noise_p       : float — probability of adding Gaussian noise to optical bands
    noise_std     : float — std of Gaussian noise (on 0-1 normalised values)
    n_optical     : int   — number of optical bands (default 7 for sentinel2_ms)
                    value augmentations are applied to bands [:n_optical] only
    """

    def __init__(
        self,
        image_size: int = 224,
        jitter_strength: float = 0.2,
        hflip_p: float = 0.5,
        vflip_p: float = 0.2,
        noise_p: float = 0.3,
        noise_std: float = 0.01,
        n_optical: int = 7,
    ):
        self.image_size = image_size
        self.jitter_strength = jitter_strength
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.noise_p = noise_p
        self.noise_std = noise_std
        self.n_optical = n_optical

    def _random_resized_crop(self, x: torch.Tensor, scale=(0.5, 1.0)) -> torch.Tensor:
        _, h, w = x.shape
        min_side = min(h, w)
        crop_scale = torch.empty(1).uniform_(scale[0], scale[1]).item()
        crop_size = max(8, int(min_side * crop_scale))
        crop_h = min(crop_size, h)
        crop_w = min(crop_size, w)
        top  = 0 if h == crop_h else int(torch.randint(0, h - crop_h + 1, (1,)).item())
        left = 0 if w == crop_w else int(torch.randint(0, w - crop_w + 1, (1,)).item())
        x = x[:, top:top + crop_h, left:left + crop_w]
        x = F.interpolate(
            x.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return x.squeeze(0)

    def _maybe_jitter_optical(self, x: torch.Tensor) -> torch.Tensor:
        """Applies brightness/contrast jitter to optical bands only."""
        s = self.jitter_strength
        n = min(self.n_optical, x.shape[0])
        optical = x[:n]
        brightness = float(torch.empty(1).uniform_(1 - s, 1 + s).item())
        contrast   = float(torch.empty(1).uniform_(1 - s, 1 + s).item())
        optical = optical * brightness
        mean    = optical.mean(dim=(1, 2), keepdim=True)
        optical = (optical - mean) * contrast + mean
        x = torch.cat([optical.clamp(0.0, 1.0), x[n:]], dim=0)
        return x

    def _maybe_noise_optical(self, x: torch.Tensor) -> torch.Tensor:
        """Adds small Gaussian noise to optical bands to simulate sensor variation."""
        if torch.rand(1).item() < self.noise_p:
            n = min(self.n_optical, x.shape[0])
            noise = torch.randn_like(x[:n]) * self.noise_std
            x = torch.cat([(x[:n] + noise).clamp(0.0, 1.0), x[n:]], dim=0)
        return x

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected tensor with shape [C,H,W], got {tuple(x.shape)}")

        # Spatial transforms — valid for all modalities
        x = self._random_resized_crop(x)

        if torch.rand(1).item() < self.hflip_p:
            x = torch.flip(x, dims=[2])
        if torch.rand(1).item() < self.vflip_p:
            x = torch.flip(x, dims=[1])

        # Value transforms — optical bands only
        x = self._maybe_jitter_optical(x)
        x = self._maybe_noise_optical(x)

        return x.clamp(0.0, 1.0)


def build_simclr_transform(image_size: int, n_optical: int = 7) -> TensorSimCLRTransform:
    return TensorSimCLRTransform(image_size=image_size, n_optical=n_optical)