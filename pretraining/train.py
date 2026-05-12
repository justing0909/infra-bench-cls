from __future__ import annotations

# Changes from original:
#   - in_channels derived from band_indices and passed to SimCLRModel so the
#     first conv layer is correctly sized for multimodal tiles.
#   - n_optical passed to build_simclr_transform so value augmentations
#     target only optical bands (not SAR or thermal).
#   - num_workers default changed from 4 to 0 — avoids multiprocessing
#     issues on Windows. Set higher on Linux/macOS or with --num-workers.
#   - band_indices and in_channels saved into checkpoint state so downstream
#     classification can verify compatibility when loading.

import argparse
import time

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .augmentations import TwoCropTransform, build_simclr_transform
from .config import TrainConfig
from .datasets import InfrastructureImageDataset, parse_band_indices
from .losses import nt_xent_loss
from downstream.common.models import SimCLRModel
from downstream.common.utils import choose_device, count_trainable_params, ensure_dir, save_checkpoint, set_seed, worker_init_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-supervised pretraining on infrastructure .npy imagery"
    )
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/pretrain_v1")
    parser.add_argument("--metadata-file", type=str, default=None)
    parser.add_argument("--image-column", type=str, default="image_path")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader workers. Default 0 is safest on Windows. Increase on Linux/macOS.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--backbone-name", type=str, default="resnet18",
                        choices=["resnet18", "resnet50"])
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument(
        "--band-indices", type=str, default="0,1,2,3,4,5,6,7,8,9",
        help=(
            "Comma-separated band indices. Default uses all 10 bands "
            "(sentinel2_ms + sentinel1 + landsat_thermal). "
            "Adjust if your tiles have a different modality combination."
        ),
    )
    parser.add_argument(
        "--n-optical", type=int, default=7,
        help=(
            "Number of optical bands for value augmentations. "
            "Default 7 for sentinel2_ms. SAR and thermal bands are excluded "
            "from jitter and noise augmentations."
        ),
    )
    parser.add_argument("--min-valid-size", type=int, default=16)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        projection_dim=args.projection_dim,
        backbone_name=args.backbone_name,
        pretrained_backbone=args.pretrained_backbone,
        mixed_precision=not args.no_mixed_precision,
        seed=args.seed,
        save_every=args.save_every,
        device=args.device,
        metadata_file=args.metadata_file,
        image_column=args.image_column,
        max_images=args.max_images,
        band_indices=args.band_indices,
        min_valid_size=args.min_valid_size,
    )


def train_one_epoch(model, loader, optimizer, scaler, device, temperature, mixed_precision):
    model.train()
    running_loss = 0.0
    n_batches = 0

    for batch in loader:
        x1, x2 = batch["image"]
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        autocast_device = "cuda" if device.type == "cuda" else "cpu"
        with autocast(device_type=autocast_device,
                      enabled=mixed_precision and device.type == "cuda"):
            _, z1 = model(x1)
            _, z2 = model(x2)
            loss = nt_xent_loss(z1, z2, temperature=temperature)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(n_batches, 1)


def main():
    args = parse_args()
    cfg = build_config(args)
    set_seed(cfg.seed)

    output_dir = ensure_dir(cfg.output_dir)
    cfg.save(output_dir / "train_config.json")

    device = choose_device(cfg.device)
    print(f"Using device: {device}")

    band_indices = parse_band_indices(cfg.band_indices)
    in_channels  = len(band_indices)
    n_optical    = min(args.n_optical, in_channels)
    print(f"Band indices: {band_indices} ({in_channels} channels, {n_optical} optical)")

    transform = TwoCropTransform(
        build_simclr_transform(cfg.image_size, n_optical=n_optical)
    )
    dataset = InfrastructureImageDataset(
        dataset_root=cfg.dataset_root,
        transform=transform,
        metadata_file=cfg.metadata_file,
        image_column=cfg.image_column,
        max_images=cfg.max_images,
        band_indices=cfg.band_indices,
        min_valid_size=cfg.min_valid_size,
    )
    print(f"Found {len(dataset):,} valid samples for pretraining")

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        worker_init_fn=worker_init_fn if cfg.num_workers > 0 else None,
    )

    model = SimCLRModel(
        backbone_name=cfg.backbone_name,
        projection_dim=cfg.projection_dim,
        pretrained_backbone=cfg.pretrained_backbone,
        in_channels=in_channels,
    ).to(device)
    print(f"Trainable params: {count_trainable_params(model):,}")

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scaler = GradScaler(
        device="cuda",
        enabled=cfg.mixed_precision and device.type == "cuda",
    )

    best_loss = float("inf")
    start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            temperature=cfg.temperature,
            mixed_precision=cfg.mixed_precision,
        )
        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch:03d}/{cfg.epochs:03d} | loss={train_loss:.4f} | {elapsed:.1f}s")

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": train_loss,
            "config": cfg.__dict__,
            # Saved explicitly so downstream classification can verify
            # the checkpoint was trained with matching band configuration.
            "band_indices": cfg.band_indices,
            "in_channels": in_channels,
        }

        if train_loss < best_loss:
            best_loss = train_loss
            save_checkpoint(output_dir / "best.pt", state)

        if epoch % cfg.save_every == 0 or epoch == cfg.epochs:
            save_checkpoint(output_dir / f"epoch_{epoch:03d}.pt", state)

    total = time.time() - start
    print(f"Training complete in {total / 60:.1f} min. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()