from __future__ import annotations

# Changes from previous version:
#   - Added tail_mean_acc and tail_std_acc — mean and std of val_acc over the
#     last --tail-epochs epochs (default 5). More honest than best_val_acc for
#     noisy training runs — reported at end and saved to metrics.csv.
#   - Added per-class accuracy in final evaluation so you can see whether the
#     model is predicting all classes or collapsing to majority.
#   - metrics.csv now includes tail_mean and tail_std columns.
#   - Added --tail-epochs argument (default 5).

import argparse
import csv
import random
import warnings

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from downstream.common.models import EncoderBackbone, LinearClassifier, SimCLRModel
from downstream.common.transforms import build_eval_transform
from downstream.common.utils import choose_device, ensure_dir, save_checkpoint, save_json, set_seed
from downstream.asset_classification.datasets import AssetClassificationDataset, LabelSpace


def parse_args():
    p = argparse.ArgumentParser(
        description="Asset classification fine-tuning / linear probing"
    )
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--metadata-file", default=None)
    p.add_argument(
        "--checkpoint", default=None,
        help="Optional pretrained checkpoint from pretraining.train"
    )
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument(
        "--band-indices", default="0,1,2",
        help=(
            "Comma-separated band indices to load. Default is '0,1,2' (RGB). "
            "For 7-band sentinel2_ms tiles pass '0,1,2,3,4,5,6'. "
            "For sentinel2_ms + sentinel1 (9 bands) pass '0,1,2,3,4,5,6,7,8'. "
            "For sentinel2_ms + sentinel1 + landsat_thermal (10 bands) pass '0,1,2,3,4,5,6,7,8,9'."
        ),
    )
    p.add_argument(
        "--backbone-name", default="resnet18",
        choices=["resnet18", "resnet50"]
    )
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument(
        "--weighted-loss", action="store_true",
        help=(
            "Use inverse-frequency class weights in CrossEntropyLoss. "
            "Recommended when classes are imbalanced (e.g. untyped substations dominate)."
        ),
    )
    p.add_argument("--min-class-count", type=int, default=1)
    p.add_argument("--train-fraction", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument(
        "--tail-epochs", type=int, default=5,
        help=(
            "Number of final epochs to average for tail_mean_acc. "
            "This is the honest working metric — less sensitive to lucky peaks. "
            "Default 5."
        ),
    )
    return p.parse_args()


def build_model(args, num_classes: int):
    band_indices = [int(x) for x in args.band_indices.split(",")]
    in_channels = len(band_indices)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
        model = SimCLRModel(
            backbone_name=config.get("backbone_name", args.backbone_name),
            pretrained_backbone=False,
            projection_dim=config.get("projection_dim", 128),
            in_channels=in_channels,
        )
        state_dict = (
            ckpt.get("model_state")
            or ckpt.get("model_state_dict")
            or ckpt.get("state_dict")
            or ckpt
        )
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Warning: {len(missing)} missing keys in checkpoint")
        if unexpected:
            print(f"  Warning: {len(unexpected)} unexpected keys in checkpoint")
        encoder = model.backbone
        encoder.feature_dim = model.feature_dim
        print(f"Loaded encoder from checkpoint: {args.checkpoint}")
        print(f"  Matched {len(state_dict) - len(missing)} / {len(state_dict)} keys")
    else:
        encoder = EncoderBackbone(
            args.backbone_name, pretrained=False, in_channels=in_channels
        )
        print(
            f"No checkpoint provided — encoder initialised randomly ({args.backbone_name}). "
            "Pass --checkpoint to load a pretrained encoder."
        )

    head = LinearClassifier(encoder.feature_dim, num_classes)
    return encoder, head


def compute_class_weights(dataset: AssetClassificationDataset,
                           train_indices: list[int],
                           device: torch.device) -> torch.Tensor:
    """
    Computes inverse-frequency weights for each class from the training subset.
    Weight for class c = total_train_samples / (n_classes * count_of_class_c).
    Normalised so weights average to 1.0, keeping the loss scale stable.
    Computed from training indices only — no val leakage.
    """
    counts = [0] * len(dataset.label_space.classes)
    for idx in train_indices:
        item = dataset[idx]
        counts[item["label"].item()] += 1

    n_classes = len(counts)
    total = sum(counts)
    weights = [
        total / (n_classes * c) if c > 0 else 0.0
        for c in counts
    ]
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    print("Class weights (inverse frequency):")
    for cls, w, c in zip(dataset.label_space.classes, weights, counts):
        print(f"  {cls}: count={c}, weight={w:.3f}")

    return weight_tensor


def split_indices(n: int, train_fraction: float, seed: int):
    idxs = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(idxs)
    cut = max(1, int(n * train_fraction))
    train_idxs = idxs[:cut]
    val_idxs = idxs[cut:]
    if not val_idxs:
        warnings.warn(
            f"Validation set is empty (n={n}, train_fraction={train_fraction}). "
            "Val accuracy will be reported as 0. Consider lowering --train-fraction.",
            stacklevel=2,
        )
    return train_idxs, val_idxs


def evaluate(encoder, head, loader, device, classes: list[str] | None = None):
    """
    Evaluates accuracy overall and optionally per class.
    Returns (overall_acc, per_class_acc_dict).
    per_class_acc_dict is None if classes not provided.
    """
    if not loader.dataset:
        return 0.0, None
    encoder.eval()
    head.eval()

    correct, total = 0, 0
    if classes:
        class_correct = [0] * len(classes)
        class_total   = [0] * len(classes)
    else:
        class_correct = class_total = None

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            logits = head(encoder(x))
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total   += y.numel()
            if classes:
                for cls_idx in range(len(classes)):
                    mask = (y == cls_idx)
                    class_correct[cls_idx] += (pred[mask] == y[mask]).sum().item()
                    class_total[cls_idx]   += mask.sum().item()

    overall = correct / max(total, 1)

    if classes and class_total:
        per_class = {
            cls: round(class_correct[i] / max(class_total[i], 1), 4)
            for i, cls in enumerate(classes)
        }
    else:
        per_class = None

    return overall, per_class


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    save_json(output_dir / "run_config.json", vars(args))

    dataset = AssetClassificationDataset(
        dataset_root=args.dataset_root,
        transform=build_eval_transform(args.image_size),
        metadata_file=args.metadata_file,
        band_indices=args.band_indices,
        max_images=args.max_images,
        min_class_count=args.min_class_count,
    )
    label_space = dataset.label_space
    classes = label_space.classes
    save_json(output_dir / "label_space.json", {"classes": classes})
    print(f"Classification samples: {len(dataset)}")
    print(f"Classes ({len(classes)}): {classes}")

    train_idxs, val_idxs = split_indices(len(dataset), args.train_fraction, args.seed)
    train_ds = Subset(dataset, train_idxs)
    val_ds   = Subset(dataset, val_idxs)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    device = choose_device(args.device)
    encoder, head = build_model(args, len(classes))
    encoder.to(device)
    head.to(device)

    if args.freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False
        print("Encoder frozen — training linear head only.")

    params = list(head.parameters()) + [
        p for p in encoder.parameters() if p.requires_grad
    ]
    optimizer = AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.weighted_loss:
        class_weights = compute_class_weights(dataset, train_idxs, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using weighted CrossEntropyLoss.")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using unweighted CrossEntropyLoss.")

    best_acc   = -1.0
    val_history: list[float] = []

    history_path = output_dir / "metrics.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "train_loss", "val_acc"]
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            encoder.train()
            head.train()
            losses = []
            for batch in train_loader:
                x = batch["image"].to(device)
                y = batch["label"].to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = head(encoder(x))
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            val_acc, _ = evaluate(encoder, head, val_loader, device)
            val_history.append(val_acc)
            train_loss = sum(losses) / max(len(losses), 1)

            writer.writerow({
                "epoch":      epoch,
                "train_loss": round(train_loss, 6),
                "val_acc":    round(val_acc, 6),
            })
            f.flush()

            print(
                f"Epoch {epoch:03d}/{args.epochs:03d} | "
                f"train_loss={train_loss:.4f} | val_acc={val_acc:.4f}"
            )

            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(output_dir / "checkpoint_best.pt", {
                    "encoder_state": encoder.state_dict(),
                    "head_state":    head.state_dict(),
                    "classes":       classes,
                })

    # --- Tail metrics ---
    tail_n    = min(args.tail_epochs, len(val_history))
    tail_vals = val_history[-tail_n:]
    tail_mean = sum(tail_vals) / len(tail_vals)
    tail_std  = (
        (sum((v - tail_mean) ** 2 for v in tail_vals) / len(tail_vals)) ** 0.5
    )

    # --- Final per-class accuracy ---
    _, per_class = evaluate(encoder, head, val_loader, device, classes=classes)

    # --- Print summary ---
    print(f"\n{'=' * 50}")
    print(f"RESULTS SUMMARY")
    print(f"{'=' * 50}")
    print(f"Best val_acc:                    {best_acc:.4f}")
    print(f"Tail mean val_acc (last {tail_n} ep): {tail_mean:.4f}")
    print(f"Tail std  val_acc (last {tail_n} ep): {tail_std:.4f}")
    print(f"\nPer-class accuracy (final epoch):")
    if per_class:
        for cls, acc in per_class.items():
            short = cls.split(".")[-1]
            print(f"  {short:35s} {acc:.4f}")

    # --- Save final summary ---
    save_json(output_dir / "results_summary.json", {
        "best_val_acc":  round(best_acc, 6),
        "tail_mean_acc": round(tail_mean, 6),
        "tail_std_acc":  round(tail_std, 6),
        "tail_epochs":   tail_n,
        "per_class_acc": per_class,
        "classes":       classes,
        "total_epochs":  args.epochs,
    })
    print(f"\nSaved results to {output_dir / 'results_summary.json'}")


if __name__ == "__main__":
    main()