"""
visualize_tiles.py
------------------
Visual inspection tool for curated infrastructure imagery tiles.

Shows a grid of tiles from a curated dataset, with:
  - RGB composite (S2 bands B04/B03/B02)
  - NIR channel (B08) if available
  - SAR VV channel if available
  - SAR VH channel if available
  - Thermal (Landsat TIR) if available — rendered with inferno colormap
  - Asset type and ID labels

Band indices are computed dynamically from the manifest modalities,
so the visualizer works correctly regardless of which modality combination
was used when fetching.

Usage:
    # Show random sample of tiles
    python visualize_tiles.py --dataset-root ../data/curated_datasets/dataset_maine_stac_v1

    # Show specific asset type only
    python visualize_tiles.py --dataset-root ../data/curated_datasets/dataset_maine_stac_v1 --asset-type energy.transmission.substation

    # Show more tiles per page
    python visualize_tiles.py --dataset-root ../data/curated_datasets/dataset_maine_stac_v1 --n 24

    # Save output instead of displaying
    python visualize_tiles.py --dataset-root ../data/curated_datasets/dataset_maine_stac_v1 --save tiles_preview.png

    # Print dataset stats only
    python visualize_tiles.py --dataset-root ../data/curated_datasets/dataset_maine_stac_v1 --stats
"""

import os
import sys
import json
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add curation root to path so MODALITY_REGISTRY is importable
sys.path.insert(0, str(Path(__file__).parent))
try:
    from helpers.tile_types import MODALITY_REGISTRY
except ImportError:
    # Fallback hardcoded registry if import fails
    MODALITY_REGISTRY = {
        "sentinel2_ms":    {"n_bands": 7,  "dtype": "uint8"},
        "sentinel2_rgb":   {"n_bands": 3,  "dtype": "uint8"},
        "sentinel1":       {"n_bands": 2,  "dtype": "float32"},
        "landsat_thermal": {"n_bands": 1,  "dtype": "float32"},
        "naip":            {"n_bands": 4,  "dtype": "uint8"},
    }


# ---------------------------------------------------------------------------
# Band index helpers
# ---------------------------------------------------------------------------

S2_RGB_INDICES = (0, 1, 2)   # B04=R, B03=G, B02=B within sentinel2_ms
S2_NIR_INDEX   = 3            # B08 within sentinel2_ms


def _compute_band_indices(modalities: list) -> dict:
    """
    Computes band indices for named channels based on active modalities.
    Returns dict with keys: nir, sar_vv, sar_vh, thermal (where available).

    Example for ["sentinel2_ms", "sentinel1", "landsat_thermal"]:
      sentinel2_ms = bands 0-6
      sentinel1    = bands 7-8  -> sar_vv=7, sar_vh=8
      landsat_thermal = band 9  -> thermal=9
    """
    idx    = {}
    offset = 0

    for m in modalities:
        info = MODALITY_REGISTRY.get(m)
        if info is None:
            continue
        n = info["n_bands"]

        if m in ("sentinel2_ms", "sentinel2_rgb"):
            idx["nir"] = offset + 3  # B08 is always 4th band in S2 stacks
        elif m == "sentinel1":
            idx["sar_vv"] = offset
            idx["sar_vh"] = offset + 1
        elif m == "landsat_thermal":
            idx["thermal"] = offset

        offset += n

    return idx


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _to_rgb(arr: np.ndarray, band_indices=(0, 1, 2)) -> np.ndarray:
    """
    Extracts RGB bands from (C, H, W) and returns (H, W, 3) uint8.
    Handles both uint8 and float32 input. Applies 2-98 percentile stretch.
    """
    r = arr[band_indices[0]].astype(np.float32)
    g = arr[band_indices[1]].astype(np.float32)
    b = arr[band_indices[2]].astype(np.float32)

    rgb = np.stack([r, g, b], axis=-1)

    if rgb.max() > 1.0:
        rgb = rgb / 255.0

    for c in range(3):
        p2, p98 = np.percentile(rgb[:, :, c], (2, 98))
        if p98 > p2:
            rgb[:, :, c] = np.clip((rgb[:, :, c] - p2) / (p98 - p2), 0, 1)

    return (rgb * 255).astype(np.uint8)


def _to_grayscale(arr: np.ndarray, band_index: int) -> np.ndarray:
    """
    Extracts a single band and returns (H, W) float32 in [0, 1].
    Applies 2-98 percentile stretch for visibility.
    """
    band = arr[band_index].astype(np.float32)
    p2, p98 = np.percentile(band, (2, 98))
    if p98 > p2:
        band = np.clip((band - p2) / (p98 - p2), 0, 1)
    else:
        band = np.zeros_like(band)
    return band


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_manifest(dataset_root: str) -> dict:
    path = os.path.join(dataset_root, "manifest.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No manifest at {path}. Run pipeline first.")
    with open(path) as f:
        return json.load(f)


def load_tile(dataset_root: str, record: dict) -> np.ndarray:
    path = os.path.join(dataset_root, "images", record["image_file"])
    return np.load(path)


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def _short_id(asset_id: str, max_len: int = 18) -> str:
    parts = asset_id.split("_")
    if len(parts) >= 3:
        return f"osm_{parts[2][:10]}"
    return asset_id[:max_len]


def _short_type(asset_type: str) -> str:
    mapping = {
        "energy.transmission.substation":         "TX substation",
        "energy.distribution.substation":         "DX substation",
        "energy.distribution.substation_untyped": "substation",
    }
    return mapping.get(asset_type, asset_type.split(".")[-1])


# ---------------------------------------------------------------------------
# Main visualizer
# ---------------------------------------------------------------------------

def visualize_tiles(
    dataset_root : str,
    n            : int  = 16,
    asset_type   : str  = None,
    save_path    : str  = None,
    seed         : int  = 42,
) -> None:
    """
    Renders a grid of tiles with all available modality panels.
    Band indices are computed dynamically from manifest modalities.
    """
    manifest   = load_manifest(dataset_root)
    records    = manifest["records"]
    modalities = manifest.get("modalities", ["sentinel2_ms"])

    print(f"Dataset: {dataset_root}")
    print(f"Total tiles: {len(records)}")
    print(f"Asset types: {', '.join(manifest.get('asset_types', []))}")
    print(f"Modalities:  {', '.join(modalities)}")

    if asset_type:
        records = [r for r in records if r["asset_type"] == asset_type]
        print(f"Filtered to {len(records)} tiles of type '{asset_type}'")

    if not records:
        print("No tiles found matching filters.")
        return

    random.seed(seed)
    sample = random.sample(records, min(n, len(records)))

    # Compute band indices from modalities
    band_idx  = _compute_band_indices(modalities)
    has_nir   = "nir"     in band_idx
    has_sar   = "sar_vv"  in band_idx
    has_therm = "thermal" in band_idx

    # Build panel list
    panels = [("RGB", "rgb", S2_RGB_INDICES)]
    if has_nir:
        panels.append(("NIR",    "gray",    band_idx["nir"]))
    if has_sar:
        panels.append(("SAR VV", "sar",     band_idx["sar_vv"]))
        panels.append(("SAR VH", "sar",     band_idx["sar_vh"]))
    if has_therm:
        panels.append(("TIR",    "thermal", band_idx["thermal"]))

    n_panels = len(panels)

    first_tile = load_tile(dataset_root, sample[0])
    n_bands    = first_tile.shape[0]

    print(f"\nShowing {len(sample)} tiles | bands={n_bands} | "
          f"panels={n_panels} | "
          f"NIR={'yes' if has_nir else 'no'} | "
          f"SAR={'yes' if has_sar else 'no'} | "
          f"TIR={'yes' if has_therm else 'no'}")

    # Grid layout
    cols      = 4
    tile_rows = (len(sample) + cols - 1) // cols
    fig_w     = cols * (n_panels * 1.6 + 0.3)
    fig_h     = tile_rows * 2.4

    modality_label = "+".join(modalities)

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#1a1a2e")
    fig.suptitle(
        f"Infrastructure Tile Preview — {Path(dataset_root).name}\n"
        f"{len(sample)} tiles | {n_bands} bands | {modality_label}",
        color="white", fontsize=11, y=0.98,
    )

    for tile_idx, record in enumerate(sample):
        try:
            arr = load_tile(dataset_root, record)
        except Exception as e:
            print(f"  Warning: could not load {record['image_file']}: {e}")
            continue

        row = tile_idx // cols
        col = tile_idx % cols

        left   = col / cols
        bottom = 1.0 - (row + 1) / tile_rows
        width  = 1.0 / cols
        height = 1.0 / tile_rows

        margin  = 0.005
        inner_w = (width - 2 * margin) / n_panels
        inner_h = height - 0.06

        label_bottom = bottom + inner_h
        label_left   = left + margin

        # Title label
        ax_title = fig.add_axes(
            [label_left, label_bottom, width - 2 * margin, 0.04]
        )
        ax_title.axis("off")
        ax_title.text(
            0.5, 0.5,
            f"{_short_type(record['asset_type'])} | {_short_id(record['asset_id'])}",
            color="#a8d8ea", fontsize=6.5, ha="center", va="center",
            transform=ax_title.transAxes,
        )

        for p_idx, (panel_label, panel_type, p_band) in enumerate(panels):
            ax_left   = left + margin + p_idx * inner_w
            ax_bottom = bottom + 0.01
            ax        = fig.add_axes(
                [ax_left, ax_bottom, inner_w - margin, inner_h - 0.02]
            )
            ax.axis("off")

            try:
                if panel_type == "rgb":
                    img = _to_rgb(arr, p_band)
                    ax.imshow(img)
                elif panel_type == "gray":
                    img = _to_grayscale(arr, p_band)
                    ax.imshow(img, cmap="YlOrBr", vmin=0, vmax=1)
                elif panel_type == "sar":
                    img = _to_grayscale(arr, p_band)
                    ax.imshow(img, cmap="Blues_r", vmin=0, vmax=1)
                elif panel_type == "thermal":
                    img = _to_grayscale(arr, p_band)
                    ax.imshow(img, cmap="inferno", vmin=0, vmax=1)

                ax.set_title(panel_label, color="#888", fontsize=5.5, pad=1)
            except Exception:
                ax.text(0.5, 0.5, "?", color="gray", ha="center", va="center",
                        transform=ax.transAxes)

    plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0, wspace=0, hspace=0)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor="#1a1a2e")
        print(f"\nSaved to {save_path}")
    else:
        plt.show()

    plt.close()


# ---------------------------------------------------------------------------
# Stats summary
# ---------------------------------------------------------------------------

def print_stats(dataset_root: str) -> None:
    manifest = load_manifest(dataset_root)
    records  = manifest["records"]

    print(f"\n{'='*50}")
    print(f"Dataset: {Path(dataset_root).name}")
    print(f"{'='*50}")
    print(f"Total tiles:  {manifest['n_tiles']}")
    print(f"Created:      {manifest.get('created_at', 'unknown')}")
    print(f"Modalities:   {', '.join(manifest.get('modalities', []))}")
    print(f"Has temporal: {manifest.get('has_temporal', False)}")

    print(f"\nBy asset type:")
    type_counts = {}
    for r in records:
        type_counts[r["asset_type"]] = type_counts.get(r["asset_type"], 0) + 1
    for k, v in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    print(f"\nBy confidence:")
    conf_counts = {}
    for r in records:
        conf_counts[r["confidence"]] = conf_counts.get(r["confidence"], 0) + 1
    for k, v in sorted(conf_counts.items()):
        print(f"  {k}: {v}")

    shapes = [r.get("image_shape", []) for r in records if r.get("image_shape")]
    if shapes:
        unique_shapes = set(tuple(s) for s in shapes)
        print(f"\nTile shapes: {unique_shapes}")

    lats = [r["lat"] for r in records]
    lons = [r["lon"] for r in records]
    print(f"\nGeographic extent:")
    print(f"  Lat: {min(lats):.3f} — {max(lats):.3f}")
    print(f"  Lon: {min(lons):.3f} — {max(lons):.3f}")

    print(f"\nModality counts:")
    mod_counts = manifest.get("modality_counts", {})
    for k, v in sorted(mod_counts.items()):
        print(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize curated infrastructure imagery tiles"
    )
    parser.add_argument(
        "--dataset-root", required=True,
        help="Path to dataset directory (contains manifest.json and images/)",
    )
    parser.add_argument(
        "--n", type=int, default=16,
        help="Number of tiles to show (default: 16)",
    )
    parser.add_argument(
        "--asset-type",
        help="Filter to a specific asset type",
    )
    parser.add_argument(
        "--save",
        help="Save output to this path instead of displaying",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for tile sampling (default: 42)",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print dataset statistics and exit",
    )
    args = parser.parse_args()

    if args.stats:
        print_stats(args.dataset_root)
    else:
        visualize_tiles(
            dataset_root = args.dataset_root,
            n            = args.n,
            asset_type   = args.asset_type,
            save_path    = args.save,
            seed         = args.seed,
        )