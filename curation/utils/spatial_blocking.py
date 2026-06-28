"""
spatial_blocking.py
-------------------
Spatial-block-aware splitting for the Infra-Bench v1 benchmark.

Random stratified splits leak nearby tiles between train/val/test (Meyer &
Pebesma 2022; PANGAEA 2024; GEO-Bench-2 2025). This module projects tile
centroids to Equal Earth (EPSG:8857), partitions the world into 200 km
blocks anchored at (0, 0) in EE coordinates, and assigns each region's
blocks (not tiles) to train/val/test such that the resulting tile counts
follow ~70/15/15 within that region.

Same-block tiles always land in the same split. Block assignment uses a
fixed seed (default 42) and is INVARIANT across all FMs and all training
seeds — only the linear-probe head's init/data-shuffle vary with the
training seed.

The 4 functions in this module are the full public API:
  - compute_spatial_blocks
  - assign_blocks_to_splits
  - save_split_artifact
  - load_split_artifact
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


# Equal Earth equal-area projection (Šavrič, Patterson, Jenny 2018).
# EPSG:8857 — same authority code that QGIS/PROJ ship with.
DEFAULT_PROJECTION = 'EPSG:8857'
DEFAULT_BLOCK_SIZE_KM = 200


# --------------------------------------------------------------------------
# 1. compute_spatial_blocks
# --------------------------------------------------------------------------

def compute_spatial_blocks(
    tiles_df: pd.DataFrame,
    block_size_km: int = DEFAULT_BLOCK_SIZE_KM,
    projection: str = DEFAULT_PROJECTION,
) -> pd.DataFrame:
    """Project tile centroids to Equal Earth and assign block ids.

    Parameters
    ----------
    tiles_df : DataFrame with at least 'lat' and 'lon' columns (WGS84).
        Other columns (asset_id, region, sector, asset_type) are preserved.
    block_size_km : block edge length in kilometres. Default 200.
    projection : pyproj-compatible target CRS. Default 'EPSG:8857'
        (Equal Earth).

    Returns
    -------
    DataFrame
        Original columns plus:
          ee_x, ee_y       : Equal Earth coordinates in metres
          block_id_x       : floor(ee_x / block_size_m)  (int, can be negative)
          block_id_y       : floor(ee_y / block_size_m)  (int, can be negative)
          block_id         : 'bx_<id_x>_by_<id_y>'      string key for grouping

    Notes
    -----
    Blocks are corner-anchored at EE (0, 0) — block_id (0, 0) is the
    200 km square whose lower-left corner is at the projection origin.
    A tile whose Equal Earth x is in [-200000, 0) lands in block_id_x = -1.
    """
    from pyproj import Transformer

    if 'lat' not in tiles_df.columns or 'lon' not in tiles_df.columns:
        raise ValueError("tiles_df must have 'lat' and 'lon' columns")

    block_size_m = block_size_km * 1000

    # EPSG:4326 (WGS84 lat/lon) -> target projection.
    # always_xy=True so we pass (lon, lat) — pyproj's preferred order.
    transformer = Transformer.from_crs('EPSG:4326', projection, always_xy=True)
    lons = tiles_df['lon'].to_numpy(dtype=np.float64)
    lats = tiles_df['lat'].to_numpy(dtype=np.float64)
    ee_x, ee_y = transformer.transform(lons, lats)

    block_id_x = np.floor(ee_x / block_size_m).astype(np.int64)
    block_id_y = np.floor(ee_y / block_size_m).astype(np.int64)

    out = tiles_df.copy()
    out['ee_x']       = ee_x
    out['ee_y']       = ee_y
    out['block_id_x'] = block_id_x
    out['block_id_y'] = block_id_y
    out['block_id']   = [f'bx_{x}_by_{y}' for x, y in zip(block_id_x, block_id_y)]
    return out


# --------------------------------------------------------------------------
# 2. assign_blocks_to_splits
# --------------------------------------------------------------------------

def assign_blocks_to_splits(
    tiles_with_blocks: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    stratify_by: str = 'region',
    class_fallback: list | None = None,
) -> pd.DataFrame:
    """Within each stratum, assign blocks to train/val/test such that the
    resulting tile counts approximate the requested fractions.

    Parameters
    ----------
    tiles_with_blocks : output of compute_spatial_blocks (must have 'block_id'
        and the stratify_by column, e.g. 'region').
    train_frac, val_frac, test_frac : target fractions per stratum.
        Should sum to 1.0; small float drift is tolerated.
    seed : RNG seed for the per-stratum block shuffle. Default 42.
    stratify_by : column name to stratify on. Default 'region'.
    class_fallback : optional list of asset_type values (matched against
        the tiles_with_blocks['asset_type'] column) whose tiles should be
        assigned via a per-class random stratified 70/15/15 split INSTEAD
        of the block-level assignment. Used for classes too rare to
        guarantee coverage under spatial blocking (e.g. n<=12 globally —
        a single block can hold all of them and would force one split to
        zero tiles). When set, the column 'asset_type' must exist in
        tiles_with_blocks.

    Returns
    -------
    DataFrame
        Original columns plus:
          split           : 'train' | 'val' | 'test'
          split_protocol  : 'spatial_block' (default) or 'class_fallback'
                            (only for tiles whose asset_type is in
                            class_fallback)

        Block-coherence holds for tiles assigned via 'spatial_block'.
        Tiles in class_fallback classes deliberately VIOLATE block
        coherence — same-block tiles in a fallback class can land in
        different splits, because their per-class split is independent of
        the block-level assignment. This is the design trade-off: we
        accept a small spatial-leakage risk for ~20 rare-class tiles in
        order to guarantee class coverage across all three splits.

    Algorithm
    ---------
    Step 1 (block-level, all classes):
      Per stratum (e.g. per region):
        1. Group all tiles by block_id; compute tile-count per block.
        2. Shuffle the block list deterministically with random.Random(seed).
        3. Compute target tile counts t_train, t_val, t_test from the fractions.
        4. Greedy: walk shuffled blocks in order. For each block, append it
           to the split whose current shortfall (target - current) is largest.
           Ties broken in train -> val -> test order.

    Step 2 (per-class fallback override, optional):
      For each class in class_fallback (in list order):
        1. Collect all tiles with that asset_type.
        2. Shuffle them with random.Random(seed) — same seed each time so
           the assignment for class X depends only on (X, seed), not on
           list order or on previous fallback classes' shuffles.
        3. Assign the first n_tr = floor(n * train_frac) to train, the
           next n_va = floor(n * val_frac) to val, and the remainder to
           test. With n>=4 this guarantees >=1 tile per split for any
           reasonable fraction split.
        4. Overwrite their split column and stamp split_protocol =
           'class_fallback'.

    The greedy block-level strategy keeps realized fractions within
    ~0.5pp of target. The per-class fallback exists ONLY to fix the
    coverage failures that block-level assignment cannot solve for the
    n<=12 tail; never use it for classes that would have working
    coverage under blocks alone.
    """
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-6:
        raise ValueError(
            f'fractions must sum to 1.0: '
            f'{train_frac} + {val_frac} + {test_frac} = '
            f'{train_frac + val_frac + test_frac}'
        )
    if stratify_by not in tiles_with_blocks.columns:
        raise ValueError(f'stratify_by column missing: {stratify_by!r}')
    if 'block_id' not in tiles_with_blocks.columns:
        raise ValueError("tiles_with_blocks must have 'block_id' "
                         '(run compute_spatial_blocks first)')

    rng = random.Random(seed)
    block_split_lookup: Dict[str, str] = {}    # block_id -> 'train'/'val'/'test'

    for stratum_val, stratum_df in tiles_with_blocks.groupby(stratify_by, sort=True):
        # block sizes within this stratum
        block_sizes = (stratum_df.groupby('block_id').size()
                       .sort_index()
                       .to_dict())
        block_ids = list(block_sizes.keys())
        rng.shuffle(block_ids)

        n_tiles_in_stratum = sum(block_sizes.values())
        targets = {
            'train': train_frac * n_tiles_in_stratum,
            'val':   val_frac   * n_tiles_in_stratum,
            'test':  test_frac  * n_tiles_in_stratum,
        }
        running = {'train': 0, 'val': 0, 'test': 0}

        for bid in block_ids:
            # Pick split with the largest shortfall = target - running.
            # Tie-break order: train > val > test (preserves spec semantics
            # if all three are exactly tied at 0).
            order = ['train', 'val', 'test']
            best_split = max(
                order, key=lambda s: (targets[s] - running[s], -order.index(s))
            )
            block_split_lookup[bid] = best_split
            running[best_split] += block_sizes[bid]

    out = tiles_with_blocks.copy()
    out['split'] = out['block_id'].map(block_split_lookup)
    if out['split'].isna().any():
        n_missing = out['split'].isna().sum()
        raise RuntimeError(
            f'{n_missing} tile(s) ended up without a split assignment — '
            f'this is a bug in assign_blocks_to_splits, not in your data.'
        )
    out['split_protocol'] = 'spatial_block'

    # ------------------ Step 2: per-class fallback override -----------------
    if class_fallback:
        if 'asset_type' not in out.columns:
            raise ValueError(
                "class_fallback requires 'asset_type' column in tiles_with_blocks"
            )
        if 'asset_id' not in out.columns:
            raise ValueError(
                "class_fallback requires 'asset_id' column in tiles_with_blocks"
            )
        for cls in class_fallback:
            mask = out['asset_type'] == cls
            n_cls = int(mask.sum())
            if n_cls == 0:
                # Class not present at all — nothing to fall back to. Warn
                # by raising, since this likely indicates a spec / data
                # mismatch the caller should investigate.
                raise ValueError(
                    f'class_fallback class {cls!r} has 0 tiles in the dataset; '
                    f'check asset_type spelling against ASSET_TYPE_MAP'
                )
            cls_ids = out.loc[mask, 'asset_id'].tolist()
            # Per-class RNG: deterministic in (cls, seed), independent of
            # the position of cls in the class_fallback list.
            cls_rng = random.Random(seed)
            shuffled = cls_ids.copy()
            cls_rng.shuffle(shuffled)
            n_tr = int(n_cls * train_frac)
            n_va = int(n_cls * val_frac)
            # Remainder to test so the three buckets sum to n_cls exactly.
            fb_assignment: Dict[str, str] = {}
            for i, aid in enumerate(shuffled):
                if i < n_tr:
                    fb_assignment[aid] = 'train'
                elif i < n_tr + n_va:
                    fb_assignment[aid] = 'val'
                else:
                    fb_assignment[aid] = 'test'
            # Overwrite the split column for these tiles and stamp the protocol.
            new_splits = out.loc[mask, 'asset_id'].map(fb_assignment)
            out.loc[mask, 'split']          = new_splits.values
            out.loc[mask, 'split_protocol'] = 'class_fallback'

    return out


# --------------------------------------------------------------------------
# 3. save_split_artifact
# --------------------------------------------------------------------------

SPLIT_ARTIFACT_COLUMNS = (
    'asset_id', 'region', 'sector', 'asset_type',
    'lat', 'lon', 'block_id_x', 'block_id_y',
    'split', 'split_protocol',
)


def save_split_artifact(tiles_with_splits: pd.DataFrame, output_path: str) -> None:
    """Persist the split assignment as parquet. Only the canonical columns
    are written, in a stable order, so downstream notebooks have a single
    schema to depend on."""
    missing = [c for c in SPLIT_ARTIFACT_COLUMNS if c not in tiles_with_splits.columns]
    if missing:
        raise ValueError(f'tiles_with_splits missing columns: {missing}')
    out = tiles_with_splits[list(SPLIT_ARTIFACT_COLUMNS)].copy()
    # Stable column dtypes for parquet round-trip
    out['asset_id']       = out['asset_id'].astype(str)
    out['split']          = out['split'].astype(str)
    out['split_protocol'] = out['split_protocol'].astype(str)
    out['block_id_x']     = out['block_id_x'].astype('int64')
    out['block_id_y']     = out['block_id_y'].astype('int64')

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)


# --------------------------------------------------------------------------
# 4. load_split_artifact
# --------------------------------------------------------------------------

def load_split_artifact(input_path: str) -> Dict[str, str]:
    """Load the split parquet and return a dict mapping asset_id -> split.

    The dict form is what every Phase 2 FM notebook will consume — it's
    O(1) per tile lookup during dataset construction.
    """
    df = pd.read_parquet(input_path, columns=['asset_id', 'split'])
    return dict(zip(df['asset_id'].astype(str), df['split'].astype(str)))
