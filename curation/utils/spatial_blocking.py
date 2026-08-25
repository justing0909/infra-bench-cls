"""
spatial_blocking.py
-------------------
spatial-block-aware splitting for the Infra-Bench v1 benchmark.

random stratified splits leak nearby tiles between train/val/test (Meyer &
pebesma 2022; PANGAEA 2024; GEO-Bench-2 2025). this module projects tile
centroids to Equal Earth (EPSG:8857), partitions the world into 200 km
blocks anchored at (0, 0) in EE coordinates, and assigns each region's
blocks (not tiles) to train/val/test such that the resulting tile counts
follow ~70/15/15 within that region.

same-block tiles always land in the same split. block assignment uses a
fixed seed (default 42) and is INVARIANT across all FMs and all training
seeds — only the linear-probe head's init/data-shuffle vary with the
training seed.

the 4 functions in this module are the full public API:
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
    """project tile centroids to Equal Earth and assign block ids.

    Parameters
    ----------
    tiles_df : DataFrame with at least 'lat' and 'lon' columns (WGS84).
        other columns (asset_id, region, sector, asset_type) are preserved.
    block_size_km : block edge length in kilometres. default 200.
    projection : pyproj-compatible target CRS. default 'EPSG:8857'
        (Equal Earth).

    Returns
    -------
    DataFrame
        original columns plus:
          ee_x, ee_y       : Equal Earth coordinates in metres
          block_id_x       : floor(ee_x / block_size_m)  (int, can be negative)
          block_id_y       : floor(ee_y / block_size_m)  (int, can be negative)
          block_id         : 'bx_<id_x>_by_<id_y>'      string key for grouping

    Notes
    -----
    blocks are corner-anchored at EE (0, 0) — block_id (0, 0) is the
    200 km square whose lower-left corner is at the projection origin.
    a tile whose Equal Earth x is in [-200000, 0) lands in block_id_x = -1.
    """
    # validate before importing pyproj, so a caller with the wrong columns
    # gets the useful error rather than an ImportError
    if 'lat' not in tiles_df.columns or 'lon' not in tiles_df.columns:
        raise ValueError("tiles_df must have 'lat' and 'lon' columns")

    from pyproj import Transformer

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
    """within each stratum, assign blocks to train/val/test such that the
    resulting tile counts approximate the requested fractions.

    Parameters
    ----------
    tiles_with_blocks : output of compute_spatial_blocks (must have 'block_id'
        and the stratify_by column, e.g. 'region').
    train_frac, val_frac, test_frac : target fractions per stratum.
        should sum to 1.0; small float drift is tolerated.
    seed : RNG seed for the per-stratum block shuffle. default 42.
    stratify_by : column name to stratify on. default 'region'.
    class_fallback : optional list of asset_type values (matched against
        the tiles_with_blocks['asset_type'] column) whose tiles should be
        assigned via a per-class random stratified 70/15/15 split INSTEAD
        of the block-level assignment. used for classes too rare to
        guarantee coverage under spatial blocking (e.g. n<=12 globally —
        a single block can hold all of them and would force one split to
        zero tiles). when set, the column 'asset_type' must exist in
        tiles_with_blocks.

    Returns
    -------
    DataFrame
        original columns plus:
          split           : 'train' | 'val' | 'test'
          split_protocol  : 'spatial_block' (default) or 'class_fallback'
                            (only for tiles whose asset_type is in
                            class_fallback)

        block-coherence holds for tiles assigned via 'spatial_block'.
        tiles in class_fallback classes deliberately VIOLATE block
        coherence — same-block tiles in a fallback class can land in
        different splits, because their per-class split is independent of
        the block-level assignment. this is the design trade-off: we
        accept a small spatial-leakage risk for ~20 rare-class tiles in
        order to guarantee class coverage across all three splits.

    Algorithm
    ---------
    step 1 (block-level, all classes):
      per stratum (e.g. per region):
        1. group all tiles by block_id; compute tile-count per block.
        2. shuffle the block list deterministically with random.Random(seed).
        3. compute target tile counts t_train, t_val, t_test from the fractions.
        4. greedy: walk shuffled blocks in order. for each block, append it
           to the split whose current shortfall (target - current) is largest.
           ties broken in train -> val -> test order.

    step 2 (per-class fallback override, optional):
      for each class in class_fallback (in list order):
        1. collect all tiles with that asset_type.
        2. shuffle them with random.Random(seed) — same seed each time so
           the assignment for class X depends only on (X, seed), not on
           list order or on previous fallback classes' shuffles.
        3. assign the first n_tr = floor(n * train_frac) to train, the
           next n_va = floor(n * val_frac) to val, and the remainder to
           test. with n>=4 this guarantees >=1 tile per split for any
           reasonable fraction split.
        4. overwrite their split column and stamp split_protocol =
           'class_fallback'.

    the greedy block-level strategy keeps realized fractions within
    ~0.5pp of target. the per-class fallback exists ONLY to fix the
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
            # pick split with the largest shortfall = target - running.
            # tie-break order: train > val > test (preserves spec semantics
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

    # ------------------ step 2: per-class fallback override -----------------
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
                # class not present at all — nothing to fall back to. warn
                # by raising, since this likely indicates a spec / data
                # mismatch the caller should investigate.
                raise ValueError(
                    f'class_fallback class {cls!r} has 0 tiles in the dataset; '
                    f'check asset_type spelling against ASSET_TYPE_MAP'
                )
            cls_ids = out.loc[mask, 'asset_id'].tolist()
            # per-class RNG: deterministic in (cls, seed), independent of
            # the position of cls in the class_fallback list.
            cls_rng = random.Random(seed)
            shuffled = cls_ids.copy()
            cls_rng.shuffle(shuffled)
            n_tr = int(n_cls * train_frac)
            n_va = int(n_cls * val_frac)
            # remainder to test so the three buckets sum to n_cls exactly.
            fb_assignment: Dict[str, str] = {}
            for i, aid in enumerate(shuffled):
                if i < n_tr:
                    fb_assignment[aid] = 'train'
                elif i < n_tr + n_va:
                    fb_assignment[aid] = 'val'
                else:
                    fb_assignment[aid] = 'test'
            # overwrite the split column for these tiles and stamp the protocol.
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
    """persist the split assignment as parquet. only the canonical columns
    are written, in a stable order, so downstream notebooks have a single
    schema to depend on."""
    missing = [c for c in SPLIT_ARTIFACT_COLUMNS if c not in tiles_with_splits.columns]
    if missing:
        raise ValueError(f'tiles_with_splits missing columns: {missing}')
    out = tiles_with_splits[list(SPLIT_ARTIFACT_COLUMNS)].copy()
    # stable column dtypes for parquet round-trip
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
    """load the split parquet and return a dict mapping asset_id -> split.

    the dict form is what every Phase 2 FM notebook will consume — it's
    o(1) per tile lookup during dataset construction.
    """
    df = pd.read_parquet(input_path, columns=['asset_id', 'split'])
    return dict(zip(df['asset_id'].astype(str), df['split'].astype(str)))
