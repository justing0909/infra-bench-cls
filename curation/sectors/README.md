# Sectors (dataset part B — the benchmark)

The 28 cells the paper evaluates on: seven regions crossed with four sectors,
targeting 1,000 tiles each, 18,756 in total. Three cells come out at 1,001 or
1,002 because each class is allocated its share independently and the rounding
does not have to sum to the target. Output goes to
`data/curated_datasets/dataset_<region>_<sector>_v1_1k/`.

Sampling is proportional within each cell, so the OSM class mix of the source
region survives into the sample. If transmission substations outnumber
distribution substations two to one in a region's extraction, they do so in
the sample too. Classes smaller than their proportional allocation are taken
whole, which is why some cells land slightly under the cap.

**You almost certainly do not need to run any of this.** The finished cells
are on Zenodo. Rebuilding them means refetching imagery from Planetary
Computer, which took a distributed effort across several volunteers. Run this
only to change the taxonomy, move the snapshot date, or add a region.

## Contents

```
sample_v1.py        deduped parquets -> per-cell sample lists
lab_fetcher.ipynb   Colab notebook one volunteer runs for one assigned cell
subsample_to_1k.py  fetched cells -> the 1k-capped cells the paper uses
```

## The chain

**1. Draw the sample lists.** Local, fast.

```bash
python -m curation.sectors.sample_v1
```

Reads `data/PIPELINE/02-deduped-assets/<region>_deduped_assets_<sector>.parquet`
and writes `data/PIPELINE/03-v1-samples/<region>_<sector>_v1_sample.parquet`
plus a `_v1_sample_summary.json`. `--target` changes the cell size and
`--seed` the draw. For energy it prefers `_energy.parquet` and falls back to
`_substations.parquet`, flagging the fallback in the summary.

**2. Fetch the imagery.** Colab, slow, one cell per person.

Upload the sample parquets to `MyDrive/infra_fm/pipeline/` and a zip of this
repository to `MyDrive/infra_fm/code/infra_fm_curation.zip`. Each volunteer
opens `lab_fetcher.ipynb`, sets `MY_ASSIGNED_PARQUET` to their cell, and runs
it top to bottom. The notebook needs a free Microsoft Planetary Computer
account and writes `dataset_<region>_<sector>_v1/` back to the shared Drive
folder. It checkpoints as it goes, so a disconnected session resumes on rerun.

**3. Cap each cell at 1,000 tiles.** Local, fast.

```bash
python -m curation.sectors.subsample_to_1k
python -m curation.sectors.subsample_to_1k --only africa energy
```

Applies the same proportional stratification as step 1 and hardlinks the
chosen `.npy` files into `dataset_<region>_<sector>_v1_1k/`, so the 1k view
costs almost no extra disk. Cells already at or under the cap pass through
unchanged, still hardlinked, so every consumer sees one uniform layout. The
step is idempotent and safe to rerun as more cells finish.

**4. Build the split.** Colab.

[`evaluation/analysis/spatial_split_verification.ipynb`](../../evaluation/analysis/spatial_split_verification.ipynb)
reads the 28 finished cells, projects centroids to Equal Earth, cuts 200 km
blocks, and assigns whole blocks to train, validation, or test at roughly
70/15/15 within each region. It writes
`MyDrive/infra_fm/data/spatial_split/asset_id_to_split_v1.parquet` only if
every verification gate passes. The block logic itself lives in
[`curation/utils/spatial_blocking.py`](../utils/spatial_blocking.py), which all
30 evaluation notebooks import to read the artifact back.

## Why the fetch is the odd step out

Steps 1, 3, and 4 are cheap. Step 2 is hours of network-bound work per cell
against a rate-limited public API, and it was parallelised across people with
Colab runtimes rather than run on one machine. That is the only reason the
chain leaves the local environment and comes back.
