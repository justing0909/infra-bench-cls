# Towards an Infrastructure Foundation Model

This repository supports an end-to-end pipeline for building curated remote-sensing datasets of infrastructure assets, pretraining an imagery encoder on those datasets, and testing the resulting representations on downstream infrastructure tasks.

The near-term goal is not to claim a fully general foundation model. The present objective is more specific and more defensible: build a domain-specific encoder that learns useful representations from weakly labeled infrastructure imagery and then test whether those representations transfer to downstream tasks.

The current implementation is **power-first**, but the intended architecture is **multisector-first**. In other words, the first working pipeline centers on power assets, while the code structure and downstream task design are meant to extend toward water/sewer, transport, rail, roads, and telecom.

## What the repository does

The current codebase supports the following flow:

1. Extract infrastructure assets from GeoFabrik OSM PBF files.
2. Deduplicate spatially proximate assets by asset type.
3. Fetch imagery tiles from Google Earth Engine or Planetary Computer.
4. Run QC and triage to retain usable crops.
5. Assemble accepted tiles into a curated dataset.
6. Run self-supervised pretraining on curated `.npy` imagery crops.
7. Evaluate learned representations on downstream tasks.

At the moment, the best-supported downstream tasks are:

- **Asset classification**: predict infrastructure asset type from imagery.
- **CAADD (Critical Asset Access During Disruption)**: a first-pass disruption accessibility task scaffolded around asset-level accessibility labels.

## Current sector scope

The curation pipeline currently focuses on the energy sector, with the ontology centered on:

- transmission substations
- distribution substations
- untyped substations
- solar farms
- wind farms
- power plants
- generators

See `ONTOLOGY.md` for the current taxonomy and OSM tag mappings.

## Pipeline modules

### Core curation pipeline

- `sources.py`: extracts assets from GeoFabrik PBF files using pyosmium.
- `deduplication.py`: removes spatial duplicates with a KDTree-style pass.
- `imagery.py`: Planetary Computer imagery fetchers and batching utilities.
- `gee_imagery.py`: Google Earth Engine Sentinel-2 fetching for larger runs.
- `qc.py`: imagery quality-control checks.
- `triage.py`: rule-based triage plus `AgentTriager` support.
- `dataset.py`: dataset assembly to `.npy` tiles, `manifest.json`, and `summary.csv`.
- `triage_optimizer.py`: Optuna search for triage thresholds against a gold subset.
- `pipeline.py`: entrypoint that wires the full six-stage curation flow together.
- `tile_types.py`: shared tile dataclasses used across fetchers.

### Representation learning and downstream evaluation

If you imported the downstream scaffold into this repository, you should also now have:

- `infra_fm/common/`: shared dataset loading, transforms, model, and IO helpers.
- `infra_fm/pretraining/train.py`: self-supervised pretraining on curated `.npy` crops.
- `infra_fm/pretraining/export_embeddings.py`: export encoder embeddings for inspection.
- `infra_fm/downstream/asset_classification/train.py`: downstream asset classification benchmark.
- `infra_fm/downstream/caadd/label_generation.py`: first-pass CAADD label generation from accessibility scores.
- `infra_fm/downstream/caadd/train.py`: first-pass CAADD image model training.
- `infra_fm/downstream/caadd/networks.py`: placeholder hooks for future transport/network logic.
- `infra_fm/downstream/caadd/hazards.py`: placeholder hooks for future hazard impact logic.
- `infra_fm/downstream/caadd/dependencies.py`: placeholder hooks for future multisector dependency logic.
- `infra_fm/scripts/inspect_dataset.py`: quick inspection of curated dataset contents.

## Current repo snapshot

This repository is further along than the earlier README implied.

What is already operational:

- the full curation pipeline via `pipeline.py`
- Google Earth Engine Sentinel-2 imagery fetching for regional runs
- QC and triage into curated `.npy` datasets
- self-supervised pretraining on curated datasets
- downstream asset classification from imagery
- a runnable first-pass CAADD scaffold

What remains intentionally incomplete:

- robust multisector curation beyond the energy sector
- a mature CAADD target derived directly from explicit road/network disruption modeling
- stronger benchmarking on larger regional corpora
- higher-resolution global imagery integration in a unified pipeline

## Setup

There is still no single pinned `requirements.txt` for the full original repository, so install the core curation dependencies directly as needed.

### Core curation dependencies

```bash
pip install osmium scipy optuna earthengine-api mgrs requests rasterio
pip install pystac-client planetary-computer pandas numpy matplotlib
```

### Downstream scaffold dependencies

If you imported the `infra_fm/` scaffold, also install:

```bash
pip install torch torchvision pandas numpy scikit-learn matplotlib
```

Optional dependencies:

- `openai` or `anthropic` if you want to use `AgentTriager`
- any local model backend required by your Ollama setup

## Running the curation pipeline

Edit the configuration block at the top of `pipeline.py` before each run.
Important settings typically include:

- `PBF_PATH`
- `OUTPUT_DIR`
- `ASSETS_CSV`
- `DEDUPED_CSV`
- `USE_GEE`
- `GEE_PROJECT`
- `GEE_COMPOSITE`
- `SOURCES`

Then run:

```bash
python pipeline.py --dry-run
python pipeline.py
```

The dry run is the fastest way to verify counts and configuration before a longer imagery job.

## Earth Engine notes

For larger regional runs, the preferred imagery route is the GEE-backed Sentinel-2 fetcher in `gee_imagery.py`.

Current settings include:

- collection: `COPERNICUS/S2_SR_HARMONIZED`
- composite options: `median`, `mosaic`, or `best`
- output resolution: 10 m Sentinel-2 RGB
- retry/backoff logic for HTTP 429 and related transient errors

It is normal to see some 429 rate-limit retries during larger runs. Those do not necessarily indicate failure if successful assets continue increasing.

## Curated dataset layout

Typical outputs include:

- `data/<region>_all_assets.csv`
- `data/<region>_deduped_assets.csv`
- `data/checkpoints/`
- `data/curated_datasets/dataset_<region>_<version>/images/*.npy`
- `data/curated_datasets/dataset_<region>_<version>/manifest.json`
- `data/curated_datasets/dataset_<region>_<version>/summary.csv`

The curated dataset intentionally stores `.npy` arrays rather than compressed image formats so downstream training uses exact pixel values and can remain agnostic to later band-handling changes.

Example image filenames include:

- `osm_way_<id>_sentinel2_gee.npy`
- `osm_node_<id>_sentinel2_gee.npy`
- `inferred_solar_cluster_<n>_sentinel2_gee.npy`

## Representation learning workflow

### 1. Inspect a curated dataset

```bash
python -m infra_fm.scripts.inspect_dataset \
  --dataset-root data/curated_datasets/dataset_maine_v1
```

### 2. Pretrain an encoder

```bash
python -m infra_fm.pretraining.train \
  --dataset-root data/curated_datasets/dataset_maine_v1 \
  --output-dir outputs/pretrain_maine_v1 \
  --epochs 15 \
  --batch-size 16 \
  --image-size 128 \
  --backbone-name resnet18 \
  --num-workers 0
```

This writes a pretraining checkpoint such as `checkpoint_best.pt`, epoch checkpoints, and a run config file.

### 3. Export embeddings

```bash
python -m infra_fm.pretraining.export_embeddings \
  --dataset-root data/curated_datasets/dataset_maine_v1 \
  --checkpoint outputs/pretrain_maine_v1/checkpoint_best.pt \
  --output-dir outputs/embeddings_maine_v1 \
  --image-size 128 \
  --batch-size 32 \
  --num-workers 0
```

## Downstream tasks

### Asset classification

This is the simplest downstream benchmark because the curated dataset already carries weak `asset_type` labels.

#### Frozen-pretrained encoder run

```bash
python -m downstream.asset_classification.train \
  --dataset-root data/curated_datasets/dataset_maine_v1 \
  --output-dir outputs/classification_maine_v1 \
  --checkpoint outputs/pretrain_maine_v1/checkpoint_best.pt \
  --freeze-encoder \
  --epochs 15 \
  --batch-size 16 \
  --image-size 128 \
  --min-class-count 3 \
  --train-fraction 0.8 \
  --max-images 129
```

#### Scratch baseline

```bash
python -m downstream.asset_classification.train \
  --dataset-root data/curated_datasets/dataset_maine_v1 \
  --output-dir outputs/classification_maine_v1_scratch \
  --epochs 15 \
  --batch-size 16 \
  --image-size 128 \
  --min-class-count 3 \
  --train-fraction 0.8 \
  --max-images 129
```

### First Maine benchmark snapshot (April 2026)

On the first curated Maine benchmark:

- curated imagery samples used for downstream classification: **129**
- number of classes retained with `--min-class-count 3`: **7**
- frozen-pretrained run best validation accuracy: **0.3846**
- scratch baseline best validation accuracy: **0.4231**

This should be interpreted carefully:

- the end-to-end curation -> pretraining -> downstream pipeline is now operational
- the downstream task is non-trivial and produces above-chance results
- the current pretraining stage has **not yet** outperformed the scratch baseline on this small Maine corpus
- this likely reflects small-data instability, limited corpus size, and the difficulty of contrastive pretraining on only ~129 curated samples

This is still a useful and honest first benchmark.

### CAADD

CAADD is represented here as **Critical Asset Access During Disruption**.

The first runnable implementation is deliberately modest. It does **not** claim to solve full interdependent infrastructure accessibility. Instead, it provides a first path for wiring the pipeline through a disruption-aware target.

Current v1 assumptions:

1. You supply or create a crude `road_access_score` in `[0, 1]` for each asset.
2. The label generator optionally applies a neutral dependency adjustment stub.
3. The accessibility score is binned into three classes:
   - `severely_disrupted`
   - `partially_accessible`
   - `mostly_accessible`
4. A first image model is trained against those labels.

#### Generate CAADD labels

```bash
python -m infra_fm.downstream.caadd.label_generation \
  --asset-table data/caadd/asset_access_table_maine.csv \
  --output-path outputs/caadd_labels_maine_v1.csv \
  --scenario-id flood_demo_001
```

The asset table must currently include at least:

- `asset_id`
- `road_access_score`

#### Train the CAADD model

```bash
python -m infra_fm.downstream.caadd.train \
  --dataset-root data/curated_datasets/dataset_maine_v1 \
  --label-table outputs/caadd_labels_maine_v1.csv \
  --output-dir outputs/caadd_maine_v1 \
  --checkpoint outputs/pretrain_maine_v1/checkpoint_best.pt \
  --freeze-encoder \
  --epochs 15 \
  --batch-size 16 \
  --image-size 128
```

## Why the CAADD code is scaffolded this way

The long-term intent is broader than power alone. The eventual data/model trajectory includes:

- power
- water / sewer
- road / rail / transport
- telecom

So the first working CAADD implementation is deliberately road-access-first and power-first, but the architecture leaves explicit extension points for:

- hazard-specific closure logic
- multisector network bundles
- dependency adjustments across sectors
- physical accessibility vs operational accessibility
- later cascading interdependencies

That trajectory is reflected in:

- `infra_fm/downstream/caadd/networks.py`
- `infra_fm/downstream/caadd/hazards.py`
- `infra_fm/downstream/caadd/dependencies.py`
- `infra_fm/downstream/caadd/label_generation.py`

## Known limitations

- Sentinel-2 at 10 m resolution is often too coarse for confident discrimination of some substation types.
- Solar assets can dominate raw OSM pulls without careful filtering.
- OSM relation handling is still best-effort for some assets.
- The current pretraining corpus for Maine is too small to draw strong conclusions about learned generality.
- The current CAADD implementation is a scaffold and requires externally supplied first-pass accessibility scores.
- The first downstream benchmark currently favors the scratch baseline over the frozen-pretrained encoder.

## Near-term next steps

1. Run the same curation -> pretraining -> downstream flow on a larger regional corpus (for example Central America).
2. Compare scratch, frozen-pretrained, and fine-tuned downstream asset classification more systematically.
3. Build a crude but explicit road-network-based accessibility table for CAADD instead of hand-supplied scores.
4. Extend the curation ontology and downstream logic toward water/sewer, transport, rail, and telecom.
5. Revisit higher-throughput Earth Engine fetching (for example regional compositing before per-asset clipping) once the first complete pipeline benchmarks are stable.
