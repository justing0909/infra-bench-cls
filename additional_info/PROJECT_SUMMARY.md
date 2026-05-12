# PROJECT_SUMMARY.md
# Towards an Infrastructure Foundation Model (TIF-M)
# Last updated: April 2026

## Overview

TIF-M is a domain-specific pretrained encoder for critical infrastructure assets
(substations, solar farms, wind farms, generators, etc.) using satellite imagery.
The goal is to build a foundation model that learns generalizable representations
of infrastructure assets from multimodal satellite imagery, validated through
downstream classification and co-location tasks.

**Advisor:** Edward Oughton
**Collaborators:** Jack Watson

---

## Current State (April 2026)

### Curation Pipeline — COMPLETE
The full end-to-end curation pipeline is implemented and validated at scale:

1. **OSM extraction** (`sources.py`) — GeoFabrik PBF → asset geometries + weak labels
   - Filter presets: `"substation"` (recommended) or `"full"`
   - Uses `osmium` with `NodeLocationsForWays`, 2-pass power-only PBF cache
   - KDTree-based O(n log n) deduplication (`deduplication.py`)

2. **Imagery fetch** (`stac_imagery.py`) — Microsoft Planetary Computer STAC
   - Primary modalities: `sentinel2_ms` (7 bands) + `sentinel1` (2 bands)
   - Optional: `landsat_thermal` (1 band, dropped for speed in recent runs)
   - Adaptive concurrency, checkpointing every 200 tiles
   - Yields vary by region: ~99% central america, ~58% australia-oceania, ~50% africa

3. **QC** (`qc.py`) — valid pixel ratio, edge artifacts, min size, value range
4. **Triage** (`triage.py`) — rule-based confidence scoring
5. **Dataset assembly** (`dataset.py`) — COCO-style manifest + .npy tiles

**Orchestration:** `run_pipeline_from_collapsed_assets.py` — batch processes all
regions sequentially, skips completed regions via `_SUCCESS` files.

### Completed Datasets (substations only, stac_v1)
| Region | Tiles | Modalities | Notes |
|---|---|---|---|
| central-america | ~1,782 | S2+SAR+TIR | First complete run |
| australia-oceania | ~3,200 | S2+SAR | ~58% yield |
| africa | ~5,500 | S2+SAR | ~50% yield, expected |
| south-america | ~9,000 | S2+SAR | Complete |

### Pending Datasets
| Region | Status | Blocker |
|---|---|---|
| north-america | Deduped parquet needed | Jack (memory constraint) |
| asia | Deduped substations parquet needed | Jack (memory constraint) |
| europe | Deduped parquet needed | Jack (memory constraint) |

### Pretraining — IN PROGRESS
- **Architecture:** SimCLR + ResNet-18
- **Ablation grid completed** on central america (1,782 tiles):
  - Best config: temperature=0.1, projection_dim=128, 100 epochs
  - Linear probe tail mean: 0.447 (vs 0.336 random init)
  - Key finding: lower temperature (0.1 > 0.2) consistently better
  - Key finding: tail mean is more honest metric than best val_acc (random init
    has high best but horrible tail mean due to majority-class collapse)
- **Multi-continent pretraining running** (~20,000 tiles, 200 epochs, temp=0.1)
  - Config: temperature=0.1, projection_dim=128, cosine LR annealing
  - Expected completion: Tuesday morning

### Downstream Tasks

#### Asset Classification (`downstream/asset_classification/`)
- 3-class problem: tx_substation, dx_substation, dx_substation_untyped
- Evaluation: linear probe (frozen encoder) + fine-tuned
- Key metric: tail mean val_acc (last 5 epochs) — more honest than best val_acc
- Per-class accuracy tracked — random init collapses to majority class
- Weighted CrossEntropyLoss handles class imbalance

**Results so far (central america, 1,782 tiles):**
| Encoder | Mode | Best | Tail mean | Tail std |
|---|---|---|---|---|
| Random init | Fine-tuned | 0.611 | 0.336 | 0.116 |
| SimCLR ep100 t=0.1 | Linear probe | 0.546 | 0.447 | 0.084 |
| SimCLR ep25 t=0.2 | Linear probe | 0.566 | 0.410 | 0.076 |

**Multi-continent classification:** pending pretraining completion.

#### Co-location (`downstream/colocation/`)
- **Status:** In development
- **Scope:** Power-only co-location (TX near DX substations, substations near
  power plants/generators) using existing asset tables
- **Method:** Spatial join within deduped asset parquets, KDTree radius query
- **Labels:** Multi-label binary vector per tile
- **Next:** Label generation for central america, then linear probe on frozen
  multi-continent encoder embeddings

---

## Architecture Decisions

### Why STAC over GEE
Planetary Computer is free, no authentication required, global coverage, supports
sentinel-1-rtc (terrain corrected). GEE remains available as fallback.

### Why substations first
- Visually consistent at 10m resolution
- Network-critical (failure propagates)
- Manageable global asset counts vs. poles (millions) or lines
- Clean bounding boxes for future Maxar 30cm handoff via NASA CSDA

### Why SimCLR
- Well-understood, reproducible
- `SimCLRModel` in `downstream/common/models.py` is the canonical definition;
  `pretraining/train.py` imports it directly (no duplicate model definition).
- Known limitation: needs 10k+ samples and 200+ epochs for strong linear probes

### Band configuration
- Training tiles: 9 bands (sentinel2_ms bands 0-6 + sentinel1 bands 7-8)
- Some early central america tiles: 10 bands (+ landsat_thermal band 9)
- `--band-indices` must match tile configuration exactly
- Augmentations: spatial transforms on all bands, value transforms on
  optical bands only (0:n_optical), never on SAR or thermal

---

## Critical Bug Fixed (April 2026)
`common/models.py` `EncoderBackbone` previously used
`nn.Sequential(*list(net.children())[:-1])` which produced key names like
`backbone.encoder.conv1.weight`. Pretraining checkpoints use `backbone.conv1.weight`.
This caused silent random init on every classification run with `strict=False`.
**Fixed:** `net.fc = nn.Identity()` + `self.encoder = net` — keys now match exactly.
Always verify with `missing, unexpected = model.load_state_dict(..., strict=False)`
and confirm 0 missing keys before trusting classification results.

---

## Key File Locations

### Curation
```
curation/
  sources.py                    # OSM extraction
  stac_imagery.py               # Planetary Computer fetch
  qc.py                         # Quality control
  triage.py                     # Confidence scoring
  dataset.py                    # Dataset assembly
  pipeline.py                   # End-to-end orchestration
  run_pipeline_from_collapsed_assets.py  # Batch runner
  extract_substations_all.py    # Batch substation extraction
  helpers/tile_types.py         # TileResult, MODALITY_REGISTRY
  utils/io_utils.py             # load_asset_table etc.
```

### Downstream
```
downstream/
  common/
    models.py       # EncoderBackbone, SimCLRModel, LinearClassifier
    datasets.py     # NpyInfrastructureDataset
    transforms.py   # MultimodalResize, MultimodalAugment
    io.py           # parse_asset_id_from_filename (handles STAC naming)
    utils.py        # set_seed, choose_device, save_checkpoint etc.
  asset_classification/
    datasets.py     # AssetClassificationDataset, LabelSpace
    train.py        # Training loop with tail metrics + per-class accuracy
  colocation/
    labels.py       # Co-location label generation (spatial join)
    datasets.py     # ColocationDataset
    train.py        # Training loop (multi-label BCE)
  pretraining/
    model.py        # SimCLRModel (canonical for pretraining)
    augmentations.py # TensorSimCLRTransform (multimodal-aware)
    losses.py       # NT-Xent loss
    train.py        # Pretraining loop with cosine LR + resume
    datasets.py     # InfrastructureImageDataset
```

### Data
```
data/
  pbf/power_only/               # Pre-filtered power-only PBFs
  PIPELINE/
    01-extracted-assets/        # Raw asset parquets
    02-deduped-assets/          # Deduplicated asset parquets
  curated_datasets/
    dataset_<region>_stac_v1/   # Completed STAC datasets
      images/                   # .npy tiles (C,H,W)
      manifest.json             # Full metadata
      summary.csv               # Lightweight summary
      _SUCCESS                  # Completion marker
  results/
    pretrain_central_america/   # 25-epoch central america checkpoint
    experiment_grid/            # Ablation results (9 experiments)
    pretrain_multicontinent/    # 200-epoch multi-continent (in progress)
    classification/             # Classification results
```

---

## Colab Notebooks (in Drive: My Drive/infra_fm/)
- `infra_fm_stac_fetch.ipynb` — STAC imagery fetch for new regions
- `infra_fm_classification.ipynb` — Classification comparison (3 experiments)
- `infra_fm_experiment_grid.ipynb` — Ablation grid (9 experiments)
- `infra_fm_multicontinent_pretrain.ipynb` — 200-epoch multi-continent run

---

## Immediate Next Steps
1. Wait for multi-continent pretraining to complete (~Tuesday AM)
2. Run classification with multi-continent checkpoint on central america
3. Get substations parquets from Jack (NA, Asia, Europe)
4. Run STAC fetch for those three regions
5. Run global pretraining once all regions complete
6. Build co-location labels + run co-location downstream task
7. Expand to full power ontology (generators, power plants) — after substations done

## Longer Term
- Maxar 30cm imagery via NASA CSDA program (access pending clarification)
- Expand asset ontology: water/sewer, telecom, transport, critical facilities
- MAE pretraining as alternative to SimCLR
- ViT backbone as alternative to ResNet-18
- Cross-sector co-location once non-power sectors are available