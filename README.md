# Towards an Infrastructure Foundation Model

This repository contains the curation pipeline for building a labeled satellite
imagery corpus of critical infrastructure assets. The immediate goal is not a
fully general foundation model. It is a domain-specific pretrained encoder that
learns useful infrastructure representations from weakly labeled remote sensing
tiles.

The April 2026 project brief that we are working from is captured in
`PROJECT_SUMMARY.md`.

## What The Repo Does

The current codebase supports an end-to-end data curation workflow:

1. Extract infrastructure assets from GeoFabrik OSM PBF files.
2. Deduplicate spatially proximate assets by asset type.
3. Fetch imagery tiles with Google Earth Engine or Planetary Computer.
4. Run basic quality control on fetched tiles.
5. Triage tiles with rule-based heuristics or an agent-backed reviewer.
6. Assemble accepted tiles into a training dataset.

The focus today is the energy sector, with the ontology centered on:

- transmission substations
- distribution substations
- untyped substations
- solar farms
- wind farms
- power plants
- generators

See `ONTOLOGY.md` for the asset taxonomy and tag mappings.

## Pipeline Modules

- `sources.py`: extracts assets from GeoFabrik PBF files.
- `deduplication.py`: removes spatial duplicates with a KDTree-based pass.
- `imagery.py`: Planetary Computer imagery fetchers and batching utilities.
- `gee_imagery.py`: Google Earth Engine Sentinel-2 fetching for larger runs.
- `qc.py`: imagery quality control checks.
- `triage.py`: rule-based triage plus `AgentTriager`.
- `dataset.py`: dataset assembly to `.npy` tiles, `manifest.json`, and `summary.csv`.
- `triage_optimizer.py`: Optuna search for triage thresholds against a gold subset.
- `pipeline.py`: script entrypoint that wires the full pipeline together.
- `tile_types.py`: shared tile dataclasses used across fetchers.

## Current Repo Snapshot

This repo is further along than the original early-stage README implied.

- `pipeline.py` runs the full six-stage flow.
- `triage.py` includes both `RuleBasedTriager` and `AgentTriager`.
- `triage_optimizer.py` is present in the repo.
- `07_results.ipynb` and the generated PNGs capture current funnel and signal analysis.

One notable gap between the project brief and this checkout is notebooks:
the brief references `01_sources.ipynb` through `06_dataset.ipynb`, but those
files are not present in this repository snapshot.

## Setup

There is no checked-in `requirements.txt` at the moment, so install the core
dependencies directly:

```bash
pip install osmium scipy optuna earthengine-api mgrs requests rasterio
pip install pystac-client planetary-computer pandas numpy matplotlib
```

Optional dependencies:

- `openai` or `anthropic` if you want to use `AgentTriager`
- any local model backend required by your Ollama setup

## Running The Pipeline

Edit the configuration block at the top of `pipeline.py` before each run.
Important settings include:

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

The dry run is the fastest way to verify counts and configuration before a long
imagery job.

## Earth Engine Notes

The default path in `pipeline.py` is the GEE-backed Sentinel-2 fetcher, which is
the preferred route for larger regional or continental runs.

Key settings:

- collection: `COPERNICUS/S2_SR_HARMONIZED`
- composite options: `median`, `mosaic`, or `best`
- project ID must match your actual GEE project

If Earth Engine is unavailable, the code can fall back to Planetary Computer
fetching.

## Output Layout

Typical outputs include:

- `data/<region>_all_assets.csv`
- `data/<region>_deduped_assets.csv`
- `data/checkpoints/`
- `data/dataset_<region>_<version>/images/*.npy`
- `data/dataset_<region>_<version>/manifest.json`
- `data/dataset_<region>_<version>/summary.csv`

The dataset output intentionally stores raw `.npy` arrays rather than compressed
image formats so downstream training can use exact pixel values.

## Results Artifacts

The repo currently includes analysis artifacts such as:

- `07_results.ipynb`
- `pipeline_funnel.png`
- `signal_distributions.png`
- `spatial_coverage.png`
- per-asset tile preview figures

These are useful for validating asset mix, spatial coverage, and the current
limits of Sentinel-2 for certain asset classes.

## Known Limitations

- Sentinel-2 at 10 m resolution is often too coarse for confident substation discrimination.
- Solar assets can dominate raw OSM pulls without careful filtering.
- Relation handling in OSM is still best-effort for some assets.
- NAIP and Maxar are not yet integrated into a unified global workflow.
- The pretraining stage described in the project brief is not yet implemented in this repo.

## Next Steps

The project brief points toward three near-term priorities:

1. Run larger regional or continental curation jobs.
2. Stabilize documentation and reproducibility around the current pipeline.
3. Start the pretraining stage once the corpus is large enough.
