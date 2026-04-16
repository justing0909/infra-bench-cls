# Project Summary

This file is a repo-local working summary derived from
`PROJECT_SUMMARY.md.pdf` (April 2026) and cross-checked against the current
repository snapshot on 2026-04-09.

## Project Goal

Build a domain-specific pretrained encoder for critical infrastructure assets
from satellite imagery. The framing is intentionally "towards" a foundation
model: the core contribution is a curated corpus, a pretraining pipeline, and
downstream evaluation rather than a fully general production FM.

The motivating failure case is that general-purpose detectors do not transfer
well to infrastructure imagery. The project summary specifically calls out
COCO-trained models labeling substations as unrelated everyday objects.

## Team

- Justin Guthrie: lead / graduate researcher
- Edward Oughton: advisor
- Jack Watson and Raghav Pant: collaborators

## Confirmed Codebase State

The current repository already contains the main curation stages described in
the PDF:

- `sources.py`
- `deduplication.py`
- `imagery.py`
- `gee_imagery.py`
- `qc.py`
- `triage.py`
- `dataset.py`
- `pipeline.py`
- `triage_optimizer.py`

Important repo/document alignment notes:

- `pipeline.py` orchestrates all six curation stages today.
- `triage.py` includes both `RuleBasedTriager` and `AgentTriager`.
- `triage_optimizer.py` exists and is wired around Optuna.
- The PDF references notebooks `01_sources.ipynb` through `06_dataset.ipynb`,
  but those notebooks are not present in this checkout.
- `07_results.ipynb` is present and appears to hold results visualizations.
- The old README understated implementation progress and still described
  multiple modules as "coming soon."

## Pipeline As Described In The Project Brief

1. Asset extraction from GeoFabrik OSM PBF files.
2. Spatial deduplication with a KDTree and per-type grouping.
3. Sentinel-2 imagery fetching through Google Earth Engine, with Planetary
   Computer as fallback.
4. Basic QC based on valid pixels, brightness, and edge artifacts.
5. Confidence triage with rule-based heuristics and an agent-backed path.
6. Dataset assembly into raw `.npy` tiles plus metadata.

## Architecture Choices Called Out In The PDF

- GeoFabrik PBFs over Overpass for scale and offline operation.
- Two-pass PBF filtering to preserve way geometry resolution.
- GEE over Planetary Computer for larger runs because clipping happens server-side.
- Median composites for cloud robustness when temporal dynamics are not central.
- KDTree deduplication for global-scale efficiency.
- Permissive triage thresholds because weak noise is less harmful for contrastive
  pretraining than for strict supervised classification.
- Raw `.npy` tile storage to preserve exact pixel values.

## Scope And Ontology

The project summary is focused on the energy sector first. Confirmed asset types
from the brief include:

- `energy.transmission.substation`
- `energy.distribution.substation`
- `energy.distribution.substation_untyped`
- `energy.generation.solar_farm`
- `energy.generation.wind_farm`
- `energy.generation.power_plant`
- `energy.generation.generator`

Lower-confidence classes such as towers and poles are described as excluded at
the medium-confidence threshold used for current runs.

## Current Limitations

The PDF emphasizes a few persistent issues:

- Sentinel-2 resolution is usually too coarse for confident substation discrimination.
- Solar mapping noise in OSM can dominate without additional filtering.
- Some OSM relation handling is still partial or best-effort.
- Global-scale GEE runs may be constrained by quota.
- NAIP and Maxar are not yet folded into a broader multi-source global pipeline.

## Near-Term Plan From The Brief

- Run larger curation jobs over North America and Europe.
- Use the results to move into pretraining.
- Add a future `pretraining.py` stage in the same repository.

The intended pretraining direction in the brief is a ScaleMAE-style encoder with
contrastive learning over same-location infrastructure tiles.

## Working Recommendation

For future edits, the most leverage appears to be:

1. Keep repo documentation synchronized with the actual pipeline state.
2. Make the regional/global run configuration easier to reproduce.
3. Add the first pretraining scaffold once the curation workflow is stable.
