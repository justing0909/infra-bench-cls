# Towards an Infrastructure Foundation Model — Curation Pipeline

**Justin Guthrie, Edward Oughton et al.**

This repository contains the data curation pipeline for building a
labeled satellite imagery corpus of critical infrastructure assets.
The corpus supports pretrained representation learning toward an
infrastructure foundation model (FM).

> This is a "towards a foundation model" effort — the goal is a
> domain-specific pretrained encoder that produces meaningful
> infrastructure asset embeddings, not a fully general FM.

---

## Motivation

Existing pretrained models (e.g. COCO-trained YOLO, general remote
sensing FMs) do not recognize critical infrastructure assets from
overhead imagery. A COCO-trained model detects "broccoli" and "clocks"
at substations and water treatment plants respectively — the wrong
visual priors entirely. Domain-specific pretraining on a curated
infrastructure corpus is needed to fix this.

---

## Pipeline overview

```
OSM / GeoFabric          Authoritative infrastructure asset locations + weak labels
      ↓
sources.py               Query asset geometries by sector and bounding box
      ↓
data/<region>_assets.csv Asset inventory (lat/lon, asset type, OSM ID, name)
      ↓
imagery.py               Fetch imagery tiles centered on each asset centroid
                         Sources: Sentinel-2 (global), NAIP (US), Maxar (planned)
      ↓
qc.py                    Basic imagery quality control (cloud cover, valid pixels,
                         edge artifacts) — coming soon
      ↓
deduplication.py         Remove near-duplicate tiles from spatially proximate assets
                         — coming soon
      ↓
triage.py                Confidence triage: accept / flag for review / reject
                         Based on label-image agreement. Agentic AI version planned.
                         — coming soon
      ↓
dataset.py               Assemble COCO-style training dataset with metadata
                         — coming soon
```

---

## Repository structure

```
infra_curation/
├── sources.py              Query OSM for asset geometries
├── imagery.py              Fetch imagery tiles (NAIP + Sentinel-2)
├── qc.py                   Basic QC — coming soon
├── deduplication.py        Deduplication — coming soon
├── triage.py               Confidence triage — coming soon
├── dataset.py              Dataset assembly — coming soon
├── ONTOLOGY.md             Infrastructure asset taxonomy + OSM tag mapping
├── notebooks/
│   ├── 01_sources.ipynb    Run + inspect source queries
│   └── 02_imagery.ipynb    Fetch + visually inspect imagery tiles
└── data/                   Asset CSVs (gitignored — not committed to repo)
```

---

## Asset ontology

See [`ONTOLOGY.md`](ONTOLOGY.md) for the full infrastructure asset
taxonomy and OSM tag mappings.

Current status:
- **Energy** — active, full hierarchy defined
- **Transport** — stub
- **Water** — stub
- **Telecom** — stub

---

## Imagery sources

| Source | Resolution | Coverage | Status |
|---|---|---|---|
| NAIP | 60cm | USA only | Active — via Microsoft Planetary Computer |
| Sentinel-2 | 10m | Global | In progress |
| Maxar (Vantor) | 30cm | Selected areas | Pending NASA CSDA approval |

The goal is three images per asset across sources where available,
supporting a multi-scale, multi-sensor pretrained encoder.

---

## Baseline

A COCO-trained YOLOv8 model produces no meaningful detections on
overhead infrastructure imagery — detecting "broccoli" at substations
and "clocks" at water treatment plants. This failure motivates
domain-specific pretraining. See `notebooks/` in the detection
notebook for the full baseline analysis.

---

## Related work

- SeCo: Seasonal Contrast — unsupervised pretraining from remote sensing data
- Tile2Vec — contrastive representation learning for satellite imagery
- ScaleMAE — scale-aware masked autoencoder for remote sensing
- xBD — building damage assessment dataset
- PGRID — power grid reconstruction from aerial imagery

---

## Dependencies

```
pystac-client
planetary-computer
rasterio
requests
pandas
numpy
matplotlib
ultralytics
```

---

## Setup

```bash
git clone https://github.com/justing0909/towards_an_infra_fm.git
cd towards_an_infra_fm
pip install -r requirements.txt
```

Run notebooks in order:
1. `notebooks/01_sources.ipynb` — query OSM, save asset CSV
2. `notebooks/02_imagery.ipynb` — fetch tiles, visual inspection

---

## Status

Early-stage pipeline. Currently functional:
- OSM querying via Overpass API with retry + endpoint rotation
- NAIP tile fetching via Planetary Computer
- Per-sector visual tile inspection

In progress:
- Sentinel-2 integration
- QC, deduplication, triage, dataset assembly
- Agentic AI-assisted confidence triage