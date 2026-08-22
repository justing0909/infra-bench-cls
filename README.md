# Infra-Bench CLS

A classification benchmark for critical-infrastructure recognition from
freely available multimodal satellite imagery.

Infra-Bench CLS pairs a curated global tile dataset — OpenStreetMap
asset locations imaged with Sentinel-1 SAR and Sentinel-2 optical bands —
with a fixed evaluation protocol for comparing Earth-observation
foundation models (FMs) and supervised baselines on a common 13-class
classification task across 7 continental regions and 4 infrastructure
sectors.

This repository contains the curation pipeline, evaluation notebooks,
paper-figure regeneration scripts, and QA notebooks used to produce the
benchmark and the numbers reported in the accompanying paper.

**Interactive results:** <https://justing0909.github.io/infra-bench-cls> —
browse all 30 conditions, filter by class, sector or region, and get a model
recommendation for your compute and label budget. Site source in
[`docs/`](docs).

---

## What the benchmark measures

- **Task:** 13-class single-label classification of critical-infrastructure
  tiles.
- **Classes** (13): transmission substation · distribution substation ·
  distribution (other) · power plant · solar farm · wind farm ·
  wastewater plant · water works · storage tank · airport · train
  station · port terminal · data center.
- **Sectors** (4): energy · water · transport · telecom.
- **Regions** (7): North America · South America · Central America ·
  Europe · Africa · Asia · Australia/Oceania.
- **Input:** Sentinel-2 L2A optical (7 bands) + Sentinel-1 SAR (VV, VH)
  co-located tiles at 10 m resolution. Some FMs use a subset of bands
  and/or handle SAR + optical fusion internally — see each notebook's
  intro for its band mapping.

## Models evaluated

Seven Earth-observation foundation models plus two supervised baselines,
evaluated in a common LP × FT × labels-efficiency matrix. See the paper
for provenance, checkpoints, and pretraining data details.

| Model | Modality | Adaptation |
|---|---|---|
| SatlasPretrain S1 SwinB (Bastani et al. 2023) | S1 | LP, FT |
| SatlasPretrain S2 SwinB (Bastani et al. 2023) | S2 | LP, FT |
| CROMA_base (Fuller et al. 2023) | S1 + S2 (joint) | LP, FT |
| Prithvi-EO-2.0-300M-TL (IBM–NASA, 2024) | S2 | LP, FT |
| AlphaEarth Foundations (Google, 2024) | Precomputed 64-D embeddings | LP only |
| OlmoEarth v1.1-Base (AI2, 2024) | S2 | LP, FT |
| DINOv3 ViT-L/16 (Meta, 2024) | RGB | LP, FT |
| Supervised ResNet-18 (baseline) | S2 + S1 | Full training |
| Random Features ResNet-18 (baseline) | S2 + S1 | LP (frozen random init) |

Each model is evaluated at 1.0× and 0.3× training-data scales for
labels-efficiency comparison — 30 experimental conditions in total.

## Evaluation protocol

- **Spatial-block split** produced by
  `evaluation/analysis/spatial_split_verification.ipynb` (artifact at
  `data/spatial_split/asset_id_to_split_v1.parquet`). Blocks assign
  entire ~0.5° regions to the same partition to prevent near-duplicate
  tiles from leaking across train/val/test. The split is invariant
  across seeds and across every model — every model sees the same
  held-out test set.
- **3 training seeds** (314, 271, 161). LP: head init + DataLoader
  shuffle only. FT: full backbone training with per-depth LLRD γ = 0.75,
  cosine LR + 5 % linear warmup.
- **Best-val checkpoint restored before the held-out test pass** for every
  seed of every model.
- **Aggregate write is gated** by
  `set(SEEDS) == set(FULL_PROTOCOL_SEEDS)` — partial reruns can never
  overwrite an existing 3-seed aggregate.
- **Per-sector F1** is the macro average of per-class F1s for classes in
  a sector, computed on the FULL test set — not filtered to in-sector
  samples.

---

## Repository structure

```
curation/           OSM extraction + Planetary Computer STAC fetch + QC
                    Local / VSCode workflow. Produces the curated .npy
                    dataset consumed by every evaluation notebook.

evaluation/         Colab notebooks — one folder per foundation model
  alphaearth/       lp_1.0x.ipynb, lp_0.3x.ipynb
  croma/            lp_{1.0x,0.3x}.ipynb, ft_{1.0x,0.3x}.ipynb
  dinov3/           lp_{1.0x,0.3x}.ipynb, ft_{1.0x,0.3x}.ipynb
  olmoearth/        lp_{1.0x,0.3x}.ipynb, ft_{1.0x,0.3x}.ipynb
  prithvi/          lp_{1.0x,0.3x}.ipynb, ft_{1.0x,0.3x}.ipynb
  satlas_s1/        lp_{1.0x,0.3x}.ipynb, ft_{1.0x,0.3x}.ipynb
  satlas_s2/        lp_{1.0x,0.3x}.ipynb, ft_{1.0x,0.3x}.ipynb
  resnet18/         supervised_{1.0x,0.3x}.ipynb,
                    random_features_lp_{1.0x,0.3x}.ipynb
  analysis/         cross-FM utility notebooks
                    (spatial_split_verification.ipynb,
                     confusion_matrices.ipynb,
                     compute_weighted_f1.ipynb, etc.)

plots/              paper_figures.py — regenerates every manuscript figure
                    from per-seed and aggregate results JSONs

qa/                 QA notebooks for the curated dataset and results

data/               Curated data (gitignored)
                    maine_subset/  — the geojson used for the Maine
                                     rapid-verification subset

ONTOLOGY.md         Full asset taxonomy + OSM tag mappings
```

## Split execution model

- **Curation runs locally in VSCode / bash.** The pipeline is CPU-bound
  and involves multi-hour jobs over continental-scale PBFs (tens of GB
  each). It writes curated `.npy` tiles plus a manifest to
  `data/curated_datasets/`.

- **Evaluation runs in Google Colab.** The notebooks import each FM's
  loader in-cell (no repo checkout required), fetch the curated dataset
  zip from Google Drive to `/content/`, and write per-seed +
  aggregate results back to Drive. Colab-side deps are installed in the
  notebooks themselves — nothing FM-specific belongs in the top-level
  `requirements.txt`.

- **Figures + QA run locally** from the results JSONs.

---

## Setup

Local environment (curation + figure regeneration + notebook editing):

```bash
uv venv                # or python -m venv .venv
source .venv/bin/activate         # macOS/Linux
# .venv\Scripts\activate          # Windows PowerShell
uv pip install -r requirements.txt
```

For running the FM evaluation notebooks, open them in
[Google Colab](https://colab.research.google.com) and follow the setup
cells at the top. Each notebook is self-contained and installs its own
FM-specific dependencies (`transformers`, `terratorch`, `olmoearth-pretrain-minimal`, `use_croma.py`, etc.).

---

## Reproducing the paper

### Figures

With per-seed and aggregate JSONs available under `results/` and
`aggregates/`:

```bash
python plots/paper_figures.py                  # all figures
python plots/paper_figures.py --figure 4       # single main-text figure
python plots/paper_figures.py --appendix A     # a single appendix set
```

Output PNGs go to `figures/` (gitignored). See
[`plots/README.md`](plots/README.md) for the expected input directory
layout and the full figure list.

### Evaluation

Each of the 30 experimental conditions has its own notebook under
`evaluation/<model>/`. To rerun a condition:

1. Open the notebook in Colab.
2. Ensure `data/spatial_split/asset_id_to_split_v1.parquet` and the
   curated dataset zips are accessible in Drive.
3. Set `SMOKE_ONLY = False` in the training-cell parameters.
4. Run all cells. Per-seed results write to
   `results/fm_eval_<name>/`; an aggregate is written when all 3 seeds
   are present.

The FT notebooks resume automatically from a `checkpoint_final.pt` if
Colab disconnects partway through a seed.

### Curation

The dataset is available from Zenodo (DOI TBD); re-running curation
from source is only needed if you want to change the class taxonomy,
regenerate at a different snapshot date, or add regions.

```bash
python curation/pipeline.py --dry-run          # verify config
python curation/pipeline.py                    # full run
```

For batch runs across regions, use
`curation/run_pipeline_from_collapsed_assets.py`. It auto-skips regions
with existing `_SUCCESS` markers.

---

## Dataset access

The curated Infra-Bench CLS dataset and its Maine rapid-verification
subset are archived on Zenodo (DOIs pending, will be added on paper
acceptance). Both are distributed under the **Open Database License
(ODbL) 1.0** because they are derived from OpenStreetMap.

Attribution required when redistributing or deriving from either
dataset:

> *Contains information from OpenStreetMap Planet PBF (Geofabrik
> snapshot 2026-05-26). © OpenStreetMap contributors, licensed under
> ODbL 1.0. Imagery derived from modified Copernicus Sentinel-1 /
> Sentinel-2 data.*

The code in this repository is separately licensed under MIT (see
`LICENSE`).

---

## Known dataset limitations

Documented here so paper readers and downstream users have full context.
None of these undermine the benchmark's methodology; they are reported
explicitly.

**~2–3 % OSM extract undercount (fixed after 2026-05-26).** The
pyosmium `idx="sparse_file_array,locations.idx"` option silently dropped
some node locations on certain PBFs, causing a small fraction of assets
to be missed and fewer multipolygon areas to be assembled than the
source actually contained. The bug was identified in late May 2026 when
transport extraction returned 1 airport instead of 88; `sources.py` was
patched to drop the `idx` parameter. The bug is uniform-random across
regions and asset types, so class proportions and per-region
representativeness are preserved. The imagery datasets shipped with the
paper correspond to extracts taken before the patch and are therefore
mildly undercount — we chose not to re-extract globally because F1
metrics are invariant to overall sample count.

**KDTree latitude bug in pre-2026-05 dedup outputs.** Earlier
deduplicated parquets were built with a scipy KDTree on raw (lat, lon)
degree pairs plus a `threshold / 111320` conversion that only held at
the equator. The fix swaps to sklearn BallTree + haversine, which is
geographically correct. Per-region impact was largest in high-latitude
Asia (−3266 rows). The current dedup parquets reflect the corrected
logic.

**Imagery datasets are sample-capped at 25 000 tiles per region.** For
regions with more than 25 000 deduped assets (Asia, North America,
Europe), the on-disk imagery represents a random sample of the deduped
parquet. This is a deliberate compute-budget decision, not a bug.

---

## Known evaluation quirks

**"Ways scanned: 0" in `curation/sources.py` progress logs is normal.**
`NodeLocationsForWays` processes all nodes first to build a location
index, then resolves ways in a second internal pass — the way count
doesn't update during the first pass.

**Large parquets cause Colab RAM issues.** Load only the columns you
need:

```python
df = pd.read_parquet(path)[['asset_id', 'asset_type', 'lat', 'lon']].copy()
```

**Prithvi's HuggingFace loader can be fragile.** Requires
`trust_remote_code=True` and `num_labels=0`; TerraTorch's
scipy/dask/rapids conflict is documented in the Prithvi notebook.

**SatlasS1 and SatlasS2 cannot be run concurrently in the same Colab
runtime** — they share `/content/datasets/` and race on the extract
step.

---

## Related work

The closest published work on classifying critical infrastructure from
satellite imagery is Ye, Ward, De Plaen, and Koks (2025), "Big Earth
Data" (DOI: 10.1080/20964471.2025.2490408), which trains supervised
Faster R-CNN with ResNet-101 on WorldView-3 30 cm imagery over Vietnam
to detect transmission towers and poles. The contributions are
complementary:

- Infra-Bench CLS is a benchmark for evaluating FMs on a fixed
  classification task; Ye et al. is a supervised object-detection
  study.
- Infra-Bench CLS uses freely available 10 m Sentinel imagery at global
  scale; Ye et al. requires commercial 30 cm tasking.
- Infra-Bench CLS uses weak OSM labels across 13 classes and 7 regions;
  Ye et al. uses manual annotation for 2 classes over Vietnam.

---

## Citation

Paper citation will be added on posting to arXiv.

---

## License

Code: **MIT** (see [`LICENSE`](LICENSE)).

Dataset: **ODbL 1.0**, distributed via Zenodo (see *Dataset access*
above).

---

## Authors

- **Justin Guthrie** (lead author) — Northeastern University, George Mason University, Enodia Inc.
- **Edward Oughton** (advisor) — George Mason University
- **Konrad Wessels** (coauthor) — George Mason University
- **Matthew Rice** (coauthor) — George Mason University
- **Isaac Corley** (coauthor) — Taylor Geospatial
