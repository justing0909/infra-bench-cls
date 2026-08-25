# Infra-Bench CLS

A classification benchmark for critical-infrastructure recognition from freely
available multimodal satellite imagery.

Infra-Bench CLS pairs a curated global tile dataset — OpenStreetMap asset
locations imaged with Sentinel-1 SAR and Sentinel-2 optical bands — with a
fixed evaluation setup for comparing Earth-observation foundation models (FMs)
and supervised baselines on a common classification task across 7 continental
regions and 4 infrastructure sectors.

This repository contains the curation pipeline, the evaluation notebooks, the
figure notebook, and the source of the results site.

**Interactive results:** <https://justing0909.github.io/infra-bench-cls> —
browse all 30 conditions, filter by class, sector or region, and get a model
recommendation for your compute and label budget. Site source in
[`docs/`](docs).

---

## What the benchmark measures

- **Task:** single-label classification of critical-infrastructure tiles.
- **Input:** Sentinel-2 L2A optical (B02, B03, B04, B08, B8A, B11, B12) plus
  Sentinel-1 SAR (VV, VH), co-located at 10 m, as 600 m × 600 m tiles centered
  on each asset. Some FMs use a subset of bands or handle SAR and optical
  fusion internally; each notebook's intro gives its band mapping.
- **Size:** 18,756 tiles across 28 cells, one per region and sector pair.
  Each cell targets 1,000 tiles; three land at 1,001 or 1,002 because the
  per-class allocation rounds independently, and many land well under because
  the region ran out of assets or the imagery fetch did not return them.
- **Sectors** (4): energy · water · transport · telecom.
- **Regions** (7): North America · South America · Central America · Europe ·
  Africa · Asia · Australia/Oceania.

### 13 classes in the data, 10 in the paper

The dataset and every evaluation run use the full 13-class taxonomy:
transmission substation · distribution substation · distribution (other) ·
power plant · solar farm · wind farm · wastewater plant · water works ·
storage tank · airport · train station · port terminal · data center.

The paper reports 10 of them. Wind farm and port terminal have 3 test tiles
each and water works has 118, too few to support a stable per-class F1, so
they are dropped from the reported aggregates. Because every stored metric is
per-class, the 10-class macro and weighted F1 are recomputed from the 13-class
outputs without retraining: drop the three classes, re-average. The test set
goes from 2,813 tiles per seed to 2,689.

Per-region F1 is the one exception and stays 13-class. The aggregate JSONs
store it already averaged over classes, so there is nothing left to re-average,
and the checkpoints needed to redo it were not retained. The manuscript says
so in the Figure 9 caption and in the limitations, and the results site raises
a callout whenever that breakdown is selected.

## Models evaluated

Seven backbone configurations — six Earth-observation foundation models and one
general-purpose vision model — plus two supervised baselines, in a common
LP × FT × labels-efficiency matrix. See the paper for provenance, checkpoints,
and pretraining data.

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

Each is evaluated at 1.0× and 0.3× training-data scales, giving 30 conditions.

## Evaluation setup

- **Spatial-block split** produced by
  [`evaluation/analysis/spatial_split_verification.ipynb`](evaluation/analysis/spatial_split_verification.ipynb).
  Tile centroids project to Equal Earth (EPSG:8857), the world is cut into
  200 km blocks, and whole blocks go to one partition so near-duplicate tiles
  cannot leak across train, validation, and test. Assignment targets 70/15/15
  within each region. The split is fixed across seeds and across every model,
  so every model sees the same held-out test set.
- **3 training seeds** (314, 271, 161). Linear probing varies head init and
  DataLoader shuffle only. Fine-tuning trains the full backbone with per-depth
  LLRD γ = 0.75 and a cosine schedule with 5% linear warmup.
- **Best-validation checkpoint restored before the held-out test pass**, for
  every seed of every model.
- **Every published aggregate is a genuine 3-seed mean.** Verified: all 30
  conditions in `docs/data/results.json` carry three per-seed values, for every
  metric and every class.
- **A partial rerun cannot overwrite a complete aggregate.** Every aggregate
  write is guarded, in one of two ways: 30 of them test
  `set(SEEDS) == set(FULL_PROTOCOL_SEEDS)` and print a note instead of writing
  when it fails, and the four combine-from-disk cells in the SatlasPretrain
  fine-tune notebooks raise `FileNotFoundError` unless all three per-seed JSONs
  are present. Each aggregate also stamps `agg['seeds']`.
- **Per-sector F1** is the macro average of per-class F1s for the classes in a
  sector, each computed on the full test set rather than filtered to in-sector
  samples.

---

## Repository structure

```
curation/           shared curation machinery
  ontology.py         13-class taxonomy + OSM tag matchers
  sources.py          PBF extraction (pyosmium)
  deduplication.py    BallTree + haversine spatial dedup
  stac_imagery.py     Planetary Computer tile fetcher
  qc.py               tile quality control
  triage.py           rule-based confidence triage
  dataset.py          manifest + images assembly
  paths.py            repo-root-anchored paths for every artifact
  visualize_tiles.py  local tile viewer
  refetch_from_manifest.py  rebuild a dataset from the exact scenes its
                    manifest names, rather than by search
  helpers/, utils/    shared dataclasses, IO, spatial blocking, timing log

  sectors/          dataset part B — the sampled cross-sector benchmark
                    sample_v1.py, lab_fetcher.ipynb, subsample_to_1k.py
  substations/      dataset part A — the full unsampled substation extraction
                    pipeline.py, resync_dataset_manifest.py, stac_fetch.ipynb,
                    global_fetch_qa.ipynb

tests/              unit tests for the split, dedup, and taxonomy
                    (python -m unittest discover -s tests -t .)

evaluation/         Colab notebooks, one folder per model — see
                    evaluation/README.md for per-model environments
  alphaearth/       lp_{1.0x,0.3x}.ipynb
  croma/            lp_{1.0x,0.3x}.ipynb, ft_{1.0x,0.3x}.ipynb
  dinov3/           lp_{1.0x,0.3x}.ipynb, ft_{1.0x,0.3x}.ipynb
  olmoearth/        lp_{1.0x,0.3x}.ipynb, ft_{1.0x,0.3x}.ipynb
  prithvi/          lp_{1.0x,0.3x}.ipynb, ft_{1.0x,0.3x}.ipynb
  satlas_s1/        lp_{1.0x,0.3x}.ipynb, ft_{1.0x,0.3x}.ipynb
  satlas_s2/        lp_{1.0x,0.3x}.ipynb, ft_{1.0x,0.3x}.ipynb
  resnet18/         supervised_{1.0x,0.3x}.ipynb,
                    random_features_lp_{1.0x,0.3x}.ipynb
  analysis/         cross-model utilities: spatial_split_verification.ipynb,
                    confusion_matrices.ipynb, compute_weighted_f1.ipynb,
                    per_sector_f1_catchall.ipynb

plots/              paper_figures.ipynb — every manuscript figure
docs/               the results site, plus tools/ that build and validate its data
data/               curated data. the repo carries the deduplicated asset
                    tables (the last step before imagery) and nothing further
                    back: the raw extractions need 51 GB of Geofabrik PBFs that
                    are not distributable. imagery comes from Zenodo.
  PIPELINE/02-deduped-assets/   inputs to sample_v1 and pipeline.py
  spatial_split/      asset_id_to_split_v1.parquet — the fixed train/val/test
                      assignment every evaluation run consumes (committed)
  alphaearth/         embeddings_2024.parquet — 64-D AlphaEarth features,
                      needed by its two notebooks and by split regeneration
  maine_subset/       the geojson behind the Maine/New Hampshire validation
  Infra-FM-timing-log.xlsx   per-region curation timings and tile counts
ONTOLOGY.md         full asset taxonomy + OSM tag mappings
```

## Where each stage runs

Curation is CPU-bound and file-heavy, so it runs locally. Training needs GPUs
nobody on this project owns, so it runs in Colab. Figures and the site come
back local, except the figure notebook itself, which reads results straight
from Drive.

| Stage | Where | Entry point |
|---|---|---|
| OSM extraction, dedup, sampling | Local | `curation/sectors`, `curation/substations` |
| Imagery fetch | Colab | `curation/sectors/lab_fetcher.ipynb` |
| Spatial split | Colab | `evaluation/analysis/spatial_split_verification.ipynb` |
| Model training and evaluation | Colab | `evaluation/<model>/*.ipynb` |
| Manuscript figures | Colab | `plots/paper_figures.ipynb` |
| Results site data | Local | `docs/tools/` |

The imagery fetch is the one stage that leaves the local environment and comes
back. It is hours of network-bound work per cell against a rate-limited public
API, and it was split across several people's Colab runtimes rather than run
on one machine.

---

## Setup

For curation, the results site, and opening notebooks locally:

```bash
uv venv                                 # or: python -m venv .venv
source .venv/bin/activate               # macOS and Linux
uv pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell or
`.venv\Scripts\activate.bat` in cmd, then run the same install. Nothing else
here is platform-specific: every path is built with `pathlib`, and the
repository contains no shell scripts.

`requirements.txt` covers this repository's own code only. The evaluation
notebooks install their FM-specific dependencies themselves
(`transformers`, `terratorch`, `olmoearth-pretrain-minimal`, `use_croma.py`,
and so on), because those conflict with each other and belong per-notebook.

`curation` is a package. Run its entry points as modules from the repository
root, not as file paths:

```bash
python -m curation.substations.pipeline --job maine --dry-run   # works
python curation/substations/pipeline.py --job maine --dry-run   # will not
```

Every path the curation code touches is anchored to the repository root by
[`curation/paths.py`](curation/paths.py), so the working directory does not
matter.

---

## Running the notebooks

Every notebook under `evaluation/` and `plots/` expects a Google Colab runtime
with Drive mounted. They read the dataset and write results to a fixed tree in
your Drive.

### Drive layout

Create this under `MyDrive`. The `datasets/` folder is where the Zenodo
download goes; leave the per-cell archives zipped, since the notebooks accept
either a `.zip` or an unpacked folder and unzip to local disk themselves.

```
MyDrive/infra_fm/
  datasets/                     dataset_<region>_<sector>_v1_1k.zip  (28 of them)
                                dataset_<region>_stac_v1.zip         (optional, part A)
  code/
    infra_fm_curation.zip       a zip of this repository, for the notebooks
                                that import curation.utils.spatial_blocking
  results/                      written by the evaluation notebooks
    fm_eval_<condition>/
  checkpoints/                  fine-tuning resume state
  .env                          optional, HF_TOKEN=... for gated weights
```

The split artifact and the AlphaEarth embeddings are **not** in that tree. They
are committed to this repository at `data/spatial_split/` and `data/alphaearth/`
— about 12 MB together — so they travel with the code and the Drive side stays
imagery and results only. The notebooks resolve them through `curation.paths`,
which points at the repository checkout locally and at the extracted code zip
in Colab, falling back to Drive if an older setup still stages them there.

Build `code/infra_fm_curation.zip` with `git archive`, from the repository root:

```bash
git archive --format=zip HEAD curation data/spatial_split data/alphaearth -o infra_fm_curation.zip
```

That is ~15 MB: the `curation` package the notebooks import, plus the split
artifact and the AlphaEarth embeddings they read. Naming those three paths
matters — omit the `data/` ones and every notebook silently falls back to
looking for them on Drive.

Use `git archive` rather than zipping the working directory. It takes only
committed files, so it cannot pick up `.env`, `.venv/`, `__pycache__/`, the
fetch checkpoints under `data/`, or `locations.idx`. That matters because this
zip goes into a Drive folder shared with whoever is running the fetch. Commit
first — `git archive` reads `HEAD`, not the working tree.

The notebooks extract it, put the repository root on `sys.path`, and import
`curation.utils.spatial_blocking`. They fall back to an inline copy if the zip
is missing, so a stale zip degrades rather than breaks.

`.env` on Drive holds `HF_TOKEN`, and `HF_TOKEN_OLMOEARTH` for that model. The
notebooks look in Colab Secrets first and fall back to this file. Weights
download from Hugging Face at run time, so nothing has to be pre-staged and
"no token found" only matters for the gated checkpoints. Keep `.env` on Drive,
never in the repository or the zip.

### In the browser

1. Go to <https://colab.research.google.com>, choose **GitHub**, and paste this
   repository's URL. Or upload the `.ipynb` directly.
2. Runtime → Change runtime type → pick a GPU. The fine-tuning notebooks need
   one; the linear probes will run without.
3. Run the cells top to bottom. The first mounts Drive and will ask for
   permission.

### From VS Code

Google ships an official [Colab extension](https://marketplace.visualstudio.com/items?itemName=Google.colab)
(`Google.colab`, released November 2025). Install it from the Extensions view,
open a `.ipynb`, click **Select Kernel**, choose **Colab**, and sign in. You
then edit locally with a Colab GPU behind it.

One catch that matters here: `drive.mount()` does not work reliably through the
extension yet. It is
[a known open issue](https://github.com/googlecolab/colab-vscode/issues/256) —
the mount call hangs and never shows the OAuth prompt. Since every notebook in
this repository starts by mounting Drive, **use the browser to run them** and
treat VS Code as the place to read and edit them. That will change as the
extension matures.

---

## Reproducing the paper

### Figures

Open [`plots/paper_figures.ipynb`](plots/paper_figures.ipynb) in Colab, point
`RESULTS_BASE` at your `MyDrive/infra_fm/results` tree, run the setup cells,
then run whichever figure cell you want. Each renders inline and writes a PNG
to `RESULTS_BASE/figures/`.

The 10-class re-fit cell must run before any figure cell; it patches the
loaders so every downstream figure gets 10-class numbers. Cells run in
manuscript order, and each PNG is written under the filename the paper uses,
so the output folder drops straight into the manuscript's `figures/`. See
[`plots/README.md`](plots/README.md) for the figure-to-function table.

### Evaluation

Each of the 30 conditions has its own notebook under `evaluation/<model>/`.
To rerun one:

1. Open it in Colab on a GPU runtime.
2. Confirm `MyDrive/infra_fm/datasets/` holds the 28 cells. The split
   artifact and the AlphaEarth embeddings come from the code zip, so there
   is nothing else to stage.
3. Set `SMOKE_ONLY = False` in the training-cell parameters.
4. Run all cells. Per-seed results land in `results/fm_eval_<name>/`, and the
   aggregate is written once all three seeds are present.

Fine-tuning notebooks resume from `checkpoint_final.pt` if Colab disconnects
part way through a seed.

### The results site

The only stage that reproduces end to end on a laptop, given a copy of the
results tree:

```bash
cd docs
python tools/build_results.py /path/to/results data/results.json
python tools/validate.py data/results.json
```

`validate.py` checks the generated file against every number the figure
notebook prints and exits non-zero on any mismatch. Serve the folder with
`python -m http.server 8899` to preview, since `fetch()` will not read the
JSON over `file://`. See [`docs/README.md`](docs/README.md).

### Curation

Only needed to change the taxonomy, resample, or add a region. The finished
dataset is on Zenodo.

**Where the chain really starts.** A clone can run from the deduplicated asset
tables onward, which is everything except the OSM extraction itself:

| If you want to do... | Is it possible? |
|---|---|
| Extract assets from OpenStreetMap | No, as it needs 51 GB of Geofabrik PBFs that are too large to distribute, and the 2026-05-26 snapshot will no longer be downloadable |
| Deduplicate those assets | No, as it reads the extraction above, which you would not have |
| Draw the per-cell samples (`sample_v1`) | Yes. The deduplicated tables are committed to this repository |
| Fetch the imagery | Yes, though it takes hours to days per cell against a rate-limited API |
| Cap cells at 1k, build the split, evaluate | Yes |

**You almost certainly want to skip to the last row.** The Zenodo archive
already contains the imagery, sampled and capped, and this repository already
contains the split assignment, so the whole left column is optional. Download
the dataset, and go straight to the evaluation notebooks.

Everything the paper reports is therefore auditable and rerunnable from the
sampling step onward, but the dataset is not regenerable from OpenStreetMap by
a third party. The Zenodo archive is the reference for the published imagery.

- Part B, the benchmark: [`curation/sectors/README.md`](curation/sectors/README.md)
- Part A, the substations: [`curation/substations/README.md`](curation/substations/README.md)

```bash
python -m curation.sectors.sample_v1                            # sample lists
# then the Colab fetch, per curation/sectors/README.md
python -m curation.sectors.subsample_to_1k                      # cap at 1k
```

---

## Using your own assets

The benchmark answers "which model should I use", and the usual next question is
"on my own sites". The tiling half of that is supported and takes a few lines.
The evaluation half has a real limit, described at the end of this section.

### Building tiles for your own points

`STACImageryFetcher` needs four columns: `asset_id`, `asset_type`, `lat`, `lon`.
Anything you can reduce to a table of labelled centroids works, whether that
starts as a shapefile, a GeoJSON, a CSV of coordinates, or a database query.
Polygons are fine; take the centroid.

```python
import json
import pandas as pd
from curation.stac_imagery import STACImageryFetcher
from curation.dataset import DatasetAssembler

def centroid(geom):
    if geom["type"] == "Point":
        return geom["coordinates"][1], geom["coordinates"][0]
    ring = geom["coordinates"][0]                      # outer ring
    return (sum(p[1] for p in ring) / len(ring),
            sum(p[0] for p in ring) / len(ring))

features = json.loads(open("my_assets.geojson").read())["features"]
df = pd.DataFrame([{
    "asset_id":   f"own_{i}",
    "asset_type": f["properties"].get("kind", "unknown"),
    "lat":        centroid(f["geometry"])[0],
    "lon":        centroid(f["geometry"])[1],
} for i, f in enumerate(features)])

fetcher = STACImageryFetcher(
    buffer_m        = 300,                             # 300 m -> 600 m tiles
    modalities      = ["sentinel2_ms", "sentinel1"],
    temporal_stack  = False,
    checkpoint_path = "data/checkpoints/my_assets.pkl",
)
tiles = [t for t in fetcher.fetch_all(df) if t.status == "ok"]
DatasetAssembler("data/curated_datasets/my_assets").assemble(tiles, None)
```

For a shapefile, read it with geopandas first
(`gdf = gpd.read_file("assets.shp").to_crs(4326)`, then use
`gdf.geometry.centroid.y` and `.x`). geopandas is not a dependency of this
repository, so install it yourself.

That writes the same layout every other cell uses: `images/` of `(C, H, W)`
`.npy` tiles, a `manifest.json`, and a `summary.csv`. `asset_type` is a free
string here; it only has to match the benchmark's ontology if you intend to
compare against the published per-class numbers.

Your tiles carry the same band order as the benchmark, `sentinel2_ms` then
`sentinel1`, so any loader in `evaluation/` reads them without modification.

### What you can do with those tiles

**Pick a model, then adapt it on your own labels.** Use the
[results site](https://justing0909.github.io/infra-bench-cls) to choose a
backbone for your compute and label budget, then take the matching notebook
under `evaluation/` and point its dataset loader at your folder. The training,
evaluation, and metric code all carry over. You supply the labels and the split.

**Reuse the spatial split if you are scoring on held-out sites.** It exists to
stop near-duplicate tiles leaking between train and test, which matters as soon
as your assets cluster geographically. `compute_spatial_blocks` and
`assign_blocks_to_splits` in `curation/utils/spatial_blocking.py` need only
`lat`, `lon`, and a column to stratify on, and hand back a `split` column.
`save_split_artifact` additionally wants the benchmark's own schema
(`asset_id`, `region`, `sector`, `asset_type`), so for your own data keep the
returned frame rather than writing the artifact.

### What you cannot do

**Run our trained models on your tiles.** The per-condition checkpoints were
not retained, so there is nothing to load. Every number reported here comes
from training runs whose weights no longer exist. Reproducing a condition means
retraining it, which the notebooks do end to end, and adapting one to your data
means training on labels you provide.

This is the practical limit of what the benchmark offers a practitioner. It
tells you which backbone to invest in and roughly what accuracy to expect per
class, sector, and region. It is not a pretrained infrastructure classifier you
can point at unlabelled imagery.

---

## Dataset access

The dataset is archived on Zenodo in two parts (DOIs pending, added on paper
acceptance).

**Part B — the benchmark.** 28 cells, `dataset_<region>_<sector>_v1_1k`, seven
regions by four sectors, each capped at 1,000 tiles, 18,756 tiles total. This
is what the paper evaluates and what every notebook in `evaluation/` expects.

**Part A — substations.** `dataset_<region>_stac_v1`, every deduplicated
substation per region with no cap, energy sector only. Not used by the paper.
It is the one product that is complete rather than sampled, which makes it
useful for substation-specific work on its own.

Both ship as a flat set of per-cell zip archives. Also included is the
hand-curated Maine and New Hampshire validation subset (n = 376) behind the
label-quality characterization in the paper.

Both parts are under the **Open Database License (ODbL) 1.0**, because they
derive from OpenStreetMap. Attribution when redistributing or deriving:

> *Contains information from OpenStreetMap Planet PBF (Geofabrik snapshot
> 2026-05-26). © OpenStreetMap contributors, licensed under ODbL 1.0. Imagery
> derived from modified Copernicus Sentinel-1 / Sentinel-2 data.*

The code in this repository is separately licensed under MIT.

---

## Known dataset limitations

Documented here so readers and downstream users have full context. None of
these undermine the methodology; they are reported explicitly.

**~2–3% OSM extract undercount, fixed after 2026-05-26.** The pyosmium
`idx="sparse_file_array,locations.idx"` option silently dropped some node
locations on certain PBFs, so a small fraction of assets were missed and fewer
multipolygon areas were assembled than the source contained. It surfaced in
late May 2026 when transport extraction returned 1 airport instead of 88;
`sources.py` was patched to drop the `idx` parameter. The loss is
uniform-random across regions and asset types, so class proportions and
per-region representativeness survive. The imagery shipped with the paper comes
from pre-patch extracts and is therefore mildly undercount. We chose not to
re-extract globally, because F1 is invariant to overall sample count.

**KDTree latitude bug in pre-2026-05 dedup outputs.** Earlier deduplicated
parquets used a scipy KDTree on raw (lat, lon) degree pairs with a
`threshold / 111320` conversion that only holds at the equator. The fix uses a
sklearn BallTree with the haversine metric, which is geographically correct.
Per-region impact was largest in high-latitude Asia, at −3,266 rows. The
current dedup parquets reflect the corrected logic.

**Part A imagery is sample-capped at 25,000 tiles per region.** For regions
with more than 25,000 deduped assets — Asia, North America, Europe — the
on-disk substation imagery is a random sample of the deduped parquet. This is
a compute-budget decision, not a bug. Part B is separately capped at 1,000
tiles per region and sector pair, as the paper describes.

**Imagery selection is not pinned in the released dataset.** The fetcher asks
Planetary Computer for the least-cloudy Sentinel-2 scene in a fixed
2021-01-01 to 2024-12-31 window, which is a question about the archive rather
than a reference to a specific scene. Every other stage is deterministic: OSM
extraction from a fixed snapshot, deduplication, the seeded per-cell sampling,
QC, triage, and the seeded 1,000-tile cap all reproduce exactly, so a re-run
returns the same assets, labels, class balance and train/val/test splits. What
can move is the imagery itself, if a scene in that window is reprocessed
upstream. Repeated fetches against the archive as it stands today return
byte-identical tiles; the risk is drift over months, not run-to-run variation.

Manifests written before this was addressed record no scene identifiers, so
tiles in the released dataset cannot be matched back to the imagery they came
from, and a future re-run offers no way to tell which tiles drifted. The
archived Zenodo copy is therefore the reference for the published results.
Manifests written now carry a `scenes` block naming the STAC collection, item
id, acquisition datetime and cloud cover per modality, and
`curation/refetch_from_manifest.py` rebuilds a dataset from exactly those items,
failing loudly on any that no longer resolve rather than quietly substituting a
different scene.

**Weak labels.** Classes come from OSM tags, not from inspection. The Maine and
New Hampshire cross-validation against ISO New England, HIFLD, and utility
hosting-capacity maps found two recurring error modes: substations mistagged
between distribution and transmission, and substations absent from OSM but
present in authoritative sources. That characterization is regional, not
global.

---

## Known quirks

**`Ways scanned: 0` in the extraction logs is normal.** pyosmium's
`NodeLocationsForWays` reads every node first to build a location index and
resolves ways on a second internal pass, so the way counter stays at zero
through the first pass.

**Large parquets exhaust Colab RAM.** Read only the columns you need:

```python
df = pd.read_parquet(path)[['asset_id', 'asset_type', 'lat', 'lon']].copy()
```

**Prithvi's Hugging Face loader is fragile.** It needs
`trust_remote_code=True` and `num_labels=0`. TerraTorch's scipy/dask/rapids
conflict is documented in the Prithvi notebook.

**SatlasS1 and SatlasS2 cannot share a Colab runtime.** They both use
`/content/datasets/` and race on the extract step. Run them in separate
runtimes.

**The timing log is incomplete.** `data/Infra-FM-timing-log.xlsx` is a running
record filled in as regions completed, so regions curated before a given
column existed have gaps. Columns named in `COLUMN_MAP` that the sheet lacks
are appended automatically on the next write.

---

## Related work

The closest published work on classifying critical infrastructure from
satellite imagery is Ye, Ward, De Plaen, and Koks (2025), "Big Earth Data"
(DOI: 10.1080/20964471.2025.2490408), which trains supervised Faster R-CNN
with ResNet-101 on WorldView-3 30 cm imagery over Vietnam to detect
transmission towers and poles. The contributions are complementary:

- Infra-Bench CLS is a benchmark for evaluating FMs on a fixed classification
  task; Ye et al. is a supervised object-detection study.
- Infra-Bench CLS uses freely available 10 m Sentinel imagery at global scale;
  Ye et al. requires commercial 30 cm tasking.
- Infra-Bench CLS uses weak OSM labels across 13 classes and 7 regions;
  Ye et al. uses manual annotation for 2 classes over Vietnam.

---

## Before release

Placeholders that must be resolved before the dataset or paper goes public:

- [ ] Mint the Zenodo DOIs and replace `XX.XXXX/zenodo.XXXXXXX` in the
      manuscript's Open Research section, the *Dataset access* section above,
      and the BibTeX in `docs/notes.html#cite`.
- [ ] Swap the `@misc` repository citation on the results site for the paper
      entry once it is on arXiv.
- [ ] Strip the drafting comments still in the manuscript source, including the
      licensing-strategy note in Open Research.
- [ ] Rebuild `docs/data/results.json` and rerun `docs/tools/validate.py` if any
      evaluation run changed.
- [ ] Rebuild the code zip with `git archive` after the final commit, so the
      Drive copy matches the released code. It carries the split artifact and
      the AlphaEarth embeddings, so a stale zip means stale splits.

## Citation

Paper citation will be added on posting to arXiv.

## License

Code: **MIT** (see [`LICENSE`](LICENSE)).

Dataset: **ODbL 1.0**, distributed via Zenodo (see *Dataset access* above).

## Authors

- **Justin Guthrie** (lead author) — Northeastern University, George Mason University, Enodia Inc.
- **Edward Oughton** (advisor) — George Mason University
- **Konrad Wessels** (coauthor) — George Mason University
- **Matthew Rice** (coauthor) — George Mason University
- **Isaac Corley** (coauthor) — Taylor Geospatial
