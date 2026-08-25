# Infra-Bench Classification (Infra-Bench CLS)

Code for the Infra-Bench CLS benchmark: curating a global tile dataset of
critical-infrastructure assets from OpenStreetMap and Sentinel imagery, and
evaluating Earth-observation foundation models on it.

The paper describes the benchmark, the models, and the results. This file is a
guide to the repository: where things live, how to run them, and what you need
before you start.

Results are browsable at <https://justing0909.github.io/infra-bench-cls>, with
the site source in [`docs/`](docs).

> **Note.** If you already have your own labelled assets and want to test
> foundation models on them, skip to
> [Using your own assets](#using-your-own-assets). You do not need to run any
> of the curation code.

---

## Repository structure

```
curation/               shared curation machinery
  ontology.py             class taxonomy and OSM tag matchers
  sources.py              PBF extraction with pyosmium
  deduplication.py        BallTree haversine spatial dedup
  stac_imagery.py         Planetary Computer tile fetcher
  qc.py                   tile quality control
  triage.py               rule-based confidence triage
  dataset.py              writes images/, manifest.json, summary.csv
  paths.py                every artifact path, anchored to the repo root
  visualize_tiles.py      local tile viewer
  refetch_from_manifest.py  rebuild a dataset from the scenes a manifest names
  helpers/, utils/        dataclasses, IO, spatial blocking, timing log

  sectors/              the sampled cross-sector benchmark (what the paper uses)
                          sample_v1.py, lab_fetcher.ipynb, subsample_to_1k.py
  substations/          the full unsampled substation extraction
                          pipeline.py, resync_dataset_manifest.py,
                          stac_fetch.ipynb, global_fetch_qa.ipynb

evaluation/             one folder per model, one notebook per condition
  alphaearth/  croma/  dinov3/  olmoearth/  prithvi/
  satlas_s1/   satlas_s2/  resnet18/
  analysis/             cross-model utilities, including the split builder
  README.md             per-model environments and known quirks

plots/                  paper_figures.ipynb, every manuscript figure
docs/                   the results site, plus tools/ to build and check its data
tests/                  python -m unittest discover -s tests -t .

data/
  PIPELINE/02-deduped-assets/   deduplicated asset tables, the inputs to curation
  spatial_split/                the fixed train/val/test assignment
  alphaearth/                   64-D AlphaEarth features
  maine_subset/                 geojson behind the Maine validation subset
  Infra-FM-timing-log.xlsx      per-region curation timings and tile counts

ONTOLOGY.md             full asset taxonomy and OSM tag mappings
```

Curation runs locally. Everything that needs a GPU runs in Google Colab. The
results site builds locally from the evaluation output.

---

## Setup

For curation, the results site, and opening notebooks locally:

```bash
uv venv                                 # or: python -m venv .venv
source .venv/bin/activate               # macOS and Linux
uv pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell or
`.venv\Scripts\activate.bat` in cmd, then run the same install. Nothing here is
platform-specific. Every path is built with `pathlib`, and there are no shell
scripts.

`requirements.txt` covers this repository's own code. The evaluation notebooks
install their model-specific dependencies themselves, because those conflict
with each other.

`curation` is a package. Run its entry points as modules from the repository
root, not as file paths:

```bash
python -m curation.substations.pipeline --job maine --dry-run   # works
python curation/substations/pipeline.py --job maine --dry-run   # will not
```

---

## Running the notebooks

Everything under `evaluation/` and `plots/` expects a Colab runtime with Drive
mounted, and reads the dataset from a fixed tree in your Drive.

### Drive layout

Create this under `MyDrive`. The `datasets/` folder is where the Zenodo
download goes. Leave the per-cell archives zipped, because the notebooks accept
a `.zip` or an unpacked folder and unzip to local disk themselves.

```
MyDrive/infra_fm/
  datasets/                  dataset_<region>_<sector>_v1_1k.zip  (28 of them)
  code/
    infra_fm_curation.zip    built from this repository, see below
  results/                   written by the evaluation notebooks
  checkpoints/               fine-tuning resume state
  .env                       optional, HF_TOKEN for gated weights
```

The split assignment and the AlphaEarth embeddings are not in that tree. They
live in this repository under `data/`, about 12 MB together, and travel with
the code zip. Build the zip from the repository root:

```bash
git archive --format=zip HEAD curation data/spatial_split data/alphaearth -o infra_fm_curation.zip
```

Name all three paths. Omit the `data/` ones and the notebooks fall back to
looking for them on Drive. Use `git archive` rather than zipping the working
directory, because it takes only committed files and so cannot pick up `.env`,
`.venv/`, or the fetch checkpoints under `data/`. That matters, since this zip
goes into a Drive folder shared with whoever runs the fetch. Commit first,
because `git archive` reads `HEAD`.

Weights download from Hugging Face at run time, so nothing else needs staging.
`.env` holds `HF_TOKEN`, and `HF_TOKEN_OLMOEARTH` for that model. The notebooks
check Colab Secrets first and fall back to the file.

### In the browser

1. Open <https://colab.research.google.com>, choose GitHub, and paste this
   repository's URL. Or upload the `.ipynb` directly.
2. Runtime, Change runtime type, pick a GPU. Fine-tuning needs one. Linear
   probes will run without.
3. Run the cells top to bottom. The first mounts Drive and asks for permission.

### From VS Code

Google ships an official
[Colab extension](https://marketplace.visualstudio.com/items?itemName=Google.colab)
(`Google.colab`). Install it, open a `.ipynb`, click Select Kernel, choose
Colab, and sign in.

One catch matters here. `drive.mount()` does not work reliably through the
extension yet, and it is
[a known open issue](https://github.com/googlecolab/colab-vscode/issues/256):
the call hangs without showing the OAuth prompt. Every notebook here starts by
mounting Drive, so use the browser to run them and VS Code to read and edit.

---

## Running each stage

### Evaluation

Each of the 30 conditions has its own notebook under `evaluation/<model>/`.

1. Open it in Colab on a GPU runtime.
2. Confirm `MyDrive/infra_fm/datasets/` holds the 28 cells.
3. Set `SMOKE_ONLY = False` in the training-cell parameters.
4. Run all cells. Per-seed results go to `results/fm_eval_<name>/`, and the
   aggregate is written once all three seeds are present.

Fine-tuning notebooks resume from `checkpoint_final.pt` if Colab disconnects
part way through a seed. See [`evaluation/README.md`](evaluation/README.md) for
per-model environments and quirks.

### Figures

Open [`plots/paper_figures.ipynb`](plots/paper_figures.ipynb) in Colab, point
`RESULTS_BASE` at your results tree, run the setup cells, then run any figure
cell. Each renders inline and writes a PNG under the filename the paper uses,
so the output folder drops straight into the manuscript.

Run the 10-class re-fit cell before any figure cell. It patches the loaders so
every downstream figure reports the 10 classes the paper uses rather than the
13 in the stored results. Skipping it produces figures with the right filenames
and the wrong numbers. [`plots/README.md`](plots/README.md) maps each figure to
the function that draws it.

### The results site

The one stage that runs end to end on a laptop, given a copy of the results
tree:

```bash
cd docs
python tools/build_results.py /path/to/results data/results.json
python tools/validate.py data/results.json
```

`validate.py` checks the generated file against every number the figure
notebook prints, and exits non-zero on a mismatch. To preview, serve the folder
with `python -m http.server 8899`, because `fetch()` will not read the JSON over
`file://`. See [`docs/README.md`](docs/README.md).

### Curation

Only needed to change the taxonomy, resample, or add a region. The finished
dataset is on Zenodo.

| If you want to do... | Is it possible? |
|---|---|
| Extract assets from OpenStreetMap | No, as it needs 51 GB of Geofabrik PBFs that are too large to distribute, and the 2026-05-26 snapshot will no longer be downloadable |
| Deduplicate those assets | No, as it reads the extraction above, which you would not have |
| Draw the per-cell samples (`sample_v1`) | Yes. The deduplicated tables are committed to this repository |
| Fetch the imagery | Yes, though it takes hours to days per cell against a rate-limited API |
| Cap cells at 1k, build the split, evaluate | Yes |

Most people want the last row only. The Zenodo archive already holds the
imagery, sampled and capped, and this repository already holds the split
assignment, so download the dataset and go straight to the evaluation
notebooks.

```bash
python -m curation.sectors.sample_v1                  # per-cell sample lists
# then the Colab fetch, see curation/sectors/README.md
python -m curation.sectors.subsample_to_1k            # cap each cell at 1k
```

Details in [`curation/sectors/README.md`](curation/sectors/README.md) for the
benchmark, and [`curation/substations/README.md`](curation/substations/README.md)
for the substation dataset.

---

## Using your own assets

If you have your own sites and want to know which model to use on them, the
tiling side is a few lines. The limit is at the end of this section.

### Building tiles for your own points

`STACImageryFetcher` needs four columns: `asset_id`, `asset_type`, `lat`,
`lon`. Anything you can reduce to a table of labelled centroids works, whether
it starts as a shapefile, a GeoJSON, a CSV, or a database query. Polygons are
fine, just take the centroid.

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
    buffer_m        = 300,                             # 300 m gives 600 m tiles
    modalities      = ["sentinel2_ms", "sentinel1"],
    temporal_stack  = False,
    checkpoint_path = "data/checkpoints/my_assets.pkl",
)
tiles = [t for t in fetcher.fetch_all(df) if t.status == "ok"]
DatasetAssembler("data/curated_datasets/my_assets").assemble(tiles, None)
```

For a shapefile, read it with geopandas first
(`gdf = gpd.read_file("assets.shp").to_crs(4326)`, then take
`gdf.geometry.centroid.y` and `.x`). geopandas is not a dependency here, so
install it yourself.

That writes the same layout every benchmark cell uses, with the same band
order, so any loader under `evaluation/` reads it unmodified. `asset_type` is a
free string. It only has to match the benchmark ontology if you want to compare
against the published per-class numbers.

### What to do next

Use the [results site](https://justing0909.github.io/infra-bench-cls) to pick a
backbone for your compute and label budget. Then take that model's notebook
under `evaluation/`, point its dataset loader at your folder, and train on your
own labels. The training, evaluation, and metric code all carry over.

If you are scoring on held-out sites, reuse the spatial split so nearby tiles
do not leak between train and test. `compute_spatial_blocks` and
`assign_blocks_to_splits` in
[`curation/utils/spatial_blocking.py`](curation/utils/spatial_blocking.py) need
only `lat`, `lon`, and a column to stratify on, and return a `split` column.
`save_split_artifact` also wants the benchmark's own schema, so for your own
data keep the returned frame instead of writing the artifact.

### What you cannot do

You cannot run our trained models on your tiles. The per-condition checkpoints
were not retained, so there is nothing to load. Reproducing a condition means
retraining it, which the notebooks do end to end, and adapting one to your data
means training on labels you provide.

The benchmark tells you which backbone to invest in and roughly what accuracy
to expect per class, sector, and region. It is not a pretrained infrastructure
classifier you can point at unlabelled imagery.

---

## Dataset access

The dataset is archived on Zenodo in two parts, with DOIs added on paper
acceptance.

The benchmark is 28 cells named `dataset_<region>_<sector>_v1_1k`, covering
seven regions (North America, South America, Central America, Europe, Africa,
Asia, Australia/Oceania) crossed with four sectors (energy, water, transport,
telecom). This is what the paper evaluates and what every notebook under
`evaluation/` expects.

The substation dataset is `dataset_<region>_stac_v1`, every deduplicated
substation per region with no cap, energy sector only. The paper does not use
it. It is the one product that is complete rather than sampled, which makes it
useful on its own.

Both ship as a flat set of per-cell zip archives, alongside the hand-curated
Maine and New Hampshire validation subset.

Both are under the Open Database License (ODbL) 1.0, because they derive from
OpenStreetMap. Attribution when redistributing or deriving:

> *Contains information from OpenStreetMap Planet PBF (Geofabrik snapshot
> 2026-05-26). © OpenStreetMap contributors, licensed under ODbL 1.0. Imagery
> derived from modified Copernicus Sentinel-1 and Sentinel-2 data.*

Code in this repository is separately licensed under MIT.

---

## Gotchas

`Ways scanned: 0` during extraction is normal. pyosmium reads every node first
to build a location index and resolves ways on a second internal pass, so the
way counter stays at zero through the first pass.

Large parquets exhaust Colab RAM. Read only the columns you need:

```python
df = pd.read_parquet(path)[['asset_id', 'asset_type', 'lat', 'lon']].copy()
```

SatlasS1 and SatlasS2 cannot share a Colab runtime. They both use
`/content/datasets/` and race on the extract step, so run them separately.

Prithvi's Hugging Face loader needs `trust_remote_code=True` and
`num_labels=0`. The TerraTorch dependency conflict is documented in its
notebook.

`data/Infra-FM-timing-log.xlsx` is incomplete. It was filled in as regions
completed, so regions curated before a given column existed have gaps. Columns
named in `COLUMN_MAP` that the sheet lacks are appended on the next write.

---

## Citation

Paper citation will be added on posting to arXiv.

## License

Code is MIT, see [`LICENSE`](LICENSE). The dataset is ODbL 1.0, distributed via
Zenodo.

## Authors

- Justin Guthrie (lead author), Northeastern University, George Mason University, Enodia Inc.
- Edward Oughton (advisor), George Mason University
- Konrad Wessels (coauthor), George Mason University
- Matthew Rice (coauthor), George Mason University
- Isaac Corley (coauthor), Taylor Geospatial
