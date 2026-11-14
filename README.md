# Infra-Bench Classification (Infra-Bench CLS)

This repository holds the code behind Infra-Bench CLS, a benchmark that curates
a global tile dataset of critical-infrastructure assets from OpenStreetMap and
Sentinel imagery and evaluates Earth-observation foundation models on it. The
paper describes the benchmark design, the models, and the results, so this file
covers only what is specific to the code: where each piece lives, how to run
it, and what has to be in place first. Results are browsable at
<https://justing0909.github.io/infra-bench-cls>, whose source is in
[`docs/`](docs).

---

## Citation

Below is the citation for this repository.

**Code.** [10.5281/zenodo.22256967](https://doi.org/10.5281/zenodo.22256967)

```bibtex
@software{guthrie2026infrabenchclscode,
  title     = {Infra-Bench Classification (Infra-Bench CLS) GitHub Repository},
  author    = {Guthrie, Justin},
  year      = {2026},
  version   = {1},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22256967},
  url       = {https://doi.org/10.5281/zenodo.22256967}
}
```

In addition, the citations for the accompanying paper and database are below:

**Paper.** [arXiv:2609.09482](https://arxiv.org/abs/2609.09482)

```bibtex
@misc{guthrie2026infrabenchcls,
  title         = {Infra-Bench CLS: A Global, Open-Source Benchmark for
                   Critical Infrastructure Classification with Earth
                   Observation Foundation Models},
  author        = {Guthrie, Justin and Oughton, Edward and Wessels, Konrad
                   and Rice, Matthew and Corley, Isaac},
  year          = {2026},
  eprint        = {2609.09482},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2609.09482}
}
```

**Dataset.** [10.5281/zenodo.22118892](https://doi.org/10.5281/zenodo.22118892)

```bibtex
@dataset{guthrie2026infrabenchclsdata,
  title     = {Infra-Bench CLS: A Global Multimodal Satellite Imagery Dataset
               for Critical Infrastructure Classification},
  author    = {Guthrie, Justin and Oughton, Edward and Wessels, Konrad
               and Rice, Matthew and Corley, Isaac},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22118892},
  url       = {https://doi.org/10.5281/zenodo.22118892}
}
```

---

## License

The code in this repository is released under the MIT License, see
[`LICENSE`](LICENSE).

The dataset is released separately under the Open Database License (ODbL) 1.0,
because it derives from OpenStreetMap. It is archived at
<https://doi.org/10.5281/zenodo.22118892>, and redistributing or deriving from
it requires the attribution given under [Dataset access](#dataset-access).

Imagery derives from the European Space Agency Sentinel-1 and Sentinel-2
missions. Asset locations come from OpenStreetMap contributors.

---

> **Note.** If you already have your own labelled assets and want to test
> foundation models on them, skip to
> [Using your own assets](#using-your-own-assets), since none of the curation
> code is needed for that.

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

  sectors/              the sampled cross-sector benchmark, which the paper uses
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
  PIPELINE/02-deduped-assets/   deduplicated asset tables, the curation inputs
  spatial_split/                the fixed train/val/test assignment
  alphaearth/                   64-D AlphaEarth features
  Infra-FM-timing-log.xlsx      per-region curation timings and tile counts

ONTOLOGY.md             full asset taxonomy and OSM tag mappings
```

Curation runs locally because it is CPU-bound and file-heavy, everything that
needs a GPU runs in Google Colab, and the results site builds locally from the
evaluation output.

---

## Setup

A local environment covers curation, the results site, and opening notebooks:

```bash
uv venv                                 # or: python -m venv .venv
source .venv/bin/activate               # macOS and Linux
uv pip install -r requirements.txt
```

Windows users should activate with `.venv\Scripts\Activate.ps1` in PowerShell
or `.venv\Scripts\activate.bat` in cmd and then run the same install. Nothing
here is platform-specific, as every path is built with `pathlib` and the
repository contains no shell scripts.

`requirements.txt` covers this repository's own code only, because the
evaluation notebooks install their model-specific dependencies themselves,
which conflict with each other and so cannot share one environment.

Since `curation` is a package, its entry points run as modules from the
repository root rather than as file paths:

```bash
python -m curation.substations.pipeline --job maine --dry-run   # works
python curation/substations/pipeline.py --job maine --dry-run   # will not
```

---

## Running the notebooks

Everything under `evaluation/` and `plots/` expects a Colab runtime with Drive
mounted, and reads the dataset from a fixed tree in your Drive.

### Drive layout

Create the following under `MyDrive`, where `datasets/` receives the Zenodo
download. The per-cell archives can stay zipped, since the notebooks accept
either a `.zip` or an unpacked folder and unzip to local disk themselves.

```
MyDrive/infra_fm/
  datasets/                  dataset_<region>_<sector>_v1_1k.zip  (28 of them)
  code/
    infra_fm_curation.zip    built from this repository, see below
  results/                   written by the evaluation notebooks
  checkpoints/               fine-tuning resume state
  .env                       optional, HF_TOKEN for gated weights
```

The split assignment and the AlphaEarth embeddings are deliberately absent from
that tree. Both live in this repository under `data/`, about 12 MB together, and
travel with the code zip, which is built from the repository root:

```bash
git archive --format=zip HEAD curation data/spatial_split data/alphaearth -o infra_fm_curation.zip
```

All three paths have to be named, because omitting the `data/` ones leaves the
notebooks silently falling back to Drive. Using `git archive` rather than
zipping the working directory matters too, since it takes only committed files
and therefore cannot pick up `.env`, `.venv/`, or the fetch checkpoints under
`data/`, and this zip goes into a Drive folder shared with whoever runs the
fetch. Commit before building it, as `git archive` reads `HEAD`.

Weights download from Hugging Face at run time, so nothing further needs
staging. The optional `.env` holds `HF_TOKEN`, along with `HF_TOKEN_OLMOEARTH`
for that one model, and the notebooks check Colab Secrets before falling back
to the file.

### In the browser

1. Open <https://colab.research.google.com>, choose GitHub, and paste this
   repository's URL, or upload the `.ipynb` directly.
2. Under Runtime, Change runtime type, pick a GPU. Fine-tuning requires one,
   while linear probes will run without.
3. Run the cells top to bottom. The first mounts Drive and asks for permission.

### From VS Code

Google ships an official
[Colab extension](https://marketplace.visualstudio.com/items?itemName=Google.colab)
(`Google.colab`), which after installation lets you open a `.ipynb`, click
Select Kernel, choose Colab, and sign in.

One catch matters here, which is that `drive.mount()` does not yet work
reliably through the extension. It is
[a known open issue](https://github.com/googlecolab/colab-vscode/issues/256)
in which the call hangs without ever showing the OAuth prompt, and since every
notebook here begins by mounting Drive, the browser remains the place to run
them while VS Code is the better place to read and edit.

---

## Running each stage

### Evaluation

Each of the 30 conditions has its own notebook under `evaluation/<model>/`, and
running one takes four steps:

1. Open it in Colab on a GPU runtime.
2. Confirm that `MyDrive/infra_fm/datasets/` holds the 28 cells.
3. Set `SMOKE_ONLY = False` in the training-cell parameters.
4. Run all cells. Per-seed results go to `results/fm_eval_<name>/`, and the
   aggregate is written once all three seeds are present.

Should Colab disconnect part way through a seed, the fine-tuning notebooks
resume from `checkpoint_final.pt`. Per-model environments and quirks are in
[`evaluation/README.md`](evaluation/README.md).

### Figures

Opening [`plots/paper_figures.ipynb`](plots/paper_figures.ipynb) in Colab and
pointing `RESULTS_BASE` at your results tree is enough to regenerate any
figure: run the setup cells, then run whichever figure cell you want. Each one
renders inline and writes a PNG under the filename the paper uses, so the
output folder drops straight into the manuscript.

The 10-class re-fit cell has to run before any figure cell, because it patches
the loaders so that every downstream figure reports the 10 classes the paper
uses rather than the 13 held in the stored results. Skipping it yields figures
with the right filenames and the wrong numbers.
[`plots/README.md`](plots/README.md) maps each figure to the function that
draws it.

### The results site

Given a copy of the results tree, the site is the one stage that runs end to
end on a laptop:

```bash
cd docs
python tools/build_results.py /path/to/results data/results.json
python tools/validate.py data/results.json
```

`validate.py` checks the generated file against every number the figure
notebook prints and exits non-zero on a mismatch. Previewing the site requires
serving the folder with `python -m http.server 8899`, because `fetch()` will
not read the JSON over `file://`. Further detail is in
[`docs/README.md`](docs/README.md).

### Curation

Curation is only needed to change the taxonomy, resample, or add a region,
since the finished dataset is on Zenodo. How much of it a clone can actually
run varies by stage:

| If you want to do... | Is it possible? |
|---|---|
| Extract assets from OpenStreetMap | No, as it needs 51 GB of Geofabrik PBFs that are too large to distribute, and the 2026-05-26 snapshot will no longer be downloadable |
| Deduplicate those assets | Yes. The extracted tables are committed to this repository, though only Maine is wired into `pipeline.py`, so the sector tables need `curation.deduplication.Deduplicator` called directly |
| Draw the per-cell samples (`sample_v1`) | Yes. The deduplicated tables are committed to this repository |
| Fetch the imagery | Yes, though it takes hours to days per cell against a rate-limited API |
| Cap cells at 1k, build the split, evaluate | Yes |

Most readers want the last row alone, and because the Zenodo archive already
holds the imagery, sampled and capped, while this repository already holds the
split assignment, downloading the dataset and going straight to the evaluation
notebooks skips everything above it.

```bash
python -m curation.sectors.sample_v1                  # per-cell sample lists
# then the Colab fetch, see curation/sectors/README.md
python -m curation.sectors.subsample_to_1k            # cap each cell at 1k
```

The benchmark chain is documented in
[`curation/sectors/README.md`](curation/sectors/README.md) and the substation
chain in
[`curation/substations/README.md`](curation/substations/README.md).

### Label review

Every benchmark label comes from OSM, so `curation/review/` puts a human
decision on each tile. It needs the v1_1k tiles under `data/curated_datasets/`.

```bash
python -m curation.review.build_queue                 # data/label_review/queue.csv
python -m curation.review.server --reviewer justin    # http://127.0.0.1:8765
```

The page is laid out like VIA: a tile list, the Sentinel-2 chip the models see
next to high-resolution imagery of the same footprint with the mapped OSM
outline, and one annotation row with the verdict and the correct class. Space
marks the OSM label true, a class key marks it false and relabels in one press,
U marks it unconfirmed, and O opens the asset in OpenStreetMap. The open
shortcut list is in the sidebar.

The high-resolution view is Esri World Imagery when the server has an ArcGIS
API key (`ARCGIS_API_KEY`, or one line in `data/label_review/arcgis_key.txt`,
which git ignores); a free ArcGIS Location Platform account provides one. Without
a key, US tiles show NAIP aerial imagery and other tiles show the Sentinel-2
chip alone. Google Maps and Street View are deliberately not used: their terms
forbid using them to create or augment another mapping dataset (see Imagery and
source licences below).

A true verdict accepts the OSM label, false plus a class relabels it, and
unconfirmed brings the tile back in the revisit pass. The page also shows two
weak hints: a keyword in the OSM name that points at another class, and the
disagreement of an out-of-fold AlphaEarth probe, which is right only about 30%
of the time. Decisions go to `data/label_review/decisions_<reviewer>.jsonl`,
which is the committed record. `--second-opinion` sends relabels and
unconfirmed tiles to a second reviewer instead.

Secondary sources add evidence to the queue. For US substation tiles,
`python -m curation.review.hifld_lines` (then rebuild the queue) finds the
HIFLD transmission lines that end at each asset, with their voltages and the
substation names HIFLD gives at their ends. A tile whose OSM label is
transmission and which has lines of 100 kV or more ending at it, on lines HIFLD
did not trace from OSM, appears in a "Source-confirmed" pass. That pass still
needs a decision per tile, but a quick one. Lines HIFLD traced from OSM are
flagged on the page and never count as confirmation.

`python -m curation.review.eia_plants` does the same for US energy tiles with
the EIA power plant inventory published through HIFLD. A solar farm, wind farm
or other power plant tile with an EIA plant of the matching kind within 1 km is
source-confirmed. A plant of the wrong kind, or a substation tile whose OSM
name matches a plant within 300 m, is flagged as a likely mislabel.

`python -m curation.review.registers` adds official registers for stations
and airports: North American Rail Network passenger station nodes (BTS), UK
national rail stations (DfT NaPTAN), FAA aviation facilities (BTS), Deutsche
Bahn's OpenStation stations, Trafikverket's railway station nodes (Sweden),
Ireland's NaPTAN rail stops, SNCF's passenger stations (France), Fintraffic
Digitraffic stations (Finland), SBB's Trafimage stations (Switzerland), and
Geoscience Australia's railway stations, without the entries it took from
OSM. Abandoned or disused stations never confirm a tile, and an OSM station on
one is flagged. Spain's ADIF network was considered but its service refuses
access.

The most obvious tiles pass without review, openly. `python -m
curation.review.auto_accept` accepts a tile when a source confirms it under a
strict rule that always includes a name match, for example a NaPTAN station
within 300 m with the same name. Those decisions go to
`decisions_source.jsonl` under the reviewer `source`, never over a person's.
A fixed 10% of eligible tiles is held back as an audit (`auto_accept_audit.csv`,
the "Audit of auto-accepts" pass on the page), and `--report` shows how often
people agreed with the source on those tiles.

That audit measures the source rules, which mostly cover stations and
airports. To estimate how accurate the OSM labels themselves are,
`python -m curation.review.class_sample` draws a fixed random sample of every
class: 10% of its tiles, at most 100 and at least 20 (or the whole class when
smaller), 940 tiles in all. It appears as the "Random label sample" pass, with
each tile's current OSM tags and footprint size as hints, and
`python -m curation.review.class_sample --report` gives each class's accuracy
with a 95% Clopper-Pearson interval and what the wrong labels should have been.

`curation/review/two_agent.py` adapts Ed Oughton's two-agent workflow
([Agentic-GeoAI, week 10](https://github.com/edwardoughton/Agentic-GeoAI/tree/main/week10))
to these tiles, keeping his routing rule and scoring. An investigator agent sees
the imagery (Sentinel-2, plus NAIP aerial imagery for US tiles), the OSM label,
tags and name, and the source evidence, and confirms, relabels or abstains. A
reviewer agent sees only the imagery, is never told the label, and names the
class itself. A tile gets a provisional label only when both agree, neither
abstains, both call the imagery adequate and both confidences reach 0.85;
everything else goes to a person. Each role runs in its own fresh agent session
that reads only its packet. Runs live in `data/label_review/agent_runs/`, never
touch the decision logs, and `score` reports coverage and selective error
against a person's decisions:

```bash
python -m curation.review.two_agent prepare --run pilot_v1 --ids <asset ids>
# run investigator/TASK.md and reviewer/TASK.md in two fresh agent sessions
python -m curation.review.two_agent finalize --run pilot_v1
python -m curation.review.two_agent score --run pilot_v1 --reviewer justin
```

#### Imagery and source licences

Label review rests only on sources whose terms allow deriving or validating
data. Google Maps and Street View are excluded: the Google Maps / Google Earth
Additional Terms (§2.4) forbid using Google Maps "to create or augment any
other mapping-related dataset", and Google's geo guidelines forbid "creating
data from Street View images".

| Source | Used for | Licence | Attribution |
|---|---|---|---|
| Copernicus Sentinel-2 / Sentinel-1 | the benchmark tiles | free, full and open | Contains modified Copernicus Sentinel data |
| Esri World Imagery | human review only: on the page with an ArcGIS key, or in Esri's Wayback app; never given to agents or models | Esri Master Agreement; "Esri and its imagery contributors" permit tracing and validating vector data; automated extraction is more restricted | Powered by Esri; Esri, Vantor, Earthstar Geographics |
| NAIP (USDA) | human and agent review, US tiles | public domain | USDA Farm Production and Conservation |
| OpenStreetMap (Overpass API) | tags and footprints | ODbL | © OpenStreetMap contributors |
| HIFLD transmission lines, EIA power plants | evidence, auto-accept | US government, public domain | HIFLD; U.S. Energy Information Administration |
| NARN passenger stations, FAA aviation facilities (BTS NTAD) | evidence, auto-accept | US government, public domain | U.S. DOT Bureau of Transportation Statistics |
| UK NaPTAN | evidence, auto-accept | Open Government Licence v3.0 | Contains public sector information licensed under the Open Government Licence v3.0 |
| Ireland NaPTAN | evidence, auto-accept | CC BY 4.0 | National Transport Authority |
| Deutsche Bahn OpenStation | evidence, auto-accept | CC0 | DB InfraGO (not required) |
| Trafikverket railway station nodes | evidence, auto-accept | CC0 | Trafikverket (not required) |
| SNCF gares de voyageurs | evidence, auto-accept | ODbL (share-alike; compatible with this dataset) | SNCF |
| Fintraffic Digitraffic stations | evidence, auto-accept | CC BY 4.0 | Fintraffic / digitraffic.fi |
| Geoscience Australia railway stations | evidence, auto-accept | CC BY 4.0 | © Commonwealth of Australia (Geoscience Australia) |
| Transpower sites and transmission lines (New Zealand) | substation evidence, auto-accept | CC BY 4.0 | Transpower Geospatial (CC BY) |
| Vector zone substations (Auckland) | substation evidence, auto-accept | CC BY 4.0 | Vector Limited, Our Data (CC BY 4.0) |
| EDF SEI Martinique HTA lines | substation evidence | Licence Ouverte 2.0 (Etalab) | EDF SEI Martinique, with the date of the last update |
| Geoscience Australia Electricity Infrastructure substations (OSM-sourced rows dropped) | substation evidence, auto-accept | CC BY 4.0 | © Commonwealth of Australia (Geoscience Australia) |
| Vicmap Features of Interest power facilities (Victoria) | substation evidence, auto-accept | CC BY 4.0 | © State of Victoria (Department of Transport and Planning) |
| NSW Features of Interest building complexes | substation evidence, auto-accept | CC BY 4.0 | © State of New South Wales (Spatial Services, a business unit of the Department of Customer Service NSW). For current information go to spatial.nsw.gov.au |
| KPLC primary substations (Kenya), via energydata.info | substation evidence, auto-accept | CC BY 4.0 | Kenya Power and Lighting Company / energydata.info |
| ZESCO distribution substations (Zambia), via energydata.info | substation evidence, auto-accept | CC BY 4.0 | ZESCO / energydata.info |
| Secretaría de Energía estaciones transformadoras AT (Argentina), 2023 portal snapshot | substation evidence, auto-accept | CC BY 4.0 | Secretaría de Energía, República Argentina |
| OSINERGMIN subestaciones de distribución (Peru) | substation evidence | ODC-BY 1.0 | OSINERGMIN |
| Enedis postes de distribution publique HTA/BT (France) | substation evidence | Licence Ouverte 2.0 (Etalab) | Enedis, with the date of the last update |

HIFLD and EIA cover the contiguous US, Alaska, Hawaii and Puerto Rico. Sources
considered and not used: LUMA Puerto Rico (its terms forbid copying, derivative
works and automated collection), Chile's substation layer (published as CC BY-NC,
an exception to the portal's CC0 default), Brazil's ANEEL BDGD (no licence
stated), Zambia's MV transformers (too dense to tell a substation apart), Eskom
(website terms limit use to non-commercial purposes), and a third-party copy of
Puerto Rico transmission lines (no licence; partly traced from OSM).

The Kenya, Zambia and Argentina layers have no stable download URL, so they are
read from zips saved by hand under `data/label_review/sources/grid_operators/manual/`
(not committed, like the other caches): `kenya_kplc_primary_substations.zip` and
`zambia_distribution_substations.zip` from energydata.info, and
`argentina_at_estaciones_transformadoras.zip` from a 2023 Wayback Machine snapshot
of datos.energia.gob.ar. Without them, `grid_operators.py` skips those three sources and says so.

The evidence tables under `data/label_review/sources/` carry names and values
from these registers, so they keep the dataset's ODbL licence and need the
attributions above wherever they are redistributed.

The same review also runs in VIA (VGG Image Annotator 2.x), writing the same
log:

```bash
python -m curation.review.via export --reviewer justin
# Project > Load in VIA, mark each chip's verdict, Project > Save, then:
python -m curation.review.via import --reviewer justin path/to/saved_project.json
```

Chips and VIA projects are regenerable and ignored by git.

---

## Using your own assets

Anyone holding their own sites and wondering which model to use on them can
build tiles for those sites in a few lines, though the evaluation half carries
a limit described at the end of this section.

### Building tiles for your own points

`STACImageryFetcher` needs four columns, `asset_id`, `asset_type`, `lat`, and
`lon`, so anything reducible to a table of labelled centroids works, whether it
begins as a shapefile, a GeoJSON, a CSV, or a database query. Polygons are
fine, taking the centroid.

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

A shapefile needs geopandas first, so read it with
`gdf = gpd.read_file("assets.shp").to_crs(4326)` and take
`gdf.geometry.centroid.y` and `.x`. geopandas is not a dependency here and has
to be installed separately.

The result is the same layout and band order every benchmark cell uses, which
means any loader under `evaluation/` reads it unmodified. `asset_type` is a
free string that only has to match the benchmark ontology if you intend to
compare against the published per-class numbers.

### What to do next

Picking a backbone for your compute and label budget is what the
[results site](https://justing0909.github.io/infra-bench-cls) is for. Take that
model's notebook under `evaluation/`, point its dataset loader at your folder,
and train on your own labels, since the training, evaluation, and metric code
all carry over unchanged.

Scoring on held-out sites is worth doing through the spatial split, so that
nearby tiles do not leak between train and test.
[`curation/utils/spatial_blocking.py`](curation/utils/spatial_blocking.py)
provides `compute_spatial_blocks` and `assign_blocks_to_splits`, which need only
`lat`, `lon`, and a column to stratify on, and return a `split` column.
`save_split_artifact` additionally expects the benchmark's own schema, so for
your own data it is simpler to keep the returned frame than to write the
artifact.

### What you cannot do

Running our trained models on your tiles is not possible, because the
per-condition checkpoints were not retained and there is consequently nothing
to load. Reproducing a condition means retraining it, which the notebooks do
end to end, and adapting one to your data means training on labels you supply.

What the benchmark offers, then, is guidance on which backbone to invest in and
roughly what accuracy to expect per class, sector, and region. It is not a
pretrained infrastructure classifier that can be pointed at unlabelled imagery.

---

## Dataset access

The dataset is archived on Zenodo in two parts, together under
<https://doi.org/10.5281/zenodo.22118892>.

The benchmark comprises 28 cells named `dataset_<region>_<sector>_v1_1k`,
covering seven regions (North America, South America, Central America, Europe,
Africa, Asia, and Australia/Oceania) crossed with four sectors (energy, water,
transport, and telecom). This is what the paper evaluates and what every
notebook under `evaluation/` expects.

The substation dataset, `dataset_<region>_stac_v1`, holds every deduplicated
substation per region with no cap, in the energy sector only. The paper does
not use it, but it is the one product that is complete rather than sampled,
which makes it useful on its own.

Both ship as a flat set of per-cell zip archives, alongside the hand-curated
Maine and New Hampshire validation subset, and both are released under the Open
Database License (ODbL) 1.0 because they derive from OpenStreetMap.
Redistributing or deriving from either requires the following attribution:

> *Contains information from OpenStreetMap Planet PBF (Geofabrik snapshot
> 2026-05-26). © OpenStreetMap contributors, licensed under ODbL 1.0. Imagery
> derived from modified Copernicus Sentinel-1 and Sentinel-2 data.*

Code in this repository is separately licensed under MIT.

---

## Gotchas

A progress line reading `Ways scanned: 0` during extraction is expected rather
than a fault, because pyosmium reads every node first to build a location index
and only resolves ways on a second internal pass, leaving the way counter at
zero throughout the first.

Loading a full parquet will exhaust Colab RAM on the larger regions, so read
only the columns needed:

```python
df = pd.read_parquet(path)[['asset_id', 'asset_type', 'lat', 'lon']].copy()
```

SatlasS1 and SatlasS2 cannot share a Colab runtime, as both use
`/content/datasets/` and race on the extract step, so they have to be run
separately. Prithvi's Hugging Face loader is similarly particular and needs
`trust_remote_code=True` together with `num_labels=0`, and its TerraTorch
dependency conflict is documented in that notebook.

Finally, `data/Infra-FM-timing-log.xlsx` is incomplete by construction. It was
filled in as regions completed, so any region curated before a given column
existed has gaps there, and columns named in `COLUMN_MAP` that the sheet lacks
are appended on the next write.

## Authors

- Justin Guthrie (lead author), George Mason University
- Edward Oughton (advisor), George Mason University
- Konrad Wessels (coauthor), George Mason University
- Matthew Rice (coauthor), George Mason University
- Isaac Corley (coauthor), Taylor Geospatial
