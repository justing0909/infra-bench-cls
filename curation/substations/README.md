# Substations (dataset part A)

Every deduplicated substation in a region, energy sector only, with no
per-cell cap. Output goes to `data/curated_datasets/dataset_<region>_stac_v1/`.

This part is **not** what the paper evaluates. The benchmark runs on the
sampled cross-sector cells built by [`curation/sectors/`](../sectors). Part A
exists because the substation extraction is the one product that is complete
rather than sampled, which makes it useful on its own for substation-specific
work. It is distributed alongside the benchmark on Zenodo.

Two regions are the exception. Europe has 505,951 deduplicated substations and
North America is comparable, so at the observed fetch rate of roughly half a
tile per second a full pass would take days. Both fetch from a proportional
sample that preserves the OSM class mix, named `*_substations_sampled.parquet`.
`SAMPLED_REGIONS` in `pipeline.py` records which regions this applies to.

## Contents

```
pipeline.py                 six-stage driver: extract, dedup, fetch, QC,
                            triage, assemble
resync_dataset_manifest.py  reconcile a fetched dataset against an updated
                            deduped parquet without refetching
stac_fetch.ipynb            Colab equivalent of pipeline.py for three regions,
                            written when the local machine could not hold the
                            fetch. superseded by pipeline.py for new runs
global_fetch_qa.ipynb       geographic bias, visual spot-check, and failed-tile
                            inspection across all seven regions
qa_outputs/                 CSVs written by global_fetch_qa.ipynb
```

## Running it

All commands run from the repository root.

```bash
python -m curation.substations.pipeline --job central-america --dry-run
```

The dry run stops before the imagery fetch and prints how many tiles it would
request. Drop `--dry-run` to fetch. Available jobs are the seven continental
regions plus `maine`; passing no `--job` prints the list.

Stages 1 and 2 are skipped when their output parquet already exists, and both
are committed to the repository, so a normal run starts at the imagery fetch.
Re-running extraction from source needs the Geofabrik PBFs under `data/pbf/`,
which are not committed.

A long fetch can be split across machines or sessions:

```bash
python -m curation.substations.pipeline --job asia --shard-count 8 --shard-index 0
```

Shards are cut on a Morton-order spatial sort, so each shard covers a
contiguous area and the eight outputs concatenate to the whole region. Fetch
progress checkpoints to `data/checkpoints/` every few hundred tiles, so an
interrupted run resumes rather than restarting.

## Reading the output

Each region directory holds `images/` of `.npy` tiles shaped `(C, H, W)`,
a `manifest.json` with one record per tile, a flat `summary.csv`, and a
`_SUCCESS` file carrying the run's settings. A batch driver can read
`_SUCCESS` to decide whether a region needs redoing: it records the
`filter_preset` used, so changing the preset invalidates the marker.

## Known log noise

`Ways scanned: 0` during extraction is expected. pyosmium's
`NodeLocationsForWays` reads every node first to build a location index and
only resolves ways on a second internal pass, so the way counter stays at zero
through the first pass.
