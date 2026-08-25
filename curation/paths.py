"""
paths.py
--------
Repo-root-anchored paths for every curation artifact.

Every path below is derived from this file's location, so curation code
produces the same layout no matter which directory you invoke it from.
Before this module existed the paths were CWD-relative ("../data/..."),
which meant the pipeline only worked when run from inside curation/ and
silently wrote checkpoints to curation/data/ when it wasn't.
"""

from pathlib import Path

# curation/paths.py -> curation/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / "data"

# staged curation outputs, in pipeline order
PIPELINE_DIR  = DATA_DIR / "PIPELINE"
EXTRACTED_DIR = PIPELINE_DIR / "01-extracted-assets"
DEDUPED_DIR   = PIPELINE_DIR / "02-deduped-assets"
SAMPLES_DIR   = PIPELINE_DIR / "03-v1-samples"

# imagery and its resumable state
CURATED_DIR     = DATA_DIR / "curated_datasets"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
SCHEDULES_DIR   = DATA_DIR / "schedules"
PBF_DIR         = DATA_DIR / "pbf"

# per-region curation timings and tile counts, filled in as regions complete
TIMING_LOG = DATA_DIR / "Infra-FM-timing-log.xlsx"

# Small enough to ship with the code, and every evaluation notebook needs them,
# so they live here rather than being staged separately on Drive. Because these
# resolve from __file__, the same constant points at the repo checkout locally
# and at the extracted code zip inside a Colab runtime.
SPLIT_DIR      = DATA_DIR / "spatial_split"
SPLIT_ARTIFACT = SPLIT_DIR / "asset_id_to_split_v1.parquet"

ALPHAEARTH_DIR        = DATA_DIR / "alphaearth"
ALPHAEARTH_EMBEDDINGS = ALPHAEARTH_DIR / "embeddings_2024.parquet"
