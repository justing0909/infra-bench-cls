"""
pipeline.py
-----------
single-process curation pipeline for the full substation dataset (part A).

produces data/curated_datasets/dataset_<region>_stac_v1/ -- every deduplicated
substation in a region, energy sector only, with no per-cell cap. this is NOT
the dataset the paper evaluates on; that is the sampled cross-sector benchmark
built by curation.sectors. see curation/substations/README.md.

orchestrates the full sequence:
    1. extract asset locations from GeoFabrik PBF (curation/sources.py)
    2. deduplicate spatially proximate assets (curation/deduplication.py)
    3. fetch imagery tiles via Planetary Computer (curation/stac_imagery.py)
    4. basic quality control (curation/qc.py)
    5. confidence triage (curation/triage.py)
    6. assemble the dataset (curation/dataset.py)

steps 1 and 2 are skipped when their output table already exists on disk, so a
rerun against the committed parquets goes straight to the imagery fetch.

Notes:
  - Planetary Computer STAC is the only imagery path. the previous Google Earth
    engine fallback was removed when STAC became the default; see git history
    for the deleted gee_imagery.py.
  - MODALITIES controls which imagery layers are fetched.
  - TEMPORAL_STACK enables seasonal composite stacking.
  - a _SUCCESS file with run metadata is written on completion, so an
    interrupted batch can resume without redoing finished regions.

usage (from the repo root):
    python -m curation.substations.pipeline --job central-america --dry-run
    python -m curation.substations.pipeline --job central-america
    python -m curation.substations.pipeline --job asia --shard-count 8 --shard-index 0
"""

import os
import time
import argparse
import json
from datetime import datetime
from typing import Dict, Optional, List

from ..paths import (
    EXTRACTED_DIR, DEDUPED_DIR, CURATED_DIR, CHECKPOINTS_DIR,
    SCHEDULES_DIR, PBF_DIR, TIMING_LOG,
)
from ..sources import GeoFabrikSource
from ..utils.io_utils import load_asset_table
from ..deduplication import Deduplicator
from ..qc import QualityChecker
from ..triage import RuleBasedTriager
from ..dataset import DatasetAssembler
from ..utils.timing_log_utils import (
    update_timing_log, update_modality_counts, file_size_kb
)


# ===========================================================================
# configuration — change these before each run
# ===========================================================================

# --- input ---
PBF_PATH = None

# --- output ---
OUTPUT_DIR = None

# intermediate table paths
ASSETS_CSV     = None
ASSETS_PARQUET = None
ASSETS_TABLE   = None
DEDUPED_CSV    = None
DEDUPED_PARQUET = None
DEDUPED_TABLE  = None

# --- asset filtering ---
FILTER_PRESET  = "substation"   # "substation" (recommended) or "full"
MIN_CONFIDENCE = "medium"
MAX_ASSETS     = None
SAMPLE_PER_TYPE = None

# --- imagery: STAC (primary) ---
USE_STAC       = True
MODALITIES     = ["sentinel2_ms", "sentinel1"]   # add "landsat_thermal", "naip" as needed
TEMPORAL_STACK = False      # set True to fetch seasonal stacks (T, C, H, W)
N_YEARS        = 2          # number of years of seasonal composites if temporal
BUFFER_M       = 300
SOURCES        = ["sentinel2"]   # legacy field — kept for dry-run display
MAX_WORKERS    = 4

# --- qC thresholds ---
MIN_VALID_RATIO = 0.80

# --- deduplication ---
DISTANCE_THRESHOLD_M = 200

# --- triage ---
CONTRADICTION_THRESHOLD = 3
LOW_THRESHOLD           = 4


# ===========================================================================
# job PRESETS
# ===========================================================================
# one preset per region, all paths anchored to the repo root by curation.paths
# so a job resolves identically whatever directory you launch from.

# the geofabrik snapshot each region was extracted from. only consulted for a
# from-scratch re-extract -- a run that finds an existing assets table on disk
# skips step 1 and never touches the PBF.
PBF_NAMES = {
    "north-america":     "north-america-latest.osm_power_only.osm.pbf",
    "europe":            "europe-latest.osm_power_only.osm.pbf",
    "central-america":   "central-america-latest.osm_power_only.osm.pbf",
    "africa":            "africa-260408.osm_power_only.osm.pbf",
    "australia-oceania": "australia-oceania-260408.osm_power_only.osm.pbf",
    "asia":              "asia-260408.osm_power_only.osm.pbf",
    "south-america":     "south-america-260410.osm_power_only.osm.pbf",
    "maine":             "maine-latest.osm_power_only.osm.pbf",
}

# regions whose deduplicated extraction was too large to fetch imagery for at
# the observed ~0.5 tiles/s, so imagery was fetched from a proportional sample
# instead. Europe alone has 505,951 deduplicated substations, roughly 88 hours
# of fetching; the sample preserves the OSM class mix and drops it to ~127k.
SAMPLED_REGIONS = {"europe", "north-america"}


def _job_preset(region: str) -> dict:
    """build one region's path set. see PBF_NAMES and SAMPLED_REGIONS above."""
    dedup_stem = f"{region}_deduped_assets_substations"
    if region in SAMPLED_REGIONS:
        dedup_stem += "_sampled"
    return {
        "pbf_path":        str(PBF_DIR / "power_only" / PBF_NAMES[region]),
        "output_dir":      str(CURATED_DIR / f"dataset_{region}_stac_v1"),
        "assets_table":    str(EXTRACTED_DIR / f"{region}_all_assets_substations.parquet"),
        "assets_csv":      str(EXTRACTED_DIR / f"{region}_all_assets_substations.csv"),
        "assets_parquet":  str(EXTRACTED_DIR / f"{region}_all_assets_substations.parquet"),
        "deduped_table":   str(DEDUPED_DIR / f"{dedup_stem}.parquet"),
        "deduped_csv":     str(DEDUPED_DIR / f"{region}_deduped_assets.csv"),
        "deduped_parquet": str(DEDUPED_DIR / f"{dedup_stem}.parquet"),
    }


JOB_PRESETS = {region: _job_preset(region) for region in PBF_NAMES}


DEFAULT_CONFIG = {
    "pbf_path":           PBF_PATH,
    "output_dir":         OUTPUT_DIR,
    "assets_csv":         ASSETS_CSV,
    "assets_parquet":     ASSETS_PARQUET,
    "assets_table":       ASSETS_TABLE,
    "deduped_csv":        DEDUPED_CSV,
    "deduped_parquet":    DEDUPED_PARQUET,
    "deduped_table":      DEDUPED_TABLE,
    "filter_preset":      FILTER_PRESET,
    "min_confidence":     MIN_CONFIDENCE,
    "max_assets":         MAX_ASSETS,
    "sample_per_type":    SAMPLE_PER_TYPE,
    "buffer_m":           BUFFER_M,
    "sources":            list(SOURCES),
    "max_workers":        MAX_WORKERS,
    "use_stac":           USE_STAC,
    "modalities":         list(MODALITIES),
    "temporal_stack":     TEMPORAL_STACK,
    "n_years":            N_YEARS,
    "min_valid_ratio":    MIN_VALID_RATIO,
    "distance_threshold_m": DISTANCE_THRESHOLD_M,
    "contradiction_threshold": CONTRADICTION_THRESHOLD,
    "low_threshold":      LOW_THRESHOLD,
    "asset_types":        None,
    "shard_count":        1,
    "shard_index":        0,
    "shard_strategy":     "spatial",
    "schedule_name":      "pipeline_run",
    "schedule_dir":       str(SCHEDULES_DIR),
}


# ===========================================================================
# helpers
# ===========================================================================

def _parse_csv_arg(value: Optional[str]) -> Optional[list]:
    if value is None:
        return None
    parts = [item.strip() for item in value.split(",")]
    parts = [item for item in parts if item]
    return parts or None


def _interleave_bits(x: int, y: int) -> int:
    result = 0
    for i in range(32):
        result |= ((x >> i) & 1) << (2 * i)
        result |= ((y >> i) & 1) << (2 * i + 1)
    return result


def _spatial_sort_key(lat: float, lon: float) -> int:
    lat_norm = min(max((lat + 90.0) / 180.0, 0.0), 1.0)
    lon_norm = min(max((lon + 180.0) / 360.0, 0.0), 1.0)
    scale = (1 << 16) - 1
    lat_i = int(round(lat_norm * scale))
    lon_i = int(round(lon_norm * scale))
    return _interleave_bits(lat_i, lon_i)


def _with_suffix(path: str, suffix: str) -> str:
    if not suffix:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}{suffix}{ext}"


def _apply_shard(df, shard_index, shard_count, shard_strategy):
    if shard_count <= 1:
        return df.reset_index(drop=True)
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"shard_index must be in [0, {shard_count - 1}]")
    shard_df = df.copy()
    if shard_strategy == "spatial":
        shard_df["_shard_key"] = [
            _spatial_sort_key(lat, lon)
            for lat, lon in zip(shard_df["lat"], shard_df["lon"])
        ]
        shard_df = shard_df.sort_values(["_shard_key", "asset_id"]).reset_index(drop=True)
    else:
        shard_df = shard_df.sort_values(["asset_id"]).reset_index(drop=True)
    base_size = len(shard_df) // shard_count
    remainder = len(shard_df) % shard_count
    start = shard_index * base_size + min(shard_index, remainder)
    stop  = start + base_size + (1 if shard_index < remainder else 0)
    shard_df = shard_df.iloc[start:stop].copy()
    shard_df = shard_df.drop(columns=["_shard_key"], errors="ignore")
    return shard_df.reset_index(drop=True)


def _build_sorted_assignment(ordered_df, shard_count):
    ordered_df  = ordered_df.reset_index(drop=True)
    base_size   = len(ordered_df) // shard_count
    remainder   = len(ordered_df) % shard_count
    asset_to_shard = {}
    for shard_index in range(shard_count):
        start = shard_index * base_size + min(shard_index, remainder)
        stop  = start + base_size + (1 if shard_index < remainder else 0)
        for asset_id in ordered_df.iloc[start:stop]["asset_id"].tolist():
            asset_to_shard[asset_id] = shard_index
    return asset_to_shard


def _build_shard_assignment(df, shard_count, shard_strategy):
    if shard_count <= 1:
        asset_to_shard = {aid: 0 for aid in df["asset_id"].tolist()}
        return {
            "asset_to_shard": asset_to_shard,
            "metrics": None,
            "summary_lines": [],
            "schedule": None,
        }

    if shard_strategy == "spatial":
        ordered_df = df.copy()
        ordered_df["_shard_key"] = [
            _spatial_sort_key(lat, lon)
            for lat, lon in zip(ordered_df["lat"], ordered_df["lon"])
        ]
        ordered_df = ordered_df.sort_values(
            ["_shard_key", "asset_id"]
        ).reset_index(drop=True)
    else:
        ordered_df = df.sort_values(["asset_id"]).reset_index(drop=True)

    asset_to_shard = _build_sorted_assignment(ordered_df, shard_count)

    return {
        "asset_to_shard": asset_to_shard,
        "metrics": None,
        "summary_lines": [],
        "schedule": None,
    }


def _save_shard_artifacts(df, config, shard_plan):
    os.makedirs(config["schedule_dir"], exist_ok=True)
    base_name   = config["schedule_name"]
    shard_count = config["shard_count"]
    strategy    = config["shard_strategy"]
    stem        = f"{base_name}_{strategy}_shards-{shard_count:02d}"

    assignment_path = os.path.join(config["schedule_dir"],
                                   f"{stem}_assignments.csv")
    summary_path    = os.path.join(config["schedule_dir"],
                                   f"{stem}_summary.json")

    assignments = df[["asset_id", "asset_type", "lat", "lon"]].copy()
    assignments["shard_index"] = assignments["asset_id"].map(
        shard_plan["asset_to_shard"]
    )
    assignments.sort_values(
        ["shard_index", "asset_type", "asset_id"]
    ).reset_index(drop=True).to_csv(assignment_path, index=False)

    summary = {
        "strategy":    strategy,
        "shard_count": shard_count,
        "schedule_name": base_name,
    }
    if shard_plan.get("metrics"):
        summary["metrics"] = shard_plan["metrics"].to_dict()
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved shard assignments: {assignment_path}")
    print(f"  Saved shard summary:     {summary_path}")


def _write_success(output_dir: str, metadata: dict) -> None:
    """
    writes a _SUCCESS file with run metadata to the dataset output directory.
    a batch driver can read this to skip regions that are already done.
    """
    path = os.path.join(output_dir, "_SUCCESS")
    with open(path, "w") as f:
        json.dump({
            "completed_at": datetime.utcnow().isoformat() + "Z",
            **metadata,
        }, f, indent=2)


def _build_runtime_config(args: Optional[argparse.Namespace]) -> dict:
    config = dict(DEFAULT_CONFIG)
    config["sources"]    = list(DEFAULT_CONFIG["sources"])
    config["modalities"] = list(DEFAULT_CONFIG["modalities"])

    if args is None:
        return config

    if args.job:
        config.update(JOB_PRESETS[args.job])

    for key, arg_name in [
        ("pbf_path",        "pbf_path"),
        ("output_dir",      "output_dir"),
        ("assets_csv",      "assets_csv"),
        ("assets_parquet",  "assets_parquet"),
        ("assets_table",    "assets_table"),
        ("deduped_csv",     "deduped_csv"),
        ("deduped_parquet", "deduped_parquet"),
        ("deduped_table",   "deduped_table"),
    ]:
        value = getattr(args, arg_name, None)
        if value:
            config[key] = value

    if not config.get("assets_parquet") and config.get("assets_csv"):
        config["assets_parquet"] = config["assets_csv"].replace(".csv", ".parquet")
    if not config.get("deduped_parquet") and config.get("deduped_csv"):
        config["deduped_parquet"] = config["deduped_csv"].replace(".csv", ".parquet")
    if not config.get("assets_table"):
        config["assets_table"] = config.get("assets_parquet") or config.get("assets_csv")
    if not config.get("deduped_table"):
        config["deduped_table"] = config.get("deduped_parquet") or config.get("deduped_csv")

    if not config.get("output_dir"):
        raise ValueError(
            "output_dir must be provided via --job or --output-dir. "
            f"Available jobs: {', '.join(sorted(JOB_PRESETS))}"
        )

    if not config.get("assets_table") and not config.get("pbf_path"):
        raise ValueError("pbf_path is required when no assets table is provided")

    config["schedule_name"] = os.path.basename(config["output_dir"])

    if args.filter_preset:
        config["filter_preset"] = args.filter_preset
    if args.min_confidence:
        config["min_confidence"] = args.min_confidence
    if args.max_assets is not None:
        config["max_assets"] = args.max_assets
    if args.sample_per_type is not None:
        config["sample_per_type"] = args.sample_per_type
    if args.sources:
        config["sources"] = _parse_csv_arg(args.sources)
    if args.modalities:
        config["modalities"] = _parse_csv_arg(args.modalities)
    if args.use_stac is not None:
        config["use_stac"] = args.use_stac
    if args.temporal_stack is not None:
        config["temporal_stack"] = args.temporal_stack

    config["asset_types"]   = _parse_csv_arg(args.asset_types)
    config["shard_count"]   = args.shard_count
    config["shard_index"]   = args.shard_index
    config["shard_strategy"] = args.shard_strategy

    if config["shard_count"] > 1:
        shard_suffix = (
            f"_shard-{config['shard_index'] + 1:02d}"
            f"-of-{config['shard_count']:02d}"
        )
        config["output_dir"] = _with_suffix(config["output_dir"], shard_suffix)

    return config


def _print_run_plan(config: dict) -> None:
    print("\nRun plan:")
    print(f"  Filter preset:  {config.get('filter_preset', 'substation')}")
    print(f"  PBF:            {config['pbf_path']}")
    print(f"  Assets file:    {config['assets_table']}")
    print(f"  Deduped file:   {config['deduped_table']}")
    print(f"  Output dir:     {config['output_dir']}")
    print(f"  Use STAC:       {config.get('use_stac', True)}")
    if config.get("use_stac"):
        print(f"  Modalities:     {config.get('modalities', [])}")
        print(f"  Temporal stack: {config.get('temporal_stack', False)}")
        if config.get("temporal_stack"):
            print(f"  N years:        {config.get('n_years', 2)}")
    if config.get("asset_types"):
        print(f"  Asset types:    {config['asset_types']}")
    if config.get("sample_per_type") is not None:
        print(f"  Sample/type:    {config['sample_per_type']}")
    if config.get("max_assets") is not None:
        print(f"  Max assets:     {config['max_assets']}")
    if config.get("shard_count", 1) > 1:
        print(
            f"  Shard:          {config['shard_index'] + 1}"
            f"/{config['shard_count']} ({config['shard_strategy']})"
        )


# ===========================================================================
# pipeline runner
# ===========================================================================

def run_pipeline(
    dry_run : bool = False,
    config  : Optional[dict] = None,
) -> dict:
    """
    runs the full curation pipeline end to end.
    returns a dict of pipeline results and stage counts.
    """
    config = dict(DEFAULT_CONFIG if config is None else config)
    config["sources"]    = list(config.get("sources", []))
    config["modalities"] = list(config.get("modalities", ["sentinel2_ms"]))

    start_time    = time.time()
    results       = {}
    stage_timings = {}

    print("=" * 60)
    print("INFRASTRUCTURE IMAGERY CURATION PIPELINE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    _print_run_plan(config)

    # ------------------------------------------------------------------
    # step 1: Extract assets
    # ------------------------------------------------------------------
    stage_start = time.time()
    print("\n[1/6] Extracting assets from GeoFabrik PBF...")

    # once a deduplicated table exists, step 2 loads that and nothing reads the
    # extraction result, so re-deriving it from a multi-gigabyte PBF is wasted
    # work -- and a hard failure for anyone holding the tables but not the PBFs.
    deduped_ready = bool(config.get("deduped_table")) and \
        os.path.exists(config["deduped_table"])

    df = None
    if config.get("assets_table") and os.path.exists(config["assets_table"]):
        print(f"  Loading existing asset table: {config['assets_table']}")
        df = load_asset_table(config["assets_table"])
        print(f"  Loaded {len(df)} assets")
    elif deduped_ready:
        print("  Skipped: the deduplicated table already exists, so the "
              "extraction that would feed it is unused.")
    else:
        src = GeoFabrikSource(
            config["pbf_path"],
            min_confidence=config["min_confidence"],
            filter_preset=config.get("filter_preset", "substation"),
        )
        df = src.extract_all()

        write_df = df.drop(columns=["osm_tags"], errors="ignore").copy()
        if config.get("assets_csv"):
            os.makedirs(os.path.dirname(config["assets_csv"]) or ".", exist_ok=True)
            write_df.to_csv(config["assets_csv"], index=False)
        if config.get("assets_parquet"):
            os.makedirs(os.path.dirname(config["assets_parquet"]) or ".", exist_ok=True)
            write_df.to_parquet(config["assets_parquet"], index=False)
        print(f"  Saved {len(df)} assets")

    results["n_extracted"] = len(df) if df is not None else None
    stage_timings["extract_assets"] = round(time.time() - stage_start, 2)
    if df is not None:
        print("  Asset counts:")
        for asset_type, count in df["asset_type"].value_counts().items():
            print(f"    {asset_type}: {count}")

    # ------------------------------------------------------------------
    # step 2: Deduplicate
    # ------------------------------------------------------------------
    stage_start = time.time()
    print("\n[2/6] Deduplicating assets...")

    if config.get("deduped_table") and os.path.exists(config["deduped_table"]):
        print(f"  Loading existing deduplicated table: {config['deduped_table']}")
        df_clean = load_asset_table(config["deduped_table"])
        print(f"  Loaded {len(df_clean)} deduplicated assets")
    else:
        dedup = Deduplicator(distance_threshold_m=config["distance_threshold_m"])
        df_clean, _ = dedup.run(df)
        if config.get("deduped_csv"):
            os.makedirs(os.path.dirname(config["deduped_csv"]) or ".", exist_ok=True)
            df_clean.to_csv(config["deduped_csv"], index=False)
        if config.get("deduped_parquet"):
            os.makedirs(os.path.dirname(config["deduped_parquet"]) or ".", exist_ok=True)
            df_clean.to_parquet(config["deduped_parquet"], index=False)
        print(f"  Saved {len(df_clean)} deduplicated assets")

    results["n_after_dedup"] = len(df_clean)

    if config.get("asset_types"):
        before = len(df_clean)
        df_clean = df_clean[df_clean["asset_type"].isin(config["asset_types"])].copy()
        print(f"  Filtered asset types: {before} -> {len(df_clean)}")

    if config.get("sample_per_type") is not None:
        df_clean = (
            df_clean.groupby("asset_type", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), config["sample_per_type"]),
                                       random_state=42))
            .reset_index(drop=True)
        )
        print(f"  Sampled {len(df_clean)} assets ({config['sample_per_type']} per type)")

    if config.get("max_assets") is not None:
        df_clean = df_clean.head(config["max_assets"])
        print(f"  Capped at {len(df_clean)} assets")

    if config.get("shard_count", 1) > 1:
        shard_stage_start = time.time()
        shard_plan = _build_shard_assignment(
            df_clean,
            shard_count=config["shard_count"],
            shard_strategy=config["shard_strategy"],
        )
        if shard_plan["summary_lines"]:
            print("  Shard planning summary:")
            for line in shard_plan["summary_lines"]:
                print(line)
        _save_shard_artifacts(df_clean, config, shard_plan)
        selected_ids = {
            aid for aid, shard in shard_plan["asset_to_shard"].items()
            if shard == config["shard_index"]
        }
        df_clean = df_clean[df_clean["asset_id"].isin(selected_ids)].copy()
        df_clean = df_clean.sort_values(["asset_id"]).reset_index(drop=True)
        print(f"  Shard kept {len(df_clean)} assets "
              f"({config['shard_index'] + 1}/{config['shard_count']})")
        stage_timings["plan_shards"] = round(time.time() - shard_stage_start, 2)
    else:
        stage_timings["plan_shards"] = 0.0

    results["n_in_job"] = len(df_clean)
    stage_timings["deduplicate"] = round(time.time() - stage_start, 2)

    if len(df_clean) == 0:
        print("\nNo assets remain after filtering. Nothing to do.")
        results["elapsed_s"] = round(time.time() - start_time, 1)
        return results

    if dry_run:
        print("\n[DRY RUN] Stopping before imagery fetch.")
        print(f"  Would fetch tiles for {len(df_clean)} assets")
        print(f"  Modalities: {config['modalities']}")
        print(f"  Temporal stack: {config.get('temporal_stack', False)}")
        results["timings_s"] = stage_timings
        results["elapsed_s"] = round(time.time() - start_time, 1)
        return results

    # ------------------------------------------------------------------
    # step 3: Fetch imagery
    # ------------------------------------------------------------------
    stage_start = time.time()
    print(f"\n[3/6] Fetching imagery tiles ({len(df_clean)} assets)...")

    checkpoint_path = str(
        CHECKPOINTS_DIR / f"{os.path.basename(config['output_dir'])}_fetch.pkl"
    )

    if config.get("use_stac", True):
        from ..stac_imagery import STACImageryFetcher

        fetcher = STACImageryFetcher(
            buffer_m             = config.get("buffer_m", BUFFER_M),
            modalities           = config["modalities"],
            temporal_stack       = config.get("temporal_stack", False),
            n_years              = config.get("n_years", N_YEARS),
            checkpoint_path      = checkpoint_path,
            adaptive_concurrency = True,
            start_workers        = 16,
            max_workers          = 64,
        )
        tiles = fetcher.fetch_all(df_clean)

    else:
        raise ValueError(
            "No imagery fetcher enabled. Set USE_STAC=True in pipeline config."
        )

    n_ok   = sum(1 for t in tiles if t.status == "ok")
    n_fail = sum(1 for t in tiles if t.status != "ok")
    results["n_tiles_fetched"] = n_ok
    results["n_tiles_failed"]  = n_fail
    stage_timings["fetch_imagery"] = round(time.time() - stage_start, 2)
    print(f"  Fetched: {n_ok} ok, {n_fail} failed")

    # ------------------------------------------------------------------
    # step 4: Quality control
    # ------------------------------------------------------------------
    stage_start = time.time()
    print("\n[4/6] Running quality control...")

    checker    = QualityChecker(min_valid_ratio=config["min_valid_ratio"])
    qc_results = checker.check_all(tiles, max_workers=config["max_workers"])
    clean      = checker.filter_ok(qc_results)

    n_qc_pass = len(clean)
    n_qc_fail = len(tiles) - n_qc_pass
    results["n_qc_passed"] = n_qc_pass
    results["n_qc_failed"] = n_qc_fail
    stage_timings["quality_control"] = round(time.time() - stage_start, 2)
    print(f"  QC passed: {n_qc_pass}, failed: {n_qc_fail}")

    # ------------------------------------------------------------------
    # step 5: Triage
    # ------------------------------------------------------------------
    stage_start = time.time()
    print("\n[5/6] Running confidence triage...")

    triager = RuleBasedTriager(
        contradiction_threshold=config["contradiction_threshold"],
        low_threshold=config["low_threshold"],
    )
    triage_results = triager.triage_all(clean, max_workers=config["max_workers"])
    accepted       = triager.filter_accepted(triage_results)
    flagged        = triager.filter_review(triage_results)

    results["n_accepted"] = len(accepted)
    results["n_flagged"]  = len(flagged)
    results["n_rejected"] = n_qc_pass - len(accepted) - len(flagged)
    stage_timings["triage"] = round(time.time() - stage_start, 2)
    print(f"  Accepted: {len(accepted)}, flagged: {len(flagged)}, "
          f"rejected: {results['n_rejected']}")

    # ------------------------------------------------------------------
    # step 6: Assemble dataset
    # ------------------------------------------------------------------
    stage_start = time.time()
    print(f"\n[6/6] Assembling dataset -> {config['output_dir']}...")

    assembler = DatasetAssembler(config["output_dir"])
    summary   = assembler.assemble(accepted, triage_results)

    results["n_dataset_tiles"] = len(summary)
    results["output_dir"]      = config["output_dir"]
    results["elapsed_s"]       = round(time.time() - start_time, 1)
    stage_timings["assemble_dataset"] = round(time.time() - stage_start, 2)
    results["timings_s"] = stage_timings

    # ------------------------------------------------------------------
    # print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Filter preset:   {config.get('filter_preset', 'substation')}")
    print(f"  Modalities:      {config.get('modalities', [])}")
    print(f"  Temporal stack:  {config.get('temporal_stack', False)}")
    print(f"  Extracted:       {results['n_extracted']:>6} assets")
    print(f"  After dedup:     {results['n_after_dedup']:>6} assets")
    print(f"  In this job:     {results['n_in_job']:>6} assets")
    print(f"  Tiles fetched:   {results['n_tiles_fetched']:>6}")
    print(f"  QC passed:       {results['n_qc_passed']:>6}")
    print(f"  Triage accepted: {results['n_accepted']:>6}")
    print(f"  Triage flagged:  {results['n_flagged']:>6}")
    print(f"  Dataset tiles:   {results['n_dataset_tiles']:>6}")
    print(f"  Elapsed:         {results['elapsed_s']}s")
    print(f"  Output:          {config['output_dir']}")
    print("  Stage timings:")
    for stage_name, seconds in stage_timings.items():
        print(f"    {stage_name}: {seconds:.2f}s")

    if results["n_tiles_fetched"] > 0:
        yield_rate = results["n_dataset_tiles"] / results["n_tiles_fetched"] * 100
        print(f"  Pipeline yield:  {yield_rate:.1f}% of fetched tiles")

    # ------------------------------------------------------------------
    # write _SUCCESS with run metadata
    # ------------------------------------------------------------------
    try:
        success_meta = {
            "filter_preset":    config.get("filter_preset", "substation"),
            "modalities":       config.get("modalities", []),
            "temporal_stack":   config.get("temporal_stack", False),
            "n_dataset_tiles":  results["n_dataset_tiles"],
            "n_extracted":      results["n_extracted"],
            "n_after_dedup":    results["n_after_dedup"],
            "elapsed_s":        results["elapsed_s"],
        }
        _write_success(config["output_dir"], success_meta)
        print(f"  Wrote _SUCCESS to {config['output_dir']}")
    except Exception as e:
        print(f"  Warning: could not write _SUCCESS: {e}")

    # ------------------------------------------------------------------
    # update timing log
    # ------------------------------------------------------------------
    try:
        log_path = str(TIMING_LOG)

        fetcher_total  = results["n_tiles_fetched"] + results["n_tiles_failed"]
        stac_accept_pct = round(
            results["n_tiles_fetched"] / fetcher_total * 100, 2
        ) if fetcher_total else None

        qc_total    = results["n_qc_passed"] + results["n_qc_failed"]
        qc_accept_pct = round(
            results["n_qc_passed"] / qc_total * 100, 2
        ) if qc_total else None

        triage_accept_pct = round(
            results["n_accepted"] / results["n_qc_passed"] * 100, 2
        ) if results["n_qc_passed"] > 0 else None

        region_name = (
            os.path.basename(config["output_dir"])
            .replace("dataset_", "")
            .replace("_stac_v1", "")
            .replace("_sentinel_v1", "")
            .replace("_v1", "")
        )

        update_timing_log(
            workbook_path     = log_path,
            region            = region_name,
            scanning_time_s   = stage_timings.get("extract_assets"),
            stac_accept_pct   = stac_accept_pct,
            qc_accept_pct     = qc_accept_pct,
            triage_accept_pct = triage_accept_pct,
            assets_extracted  = results.get("n_extracted"),
            assets_after_dedup= results.get("n_after_dedup"),
            total_tiles_fetched= results.get("n_tiles_fetched"),
            dataset_tiles     = results.get("n_dataset_tiles"),
            total_time_elapsed_s = results.get("elapsed_s"),
            collapsed_file_size_kb = file_size_kb(config.get("assets_table")),
            filter_preset     = config.get("filter_preset", "substation"),
            modalities        = "+".join(config.get("modalities", [])),
            temporal_stack    = str(config.get("temporal_stack", False)),
        )

        # per-modality tile counts from manifest
        try:
            manifest = assembler.load_manifest()
            mod_counts = manifest.get("modality_counts", {})
            n_temporal = sum(
                1 for r in manifest.get("records", [])
                if r.get("temporal_file")
            )
            update_modality_counts(log_path, region_name, mod_counts, n_temporal)
        except Exception:
            pass

        print(f"  Updated timing log: {log_path}")
    except Exception as e:
        print(f"  Warning: could not update timing log: {e}")

    return results


# ===========================================================================
# entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infrastructure curation pipeline")

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--job", choices=sorted(JOB_PRESETS))
    parser.add_argument("--pbf-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--assets-csv")
    parser.add_argument("--assets-parquet")
    parser.add_argument("--assets-table")
    parser.add_argument("--deduped-csv")
    parser.add_argument("--deduped-parquet")
    parser.add_argument("--deduped-table")
    parser.add_argument("--asset-types",
                        help="Comma-separated asset types to keep")
    parser.add_argument("--filter-preset", choices=["full", "substation"],
                        help="Asset filter preset (default: substation)")
    parser.add_argument("--sources",
                        help="Comma-separated imagery sources (legacy, kept for compat)")
    parser.add_argument("--modalities",
                        help="Comma-separated modalities: sentinel2_ms,sentinel1,landsat_thermal,naip")
    parser.add_argument("--use-stac",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Use Planetary Computer STAC (default: True)")
    parser.add_argument("--temporal-stack",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Fetch seasonal temporal stacks")
    parser.add_argument("--min-confidence", choices=["high", "medium", "low"])
    parser.add_argument("--max-assets", type=int)
    parser.add_argument("--sample-per-type", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-strategy",
                        choices=["spatial", "asset_id"],
                        default="spatial")

    args = parser.parse_args()
    run_pipeline(dry_run=args.dry_run, config=_build_runtime_config(args))