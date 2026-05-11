"""
pipeline.py
-----------
End-to-end infrastructure imagery curation pipeline.

Orchestrates the full sequence:
    1. Extract asset locations from GeoFabrik PBF (sources.py)
    2. Deduplicate spatially proximate assets (deduplication.py)
    3. Fetch imagery tiles — STAC (stac_imagery.py) or GEE fallback (gee_imagery.py)
    4. Basic quality control (qc.py)
    5. Confidence triage (triage.py)
    6. Assemble training dataset (dataset.py)

Key changes from previous version:
  - STAC (Planetary Computer) is now the primary imagery path (USE_STAC=True)
  - GEE remains available as a fallback (USE_GEE=True, USE_STAC=False)
  - solar_collapse import removed — substation-only scope makes it unnecessary
  - filter_preset="substation" is the new default in GeoFabrikSource calls
  - MODALITIES controls which imagery layers are fetched (multimodal)
  - TEMPORAL_STACK enables seasonal composite stacking
  - Dataset output dirs use _stac_v1 suffix for STAC runs
  - _SUCCESS files written with run metadata for resumability

Usage:
    python pipeline.py --dry-run
    python pipeline.py --job central-america --dry-run
    python pipeline.py --job central-america --shard-count 8 --shard-index 0
"""

import os
import time
import argparse
import json
from datetime import datetime
from typing import Dict, Optional, List
import pandas as pd
from sources import GeoFabrikSource
from utils.io_utils import load_asset_table
from deduplication import Deduplicator
from qc import QualityChecker
from triage import RuleBasedTriager
from dataset import DatasetAssembler, dataset_output_dir
from utils.timing_log_utils import (
    update_timing_log, update_modality_counts, file_size_mb
)


# ===========================================================================
# CONFIGURATION — change these before each run
# ===========================================================================

# --- Input ---
PBF_PATH = None

# --- Output ---
OUTPUT_DIR = None

# Intermediate table paths
ASSETS_CSV     = None
ASSETS_PARQUET = None
ASSETS_TABLE   = None
DEDUPED_CSV    = None
DEDUPED_PARQUET = None
DEDUPED_TABLE  = None

# --- Asset filtering ---
FILTER_PRESET  = "substation"   # "substation" (recommended) or "full"
MIN_CONFIDENCE = "medium"
MAX_ASSETS     = None
SAMPLE_PER_TYPE = None

# --- Imagery: STAC (primary) ---
USE_STAC       = True
MODALITIES     = ["sentinel2_ms", "sentinel1"]   # add "landsat_thermal", "naip" as needed
TEMPORAL_STACK = False      # set True to fetch seasonal stacks (T, C, H, W)
N_YEARS        = 2          # number of years of seasonal composites if temporal
BUFFER_M       = 300
SOURCES        = ["sentinel2"]   # legacy field — kept for dry-run display
MAX_WORKERS    = 4

# --- Imagery: GEE (fallback, used when USE_STAC=False) ---
USE_GEE        = False
GEE_PROJECT    = "towards-an-infra-fm"
GEE_COMPOSITE  = "median"
GEE_BUFFER_M   = 300

# --- QC thresholds ---
MIN_VALID_RATIO = 0.80

# --- Deduplication ---
DISTANCE_THRESHOLD_M = 200

# --- Triage ---
CONTRADICTION_THRESHOLD = 3
LOW_THRESHOLD           = 4


# ===========================================================================
# JOB PRESETS
# All output_dirs use _stac_v1 suffix for new STAC runs.
# ===========================================================================

JOB_PRESETS = {
    "north-america": {
        # "pbf_path":        "../data/pbf/power_only/north-america-latest.osm_power_only.osm.pbf",
        "output_dir":      "../data/curated_datasets/dataset_north-america_stac_v1",
        "assets_table":    "../data/PIPELINE/01-extracted-assets/north-america_all_assets_substations.parquet",
        "assets_csv":      "../data/PIPELINE/01-extracted-assets/north-america_all_assets_collapsed.csv",
        "assets_parquet":  "../data/PIPELINE/01-extracted-assets/north-america_all_assets_substations.parquet",
        "deduped_table":   "../data/PIPELINE/02-deduped-assets/north-america_deduped_assets_substations_sampled.parquet",
        "deduped_csv":     "../data/PIPELINE/02-deduped-assets/north-america_deduped_assets.csv",
        "deduped_parquet": "../data/PIPELINE/02-deduped-assets/north-america_deduped_assets_substations_sampled.parquet",
    },
    "europe": {
        # "pbf_path":        "../data/pbf/power_only/europe-latest.osm_power_only.osm.pbf",
        "output_dir":      "../data/curated_datasets/dataset_europe_stac_v1",
        "assets_table":    "../data/PIPELINE/01-extracted-assets/europe_all_assets_substations.parquet",
        "assets_csv":      "../data/PIPELINE/01-extracted-assets/europe_all_assets_collapsed.csv",
        "assets_parquet":  "../data/PIPELINE/01-extracted-assets/europe_all_assets_substations.parquet",
        "deduped_table":   "../data/PIPELINE/02-deduped-assets/europe_deduped_assets_substations_sampled.parquet",
        "deduped_csv":     "../data/PIPELINE/02-deduped-assets/europe_deduped_assets.csv",
        "deduped_parquet": "../data/PIPELINE/02-deduped-assets/europe_deduped_assets_substations_sampled.parquet",
    },
    "central-america": {
        # "pbf_path":        "../data/pbf/power_only/central-america-260408.osm_power_only.osm.pbf",
        "output_dir":      "../data/curated_datasets/dataset_central-america_stac_v1",
        "assets_table":    "../data/PIPELINE/01-extracted-assets/central-america_all_assets_substations.parquet",
        "assets_csv":      "../data/PIPELINE/01-extracted-assets/central-america_all_assets_substations.csv",
        "assets_parquet":  "../data/PIPELINE/01-extracted-assets/central-america_all_assets_substations.parquet",
        "deduped_table":   "../data/PIPELINE/02-deduped-assets/central-america_deduped_assets_substations.parquet",
        "deduped_csv":     "../data/PIPELINE/02-deduped-assets/central-america_deduped_assets.csv",
        "deduped_parquet": "../data/PIPELINE/02-deduped-assets/central-america_deduped_assets_substations.parquet",
    },
    "africa": {
        # "pbf_path":        "../data/pbf/power_only/africa-260408.osm_power_only.osm.pbf",
        "output_dir":      "../data/curated_datasets/dataset_africa_stac_v1",
        "assets_table":    "../data/PIPELINE/01-extracted-assets/africa_all_assets_substations.parquet",
        "assets_csv":      "../data/PIPELINE/01-extracted-assets/africa_all_assets_substations.csv",
        "assets_parquet":  "../data/PIPELINE/01-extracted-assets/africa_all_assets_substations.parquet",
        "deduped_table":   "../data/PIPELINE/02-deduped-assets/africa_deduped_assets_substations.parquet",
        "deduped_csv":     "../data/PIPELINE/02-deduped-assets/africa_deduped_assets.csv",
        "deduped_parquet": "../data/PIPELINE/02-deduped-assets/africa_deduped_assets_substations.parquet",
    },
    "australia-oceania": {
        # "pbf_path":        "../data/pbf/power_only/australia-oceania-260408.osm_power_only.osm.pbf",
        "output_dir":      "../data/curated_datasets/dataset_australia-oceania_stac_v1",
        "assets_table":    "../data/PIPELINE/01-extracted-assets/australia-oceania_all_assets_substations.parquet",
        "assets_csv":      "../data/PIPELINE/01-extracted-assets/australia-oceania_all_assets_substations.csv",
        "assets_parquet":  "../data/PIPELINE/01-extracted-assets/australia-oceania_all_assets_substations.parquet",
        "deduped_table":   "../data/PIPELINE/02-deduped-assets/australia-oceania_deduped_assets_substations.parquet",
        "deduped_csv":     "../data/PIPELINE/02-deduped-assets/australia-oceania_deduped_assets.csv",
        "deduped_parquet": "../data/PIPELINE/02-deduped-assets/australia-oceania_deduped_assets_substations.parquet",
    },
    "asia": {
        # "pbf_path":        "../data/pbf/power_only/asia-260408.osm_power_only.osm.pbf",
        "output_dir":      "../data/curated_datasets/dataset_asia_stac_v1",
        "assets_table":    "../data/PIPELINE/01-extracted-assets/asia_all_assets_substations.parquet",
        "assets_csv":      "../data/PIPELINE/01-extracted-assets/asia_all_assets_substations.csv",
        "assets_parquet":  "../data/PIPELINE/01-extracted-assets/asia_all_assets_substations.parquet",
        "deduped_table":   "../data/PIPELINE/02-deduped-assets/asia_deduped_assets_substations.parquet",
        "deduped_csv":     "../data/PIPELINE/02-deduped-assets/asia_deduped_assets.csv",
        "deduped_parquet": "../data/PIPELINE/02-deduped-assets/asia_deduped_assets_substations.parquet",
    },
    "south-america": {
        # "pbf_path":        "../data/pbf/power_only/south-america-260410.osm_power_only.osm.pbf",
        "output_dir":      "../data/curated_datasets/dataset_south-america_stac_v1",
        "assets_table":    "../data/PIPELINE/01-extracted-assets/south-america_all_assets_substations.parquet",
        "assets_csv":      "../data/PIPELINE/01-extracted-assets/south-america_all_assets_substations.csv",
        "assets_parquet":  "../data/PIPELINE/01-extracted-assets/south-america_all_assets_substations.parquet",
        "deduped_table":   "../data/PIPELINE/02-deduped-assets/south-america_deduped_assets_substations.parquet",
        "deduped_csv":     "../data/PIPELINE/02-deduped-assets/south-america_deduped_assets.csv",
        "deduped_parquet": "../data/PIPELINE/02-deduped-assets/south-america_deduped_assets_substations.parquet",
    },
    "maine": {
        # "pbf_path":        "../data/pbf/power_only/maine-latest.osm_power_only.osm.pbf",
        "output_dir":      "../data/curated_datasets/dataset_maine_stac_v1",
        "assets_table":    "../data/PIPELINE/01-extracted-assets/maine_all_assets_substations.parquet",
        "assets_csv":      "../data/PIPELINE/01-extracted-assets/maine_all_assets_substations.csv",
        "assets_parquet":  "../data/PIPELINE/01-extracted-assets/maine_all_assets_substations.parquet",
        "deduped_table":   "../data/PIPELINE/02-deduped-assets/maine_deduped_assets_substations.parquet",
        "deduped_csv":     "../data/PIPELINE/02-deduped-assets/maine_deduped_assets.csv",
        "deduped_parquet": "../data/PIPELINE/02-deduped-assets/maine_deduped_assets_substations.parquet",
    },
}

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
    "use_gee":            USE_GEE,
    "gee_project":        GEE_PROJECT,
    "gee_composite":      GEE_COMPOSITE,
    "gee_buffer_m":       GEE_BUFFER_M,
    "min_valid_ratio":    MIN_VALID_RATIO,
    "distance_threshold_m": DISTANCE_THRESHOLD_M,
    "contradiction_threshold": CONTRADICTION_THRESHOLD,
    "low_threshold":      LOW_THRESHOLD,
    "asset_types":        None,
    "shard_count":        1,
    "shard_index":        0,
    "shard_strategy":     "spatial",
    "schedule_name":      "pipeline_run",
    "schedule_dir":       os.path.join("data", "schedules"),
}


# ===========================================================================
# Helpers
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
    try:
        from hypergraphs.hypergraph_scheduler import (
            HypergraphScheduler, compute_assignment_metrics
        )
    except ImportError:
        compute_assignment_metrics = None
        HypergraphScheduler = None

    if shard_count <= 1:
        asset_to_shard = {aid: 0 for aid in df["asset_id"].tolist()}
        return {
            "asset_to_shard": asset_to_shard,
            "metrics": None,
            "summary_lines": [],
            "schedule": None,
        }

    if shard_strategy == "hypergraph" and HypergraphScheduler:
        scheduler = HypergraphScheduler(num_shards=shard_count)
        schedule  = scheduler.schedule(df)
        metrics   = compute_assignment_metrics(
            df, schedule.asset_to_shard, num_shards=shard_count,
            coarse_cell_km=scheduler.coarse_cell_km,
        )
        return {
            "asset_to_shard": schedule.asset_to_shard,
            "metrics": metrics,
            "summary_lines": schedule.summary_lines(),
            "schedule": schedule,
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
    if compute_assignment_metrics:
        metrics = compute_assignment_metrics(df, asset_to_shard,
                                             num_shards=shard_count)
        summary_lines = metrics.summary_lines()
    else:
        metrics = None
        summary_lines = []

    return {
        "asset_to_shard": asset_to_shard,
        "metrics": metrics,
        "summary_lines": summary_lines,
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
    Writes a _SUCCESS file with run metadata to the dataset output directory.
    Used by run_pipeline_from_collapsed_assets.py to detect completed regions.
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
        raise ValueError("output_dir must be provided via --job or --output-dir")

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
    if args.use_gee is not None:
        config["use_gee"] = args.use_gee
    if args.gee_project:
        config["gee_project"] = args.gee_project
    if args.gee_composite:
        config["gee_composite"] = args.gee_composite

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
    if config.get("use_gee") and not config.get("use_stac"):
        print(f"  GEE project:    {config['gee_project']}")
        print(f"  Composite:      {config['gee_composite']}")
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
# Pipeline runner
# ===========================================================================

def run_pipeline(
    dry_run : bool = False,
    config  : Optional[dict] = None,
) -> dict:
    """
    Runs the full curation pipeline end to end.
    Returns a dict of pipeline results and stage counts.
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
    # Step 1: Extract assets
    # ------------------------------------------------------------------
    stage_start = time.time()
    print("\n[1/6] Extracting assets from GeoFabrik PBF...")

    if config.get("assets_table") and os.path.exists(config["assets_table"]):
        print(f"  Loading existing asset table: {config['assets_table']}")
        df = load_asset_table(config["assets_table"])
        print(f"  Loaded {len(df)} assets")
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

    results["n_extracted"] = len(df)
    stage_timings["extract_assets"] = round(time.time() - stage_start, 2)
    print("  Asset counts:")
    for asset_type, count in df["asset_type"].value_counts().items():
        print(f"    {asset_type}: {count}")

    # ------------------------------------------------------------------
    # Step 2: Deduplicate
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
    # Step 3: Fetch imagery
    # ------------------------------------------------------------------
    stage_start = time.time()
    print(f"\n[3/6] Fetching imagery tiles ({len(df_clean)} assets)...")

    checkpoint_path = os.path.join(
        "data", "checkpoints",
        f"{os.path.basename(config['output_dir'])}_fetch.pkl",
    )

    if config.get("use_stac", True):
        from stac_imagery import STACImageryFetcher

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

    elif config.get("use_gee", False):
        from gee_imagery import GEEImageryFetcher

        fetcher = GEEImageryFetcher(
            project        = config["gee_project"],
            buffer_m       = config.get("gee_buffer_m", GEE_BUFFER_M),
            composite      = config.get("gee_composite", GEE_COMPOSITE),
            checkpoint_path= checkpoint_path,
        )
        tiles = fetcher.fetch_all(df_clean)

    else:
        raise ValueError(
            "No imagery fetcher enabled. Set USE_STAC=True (recommended) "
            "or USE_GEE=True in pipeline config."
        )

    n_ok   = sum(1 for t in tiles if t.status == "ok")
    n_fail = sum(1 for t in tiles if t.status != "ok")
    results["n_tiles_fetched"] = n_ok
    results["n_tiles_failed"]  = n_fail
    stage_timings["fetch_imagery"] = round(time.time() - stage_start, 2)
    print(f"  Fetched: {n_ok} ok, {n_fail} failed")

    # ------------------------------------------------------------------
    # Step 4: Quality control
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
    # Step 5: Triage
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
    # Step 6: Assemble dataset
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
    # Print summary
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
    # Write _SUCCESS with run metadata
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
    # Update timing log
    # ------------------------------------------------------------------
    try:
        log_path = "data/Infra-FM-timing-log.xlsx"

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
            collapsed_file_size_mb = file_size_mb(config.get("assets_table")),
            filter_preset     = config.get("filter_preset", "substation"),
            modalities        = "+".join(config.get("modalities", [])),
            temporal_stack    = str(config.get("temporal_stack", False)),
        )

        # Per-modality tile counts from manifest
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
# Entry point
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
    parser.add_argument("--use-gee",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Use Google Earth Engine (fallback)")
    parser.add_argument("--gee-project")
    parser.add_argument("--gee-composite", choices=["median", "mosaic", "best"])
    parser.add_argument("--min-confidence", choices=["high", "medium", "low"])
    parser.add_argument("--max-assets", type=int)
    parser.add_argument("--sample-per-type", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-strategy",
                        choices=["spatial", "asset_id", "hypergraph"],
                        default="spatial")

    args = parser.parse_args()
    run_pipeline(dry_run=args.dry_run, config=_build_runtime_config(args))