"""
pipeline.py
-----------
End-to-end infrastructure imagery curation pipeline.

Orchestrates the full sequence:
    1. Extract asset locations from GeoFabrik PBF (sources.py)
    2. Deduplicate spatially proximate assets (deduplication.py)
    3. Fetch imagery tiles - NAIP + Sentinel-2 (imagery.py / gee_imagery.py)
    4. Basic quality control (qc.py)
    5. Confidence triage (triage.py)
    6. Assemble training dataset (dataset.py)

Designed to be run as a script for full pipeline runs, or imported
and called programmatically for notebook use.

Usage:
    python pipeline.py --dry-run
    python pipeline.py --job north-america --dry-run
    python pipeline.py --job north-america --shard-count 8 --shard-index 0
"""

import os
import time
import argparse
import json
from datetime import datetime
from typing import Dict, Optional
import pandas as pd
from sources import GeoFabrikSource
from solar_collapse import collapse_solar
from utils.io_utils import load_asset_table
from deduplication import Deduplicator
from qc import QualityChecker
from triage import RuleBasedTriager
from dataset import DatasetAssembler
from utils.timing_log_utils import update_timing_log, file_size_mb


# ===========================================================================
# CONFIGURATION - change these before each run
# ===========================================================================

# --- Input ---
# Download from https://download.geofabrik.de/
PBF_PATH = None

# --- Output ---
# Convention: data/dataset_<region>_<version>/
OUTPUT_DIR = None

# Intermediate CSV/parquet paths
ASSETS_CSV = None
ASSETS_PARQUET = None
ASSETS_TABLE = None

DEDUPED_CSV = None
DEDUPED_PARQUET = None
DEDUPED_TABLE = None

# --- Asset filtering ---
MIN_CONFIDENCE = "medium"  # "high", "medium", or "low"
MAX_ASSETS = None          # set to an int to cap (useful for testing)
SAMPLE_PER_TYPE = None     # set to an int to sample evenly per asset type

# --- Imagery ---
BUFFER_M = 150
SOURCES = ["sentinel2"]
MAX_WORKERS = 4

# --- GEE (recommended for global runs) ---
USE_GEE = True
GEE_PROJECT = "towards-an-infra-fm"
GEE_COMPOSITE = "median"   # "median", "mosaic", or "best"
GEE_BUFFER_M = 300

# --- Agentic scene selection (only used when USE_GEE = False) ---
USE_AGENT_SCENE_SELECTOR = True
AGENT_BACKEND = "ollama"
AGENT_API_KEY = ""

# --- QC thresholds ---
MIN_VALID_RATIO = 0.80
MAX_BRIGHTNESS = 220
MIN_BRIGHTNESS = 15

# --- Deduplication ---
DISTANCE_THRESHOLD_M = 200

# --- Triage ---
CONTRADICTION_THRESHOLD = 3
LOW_THRESHOLD = 4

# --- Job presets from the April 2026 project brief ---
JOB_PRESETS = {
    "north-america": {

        # starting input and ending output
        "pbf_path": "../data/pbf/power_only/north-america-latest.osm_power_only.osm.pbf",
        "output_dir": "../data/curated_datasets/dataset_north-america_sentinel_v1",

        # 01-collapsed-assets
        "assets_csv": "../data/PIPELINE/01-extracted-assets/north-america_all_assets_collapsed.csv",
        "assets_parquet": "../data/PIPELINE/01-extracted-assets/north-america_all_assets_collapsed.parquet",
        "assets_table": "../data/PIPELINE/01-extracted-assets/north-america_all_assets_collapsed.parquet",

        # 02-deduped-assets
        "deduped_csv": "../data/PIPELINE/02-deduped-assets/north-america_deduped_assets.csv",
        "deduped_parquet": "../data/PIPELINE/02-deduped-assets/north-america_deduped_assets.parquet",
        "deduped_table": "../data/PIPELINE/02-deduped-assets/north-america_deduped_assets.parquet",
    },
    "europe": {

        # starting input and ending output
        "pbf_path": "../data/pbf/power_only/europe-latest.osm_power_only.osm.pbf",
        "output_dir": "../data/curated_datasets/dataset_europe_sentinel_v1",

        # 01-collapsed-assets
        "assets_csv": "../data/PIPELINE/01-extracted-assets/europe_all_assets_collapsed.csv",
        "assets_parquet": "../data/PIPELINE/01-extracted-assets/europe_all_assets_collapsed.parquet",
        "assets_table": "../data/PIPELINE/01-extracted-assets/europe_all_assets_collapsed.parquet",

        # 02-deduped-assets
        "deduped_csv": "../data/PIPELINE/02-deduped-assets/europe_deduped_assets.csv",
        "deduped_parquet": "../data/PIPELINE/02-deduped-assets/europe_deduped_assets.parquet",
        "deduped_table": "../data/PIPELINE/02-deduped-assets/europe_deduped_assets.parquet",
    },
    "central-america": {

        # starting input and ending output
        "pbf_path": "../data/pbf/power_only/central-america-260408.osm_power_only.osm.pbf",
        "output_dir": "../data/curated_datasets/dataset_central-america_sentinel_v1",

        # 01-collapsed-assets
        "assets_csv": "../data/PIPELINE/01-extracted-assets/central-america_all_assets_collapsed.csv",
        "assets_parquet": "../data/PIPELINE/01-extracted-assets/central-america_all_assets_collapsed.parquet",
        "assets_table": "../data/PIPELINE/01-extracted-assets/central-america_all_assets_collapsed.parquet",

        # 02-deduped-assets
        "deduped_csv": "../data/PIPELINE/02-deduped-assets/central-america_deduped_assets.csv",
        "deduped_parquet": "../data/PIPELINE/02-deduped-assets/central-america_deduped_assets.parquet",
        "deduped_table": "../data/PIPELINE/02-deduped-assets/central-america_deduped_assets.parquet",
    },
    "africa": {
        # starting input and ending output
        "pbf_path": "../data/pbf/power_only/africa-260408.osm_power_only.osm.pbf",
        "output_dir": "../data/curated_datasets/dataset_africa_sentinel_v1",

        # 01-collapsed-assets
        "assets_csv": "../data/PIPELINE/01-extracted-assets/africa_all_assets_collapsed.csv",
        "assets_parquet": "../data/PIPELINE/01-extracted-assets/africa_all_assets_collapsed.parquet",
        "assets_table": "../data/PIPELINE/01-extracted-assets/africa_all_assets_collapsed.parquet",

        # 02-deduped-assets
        "deduped_csv": "../data/PIPELINE/02-deduped-assets/africa_deduped_assets.csv",
        "deduped_parquet": "../data/PIPELINE/02-deduped-assets/africa_deduped_assets.parquet",
        "deduped_table": "../data/PIPELINE/02-deduped-assets/africa_deduped_assets.parquet",
    },
    "australia-oceania": {

        # starting input and ending output
        "pbf_path": "../data/pbf/power_only/australia-oceania-260408.osm_power_only.osm.pbf",
        "output_dir": "../data/curated_datasets/dataset_australia-oceania_sentinel_v1",

        # 01-collapsed-assets
        "assets_csv": "../data/PIPELINE/01-extracted-assets/australia-oceania_all_assets_collapsed.csv",
        "assets_parquet": "../data/PIPELINE/01-extracted-assets/australia-oceania_all_assets_collapsed.parquet",
        "assets_table": "../data/PIPELINE/01-extracted-assets/australia-oceania_all_assets_collapsed.parquet",

        # 02-deduped-assets
        "deduped_csv": "../data/PIPELINE/02-deduped-assets/australia-oceania_deduped_assets.csv",
        "deduped_parquet": "../data/PIPELINE/02-deduped-assets/australia-oceania_deduped_assets.parquet",
        "deduped_table": "../data/PIPELINE/02-deduped-assets/australia-oceania_deduped_assets.parquet",
    },
    "asia": {

        # starting input and ending output
        "pbf_path": "../data/pbf/power_only/asia-260408.osm_power_only.osm.pbf",
        "output_dir": "../data/curated_datasets/dataset_asia_sentinel_v1",

        # 01-collapsed-assets
        "assets_csv": "../data/PIPELINE/01-extracted-assets/asia_all_assets_collapsed.csv",
        "assets_parquet": "../data/PIPELINE/01-extracted-assets/asia_all_assets_collapsed.parquet",
        "assets_table": "../data/PIPELINE/01-extracted-assets/asia_all_assets_collapsed.parquet",

        # 02-deduped-assets
        "deduped_csv": "../data/PIPELINE/02-deduped-assets/asia_deduped_assets.csv",
        "deduped_parquet": "../data/PIPELINE/02-deduped-assets/asia_deduped_assets.parquet",
        "deduped_table": "../data/PIPELINE/02-deduped-assets/asia_deduped_assets.parquet",
    },
    "south-america" : {

        # starting input and ending output
        "pbf_path": "../data/pbf/power_only/south-america-260410.osm_power_only.osm.pbf",
        "output_dir": "../data/curated_datasets/dataset_south-america_sentinel_v1",

        # 01-collapsed-assets
        "assets_csv": "../data/PIPELINE/01-extracted-assets/south-america_all_assets_collapsed.csv",
        "assets_parquet": "../data/PIPELINE/01-extracted-assets/south-america_all_assets_collapsed.parquet",
        "assets_table": "../data/PIPELINE/01-extracted-assets/south-america_all_assets_collapsed.parquet",

        # 02-deduped-assets
        "deduped_csv": "../data/PIPELINE/02-deduped-assets/south-america_deduped_assets.csv",
        "deduped_parquet": "../data/PIPELINE/02-deduped-assets/south-america_deduped_assets.parquet",
        "deduped_table": "../data/PIPELINE/02-deduped-assets/south-america_deduped_assets.parquet",
    },
    "maine": {

        # starting input and ending output
        "pbf_path": "../data/pbf/power_only/maine-latest.osm_power_only.osm.pbf",
        "output_dir": "../data/curated_datasets/dataset_maine_v1",

        # 01-extracted-assets
        "assets_csv": "../data/PIPELINE/01-extracted-assets/maine_all_assets_collapsed.csv",
        "assets_parquet": "../data/PIPELINE/01-extracted-assets/maine_all_assets_collapsed.parquet",
        "assets_table": "../data/PIPELINE/01-extracted-assets/maine_all_assets_collapsed.parquet",

        # 02-deduped-assets
        "deduped_csv": "../data/PIPELINE/02-deduped-assets/maine_deduped_assets.csv",
        "deduped_parquet": "../data/PIPELINE/02-deduped-assets/maine_deduped_assets.parquet",
        "deduped_table": "../data/PIPELINE/02-deduped-assets/maine_deduped_assets.parquet",
    }
}

DEFAULT_CONFIG = {
    "pbf_path": PBF_PATH,
    "output_dir": OUTPUT_DIR,
    "assets_csv": ASSETS_CSV,
    "assets_parquet": ASSETS_PARQUET,
    "assets_table": ASSETS_TABLE,
    "deduped_csv": DEDUPED_CSV,
    "deduped_parquet": DEDUPED_PARQUET,
    "deduped_table": DEDUPED_TABLE,
    "min_confidence": MIN_CONFIDENCE,
    "max_assets": MAX_ASSETS,
    "sample_per_type": SAMPLE_PER_TYPE,
    "buffer_m": BUFFER_M,
    "sources": list(SOURCES),
    "max_workers": MAX_WORKERS,
    "use_gee": USE_GEE,
    "gee_project": GEE_PROJECT,
    "gee_composite": GEE_COMPOSITE,
    "gee_buffer_m": GEE_BUFFER_M,
    "use_agent_scene_selector": USE_AGENT_SCENE_SELECTOR,
    "agent_backend": AGENT_BACKEND,
    "agent_api_key": AGENT_API_KEY,
    "min_valid_ratio": MIN_VALID_RATIO,
    "max_brightness": MAX_BRIGHTNESS,
    "min_brightness": MIN_BRIGHTNESS,
    "distance_threshold_m": DISTANCE_THRESHOLD_M,
    "contradiction_threshold": CONTRADICTION_THRESHOLD,
    "low_threshold": LOW_THRESHOLD,
    "asset_types": None,
    "shard_count": 1,
    "shard_index": 0,
    "shard_strategy": "spatial",
    "schedule_name": "pipeline_run",
    "schedule_dir": os.path.join("data", "schedules"),
    "use_collapsed_power_extraction": True,
    "cluster_radius_m": 250.0,
    "plant_buffer_m": 35.0,
    "assets_parquet": ASSETS_PARQUET,
    "assets_table": ASSETS_TABLE,
    "deduped_parquet": DEDUPED_PARQUET,
    "deduped_table": DEDUPED_TABLE,
}


# ===========================================================================
# Helpers
# ===========================================================================
def _parse_csv_arg(value: Optional[str]) -> Optional[list[str]]:
    """Parses comma-separated CLI values into a clean list."""
    if value is None:
        return None
    parts = [item.strip() for item in value.split(",")]
    parts = [item for item in parts if item]
    return parts or None


def _interleave_bits(x: int, y: int) -> int:
    """Builds a Morton/Z-order key for stable spatial sharding."""
    result = 0
    for i in range(32):
        result |= ((x >> i) & 1) << (2 * i)
        result |= ((y >> i) & 1) << (2 * i + 1)
    return result


def _spatial_sort_key(lat: float, lon: float) -> int:
    """
    Maps lat/lon to a Morton key.

    This is the current production sharding strategy: we keep nearby assets in
    the same shard so big jobs are spatially coherent, resumable, and easier to
    inspect.
    """
    lat_norm = min(max((lat + 90.0) / 180.0, 0.0), 1.0)
    lon_norm = min(max((lon + 180.0) / 360.0, 0.0), 1.0)
    scale = (1 << 16) - 1
    lat_i = int(round(lat_norm * scale))
    lon_i = int(round(lon_norm * scale))
    return _interleave_bits(lat_i, lon_i)


def _with_suffix(path: str, suffix: str) -> str:
    """Adds a suffix before a file extension, or to the end of a directory-like path."""
    if not suffix:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}{suffix}{ext}"


def _apply_shard(
    df: pd.DataFrame,
    shard_index: int,
    shard_count: int,
    shard_strategy: str,
) -> pd.DataFrame:
    """Splits a dataframe into one deterministic shard."""
    if shard_count <= 1:
        return df.reset_index(drop=True)

    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            f"shard_index must be in [0, {shard_count - 1}], got {shard_index}"
        )

    shard_df = df.copy()
    if shard_strategy == "hypergraph":
        from hypergraphs.hypergraph_scheduler import HypergraphScheduler

        scheduler = HypergraphScheduler(num_shards=shard_count)
        schedule = scheduler.schedule(shard_df)
        print("  Hypergraph schedule summary (experimental strategy):")
        for line in schedule.summary_lines():
            print(line)

        selected_asset_ids = set(schedule.assets_for_shard(shard_index))
        shard_df = shard_df[shard_df["asset_id"].isin(selected_asset_ids)].copy()
        return shard_df.sort_values(["asset_id"]).reset_index(drop=True)

    if shard_strategy == "spatial":
        shard_df["_shard_key"] = [
            _spatial_sort_key(lat, lon)
            for lat, lon in zip(shard_df["lat"], shard_df["lon"])
        ]
        shard_df = shard_df.sort_values(
            ["_shard_key", "asset_id"]
        ).reset_index(drop=True)
    else:
        shard_df = shard_df.sort_values(["asset_id"]).reset_index(drop=True)

    base_size = len(shard_df) // shard_count
    remainder = len(shard_df) % shard_count
    start = shard_index * base_size + min(shard_index, remainder)
    stop = start + base_size + (1 if shard_index < remainder else 0)

    shard_df = shard_df.iloc[start:stop].copy()
    shard_df = shard_df.drop(columns=["_shard_key"], errors="ignore")
    return shard_df.reset_index(drop=True)


def _build_sorted_assignment(
    ordered_df: pd.DataFrame,
    shard_count: int,
) -> Dict[str, int]:
    """Assigns ordered rows to shards using the same contiguous split logic."""
    ordered_df = ordered_df.reset_index(drop=True)
    base_size = len(ordered_df) // shard_count
    remainder = len(ordered_df) % shard_count
    asset_to_shard: Dict[str, int] = {}

    for shard_index in range(shard_count):
        start = shard_index * base_size + min(shard_index, remainder)
        stop = start + base_size + (1 if shard_index < remainder else 0)
        shard_asset_ids = ordered_df.iloc[start:stop]["asset_id"].tolist()
        for asset_id in shard_asset_ids:
            asset_to_shard[asset_id] = shard_index

    return asset_to_shard


def _build_shard_assignment(
    df: pd.DataFrame,
    shard_count: int,
    shard_strategy: str,
) -> dict:
    """Builds a full asset -> shard plan plus comparable metrics."""
    from hypergraphs.hypergraph_scheduler import HypergraphScheduler, compute_assignment_metrics

    if shard_count <= 1:
        asset_to_shard = {asset_id: 0 for asset_id in df["asset_id"].tolist()}
        metrics = compute_assignment_metrics(df, asset_to_shard, num_shards=1)
        return {
            "asset_to_shard": asset_to_shard,
            "metrics": metrics,
            "summary_lines": metrics.summary_lines(),
            "schedule": None,
        }

    if shard_strategy == "hypergraph":
        scheduler = HypergraphScheduler(num_shards=shard_count)
        schedule = scheduler.schedule(df)
        metrics = compute_assignment_metrics(
            df,
            schedule.asset_to_shard,
            num_shards=shard_count,
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
        ordered_df = ordered_df.sort_values(["_shard_key", "asset_id"]).reset_index(drop=True)
        asset_to_shard = _build_sorted_assignment(ordered_df, shard_count)
    else:
        ordered_df = df.sort_values(["asset_id"]).reset_index(drop=True)
        asset_to_shard = _build_sorted_assignment(ordered_df, shard_count)

    metrics = compute_assignment_metrics(df, asset_to_shard, num_shards=shard_count)
    return {
        "asset_to_shard": asset_to_shard,
        "metrics": metrics,
        "summary_lines": metrics.summary_lines(),
        "schedule": None,
    }


def _save_shard_artifacts(
    df: pd.DataFrame,
    config: dict,
    shard_plan: dict,
) -> None:
    """Writes shard assignments and summary diagnostics to disk."""
    os.makedirs(config["schedule_dir"], exist_ok=True)

    base_name = config["schedule_name"]
    shard_count = config["shard_count"]
    strategy = config["shard_strategy"]
    stem = f"{base_name}_{strategy}_shards-{shard_count:02d}"

    assignment_path = os.path.join(
        config["schedule_dir"],
        f"{stem}_assignments.csv",
    )
    summary_path = os.path.join(
        config["schedule_dir"],
        f"{stem}_summary.json",
    )

    assignments = df[["asset_id", "asset_type", "lat", "lon"]].copy()
    assignments["shard_index"] = assignments["asset_id"].map(
        shard_plan["asset_to_shard"]
    )
    assignments = assignments.sort_values(
        ["shard_index", "asset_type", "asset_id"]
    ).reset_index(drop=True)
    assignments.to_csv(assignment_path, index=False)

    summary = {
        "strategy": strategy,
        "shard_count": shard_count,
        "schedule_name": base_name,
        "metrics": shard_plan["metrics"].to_dict(),
        "summary_lines": shard_plan["summary_lines"],
    }
    if shard_plan["schedule"] is not None:
        summary["schedule"] = shard_plan["schedule"].to_dict()

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved shard assignments: {assignment_path}")
    print(f"  Saved shard summary:     {summary_path}")


def _build_runtime_config(args: Optional[argparse.Namespace]) -> dict:
    """Combines neutral defaults, job presets, and CLI overrides."""
    config = dict(DEFAULT_CONFIG)
    config["sources"] = list(DEFAULT_CONFIG["sources"])

    if args is None:
        return config

    if args.job:
        config.update(JOB_PRESETS[args.job])

    # Explicit CLI overrides
    for key, arg_name in [
        ("pbf_path", "pbf_path"),
        ("output_dir", "output_dir"),
        ("assets_csv", "assets_csv"),
        ("assets_parquet", "assets_parquet"),
        ("assets_table", "assets_table"),
        ("deduped_csv", "deduped_csv"),
        ("deduped_parquet", "deduped_parquet"),
        ("deduped_table", "deduped_table"),
    ]:
        value = getattr(args, arg_name, None)
        if value:
            config[key] = value

    # Derive related paths only if missing
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

    # Only require pbf_path if we might need extraction
    assets_table = config.get("assets_table")
    if not assets_table and not config.get("pbf_path"):
        raise ValueError(
            "pbf_path is required when no assets table is provided"
        )

    config["schedule_name"] = os.path.basename(config["output_dir"]) if config.get("output_dir") else "pipeline_run"

    if args.min_confidence:
        config["min_confidence"] = args.min_confidence
    if args.max_assets is not None:
        config["max_assets"] = args.max_assets
    if args.sample_per_type is not None:
        config["sample_per_type"] = args.sample_per_type
    if args.sources:
        config["sources"] = _parse_csv_arg(args.sources)
    if args.use_gee is not None:
        config["use_gee"] = args.use_gee
    if args.gee_project:
        config["gee_project"] = args.gee_project
    if args.gee_composite:
        config["gee_composite"] = args.gee_composite

    config["asset_types"] = _parse_csv_arg(args.asset_types)
    config["shard_count"] = args.shard_count
    config["shard_index"] = args.shard_index
    config["shard_strategy"] = args.shard_strategy

    if config["shard_count"] > 1:
        shard_suffix = (
            f"_shard-{config['shard_index'] + 1:02d}-of-{config['shard_count']:02d}"
        )
        config["output_dir"] = _with_suffix(config["output_dir"], shard_suffix)

    return config


def _print_run_plan(config: dict) -> None:
    """Prints the effective runtime plan before the heavy work starts."""
    print("\nRun plan:")
    print(f"  PBF:          {config['pbf_path']}")
    print(f"  Assets File:   {config['assets_table']}")
    print(f"  Deduped File:  {config['deduped_table']}")
    print(f"  Output dir:   {config['output_dir']}")
    print(f"  Sources:      {config['sources']}")
    print(f"  Use GEE:      {config['use_gee']}")
    if config["use_gee"]:
        print(f"  GEE project:  {config['gee_project']}")
        print(f"  Composite:    {config['gee_composite']}")
    if config["asset_types"]:
        print(f"  Asset types:  {config['asset_types']}")
    if config["sample_per_type"] is not None:
        print(f"  Sample/type:  {config['sample_per_type']}")
    if config["max_assets"] is not None:
        print(f"  Max assets:   {config['max_assets']}")
    if config["shard_count"] > 1:
        print(
            f"  Shard:        {config['shard_index'] + 1}/{config['shard_count']} "
            f"({config['shard_strategy']})"
        )


# ===========================================================================
# Pipeline runner
# ===========================================================================

def run_pipeline(
    dry_run: bool = False,
    config: Optional[dict] = None,
) -> dict:
    """
    Runs the full curation pipeline end to end.

    Parameters
    ----------
    dry_run : bool
        If True, print the plan and asset counts but skip imagery fetching.
    config : dict, optional
        Runtime configuration. If omitted, the file-level defaults are used.

    Returns
    -------
    dict of pipeline results and counts
    """
    config = dict(DEFAULT_CONFIG if config is None else config)
    config["sources"] = list(config.get("sources", []))

    if not config["sources"]:
        raise ValueError("At least one imagery source must be specified.")

    start_time = time.time()
    results = {}
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

    if os.path.exists(config["assets_table"]):
        print(f"  Loading existing asset table: {config['assets_table']}")
        df = load_asset_table(config["assets_table"])
        print(f"  Loaded {len(df)} assets")
    else:
        if config.get("use_collapsed_power_extraction", False):
            df = collapse_solar(
                pbf_path=config["pbf_path"],
                output_parquet=config["assets_parquet"],
                output_csv=None,
                cluster_radius_m=config.get("cluster_radius_m", 250.0),
                plant_buffer_m=config.get("plant_buffer_m", 35.0),
            )
        else:
            src = GeoFabrikSource(
                config["pbf_path"],
                min_confidence=config["min_confidence"],
            )
            df = src.extract_all()

        write_df = df.drop(columns=["osm_tags"], errors="ignore").copy()

        if config.get("assets_csv"):
            os.makedirs(os.path.dirname(config["assets_csv"]) or ".", exist_ok=True)
            write_df.to_csv(config["assets_csv"], index=False)

        if config.get("assets_parquet"):
            os.makedirs(os.path.dirname(config["assets_parquet"]) or ".", exist_ok=True)
            write_df.to_parquet(config["assets_parquet"], index=False)

        print(f"  Saved {len(df)} assets to:")
        if config.get("assets_csv"):
            print(f"    CSV:     {config['assets_csv']}")
        if config.get("assets_parquet"):
            print(f"    Parquet: {config['assets_parquet']}")

            print(f"  Saved {len(df)} assets to:")
            print(f"    CSV:     {config['assets_csv']}")
            print(f"    Parquet: {config['assets_parquet']}")

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

    if os.path.exists(config["deduped_table"]):
        print(f"  Loading existing deduplicated table: {config['deduped_table']}")
        df_clean = load_asset_table(config["deduped_table"])
        print(f"  Loaded {len(df_clean)} deduplicated assets")
    else:
        dedup = Deduplicator(
            distance_threshold_m=config["distance_threshold_m"]
        )
        df_clean, _df_removed = dedup.run(df)

        if config.get("deduped_csv"):
            os.makedirs(os.path.dirname(config["deduped_csv"]) or ".", exist_ok=True)
            df_clean.to_csv(config["deduped_csv"], index=False)

        if config.get("deduped_parquet"):
            os.makedirs(os.path.dirname(config["deduped_parquet"]) or ".", exist_ok=True)
            df_clean.to_parquet(config["deduped_parquet"], index=False)

        print(f"  Saved {len(df_clean)} assets to:")
        if config.get("deduped_csv"):
            print(f"    CSV:     {config['deduped_csv']}")
        if config.get("deduped_parquet"):
            print(f"    Parquet: {config['deduped_parquet']}")

        print(f"  Saved {len(df_clean)} assets to:")
        print(f"    CSV:     {config['deduped_csv']}")
        print(f"    Parquet: {config['deduped_parquet']}")

    results["n_after_dedup"] = len(df_clean)

    if config["asset_types"]:
        before = len(df_clean)
        df_clean = df_clean[df_clean["asset_type"].isin(config["asset_types"])].copy()
        print(
            f"  Filtered asset types: {before} -> {len(df_clean)} "
            f"for {config['asset_types']}"
        )

    if config["sample_per_type"] is not None:
        df_clean = (
            df_clean.groupby("asset_type", group_keys=False)
            .apply(
                lambda g: g.sample(
                    min(len(g), config["sample_per_type"]),
                    random_state=42,
                )
            )
            .reset_index(drop=True)
        )
        print(
            f"  Sampled {len(df_clean)} assets "
            f"({config['sample_per_type']} per type)"
        )

    if config["max_assets"] is not None:
        df_clean = df_clean.head(config["max_assets"])
        print(f"  Capped at {len(df_clean)} assets")

    if config["shard_count"] > 1:
        shard_stage_start = time.time()
        shard_plan = _build_shard_assignment(
            df_clean,
            shard_count=config["shard_count"],
            shard_strategy=config["shard_strategy"],
        )
        print("  Shard planning summary:")
        for line in shard_plan["summary_lines"]:
            print(line)
        _save_shard_artifacts(df_clean, config, shard_plan)

        selected_asset_ids = {
            asset_id
            for asset_id, assigned_shard in shard_plan["asset_to_shard"].items()
            if assigned_shard == config["shard_index"]
        }
        df_clean = df_clean[df_clean["asset_id"].isin(selected_asset_ids)].copy()
        df_clean = df_clean.sort_values(["asset_id"]).reset_index(drop=True)
        print(
            f"  Shard selection kept {len(df_clean)} assets "
            f"for shard {config['shard_index'] + 1}/{config['shard_count']}"
        )
        stage_timings["plan_shards"] = round(time.time() - shard_stage_start, 2)
    else:
        stage_timings["plan_shards"] = 0.0
    results["n_in_job"] = len(df_clean)
    stage_timings["deduplicate"] = round(time.time() - stage_start, 2)

    if len(df_clean) == 0:
        print("\nNo assets remain after filtering and sharding. Nothing to do.")
        results["elapsed_s"] = round(time.time() - start_time, 1)
        return results

    if dry_run:
        print("\n[DRY RUN] Stopping before imagery fetch.")
        print(f"  Would fetch tiles for {len(df_clean)} assets")
        print(f"  Sources: {config['sources']}")
        print(f"  Estimated tiles: {len(df_clean) * len(config['sources'])}")
        results["timings_s"] = stage_timings
        results["elapsed_s"] = round(time.time() - start_time, 1)
        return results

    # ------------------------------------------------------------------
    # Step 3: Fetch imagery
    # ------------------------------------------------------------------
    stage_start = time.time()
    print(
        f"\n[3/6] Fetching imagery tiles "
        f"({len(df_clean)} assets x {len(config['sources'])} sources)..."
    )

    if config["use_gee"]:
        from gee_imagery import GEEImageryFetcher

        checkpoint_path = os.path.join(
            "data",
            "checkpoints",
            f"{os.path.basename(config['output_dir'])}_gee_fetch.pkl",
        )
        fetcher = GEEImageryFetcher(
            project=config["gee_project"],
            buffer_m=config["gee_buffer_m"],
            composite=config["gee_composite"],
            checkpoint_path=checkpoint_path,
        )
        tiles = fetcher.fetch_all(df_clean)
    else:
        from legacy.imagery import AgentSceneSelector, BatchedImageryFetcher

        scene_selector = None
        if config["use_agent_scene_selector"]:
            scene_selector = AgentSceneSelector(
                backend=config["agent_backend"],
                api_key=config["agent_api_key"] or None,
            )

        checkpoint_path = os.path.join(
            "data",
            "checkpoints",
            f"{os.path.basename(config['output_dir'])}_fetch.pkl",
        )
        fetcher = BatchedImageryFetcher(
            buffer_m=config["buffer_m"],
            checkpoint_path=checkpoint_path,
            checkpoint_every=50,
            naip_fallback="naip" in config["sources"],
        )
        # Keep the variable alive for future fetcher integration, even though
        # BatchedImageryFetcher currently does not accept it directly.
        _ = scene_selector
        tiles = fetcher.fetch_all(df_clean)

    n_ok = sum(1 for t in tiles if t.status == "ok")
    n_fail = sum(1 for t in tiles if t.status != "ok")
    results["n_tiles_fetched"] = n_ok
    results["n_tiles_failed"] = n_fail
    stage_timings["fetch_imagery"] = round(time.time() - stage_start, 2)
    print(f"  Fetched: {n_ok} ok, {n_fail} failed")

    # ------------------------------------------------------------------
    # Step 4: Quality control
    # ------------------------------------------------------------------
    stage_start = time.time()
    print("\n[4/6] Running quality control...")

    checker = QualityChecker(
        min_valid_ratio=config["min_valid_ratio"],
        max_brightness=config["max_brightness"],
        min_brightness=config["min_brightness"],
    )
    qc_results = checker.check_all(
        tiles,
        max_workers=config["max_workers"],
    )
    clean = checker.filter_ok(qc_results)

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
    triage_results = triager.triage_all(
        clean,
        max_workers=config["max_workers"],
    )
    accepted = triager.filter_accepted(triage_results)
    flagged = triager.filter_review(triage_results)

    results["n_accepted"] = len(accepted)
    results["n_flagged"] = len(flagged)
    results["n_rejected"] = n_qc_pass - len(accepted) - len(flagged)
    stage_timings["triage"] = round(time.time() - stage_start, 2)
    print(
        f"  Accepted: {len(accepted)}, flagged: {len(flagged)}, "
        f"rejected: {results['n_rejected']}"
    )

    # ------------------------------------------------------------------
    # Step 6: Assemble dataset
    # ------------------------------------------------------------------
    stage_start = time.time()
    print(f"\n[6/6] Assembling dataset -> {config['output_dir']}...")

    assembler = DatasetAssembler(config["output_dir"])
    summary = assembler.assemble(accepted, triage_results)

    results["n_dataset_tiles"] = len(summary)
    results["output_dir"] = config["output_dir"]
    results["elapsed_s"] = round(time.time() - start_time, 1)
    stage_timings["assemble_dataset"] = round(time.time() - stage_start, 2)
    results["timings_s"] = stage_timings

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
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
    # Timing log update
    # ------------------------------------------------------------------
    try:
        log_path = "data/Infra-FM-timing-log.xlsx"

        gee_total = results["n_tiles_fetched"] + results["n_tiles_failed"]
        gee_accept_pct = round(results["n_tiles_fetched"] / gee_total * 100, 2) if gee_total else None

        qc_total = results["n_qc_passed"] + results["n_qc_failed"]
        qc_accept_pct = round(results["n_qc_passed"] / qc_total * 100, 2) if qc_total else None

        triage_accept_pct = (
            round(results["n_accepted"] / results["n_qc_passed"] * 100, 2)
            if results["n_qc_passed"] > 0 else None
        )

        region_name = os.path.basename(config["output_dir"]).replace("dataset_", "").replace("_sentinel_v1", "").replace("_v1", "")

        update_timing_log(
            workbook_path=log_path,
            region=region_name,
            scanning_time_s=stage_timings.get("extract_assets"),
            gee_accept_pct=gee_accept_pct,
            qc_accept_pct=qc_accept_pct,
            triage_accept_pct=triage_accept_pct,
            assets_extracted=results.get("n_extracted"),
            assets_after_dedup=results.get("n_after_dedup"),
            total_tiles_fetched=results.get("n_tiles_fetched"),
            dataset_tiles=results.get("n_dataset_tiles"),
            total_time_elapsed_s=results.get("elapsed_s"),
            collapsed_file_size_mb=file_size_mb(config.get("assets_table")),
        )
        print(f"  Updated timing log: {log_path}")
    except Exception as e:
        print(f"  Warning: could not update timing log: {e}")

    return results


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infrastructure curation pipeline")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without fetching imagery",
    )
    parser.add_argument(
        "--job",
        choices=sorted(JOB_PRESETS),
        help="Load a predefined regional job from the project brief",
    )
    parser.add_argument("--pbf-path", help="Override the PBF input path")
    parser.add_argument("--output-dir", help="Override the dataset output directory")
    parser.add_argument("--assets-csv", help="Override the extracted assets CSV path")
    parser.add_argument("--deduped-csv", help="Override the deduplicated assets CSV path")
    parser.add_argument(
        "--asset-types",
        help="Comma-separated asset types to keep after deduplication",
    )
    parser.add_argument(
        "--sources",
        help="Comma-separated imagery sources, e.g. sentinel2 or sentinel2,naip",
    )
    parser.add_argument(
        "--min-confidence",
        choices=["high", "medium", "low"],
        help="Minimum ontology confidence to include during extraction",
    )
    parser.add_argument(
        "--max-assets",
        type=int,
        help="Cap the number of assets after filtering for smaller test runs",
    )
    parser.add_argument(
        "--sample-per-type",
        type=int,
        help="Sample an even number of assets per type after filtering",
    )
    parser.add_argument(
        "--use-gee",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use Google Earth Engine instead of the Planetary Computer path",
    )
    parser.add_argument("--gee-project", help="Override the Google Earth Engine project ID")
    parser.add_argument(
        "--gee-composite",
        choices=["median", "mosaic", "best"],
        help="Override the Sentinel-2 compositing strategy",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Split the filtered asset list into this many deterministic shards",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard index to run when --shard-count > 1",
    )
    parser.add_argument(
        "--shard-strategy",
        choices=["spatial", "asset_id", "hypergraph"],
        default="spatial",
        help="How to split assets into shards. Use 'spatial' for the current recommended path; 'hypergraph' remains experimental.",
    )

    parser.add_argument("--assets-table", help="Override the extracted assets table path (.parquet or .csv)")
    parser.add_argument("--deduped-table", help="Override the deduplicated assets table path (.parquet or .csv)")
    parser.add_argument("--assets-parquet", help="Override the extracted assets parquet path")
    parser.add_argument("--deduped-parquet", help="Override the deduplicated assets parquet path")

    args = parser.parse_args()

    run_pipeline(
        dry_run=args.dry_run,
        config=_build_runtime_config(args),
    )
