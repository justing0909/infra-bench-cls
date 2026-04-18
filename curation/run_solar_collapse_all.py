import subprocess
import sys
import os

regions = [
    "central-america",
    "australia-oceania",
    "south-america",
    "africa",
    "asia",
    "north-america",
    "europe",
]

# Map regions to the exact power_only filenames you currently have
power_pbf_map = {
    "central-america": "data/pbf/power_only/central-america-260408.osm_power_only.osm.pbf",
    "australia-oceania": "data/pbf/power_only/australia-oceania-260408.osm_power_only.osm.pbf",
    "south-america": "data/pbf/power_only/south-america-260410.osm_power_only.osm.pbf",
    "africa": "data/pbf/power_only/africa-260408.osm_power_only.osm.pbf",
    "asia": "data/pbf/power_only/asia-260408.osm_power_only.osm.pbf",
    "north-america": "data/pbf/power_only/north-america-latest.osm_power_only.osm.pbf",
    "europe": "data/pbf/power_only/europe-latest.osm_power_only.osm.pbf",
}

def run(cmd):
    print("\n================================================")
    print("RUNNING:", " ".join(cmd))
    print("================================================\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("\nERROR: command failed")
        sys.exit(1)

for region in regions:
    output_parquet = f"data/PIPELINE/01-extracted-assets/{region}_all_assets_collapsed.parquet"

    if os.path.exists(output_parquet):
        print(f"Skipping {region}: already exists -> {output_parquet}")
        continue

    print(f"\n\n#############################")
    print(f"PROCESSING REGION: {region}")
    print(f"#############################\n")

    power_pbf = power_pbf_map[region]

    run([
        "python",
        "solar_collapse.py",
        "--pbf",
        power_pbf,
        "--output-parquet",
        output_parquet,
    ])


print("\nALL SOLAR COLLAPSE RUNS COMPLETE")