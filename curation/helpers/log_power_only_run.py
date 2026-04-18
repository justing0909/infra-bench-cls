from __future__ import annotations

import argparse
import time

from curation.utils.timing_log_utils import update_timing_log, file_size_kb, file_size_mb


def main() -> None:
    parser = argparse.ArgumentParser(description="Record pre-filter/power_only timing stats")
    parser.add_argument("--region", required=True)
    parser.add_argument("--raw-pbf", required=True)
    parser.add_argument("--power-only-pbf", required=True)
    parser.add_argument("--elapsed-s", type=float, required=True)

    args = parser.parse_args()

    update_timing_log(
        workbook_path="Infra-FM-timing-log.xlsx",
        region=args.region,
        starting_file_size_kb=file_size_kb(args.raw_pbf),
        pre_filter_time_s=round(args.elapsed_s, 2),
        power_only_file_size_mb=file_size_mb(args.power_only_pbf),
    )

    print("Updated power_only timing stats in workbook.")
    print(f"  region: {args.region}")
    print(f"  raw size KB: {file_size_kb(args.raw_pbf)}")
    print(f"  power_only size MB: {file_size_mb(args.power_only_pbf)}")
    print(f"  pre-filter time s: {round(args.elapsed_s, 2)}")


if __name__ == "__main__":
    main()