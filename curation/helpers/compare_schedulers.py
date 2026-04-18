"""
compare_schedulers.py
---------------------
Benchmarks shard planning strategies on an asset CSV.

Example:
    python compare_schedulers.py --input-csv data/us-northeast_deduped_assets.csv
    python compare_schedulers.py --input-csv data/us-northeast_deduped_assets.csv --shard-counts 4,8
"""

import argparse
import json
import os

import pandas as pd

from curation.pipeline import _build_shard_assignment


def _parse_csv_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare shard planning strategies")
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Deduplicated asset CSV to analyze",
    )
    parser.add_argument(
        "--shard-counts",
        default="4,8",
        help="Comma-separated shard counts to compare",
    )
    parser.add_argument(
        "--strategies",
        default="spatial,hypergraph",
        help="Comma-separated strategies to compare",
    )
    parser.add_argument(
        "--asset-types",
        help="Optional comma-separated asset types to keep before evaluation",
    )
    parser.add_argument(
        "--max-assets",
        type=int,
        help="Optional cap for faster experiments",
    )
    parser.add_argument(
        "--output-prefix",
        help="Optional custom output prefix inside data/schedules/",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.asset_types:
        keep_types = set(_parse_csv_arg(args.asset_types))
        df = df[df["asset_type"].isin(keep_types)].copy()
    if args.max_assets is not None:
        df = df.head(args.max_assets).copy()

    shard_counts = [int(value) for value in _parse_csv_arg(args.shard_counts)]
    strategies = _parse_csv_arg(args.strategies)

    records = []
    details = {}

    for shard_count in shard_counts:
        for strategy in strategies:
            plan = _build_shard_assignment(
                df,
                shard_count=shard_count,
                shard_strategy=strategy,
            )
            metrics = plan["metrics"]
            key = f"{strategy}__{shard_count}"
            details[key] = {
                "strategy": strategy,
                "shard_count": shard_count,
                "summary_lines": plan["summary_lines"],
                "metrics": metrics.to_dict(),
            }
            if plan["schedule"] is not None:
                details[key]["schedule"] = plan["schedule"].to_dict()

            records.append(
                {
                    "strategy": strategy,
                    "shard_count": shard_count,
                    "n_assets": metrics.n_assets,
                    "n_hyperedges": metrics.n_hyperedges,
                    "cut_weight_total": metrics.cut_weight_total,
                    "size_std": metrics.size_std,
                    "size_min": metrics.size_min,
                    "size_max": metrics.size_max,
                    "size_ratio": metrics.size_ratio,
                    "type_balance_score": metrics.type_balance_score,
                    "shard_sizes_json": json.dumps(metrics.shard_sizes, sort_keys=True),
                }
            )

    comparison_df = pd.DataFrame(records).sort_values(
        ["shard_count", "strategy"]
    ).reset_index(drop=True)

    output_dir = os.path.join("data", "schedules")
    os.makedirs(output_dir, exist_ok=True)

    dataset_name = (
        args.output_prefix
        or os.path.splitext(os.path.basename(args.input_csv))[0]
    )
    csv_path = os.path.join(output_dir, f"{dataset_name}_scheduler_comparison.csv")
    json_path = os.path.join(output_dir, f"{dataset_name}_scheduler_comparison.json")

    comparison_df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2)

    print("Scheduler comparison:")
    print(
        comparison_df[
            [
                "strategy",
                "shard_count",
                "cut_weight_total",
                "size_std",
                "size_ratio",
                "type_balance_score",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved comparison CSV:  {csv_path}")
    print(f"Saved comparison JSON: {json_path}")


if __name__ == "__main__":
    main()
