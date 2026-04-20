import pandas as pd
import numpy as np

summary = pd.read_csv("data/curated_datasets/dataset_maine_v1/summary.csv")

# keep one row per asset if needed
if "asset_id" not in summary.columns:
    raise ValueError("summary.csv must contain asset_id")

if "asset_type" not in summary.columns:
    raise ValueError("summary.csv must contain asset_type for this crude heuristic")

base_scores = {
    "energy.transmission.substation": 0.80,
    "energy.distribution.substation": 0.70,
    "energy.distribution.substation_untyped": 0.65,
    "energy.generation.power_plant": 0.60,
    "energy.generation.generator": 0.55,
    "energy.generation.solar_farm": 0.45,
    "energy.generation.wind_farm": 0.40,
}

df = summary[["asset_id", "asset_type"]].copy()
df["road_access_score"] = df["asset_type"].map(base_scores).fillna(0.50)

# small jitter so every class is not perfectly tied
rng = np.random.default_rng(42)
df["road_access_score"] = (df["road_access_score"] + rng.normal(0, 0.05, len(df))).clip(0, 1)

df.to_csv("downstream/caadd/asset_access_table_maine.csv", index=False)
print(f"Saved {len(df)} rows to downstream/caadd/asset_access_table_maine.csv")