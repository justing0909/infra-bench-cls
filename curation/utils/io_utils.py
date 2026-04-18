import json
import pandas as pd

def load_asset_table(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported asset table format: {path}")

    if "osm_tags" in df.columns:
        try:
            df["osm_tags"] = df["osm_tags"].apply(
                lambda x: json.loads(x) if isinstance(x, str) and x.startswith("{") else x
            )
        except Exception:
            pass

    return df