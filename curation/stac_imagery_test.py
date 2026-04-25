# test_single_tile.py
import pystac_client
import planetary_computer
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
import numpy as np

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

bbox = (-70.55, 43.61, -70.45, 43.71)

# --- Sentinel-2 ---
print("Testing Sentinel-2...")
search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime="2021-01-01/2024-12-31",
    limit=50,
)
items = list(search.items())
items.sort(key=lambda x: x.properties.get("eo:cloud_cover", 100))
clean = [i for i in items if i.properties.get("eo:cloud_cover", 100) < 20]
print(f"  Found {len(clean)} clean scenes")

item = planetary_computer.sign(clean[0])
print(f"  Using: {item.id}")

href = item.assets["B04"].href
with rasterio.open(href) as src:
    bbox_native = transform_bounds(
        CRS.from_epsg(4326), src.crs,
        bbox[0], bbox[1], bbox[2], bbox[3],
    )
    window = rasterio.windows.from_bounds(*bbox_native, transform=src.transform)
    data = src.read(1, window=window)
    print(f"  B04 shape: {data.shape}")
    print(f"  B04 min/max: {data.min()} / {data.max()}")

# --- Sentinel-1 ---
print("\nTesting Sentinel-1...")
search = catalog.search(
    collections=["sentinel-1-grd"],
    bbox=bbox,
    datetime="2021-01-01/2024-12-31",
    limit=10,
)
items = list(search.items())
if items:
    item = planetary_computer.sign(items[0])
    print(f"  Using: {item.id}")
    print(f"  Assets: {[k for k in item.assets.keys() if k in ['VV', 'VH']]}")
    if "VV" in item.assets:
        href = item.assets["VV"].href
        with rasterio.open(href) as src:
            bbox_native = transform_bounds(
                CRS.from_epsg(4326), src.crs,
                bbox[0], bbox[1], bbox[2], bbox[3],
            )
            window = rasterio.windows.from_bounds(*bbox_native, transform=src.transform)
            data = src.read(1, window=window)
            print(f"  VV shape: {data.shape}")
            print(f"  VV min/max: {data.min():.3f} / {data.max():.3f}")
else:
    print("  No Sentinel-1 scenes found")

# --- Full STACImageryFetcher test ---
print("\nTesting STACImageryFetcher end-to-end...")
import pandas as pd
from stac_imagery import STACImageryFetcher

df = pd.DataFrame([{
    "asset_id":   "test_001",
    "asset_type": "energy.distribution.substation_untyped",
    "lat":        43.66,
    "lon":        -70.25,
}])

fetcher = STACImageryFetcher(
    buffer_m=300,
    modalities=["sentinel2_ms"],
    temporal_stack=False,
)
results = fetcher.fetch_all(df)
r = results[0]
print(f"  status:      {r.status}")
print(f"  error_msg:   {r.error_msg}")
print(f"  image shape: {r.image_shape}")
print(f"  n_bands:     {r.n_bands}")
print(f"  image_date:  {r.image_date}")