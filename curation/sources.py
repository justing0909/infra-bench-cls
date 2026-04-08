"""
sources.py
----------
Queries authoritative sources (OSM via Overpass API) for infrastructure
asset geometries and weak labels across four sectors:
  - power       (substations)
  - water       (water treatment plants)
  - telecom     (communication towers)
  - transport   (airports, train stations)

Each query returns a pandas DataFrame with one row per asset:
  - asset_id      : OSM element ID
  - asset_type    : sector label (e.g. "power_substation")
  - lat / lon     : centroid coordinates
  - name          : OSM name tag if available
  - source        : always "osm" for this module
  - osm_tags      : dict of all OSM tags (for provenance)

Usage:
    from sources import InfrastructureSource
    src = InfrastructureSource(bbox=(-70.9, 43.5, -66.9, 47.5))  # Maine
    df = src.query_all()
"""

import time
import requests
import pandas as pd


# --- OSM tag definitions per sector -------------------------------------------
# Each entry maps a human-readable asset_type to its Overpass filter string.
# nwr = nodes, ways, and relations (catches all geometry types in OSM).

SECTOR_QUERIES = {
    "power_substation":        'nwr["power"="substation"]',
    "water_treatment":         'nwr["man_made"="wastewater_plant"]'
                               + '\n  nwr["man_made"="water_works"]',
    "telecom_tower":           'nwr["man_made"="mast"]["tower:type"="communication"]'
                               + '\n  nwr["man_made"="tower"]["tower:type"="communication"]',
    "transport_airport":       'nwr["aeroway"="aerodrome"]',
    "transport_train_station": 'nwr["railway"="station"]',
}

# Public Overpass mirrors — tried in order, rotated on failure.
# All support the same API; having fallbacks avoids single-server flakiness.
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",       # main DE server
    "https://overpass.kumi.systems/api/interpreter",  # kumi.systems mirror
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",  # mail.ru mirror
]

REQUEST_DELAY  = 2.0   # seconds between sector requests — be polite
RETRY_ATTEMPTS = 3     # how many times to retry a failing request
RETRY_BACKOFF  = 5.0   # seconds to wait between retries (doubles each attempt)


class InfrastructureSource:
    """
    Queries OSM for infrastructure assets within a bounding box.

    Parameters
    ----------
    bbox : tuple of (min_lon, min_lat, max_lon, max_lat)
        Geographic bounding box for the query area.
        Note: Overpass expects (south, west, north, east) — we convert internally.
    timeout : int
        Overpass query timeout in seconds.
    """

    def __init__(self, bbox: tuple, timeout: int = 90):
        self.bbox = bbox                    # (min_lon, min_lat, max_lon, max_lat)
        self.timeout = timeout
        # Overpass uses (south, west, north, east) order
        min_lon, min_lat, max_lon, max_lat = bbox
        self._overpass_bbox = f"{min_lat},{min_lon},{max_lat},{max_lon}"

    def _build_query(self, filter_lines: str) -> str:
        """
        Wraps one or more Overpass filter lines in a bbox-scoped query.
        Returns results with centroid coordinates (out center).
        """
        lines = "\n  ".join(
            f"{line}({self._overpass_bbox});"
            for line in filter_lines.strip().split("\n")
            if line.strip()
        )
        return f"""
[out:json][timeout:{self.timeout}];
(
  {lines}
);
out center tags;
""".strip()

    def _run_query(self, query: str) -> list:
        """
        Sends query to Overpass with retry logic and endpoint rotation.

        Strategy:
          - Try each endpoint in OVERPASS_ENDPOINTS in order.
          - On failure, wait with exponential backoff and try the next endpoint.
          - After exhausting all endpoints, start over from the first (up to
            RETRY_ATTEMPTS total rounds).
          - Raises RuntimeError if all attempts across all endpoints fail.
        """
        last_error = None
        wait = RETRY_BACKOFF

        for attempt in range(1, RETRY_ATTEMPTS + 1):
            for endpoint in OVERPASS_ENDPOINTS:
                try:
                    response = requests.post(
                        endpoint,
                        data={"data": query},
                        timeout=self.timeout + 15,
                    )
                    response.raise_for_status()
                    return response.json().get("elements", [])

                except requests.exceptions.Timeout:
                    last_error = f"Timeout on {endpoint}"
                    print(f"    timeout on {endpoint} (attempt {attempt}) — trying next")
                except requests.exceptions.ConnectionError:
                    last_error = f"Connection error on {endpoint}"
                    print(f"    connection error on {endpoint} (attempt {attempt}) — trying next")
                except requests.exceptions.HTTPError as e:
                    last_error = str(e)
                    print(f"    HTTP error on {endpoint}: {e} — trying next")

            # All endpoints failed this round — wait before retrying
            if attempt < RETRY_ATTEMPTS:
                print(f"    all endpoints failed, waiting {wait:.0f}s before retry {attempt + 1}...")
                time.sleep(wait)
                wait *= 2  # exponential backoff

        raise RuntimeError(
            f"Query failed after {RETRY_ATTEMPTS} attempts across all endpoints. "
            f"Last error: {last_error}"
        )

    def _elements_to_df(self, elements: list, asset_type: str) -> pd.DataFrame:
        """
        Converts raw Overpass elements to a tidy DataFrame row per asset.
        Handles nodes (direct lat/lon) and ways/relations (center lat/lon).
        """
        rows = []
        for el in elements:
            # nodes have lat/lon directly; ways/relations have a 'center' key
            lat = el.get("lat") or (el.get("center") or {}).get("lat")
            lon = el.get("lon") or (el.get("center") or {}).get("lon")

            if lat is None or lon is None:
                continue  # skip elements with no resolvable centroid

            tags = el.get("tags", {})
            rows.append({
                "asset_id":   f"osm_{el['type']}_{el['id']}",
                "asset_type": asset_type,
                "lat":        float(lat),
                "lon":        float(lon),
                "name":       tags.get("name", ""),
                "source":     "osm",
                "osm_tags":   tags,
            })
        return pd.DataFrame(rows)

    def query_sector(self, asset_type: str) -> pd.DataFrame:
        """
        Queries OSM for a single sector and returns a DataFrame.

        Parameters
        ----------
        asset_type : str
            One of the keys in SECTOR_QUERIES.
        """
        if asset_type not in SECTOR_QUERIES:
            raise ValueError(
                f"Unknown asset_type '{asset_type}'. "
                f"Choose from: {list(SECTOR_QUERIES.keys())}"
            )

        filter_lines = SECTOR_QUERIES[asset_type]
        query = self._build_query(filter_lines)
        print(f"  [{asset_type}] querying...")
        elements = self._run_query(query)
        df = self._elements_to_df(elements, asset_type)
        print(f"  [{asset_type}] found {len(df)} assets")
        return df

    def query_all(self) -> pd.DataFrame:
        """
        Queries all sectors and returns a combined DataFrame.
        Sectors that fail after all retries are logged and skipped —
        the pipeline continues with whatever sectors succeeded.
        """
        frames = []
        failed = []

        for asset_type in SECTOR_QUERIES:
            try:
                df = self.query_sector(asset_type)
                frames.append(df)
            except RuntimeError as e:
                print(f"  [{asset_type}] FAILED after all retries: {e}")
                failed.append(asset_type)
            except Exception as e:
                print(f"  [{asset_type}] unexpected error: {e}")
                failed.append(asset_type)
            time.sleep(REQUEST_DELAY)

        if failed:
            print(f"\nWarning: {len(failed)} sector(s) could not be retrieved: {failed}")
            print("Re-run query_sector() for these individually, or try again later.")

        if not frames:
            print("No sectors returned data. Returning empty DataFrame.")
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        print(f"\nTotal assets found: {len(combined)} across {len(frames)} sector(s)")
        return combined


# --- Configuration ------------------------------------------------------------
# !! CHANGE THIS NAME before each new query run to avoid overwriting earlier data.
# Convention: data/<region>_<scope>_assets.csv
# Examples:
#   "data/maine_all_assets.csv"
#   "data/maine_power_only.csv"
#   "data/portland_me_assets.csv"
#   "data/new_hampshire_assets.csv"

OUTPUT_CSV = "data/maine_all_assets.csv"

QUERY_BBOX = (-71.1, 43.0, -66.9, 47.5)   # (min_lon, min_lat, max_lon, max_lat)


# --- Quick demo ---------------------------------------------------------------
if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)

    print(f"Output will be saved to: {OUTPUT_CSV}")
    print("(Change OUTPUT_CSV above before re-running for a different region.)\n")

    print("Querying OSM for infrastructure assets...")
    src = InfrastructureSource(bbox=QUERY_BBOX)
    df = src.query_all()

    if not df.empty:
        # drop osm_tags before saving — it's a dict column, not CSV-friendly
        df.drop(columns=["osm_tags"], errors="ignore").to_csv(OUTPUT_CSV, index=False)
        print(f"\nSaved {len(df)} assets to {OUTPUT_CSV}")

        print("\nSample results:")
        print(df[["asset_type", "lat", "lon", "name"]].head(10).to_string(index=False))

        print("\nAsset counts by sector:")
        print(df["asset_type"].value_counts().to_string())