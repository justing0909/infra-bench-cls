from __future__ import annotations

import os
from typing import Any, Optional

import openpyxl


SHEET_NAME = "Sheet1"
HEADER_ROW = 2
DATA_START_ROW = 3

COLUMN_MAP = {
    "region": "sentinel data",
    "starting_file_size_kb": "starting file size (KB)",
    "pre_filter_time_s": "pre-filter time (s)", # pre-filter time (s)
    "power_only_file_size_mb": 'pre-filter ending file size (MB) [the "power_only" dataset]',
    "compression_time_s": 'compression to "non_solar_gen" time (s)',
    "collapsed_file_size_mb": 'non_solar_gen file size (MB)',
    "scanning_time_s": 'scanning time of non_solar_gen file (s)',
    "gee_accept_pct": "GEE Percent of accepted tiles (%)",
    "qc_accept_pct": "QC Percent of accepted tiles (%)",
    "triage_accept_pct": "triage percent of accepted tiles (%)",
    "assets_extracted": "assets extracted",
    "assets_after_dedup": "assets after dedup",
    "total_tiles_fetched": "total tiles fetched",
    "dataset_tiles": "dataset tiles",
    "total_time_elapsed_s": "total time elapsed (s)",
}


def _normalize_region(region: str) -> str:
    return region.strip().lower().replace("_", " ")


def _find_header_columns(ws) -> dict[str, int]:
    headers = {}
    for col in range(1, ws.max_column + 1):
        value = ws.cell(row=HEADER_ROW, column=col).value
        if isinstance(value, str):
            headers[value.strip()] = col
    return headers


def _find_or_create_region_row(ws, region: str) -> int:
    norm = _normalize_region(region)

    for row in range(DATA_START_ROW, ws.max_row + 1):
        value = ws.cell(row=row, column=1).value
        if isinstance(value, str) and _normalize_region(value) == norm:
            return row

    row = ws.max_row + 1
    ws.cell(row=row, column=1).value = region
    return row


def update_timing_log(
    workbook_path: str,
    region: str,
    **fields: Any,
) -> None:
    """
    Update one region row in the timing workbook with partial stats.
    Fields not supplied are left untouched.
    """
    if not os.path.exists(workbook_path):
        raise FileNotFoundError(f"Workbook not found: {workbook_path}")

    wb = openpyxl.load_workbook(workbook_path)
    ws = wb[SHEET_NAME]

    header_cols = _find_header_columns(ws)
    row = _find_or_create_region_row(ws, region)

    # Ensure region name itself is written
    ws.cell(row=row, column=1).value = region

    for key, value in fields.items():
        if key not in COLUMN_MAP:
            raise KeyError(f"Unknown timing-log field: {key}")

        column_name = COLUMN_MAP[key]
        if column_name not in header_cols:
            raise KeyError(f"Column not found in workbook: {column_name}")

        col = header_cols[column_name]
        ws.cell(row=row, column=col).value = value

    wb.save(workbook_path)


def file_size_kb(path: str) -> Optional[float]:
    if not path or not os.path.exists(path):
        return None
    return round(os.path.getsize(path) / 1024.0, 2)


def file_size_mb(path: str) -> Optional[float]:
    if not path or not os.path.exists(path):
        return None
    return round(os.path.getsize(path) / (1024.0 * 1024.0), 2)