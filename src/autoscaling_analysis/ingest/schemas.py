# src/autoscaling_analysis/ingest/schemas.py

from typing import Dict, List


# ==============================
# Raw parsed dataframe schema
# ==============================

RAW_COLUMNS: List[str] = [
    "datetime",
    "host",
    "method",
    "url",
    "version",
    "status",
    "bytes",
    "bytes_missing_flag",
]

# Pandas extension dtypes (match notebook CELL 2)
RAW_DTYPES: Dict[str, str] = {
    "status": "Int16",
    "bytes": "Int64",
    "bytes_missing_flag": "Int8",
}