"""
Ingestion layer:
- Parse raw NASA access logs
- Normalize dtypes
- Stream large files safely
"""

from .parse_logs import (
    parse_line,
    parse_file_streaming,
    iter_parsed_events,
    compile_regex_from_config,
)

from .schemas import RAW_COLUMNS, RAW_DTYPES

__all__ = [
    "parse_line",
    "parse_file_streaming",
    "iter_parsed_events",
    "compile_regex_from_config",
    "RAW_COLUMNS",
    "RAW_DTYPES",
]