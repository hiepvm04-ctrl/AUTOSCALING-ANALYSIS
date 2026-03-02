# src/autoscaling_analysis/ingest/parse_logs.py

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Iterator, Optional, Tuple, Dict, Any, List

import pandas as pd

from .schemas import RAW_COLUMNS, RAW_DTYPES


# ============================================================
# Default patterns (match notebook exactly)
# ============================================================

DEFAULT_DT_FORMAT = "%d/%b/%Y:%H:%M:%S %z"

DEFAULT_LOG_RE = re.compile(
    r'^(?P<host>\S+)\s+\S+\s+\S+\s+\[(?P<ts>[^\]]+)\]\s+'
    r'"(?P<request>[^"]*)"\s+(?P<status>\d{3})\s+(?P<bytes>\S+)\s*$'
)

DEFAULT_REQ_RE = re.compile(
    r'^(?P<method>[A-Z]+)\s+(?P<url>\S+)\s+(?P<version>HTTP/\d\.\d)$'
)


# ============================================================
# Dataclass for one parsed line
# ============================================================

@dataclass(frozen=True)
class ParsedEvent:
    datetime: object
    host: str
    method: str
    url: str
    version: str
    status: object
    bytes: object
    bytes_missing_flag: int


# ============================================================
# Line parser
# ============================================================

def parse_line(
    line: str,
    *,
    log_re: re.Pattern = DEFAULT_LOG_RE,
    req_re: re.Pattern = DEFAULT_REQ_RE,
    dt_format: str = DEFAULT_DT_FORMAT,
) -> Optional[ParsedEvent]:
    """
    Parse a single log line.

    Returns:
        ParsedEvent or None if regex doesn't match.
    """

    m = log_re.match(line)
    if not m:
        return None

    host = m.group("host")
    ts_raw = m.group("ts")
    req_raw = m.group("request")
    status_raw = m.group("status")
    bytes_raw = m.group("bytes")

    # --- datetime ---
    try:
        dt = datetime.strptime(ts_raw, dt_format)
    except Exception:
        dt = pd.NaT

    # --- request ---
    method = url = version = "UNKNOWN"
    rm = req_re.match(req_raw.strip())
    if rm:
        method = rm.group("method")
        url = rm.group("url")
        version = rm.group("version")

    # --- status ---
    try:
        status = int(status_raw)
    except Exception:
        status = pd.NA

    # --- bytes ---
    if bytes_raw in ("-", ""):
        bval = pd.NA
        miss = 1
    else:
        try:
            bval = int(bytes_raw)
            miss = 0
        except Exception:
            bval = pd.NA
            miss = 1

    return ParsedEvent(
        datetime=dt,
        host=host,
        method=method,
        url=url,
        version=version,
        status=status,
        bytes=bval,
        bytes_missing_flag=miss,
    )


# ============================================================
# Iterator (memory safe)
# ============================================================

def iter_parsed_events(
    lines: Iterable[str],
    *,
    log_re: re.Pattern = DEFAULT_LOG_RE,
    req_re: re.Pattern = DEFAULT_REQ_RE,
    dt_format: str = DEFAULT_DT_FORMAT,
) -> Iterator[ParsedEvent]:
    for line in lines:
        line = line.rstrip("\n")
        ev = parse_line(
            line,
            log_re=log_re,
            req_re=req_re,
            dt_format=dt_format,
        )
        if ev is not None:
            yield ev


# ============================================================
# Normalize dataframe dtypes
# ============================================================

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    for col, dtype in RAW_DTYPES.items():
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
    return df


# ============================================================
# Streaming file parser
# ============================================================

def parse_file_streaming(
    path: str,
    *,
    chunk_lines: int = 300_000,
    log_re: re.Pattern = DEFAULT_LOG_RE,
    req_re: re.Pattern = DEFAULT_REQ_RE,
    dt_format: str = DEFAULT_DT_FORMAT,
    encoding_errors: str = "replace",
) -> pd.DataFrame:
    """
    Stream large log file into dataframe.
    """

    parts: List[pd.DataFrame] = []
    buf: List[Tuple] = []

    with open(path, "r", errors=encoding_errors) as f:
        for line in f:
            ev = parse_line(
                line.rstrip("\n"),
                log_re=log_re,
                req_re=req_re,
                dt_format=dt_format,
            )
            if ev is None:
                continue

            buf.append((
                ev.datetime,
                ev.host,
                ev.method,
                ev.url,
                ev.version,
                ev.status,
                ev.bytes,
                ev.bytes_missing_flag,
            ))

            if len(buf) >= int(chunk_lines):
                df = pd.DataFrame(buf, columns=RAW_COLUMNS)
                parts.append(_normalize_df(df))
                buf = []

    if buf:
        df = pd.DataFrame(buf, columns=RAW_COLUMNS)
        parts.append(_normalize_df(df))

    if not parts:
        return pd.DataFrame(columns=RAW_COLUMNS)

    return pd.concat(parts, ignore_index=True)


# ============================================================
# Build regex from config.yaml
# ============================================================

def compile_regex_from_config(cfg: Dict[str, Any]) -> Tuple[re.Pattern, re.Pattern, str]:
    """
    Extract regex + datetime format from loaded config.
    """

    dt_format = cfg.get("time", {}).get("dt_format", DEFAULT_DT_FORMAT)
    log_pat = cfg.get("ingest", {}).get("log_regex", DEFAULT_LOG_RE.pattern)
    req_pat = cfg.get("ingest", {}).get("req_regex", DEFAULT_REQ_RE.pattern)

    return re.compile(log_pat), re.compile(req_pat), dt_format
